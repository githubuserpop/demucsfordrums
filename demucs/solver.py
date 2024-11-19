# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
"""Main training loop."""

import logging

from dora import get_xp
from dora.utils import write_and_rename
from dora.log import LogProgress, bold
import torch
import torch.nn.functional as F

from . import augment, distrib, states, pretrained
from .apply import apply_model
from .ema import ModelEMA
from .evaluate import evaluate, new_sdr
from .svd import svd_penalty
from .drum_losses import DrumPatternLoss
from .utils import pull_metric, EMA
from .train import load_custom_data  # Import the custom data loading function

logger = logging.getLogger(__name__)


def _summary(metrics):
    return " | ".join(f"{key.capitalize()}={val}" for key, val in metrics.items())


class Solver(object):
    def __init__(self, loaders, model, optimizer, args):
        self.args = args
        self.loaders = loaders or {
            'train': load_custom_data(TRAINING_FOLDERS, args.batch_size),
            'valid': load_custom_data(TRAINING_FOLDERS, args.batch_size)
        }

        self.model = model
        self.optimizer = optimizer
        self.quantizer = states.get_quantizer(self.model, args.quant, self.optimizer)
        self.dmodel = distrib.wrap(model)
        self.device = next(iter(self.model.parameters())).device

        # Exponential moving average of the model, either updated every batch or epoch.
        # The best model from all the EMAs and the original one is kept based on the valid
        # loss for the final best model.
        self.emas = {'batch': [], 'epoch': []}
        for kind in self.emas.keys():
            decays = getattr(args.ema, kind)
            device = self.device if kind == 'batch' else 'cpu'
            if decays:
                for decay in decays:
                    self.emas[kind].append(ModelEMA(self.model, decay, device=device))

        # data augment
        augments = [augment.Shift(shift=int(args.dset.samplerate * args.dset.shift),
                                  same=args.augment.shift_same)]
        if args.augment.flip:
            augments += [augment.FlipChannels(), augment.FlipSign()]
        for aug in ['scale', 'remix']:
            kw = getattr(args.augment, aug)
            if kw.proba:
                augments.append(getattr(augment, aug.capitalize())(**kw))
        self.augment = torch.nn.Sequential(*augments)

        xp = get_xp()
        self.folder = xp.folder
        # Checkpoints
        self.checkpoint_file = xp.folder / 'checkpoint.th'
        self.best_file = xp.folder / 'best.th'
        logger.debug("Checkpoint will be saved to %s", self.checkpoint_file.resolve())
        self.best_state = None
        self.best_changed = False

        self.link = xp.link
        self.history = self.link.history

        self._reset()

    def _serialize(self, epoch):
        package = {}
        package['state'] = self.model.state_dict()
        package['optimizer'] = self.optimizer.state_dict()
        package['history'] = self.history
        package['best_state'] = self.best_state
        package['args'] = self.args
        for kind, emas in self.emas.items():
            for k, ema in enumerate(emas):
                package[f'ema_{kind}_{k}'] = ema.state_dict()
        with write_and_rename(self.checkpoint_file) as tmp:
            torch.save(package, tmp)

        save_every = self.args.save_every
        if save_every and (epoch + 1) % save_every == 0 and epoch + 1 != self.args.epochs:
            with write_and_rename(self.folder / f'checkpoint_{epoch + 1}.th') as tmp:
                torch.save(package, tmp)

        if self.best_changed:
            # Saving only the latest best model.
            with write_and_rename(self.best_file) as tmp:
                package = states.serialize_model(self.model, self.args)
                package['state'] = self.best_state
                torch.save(package, tmp)
            self.best_changed = False

    def _reset(self):
        """Reset state of the solver, potentially using checkpoint."""
        if self.checkpoint_file.exists():
            logger.info(f'Loading checkpoint model: {self.checkpoint_file}')
            package = torch.load(self.checkpoint_file, 'cpu')
            self.model.load_state_dict(package['state'])
            self.optimizer.load_state_dict(package['optimizer'])
            self.history[:] = package['history']
            self.best_state = package['best_state']
            for kind, emas in self.emas.items():
                for k, ema in enumerate(emas):
                    ema.load_state_dict(package[f'ema_{kind}_{k}'])
        elif self.args.continue_pretrained:
            model = pretrained.get_model(
                name=self.args.continue_pretrained,
                repo=self.args.pretrained_repo)
            self.model.load_state_dict(model.state_dict())
        elif self.args.continue_from:
            name = 'checkpoint.th'
            root = self.folder.parent
            cf = root / str(self.args.continue_from) / name
            logger.info("Loading from %s", cf)
            package = torch.load(cf, 'cpu')
            self.best_state = package['best_state']
            if self.args.continue_best:
                self.model.load_state_dict(package['best_state'], strict=False)
            else:
                self.model.load_state_dict(package['state'], strict=False)
            if self.args.continue_opt:
                self.optimizer.load_state_dict(package['optimizer'])

    def _format_train(self, metrics: dict) -> dict:
        """Formatting for train/valid metrics."""
        losses = {
            'loss': format(metrics['loss'], ".4f"),
            'reco': format(metrics['reco'], ".4f"),
        }
        if 'nsdr' in metrics:
            losses['nsdr'] = format(metrics['nsdr'], ".3f")
        if self.quantizer is not None:
            losses['ms'] = format(metrics['ms'], ".2f")
        if 'grad' in metrics:
            losses['grad'] = format(metrics['grad'], ".4f")
        if 'best' in metrics:
            losses['best'] = format(metrics['best'], '.4f')
        if 'bname' in metrics:
            losses['bname'] = metrics['bname']
        if 'penalty' in metrics:
            losses['penalty'] = format(metrics['penalty'], ".4f")
        if 'hloss' in metrics:
            losses['hloss'] = format(metrics['hloss'], ".4f")
        return losses

    def _format_test(self, metrics: dict) -> dict:
        """Formatting for test metrics."""
        losses = {}
        if 'sdr' in metrics:
            losses['sdr'] = format(metrics['sdr'], '.3f')
        if 'nsdr' in metrics:
            losses['nsdr'] = format(metrics['nsdr'], '.3f')
        for source in self.model.sources:
            key = f'sdr_{source}'
            if key in metrics:
                losses[key] = format(metrics[key], '.3f')
            key = f'nsdr_{source}'
            if key in metrics:
                losses[key] = format(metrics[key], '.3f')
        return losses

    def train(self):
        # Optimizing the model
        if self.history:
            logger.info("Replaying metrics from previous run")
        for epoch, metrics in enumerate(self.history):
            formatted = self._format_train(metrics['train'])
            logger.info(
                bold(f'Train Summary | Epoch {epoch + 1} | {_summary(formatted)}'))
            formatted = self._format_train(metrics['valid'])
            logger.info(
                bold(f'Valid Summary | Epoch {epoch + 1} | {_summary(formatted)}'))
            if 'test' in metrics:
                formatted = self._format_test(metrics['test'])
                if formatted:
                    logger.info(bold(f"Test Summary | Epoch {epoch + 1} | {_summary(formatted)}"))

        epoch = 0
        for epoch in range(len(self.history), self.args.epochs):
            # Train one epoch
            self.model.train()  # Turn on BatchNorm & Dropout
            metrics = {}
            logger.info('-' * 70)
            logger.info("Training...")
            metrics['train'] = self._run_one_epoch(epoch)
            formatted = self._format_train(metrics['train'])
            logger.info(
                bold(f'Train Summary | Epoch {epoch + 1} | {_summary(formatted)}'))

            # Cross validation
            logger.info('-' * 70)
            logger.info('Cross validation...')
            self.model.eval()  # Turn off Batchnorm & Dropout
            with torch.no_grad():
                valid = self._run_one_epoch(epoch, train=False)
                bvalid = valid
                bname = 'main'
                state = states.copy_state(self.model.state_dict())
                metrics['valid'] = {}
                metrics['valid']['main'] = valid
                key = self.args.test.metric
                for kind, emas in self.emas.items():
                    for k, ema in enumerate(emas):
                        with ema.swap():
                            valid = self._run_one_epoch(epoch, train=False)
                        name = f'ema_{kind}_{k}'
                        metrics['valid'][name] = valid
                        a = valid[key]
                        b = bvalid[key]
                        if key.startswith('nsdr'):
                            a = -a
                            b = -b
                        if a < b:
                            bvalid = valid
                            state = ema.state
                            bname = name
                    metrics['valid'].update(bvalid)
                    metrics['valid']['bname'] = bname

            valid_loss = metrics['valid'][key]
            mets = pull_metric(self.link.history, f'valid.{key}') + [valid_loss]
            if key.startswith('nsdr'):
                best_loss = max(mets)
            else:
                best_loss = min(mets)
            metrics['valid']['best'] = best_loss
            if self.args.svd.penalty > 0:
                kw = dict(self.args.svd)
                kw.pop('penalty')
                with torch.no_grad():
                    penalty = svd_penalty(self.model, exact=True, **kw)
                metrics['valid']['penalty'] = penalty

            formatted = self._format_train(metrics['valid'])
            logger.info(
                bold(f'Valid Summary | Epoch {epoch + 1} | {_summary(formatted)}'))

            # Save the best model
            if valid_loss == best_loss or self.args.dset.train_valid:
                logger.info(bold('New best valid loss %.4f'), valid_loss)
                self.best_state = states.copy_state(state)
                self.best_changed = True

            # Eval model every `test.every` epoch or on last epoch
            should_eval = (epoch + 1) % self.args.test.every == 0
            is_last = epoch == self.args.epochs - 1
            # # Tries to detect divergence in a reliable way and finish job
            # # not to waste compute.
            # # Commented out as this was super specific to the MDX competition.
            # reco = metrics['valid']['main']['reco']
            # div = epoch >= 180 and reco > 0.18
            # div = div or epoch >= 100 and reco > 0.25
            # div = div and self.args.optim.loss == 'l1'
            # if div:
            #     logger.warning("Finishing training early because valid loss is too high.")
            #     is_last = True
            if should_eval or is_last:
                # Evaluate on the testset
                logger.info('-' * 70)
                logger.info('Evaluating on the test set...')
                # We switch to the best known model for testing
                if self.args.test.best:
                    state = self.best_state
                else:
                    state = states.copy_state(self.model.state_dict())
                compute_sdr = self.args.test.sdr and is_last
                with states.swap_state(self.model, state):
                    with torch.no_grad():
                        metrics['test'] = evaluate(self, compute_sdr=compute_sdr)
                formatted = self._format_test(metrics['test'])
                logger.info(bold(f"Test Summary | Epoch {epoch + 1} | {_summary(formatted)}"))
            self.link.push_metrics(metrics)

            if distrib.rank == 0:
                # Save model each epoch
                self._serialize(epoch)
                logger.debug("Checkpoint saved to %s", self.checkpoint_file.resolve())
            if is_last:
                break

    def _run_one_epoch(self, epoch, train=True):
        args = self.args
        data_loader = self.loaders['train'] if train else self.loaders['valid']
        if distrib.world_size > 1 and train:
            data_loader.sampler.set_epoch(epoch)

        label = "Train" if train else "Valid"
        name = f"{label} | Epoch {epoch + 1}"
        total = min(len(data_loader), args.max_batches or float('inf'))
        logprog = LogProgress(logger, data_loader, total=total,
                              updates=self.args.misc.num_prints, name=name)
        averager = EMA()

        for idx, sources in enumerate(logprog):
            sources = sources.to(self.device)
            if train:
                self.optimizer.zero_grad()
            else:
                mix = sources[:, 0]

            # Forward pass with unpacked drum component outputs
            kick_pred, clap_pred, snare_pred, hihat_pred, openhat_pred, perc_pred = self.dmodel(mix)

            # Separate target sources for each drum component
            if sources.shape[1] != 6:
                raise ValueError("Expected 6 sources for drum components")
            kick_target, clap_target, snare_target, hihat_target, openhat_target, perc_target = sources

            # Initialize pattern-aware loss
            pattern_loss = DrumPatternLoss(sample_rate=args.dset.samplerate).to(self.device)

            # Calculate both standard and pattern-aware losses for each component
            loss_fn = F.l1_loss if args.optim.loss == 'l1' else F.mse_loss

            # Kick drum losses
            kick_recon_loss = loss_fn(kick_pred, kick_target)
            kick_pattern_loss = pattern_loss(kick_pred, kick_target, 'kick')
            kick_loss = kick_recon_loss + 0.5 * kick_pattern_loss

            # Snare/Clap losses
            clap_recon_loss = loss_fn(clap_pred, clap_target)
            clap_pattern_loss = pattern_loss(clap_pred, clap_target, 'clap')
            clap_loss = clap_recon_loss + 0.4 * clap_pattern_loss

            snare_recon_loss = loss_fn(snare_pred, snare_target)
            snare_pattern_loss = pattern_loss(snare_pred, snare_target, 'snare')
            snare_loss = snare_recon_loss + 0.4 * snare_pattern_loss

            # Hi-hat losses
            hihat_recon_loss = loss_fn(hihat_pred, hihat_target)
            hihat_pattern_loss = pattern_loss(hihat_pred, hihat_target, 'hihat')
            hihat_loss = hihat_recon_loss + 0.3 * hihat_pattern_loss

            openhat_recon_loss = loss_fn(openhat_pred, openhat_target)
            openhat_pattern_loss = pattern_loss(openhat_pred, openhat_target, 'openhat')
            openhat_loss = openhat_recon_loss + 0.3 * openhat_pattern_loss

            # Percussion losses
            perc_recon_loss = loss_fn(perc_pred, perc_target)
            perc_pattern_loss = pattern_loss(perc_pred, perc_target, 'perc')
            perc_loss = perc_recon_loss + 0.3 * perc_pattern_loss

            # Aggregate the individual component losses
            total_loss = (kick_loss + clap_loss + snare_loss + hihat_loss + openhat_loss + perc_loss) / 6

            # Collect losses for logging (convert to scalar with .item())
            losses = {
                'loss': total_loss.item(),
                'kick_loss': kick_loss.item(),
                'clap_loss': clap_loss.item(),
                'snare_loss': snare_loss.item(),
                'hihat_loss': hihat_loss.item(),
                'openhat_loss': openhat_loss.item(),
                'perc_loss': perc_loss.item(),
            }

            # Backpropagate and optimize in training mode
            if train:
                total_loss.backward()
                self.optimizer.step()

            # Log progress and update averages
            losses = averager(losses)
            logs = self._format_train(losses)
            logprog.update(**logs)

            if args.max_batches and idx >= args.max_batches - 1:
                break
            if self.args.debug and train:
                break
            if self.args.flag == 'debug':
                break

        # Convert all loss tensors to scalars before averaging
        avg_losses = {k: v if isinstance(v, float) else v.item() for k, v in losses.items()}
        return distrib.average(avg_losses, idx + 1)