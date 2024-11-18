# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the LICENSE file in the root directory of this source tree.

"""Test time evaluation, using either the original SDR or the MDX 2021 competition SDR (reported as `nsdr`).
"""

import logging
from concurrent import futures

from dora.log import LogProgress
import numpy as np
import musdb
import museval
import torch as th

from .apply import apply_model
from .audio import convert_audio, save_audio
from . import distrib
from .utils import DummyPoolExecutor

logger = logging.getLogger(__name__)

def new_sdr(references, estimates):
    """Compute SDR based on the MDX competition definition."""
    delta = 1e-7  # for numerical stability
    num = th.sum(th.square(references), dim=(2, 3)) + delta
    den = th.sum(th.square(references - estimates), dim=(2, 3)) + delta
    scores = 10 * th.log10(num / den)
    return scores

def eval_track(references, estimates, win_size, hop_size, compute_sdr=True):
    """Evaluate a single track and calculate SDR scores."""
    references, estimates = references.transpose(1, 2).double(), estimates.transpose(1, 2).double()
    nsdr_scores = new_sdr(references.cpu()[None], estimates.cpu()[None])[0]
    
    if not compute_sdr:
        return None, nsdr_scores
    
    refs_np, ests_np = references.numpy(), estimates.numpy()
    museval_scores = museval.metrics.bss_eval(
        refs_np, ests_np, compute_permutation=False, window=win_size,
        hop=hop_size, framewise_filters=False, bsseval_sources_version=False
    )[:-1]
    
    return museval_scores, nsdr_scores

def evaluate(solver, compute_sdr=False):
    """Evaluate the model using museval or MDX SDR, and save metrics."""
    args = solver.args
    output_dir = solver.folder / "results"
    json_folder = output_dir / "test"
    output_dir.mkdir(exist_ok=True, parents=True)
    json_folder.mkdir(exist_ok=True, parents=True)

    # Load test set
    test_set = musdb.DB(args.dset.musdb, subsets=["test"], is_wav=args.test.nonhq is None)
    src_rate = args.dset.musdb_samplerate
    eval_device = 'cpu'
    
    model = solver.model
    win_size, hop_size = int(model.samplerate), int(model.samplerate)

    indexes = range(distrib.rank, len(test_set), distrib.world_size)
    progress = LogProgress(logger, indexes, updates=args.misc.num_prints, name='Eval')
    eval_tasks = []
    
    executor_cls = futures.ProcessPoolExecutor if args.test.workers else DummyPoolExecutor
    with executor_cls(args.test.workers) as pool:
        for idx in progress:
            track = test_set.tracks[idx]
            mix = th.from_numpy(track.audio).t().float().to(solver.device)
            ref = mix.mean(dim=0)
            mix = (mix - ref.mean()) / ref.std()
            mix = convert_audio(mix, src_rate, model.samplerate, model.audio_channels)
            
            estimates = apply_model(model, mix[None], shifts=args.test.shifts,
                                    split=args.test.split, overlap=args.test.overlap)[0]
            estimates = (estimates * ref.std() + ref.mean()).to(eval_device)

            references = th.stack([th.from_numpy(track.targets[name].audio).t() for name in model.sources]).to(eval_device)
            references = convert_audio(references, src_rate, model.samplerate, model.audio_channels)
            
            if args.test.save:
                track_output_dir = solver.folder / "wav" / track.name
                track_output_dir.mkdir(exist_ok=True, parents=True)
                for source_name, est in zip(model.sources, estimates):
                    save_audio(est.cpu(), track_output_dir / f"{source_name}.mp3", model.samplerate)
            
            eval_tasks.append((track.name, pool.submit(eval_track, references, estimates, win_size, hop_size, compute_sdr)))

        eval_tasks = LogProgress(logger, eval_tasks, updates=args.misc.num_prints, name='Eval (BSS)')
        track_scores = {}

        for track_name, task in eval_tasks:
            museval_scores, nsdr_scores = task.result()
            track_scores[track_name] = {source: {'nsdr': [float(nsdr_scores[i])]} for i, source in enumerate(model.sources)}

            if museval_scores:
                sdr, isr, sir, sar = museval_scores
                for i, source in enumerate(model.sources):
                    metrics = {
                        "SDR": sdr[i].tolist(),
                        "SIR": sir[i].tolist(),
                        "ISR": isr[i].tolist(),
                        "SAR": sar[i].tolist()
                    }
                    track_scores[track_name][source].update(metrics)

    # Consolidate results across distributed processes
    consolidated_scores = {}
    for src in range(distrib.world_size):
        consolidated_scores.update(distrib.share(track_scores, src))

    metrics_summary = {}
    metric_keys = next(iter(consolidated_scores.values()))[model.sources[0]].keys()
    for metric in metric_keys:
        avg_metric, median_metric = 0, 0
        for source in model.sources:
            medians = [np.nanmedian(consolidated_scores[track][source][metric]) for track in consolidated_scores.keys()]
            mean_metric = np.mean(medians)
            median_metric_src = np.median(medians)
            
            metrics_summary[f"{metric.lower()}_{source}"] = mean_metric
            metrics_summary[f"{metric.lower()}_med_{source}"] = median_metric_src
            
            avg_metric += mean_metric / len(model.sources)
            median_metric += median_metric_src / len(model.sources)
        
        metrics_summary[metric.lower()] = avg_metric
        metrics_summary[f"{metric.lower()}_med"] = median_metric

    return metrics_summary