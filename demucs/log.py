"""
Logging utilities for Demucs.
"""
import logging
import sys

logger = logging.getLogger(__name__)

def fatal(*args):
    """Log a fatal error and exit."""
    logger.fatal(*args)
    sys.exit(1)
