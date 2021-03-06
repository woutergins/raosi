from .core import *

import logging

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)
logger.addHandler(logging.NullHandler())

ch = logging.StreamHandler()
ch.setLevel(logging.INFO)
formatter = logging.Formatter("%(asctime)s:%(name)-12s:%(levelname)-8s:%(message)s")
ch.setFormatter(formatter)

logger.addHandler(ch)


def setLogLevel(level):
    """Set the log level for the"""
    logger.setLevel(level.upper())
    ch.setLevel(level.upper())
