import logging
from . import dist_utils
from . import config


logger = logging.getLogger(config.APP_NAME)


def log_warning(msg):
    if dist_utils.is_main_proc():
        logger.warn(msg)


def log_debug(msg):
    if dist_utils.is_main_proc():
        logger.debug(msg)


def log_info(msg):
    if dist_utils.is_main_proc():
        logger.info(msg)
