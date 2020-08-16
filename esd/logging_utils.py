import logging
from . import dist_utils
from . import config


logger = logging.getLogger(config.APP_NAME)


def log_warning(msg):
    """
    Logs the given message as a warning.
    :param msg: string containing the desired message.
    """
    if dist_utils.is_main_proc():
        logger.warn(msg)


def log_debug(msg):
    """
    Logs the given message as a debug message.
    :param msg: string containing the desired message.
    """
    if dist_utils.is_main_proc():
        logger.debug(msg)


def log_info(msg):
    """
    Logs the given message as an information.
    :param msg: string containing the desired message.
    """
    if dist_utils.is_main_proc():
        logger.info(msg)
