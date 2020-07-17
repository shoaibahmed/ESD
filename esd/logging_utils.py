import logging
from . import dist_utils
from . import config


def log_warning(msg):
    if dist_utils.is_main_proc():
        logging.warn(f"[{config.APP_NAME}] {msg}")


def log_debug(msg):
    if dist_utils.is_main_proc():
        logging.debug(f"[{config.APP_NAME}] {msg}")


def log_info(msg):
    if dist_utils.is_main_proc():
        logging.info(f"[{config.APP_NAME}] {msg}")
