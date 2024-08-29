import logging

def create_logger(name='global_logger', log_file=None):
    """ use different log level for file and stream
    """
    logger = logging.getLogger(name)
    formatter = logging.Formatter('[%(asctime)s] %(message)s')
    logger.setLevel(logging.DEBUG)

    sh = logging.StreamHandler()
    sh.setFormatter(formatter)
    sh.setLevel(logging.INFO)
    # sh.setLevel(logging.ERROR)
    logger.addHandler(sh)

    if log_file is not None:
        print("==== logger path", log_file)
        fh = logging.FileHandler(log_file)
        fh.setFormatter(formatter)
        fh.setLevel(logging.DEBUG)
        logger.addHandler(fh)

    return logger
