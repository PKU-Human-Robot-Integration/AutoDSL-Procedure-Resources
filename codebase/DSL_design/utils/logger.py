import logging
import os

def create_logger(filename, file_handle=True):
    '''
    Creates a logger with both console and file handlers.

        @Arguments:
            filename (str): Name of the log file.
            file_handle (bool, optional): Flag indicating whether to create a file handler. Defaults to True.

        @Returns:
            logging.Logger: Configured logger object.
    '''

    # create logger
    logger = logging.getLogger(filename)
    logger.propagate = False
    logger.setLevel(logging.DEBUG)
    # create console handler with a higher log level
    ch = logging.StreamHandler()
    ch.setLevel(logging.INFO)
    stream_formatter = logging.Formatter('%(message)s')
    ch.setFormatter(stream_formatter)
    logger.addHandler(ch)

    if file_handle:
        # create file handler which logs even debug messages
        fh = logging.FileHandler(filename, mode='a')
        fh.setLevel(logging.DEBUG)
        file_formatter = logging.Formatter('[%(asctime)s] %(message)s')
        fh.setFormatter(file_formatter)
        logger.addHandler(fh)

    return logger