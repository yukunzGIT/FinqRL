import os

from nltk.stem.porter import PorterStemmer as PS # PorterStemmer is used for stemming words (reducing words to their root form)
# Ex: stemmer = PS()
# stemmer.stem("running")  # Returns "run"

import logging #  this module is used for creating logs.
# getLogger retrieves a logger instance, StreamHandler logs to the console, FileHandler logs to a file, and Formatter customizes the log format.
from logging import getLogger, StreamHandler, Formatter, FileHandler


def init_logger(logfile="./log.txt"): # the default log file path.
    # Ex: logger = init_logger()
    # logger.info("Application started.")
    logger = getLogger(__name__) #  retrieves a logger instance named after the current module.

    # Sets the logging level to INFO. Only messages of level INFO or higher will be logged.
    logger.setLevel(logging.INFO) 

    # log to console and to file both
    stream_handler = StreamHandler() # logs messages to the console.
    file_handler = FileHandler(logfile, 'a', encoding='utf-8') #  logs messages to a file in append mode ('a'), using UTF-8 encoding.

    # Sets the logging level for both handlers to INFO.
    stream_handler.setLevel(logging.INFO) 
    file_handler.setLevel(logging.INFO)

    # Specifies the format of log messages.
    # Example output: [2024-12-02 10:00:00][INFO] - This is a log message.
    handler_format = Formatter('[%(asctime)s][%(levelname)s] - %(message)s')
    stream_handler.setFormatter(handler_format) # Assigns the format to the handlers.
    file_handler.setFormatter(handler_format) # Assigns the format to the handlers.

    # Adds both handlers to the logger and returns the logger instance.
    logger.addHandler(stream_handler) 
    logger.addHandler(file_handler)
    return logger


def load_stop_words(stopwords_file=None):  # If no file is specified, it defaults to a file named stopwords.txt in the script's directory.
    # Loads a list of stop words (common words to exclude during text processing).
    # Reads the stopwords file line by line, stripping whitespace, and appends each word to the stopwords list. then converts the list into a set for faster lookups.
    if stopwords_file is None:
        srcdir = os.path.dirname(os.path.abspath(__file__))
        stopwords_file = os.path.join(srcdir, "stopwords.txt")
    stopwords = []
    with open(stopwords_file, "r") as f:
        for l in f:
            stopwords.append(l.strip())
    return set(stopwords)


def stem(x):
    # Stems a given word using the PorterStemmer.
    return PS().stem(x)
