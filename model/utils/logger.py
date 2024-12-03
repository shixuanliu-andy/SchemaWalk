from __future__ import absolute_import
from __future__ import division
import sys, logging

def setlogger(options):
    logger = logging.getLogger()
    logging.basicConfig(stream=sys.stdout, level=logging.DEBUG)
    logger.setLevel(logging.INFO)
    fmt = logging.Formatter('%(asctime)s: [ %(message)s ]',
                            '%m/%d/%Y %I:%M:%S %p')
    console = logging.StreamHandler()
    console.setFormatter(fmt)
    logger.addHandler(console)
    logfile = logging.FileHandler(options['log_file_name'], 'w')
    logfile.setFormatter(fmt)
    logger.addHandler(logfile)
    return logger