import os
import sys
import logging
import random
import torch
from pprint import pprint
from .options import user_config

def set_configure(data_name, candidate_per_rels, logger, train=True, test=True, use_logger=True):
    options = user_config(data_name, candidate_per_rels)
    target_rel = options['target_rel']
    if train is True and test is False:
        raise Exception('If train, test is required')
    if train:
        os.makedirs(options['output_dir'])
        os.mkdir(options['model_save_dir'])
        options_print = {u:v for u,v in options.items() if type(v) is not dict}
        pprint(options_print, stream=open(options['output_dir']+'/config.txt', 'w'))
    if not train and test: # Load last model in dirs
        model_dirs = sorted([i for i in os.listdir(options['base_output_dir']) if target_rel in i and len(os.listdir(options['base_output_dir']+i+'/model/'))!=0])
        last_dir = options['base_output_dir'] + model_dirs[-1]
        options['model_save_dir'] = last_dir + '/model/'
        options['log_file_name'] = last_dir + '/test_log.txt'
    if not test: # Load last metapath dir
        mp_dirs = sorted([i for i in os.listdir(options['base_metapath_dir']) if target_rel in i and len(os.listdir(options['base_metapath_dir']+i))!=0])
        options['metapath_dir'] = options['base_metapath_dir'] + mp_dirs[-1]
        options['log_file_name'] = options['metapath_dir'] + '/prediction_log.txt'
    if use_logger:
        logger = set_logger(options, logger)
    set_seed(options['seed'])
    return options, logger

def set_logger(options, existing_logger=None):
    if existing_logger is None:
        logger = logging.getLogger()
    else:
        logger = existing_logger
        for hdlr in logger.handlers[:]:
            logger.removeHandler(hdlr)
    logging.basicConfig(stream=sys.stdout, level=logging.INFO)
    logger_format = logging.Formatter('%(asctime)s: [ %(message)s ]', '%m/%d/%Y %I:%M:%S %p')
    console = logging.StreamHandler(); console.setFormatter(logger_format); logger.addHandler(console)
    filehdlr = logging.FileHandler(options['log_file_name'], 'w'); filehdlr.setFormatter(logger_format); logger.addHandler(filehdlr)
    return logger

def set_seed(seed):
    random.seed(seed)
    torch.manual_seed(seed)