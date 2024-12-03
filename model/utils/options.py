import time, json
import argparse

def general_config():
    parser = argparse.ArgumentParser()
    parser.add_argument('--target_rel_index', default=-1, type=int) # Multi-relation model if set to -1
    parser.add_argument('--gpu', default='0', type=str)
    parser.add_argument('--reward_device', default='auto', type=str, help='cpu/gpu/auto')
    parser.add_argument('--transductive', default=False, action='store_true')
    parser.add_argument('--path_length', default=4, type=int, help='Number of Relations in the Metapath')
    parser.add_argument('--base_iterations', default=0, type=int)
    parser.add_argument('--hc_coff', default=5.0, type=float)
    parser.add_argument('--conf_coff', default=1.0, type=float)
    parser.add_argument('--global_coverage', default=False, action='store_true')
    parser.add_argument('--positive_reward', default=1.0, type=float)
    parser.add_argument('--negative_reward', default=0.0, type=float)
    ### Learning Parameters
    parser.add_argument('--load_concept_embeddings', default=False, action='store_true')
    parser.add_argument('--load_relation_embeddings', default=True, action='store_false')
    parser.add_argument('--train_concept_embeddings', default=True, action='store_false')
    parser.add_argument('--train_relation_embeddings', default=False, action='store_true')
    parser.add_argument('--use_concept_embeddings', default=True, action='store_false')
    ### Training Parameters
    parser.add_argument('--baseline_rate', default=0.05, type=float)
    parser.add_argument('--gamma', default=0.99, type=float)
    parser.add_argument('--grad_clip_norm', default=5, type=int)
    parser.add_argument('--beta', default=-1, type=float)
    parser.add_argument('--hidden_size', default=0, type=int)
    ### Prediction Parameters
    parser.add_argument('--pool', default='max', type=str, help='max/mean/sum')
    parser.add_argument('--lp_pool', default='sum', type=str, help='max/sum/bool')
    parser.add_argument('--aggregation', default='bool', type=str, help='bool/count')
    ### Others
    parser.add_argument('--seed', default=42, type=int)
    parser.add_argument('--eval_times', default=5, type=int)
    parser.add_argument('--pred_times', default=5, type=int)
    options = vars(parser.parse_args())
    
    ### Agent and Policy Network
    options['agent_type'] = 'vpg'
    # options['agent_type'] = 'ppo'
    options['encoder_type'] = 'LSTMEncoder'
    # options['encoder_type'] = 'TransformerEncoder'
    options['decoder_type'] = 'LinearDecoder'
    
    ### Deep Learning Parameters
    options['embedding_size'] = 64
    options['sliding_window'] = options['path_length'] # for Transformer if used
    ### PPO (if used) Parameters
    options['lambda'] = 0.90
    options['replayer_size'] = 2400
    options['replay_mini'] = 800
    options['update_times'] = 1
    options['clip_ratio'] = 0.2
    return options

def user_config(data_name, candidate_per_rels):
    options = general_config()
    options['data_name'] = data_name
    target_rel = 'multi' if options['target_rel_index']==-1 else candidate_per_rels[data_name][options['target_rel_index']]
    options['target_rel'] = target_rel
    options['eval_mode'] = 'KGC' if target_rel=='multi' else 'LP'
    if target_rel!='multi': options['eval_times'] = 5
    ### Directories
    options['data_input_dir'] = "../datasets/data/{}/".format(data_name)
    options['instance_input_dir'] = options['data_input_dir'] + 'instance/'
    options['schema_input_dir'] = options['data_input_dir'] + 'schema/'
    options['repo_buffer_dir'] = options['data_input_dir'] + 'buffer/'
    options['link_prediction_dir'] = options['data_input_dir'] + 'link_prediction/'
    options['base_output_dir'] = "../output/{}/".format(data_name)
    options['base_metapath_dir'] = "../metapaths/{}/".format(data_name)
    ### Schema Files
    options['schema_graph'] = options['schema_input_dir']+'schema_graph.txt'
    options['schema_fact_learn'] = options['schema_input_dir']+'schema_facts_learn.txt'
    options['relation_vocab'] = json.load(open(options['schema_input_dir'] + 'relation_vocab.json'))
    options['concept_vocab'] = json.load(open(options['schema_input_dir'] + 'concept_vocab.json'))
    options['rel_to_train'] = options['schema_input_dir']+'train_relations.txt' if target_rel=='multi' else None
    options['rel_to_test'] = options['schema_input_dir']+'test_relations.txt' if target_rel=='multi' else [target_rel]
    options['pretrained_embeddings_action'] = options['schema_input_dir'] + 'relation_embeddings.txt'
    options['pretrained_embeddings_concept'] = options['schema_input_dir'] + 'concept_embeddings.txt'
    ### Instance Files
    options['instance_graph'] = options['instance_input_dir']+'instance_graph.txt'
    options['entity_vocab'] = json.load(open(options['instance_input_dir'] + 'entity_vocab.json'))
    options['rel_dict'] = json.load(open(options['instance_input_dir'] + 'comb_rel_dict.json'))
    options['comb_rel_vocab'] = json.load(open(options['instance_input_dir'] + 'comb_rel_vocab.json'))
    options['entity_type'] = options['instance_input_dir']+'entity_type.txt'
    ### Schema-Instance Mapping
    options['comb_rel_decompose'] = json.load(open(options['data_input_dir'] + 'comb_rel_decompose.json'))

    if 'yago' in data_name:
        ### Walk Parameters
        options["max_num_actions"] = 140
        ### Training Parameters
        options['base_iterations'] = 500 if options['base_iterations'] == 0 else options['base_iterations']
        options['base_rounds'] = 5 if target_rel == 'multi' else 1
        options['batch_size'] = 20 if target_rel == 'multi' else 80
        options["num_rollouts"] = 40 if target_rel == 'multi' else 10
        ### Learning Parameters
        options['beta'] = 0.05 if options['beta'] == -1 else options['beta']
        if target_rel != 'multi': options['beta'] = 0.1
        options['beta_dacay_rate'] = 0.8 if target_rel == 'multi' else 0.9
        options['beta_dacay_time'] = 500 if target_rel == 'multi' else 200
        options["learning_rate"] = 4e-5 if target_rel == 'multi' else 5e-4
        options["save_every"] = 3000
        ### Deep Parameters
        options['hidden_size'] = 200 if options['hidden_size'] == 0 else options['hidden_size']
        ### Testing Parameters
        options['test_iterations'] = 20 if target_rel == 'multi' else 20
        options['test_batch_size'] = 10
        options["test_rollouts"] = 400 if target_rel == 'multi' else 400
        options["pred_ratio"] = 0.1
        options["tolerance"] = 3

    if 'dbpedia' in data_name:
        ### Walk Parameters
        options["max_num_actions"] = 550
        ### Training Parameters
        options['base_iterations'] = 500 if options['base_iterations'] == 0 else options['base_iterations']
        options['base_rounds'] = 5 if target_rel == 'multi' else 1
        options['batch_size'] = 20
        options["num_rollouts"] = 40
        ### Learning Parameters
        options['beta'] = 0.05 if options['beta'] == -1 else options['beta']
        options['beta_dacay_rate'] = 0.8
        options['beta_dacay_time'] = 500
        options["learning_rate"] = 4e-5
        options["save_every"] = 2000
        ### Deep Parameters
        options['hidden_size'] = 200 if options['hidden_size'] == 0 else options['hidden_size']
        ### Testing Parameters
        options['test_iterations'] = 20
        options['test_batch_size'] = 10
        options["test_rollouts"] = 400
        options["pred_ratio"] = 0.05
        options["tolerance"] = 3
    
    if 'nell' in data_name:
        ### Walk Parameters
        options["max_num_actions"] = 670
        ### Training Parameters
        options['base_iterations'] = 500 if options['base_iterations'] == 0 else options['base_iterations']
        options['base_rounds'] = 5 if target_rel == 'multi' else 1
        options['batch_size'] = 20 if target_rel == 'multi' else 40
        options["num_rollouts"] = 40 if target_rel == 'multi' else 40
        ### Learning Parameters
        options['beta'] = 0.05 if options['beta'] == -1 else options['beta']
        options['beta_dacay_rate'] = 0.98 if target_rel == 'multi' else 0.9
        options['beta_dacay_time'] = 500 if target_rel == 'multi' else 200
        options["learning_rate"] = 4e-5 if target_rel == 'multi' else 5e-4
        options["save_every"] = 2000
        ### Deep Parameters
        options['hidden_size'] = 200 if options['hidden_size'] == 0 else options['hidden_size']
        ### Testing Parameters
        options['test_iterations'] = 50 if target_rel == 'multi' else 50
        options['test_batch_size'] = 10 if target_rel == 'multi' else 5
        options["test_rollouts"] = 500 if target_rel == 'multi' else 800
        options["pred_ratio"] = 0.05
        options["tolerance"] = 10

    strTime = time.strftime("%Y-%m-%d-%H-%M-%S", time.localtime(int(time.time())))
    options['metapath_dir'] = options['base_metapath_dir'] + '{}_{}/'.format(target_rel, strTime)
    options['output_dir'] = options['base_output_dir']+ '{}_{}_{}_{}_{}_{}/'.format(target_rel, strTime, 
                                                                                    options['agent_type'], options['encoder_type'],
                                                                                    options['beta'], options['base_iterations'])
    options['model_save_dir'] = options['output_dir'] + 'model/'
    options['log_file_name'] = options['output_dir'] + 'log.txt'
    return options
