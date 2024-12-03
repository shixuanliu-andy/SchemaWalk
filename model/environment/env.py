import os
import json
import time
from tqdm import tqdm
from itertools import chain
from collections import defaultdict
import numpy as np
from scipy.sparse import csr_matrix, identity
import torch
from .batch import BatchLoader
from .graph import Graph

class Environment:
    def __init__(self, params, target_rels, device, predictor=None, train=True, use_buffer=False):
        # Load Params
        self.relation_vocab = params['relation_vocab']
        self.rev_relation_vocab = {j:i for i,j in params['relation_vocab'].items()}
        self.batch_size = params['batch_size'] if train else params['test_batch_size']
        self.num_rollouts = params['num_rollouts'] if train else params['test_rollouts']
        self.path_len = params['path_length']
        self.pos_reward, self.neg_reward = params['positive_reward'], params['negative_reward']
        self.comb_rel_mapping = {tuple(j):int(i) for i,j in params['comb_rel_decompose'].items()}
        self.get_rel_count(params['rel_dict'])
        self.repo_buffer_dir = params['repo_buffer_dir']
        self.transductive = params['transductive']
        # Create Sparse Matrices Mapping
        self.N = len(params['entity_vocab'])
        self.comb_rel_dict = {int(i):j for i,j in params['rel_dict'].items()}
        # Init Components
        self.device = device
        self.use_buffer = use_buffer
        self.reward_device = params['reward_device']
        self.global_coverage = params['global_coverage']
        self.aux_rels = torch.arange(4, device=device)
        self.batchloader = BatchLoader(params, target_rels, self.batch_size, device)
        self.graph = Graph(params, self.batchloader.target_rels, device)
        self.target_rels, self.target_rels_str = self.batchloader.target_rels, target_rels
        self.train = train
        self.predictor = predictor
        self.reset()
    
    def reset(self):
        self.repos = defaultdict(dict)
        if self.transductive and self.train: # Filter Instance Testset if Transductive
            if len(self.target_rels)>1:
                for rel, testset in self.predictor.dataset.items():
                    self.filter_testset(rel, testset)
            else:
                self.filter_testset(self.target_rels[0], self.predictor.testset)
        if self.use_buffer and self.train:
            buffer_files = set(os.listdir(self.repo_buffer_dir)) & set(self.target_rels_str)
            for f in buffer_files:
                repo_dict = {tuple(map(int, i.split('_'))):j for i,j in json.load(open(self.repo_buffer_dir+f)).items() if len(i)>0}
                self.repos[self.relation_vocab[f]] = repo_dict
        self.change_target_relation(0)

    def filter_testset(self, rel, testset):
        comb_rels = [j for (rel_,c1,c2),j in self.comb_rel_mapping.items() if rel_==rel]
        for id in comb_rels:
            self.comb_rel_dict[id] = list(set([tuple(i) for i in self.comb_rel_dict[id]])-set(testset))

    def get_rel_count(self, comb_rel_dict):
        self.rel_pairs = defaultdict(list)
        for (rel, c1, c2), comb_rel in self.comb_rel_mapping.items():
            self.rel_pairs[rel].extend([tuple(i) for i in comb_rel_dict[str(comb_rel)]])
        self.rel_count = {i:len(set(j)) for i,j in self.rel_pairs.items()}

    def run_episodes(self):
        params_walk = self.batch_size, self.num_rollouts, self.path_len
        params_reward = self.pos_reward, self.neg_reward, self.aux_rels, self.comb_rel_dict, self.N, self.comb_rel_mapping, self.rel_count, self.global_coverage
        for data in self.batchloader.yield_next_batch():
            yield Episode(self.graph, data, params_walk, params_reward, self.device, self.reward_device, self.repos)
            
    def change_target_relation(self, index=None):
        if self.use_buffer and index is None:
            repo_dict = {'_'.join(map(str,i)):j for i,j in self.repos[self.cur_rel].items()}
            json.dump(repo_dict, open(self.repo_buffer_dir+self.rev_relation_vocab[self.cur_rel], "w"))
        self.rel_index = (self.rel_index+1)%len(self.target_rels) if index is None else index
        self.cur_rel = self.target_rels[self.rel_index]
        self.graph.change_target_relation(self.rel_index)
        self.batchloader.change_target_relation(self.rel_index)
        return self.graph.target_rels[self.rel_index]
    
    def get_repo_count(self):
        return len(self.repos[self.cur_rel])
    
class Episode:
    def __init__(self, graph, data, params_walk, params_reward, device, reward_device, repos):
        self.graph = graph
        # Unpack Params
        self.batch_size, self.num_rollouts, self.path_len = params_walk
        self.sample_size = self.batch_size*self.num_rollouts
        self.pos_reward, self.neg_reward, self.aux_rels, self.comb_rel_dict, self.N, self.comb_rel_mapping, self.rel_count, self.global_coverage = params_reward
        # Init Data
        start_concepts, query_relation, end_concepts, self.all_concepts = data
        self.start_concepts = torch.repeat_interleave(start_concepts, self.num_rollouts)
        self.current_concepts = self.start_concepts.clone()
        self.end_concepts = torch.repeat_interleave(end_concepts, self.num_rollouts)
        self.query_relation_list = query_relation.cpu().tolist()
        self.query_relation = torch.repeat_interleave(query_relation, self.num_rollouts)
        self.comb_query_relation = torch.stack([query_relation, start_concepts, end_concepts], axis=-1).cpu()
        # Init Walk
        self.current_hop = 0
        self.device = device
        self.reward_device = reward_device
        self.repos = repos
        self.csr_dict_pool = {}
        self.update_state_info()

    def step(self, action):
        self.current_hop += 1
        self.current_concepts = torch.gather(self.next_concepts, dim=1, index=action).squeeze()
        self.chosen_relations = torch.gather(self.next_relations, dim=1, index=action).squeeze()
        self.update_state_info()
    
    def beam_update(self, path_idx):
        self.current_concepts = self.current_concepts[path_idx]
        self.next_concepts = self.next_concepts[path_idx, :]
        self.next_relations = self.next_relations[path_idx, :]
        
    def update_state_info(self):
        next_actions = self.graph.return_next_actions(self.current_concepts, self.start_concepts, self.end_concepts, 
                                                      self.query_relation, self.all_concepts, self.current_hop==self.path_len-1, self.num_rollouts)
        self.next_concepts, self.next_relations = next_actions[:, :, 0], next_actions[:, :, 1]
        
    def get_reward(self, concept_paths, rel_paths):
        conf, hc = torch.zeros(self.sample_size, device=self.device), torch.zeros(self.sample_size, device=self.device)
        concept_paths, rel_paths = concept_paths.cpu(), rel_paths.cpu()
        rel_mask = ~torch.isin(rel_paths, self.aux_rels.cpu())
        concept_mask = torch.cat([torch.ones([self.sample_size, 1]).bool(), rel_mask], axis=-1)
        for ind in range(self.sample_size):
            if self.current_concepts[ind] == self.end_concepts[ind]:
                concept_path = concept_paths[ind][concept_mask[ind]].tolist()
                rel_path = rel_paths[ind][rel_mask[ind]].tolist()
                metapath = tuple([self.comb_rel_mapping[(rel_path[i],concept_path[i],concept_path[i+1])] for i in range(len(rel_path))])
                batch_ind = ind//self.num_rollouts
                target_rel = self.query_relation_list[batch_ind]
                if len(metapath) == 0:
                    conf_res, hc_res = 0, 0
                elif metapath in self.repos[target_rel]:
                    conf_res, hc_res = self.repos[target_rel][metapath]
                else:
                    comb_query_relation = self.comb_rel_mapping[tuple(self.comb_query_relation[batch_ind].tolist())]
                    conf_res, hc_res = self.get_conf_hc(comb_query_relation, metapath, self.rel_count[self.query_relation_list[batch_ind]])
                    self.repos[target_rel][metapath] = (conf_res, hc_res)
                conf[ind], hc[ind] = conf_res, hc_res
        return conf, hc

    def get_conf_hc(self, query_rel, meta_path, rel_count):
        if self.reward_device == 'auto':
            try:
                conf, hc = self.get_conf_hc_gpu_optim(query_rel, meta_path, rel_count)
            except:
                conf, hc = self.get_conf_hc_cpu(query_rel, meta_path, rel_count)
        if self.reward_device == 'gpu':
            conf, hc = self.get_conf_hc_gpu_optim(query_rel, meta_path, rel_count)
        if self.reward_device == 'cpu':
            conf, hc = self.get_conf_hc_cpu(query_rel, meta_path, rel_count)
        return conf, hc

    def get_conf_hc_gpu_optim(self, query_rel, meta_path, rel_count):
        csr_dict = dict()
        # Remap entities:
        ent_set = set(chain.from_iterable(list(chain.from_iterable([self.comb_rel_dict[rel] for rel in list(meta_path)+[query_rel]]))))
        ent_tensor, _ = torch.tensor(list(ent_set)).to(self.device).sort()
        ent_tensor = torch.stack([ent_tensor,ent_tensor])
        for rel in list(meta_path) + [query_rel]:
            if rel in self.csr_dict_pool:
                csr_index = self.csr_dict_pool[rel]
            else:
                if len(self.comb_rel_dict[rel])==0: return 0, 0
                csr_index = torch.tensor(self.comb_rel_dict[rel]).to(self.device).T
                self.csr_dict_pool[rel] = csr_index
            csr_index = torch.searchsorted(ent_tensor, csr_index)
            csr_dict[rel] = torch.sparse_coo_tensor(csr_index, torch.ones(csr_index.size(1), device=self.device),
                                                    [ent_tensor.size(1),ent_tensor.size(1)])
        result = csr_dict[meta_path[0]]
        for rel in meta_path[1:]:
            result = result.mm(csr_dict[rel])
        result_masked = result.mul(csr_dict[query_rel])
        body = result._nnz() # number of pairs that have the meta-path
        head_body = result_masked._nnz() # number of pairs that have both query comb_relation and meta-path
        if self.global_coverage:
            (conf, hc) = (0, head_body/rel_count) if body == 0 else (head_body/body, head_body/rel_count)
        else:
            (conf, hc) = (0, head_body/csr_dict[query_rel]) if body == 0 else (head_body/body, head_body/csr_dict[query_rel])
        return conf, hc
    
    def get_conf_hc_gpu(self, query_rel, meta_path, rel_count):
        csr_dict = dict()
        # Remap entities:
        ent_set = set(chain.from_iterable(list(chain.from_iterable([self.comb_rel_dict[rel] for rel in list(meta_path)+[query_rel]]))))
        ent_dict = {ent: id for id, ent in enumerate(ent_set)}
        for rel in list(meta_path) + [query_rel]:
            indices = [list(map(ent_dict.get, i)) for i in self.comb_rel_dict[rel]]
            csr_dict[rel] = torch.sparse_coo_tensor(torch.tensor(indices).T, torch.ones(len(indices)), [len(ent_dict),len(ent_dict)]).to(self.device)
        result = csr_dict[meta_path[0]]
        for rel in meta_path[1:]:
            result = result.mm(csr_dict[rel])
        result_masked = result.mul(csr_dict[query_rel])
        body = result._nnz() # number of pairs that have the meta-path
        head_body = result_masked._nnz() # number of pairs that have both query comb_relation and meta-path
        (conf, hc) = (0, head_body/rel_count) if body == 0 else (head_body/body, head_body/rel_count)
        return conf, hc
    
    def get_conf_hc_cpu(self, query_rel, meta_path, rel_count):
        csr_dict = defaultdict(csr_matrix)
        # ent_set = np.unique(list(chain.from_iterable([self.comb_rel_dict[rel] for rel in list(meta_path)+[query_rel]]))).tolist()
        # ent_dict = dict([(ent, id) for id, ent in enumerate(ent_set)]); N = len(ent_dict)
        for rel in list(meta_path) + [query_rel]:
            if len(self.comb_rel_dict[rel])==0: return 0, 0
            indices, data = np.array(self.comb_rel_dict[rel]), np.ones(len(self.comb_rel_dict[rel]))
            # indices = np.array([list(map(ent_dict.get, i)) for i in indices])
            csr_dict[rel] = csr_matrix((data, (indices[:,0], indices[:,1])), shape=(self.N, self.N))
        result = identity(self.N)
        for rel in meta_path:
            result *= csr_dict[rel]
        result_masked = result.multiply(csr_dict[query_rel])
        body = result.nnz # number of pairs that have the meta-path
        head_body = result_masked.nnz # number of pairs that have both query relation and meta-path
        if self.global_coverage:
            (conf, hc) = (0, head_body/rel_count) if body == 0 else (head_body/body, head_body/rel_count)
        else:
            (conf, hc) = (0, head_body/csr_dict[query_rel].size) if body == 0 else (head_body/body, head_body/csr_dict[query_rel].size)
        return conf, hc
    
    def get_arrival(self, rel_path):
        arrival = (self.current_concepts == self.end_concepts)
        moved_samples = ~(torch.all(torch.isin(rel_path, self.aux_rels), dim=-1))
        ans = torch.where(arrival & moved_samples, self.pos_reward, self.neg_reward)
        return ans