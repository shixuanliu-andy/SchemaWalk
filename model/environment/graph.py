from collections import defaultdict
import numpy as np
import torch
import csv

class Graph:
    def __init__(self, params, target_rels, device):
        self.schema_dir = params['schema_graph']
        self.relation_vocab = params['relation_vocab']
        self.concept_vocab = params['concept_vocab']
        self.cPAD, self.rPAD = self.concept_vocab['PAD'], self.relation_vocab['PAD']
        self.max_num_actions = params['max_num_actions']
        self.target_rels = target_rels
        self.device = device
        self.load_graph()
        self.gen_graph_array()

    def load_graph(self):
        self.base_store = defaultdict(list)
        with open(self.schema_dir) as f:
            for e1, r, e2 in csv.reader(f, delimiter='\t'):
                e1, r, e2 = self.concept_vocab[e1], self.relation_vocab[r], self.concept_vocab[e2]
                self.base_store[e1].append((r, e2))
        self.cur_rel = self.target_rels[0]

    def gen_graph_array(self):
        self.base_array_store = np.ones((len(self.concept_vocab), self.max_num_actions, 2), dtype=int)
        self.base_array_store[:, :, 0] *= self.cPAD
        self.base_array_store[:, :, 1] *= self.rPAD
        self.masked_array_store = dict()
        for e1 in self.base_store:
            action_index = 1
            self.base_array_store[e1, 0, 0] = e1
            self.base_array_store[e1, 0, 1] = self.relation_vocab['STAY']
            for r, e2 in self.base_store[e1]:
                if r in self.target_rels:
                    if r not in self.masked_array_store:
                        self.masked_array_store[r] = np.ones((len(self.concept_vocab), self.max_num_actions, 2), dtype=int)
                    self.masked_array_store[r][e1, action_index, :] = 0
                if action_index == self.max_num_actions:
                    break
                self.base_array_store[e1, action_index, 0] = e2
                self.base_array_store[e1, action_index, 1] = r
                action_index += 1
        self.base_array_store = torch.from_numpy(self.base_array_store).to(self.device)
        self.masked_array_store = {i:torch.from_numpy(j).to(self.device) for i,j in self.masked_array_store.items()}
        self.array_store = self.base_array_store*self.masked_array_store[self.cur_rel]

    def change_target_relation(self, rel_index):
        # Masks edges for current relation
        self.array_store = self.base_array_store*self.masked_array_store[self.target_rels[rel_index]]

    def return_next_actions(self, current_concepts, start_concepts, end_concepts, qrs, all_concepts, last_step, num_rollouts):
        # current_entities: [batch_size*roll_outs, num_actions, 2]
        returns = self.array_store[current_concepts, :, :].clone()
        mask_start = (current_concepts==start_concepts)
        for i in range(current_concepts.size(0)):
            if mask_start[i]:
                mask = (returns[i, :, 0]==end_concepts[i]) & (returns[i, :, 1]==qrs[i])
                returns[i, mask, 0], returns[i, mask, 1] = self.cPAD, self.rPAD
            if last_step:
                mask = torch.isin(returns[i, :, 0], all_concepts[i//num_rollouts]) & (returns[i, :, 0]!=end_concepts[i])
                returns[i, mask, 0], returns[i, mask, 1] = self.cPAD, self.rPAD
        return returns
