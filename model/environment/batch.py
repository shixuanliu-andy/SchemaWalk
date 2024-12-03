import numpy as np
from collections import defaultdict
import csv
import torch

class BatchLoader:
    def __init__(self, params, target_rels, batch_size, device):
        self.schema_dir = params['schema_graph']
        self.schema_fact_learn = params['schema_fact_learn']
        self.concept_vocab = params['concept_vocab']
        self.relation_vocab = params['relation_vocab']
        self.target_rels_str = target_rels
        self.target_rels = [self.relation_vocab[i] for i in target_rels]
        self.batch_size = batch_size
        self.device = device
        self.load_schema_triples()

    def load_schema_triples(self):
        self.store_all_correct = defaultdict(set)
        self.store_all_rel = defaultdict(list)
        with open(self.schema_fact_learn) as f:
            for e1, r, e2 in csv.reader(f, delimiter='\t'):
                e1, r, e2 = self.concept_vocab[e1], self.relation_vocab[r], self.concept_vocab[e2]
                if r in self.target_rels:
                    self.store_all_rel[r].append([e1,r,e2])
                self.store_all_correct[(e1, r)].add(e2)
        self.store_all_correct = {i:torch.tensor(list(j)).to(self.device) for i,j in self.store_all_correct.items()}
        self.rel_triple_num = {i:len(j) for i,j in self.store_all_rel.items()}
        self.target_rels = list(sorted(self.target_rels, key=lambda x:self.rel_triple_num[x], reverse=True))
        self.store_numpy = np.array(self.store_all_rel[self.target_rels[0]])
        self.store = torch.from_numpy(self.store_numpy).to(self.device)

    def change_target_relation(self, rel_index):
        self.store_numpy = np.array(self.store_all_rel[self.target_rels[rel_index]])
        self.store = torch.from_numpy(self.store_numpy).to(self.device)

    def yield_next_batch(self):
        while True:
            batch_idx = np.random.randint(0, self.store.shape[0], size=self.batch_size)
            batch, batch_np = self.store[batch_idx, :], self.store_numpy[batch_idx, :]
            c1_s, r_s, c2_s = batch[:, 0], batch[:, 1], batch[:, 2]
            all_concepts = [self.store_all_correct[(batch_np[i, 0], batch_np[i, 1])] for i in range(self.batch_size)]
            yield c1_s, r_s, c2_s, all_concepts