import os
import time
import shutil
from itertools import chain
from collections import defaultdict
import torch
from agent import VPGAgent
from environment import Environment
from prediction import Prediction
from utils.configure import set_configure
from utils.utils import load_txt, write_txt
import numpy as np

class Trainer:
    def __init__(self, params, logger, device, predictor=None, train=True):
        self.train = train
        self.params = params
        for key, val in params.items():
            setattr(self, key, val);
        self.logger = logger
        self.device = device
        self.rev_relation_vocab = {i:j for j,i in self.relation_vocab.items()}
        self.rev_comb_relation_vocab = {i:j for j,i in self.comb_rel_vocab.items()}
        # Create Environment
        self.rel_to_train_txt = load_txt(self.rel_to_train, merge_list=True) if type(self.rel_to_train)==str else []
        self.rel_to_test_txt = load_txt(self.rel_to_test, merge_list=True) if type(self.rel_to_test)==str else self.rel_to_test
        if self.transductive:
            self.rel_to_train_txt.extend(self.rel_to_test_txt)
        if train:
            self.environment_train = Environment(params, self.rel_to_train_txt, device, predictor=predictor, train=True)
        self.environment_test = Environment(params, self.rel_to_test_txt, device, train=False)
        self.agent = VPGAgent(params, device, num_relations=len(self.rel_to_train_txt))
        self.agent_test = VPGAgent(params, device, train=False)

    def Train(self):
        self.record = defaultdict(list)
        self.total_counter = 0
        best_result = 0
        for episode in self.environment_train.run_episodes():
            self.total_counter += 1
            self.agent.reset(episode.query_relation, episode.end_concepts)
            prev_relation = torch.ones(self.batch_size*self.num_rollouts, dtype=torch.int64, device=self.device)*self.relation_vocab['START']
            concept_path, rel_path = [episode.current_concepts], []
            time1 = time.time()
            for i in range(self.path_length):
                scores, idx = self.agent.decide(prev_relation, episode.current_concepts, episode.next_relations, episode.next_concepts)
                episode.step(idx)
                concept_path.append(episode.current_concepts); rel_path.append(episode.chosen_relations)
                prev_relation = episode.chosen_relations
            concept_path, rel_path = torch.stack(concept_path, axis=-1), torch.stack(rel_path, axis=-1)
            time2 = time.time()
            conf, hc = episode.get_reward(concept_path, rel_path)
            arrival = episode.get_arrival(rel_path)
            rewards = (self.hc_coff * hc+ self.conf_coff * conf + arrival)/(self.hc_coff + self.conf_coff + 1)
            time3 = time.time()
            loss = self.agent.update(rewards, self.total_counter)
            time4 = time.time()
            self.record['loss'].append(loss)
            self.logger.info('Forward: {0:7.5f}s, Reward: {1:7.5f}s, Backward: {2:7.5f}s'.format(time2-time1, time3-time2, time4-time3))
            self.log_metrics(conf, hc, arrival, self.environment_train.cur_rel, loss)
            if self.total_counter % self.save_every == 0 and len(self.rel_to_train_txt)>0:
                with torch.no_grad(): self.Test(eval_mode=True)
                _, _, _, _, mrr = predictor.predict(kg_valid=True)
                if mrr>best_result:
                    best_result = mrr; self.save_model()
                    self.logger.info(f'Stored Model at Epoch {self.total_counter}, MRR: {mrr}')
               # else:
                #    self.load_model()
                self.train = True
            if self.total_counter % self.base_iterations == 0 and self.total_counter != 0:
                self.environment_train.change_target_relation()
            if self.total_counter // self.base_iterations == len(self.rel_to_train_txt)*self.base_rounds:
                break
        # self.save_model()
        del self.agent
        return self.record

    def Test(self, eval_mode=False, beam=True):
        self.train = False
        self.test_counter = 0
        if eval_mode:
            self.agent_test.model.load_state_dict(self.agent.model.state_dict())
        else:
            self.load_model()
        self.prepare_metapath_folder()
        rel_count, early_stop = 0, 0
        for episode in self.environment_test.run_episodes():
            metapath_count = self.environment_test.get_repo_count()
            self.test_counter += 1
            self.agent_test.reset(episode.query_relation, episode.end_concepts)
            prev_relation = torch.ones(self.test_batch_size*self.test_rollouts, dtype=torch.int64, device=self.device)*self.relation_vocab['START']
            concept_path, rel_path = [episode.current_concepts], []
            beam_probs = torch.zeros((self.test_batch_size*self.test_rollouts, 1), device=self.device)
            for i in range(self.path_length):
                test_scores, idx = self.agent_test.decide(prev_relation, episode.current_concepts, episode.next_relations, episode.next_concepts)
                if beam:
                    new_scores = test_scores + beam_probs
                    if i == 0:
                        idx = torch.argsort(new_scores)[:, -self.test_rollouts:]
                        ranged_idx = torch.arange(self.test_rollouts, device=self.device).repeat(self.test_batch_size).unsqueeze(-1)%self.max_num_actions
                        idx = torch.gather(idx, dim=1, index=ranged_idx).squeeze()
                    else:
                        scores = new_scores.reshape(-1, self.test_rollouts*self.max_num_actions)  # [B, (k*max_num_actions)]
                        idx = torch.argsort(scores)[:, -self.test_rollouts:].reshape((-1))
                    path_idx, idx = torch.div(idx, self.max_num_actions, rounding_mode='floor'), idx%self.max_num_actions
                    path_idx += torch.arange(self.test_batch_size, device=self.device).repeat_interleave(self.test_rollouts)*self.test_rollouts
                    episode.beam_update(path_idx)
                    concept_path = [i[path_idx] for i in concept_path]
                    if i>0:
                        rel_path = [i[path_idx] for i in rel_path]
                    self.agent_test.update_beam_memory(path_idx)
                    beam_probs = new_scores[path_idx, idx].unsqueeze(-1)
                    idx = idx.unsqueeze(-1)
                episode.step(idx)
                concept_path.append(episode.current_concepts); rel_path.append(episode.chosen_relations)
                prev_relation = episode.chosen_relations
            concept_path, rel_path = torch.stack(concept_path, axis=-1), torch.stack(rel_path, axis=-1)
            conf, hc = episode.get_reward(concept_path, rel_path)
            arrival = episode.get_arrival(rel_path)
            self.log_metrics(conf, hc, arrival, self.environment_test.cur_rel)
            if metapath_count == self.environment_test.get_repo_count():
                early_stop += 1
            if (self.test_counter % self.test_iterations == 0 and self.test_counter != 0) or early_stop>=self.params['tolerance']:
                self.store_metapaths(self.environment_test.cur_rel)
                self.environment_test.change_target_relation()
                rel_count += 1; self.test_counter = self.test_iterations * rel_count; early_stop = 0
            if self.test_counter // self.test_iterations == len(self.rel_to_test_txt):
                break
        self.environment_test.reset()
    
    def log_metrics(self, conf, hc, arrival, rel, loss=None):
        arrival = arrival.mean().cpu().item()
        global_conf, global_hc = conf.mean().cpu().item(), hc.mean().cpu().item()
        conf_mean = 0 if len(conf[conf>0])==0 else conf[conf>0].mean().cpu().item()
        hc_mean = 0 if len(hc[hc>0])==0 else hc[hc>0].mean().cpu().item()
        if self.train:
            self.record['conf'].append(global_conf)
            self.record['hc'].append(global_hc)
            self.record['arrival'].append(arrival)
            self.logger.info(("Counter: {0:4d}, Mean Conf: {1:7.5f}, Global Conf: {2:7.5f}, Mean HC: {3:7.5f}, Global HC: {4:7.5f}, " +\
                              "Arrival: {5:7.3f}, Loss: {6:7.4f}, Len Repo: {7:4d}").
            format(self.total_counter, conf_mean, global_conf, hc_mean, global_hc, arrival, loss, len(self.environment_train.repos[rel])))
        else:
            self.logger.info(("Counter: {0:4d}, Mean Conf: {1:7.5f}, Global Conf: {2:7.5f}, Mean HC: {3:7.5f}, Global HC: {4:7.5f}, " +\
                              "Arrival: {5:7.3f}, Len Repo: {6:4d}").
            format(self.test_counter, conf_mean, global_conf, hc_mean, global_hc, arrival, len(self.environment_test.repos[rel])))

    def store_metapaths(self, rel):
        rel_dict = {mp: (conf, hc) for mp, (conf, hc) in self.environment_test.repos[rel].items() if conf > 0 or hc > 0}
        mp_list = [[[self.rev_comb_relation_vocab[i] for i in mp], [conf], [hc]] for mp, (conf, hc) in rel_dict.items()]
        write_txt(list(chain.from_iterable(mp_list)), self.metapath_dir+self.rev_relation_vocab[rel])

    def prepare_metapath_folder(self):
        if os.path.exists(self.metapath_dir):
            shutil.rmtree(self.metapath_dir)
        os.mkdir(self.metapath_dir)
    
    def save_model(self):
        torch.save(self.agent.model.state_dict(), self.model_save_dir+'model.pkt')

    def load_model(self):
        if self.train:
            self.agent.model.load_state_dict(torch.load(self.model_save_dir+'model.pkt', map_location=self.device))
            self.agent.model.train()
        else:
            self.agent_test.model.load_state_dict(torch.load(self.model_save_dir+'model.pkt', map_location=self.device))
            self.agent_test.model.eval()

if __name__ == '__main__':
    train, test = True, True
    # train, test = False, True
    # train, test = False, False
    # data_name = 'yago'
    # data_name = 'nell'
    data_name = 'dbpedia'
    candidate_per_rels = {'yago': ['isCitizenOf', 'diedIn', 'graduatedFrom'],
                          'nell': ['teamplaysagainstteam', 'competeswith', 'worksfor']}
    logger = None
    options, logger = set_configure(data_name, candidate_per_rels, logger, train, test)
    # device = torch.device("cpu")
    device = torch.device(f"cuda:{options['gpu']}" if torch.cuda.is_available() else "cpu")
    predictor = Prediction(options, logger, device)
    if options['eval_mode'] == 'KGC':
        trainer = Trainer(options, logger, device, predictor, train)
        if train:
            record = trainer.Train()
        hit_1s, hit_3s, hit_5s, hit_10s, mrrs = [], [], [], [], []
        for _ in range(options['eval_times']): # For KGC, Train once, Test few times
            if test:
                with torch.no_grad(): trainer.Test()
            hit_1, hit_3, hit_5, hit_10, mrr = predictor.predict()
            hit_1s.append(hit_1); hit_3s.append(hit_3); hit_5s.append(hit_5); hit_10s.append(hit_10); mrrs.append(mrr)
            logger.info(f'Hits@1: {hit_1}, Hits@3: {hit_3}, Hits@5: {hit_5}, Hits@10: {hit_10}, MRR: {mrr}')
        hit_1, hit_3, hit_5, hit_10, mrr = np.mean(hit_1s), np.mean(hit_3s), np.mean(hit_5s), np.mean(hit_10s), np.mean(mrrs)
        logger.info(f'Mean - Hit@1: {hit_1}, Hit@3: {hit_3}, Hit@5: {hit_5}, Hit@10: {hit_10}, MRR: {mrr}')
    
    if options['eval_mode'] == 'LP':
        aps, AUCs = [], []
        for _ in range(options['eval_times']): # For LP, (Train & Test) few times
            trainer = Trainer(options, logger, device, predictor, train)
            if train:
                record = trainer.Train()
            if test:
                with torch.no_grad(): trainer.Test()
            ap, AUC = predictor.predict()
            aps.append(ap); AUCs.append(AUC); logger.info(f'AP: {ap}, AUC: {AUC}')
            options, logger = set_configure(data_name, candidate_per_rels, logger, train, test)
            predictor = Prediction(options, logger, device)
        logger.info(f'Mean - AP: {np.mean(aps)}, AUC: {np.mean(AUCs)}')
