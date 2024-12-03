import os
import time
import random
import itertools
from collections import defaultdict
import torch
from agent import VPGAgent, Replayer
from utils.configure import set_configure
from utils.utils import load_txt, write_txt
from environment import Environment

class Tester:
    def __init__(self, params, logger, device):
        for key, val in params.items():
            setattr(self, key, val);
        self.logger = logger
        self.device = device
        self.rev_relation_vocab = {i:j for j,i in self.relation_vocab.items()}
        self.rev_comb_relation_vocab = {i:j for j,i in self.comb_rel_vocab.items()}
        # Create Environment
        self.rel_to_test_txt = load_txt(self.rel_to_test, filter_=False, single_list=True)
        self.environment_test = Environment(params, self.rel_to_test_txt, device, train=False)
        self.agent = VPGAgent(params, device)

    def test(self):
        self.total_counter = 0
        self.environment = self.environment_test
        self.load_model()
        for episode in self.environment.run_episodes():
            self.total_counter += 1
            self.agent.reset(episode.query_relation, episode.end_concepts)
            prev_relation = torch.ones(self.batch_size*self.num_rollouts, dtype=torch.int64, device=self.device)*self.relation_vocab['START']
            concept_path, rel_path = [episode.current_concepts], []
            for i in range(self.path_length):
                scores, idx = self.agent.decide(prev_relation, episode.current_concepts, episode.next_relations, episode.next_concepts)
                episode.step(idx)
                concept_path.append(episode.current_concepts)
                rel_path.append(episode.chosen_relations)
                prev_relation = episode.chosen_relations
            concept_path, rel_path = torch.stack(concept_path, axis=-1), torch.stack(rel_path, axis=-1)
            conf, hc = episode.get_reward(concept_path, rel_path)
            arrival = episode.get_arrival(rel_path)
            self.log_metrics(conf, hc, arrival, self.environment.cur_rel)
            if self.total_counter % self.test_iterations == 0 and self.total_counter != 0:
                self.store_metapaths(self.environment.cur_rel)
                self.environment.change_target_relation()
            if self.total_counter // self.test_iterations == len(self.rel_to_test_txt):
                break
        return

    def store_metapaths(self, rel):
        if not os.path.exists(self.metapath_dir):
            os.mkdir(self.metapath_dir)
        rel_dict = self.environment.repos[rel]
        rel_dict = {mp: (conf, hc) for mp, (conf, hc) in rel_dict.items() if conf > 0 or hc > 0}
        mp_list = [[[self.rev_comb_relation_vocab[i] for i in mp], [conf], [hc]] for mp, (conf, hc) in rel_dict.items()]
        mp_list = list(itertools.chain.from_iterable(mp_list))
        write_txt(mp_list, self.metapath_dir+self.rev_relation_vocab[rel])
    
    def log_metrics(self, conf, hc, arrival, rel):
        arrival = arrival.mean().cpu().item()
        global_conf, global_hc = conf.mean().cpu().item(), hc.mean().cpu().item()
        conf_mean = 0 if len(conf[conf>0])==0 else conf[conf>0].mean().cpu().item()
        hc_mean = 0 if len(hc[hc>0])==0 else hc[hc>0].mean().cpu().item()
        self.logger.info(("Counter: {0:4d}, Mean Conf: {1:7.5f}, Global Conf: {2:7.5f}, Mean HC: {3:7.5f}, Global HC: {4:7.5f}, " +\
                          "Arrival: {5:7.3f}, Len Repo: {6:4d}").
                         format(self.total_counter, conf_mean, global_conf, hc_mean, global_hc, arrival, len(self.environment.repos[rel])))

    def save_model(self):
        torch.save(self.agent.model.state_dict(), self.model_save_dir + 'model.pkt')

    def load_model(self):
        self.agent.model.load_state_dict(torch.load(self.model_save_dir + 'model.pkt', map_location=self.device))


if __name__ == '__main__':
    data_name = 'yago' # nell or chembio
    target_rel = None
    logger = None
    options, logger = set_configure(data_name, target_rel, logger, train=False)
    device = torch.device(f"cuda:{options['gpu']}" if torch.cuda.is_available() else "cpu")
    # device = torch.device("cpu")
    trainer = Tester(options, logger, device)
    trainer.test()