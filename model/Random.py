import os
import shutil
from tqdm import tqdm
import torch
from itertools import chain
from utils.configure import set_configure
from utils.utils import write_txt, load_txt
from environment import Environment
from prediction import Prediction
import numpy as np

class RandomWalk:
    def __init__(self, params, logger, device):
        for key, val in params.items():
            setattr(self, key, val);
        self.logger = logger
        self.rev_relation_vocab = {i:j for j,i in self.relation_vocab.items()}
        self.rev_comb_relation_vocab = {i:j for j,i in self.comb_rel_vocab.items()}
        # Create Environment
        self.rel_to_test_txt = load_txt(self.rel_to_test, merge_list=True)
        self.environment_test = Environment(params, self.rel_to_test_txt, device, train=False)

    def walk(self):
        self.total_counter = 0
        self.prepare_metapath_folder()
        for episode in self.environment_test.run_episodes():
            self.total_counter += 1
            concept_path, rel_path = [episode.current_concepts], []
            for i in range(self.path_length):
                prelim_scores = torch.ones_like(episode.next_relations)
                mask = (episode.next_relations == torch.ones_like(episode.next_relations)*self.relation_vocab['PAD'])
                scores = torch.where(mask, torch.ones_like(prelim_scores)*-99999.0, prelim_scores)  # [B, MAX_NUM_ACTIONS]
                idx = torch.multinomial(torch.softmax(scores, dim=1), 1)
                episode.step(idx)
                concept_path.append(episode.current_concepts); rel_path.append(episode.chosen_relations)
            concept_path, rel_path = torch.stack(concept_path, axis=-1), torch.stack(rel_path, axis=-1)
            conf, hc = episode.get_reward(concept_path, rel_path)
            arrival = episode.get_arrival(rel_path)
            self.log_metrics(conf, hc, arrival, self.environment_test.cur_rel)
            if self.total_counter % self.test_iterations == 0 and self.total_counter != 0:
                self.store_metapaths(self.environment_test.cur_rel)
                self.environment_test.change_target_relation()
            if self.total_counter // self.test_iterations == len(self.rel_to_test_txt):
                break
        self.environment_test.reset()

    def prepare_metapath_folder(self):
        if os.path.exists(self.metapath_dir):
            shutil.rmtree(self.metapath_dir)
        os.mkdir(self.metapath_dir)
            
    def log_metrics(self, conf, hc, arrival, rel, loss=None):
        arrival = arrival.mean().cpu().item()
        global_conf, global_hc = conf.mean().cpu().item(), hc.mean().cpu().item()
        conf_mean = 0 if len(conf[conf>0])==0 else conf[conf>0].mean().cpu().item()
        hc_mean = 0 if len(hc[hc>0])==0 else hc[hc>0].mean().cpu().item()
        self.logger.info(("Counter: {0:4d}, Mean Conf: {1:7.5f}, Global Conf: {2:7.5f}, Mean HC: {3:7.5f}, Global HC: {4:7.5f}, " +\
                          "Arrival: {5:7.3f}, Len Repo: {6:4d}").
        format(self.total_counter, conf_mean, global_conf, hc_mean, global_hc, arrival, len(self.environment_test.repos[rel])))

    def store_metapaths(self, rel):
        mp_ans = {mp: (conf, hc) for mp, (conf, hc) in self.environment_test.repos[rel].items() if conf > 0 or hc > 0}
        mp_list = [[[self.rev_comb_relation_vocab[i] for i in mp], [conf], [hc]] for mp, (conf, hc) in mp_ans.items()]
        write_txt(list(chain.from_iterable(mp_list)), self.metapath_dir+self.rev_relation_vocab[rel])

if __name__ == '__main__':
    train, test = False, True
    data_name = 'yago'
    # data_name = 'nell'
    target_rel = None
    logger = None
    options, logger = set_configure(data_name, target_rel, logger, train, test)
    # options['test_iterations'] = 20
    # device = torch.device("cpu")
    device = torch.device(f"cuda:{options['gpu']}" if torch.cuda.is_available() else "cpu")
    walker = RandomWalk(options, logger, device)
    prediction = Prediction(options, logger, device)
    if test:
        hit_1s, hit_3s, hit_5s, hit_10s, mrrs = [], [], [], [], []
        for _ in range(options['test_times']):
            with torch.no_grad():
                walker.walk()
            hit_1, hit_3, hit_5, hit_10, mrr = prediction.predict()
            hit_1s.append(hit_1); hit_3s.append(hit_3); hit_5s.append(hit_5); hit_10s.append(hit_10); mrrs.append(mrr)
            logger.info(f'Hits@1: {hit_1}, Hits@3: {hit_3}, Hits@5: {hit_5}, Hits@10: {hit_10}, MRR: {mrr}')
        hit_1, hit_3, hit_5, hit_10, mrr = np.mean(hit_1s), np.mean(hit_3s), np.mean(hit_5s), np.mean(hit_10s), np.mean(mrrs)
        logger.info(f'Mean - Hit@1: {hit_1}, Hit@3: {hit_3}, Hit@5: {hit_5}, Hit@10: {hit_10}, MRR: {mrr}')