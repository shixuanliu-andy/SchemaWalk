import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from encoder import LSTMEncoder, TransformerEncoder
from decoder import LinearDecoder

class Policy(nn.Module):
    def __init__(self, params, device, train=True):
        super(Policy, self).__init__()
        self.device = device
        self.use_concept_embeddings = params['use_concept_embeddings']
        if params['load_relation_embeddings']:
            self.relation_embedding_table = nn.Embedding.from_pretrained(torch.from_numpy(np.loadtxt(open(params['pretrained_embeddings_action']))).float()).to(device)
        else:
            self.relation_embedding_table = nn.Embedding(len(params['relation_vocab']), params['embedding_size']).to(device)
        if params['load_concept_embeddings']:
            self.concept_embedding_table = nn.Embedding.from_pretrained(torch.from_numpy(np.loadtxt(open(params['pretrained_embeddings_concept']))).float()).to(device)
        else:
            self.concept_embedding_table = nn.Embedding(len(params['concept_vocab']), params['embedding_size']).to(device)
        self.relation_embedding_table.weight.requires_grad = params['train_relation_embeddings']
        self.concept_embedding_table.weight.requires_grad = params['train_concept_embeddings']

        self.encoder_type = params['encoder_type']
        if params['encoder_type'] == 'LSTMEncoder':
            self.encoder = LSTMEncoder(params, device, train).to(device)
        if params['decoder_type'] == 'LinearDecoder':
            self.decoder = LinearDecoder(params).to(device)

    def set_targets_embedding(self, query_relation, target):
        self.query = self.relation_embedding_table(query_relation)
        self.target = self.concept_embedding_table(target)
    
    def action_encoder(self, next_relations, next_concepts):
        relation_embedding = self.relation_embedding_table(next_relations)
        concept_embedding = self.concept_embedding_table(next_concepts)
        action_embedding = torch.cat([relation_embedding, concept_embedding], axis=-1) if self.use_concept_embeddings else relation_embedding
        return action_embedding

    def forward(self, prev_relation, current_concepts, candidate_relations, candidate_concepts):
        prev_action_embedding = self.action_encoder(prev_relation, current_concepts)
        prev_concept = self.concept_embedding_table(current_concepts)
        if self.encoder_type == 'LSTMEncoder':
            output = self.encoder(prev_action_embedding)
        if self.use_concept_embeddings:
            state_query_concat = torch.cat([torch.cat([output, prev_concept], axis=-1), self.query, self.target-self.query], axis=-1)
        else:
            state_query_concat = torch.cat([output, self.query], axis=-1)
        candidate_action_embeddings = self.action_encoder(candidate_relations, candidate_concepts)
        scores, logits, action = self.decoder(state_query_concat, candidate_action_embeddings, candidate_relations)
        return scores, logits, action

class ReactiveBaseline:
    def __init__(self, lr, device):
        self.lr = lr
        self.b = torch.zeros(1, device=device)

    def update(self, target):
        self.b = torch.add((1-self.lr)*self.b, self.lr*target)

class VPGAgent:
    def __init__(self, params, device, num_relations=1, train=True):
        self.path_length = params['path_length']
        self.max_num_actions = params['max_num_actions']
        self.num_rollouts = params['num_rollouts'] if train else params['test_rollouts']
        self.batch_size = params['batch_size'] if train else params['test_batch_size']
        self.sample_size = self.batch_size * self.num_rollouts
        self.num_relations = num_relations
        # Deep Parameters
        self.hidden_size = params['hidden_size']
        self.use_concept_embeddings = params['use_concept_embeddings']
        self.encoder_type = params['encoder_type']
        # RL Parameters
        self.gamma = params['gamma']
        self.decaying_beta = params['beta']
        self.beta_dacay_time = params['beta_dacay_time']
        self.beta_dacay_rate = params['beta_dacay_rate']
        self.baseline = ReactiveBaseline(lr=params['baseline_rate'], device=device)
        self.grad_clip_norm = params['grad_clip_norm']
        # Set Model
        self.device = device
        self.model = Policy(params, device, train)
        self.optimizer = optim.Adam(self.model.parameters(), lr=params['learning_rate'])

    def decide(self, prev_relation, current_concepts, candidate_relations, candidate_concepts):
        scores, logits, action = self.model(prev_relation, current_concepts, candidate_relations, candidate_concepts)
        one_hot = torch.zeros_like(logits).scatter(1, action, 1)
        loss = -torch.sum(torch.mul(logits, one_hot), dim=1)
        self.all_logits.append(logits)
        self.all_loss.append(loss)
        return scores, action
    
    def reset(self, query_relation, target):
        self.all_loss, self.all_logits = [], []
        self.model.set_targets_embedding(query_relation, target)
        self.model.encoder.init_state()
    
    def get_returns(self, rewards):
        running_add = torch.zeros([rewards.shape[0]], device=self.device)
        returns = torch.zeros([rewards.shape[0], self.path_length], device=self.device)
        returns[:, self.path_length-1] = rewards
        for t in reversed(range(self.path_length)):
            running_add = self.gamma * running_add + returns[:, t]
            returns[:, t] = running_add
        return returns

    def update(self, rewards, time):
        returns = self.get_returns(rewards)
        loss = torch.stack(self.all_loss, dim=1)  # [B, T]
        baseline_reward = returns - self.baseline.b
        reward_mean, reward_std = torch.mean(baseline_reward), torch.std(baseline_reward) + 1e-6
        loss = torch.mul(loss, torch.div(baseline_reward - reward_mean, reward_std))  # [B, T]
        all_logits = torch.stack(self.all_logits, dim=2)  # [B, MAX_NUM_ACTIONS, T]
        entropy_loss = -torch.mean(torch.sum(torch.mul(torch.exp(all_logits), all_logits), dim=1))  # scalar
        total_loss = torch.mean(loss) - self.decaying_beta * entropy_loss  # scalar
        self.baseline.update(torch.mean(returns))
        self.model.zero_grad()
        total_loss.backward()
        torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=self.grad_clip_norm, norm_type=2)
        self.optimizer.step()
        self.beta_decay(time)
        return total_loss

    def beta_decay(self, time):
        if time % (self.beta_dacay_time*self.num_relations) == 0:
            self.decaying_beta *= self.beta_dacay_rate

    def update_beam_memory(self, path_idx):
        self.model.encoder.beam_update(path_idx)