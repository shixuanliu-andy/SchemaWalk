#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Feb  9 21:49:08 2021

@author: Shixuan Liu
"""
import torch
import torch.nn as nn
import torch.optim as optim

class TransformerEncoder:
    def __init__(self, params):
        self.hidden_size = params['hidden_size']
        self.LSTM_Layers = params['LSTM_layers']
        self.use_entity_embeddings = params['use_entity_embeddings']
        self.action_vocab_size = len(params['relation_vocab'])
        self.entity_vocab_size = len(params['entity_vocab'])
        self.embedding_size = params['embedding_size']
        self.train_entities = params['train_entity_embeddings']
        self.train_relations = params['train_relation_embeddings']
        self.entity_embedding_size = params['embedding_size']
        self.is_training = True
        self.num_heads = 16
        self.dropout_rate = 0.1
        self.num_stacks = 1
        if params['use_entity_embeddings']:
            self.entity_initializer = tf.contrib.layers.xavier_initializer()
            self.trans_input_size = 5*self.embedding_size
        else:
            self.entity_initializer = tf.zeros_initializer()
            self.trans_input_size = 3*self.embedding_size
        self.feedforward_units = [4*self.trans_input_size, self.trans_input_size]
        self.initializer = tf.contrib.layers.xavier_initializer()
        
        with tf.variable_scope("action_lookup_table", reuse=tf.AUTO_REUSE):
            self.action_embedding_placeholder = tf.placeholder(tf.float32, [self.action_vocab_size, self.embedding_size])
            self.relation_lookup_table = tf.get_variable("relation_lookup_table", shape=[self.action_vocab_size, self.embedding_size], dtype=tf.float32, initializer=tf.contrib.layers.xavier_initializer(), trainable=self.train_relations)
            self.relation_embedding_init = self.relation_lookup_table.assign(self.action_embedding_placeholder)

        with tf.variable_scope("entity_lookup_table", reuse=tf.AUTO_REUSE):
            self.entity_embedding_placeholder = tf.placeholder(tf.float32, [self.entity_vocab_size, self.embedding_size])
            self.entity_lookup_table = tf.get_variable("entity_lookup_table", shape=[self.entity_vocab_size, self.entity_embedding_size], dtype=tf.float32, initializer=self.entity_initializer, trainable=self.train_entities)
            self.entity_embedding_init = self.entity_lookup_table.assign(self.entity_embedding_placeholder)

    def __call__(self, prev_relation, current_entities, candidate_relations, candidate_entities,
                 query_embedding, target_embedding, history_input_window):
        prev_action_embedding = self.action_encoder(prev_relation, current_entities)
        prev_entity = tf.nn.embedding_lookup(self.entity_lookup_table, current_entities)
        candidate_action_embeddings = self.action_encoder(candidate_relations, candidate_entities)
        if self.use_entity_embeddings:
            state_query_concat = tf.concat([tf.concat([prev_action_embedding, prev_entity], axis=-1),
                                            query_embedding, target_embedding - query_embedding], axis=-1)
        else:
            state_query_concat = tf.concat([prev_action_embedding, query_embedding], axis=-1)
        trans_input = tf.expand_dims(state_query_concat, axis=1)
        with tf.variable_scope("encoder_transformer"):
            input_window = tf.concat([history_input_window, trans_input], axis=1)
            output = self.transformer(input_window)
            output = tf.reduce_mean(output, axis=1)
        return output, candidate_action_embeddings, trans_input
    
    def action_encoder(self, next_relations, next_entities):
        with tf.variable_scope("lookup_table_edge_encoder", reuse=tf.AUTO_REUSE):
            relation_embedding = tf.nn.embedding_lookup(self.relation_lookup_table, next_relations)
            entity_embedding = tf.nn.embedding_lookup(self.entity_lookup_table, next_entities)
            if self.use_entity_embeddings:
                action_embedding = tf.concat([relation_embedding, entity_embedding], axis=-1)
            else:
                action_embedding = relation_embedding
        return action_embedding
    
    def transformer(self, inputs):
        with tf.variable_scope("embedding"):
            W_embed = tf.get_variable("weights", [1, self.trans_input_size, self.trans_input_size], initializer=self.initializer)
            self.embedded_input = tf.nn.conv1d(inputs, W_embed, 1, "VALID", name="embedded_input")
            self.enc = tf.layers.batch_normalization(self.embedded_input, axis=2, training=self.is_training, name='layer_norm', reuse=None)
        with tf.variable_scope("stack"):
            for i in range(self.num_stacks):
                with tf.variable_scope("block_{}".format(i)):
                    self.enc = self.multihead_attention(self.enc)
                    self.enc = self.feedforward(self.enc)
            self.encoder_output = self.enc
        return self.encoder_output
      
    def multihead_attention(self, inputs):
        with tf.variable_scope("multihead_attention", reuse=None):
            Q = tf.layers.dense(inputs, self.trans_input_size, activation=tf.nn.relu) # [batch_size, seq_length, n_hidden]
            K = tf.layers.dense(inputs, self.trans_input_size, activation=tf.nn.relu) # [batch_size, seq_length, n_hidden]
            V = tf.layers.dense(inputs, self.trans_input_size, activation=tf.nn.relu) # [batch_size, seq_length, n_hidden]
            # Split and concat
            Q_ = tf.concat(tf.split(Q, self.num_heads, axis=2), axis=0) # [batch_size, seq_length, n_hidden/num_heads]
            K_ = tf.concat(tf.split(K, self.num_heads, axis=2), axis=0) # [batch_size, seq_length, n_hidden/num_heads]
            V_ = tf.concat(tf.split(V, self.num_heads, axis=2), axis=0) # [batch_size, seq_length, n_hidden/num_heads]
            # Multiplication
            outputs = tf.matmul(Q_, tf.transpose(K_, [0, 2, 1])) # num_heads*[batch_size, seq_length, seq_length]
            outputs = outputs / (K_.get_shape().as_list()[-1] ** 0.5)
            outputs = tf.nn.softmax(outputs) # num_heads*[batch_size, seq_length, seq_length]
            outputs = tf.layers.dropout(outputs, rate=self.dropout_rate, training=tf.convert_to_tensor(self.is_training))
            outputs = tf.matmul(outputs, V_) # num_heads*[batch_size, seq_length, n_hidden/num_heads]
            outputs = tf.concat(tf.split(outputs, self.num_heads, axis=0), axis=2 ) # [batch_size, seq_length, n_hidden]
            outputs += inputs # [batch_size, seq_length, n_hidden]
            outputs = tf.layers.batch_normalization(outputs, axis=2, training=self.is_training, name='ln', reuse=None)  # [batch_size, seq_length, n_hidden]
        return outputs
    
    def feedforward(self, inputs):
        with tf.variable_scope("feedforward"):
            outputs = tf.layers.conv1d(inputs, self.feedforward_units[0], 1, activation=tf.nn.relu, use_bias=True)
            outputs = tf.layers.conv1d(outputs, self.feedforward_units[1], 1, activation=None, use_bias=True)
            outputs += inputs
            outputs = tf.layers.batch_normalization(outputs, axis=2, training=self.is_training, name='ln', reuse=None)  # [batch_size, seq_length, n_hidden]
        return outputs