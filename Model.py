#!/usr/bin/env python
# coding: utf-8
'''
Last update: 21.03.03. KS.Kwon

'''

import tensorflow as tf
from tensorflow import keras

from Utils import *
from Layer import *
from Metrics import *


class GAE(keras.models.Model):
    def __init__(self, 
                 max_poc_node, 
                 n_poc_feature,
                 batch_size,
                 W_poc_layer,
                 b_poc_layer,
                 b_poc_recon,
                 **kwargs):
        super().__init__(**kwargs)

        self._hidden_features = [300, 200]

        self.Pocket_encoder1 = GraphConv(           
            max_poc_node=max_poc_node, 
            n_feature=n_poc_feature, 
            n_hidden=self._hidden_features[0],
            batch_size=batch_size,
            W_layer=W_poc_layer[0],
            b_layer=b_poc_layer[0],
            b_recon=b_poc_recon[0],
            activation=tf.nn.leaky_relu
        )
        self.Pocket_encoder2 = GraphConv(
            max_poc_node=max_poc_node, 
            n_feature=self._hidden_features[0], 
            n_hidden=self._hidden_features[1],
            batch_size=batch_size,
            W_layer=W_poc_layer[1],
            b_layer=b_poc_layer[1],
            b_recon=b_poc_recon[1],
            activation=tf.nn.leaky_relu
        )
       

    def call(self, inputs, training=False, featurize=False):
        M_poc_feature, M_poc_adj, poc_mask, poc_d_score = inputs

        M_poc_hidden1 = self.Pocket_encoder1((M_poc_feature, M_poc_adj, poc_mask))
        M_poc_hidden2 = self.Pocket_encoder2((M_poc_hidden1, M_poc_adj, poc_mask))
        recon_hidden2 = self.Pocket_encoder2.reconstruction((M_poc_hidden2, poc_mask), activation=lambda x:x)
        recon_hidden1 = self.Pocket_encoder1.reconstruction((recon_hidden2, poc_mask), activation=lambda x:x)

        ###  Multiply centroid distance score
        #poc_d_score = tf.expand_dims(poc_d_score, axis=2)
        #M_poc_hidden2 = M_poc_hidden2 * poc_d_score

        if featurize:
            AX = tf.matmul(M_poc_adj, M_poc_feature)
            return M_poc_hidden2, AX, recon_hidden1
        else:
            return M_poc_hidden2, recon_hidden1
    
    def loss(self, M_poc_feature, M_poc_adj, recon_hidden1):
        AX = tf.matmul(M_poc_adj, M_poc_feature)
        
        return tf.sqrt(tf.reduce_mean(tf.square(AX - recon_hidden1)))

    def _convToFP(self, matrix, mask):
        n = tf.reduce_sum(mask, axis=1)
        vector = tf.reduce_sum(matrix, axis=1)
        FP = vector / tf.expand_dims(n, axis=-1)

        return FP