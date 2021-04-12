#!/usr/bin/env python
# coding: utf-8
'''
Last update: 21.01.21. by KS.Kwon

'''

import tensorflow as tf
import numpy as np

from tensorflow import keras
from tensorflow.keras import layers

from numpy.linalg import inv
from scipy.linalg import fractional_matrix_power

from inits import glorot, He, zeros


def get_inv_diag_matrix(M_adjacency, batch_size, max_poc_node):
    deg = tf.reduce_sum(M_adjacency, 1)
    inv_deg = []

    for b in range(batch_size):
        k = [i for i in deg[b] if i != 0]
        inv_k = inv(np.diag(k))
        inv_d = np.pad(inv_k, ((0, max_poc_node-len(k)), (0, max_poc_node-len(k))), 'constant', constant_values=(0))
        #inv_d = fractional_matrix_power(inv_d, 0.5)
        inv_deg.append(inv_d)
    
    return np.array(inv_deg)


class GraphConv(layers.Layer):
    def __init__(self,
                 max_poc_node, 
                 n_feature, 
                 n_hidden,  
                 batch_size,
                 W_layer,
                 b_layer,
                 b_recon,
                 dropout=0.5,
                 activation=tf.nn.relu,
                 **kwargs
                ):
        super().__init__(**kwargs)
        
        self.max_poc_node = max_poc_node
        self.n_feature = n_feature
        self.n_hidden = n_hidden
        self.batch_size = batch_size
        self.dropout = dropout
        self.activation = activation

        if W_layer is None:
            self.W_layer = tf.Variable(name='W', initial_value=He([self.n_feature, self.n_hidden]))
        else:
            self.W_layer = tf.Variable(name='W', initial_value=W_layer)
        
        if b_layer is None:
            self.b_layer = tf.Variable(name='b_layer', initial_value=zeros([self.n_hidden,]))
        else:
            self.b_layer = tf.Variable(name='b_layer', initial_value=b_layer)

        if b_recon is None:
            self.b_recon = tf.Variable(name='b_recon', initial_value=zeros([self.n_feature,]))
        else:
            self.b_recon = tf.Variable(name='b_recon', initial_value=b_recon)

    def call(self, inputs, training=False):
        M_features, M_adjacency, mask = inputs

        ### To do: How to normalize the adjacency matrix
        #inv_deg = get_inv_diag_matrix(M_adjacency, self.batch_size, self.max_poc_node)
        #inv_deg = tf.convert_to_tensor(inv_deg)

        AX = tf.matmul(M_adjacency, M_features)
        hidden_feature = tf.matmul(AX, self.W_layer) + tf.expand_dims(self.b_layer, 0) 

        if training:
            hidden_feature = tf.nn.dropout(hidden_feature, self.dropout)

        activations = self.activation(hidden_feature) * tf.expand_dims(mask, axis=-1)

        return activations

    def reconstruction(self, inputs, activation=tf.nn.leaky_relu, training=False):
        hidden_features, mask = inputs

        ### To do: How to normalize the adjacency matrix
        #inv_deg = get_inv_diag_matrix(M_adjacency, self.batch_size, self.max_poc_node)
        #inv_deg = tf.convert_to_tensor(inv_deg)
        
        recon_hidden_features = tf.matmul(hidden_features, self.W_layer, transpose_b=True) # [max_poc_nodes, n_hidden]
        recon_output = recon_hidden_features + tf.expand_dims(self.b_recon, 0)  # 

        if training:
            recon_output = tf.nn.dropout(recon_output, self.dropout)

        recon_activations = self.activation(recon_output) * tf.expand_dims(mask, axis=-1)

        return recon_activations