import os
import time
import numpy as np
import random
import csv
from collections import defaultdict


import tensorflow as tf
from tensorflow.keras import optimizers

from Utils import *
from Model import GAE
from Layer import GraphConv
from config import configuration
from Metrics import *

seed = 123
np.random.seed(seed)
tf.random.set_seed(seed)


def train_GAE(f_config, transfer='Autoencoder'):
    print('======= Parameters ========')
    f_train_list, f_valid_list, pocket_dir, max_poc_node, epochs, learning_rate, batch_size, l2_param, checkpoint, prefix = configuration(f_config)
    print('===========================\n')

    print(time.strftime('Start: %x %X', time.localtime(time.time())))
    
    checkpoint_save = os.path.join(checkpoint, 'GAE')

    # Data loading 
    train_list = input_list_parsing(f_train_list, pocket_dir)
    valid_list = input_list_parsing(f_valid_list, pocket_dir)
    
    random.shuffle(train_list)

    tr_total_batch = int(len(train_list) / batch_size)
    va_total_batch = int(len(valid_list) / batch_size)

    # Weights loading 
    os.makedirs(checkpoint_save)
    W_poc_layer = [None, None]
    b_poc_layer = [None, None]
    b_poc_recon = [None, None]
   
    # Define model feature dimensions
    poc_check_data = np.load(train_list[0], allow_pickle=True)
    n_poc_feature = poc_check_data[0].shape[-1]

    # Define a model
    model = GAE(max_poc_node=max_poc_node, 
                n_poc_feature=n_poc_feature,
                batch_size=batch_size, 
                W_poc_layer=W_poc_layer,
                b_poc_layer=b_poc_layer,
                b_poc_recon=b_poc_recon
                )

    optimizer = optimizers.RMSprop(learning_rate = learning_rate)
    
    O_loss = open(f'{checkpoint_save}/loss.csv', 'w')
    O_loss.write('Epoch,Train,Valid\n')

    loss_dict = defaultdict(list)
    for epoch in range(1, epochs+1):
        print('Epoch', epoch, time.strftime('Start: [%x %X]', time.localtime(time.time())))

        # Training
        for b_idx in range(1, tr_total_batch+1):
            # Mini batch data load
            batch_list = np.array(train_list[batch_size*(b_idx-1):batch_size*b_idx])
            
            tr_M_poc_feat, tr_M_poc_adj, tr_poc_mask, tr_poc_d_score, tr_PIDs = poc_load_data(batch_list)
            
            # Convert to tensor variable
            tr_M_poc_feat = tf.convert_to_tensor(np.array(tr_M_poc_feat), dtype='float32')
            tr_M_poc_adj = tf.convert_to_tensor(np.array(tr_M_poc_adj), dtype='float32')
            tr_poc_mask = tf.convert_to_tensor(np.array(tr_poc_mask), dtype='float32')
            tr_poc_d_score = tf.convert_to_tensor(np.array(tr_poc_d_score), dtype='float32')

            tr_PIDs = tf.convert_to_tensor(np.array(tr_PIDs))
            
            # Model call and calculate gradients
            with tf.GradientTape() as tape:
                M_poc_hidden2, recon_hidden1 = model((tr_M_poc_feat, tr_M_poc_adj, tr_poc_mask, tr_poc_d_score), training=True)
                tr_loss = model.loss(tr_M_poc_feat, tr_M_poc_adj, recon_hidden1)

                if l2_param != 0.:
                    for param in model.trainable_variables:
                        tr_loss = tf.add(tr_loss, l2_param*tf.nn.l2_loss(param))
            
            # Weight update
            grads = tape.gradient(tr_loss, model.trainable_variables)
            optimizer.apply_gradients(zip(grads, model.trainable_variables))

        # Validation
        tr_avg_loss = 0.
        va_avg_loss = 0.
        for b_idx in range(1, tr_total_batch+1):
            # Mini batch data load
            batch_list = np.array(train_list[batch_size*(b_idx-1):batch_size*b_idx])
            
            tr_M_poc_feat, tr_M_poc_adj, tr_poc_mask, tr_poc_d_score, tr_PIDs = poc_load_data(batch_list)
            
            # Convert to tensor variable
            tr_M_poc_feat = tf.convert_to_tensor(np.array(tr_M_poc_feat), dtype='float32')
            tr_M_poc_adj = tf.convert_to_tensor(np.array(tr_M_poc_adj), dtype='float32')
            tr_poc_mask = tf.convert_to_tensor(np.array(tr_poc_mask), dtype='float32')
            tr_poc_d_score = tf.convert_to_tensor(np.array(tr_poc_d_score), dtype='float32')

            tr_PIDs = tf.convert_to_tensor(np.array(tr_PIDs))

            # Model call
            tr_M_poc_hidden2, tr_recon_hidden1 = model((tr_M_poc_feat, tr_M_poc_adj, tr_poc_mask, tr_poc_d_score), training=False)
            tr_loss = model.loss(tr_M_poc_feat, tr_M_poc_adj, tr_recon_hidden1)
            
            # Metric 
            #tr_avg_loss += tr_loss
            tr_avg_loss += np.sum(tr_loss)

        for b_idx in range(1, va_total_batch+1):
            # Mini batch data load
            batch_list = np.array(valid_list[batch_size*(b_idx-1):batch_size*b_idx])

            va_M_poc_feat, va_M_poc_adj, va_poc_mask, va_poc_d_score, va_PIDs = poc_load_data(batch_list)
                        
            # Convert to tensor variable
            va_M_poc_feat = tf.convert_to_tensor(np.array(va_M_poc_feat), dtype='float32')
            va_M_poc_adj = tf.convert_to_tensor(np.array(va_M_poc_adj), dtype='float32')
            va_poc_mask = tf.convert_to_tensor(np.array(va_poc_mask), dtype='float32')
            va_poc_d_score = tf.convert_to_tensor(np.array(va_poc_d_score), dtype='float32')

            va_PIDs = tf.convert_to_tensor(np.array(va_PIDs))
            
            # Model call
            va_M_poc_hidden2, va_recon_hidden1 = model((va_M_poc_feat, va_M_poc_adj, va_poc_mask, va_poc_d_score), training=False)
            va_loss = model.loss(va_M_poc_feat, va_M_poc_adj, va_recon_hidden1)
            
            # Metric
            #va_avg_loss += va_loss
            va_avg_loss += np.sum(va_loss)

        # Save metrics
        tr_avg_loss = round(float(tr_avg_loss / tr_total_batch), 4)
        va_avg_loss = round(float(va_avg_loss / va_total_batch), 4)
        O_loss.write(f'[Epoch {epoch}],{tr_avg_loss},{va_avg_loss}\n')
        loss_dict['tr'].append(tr_avg_loss)
        loss_dict['va'].append(va_avg_loss)

        print(f'Train loss: {tr_avg_loss}, Valid loss: {va_avg_loss}')

        # Save parameters every 5 epochs
        if epoch % 5 == 0:
            params = np.array([
                [model.Pocket_encoder1.W_layer, model.Pocket_encoder2.W_layer], 
                [model.Pocket_encoder1.b_layer, model.Pocket_encoder2.b_layer],
                [model.Pocket_encoder1.b_recon, model.Pocket_encoder2.b_recon]
            ])
            np.save(f'{checkpoint_save}/params.npy', params)
            np.save(f'{checkpoint_save}/params_{epoch}.npy', params)
            Loss_plot(loss_dict, checkpoint, prefix, epochs)

    O_loss.close()
    
    Loss_plot(loss_dict, checkpoint, prefix, epochs)
    print(time.strftime('End: %x %X', time.localtime(time.time())))


def eval_GAE(f_config):
    print('======= Parameters ========')
    f_test_list, pocket_dir, max_poc_node, batch_size, checkpoint, prefix = configuration(f_config)    
    print('===========================\n')

    checkpoint_load = os.path.join(checkpoint, 'GAE')

    print(time.strftime('Start: %x %X', time.localtime(time.time())))

    # Data loading 
    test_list = input_list_parsing(f_test_list, pocket_dir)
    te_total_batch = int(len(test_list) / batch_size)
    
    # Weights loading
    params = os.path.join(checkpoint_load, 'params.npy')
    W_poc_layer, b_poc_layer, b_poc_recon = np.load(params, allow_pickle=True)

    # Model assessment
    poc_check_data = np.load(test_list[0], allow_pickle=True)
    n_poc_feature = poc_check_data[0].shape[-1]
    
    model = GAE(max_poc_node=max_poc_node, 
                n_poc_feature=n_poc_feature,
                batch_size=batch_size, 
                W_poc_layer=W_poc_layer,
                b_poc_layer=b_poc_layer,
                b_poc_recon=b_poc_recon
                )
    
    avg_loss = 0.
    hidden_features = []
    PIDs = []
    for b_idx in range(1, te_total_batch+1):
        batch_list = np.array(test_list[batch_size*(b_idx-1):batch_size*b_idx])

        te_M_poc_feat, te_M_poc_adj, te_poc_mask, te_poc_d_score, te_PIDs = poc_load_data(batch_list)

        te_M_poc_feat = tf.convert_to_tensor(np.array(te_M_poc_feat), dtype='float32')
        te_M_poc_adj = tf.convert_to_tensor(np.array(te_M_poc_adj), dtype='float32')
        te_poc_mask = tf.convert_to_tensor(np.array(te_poc_mask), dtype='float32')
        te_poc_d_score = tf.convert_to_tensor(np.array(te_poc_d_score), dtype='float32')

        M_poc_hidden2, recon_hidden1 = model((te_M_poc_feat, te_M_poc_adj, te_poc_mask, te_poc_d_score), training=False)
        te_loss = model.loss(te_M_poc_feat, te_M_poc_adj, recon_hidden1)

        avg_loss += te_loss
        PIDs.extend(te_PIDs)
        hidden_features.extend(M_poc_hidden2)

    avg_loss = round(float(avg_loss / te_total_batch), 4)
    print(f'Average loss:\t{avg_loss}')

    predict_dir = os.path.join(checkpoint, prefix)
    Prediction(predict_dir, PIDs, hidden_features)

    tSNE_fpath = os.path.join(checkpoint, f'{prefix}_2D_tSNE.png')
    tSNE(tSNE_fpath, PIDs, hidden_features)

    print(time.strftime('End: %x %X', time.localtime(time.time())))


def print_GAE_representation(f_config):
    print('======= Parameters ========')
    f_test_list, pocket_dir, max_poc_node, batch_size, checkpoint, prefix = configuration(f_config)    
    print('===========================\n')

    checkpoint_load = os.path.join(checkpoint, 'GAE')

    print(time.strftime('Start: %x %X', time.localtime(time.time())))

    # Data loading
    test_list = input_list_parsing(f_test_list, pocket_dir)
    te_total_batch = int(len(test_list) / batch_size)
    
    # Weights loading
    params = os.path.join(checkpoint_load, 'params.npy')
    W_poc_layer, b_poc_layer, b_poc_recon = np.load(params, allow_pickle=True)
    
    # Model assessment
    poc_check_data = np.load(test_list[0], allow_pickle=True)
    n_poc_feature = poc_check_data[0].shape[-1]

    model = GAE(max_poc_node=max_poc_node, 
                n_poc_feature=n_poc_feature,
                batch_size=batch_size, 
                W_poc_layer=W_poc_layer,
                b_poc_layer=b_poc_layer,
                b_poc_recon=b_poc_recon
                )
    
    hidden_features = []
    AXs = []
    recon_matrices = []
    PIDs = []
    for b_idx in range(1, te_total_batch+1):
        batch_list = np.array(test_list[batch_size*(b_idx-1):batch_size*b_idx])

        te_M_poc_feat, te_M_poc_adj, te_poc_mask, te_poc_d_score, te_PIDs = poc_load_data(batch_list)

        te_M_poc_feat = tf.convert_to_tensor(np.array(te_M_poc_feat), dtype='float32')
        te_M_poc_adj = tf.convert_to_tensor(np.array(te_M_poc_adj), dtype='float32')
        te_poc_mask = tf.convert_to_tensor(np.array(te_poc_mask), dtype='float32')
        te_poc_d_score = tf.convert_to_tensor(np.array(te_poc_d_score), dtype='float32')

        M_poc_hidden2, AX, recon_hidden1 = model((te_M_poc_feat, te_M_poc_adj, te_poc_mask, te_poc_d_score), training=False, featurize=True)

        hidden_features.extend(M_poc_hidden2)
        AXs.extend(AX)
        recon_matrices.extend(recon_hidden1)
        PIDs.extend(te_PIDs)
    
    predict_dir = os.path.join(checkpoint, prefix)
    Prediction(predict_dir, PIDs, hidden_features)

    recon_dir = os.path.join(checkpoint, f'{prefix}_recon')
    Reconstruction(recon_dir, PIDs, AXs, recon_matrices)

    print(time.strftime('End: %x %X', time.localtime(time.time())))


if __name__ == '__main__':
    
    f_config = sys.argv[1]
    tasks = sys.argv[2]

    if tasks == 'train':
        train_GAE(f_config, transfer=False)
    elif tasks == 'eval':
        eval_GAE(f_config)
    elif tasks == 'featurize':
        print_GAE_representation(f_config)
