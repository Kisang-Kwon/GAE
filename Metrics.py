import os
import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd

from collections import defaultdict

from sklearn.metrics import roc_curve, auc, confusion_matrix
from sklearn.manifold import TSNE
from pandas import Series


def Loss_plot(loss_dict, checkpoint, prefix, epochs):
    fig1 = plt.figure()
    plt.plot(range(1, len(loss_dict['tr'])+1), loss_dict['tr'])
    plt.plot(range(1, len(loss_dict['va'])+1), loss_dict['va'])

    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.title('Loss plot')
    plt.legend(['Training', 'Validation'], loc='upper right')
    plt.xlim(0, epochs)
    plt.ylim(0.0, 5.0)

    filename = os.path.join(checkpoint, f'{prefix}_training_loss.png')
    plt.savefig(filename)
    plt.close(fig1)


def Prediction(outdir, PIDs, hidden_features):
    if os.path.isdir(outdir) is False:
        os.mkdir(outdir)

    for idx, PID in enumerate(PIDs):
        out_fpath = os.path.join(outdir, f'{PID}.npy')
        h_feature = np.array(hidden_features[idx])
        np.save(out_fpath, h_feature)


def Reconstruction(outdir, PIDs, AXs, recon_features):
    if os.path.isdir(outdir) is False:
        os.mkdir(outdir)

    for idx, PID in enumerate(PIDs):
        H_fpath = os.path.join(outdir, f'{PID}_H.npy')
        R_fpath = os.path.join(outdir, f'{PID}_R.npy')
        
        h_feature = np.array(AXs[idx])
        r_feature = np.array(recon_features[idx])

        np.save(R_fpath, r_feature)
        np.save(H_fpath, h_feature)


def tSNE(out_fpath, PIDs, hidden_features):
    FPs = tf.reduce_sum(hidden_features, axis=1)
    df_FPs = pd.DataFrame(FPs, index=PIDs)

    tsne = TSNE(learning_rate=100)
    transform_2d = tsne.fit_transform(df_FPs)

    xs = transform_2d[:, 0]
    ys = transform_2d[:, 1]

    size = 20*np.ones([len(xs)])
    fig = plt.figure(figsize=(10,10))
    plt.scatter(xs, ys, s = size)
    #plt.scatter(xs, ys, s = size, c = labels['label'])
    plt.suptitle('2D t-SNE', fontsize=20, y=0.95)
    plt.savefig(out_fpath, bbox_inches='tight')
    plt.close()
