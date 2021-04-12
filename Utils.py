import os
import sys
import numpy as np
import tensorflow as tf
import csv

from multiprocessing import Pool
from rdkit.Chem import ChemicalFeatures
from rdkit import Chem, RDConfig

seed = 123
np.random.seed(seed)
tf.random.set_seed(seed)


def input_list_parsing(f_input_list, pocket_dir):
    Fopen = open(f_input_list)

    input_list = []
    for line in Fopen:
        line = line.rstrip()
        f_poc = os.path.join(pocket_dir, f'{line}.npy')
        input_list.append(f_poc)
    
    return input_list


def read_npy(f_npy):
    arr = np.load(f_npy, allow_pickle=True)
    return arr


def poc_load_data(input_list, n_threads=4):
    
    batch_data = []
    #for i in input_list:
    #    batch_data.append(np.load(i, allow_pickle=True))
    p = Pool(n_threads)
    batch_data = p.map(read_npy, input_list)
    batch_data = np.array(batch_data)
    p.close()

    poc_feat = []
    poc_adj = []
    poc_mask = []
    poc_d_score = []
    PIDs = []

    for e in batch_data:
        pf, pa, pm, d_score, pid, _ = e
        
        poc_feat.append(pf)
        poc_adj.append(pa)
        poc_mask.append(pm)
        poc_d_score.append(d_score)
        PIDs.append(pid)

    return poc_feat, poc_adj, poc_mask, poc_d_score, PIDs
    

def lig_load_data(input_list, n_threads=4): 

    batch_data = []
    #for i in input_list:
    #    batch_data.append(np.load(i, allow_pickle=True))
    p = Pool(n_threads)
    batch_data = p.map(read_npy, input_list)
    batch_data = np.array(batch_data)
    p.close()
    
    mol_atom_feat = []
    mol_bond_feat = []
    mol_atom_adj = []
    mol_bond_adj = []
    mol_mask = []
    CIDs = []
    
    for e in batch_data:
        af, bf, aa, ba, mm, cid = e
        
        mol_atom_feat.append(af)
        mol_bond_feat.append(bf)
        mol_atom_adj.append(aa)
        mol_bond_adj.append(ba)
        mol_mask.append(mm)
        CIDs.append(cid)

    return mol_atom_feat, mol_bond_feat, mol_atom_adj, mol_bond_adj, mol_mask, CIDs


def load_labels(input_list):
    labels = []
    for l in input_list:
        label = np.zeros([2])
        l = int(l)
        label[l] = 1
        labels.append(label)

    return labels
