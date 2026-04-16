from .gen_model import GEN
from .made import MADE
import torch
import numpy as np 
import torch.nn as nn 
import torch.nn.functional as F
import sys
import os 
from sklearn.datasets import make_blobs
import math

# synthesize 3 types of samples.
def main(feat_dir, model_dir, TRAIN, index, cuda_device):

    be = np.load(os.path.join(feat_dir, 'be_%s.npy'%(TRAIN)), allow_pickle=True)[:, :32]
    ma = np.load(os.path.join(feat_dir, 'ma_%s.npy'%(TRAIN)), allow_pickle=True)[:, :32]

    output_size = be.shape[1]
    hiddens = [8, 16]
    device = int(cuda_device) if cuda_device != 'None' else None
    use_cuda = device is not None
    train_type_be = 'be_' + TRAIN
    train_type_ma = 'ma_' + TRAIN

    load_name_be = f"gen_GAN_{train_type_be}_{'_'.join(str(d) for d in hiddens)}.pt"
    load_name_ma1 = f"gen1_GAN_{train_type_ma}_{'_'.join(str(d) for d in hiddens)}.pt"
    load_name_ma2 = f"gen2_GAN_{train_type_ma}_{'_'.join(str(d) for d in hiddens)}.pt"
    BeGenModel = torch.load(os.path.join(model_dir, load_name_be), weights_only=False)
    MaGenModel_1 = torch.load(os.path.join(model_dir, load_name_ma1), weights_only=False)
    MaGenModel_2 = torch.load(os.path.join(model_dir, load_name_ma2), weights_only=False)

    if use_cuda:
        torch.cuda.set_device(device)
        BeGenModel.to_cuda(device)
        BeGenModel = BeGenModel.cuda()
        MaGenModel_1.to_cuda(device)
        MaGenModel_1 = MaGenModel_1.cuda()
        MaGenModel_2.to_cuda(device)
        MaGenModel_2 = MaGenModel_2.cuda()

    def move_tensor(tensor):
        return tensor.cuda() if use_cuda else tensor

    def generate(train_type, GenModel, total_size, seed):

        data_train = np.load(os.path.join(feat_dir, train_type + '.npy'), allow_pickle=True)[:, :output_size]
        mu_train = torch.Tensor(data_train.mean(axis=0))
        s_train = torch.Tensor(data_train.std(axis=0))

        mu_train = move_tensor(mu_train)
        s_train = move_tensor(s_train)

        X, _ = make_blobs(n_samples=total_size, centers=[[0, 0]], n_features=2, random_state=seed)
        X = move_tensor(torch.Tensor(X))
        batch = GenModel.forward(X)
        batch1 = batch * s_train + mu_train
        gen_data = batch1.detach().cpu()
        
        return np.array(gen_data)

    gen_data_be = generate(train_type_be, BeGenModel, int(be.shape[0]) * 2, np.random.randint(1000))
    gen_data_ma1 = generate(train_type_ma, MaGenModel_1, int(ma.shape[0]) * 2, np.random.randint(1000))
    gen_data_ma2 = generate(train_type_ma, MaGenModel_2, int(ma.shape[0]) * 2, np.random.randint(1000))

    np.save(os.path.join(feat_dir, 'be_%s_generated_GAN_%d.npy'%(TRAIN, index)), gen_data_be)
    np.save(os.path.join(feat_dir, 'ma_%s_generated_GAN_1_%d.npy'%(TRAIN, index)), gen_data_ma1)
    np.save(os.path.join(feat_dir, 'ma_%s_generated_GAN_2_%d.npy'%(TRAIN, index)), gen_data_ma2)
