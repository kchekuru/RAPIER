# -*- coding:utf-8 -*-
import torch 
import torch.nn as nn
import torch.nn.functional as F

from .model import MLP
import sys, os
import numpy as np
from tqdm import tqdm

from .loss import loss_coteaching, loss_simple


def _drop_nonfinite_rows(array, name):
    """Drop rows containing NaN/Inf to keep classifier training numerically stable."""
    if array.ndim != 2:
        return array
    finite_mask = np.isfinite(array).all(axis=1)
    dropped = int((~finite_mask).sum())
    if dropped > 0:
        print(f'[warn] {name}: dropping {dropped} non-finite rows out of {array.shape[0]}')
    return array[finite_mask]

# Hyper Parameters
batch_size = 128
learning_rate = 1e-4
epochs = 100
num_gradual = 10
forget_rate = 0.1
exponent = 1
rate_schedule = np.ones(epochs) * forget_rate
rate_schedule[:num_gradual] = np.linspace(0, forget_rate ** exponent, num_gradual)
grad_clip = 1.0
weight_decay = 1e-5

def accuracy(logit, target):
    """Computes the precision@k for the specified values of k"""
    output = F.softmax(logit, dim=1)
    batch_size = target.size(0)

    _, pred = output.topk(1, 1, True, True)
    pred = pred.t()
    correct = pred.eq(target.view(1, -1).expand_as(pred))

    correct_k = correct[:1].view(-1).float().sum(0, keepdim=True)
    return correct_k.mul_(100.0 / batch_size)

# Train the Model
def train(train_loader, epoch, model1, optimizer1, model2, optimizer2, device):
    
    train_total1=0
    train_correct1=0 
    train_total2=0
    train_correct2=0 

    for i, data_labels in enumerate(train_loader):
        
        feats = data_labels[:, :-1].to(dtype=torch.float32)
        labels = data_labels[:, -1].to(dtype=int)
        if device != None:
            torch.cuda.set_device(device)
            feats = feats.cuda()
            labels = labels.cuda()
    
        logits1 = model1(feats)
        prec1 = accuracy(logits1, labels)
        train_total1 += 1
        train_correct1 += prec1

        logits2 = model2(feats)
        prec2 = accuracy(logits2, labels)
        train_total2 += 1
        train_correct2 += prec2
        
        # Use stable simple cross-entropy loss instead of co-teaching
        loss_1 = loss_simple(logits1, labels)
        loss_2 = loss_simple(logits2, labels)

        optimizer1.zero_grad()
        loss_1.backward()
        torch.nn.utils.clip_grad_norm_(model1.parameters(), grad_clip)
        optimizer1.step()
        
        optimizer2.zero_grad()
        loss_2.backward()
        torch.nn.utils.clip_grad_norm_(model2.parameters(), grad_clip)
        optimizer2.step()

    train_acc1=float(train_correct1)/float(train_total1)
    train_acc2=float(train_correct2)/float(train_total2)
    return train_acc1, train_acc2

# predict unknown traffic data's label
def predict(test_loader, model, device, alpha=0.5):

    preds = []
    for i, data in enumerate(test_loader):
        
        # Forward + Backward + Optimize
        feats = data.to(dtype=torch.float32)
        
        if device != None:
            torch.cuda.set_device(device)
            feats = feats.cuda()
        
        logits = model(feats)
        outputs = F.softmax(logits, dim=1)
        preds.append((outputs[:, 1] > alpha).detach().cpu().numpy())

    return np.concatenate(preds, axis=0)

def main(feat_dir, model_dir, result_dir, TRAIN, cuda_device, parallel=5):
    
    if str(cuda_device) != 'None':
        cuda_device = int(cuda_device)
    # get the origin training set
    be = np.load(os.path.join(feat_dir, 'be_corrected.npy'), allow_pickle=True)[:, :32]
    ma = np.load(os.path.join(feat_dir, 'ma_corrected.npy'), allow_pickle=True)[:, :32]
    be = _drop_nonfinite_rows(be, 'be_corrected.npy')
    ma = _drop_nonfinite_rows(ma, 'ma_corrected.npy')
    be_shape = be.shape[0]
    ma_shape = ma.shape[0]

    for index in range(parallel):

        # add synthesized traffic features into the origin training set
        be_gen = np.load(os.path.join(feat_dir, 'be_%s_generated_GAN_%d.npy' % (TRAIN, index)), allow_pickle=True)
        ma_gen1 = np.load(os.path.join(feat_dir, 'ma_%s_generated_GAN_1_%d.npy' % (TRAIN, index)), allow_pickle=True)
        ma_gen2 = np.load(os.path.join(feat_dir, 'ma_%s_generated_GAN_2_%d.npy' % (TRAIN, index)), allow_pickle=True)
        be_gen = _drop_nonfinite_rows(be_gen, 'be_%s_generated_GAN_%d.npy' % (TRAIN, index))
        ma_gen1 = _drop_nonfinite_rows(ma_gen1, 'ma_%s_generated_GAN_1_%d.npy' % (TRAIN, index))
        ma_gen2 = _drop_nonfinite_rows(ma_gen2, 'ma_%s_generated_GAN_2_%d.npy' % (TRAIN, index))
        np.random.shuffle(be_gen)
        np.random.shuffle(ma_gen1)
        np.random.shuffle(ma_gen2)
        be = np.concatenate([
            be, 
            be_gen[:be_shape // (parallel)], 
        ], axis=0)
        
        ma = np.concatenate([
            ma,
            ma_gen1[:ma_shape // (parallel) // 5],
            ma_gen2[:ma_shape // (parallel) // 5],
        ], axis=0)

    print(be.shape, ma.shape)

    train_data = np.concatenate([be, ma], axis=0)
    train_label = np.concatenate([np.zeros(be.shape[0]), np.ones(ma.shape[0])], axis=0)
    train_dataset = np.concatenate((train_data, train_label[:, None]), axis=1)
    train_dataset = _drop_nonfinite_rows(train_dataset, 'train_dataset')

    test_data_label = np.load(os.path.join(feat_dir, 'test.npy'), allow_pickle=True)
    test_data_label = _drop_nonfinite_rows(test_data_label, 'test.npy')
    test_data = test_data_label[:, :32]
    test_label = test_data_label[:, -1]

    device = int(cuda_device) if cuda_device != 'None' else None
    # define drop rate schedule

    if device != None:
        torch.cuda.set_device(device)
    # Data Loader (Input Pipeline)
    print('loading dataset...')
    train_loader = torch.utils.data.DataLoader(dataset=train_dataset, batch_size=batch_size, shuffle=True)
    # Define models
    print('building model...')
    mlp1 = MLP(input_size=32, hiddens=[16, 8], output_size=2, device=device)
    if device != None:
        mlp1.to_cuda(device)
        mlp1 = mlp1.cuda()
    optimizer1 = torch.optim.Adam(mlp1.parameters(), lr=learning_rate, weight_decay=weight_decay)
    
    mlp2 = MLP(input_size=32, hiddens=[16, 8], output_size=2, device=device)
    if device != None:
        mlp2.to_cuda(device)
        mlp2 = mlp2.cuda()
    optimizer2 = torch.optim.Adam(mlp2.parameters(), lr=learning_rate, weight_decay=weight_decay)

    epoch=0
    mlp1.train()
    mlp2.train()
    for epoch in tqdm(range(epochs)):
        train(train_loader, epoch, mlp1, optimizer1, mlp2, optimizer2, device)
    
    mlp1.eval()
    test_loader = torch.utils.data.DataLoader(dataset=test_data, batch_size=batch_size, shuffle=False)
    preds = predict(test_loader, mlp1, device)
    np.save(os.path.join(result_dir, 'prediction.npy'), preds)

    scores = np.zeros((2, 2))
    for label, pred in zip(test_label, preds):
        scores[int(label), int(pred)] += 1
    TP = scores[1, 1]
    FP = scores[0, 1]
    TN = scores[0, 0]
    FN = scores[1, 0]
    
    Accuracy = (TP + TN) / (TP + TN + FP + FN)
    Recall = TP / (TP + FN)
    Precision = TP / (TP + FP)
    F1score = 2 * Recall * Precision / (Recall + Precision)
    print(Recall, Precision, F1score)
    
    with open(os.path.join(result_dir, 'detection_result.txt'), 'w') as fp:
        fp.write('Testing data: Benign/Malicious = %d/%d\n'%((TN + FP), (TP + FN)))
        fp.write('Recall: %.2f, Precision: %.2f, F1: %.2f\n'%(Recall, Precision, F1score))
        fp.write('Acc: %.2f\n'%(Accuracy))

    mlp1 = mlp1.cpu()
    mlp1.to_cpu()
    torch.save(mlp1, os.path.join(model_dir, 'Detection_Model.pkl'))

