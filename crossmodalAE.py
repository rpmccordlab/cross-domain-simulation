# -*- coding: utf-8 -*-
"""
Created on Fri Feb 11 10:56:42 2022

@author: xuyan
"""

##Training network
import gc
#import math

import torch
import torch.nn as nn
import torch.optim as optim
import torch.backends.cudnn as cudnn
cudnn.deterministic = True
cudnn.benchmark = True

#import umap
import numpy as np

import random
seed = 1
random.seed(seed)
torch.manual_seed(seed)
torch.cuda.manual_seed_all(seed)
np.random.seed(seed)

#import copy
from network import *

def set_requires_grad(model, requires_grad=True):
    for param in model.parameters():
        param.requires_grad = requires_grad
  
def inv_lr_scheduler(optimizer, iter_num, gamma, power, lr=0.001, weight_decay=0.0005):
    """Decay learning rate by a factor of 0.1 every lr_decay_epoch epochs."""
    lr = lr * (1 + gamma * iter_num) ** (-power)
    i=0
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr * param_group['lr_mult']
        param_group['weight_decay'] = weight_decay * param_group['decay_mult']
        i+=1

    return optimizer

schedule_dict = {"inv":inv_lr_scheduler}

def crossmodalAE(epoch=50, batch_size=512,source_trainset=None, target_trainset=None):
    
    ##this training function is fully unsupervised learning
    encoderA = Encoder(input_dim=source_trainset.shape[1]).cuda()
    encoderB = Encoder(input_dim=target_trainset.shape[1]).cuda()
    encoderA.cuda()
    encoderB.cuda()
    advnet = AdvNet().cuda()
    decoderA = Decoder(output_dim=source_trainset.shape[1]).cuda()
    decoderB = Decoder(output_dim=target_trainset.shape[1]).cuda()
    
    
    config_optimizer = {"lr_type": "inv", "lr_param": {"lr": 0.001, "gamma": 0.001, "power": 0.75}}
    parameter_listC = encoderA.get_parameters()+encoderB.get_parameters() +\
        decoderA.get_parameters() + decoderB.get_parameters()
    parameter_listD = encoderA.get_parameters() + advnet.get_parameters() + encoderB.get_parameters()
    #parameter_listD = encoder.get_parameters() + disT.get_parameters() + decoderS.get_parameters()
    optimizerC = optim.SGD(parameter_listC, lr=5e-3, weight_decay=5e-4, momentum=0.9, nesterov=True)#lr=1e-3
    optimizerD = optim.SGD(parameter_listD, lr=5e-3, weight_decay=5e-4, momentum=0.9, nesterov=True)#lr=1e-3
    schedule_param = config_optimizer["lr_param"]
    lr_scheduler = schedule_dict[config_optimizer["lr_type"]]
    l1_crit = torch.nn.MSELoss()#CosineEmbeddingLoss()#L1Loss()#
    ###########################################################################
    for e in range(1,epoch+1):
        n = source_trainset.shape[0]
        r = np.random.permutation(n)
        X_source = torch.tensor(source_trainset[r,:]).float()
        
        n = target_trainset.shape[0]
        r = np.random.permutation(n)
        X_target = torch.tensor(target_trainset[r,:]).float()
        
        n=min(target_trainset.shape[0],source_trainset.shape[0])
        for j in range(n//batch_size):
            
            source_inputs = X_source[j*batch_size:(j+1)*batch_size,:].cuda()
            nsource_inputs1 = source_inputs + torch.normal(0,1,source_inputs.shape).cuda()
            target_inputs = X_target[j*batch_size:(j+1)*batch_size,:].cuda()
            ntarget_inputs1 = target_inputs + torch.normal(0,1,target_inputs.shape).cuda()
            
            optimizerC = lr_scheduler(optimizerC, epoch, **schedule_param)
            optimizerD = lr_scheduler(optimizerD, epoch, **schedule_param)
            optimizerC.zero_grad()
            optimizerD.zero_grad()
            
            feature_source1 = encoderA(nsource_inputs1)
            feature_target1 = encoderB(ntarget_inputs1)
            res_source1 = decoderA(feature_source1)
            res_target1 = decoderB(feature_target1)
            
            clf_loss = (l1_crit(res_source1,source_inputs)+l1_crit(res_target1,target_inputs))*0.5
            clf_loss.backward()
            optimizerC.step()
            
            feature_source = encoderA(source_inputs)
            feature_target = encoderB(target_inputs)
            prob_source = advnet.forward(feature_source)
            prob_target = advnet.forward(feature_target)
            wasserstein_distance = (prob_source.mean() - prob_target.mean())*3.0
            adv_loss = -wasserstein_distance
            adv_loss.backward()
            optimizerD.step()
            
        gc.collect()
        
    return encoderA, encoderB
