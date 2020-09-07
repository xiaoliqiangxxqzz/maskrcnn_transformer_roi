import torch
from torch import nn
import sys
from . import models
from . import ctc
from .utils import *
import torch.optim as optim
import numpy as np
import time
from torch.optim.lr_scheduler import ReduceLROnPlateau
import os
import pickle

from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix
from sklearn.metrics import precision_recall_fscore_support
from sklearn.metrics import accuracy_score, f1_score
from .eval_metrics import *


####################################################################
#
# Construct the model and the CTC module (which may not be needed)
#
####################################################################

def get_CTC_module(hyp_params):
    a2l_module = getattr(ctc, 'CTCModule')(in_dim=hyp_params.orig_d_a, out_seq_len=hyp_params.l_len)
    v2l_module = getattr(ctc, 'CTCModule')(in_dim=hyp_params.orig_d_v, out_seq_len=hyp_params.l_len)
    return a2l_module, v2l_module

def tsf_initiate(hyp_params):
    print("1111111")
    model = getattr(models, hyp_params.model+'Model')(hyp_params)
    print("222222")

    if hyp_params.use_cuda:
        model = model.cuda()

    optimizer = getattr(optim, hyp_params.optim)(model.parameters(), lr=hyp_params.lr)
    criterion = getattr(nn, hyp_params.criterion)()
    if hyp_params.aligned or hyp_params.model=='MULT':
        ctc_criterion = None
        ctc_a2l_module, ctc_v2l_module = None, None
        ctc_a2l_optimizer, ctc_v2l_optimizer = None, None
    else:
        from warpctc_pytorch import CTCLoss
        ctc_criterion = CTCLoss()
        ctc_a2l_module, ctc_v2l_module = get_CTC_module(hyp_params)
        if hyp_params.use_cuda:
            ctc_a2l_module, ctc_v2l_module = ctc_a2l_module.cuda(), ctc_v2l_module.cuda()
        ctc_a2l_optimizer = getattr(optim, hyp_params.optim)(ctc_a2l_module.parameters(), lr=hyp_params.lr)
        ctc_v2l_optimizer = getattr(optim, hyp_params.optim)(ctc_v2l_module.parameters(), lr=hyp_params.lr)
    
    scheduler = ReduceLROnPlateau(optimizer, mode='min', patience=hyp_params.when, factor=0.1, verbose=True)
    settings = {'model': model,
                'optimizer': optimizer,
                'criterion': criterion,
                'ctc_a2l_module': ctc_a2l_module,
                'ctc_v2l_module': ctc_v2l_module,
                'ctc_a2l_optimizer': ctc_a2l_optimizer,
                'ctc_v2l_optimizer': ctc_v2l_optimizer,
                'ctc_criterion': ctc_criterion,
                'scheduler': scheduler}
    return settings


####################################################################
#
# Training and evaluation scripts
#
####################################################################

def train_model(model,settings, hyp_params, train_loader, valid_loader, test_loader):
    criterion = settings['criterion']    
    
    ctc_a2l_module = settings['ctc_a2l_module']
    ctc_v2l_module = settings['ctc_v2l_module']
    ctc_a2l_optimizer = settings['ctc_a2l_optimizer']
    ctc_v2l_optimizer = settings['ctc_v2l_optimizer']
    ctc_criterion = settings['ctc_criterion']
    
    

    def train(model, criterion, ctc_a2l_module, ctc_v2l_module, ctc_a2l_optimizer, ctc_v2l_optimizer, ctc_criterion):
        epoch_loss = 0
        proc_loss, proc_size = 0, 0
        for i_batch, (batch_X, batch_Y) in enumerate(train_loader):
            sample_ind, text, audio, vision = batch_X
            eval_attr = batch_Y.squeeze(-1)   # if num of labels is 1

            # model.zero_grad()
            # if ctc_criterion is not None:
            #     ctc_a2l_module.zero_grad()
            #     ctc_v2l_module.zero_grad()
                
            if hyp_params.use_cuda:
                with torch.cuda.device(0):
                    text, audio, vision, eval_attr = text.cuda(), audio.cuda(), vision.cuda(), eval_attr.cuda()
                    if hyp_params.dataset == 'iemocap':
                        eval_attr = eval_attr.long()
            
            batch_size = text.size(0)
            batch_chunk = hyp_params.batch_chunk
                
            combined_loss = 0
            net = nn.DataParallel(model) if batch_size > 10 else model
            if batch_chunk > 1:
                raw_loss = combined_loss = 0
                text_chunks = text.chunk(batch_chunk, dim=0)
                audio_chunks = audio.chunk(batch_chunk, dim=0)
                vision_chunks = vision.chunk(batch_chunk, dim=0)
                eval_attr_chunks = eval_attr.chunk(batch_chunk, dim=0)
                
                for i in range(batch_chunk):
                    text_i, audio_i, vision_i = text_chunks[i], audio_chunks[i], vision_chunks[i]
                    eval_attr_i = eval_attr_chunks[i]
                    preds_i, hiddens_i = net(text_i, audio_i, vision_i)
                    
                    if hyp_params.dataset == 'iemocap':
                        preds_i = preds_i.view(-1, 2)
                        eval_attr_i = eval_attr_i.view(-1)
                    raw_loss_i = criterion(preds_i, eval_attr_i) / batch_chunk
                    raw_loss += raw_loss_i
                    raw_loss_i.backward()
            else:
                print(text.shape)
                feature = net(text, audio, vision)
                
            

                
        return feature

    return train(model, criterion, ctc_a2l_module, ctc_v2l_module, ctc_a2l_optimizer, ctc_v2l_optimizer, ctc_criterion)
    # best_valid = 1e8
    # for epoch in range(1, hyp_params.num_epochs+1):
    #     start = time.time()
    #     train(model, optimizer, criterion, ctc_a2l_module, ctc_v2l_module, ctc_a2l_optimizer, ctc_v2l_optimizer, ctc_criterion)
        
    #     end = time.time()
    #     duration = end-start
    #     scheduler.step(val_loss)    # Decay learning rate by validation loss

    #     print("-"*50)
    #     print('Epoch {:2d} | Time {:5.4f} sec | Valid Loss {:5.4f} | Test Loss {:5.4f}'.format(epoch, duration, val_loss, test_loss))
    #     print("-"*50)
        
    #     if val_loss < best_valid:
    #         print(f"Saved model at pre_trained_models/{hyp_params.name}.pt!")
    #         save_model(hyp_params, model, name=hyp_params.name)
    #         best_valid = val_loss

    # model = load_model(hyp_params, name=hyp_params.name)
    # _, results, truths = evaluate(model, ctc_a2l_module, ctc_v2l_module, criterion, test=True)

    # if hyp_params.dataset == "mosei_senti":
    #     eval_mosei_senti(results, truths, True)
    # elif hyp_params.dataset == 'mosi':
    #     eval_mosi(results, truths, True)
    # elif hyp_params.dataset == 'iemocap':
    #     eval_iemocap(results, truths)

    # sys.stdout.flush()
    # input('[Press Any Key to start another run]')
