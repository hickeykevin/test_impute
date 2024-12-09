#!/usr/bin/env python
# coding: utf-8

#from torch.utils.data import Dataset
import os
import numpy as np
import pandas as pd
import math
from sklearn import metrics
# from imblearn.metrics import specificity_score
# from utils import makedir
import re
from sklearn.preprocessing import MinMaxScaler

#Model 
import torch
import torch.nn as nn                     # general structure of a net 
from torch.nn.parameter import Parameter  # for weights and bias
import torch.nn.functional as F           # for Relu
from torch.autograd import Variable
from torch.utils.data import Dataset, DataLoader # to data loader
#from torch.utils import data
import torch.optim as optim

import json                              # import data from json file
import argparse

 

def binary_cross_entropy_with_logits(input, target, weight=None, size_average=True, reduce=True):
    if not (target.size() == input.size()):
        raise ValueError("Target size ({}) must be the same as input size ({})".format(target.size(), input.size()))

    max_val = (-input).clamp(min=0)
    loss = input - input * target + max_val + ((-max_val).exp() + (-input - max_val).exp()).log()

    if weight is not None:
        loss = loss * weight

    if not reduce:
        return loss
    elif size_average:
        return loss.mean()
    else:
        return loss.sum()

class FeatureRegression(nn.Module):
    def __init__(self, input_size):
        super(FeatureRegression, self).__init__()
        self.build(input_size)

    def build(self, input_size):
        self.W = Parameter(torch.Tensor(input_size, input_size))
        self.b = Parameter(torch.Tensor(input_size))

        m = torch.ones(input_size, input_size) - torch.eye(input_size, input_size)
        self.register_buffer('m', m)

        self.reset_parameters()

    def reset_parameters(self):
        stdv = 1. / math.sqrt(self.W.size(0))
        self.W.data.uniform_(-stdv, stdv)
        if self.b is not None:
            self.b.data.uniform_(-stdv, stdv)

    def forward(self, x):
        z_h = F.linear(x, self.W * Variable(self.m), self.b)
        return z_h

class TemporalDecay(nn.Module):
    def __init__(self, input_size, output_size, diag = False):
        super(TemporalDecay, self).__init__()
        self.diag = diag

        self.build(input_size, output_size)

    def build(self, input_size, output_size):
        self.W = Parameter(torch.Tensor(output_size, input_size))
        self.b = Parameter(torch.Tensor(output_size))

        if self.diag == True:
            assert(input_size == output_size)
            m = torch.eye(input_size, input_size)
            self.register_buffer('m', m)

        self.reset_parameters()

    def reset_parameters(self):
        stdv = 1. / math.sqrt(self.W.size(0))
        self.W.data.uniform_(-stdv, stdv)
        if self.b is not None:
            self.b.data.uniform_(-stdv, stdv)

    def forward(self, d):
        if self.diag == True:
            gamma = F.relu(F.linear(d, self.W * Variable(self.m), self.b))
        else:
            gamma = F.relu(F.linear(d, self.W, self.b))
        gamma = torch.exp(-gamma)
        return gamma

class Model_rits(nn.Module):
    def __init__(self, rnn_name='LSTM'):
        super(Model_rits, self).__init__()
        self.rnn_name = rnn_name
        self.build()

    def build(self):
        if self.rnn_name=='LSTM':
            self.rnn_cell = nn.LSTMCell(N_SERIES * 2, RNN_HID_SIZE)
        elif self.rnn_name=='GRU':
            self.rnn_cell = nn.GRUCell(N_SERIES * 2, RNN_HID_SIZE)

        self.temp_decay_h = TemporalDecay(input_size = N_SERIES, output_size = RNN_HID_SIZE, diag = False)
        self.temp_decay_x = TemporalDecay(input_size = N_SERIES, output_size = N_SERIES, diag = True)

        self.hist_reg = nn.Linear(RNN_HID_SIZE, N_SERIES)
        self.feat_reg = FeatureRegression(N_SERIES)

        self.weight_combine = nn.Linear(N_SERIES * 2, N_SERIES)

        self.dropout = nn.Dropout(p = 0.25)
        self.out = nn.Linear(RNN_HID_SIZE, 1)

    
    def _assemble_input_for_training(self, data): 

        """
        Collate function for the BRITS dataloader.

        Args:
            data (List[Dict]): List of records containing time series data from BRITSDataFormat.

        Returns:
            Dict: A dictionary containing the collated data.

        Raises:
            AssertionError: If the required keys are not found in the input list.
        """
        # assuming data is a dict of tensors with keys: 'X', 'missing_mask', 'deltas', 'back_X', 'back_missing_mask', 'back_deltas', 'label'
        for k, v in data.items():
            if k == 'label':
                data[k] = v.long()
            else:
                data[k] = v.type_as(next(iter(self.temp_decay_h.parameters())))
        final_dict = {
            'forward': {"X": data['X'], "missing_mask": data['missing_mask'], "deltas": data['deltas']}, #TODO: check if this is correct
            'backward': {"X": data['back_X'], "missing_mask": data['back_missing_mask'], "deltas": data['back_deltas']},
            'label': data['label']
        }
        return final_dict 

    def forward(self, data, direct):
        # Original sequence with 24 time steps
        values = data[direct]['values']
        masks = data[direct]['masks']
        deltas = data[direct]['deltas']

        evals = data[direct]['evals']
        eval_masks = data[direct]['eval_masks']

        labels = data['label'].view(-1, 1)
        is_train = data['is_train'].view(-1, 1)

        h = Variable(torch.zeros((values.size()[0], RNN_HID_SIZE)))
        c = Variable(torch.zeros((values.size()[0], RNN_HID_SIZE)))

        #if torch.cuda.is_available():
        #    h, c = h.cuda(), c.cuda()

        x_loss = 0.0
        y_loss = 0.0

        imputations = []

        for t in range(SEQ_LEN):
            x = values[:, t, :]
            m = masks[:, t, :]
            d = deltas[:, t, :]

            gamma_h = self.temp_decay_h(d)
            gamma_x = self.temp_decay_x(d)

            h = h * gamma_h

            x_h = self.hist_reg(h)
            x_loss += torch.sum(torch.abs(x - x_h) * m) / (torch.sum(m) + 1e-5)

            x_c =  m * x +  (1 - m) * x_h

            z_h = self.feat_reg(x_c)
            x_loss += torch.sum(torch.abs(x - z_h) * m) / (torch.sum(m) + 1e-5)

            alpha = self.weight_combine(torch.cat([gamma_x, m], dim = 1))

            c_h = alpha * z_h + (1 - alpha) * x_h
            x_loss += torch.sum(torch.abs(x - c_h) * m) / (torch.sum(m) + 1e-5)

            c_c = m * x + (1 - m) * c_h

            inputs = torch.cat([c_c, m], dim = 1)
            if self.rnn_name=='LSTM':
                h, c = self.rnn_cell(inputs, (h, c))
            elif self.rnn_name=='GRU':
                h = self.rnn_cell(inputs, h)

            imputations.append(c_c.unsqueeze(dim = 1))

        imputations = torch.cat(imputations, dim = 1)

        y_h = self.out(h)
        y_loss = binary_cross_entropy_with_logits(y_h, labels, reduce = False)
        y_loss = torch.sum(y_loss * is_train) / (torch.sum(is_train) + 1e-5)

        y_h = torch.sigmoid(y_h)

        return {'loss': x_loss / SEQ_LEN + y_loss * 0.1, 'predictions': y_h,\
                'imputations': imputations, 'label': labels, 'is_train': is_train,\
                'evals': evals, 'eval_masks': eval_masks}

    def run_on_batch(self, data, optimizer):
        ret = self(data, direct = 'forward')

        if optimizer is not None:
            optimizer.zero_grad()
            ret['loss'].backward()
            optimizer.step()

        return ret

#--- Run BRITS model using rits model
class Model_brits(nn.Module):
    def __init__(self, rnn_name='LSTM'):
        super(Model_brits, self).__init__()
        self.rnn_name = rnn_name
        self.build()

    def build(self):
        self.rits_f = Model_rits(self.rnn_name)
        self.rits_b = Model_rits(self.rnn_name)

    def forward(self, data):
        ret_f = self.rits_f(data, 'forward')
        ret_b = self.reverse(self.rits_b(data, 'backward'))

        ret = self.merge_ret(ret_f, ret_b)

        return ret

    def merge_ret(self, ret_f, ret_b):
        loss_f = ret_f['loss']
        loss_b = ret_b['loss']
        loss_c = self.get_consistency_loss(ret_f['imputations'], ret_b['imputations'])

        loss = loss_f + loss_b + loss_c

        predictions = (ret_f['predictions'] + ret_b['predictions']) / 2
        imputations = (ret_f['imputations'] + ret_b['imputations']) / 2

        ret_f['loss'] = loss
        ret_f['predictions'] = predictions
        ret_f['imputations'] = imputations

        return ret_f

    def get_consistency_loss(self, pred_f, pred_b):
        loss = torch.pow(pred_f - pred_b, 2.0).mean()
        return loss

    def reverse(self, ret):
        def reverse_tensor(tensor_):
            if tensor_.dim() <= 1:
                return tensor_
            indices = range(tensor_.size()[1])[::-1]
            indices = Variable(torch.LongTensor(indices), requires_grad = False)

            #if torch.cuda.is_available():
            #    indices = indices.cuda()

            return tensor_.index_select(1, indices)

        for key in ret:
            ret[key] = reverse_tensor(ret[key])

        return ret

    def run_on_batch(self, data, optimizer):
        ret = self(data)

        if optimizer is not None:
            optimizer.zero_grad()
            ret['loss'].backward()
            optimizer.step()

        return ret

#--- Run RITS model with Attentions


class Model_rits_att(nn.Module):
    def __init__(self, rnn_name='LSTM'):
        super(Model_rits_att, self).__init__()
        self.rnn_name = rnn_name
        self.build()
        
        # Attention following AudiBert 
        self.W_s1 = nn.Linear(RNN_HID_SIZE, 350)
        self.W_s2 = nn.Linear(350, 30)
    def build(self):
        if self.rnn_name=='LSTM':
            self.rnn_cell = nn.LSTMCell(N_SERIES * 2, RNN_HID_SIZE)
        elif self.rnn_name=='GRU':
            self.rnn_cell = nn.GRUCell(N_SERIES * 2, RNN_HID_SIZE)

        self.temp_decay_h = TemporalDecay(input_size = N_SERIES, output_size = RNN_HID_SIZE, diag = False)
        self.temp_decay_x = TemporalDecay(input_size = N_SERIES, output_size = N_SERIES, diag = True)

        self.hist_reg = nn.Linear(RNN_HID_SIZE, N_SERIES)
        self.feat_reg = FeatureRegression(N_SERIES)

        self.weight_combine = nn.Linear(N_SERIES * 2, N_SERIES)

        self.dropout = nn.Dropout(p = 0.25)
        #self.out = nn.Linear(RNN_HID_SIZE, 1)
        self.out = nn.Linear(RNN_HID_SIZE*30, 1)
        
    def attention_rnn(self, rnn_output):
        #attn_weight_matrix = self.W_s2(F.tanh(self.W_s1(lstm_output)))
        attn_weight_matrix = self.W_s2(torch.tanh(self.W_s1(rnn_output)))
        attn_weight_matrix = attn_weight_matrix.permute(0, 2, 1)
        attn_weight_matrix = F.softmax(attn_weight_matrix, dim=2)
        return attn_weight_matrix
    
    def _assemble_input_for_training(self, data): 

        """
        Collate function for the BRITS dataloader.

        Args:
            data (List[Dict]): List of records containing time series data from BRITSDataFormat.

        Returns:
            Dict: A dictionary containing the collated data.

        Raises:
            AssertionError: If the required keys are not found in the input list.
        """
        # assuming data is a dict of tensors with keys: 'X', 'missing_mask', 'deltas', 'back_X', 'back_missing_mask', 'back_deltas', 'label'
        for k, v in data.items():
            if k == 'label':
                data[k] = v.long()
            else:
                data[k] = v.type_as(next(iter(self.W_s1.parameters())))
        final_dict = {
            'forward': {"X": data['X'], "missing_mask": data['missing_mask'], "deltas": data['deltas']}, #TODO: check if this is correct
            'backward': {"X": data['back_X'], "missing_mask": data['back_missing_mask'], "deltas": data['back_deltas']},
            'label': data['label']
        }
        return final_dict 

    def forward(self, data, direct):
        # Original sequence with 24 time steps
        values = data[direct]['values']
        #print('values.shape:',values.shape)          #torch.Size([1, 40, 12])
        masks = data[direct]['masks']
        deltas = data[direct]['deltas']
        
        # to store historical hidden size from rnn
        H_rnn = torch.zeros(values.shape[0], SEQ_LEN, RNN_HID_SIZE)  # (batch, sequence, hiden_dize)

        evals = data[direct]['evals']
        eval_masks = data[direct]['eval_masks']

        labels = data['label'].view(-1, 1)
        is_train = data['is_train'].view(-1, 1)

        h = Variable(torch.zeros((values.size()[0], RNN_HID_SIZE)))
        c = Variable(torch.zeros((values.size()[0], RNN_HID_SIZE)))

        #if torch.cuda.is_available():
        #    h, c = h.cuda(), c.cuda()
        x_loss = 0.0
        y_loss = 0.0

        imputations = []

        for t in range(SEQ_LEN):
            x = values[:, t, :] # every sample in batch, t'th time step, all features
            #print('x.shape:',x.shape)     # x.shape: torch.Size([1, 12])
            m = masks[:, t, :] # every sample in batch, t'th time step, all features
            d = deltas[:, t, :] # every sample in batch, t'th time step, all features

            gamma_h = self.temp_decay_h(d) # reshaped deltas to rnn_hidden_size
            gamma_x = self.temp_decay_x(d) # reshaped deltas to number time steps

            h = h * gamma_h

            x_h = self.hist_reg(h) # reshapes hidden state to (number of features)
            x_loss += torch.sum(torch.abs(x - x_h) * m) / (torch.sum(m) + 1e-5)
            # calculates avg absolute error for present predicted values (Based on hidden state) and actual values

            x_c =  m * x +  (1 - m) * x_h # this just puts the real values with the predicted values together

            z_h = self.feat_reg(x_c) # some? reshaping shape: (bs, n_features), all values
            x_loss += torch.sum(torch.abs(x - z_h) * m) / (torch.sum(m) + 1e-5) # some loss of real values and imputed values for the index of real values

            alpha = self.weight_combine(torch.cat([gamma_x, m], dim = 1)) # shape: (bs, n_features) all values

            c_h = alpha * z_h + (1 - alpha) * x_h
            x_loss += torch.sum(torch.abs(x - c_h) * m) / (torch.sum(m) + 1e-5)

            c_c = m * x + (1 - m) * c_h

            inputs = torch.cat([c_c, m], dim = 1)
            if self.rnn_name=='LSTM':
                h, c = self.rnn_cell(inputs, (h, c))         # h lstm: torch.Size([1, 32]
                #print('h lstm:', h.shape)  
            elif self.rnn_name=='GRU':
                h = self.rnn_cell(inputs, h)                 # h GRU: torch.Size([1, 32]
                #print('h GRU:', h.shape)
            H_rnn[:,t,:] = h
            imputations.append(c_c.unsqueeze(dim = 1))

        imputations = torch.cat(imputations, dim = 1)
        
        # Attentions 
        attn_weight_matrix = self.attention_rnn(H_rnn)
        hidden_matrix = torch.bmm(attn_weight_matrix, H_rnn)
        attention_output = hidden_matrix.view(-1, hidden_matrix.size()[1]*hidden_matrix.size()[2])
        #print('attention_output.shape:',attention_output.shape)
        #print('h.shape:',h.shape)

        #y_h = self.out(h)
        y_h = self.out(attention_output)
        y_loss = binary_cross_entropy_with_logits(y_h, labels, reduce = False)
        y_loss = torch.sum(y_loss * is_train) / (torch.sum(is_train) + 1e-5)

        y_h = torch.sigmoid(y_h)

        return {'loss_imp': x_loss / SEQ_LEN, 'loss_clf': y_loss * 0.1, "loss": (x_loss / SEQ_LEN + y_loss * 0.1), 
                'predictions': y_h,\
                'imputations': imputations, 'label': labels, 'is_train': is_train,\
                'evals': evals, 'eval_masks': eval_masks}
    
    def forward2(self, data, direct):
        is_train = data['is_train'].view(-1, 1)
        non_zero = torch.where(is_train == 1)[0]
        # Original sequence with 24 time steps
        values = data[direct]['values'][non_zero]
        #print('values.shape:',values.shape)          #torch.Size([1, 40, 12])
        masks = data[direct]['masks'][non_zero]
        deltas = data[direct]['deltas'][non_zero]
        
        # to store historical hidden size from rnn
        H_rnn = torch.zeros(values.shape[0], SEQ_LEN, RNN_HID_SIZE)  # (batch, sequence, hiden_dize)

        evals = data[direct]['evals']
        eval_masks = data[direct]['eval_masks']

        labels = data['label'].view(-1, 1)[non_zero]

        h = Variable(torch.zeros((values.size()[0], RNN_HID_SIZE)))
        c = Variable(torch.zeros((values.size()[0], RNN_HID_SIZE)))

        #if torch.cuda.is_available():
        #    h, c = h.cuda(), c.cuda()
        x_loss = 0.0
        y_loss = 0.0

        imputations = []

        for t in range(SEQ_LEN):
            x = values[:, t, :] # every sample in batch, t'th time step, all features
            #print('x.shape:',x.shape)     # x.shape: torch.Size([1, 12])
            m = masks[:, t, :] # every sample in batch, t'th time step, all features
            d = deltas[:, t, :] # every sample in batch, t'th time step, all features

            gamma_h = self.temp_decay_h(d) # reshaped deltas to rnn_hidden_size
            gamma_x = self.temp_decay_x(d) # reshaped deltas to number time steps

            h = h * gamma_h

            x_h = self.hist_reg(h) # reshapes hidden state to (number of features)
            x_loss += torch.sum(torch.abs(x - x_h) * m) / (torch.sum(m) + 1e-5)
            # calculates avg absolute error for present predicted values (Based on hidden state) and actual values

            x_c =  m * x +  (1 - m) * x_h # this just puts the real values with the predicted values together

            z_h = self.feat_reg(x_c) # some? reshaping shape: (bs, n_features), all values
            x_loss += torch.sum(torch.abs(x - z_h) * m) / (torch.sum(m) + 1e-5) # some loss of real values and imputed values for the index of real values

            alpha = self.weight_combine(torch.cat([gamma_x, m], dim = 1)) # shape: (bs, n_features) all values

            c_h = alpha * z_h + (1 - alpha) * x_h
            x_loss += torch.sum(torch.abs(x - c_h) * m) / (torch.sum(m) + 1e-5)

            c_c = m * x + (1 - m) * c_h

            inputs = torch.cat([c_c, m], dim = 1)
            if self.rnn_name=='LSTM':
                h, c = self.rnn_cell(inputs, (h, c))         # h lstm: torch.Size([1, 32]
                #print('h lstm:', h.shape)  
            elif self.rnn_name=='GRU':
                h = self.rnn_cell(inputs, h)                 # h GRU: torch.Size([1, 32]
                #print('h GRU:', h.shape)
            H_rnn[:,t,:] = h
            imputations.append(c_c.unsqueeze(dim = 1))

        imputations = torch.cat(imputations, dim = 1)
        
        # Attentions 
        attn_weight_matrix = self.attention_rnn(H_rnn)
        hidden_matrix = torch.bmm(attn_weight_matrix, H_rnn)
        attention_output = hidden_matrix.view(-1, hidden_matrix.size()[1]*hidden_matrix.size()[2])
        #print('attention_output.shape:',attention_output.shape)
        #print('h.shape:',h.shape)

        #y_h = self.out(h)
        y_h = self.out(attention_output)
        y_loss = binary_cross_entropy_with_logits(y_h, labels, reduce = False)
        y_loss = torch.sum(y_loss * is_train[non_zero]) / (torch.sum(is_train[non_zero]) + 1e-5)

        y_h = torch.sigmoid(y_h)

        return {'loss_imp': x_loss / SEQ_LEN, 'loss_clf': y_loss * 0.1, "loss": x_loss/SEQ_LEN + y_loss*0.1,  
                'predictions': y_h,
                'imputations': imputations, 
                'label': labels, 'is_train': is_train,\
                'evals': evals, 'eval_masks': eval_masks}

    def run_on_batch(self, data, optimizer):
        ret = self(data, direct = 'forward')

        if optimizer is not None:
            optimizer.zero_grad()
            ret['loss'].backward()
            optimizer.step()

        return ret
    


#--- Run BRITS model with Attentions 
class Model_brits_att(nn.Module):
    def __init__(self, rnn_name='LSTM'):
        super(Model_brits_att, self).__init__()
        self.rnn_name = rnn_name
        self.build()

    def build(self):
        self.rits_f = Model_rits_att(self.rnn_name)
        self.rits_b = Model_rits_att(self.rnn_name)

    def forward(self, data):
        ret_f = self.rits_f(data, 'forward')
        ret_b = self.reverse(self.rits_b(data, 'backward'))

        ret = self.merge_ret(ret_f, ret_b) #TODO finish making "2" so you can check if yloss is same for full vs nonzero, and if x_loss is same, gradient update same/different

        return ret
    
    def forward2(self, data):
        ret_f = self.rits_f.forward2(data, 'forward')
        ret_b = self.reverse(self.rits_b.forward2(data, 'backward'))

        ret = self.merge_ret2(ret_f, ret_b)

        return ret

    def merge_ret2(self, ret_f, ret_b):
        loss_f = ret_f['loss']
        loss_b = ret_b['loss']
        loss_c = self.get_consistency_loss(ret_f['imputations'], ret_b['imputations'])

        loss = loss_f + loss_b + loss_c

        predictions = (ret_f['predictions'] + ret_b['predictions']) / 2
        imputations = (ret_f['imputations'] + ret_b['imputations']) / 2

        ret_f['loss'] = loss
        ret_f['predictions'] = predictions
        ret_f['imputations'] = imputations

        return ret_f
    def merge_ret(self, ret_f, ret_b):
        loss_f = ret_f['loss']
        loss_b = ret_b['loss']
        loss_c = self.get_consistency_loss(ret_f['imputations'], ret_b['imputations'])

        loss = loss_f + loss_b + loss_c

        predictions = (ret_f['predictions'] + ret_b['predictions']) / 2
        imputations = (ret_f['imputations'] + ret_b['imputations']) / 2

        ret_f['loss'] = loss
        ret_f['predictions'] = predictions
        ret_f['imputations'] = imputations

        return ret_f

    def get_consistency_loss(self, pred_f, pred_b):
        loss = torch.pow(pred_f - pred_b, 2.0).mean()
        return loss

    def reverse(self, ret):
        def reverse_tensor(tensor_):
            if tensor_.dim() <= 1:
                return tensor_
            indices = range(tensor_.size()[1])[::-1]
            indices = Variable(torch.LongTensor(indices), requires_grad = False)

            #if torch.cuda.is_available():
            #    indices = indices.cuda()

            return tensor_.index_select(1, indices)

        for key in ret:
            ret[key] = reverse_tensor(ret[key])

        return ret

    def run_on_batch(self, data, optimizer):
        ret1 = self.forward(data)
        ret2 = self.forward2(data)
        if optimizer is not None:
            optimizer.zero_grad()
            ret1['loss_imp'].backward(retain_graph=True)
            gradients_loss1_imp = [param.grad.clone() for name, param in self.named_parameters() if param.grad is not None]
            
            optimizer.zero_grad()
            ret1['loss_clf'].backward(retain_graph=True)
            gradients_loss1_clf = [param.grad.clone() for name, param in self.named_parameters() if param.grad is not None]
            optimizer.zero_grad()

            ret2['loss_imp'].backward(retain_graph=True)
            gradients_loss2_imp = [param.grad.clone() for name, param in self.named_parameters() if param.grad is not None]
            optimizer.zero_grad()

            ret2['loss_clf'].backward(retain_graph=True)
            gradients_loss2_clf = [param.grad.clone() for name, param in self.named_parameters() if param.grad is not None]
            optimizer.zero_grad()

            import pdb; pdb.set_trace()
            ret1['loss'].backward(retain_graph=True)
            gradients_loss_full1 = [param.grad.clone() for name, param in self.named_parameters() if param.grad is not None]
            optimizer.zero_grad()

            ret2['loss'].backward(retain_graph=True)
            gradients_loss_full2 = [param.grad.clone() for name, param in self.named_parameters() if param.grad is not None]


            # check if gradients for imp_loss on ret1 and ret2 are the same
            import pdb; pdb.set_trace()
            from itertools import zip_longest
            print(f"Are is_train=0 present in batch?: {torch.any(data['is_train'] == 0)}")
            # loop through zipeed gradients_loss1_imp and gradients_loss2_imp, along with zipped gradients_loss1_clf and gradients_loss2_clf
            for i, ((grad1_imp, grad2_imp), (grad1_clf, grad2_clf), (gradfull1, gradfull2)) in enumerate(zip_longest(zip(gradients_loss1_imp, gradients_loss2_imp), zip(gradients_loss1_clf, gradients_loss2_clf), zip(gradients_loss_full1, gradients_loss_full2))):
                layer = list(self.named_parameters())[i][0]
                print(f"Are all grads equal for layer {layer} with respect to imp_loss?: {torch.allclose(grad1_imp, grad2_imp, 1e-2)}")
                print(f"Are all grads equal for layer {layer} with respect to for clf_loss?: {torch.allclose(grad1_clf, grad2_clf, 1e-2)}")
                print(f"Are all grads equal for layer {layer} with respect to full_loss?: {torch.allclose(gradfull1, gradfull2, 1e-2)}")
                print()
            

        return ret1, ret2

# ----- function to transform to Variable ----#
def to_var(var):
    if torch.is_tensor(var):
        var = Variable(var)
        #if torch.cuda.is_available():
        #    var = var.cuda()
        return var
    if isinstance(var, int) or isinstance(var, float) or isinstance(var, str):
        return var
    if isinstance(var, dict):
        for key in var:
            var[key] = to_var(var[key])
        return var
    if isinstance(var, list):
        var = map(lambda x: to_var(x), var)
        return var

#------ Evaluate function -----------#
def evaluate(model, val_iter, use_ricardo_loader: bool):
    model.eval()
    labels = []
    preds = []

    evals = []
    imputations = []
    params = next(iter(model.parameters()))
    for idx, data in enumerate(val_iter):
        if not use_ricardo_loader:
            data = assemble_input_for_training(data, params=params)
        # data = to_var(data)
        ret = model.run_on_batch(data, None)

        pred = ret['predictions'].data.cpu().numpy()
        label = ret['label'].data.cpu().numpy()
        is_train = ret['is_train'].data.cpu().numpy()

        eval_masks = ret['eval_masks'].data.cpu().numpy()
        eval_ = ret['evals'].data.cpu().numpy()
        imputation = ret['imputations'].data.cpu().numpy()

        evals += eval_[np.where(eval_masks == 1)].tolist()
        imputations += imputation[np.where(eval_masks == 1)].tolist()

        # collect test label & prediction
        pred = pred[np.where(is_train == 0)]
        label = label[np.where(is_train == 0)]

        labels += label.tolist()
        preds += pred.tolist()

    labels = np.asarray(labels).astype('int32') 
    preds = np.asarray(preds)
    dummy_pred = np.zeros_like(preds)
    dummy_pred[preds >= 0.5] = 1

    AUC = metrics.roc_auc_score(labels, preds)
    BA = metrics.balanced_accuracy_score(labels,dummy_pred)
    F1 = metrics.f1_score(labels,dummy_pred)
    Recall = metrics.recall_score(labels,dummy_pred)
    # SF = specificity_score(labels,dummy_pred)

    evals = np.asarray(evals)
    imputations = np.asarray(imputations)
    #print('imputations:', imputations)

    if len(imputations)!=0:  
        MAE = np.abs(evals - imputations).mean()
        MRE = np.abs(evals - imputations).sum() / np.abs(evals).sum()
    else: # when the missing ration is setting to zero
        MAE = 0
        MRE = 0
    
    #print ('MAE', np.abs(evals - imputations).mean())
    return MAE, MRE, AUC, BA, F1, Recall, 


def assemble_input_for_training(data, params): 

    """
    Collate function for the BRITS dataloader.

    Args:
        data (List[Dict]): List of records containing time series data from BRITSDataFormat.

    Returns:
        Dict: A dictionary containing the collated data.

    Raises:
        AssertionError: If the required keys are not found in the input list.
    """
    # assuming data is a dict of tensors with keys: 'X', 'missing_mask', 'deltas', 'back_X', 'back_missing_mask', 'back_deltas', 'label'
    for k, v in data.items():
        if k == 'label':
            data[k] = v.long()
        else:
            data[k] = v.type_as(params)
    final_dict = {
        'forward': {
            "values": data['X'], 
            "masks": data['missing_mask'], 
            "deltas": data['deltas'], 
            'evals': data['X_ori'], 
            'eval_masks': data['indicating_mask']
            },
        
        'backward': {
            "values": data['back_X'], 
            "masks": data['back_missing_mask'], 
            "deltas": data['back_deltas'], 
            'evals': torch.flip(data['X_ori'], dims=[1]),
            'eval_masks': torch.flip(data['indicating_mask'], dims=[1])
            },
        
        'label': data['label'],
        'is_train': data['is_train']
    }
    return final_dict


#----- function to transform to Variable ----#
def to_var(var):
    if torch.is_tensor(var):
        var = Variable(var)
        #if torch.cuda.is_available():
        #    var = var.cuda()
        return var
    if isinstance(var, int) or isinstance(var, float) or isinstance(var, str):
        return var
    if isinstance(var, dict):
        for key in var:
            var[key] = to_var(var[key])
        return var
    if isinstance(var, list):
        var = map(lambda x: to_var(x), var)
        return var
    
def miss_f(x, n_miss, random = True): 
    x1 = x.copy()
    #print('len(x1):', len(x1))   # 40
    if random:
        # index for missing
        m_indx = np.random.choice(list(range(len(x1))), n_miss, replace= False)
    else:
        np.random.seed(14)
        m_indx = np.random.choice(list(range(len(x1))), n_miss, replace= False)
        
    eval_mask1 = np.zeros(len(x))
    mask1 = np.ones(len(x))
    for i in m_indx:
        x1[i] = 0 
        eval_mask1[i] = 1
        mask1[i] = 0
    
    return x1, eval_mask1.tolist(), mask1

def miss_f2(x, n_miss, random = True):
    # Consecutive Missing Values (CMV) 
    x1 = x.copy()
    #print('len(x1):', len(x1))   # 40
    if random:
        # index for missing
        seq_len0 = len(x1)
        m_indx0 = np.random.choice(list(range(seq_len0)), 1, replace= False)[0]
        if (m_indx0 + n_miss) > seq_len0: # superior corner problem
            m_indx0 = m_indx0 - n_miss + 1
        m_indx = list(range(m_indx0,m_indx0+n_miss))
    else:
        np.random.seed(14)
        m_indx = np.random.choice(list(range(len(x1))), n_miss, replace= False)
        
    eval_mask1 = np.zeros(len(x))
    mask1 = np.ones(len(x))
    for i in m_indx:
        x1[i] = 0 
        eval_mask1[i] = 1
        mask1[i] = 0
    return x1, eval_mask1.tolist(), mask1


def mask_f(x):
    x2 = np.zeros((1, len(x)))
    for i, dx in enumerate(x):
        if dx != 0:
            x2[0, i] = 1

    return x2.tolist()[0]
    
def delta_f(x):
    d1 = np.zeros(len(x))
    m1 = mask_f(x)
    count = 0
    for i in range(len(x)):
        if i > 0 and m1[i] == 1:
            d1[i] = count
        elif i > 0 and m1[i] == 0:
            count += 1
            d1[i] = count
        else:
            d1[i] = 0
    
    return d1

def get_data(x, y, n_miss, type_missing='Random'):
    # input : x[n_features[n_tiempo]]
    n_feature = len(x) 
    #print('n_feature:',n_feature)  # 12
    n_time = len(x[0])            
    #print('n_time:', n_time)       # 40
    
    # forwards: initial values
    x0 = np.zeros((n_feature, n_time))
    values_f = x0.copy()
    masks_f = x0.copy()
    deltas_f = x0.copy()
    
    evals_f = x0.copy() 
    eval_masks_f = x0.copy()
    
    # backward: initial values
    values_b = x0.copy() 
    masks_b = x0.copy()
    deltas_b = x0.copy()
    
    evals_b = x0.copy() 
    eval_masks_b = x0.copy()
    
    for i in range(n_feature):
        if type_missing=='Random':
            x_miss = miss_f(x[i], n_miss, True)
        elif type_missing=='CMV': # Consecutive Missing Values (CMV)
            x_miss = miss_f2(x[i], n_miss, True) 
        # Forwards
        values_f[i,:] = x_miss[0]
        #masks_f[i,:] = mask_f(x_miss[0])
        masks_f[i,:] = x_miss[2]
        deltas_f[i,:] = delta_f(x_miss[0])
        
        evals_f[i,:] = x[i] 
        eval_masks_f[i,:] = x_miss[1]
        
        # Backward
        values_b[i,:] = np.flip(x_miss[0], axis = 0).tolist() 
        masks_b[i,:] = np.flip(mask_f(x_miss[0]), axis = 0).tolist()
        deltas_b[i,:] = delta_f(x_miss[0]) * -1
        
        evals_b[i,:] = np.flip(x[i], axis = 0).tolist() 
        eval_masks_b[i,:] = np.flip(x_miss[1], axis = 0).tolist()

    # prepare data
    
    # Create format to data loader
    forward = []
    backward = []
    for i in range(n_time):
        forward.append({'values': values_f[:, i].tolist(), 'masks': masks_f[:, i].tolist(), 
                        'deltas': deltas_f[:, i].tolist(), 'evals': evals_f[:, i].tolist(), 
                        'eval_masks': eval_masks_f[:, i].tolist()})
        backward.append({'values': values_b[:, i].tolist(), 'masks': masks_b[:, i].tolist(),
                         'deltas': deltas_b[:, i].tolist(), 'evals': evals_b[:, i].tolist(),
                         'eval_masks': eval_masks_b[:, i].tolist()})
    
    data = {'forward': forward, 'backward': backward, 'label' : y}
    
    return data

# Function to get X, and y data
def daicwoz_missing(question='easy_sleep', open_face='eye_gaze', delta_steps=1, delta_average=False):
    seq_length = 40
    # call all dataset
    sessions1 = pd.read_csv('data/daicwoz/data/'+question+'/'+'train.tsv', delimiter='\t',
                            header=None,names=['index', 'label', 'label_notes', 'sentence', 'audio_features'])
    sessions2 = pd.read_csv('data/daicwoz/data/'+question+'/'+'dev.tsv', delimiter='\t',
                            header=None, names=['index', 'label', 'label_notes', 'sentence', 'audio_features'])

    # train sessions
    train_sessions = []
    for txt in sessions1.audio_features:
        s1 = re.search("[0-9]{3}",txt)
        train_sessions.append(s1.group())

    # dev sessions
    test_sessions = []
    for txt in sessions2.audio_features:
        s1 = re.search("[0-9]{3}",txt)
        test_sessions.append(s1.group())
    # subset for training and testing
    df_train_label = pd.DataFrame({'id':train_sessions,'label':sessions1.label.tolist(), 'is_train':1})
    df_test_label = pd.DataFrame({'id':test_sessions,'label':sessions2.label.tolist(), 'is_train':0})
    df_label = pd.concat([df_train_label, df_test_label], ignore_index=True)
    df_label['index'] = list(range(0, len(df_label)))

    # get statistics by sessions
    #sessions = train_sessions + test_sessions
    xt_tensor = []
    bad_sessions = []

    # to scale variables 
    scaler = MinMaxScaler()
    for session in df_label.id.tolist():
        #print(session)
        #id_list.append(session)
        # get face features 
        try:
            df1 = pd.read_csv('data/daicwoz/data_open_face_processed/'+open_face+'/'+question+
                          '/'+session+'.csv')
            lengh_serie = len(df1)
            if delta_average:
                # create id columns with sequences
                result1 = []
                for d in range(int(lengh_serie/delta_steps+1)):
                    result1.extend(delta_steps * [d])
                df1['colsum1'] = result1[0:lengh_serie]
                df1 = df1.groupby(['colsum1']).mean()
            else:
                list_sample_time_steps = list(range(0, lengh_serie,delta_steps))
                df1 = df1.iloc[list_sample_time_steps]
            # drop rows with zero: error o no information captured
            sum_rows1 = df1.iloc[:,1:].sum(axis=1).tolist()
            indices = [i for i, x in enumerate(sum_rows1) if x ==0]
            df1 = df1.drop(df1.index[indices])

            # remove timestamp
            df1 = df1.iloc[:,1:]
            if len(df1)>=seq_length:
                # first i-th 
                df1 = df1.iloc[0:seq_length,:]
            elif len(df1)<seq_length:
                # add zeros (pandas or tensor)
                # creat matrix with zeros 
                matrix_zeros = np.zeros((seq_length -len(df1), df1.shape[1]))
                df_zeros = pd.DataFrame(matrix_zeros, columns=df1.columns)
                df1 = pd.concat([df1, df_zeros], ignore_index=True)
                #df1 = df1.append(df_zeros, ignore_index=True)
            # transpos and convert: xt_tensor[n_sample[n_features[n_sequence]]]
            xt_tensor.append(df1.transpose().to_numpy().tolist())

        except:
            print('Problem with id:', session)
            bad_sessions.append(session)
            #time_steps_list.append(0)
            #seconds_list.append(0) 

    # torch tensor
    labels = df_label.label.tolist()
    
    # index
    index_train = df_label[df_label['is_train']==1].index.tolist()
    index_test = df_label[df_label['is_train']==0].index.tolist()
    
    # id
    id_train = df_label[df_label['is_train']==1].id.tolist()
    id_test = df_label[df_label['is_train']==0].id.tolist()
    
    return xt_tensor, labels, index_train, index_test,id_train,id_test

class MySet(Dataset):
    def __init__(self, question, open_face, ratio_missing, type_missing):
        super(MySet, self).__init__()
        X,y,train_idx,test_idx,train_id,test_id = daicwoz_missing(question,open_face,
                                        delta_steps=1, delta_average=False)
        # Creat a list for each sample
        data0 = []
        #ratio_missing = 0.05
        n_miss = int(40 * ratio_missing) # 40 = sequence length
        #print('n_miss:', n_miss)
        for i in range(len(y)):
            data0.append(get_data(X[i],y[i], n_miss, type_missing))
        self.content = data0
        self.test_indices = test_idx
        self.train_id = train_id
        self.test_id = test_id
        
    def __len__(self):
        return len(self.content)

    def __getitem__(self, idx):
        rec = self.content[idx]
        if idx in self.test_indices:
            rec['is_train'] = 0
            #rec['id'] = self.test_id[idx]
        else:
            rec['is_train'] = 1
            #rec['id'] = self.train_id[idx]
        return rec
       
def collate_fn(recs):
    forward = [x['forward'] for x in recs]
    backward = [x['backward'] for x in recs]

    def to_tensor_dict(recs):
        values = torch.FloatTensor([[x['values'] for x in r] for r in recs])
        masks = torch.FloatTensor([[x['masks'] for x in r] for r in recs])
        deltas = torch.FloatTensor([[x['deltas'] for x in r] for r in recs])
        #forwards = torch.FloatTensor([[x['forwards'] for x in r] for r in recs])
        
        evals = torch.FloatTensor([[x['evals'] for x in r] for r in recs])
        eval_masks = torch.FloatTensor([[x['eval_masks'] for x in r] for r in recs])
        
        return {'values': values, 'masks': masks, 'deltas': deltas, 'evals': evals, 'eval_masks': eval_masks}

    ret_dict = {'forward': to_tensor_dict(forward), 'backward': to_tensor_dict(backward)}

    ret_dict['label'] = torch.FloatTensor([x['label'] for x in recs])
    ret_dict['is_train'] = torch.FloatTensor([x['is_train'] for x in recs])
    #ret_dict['id'] = torch.FloatTensor([int(x['id']) for x in recs])

    return ret_dict

def get_loader(question,open_face,ratio_missing,batch_size,type_missing,shuffle=True):
    data_set = MySet(question, open_face, ratio_missing, type_missing)
    data_iter = DataLoader(dataset = data_set, \
                              batch_size = batch_size, \
                              #num_workers = 4, \
                              shuffle = shuffle, \
                              pin_memory = True, \
                              collate_fn = collate_fn
    )

    return data_iter
    
if __name__ == '__main__':
    from lightning.pytorch import seed_everything
    seed_everything(41)
        # Hyperparameters 
    SEQ_LEN = 40                           # number of period in the ts, t = 1, 2, 3, 4, 5.
    RNN_HID_SIZE = 32                     # hidden node of the rnn 
    batch_size = 32
    model_name = 'BRITS_ATT' # RITS
    question = 'feeling_lately'
    open_face = 'eye_gaze'
    epochs = 100
    #N_SERIES = 12                          # number of series Rd, 12:eye, 136:landmark,14:action unit
    lr = 1e-3
    repetitions = 1
    ratio_missing = 0.05
    type_missing = 'Random' # CMV
    rnn_name = 'LSTM' # GRU
    experiment_name = 'exp01'
    use_ricardo_loader = True


    parser = argparse.ArgumentParser()
    parser.add_argument("--bs", type=int, default=32, help="Batch size. Default=1")
    parser.add_argument("--model_name", default="BRITS_ATT",help="Model Name. Default=BRITS. Options=[BRITS, RITS, BRITS_ATT]")
    parser.add_argument("--question", default="doing_today", help="Question dataset. Default=doing_today")
    parser.add_argument("--open_face", default="landmark", help="Facial features. Default=eye_gaze, options=[eye_gaze, landmark, action_unit, all]")
    parser.add_argument("--epochs", type=int, default=30, help="Number of Epochs to train. Default=1")
    parser.add_argument("--lr", type=float, default=0.001, help="Use folloup question. Default=1e-3")
    parser.add_argument("--rep", type=int, default=1, help="Number of repetitions. Default=1")
    parser.add_argument("--ratio_miss", type=float, default=0.10, help="Ratio of missing values. Default=0.05")
    parser.add_argument("--type_missing", default="Random", help="Type of missing values removing (Random or Consecutive). Default=Random")
    parser.add_argument("--rnn_name", default="LSTM", help="RNN cell layer (LSTM or GRU). Default=LSTM")

    args = parser.parse_args()


    if args.__dict__["bs"]  is not None:
        batch_size = args.__dict__["bs"]
    if args.__dict__["model_name"]  is not None:
        model_name = args.__dict__["model_name"]
    if args.__dict__["question"]  is not None:
        question = args.__dict__["question"]
    if args.__dict__["open_face"]  is not None:
        open_face = args.__dict__["open_face"]    
    if args.__dict__["epochs"]  is not None:
        epochs = args.__dict__["epochs"]
    if args.__dict__["lr"]  is not None:
        lr = args.__dict__["lr"]  
    if args.__dict__["rep"]  is not None:
        repetitions = args.__dict__["rep"] 
    if args.__dict__["ratio_miss"]  is not None:
        ratio_missing = args.__dict__["ratio_miss"] 
    if args.__dict__["type_missing"]  is not None:
        type_missing = args.__dict__["type_missing"] 
    if args.__dict__["rnn_name"]  is not None:
        rnn_name = args.__dict__["rnn_name"]
    
    
    from src.datamodules.daicwoz import DAICWOZDatamodule
    if not use_ricardo_loader:
        dm = DAICWOZDatamodule(
            label='phq8',
            open_face=args.open_face,
            question=args.question,
            delta_steps=1,
            delta_average=False,
            regen=True,
            ratio_missing=args.ratio_miss,
            type_missing=args.type_missing,
            val_ratio=0.1, # dummy
            batch_size=args.bs,
            num_workers=0,
            ricardo=False
        )
        dm.prepare_data()
        dm.setup()
        train_loader = dm.train_dataloader()
        test_loader = dm.test_dataloader()
    else:
        ricardo_loader = get_loader(args.question,args.open_face,args.ratio_miss,args.bs,args.type_missing)
        train_loader = ricardo_loader 
        test_loader = ricardo_loader
    

    #-----------------------------------#
    #--------- Train BRITS -------------#
    #-----------------------------------#
    if open_face=='action_unit':
        N_SERIES = 14  # 14, for action unit
    elif open_face=='eye_gaze':
        N_SERIES = 12
    elif open_face=='landmark':
        N_SERIES = 136
    elif open_face=='all':
        N_SERIES = (14+12+136)

    print('Question:', question) 
    for rep in range(1):
        if model_name=="BRITS":
            model = Model_brits(rnn_name)
        elif model_name=="RITS":
            model = Model_rits(rnn_name)
        elif model_name=="BRITS_ATT":
            model = Model_brits_att(rnn_name)

        params = next(iter(model.parameters()))
        optimizer = optim.Adam(model.parameters(), lr = lr) # switch optimizer?? adamW
        # data_iter = get_loader(question,open_face,ratio_missing,batch_size,type_missing)

        Loss_brits = []
        MAE_brits = []
        MRE_brits = []
        AUC_brits = []
        BA_brits = []
        F1_brits = []
        Recall_brits = []
        SF_brits = []

        for epoch in range(epochs):
            model.train()

            run_loss = 0.0 
            for idx, data in enumerate(train_loader):
                if not use_ricardo_loader:
                    data = assemble_input_for_training(data, params=params)
                data = to_var(data)
                ret1, ret2 = model.run_on_batch(data, optimizer)
                
            # evaluate model 
            eval1 = evaluate(model, test_loader, use_ricardo_loader=True if use_ricardo_loader else False)
            MAE_brits.append(eval1[0])
            MRE_brits.append(eval1[1])
            AUC_brits.append(eval1[2])
            BA_brits.append(eval1[3])
            F1_brits.append(eval1[4])
            Recall_brits.append(eval1[5])
            # SF_brits.append(eval1[6])

            run_loss += float(ret['loss'].data.cpu().numpy())
            Loss_brits.append(run_loss)
            if epoch % 1 == 0:
                print('Rep:', rep + 1,'/',repetitions,', Epoch:',epoch,'/', epochs - 1, 
                    ', loss:', round(run_loss, 2), ', BA :', round(eval1[3],2))
                
                #true_missing.append(ret['imputations'][0,3,0].data.cpu().numpy().tolist())
        # summary dataset
        if rep==0:
            # create summary dataset 
            df_metrics = pd.DataFrame({'Epoch':range(len(Loss_brits)), 'Loss':Loss_brits,'MAE':MAE_brits,
                                'MRE':MRE_brits, 'AUC':AUC_brits, 'BA':BA_brits, 'F1':F1_brits,
                                'Recall':Recall_brits, 'Model':model_name})
            df_metrics['batch_size'] = batch_size
            df_metrics['lr'] = lr
            df_metrics['Question'] = question
            df_metrics['Facial feature'] = open_face
            df_metrics['Ratio missing'] = ratio_missing
            df_metrics['Repetition'] = rep
            df_metrics['Type missing'] = type_missing
            df_metrics['RNN name'] = rnn_name

        else:
            df_metrics1 = pd.DataFrame({'Epoch':range(len(Loss_brits)), 'Loss':Loss_brits,'MAE':MAE_brits,
                                'MRE':MRE_brits, 'AUC':AUC_brits, 'BA':BA_brits, 'F1':F1_brits,
                                'Recall':Recall_brits, 'SF':SF_brits, 'Model':model_name})
            df_metrics1['batch_size'] = batch_size
            df_metrics1['lr'] = lr
            df_metrics1['Question'] = question
            df_metrics1['Facial feature'] = open_face
            df_metrics1['Ratio missing'] = ratio_missing
            df_metrics1['Repetition'] = rep
            df_metrics['Type missing'] = type_missing
            df_metrics['RNN name'] = rnn_name

            # concatenate
            df_metrics = pd.concat([df_metrics, df_metrics1], ignore_index=True)

    print('End of Experiment!')
    print('')

    # save results
    from pathlib import Path
    save_path = Path('ricardo_results') / experiment_name / 'df_metrics_'/ f"{question}_{open_face}_bs_{str(batch_size)} _epoch_{str(epochs)}_lr_{str(lr)}_rep_{str(repetitions)}_rm_{str(ratio_missing)}_model_name_{rnn_name}_{type_missing}_v1.csv"
    save_path.parent.mkdir(parents=True, exist_ok=True)
    df_metrics.to_csv(save_path, index=False)
    # df_metrics.to_csv(f"experiment_name+'/df_metrics_'+question+'_'+open_face+'_bs'+str(batch_size) +'_epoch'+ str(epochs)+'_lr'+str(lr).replace('0.','')+'_rep'+ str(repetitions)+'_rm'+str(ratio_missing).replace('0.','') +'_'+model_name+'_'+rnn_name+'_'+type_missing+'_v1.csv', index=False)
    print(df_metrics.iloc[:, :-4].tail(5))





