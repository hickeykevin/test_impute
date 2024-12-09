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
    def __init__(
            self, 
            rnn_hid_size,
            n_series,
            seq_len,
            rnn_name='LSTM',
            ):
        super(Model_rits, self).__init__()
        self.rnn_name = rnn_name
        self.rnn_hid_size = rnn_hid_size
        self.n_series = n_series
        self.seq_len = seq_len
        self.build()

    def build(self):
        if self.rnn_name=='LSTM':
            self.rnn_cell = nn.LSTMCell( self.n_series * 2, self.rnn_hid_size)
        elif self.rnn_name=='GRU':
            self.rnn_cell = nn.GRUCell( self.n_series * 2, self.rnn_hid_size)

        self.temp_decay_h = TemporalDecay(input_size =  self.n_series, output_size = self.rnn_hid_size, diag = False)
        self.temp_decay_x = TemporalDecay(input_size =  self.n_series, output_size =  self.n_series, diag = True)

        self.hist_reg = nn.Linear(self.rnn_hid_size,  self.n_series)
        self.feat_reg = FeatureRegression( self.n_series)

        self.weight_combine = nn.Linear( self.n_series * 2,  self.n_series)

        self.dropout = nn.Dropout(p = 0.25)
        self.out = nn.Linear(self.rnn_hid_size, 1)

    
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

        h = Variable(torch.zeros((values.size()[0], self.rnn_hid_size)))
        c = Variable(torch.zeros((values.size()[0], self.rnn_hid_size)))

        #if torch.cuda.is_available():
        #    h, c = h.cuda(), c.cuda()

        x_loss = 0.0
        y_loss = 0.0

        imputations = []

        for t in range(self.seq_len):
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

        return {'loss': x_loss / self.seq_len + y_loss * 0.1, 'predictions': y_h,\
                'imputations': imputations, 'label': labels, 'is_train': is_train,\
                'evals': evals, 'eval_masks': eval_masks}

    def run_on_batch(self, data):
        return self(data)
    # def run_on_batch(self, data, optimizer):
    #     ret = self(data, direct = 'forward')

    #     if optimizer is not None:
    #         optimizer.zero_grad()
    #         ret['loss'].backward()
    #         optimizer.step()

    #     return ret
    
#--- Run BRITS model using rits model
class Model_brits(nn.Module):
    def __init__(
            self, 
            rnn_hid_size: int, 
            n_series: int,
            seq_len: int,
            rnn_name='LSTM',
        ):
        super(Model_brits, self).__init__()
        self.rnn_name = rnn_name
        self.rnn_hid_size = rnn_hid_size
        self.n_series = n_series
        self.seq_len = seq_len
        self.build()

    def build(self):
        self.rits_f = Model_rits(rnn_name=self.rnn_name, rnn_hid_size=self.rnn_hid_size, n_series=self.n_series)
        self.rits_b = Model_rits(rnn_name=self.rnn_name, rnn_hid_size=self.rnn_hid_size, n_series=self.n_series)

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

    def run_on_batch(self, data):
        return self(data)

    # def run_on_batch(self, data, optimizer):
    #     ret = self(data)

    #     if optimizer is not None:
    #         optimizer.zero_grad()
    #         ret['loss'].backward()
    #         optimizer.step()

    #     return ret

#--- Run RITS model with Attentions
class Model_rits_att(nn.Module):
    def __init__(self, 
                 n_series,
                 rnn_hid_size,
                 seq_len,
                 rnn_name='LSTM'):
        super(Model_rits_att, self).__init__()
        self.rnn_name = rnn_name
        self.n_series = n_series
        self.rnn_hid_size = rnn_hid_size
        self.seq_len = seq_len
        self.build()
        
        # Attention following AudiBert 
        self.W_s1 = nn.Linear(self.rnn_hid_size, 350)
        self.W_s2 = nn.Linear(350, 30)
    def build(self):
        if self.rnn_name=='LSTM':
            self.rnn_cell = nn.LSTMCell( self.n_series * 2, self.rnn_hid_size)
        elif self.rnn_name=='GRU':
            self.rnn_cell = nn.GRUCell( self.n_series * 2, self.rnn_hid_size)

        self.temp_decay_h = TemporalDecay(input_size =  self.n_series, output_size = self.rnn_hid_size, diag = False)
        self.temp_decay_x = TemporalDecay(input_size =  self.n_series, output_size =  self.n_series, diag = True)

        self.hist_reg = nn.Linear(self.rnn_hid_size,  self.n_series)
        self.feat_reg = FeatureRegression( self.n_series)

        self.weight_combine = nn.Linear( self.n_series * 2,  self.n_series)

        self.dropout = nn.Dropout(p = 0.25)
        #self.out = nn.Linear(self.rnn_hid_size, 1)
        self.out = nn.Linear(self.rnn_hid_size*30, 1)
        
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
        H_rnn = torch.zeros(values.shape[0], self.seq_len, self.rnn_hid_size)  # (batch, sequence, hiden_dize)

        evals = data[direct]['evals']
        eval_masks = data[direct]['eval_masks']

        labels = data['label'].view(-1, 1)
        is_train = data['is_train'].view(-1, 1)

        h = Variable(torch.zeros((values.size()[0], self.rnn_hid_size)))
        c = Variable(torch.zeros((values.size()[0], self.rnn_hid_size)))

        #if torch.cuda.is_available():
        #    h, c = h.cuda(), c.cuda()
        x_loss = 0.0
        y_loss = 0.0

        imputations = []

        for t in range(self.seq_len):
            x = values[:, t, :]
            #print('x.shape:',x.shape)     # x.shape: torch.Size([1, 12])
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

        return {'loss': x_loss / self.seq_len + y_loss * 0.1, 'predictions': y_h,\
                'imputations': imputations, 'label': labels, 'is_train': is_train,\
                'evals': evals, 'eval_masks': eval_masks}
    
    def run_on_batch(self, data):
        return self(data)

    # def run_on_batch(self, data, optimizer):
    #     ret = self(data, direct = 'forward')

    #     if optimizer is not None:
    #         optimizer.zero_grad()
    #         ret['loss'].backward()
    #         optimizer.step()

    #     return ret

#--- Run BRITS model with Attentions 
class Model_brits_att(nn.Module):
    def __init__(
            self, 
            rnn_hid_size: int,
            n_series: int,
            seq_len: int,
            rnn_name='LSTM'
        ):
        super(Model_brits_att, self).__init__()
        self.rnn_name = rnn_name
        self.rnn_hid_size = rnn_hid_size
        self.n_series = n_series
        self.seq_len = seq_len
        self.build()

    def build(self):
        self.rits_f = Model_rits_att(rnn_name=self.rnn_name, rnn_hid_size=self.rnn_hid_size, n_series=self.n_series, seq_len=self.seq_len)
        self.rits_b = Model_rits_att(rnn_name=self.rnn_name, rnn_hid_size=self.rnn_hid_size, n_series=self.n_series, seq_len=self.seq_len)

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
    
    def run_on_batch(self, data):
       return self(data)

    # def run_on_batch(self, data, optimizer):
    #     ret = self(data)

    #     if optimizer is not None:
    #         optimizer.zero_grad()
    #         ret['loss'].backward()
    #         optimizer.step()

    #     return ret



