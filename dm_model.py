from logging import raiseExceptions
import os 
import sys
from typing import Dict, overload
import torch
import torch.nn.functional as F
from torch import nn
import pytorch_lightning as pl 
import torchvision.models as tvmodels
from sklearn.metrics import accuracy_score, precision_score, recall_score
# from torchmetrics import MetricCollection, Accuracy, Precision, Recall
from datetime import datetime
import numpy as np
import pickle

from networks import LSTARNet

class _BaseModel(pl.LightningModule):
    def __init__(self, modal='dual'):
        super().__init__()
        self.modal = modal
        self.batch_size = None
        self.evaluate = False

    def _load_model(self):
        self.encoder = None
        self.head = None
    
    def _setup(self):
        # model
        self._load_model()
        # loss
        self.loss = nn.MSELoss()
    
    def _get_input(self, batch, modal='dual'):
        y = batch['y'][self.target]
        y = self.fill_nan(y)

        xt, xs = None, None
        if modal == 'timeseries' or modal == 'dual':
            xt = batch['X']['timeseries']
            xt_len=torch.LongTensor(list(map(len,xt)))
            xt = nn.utils.rnn.pad_sequence(xt, batch_first=True, padding_value=0.0)
            # xt = nn.utils.rnn.pack_padded_sequence(xt, xt_len.cpu().numpy(), batch_first=True, enforce_sorted=False)

        if modal == 'slice' or modal == 'dual':
            xs = batch['X']['slice']

        return xs.float().to(memory_format=torch.contiguous_format) if xs is not None else xs, \
               xt.float().to(memory_format=torch.contiguous_format) if xt is not None else xt, \
               y.float().to(memory_format=torch.contiguous_format)
    
    def forward(self, x):
        pass
    
    def fill_nan(self, x):
        if len(x.shape) == 1 and x.shape[0] == 1: #single input
            fill = 0.0
        else:
            fill = torch.nanmedian(x)
        return torch.nan_to_num(x, fill)

    def training_step(self, batch, batch_idx):
        xs, xt, y = self._get_input(batch, self.modal)
        y_hat = self([xs, xt])

        loss = self.loss(y_hat, y)

        self.log_dict({'train/loss': loss}, batch_size=self.batch_size,
             prog_bar=True, logger=True, on_step=True, on_epoch=True)

        return {'loss':loss, 'y_hat':y_hat.detach()}

    def validation_step(self, batch, batch_idx):
        xs, xt, y = self._get_input(batch, self.modal)
        y_hat = self([xs, xt])

        loss = self.loss(y_hat, y)
        self.log_dict({'val/loss': loss}, batch_size=self.batch_size,
            prog_bar=True, logger=True, on_step=False, on_epoch=True)
        
        return {'loss':loss, 'y_hat':y_hat.detach()}

    def predict_step(self, batch, batch_idx):
        xs, xt, y = self._get_input(batch, self.modal)
        y_hat = self([xs, xt])

        return {'y_hat': y_hat.detach().cpu().numpy(), 
                'y': y.detach().cpu().numpy()}

    def configure_optimizers(self):
        pass

class TimeseriesModel(_BaseModel):
    def __init__(self, params):
        super().__init__('timeseries')
        self.arch = params.timeseries.arch
        self.name = self.arch
        self.target = params.run.target
        self.learning_rate = params.train.learning_rate
        self.batch_size = params.data.batch_size
        self.model_params = params.timeseries.model_params
        self.save_hyperparameters()
        self._setup()

    def _load_model(self):
        if self.arch == 'LSTARNet':
            net = LSTARNet.Model(self.model_params)
            self.encoder = net
            self.head = nn.Sequential(
                nn.Linear(512, 256),
                nn.Dropout(p=0.25),
                nn.Linear(256, 1))
        else:
            raise ValueError(f'Timeseries arch={self.arch} is not supported')
    
    def forward(self, x):
        c = self.encoder(x[1])
        y_hat = self.head(c)
        y_hat = torch.squeeze(y_hat)

        return y_hat

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(list(self.encoder.parameters()) + \
                                     list(self.head.parameters()), 
                                     lr=self.learning_rate)
        
        return optimizer

class SliceModel(_BaseModel):
    def __init__(self, params):
        super().__init__('slice')
        self.arch = params.slice.arch
        self.name = self.arch
        self.target = params.run.target
        self.learning_rate = params.train.learning_rate
        self.batch_size = params.data.batch_size
        self.model_params = params.slice.model_params
        self.save_hyperparameters()
        self._setup()

    def _load_model(self):
        if self.arch == 'ResNet18':
            net = tvmodels.resnet18(pretrained=True)
            self.encoder = nn.Sequential(*(list(net.children())[:-1]))
            self.head = nn.Sequential(
                nn.Linear(512, 256),
                nn.Dropout(p=0.5),
                nn.Linear(256, 1))
        else:
            raise ValueError(f'Slice arch={self.arch} is not supported')
    
    def forward(self, x):
        c = self.encoder(x[0])
        c = torch.squeeze(c)
        y_hat = self.head(c)
        y_hat = torch.squeeze(y_hat)
        
        return y_hat

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(list(self.encoder.parameters()) + \
                                     list(self.head.parameters()), 
                                     lr=self.learning_rate)
        
        return optimizer

class DualModalModel(_BaseModel):
    def __init__(self, params, timeseries_model=None, slice_model=None):
        super().__init__('dual')
        self.train_head_only = params.dual.train_head_only
        self.name = 'Dual' + ('head_only' if self.train_head_only else 'whole')
        self.timeseries_model = timeseries_model
        self.slice_model = slice_model
        self.target = params.run.target
        self.learning_rate = params.train.learning_rate
        self.batch_size = params.data.batch_size
        self.save_hyperparameters(ignore=['timeseries_model', 'slice_model'])
        self._setup()
    
    def _load_model(self):
        if self.timeseries_model is not None:
            self.t_encoder = self.timeseries_model.encoder
        if self.slice_model is not None:
            self.s_encoder = self.slice_model.encoder
        if self.timeseries_model is not None and self.slice_model is not None:
            self.head = nn.Sequential(
                nn.Linear(512*2, 256),
                nn.Dropout(p=0.1),
                nn.Linear(256, 64),
                nn.Dropout(p=0.1),
                nn.Linear(64, 1)
            )
    
    def forward(self, x):
        s = self.s_encoder(x[0])
        s = torch.squeeze(s, dim=-1)
        s = torch.squeeze(s, dim=-1)

        t = self.t_encoder(x[1])
        t = torch.squeeze(t, dim=-1)

        st = torch.cat([s, t], dim=1)
        y_hat = self.head(st)
        y_hat = torch.squeeze(y_hat)

        return y_hat

    def configure_optimizers(self):
        if self.train_head_only:
            optimizer = torch.optim.Adam(list(self.head.parameters()), lr=self.learning_rate)
        else:
            optimizer = torch.optim.Adam(list(self.s_encoder.parameters()) + \
                                         list(self.t_encoder.parameters()) + \
                                         list(self.head.parameters()), lr=self.learning_rate)
        return optimizer

    