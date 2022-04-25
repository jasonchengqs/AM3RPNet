from time import time
import torch
from torch.utils.data import Dataset, DataLoader
import torchvision as T
from skimage import io, transform
import pytorch_lightning as pl
import glob
import os
import sys
import pandas as pd
import numpy as np
import joblib
from utils.split import *
from utils.transforms import *
from PIL import Image, ImageOps

SLICE_MAX_SIZE = 1536 # 256 * 6
TIMESERIES_MAX_SIZE = 2680

class DualModalDataset(Dataset):
    def __init__(self, data, slices_transforms=None, timeseries_pad_value=-1000):
        super().__init__()
        
        self.data = data
        self.slices_transforms = slices_transforms
        self.timeseries_pad_value = timeseries_pad_value
        self.pad_slice = PadImageToSize(SLICE_MAX_SIZE)
        
    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        pair = self.data[idx]
        blk = []
        for s in pair[0]['slice']:
            img = ImageOps.grayscale(Image.open(s))
            img = np.array(self.pad_slice(img))
            blk.append(img.reshape(img.shape[0], img.shape[1], 1))
        blk = np.concatenate(blk, axis=2)
        # print(np.max(blk))
        ts = joblib.load(pair[0]['timeseries'])
        ts = np.stack(list(ts.values()), axis=1)
        ts = torch.from_numpy(ts)
        # ts = self.pad_timeseries(ts)
        
        if self.slices_transforms is not None:
            blk = self.slices_transforms(blk)

        sample = {'X': {'slice': blk, 'timeseries': ts},
                  'y': pair[1]}

        return sample


class DualModalDataModule(pl.LightningDataModule):
    def __init__(self, configs):
        super().__init__()
        self.slices_dir = configs.run.slices_dir
        self.timeseries_dir = configs.run.timeseries_dir
        self.layer_measures_dir = configs.run.layer_measures_dir
        self.seed = configs.run.seed
        self.batch_size = configs.data.batch_size
        # self.batch_size = None
        self.slices_padding = configs.data.slices_padding

    def pair(self, slices_files, timeseries_files, layer_measures_files):
        def _group(files):
            grp = {}
            for i in range(len(files)):
                obj = files[i].split('/')[-2]
                if obj not in grp:
                    grp[obj] = [files[i]]
                else:
                    grp[obj].append(files[i])
            for obj in grp:
                grp[obj] = sorted(grp[obj])
            
            return grp

        slices_files = _group(slices_files)
        timeseries_files = _group(timeseries_files)
        layer_measures_files = sorted(layer_measures_files)

        pairs = []
        for obj in slices_files.keys():
            if len(slices_files[obj]) - len(timeseries_files[obj]) > 2 * self.slices_padding:
                dummpy = len(slices_files[obj]) - len(timeseries_files[obj]) - 2 * self.slices_padding
                slices_files[obj] = slices_files[obj][dummpy//2:-(dummpy-dummpy//2)]
            
            df = pd.read_excel(os.path.join(self.layer_measures_dir, f'{obj}_layer_data.xls'))
            layer_measures = df.loc[:, ['Layer Time [h]', 'Total Energy [Wh]']].values 
            
            assert len(slices_files[obj])-2*self.slices_padding == len(timeseries_files[obj]) == len(layer_measures), \
                f'{obj} layer length not match: slices={len(slices_files[obj])}, timeseries={len(timeseries_files[obj])}, measures={len(layer_measures)}'
            
            obj_pairs = [[{'slice': slices_files[obj][s-self.slices_padding:s+self.slices_padding+1], 'timeseries':t}, {'time':y[0], 'energy':y[1]}] 
                for s, t, y in zip(np.arange(len(timeseries_files[obj]), dtype=int)+self.slices_padding, timeseries_files[obj], layer_measures)]
            pairs.extend(obj_pairs)
        
        return np.array(pairs)
    
    def prepare_data(self):
        # load all slices
        slices_files = glob.glob(os.path.join(self.slices_dir, '**', '*.png')) 
        # + glob.glob(os.path.join(self.slices_dir, '**', '*.png'))
        
        # load all timeseries
        timeseries_files = glob.glob(os.path.join(self.timeseries_dir, '**', '*.pkl')) 
        # + glob.glob(os.path.join(self.timeseries_dir, '**', '*.pkl'))
        
        # load all layer resource measurements
        layer_measures_files = glob.glob(os.path.join(self.layer_measures_dir, '**', '*.xls')) + \
            glob.glob(os.path.join(self.layer_measures_dir, '**', '*.pkl'))
        
        self.data = self.pair(slices_files, timeseries_files, layer_measures_files)

    def get_stratify_splits(self, X, y, holdout=0.2, n_splits=1):
        """Get stratified training and validation splits.
        """
        splits = {}
        skf = StratifiedShuffleSplitReimplement(
            n_splits=n_splits, 
            test_size=holdout, 
            train_size=1-holdout, 
            random_state=self.seed)
        skf.get_n_splits(X, y)

        for train_index, test_index in skf.split(X, y):
            return train_index, test_index 

    def get_random_splits(self, data, holdout=0.2):
        """Get random training and validation splits.
        """
        np.random.seed(self.seed)
        total = len(data)
        split_idx = int(holdout*total)
        index = np.arange(total, dtype=int)
        np.random.shuffle(index)
        
        return data[index[split_idx:]], data[index[:split_idx]]

    def setup(self, stage):
        
        def _stat(dat):
            return len(dat), np.sum(dat==1), np.round(np.sum(dat==1) / len(dat) * 100, 2)
        
        train_data, test_data = self.get_random_splits(self.data)

        _base_transforms = [
            # PadImageToSize(max_size=MAX_SIZE),
            T.transforms.ToPILImage(),
            T.transforms.ToTensor(),
            T.transforms.Resize((256, 256)),
            T.transforms.Normalize([0.485, 0.456, 0.406], 
                                   [0.229, 0.224, 0.225])
        ]
        if stage == 'fit':
            train_data, val_data = self.get_random_splits(train_data)
            transforms = T.transforms.Compose(
                _base_transforms + \
                [
                    T.transforms.RandomHorizontalFlip(),
                    T.transforms.RandomVerticalFlip(),
                    T.transforms.RandomInvert(p=0.25)
                ]
            )
            self.train_dataset = DualModalDataset(
                data=train_data, 
                slices_transforms=transforms)
            transforms = T.transforms.Compose(_base_transforms)
            self.val_dataset = DualModalDataset(
                data=val_data, 
                slices_transforms=transforms)
            
            print(f'Total={len(self.data)}, Train set={len(self.train_dataset)} | Validation set={len(self.val_dataset)}')

        elif stage in ('test', 'predict', None):
            transforms = T.transforms.Compose(_base_transforms)
            self.test_dataset = DualModalDataset(
                data=test_data, 
                slices_transforms=transforms)
            print(f'Total={len(self.data)}, Test set={len(self.test_dataset)}')

    def train_dataloader(self):
        return DataLoader(self.train_dataset, batch_size=self.batch_size,
                          collate_fn=collate_based_on_len,
                          num_workers=8, pin_memory=True, shuffle=True)

    def val_dataloader(self):
        return DataLoader(self.val_dataset, batch_size=self.batch_size,
                          collate_fn=collate_based_on_len,
                          num_workers=8, pin_memory=True, shuffle=False)

    def test_dataloader(self):
        return DataLoader(self.test_dataset, batch_size=self.batch_size,
                          collate_fn=collate_based_on_len,
                          num_workers=8, pin_memory=True, shuffle=False)

    def predict_dataloader(self):
        return DataLoader(self.test_dataset, batch_size=self.batch_size,
                          collate_fn=collate_based_on_len,
                          num_workers=8, pin_memory=True, shuffle=False)

