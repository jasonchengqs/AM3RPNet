import torch
import pytorch_lightning as pl
import dm_model
import r3dim_data
import os, sys
import argparse
from omegaconf import OmegaConf
import joblib
import numpy as np

def parse_args():
    """Parser for command line arguments.
    """
    parser = argparse.ArgumentParser(description='Run configuration.')
    parser.add_argument('--gpu', type=int, default=-1, 
                        help='choice of GPU device, -1 means using all')
    parser.add_argument('--mode', type=str, default='debug', 
                        help='train | test | debug')
    parser.add_argument('--config-file', type=str, default='',
                        help='absolute path to config file.')             
    args = parser.parse_args()
    return args


def train(data, args, configs):
    if configs.run.timeseries_only:
        if configs.run.resume:
            model = dm_model.TimeseriesModel.load_from_checkpoint(configs.timeseries.ckpt)
        else:
            model = dm_model.TimeseriesModel(configs)
        save_dir = os.path.join(configs.run.save_dir, 'timeseries', configs.run.target)
    elif configs.run.slice_only:
        if configs.run.resume:
            model = dm_model.SliceModel.load_from_checkpoint(configs.slice.ckpt)
        else:
            model = dm_model.SliceModel(configs)
        save_dir = os.path.join(configs.run.save_dir, 'slice', configs.run.target)
    else:
        if configs.run.resume:
            model = dm_model.DualModalModel.load_from_checkpoint(configs.dual.ckpt)
        else:
            if configs.dual.use_pretrained_singles:
                t_model = dm_model.TimeseriesModel.load_from_checkpoint(configs.timeseries.ckpt)
                s_model = dm_model.SliceModel.load_from_checkpoint(configs.slice.ckpt)
            else:
                t_model = dm_model.TimeseriesModel(configs)
                s_model = dm_model.SliceModel(configs)
            model = dm_model.DualModalModel(configs, timeseries_model=t_model, slice_model=s_model)
        save_dir = os.path.join(configs.run.save_dir, 'dual', configs.run.target)
            
    ckpt_dir = os.path.join(save_dir, 'ckpt')
    save_checkpoint = pl.callbacks.ModelCheckpoint(
            dirpath=ckpt_dir,
            monitor='val/loss',
            filename=model.name)

    trainer = pl.Trainer(
            max_epochs=configs.train.max_epoch,
            progress_bar_refresh_rate=10,
            default_root_dir=save_dir,
            accelerator='gpu',
            devices=AVAIL_GPUS,
            check_val_every_n_epoch=1,
            callbacks=[save_checkpoint]
        )
    trainer.fit(model, datamodule=data)


def test(data, args, configs):
    if configs.run.timeseries_only:
        model = dm_model.TimeseriesModel.load_from_checkpoint(configs.timeseries.ckpt)
        save_dir = os.path.join(configs.run.save_dir,'timeseries', configs.run.target)
    elif configs.run.slice_only:
        model = dm_model.SliceModel.load_from_checkpoint(configs.slice.ckpt)
        save_dir = os.path.join(configs.run.save_dir,'slice', configs.run.target)
    else:
        model = dm_model.DualModalModel.load_from_checkpoint(configs.dual.ckpt)
        save_dir = os.path.join(configs.run.save_dir,'dual', configs.run.target)

    tester = pl.Trainer(progress_bar_refresh_rate=10,
                        enable_checkpointing=False, 
                        logger=False,
                        accelerator='gpu',
                        devices=AVAIL_GPUS,)
    res_dict = tester.predict(model, datamodule=data)
    
    def _concate_dict(res_dict, key=''):
        ret = []
        for r in res_dict:
            try:
                ret.extend(r[key])
            except:
                ret.append(r[key])
        return ret

    preds = _concate_dict(res_dict, 'y_hat')
    targets = _concate_dict(res_dict, 'y')

    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
    joblib.dump({'preds':preds, 'targets':targets}, 
        os.path.join(f'{save_dir}', f'preds_{configs.run.target}.pkl'))


if __name__ == "__main__":
    args = parse_args()
    AVAIL_GPUS = [d for d in range(torch.cuda.device_count())]
    if args.gpu != -1:
        AVAIL_GPUS = [AVAIL_GPUS[args.gpu]]
    configs = OmegaConf.load(args.config_file)

    if args.mode == 'train':
        configs = configs.train
        data = r3dim_data.DualModalDataModule(configs)
        train(data, args, configs)
    elif args.mode == 'test':
        configs = configs.test
        data = r3dim_data.DualModalDataModule(configs)
        test(data, args, configs)
