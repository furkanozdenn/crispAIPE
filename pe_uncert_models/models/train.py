"""Training script for CRISPRoposer using PyTorch Lightning
"""

import datetime
import time
import os 
import sys
import json
import platform

import numpy as np
import pandas as pd

import argparse
from argparse import ArgumentParser

import pytorch_lightning as pl
from pytorch_lightning import Trainer
from pytorch_lightning.callbacks import ModelCheckpoint
from pytorch_lightning.loggers import CSVLogger
from pytorch_lightning.callbacks.early_stopping import EarlyStopping

# Ensure this worktree's code takes priority over editable installs
_project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), os.pardir, os.pardir))
sys.path.insert(0, _project_root)

from pe_uncert_models.models.crispAIPE import crispAIPE
from pe_uncert_models.data_utils.data import PE_Dataset


from scipy.stats import spearmanr

import torch
import pdb

if __name__ == '__main__':
    parser = ArgumentParser()

    parser.add_argument('--config', type=str, required=True, help='path to config file')

    args = parser.parse_args()

    with open(args.config, 'r') as f:
        config = json.load(f)

    config_model = config['model_parameters']
    config_data = config['data_parameters']
    config_training = config['training_parameters']

    # Resolve relative data paths against the config file's directory
    config_dir = os.path.dirname(os.path.abspath(args.config))
    for path_key in ['train_data_path', 'val_data_path', 'test_data_path', 'vocab_path']:
        if path_key in config_data and not os.path.isabs(config_data[path_key]):
            config_data[path_key] = os.path.normpath(
                os.path.join(config_dir, config_data[path_key])
            )
    if 'log_dir' in config_training and not os.path.isabs(config_training['log_dir']):
        config_training['log_dir'] = os.path.normpath(
            os.path.join(config_dir, config_training['log_dir'])
        )

    for key, value in config.items():
        parser.add_argument(f'--{key}', type=type(value), default=value)

    cl_args = parser.parse_args()

    now = datetime.datetime.now()
    date_suffix = now.strftime("%Y-%m-%d-%H-%M-%S")
    config_file_name = args.config.split('/')[-1].split('.')[0]
    save_dir = os.path.join(config_training['log_dir'], config_file_name, date_suffix)

    if not os.path.exists(save_dir):
        os.makedirs(save_dir)

    csv_logger = CSVLogger(
        save_dir=save_dir,
        name='csv_logs',
    )
    csv_logger.log_hyperparams(config)

    early_stop_callback = EarlyStopping(
        monitor='val_loss',
        patience=config_training['patience'],
        verbose=True,
        mode='min',
        min_delta=0.001
    )

    checkpoint_callback = ModelCheckpoint(
        monitor='val_loss',
        dirpath=save_dir,
        filename='best_model-{epoch:02d}-val_loss_{val_loss:.4f}',
        save_top_k=1,
        mode='min',
        verbose=True,
        save_last=True,
    )

    data = PE_Dataset(data_config=config_data)
    model = crispAIPE(hparams={**config_model, **config_data, **config_training})

    if config_training['cpu']:
        trainer = pl.Trainer(
            max_epochs=config_training['max_epochs'],
            accelerator='cpu',
            log_every_n_steps=10,
            logger=csv_logger,
            callbacks=[early_stop_callback, checkpoint_callback]
        )
    elif platform.system() == 'Darwin' and 'arm' in platform.machine():
        if torch.backends.mps.is_available():
            print("Using Apple Silicon GPU (MPS)")
            torch.set_default_device('mps')
            os.environ['PYTORCH_ENABLE_MPS_FALLBACK'] = '1'

            trainer = pl.Trainer(
                max_epochs=config_training['max_epochs'],
                accelerator='mps',
                devices=1,
                log_every_n_steps=10,
                logger=csv_logger,
                callbacks=[early_stop_callback, checkpoint_callback],
                deterministic=False,
            )

            print(f"Default device: {torch.device('mps')}")
            print(f"PyTorch MPS device available: {torch.backends.mps.is_available()}")
            print(f"PyTorch MPS device built: {torch.backends.mps.is_built()}")
        else:
            print("MPS is not available, falling back to CPU")
            trainer = pl.Trainer(
                max_epochs=config_training['max_epochs'],
                accelerator='cpu',
                log_every_n_steps=10,
                logger=csv_logger,
                callbacks=[early_stop_callback, checkpoint_callback]
            )
    else:
        print(f'Using GPUs: {config_training["gpus"]}')
        trainer = pl.Trainer(
            max_epochs=config_training['max_epochs'],
            strategy='dp',
            accelerator='gpu',
            devices=config_training['gpus'],
            log_every_n_steps=10,
            logger=csv_logger,
            callbacks=[early_stop_callback, checkpoint_callback]
        )

    trainer.fit(
        model=model,
        train_dataloaders=data.train_dataloader(),
        val_dataloaders=data.val_dataloader(),
    )

    trainer.save_checkpoint(os.path.join(save_dir, 'model.ckpt'))
    train_time = datetime.datetime.now() - now
    hours, remainder = divmod(train_time.seconds, 3600)
    print(f'Training complete, took {hours}h:{remainder//60}m')

    print("Evaluating model")
