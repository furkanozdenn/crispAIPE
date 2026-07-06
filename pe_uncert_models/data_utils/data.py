"""
Prepare data for PE uncertainty models
- pridict-v1
"""

import os 
import sys
import pdb

from sklearn.model_selection import train_test_split

import torch
from torch.utils.data import DataLoader, TensorDataset

import pytorch_lightning as pl

import numpy as np
import pandas as pd

from . import data_utils


class PE_Dataset(pl.LightningDataModule):

    """
    Args:
        "data": "pridict-v1",
        "vocab_char_dict": "ACGTN",
        "train_data_path": "../../data/pridict_data/pridict-90k-cleaned_train.csv",
        "test_data_path": "../../data/pridict_data/pridict-90k-cleaned_test.csv",
        "val_data_path": (optional) explicit validation set path for target-disjoint splits,
        "batch_size": 128,
        "val_split": 0.1,
        "pegrna_length": 100
    """


    def __init__(self, data_config):
        super().__init__()
        self.data_config = data_config

        self.data = data_config["data"]
        self.vocab_char_dict = data_config["vocab_char_dict"]
        self.train_data_path = data_config["train_data_path"]
        self.test_data_path = data_config["test_data_path"]
        self.val_data_path = data_config.get("val_data_path", None)
        self.batch_size = data_config["batch_size"]
        self.val_split = data_config["val_split"]
        self.pegrna_length = data_config["pegrna_length"]
        self.sequence_length = data_config["sequence_length"]

        print(f"loading training data from {self.train_data_path}")
        self.train_data = pd.read_csv(self.train_data_path)
        if self.val_data_path:
            print(f"loading validation data from {self.val_data_path}")
            self.val_data = pd.read_csv(self.val_data_path)
        print(f"loading testing data from {self.test_data_path}")
        self.test_data = pd.read_csv(self.test_data_path)

        self._prepare_data()
        self._setup()


    def train_dataloader(self, shuffle_bool=True):
        # Setup MPS generator if needed
        generator = None
        if torch.backends.mps.is_available() and torch.device(torch.empty(1).device).type == 'mps':
            generator = torch.Generator(device='mps')
        elif torch.cuda.is_available():
            generator = torch.Generator(device='cuda')
        
        return DataLoader(
            self.train_dataset,
            batch_size=self.batch_size,
            shuffle=shuffle_bool,
            generator=generator if shuffle_bool else None
        )

    def val_dataloader(self, shuffle_bool=False):
        # Setup MPS generator if needed
        generator = None
        if shuffle_bool:
            if torch.backends.mps.is_available() and torch.device(torch.empty(1).device).type == 'mps':
                generator = torch.Generator(device='mps')
            elif torch.cuda.is_available():
                generator = torch.Generator(device='cuda')
        
        return DataLoader(
            self.val_dataset,
            batch_size=self.batch_size,
            shuffle=shuffle_bool,
            generator=generator if shuffle_bool else None
        )

    def test_dataloader(self, shuffle_bool=False):
        # Setup MPS generator if needed
        generator = None
        if shuffle_bool:
            if torch.backends.mps.is_available() and torch.device(torch.empty(1).device).type == 'mps':
                generator = torch.Generator(device='mps')
            elif torch.cuda.is_available():
                generator = torch.Generator(device='cuda')
        
        return DataLoader(
            self.test_dataset,
            batch_size=self.batch_size,
            shuffle=shuffle_bool,
            generator=generator if shuffle_bool else None
        )
    
    def load_vocab(self):
        pass
    
    def _extract_split(self, df):
        """Extract and encode all feature columns from a dataframe."""
        cols = data_utils.dataset_to_data_columns_dict(self.data)
        initial_seq = data_utils.df_one_hot_encode_seq(df[cols['initial_sequence']], vocab=None)
        mutated_seq = data_utils.df_one_hot_encode_seq(df[cols['mutated_sequence']], vocab=None)
        protospacer_mask = data_utils.get_binary_location_mask(df[cols['protospacer_location']], self.sequence_length)
        pbs_mask = data_utils.get_binary_location_mask(df[cols['pbs_location']], self.sequence_length)
        rt_init_mask = data_utils.get_binary_location_mask(df[cols['rt_initial_location']], self.sequence_length)
        rt_mut_mask = data_utils.get_binary_location_mask(df[cols['rt_mutated_location']], self.sequence_length)
        return (
            initial_seq, mutated_seq,
            df[cols['total_read_count']], df[cols['edited_percentage']],
            df[cols['unedited_percentage']], df[cols['indel_percentage']],
            protospacer_mask, pbs_mask, rt_init_mask, rt_mut_mask,
        )

    def _prepare_data(self):
        train_df = data_utils.read_data(self.train_data_path)
        test_df = data_utils.read_data(self.test_data_path)

        self.train_tuple = self._extract_split(train_df)
        self.test_tuple = self._extract_split(test_df)

        if self.val_data_path:
            val_df = data_utils.read_data(self.val_data_path)
            self.val_tuple = self._extract_split(val_df)
            self.val_data_size = len(val_df)
            print(f"val data size: {self.val_data_size}")

        self.train_data_size = len(train_df)
        self.test_data_size = len(test_df)

        print(f"train data size: {self.train_data_size}")
        print(f"test data size: {self.test_data_size}")

    def _tuple_to_tensors(self, data_tuple):
        """Convert a data tuple of numpy arrays/series to a list of tensors."""
        (initial_seq, mutated_seq, total_read_count, edited_pct,
         unedited_pct, indel_pct, proto_mask, pbs_mask, rt_init_mask, rt_mut_mask) = data_tuple
        return [
            torch.tensor(np.stack(initial_seq)).to(torch.int64),
            torch.tensor(np.stack(mutated_seq)).to(torch.int64),
            torch.tensor(np.stack(total_read_count)).to(torch.int64),
            torch.tensor(np.stack(edited_pct)).to(torch.float32),
            torch.tensor(np.stack(unedited_pct)).to(torch.float32),
            torch.tensor(np.stack(indel_pct)).to(torch.float32),
            torch.tensor(np.stack(proto_mask)).to(torch.int64),
            torch.tensor(np.stack(pbs_mask)).to(torch.int64),
            torch.tensor(np.stack(rt_init_mask)).to(torch.int64),
            torch.tensor(np.stack(rt_mut_mask)).to(torch.int64),
        ]

    def _setup(self):
        if self.val_data_path:
            train_tensors = self._tuple_to_tensors(self.train_tuple)
            val_tensors = self._tuple_to_tensors(self.val_tuple)
        else:
            raw = self.train_tuple
            split_result = train_test_split(
                *raw, test_size=self.val_split, random_state=42
            )
            n_fields = len(raw)
            train_arrays = split_result[0::2]
            val_arrays = split_result[1::2]
            train_tensors = self._tuple_to_tensors(train_arrays)
            val_tensors = self._tuple_to_tensors(val_arrays)

        test_tensors = self._tuple_to_tensors(self.test_tuple)

        self.train_dataset = torch.utils.data.TensorDataset(*train_tensors)
        self.val_dataset = torch.utils.data.TensorDataset(*val_tensors)
        self.test_dataset = torch.utils.data.TensorDataset(*test_tensors)

