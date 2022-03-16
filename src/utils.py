import json
import logging
import torch

import os

from torch.utils.data import DataLoader
from transformers import BertTokenizerFast

from .klue_dataset import build_dataset
from .model import BERTforKLUE


def get_tokenizer():
    tokenizer_bert = BertTokenizerFast.from_pretrained("kykim/bert-kor-base")
    return tokenizer_bert


def get_dataloader(path_to_data, tokenizer, cfg):
    logging.info(f"Loding {path_to_data}")
    raw_data = open(path_to_data)
    raw_data = json.load(raw_data)
    dataset = build_dataset(raw_data, tokenizer, cfg)
    dataloader = DataLoader(dataset, batch_size=cfg.batch_size)
    return dataloader


def get_model(cfg):
    model = BERTforKLUE(cfg)
    return model


def save_checkpoint(cfg, model, epoch):
    #logging.info(f">>> Saving model at {args.save_path} for iteration {iteration}...")
    save_path = f"{cfg.save_path}/{epoch}_checkpoint.pt"
     
    checkpoint = {
        'model': model.state_dict()
    }
    
    torch.save(checkpoint, save_path)
