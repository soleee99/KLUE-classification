import hydra
import logging
import os
import sys
import torch

from omegaconf import OmegaConf
from torch.optim import Adam

from src.utils import get_tokenizer, get_dataloader, get_model
from src.train import train
from src.eval import eval

def main(cfg):
    logging.info('=====================Configs=====================')
    logging.info(OmegaConf.to_yaml(cfg))

    # Open logging files
    train_log_f = open(cfg.train_log_path, 'w')
    test_log_f = open(cfg.test_log_path, 'w')

    # Get Tokenizer
    tokenizer = get_tokenizer()

    # Get Dataloader
    logging.info('=====================Getting Dataloader=====================')
    train_ds = get_dataloader(cfg.train_path, tokenizer, cfg)
    dev_ds = get_dataloader(cfg.dev_path, tokenizer, cfg)
    test_ds = get_dataloader(cfg.test_path, tokenizer, cfg)

    # Get device
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    logging.info(f"Device type is: {device}")

    # Get Model and Optimizer
    logging.info('=====================Getting Model & Optimizer=====================')
    model = get_model(cfg)
    optimizer = Adam(model.parameters(), lr=cfg.lr)
    model = model.to(device)

    train(cfg, model, optimizer, train_ds, dev_ds, device, train_log_f)

    eval(model, test_ds, device, test_log_f)

if __name__ == '__main__':
    cfg_path = sys.argv[1]
    sys.argv = sys.argv[1:]

    if os.path.isdir(cfg_path):
        cfg_dir = cfg_path
        cfg_file = None
    else:
        cfg_dir = os.path.dirname(cfg_path)
        cfg_file = os.path.basename(cfg_path)

    if cfg_file.endswith('.yaml'):
        cfg_file = cfg_file[:-5]
        
    logging.info(cfg_dir, cfg_file)

    logging.basicConfig(level=logging.INFO,
                        stream=sys.stdout,
                        format='[%(asctime)s] %(message)s',
                        datefmt="%m-%d %H:%M:%S"
                        )
    
    hydra.main(config_path=cfg_dir, config_name=cfg_file)(main)()

