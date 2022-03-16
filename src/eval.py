import logging
from tqdm import tqdm
import torch

def eval(model, ds, device, log):
    model.eval()
    num_correct = 0
    num_samples = 0

    for iteration, batch in enumerate(tqdm(ds)):
        num_samples += int(batch[0].shape[0])

        input_ids = batch[0].to(device)    # (bsz, max_seq_len)
        labels = batch[1].to(device)       # (bsz)
        loss, logits = model(input_ids, labels)
        pred = torch.argmax(logits, dim=-1)
        num_correct += torch.sum((pred == labels).int())

    log.write(f"[Accuracy]: {num_correct / num_samples}\n")
    log.flush()

    model.train()
