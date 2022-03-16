import logging
from tqdm import tqdm
from .eval import eval
from .utils import save_checkpoint

def train(cfg, model, optimizer, train_ds, dev_ds, device, train_log):
    logging.info('=====================Train=====================')
    model.train()

    epochs = cfg.epochs
    val_interval = cfg.val_interval
    log_interval = cfg.log_interval
    save_interval = cfg.save_interval

    for epoch in range(epochs):
        logging.info("")
        logging.info(f"Start epoch {epoch+1} / {epochs}...")
        accumulated_loss = 0.0
        
        for iteration, batch in enumerate(tqdm(train_ds)):
            input_ids = batch[0].to(device)    # (bsz, max_seq_len)
            labels = batch[1].to(device)       # (bsz)
            
            loss, _ = model(input_ids, labels)
            accumulated_loss += loss

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
        
        if (epoch + 1) % log_interval == 0:
            train_log.write(f"[Epoch {epoch+1} / {epochs}] loss: {accumulated_loss}\n")
            train_log.flush()
        if (epoch + 1) % val_interval == 0:
            eval(model, dev_ds, device, train_log)
        if (epoch + 1) % save_interval == 0:
            save_checkpoint(cfg, model, epoch)
