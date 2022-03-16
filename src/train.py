import logging

def train(cfg, model, optimizer, train_ds, val_ds):
    logging.info('=====================Train=====================')
    model.train()

    epochs = cfg.epochs
    total_iterations = len(train_ds)

    for epoch in range(epochs):
        logging.info("")
        logging.info(f"Start epoch {epoch+1} / {epochs}...")

        for iteration, batch in enumerate(train_ds):
            print(batch)
            break
