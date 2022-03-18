# KLUE-classification
💬 Text Classification using KLUE benchmark  
This repository uses the [bert-kor-base](https://huggingface.co/kykim/bert-kor-base/tree/main) model to do text classification with the KLUE benchmark.

### 📌 Some Requirements
- place the data in `./data/ynat-v1.1`
- the `hydra-core` package (refer to `requirements.txt`)
-------------
## 🚀 How to Run 
The script below trains the bert-kor-base model.
```
run.sh {LR} {EPOCHS} {BSZ} {MAX_SEQ_LEN} {FT_TYPES}
```
- `FT_TYPES` are used to choose the finetuning layer. Possible options are;
  - `base`: a simple, single linear layer
  - `one`: two linear layers

## ⚙️ Configurations
All configurations are in `conf/base.yaml`, which can be overriden in the commandline.
The below are specific explanations of each configuration
```
# Paths
train_path        path of training json file
dev_path          path of dev json file
save_path         path to save checkpoints of model
train_log_path    path to keep the train log file
test_log_path     path to keep the test log file (for testing without training)
load_path         false or actual path of ckpt to load

# Training
epochs            number of epochs to train the model
batch_size        batch size for train & dev
lr                learning rate of Adam optimizer
finetune_type     choosing the finetuning layer
max_seq_len       maximum sequence length of input to BERT

# Logging & Validating
val_interval      performs validation with the dev set every {val_interval} epochs
log_interval      prints the training loss every {log_interval} epochs
save_interval     saves the model checkpoint every {save_interval} epochs

# Preprocessing
remove_stopwords  true/false for removing stopword-containing tokens when preprocessing
remove_num        true/false for removing number-containing tokens when preprocessing

# Others
test_only         true for testing without training
seed              specify an integer seed
```

## 🖍 Checking Outputs
Running `run.sh`...
- In training mode, the **train loss and test accuracies** will be stored in `./outputs/${FT_TYPE}/lr-${LR}_epoch-${EPOCHS}_bsz-${BSZ}_msl-${MAX_SEQ_LEN}/${NOW}/log/train.log`
- In testing mode, the test accuracy will be stored in `./outputs/${FT_TYPE}/lr-${LR}_epoch-${EPOCHS}_bsz-${BSZ}_msl-${MAX_SEQ_LEN}/${NOW}/log/test.log`

