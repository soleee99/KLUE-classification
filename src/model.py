import torch
import torch.nn as nn
from transformers import BertModel

tokenizer_bert = BertTokenizerFast.from_pretrained("kykim/bert-kor-base")
model_bert = BertModel.from_pretrained("kykim/bert-kor-base")

class BERTforKLUE(nn.Module):
    """
    Custom implementation instead of using BERTforClassification to try additional stuff
    such as changing the classification layer
    """
    def __init__(self, cfg):
        self.num_labels = 7
        self.loss_fct = nn.CrossEntropyLoss()

        self.bert = BertModel.from_pretrained("kykim/bert-kor-base")
        self.dropout = nn.Dropout()
        
        self.classifier = None
        if cfg.finetune_type == 'base':
            self.classifier = nn.Linear(cfg.hidden_size, self.num_labels)
        
        
    def forward(self, input_ids, labels):
        out = self.bert(input_ids)
        out = self.dropout(out[0])
        logits = self.classifier(out)

        loss = self.loss_fct(logits, labels)
        return loss


