import torch
import torch.nn as nn
from transformers import BertModel
import math

""" Activation """
def gelu(x):
    return 0.5 * x * (1 + torch.tanh(math.sqrt(2 / math.pi) * (x + 0.044715 * torch.pow(x, 3))))

class ClassifierBase(nn.Module):
    def __init__(self, hidden_size, num_labels):
        super(ClassifierBase, self).__init__()
        self.classifier = nn.Linear(hidden_size, num_labels)
    def forward(self, inp):
        out = gelu(inp)
        out = self.classifier(out)
        return out



class BERTforKLUE(nn.Module):
    """
    Custom implementation instead of using BERTforClassification to try additional stuff
    such as changing the classification layer
    """
    def __init__(self, cfg):
        super(BERTforKLUE, self).__init__()

        self.num_labels = 7     # number of labels for ynat dataset
        self.hidden_size = 768  # BERT_base hidden size
        self.loss_fct = nn.CrossEntropyLoss()

        self.bert = BertModel.from_pretrained("kykim/bert-kor-base")
        self.dropout = nn.Dropout()
        
        self.classifier = None
        if cfg.finetune_type == 'base':
            self.classifier = ClassifierBase(self.hidden_size, self.num_labels)
        
        
    def forward(self, input_ids, labels):
        out = self.bert(input_ids)
        out = self.dropout(out[0])      # (bsz, max_seq_len, hidden_Size)
        out = out[:,0,:].view(-1, self.hidden_size)     # take the CLS token hidden vector
        
        logits = self.classifier(out)   # (bsz, num_labels)
        loss = self.loss_fct(logits, labels)
        
        return loss, logits


