import numpy as np
from torch.utils.data import Dataset

# label mapper from 'labels' to a label index
label_map = {"IT과학": 0,
            "경제": 1,
            "사회": 2, 
            "생활문화": 3,
            "세계": 4,
            "스포츠": 5,
            "정치": 6
            }


# customized KLUE dataset 
class KLUEDataset(Dataset):
    def __init__(self, data, tokenizer, cfg):
        self.data = data
        self.tokenizer = tokenizer
        self.max_seq_len = cfg.max_seq_len
    
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        curr_item = self.data[idx]
        title = self.tokenizer(curr_item['title'],
                               padding='max_length', 
                               truncation=True, 
                               max_length=self.max_seq_len
                               )['input_ids']  
        label = label_map[curr_item['label']]

        return np.array(title), np.array(label)


def build_dataset(raw_data, tokenizer, cfg):
    return KLUEDataset(raw_data, tokenizer, cfg)
