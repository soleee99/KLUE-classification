import numpy as np
from torch.utils.data import Dataset


korean_stopwords = ["이", "있", "하", "것", "들", "그", "되", "수", "이", "보",
                    "않", "없", "나", "사람", "주", "아니", "등", "같", "우리",
                    "때", "년", "가", "한", "지", "대하", "오", "말", "일", "그렇",
                    "위하", "때문", "그것", "두", "말하", "알", "그러나", "받", "못하",
                    "그런", "또", "문제", "더", "사회", "많", "그리고", "좋", "크",
                    "따르", "중", "나오", "가지", "씨", "시키", "만들", "지금",
                    "생각하", "그러", "속", "하나", "집", "살", "모르", "적", "월",
                    "데", "자신", "인", "어떤", "내", "경우", "명", "생각", "시간",
                    "그녀", "다시", "이런", "앞", "보이", "번", "나", "다른", "어떻",
                    "여자", "개", "전", "들", "사실", "이렇", "점", "싶", "말", "정도",
                    "좀", "원", "잘", "통하", "소리", "놓"]



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
        self.remove_stopwords = cfg.remove_stopwords
        self.remove_num = cfg.remove_num
    
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

        if self.remove_stopwords:
            decoded = [self.tokenizer.decode(input_id) for input_id in title]
            preproc_title = []
            for i, word in enumerate(decoded):
                for stopword in korean_stopwords:
                    if stopword not in word:
                        preproc_title.append(title[i])
            title = preproc_title + [0] * (self.max_seq_len - len(preproc_title))
        #print(f"{[self.tokenizer.decode(input_id) for input_id in title]}\n")

        if self.remove_num:
            def has_numbers(inputString):
                return any(char.isdigit() for char in inputString)
            decoded = [self.tokenizer.decode(input_id) for input_id in title]
            preproc_title = []
            for i, word in enumerate(decoded):
                if not has_numbers(word):
                    preproc_title.append(title[i])
            title = preproc_title + [0] * (self.max_seq_len - len(preproc_title))

        return np.array(title), np.array(label)


def build_dataset(raw_data, tokenizer, cfg):
    return KLUEDataset(raw_data, tokenizer, cfg)
