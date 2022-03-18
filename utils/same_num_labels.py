import json
import random

label_map = {"IT과학": 0,
            "경제": 1,
            "사회": 2, 
            "생활문화": 3,
            "세계": 4,
            "스포츠": 5,
            "정치": 6
            }

train = open("../data/ynat-v1.1/ynat-v1.1_train.json")
train = json.load(train)

ds = {0:[], 1:[], 2:[], 3:[], 4:[], 5:[], 6:[]}
for item in train:
    ds[label_map[item['label']]].append(item)

min_cnt = 10000
for label in ds:
    print(f"for label {label}: {len(ds[label])}")
    if len(ds[label]) < min_cnt:
        min_cnt = len(ds[label])

same_num_ds = ds[0][:min_cnt]
for i in range(1, 7):
    same_num_ds += ds[i][:min_cnt]

print(f"take {min_cnt} for each label class.")
print(f"resulting train set has {len(same_num_ds)} samples.")

random.shuffle(same_num_ds)
with open("../data/ynat-v1.1/ynat-v1.1_train_same_num_label.json", "w") as train_file:
    json.dump(same_num_ds, train_file)
