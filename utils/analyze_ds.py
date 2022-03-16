import json

train = open("../data/ynat-v1.1/ynat-v1.1_train.json")
train = json.load(train)
print(f"train ds samples: {len(train)}")

dev = open("../data/ynat-v1.1/ynat-v1.1_dev_origin.json")
dev = json.load(dev)

test = dev[:int(len(dev)*0.8)]
dev = dev[len(test):]

print(f"dev ds samples: {len(dev)}")
print(f"test ds samples: {len(test)}")

with open("../data/ynat-v1.1/ynat-v1.1_dev.json", "w") as dev_file:
    json.dump(dev, dev_file)

with open("../data/ynat-v1.1/ynat-v1.1_test.json", "w") as test_file:
    json.dump(test, test_file)
