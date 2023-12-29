import json
import random
import os

os.chdir("./data/")

def read_jsonl(file_path):
    data = []
    with open(file_path, 'r', encoding='utf-8') as f:
        for line in f:
            data.append(json.loads(line.strip()))
    return data

def write_jsonl(data, file_path):
    with open(file_path, 'w', encoding='utf-8') as f:
        for item in data:
            s = json.dumps(item, ensure_ascii=False)
            f.write(s)
            f.write('\n')

def split_data(file_path, train_file_path, test_file_path, test_size=0.2):
    data = read_jsonl(file_path)
    random.shuffle(data)
    data = data[60000:]
    split_point = int(len(data) * (1 - test_size))
    train_data = data[:split_point]
    test_data = data[split_point:]
    write_jsonl(train_data, train_file_path)
    write_jsonl(test_data, test_file_path)

# 使用方法
split_data('data.jsonl', 'train_data.jsonl', 'validation_data.jsonl', test_size=0.2)