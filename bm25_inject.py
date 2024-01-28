import json
from elasticsearch import Elasticsearch
import os
from elasticsearch import Elasticsearch, helpers

'''
作业一：
对于candidates的所有文档在本地建立索引
'''

def load_json(file):
    with open(file, 'r', encoding='utf-8') as f:
        obj = json.load(f)
    return obj
data_path = '/mnt/d/github/THU_PASS/year3/Information_Retreival/Assignments/data/data/'

es = Elasticsearch(hosts='http://127.0.0.1:9200')

resp = es.info()
print(resp)

query_path = data_path + 'query.json'
q = load_json(query_path)
try:
    es.indices.delete(index='total')
except: pass
try:
    es.indices.delete(index='law')
except: pass

# 创建索引时指定 smartcn 分词器，对指定字段分词
body = {
    "settings": {
        "analysis": {
            "analyzer": {
                "my_analyzer": {
                    "type": "smartcn"
                }
            }
        }
    },
    "mappings": {
        "properties": {
            "ajName": {
                "type": "text",
                "analyzer": "my_analyzer"
            },
            "ajjbqk": {
                "type": "text",
                "analyzer": "my_analyzer"
            },
            "cpfxgc": {
                "type": "text",
                "analyzer": "my_analyzer"
            },
            "pjjg": {
                "type": "text",
                "analyzer": "my_analyzer"
            },
            "qw": {
                "type": "text",
                "analyzer": "my_analyzer"
            }
        }
    }
}

'''
批量插入数据
'''

actions = []
query_ids = [1, 4738, 27, 1972, 5156, 24, 5223, 0, 861, 837]
q = [x for x in q if x['ridx'] in query_ids]
for x in q:
    qid = x['ridx']
    candidate_root = data_path + 'candidates/' + str(qid) + '/'
    try:
        es.indices.delete(index=str(qid))
    except: pass
    es.indices.create(index=str(qid), body=body)
    for file in os.listdir(candidate_root):
        if file.endswith('.json'):
            file_path = candidate_root + file
            candidate = load_json(file_path)
            action = {
                "_index": qid,
                "_id": int(file[:-5]),
                "_source": candidate
            }
            actions.append(action)

try:
    helpers.bulk(es, actions)
except helpers.BulkIndexError as e:
    for error in e.errors:
        print(error)