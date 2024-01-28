from elasticsearch import Elasticsearch
import os
from elasticsearch import Elasticsearch, helpers
import json

'''
作业一：
针对作业一的检索，在此尝试了多种优化方法，并将最好的一种应用于作业二的views.py中。检索结果存放到result_test.json中
'''

es = Elasticsearch(hosts='http://127.0.0.1:9200')

resp = es.info()
print(resp)
def load_json(file):
    with open(file, 'r', encoding='utf-8') as f:
        obj = json.load(f)
    return obj
data_path = '/mnt/d/github/THU_PASS/year3/Information_Retreival/Assignments/data/data/'

query_path = data_path + 'query.json'
q = load_json(query_path)
query_ids = [1, 4738, 27, 1972, 5156, 24, 5223, 0, 861, 837]
q = [x for x in q if x['ridx'] in query_ids]

result = {}
fields = ['ajjbqk^4', 'qw^4', "ajName^1", "cpfxgc^2", "pjjg^1"]

length = len(q)
for i in range(len(query_ids)):
    query_terms = q[i]['words']
    query_body = {
        "size": 30,
        "query": {
            "bool": {
                "should": [{
                        "multi_match": {
                            "query": term,
                            "fields": fields
                        }
                    } for term in query_terms]
            }
        }
    }

    response = es.search(index=str(q[i]['ridx']), body=query_body)
    result[str(q[i]['ridx'])] = {str(hit['_id']): hit['_score'] for hit in response['hits']['hits']}
    
with open('result_bm25_4.json', 'w', encoding='utf-8') as f:
    json.dump(result, f, ensure_ascii=False, indent=4)