import json
import math

data_path = '/mnt/d/github/THU_PASS/year3/Information_Retreival/Assignments/data/data/'
query_path = data_path + 'query.json'
candidates_path = data_path + 'candidates/'
gt_path = data_path + 'label_top30_dict.json'

def load_json(file):
    with open(file, 'r', encoding='utf-8') as f:
        obj = json.load(f)
    return obj


'''
作业一：
计算NDCG@k
'''
def NDCG_k(k: int, query: int):
    # IDCG_k
    gt = load_json(gt_path)
    gt_dict = gt[str(query)]
    gt = list(gt_dict.items())
    gt.sort(key=lambda x: x[1], reverse=True)
    idcg = 0
    for i in range(k):
        idcg += gt[i][1] / math.log2(i + 2)

    # DCG_k
    result = load_json('result_bm25_4.json')
    ob = result[str(query)]
    ob = list(ob.items())
    ob.sort(key=lambda x: x[1], reverse=True)
    dcg = 0
    for i in range(k):
        dcg += gt_dict[ob[i][0]] / math.log2(i + 2) if ob[i][0] in gt_dict else 0
    
    return dcg / idcg

'''
从result_test.json中读取结果
计算指标并打印
'''
q = load_json(query_path)
query_ids = [1, 4738, 27, 1972, 5156, 24, 5223, 0, 861, 837]
q = [x for x in q if x['ridx'] in query_ids]
length = len(q)
metrixs = {}
andcg_5 = 0
andcg_10 = 0
andcg_30 = 0
for x in q:
    query = x['ridx']
    ndcg_5 = NDCG_k(5, query)
    ndcg_10 = NDCG_k(10, query)
    ndcg_30 = NDCG_k(30, query)
    metrixs[str(query)] = {"NDCG@5": ndcg_5, "NDCG@10": ndcg_10, "NDCG@30": ndcg_30}
    andcg_5 += ndcg_5
    andcg_10 += ndcg_10
    andcg_30 += ndcg_30
with open('metrix_4.json', 'w', encoding='utf-8') as f:
    json.dump(metrixs, f, ensure_ascii=False, indent=4)

print("NDCG@5:", andcg_5 / length)
print("NDCG@10:", andcg_10 / length)
print("NDCG@30:", andcg_30 / length)