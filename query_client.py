#!/usr/bin/env python

import asyncio
import json
from websockets.sync.client import connect

query_ids = [1, 4738, 27, 1972, 5156, 24, 5223, 0, 861, 837]

data_path = '/mnt/d/github/THU_PASS/year3/Information_Retreival/Assignments/data/data/'
query_path = data_path + 'query.json'

def load_json(file):
    with open(file, 'r', encoding='utf-8') as f:
        obj = json.load(f)
    return obj

def hello():
    q = load_json(query_path)
    q = [x for x in q if x['ridx'] in query_ids]
    print(len(q))
    msglist = {}
    for query in q:
        with connect("ws://localhost:8765") as websocket:
            print(len(query['q']))
            __query = {
                "mode": "query",
                "content": query['q']
            }
            websocket.send(json.dumps(__query))
            payload = json.loads(websocket.recv())
            context_lst = payload[0]['ctxs']
            contents = {}
            for context in context_lst:
                contents[context['id']] = float(context['score'])
            
            msglist[str(query['ridx'])] = contents

    with open('received_contents_.json', 'w') as wf:
        print(msglist, file=wf)

hello()