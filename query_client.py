#!/usr/bin/env python

import asyncio
import json
from websockets.sync.client import connect

def hello():
    with connect("ws://localhost:8765") as websocket:
        query = {
            "mode": "query",
            "content": "莫新国在长沙市酒后驾驶车辆被交通警察检查，血液酒精含量195毫克／100毫升。他随后被带到医院提取血样，检验结果显示酒精含量201.1毫克／100毫升。被告人莫新国被认定为精神残疾人，但在案发时被鉴定为处于普通醉酒状态，有完全刑事责任能力。他在日后主动投案，如实供述罪行。"
        }
        websocket.send(json.dumps(query))
        message = websocket.recv()
        #print(f"Received: {message}")
    with open('received_contents_other.json', 'w') as wf:
        print(message, file=wf)

hello()