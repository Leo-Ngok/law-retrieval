

import json
import os
from dpr.data.biencoder_data import BiEncoderPassage

def gen_doc_dict(dir: str) -> dict[str, BiEncoderPassage]:
    ctx = {}
    for filename in os.listdir(dir):
        fpfx = filename.split('.')[0]
        with open(os.path.join(dir, filename), 'r') as rf:
            dt = json.load(rf)
            ctx[fpfx] = BiEncoderPassage(
                dt["ajjbqk"],
                dt["ajName"]
            )
    return ctx

if __name__ == '__main__':
    #gen_doc_dict(sys.argv[1])
    gen_doc_dict('/mnt/d/github/THU_PASS/year3/Information_Retreival/Assignments/data/data/documents')