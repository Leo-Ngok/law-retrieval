

import json
import os
from dpr.data.biencoder_data import BiEncoderPassage
from dpr.data.retriever_data import JsonLawCtxSrc, RetrieverData
from pickle import load,dump
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

def _get_all_passages(ctx_sources: list[RetrieverData]):
    all_passages = {}
    for ctx_src in ctx_sources:
        ctx_src.load_data_to(all_passages)
        print("Loaded ctx data: %d", len(all_passages))

    if len(all_passages) == 0:
        raise RuntimeError("No passages data found. Please specify ctx_file param properly.")
    return all_passages

if __name__ == '__main__':
    #gen_doc_dict(sys.argv[1])
    #gen_doc_dict('/mnt/d/github/THU_PASS/year3/Information_Retreival/Assignments/data/data/documents')
    ctx_src = JsonLawCtxSrc(query_ridx_path="/mnt/d/github/THU_PASS/year3/Information_Retreival/Assignments/data/data/documents", id_prefix="")
    passages = _get_all_passages([ctx_src])
    with open('passage.pickl', 'wb') as wf:
        dump(passages, wf)
