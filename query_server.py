#!/usr/bin/env python

import asyncio
import json
from websockets.server import serve



#!/usr/bin/env python3
# Copyright (c) Facebook, Inc. and its affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

"""
 Command line tool to get dense results and validate them
"""

import glob
import json
import logging
import pickle
import time
import zlib
from typing import Any, List, Optional, Tuple, Dict, Iterator

import hydra
import numpy as np
import torch
from omegaconf import DictConfig, OmegaConf
from torch import Tensor as T
from torch import nn
from dpr.models.hf_models import BertTensorizer

from dpr.utils.data_utils import RepTokenSelector
from dpr.data.qa_validation import calculate_matches, calculate_chunked_matches, calculate_matches_from_meta
from dpr.data.retriever_data import JsonLawQASrc, KiltCsvCtxSrc, TableChunk
from dpr.indexer.faiss_indexers import (
    DenseFlatIndexer,
    DenseIndexer,
)
from dpr.models import init_biencoder_components
from dpr.models.biencoder import (
    BiEncoder,
    _select_span_with_token,
)
from dpr.options import setup_logger, setup_cfg_gpu, set_cfg_params_from_state
from dpr.utils.data_utils import Tensorizer
from dpr.utils.model_utils import setup_for_distributed_mode, get_model_obj, load_states_from_checkpoint

logger = logging.getLogger()
setup_logger(logger)


def generate_question_vectors(
    question_encoder: torch.nn.Module,
    tensorizer: Tensorizer,
    questions: List[str],
    bsz: int,
    query_token: Optional[str] = None,
    selector: Optional[RepTokenSelector] = None,
) -> T:
    n = len(questions)
    query_vectors = []

    with torch.no_grad():
        for j, batch_start in enumerate(range(0, n, bsz)):
            batch_questions = questions[batch_start : batch_start + bsz]

            if query_token:
                # TODO: tmp workaround for EL, remove or revise
                if query_token == "[START_ENT]":
                    batch_tensors = [
                        _select_span_with_token(q, tensorizer, token_str=query_token) for q in batch_questions
                    ]
                else:
                    batch_tensors = [tensorizer.text_to_tensor(" ".join([query_token, q])) for q in batch_questions]
            elif isinstance(batch_questions[0], T):
                batch_tensors = [q for q in batch_questions]
            else:
                # this case is chosen in this assignment.
                batch_tensors = [tensorizer.text_to_tensor(q) for q in batch_questions]

            q_ids_batch = torch.stack(batch_tensors, dim=0).cuda() # type: ignore
            q_seg_batch = torch.zeros_like(q_ids_batch).cuda()
            q_attn_mask = tensorizer.get_attn_mask(q_ids_batch)

            if selector:
                rep_positions = selector.get_positions(q_ids_batch, tensorizer)

                _, out, _ = BiEncoder.get_representation(
                    question_encoder,
                    q_ids_batch,
                    q_seg_batch,
                    q_attn_mask,
                    representation_token_pos=rep_positions,
                )
            else: # selector is None
                _, out, _ = question_encoder(q_ids_batch, q_seg_batch, q_attn_mask)

            query_vectors.extend(out.cpu().split(1, dim=0))

            if len(query_vectors) % 100 == 0:
                logger.info("Encoded queries %d", len(query_vectors))

    query_tensor = torch.cat(query_vectors, dim=0)
    logger.info("Total encoded queries tensor %s", query_tensor.size())
    assert query_tensor.size(0) == len(questions)
    return query_tensor


class DenseRetriever(object):
    def __init__(self, question_encoder: nn.Module, batch_size: int, tensorizer: Tensorizer):
        self.question_encoder = question_encoder
        self.batch_size = batch_size
        self.tensorizer = tensorizer
        self.selector = None

    def generate_question_vectors(self, questions: List[str], query_token: Optional[str] = None) -> T:

        bsz = self.batch_size
        self.question_encoder.eval()
        return generate_question_vectors(
            self.question_encoder,
            self.tensorizer,
            questions,
            bsz,
            query_token=query_token,
            selector=self.selector,
        )


class LocalFaissRetriever(DenseRetriever):
    """
    Does passage retrieving over the provided index and question encoder
    """

    def __init__(
        self,
        question_encoder: nn.Module,
        batch_size: int,
        tensorizer: Tensorizer,
        index: DenseIndexer,
    ):
        super().__init__(question_encoder, batch_size, tensorizer)
        self.index = index

    def index_encoded_data(
        self,
        vector_files: List[str],
        buffer_size: int, # 49920
        path_id_prefixes: Optional[List] = None,
    ):
        """
        Indexes encoded passages takes form a list of files
        :param vector_files: file names to get passages vectors from
        :param buffer_size: size of a buffer (amount of passages) to send for the indexing at once
        :return:
        """
        buffer = []
        assert self.index
        for i, item in enumerate(iterate_encoded_files(vector_files, path_id_prefixes=path_id_prefixes)):
            buffer.append(item)
            if 0 < buffer_size == len(buffer):
                self.index.index_data(buffer)
                buffer = []
        self.index.index_data(buffer)
        logger.info("Data indexing completed.")

    def get_top_docs(self, query_vectors: np.ndarray, top_docs: int = 100) -> List[Tuple[List[object], List[float]]]:
        """
        Does the retrieval of the best matching passages given the query vectors batch
        :param query_vectors:
        :param top_docs:
        :return:
        """
        time0 = time.time()
        assert self.index
        results = self.index.search_knn(query_vectors, top_docs)
        logger.info("index search time: %f sec.", time.time() - time0)
        return results



def validate(
    passages: Dict[object, Tuple[str, str]],
    answers: List[List[str]],
    result_ctx_ids: List[Tuple[List[object], List[float]]],
    workers_num: int,
    match_type: str,
) -> List[List[bool]]:
    logger.info("validating passages. size=%d", len(passages))
    match_stats = calculate_matches(passages, answers, result_ctx_ids, workers_num, match_type)
    top_k_hits = match_stats.top_k_hits

    logger.info("Validation results: top k documents hits %s", top_k_hits)
    top_k_hits = [v / len(result_ctx_ids) for v in top_k_hits]
    logger.info("Validation results: top k documents hits accuracy %s", top_k_hits)
    return match_stats.questions_doc_hits


def save_results(
    passages: Dict[object, Tuple[str, str]],
    questions: List[str],
    answers: List[List[str]],
    top_passages_and_scores: List[Tuple[List[object], List[float]]],
    per_question_hits: List[List[bool]],
    out_file: str,
):
    # join passages text with the result ids, their questions and assigning has|no answer labels
    merged_data = []
    # assert len(per_question_hits) == len(questions) == len(answers)
    for i, q in enumerate(questions):
        q_answers = answers[i]
        results_and_scores = top_passages_and_scores[i]
        hits = per_question_hits[i]
        docs = [passages[doc_id] for doc_id in results_and_scores[0]]
        scores = [str(score) for score in results_and_scores[1]]
        ctxs_num = len(hits)

        results_item = {
            "question": q,
            "answers": q_answers,
            "ctxs": [
                {
                    "id": results_and_scores[0][c],
                    "title": docs[c][1],
                    "text": docs[c][0],
                    "score": scores[c],
                    "has_answer": hits[c],
                }
                for c in range(ctxs_num)
            ],
        }

        # if questions_extra_attr and questions_extra:
        #    extra = questions_extra[i]
        #    results_item[questions_extra_attr] = extra

        merged_data.append(results_item)

    with open(out_file, "w") as writer:
        writer.write(json.dumps(merged_data, indent=4,ensure_ascii=False) + "\n")
    logger.info("Saved results * scores  to %s", out_file)



def iterate_encoded_files(vector_files: list, path_id_prefixes: Optional[List] = None) -> Iterator[Tuple]:
    for i, file in enumerate(vector_files):
        logger.info("Reading file %s", file)
        id_prefix = None
        if path_id_prefixes:
            id_prefix = path_id_prefixes[i]
        with open(file, "rb") as reader:
            doc_vectors = pickle.load(reader)
            logger.info("length of doc vectors: %d",len(doc_vectors))
            for doc in doc_vectors:
                doc = list(doc)
                if id_prefix and not str(doc[0]).startswith(id_prefix):
                    doc[0] = id_prefix + str(doc[0])
                yield doc # type: ignore



def get_all_passages():
    with open('/mnt/d/github/law-retrieval/filtered_passage_100.pickl', 'rb') as prf:
        return pickle.load(prf)

def pack_results(
    passages: Dict[object, Tuple[str, str]],
    questions: List[str],
    answers: List[List[str]],
    top_passages_and_scores: List[Tuple[List[object], List[float]]],
    per_question_hits: List[List[bool]],
):
    # join passages text with the result ids, their questions and assigning has|no answer labels
    merged_data = []
    # assert len(per_question_hits) == len(questions) == len(answers)
    for i, q in enumerate(questions):
        q_answers = answers[i]
        results_and_scores = top_passages_and_scores[i]
        hits = per_question_hits[i]
        docs = [passages[doc_id] for doc_id in results_and_scores[0]]
        scores = [str(score) for score in results_and_scores[1]]
        ctxs_num = len(hits)

        results_item = {
            "question": q,
            "answers": q_answers,
            "ctxs": [
                {
                    "id": results_and_scores[0][c],
                    "title": docs[c][1],
                    "text": docs[c][0],
                    "score": scores[c],
                    "has_answer": hits[c],
                }
                for c in range(ctxs_num)
            ],
        }

        # if questions_extra_attr and questions_extra:
        #    extra = questions_extra[i]
        #    results_item[questions_extra_attr] = extra

        merged_data.append(results_item)
    return merged_data

init_res: Any

@hydra.main(config_path="conf", config_name="dense_retriever")
def retrieve_main(cfg: DictConfig):
    logger.info("parameter initialization.")
    # region parameter initialization.
    cfg = setup_cfg_gpu(cfg)
    saved_state = load_states_from_checkpoint(cfg.model_file)

    set_cfg_params_from_state(saved_state.encoder_params, cfg)

    logger.info("CFG (after gpu  configuration):")
    logger.info("%s", OmegaConf.to_yaml(cfg))
    # endregion

    logger.info("construct model")
    # region construct model
    tensorizer, encoder, _ = init_biencoder_components(cfg.encoder.encoder_model_type, cfg, inference_only=True)

    logger.info("Loading saved model state ...")
    encoder.load_state(saved_state, strict=False)

    encoder_path = cfg.encoder_path # either question model or context model.
    if encoder_path:
        logger.info("Selecting encoder: %s", encoder_path)
        encoder = getattr(encoder, encoder_path)
    else:
        logger.info("Selecting standard question encoder")
        encoder = encoder.question_model

    encoder, _ = setup_for_distributed_mode(encoder, None, cfg.device, cfg.n_gpu, cfg.local_rank, cfg.fp16)
    encoder.eval()

    model_to_load = get_model_obj(encoder)
    vector_size = model_to_load.get_out_size()
    logger.info("Encoder vector_size=%d", vector_size)
    # endregion
    
    logger.info("setup retriever.")
    # region setup retriever
    index:DenseFlatIndexer = hydra.utils.instantiate(cfg.indexers[cfg.indexer])
    logger.info("Local Index class %s ", type(index))
    index_buffer_sz = index.buffer_size
    index.init_index(vector_size)
    retriever = LocalFaissRetriever(encoder, cfg.batch_size, tensorizer, index)
    # endregion
    
    logger.info("index the passages.")
    # region index all passages
    all_passages = get_all_passages()

    ctx_files_patterns = cfg.encoded_ctx_files

    logger.info("ctx_files_patterns: %s", ctx_files_patterns)
    input_paths = []
    for i, pattern in enumerate(ctx_files_patterns):
        pattern_files = glob.glob(pattern)
        input_paths.extend(pattern_files)
    logger.info("Reading all passages data from files: %s", input_paths)
    retriever.index_encoded_data(input_paths, index_buffer_sz, path_id_prefixes=None)
    # endregion
    global init_res
    init_res = (all_passages, retriever, cfg.n_docs, cfg.validation_workers, cfg.match)

def query(all_passages, retriever:LocalFaissRetriever, query_ctx:str, doc_count: int, validation_workers: int, match_type: str):
    questions = [query_ctx]
    answers = [[]]
    questions_tensor = retriever.generate_question_vectors(questions, query_token=None)
    
    # get top k results
    top_results_and_scores = retriever.get_top_docs(questions_tensor.numpy(), doc_count)
    # convert retrieval results to readable documents
    questions_doc_hits = validate(
        all_passages,
        answers,
        top_results_and_scores,
        validation_workers,
        match_type,
    )
    # save the result to the file specified.
    results = pack_results(all_passages,questions, answers, top_results_and_scores, questions_doc_hits)
    return results

async def echo(websocket):
    global init_res
    all_passages, retriever, n_docs, validation_workers, match_type =init_res
    async for message in websocket:
        message = json.loads(message)
        if message['mode'] == 'query':
            qry = message['content']
            msg = query(all_passages, retriever, qry, n_docs, validation_workers, match_type)
            await websocket.send(json.dumps(msg, ensure_ascii=False))

async def main():
    async with serve(echo, "localhost", 8765):
        await asyncio.Future()  # run forever

query_ids = [5223] # 5156 5223 #[1, 4738, 27, 1972, 5156, 24, 5223, 0, 861, 837]

data_path = '/mnt/d/github/THU_PASS/year3/Information_Retreival/Assignments/data/data/'
query_path = data_path + 'query.json'

def load_json(file):
    with open(file, 'r', encoding='utf-8') as f:
        obj = json.load(f)
    return obj

def hello():
    global init_res
    all_passages, retriever, n_docs, validation_workers, match_type =init_res
    q = load_json(query_path)
    q = [x for x in q if x['ridx'] in query_ids]
    print(len(q))
    msglist = {}
    for query__ in q:
        msg = query(all_passages[str(query__['ridx'])], retriever, query__['q'], n_docs, validation_workers, match_type)

        payload = msg
        context_lst = payload[0]['ctxs']
        contents = {}
        for context in context_lst:
            contents[context['id']] = float(context['score'])
        
        msglist[str(query__['ridx'])] = contents

    with open(f'received_contents_{str(query_ids[0])}.json', 'w') as wf:
        print(msglist, file=wf)

if __name__ == "__main__":
    retrieve_main()
    logger.info('initialization done')
    hello()
    #asyncio.run(main())