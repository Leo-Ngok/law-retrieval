torchrun --nproc_per_node 1 \
    query_server.py \
    encoder.pretrained_model_cfg="/mnt/d/hf/Lawformer" \
    encoder.sequence_length=512 \
    model_file="/mnt/d/model_train/law_model/dpr_biencoder.1" \
    indexer=flat \
    qa_dataset=lecard_short \
    ctx_datatsets=[lecard_short] \
    encoded_ctx_files=[\"/mnt/d/model_train/law_model/embeddings.pickl_0\"] \
    out_file=result.txt \
    fp16=True