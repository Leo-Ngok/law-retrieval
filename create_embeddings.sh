

torchrun --nproc_per_node 1 \
    gen_embeddings.py \
    fp16=True \
    encoder.pretrained_model_cfg="/root/autodl-tmp/DPR/downloads/model/Lawformer_model" \
    encoder.sequence_length=512 \
    model_file="/root/autodl-tmp/DPR/model_train/law_model/dpr_biencoder.19" \
    ctx_sources=law_sources \
    ctx_src=default \
    batch_size=128 \
    +base_path="/root/autodl-tmp/documents" \
    out_file="embeddings.pickl" | tee debug.log

    