

torchrun --nproc_per_node 1 \
    gen_embeddings.py \
    fp16=True \
    encoder.pretrained_model_cfg="/mnt/d/hf/Lawformer" \
    encoder.sequence_length=512 \
    model_file="/mnt/d/model_train/law_model/dpr_biencoder.1" \
    ctx_sources=law_sources \
    ctx_src=default \
    +base_path="/mnt/d/github/THU_PASS/year3/Information_Retreival/Assignments/data/data/documents" \
    out_file="embeddings.pickl" | tee debug.log

    