# 本仓库使用hydra解析参数
# 在本文件中未指定的参数，使用./conf/biencoder_train_cfg.yaml中的默认配置，若在本文件中指定参数，则覆盖上述yaml文件中的默认配置
# 若需要修改训练超参数，可在.conf/train/biencoder_Law.yaml中修改，或者在本文件中使用train.xxx=xxx指定
# 若需要从已经得到的checkpoint继续训练，可将下方checkpoint_file_name指定为checkpoint路径

# TODO for class
log_path="./train.log"
train_data_path="/mnt/d/github/law-retrieval/data/train_data.jsonl"
dev_data_path="/mnt/d/github/law-retrieval/data/validation_data.jsonl"
output_checkpoint_dir="/mnt/d/model_train/law_model"
model_file_path="/mnt/d/hf/Lawformer"

torchrun --nproc_per_node 1 \
    train_dense_encoder.py \
    fp16=True \
    encoder.pretrained_model_cfg=${model_file_path} \
    encoder.sequence_length=256 \
    train=biencoder_Law \
    train.num_train_epochs=4 \
    val_av_rank_start_epoch=4 \
    checkpoint_file_name=dpr_biencoder \
    train_datasets=[Law_data_train] \
    datasets.Law_data_train.file=${train_data_path} \
    dev_datasets=[Law_data_dev] \
    datasets.Law_data_dev.file=${dev_data_path} \
    output_dir=${output_checkpoint_dir} \
    | tee ${log_path}