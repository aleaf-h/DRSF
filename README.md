Training

CUDA_VISIBLE_DEVICES=0 python -m torch.distributed.launch --nproc_per_node=1 --master_port 12345 train.py --config path/to/config.yaml

Testing

python test.py --config path/to/config.yaml
