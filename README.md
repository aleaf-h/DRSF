### Training
- To train a model, run the following script in your console. 
```{bash}
CUDA_VISIBLE_DEVICES=0 python -m torch.distributed.launch --nproc_per_node=1 --master_port 67890 train.py --config path/to/config.yaml
```
- `--config`: Specify the path of the config file. 

### Testing
- To test a model, run the following script in your console. 
```{bash}
python test.py --config path/to/config.yaml
```
- `--config`: Specify the path of the config file.
