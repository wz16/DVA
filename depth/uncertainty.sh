PYTHONPATH="$(dirname $0)/..":"$(dirname $0)/../stable-diffusion":$PYTHONPATH \
python3 -m torch.distributed.launch --nproc_per_node=1 --use_env uncertainty.py
