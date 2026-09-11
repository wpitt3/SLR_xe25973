# Bash scripts

## General
### ssh onto spot (local)
```
ssh -i /usr/local/development/SLT_xe25973/pems/ap-northeast-2_xe25973.pem ubuntu@$SPOT_IP
```

### copy config to spot / setup MSKA
#### local
```
scp -i /usr/local/development/SLT_xe25973/pems/ap-northeast-2_xe25973.pem MSKA/configs/csl-daily_s2g.yaml ubuntu@$SPOT_IP:~
```
#### remote
```
git clone https://github.com/sutwangyan/MSKA
rm MSKA/configs/csl-daily_s2g.yaml 
mv csl-daily_s2g.yaml MSKA/configs/csl-daily_s2g.yaml 
```
### setup env
```
source /opt/pytorch/bin/activate
pip install pyyaml einops opencv-python-headless loguru wandb transformers matplotlib seaborn
pip uninstall -y tensorflow triton pytorch-triton
pip install tensorflow-cpu portalocker regex tabulate colorama rouge-chinese jieba nltk transformers einops pyyaml opencv-python-headless 
python -c "import torch, tensorflow as tf; print(torch.cuda.is_available(), tf.__version__)"
python -c "import numpy;print(numpy.__version__)"
```

## Pre-train
### Copy script to spot

#### local (on changes)
```
tar -czf pts.tar.gz src/
```
#### local
```
scp -i /usr/local/development/SLT_xe25973/pems/ap-northeast-2_xe25973.pem pts.tar.gz ubuntu@$SPOT_IP:~
```
#### remote
```
tar xzf pts.tar.gz 
mv src/* MSKA
```

### Copy shards to spot

#### remote
```
sudo mkdir -p /opt/dlami/nvme/data && sudo chown $USER /opt/dlami/nvme/data
ln -sfn /opt/dlami/nvme/data ~/data
df -h /opt/dlami/nvme
aws s3 sync s3://mska-ap-northeast-2-xe25973/data/shards/ ~/data/csl-news

mkdir -p ~/data/csl-news-heldout
mv data/csl-news/csl-news-00074.tar data/csl-news-heldout/
mv data/csl-news/csl-news-00075.tar data/csl-news-heldout/
```

### Run pretraining
```
tmux new -s pretrain
source /opt/pytorch/bin/activate

python -u pretrain.py --shards ~/data/csl-news --val-shards ~/data/csl-news-heldout \
  --config configs/csl-daily_s2g.yaml --fraction 0.20 \
  --steps 30000 --batch-size 10 --eval-every 250 --log-every 100 \
  --local-frac 0.217 --p-span 0.2 --p-joint 0.4 --p-stream 0.4 --span-frac 0.5 \
  --span-min 12 --span-max 40 --stream-win 0.7 --joint-frac 0.5 --joint-win 0.6 --out pretrain_out 2>&1 | tee pretrain.log


```

### Save pretrain
```
aws s3 sync pretrain_out/ s3://mska-ap-northeast-2-xe25973/pretrain/
```

### Run probe 
```
python -u probe_cached.py --config configs/csl-daily_s2g.yaml --encoder random --epochs 40 2>&1 | tee random.log
python -u probe_cached.py --config configs/csl-daily_s2g.yaml --encoder /pretrain_out/enc.pth --epochs 40 2>&1 | tee x_probe.log
```



## Fine tune

### Copy new train.py
``` local
scp -i /usr/local/development/SLT_xe25973/pems/ap-northeast-2_xe25973.pem MSKA/train.py ubuntu@$SPOT_IP:~/MSKA/
```

### Copy csl_daily_data to spot
#### remote
```
cd MSKA
aws s3 cp s3://mska-ap-northeast-2-xe25973/data/csl_daily_data.tgz .
tar xzf csl_daily_data.tgz
```

### tmux

#### new window
```
tmux new -s train
```
#### escape
```
ctrl b d
```
#### attach specific
```
tmux a -t train
```
#### attach last 
```
tmux a
```
#### show all
```
tmux ls
```

### run fine tune
```
tmux new -s train
source /opt/pytorch/bin/activate
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
torchrun --nproc_per_node=1 train.py --config configs/csl-daily_s2g.yaml \
  --world_size 1 --batch-size 8 --epochs 100 --seed 0 2>&1 | tee baseline_rtm.log
```

### regularly clone checkpoints to s3
```
tmux new -s save
while true; do
  aws s3 sync outputs/CSL-Daily_SLR/ s3://mska-ap-northeast-2-xe25973/frozen_default/ --exclude "*.log"
  aws s3 cp ft_freeze10_pre020.log s3://mska-ap-northeast-2-xe25973/frozen_default/
  sleep 900
done
```

### continue fine tune
```
aws s3 sync s3://mska-ap-northeast-2-xe25973/checkpoints outputs 

tmux new -s train
source /opt/pytorch/bin/activate
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
torchrun --nproc_per_node=1 train.py --config configs/csl-daily_s2g.yaml \
  --world_size 1 --batch-size 8 --epochs 100 --seed 0 \
  --resume outputs/CSL-Daily_SLR/checkpoint.pth 2>&1 | tee -a baseline_rtm.log
```

### line_probe 
```
torchrun --nproc_per_node=1 train.py --config configs/csl-daily_s2g.yaml \
  --world_size 1 --batch-size 8 --epochs 100 --seed 0 \
  --finetune pretrain_out/enc_0.20.pth \
  --ft-recipe low-enc-lr --enc-lr 1e-4 \
  --resume outputs/CSL-Daily_SLR/checkpoint.pth \
  2>&1 | tee ft_jit_arm2.log
```


###  fine tune after pretrain a
```
torchrun --nproc_per_node=1 train.py --config configs/csl-daily_s2g.yaml \
  --world_size 1 --batch-size 8 --epochs 100 --seed 0 \
  --finetune pretrain_out/enc_0.20.pth \
  --ft-recipe normal \
  2>&1 | tee ft_arm3_pre020.log
```


###  fine tune after pretrain b
```
torchrun --nproc_per_node=1 train.py --config configs/csl-daily_s2g.yaml \
  --world_size 1 --batch-size 8 --epochs 100 --seed 0 \
  --finetune pretrain_out/enc_0.20.pth \
  --ft-recipe low-enc-lr --enc-lr 1e-4 \
  2>&1 | tee ft_jit_arm2.log
```

### continue fine tune b 
```
torchrun --nproc_per_node=1 train.py --config configs/csl-daily_s2g.yaml \
  --world_size 1 --batch-size 8 --epochs 100 --seed 0 \
  --finetune pretrain_out/enc_0.20.pth \
  --ft-recipe low-enc-lr --enc-lr 1e-4 \
  --resume outputs/CSL-Daily_SLR/checkpoint.pth \
  2>&1 | tee ft_jit_arm2.log
```


while true; do
  aws s3 sync robust_logs/ s3://mska-ap-northeast-2-xe25973/robust_logs/
  sleep 900
done