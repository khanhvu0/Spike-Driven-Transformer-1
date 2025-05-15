Instructions for training with cifar10-dvs

conda env create -f environment.yml
conda activate sdt

Create data/cifar10-dvs directory
Change data paths in prepare-cifar10dvs.py and run it

For dvs training with time step reduction:
 CUDA_VISIBLE_DEVICES=0 python -m torch.distributed.launch --nproc_per_node=1 --master_port 29501 train.py -c conf/cifar10-dvs/2_256_200E_t16_TET.yml --model sdt --spike-mode lif --early-exit --exit-threshold 0.9 --exit-metric confidence