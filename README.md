# Wfn-lsip
The code of paper ‘Accurate global weather forecasting with a low computational cost model’.

To run the code, the following steps need to take:

1. You need to replace the files in 'data_npy' and 'weights' with the files in https://pan.baidu.com/s/1m0gLfYBGudxVMQFH5C8B5g?pwd=2nw6 提取码(password): 2nw6

2. The package need to install is in 'requirements.txt' file. The packages with the similar version are also OK.

3. Run the code using the command:

  `torchrun --nproc_per_node=1 --master_port 55562 Wfn-lsip/main_latent.py --epoch 65 --ex_name 'baseline' --batch_size 32 --val_batch_size 32 --lr 1e-4 --test 1 --drop 0.2 --clip_grad 5`
