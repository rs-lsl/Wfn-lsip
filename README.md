# Hd-lsip
The code of paper ‘High-Dimensional Latent Space Iterative Prediction: A Robust Framework for Enhanced Weather Forecasting Accuracy’.

To run the code, the following steps need to take:

1. The package need to install is in 'requirements.txt' file. The packages with the similar version are also OK.

2. ERA5 dataset at each time step should be stored as individual files and read by filename in the dataloader function (you need to download the ERA5 dataset in advance).

3. Adjust all 'paths' and run the code using the command:

  training mode: `torchrun --nproc_per_node=2 --master_port 55562 main_latent.py --epoch 65 --ex_name 'baseline' --batch_size 16 --val_batch_size 32 --lr 1e-4 --test 0

  testing mode: `torchrun --nproc_per_node=1 --master_port 55562 main_latent.py --epoch 65 --ex_name 'baseline' --batch_size 16 --val_batch_size 32 --lr 1e-4 --test 1
