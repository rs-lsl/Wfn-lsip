# Hd-lsip
The code of paper ‘High-Dimensional Latent Space Iterative Prediction: A Robust Framework for Enhanced Weather Forecasting Accuracy’.

To run the code, the following steps need to take:

1. The package need to install is in 'requirements.txt' file. The packages with the similar version are also OK.

2. Run the code using the command:

  `torchrun --nproc_per_node=1 --master_port 55562 main_latent.py --epoch 65 --ex_name 'baseline' --batch_size 16 --val_batch_size 32 --lr 1e-4 --test 0 --drop 0.2 --clip_grad 5`
