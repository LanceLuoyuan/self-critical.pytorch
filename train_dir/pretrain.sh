#!/bin/sh

python train.py --id att2in --caption_model att2in --input_json data/cocotalk.json --input_fc_dir data/cocotalk_fc --input_att_dir data/cocotalk_att --input_label_h5 data/cocotalk_label.h5 --batch_size 10 --learning_rate 5e-4 --learning_rate_decay_start 0 --scheduled_sampling_start 0 --language_eval 1 --checkpoint_path log_att2in --save_checkpoint_every 6000 --val_images_use 5000 --max_epochs 30 --start_from log_att2in
