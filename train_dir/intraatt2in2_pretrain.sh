#!/bin/sh

CUDA_VISIBLE_DEVICES=0 python train.py --id intraatt2in2 --caption_model intraatt2in2 --input_json data/new_cocotalk.json --input_fc_dir data/cocobu_fc --input_att_dir data/cocobu_att --input_label_h5 data/new_cocotalk_label.h5 --batch_size 45  --learning_rate 5e-4 --learning_rate_decay_start 0 --scheduled_sampling_start 0 --language_eval 1 --checkpoint_path log_intraatt2in2 --save_checkpoint_every 6000 --val_images_use 5000 --max_epochs 30 --att_type intra --use_crf True 
