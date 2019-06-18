#!/bin/sh

CUDA_VISIBLE_DEVICES=0 python train.py --id intraatt2in2_rl --caption_model intraatt2in2 --input_json data/new_cocotalk.json --input_fc_dir data/cocobu_fc --input_att_dir data/cocobu_att --input_label_h5 data/new_cocotalk_label.h5 --batch_size 40  --start_from log_intraatt2in2_rl --learning_rate 5e-5 --checkpoint_path log_intraatt2in2_rl --save_checkpoint_every 6000 --language_eval 1 --val_images_use 5000 --self_critical_after 30 --att_type intra 
