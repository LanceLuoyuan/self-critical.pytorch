#!/bin/bash

function repeat()
{
    while true
    do
        $@ && return
        sleep 1
    done
}

repeat nohup ./train_dir/att2in2_pretrain.sh >> log/att2in2/pre.out
