#!/bin/bash

###task-train
echo ../tr_output/udp.err
sbatch -o ../tr_output/udp.out -e ../tr_output/udp.err h_train_udp.slurm udp
echo ../tr_output/pos.err
sbatch -o ../tr_output/pos.out -e ../tr_output/pos.err h_train_udp.slurm pos
echo ../tr_output/nli.err
sbatch -o ../tr_output/nli.out -e ../tr_output/nli.err h_train_udp.slurm nli