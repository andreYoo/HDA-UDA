#/bin/bash


#Experiment for normal training models




#Experiments for domain adaptation one-class classification
python main.py arrhythmia thyroid arrhythmia_mlp thyroid_mlp ../log/da_test ../data --ratio_known_outlier 0.01 --ratio_pollution -1.0 --lr 0.001 --n_epochs 150 --lr_milestone 30 --batch_size 128 --weight_decay 0.5e-6 --pretrain True --ae_lr 0.0001 --ae_n_epochs 50 --ae_batch_size 128 --ae_weight_decay 0.5e-3 --normal_class 0 --known_outlier_class 1 --n_known_outlier_classes 1;
