#!/bin/bash

for arg in "$@"; do
    # if [[ $arg -eq 1 ]]; then
    #     echo "Input is equal to 1. Running s3_s2_comb_nonpre"
    #     bash run_train2.sh train_s3.py s3_s2_comb_nonpre --train_rev_v2c --num_epochs 300
    
    # elif [[ $arg -eq 2 ]]; then
    #     echo "Input is equal to 2. Running s3_s2_comb_nonpre_s02"
    #     bash run_train2.sh train_s3.py s3_s2_comb_nonpre_s02 --train_rev_v2c --num_epochs 300 --subj_id 02
    
    # elif [[ $arg -eq 3 ]]; then
    #     echo "Input is equal to 3. Running s3_s2_comb_nonpre_s03"
    #     bash run_train2.sh train_s3.py s3_s2_comb_nonpre_s03 --train_rev_v2c --num_epochs 300 --subj_id 03
    
    # elif [[ $arg -eq 4 ]]; then
    #     echo "Input is equal to 4. Running s3_s2_comb_nonpre_s04"
    #     bash run_train2.sh train_s3.py s3_s2_comb_nonpre_s04 --train_rev_v2c --num_epochs 300 --subj_id 04

    # elif [[ $arg -eq 5 ]]; then
    #     echo "Input is equal to 5. Running e2e"
    #     bash run_train2.sh train_prior.py prior_e2e --train_rev_v2c --num_epochs 300 --subj_id 01
    
    # if [[ $arg -eq 1 ]]; then
    #     echo "Input is equal to 1. Running s3_pre_noise_ln"
    #     bash run_train2.sh train_s3.py s3_pre_noise_ln --train_rev_v2c --num_epochs 300 --pre_noise_norm ln
    # elif [[ $arg -eq 2 ]]; then
    #     echo "Input is equal to 2. Running s3_pre_noise_bn"
    #     bash run_train2.sh train_s3.py s3_pre_noise_bn --train_rev_v2c --num_epochs 300 --pre_noise_norm bn
    # elif [[ $arg -eq 3 ]]; then
    #     echo "Input is equal to 3. Running s3_pre_noise_bn_1000ts"
    #     bash run_train2.sh train_s3.py s3_pre_noise_bn_1000ts --train_rev_v2c --num_epochs 300 --pre_noise_norm bn
    # elif [[ $arg -eq 4 ]]; then
    #     echo "Input is equal to 4. Running s3_pre_noise_ln_1000ts"
    #     bash run_train2.sh train_s3.py s3_pre_noise_ln_1000ts --train_rev_v2c --num_epochs 300 --pre_noise_norm ln

    # if [[ $arg -eq 1 ]]; then
    #     echo "Input is equal to 1. Running prior_fwd_bn_hialph"
    #     bash run_train2.sh train_forward_prior.py prior_fwd_bn_hialph --bidir_mixco --wandb_log

    # if [[ $arg -eq 1 ]]; then
    #     echo "Input is equal to 1. Running prior_bwd_normfwd"
    #     bash run_train2.sh train_s3.py prior_bwd_normfwd --train_rev_v2c --num_epochs 300
    
    if [[ $arg -eq 1 ]]; then
        echo "Input is equal to 1. Running mlp_fpn"
        bash run_train2.sh train_mlp.py mlp_fpn --num_epochs 300
    elif [[ $arg -eq 2 ]]; then
        echo "Input is equal to 2. Running mlp_fpn_hidrop"
        bash run_train2.sh train_mlp.py mlp_fpn_hidrop --num_epochs 300
    elif [[ $arg -eq 3 ]]; then
        echo "Input is equal to 3. Running mlp_fpn_mixer"
        bash run_train2.sh train_mlp.py mlp_fpn_mixer --num_epochs 300 --use_token_mixer
    elif [[ $arg -eq 4 ]]; then
        echo "Input is equal to 4. Running mlp_fpn_cls_corrloss"
        bash run_train2.sh train_mlp.py mlp_fpn_cls_corrloss --num_epochs 300 --only_cls --n_blocks 0
    elif [[ $arg -eq 5 ]]; then
        echo "Input is equal to 5. Running prior_bwd_normfwd_corrloss"
        bash run_train2.sh train_s3.py prior_bwd_normfwd_corrloss --train_rev_v2c --num_epochs 300

    elif [[ $arg -eq 6 ]]; then
        echo "Input is equal to 6. Running mlp_fpn_cls_corrloss_voxelbatch"
        bash run_train2.sh train_mlp.py mlp_fpn_cls_corrloss_voxelbatch --num_epochs 300 --only_cls --n_blocks 0 --voxel_batch 0

    elif [[ $arg -eq 7 ]]; then
        echo "Input is equal to 7. Running eeg_prior"
        bash run_train2.sh train_forward_prior_eeg.py eeg_prior_conv_nomix --bidir_mixco --wandb_log --mixup_pct 0
    elif [[ $arg -eq 8 ]]; then
        echo "Input is equal to 8. Running eeg_prior"
        bash run_train2.sh train_forward_prior_eeg.py eeg_prior_conv_nomix_sd2 --bidir_mixco --wandb_log --mixup_pct 0 --no_versatile

     elif [[ $arg -eq 9 ]]; then
        echo "Input is equal to 9. Running eeg_prior_mbd"
        bash run_train2.sh train_forward_prior_eeg.py eeg_prior_mbd_conv_nomix_sd --bidir_mixco --wandb_log --mixup_pct 0 --no_versatile --subj_id=mbd --num_epochs 300
    elif [[ $arg -eq 10 ]]; then
        echo "Input is equal to 10. Running eeg_prior_mbd_oldnet"
        bash run_train2.sh train_forward_prior_eeg.py eeg_prior_mbd_oldnet --bidir_mixco --wandb_log --mixup_pct 0 --no_versatile --subj_id=mbd --num_epochs 300
    
    else
        echo "Invalid input. Ignoring argument $arg."
    fi
done
