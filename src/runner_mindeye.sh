#!/bin/bash

for arg in "$@"; do
    # if [[ $arg -eq 1 ]]; then
    #     echo "Input is equal to 1. Running bimixco."
    #     bash run_train2.sh train_voxel2clip_257.py voxel2clip_257_final_subj01_oldaugs_bimixco_300 --num_epochs=300 --mixup_pct=1 --bidir_mixco 
    # elif [[ $arg -eq 2 ]]; then
    #     echo "Input is equal to 2. Running bince."
    #     bash run_train2.sh train_voxel2clip_257.py voxel2clip_257_final_subj01_oldaugs_bince --mixup_pct=1 --mixco_sel_thresh=0 --bidir_mixco 
    # elif [[ $arg -eq 3 ]]; then
    #     echo "Input is equal to 3. Running unince."
    #     bash run_train2.sh train_voxel2clip_257.py voxel2clip_257_final_subj01_oldaugs_unince --mixup_pct=1 --mixco_sel_thresh=0
    # elif [[ $arg -eq 4 ]]; then
    #     echo "Input is equal to 4. Running autoencoder."
    #     bash run_train2.sh train_autoencoder.py autoencoder_final_cont_noreconst
    # else
    #     echo "Invalid input. Ignoring argument $arg."
    # fi
    
    # if [[ $arg -eq 1 ]]; then
    #     echo "Input is equal to 1. Running bimixco_byol."
    #     bash run_train2.sh train_prior_257.py prior_257_final_subj01_bimixco_byol --mixup_pct=1 --bidir_mixco --v2c_projector --wandb_log
    # elif [[ $arg -eq 2 ]]; then
    #     echo "Input is equal to 2. Running bimixco_softclip_byol."
    #     bash run_train2.sh train_prior_257.py prior_257_final_subj01_bimixco_softclip_byol --mixup_pct=0.33 --bidir_mixco --v2c_projector --wandb_log
    
    # if [[ $arg -eq 1 ]]; then
    #     echo "Input is equal to 1. Running softcont_cont_flatten."
    #     bash run_train2.sh train_prior_257.py prior_257_final_subj01_bimixco_softcont_byol --mixup_pct=0.33 --bidir_mixco --v2c_projector --wandb_log --soft_loss_type=cont_flatten --ckpt_path=../train_logs/models/prior_257_final_subj01_bimixco_softclip_byol/epoch074.pth
    # elif [[ $arg -eq 2 ]]; then
    #     echo "Input is equal to 2. Running softcont_cont_inter."
    #     bash run_train2.sh train_prior_257.py prior_257_final_subj01_bimixco_softcont_inter_byol --mixup_pct=0.33 --bidir_mixco --v2c_projector --wandb_log --soft_loss_type=cont_inter --ckpt_path=../train_logs/models/prior_257_final_subj01_bimixco_softclip_byol/epoch074.pth
    
    # if [[ $arg -eq 1 ]]; then
    #     echo "Input is equal to 1. Running subj02."
    #     bash run_train2.sh train_prior_257.py prior_257_final_subj02_bimixco_softclip_byol --mixup_pct=0.33 --bidir_mixco --v2c_projector --subj_id 02
    # elif [[ $arg -eq 2 ]]; then
    #     echo "Input is equal to 2. Running subj05."
    #     bash run_train2.sh train_prior_257.py prior_257_final_subj05_bimixco_softclip_byol --mixup_pct=0.33 --bidir_mixco --v2c_projector --subj_id 05
    # elif [[ $arg -eq 3 ]]; then
    #     echo "Input is equal to 3. Running subj07."
    #     bash run_train2.sh train_prior_257.py prior_257_final_subj07_bimixco_softclip_byol --mixup_pct=0.33 --bidir_mixco --v2c_projector --subj_id 07

    
    # if [[ $arg -eq 1 ]]; then
    #     echo "Input is equal to 1. Running autoenc_4x."
    #     bash run_train.sh train_autoencoder.py autoencoder_subj01_4x 1 --ups_mode=4x
    # elif [[ $arg -eq 2 ]]; then
    #     echo "Input is equal to 2. Running autoenc_8x."
    #     bash run_train.sh train_autoencoder.py autoencoder_subj01_8x 1 --ups_mode=8x
    # elif [[ $arg -eq 3 ]]; then
    #     echo "Input is equal to 3. Running autoenc_16x."
    #     bash run_train.sh train_autoencoder.py autoencoder_subj01_16x 1 --ups_mode=16x
    # elif [[ $arg -eq 4 ]]; then
    #     echo "Input is equal to 4. Running autoenc_4x_cont."
    #     bash run_train.sh train_autoencoder.py autoencoder_subj01_4x_cont 8 --ups_mode=4x --use_reconst=True --batch_size=8 --use_cont=True
    # elif [[ $arg -eq 5 ]]; then
    #     echo "Input is equal to 5. Running autoenc_4x_sobel."
    #     bash run_train.sh train_autoencoder.py autoencoder_subj01_4x_sobel 8 --ups_mode=4x --use_reconst=True --batch_size=8 --use_sobel_loss=True
    # elif [[ $arg -eq 6 ]]; then
    #     echo "Input is equal to 6. Running autoenc_4x_locont_no_reconst."
    #     bash run_train.sh train_autoencoder.py autoencoder_subj01_4x_locont_no_reconst 2 --ups_mode=4x --use_cont=True
    # elif [[ $arg -eq 7 ]]; then
    #     echo "Input is equal to 7. Running autoenc_4x_sobel_blur."
    #     bash run_train.sh train_autoencoder.py autoencoder_subj01_4x_sobel_blur 8 --ups_mode=4x --use_reconst=True --batch_size=8 --use_sobel_loss=True --use_blurred_training=True
    
    # if [[ $arg -eq 1 ]]; then
    #     echo "Input is equal to 1. Running autoenc_subj02."
    #     bash run_train.sh train_autoencoder.py autoencoder_subj02_4x_locont_no_reconst 8 --ups_mode=4x --use_cont=True --batch_size=8 --subj_id=02
    # elif [[ $arg -eq 2 ]]; then
    #     echo "Input is equal to 2. Running autoenc_subj05."
    #     bash run_train.sh train_autoencoder.py autoencoder_subj05_4x_locont_no_reconst 8 --ups_mode=4x --use_cont=True --batch_size=8 --subj_id=05
    # elif [[ $arg -eq 3 ]]; then
    #     echo "Input is equal to 3. Running autoenc_subj07."
    #     bash run_train.sh train_autoencoder.py autoencoder_subj07_4x_locont_no_reconst 8 --ups_mode=4x --use_cont=True --batch_size=8 --subj_id=07
    
    # if [[ $arg -eq 1 ]]; then
    #     echo "Input is equal to 1. Running prior_subj01."
    #     bash run_train2.sh train_prior_257.py prior_1x768_final_subj01_bimixco_softclip_byol --mixup_pct=0.33 --bidir_mixco --v2c_projector --wandb_log --num_epochs=300 --no_versatile
    # elif [[ $arg -eq 2 ]]; then
    #     echo "Input is equal to 2. Running prior_subj02."
    #     bash run_train2.sh train_prior_257.py prior_1x768_final_subj02_bimixco_softclip_byol --mixup_pct=0.33 --bidir_mixco --v2c_projector --wandb_log --num_epochs=300 --no_versatile --subj_id=02
    # elif [[ $arg -eq 3 ]]; then
    #     echo "Input is equal to 3. Running prior_subj05."
    #     bash run_train2.sh train_prior_257.py prior_1x768_final_subj05_bimixco_softclip_byol --mixup_pct=0.33 --bidir_mixco --v2c_projector --wandb_log --num_epochs=300 --no_versatile --subj_id=05
    # elif [[ $arg -eq 4 ]]; then
    #     echo "Input is equal to 4. Running prior_subj07."
    #     bash run_train2.sh train_prior_257.py prior_1x768_final_subj07_bimixco_softclip_byol --mixup_pct=0.33 --bidir_mixco --v2c_projector --wandb_log --num_epochs=300 --no_versatile --subj_id=07
    # elif [[ $arg -eq 5 ]]; then
    #     echo "Input is equal to 5. Running bimixco_softclip_byol."
    #     bash run_train2.sh train_prior_257.py prior_257_subj01_bimixco_softclip_byol_1000st --mixup_pct=0.33 --bidir_mixco --v2c_projector --wandb_log --num_epochs=240
    # elif [[ $arg -eq 6 ]]; then
    #     echo "Input is equal to 6. Running bimixco_softclip_byol_causal."
    #     bash run_train2.sh train_prior_257.py prior_257_subj01_bimixco_softclip_byol_causal --mixup_pct=0.33 --bidir_mixco --v2c_projector --wandb_log --num_epochs=240 --causal

    # if [[ $arg -eq 1 ]]; then
    #     echo "Input is equal to 1. Running n0"
    #     bash run_train2.sh train_voxel2clip_257.py voxel2clip_257_subj01_n0  --model_type n0
    # elif [[ $arg -eq 2 ]]; then
    #     echo "Input is equal to 2. Running n2"
    #     bash run_train2.sh train_voxel2clip_257.py voxel2clip_257_subj01_n2  --model_type n2
    # elif [[ $arg -eq 3 ]]; then
    #     echo "Input is equal to 3. Running n2_res"
    #     bash run_train2.sh train_voxel2clip_257.py voxel2clip_257_subj01_n2res  --model_type n2_res
    # elif [[ $arg -eq 4 ]]; then
    #     echo "Input is equal to 4. Running n4"
    #     bash run_train2.sh train_voxel2clip_257.py voxel2clip_257_subj01_n4  --model_type n4
    # elif [[ $arg -eq 5 ]]; then
    #     echo "Input is equal to 5. Running n4_res"
    #     bash run_train2.sh train_voxel2clip_257.py voxel2clip_257_subj01_n4res  --model_type n4_res
    
    # elif [[ $arg -eq 6 ]]; then
    #     echo "Input is equal to 6. Running n4_res_1x768"
    #     bash run_train2.sh train_voxel2clip_257.py voxel2clip_1x768_subj01_n4res  --model_type n4_res

    if [[ $arg -eq 1 ]]; then
        echo "Input is equal to 1. Running prior_nodetr_noncausal_posemb_240_cont"
        bash run_train2.sh train_prior_257.py prior_nodetr_noncausal_posemb_240_cont_2 --wandb_log
    elif [[ $arg -eq 2 ]]; then
        echo "Input is equal to 2. Running voxel2clip_allflat_nodetr_normedmse_byol_240"
        bash run_train2.sh train_voxel2clip_257.py voxel2clip_allflat_nodetr_normedmse_byol_240_2 --num_epochs 240 --use_mse --v2c_projector --wandb_log
    elif [[ $arg -eq 3 ]]; then
        echo "Input is equal to 3. Running voxel2clip_allflat_nodetr_normedmse_240_vdpipe"
        bash run_train2.sh train_voxel2clip_257.py vvoxel2clip_allflat_nodetr_normedmse_240_vdpipe_2  --num_epochs 240 --use_mse --wandb_log

    else
        echo "Invalid input. Ignoring argument $arg."
    fi
done