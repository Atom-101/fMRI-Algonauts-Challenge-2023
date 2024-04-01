# # Import packages & functions

import os
import shutil
import sys
import traceback
import time
import json
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from tqdm import tqdm
import kornia
from kornia.augmentation.container import AugmentationSequential

import utils
from utils import torch_to_matplotlib, torch_to_Image
from models import HiddenClipper, OpenClipper, BrainNetworkNoDETR, ReversibleBrainNetwork, VersatileDiffusionPriorNetwork, BrainDiffusionPrior, BrainNetworkFPN

import torch.distributed as dist
from accelerate import Accelerator

import argparse

import math
import random
import webdataset as wds
from torchmetrics import PearsonCorrCoef


def parse_args():
    parser = argparse.ArgumentParser(description="Train prior")
    parser.add_argument(
        "--model_name",
        type=str,
        default="prior_257_test",
        help="name of model, used for wandb logging",
    )
    parser.add_argument(
        "--clip_variant",
        type=str,
        default="ViT-L/14",
        choices=["RN50", "ViT-L/14", "ViT-B/32"],
        help='clip variant',
    )
    parser.add_argument(
        "--wandb_log",
        action="store_true",
        help="whether to log to wandb",
    )
    parser.add_argument(
        "--wandb_project",
        type=str,
        default="stability",
        help="wandb project name",
    )
    parser.add_argument(
        "--remote_data",
        action="store_true",
        help="whether to pull data from huggingface",
    )
    parser.add_argument(
        "--wds_cache_dir",
        type=str,
        default='/tmp/wds-cache',
        help="directory for caching webdatasets fetched from huggingface",
    )
    parser.add_argument(
        "--disable_image_aug",
        action="store_true",
        help="whether to disable image augmentation (only used for modality=image)",
    )
    parser.add_argument(
        "--outdir",
        type=str,
        default=None,
        help="output location",
    )
    parser.add_argument(
        "--learned_query_mode",
        type=str,
        default="pos_emb",
        choices=["none", "token", "pos_emb", "all_pos_emb"],
        help="output location",
    )
    parser.add_argument(
        "--cont_loss_type",
        type=str,
        default="flatten",
        choices=["all", "flatten"],
        help="loss type",
    )
    parser.add_argument(
        "--ckpt_path",
        type=str,
        default=None
    )
    parser.add_argument(
        "--mixup_pct",
        type=float,
        default=0.5
    )
    parser.add_argument(
        "--num_epochs",
        type=int,
        default=240
    )
    parser.add_argument(
        "--mixco_sel_thresh",
        type=float,
        default=0.5
    )
    parser.add_argument(
        "--subj_id",
        choices=["01", "02", "03", "04", "05", "06", "07", "08"],
        default="01"
    )
    parser.add_argument(
        "--train_rev_v2c",
        action="store_true",
    )
    parser.add_argument(
        "--use_token_mixer",
        action="store_true",
    )
    parser.add_argument(
        "--only_cls",
        action="store_true",
    )
    parser.add_argument(
        "--n_blocks",
        type=int,
        default=4
    )
    parser.add_argument(
        "--voxel_batch",
        type=int,
        default=None
    )
    return parser.parse_args()

if __name__ == '__main__':
    # Multi-GPU config #
    accelerator = Accelerator()
    print = accelerator.print # only print if local_rank=0

    device = accelerator.device
    print("device:",device)

    args = parse_args()
    print('args', args)

    model_name = args.model_name
    clip_variant = args.clip_variant  # "convnext_xxlarge"  # "ViT-L/14" # ("RN50", "ViT-L/14", "ViT-B/32")
    weights_path = None
    
    # params for all models
    seed = 0
    batch_size = 64
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    num_epochs = args.num_epochs
    lr_scheduler = 'cycle'
    initial_lr = 1e-3 #3e-5
    max_lr = 3e-3
    
    wandb_log = args.wandb_log
    wandb_project = 'laion-fmri'
    wandb_run_name = ''
    wandb_notes = ''
    
    ckpt_saving = True
    use_mp = False
    distributed = False
    save_at_end = False
    subj_id = args.subj_id

    cache_dir = 'cache'
    mixup_pct = args.mixup_pct

    resume_from_ckpt = args.ckpt_path is not None
    ckpt_path = args.ckpt_path

    if args.outdir is None:
        # outdir = os.path.expanduser(f'../train_logs/models/{model_name}/test')
        outdir = f'../train_logs/models/{args.model_name}'
    else:
        outdir = args.outdir
    os.makedirs(outdir, exist_ok=True)
    
    # uses tf32 data type which is faster than standard float32
    torch.backends.cuda.matmul.allow_tf32 = True
    utils.seed_everything(seed, cudnn_deterministic=False)
    
    num_devices = torch.cuda.device_count()
    if num_devices==0: num_devices = 1
    num_workers = 1

    # auto resume
    if os.path.exists(os.path.join(outdir, 'last.pth')) or os.path.exists(os.path.join(outdir, 'last_old.pth')):
        if os.path.exists(os.path.join(outdir, 'last_old.pth')):
            if os.path.exists(os.path.join(outdir, 'last.pth')):
                # this is corrupted
                os.remove(os.path.join(outdir, f'last.pth'))
            # set last_old as last
            shutil.move(os.path.join(outdir, f'last_old.pth'), os.path.join(outdir, f'last.pth'))
        
        ckpt_path = os.path.join(outdir, 'last.pth')
        resume_from_ckpt = True

    if not args.disable_image_aug:
        train_augs = AugmentationSequential(
            kornia.augmentation.RandomResizedCrop((240,240), (0.6,1), p=0.3),
            kornia.augmentation.Resize((224, 224)),
            kornia.augmentation.RandomGaussianBlur(kernel_size=(7,7), sigma=(5,5), p=0.3), #MedianBlur is better but computationally inefficient
            # kornia.augmentation.RandomHorizontalFlip(p=0.5),
            kornia.augmentation.ColorJiggle(brightness=0.2, contrast=0.2, saturation=0.3, hue=0., p=0.3),
        )
    else:
        train_augs = None

    clip_extractor = HiddenClipper(clip_variant, device=device, train_transforms=train_augs)

    print(accelerator.state)
    local_rank = accelerator.state.local_process_index
    world_size = accelerator.state.num_processes
    if num_devices <= 1 and world_size <= 1:
        distributed = False
    else:
        distributed = True

    print('Pulling NSD webdataset data...')
    # local paths
    if args.subj_id in [3,6]:
        max_tar = 90
    elif args.subj_id in [4,8]:
        max_tar = 87
    else:
        max_tar = 98
    train_url = f"/fsx/proj-fmri/shared/algonauts_wds2/subj{args.subj_id}_{{3..{max_tar}}}.tar"
    val_url = f"/fsx/proj-fmri/shared/algonauts_wds2/subj{args.subj_id}_{{0..2}}.tar"
    meta_url = f"/fsx/proj-fmri/shared/algonauts_wds2/metadata_subj{args.subj_id}.json"

    train_dl, val_dl, num_train, num_val = utils.get_dataloaders_wds2(
        batch_size,
        num_devices=num_devices,
        num_workers=num_workers,
        train_url=train_url,
        val_url=val_url,
        meta_url=meta_url,
        val_batch_size=150,
        cache_dir=args.wds_cache_dir,
        seed=seed,
        local_rank=local_rank
    )

    print('Creating voxel2clip...')
    # size of the CLIP embedding for each variant
    clip_sizes = {"RN50": 1024, "ViT-L/14": 1024, "ViT-B/32": 512}
    clip_depths = {"RN50": 5, "ViT-L/14": 24, "ViT-B/32": 12}
    # output dim for voxel2clip model
    out_dim, clip_depth  = clip_sizes[clip_variant], clip_depths[clip_variant]

    clip2voxel_kwargs = dict(out_dim=out_dim, clip_depth=clip_depth, norm_type='ln', act_first=False, 
                             encoder_tokens=257 if not args.only_cls else 1, token_mixer=args.use_token_mixer,
                             n_blocks=args.n_blocks, h=2304 if not args.only_cls else 8192)
    in_dims = {'01': 39548, '02': 39548, '03': 39548, '04': 39548, '05': 39548, '06': 39198, '07': 39548, '08': 39511}
    clip2voxel_kwargs["in_dim"] = in_dims[subj_id]
    if args.voxel_batch is not None:
        clip2voxel_kwargs["in_dim"] = 3072
    clip2voxel = BrainNetworkFPN(**clip2voxel_kwargs)
    clip2voxel.to(device)
    
    torch.cuda.empty_cache()
    
    no_decay = ['bias', 'LayerNorm.bias', 'LayerNorm.weight']
    opt_grouped_parameters = [
        {'params': [p for n, p in clip2voxel.named_parameters() if not any(nd in n for nd in no_decay)], 'weight_decay': 1e-2},
        {'params': [p for n, p in clip2voxel.named_parameters() if any(nd in n for nd in no_decay)], 'weight_decay': 0.0}
    ]
    optimizer = torch.optim.AdamW(opt_grouped_parameters, lr=initial_lr) # lr doesnt get used if lr_scheduler='cycle'

    if lr_scheduler == 'fixed':
        lr_scheduler = None
    elif lr_scheduler == 'cycle':
        global_batch_size = batch_size * num_devices
        total_steps = num_epochs*(num_train//global_batch_size)
        lr_scheduler = torch.optim.lr_scheduler.OneCycleLR(
            optimizer, 
            max_lr=max_lr,
            total_steps=total_steps,
            final_div_factor=1000,
            last_epoch=-1, pct_start=2/num_epochs
        )

    def save_ckpt(tag):
        if tag == "last" and os.path.exists(os.path.join(outdir, f'{tag}.pth')):
            shutil.copyfile(os.path.join(outdir, f'{tag}.pth'), os.path.join(outdir, f'{tag}_old.pth'))
            # shutil.move(os.path.join(outdir, f'{tag}.pth'), os.path.join(outdir, f'{tag}_old.pth'))
        
        ckpt_path = os.path.join(outdir, f'{tag}.pth')
        print(f'saving {ckpt_path}',flush=True)
        if tag == "last":
            torch.save({
                'epoch': epoch,
                'model_state_dict': clip2voxel.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'train_losses': losses,
                'val_losses': val_losses,
                "val/val_corr": val_corr,
                'lrs': lrs,
                'best_val_corr': best_val_corr
                }, ckpt_path)
        else:
            torch.save({
                'epoch': epoch,
                'model_state_dict': clip2voxel.state_dict(),
                "val/val_corr": val_corr,
                'best_val_corr': best_val_corr
                }, ckpt_path)
        
        if tag == "last" and os.path.exists(os.path.join(outdir, f'{tag}_old.pth')):
            os.remove(os.path.join(outdir, f'{tag}_old.pth'))

    print("\nDone with model preparations!")
    
    #--------WANDB-----------------
    if local_rank==0 and args.wandb_log:
        wandb_run = args.model_name
        wandb_notes = ''

        import wandb
        print(f"wandb {args.wandb_project} run {wandb_run}")
        wandb.login(host='https://stability.wandb.io')#, relogin=True)
        wandb_config = {
            "model_name": args.model_name,
            "modality": args.modality,
            "voxel_dims": args.voxel_dims,
            "clip_variant": args.clip_variant,
            "batch_size": batch_size,
            "num_epochs": num_epochs,
            "disable_image_aug": args.disable_image_aug,
            "max_lr": max_lr,
            "lr_scheduler": lr_scheduler,
            # "clamp_embs": clamp_embs,
            "mixup_pct": mixup_pct,
            "num_train": num_train,
            "num_val": num_val,
            "seed": seed,
            "distributed": distributed,
            "num_devices": num_devices,
            "world_size": world_size,
            # "resume_from_ckpt": resume_from_ckpt,
            # "ckpt_path": ckpt_path,
            "train_url": train_url,
            "val_url": val_url,
        }
        print("wandb_config:\n",wandb_config)
        wandb.init(
            id = model_name,
            project=args.wandb_project,
            name=wandb_run,
            config=wandb_config,
            notes=wandb_notes,
            resume="allow"
        )
            
    # #----ACCELERATE------------
    # rev_diffusion_prior, rev_v2c, optimizer, train_dl, val_dl, lr_scheduler = accelerator.prepare(
    #     rev_diffusion_prior, rev_v2c, optimizer, train_dl, val_dl, lr_scheduler
    # )

    epoch = 0
    losses, mse_losses, val_losses, lrs = [], [], [], []
    best_val_corr = 0
    soft_loss_temps = utils.cosine_anneal(0.002, 0.008, num_epochs)

    voxel0 = image0 = val_voxel0 = val_image0 = None

    # Optionally resume from checkpoint #
    if resume_from_ckpt:
        print("\n---resuming from ckpt_path---\n", ckpt_path)
        checkpoint = torch.load(ckpt_path, map_location=device)
        epoch = checkpoint['epoch']+1
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])        
        clip2voxel.load_state_dict(checkpoint['model_state_dict'])
        if 'best_val_corr' in checkpoint:
            best_val_corr = checkpoint['best_val_corr']
        global_batch_size = batch_size * num_devices
        total_steps_done = epoch*(num_train//global_batch_size)
        for _ in range(total_steps_done):
            lr_scheduler.step()
        del checkpoint
        torch.cuda.empty_cache()

    progress_bar = tqdm(range(epoch,num_epochs), disable=(local_rank!=0))
    pearson = PearsonCorrCoef(in_dims[subj_id] if args.voxel_batch is None else 3072).to(device)
    dino_loss = utils.DINOLoss(in_dims[subj_id], student_temp=0.01).to(device)
    g_cuda = torch.Generator(device=device)

    for epoch in progress_bar:
        clip2voxel.train()

        sims = 0.
        sims_base = 0.
        val_sims = 0.
        val_sims_base = 0.
        fwd_percent_correct = 0.
        bwd_percent_correct = 0.
        val_corr = 0.
        train_corr = 0.
        loss_mse_sum = 0.
        loss_corr_sum = 0.
        val_loss_mse_sum = 0.

        for train_i, (voxel, image) in enumerate(train_dl):
            optimizer.zero_grad()

            voxel = voxel.float().to(device).mean(1)
            # voxel = utils.voxel_select(voxel)
            image = image.float()

            if epoch < int(mixup_pct * num_epochs):
                image, perm, betas, select = utils.mixco(image, beta=0.3, s_thresh=0.8)
            else:
                image, perm, betas, select = utils.mixco(image, beta=0.15, s_thresh=0.5)
            betas_shape = [-1] + [1]*(len(voxel.shape)-1)
            voxel = voxel * betas.reshape(*betas_shape).to(device) + voxel[perm] * (1-betas.reshape(*betas_shape)).to(device)
            if args.voxel_batch is not None:
                voxel = voxel[:,(args.voxel_batch)*3072:(args.voxel_batch+1)*3072]
            
            clip_latent = clip_extractor.embed_image(image, apply_transforms=True).float()
            clip_latent.to(voxel.dtype)  # b, 24, 257, 768
            if args.only_cls:
                clip_latent = clip_latent[:, :, :1]
            
            pred_voxel = clip2voxel(clip_latent)
            # recons_mse = F.mse_loss(pred_voxel, voxel)
            # recons_corr_loss = dino_loss(pred_voxel, voxel, temp=soft_loss_temps[epoch])
            recons_mse = torch.tensor(0)
            recons_corr_loss =  1 - torch.nanmean(utils.pairwise_pearson_correlation(pred_voxel+(torch.randn_like(pred_voxel)*1e-8), voxel))
            loss = 0.1 * recons_mse + recons_corr_loss

            recons_corr = pearson(pred_voxel, voxel)
            train_corr += ((recons_corr**2)*100).mean().item()

            loss_mse_sum += recons_mse.item()
            loss_corr_sum += recons_corr_loss.item()
            utils.check_loss(loss)
            losses.append(loss.item())
            lrs.append(optimizer.param_groups[0]['lr'])
            del pred_voxel, clip_latent, voxel

            accelerator.backward(loss)
            optimizer.step()

            if lr_scheduler is not None:
                lr_scheduler.step()
            
            # logs = {"train/loss": np.mean(losses[-(train_i+1):]),
            #         "train/lr": lrs[-1],
            #         "train/num_steps": len(losses),
            #         "val/val_corr": val_corr,
            #         "train/train_corr": train_corr / (train_i + 1),
            #         "train/mse_loss": loss_mse_sum / (train_i + 1),
            #         "train/recons_corr_loss": loss_corr_sum / (train_i + 1),
            #     }
            # progress_bar.set_postfix(**logs)
        
        if local_rank==0:
            clip2voxel.eval()
            val_corr = 0.
            for val_i, (voxel, image) in enumerate(val_dl): 
                with torch.inference_mode():
                    voxel = voxel.float().to(device)
                    voxel = voxel.mean(1)
                    if args.voxel_batch is not None:
                        voxel = voxel[:,(args.voxel_batch)*3072:(args.voxel_batch+1)*3072]
                    clip_latent = clip_extractor.embed_image(image, apply_transforms=True).float()
                    clip_latent.to(voxel.dtype)  # b, 24, 257, 768
                    if args.only_cls:
                        clip_latent = clip_latent[:, :, :1]

                    pred_voxel = clip2voxel(clip_latent)

                    corr_coefs = pearson(pred_voxel, voxel)  # 30k
                    val_corr += ((corr_coefs**2)*100).mean().item()
            
            logs = {"train/loss": np.mean(losses[-(train_i+1):]),
                    "train/lr": lrs[-1],
                    "train/num_steps": len(losses),
                    "val/val_corr": val_corr/(val_i+1),
                    "train/train_corr": train_corr / (train_i + 1),
                    "train/mse_loss": loss_mse_sum / (train_i + 1),
                    "train/recons_corr_loss": loss_corr_sum / (train_i + 1),
                }
            progress_bar.set_postfix(**logs)
            
            if args.wandb_log:
                while True:
                    try:
                        wandb.log(logs)
                        break
                    except:
                        print('Wandb log failed. Retrying')
                        time.sleep(1)
            
            if val_corr > best_val_corr:
                print(f'Saving new best at epoch {epoch}')
                save_ckpt('best')
                best_val_corr = float(val_corr)

        if ckpt_saving and local_rank==0:
            try:
                save_ckpt(f'last')
            except: 
                pass

    if args.wandb_log and local_rank==0:
        wandb.finish()

    print("\n===Finished!===\n")