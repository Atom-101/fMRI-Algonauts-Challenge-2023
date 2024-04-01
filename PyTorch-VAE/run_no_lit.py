import os
import shutil
import yaml
import time
import argparse
import numpy as np
from pathlib import Path
from models import *
import torch.backends.cudnn as cudnn
import random
import torch
import webdataset as wds
import wandb
from tqdm import tqdm

from torchvision.utils import make_grid
from torchvision import transforms


def torch_to_Image(x):
    if x.ndim==4:
        x=x[0]
    return transforms.ToPILImage()(x)

def seed_everything(seed=0, cudnn_deterministic=False):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    if cudnn_deterministic:
        torch.backends.cudnn.deterministic = True
    else:
        ## needs to be False to use conv3D
        print('Note: not using cudnn.deterministic')

seed_everything(0, cudnn_deterministic=False)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Generic runner for VAE models')
    parser.add_argument('--config',  '-c',
                        dest="filename",
                        metavar='FILE',
                        help =  'path to the config file',
                        default='configs/vae.yaml')
    parser.add_argument(
            "--model_name",
            type=str,
            default="mnist_vae",
            help="name of model, used for wandb logging")
    parser.add_argument(
            "--wandb_log",
            action="store_true",
            help="whether to log to wandb")

    args = parser.parse_args()
    with open(args.filename, 'r') as file:
        try:
            config = yaml.safe_load(file)
        except yaml.YAMLError as exc:
            print(exc)
    resume_from_ckpt = False
    epoch = 0
    outdir = f'../train_logs/models/{args.model_name}'    
    os.makedirs(outdir, exist_ok=True)
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

    def save_ckpt(tag):
        if tag == "last" and os.path.exists(os.path.join(outdir, f'{tag}.pth')):
            shutil.copyfile(os.path.join(outdir, f'{tag}.pth'), os.path.join(outdir, f'{tag}_old.pth'))
            # shutil.move(os.path.join(outdir, f'{tag}.pth'), os.path.join(outdir, f'{tag}_old.pth'))
        
        ckpt_path = os.path.join(outdir, f'{tag}.pth')
        print(f'saving {ckpt_path}',flush=True)
        if tag=='last':
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'scheduler_state_dict': scheduler.state_dict(),
                'train_losses': losses,
                'val_losses': val_losses,
                'lrs': lrs,
                }, ckpt_path)
        else:
                torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'train_losses': losses,
                'val_losses': val_losses,
                'lrs': lrs,
                }, ckpt_path)
                
        if tag == "last" and os.path.exists(os.path.join(outdir, f'{tag}_old.pth')):
            os.remove(os.path.join(outdir, f'{tag}_old.pth'))

    model = vae_models[config['model_params']['name']](**config['model_params'])
    model.cuda()

    train_data = wds.WebDataset('/fsx/proj-fmri/shared/eeg_mnist_train.tar', resampled=False, nodesplitter=wds.split_by_node)\
                .shuffle(500, initial=500, rng=random.Random(0))\
                .decode("torch")\
                .rename(images="jpg;png")\
                .to_tuple("images")\
                .batched(1024, partial=False)\
                .with_epoch(45)
    train_dl = torch.utils.data.DataLoader(train_data, 
                            num_workers=1,
                            batch_size=None, shuffle=False, persistent_workers=True)

    val_data = wds.WebDataset('/fsx/proj-fmri/shared/eeg_mnist_test.tar', resampled=False, nodesplitter=wds.split_by_node)\
                .decode("torch")\
                .rename(images="jpg;png")\
                .to_tuple("images")\
                .batched(4096, partial=False)\
                .with_epoch(1)
    val_dl = torch.utils.data.DataLoader(val_data, num_workers=1,
                    batch_size=None, shuffle=False, persistent_workers=True)

    optimizer = torch.optim.AdamW(model.parameters(), lr=config['exp_params']['LR']) # lr doesnt get used if lr_scheduler='cycle'
    scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer, gamma = config['exp_params']['scheduler_gamma'])

    #--------WANDB-----------------
    if args.wandb_log:
        wandb_run = args.model_name
        wandb_notes = ''

        import wandb
        print(f"wandb stability run {wandb_run}")
        wandb.login(host='https://stability.wandb.io')#, relogin=True)
        wandb.init(
            id = args.model_name,
            project='stability',
            name=wandb_run,
            config={},
            notes=wandb_notes,
            resume="allow",
            dir='../train_logs/'
        )

    if resume_from_ckpt:
        print("\n---resuming from ckpt_path---\n", ckpt_path)
        checkpoint = torch.load(ckpt_path, map_location='cuda')
        epoch = checkpoint['epoch']+1
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])        
        scheduler.load_state_dict(checkpoint['scheduler_state_dict'])  
        model.load_state_dict(checkpoint['model_state_dict'])
        del checkpoint
        torch.cuda.empty_cache()

    image0 = None
    losses = []
    lrs = []
    val_losses = []
    progress_bar = tqdm(range(epoch,config['trainer_params']['max_epochs']))
    for epoch in progress_bar:
        loss_mse_sum = 0.
        loss_kld_sum = 0.
        model.train()

        try:
            for train_i, (image,) in enumerate(train_dl):
                optimizer.zero_grad()

                image = 2 * image[:, :1].float().cuda() - 1
                image = F.interpolate(image, (32,32), mode='bilinear', align_corners=False)
                if image0 is None:
                    image0 = image[-256:]

                results = model(image)
                loss = model.loss_function(*results, M_N=config['exp_params']['kld_weight'])

                losses.append(loss['loss'].item())
                loss_mse_sum += loss['Reconstruction_Loss'].item()
                loss_kld_sum += -loss['KLD'].item()
                lrs.append(optimizer.param_groups[0]['lr'])

                loss['loss'].backward()
                optimizer.step()
        except ValueError:
            pass
        scheduler.step()

        model.eval()
        with torch.inference_mode():
            try:
                for val_i, (image,) in enumerate(val_dl): 
                
                    image = 2 * image[:, :1].float().cuda() - 1
                    image = F.interpolate(image, (32,32), mode='bilinear', align_corners=False)

                    results = model(image)
                    val_loss = model.loss_function(*results, M_N=config['exp_params']['kld_weight'])

                    val_losses.append(val_loss['loss'].item())
                    break
                logs = {"train/loss": np.mean(losses[-(train_i+1):]),
                    "val/loss": np.mean(val_losses[-(val_i+1):]),
                    "train/lr": lrs[-1],
                    "train/num_steps": len(losses),
                    "val/num_steps": len(val_losses),
                    "train/mse_loss": loss_mse_sum/ (train_i + 1),
                    "train/kld_loss": loss_kld_sum/ (train_i + 1),
                    "val/mse_loss": val_loss['Reconstruction_Loss'],
                    "val/kld_loss": -val_loss['KLD'],
                }
                
                image = image[-256:]
                recons = model.generate(image)
                orig_grid = make_grid(image/2 + 0.5 , nrow=16, padding=2)
                recons_grid = make_grid(recons/2 + 0.5 , nrow=16, padding=2)
                full_grid = torch_to_Image(torch.cat([recons_grid, orig_grid], dim=-1))

                if args.wandb_log:
                    logs['val/samples'] = wandb.Image(full_grid)
                full_grid.save(os.path.join(outdir, f'samples-val.png'))
            
            except ValueError:
                logs = {"train/loss": np.mean(losses[-(train_i+1):]),
                    "train/lr": lrs[-1],
                    "train/num_steps": len(losses),
                    "train/mse_loss": loss_mse_sum/ (train_i + 1),
                    "train/kld_loss": loss_kld_sum/ (train_i + 1),
                }
        
            
            progress_bar.set_postfix(**logs)
            
            # image = image[-256:]
            # recons = model.generate(image)
            # orig_grid = make_grid(image/2 + 0.5 , nrow=16, padding=2)
            # recons_grid = make_grid(recons/2 + 0.5 , nrow=16, padding=2)
            # full_grid = torch_to_Image(torch.cat([recons_grid, orig_grid], dim=-1))

            # if args.wandb_log:
            #     logs['val/samples'] = wandb.Image(full_grid)
            # full_grid.save(os.path.join(outdir, f'samples-val.png'))

            recons = model.generate(image0)
            orig_grid = make_grid(image0/2 + 0.5 , nrow=16, padding=2)
            recons_grid = make_grid(recons/2 + 0.5 , nrow=16, padding=2)
            full_grid = torch_to_Image(torch.cat([recons_grid, orig_grid], dim=-1))

            if args.wandb_log:
                logs['train/samples'] = wandb.Image(full_grid)
            full_grid.save(os.path.join(outdir, f'samples-train.png'))

        save_ckpt('last')
        if args.wandb_log:
            while True:
                try:
                    wandb.log(logs)
                    break
                except:
                    print('Wandb log failed. Retrying')
                    time.sleep(1)

    if args.wandb_log:
        wandb.finish()

