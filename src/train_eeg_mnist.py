import os
import shutil
import yaml
import time
import argparse
import numpy as np
from pathlib import Path
import torch.backends.cudnn as cudnn
import random
import torch
import webdataset as wds
import wandb
from tqdm import tqdm
import torch.nn.functional as F
from torchvision.utils import make_grid
from torchvision import transforms
import utils
from models import BrainNetworkMnistEEG, VanillaVAE


def torch_to_Image(x):
    if x.ndim==4:
        x=x[0]
    return transforms.ToPILImage()(x)

def soft_clip_loss(preds, targs, temp=0.04):
    preds = F.normalize(preds, 2, dim=-1)
    targs = F.normalize(targs, 2, dim=-1)

    clip_clip = (targs @ targs.T)/temp
    brain_clip = (preds @ targs.T)/temp
    
    loss1 = -(brain_clip.log_softmax(-1) * clip_clip.softmax(-1)).sum(-1).mean()
    loss2 = -(brain_clip.T.log_softmax(-1) * clip_clip.softmax(-1)).sum(-1).mean()
    
    loss = (loss1 + loss2)/2
    return loss

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
                        default='../PyTorch-VAE/configs/vae.yaml')
    parser.add_argument(
            "--model_name",
            type=str,
            default="eeg_mnist",
            help="name of model, used for wandb logging")
    parser.add_argument(
            "--wandb_log",
            action="store_true",
            help="whether to log to wandb")
    parser.add_argument(
            "--num_epochs",
            type=int,
            default=800,
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

    vae = VanillaVAE(**config['model_params'])
    vae.load_state_dict(torch.load('../train_logs/models/mnist_vae/last.pth')['model_state_dict'])
    model = BrainNetworkMnistEEG(big=False, h=128)
    model.decoder.load_state_dict(vae.decoder.state_dict())
    model.final_layer.load_state_dict(vae.final_layer.state_dict())
    # del vae
    model.cuda()
    vae.cuda()
    vae.eval()

    train_data = wds.WebDataset('/fsx/proj-fmri/shared/eeg_mnist_train_1.tar', resampled=False, nodesplitter=wds.split_by_node)\
                .shuffle(500, initial=500, rng=random.Random(0))\
                .decode("torch")\
                .rename(images="jpg;png", eeg='eeg.npy')\
                .to_tuple("eeg", "images")\
                .batched(1024, partial=False)\
                .with_epoch(45)
    train_dl = torch.utils.data.DataLoader(train_data, 
                            num_workers=1,
                            batch_size=None, shuffle=False, persistent_workers=True)

    val_data = wds.WebDataset('/fsx/proj-fmri/shared/eeg_mnist_test_1.tar', resampled=False, nodesplitter=wds.split_by_node)\
                .decode("torch")\
                .rename(images="jpg;png", eeg='eeg.npy')\
                .to_tuple("eeg", "images")\
                .batched(4096, partial=False)\
                .with_epoch(1)
    val_dl = torch.utils.data.DataLoader(val_data, num_workers=1,
                    batch_size=None, shuffle=False, persistent_workers=True)

    opt_grouped_parameters = [
        {'params': model.conv0.parameters(), 'weight_decay': 1e-2},
        {'params': model.lin0.parameters(), 'weight_decay': 1e-2},
        # {'params': model.decoder.parameters(), 'weight_decay': 1e-4},
        # {'params': model.final_layer.parameters(), 'weight_decay': 1e-4},
    ]
    optimizer = torch.optim.AdamW(opt_grouped_parameters, lr=0) # lr doesnt get used if lr_scheduler='cycle'
    scheduler = torch.optim.lr_scheduler.OneCycleLR(
        optimizer, max_lr=[1e-3, 1e-3], # max_lr=[1e-3, 1e-3, 5e-5, 5e-5],
        total_steps=45*args.num_epochs,
        pct_start=0.005
    )

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
    progress_bar = tqdm(range(epoch, args.num_epochs))
    for epoch in progress_bar:
        loss_mse_sum = 0.
        loss_kld_sum = 0.
        model.train()

        # try:
        for train_i, (eeg, image,) in enumerate(train_dl):
            optimizer.zero_grad()
            
            eeg = eeg.float().cuda()
            image = 2 * image[:, :1].float().cuda() - 1
            image = F.interpolate(image, (32,32), mode='bilinear', align_corners=False)
            if image0 is None:
                image0 = image[-256:]
                eeg0 = eeg[-256:]

            z = vae.reparameterize(*vae.encode(image))
            targ = vae.decoder_input(z).reshape(-1, vae.hidden_dims[-1], 2, 2)
            
            results, enc = model(eeg, return_enc=True)
            loss = F.l1_loss(results, image) + F.mse_loss(enc, targ) + soft_clip_loss(enc.flatten(1), targ.flatten(1))
            # loss = F.mse_loss(results, image) + F.mse_loss(enc, targ)

            losses.append(loss.item())
            lrs.append(optimizer.param_groups[0]['lr'])

            loss.backward()
            optimizer.step()
            scheduler.step()
        # except ValueError:
        #     pass

        if epoch%10==9:
            model.eval()
            with torch.inference_mode():
                try:
                    for val_i, (eeg, image) in enumerate(val_dl): 
                    
                        eeg = eeg.float().cuda()
                        image = 2 * image[:, :1].float().cuda() - 1
                        image = F.interpolate(image, (32,32), mode='bilinear', align_corners=False)

                        results = model(eeg)
                        val_loss = F.l1_loss(results, image)
                        # val_loss = F.mse_loss(results, image)

                        val_losses.append(val_loss.item())
                        break
                    logs = {"train/loss": np.mean(losses[-(train_i+1):]),
                        "val/loss": np.mean(val_losses[-(val_i+1):]),
                        "train/lr": lrs[-1],
                        "train/num_steps": len(losses),
                        "val/num_steps": len(val_losses),
                    }
                    
                    image = image[-256:]
                    eeg = eeg[-256:]
                    recons = model(eeg)
                    orig_grid = make_grid(image/2 + 0.5 , nrow=16, padding=2, pad_value=0.5)
                    recons_grid = make_grid(recons/2 + 0.5 , nrow=16, padding=2, pad_value=0.5)
                    full_grid = torch_to_Image(torch.cat([recons_grid, orig_grid], dim=-1))

                    if args.wandb_log:
                        logs['val/samples'] = wandb.Image(full_grid)
                    full_grid.save(os.path.join(outdir, f'samples-val.png'))
                
                except ValueError:
                    logs = {"train/loss": np.mean(losses[-(train_i+1):]),
                        "train/lr": lrs[-1],
                        "train/num_steps": len(losses),
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

                recons = model(eeg0)
                orig_grid = make_grid(image0/2 + 0.5 , nrow=16, padding=2, pad_value=0.5)
                recons_grid = make_grid(recons/2 + 0.5 , nrow=16, padding=2, pad_value=0.5)
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

