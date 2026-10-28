import argparse
from json import load
import os
import time

from tqdm import tqdm
import torch
from torch.nn.utils import clip_grad_norm_
from torch.utils.data import DataLoader, Dataset
from torch.utils.tensorboard import SummaryWriter
import numpy as np


import os, sys
sys.path.append(os.path.join(os.path.dirname(__file__), '../'))
sys.path.append(os.path.join(os.path.dirname(__file__), '../../'))
from models.unet1d import Unet1D
from models.fno import get_model as get_fno
from src.functional import *
from src.utilities import seed_all, load_config, get_optimizer, get_scheduler, count_parameters

# def get_AllenCahn_dataloader(grid_size, batch_size, train_test_split, subsampling_ratio = None, loc="./data/ac/hwm10.pt"):
#     """
#     get dataloader for 2D Allen Cahn

#     Parameters
#     ----------
#     grid_size: resolutions of the field
#     batch_size: batch size
#     train_test_split: a ratio representing the splitting of training/testing data
#     subsampling_ratio: used for subsampling a small portion of data, for convenient small scale experiments
#     """
#     loc = loc + f"AllenCahn1D_grid{grid_size}_samples.npy"
    
#     data_raw = np.load(loc)
#     print(f"[Data] raw data shape is {data_raw.shape}")
#     torch_data = torch.from_numpy(data_raw)
#     torch_data = torch.reshape(torch_data, (-1, torch_data.shape[-1]))
#     print(f"[Data] flattened torchdata, shape is {torch_data.shape}")
#     norm_per_pixel = torch.norm(torch_data,dim=(1),p='fro').mean()/torch_data.shape[-1]
#     print(f"[Data] norm per pixel {norm_per_pixel}")

#     if subsampling_ratio:
#         torch_data = torch_data[:int(subsampling_ratio*torch_data.size()[0]),...]

#     num_train = int(torch_data.size()[0]*train_test_split)
#     print(f'---- [processing] train_test_split {train_test_split}, num of training {num_train}, testing {torch_data.size()[0]-num_train}')
    
#     train_loader = torch.utils.data.DataLoader(torch.utils.data.TensorDataset(torch_data[:num_train,None,:grid_size].float()), batch_size=batch_size, shuffle=True)
#     test_loader = torch.utils.data.DataLoader(torch.utils.data.TensorDataset(torch_data[num_train:,None,:grid_size].float()), batch_size=batch_size, shuffle=False)

#     return train_loader, test_loader, norm_per_pixel


def get_AllenCahn_dataloader(grid_size, batch_size, train_test_split, subsampling_ratio = None, loc="./data/ac/hwm10.pt"):
    """
    get dataloader for 2D Allen Cahn

    Parameters
    ----------
    grid_size: resolutions of the field
    batch_size: batch size
    train_test_split: a ratio representing the splitting of training/testing data
    subsampling_ratio: used for subsampling a small portion of data, for convenient small scale experiments
    """
    
    torch_data = torch.load(loc, map_location="cpu", weights_only=False)
    print(f"[Data] raw data shape is {torch_data.shape}")
    # torch_data = torch.from_numpy(data_raw)
    # torch_data = torch.reshape(torch_data, (-1, torch_data.shape[-1]))
    print(f"[Data] flattened torchdata, shape is {torch_data.shape}")
    norm_per_pixel = torch.norm(torch_data,dim=(1),p='fro').mean()/torch_data.shape[-1]
    print(f"[Data] norm per pixel {norm_per_pixel}")

    if subsampling_ratio:
        torch_data = torch_data[:int(subsampling_ratio*torch_data.size()[0]),...]

    num_train = int(torch_data.size()[0]*train_test_split)
    print(f'---- [processing] train_test_split {train_test_split}, num of training {num_train}, testing {torch_data.size()[0]-num_train}')
    
    train_loader = torch.utils.data.DataLoader(torch.utils.data.TensorDataset(torch_data[:num_train,None,:grid_size].float()), batch_size=batch_size, shuffle=True)
    test_loader = torch.utils.data.DataLoader(torch.utils.data.TensorDataset(torch_data[num_train:,None,:grid_size].float()), batch_size=batch_size, shuffle=False)

    return train_loader, test_loader, norm_per_pixel


def get_model(model_cfg):
    if model_cfg.type == 'unet1d':
        return Unet1D(**model_cfg)
    elif model_cfg.type == 'fno':
        return get_fno(model_cfg)
    else:
        raise NotImplementedError


def get_flow_model(noise_cfg, model_cfg, device, ratio=None):
    """
    Build the functional flow model.
    :param model_cfg: model configs passed to the flow model, type indicates the model type
    :param encoder_cfg: encoder configs passed to the encoder model
    :param conditional: whether the model is conditional
    :return: the flow model
    """
    if noise_cfg.intp_type == 'scale':
        assert ratio is not None
        intp = scale_intp(ratio=ratio)
    else:
        intp = interpolation()
        
    if noise_cfg.noise_type == 'matern':
        noise = GPPrior(dim=noise_cfg.dim, L=noise_cfg.L, res=noise_cfg.res, nu=noise_cfg.nu, ell=noise_cfg.ell, device=device)
    elif noise_cfg.noise_type == 'white':
        noise = WNPrior(dim=noise_cfg.dim, L=noise_cfg.L, res=noise_cfg.res, device=device)
    elif noise_cfg.noise_type == 'gaussian':
        noise = GNPrior(dim=noise_cfg.dim, res=noise_cfg.res, device=device)
    else:
        return NotImplementedError

    return FFM(get_model(model_cfg), noise, intp).to(device)

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', type=str, default='./configs/ac_configs.yaml')
    parser.add_argument('--mode', type=str, choices=['train', 'inf'], default='train')
    parser.add_argument('--device', type=str, default='cuda:0')
    parser.add_argument('--logdir', type=str, default='./logs/ac')
    parser.add_argument('--savename', type=str, default='ac_matern_fno_005')
    parser.add_argument('--resume', type=str, default=None)
    parser.add_argument('--data-file', default='./data/ac/hwm10.pt')
    parser.add_argument('--noise-type', choices=['matern', 'gaussian', 'white'])
    parser.add_argument('--samples-output', default=None)
    parser.add_argument('--n-eval', type=int, default=None)
    args = parser.parse_args()
    if args.mode == 'inf' and args.resume is None:
        parser.error('--mode inf requires --resume')

    config = load_config(args.config)
    if args.noise_type is not None:
        config.noise.noise_type = args.noise_type
    if args.n_eval is not None:
        config.n_eval = args.n_eval
    seed_all(config.train.seed)
    print(config)
    logdir = os.path.join(args.logdir, args.savename)
    if not os.path.exists(logdir):
        os.makedirs(logdir, exist_ok=True)
    writer = SummaryWriter(logdir)

    print('Loading datasets...')
    # train_set, test_set = get_dataset(config.datasets)

    data_cfg = config.datasets
    train_loader, test_loader, norm_per_pixel = get_AllenCahn_dataloader(
        data_cfg.grid_size, 
        data_cfg.batch_size, 
        data_cfg.train_test_split, 
        data_cfg.subsampling_ratio,
        loc=args.data_file)

    if config.noise.intp_type == 'scale':
        ratio = 4e-4
    else:
        ratio = None

    print('Building model...')
    if config.model_type == 'unet':
        cfg_model = config.unet
    elif config.model_type == 'fno':
        cfg_model = config.fno
    else:
        NotImplementedError
    model = get_flow_model(config.noise, cfg_model, args.device, ratio)
    print(f'Number of parameters: {count_parameters(model)}')

    optimizer = get_optimizer(config.train.optimizer, model)
    scheduler = get_scheduler(config.train.scheduler, optimizer)
    optimizer.zero_grad()

    if args.resume is not None:
        print(f'Resuming from checkpoint: {args.resume}')
        ckpt = torch.load(args.resume, map_location=args.device, weights_only=False)
        model.load_state_dict(ckpt['model'])
        if 'optimizer' in ckpt:
            print('Resuming optimizer states...')
            optimizer.load_state_dict(ckpt['optimizer'])
        if 'scheduler' in ckpt:
            print('Resuming scheduler states...')
            scheduler.load_state_dict(ckpt['scheduler'])
        torch.cuda.empty_cache()
    global_step = 0


    def train():
        global global_step

        epoch = 0
        while True:
            model.train()
            epoch_losses = []
            for [x] in train_loader:
                x = x.to(args.device)
                loss = model.get_loss(x)
                epoch_losses.append(loss.item())
                loss.backward()
                grad_norm = clip_grad_norm_(model.parameters(), config.train.max_grad_norm)
                optimizer.step()
                optimizer.zero_grad()

                # Logging
                writer.add_scalar('train/loss', loss.item(), global_step)
                writer.add_scalar('train/grad', grad_norm.item(), global_step)
                writer.add_scalar('train/lr', optimizer.param_groups[0]['lr'], global_step)
                if global_step % config.train.log_freq == 0:
                    print(f'Epoch {epoch} Step {global_step} train loss {loss.item():.6f}')
                global_step += 1
                if global_step % config.train.val_freq == 0:
                    avg_val_loss = validate()
                    # sample_uncond()
                    if config.train.scheduler.type == 'plateau':
                        scheduler.step(avg_val_loss)
                    else:
                        scheduler.step()

                    model.train()
                    torch.save({
                        'config': config,
                        'model': model.state_dict(),
                        'step': global_step,
                    }, os.path.join(logdir, 'latest.pt'))
                    if global_step % config.train.save_freq == 0:
                        ckpt_path = os.path.join(logdir, f'{global_step}.pt')
                        torch.save({
                            'config': config,
                            'model': model.state_dict(),
                            'optimizer': optimizer.state_dict(),
                            'scheduler': scheduler.state_dict(),
                            'avg_val_loss': avg_val_loss,
                        }, ckpt_path)
                if global_step >= config.train.max_iter:
                    return

            epoch_loss = sum(epoch_losses) / len(epoch_losses)
            print(f'Epoch {epoch} train loss {epoch_loss:.6f}')
            epoch += 1


    def validate():
        with torch.no_grad():
            model.eval()

            val_losses = []
            total = config.train.valid_max_batch or len(test_loader)
            total = min(total, len(test_loader))
            for i, [x] in tqdm(enumerate(test_loader), total=total):
                if i >= total:
                    break
                x = x.to(args.device)
                loss = model.get_loss(x)
                val_losses.append(loss.item())
        val_loss = sum(val_losses) / len(val_losses)
        writer.add_scalar('valid/loss', val_loss, global_step)
        print(f'Step {global_step} valid loss {val_loss:.6f}')
        return val_loss


    @torch.no_grad()
    def sample_uncond():
        model.eval()
        gen = model.sample(config.n_sample, config.n_eval, config.sample_dims, args.device)
        # torch.save(gen, 'noise.pt')
        # for i in range(config.n_sample):
        return gen


    try:
        if args.mode == 'train':
            # sample_uncond()
            train()
            print('Training finished!')
        if args.mode == 'inf' and args.resume is None:
            print('[WARNING]: inference mode without loading a pretrained model')

        samples = sample_uncond().cpu()
        samples_output = args.samples_output or os.path.join(logdir, 'generated_samples.pt')
        os.makedirs(os.path.dirname(os.path.abspath(samples_output)), exist_ok=True)
        torch.save(samples, samples_output)
        writer.close()
        print(f'Sampling finished: {samples_output}')
        time.sleep(5)
    except KeyboardInterrupt:
        print('Terminating...')
