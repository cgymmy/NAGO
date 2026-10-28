import torch
import torch.nn as nn
import torch.distributed as torch_dist
import numpy as np
import argparse
from torch.utils.tensorboard import SummaryWriter
from torch.nn.utils import clip_grad_norm_
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data.distributed import DistributedSampler
import time


import os, sys
sys.path.append(os.path.join(os.path.dirname(__file__), '../'))
sys.path.append(os.path.join(os.path.dirname(__file__), '../../'))
from models.fno import get_model as get_fno
from models.unet import Unet
from src.functional import *
from src.utilities import seed_all, load_config, get_optimizer, get_scheduler, count_parameters
from expt.sns.metrics import get_energy_spectrum

def itr_merge(itrs):
    for itr in itrs:
        for v in enumerate(itr):
            yield v

def get_many_forecasting_dataloader(
    list_loc,
    time_lag,
    lo_size,
    hi_size,
    batch_size,
    train_test_split,
    subsampling_ratio=None,
    num_workers=0,
    rank=0,
    world_size=1,
    processed_data_root=None,
):
    """
    get forecasting dataloader for 2D Stochastic NSE

    Parameters
    ----------
    list_loc: list of locations of data files, of dim num_trajectory*num_snapshots*res*res
    time_lag: creat forecasting data with time_lag 
        (i.e. x_t = data[:,:-time_lag,...] and x_{t+tau}=data[:,time_lag:,...])
    lo_size, hi_size: resolutions of x_t and x_{t+tau}
    batch_size: batch size
    train_test_split: a ratio representing the splitting of training/testing data
    subsampling_ratio: used for subsampling a small portion of data, for convenient small scale experiments
    """

    avg_pixel_norm = 3.0679163932800293 # avg data norm computed a priori

    is_main_process = rank == 0
    if is_main_process:
        print(f'[prepare dataset] time lag {time_lag}')
    
    list_len = len(list_loc)
    list_train_loaders = []
    list_test_loaders = []
    list_train_samplers = []

    def get_processed_path(data_loc):
        if processed_data_root is None:
            return None
        data_name = os.path.splitext(os.path.basename(data_loc))[0]
        split_tag = f'{train_test_split:g}'.replace('.', 'p')
        subsample = 1.0 if subsampling_ratio is None else subsampling_ratio
        subsample_tag = f'{subsample:g}'.replace('.', 'p')
        filename = (
            f'{data_name}_lag{time_lag}_hi{hi_size}_'
            f'split{split_tag}_sub{subsample_tag}.pt'
        )
        return os.path.join(processed_data_root, filename)

    def load_or_build_processed(data_loc):
        processed_path = get_processed_path(data_loc)
        if processed_path is not None and os.path.exists(processed_path):
            if is_main_process:
                print(f'---- [cache] loading processed data {processed_path}')
            cached = torch.load(processed_path, map_location='cpu')
            return cached['train'], cached['test']

        if world_size > 1 and processed_path is not None and not is_main_process:
            torch_dist.barrier()
            if not os.path.exists(processed_path):
                raise FileNotFoundError(f'Processed data cache was not created: {processed_path}')
            if is_main_process:
                print(f'---- [cache] loading processed data {processed_path}')
            cached = torch.load(processed_path, map_location='cpu')
            return cached['train'], cached['test']

        data_raw, time_raw = torch.load(data_loc, map_location='cpu')
        Ntj, Nts, Nx, Ny = data_raw.size()
        tmp = torch.norm(data_raw, dim=(2, 3), p='fro').mean() / np.sqrt(Nx * Ny)
        if is_main_process:
            print(f'---- [dataset] average pixel norm of data set {tmp.item()}')
        data_raw = data_raw / avg_pixel_norm

        if time_lag > 0:
            data_post = data_raw[:, time_lag:, ...]
        else:
            data_post = data_raw

        if is_main_process:
            print(f'---- [processing] low resolution {lo_size}, high resolution {hi_size}')
        hi = torch.nn.functional.interpolate(
            data_post,
            size=(hi_size, hi_size),
            mode='bilinear',
        ).reshape([-1, hi_size, hi_size])
        hi = hi[:, None, :, :]

        if subsampling_ratio:
            hi = hi[:int(subsampling_ratio * hi.size()[0]), ...]

        num_train = int(hi.size()[0] * train_test_split)
        if is_main_process:
            print(
                f'---- [processing] train_test_split {train_test_split}, '
                f'num of training {num_train}, testing {hi.size()[0] - num_train}'
            )
        train = hi[:num_train, ...].contiguous()
        test = hi[num_train:, ...].contiguous()

        if processed_path is not None:
            os.makedirs(os.path.dirname(processed_path), exist_ok=True)
            tmp_path = f'{processed_path}.tmp'
            if is_main_process:
                print(f'---- [cache] saving processed data {processed_path}')
            torch.save(
                {
                    'train': train,
                    'test': test,
                    'meta': {
                        'source': data_loc,
                        'time_lag': time_lag,
                        'lo_size': lo_size,
                        'hi_size': hi_size,
                        'train_test_split': train_test_split,
                        'subsampling_ratio': subsampling_ratio,
                        'avg_pixel_norm': avg_pixel_norm,
                    },
                },
                tmp_path,
            )
            os.replace(tmp_path, processed_path)
            if world_size > 1:
                torch_dist.barrier()

        del data_raw
        return train, test

    for i in range(list_len):
        if is_main_process:
            print(f'---- [data set loc {i}] {list_loc[i]}')
        train, test = load_or_build_processed(list_loc[i])
        if is_main_process and processed_data_root is not None:
            print(f'---- [processing] num of training {train.size(0)}, testing {test.size(0)}')

        train_dataset = torch.utils.data.TensorDataset(train)
        test_dataset = torch.utils.data.TensorDataset(test)
        train_sampler = None
        test_sampler = None
        if world_size > 1:
            train_sampler = DistributedSampler(
                train_dataset,
                num_replicas=world_size,
                rank=rank,
                shuffle=True,
            )
            test_sampler = DistributedSampler(
                test_dataset,
                num_replicas=world_size,
                rank=rank,
                shuffle=False,
            )

        loader_kwargs = {
            'batch_size': batch_size,
            'num_workers': num_workers,
            'pin_memory': torch.cuda.is_available(),
            'persistent_workers': num_workers > 0,
        }
        train_loader = torch.utils.data.DataLoader(
            train_dataset,
            shuffle=train_sampler is None,
            sampler=train_sampler,
            **loader_kwargs,
        )
        test_loader = torch.utils.data.DataLoader(
            test_dataset,
            shuffle=False,
            sampler=test_sampler,
            **loader_kwargs,
        )
      
        list_train_loaders.append(train_loader)
        list_test_loaders.append(test_loader)
        list_train_samplers.append(train_sampler)
    
    new_avg_pixel_norm = 1.0
    
    return (
        list_train_loaders,
        list_test_loaders,
        list_train_samplers,
        avg_pixel_norm,
        new_avg_pixel_norm,
    )


def get_model(model_cfg):
    if model_cfg.type == 'unet':
        return Unet(**model_cfg)
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
        noise = GPPrior(dim=noise_cfg.dim, L=noise_cfg.L * 2 * pi, res=noise_cfg.res, nu=noise_cfg.nu, ell=noise_cfg.ell, device=device)
    elif noise_cfg.noise_type == 'white':
        noise = WNPrior(dim=noise_cfg.dim, L=noise_cfg.L * 2 * pi, res=noise_cfg.res, device=device)
    elif noise_cfg.noise_type == 'gaussian':
        noise = GNPrior(dim=noise_cfg.dim, res=noise_cfg.res, device=device)
    else:
        return NotImplementedError
    return FFM(get_model(model_cfg), noise, intp).to(device)


def setup_distributed(requested_device):
    world_size = int(os.environ.get('WORLD_SIZE', '1'))
    distributed = world_size > 1
    if distributed:
        if not torch.cuda.is_available():
            raise RuntimeError('Distributed SNS training requires CUDA.')
        local_rank = int(os.environ['LOCAL_RANK'])
        rank = int(os.environ['RANK'])
        torch.cuda.set_device(local_rank)
        torch_dist.init_process_group(backend='nccl', init_method='env://')
        device = torch.device('cuda', local_rank)
    else:
        local_rank = 0
        rank = 0
        device = torch.device(requested_device)
        if device.type == 'cuda':
            torch.cuda.set_device(device)
    return distributed, rank, local_rank, world_size, device


def reduce_mean(value, device, world_size):
    value = torch.as_tensor(value, dtype=torch.float64, device=device)
    if world_size > 1:
        torch_dist.all_reduce(value, op=torch_dist.ReduceOp.SUM)
        value /= world_size
    return value.item()


def flow_state_dict(model):
    """Return checkpoints without DDP's implementation-specific `module` prefix."""
    state_dict = model.state_dict()
    return {
        key.replace('model.module.', 'model.', 1): value
        for key, value in state_dict.items()
    }


def normalize_checkpoint_state_dict(state_dict):
    """Accept both canonical checkpoints and checkpoints saved from a wrapped model."""
    return {
        key.replace('model.module.', 'model.', 1): value
        for key, value in state_dict.items()
    }


def default_experiment_name(config):
    """Build a reproducible log-folder name from resolution/model/noise."""
    resolution = int(config.datasets.hi_size)
    model_name = str(config.model_type)
    if model_name == 'fno':
        modes = config.fno.n_modes
        mode_name = 'x'.join(str(int(v)) for v in modes)
        model_name = f'fno_mode{mode_name}'
    noise_name = str(config.noise.noise_type)
    parts = [str(resolution), model_name, noise_name]
    if noise_name == 'matern':
        parts.extend([
            f'nu{config.noise.nu:g}',
            f'ell{config.noise.ell:g}',
        ])
    return '_'.join(parts)




if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', type=str, default='./configs/sns_configs.yaml')
    parser.add_argument('--mode', type=str, choices=['train', 'inf'], default='train')
    parser.add_argument('--device', type=str, default='cuda:0')
    parser.add_argument('--logdir', type=str, default='./logs/sns')
    parser.add_argument('--data-root', type=str, default='data/sns')
    parser.add_argument('--processed-data-root', type=str, default=None)
    parser.add_argument(
        '--savename', type=str, default=None,
        help='log folder name; defaults to resolution_model-noise-parameters',
    )
    parser.add_argument('--resume', type=str, default=None)
    parser.add_argument('--noise-type', type=str, default=None,
                        choices=['matern', 'gaussian', 'white'])
    parser.add_argument('--fno-modes', type=int, nargs=2, default=None,
                        metavar=('MODE_X', 'MODE_Y'))
    parser.add_argument(
        '--resolution', type=int, default=None,
        help='override the SNS training/noise/sample resolution',
    )
    parser.add_argument('--matern-nu', type=float, default=None)
    parser.add_argument('--matern-ell', type=float, default=None)
    parser.add_argument('--global-batch-size', type=int, default=None)
    parser.add_argument('--num-workers', type=int, default=None)
    parser.add_argument(
        '--low-memory-optimizer',
        action='store_true',
        help='disable Adam foreach temporaries to reduce optimizer-step peak memory',
    )
    args = parser.parse_args()

    distributed, rank, local_rank, world_size, device = setup_distributed(args.device)
    is_main_process = rank == 0

    config = load_config(args.config)
    if args.noise_type is not None:
        config.noise.noise_type = args.noise_type
    if args.fno_modes is not None:
        config.fno.n_modes = list(args.fno_modes)
    if args.resolution is not None:
        config.datasets.lo_size = args.resolution
        config.datasets.hi_size = args.resolution
        config.noise.res = args.resolution
        config.sample_dims[-2:] = [args.resolution, args.resolution]
    if args.matern_nu is not None:
        config.noise.nu = args.matern_nu
    if args.matern_ell is not None:
        config.noise.ell = args.matern_ell
    if args.global_batch_size is not None:
        config.datasets.batch_size = args.global_batch_size
    if args.num_workers is not None:
        config.datasets.num_workers = args.num_workers
    if args.low_memory_optimizer:
        config.train.optimizer.foreach = False
    if args.savename is None:
        args.savename = default_experiment_name(config)
    # Different rank seeds make flow-matching time/noise samples independent.
    seed_all(config.train.seed + rank)
    if is_main_process:
        print(config)
        if distributed:
            print(f'Using DistributedDataParallel on {world_size} GPUs')
    logdir = os.path.join(args.logdir, args.savename)
    if is_main_process:
        os.makedirs(logdir, exist_ok=True)
    if distributed:
        torch_dist.barrier()
    writer = SummaryWriter(logdir) if is_main_process else None

    if is_main_process:
        print('Loading datasets...')

    data_cfg = config.datasets
    if data_cfg.batch_size % world_size != 0:
        raise ValueError(
            f'datasets.batch_size ({data_cfg.batch_size}) must be divisible by '
            f'the number of processes ({world_size}).'
        )
    per_device_batch_size = data_cfg.batch_size // world_size
    if is_main_process:
        print(
            f'Global batch size {data_cfg.batch_size}; '
            f'per-GPU batch size {per_device_batch_size}'
        )
    list_suffix = [f"data_file0{i}" for i in np.arange(1,data_cfg.num_dataset+1)]
    list_data_loc = [os.path.join(args.data_root, i + '.pt') for i in list_suffix]
    processed_data_root = args.processed_data_root or getattr(data_cfg, 'processed_root', None)
    (
        list_train_loaders,
        list_test_loaders,
        list_train_samplers,
        original_avg_pixel_norm,
        avg_pixel_norm,
    ) = get_many_forecasting_dataloader(
        list_data_loc,
        data_cfg.time_lag,
        data_cfg.lo_size,
        data_cfg.hi_size,
        per_device_batch_size,
        data_cfg.train_test_split,
        data_cfg.subsampling_ratio,
        getattr(data_cfg, 'num_workers', 0),
        rank,
        world_size,
        processed_data_root,
    )

    if config.noise.intp_type == 'scale':
        ratio = 4e-4
    else:
        ratio = None

    if is_main_process:
        print('Building model...')
    if config.model_type == 'unet':
        cfg_model = config.unet
    elif config.model_type == 'fno':
        cfg_model = config.fno
    else:
        NotImplementedError

    model = get_flow_model(config.noise, cfg_model, device, ratio)
    if is_main_process:
        print(f'Number of parameters: {count_parameters(model)}')

    resume_ckpt = None
    if args.resume is not None:
        if is_main_process:
            print(f'Resuming from checkpoint: {args.resume}')
        # Training checkpoints are trusted local files and may contain the
        # EasyDict config/optimizer objects. PyTorch 2.6 defaults to
        # weights_only=True, which cannot unpickle those objects.
        resume_ckpt = torch.load(args.resume, map_location=device, weights_only=False)
        model.load_state_dict(normalize_checkpoint_state_dict(resume_ckpt['model']))
        torch.cuda.empty_cache()

    # Wrapping the neural operator (rather than FFM) lets get_loss continue to
    # construct flow-matching samples while all parameter gradients use DDP.
    if distributed:
        model.model = DDP(
            model.model,
            device_ids=[local_rank],
            output_device=local_rank,
            gradient_as_bucket_view=True,
        )

    optimizer = get_optimizer(config.train.optimizer, model)
    scheduler = get_scheduler(config.train.scheduler, optimizer)
    if resume_ckpt is not None and 'optimizer' in resume_ckpt:
        if is_main_process:
            print('Resuming optimizer states...')
        optimizer.load_state_dict(resume_ckpt['optimizer'])
    if resume_ckpt is not None and 'scheduler' in resume_ckpt:
        if is_main_process:
            print('Resuming scheduler states...')
        scheduler.load_state_dict(resume_ckpt['scheduler'])
    optimizer.zero_grad(set_to_none=True)
    global_step = resume_ckpt.get('step', 0) if resume_ckpt is not None else 0


    def train():
        global global_step

        epoch = 0
        spectral_info = []
        while True:
            for sampler in list_train_samplers:
                if sampler is not None:
                    sampler.set_epoch(epoch)
            model.train()
            epoch_losses = []
            for idx, [x] in itr_merge(list_train_loaders):
                x = x.to(device, non_blocking=True)
                loss = model.get_loss(x)
                epoch_losses.append(loss.item())
                loss.backward()
                grad_norm = clip_grad_norm_(model.parameters(), config.train.max_grad_norm)
                optimizer.step()
                optimizer.zero_grad(set_to_none=True)

                # Logging
                if global_step % config.train.log_freq == 0:
                    train_loss = reduce_mean(loss.detach(), device, world_size)
                    mean_grad_norm = reduce_mean(grad_norm.detach(), device, world_size)
                    if is_main_process:
                        writer.add_scalar('train/loss', train_loss, global_step)
                        writer.add_scalar('train/grad', mean_grad_norm, global_step)
                        writer.add_scalar('train/lr', optimizer.param_groups[0]['lr'], global_step)
                        print(f'Epoch {epoch} Step {global_step} train loss {train_loss:.6f}')
                global_step += 1
                if global_step % config.train.val_freq == 0:
                    avg_val_loss = validate()
                    # gen = sample_uncond()
                    # kvals, Abins_w1, Abins_v1 = get_energy_spectrum(gen.squeeze().detach().cpu())
                    # spectral_info.append([Abins_w1, Abins_v1])
                    # torch.save(spectral_info, os.path.join(logdir, 'spectral_info.pt'))

                    if config.train.scheduler.type == 'plateau':
                        scheduler.step(avg_val_loss)
                    else:
                        scheduler.step()

                    model.train()

                    if is_main_process:
                        torch.save({
                            'config': config,
                            'model': flow_state_dict(model),
                            'step': global_step,
                            'optimizer': optimizer.state_dict(),
                            'scheduler': scheduler.state_dict(),
                            'avg_val_loss': avg_val_loss,
                        }, os.path.join(logdir, 'latest.pt'))

                        if global_step % config.train.save_freq == 0:
                            ckpt_path = os.path.join(logdir, f'{global_step}.pt')
                            torch.save({
                                'config': config,
                                'model': flow_state_dict(model),
                                'step': global_step,
                                'optimizer': optimizer.state_dict(),
                                'scheduler': scheduler.state_dict(),
                                'avg_val_loss': avg_val_loss,
                            }, ckpt_path)
                if global_step >= config.train.max_iter:
                    # torch.save(spectral_info, os.path.join(logdir, 'spectral_info.pt'))
                    return 

            epoch_loss = reduce_mean(
                sum(epoch_losses) / len(epoch_losses),
                device,
                world_size,
            )
            if is_main_process:
                print(f'Epoch {epoch} train loss {epoch_loss:.6f}')
            epoch += 1
        
        


    def validate():
        with torch.no_grad():
            model.eval()

            loss_sum = torch.zeros((), dtype=torch.float64, device=device)
            loss_count = torch.zeros((), dtype=torch.float64, device=device)
            total = max(1, int(np.ceil(config.train.valid_max_batch / world_size)))
            for i, (_, [x]) in tqdm(
                enumerate(itr_merge(list_test_loaders)),
                total=total,
                disable=not is_main_process,
            ):
                if i >= total:
                    break
                x = x.to(device, non_blocking=True)
                loss = model.get_loss(x)
                loss_sum += loss.detach().double() * x.size(0)
                loss_count += x.size(0)
        if distributed:
            torch_dist.all_reduce(loss_sum, op=torch_dist.ReduceOp.SUM)
            torch_dist.all_reduce(loss_count, op=torch_dist.ReduceOp.SUM)
        val_loss = (loss_sum / loss_count).item()
        if is_main_process:
            writer.add_scalar('valid/loss', val_loss, global_step)
            print(f'Step {global_step} valid loss {val_loss:.6f}')
        return val_loss


    @torch.no_grad()
    def sample_uncond():
        model.eval()
        gen = model.sample(config.n_sample, config.n_eval, config.sample_dims, device)
        return gen


    try:
        if args.mode == 'train':
            # sample_uncond()
            train()
            if is_main_process:
                print('Training finished!')
        if is_main_process and args.mode == 'inf' and args.resume is None:
            print('[WARNING]: inference mode without loading a pretrained model')

        # sample_uncond()
        if is_main_process:
            print('Sampling finished!')
        time.sleep(5)
    except KeyboardInterrupt:
        if is_main_process:
            print('Terminating...')
    finally:
        if writer is not None:
            writer.close()
        if distributed and torch_dist.is_initialized():
            torch_dist.destroy_process_group()
