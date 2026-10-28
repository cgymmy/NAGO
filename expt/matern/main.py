import argparse
import os
import time

from tqdm import tqdm
import torch
import torch.distributed as torch_dist
from torch.nn.utils import clip_grad_norm_
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DataLoader, Dataset
from torch.utils.data.distributed import DistributedSampler
from torch.utils.tensorboard import SummaryWriter



import os, sys
sys.path.append(os.path.join(os.path.dirname(__file__), '../'))
sys.path.append(os.path.join(os.path.dirname(__file__), '../../'))
from models.fno import get_model as get_fno
from models.unet import Unet
from src.functional import *
from src.utilities import seed_all, load_config, get_optimizer, get_scheduler, count_parameters
from src.random_field import get_Fourier_spectrum

class GPPriorDataset(Dataset):
    def __init__(self, data_cfg, n_samples, device="cpu", normalize=True):
        '''
        [B, 1, *dim]
        '''
        self.device = device
        self.normalize = normalize
        gp = GPPrior(
            dim=data_cfg.dim,
            L=data_cfg.L,
            res=data_cfg.res,
            nu=data_cfg.nu,
            ell=data_cfg.ell,
            device=device,
        )
        self.data = gp.sample(n_samples=n_samples, normalize=normalize).unsqueeze(1)

    def __len__(self):
        return self.data.shape[0]

    def __getitem__(self, idx):
        return self.data[idx]

def get_ratio_intp(data_cfg, noise_cfg, device='cpu'):
    res = noise_cfg.res
    mu1 = GPPrior(dim=noise_cfg.dim, L=noise_cfg.L, res=noise_cfg.res, nu=noise_cfg.nu, ell=noise_cfg.ell, device=device)
    sgm1 = sqrt(mu1.sigma2)
    _, sd1, _, _, _ = mu1.generate(1, None, True)
    sd1 = 1.0 / (sd1 * sgm1)**2
    mu2 = GPPrior(dim=data_cfg.dim, L=data_cfg.L, res=data_cfg.res, nu=data_cfg.nu, ell=data_cfg.ell, device=device)
    sgm2 = sqrt(mu2.sigma2)
    _, sd2, _, _, _ = mu2.generate(1, None, True)
    sd2 = 1.0 / (sd2 * sgm2)**2
    ratio = sd2[res//2, res//2] / sd1[res//2, res//2]
    return ratio


def setup_distributed(requested_device):
    world_size = int(os.environ.get('WORLD_SIZE', '1'))
    distributed = world_size > 1
    if distributed:
        if not torch.cuda.is_available():
            raise RuntimeError('Distributed Matérn training requires CUDA.')
        local_rank = int(os.environ['LOCAL_RANK'])
        rank = int(os.environ['RANK'])
        torch.cuda.set_device(local_rank)
        torch_dist.init_process_group(backend='nccl', init_method='env://')
        device = torch.device('cuda', local_rank)
    else:
        rank = 0
        local_rank = 0
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
    """Save checkpoints without DDP's implementation-specific module prefix."""
    return {
        key.replace('model.module.', 'model.', 1): value
        for key, value in model.state_dict().items()
    }


def normalize_checkpoint_state_dict(state_dict):
    """Load checkpoints produced by either single-GPU or DDP training."""
    return {
        key.replace('model.module.', 'model.', 1): value
        for key, value in state_dict.items()
    }


def default_experiment_name(config):
    """Match the SNS resolution/model/noise checkpoint naming convention."""
    resolution = int(config.datasets.res)
    model_name = str(config.model_type)
    if model_name == 'fno':
        mode_name = 'x'.join(str(int(v)) for v in config.fno.n_modes)
        model_name = f'fno_mode{mode_name}'
    noise_name = str(config.noise.noise_type)
    parts = [str(resolution), model_name, noise_name]
    if noise_name == 'matern':
        parts.extend([
            f'nu{config.noise.nu:g}',
            f'ell{config.noise.ell:g}',
        ])
    return '_'.join(parts)



def get_flow_model(noise_cfg, net, device, ratio=None):
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
    return FFM(net, noise, intp).to(device)

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', type=str, default='./configs/matern_configs.yaml')
    parser.add_argument('--mode', type=str, choices=['train', 'inf'], default='train')
    parser.add_argument('--device', type=str, default='cuda:0')
    parser.add_argument('--logdir', type=str, default='./logs/matern')
    parser.add_argument(
        '--savename', type=str, default=None,
        help='log folder name; defaults to the SNS naming convention',
    )
    parser.add_argument('--resume', type=str, default=None)
    parser.add_argument('--noise-type', type=str, default=None,
                        choices=['matern', 'gaussian', 'white'])
    parser.add_argument('--resolution', type=int, default=None)
    parser.add_argument('--fno-modes', type=int, nargs=2, default=None,
                        metavar=('MODE_X', 'MODE_Y'))
    parser.add_argument('--nu', type=float, default=1.0,
                    help='Smoothness parameter of the Matérn kernel')
    parser.add_argument('--ell', type=float, default=0.05,
                    help='Length-scale parameter of the Matérn kernel')
    parser.add_argument('--steps', type=int, default=5)
    parser.add_argument('--num-workers', type=int, default=2,
                        help='DataLoader workers per GPU/process')
    args = parser.parse_args()

    distributed, rank, local_rank, world_size, device = setup_distributed(args.device)
    is_main_process = rank == 0
    config = load_config(args.config)
    if args.noise_type is not None:
        config.noise.noise_type = args.noise_type
    if args.resolution is not None:
        config.datasets.res = args.resolution
        config.noise.res = args.resolution
        config.sample_dims[-2:] = [args.resolution, args.resolution]
    if args.fno_modes is not None:
        config.fno.n_modes = list(args.fno_modes)
    config.noise.nu = args.nu
    config.noise.ell = args.ell

    # All ranks construct the same synthetic dataset so DistributedSampler can
    # partition it without communicating a large CPU tensor.
    seed_all(config.train.seed)
    # print(config)
    if args.savename is None or args.savename == 'default':
        savename = default_experiment_name(config)
    elif args.savename == 'simple':
        savename = f"{config.noise.noise_type}_{config.model_type}"
    else:
        savename = args.savename
    logdir = os.path.join(args.logdir, savename)
    if is_main_process:
        os.makedirs(logdir, exist_ok=True)
    if distributed:
        torch_dist.barrier(device_ids=[local_rank])
    writer = SummaryWriter(logdir) if is_main_process else None

    if is_main_process:
        print('Loading datasets...')
    # train_set, test_set = get_dataset(config.datasets)

    data_cfg = config.datasets
    train_set = GPPriorDataset(data_cfg, n_samples=8000, normalize=True)
    test_set  = GPPriorDataset(data_cfg, n_samples=200,  normalize=True)

    if config.train.batch_size % world_size != 0:
        raise ValueError(
            f'train.batch_size ({config.train.batch_size}) must be divisible by '
            f'the number of processes ({world_size}).'
        )
    per_device_batch_size = config.train.batch_size // world_size
    train_sampler = DistributedSampler(
        train_set, num_replicas=world_size, rank=rank, shuffle=True,
    ) if distributed else None
    test_sampler = DistributedSampler(
        test_set, num_replicas=world_size, rank=rank, shuffle=False,
    ) if distributed else None
    loader_kwargs = {
        'batch_size': per_device_batch_size,
        'num_workers': args.num_workers,
        'pin_memory': device.type == 'cuda',
        'persistent_workers': args.num_workers > 0,
    }
    train_loader = DataLoader(
        train_set,
        shuffle=train_sampler is None,
        sampler=train_sampler,
        **loader_kwargs,
    )
    test_loader = DataLoader(
        test_set,
        shuffle=False,
        sampler=test_sampler,
        **loader_kwargs,
    )
    if is_main_process:
        print(
            f'Using {world_size} GPU(s): global batch size '
            f'{config.train.batch_size}, per-GPU batch size {per_device_batch_size}'
        )

    # Dataset construction must be identical; stochastic flow samples should
    # differ across ranks.
    seed_all(config.train.seed + rank)

    if config.noise.intp_type == 'scale':
        ratio = get_ratio_intp(data_cfg, config.noise, device)
    else:
        ratio = None

    if is_main_process:
        print('Building model...')
    if config.model_type == 'fno':
        net = get_fno(config.fno)
    elif config.model_type == 'unet':
        # net = Unet2D(**config.unet)
        net = Unet(**config.unet)

    else:
        raise NotImplementedError
    
    model = get_flow_model(config.noise, net, device, ratio)
    if is_main_process:
        print(f'Number of parameters: {count_parameters(model)}')

    resume_ckpt = None
    if args.resume is not None:
        if is_main_process:
            print(f'Resuming from checkpoint: {args.resume}')
        resume_ckpt = torch.load(args.resume, map_location=device, weights_only=False)
        model.load_state_dict(
            normalize_checkpoint_state_dict(resume_ckpt['model'])
        )
        torch.cuda.empty_cache()

    # FFM owns non-module GP state, so only wrap its trainable neural operator.
    if distributed and args.mode == 'train':
        model.model = DDP(
            model.model,
            device_ids=[local_rank],
            output_device=local_rank,
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
            if train_sampler is not None:
                train_sampler.set_epoch(epoch)
            model.train()
            epoch_losses = []
            for x in train_loader:
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
                    mean_grad_norm = reduce_mean(
                        grad_norm.detach(), device, world_size
                    )
                    if is_main_process:
                        writer.add_scalar('train/loss', train_loss, global_step)
                        writer.add_scalar('train/grad', mean_grad_norm, global_step)
                        writer.add_scalar(
                            'train/lr', optimizer.param_groups[0]['lr'], global_step
                        )
                        print(
                            f'Epoch {epoch} Step {global_step} '
                            f'train loss {train_loss:.6f}'
                        )
                global_step += 1
                if global_step % config.train.val_freq == 0:
                    avg_val_loss = validate()

                    # gen = sample_uncond()
                    # k, w = get_Fourier_spectrum(gen.squeeze().detach().cpu())
                    # spectral_info.append(w)
                    # torch.save(spectral_info, os.path.join(logdir, 'spectral_info.pt'))

                    if config.train.scheduler.type == 'plateau':
                        scheduler.step(avg_val_loss)
                    else:
                        scheduler.step()

                    model.train()
                    if is_main_process:
                        checkpoint = {
                            'config': config,
                            'model': flow_state_dict(model),
                            'step': global_step,
                            'optimizer': optimizer.state_dict(),
                            'scheduler': scheduler.state_dict(),
                            'avg_val_loss': avg_val_loss,
                        }
                        torch.save(checkpoint, os.path.join(logdir, 'latest.pt'))
                        if global_step % config.train.save_freq == 0:
                            torch.save(
                                checkpoint,
                                os.path.join(logdir, f'{global_step}.pt'),
                            )
                if global_step >= config.train.max_iter:
                    return

            epoch_loss = reduce_mean(
                sum(epoch_losses) / len(epoch_losses), device, world_size
            )
            if is_main_process:
                print(f'Epoch {epoch} train loss {epoch_loss:.6f}')
            epoch += 1


    def validate():
        with torch.no_grad():
            model.eval()

            loss_sum = torch.zeros((), dtype=torch.float64, device=device)
            loss_count = torch.zeros((), dtype=torch.float64, device=device)
            total = config.train.valid_max_batch or len(test_loader)
            total = min(total, len(test_loader))
            for i, x in tqdm(
                enumerate(test_loader),
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
        # gen = model.sample(config.n_sample, config.n_eval, config.sample_dims, args.device)
        gen = model.sample(config.n_sample, args.steps, config.sample_dims, device)
        
        # torch.save(gen, 'noise.pt')
        return gen


    try:
        if args.mode == 'train':
            # sample_uncond()
            train()
            if is_main_process:
                print('Training finished!')
        if args.mode == 'inf' and is_main_process:
            ckpt = torch.load(
                logdir + '/latest.pt', map_location=device, weights_only=False
            )
            model.load_state_dict(
                normalize_checkpoint_state_dict(ckpt['model'])
            )
            gen = sample_uncond()
            k, w = get_Fourier_spectrum(gen.squeeze().detach().cpu())
            torch.save([k, w], os.path.join(logdir, 'spectral_results.pt'))


        if is_main_process:
            print('Sampling finished!')
        time.sleep(2)
    except KeyboardInterrupt:
        if is_main_process:
            print('Terminating...')
    finally:
        if writer is not None:
            writer.close()
        if distributed and torch_dist.is_initialized():
            torch_dist.destroy_process_group()
