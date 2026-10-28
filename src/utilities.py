# Modifications for PCFM © 2025 Pengfei Cai (Learning Matter @ MIT) and Utkarsh (Julia Lab @ MIT), licensed under the MIT License.
# Original portions © Amazon.com, Inc. or its affiliates, licensed under the Apache License 2.0.

import random

import yaml
from easydict import EasyDict
import numpy as np

import io

import torch
from torchvision.transforms import ToTensor
from PIL import Image
from matplotlib import pyplot as plt
import plotly.graph_objects as go

plt.switch_backend('agg')

def load_config(config_path):
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    return EasyDict(config)


def seed_all(seed):
    if seed is None:
        return
    # torch.backends.cudnn.deterministic = True
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)


def recursive_to_device(x, device):
    if isinstance(x, torch.Tensor):
        return x.to(device)
    elif isinstance(x, dict):
        return {k: recursive_to_device(v, device) for k, v in x.items()}
    elif isinstance(x, (list, tuple)):
        return [recursive_to_device(v, device) for v in x]
    else:
        return x


def get_optimizer(cfg, model):
    optimizer_kwargs = {}
    foreach = getattr(cfg, 'foreach', None)
    if foreach is not None:
        optimizer_kwargs['foreach'] = foreach
    if cfg.type == 'adam':
        return torch.optim.Adam(
            model.parameters(),
            lr=cfg.lr,
            weight_decay=cfg.weight_decay,
            betas=(cfg.beta1, cfg.beta2,),
            **optimizer_kwargs,
        )
    elif cfg.type == 'adamw':
        return torch.optim.AdamW(
            model.parameters(),
            lr=cfg.lr,
            weight_decay=cfg.weight_decay,
            betas=(cfg.beta1, cfg.beta2,),
            **optimizer_kwargs,
        )
    else:
        raise NotImplementedError(f'Optimizer not supported: {cfg.type}')


def get_scheduler(cfg, optimizer):
    if cfg is None:
        return None
    if cfg.type == 'plateau':
        return torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer,
            factor=cfg.factor,
            patience=cfg.patience,
            min_lr=cfg.min_lr,
        )
    elif cfg.type == 'step':
        return torch.optim.lr_scheduler.StepLR(
            optimizer,
            step_size=cfg.step_size,
            gamma=cfg.gamma,
        )
    elif cfg.type == 'multistep':
        return torch.optim.lr_scheduler.MultiStepLR(
            optimizer,
            milestones=cfg.milestones,
            gamma=cfg.gamma,
        )
    elif cfg.type == 'exp':
        return torch.optim.lr_scheduler.ExponentialLR(
            optimizer,
            gamma=cfg.gamma,
        )
    else:
        raise NotImplementedError(f'Scheduler not supported: {cfg.type}')


def count_parameters(model):
    return sum(p.numel() for p in model.parameters())





def plt2tensor(fig):
    buf = io.BytesIO()
    fig.savefig(buf, format='jpg')
    buf.seek(0)
    image = Image.open(buf)
    image = ToTensor()(image)
    plt.close()
    return image


def draw_2d(x, vmin=0, vmax=1, cmap='bwr'):
    # x: (nx, nt)
    plt.figure(figsize=(4, 4))
    plt.imshow(x.detach().cpu().numpy(), cmap=cmap, aspect='auto', vmin=vmin, vmax=vmax)
    plt.axis('off')
    return plt2tensor(plt.gcf())


def draw_3d(x, downsample=2, nrow=5, vmin=-2.5, vmax=2.5, cmap='bwr'):
    # x: (nx, ny, nt)
    if downsample > 1:
        idx = torch.tensor(list(range(0, x.size(-1), downsample)), device=x.device, dtype=torch.long)
        x = x[..., idx]
    ncol = x.size(-1) // nrow
    fig, axes = plt.subplots(nrow, ncol, figsize=(8, 8))
    fig.subplots_adjust(hspace=0., wspace=0.)
    for ax, im in zip(axes.flat, x.detach().cpu().permute(2, 0, 1)):
        ax.imshow(im, cmap=cmap, aspect='auto', vmin=vmin, vmax=vmax)
        ax.axis('off')
    return plt2tensor(plt.gcf())


def draw(x, **kwargs):
    if x.dim() == 2:
        return draw_2d(x, **kwargs)
    if x.dim() == 3:
        return draw_3d(x, **kwargs)
    raise ValueError(f'Unsupported dimension {x.dim()}')



def plot3d(points):
    x, y, z = points[:,0], points[:,1], points[:,2]

    fig = go.Figure(data=[go.Scatter3d(
        x=x, y=y, z=z,
        mode='markers',
        marker=dict(size=1, color=z, colorscale='Viridis', opacity=0.8)
    )])

    fig.update_layout(
        scene=dict(
            xaxis_title='X', yaxis_title='Y', zaxis_title='Z'
        ),
        title="3D Sphere Sampling"
    )
    # fig.show()
    return fig

def plot_traj3d(x: np.ndarray, title="3D Geodesic Trajectories"):
    """
    x: np.ndarray [B, T, 3] — B 条测地线轨迹，每条 T 个时间步
    """
    B, T, _ = x.shape
    fig = go.Figure()

    # 绘制每条轨迹
    for i in range(B):
        traj = x[i]  # [T, 3]
        color_vals = np.linspace(0, 1, T)  # 颜色映射 0~1 随时间
        
        fig.add_trace(go.Scatter3d(
            x=traj[:, 0],
            y=traj[:, 1],
            z=traj[:, 2],
            mode='markers+lines',
            marker=dict(
                size=1,
                color=color_vals,
                colorscale='Viridis',
                opacity=0.8,
                colorbar=dict(title="t (0→1)") if i == 0 else None
            ),
            line=dict(color='rgba(0,0,0,0.3)', width=2),
            name=f"Trajectory {i+1}"
        ))

    # 起点和终点（合并绘制提高可读性）
    fig.add_trace(go.Scatter3d(
        x=x[:, 0, 0], y=x[:, 0, 1], z=x[:, 0, 2],
        mode='markers',
        marker=dict(size=1, color='green', symbol='circle'),
        name='Start (t=0)'
    ))

    fig.add_trace(go.Scatter3d(
        x=x[:, -1, 0], y=x[:, -1, 1], z=x[:, -1, 2],
        mode='markers',
        marker=dict(size=1, color='red', symbol='circle'),
        name='End (t=1)'
    ))

    fig.update_layout(
        title=title,
        scene=dict(
            xaxis_title='X',
            yaxis_title='Y',
            zaxis_title='Z',
            aspectmode='data'
        ),
        legend=dict(
            x=0.02, y=0.98,
            bgcolor='rgba(255,255,255,0.6)'
        ),
        margin=dict(l=0, r=0, b=0, t=40)
    )

    # fig.show()
    return fig
