# Modifications for PCFM © 2025 Pengfei Cai (Learning Matter @ MIT) and Utkarsh (Julia Lab @ MIT), licensed under the MIT License.
# Original portions © Amazon.com, Inc. or its affiliates, licensed under the Apache License 2.0.

from __future__ import annotations

import torch
import torch.nn as nn

from torchdiffeq import odeint
from tqdm import tqdm
from .random_field import *


class GPPrior(MaternFieldGenerator):
    def __init__(self, dim: int = 1, L=1, res=2 ** 10, nu=1.5, ell: float = 0.1, device: torch.device = 'cpu') -> None:
        super().__init__(dim, L, res - 1, nu, ell, device)
    
    def sample(self, n_samples: int, normalize: bool = True):
        _, _, _, u, _ = self.generate(num_realizations=n_samples, normalize=normalize)
        return u

class WNPrior(WhiteNoiseGenerator):
    def __init__(self, dim: int = 1, L=1, res=2 ** 10, device: torch.device = 'cpu') -> None:
        super().__init__(dim, L, res - 1, device)
    
    def sample(self, n_samples: int, normalize: bool = False):
        _, _, _, u, _ = self.generate(num_realizations=n_samples, normalize=normalize)
        return u

class GNPrior():
    def __init__(self, dim: int = 2, res: int=64, device: torch.device=None) -> None:
        self.res = res
        self.dim = dim
        self.device = device
    def sample(self, n_samples: int):
        if self.dim == 1:
            return torch.randn(n_samples, self.res, device=self.device)
        elif self.dim == 2:
            return torch.randn(n_samples, self.res, self.res, device=self.device)


def make_grid(dims: tuple[int], device='cpu', start: float | tuple[float] = 0., end: float | tuple[float] = 1.):
    ndim = len(dims)
    if not isinstance(start, (tuple, list)):
        start = [start] * ndim
    if not isinstance(end, (tuple, list)):
        end = [end] * ndim
    if ndim == 1:
        return torch.linspace(start[0], end[0], dims[0], dtype=torch.float, device=device).unsqueeze(-1)
    xs = torch.meshgrid([
        torch.linspace(start[i], end[i], dims[i], dtype=torch.float, device=device)
        for i in range(ndim)
    ], indexing='ij')
    grid = torch.stack(xs, dim=-1).view(-1, ndim)
    return grid


class interpolation():
    def __init__(self, type: str='linear'):
        if type.lower() == 'linear':
            self.alpha = lambda t: 1 - t
            self.beta = lambda t: t
            self.dalpha = lambda t: -1   
            self.dbeta = lambda t: 1
        else:
            pass

class scale_intp(interpolation):
    def __init__(self, type: str='scale', ratio: float=1.0):
        super().__init__(type)
        self.alpha = lambda t: torch.sqrt((ratio - ratio**t)/(ratio - 1))
        self.beta = lambda t: torch.sqrt((ratio**t - 1)/(ratio - 1))
        self.dalpha = lambda t: -0.5*1.0/self.alpha(t)*(ratio**t)*log(ratio)/(ratio - 1)
        self.dbeta = lambda t: 0.5*1.0/self.beta(t)*(ratio**t)*log(ratio)/(ratio - 1)


class FFM(nn.Module):
    """
    Functional Flow Matching (FFM) model for generative modeling of PDE solutions.
    """

    def __init__(self, model, noise = None, intp = interpolation()):
        super().__init__()
        self.model = model
        self.gp = noise
        ##
        self.alpha = intp.alpha
        self.beta = intp.beta
        self.dalpha = intp.dalpha
        self.dbeta = intp.dbeta

    def forward(self, t, xt):
        pred = self.model(t, xt)
        return pred

    def simulate(self, t, x):
        batch_size, c, *dim = x.size()
        noise = self.gp.sample(n_samples=batch_size * c).reshape(batch_size, c, *dim)
        t = t[(slice(None),) + (None,) * (x.dim() - 1)]
        return self.alpha(t) * noise + self.beta(t) * x, self.dalpha(t) * noise + self.dbeta(t) * x

    def get_loss(self, x):
        t = torch.rand(x.size(0), dtype=torch.float, device=x.device)
        xt, vf = self.simulate(t, x)
        pred_vf = self(t, xt)
        loss = (pred_vf - vf).pow(2).mean()
        return loss

    @torch.no_grad()
    def sample(
        self, n_sample, n_eval, dims, device, method='rk4',
        rtol=1e-5, atol=1e-5, return_traj=False,
        progress=False, progress_desc='ODE sampling',
    ):
        """
        Sample from the FFM model by integrating the GP prior with the learned vector field.
        :param n_sample: number of samples to generate
        :param n_eval: number of evaluation steps
        :param dims: dimensions of the generated grid
        :param device: device to run on
        :param rtol: relative tolerance for ODE solver
        :param atol: absolute tolerance for ODE solver
        :param return_traj: whether to return the full trajectory
        :return: generated samples of shape (n_sample, *dims) or trajectory of shape (n_eval + 1, n_sample, *dims)
        """
        c = dims[0]
        noise = self.gp.sample(n_samples=n_sample * c).reshape(n_sample, *dims)
        
        ts = torch.linspace(0, 1, n_eval + 1, device=device)
        if progress:
            states = [noise] if return_traj else None
            current = noise
            for i in tqdm(range(n_eval), total=n_eval, desc=progress_desc, unit='step'):
                current = odeint(
                    self.model, current, ts[i:i + 2],
                    method=method, rtol=rtol, atol=atol,
                )[-1]
                if return_traj:
                    states.append(current)
            xs = torch.stack(states) if return_traj else current
        else:
            xs = odeint(self.model, noise, ts, method=method, rtol=rtol, atol=atol)
        if return_traj:
            return xs.detach()
        return xs.detach() if progress else xs[-1].detach()
        
    # @torch.no_grad()
    def tweedie(self, xt, t):
        '''[N,c,H,W]'''
        # batch_size, c, *dim = xt.size()
        vt = self.model(t, xt)
        return (self.beta(t) - self.alpha(t) / self.dalpha(t) * self.dbeta(t))**(-1) * (xt - self.alpha(t) / self.dalpha(t) * vt)

    @torch.no_grad()
    def sample_one_step(self, xt, t, tp1, device, method='rk4', rtol=0.00001, atol=0.00001):
        [n_sample, c, res, res] = xt.shape
        ts = torch.tensor([t, tp1], device=device)
        return odeint(self.model, xt, ts, method=method, rtol=rtol, atol=atol)[-1]

    def sample_with_guide(self, loss_fn: callable, n_sample, n_eval, dims, device, 
    method='rk4', rtol=0.00001, atol=0.00001, save_path=None):
        c = dims[0]
        ts = torch.linspace(0, 1, n_eval + 1, device=device)
        xt = self.gp.sample(n_samples=n_sample * c).reshape(n_sample, *dims)
        loss_list = []
        pbar = tqdm(range(n_eval))
        for i in pbar:
            xt = xt.detach().requires_grad_(True)
            hat_xt = self.tweedie(xt, ts[i])
            loss = loss_fn(hat_xt[:, 0, ...], ts[i])
            grad_J = torch.autograd.grad(outputs=loss, inputs=xt)[0]
            xt = xt - 1.0 * grad_J
            pbar.set_description(f'loss: {loss.item()/n_sample, grad_J.norm().item()}')
            loss_list.append(loss.item()/n_sample)
            xt = self.sample_one_step(xt, ts[i], ts[i+1], device, method, rtol, atol)
        if save_path is not None:
            loss_tensor = torch.tensor(loss_list)
            torch.save(loss_tensor, save_path)

        return xt.detach()
    

class FFM_guide(nn.Module):
    def __init__(self, model, loss_fn):
        super().__init__()
        intp = interpolation()
        self.alpha = intp.alpha
        self.beta = intp.beta
        self.dalpha = intp.dalpha
        self.dbeta = intp.dbeta

        self.model = model
        self.loss_fn = loss_fn
    
    def forward(self, t, x):
        x = x.requires_grad_(True)
        hat_x = self.tweedie(x, t)
        loss = self.loss_fn(hat_x[:, 0, ...], t)
        grad_J = torch.autograd.grad(outputs=loss, inputs=x, create_graph=False, retain_graph=False)[0]
        with torch.no_grad():
            vt = self.model(t, x - grad_J)
        return vt - grad_J
    
    # @torch.no_grad()
    def tweedie(self, xt, t):
        '''[N, c, H, W]'''
        vt = self.model(t, xt)
        return (self.beta(t) - self.alpha(t) / self.dalpha(t) * self.dbeta(t))**(-1) * (xt - self.alpha(t) / self.dalpha(t) * vt)


            




class SFFM(FFM):
    def __init__(self, model, noise = None, intp = interpolation()):
        super(SFFM, self).__init__(model, noise, intp)
    
    def sample(self, n_sample, n_eval, dims, device, method='rk4', rtol=0.00001, atol=0.00001, return_traj=False):
        def rhs(t, xt):
            batch_size, c, *dims = xt.size()
            if t.dim() == 0 or t.numel() == 1:
                t = torch.ones(batch_size, device=t.device) * t
            bt = (self.dalpha(t)/self.alpha(t))[:,None,None]*xt
            coef = self.dbeta(t) - self.dalpha(t)*self.beta(t)/self.alpha(t)
            coef = coef[:,None,None]
            orig_t = 1/(1+self.alpha(t)/self.beta(t)) # s
            orig_x = orig_t[:,None,None]/self.beta(t)[:,None,None]*xt
            orig_bt = self.model(orig_t, orig_x) 
            bt += coef*((1-orig_t[:,None,None])*orig_bt + orig_x)
            return bt.detach()
        c = dims[0]
        noise = self.gp.sample(n_samples=n_sample * c).reshape(n_sample, *dims)
        
        ts = torch.linspace(1e-4, 1-1e-4, n_eval + 1, device=device)
        xs = odeint(rhs, noise, ts, method=method, rtol=rtol, atol=atol)
        if return_traj:
            return xs.detach()
        return xs[-1].detach()
