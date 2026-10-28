import torch
from math import *
from scipy.special import kv, gamma
from scipy.stats import binned_statistic
import numpy as np

device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
class SpectralNoiseGenerator(object):
    def __init__(self, dim: int = 1, device = device) -> None:
        self.dim = dim
        self.device = device

    def generate(self, 
                num_realizations: int=1, 
                L: float=1.0,
                N:int = 2**10 -1, 
                spectral_density: callable = None,
                seed: int=None, 
                normalize: bool=False):
        """
        Generate a realization of a Matérn Gaussian random field using spectral synthesis.

        Parameters:
            dim (int): Dimension of the field (1 or 2).
            num_realizations (int): Number of realizations.
            L (float): Domain size.
            N (int): Number of spatial discretization points.
            spectral_density (callable): Function to compute spectral density.
            seed (int): Random seed for reproducibility.
            normalize (bool): Whether to normalize the output to zero mean and unit variance.

        Returns:
            Tuple of frequency components, spectral amplitude H, spatial grid (x[, y]), 
            field realizations u, and noise W.
        """
        if self.dim == 1:
            return self.spectral_1DRF(num_realizations=num_realizations, L=L, N=N, spectral_density=spectral_density, seed=seed, normalize=normalize)
        elif self.dim == 2:
            return self.spectral_2DRF(num_realizations=num_realizations, L=L, N=N, spectral_density=spectral_density, seed=seed, normalize=normalize)
        else:
            raise ValueError(f"Dimension {self.dim} not supported")


    def spectral_1DRF(self, 
                      num_realizations: int=1, 
                      L: float=100.0,
                      N:int = 2**10 -1, 
                      spectral_density: callable = None,
                      seed: int=None, 
                      normalize: bool=False):
        
        h = L / N
        x = torch.linspace(0, L, N+1, device=self.device)
        freq = torch.fft.fftfreq(N+1, d=h).to(self.device) * (2 * pi)
        if seed is not None:
            generator = torch.Generator(device=self.device)
            generator.manual_seed(seed)
            W = torch.normal(mean=0.0, std=1.0/sqrt(h), size=(num_realizations, N+1), generator=generator, device=self.device)
        else:
            W = torch.normal(mean=0.0, std=1.0/sqrt(h), size=(num_realizations, N+1), device=self.device)

        W_hat = torch.fft.fft(W, dim=-1)
        H = spectral_density(freq)
        u = torch.fft.ifft(W_hat / H[None, :], dim=-1).real
        if normalize is True:
            # print(sigma2, torch.mean(u.var(dim = -1)))
            u = u - u.mean()
            u = u/u.std()
        return [freq], H, [x], u, W

    def spectral_2DRF(self,
                      N: int = 2**10 - 1,
                      num_realizations: int = 100,
                      L: float = 100.0,
                      spectral_density: callable = None,
                      seed: int = None, 
                      normalize: bool = False):
        h = L / N
        x = torch.linspace(0, L, N + 1, device=self.device)
        y = torch.linspace(0, L, N + 1, device=self.device)
        X, Y = torch.meshgrid(x, y, indexing='ij')
        freq = torch.fft.fftfreq(N + 1, d=h).to(self.device) * (2 * pi)
        Kx, Ky = torch.meshgrid(freq, freq, indexing='ij')
        if seed is not None:
            generator = torch.Generator(device=self.device)
            generator.manual_seed(seed)
            W = torch.normal(mean=0.0, std=1.0/h,
                            size=(num_realizations, N + 1, N + 1), 
                            generator=generator, device=self.device)
        else:
            W = torch.normal(mean=0.0, std=1.0/h,
                            size=(num_realizations, N + 1, N + 1),
                            device=self.device)
        W_hat = torch.fft.fft2(W)
        H = spectral_density(Kx, Ky)  # shape: (N+1, N+1)
        u = torch.fft.ifft2(W_hat / H[None, :, :]).real

        if normalize:
            u = u - u.mean()
            std_val = u.std()
            u = u / std_val
        return [Kx, Ky], H, [x, y], u, W

    def compute_covariance(self, data, L: float = None):
        N = data.shape[1]
        dh = L / (N - 1)
        if self.dim == 1:
            maxlag = N // 2 
            sample_cov = torch.empty(maxlag, device=device)
            for lag in range(maxlag):
                cov_lag = (data[:, :N-lag] * data[:, lag:]).mean()
                sample_cov[lag] = cov_lag
            lags = torch.arange(maxlag, device=device) * L / (N - 1)
            lags = lags.cpu().numpy()
            sample_cov = sample_cov.cpu().numpy()
            return lags, sample_cov
        elif self.dim == 2:
            z_hat = torch.fft.rfft2(data)
            acf = torch.fft.irfft2(torch.conj(z_hat) * z_hat)
            acf = torch.fft.fftshift(acf).mean(dim=0) / data[0].numel()
            acf_r = acf.view(-1).cpu().detach().numpy()
            lags_x, lags_y = torch.meshgrid(torch.arange(N) - N//2, torch.arange(N) - N//2)
            lags_r = torch.sqrt(lags_x**2 + lags_y**2).view(-1).cpu().detach().numpy()
            idx = np.argsort(lags_r)
            lags_r = lags_r[idx]
            acf_r = acf_r[idx]
            bin_means, bin_edges, binnumber = binned_statistic(lags_r, acf_r, 'mean', bins=np.linspace(0.0, N, N + 1))
            r = bin_edges * dh
            return r[:-1], bin_means
        else:
            raise ValueError(f"Dimension {self.dim} not supported")

class MaternFieldGenerator(SpectralNoiseGenerator):
    def __init__(self, dim: int = 1, 
                 L = 1.0,
                 res = 2**10 - 1,
                 nu = 1.5,
                 ell: float = 0.1,
                 device: torch.device = device) -> None:
        '''
        nu: smoothness parameter
        ell: related length scale
        '''
        super().__init__(dim, device)
        self.dim = dim
        self.device = device
        self.L = L
        self.res = res
        self.dh = L / res

        self.nu = nu
        self.ell = ell
        self.alpha = nu + dim/2
        self.kappa = sqrt(2*nu)/ell
        self.sigma2 = gamma(nu) / (gamma(self.alpha) * self.kappa ** (2 * nu) * (4 *pi)**(dim/2))

        self.name = f'Matern_{self.dim}D_nu{self.nu}_ell{self.ell:.2f}_res{self.res}_L{self.L:.2f}'

        if self.dim == 1:
            self.spectral_density = lambda freq: (self.kappa**2 + freq**2)**(self.alpha/2)
        elif self.dim == 2:
            self.spectral_density = lambda Kx, Ky: (self.kappa**2 + Kx**2 + Ky**2)**(self.alpha/2)
        else:
            raise ValueError(f"Dimension {self.dim} not supported")

    def generate(self,
                 num_realizations: int=1, 
                 seed: int=None, 
                 normalize: bool=False):
        return super().generate(num_realizations=num_realizations, L=self.L, N=self.res, spectral_density=self.spectral_density, seed=seed, normalize=normalize)
    
    def theoretical_covariance(self, r, normalize: bool = False):
        cov = np.empty_like(r)
        nonzero = r > 1e-10
        if normalize is True:
            sigma2 = 1
        else:
            sigma2 = self.sigma2
        cov[~nonzero] = sigma2
        cov[nonzero] = (sigma2 / (2**(self.nu-1)*gamma(self.nu))) * (self.kappa * r[nonzero])**self.nu * kv(self.nu, self.kappa * r[nonzero])
        return cov

class GaussianFieldGenerator(SpectralNoiseGenerator):
    def __init__(self, dim=1, alpha=0.005, L=1.0, res=2**10 - 1, device=device):
        super().__init__(dim, device)
        self.dim = dim
        self.device = device
        self.alpha = alpha
        self.L = L
        self.res = res
        self.dh = L / res

        self.sigma2 = 1/(4*pi * self.alpha)**(self.dim/2)

        self.name = f'Gaussian_{self.dim}D_alpha{self.alpha:.2f}_res{self.res}_L{self.L:.2f}'

        if self.dim == 1:
            self.spectral_density = lambda freq: torch.exp(self.alpha * freq**2/2)
        elif self.dim == 2:
            self.spectral_density = lambda Kx, Ky: torch.exp(self.alpha * (Kx**2 + Ky**2)/2)
        else:
            raise ValueError(f"Dimension {self.dim} not supported")

    def generate(self, num_realizations=1, seed=None, normalize=False):
        return super().generate(num_realizations=num_realizations, L=self.L, N=self.res, spectral_density=self.spectral_density, seed=seed, normalize=normalize)
    
    def theoretical_covariance(self, r, normalize: bool = False):
        cov = np.empty_like(r)
        nonzero = r > 1e-10
        if normalize is True:
            sigma2 = 1
        else:
            sigma2 = self.sigma2
        cov[~nonzero] = sigma2
        cov[nonzero] = sigma2 * np.exp(- r[nonzero]**2/(4 * self.alpha))
        return cov
    
class WienerProcessGenerator:
    def __init__(self, T=1.0, temporal_res=2**10 - 1, random_field_generator: SpectralNoiseGenerator = None, device=device):
        self.dim = random_field_generator.dim
        self.device = device
        self.T = T
        self.temporal_res = temporal_res
        self.dt = T / temporal_res
        self.random_field_generator = random_field_generator

    def Brownian_motion(self, num_realizations: int = 1, seed: int = None):
        t = torch.linspace(0, self.T, self.temporal_res+1, device=self.device)
        Wt = torch.zeros((num_realizations, self.temporal_res+1,), device=self.device)
        if seed is not None:
            generator = torch.Generator(device=self.device)
            generator.manual_seed(seed)
            dW = torch.normal(mean=0.0, std=self.dt**0.5, size=(num_realizations, self.temporal_res,), generator=generator, device=self.device)
        else:
            dW = torch.normal(mean=0.0, std=self.dt**0.5, size=(num_realizations, self.temporal_res,), device=self.device)
        Wt[:, 1:] = torch.cumsum(dW, dim=1)
        return t, Wt
    
    def generate(self, num_realizations: int = 1, seed: int = None, normalize: bool = True):
        t = torch.linspace(0, self.T, self.temporal_res+1, device=self.device)
        dwp_list = [];
        for i in range(self.temporal_res):
            _, _, coords, u, _ = self.random_field_generator.generate(num_realizations=num_realizations, seed=seed, normalize=normalize)
            u = u.unsqueeze(1) * sqrt(self.dt)
            dwp_list.append(u)

        dwp = torch.cat(dwp_list, dim=1)
        # print(dwp.shape)
        Q_wp = torch.cumsum(dwp, dim=1)
        zeros = torch.zeros_like(dwp[:, [0], ...])
        Q_wp = torch.cat([zeros, Q_wp], dim=1)
        return t, coords, dwp, Q_wp


class CylinderWhiteNoiseGenerator():
    def __init__(self, dim: int = 1, T: float = 1.0, temporal_res: int = 50, 
                 L: float = 1.0, spatial_res: int = 2**10 - 1, device = device) -> None:
        self.dim = dim
        self.device = device
        self.L, self.T = L, T
        self.tres, self.xres = temporal_res, spatial_res
        self.dt, self.dx = T / temporal_res, L / spatial_res
        self.x = torch.linspace(0, L, spatial_res + 1, device=device)
        self.basis = lambda j, x: sqrt(2 / L) * torch.sin(j * pi * x / L)

    def Brownian_motion(self, num_realizations: int = 1, seed: int = None):
        t = torch.linspace(0, self.T, self.tres+1, device=self.device)
        Wt = torch.zeros((num_realizations, self.tres+1,), device=self.device)
        if seed is not None:
            generator = torch.Generator(device=self.device)
            generator.manual_seed(seed)
            dW = torch.normal(mean=0.0, std=self.dt**0.5, size=(num_realizations, self.tres,), generator=generator, device=self.device)
        else:
            dW = torch.normal(mean=0.0, std=self.dt**0.5, size=(num_realizations, self.tres,), device=self.device)
        Wt[:, 1:] = torch.cumsum(dW, dim=1)
        return t, Wt
    def generate(self, J: int = 12, seed: int = None):
        t, Wt = self.Brownian_motion(num_realizations=J + 1, seed=seed)
        space_corr = torch.tensor([[self.basis(j, x) for j in range(J+1)] for x in self.x]).to(self.device)
        wxt = space_corr @ Wt
        return wxt