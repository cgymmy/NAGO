import torch
from math import *
import scipy.stats as stats
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


class AnisotropicMaternFieldGenerator(SpectralNoiseGenerator):
    """Two-dimensional stationary Matérn field with geometric anisotropy.

    ``ell_x`` and ``ell_y`` are correlation-length parameters along the
    rotated principal axes. ``theta`` is the counter-clockwise rotation of the
    long/short-axis coordinate system in radians.  The parameterization
    reduces exactly to :class:`MaternFieldGenerator` when
    ``ell_x == ell_y``.
    """

    def __init__(
        self,
        L=1.0,
        res=2**10 - 1,
        nu=1.5,
        ell_x: float = 0.2,
        ell_y: float = 0.05,
        theta: float = 0.0,
        device: torch.device = device,
    ) -> None:
        super().__init__(dim=2, device=device)
        if nu <= 0:
            raise ValueError(f"nu must be positive, got {nu}")
        if ell_x <= 0 or ell_y <= 0:
            raise ValueError(
                f"ell_x and ell_y must be positive, got {ell_x}, {ell_y}"
            )

        self.dim = 2
        self.device = device
        self.L = L
        self.res = res
        self.dh = L / res
        self.nu = nu
        self.ell_x = float(ell_x)
        self.ell_y = float(ell_y)
        self.theta = float(theta)
        self.alpha = nu + self.dim / 2

        # The geometric-mean reference scale makes this operator identical to
        # the existing isotropic implementation when ell_x == ell_y.
        self.ell_ref = sqrt(self.ell_x * self.ell_y)
        self.kappa = sqrt(2 * nu) / self.ell_ref
        self.sigma2 = gamma(nu) / (
            gamma(self.alpha)
            * self.kappa ** (2 * nu)
            * (4 * pi) ** (self.dim / 2)
        )
        self._scale_x = self.ell_x / self.ell_ref
        self._scale_y = self.ell_y / self.ell_ref
        self._cos_theta = cos(self.theta)
        self._sin_theta = sin(self.theta)

        theta_deg = self.theta * 180.0 / pi
        self.name = (
            f"AnisotropicMatern_2D_nu{self.nu}_ellx{self.ell_x:.3f}"
            f"_elly{self.ell_y:.3f}_theta{theta_deg:.1f}"
            f"_res{self.res}_L{self.L:.2f}"
        )
        self.spectral_density = self._spectral_density

    def _rotated_frequencies(self, Kx, Ky):
        Kx_rot = self._cos_theta * Kx + self._sin_theta * Ky
        Ky_rot = -self._sin_theta * Kx + self._cos_theta * Ky
        return Kx_rot, Ky_rot

    def _spectral_density(self, Kx, Ky):
        Kx_rot, Ky_rot = self._rotated_frequencies(Kx, Ky)
        squared_radius = (
            (self._scale_x * Kx_rot) ** 2
            + (self._scale_y * Ky_rot) ** 2
        )
        return (self.kappa**2 + squared_radius) ** (self.alpha / 2)

    def generate(
        self,
        num_realizations: int = 1,
        seed: int = None,
        normalize: bool = False,
    ):
        return super().generate(
            num_realizations=num_realizations,
            L=self.L,
            N=self.res,
            spectral_density=self.spectral_density,
            seed=seed,
            normalize=normalize,
        )

    def anisotropic_radius(self, dx, dy):
        """Return the equivalent isotropic distance for spatial displacements."""
        dx_rot = self._cos_theta * dx + self._sin_theta * dy
        dy_rot = -self._sin_theta * dx + self._cos_theta * dy
        return self.ell_ref * np.sqrt(
            (dx_rot / self.ell_x) ** 2 + (dy_rot / self.ell_y) ** 2
        )

    def theoretical_covariance(self, dx, dy, normalize: bool = False):
        r = np.asarray(self.anisotropic_radius(dx, dy))
        cov = np.empty_like(r, dtype=float)
        nonzero = r > 1e-10
        sigma2 = 1.0 if normalize else self.sigma2
        cov[~nonzero] = sigma2
        scaled_r = self.kappa * r[nonzero]
        cov[nonzero] = (
            sigma2
            / (2 ** (self.nu - 1) * gamma(self.nu))
            * scaled_r**self.nu
            * kv(self.nu, scaled_r)
        )
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


class WhiteNoiseGenerator(SpectralNoiseGenerator):
    def __init__(self, dim: int = 1, 
                 L = 1.0,
                 res = 2**10 - 1,
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

        self.name = f'White noise'

        if self.dim == 1:
            self.spectral_density = lambda freq: torch.ones_like(freq)
        elif self.dim == 2:
            self.spectral_density = lambda Kx, Ky: torch.ones_like(Kx)
        else:
            raise ValueError(f"Dimension {self.dim} not supported")

    def generate(self,
                 num_realizations: int=1, 
                 seed: int=None,
                 normalize: bool=False):
        return super().generate(num_realizations=num_realizations, L=self.L, N=self.res, spectral_density=self.spectral_density, seed=seed, normalize=normalize)

    # def generate(self,
    #             num_realizations: int=1, 
    #             seed: int=None, 
    #             normalize: bool=False):
    #     noise = torch.randn(num_realizations, self.res + 1)
    #     sample = torch.fft.ifft(noise, norm='forward')
    #     sample = sample.real
    #     return sample

    

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

def compute_covariance(data, L: float = None):
    N = data.shape[1]
    dh = L / (N - 1)
    dim = len(data.shape) - 1
    if dim == 1:
        maxlag = N // 2 
        sample_cov = torch.empty(maxlag, device=device)
        for lag in range(maxlag):
            cov_lag = (data[:, :N-lag] * data[:, lag:]).mean()
            sample_cov[lag] = cov_lag
        lags = torch.arange(maxlag, device=device) * L / (N - 1)
        lags = lags.cpu().numpy()
        sample_cov = sample_cov.cpu().numpy()
        return lags, sample_cov
    elif dim == 2:
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
        raise ValueError(f"Dimension {dim} not supported")


def get_Fourier_spectrum(data):
    '''
    data: (B, H, W)
    '''
    data_hat = torch.fft.fftn(data,dim=(1,2),norm = "forward")
    fourier_amplitudes = np.abs(data_hat)**2
    fourier_amplitudes = fourier_amplitudes.mean(dim=0)
    npix = data_hat.shape[-1]
    kfreq = np.fft.fftfreq(npix) * npix
    kfreq2D = np.meshgrid(kfreq, kfreq)
    knrm = np.sqrt(kfreq2D[0]**2 + kfreq2D[1]**2)
    knrm = knrm.flatten()
    fourier_amplitudes = fourier_amplitudes.flatten()
    
    kbins = np.arange(0.5, npix//2+1, 1.)
    kvals = 0.5 * (kbins[1:] + kbins[:-1])
    Abins_w, _, _ = stats.binned_statistic(knrm, fourier_amplitudes,
                                     statistic = "mean",
                                     bins = kbins)
    # Abins_w *= np.pi * (kbins[1:]**2 - kbins[:-1]**2)
    
    return kvals, Abins_w

def get_energy_spectrum1d(vorticity_trajectory):
    vorticity_hat = torch.fft.fft(vorticity_trajectory,dim=(1),norm = "forward")
    fourier_amplitudes = np.abs(vorticity_hat)**2 
    fourier_amplitudes = fourier_amplitudes.mean(dim=0)
    npix = vorticity_hat.shape[-1]
    kfreq = np.fft.fftfreq(npix) * npix
    knrm = np.sqrt(kfreq**2)
    knrm = knrm.flatten()
    fourier_amplitudes = fourier_amplitudes.flatten()
    
    kvals = knrm
    Abins_w = fourier_amplitudes
    
    return kvals, Abins_w

def log_mse(pred, target, eps=1e-12):
    '''return numpy array'''
    if isinstance(pred, torch.Tensor):
        pred = pred.detach().cpu().numpy()
    if isinstance(target, torch.Tensor):
        target = target.detach().cpu().numpy()

    pred  = np.clip(pred,  eps, None)
    target = np.clip(target, eps, None)

    diff = np.log(pred) - np.log(target)
    return np.mean(diff**2)
