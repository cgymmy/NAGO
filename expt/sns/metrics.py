"""Numerical SNS spectrum metrics; no plotting dependencies."""

import torch
import numpy as np
import scipy.stats as stats

def get_energy_spectrum(vorticity_trajectory):
    vorticity_hat = torch.fft.fftn(vorticity_trajectory,dim=(1,2),norm = "forward")
    fourier_amplitudes = np.abs(vorticity_hat)**2 * (2*np.pi)
    fourier_amplitudes = fourier_amplitudes.mean(dim=0)
    npix = vorticity_hat.shape[-1]
    kfreq = np.fft.fftfreq(npix) * npix
    kfreq2D = np.meshgrid(kfreq, kfreq)
    knrm = np.sqrt(kfreq2D[0]**2 + kfreq2D[1]**2)
    knrm = knrm.flatten()
    fourier_amplitudes = fourier_amplitudes.flatten()
    
    laplace = knrm ** 2
    laplace[0] = 1.0
    vhat_square = (fourier_amplitudes)/laplace
    
    kbins = np.arange(0.5, npix//2+1, 1.)
    kvals = 0.5 * (kbins[1:] + kbins[:-1])
    Abins_w, _, _ = stats.binned_statistic(knrm, fourier_amplitudes,
                                     statistic = "mean",
                                     bins = kbins)
    # Abins_w *= np.pi * (kbins[1:]**2 - kbins[:-1]**2)
    
    Abins_v, _, _ = stats.binned_statistic(knrm, vhat_square,
                                     statistic = "mean",
                                     bins = kbins)
    # Abins_v *= np.pi * (kbins[1:]**2 - kbins[:-1]**2)
    return kvals, Abins_w, Abins_v
