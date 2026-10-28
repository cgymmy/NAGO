import torch
import math
import matplotlib.pyplot as plt
import matplotlib
import tqdm
from gen_data.src.random_field import *
from timeit import default_timer
import scipy.io

def get_dW(a, dt, res, batch_size):
  dW = torch.randn(batch_size, 8) * sqrt(dt)
  mesh = torch.linspace(0, a[0], res)
  X, Y = torch.meshgrid(mesh, mesh, indexing='ij')
  force = lambda x, y: torch.stack([torch.sin(6*x), torch.cos(6*x),
                      torch.cos(7 * x), torch.sin(7 * x),
                      torch.sin(5*(x+y)), torch.cos(5*(x+y)),
                      torch.cos(8*(x+y)), torch.sin(8*(x+y))], dim=0)
  return torch.einsum("bk, knm -> bnm", dW, force(X, Y))
  

#a: domain where we are solving
#w0: initial vorticity
#f: deterministic forcing term 
#visc: viscosity (1/Re)
#T: final time
#delta_t: internal time-step for solve (descrease if blow-up)
#record_steps: number of in-time snapshots to record
#sigma: coefficient of the random noise
#gen_force: generator of the random noise

def navier_stokes_2d(a, w0, visc=1e-3, alpha=0.1, T=1.0, dt=1e-1, sigma: float = 0.05):

    #Grid size - must be power of 2

    delta_t = 1e-4
    record_steps = int(dt / delta_t)
    num_t = int(T/dt) + 1
    N1, N2 = w0.size()[-2], w0.size()[-1] 

    #Maximum frequency
    k_max1 = math.floor(N1/2.0)
    k_max2 = math.floor(N1/2.0)

    #Number of steps to final time
    steps = math.ceil(T/delta_t)

    #Initial vorticity to Fourier space
    w_h = torch.fft.fftn(w0, dim=[1,2])
    w_h = torch.stack([w_h.real, w_h.imag],dim=-1)


    #Wavenumbers in y-direction
    k_y = torch.cat((torch.arange(start=0, end=k_max2, step=1, device=w0.device), torch.arange(start=-k_max2, end=0, step=1, device=w0.device)), 0).repeat(N1,1)
    #Wavenumbers in x-direction
    k_x = torch.cat((torch.arange(start=0, end=k_max1, step=1, device=w0.device), torch.arange(start=-k_max1, end=0, step=1, device=w0.device)), 0).repeat(N2,1).transpose(0,1)
    #Negative Laplacian in Fourier space
    lap = 4*(math.pi**2)*(k_x**2/a[0]**2 + k_y**2/a[1]**2)
    # lap_ = lap.clone()
    lap[0,0] = 1.0
    #Dealiasing mask
    dealias = torch.unsqueeze(torch.logical_and(torch.abs(k_y) <= (2.0/3.0)*k_max2, torch.abs(k_x) <= (2.0/3.0)*k_max1).float(), 0)

    #Saving solution and time
    sol = torch.zeros(*w0.size(), num_t, device=w0.device)
    batch_size = w0.size(0)

    #Record counter
    c = 0
    #Physical time
    t = 0.0
    f_idx = -1
    w = w0

    for j in tqdm.trange(steps + 1):
        if j % record_steps == 0:
          # print(j // record_steps)
          sol[...,c] = w
          c += 1
        if j == steps:
          break
        #Stream function in Fourier space: solve Poisson equation
        psi_h = w_h.clone()
        psi_h[...,0] = psi_h[...,0]/lap
        psi_h[...,1] = psi_h[...,1]/lap

        #Velocity field in x-direction = psi_y
        q = psi_h.clone()
        temp = q[...,0].clone()
        q[...,0] = -2*math.pi*k_y*q[...,1]
        q[...,1] = 2*math.pi*k_y*temp
        q = torch.fft.ifftn(torch.view_as_complex(q/a[1]), dim=[1,2], s=(N1,N2)).real

        #Velocity field in y-direction = -psi_x
        v = psi_h.clone()
        temp = v[...,0].clone()
        v[...,0] = 2*math.pi*k_x*v[...,1]
        v[...,1] = -2*math.pi*k_x*temp
        v = torch.fft.ifftn(torch.view_as_complex(v/a[0]), dim=[1,2], s=(N1,N2)).real

        #Partial x of vorticity
        w_x = w_h.clone()
        temp = w_x[...,0].clone()
        w_x[...,0] = -2*math.pi*k_x*w_x[...,1]
        w_x[...,1] = 2*math.pi*k_x*temp
        w_x = torch.fft.ifftn(torch.view_as_complex(w_x/a[0]), dim=[1,2], s=(N1,N2)).real

        #Partial y of vorticity
        w_y = w_h.clone()
        temp = w_y[...,0].clone()
        w_y[...,0] = -2*math.pi*k_y*w_y[...,1]
        w_y[...,1] = 2*math.pi*k_y*temp
        w_y = torch.fft.ifftn(torch.view_as_complex(w_y/a[1]), dim=[1,2], s=(N1,N2)).real

        #Non-linear term (u.grad(w)): compute in physical space then back to Fourier space
        F_h = torch.fft.fftn(q*w_x + v*w_y, dim=[1,2])
        F_h = torch.stack([F_h.real, F_h.imag],dim=-1) 

        #Dealias
        F_h[...,0] = dealias* F_h[...,0]
        F_h[...,1] = dealias* F_h[...,1]



          
        #Cranck-Nicholson update
        # if j % record_steps == 0:
        #   f_idx += 1
        #   dW = sigma * noise_f[:, f_idx, :, :] * math.sqrt(delta_t)
        dW = get_dW(a, delta_t, N1, batch_size).to(w0.device)
        gudWh = torch.fft.fft2(dW , dim=[-2,-1])
        gudWh = torch.stack([gudWh.real, gudWh.imag],dim=-1) 


        w_h[...,0] = (-delta_t*F_h[...,0] - alpha * delta_t*w_h[...,0] + gudWh[...,0] + (1.0 - 0.5*delta_t*visc*lap)*w_h[...,0])/(1.0 + 0.5*delta_t*visc*lap)
        w_h[...,1] = (-delta_t*F_h[...,1] - alpha * delta_t*w_h[...,1] + gudWh[...,1] + (1.0 - 0.5*delta_t*visc*lap)*w_h[...,1])/(1.0 + 0.5*delta_t*visc*lap)

        #Update real time (used only for recording)
        t += delta_t
        w = torch.fft.ifftn(torch.view_as_complex(w_h), dim=[1,2], s=(N1,N2)).real

        # if (j+1) % record_steps == 0:
        #     #Solution in physical space
        #     w = torch.fft.ifftn(torch.view_as_complex(w_h), dim=[1,2], s=(N1,N2)).real
        #     #Record solution and time
        #     sol[...,c] = w
        #     c += 1
    return sol
