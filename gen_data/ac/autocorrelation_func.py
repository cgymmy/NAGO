
import numpy as np

def autocorrelation_fft(x, max_lag=None):

    """
    Efficiently compute autocorrelation function using FFT.
    
    Parameters:
    -----------
    x : array
        1D array of samples
    max_lag : int, optional
        Maximum lag to compute (default: len(x)//3)
        
    Returns:
    --------
    acf : array
        Autocorrelation function values
    """
    n = len(x)
    if max_lag is None:
        max_lag = min(n // 3, 20000)  # Cap at 20000 to prevent slow computation
    
    # Remove mean and normalize
    x_norm = x - np.mean(x)
    var = np.var(x_norm)
    x_norm = x_norm / np.sqrt(var)
    
    # Compute autocorrelation using FFT
    # Pad the signal with zeros to avoid circular correlation
    fft = np.fft.fft(x_norm, n=2*n)
    acf = np.fft.ifft(fft * np.conjugate(fft))[:n]
    acf = acf.real / n  # Normalize
    
    return acf[:max_lag]

def integrated_autocorr_time(x, M=5, c=10):
    """
    Estimate the integrated autocorrelation time using a self-consistent window.
    Based on the algorithm described by Goodman and Weare.
    
    Parameters:
    -----------
    x : array
        1D array of samples
    M : int, default=5
        Window size multiplier (typically 5-10)
    c : int, default=10
        Maximum lag cutoff for window determination
        
    Returns:
    --------
    tau : float
        Integrated autocorrelation time
    acf : array
        Autocorrelation function values
    ess : float
        Effective sample size
    """
    n = len(x)
    orig_x = x.copy()
    
    # Initial pairwise reduction if needed
    k = 0
    max_iterations = 10  # Prevent infinite loop
    
    while k < max_iterations:
        # Calculate autocorrelation function
        acf = autocorrelation_fft(x)
        
        # Calculate integrated autocorrelation time with self-consistent window
        tau = 1.0  # Initialize with the first term
        
        # Find the window size where window <= M * tau
        for window in range(1, len(acf)):
            # Update tau with this window
            tau_window = 1.0 + 2.0 * sum(acf[1:window+1])
            
            # Check window consistency: window <= M*tau
            if window <= M * tau_window:
                tau = tau_window
            else:
                break
        
        # If we have a robust estimate, we're done
        if n >= c * tau:
            # Scale tau back to the original time scale: tau_0 = 2^k * tau_k
            tau = tau * (2**k)
            break
            
        # If we don't have a robust estimate, perform pairwise reduction
        k += 1
        n_half = len(x) // 2
        x_new = np.zeros(n_half)
        for i in range(n_half):
            if 2*i + 1 < len(x):
                x_new[i] = 0.5 * (x[2*i] + x[2*i+1])
            else:
                x_new[i] = x[2*i]
        x = x_new
        n = len(x)
    
    # If we exited without a robust estimate, compute one final estimate
    if k >= max_iterations or n < c * tau:
        acf = autocorrelation_fft(orig_x)
        tau_reduced = 1.0 + 2.0 * sum(acf[1:min(len(acf), int(M)+1)])
        # Scale tau back to the original time scale
        tau = tau_reduced * (2**k)
    
    # Calculate effective sample size using original series length
    ess = len(orig_x) / tau
    
    return tau, acf, ess
