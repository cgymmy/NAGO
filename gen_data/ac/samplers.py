import numpy as np

def side_move(log_prob, initial, n_samples, n_walkers=20, gamma=1.6869, normal_std=1.0, n_thin = 1):
    """
    The "side move" algorithm.
    
    Parameters:
    -----------
    log_prob : function
        Log probability density function that accepts and returns arrays
    initial : array
        Initial state (will be used as mean for initializing walkers)
    n_samples : int
        Number of samples to draw per walker
    n_walkers : int
        Number of walkers in the ensemble (must be even and >= 4)
    gamma : float
        Differential evolution scale parameter
    normal_std : float
        Standard deviation of normal noise multiplier
    n_thin : int
        Thinning factor - store every n_thin sample (default: 1, no thinning)
    
    Returns:
    --------
    samples : array
        Samples from all walkers (shape: n_walkers, n_samples, dim)
    acceptance_rates : array
        Acceptance rates for all walkers
    """
    # Ensure even number of walkers and at least 4
    if n_walkers < 4:
        n_walkers = 4
    if n_walkers % 2 != 0:
        n_walkers += 1
    
    dim = len(initial)
    half_walkers = n_walkers // 2
    
    # Scale gamma by dimension (optimal scaling for Gaussian targets)
    gamma_scale = gamma / np.sqrt(dim)
    
    # Initialize walkers with small random perturbations around initial
    walkers = np.tile(initial, (n_walkers, 1)) + 0.1 * np.random.randn(n_walkers, dim)
    
    # Vectorized evaluation of initial log probabilities
    walker_log_probs = log_prob(walkers)
    
    # Calculate total iterations needed based on thinning factor
    total_iterations = n_samples * n_thin
    
    # Storage for samples and tracking acceptance
    samples = np.zeros((n_walkers, n_samples, dim))
    accepts = np.zeros(n_walkers)
    
    # Sample index to track where to store thinned samples
    sample_idx = 0
    
    # Main sampling loop
    for i in range(total_iterations):
        
        # Store current state from all walkers (only every n_thin iterations)
        if i % n_thin == 0 and sample_idx < n_samples:
            samples[:, sample_idx] = walkers
            sample_idx += 1
            
        # # Store current state from all walkers
        # samples[:, i] = walkers
        
        # Update first half using second half
        # Create arrays of indices for the first and second walker sets
        first_half_indices = np.arange(half_walkers)
        second_half_indices = np.arange(half_walkers, n_walkers)
        
        # Random indices for r1 walkers in second half
        r1_indices_first_update = np.random.choice(second_half_indices, size=half_walkers, replace=False)
        
        # Create a shuffled copy of second half indices for r2 that guarantees r1 != r2
        r2_pool = second_half_indices.copy()
        np.random.shuffle(r2_pool)
        r2_indices_first_update = np.array([
            r2_pool[idx] if r2_pool[idx] != r1 else r2_pool[(idx + 1) % len(r2_pool)]
            for idx, r1 in enumerate(r1_indices_first_update)
        ])
        
        # Generate normal noise for all proposals at once
        normal_noise_first = np.random.normal(0, normal_std, size=(half_walkers, 1))
        
        # Compute difference vectors and proposals in a vectorized way
        difference_vectors_first = walkers[r1_indices_first_update] - walkers[r2_indices_first_update]
        proposals_first = walkers[:half_walkers] + gamma_scale * difference_vectors_first * normal_noise_first
        
        # Evaluate all proposals at once
        proposal_log_probs_first = log_prob(proposals_first)
        
        # Calculate acceptance probabilities
        log_accept_probs_first = proposal_log_probs_first - walker_log_probs[:half_walkers]
        
        # Generate random numbers for acceptance
        random_uniforms_first = np.log(np.random.uniform(size=half_walkers))
        
        # Determine accepted proposals
        accepted_first = random_uniforms_first < log_accept_probs_first
        
        # Update walkers and log probs where accepted
        walkers[:half_walkers][accepted_first] = proposals_first[accepted_first]
        walker_log_probs[:half_walkers][accepted_first] = proposal_log_probs_first[accepted_first]
        
        # Track acceptance for all walkers in first half
        accepts[:half_walkers][accepted_first] += 1
        
        # Update second half using updated first half
        # Random indices for r1 walkers in first half
        r1_indices_second_update = np.random.choice(first_half_indices, size=half_walkers, replace=False)
        
        # Create a shuffled copy of first half indices for r2 that guarantees r1 != r2
        r2_pool = first_half_indices.copy()
        np.random.shuffle(r2_pool)
        r2_indices_second_update = np.array([
            r2_pool[idx] if r2_pool[idx] != r1 else r2_pool[(idx + 1) % len(r2_pool)]
            for idx, r1 in enumerate(r1_indices_second_update)
        ])
        
        # Generate normal noise for all proposals at once
        normal_noise_second = np.random.normal(0, normal_std, size=(half_walkers, 1))
        
        # Compute difference vectors and proposals in a vectorized way
        difference_vectors_second = walkers[r1_indices_second_update] - walkers[r2_indices_second_update]
        proposals_second = walkers[half_walkers:] + gamma_scale * difference_vectors_second * normal_noise_second
        
        # Evaluate all proposals at once
        proposal_log_probs_second = log_prob(proposals_second)
        
        # Calculate acceptance probabilities
        log_accept_probs_second = proposal_log_probs_second - walker_log_probs[half_walkers:]
        
        # Generate random numbers for acceptance
        random_uniforms_second = np.log(np.random.uniform(size=half_walkers))
        
        # Determine accepted proposals
        accepted_second = random_uniforms_second < log_accept_probs_second
        
        # Update walkers and log probs where accepted
        walkers[half_walkers:][accepted_second] = proposals_second[accepted_second]
        walker_log_probs[half_walkers:][accepted_second] = proposal_log_probs_second[accepted_second]
        
        # Track acceptance for all walkers in second half
        accepts[half_walkers:][accepted_second] += 1
    
    # Return results from all walkers
    acceptance_rates = accepts / total_iterations
    
    return samples, acceptance_rates

def stretch_move(log_prob, initial, n_samples, n_walkers=20, a=2.0, n_thin = 1):
    """
    Vectorized implementation of the Ensemble MCMC with stretch moves.
    
    Parameters:
    -----------
    log_prob : function
        Log probability density function that accepts array of shape (n_walkers, dim)
        and returns array of shape (n_walkers,)
    initial : array
        Initial state (will be used as mean for initializing walkers)
    n_samples : int
        Number of samples to draw per walker
    n_walkers : int
        Number of walkers in the ensemble (must be even)
    a : float
        Stretch parameter (controls proposal scale)
    n_thin : int
        Thinning factor - store every n_thin sample (default: 1, no thinning)
        
    Returns:
    --------
    samples : array
        Samples from all walkers (shape: n_walkers, n_samples, dim)
    acceptance_rates : array
        Acceptance rates for all walkers
    """
    # Ensure even number of walkers
    if n_walkers % 2 != 0:
        n_walkers += 1
        
    dim = len(initial)
    half_walkers = n_walkers // 2
    
    # Initialize walkers with small random perturbations around initial
    walkers = np.tile(initial, (n_walkers, 1)) + 0.1 * np.random.randn(n_walkers, dim)
    
    # Vectorized evaluation of initial log probabilities
    walker_log_probs = log_prob(walkers)
    
    # Calculate total iterations needed based on thinning factor
    total_iterations = n_samples * n_thin
    
    # Storage for samples and tracking acceptance
    samples = np.zeros((n_walkers, n_samples, dim))
    accepts = np.zeros(n_walkers)
    
    # Sample index to track where to store thinned samples
    sample_idx = 0
    
    # Main sampling loop
    for i in range(total_iterations):
        # Store current state from all walkers (only every n_thin iterations)
        if i % n_thin == 0 and sample_idx < n_samples:
            samples[:, sample_idx] = walkers
            sample_idx += 1
        
        # Update first half, then second half
        for half in [0, 1]:
            # Set indices for active and complementary walker sets
            active_indices = np.arange(half * half_walkers, (half + 1) * half_walkers)
            comp_indices = np.arange((1 - half) * half_walkers, (2 - half) * half_walkers)
            
            # Randomly select complementary walkers for each active walker
            selected_comp_indices = np.random.choice(comp_indices, size=half_walkers, replace=True)
            
            # Generate stretch factors (z) for all active walkers at once
            z_factors = ((a - 1.0) * np.random.uniform(size=half_walkers) + 1.0) ** 2.0 / a
            
            # Extract active and selected complementary walkers
            active_walkers = walkers[active_indices]
            comp_walkers = walkers[selected_comp_indices]
            
            # Vectorized proposal generation for all active walkers at once
            # z * (Xactive - Xcomp) + Xcomp = Xcomp + z * (Xactive - Xcomp)
            proposals = comp_walkers + (active_walkers - comp_walkers) * z_factors[:, np.newaxis]
            
            # Evaluate all proposals at once
            proposal_log_probs = log_prob(proposals)
            
            # Calculate log acceptance probabilities
            log_accept_probs = (dim - 1) * np.log(z_factors) + proposal_log_probs - walker_log_probs[active_indices]
            
            # Generate random numbers for acceptance decisions
            random_uniforms = np.log(np.random.uniform(size=half_walkers))
            
            # Determine which proposals are accepted
            accepted = random_uniforms < log_accept_probs
            
            # Update walkers and log probabilities in one step
            walkers[active_indices[accepted]] = proposals[accepted]
            walker_log_probs[active_indices[accepted]] = proposal_log_probs[accepted]
            
            # Track acceptance for all walkers
            accepts[active_indices[accepted]] += 1
    
    # Return results from all walkers
    acceptance_rates = accepts / total_iterations
    return samples, acceptance_rates

def hmc(log_prob, initial, n_samples, grad_fn, epsilon=0.1, L=10, n_chains=1, n_thin = 1):
    """
    Vectorized Hamiltonian Monte Carlo implementation.
    Processes all chains simultaneously for improved efficiency.
    
    n_thin : int
        Thinning factor - store every n_thin sample (default: 1, no thinning)
    """
    
    dim = len(initial)
    
    # Initialize multiple chains with small random perturbations around initial
    chains = np.tile(initial, (n_chains, 1)) + 0.1 * np.random.randn(n_chains, dim)
    
    # Vectorized evaluation of initial log probabilities
    chain_log_probs = log_prob(chains)
    
    # Calculate total iterations needed based on thinning factor
    total_iterations = 1 + (n_samples - 1) * n_thin  # +1 for initial state
    
    # Storage for samples and tracking acceptance
    samples = np.zeros((n_chains, n_samples, dim))
    accepts = np.zeros(n_chains)
    
    # Store initial state
    samples[:, 0, :] = chains
    
    # Sample index to track where to store thinned samples (start at 1 since we stored initial state)
    sample_idx = 1
    
    # Main sampling loop
    for i in range(1, total_iterations):
        # Generate momentum variables for all chains at once
        p = np.random.normal(size=(n_chains, dim))
        current_p = p.copy()
        
        # Leapfrog integration (vectorized over all chains)
        x = chains.copy()
        
        # Get gradients for all chains
        x_grad = grad_fn(x)
        # Handle NaN values safely
        x_grad = np.nan_to_num(x_grad, nan=0.0)
        
        # Half step for momentum
        p -= 0.5 * epsilon * x_grad
        
        # Full steps for position and momentum
        # L_steps = np.random.randint(L//2, 2*L)
        L_steps = L
        for j in range(L_steps):
            # Full step for position
            x += epsilon * p
            
            if j < L_steps - 1:
                # Get gradients for all chains
                x_grad = grad_fn(x)
                # Handle NaN values safely
                x_grad = np.nan_to_num(x_grad, nan=0.0)
                
                # Full step for momentum
                p -= epsilon * x_grad
        
        # Get final gradients for all chains
        x_grad = grad_fn(x)
        # Handle NaN values safely
        x_grad = np.nan_to_num(x_grad, nan=0.0)
        
        # Half step for momentum
        p -= 0.5 * epsilon * x_grad
        
        # Flip momentum for reversibility
        p = -p
        
        # Metropolis acceptance (vectorized)
        proposal_log_probs = log_prob(x)
        
        # Compute log acceptance ratio directly in log space - avoiding exp
        # current_H = -chain_log_probs + 0.5 * sum(current_p²)
        # proposal_H = -proposal_log_probs + 0.5 * sum(p²)
        # log_accept_prob = min(0, current_H - proposal_H)
        current_K = 0.5 * np.sum(current_p**2, axis=1)
        proposal_K = 0.5 * np.sum(p**2, axis=1)
        
        # Calculate log acceptance probability directly to avoid overflow
        log_accept_prob = np.minimum(0, proposal_log_probs - chain_log_probs - proposal_K + current_K)
        
        # Generate uniform random numbers for acceptance decision
        log_u = np.log(np.random.uniform(size=n_chains))
        
        # Create mask for accepted proposals
        accept_mask = log_u < log_accept_prob
        
        # Update chains and log probabilities where accepted
        chains[accept_mask] = x[accept_mask]
        chain_log_probs[accept_mask] = proposal_log_probs[accept_mask]
        
        # Track acceptances
        accepts += accept_mask
        
        # Store current state for all chains (only every n_thin iterations after the first)
        if (i - 1) % n_thin == 0 and sample_idx < n_samples:
            samples[:, sample_idx, :] = chains
            sample_idx += 1
    
    # Calculate acceptance rates for all chains
    acceptance_rates = accepts / (total_iterations - 1)
    
    return samples, acceptance_rates

def hamiltonian_side_move(gradient_func, potential_func, initial, n_samples, n_chains_per_group=5, 
                       epsilon=0.01, n_leapfrog=10, beta=1.0, n_thin = 1):
    """
    Vectorized Ensemble Hamiltonian Side Move sampler.
    Each particle randomly selects one particle from the complementary group for preconditioning.

    n_thin : int
        Thinning factor - store every n_thin sample (default: 1, no thinning)
    """
    
    # Initialize
    orig_dim = initial.shape
    flat_dim = np.prod(orig_dim)
    total_chains = 2 * n_chains_per_group
    
    # Create initial states with small random perturbations
    states = np.tile(initial.flatten(), (total_chains, 1)) + 0.1 * np.random.randn(total_chains, flat_dim)
    
    # Split into two groups
    group1_indices = np.arange(n_chains_per_group)
    group2_indices = np.arange(n_chains_per_group, total_chains)
    
    # Calculate total iterations needed based on thinning factor
    total_iterations = n_samples * n_thin
    
    # Storage for samples and acceptance tracking
    samples = np.zeros((total_chains, n_samples, flat_dim))
    accepts = np.zeros(total_chains)
    
    # Sample index to track where to store thinned samples
    sample_idx = 0
    
    # Precompute some constants for efficiency
    beta_eps = beta * epsilon
    beta_eps_half = beta_eps / 2
    
    # Main sampling loop
    for i in range(total_iterations):
        # Store current state from all chains (only every n_thin iterations)
        if i % n_thin == 0 and sample_idx < n_samples:
            samples[:, sample_idx] = states
            sample_idx += 1
        
        #---------------------------------------------
        # First group update - VECTORIZED
        #---------------------------------------------
        
        # For each particle in group 1, randomly select TWO particles from group 2
        random_indices1_from_group2 = np.random.choice(group2_indices, size=n_chains_per_group)
        random_indices2_from_group2 = np.random.choice(group2_indices, size=n_chains_per_group)
        
        # Ensure the two selected particles are different
        for j in range(n_chains_per_group):
            while random_indices1_from_group2[j] == random_indices2_from_group2[j]:
                random_indices2_from_group2[j] = np.random.choice(group2_indices)
        
        # Get the two sets of selected particles from group 2 (shape: n_chains_per_group x flat_dim)
        selected_particles1_group2 = states[random_indices1_from_group2]
        selected_particles2_group2 = states[random_indices2_from_group2]
        
        # Use the difference between the two particles (shape: n_chains_per_group x flat_dim)
        diff_particles_group2 = (selected_particles1_group2 - selected_particles2_group2) / np.sqrt(2*n_chains_per_group)
        
        # Generate momentum - one scalar per chain (shape: n_chains_per_group)
        p1 = np.random.randn(n_chains_per_group)
        
        # Store current state and energy
        current_q1 = states[group1_indices].copy()
        current_q1_reshaped = current_q1.reshape(n_chains_per_group, *orig_dim)
        current_U1 = potential_func(current_q1_reshaped)

        current_K1 = np.clip(0.5 * p1**2, 0, 1000)
        
        # Leapfrog integration with preconditioning
        q1 = current_q1.copy()
        p1_current = p1.copy()
        
        # Initial half-step for momentum - VECTORIZED
        grad1 = gradient_func(q1.reshape(n_chains_per_group, *orig_dim)).reshape(n_chains_per_group, -1)
        # Handle NaNs safely
        grad1 = np.nan_to_num(grad1, nan=0.0)
        
        # Compute dot products between gradients and difference particles - VECTORIZED
        # This gives one scalar per chain (shape: n_chains_per_group)
        gradient_projections = np.sum(grad1 * diff_particles_group2, axis=1)
        p1_current -= beta_eps_half * gradient_projections
        
        # Full leapfrog steps
        # n_steps = np.random.randint(n_leapfrog//2, 2*n_leapfrog)
        n_steps = n_leapfrog
        for step in range(n_steps):
            # Full position step - VECTORIZED with broadcasting
            # For each chain j, we're doing: q1[j] += beta_eps * p1_current[j] * diff_particles_group2[j]
            q1 += beta_eps * (p1_current[:, np.newaxis] * diff_particles_group2)
            
            if step < n_steps - 1:
                # Full momentum step (except at last iteration) - VECTORIZED
                grad1 = gradient_func(q1.reshape(n_chains_per_group, *orig_dim)).reshape(n_chains_per_group, -1)
                # Handle NaNs safely
                grad1 = np.nan_to_num(grad1, nan=0.0)
                
                gradient_projections = np.sum(grad1 * diff_particles_group2, axis=1)
                p1_current -= beta_eps * gradient_projections
        
        # Final half-step for momentum - VECTORIZED
        grad1 = gradient_func(q1.reshape(n_chains_per_group, *orig_dim)).reshape(n_chains_per_group, -1)
        # Handle NaNs safely
        grad1 = np.nan_to_num(grad1, nan=0.0)
        
        gradient_projections = np.sum(grad1 * diff_particles_group2, axis=1)
        p1_current -= beta_eps_half * gradient_projections
        
        # Compute proposed energy
        proposed_U1 = potential_func(q1.reshape(n_chains_per_group, *orig_dim))
        proposed_K1 = np.clip(0.5 *p1_current**2, 0, 1000)
        

        # Metropolis acceptance - IMPROVED FOR NUMERICAL STABILITY
        dH1 = (proposed_U1 + proposed_K1) - (current_U1 + current_K1)
        
        # Calculate acceptance probabilities with numerical safeguards for exp()
        # Instead of: accept_probs1 = np.minimum(1.0, np.exp(-dH1))
        accept_probs1 = np.ones_like(dH1)  # Default to accept
        # Only calculate exp for positive dH (negative cases auto-accept with prob 1.0)
        exp_needed = dH1 > 0
        if np.any(exp_needed):
            # Clip extremely high values before exponentiating
            safe_dH = np.clip(dH1[exp_needed], None, 100)  # No lower bound, upper bound of 100
            accept_probs1[exp_needed] = np.exp(-safe_dH)
        
        accepts1 = np.random.random(n_chains_per_group) < accept_probs1
        
        # Update states - VECTORIZED
        states[group1_indices[accepts1]] = q1[accepts1]
        # Track acceptance for first chains
        accepts[group1_indices] += accepts1
        
        #---------------------------------------------
        # Second group update - VECTORIZED similarly
        #---------------------------------------------
        
        # For each particle in group 2, randomly select TWO particles from group 1
        random_indices1_from_group1 = np.random.choice(group1_indices, size=n_chains_per_group)
        random_indices2_from_group1 = np.random.choice(group1_indices, size=n_chains_per_group)
        
        # Ensure the two selected particles are different
        for j in range(n_chains_per_group):
            while random_indices1_from_group1[j] == random_indices2_from_group1[j]:
                random_indices2_from_group1[j] = np.random.choice(group1_indices)
        
        # Get the two sets of selected particles from group 1 (shape: n_chains_per_group x flat_dim)
        selected_particles1_group1 = states[random_indices1_from_group1]
        selected_particles2_group1 = states[random_indices2_from_group1]
        
        # Use the difference between the two particles (shape: n_chains_per_group x flat_dim)
        diff_particles_group1 = (selected_particles1_group1 - selected_particles2_group1) / np.sqrt(2*n_chains_per_group)
        
        # Generate momentum - one scalar per chain (shape: n_chains_per_group)
        p2 = np.random.randn(n_chains_per_group)
        
        # Store current state and energy
        current_q2 = states[group2_indices].copy()
        current_q2_reshaped = current_q2.reshape(n_chains_per_group, *orig_dim)
        current_U2 = potential_func(current_q2_reshaped)
        current_K2 = np.clip(0.5 *p2**2, 0, 1000)
        
        # Leapfrog integration with preconditioning
        q2 = current_q2.copy()
        p2_current = p2.copy()
        
        # Initial half-step for momentum - VECTORIZED
        grad2 = gradient_func(q2.reshape(n_chains_per_group, *orig_dim)).reshape(n_chains_per_group, -1)
        # Handle NaNs safely
        grad2 = np.nan_to_num(grad2, nan=0.0)
        
        gradient_projections = np.sum(grad2 * diff_particles_group1, axis=1)
        p2_current -= beta_eps_half * gradient_projections
        
        # Full leapfrog steps
        n_steps = n_leapfrog
        for step in range(n_steps):
            # Full position step - VECTORIZED with broadcasting
            q2 += beta_eps * (p2_current[:, np.newaxis] * diff_particles_group1)
            
            if step < n_steps - 1:
                # Full momentum step (except at last iteration) - VECTORIZED
                grad2 = gradient_func(q2.reshape(n_chains_per_group, *orig_dim)).reshape(n_chains_per_group, -1)
                # Handle NaNs safely
                grad2 = np.nan_to_num(grad2, nan=0.0)
                
                gradient_projections = np.sum(grad2 * diff_particles_group1, axis=1)
                p2_current -= beta_eps * gradient_projections
        
        # Final half-step for momentum - VECTORIZED
        grad2 = gradient_func(q2.reshape(n_chains_per_group, *orig_dim)).reshape(n_chains_per_group, -1)
        # Handle NaNs safely
        grad2 = np.nan_to_num(grad2, nan=0.0)
        
        gradient_projections = np.sum(grad2 * diff_particles_group1, axis=1)
        p2_current -= beta_eps_half * gradient_projections
        
        # Compute proposed energy
        proposed_U2 = potential_func(q2.reshape(n_chains_per_group, *orig_dim))
        proposed_K2 = np.clip(0.5 *p2_current**2, 0, 1000)
        
        # Metropolis acceptance - IMPROVED FOR NUMERICAL STABILITY
        dH2 = (proposed_U2 + proposed_K2) - (current_U2 + current_K2)
        
        # Calculate acceptance probabilities with numerical safeguards for exp()
        accept_probs2 = np.ones_like(dH2)  # Default to accept
        # Only calculate exp for positive dH (negative cases auto-accept with prob 1.0)
        exp_needed = dH2 > 0
        if np.any(exp_needed):
            # Clip extremely high values before exponentiating
            safe_dH = np.clip(dH2[exp_needed], None, 100)  # No lower bound, upper bound of 100
            accept_probs2[exp_needed] = np.exp(-safe_dH)
        
        accepts2 = np.random.random(n_chains_per_group) < accept_probs2
        
        # Update states - VECTORIZED
        states[group2_indices[accepts2]] = q2[accepts2]
        
        # Track acceptance for second chains
        accepts[group2_indices] += accepts2
    
    # Reshape final samples to original dimensions
    samples = samples.reshape((total_chains, n_samples) + orig_dim)
    
    # Compute acceptance rates for all chains
    acceptance_rates = accepts / total_iterations
    
    return samples, acceptance_rates

def hamiltonian_walk_move(gradient_func, potential_func, initial, n_samples, n_chains_per_group=5, 
                       epsilon=0.01, n_leapfrog=10, beta=0.05, n_thin = 1):
    """
    Vectorized Hamiltonian Walk Move sampler.
    n_thin : int
        Thinning factor - store every n_thin sample (default: 1, no thinning)
    """
    
    # Initialize
    orig_dim = initial.shape
    flat_dim = np.prod(orig_dim)
    total_chains = 2 * n_chains_per_group
    
    # Create initial states with small random perturbations
    states = np.tile(initial.flatten(), (total_chains, 1)) + 0.1 * np.random.randn(total_chains, flat_dim)
    
    # Split into two groups
    group1 = slice(0, n_chains_per_group)
    group2 = slice(n_chains_per_group, total_chains)
    
    # Calculate total iterations needed based on thinning factor
    total_iterations = n_samples * n_thin
    
    # Storage for samples and acceptance tracking
    samples = np.zeros((total_chains, n_samples, flat_dim))
    accepts = np.zeros(total_chains)
    
    # Sample index to track where to store thinned samples
    sample_idx = 0
    
    # Precompute some constants for efficiency
    beta_eps = beta * epsilon
    beta_eps_half = beta_eps / 2
    
    # Main sampling loop
    for i in range(total_iterations):
        # Store current state from all chains (only every n_thin iterations)
        if i % n_thin == 0 and sample_idx < n_samples:
            samples[:, sample_idx] = states
            sample_idx += 1
        
        # Compute centered ensembles for preconditioning
        centered2 = (states[group2] - np.mean(states[group2], axis=0)) / np.sqrt(n_chains_per_group)
        
        # First group update
        # Generate momentum - fully vectorized with correct dimensions
        p1 = np.random.randn(n_chains_per_group, n_chains_per_group)
        
        # Store current state and energy
        current_q1 = states[group1].copy()
        current_q1_reshaped = current_q1.reshape(n_chains_per_group, *orig_dim)
        current_U1 = potential_func(current_q1_reshaped)
        current_K1 =  np.clip(0.5 *np.sum(p1**2, axis=1), 0, 1000)
        
        # Leapfrog integration with preconditioning
        q1 = current_q1.copy()
        p1_current = p1.copy()
        
        # Initial half-step for momentum - vectorized
        grad1 = gradient_func(q1.reshape(n_chains_per_group, *orig_dim)).reshape(n_chains_per_group, -1)
        # Handle NaNs safely
        grad1 = np.nan_to_num(grad1, nan=0.0)
        
        # Matrix multiplication for projection - fully vectorized
        p1_current -= beta_eps_half * np.dot(grad1, centered2.T)
        
        # Full leapfrog steps
        # n_steps = np.random.randint(n_leapfrog//2, 2*n_leapfrog)
        n_steps = n_leapfrog
        for step in range(n_steps):
            # Full position step - vectorized matrix multiplication
            q1 += beta_eps * np.dot(p1_current, centered2)
            
            if step < n_steps - 1:
                # Full momentum step (except at last iteration) - vectorized
                grad1 = gradient_func(q1.reshape(n_chains_per_group, *orig_dim)).reshape(n_chains_per_group, -1)
                # Handle NaNs safely
                grad1 = np.nan_to_num(grad1, nan=0.0)
                
                p1_current -= beta_eps * np.dot(grad1, centered2.T)
        
        # Final half-step for momentum - vectorized
        grad1 = gradient_func(q1.reshape(n_chains_per_group, *orig_dim)).reshape(n_chains_per_group, -1)
        # Handle NaNs safely
        grad1 = np.nan_to_num(grad1, nan=0.0)
        
        p1_current -= beta_eps_half * np.dot(grad1, centered2.T)
        
        # Compute proposed energy
        proposed_U1 = potential_func(q1.reshape(n_chains_per_group, *orig_dim))
        proposed_K1 = np.clip(0.5 *np.sum(p1_current**2, axis=1), 0, 1000)
        
        # Metropolis acceptance - IMPROVED FOR NUMERICAL STABILITY
        dH1 = (proposed_U1 + proposed_K1) - (current_U1 + current_K1)
        
        # Calculate acceptance probabilities with numerical safeguards for exp()
        accept_probs1 = np.ones_like(dH1)  # Default to accept
        # Only calculate exp for positive dH (negative cases auto-accept with prob 1.0)
        exp_needed = dH1 > 0
        if np.any(exp_needed):
            # Clip extremely high values before exponentiating
            safe_dH = np.clip(dH1[exp_needed], None, 100)  # No lower bound, upper bound of 100
            accept_probs1[exp_needed] = np.exp(-safe_dH)
        
        accepts1 = np.random.random(n_chains_per_group) < accept_probs1
        
        # Update states - vectorized
        states[group1][accepts1] = q1[accepts1]
        accepts[group1] += accepts1

        # Second group update - vectorized the same way
        centered1 = (states[group1] - np.mean(states[group1], axis=0)) / np.sqrt(n_chains_per_group)
        
        p2 = np.random.randn(n_chains_per_group, n_chains_per_group)
        
        current_q2 = states[group2].copy()
        current_q2_reshaped = current_q2.reshape(n_chains_per_group, *orig_dim)
        current_U2 = potential_func(current_q2_reshaped)
        current_K2 = np.clip(0.5 *np.sum(p2**2, axis=1), 0, 1000)
        
        q2 = current_q2.copy()
        p2_current = p2.copy()
        
        # Initial half-step for momentum - vectorized
        grad2 = gradient_func(q2.reshape(n_chains_per_group, *orig_dim)).reshape(n_chains_per_group, -1)
        # Handle NaNs safely
        grad2 = np.nan_to_num(grad2, nan=0.0)
        
        p2_current -= beta_eps_half * np.dot(grad2, centered1.T)
        
        # Full leapfrog steps
        n_steps = n_leapfrog
        for step in range(n_steps):
            # Full position step - vectorized
            q2 += beta_eps * np.dot(p2_current, centered1)
            
            if step < n_steps - 1:
                # Full momentum step (except at last iteration) - vectorized
                grad2 = gradient_func(q2.reshape(n_chains_per_group, *orig_dim)).reshape(n_chains_per_group, -1)
                # Handle NaNs safely
                grad2 = np.nan_to_num(grad2, nan=0.0)
                
                p2_current -= beta_eps * np.dot(grad2, centered1.T)
        
        # Final half-step for momentum - vectorized
        grad2 = gradient_func(q2.reshape(n_chains_per_group, *orig_dim)).reshape(n_chains_per_group, -1)
        # Handle NaNs safely
        grad2 = np.nan_to_num(grad2, nan=0.0)
        
        p2_current -= beta_eps_half * np.dot(grad2, centered1.T)
        
        # Compute proposed energy
        proposed_U2 = potential_func(q2.reshape(n_chains_per_group, *orig_dim))
        proposed_K2 = np.clip(0.5 *np.sum(p2_current**2, axis=1), 0, 1000)
        
        # Metropolis acceptance - IMPROVED FOR NUMERICAL STABILITY
        dH2 = (proposed_U2 + proposed_K2) - (current_U2 + current_K2)
        
        # Calculate acceptance probabilities with numerical safeguards for exp()
        accept_probs2 = np.ones_like(dH2)  # Default to accept
        # Only calculate exp for positive dH (negative cases auto-accept with prob 1.0)
        exp_needed = dH2 > 0
        if np.any(exp_needed):
            # Clip extremely high values before exponentiating
            safe_dH = np.clip(dH2[exp_needed], None, 100)  # No lower bound, upper bound of 100
            accept_probs2[exp_needed] = np.exp(-safe_dH)
        
        accepts2 = np.random.random(n_chains_per_group) < accept_probs2
        
        # Update states
        states[group2][accepts2] = q2[accepts2]
        
        # Track acceptance for second chains
        accepts[group2] += accepts2
    
    # Reshape final samples to original dimensions
    samples = samples.reshape((total_chains, n_samples) + orig_dim)
    
    # Compute acceptance rates for all chains
    acceptance_rates = accepts / total_iterations
    
    return samples, acceptance_rates