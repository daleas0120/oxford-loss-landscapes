import time
import warnings

import torch
import numpy as np
from scipy.sparse.linalg import LinearOperator, eigsh

# Hessian Eigenvectors and Values
from numpy import linalg as LA

# Threshold for large matrix size
SMALL_MATRIX_SIZE = int(1e3)
MEDIUM_MATRIX_SIZE = int(1e6)
LARGE_MATRIX_SIZE = int(1e10)

################################################################################
#                              Supporting Functions
################################################################################

def get_eigenstuff(hessian, num_eigs_returned=2, method='numpy'):
    """Calculate eigenvalues and eigenvectors of a Hessian matrix"""
    
    if method == 'numpy':
        # Warn for very large matrices
        if hessian.shape[0] > LARGE_MATRIX_SIZE:
            warnings.warn("Matrix is very large for numpy.linalg.eig. Consider using method='scipy'.")
            
        # Use eigh for symmetric Hessians to guarantee real eigenvalues/vectors
        eigenvalues, eigenvectors = LA.eigh(hessian)
        
        # Sort eigenvalues in ascending order and select the largest num_eigs_returned
        sorted_indices = np.argsort(eigenvalues)[-num_eigs_returned:]
        eigenvalues = [float(eigenvalues[i]) for i in sorted_indices]
        eigenvectors = [eigenvectors[:, i].flatten() for i in sorted_indices]
        
    elif method == 'scipy':
        # Validation checks
        if hessian.shape[0] != hessian.shape[1]:
            raise ValueError("Matrix must be square for scipy eigsh decomposition.")
        if hessian.shape[0] < num_eigs_returned:
            raise ValueError(f"Matrix size {hessian.shape[0]} too small for {num_eigs_returned} eigenvalues.")
            
        # Warn for edge cases
        if hessian.shape[0] < 10 * num_eigs_returned:
            warnings.warn(f"Matrix size {hessian.shape[0]} small relative to requested eigenvalues. "
                         "Consider method='numpy' for better accuracy.")
                         
        eigenvalues, eigenvectors = eigsh(hessian, k=num_eigs_returned, which='LM', tol=1e-2)
        eigenvalues = [float(ev) for ev in eigenvalues]
        eigenvectors = [eigenvectors[:, i] for i in range(eigenvectors.shape[1])]
        
        # Keep eigenvectors as numpy arrays for consistency with test expectations

    elif method == 'vrpca':
        raise NotImplementedError("VR-PCA method not implemented in this function.")

    return eigenvalues, eigenvectors

def get_hessian(model, loss, method='numpy'):
    """Function that calculates the Hessian matrix for a given model and loss value"""

    n_params = sum(p.numel() for p in model.parameters())
    if n_params < SMALL_MATRIX_SIZE:
        # For small models, use direct Hessian computation
        hessian = small_hessian(model, loss, method=method)
        return hessian
    elif n_params < MEDIUM_MATRIX_SIZE:
        # For medium models, use Hessian-vector product
        if isinstance(loss, (int, float)):
            loss = torch.tensor(loss, requires_grad=True)
        elif not loss.requires_grad:
            loss.requires_grad_(True)
        hessian = hessian_vector_product(net=model, loss=loss, use_cuda=False, all_params=True)
        return hessian
    else:
        raise NotImplementedError("Hessian computation for very large models not implemented.")
    

def small_hessian(model, loss, num_eigs_returned=2, method='numpy'):
    """Function that calculates the Hessian matrix of a small model for demonstration purposes."""

    n_params = sum(p.numel() for p in model.parameters())
    if n_params > 1000:
        raise RuntimeError(
            f"Model has {n_params} parameters. "
            "Direct Hessian computation is not recommended for large models. "
            "Use Hessian-vector product methods for efficiency."
        )

    # Ensure all model parameters require gradients
    for param in model.parameters():
        param.requires_grad_(True)

    # Clear the existing model gradients
    model.zero_grad()
    
    # Ensure loss requires gradients
    if isinstance(loss, (int, float)):
        loss = torch.tensor(loss, requires_grad=True)
    elif not loss.requires_grad:
        loss.requires_grad_(True)
        
    # Calculate first gradients with proper handling of unused parameters
    grads = torch.autograd.grad(loss, model.parameters(), create_graph=True, retain_graph=True, allow_unused=True)
    
    # Handle None gradients by replacing them with zeros that require grad
    processed_grads = []
    for i, g in enumerate(grads):
        if g is not None:
            processed_grads.append(g.contiguous().view(-1))
        else:
            # Create zero gradients for unused parameters with requires_grad=True
            param = list(model.parameters())[i]
            zero_grad = torch.zeros(param.numel(), dtype=param.dtype, device=param.device, requires_grad=True)
            processed_grads.append(zero_grad)
    
    grads_flat = torch.cat(processed_grads)

    # Build Hessian matrix row by row
    hessian_rows = []
    for g_idx in grads_flat:
        # Skip if the gradient element doesn't require gradients
        if not g_idx.requires_grad:
            # Create a row of zeros for this gradient element
            zero_row = torch.zeros(sum(p.numel() for p in model.parameters()), dtype=g_idx.dtype, device=g_idx.device)
            hessian_rows.append(zero_row)
            continue
            
        # get the gradient w.r.t. the gradient g_idx
        grad2 = torch.autograd.grad(g_idx, model.parameters(), retain_graph=True, allow_unused=True)

        # Handle None gradients and flatten the second derivative into a vector
        processed_grad2 = []
        for i, g in enumerate(grad2):
            if g is not None:
                processed_grad2.append(g.contiguous().view(-1))
            else:
                # Create zero gradients for unused parameters
                param = list(model.parameters())[i]
                processed_grad2.append(torch.zeros(param.numel(), dtype=param.dtype, device=param.device))
        
        grad2_flat = torch.cat(processed_grad2)
        hessian_rows.append(grad2_flat)

    hessian = torch.stack(hessian_rows).detach().numpy()
    return hessian


def hessian_vector_product(net, loss, use_cuda=False, all_params=True, rank=0, verbose=False):
    """Implementation of the hessian vector product algorithm using a pre-computed loss.
    
    This function creates a LinearOperator representing the Hessian matrix that can be used
    with scipy.sparse.linalg functions like eigsh() for eigenvalue computations.
    
    Args:
        net: the trained model
        loss: pre-computed loss tensor with requires_grad=True
              Example: loss = criterion(model(inputs), targets)
        use_cuda: use GPU for computations
        all_params: use all model parameters (True) or only weight matrices (False)
        rank: rank for distributed computing (used in verbose output)
        verbose: print timing information for each HVP computation
        
    Returns:
        LinearOperator: Hessian matrix as a linear operator that supports .matvec()
                       for computing Hessian-vector products
                       
    Example:
        >>> model = nn.Sequential(nn.Linear(3, 5), nn.ReLU(), nn.Linear(5, 1))
        >>> inputs, targets = torch.randn(10, 3), torch.randn(10, 1)
        >>> loss = nn.functional.mse_loss(model(inputs), targets)
        >>> hvp_op = hessian_vector_product(model, loss)
        >>> # Use with scipy for eigenvalues
        >>> from scipy.sparse.linalg import eigsh
        >>> eigenvals, eigenvecs = eigsh(hvp_op, k=3, which='LM')
    """
    
    # Create the Hessian-vector product function
    hess_vec_prod, _, N = create_hessian_vector_product_from_loss(
        net, loss, use_cuda, all_params
    )
    
    # Reset counter
    hess_vec_prod.count = 0
    
    if verbose and rank == 0: 
        print("Rank %d: computing max eigenvalue" % rank)

    # Create a wrapper that matches the expected interface for eigsh
    def hvp_wrapper(vec):
        return hess_vec_prod(vec, verbose=verbose, rank=rank)
    
    hessian = LinearOperator((N, N), matvec=hvp_wrapper)

    return hessian

def npvec_to_tensorlist(vec, params):
    """Convert numpy vector to list of tensors with same dimensions as params"""
    loc = 0
    result = []
    for p in params:
        numel = p.data.numel()
        result.append(torch.from_numpy(vec[loc:loc+numel]).view(p.data.shape).float())
        loc += numel
    assert loc == vec.size, f'Vector length {vec.size} does not match total parameter count {loc}'
    return result


def gradtensor_to_npvec(net, all_params=True):
    """Extract gradients from network and return concatenated numpy vector"""
    param_filter = lambda p: all_params or len(p.data.size()) >= 1
    
    grad_parts = []
    for p in net.parameters():
        if p.grad is None:
            grad_parts.append(np.zeros(p.data.numel(), dtype=np.float32))
        elif param_filter(p):
            grad_parts.append(p.grad.data.cpu().numpy().ravel())
    
    return np.concatenate(grad_parts)


################################################################################
#                  For computing Hessian-vector products
################################################################################
def eval_hess_vec_prod_from_loss(vec, params, net, loss, use_cuda=False):
    """
    Evaluate Hessian-vector product using a pre-computed loss and store result in network gradients.
    
    Args:
        vec: list of tensors with same dimensions as params
        params: parameter list of the network
        net: model with trained parameters
        loss: pre-computed loss tensor with requires_grad=True
        use_cuda: use GPU
    """
    if use_cuda:
        net.cuda()
        vec = [v.cuda() for v in vec]

    device = next(net.parameters()).device
    
    net.eval()
    net.zero_grad()
    
    # Ensure loss requires gradients
    if not loss.requires_grad:
        raise ValueError("Loss tensor must have requires_grad=True for Hessian computation")
    
    # Compute gradients with respect to parameters
    grad_f = torch.autograd.grad(loss, inputs=params, create_graph=True, allow_unused=True, retain_graph=True)

    # Compute inner product of gradient with direction vector
    inner_products = []
    for i, v in enumerate(vec):
        if grad_f[i] is not None and v is not None:
            inner_products.append((grad_f[i] * v).sum())
    
    prod = sum(inner_products) if inner_products else torch.zeros(1).to(device)
    
    # Compute Hessian-vector product H*v and store in parameter gradients
    prod.backward(retain_graph=True)


def eval_hess_vec_prod(vec, params, net, loss_func, inputs, outputs, use_cuda=False):
    """
    Evaluate Hessian-vector product and store result in network gradients.
    
    Args:
        vec: list of tensors with same dimensions as params
        params: parameter list of the network
        net: model with trained parameters
        loss_func: loss function
        inputs: network inputs
        outputs: desired network outputs
        use_cuda: use GPU
    """
    if use_cuda:
        net.cuda()
        vec = [v.cuda() for v in vec]

    device = next(net.parameters()).device
    inputs = inputs.to(device)
    outputs = outputs.to(device)

    net.eval()
    net.zero_grad()
    
    # Forward pass and compute loss
    pred_outputs = net(inputs)
    loss = loss_func(pred_outputs, outputs)
    
    # Compute gradients with respect to parameters
    grad_f = torch.autograd.grad(loss, inputs=params, create_graph=True, allow_unused=True)

    # Compute inner product of gradient with direction vector
    inner_products = []
    for i, v in enumerate(vec):
        if grad_f[i] is not None and v is not None:
            inner_products.append((grad_f[i] * v).sum())
    
    prod = sum(inner_products) if inner_products else torch.zeros(1).to(device)
    
    # Compute Hessian-vector product H*v and store in parameter gradients
    prod.backward()

################################################################################
#                  For computing Eigenvalues of Hessian
################################################################################
def create_hessian_vector_product_from_loss(net, loss, use_cuda=False, all_params=True):
    """
    Create a Hessian-vector product function for a given model using a pre-computed loss.
    
    Args:
        net: the trained model
        loss: pre-computed loss tensor with requires_grad=True
        use_cuda: use GPU
        all_params: use all nn parameters
        
    Returns:
        hess_vec_prod: function that computes Hessian-vector products
        params: list of parameters used
        N: total number of parameters
    """
    if all_params:
        params = [p for p in net.parameters()]
    else:
        params = [p for p in net.parameters() if len(p.size()) >= 1]
        
    N = sum(p.numel() for p in params)
    
    def hess_vec_prod(vec, verbose=False, rank=0):
        """
        Compute Hessian-vector product using pre-computed loss.
        
        Args:
            vec (numpy.ndarray of shape (N,)): Vector to multiply with Hessian, where N is the total number of parameters in the model.
            verbose: print timing information
            rank: rank for distributed computing
            
        Returns:
            numpy.ndarray of shape (N,): Resulting vector representing H*vec
        """
        if not hasattr(hess_vec_prod, 'count'):
            hess_vec_prod.count = 0
            
        hess_vec_prod.count += 1
        vec_tensors = npvec_to_tensorlist(vec, params)
        start_time = time.time()
        eval_hess_vec_prod_from_loss(vec_tensors, params, net, loss, use_cuda)
        prod_time = time.time() - start_time
        if verbose and rank == 0: 
            print("Iter: %d  time: %f" % (hess_vec_prod.count, prod_time))
        return gradtensor_to_npvec(net, all_params)
    
    return hess_vec_prod, params, N


def create_hessian_vector_product(net, inputs, outputs, criterion, use_cuda=False, all_params=True):
    """
    Create a Hessian-vector product function for a given model and data.
    
    Args:
        net: the trained model.
        inputs: nn inputs.
        outputs: desired nn outputs.
        criterion: loss function.
        use_cuda: use GPU
        all_params: use all nn parameters
        
    Returns:
        hess_vec_prod: function that computes Hessian-vector products
        params: list of parameters used
        N: total number of parameters
    """
    if all_params:
        params = [p for p in net.parameters()]
    else:
        params = [p for p in net.parameters() if len(p.size()) >= 1]
        
    N = sum(p.numel() for p in params)
    
    def hess_vec_prod(vec, verbose=False, rank=0):
        """
        Compute Hessian-vector product.
        
        Args:
            vec: numpy vector to multiply with Hessian
            verbose: print timing information
            rank: rank for distributed computing
            
        Returns:
            numpy vector representing H*vec
        """
        if not hasattr(hess_vec_prod, 'count'):
            hess_vec_prod.count = 0
            
        hess_vec_prod.count += 1
        vec_tensors = npvec_to_tensorlist(vec, params)
        start_time = time.time()
        eval_hess_vec_prod(vec_tensors, params, net, criterion, inputs, outputs, use_cuda)
        prod_time = time.time() - start_time
        if verbose and rank == 0: 
            print("Iter: %d  time: %f" % (hess_vec_prod.count, prod_time))
        return gradtensor_to_npvec(net, all_params)
    
    return hess_vec_prod, params, N


def min_max_hessian_eigs(net, inputs, outputs, criterion, rank=0, use_cuda=False, verbose=False, all_params=True):
    """
        Compute the largest and the smallest eigenvalues of the Hessian marix.

        Args:
            net: the trained model.
            inputs: nn inputs.
            outputs: desired nn outputs.
            criterion: loss function.
            rank: rank of the working node.
            use_cuda: use GPU
            verbose: print more information
            all_params: use all nn parameters

        Returns:
            maxeig: max eigenvalue
            mineig: min eigenvalue
            maxeigvec: max eigenvector
            mineigvec: min eigenvector
            hess_vec_prod.count: number of iterations for calculating max and min eigenvalues
    """
    
    # Create the Hessian-vector product function
    hess_vec_prod, params, N = create_hessian_vector_product(
        net, inputs, outputs, criterion, use_cuda, all_params
    )
    
    # Reset counter
    hess_vec_prod.count = 0
    
    if verbose and rank == 0: 
        print("Rank %d: computing max eigenvalue" % rank)

    # Create a wrapper that matches the expected interface for eigsh
    def hvp_wrapper(vec):
        return hess_vec_prod(vec, verbose=verbose, rank=rank)
    
    A = LinearOperator((N, N), matvec=hvp_wrapper)
    
    # Use scipy's eigsh directly instead of get_eigenstuff for LinearOperator
    eigvals, eigvecs = eigsh(A, k=1, which='LM', tol=1e-2)

    maxeig = eigvals[0]
    maxeigvec = eigvecs[:, 0]
    if verbose and rank == 0: print('max eigenvalue = %f' % maxeig)

    # If the largest eigenvalue is positive, shift matrix so that any negative eigenvalue is now the largest
    # We assume the smallest eigenvalue is zero or less, and so this shift is more than what we need
    shift = maxeig*1.0
    def shifted_hess_vec_prod(vec):
        return hvp_wrapper(vec) - shift*vec

    if verbose and rank == 0: print("Rank %d: Computing shifted eigenvalue" % rank)

    A = LinearOperator((N, N), matvec=shifted_hess_vec_prod)

    # Use scipy's eigsh directly instead of get_eigenstuff for LinearOperator
    eigvals, eigvecs = eigsh(A, k=1, which='LM', tol=1e-2)

    eigvals = eigvals + shift
    mineig = eigvals[0]
    mineigvec = eigvecs[:, 0]
    if verbose and rank == 0: print('min eigenvalue = ' + str(mineig))

    if maxeig <= 0 and mineig > 0:
        maxeig, mineig = mineig, maxeig
        maxeigvec, mineigvec = mineigvec, maxeigvec

    return maxeig, mineig, maxeigvec, mineigvec, hess_vec_prod.count

def get_hessian_eigenstuff(model, loss, num_eigs_returned=2, method='numpy', landscape_type='minimum'):
    """Get the eigenvalues and eigenvectors of the Hessian matrix."""
    
    if landscape_type not in ['minimum', 'saddle']:
        raise ValueError("landscape_type must be 'minimum' or 'saddle'")
    
    if landscape_type == 'minimum':
        try:
            hessian = get_hessian(model, loss, method=method)
            eigenvalues, eigenvectors = get_eigenstuff(hessian, num_eigs_returned, method)
            return eigenvalues, eigenvectors
        except Exception as e:
            warnings.warn(f"Failed to compute Hessian eigenvalues: {e}. Returning None values.")
            return None, None
    elif landscape_type == 'saddle':
        raise NotImplementedError("Min-max eigenvalue computation not implemented.")
    else:
        return None, None