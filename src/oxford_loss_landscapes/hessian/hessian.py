import torch
import time
import numpy as np
from torch import nn
from torch.autograd import Variable
from scipy.sparse.linalg import LinearOperator, eigsh
from typing import Optional

try:
    # Optional import: available when VR-PCA is installed within this package
    from .vrpca import VRPCAConfig, min_hessian_eigenpair_vrpca, top_hessian_eigenpair_vrpca
except Exception:  # pragma: no cover - keep classical path usable even if VRPCA missing
    VRPCAConfig = None  # type: ignore[assignment]
    top_hessian_eigenpair_vrpca = None  # type: ignore[assignment]
    min_hessian_eigenpair_vrpca = None  # type: ignore[assignment]

################################################################################
#                              Supporting Functions
################################################################################
def npvec_to_tensorlist(vec, params):
    """ Convert a numpy vector to a list of tensor with the same dimensions as params

        Args:
            vec: a 1D numpy vector
            params: a list of parameters from net

        Returns:
            rval: a list of tensors with the same shape as params
    """
    loc = 0
    rval = []
    for p in params:
        numel = p.data.numel()
        rval.append(torch.from_numpy(vec[loc:loc+numel]).view(p.data.shape).float())
        loc += numel
    assert loc == vec.size, f'ERROR: The vector has a {loc} elements and the net has {vec.size} parameters'
    return rval


def gradtensor_to_npvec(net, all_params=True):
    """ Extract gradients from net, and return a concatenated numpy vector.

        Args:
            net: trained model
            all_params: If all_params, then gradients w.r.t. BN parameters and bias
            values are also included. Otherwise only gradients with dim > 1 are considered.

        Returns:
            a concatenated numpy vector containing all gradients
    """
    param_filter = lambda p: all_params or len(p.data.size()) >= 1

    # tmp_list = [p.grad.data.cpu().numpy().ravel() for p in net.parameters() if filter(p)]
    tmp_list = []
    for p in net.parameters():
        if p.grad is None:
            tmp_list.append(np.zeros(p.data.numel(), dtype=np.float32))
        elif param_filter(p):
            tmp_list.append(p.grad.data.cpu().numpy().ravel())
    # print("grad tensor list size: ", len(tmp_list))

    tmp_list = np.concatenate(tmp_list)

    # print(f'grad tensor list shape: {tmp_list.shape}, dtype: {tmp_list.dtype}')

    return tmp_list


################################################################################
#                  For computing Hessian-vector products
################################################################################
def eval_hess_vec_prod(vec, params, net, loss_func, inputs, outputs, use_cuda=False):
    """
    Evaluate product of the Hessian of the loss function with a direction vector "vec".
    The product result is saved in the grad of net.

    Args:
        vec: a list of tensor with the same dimensions as "params".
        params: the parameter list of the net (ignoring biases and BN parameters).
        net: model with trained parameters.
        criterion: loss function.
        inputs: nn inputs.
        outputs: desired nn outputs.
        use_cuda: use GPU.
    """

    if use_cuda:
        net.cuda()
        vec = [v.cuda() for v in vec]

    device = next(net.parameters()).device
    inputs = inputs.to(device)
    outputs = outputs.to(device)

    net.eval()
    net.zero_grad() # clears grad for every parameter in the net
    
    ### OG IMPLEMENTATION
    # pred_outputs = net(inputs.unsqueeze(-1)).flatten()
    pred_outputs = net(inputs)
    loss = loss_func(pred_outputs,outputs)
    
    grad_f = torch.autograd.grad(loss, inputs=params, create_graph=True, allow_unused=True)

    # Compute inner product of gradient with the direction vector
    prod = Variable(torch.zeros(1)).type(type(grad_f[0].data))

    #for i in range(len(vec)):
    tmp = []
    for i in range(len(vec)):
        if (grad_f[i] is not None) and (vec[i] is not None):
            tmp.append((grad_f[i] * vec[i]).cpu().sum())
    prod =+ sum(tmp)
    # prod += sum([(grad_f[i] * vec[i]).cpu().sum() for i in range(len(vec))])

    # Compute the Hessian-vector product, H*v
    # prod.backward() computes dprod/dparams for every parameter in params and
    # accumulate the gradients into the params.grad attributes
    prod.backward()

################################################################################
#                  For computing Eigenvalues of Hessian
################################################################################
def min_max_hessian_eigs(
    net,
    inputs,
    outputs,
    criterion,
    rank=0,
    use_cuda=False,
    verbose=False,
    all_params=True,
    *,
    backend: str = "classical",
    vrpca_config: Optional["VRPCAConfig"] = None,
    compute_min: bool = True,
):
    """
    Compute the largest and the smallest eigenvalues of the Hessian matrix.

    Args:
        net: the trained model.
        inputs: nn inputs.
        outputs: desired nn outputs.
        criterion: loss function.
        rank: rank of the working node.
        use_cuda: use GPU.
        verbose: print more information.
        all_params: use all nn parameters.
        backend: one of {"classical", "vrpca"}. When "vrpca", both extreme
            eigenpairs are estimated with the stochastic VR-PCA solver.
        vrpca_config: optional VR-PCA configuration dataclass; ignored unless
            ``backend == 'vrpca'``.
        compute_min: when True (default), also compute the minimum eigenpair.
            For the VR-PCA backend this uses a second run on the negated Hessian.

    Returns:
        maxeig: max eigenvalue
        mineig: min eigenvalue (or None when ``backend == 'vrpca'`` and ``compute_min`` is False)
        maxeigvec: dominant eigenvector (as a numpy array shaped like the flattened parameters)
        mineigvec: minimum eigenvector (or None when omitted)
        iters_or_cost: classical iteration count, or an HVP-equivalent cost when ``backend == 'vrpca'``
    """

    if backend not in {"classical", "vrpca"}:
        raise ValueError("backend must be one of {'classical', 'vrpca'}")

    # VR-PCA path: dominant eigenpair via stochastic solver with optional
    # VR-PCA computation of the minimum eigenpair.
    if backend == "vrpca":
        if top_hessian_eigenpair_vrpca is None:
            raise RuntimeError("VR-PCA backend requested but not available in this build")

        result = top_hessian_eigenpair_vrpca(
            net=net,
            inputs=inputs,
            targets=outputs,
            criterion=criterion,
            all_params=all_params,
            use_cuda=use_cuda,
            config=vrpca_config,
        )

        maxeig = float(result.eigenvalue)
        maxeigvec = result.eigenvector.detach().cpu().numpy()
        if not compute_min:
            return maxeig, None, maxeigvec, None, float(result.hvp_equivalent_calls)

        if min_hessian_eigenpair_vrpca is None:
            raise RuntimeError("VR-PCA minimum eigenpair helper unavailable")

        min_result = min_hessian_eigenpair_vrpca(
            net=net,
            inputs=inputs,
            targets=outputs,
            criterion=criterion,
            all_params=all_params,
            use_cuda=use_cuda,
            config=vrpca_config,
        )

        mineig = float(min_result.eigenvalue)
        mineigvec = min_result.eigenvector.detach().cpu().numpy()
        total_cost = float(result.hvp_equivalent_calls + min_result.hvp_equivalent_calls)
        return maxeig, mineig, maxeigvec, mineigvec, total_cost

    if all_params:
        params = [p for p in net.parameters()]
    else:
        params = [p for p in net.parameters() if len(p.size()) >= 1]

    N = sum(p.numel() for p in params)

    def hess_vec_prod(vec):
        hess_vec_prod.count += 1  # simulates a static variable
        vec = npvec_to_tensorlist(vec, params)
        start_time = time.time()
        eval_hess_vec_prod(vec, params, net, criterion, inputs, outputs, use_cuda)
        prod_time = time.time() - start_time
        if verbose and rank == 0:
            print("Iter: %d  time: %f" % (hess_vec_prod.count, prod_time))
        return gradtensor_to_npvec(net, all_params)

    hess_vec_prod.count = 0
    if verbose and rank == 0:
        print("Rank %d: computing max eigenvalue" % rank)

    A = LinearOperator((N, N), matvec=hess_vec_prod)

    eigvals, eigvecs = eigsh(A, k=1, which='LM', tol=1e-2)
    maxeig = eigvals[0]
    maxeigvec = eigvecs
    if verbose and rank == 0:
        print('max eigenvalue = %f' % maxeig)

    # If the largest eigenvalue is positive, shift matrix so that any negative eigenvalue is now the largest
    # We assume the smallest eigenvalue is zero or less, and so this shift is more than what we need
    shift = maxeig * 1.0

    def shifted_hess_vec_prod(vec):
        return hess_vec_prod(vec) - shift * vec

    if verbose and rank == 0:
        print("Rank %d: Computing shifted eigenvalue" % rank)

    A = LinearOperator((N, N), matvec=shifted_hess_vec_prod)
    eigvals, eigvecs = eigsh(A, k=1, which='LM', tol=1e-2)
    eigvals = eigvals + shift
    mineig = eigvals[0]
    mineigvec = eigvecs
    if verbose and rank == 0:
        print('min eigenvalue = ' + str(mineig))

    if maxeig <= 0 and mineig > 0:
        maxeig, mineig = mineig, maxeig
        maxeigvec, mineigvec = mineigvec, maxeigvec

    return maxeig, mineig, maxeigvec, mineigvec, hess_vec_prod.count
