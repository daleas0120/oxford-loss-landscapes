import torch

def hessian_trace(model, loss_fn, inputs, targets, num_random_vectors=10):
    """
    Estimates the trace of the Hessian of the loss with respect to model parameters using Hutchinson's method.

    Args:
        model: torch.nn.Module
        loss_fn: loss function
        inputs: input tensor(s) for the model
        targets: target tensor(s) for the loss
        num_random_vectors: number of Hutchinson samples

    Returns:
        Estimated Hessian trace (float)
    """
    device = next(model.parameters()).device
    inputs = inputs.to(device)
    targets = targets.to(device)
    model.zero_grad()
    outputs = model(inputs)
    loss = loss_fn(outputs, targets)
    params = [p for p in model.parameters() if p.requires_grad]
    trace_estimate = 0.0

    for _ in range(num_random_vectors):
        # Create a random vector v with the same shape as all parameters
        v = [torch.randint_like(p, high=2, device=device).float() * 2 - 1 for p in params]  # Rademacher distribution
        # Compute the gradient of the loss
        grad = torch.autograd.grad(loss, params, create_graph=True)
        # Compute the dot product of grad and v
        grad_dot_v = sum((g * v_elem).sum() for g, v_elem in zip(grad, v))
        # Compute the gradient of grad_dot_v (i.e., Hessian-vector product)
        hvp = torch.autograd.grad(grad_dot_v, params, retain_graph=True)
        # Accumulate the trace estimate
        trace_estimate += sum((hvp_elem * v_elem).sum().item() for hvp_elem, v_elem in zip(hvp, v))

    return trace_estimate / num_random_vectors
