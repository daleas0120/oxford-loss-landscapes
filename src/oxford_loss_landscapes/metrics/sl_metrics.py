"""
A library of pre-written evaluation functions for PyTorch loss functions.

The classes and functions in this module cover common loss landscape evaluations. In particular,
computing the loss, the gradient of the loss (w.r.t. model parameters) and Hessian of the loss
(w.r.t. model parameters) for some supervised learning loss is easily accomplished.
"""


import numpy as np
import torch
import torch.autograd
from .metric import Metric
from ..model_interface.model_parameters import rand_u_like
from ..model_interface.model_wrapper import ModelWrapper

class Loss(Metric):
    """Computes a specified loss function over specified input-output pairs."""
    def __init__(self, loss_fn, inputs: torch.Tensor, target: torch.Tensor):
        super().__init__()
        self.loss_fn = loss_fn
        self.inputs = inputs
        self.target = target

    def __call__(self, model_wrapper: ModelWrapper) -> float:
        return self.loss_fn(model_wrapper.forward(self.inputs), self.target).item()


class LossGradient(Metric):
    """ Computes the gradient of a specified loss function w.r.t. the model parameters
    over specified input-output pairs. """
    def __init__(self, loss_fn, inputs: torch.Tensor, target: torch.Tensor):
        super().__init__()
        self.loss_fn = loss_fn
        self.inputs = inputs
        self.target = target

    def __call__(self, model_wrapper: ModelWrapper) -> np.ndarray:
        loss = self.loss_fn(model_wrapper.forward(self.inputs), self.target)
        gradient = torch.autograd.grad(loss, model_wrapper.named_parameters()).detach().numpy()
        model_wrapper.zero_grad()
        return gradient


class LossPerturbations(Metric):
    """ Computes random perturbations in the loss value along a sample or random directions.
    These perturbations can be used to reason probabilistically about the curvature of a
    point on the loss landscape, as demonstrated in the paper by Schuurmans et al
    (https://arxiv.org/abs/1811.11214)."""
    def __init__(self, loss_fn, inputs: torch.Tensor, target: torch.Tensor, n_directions, alpha):
        super().__init__()
        self.loss_fn = loss_fn
        self.inputs = inputs
        self.target = target
        self.n_directions = n_directions
        self.alpha = alpha

    def __call__(self, model_wrapper: ModelWrapper) -> np.ndarray:
        # start point and directions
        start_point = model_wrapper.get_module_parameters()
        start_loss = self.loss_fn(model_wrapper.forward(self.inputs), self.target).item()

        # compute start loss and perturbed losses
        results = []
        for idx in range(self.n_directions):
            direction = rand_u_like(start_point)
            start_point.add_(direction)

            loss = self.loss_fn(model_wrapper.forward(self.inputs), self.target).item()
            results.append(loss - start_loss)

            start_point.sub_(direction)

        return np.array(results)




class TransformerLoss(Metric):
    """Loss metric specifically designed for transformer models."""
    
    def __init__(self, loss_fn, inputs, targets, attention_mask=None):
        super().__init__()
        self.loss_fn = loss_fn
        self.inputs = inputs
        self.targets = targets
        self.attention_mask = attention_mask

    def __call__(self, model_wrapper: ModelWrapper) -> float:
        # Prepare inputs for transformer
        if isinstance(self.inputs, dict):
            outputs = model_wrapper.forward(self.inputs)
        else:
            model_inputs = {
                'input_ids': self.inputs,
                'attention_mask': self.attention_mask
            }
            if self.attention_mask is not None:
                model_inputs['attention_mask'] = self.attention_mask
            outputs = model_wrapper.forward(model_inputs)
        
        # Extract logits (handle different transformer output formats)
        if hasattr(outputs, 'logits'):
            logits = outputs.logits
        elif hasattr(outputs, 'last_hidden_state'):
            # For encoder-only models, you might need a classification head
            logits = outputs.last_hidden_state
        else:
            logits = outputs
            
        return self.loss_fn(logits, self.targets).item()


class LanguageModelingLoss(Metric):
    """Specialized loss for language modeling tasks."""
    
    def __init__(self, inputs, shift_labels=True):
        super().__init__()
        self.inputs = inputs
        self.shift_labels = shift_labels
        self.loss_fn = torch.nn.CrossEntropyLoss()

    def __call__(self, model_wrapper: ModelWrapper) -> float:
        outputs = model_wrapper.forward(self.inputs)
        
        if hasattr(outputs, 'logits'):
            logits = outputs.logits
        else:
            logits = outputs
            
        if self.shift_labels:
            # Standard language modeling: predict next token
            shift_logits = logits[..., :-1, :].contiguous()
            shift_labels = self.inputs[..., 1:].contiguous()
            
            # Flatten for loss computation
            shift_logits = shift_logits.view(-1, shift_logits.size(-1))
            shift_labels = shift_labels.view(-1)
            
            return self.loss_fn(shift_logits, shift_labels).item()
        else:
            # Direct comparison (for other tasks)
            return self.loss_fn(logits.view(-1, logits.size(-1)), 
                              self.inputs.view(-1)).item()