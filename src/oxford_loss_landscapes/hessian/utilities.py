# Some code snippets are taken from
# https://github.com/alperyeg/enkf-dnn-lod2020
# and
# https://github.com/tomgoldstein/loss-landscape

import torch
from torchdiffeq import odeint
from copy import deepcopy
import numpy as np

def weights_init(m):
    if isinstance(m, torch.nn.Linear):
        torch.nn.init.constant_(m.weight,val=1e-1)#kaiming_uniform_(m.weight)
        torch.nn.init.constant_(m.bias,val=1e-1)
        
def get_weights(net):
    """ 
    Extract parameters from net, and return a list of tensors.

    Parameters: 
    net (torch nn): neural net class object 
  
    Returns: 
    list of tensors: neural network parameters 
  
    """
        
    return [deepcopy(p.data) for p in net.parameters()]

def set_weights(net, weights):
    """ 
    Set neural network parameters.

    Parameters: 
    net (torch nn): neural net class object 
    weights (list of torch tensors): nn parameters

  
    """
    
    for (p, w) in zip(net.parameters(), weights):
        p.data.copy_(w.type(type(p.data)))

def tensorlist_to_tensor(weights):
    """ Concatnate a list of tensors into one tensor.

        Args:
            weights: a list of parameter tensors

        Returns:
            concatenated 1D tensor
    """
    
    return torch.cat([w.view(w.numel()) if w.dim() > 1 else torch.FloatTensor(w) for w in weights])

def npvec_to_tensorlist(flattened_weights, params):
    """ Convert a numpy vector to a list of tensors with the same shape as "params".

        Args:
            flattened_weights: a list of numpy vectors
            base: a list of parameter tensors from net

        Returns:
            a list of tensors with the same shape as base
    """
    
    if isinstance(params, list):
        w2 = deepcopy(params)
        idx = 0
        for w in w2:
            w.copy_(flattened_weights[idx:idx + w.numel()].clone().detach().view(w.size()))
            idx += w.numel()
        assert(idx == len(flattened_weights)), 'Flattened weights length does not match total number of parameters.'
        return w2
    else:
        s2 = []
        idx = 0
        for (k, w) in params.items():
            s2.append(flattened_weights[idx:idx + w.numel()]).clone().detach().view(w.size())
            idx += w.numel()
        assert(idx == len(flattened_weights))
        return s2
        
def covariance(x, y, number_ensembles):
    """ Compute covriance associated with tensors x,y.

        Args:
            x,y: torch tensors
            number_ensembles: number of EKI ensemble members

        Returns:
            covariance matrix
    """
    
    x_mean = x.mean(0)
    y_mean = y.mean(0)
    #return torch.mean(torch.stack([torch.outer(x[j] - x_mean, y[j] - y_mean) \
    #for j in range(number_ensembles)]),axis=0)
    return torch.tensordot(x - x_mean, y - y_mean,
                           dims=([0], [0])) / number_ensembles

def update_step(u, 
                F_u, 
                y, 
                Sigma, 
                ensemble_members,
                rescale_time_step = False,
                h0=1.0,
                eps=0.5):
    """ EKI update step.

        Args:
            u: ensemble
            F_u: predicted state
            y: target state
            Sigma: noise term
            ensemble_members: number of EKI ensemble members
            rescale_time_step: if true, time step will be rescaled
            h0: unrescaled timestep
            eps: rescaling hyperparameter

        Returns:
            updated ensemble
    """
    
    Cuw_u = covariance(u,F_u,ensemble_members)
    
    Sigma_repeat = Sigma.unsqueeze(0).repeat(ensemble_members,1,1)
    Sigma_block = torch.block_diag(*Sigma_repeat)
    lstsq = (torch.lstsq((F_u-y).flatten(),Sigma_block)[0]).reshape(F_u.size())
    
    h = 1
    delta_u = (Cuw_u @ lstsq.T).T
    u_new = u - h*delta_u
    
    if rescale_time_step:
        u_normalized = u.flatten()/torch.norm(u.flatten())**2
        delta_u = (Cuw_u @ lstsq.T).T.flatten()
        D_u = torch.outer(delta_u,u_normalized) 
        h = h0/(torch.norm(D_u)+eps)
        u_new = u.flatten() - h*delta_u
        u_new = u_new.reshape(u.size())

    return u_new
    
def generate_initial_ensemble(func, 
                              x0, 
                              t,
                              ensemble_members,
                              number_layers=-1, 
                              number_neurons=-1,
                              solve_oc_problem=True):
    """ Generate initial EKI ensemble.

        Args:
            func: neural ODE
            x0: initial condition of neural ODE
            t: time array
            ensemble_members: number of EKI ensemble members
            number_layers: number of RNN layers
            number_neurons: number of RNN neurons
            solve_oc_problem: solution of optimal control problem if "true"

        Returns:
            initial ensemble u and corresponding outputs F(u)
    """
    
    initial_ensemble = torch.tensor([])
    G_u = torch.tensor([])

    energies = torch.tensor([])
    
    for j in range(ensemble_members):

        if number_layers != -1:
            func_enkf = func(number_layers, number_neurons)
        else:
            func_enkf = func()
        
        pred_x = odeint(func_enkf, x0, t, method='dopri5')
        
        if solve_oc_problem:
            reached_state = torch.tensor([pred_x[-1][0].clone().detach()])
            G_u = torch.cat((G_u, reached_state))

            control_energy = torch.tensor([pred_x[-1][1].clone().detach()])
            energies = torch.cat((energies,control_energy))
            
        else:
            G_u = torch.cat((G_u, pred_x.clone().detach()))
        
        params = get_weights(func_enkf)
        flattened_params = tensorlist_to_tensor(params)
        
        initial_ensemble = torch.cat((initial_ensemble, flattened_params))
                
    u = initial_ensemble.reshape(ensemble_members,-1)
    
    if solve_oc_problem:
        F_u = torch.vstack((G_u,energies)).T
    
        return u, F_u
    
    else:
        G_u = G_u.flatten()
        G_u = G_u.reshape(ensemble_members,-1)
        
        return u, G_u
    
def compute_F_u(u,
                func,
                x0,
                t,
                ensemble_members,
                number_layers=-1, 
                number_neurons=-1,
                solve_oc_problem=True):          
    """ Compute neural ODE predictions.

        Args:
            u: ensemble
            func: neural ODE
            x0: initial condition of neural ODE
            t: time array
            ensemble_members: number of EKI ensemble members
            number_layers: number of RNN layers
            number_neurons: number of RNN neurons
            solve_oc_problem: solution of optimal control problem if "true"

        Returns:
            neural ODE predictions F(u)
    """
    
    G_u = torch.tensor([])
    energies = torch.tensor([])
    
    for j in range(ensemble_members):

        if number_layers != -1:
            func_enkf = func(number_layers, number_neurons)
        else:
            func_enkf = func()
            
        params = get_weights(func_enkf)
        updated_params = npvec_to_tensorlist(u[j],params)
        set_weights(func_enkf,updated_params)
    
        pred_x = odeint(func_enkf, x0, t, method='dopri5')
        
        if solve_oc_problem:
             reached_state = torch.tensor([pred_x[-1][0].clone().detach()])
             G_u = torch.cat((G_u, reached_state))
        
             control_energy = torch.tensor([pred_x[-1][1].clone().detach()])
             energies = torch.cat((energies,control_energy))
             
        else:
            G_u = torch.cat((G_u, pred_x.clone().detach()))
    
    if solve_oc_problem:
        F_u = torch.vstack((G_u,energies)).T
    
        return F_u
        
    else:
        G_u = G_u.flatten()
        G_u = G_u.reshape(ensemble_members,-1)
        
        return G_u
    
def add_new_ensemble_members(u, 
                             func,
                             new_ensemble_members,
                             number_layers=-1, 
                             number_neurons=-1,
                             new_ensemble_prefactor=1):
    """ Add new ensemble members.

        Args:
            u: ensemble
            func: neural ODE
            new_ensemble_members: number of new ensemble members
            number_layers: number of RNN layers
            number_neurons: number of RNN neurons
            new_ensemble_prefactor: prefactor of mean perturbation

        Returns:
            extended ensemble
    """
    
    u_mean = torch.mean(u,axis=0)
    
    ensemble_members = u.size()[0]
    number_params = u.size()[1]
        
    u = torch.flatten(u)
    
    for j in range(new_ensemble_members):
    
        if number_layers != -1:
            func_enkf = func(number_layers, number_neurons)
        else:
            func_enkf = func()
            
        params = get_weights(func_enkf)
        flattened_params = tensorlist_to_tensor(params)
        u = torch.cat((u, u_mean+new_ensemble_prefactor*flattened_params))
        
    ensemble_members += new_ensemble_members

    u = u.reshape(ensemble_members,number_params)
    
    return u, ensemble_members


def copy_wts_into_model(new_wts_vector, model):
    """
    Copy a new weights vector into a model.
    """

    layer_names = model.state_dict().keys()
    print(layer_names)

    new_model_wt_dict = deepcopy(model.state_dict())
    empty_model = deepcopy(model)

    for layer, new_param in zip(layer_names, new_wts_vector):
        if new_param.shape == new_model_wt_dict[layer].shape:
            new_model_wt_dict[layer] = new_param
        else:
            print(layer+" incompatible")

    err_layers = empty_model.load_state_dict(new_model_wt_dict, strict=False)
    print(err_layers)

    return empty_model
    