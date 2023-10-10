# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved
from copy import deepcopy
from typing import Iterable, List, Optional, Tuple

import numpy as np
import time
import torch
from omegaconf import OmegaConf

from mtrl.agent import grad_manipulation as grad_manipulation_agent
from mtrl.utils.types import ConfigType, TensorType
#from mtrl.agent.mgda import MinNormSolver


def euclidean_proj_simplex(v, s=1):
    """ Compute the Euclidean projection on a positive simplex
    Solves the optimisation problem (using the algorithm from [1]):
        min_w 0.5 * || w - v ||_2^2 , s.t. \sum_i w_i = s, w_i >= 0 
    Parameters
    ----------
    v: (n,) numpy array,
       n-dimensional vector to project
    s: int, optional, default: 1,
       radius of the simplex
    Returns
    -------
    w: (n,) numpy array,
       Euclidean projection of v on the simplex
    Notes
    -----
    The complexity of this algorithm is in O(n log(n)) as it involves sorting v.
    Better alternatives exist for high-dimensional sparse vectors (cf. [1])
    However, this implementation still easily scales to millions of dimensions.
    References
    ----------
    [1] Efficient Projections onto the .1-Ball for Learning in High Dimensions
        John Duchi, Shai Shalev-Shwartz, Yoram Singer, and Tushar Chandra.
        International Conference on Machine Learning (ICML 2008)
        http://www.cs.berkeley.edu/~jduchi/projects/DuchiSiShCh08.pdf
    [2] Projection onto the probability simplex: An efficient algorithm with a simple proof, and an application
        Weiran Wang, Miguel Á. Carreira-Perpiñán. arXiv:1309.1541
        https://arxiv.org/pdf/1309.1541.pdf
    [3] https://gist.github.com/daien/1272551/edd95a6154106f8e28209a1c7964623ef8397246#file-simplex_projection-py
    """
    assert s > 0, "Radius s must be strictly positive (%d <= 0)" % s
    v = v.astype(np.float64)
    n, = v.shape  # will raise ValueError if v is not 1-D
    # check if we are already on the simplex
    if v.sum() == s and np.alltrue(v >= 0):
        # best projection: itself!
        return v
    # get the array of cumulative sums of a sorted (decreasing) copy of v
    u = np.sort(v)[::-1]
    cssv = np.cumsum(u)
    # get the number of > 0 components of the optimal solution
    rho = np.nonzero(u * np.arange(1, n + 1) > (cssv - s))[0][-1]
    # compute the Lagrange multiplier associated to the simplex constraint
    theta = float(cssv[rho] - s) / (rho + 1)
    # compute the projection by thresholding v using theta
    w = (v - theta).clip(min=0)
    return w


def _check_param_device(param: TensorType, old_param_device: Optional[int]) -> int:
    """This helper function is to check if the parameters are located
        in the same device. Currently, the conversion between model parameters
        and single vector form is not supported for multiple allocations,
        e.g. parameters in different GPUs, or mixture of CPU/GPU.

        The implementation is taken from: https://github.com/pytorch/pytorch/blob/22a34bcf4e5eaa348f0117c414c3dd760ec64b13/torch/nn/utils/convert_parameters.py#L57

    Args:
        param ([TensorType]): a Tensor of a parameter of a model.
        old_param_device ([int]): the device where the first parameter
            of a model is allocated.

    Returns:
        old_param_device (int): report device for the first time

    """
    # Meet the first parameter
    if old_param_device is None:
        old_param_device = param.get_device() if param.is_cuda else -1
    else:
        warn = False
        if param.is_cuda:  # Check if in same GPU
            warn = param.get_device() != old_param_device
        else:  # Check if in CPU
            warn = old_param_device != -1
        if warn:
            raise TypeError("Found two parameters on different devices, "
                            "this is currently not supported.")
    return old_param_device


def apply_vector_grad_to_parameters(vec: TensorType, parameters: Iterable[TensorType], accumulate: bool = False):
    """Apply vector gradients to the parameters

    Args:
        vec (TensorType): a single vector represents the gradients of a model.
        parameters (Iterable[TensorType]): an iterator of Tensors that are the
            parameters of a model.
    """
    # Ensure vec of type Tensor
    if not isinstance(vec, torch.Tensor):
        raise TypeError("expected torch.Tensor, but got: {}".format(torch.typename(vec)))
    # Flag for the device where the parameter is located
    param_device = None

    # Pointer for slicing the vector for each parameter
    pointer = 0
    for param in parameters:
        # Ensure the parameters are located in the same device
        param_device = _check_param_device(param, param_device)

        # The length of the parameter
        num_param = param.numel()
        # Slice the vector, reshape it, and replace the old grad of the parameter
        if accumulate:
            param.grad = (param.grad + vec[pointer:pointer + num_param].view_as(param).data)
        else:
            param.grad = vec[pointer:pointer + num_param].view_as(param).data

        # Increment the pointer
        pointer += num_param


class Agent(grad_manipulation_agent.Agent):

    def __init__(
        self,
        env_obs_shape: List[int],
        action_shape: List[int],
        action_range: Tuple[int, int],
        device: torch.device,
        agent_cfg: ConfigType,
        multitask_cfg: ConfigType,
        cfg_to_load_model: Optional[ConfigType] = None,
        should_complete_init: bool = True,
    ):
        """Regularized gradient algorithm."""
        agent_cfg_copy = deepcopy(agent_cfg)
        del agent_cfg_copy['sdmgrad_lamda']
        del agent_cfg_copy['sdmgrad_method']

        OmegaConf.set_struct(agent_cfg_copy, False)
        agent_cfg_copy.cfg_to_load_model = None
        agent_cfg_copy.should_complete_init = False
        agent_cfg_copy.loss_reduction = "none"
        OmegaConf.set_struct(agent_cfg_copy, True)

        super().__init__(
            env_obs_shape=env_obs_shape,
            action_shape=action_shape,
            action_range=action_range,
            multitask_cfg=multitask_cfg,
            agent_cfg=agent_cfg_copy,
            device=device,
        )
        self.agent._compute_gradient = self._compute_gradient
        self._rng = np.random.default_rng()

        self.sdmgrad_lamda = agent_cfg['sdmgrad_lamda']
        self.sdmgrad_method = agent_cfg['sdmgrad_method']

        fn_maps = {
            "sdmgrad": self.sdmgrad,
        }
        for k in range(2, 50):
            fn_maps[f"sdmgrad_os{k}"] = self.sdmgrad_os

        fn_names = ", ".join(fn_maps.keys())
        assert self.sdmgrad_method in fn_maps, \
                f"[error] unrealized fn {self.sdmgrad_method}, currently we have {fn_names}"
        self.sdmgrad_fn = fn_maps[self.sdmgrad_method]
        self.wi_map = {}
        self.num_param_block = -1
        self.conflicts = []
        self.last_w = None
        self.save_target = 500000
        if "os" in self.sdmgrad_method:
            num_tasks = multitask_cfg['num_envs']
            self.os_n = int(self.sdmgrad_method[self.sdmgrad_method.find("os") + 2:])

        if should_complete_init:
            self.complete_init(cfg_to_load_model=cfg_to_load_model)

    def _compute_gradient(
        self,
        loss: TensorType,  # batch x 1
        parameters: List[TensorType],
        step: int,
        component_names: List[str],
        env_metadata: grad_manipulation_agent.EnvMetadata,
        retain_graph: bool = False,
        allow_unused: bool = False,
    ) -> None:

        #t0 = time.time()
        task_loss = self._convert_loss_into_task_loss(loss=loss, env_metadata=env_metadata)
        num_tasks = task_loss.shape[0]
        grad = []

        if "os" in self.sdmgrad_method:
            n = self.os_n
            while True:
                indices = np.random.binomial(1, n / num_tasks, num_tasks)
                sample_indices = np.where(indices == 1)[0]
                n_sample = sample_indices.shape[0]
                if n_sample:
                    break
            
            grad_os = []
            losses = [0] * n_sample
            for j in range(n_sample):
                losses[j] = task_loss[sample_indices[j]]
            for loss in losses:
                grad_os.append(
                    tuple(_grad.contiguous() for _grad in torch.autograd.grad(
                        loss,
                        parameters,
                        retain_graph=True,
                        allow_unused=allow_unused,
                    )))

            zero_grad = tuple(torch.zeros_like(item) for item in grad_os[0])
            for index in range(num_tasks):
                if index in sample_indices:
                    grad.append(grad_os[sample_indices.tolist().index(index)])
                else:
                    grad.append(zero_grad)
        else:
            for index in range(num_tasks):
                grad.append(
                    tuple(_grad.contiguous() for _grad in torch.autograd.grad(
                        task_loss[index],
                        parameters,
                        retain_graph=(retain_graph or index != num_tasks - 1),
                        allow_unused=allow_unused,
                    )))

        grad_vec = torch.cat(
            list(map(lambda x: torch.nn.utils.parameters_to_vector(x).unsqueeze(0), grad)),
            dim=0,
        )  # num_tasks x dim

        regularized_grad = self.sdmgrad_fn(grad_vec, num_tasks)
        apply_vector_grad_to_parameters(regularized_grad, parameters)

    def sdmgrad(self, grad_vec, num_tasks):
        """
        grad_vec: [num_tasks, dim]
        """
        grads = grad_vec

        GG = torch.mm(grads, grads.t()).cpu()
        scale = torch.mean(torch.sqrt(torch.diag(GG) + 1e-4))
        GG = GG / scale.pow(2)
        Gg = torch.mean(GG, dim=1)
        gg = torch.mean(Gg)

        if not hasattr(self, "w"):
            self.w = torch.ones(num_tasks) / num_tasks
        w = self.w
        w.requires_grad = True
        w_opt = torch.optim.SGD([w], lr=10, momentum=0.5)

        lamda = self.sdmgrad_lamda

        for i in range(20):
            w_opt.zero_grad()
            obj = 0.5 * torch.dot(w, torch.mv(GG, w)) + lamda * torch.dot(w, Gg) + 0.5 * lamda**2 * gg
            obj.backward()
            w_opt.step()
            proj = euclidean_proj_simplex(w.data.cpu().numpy())
            w.data.copy_(torch.from_numpy(proj).data)
        w.requires_grad = False

        g0 = torch.mean(grads, dim=0)
        gw = torch.mv(grads.t(), w.to(grads.device))
        g = (gw + lamda * g0) / (1 + lamda)
        return g

    def sdmgrad_os(self, grad_vec, num_tasks):
        """
        objective sampling
        grad_vec: [num_tasks, dim]
        """
        grads = grad_vec

        GG = torch.mm(grads, grads.t()).cpu()
        scale = (torch.diag(GG) + 1e-4).sqrt().mean()
        GG = GG / scale.pow(2)
        Gg = torch.mean(GG, dim=1)
        gg = torch.mean(Gg)

        if not hasattr(self, "w_os"):
            self.w_os = torch.ones(num_tasks) / num_tasks
        w = self.w_os
        w.requires_grad = True
        w_opt = torch.optim.SGD([w], lr=10, momentum=0.5)

        lamda = self.sdmgrad_lamda

        for i in range(20):
            w_opt.zero_grad()
            obj = 0.5 * torch.dot(w, torch.mv(GG, w)) + lamda * torch.dot(w, Gg) + 0.5 * lamda**2 * gg
            obj.backward()
            w_opt.step()
            proj = euclidean_proj_simplex(w.data.cpu().numpy())
            w.data.copy_(torch.from_numpy(proj).data)
        w.requires_grad = False

        g0 = torch.mean(grads, dim=0)
        gw = torch.mv(grads.t(), w.to(grads.device))
        gamma = num_tasks / self.os_n
        g = gamma * (gw + lamda * g0) / (1 + lamda)
        return g
