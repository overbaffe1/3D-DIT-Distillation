import numpy as np
import math
import torch
from copy import deepcopy
from torch.optim.lr_scheduler import _LRScheduler
import gc

def get_expon_lr_func(
    lr_init, lr_final, lr_delay_steps=0, lr_delay_mult=1.0, max_steps=100000):
    """
    Copied from Plenoxels

    Continuous learning rate decay function. Adapted from JaxNeRF
    The returned rate is lr_init when step=0 and lr_final when step=max_steps, and
    is log-linearly interpolated elsewhere (equivalent to exponential decay).
    If lr_delay_steps>0 then the learning rate will be scaled by some smooth
    function of lr_delay_mult, such that the initial learning rate is
    lr_init*lr_delay_mult at the beginning of optimization but will be eased back
    to the normal learning rate when steps>lr_delay_steps.
    :param conf: config subtree 'lr' or similar
    :param max_steps: int, the number of steps during optimization.
    :return HoF which takes step as input
    """

    def helper(step):
        if step < 0 or (lr_init == 0.0 and lr_final == 0.0):
            # Disable this parameter
            return 0.0
        if lr_delay_steps > 0:
            # A kind of reverse cosine decay.
            delay_rate = lr_delay_mult + (1 - lr_delay_mult) * np.sin(
                0.5 * np.pi * np.clip(step / lr_delay_steps, 0, 1)
            )
        else:
            delay_rate = 1.0
        t = np.clip(step / max_steps, 0, 1)
        log_lerp = np.exp(np.log(lr_init) * (1 - t) + np.log(lr_final) * t)
        return delay_rate * log_lerp

    return helper


def process_training_config_and_return_lambda_lr(config: dict,
                                                 lr_mult_default=5,
                                                 max_steps_default=100000):
    """
    Method to process dictionary with training option for learning different learning schedules.

    :param config:
    :return:
    """

    lr = config.get("lr", None)
    lr_multiplier = config.get("lr_multiplier", lr_mult_default)
    max_steps = config.get("max_steps", max_steps_default)

    if lr is None:
        lr_lambda = lambda epoch: (1 / lr_multiplier) ** (epoch / max_steps)
        return lr_lambda

    lr_final = config.get("final_lr", None)
    lr_final = lr / lr_multiplier if lr_final is None else lr_final
    lr_lambda = lambda epoch: (lr_final / lr) * (epoch / max_steps)

    return lr_lambda

def get_learning_params(config, lr_mult_default=5, max_steps_default=100000):
    """
    Reads dict with optimization config and tries to extract information about learning
    rate for this block of parameters and params for lr scheduler initialization

    if field 'lr' is empty then return all None. Means that we don't need to specify any param groups lr.
    if not try to extract 'lr_final'.
        if 'lr_final' is empty fill lr_final with lr/lr_multiplier that recieved with also this rule

    if field 'max_steps' is empty than fill with max_steps_default.

    from lr,lr_final, max_steps we can get lambda function to update learning rate.
    """
    lr = config.get("lr", None)

    if lr is None:
        return None,None,None

    lr_multiplier = config.get("lr_multiplier", lr_mult_default)
    max_steps = config.get("max_steps", max_steps_default)

    warmup_steps = config.get("warmup_steps", 0)
    warmup_init_lr = config.get("warmup_init_lr", lr)

    lr_final = config.get("lr_final", None)
    lr_final = lr/lr_multiplier if lr_final is None else lr_final

    return lr, lr_final, max_steps, warmup_steps, warmup_init_lr


def get_lambdas(epoch, lr, lr_final, max_steps):
    return (lr_final / lr) ** (epoch / max_steps)


def warmup(epoch, lr, warmup_init_lr, warmup_steps):
    return (warmup_init_lr / lr) ** (1 - epoch / warmup_steps)


class CosineAnnealingWarmupRestarts(_LRScheduler):
    """
        optimizer (Optimizer): Wrapped optimizer.
        first_cycle_steps (int): First cycle step size.
        cycle_mult(float): Cycle steps magnification. Default: -1.
        max_lr(float): First cycle's max learning rate. Default: 0.1.
        min_lr(float): Min learning rate. Default: 0.001.
        warmup_steps(int): Linear warmup step size. Default: 0.
        gamma(float): Decrease rate of max learning rate by cycle. Default: 1.
        last_epoch (int): The index of last epoch. Default: -1.
    """

    def __init__(self,
                 optimizer: torch.optim.Optimizer,
                 first_cycle_steps: int,
                 cycle_mult: float = 1.,
                 max_lr: float = 0.1,
                 min_lr: float = 0.001,
                 warmup_steps: int = 0,
                 gamma: float = 1.,
                 last_epoch: int = -1
                 ):
        assert warmup_steps < first_cycle_steps

        self.first_cycle_steps = first_cycle_steps  # first cycle step size
        self.cycle_mult = cycle_mult  # cycle steps magnification
        self.base_max_lr = max_lr  # first max learning rate
        self.max_lr = max_lr  # max learning rate in the current cycle
        self.min_lr = min_lr  # min learning rate
        self.warmup_steps = warmup_steps  # warmup step size
        self.gamma = gamma  # decrease rate of max learning rate by cycle

        self.cur_cycle_steps = first_cycle_steps  # first cycle step size
        self.cycle = 0  # cycle count
        self.step_in_cycle = last_epoch  # step size of the current cycle

        super(CosineAnnealingWarmupRestarts, self).__init__(optimizer, last_epoch)

        # set learning rate min_lr
        self.init_lr()

    def init_lr(self):
        self.base_lrs = []
        for param_group in self.optimizer.param_groups:
            param_group['lr'] = self.min_lr
            self.base_lrs.append(self.min_lr)

    def get_lr(self):
        if self.step_in_cycle == -1:
            return self.base_lrs
        elif self.step_in_cycle < self.warmup_steps:
            return [(self.max_lr - base_lr) * self.step_in_cycle / self.warmup_steps + base_lr for base_lr in
                    self.base_lrs]
        else:
            return [base_lr + (self.max_lr - base_lr) \
                    * (1 + math.cos(math.pi * (self.step_in_cycle - self.warmup_steps) \
                                    / (self.cur_cycle_steps - self.warmup_steps))) / 2
                    for base_lr in self.base_lrs]

    def step(self, epoch=None):
        if epoch is None:
            epoch = self.last_epoch + 1
            self.step_in_cycle = self.step_in_cycle + 1
            if self.step_in_cycle >= self.cur_cycle_steps:
                self.cycle += 1
                self.step_in_cycle = self.step_in_cycle - self.cur_cycle_steps
                self.cur_cycle_steps = int(
                    (self.cur_cycle_steps - self.warmup_steps) * self.cycle_mult) + self.warmup_steps
        else:
            if epoch >= self.first_cycle_steps:
                if self.cycle_mult == 1.:
                    self.step_in_cycle = epoch % self.first_cycle_steps
                    self.cycle = epoch // self.first_cycle_steps
                else:
                    n = int(math.log((epoch / self.first_cycle_steps * (self.cycle_mult - 1) + 1), self.cycle_mult))
                    self.cycle = n
                    self.step_in_cycle = epoch - int(
                        self.first_cycle_steps * (self.cycle_mult ** n - 1) / (self.cycle_mult - 1))
                    self.cur_cycle_steps = self.first_cycle_steps * self.cycle_mult ** (n)
            else:
                self.cur_cycle_steps = self.first_cycle_steps
                self.step_in_cycle = epoch

        self.max_lr = self.base_max_lr * (self.gamma ** self.cycle)
        self.last_epoch = math.floor(epoch)
        for param_group, lr in zip(self.optimizer.param_groups, self.get_lr()):
            param_group['lr'] = lr
            
def cosine_lambda(step, lr_min, lr_max, first_cycle_steps):
    first_cycle_steps = max(first_cycle_steps, 1)
    step = step + first_cycle_steps  # stupid fix
    cycle_idx = 0
    while step >= first_cycle_steps * 2 ** cycle_idx:
        cycle_idx += 1
    cycle_start = 0 if cycle_idx == 0 else first_cycle_steps * 2 ** (cycle_idx - 1)
    cycle_denom = first_cycle_steps if cycle_idx == 0 else first_cycle_steps * 2 ** (cycle_idx - 1)
    return lr_min + 0.5 * (lr_max - lr_min) * (1 + np.cos(np.pi * (step - cycle_start) / max(1, cycle_denom)))


class Lamb(torch.optim.Optimizer):
    r"""Implements Lamb algorithm.

    It has been proposed in `Large Batch Optimization for Deep Learning: Training BERT in 76 minutes`_.

    Arguments:
        params (iterable): iterable of parameters to optimize or dicts defining
            parameter groups
        lr (float, optional): learning rate (default: 1e-3)
        betas (Tuple[float, float], optional): coefficients used for computing
            running averages of gradient and its square (default: (0.9, 0.999))
        eps (float, optional): term added to the denominator to improve
            numerical stability (default: 1e-8)
        weight_decay (float, optional): weight decay (L2 penalty) (default: 0)
        adam (bool, optional): always use trust ratio = 1, which turns this into
            Adam. Useful for comparison purposes.

    .. _Large Batch Optimization for Deep Learning: Training BERT in 76 minutes:
        https://arxiv.org/abs/1904.00962
    """

    def __init__(
        self, params, lr=1e-3, betas=(0.9, 0.999), eps=1e-6, weight_decay=0, adam=False
    ):
        if not 0.0 <= lr:
            raise ValueError("Invalid learning rate: {}".format(lr))
        if not 0.0 <= eps:
            raise ValueError("Invalid epsilon value: {}".format(eps))
        if not 0.0 <= betas[0] < 1.0:
            raise ValueError("Invalid beta parameter at index 0: {}".format(betas[0]))
        if not 0.0 <= betas[1] < 1.0:
            raise ValueError("Invalid beta parameter at index 1: {}".format(betas[1]))
        defaults = dict(lr=lr, betas=betas, eps=eps, weight_decay=weight_decay)
        self.adam = adam
        super(Lamb, self).__init__(params, defaults)

    def step(self, closure=None):
        """Performs a single optimization step.

        Arguments:
            closure (callable, optional): A closure that reevaluates the model
                and returns the loss.
        """
        loss = None
        if closure is not None:
            loss = closure()

        for group in self.param_groups:
            for p in group["params"]:
                if p.grad is None:
                    continue
                grad = p.grad.data
                if grad.is_sparse:
                    raise RuntimeError(
                        "Lamb does not support sparse gradients, consider SparseAdam instad."
                    )

                state = self.state[p]

                # State initialization
                if len(state) == 0:
                    state["step"] = 0
                    # Exponential moving average of gradient values
                    state["exp_avg"] = torch.zeros_like(p.data)
                    # Exponential moving average of squared gradient values
                    state["exp_avg_sq"] = torch.zeros_like(p.data)

                exp_avg, exp_avg_sq = state["exp_avg"], state["exp_avg_sq"]
                beta1, beta2 = group["betas"]

                state["step"] += 1

                # Decay the first and second moment running average coefficient
                # m_t
                exp_avg.mul_(beta1).add_(grad, alpha=1 - beta1)
                # v_t
                exp_avg_sq.mul_(beta2).addcmul_(grad, grad, value=1 - beta2)

                # Paper v3 does not use debiasing.
                bias_correction1 = 1 - beta1 ** state["step"]
                bias_correction2 = 1 - beta2 ** state["step"]
                # Apply bias to lr to avoid broadcast.
                step_size = group["lr"] * math.sqrt(bias_correction2) / bias_correction1

                weight_norm = p.data.pow(2).sum().sqrt() #.clamp(0, 10)

                adam_step = exp_avg / exp_avg_sq.sqrt().add(group["eps"])
                if group["weight_decay"] != 0:
                    adam_step.add_(p.data, alpha=group["weight_decay"] * group["lr"])

                adam_norm = adam_step.pow(2).sum().sqrt()
                if weight_norm == 0 or adam_norm == 0:
                    trust_ratio = 1
                else:
                    trust_ratio = weight_norm / adam_norm
                state["weight_norm"] = weight_norm
                state["adam_norm"] = adam_norm
                state["trust_ratio"] = trust_ratio
                if self.adam:
                    trust_ratio = 1

                p.data.add_(adam_step, alpha=-step_size * trust_ratio)

        return loss


class EMA(torch.nn.Module):
    def __init__(self, model, decay=0.9999, device=None):
        super(EMA, self).__init__()
        self.module = deepcopy(model)
        self.module.eval()
        self.decay = decay
        self.device = device
        if self.device is not None:
            self.module.to(device=device)
        for param in self.module.parameters():
            param.requires_grad_(False)

    def _update(self, model, update_fn):
        with torch.no_grad():
            for ema_v, model_v in zip(
                self.module.state_dict().values(), model.state_dict().values()
            ):
                if self.device is not None:
                    model_v = model_v.to(device=self.device, dtype=ema_v.dtype)
                ema_v.copy_(update_fn(ema_v, model_v))

    def update(self, model):
        self._update(
            model, update_fn=lambda e, m: self.decay * e + (1.0 - self.decay) * m
        )

    def set(self, model):
        self._update(model, update_fn=lambda e, m: m)

class ParamEMA:
    def __init__(self, model, decay=0.9999, device=None):
        super(ParamEMA, self).__init__()
        self.model = [model] # save for inference
        self.decay = decay
        self.device = device
        self.params = {n: p.detach().clone() for n, p in model.state_dict().items()}
        if self.device is not None:
            self.params = {n: p.to(device) for n, p in self.params.items()}
    
    def __call__(self, *args, **kwargs):
        out = self.model[0](*args, **kwargs)
        return out
    
    def eval(self):
        self.orig_state_dict = {k: v.clone() for k, v in self.model[0].state_dict().items()}
        self.model[0].load_state_dict(self.params)
        
    def train(self):
        self.model[0].load_state_dict(self.orig_state_dict)
        del self.orig_state_dict
    
    def to(self, device):
        self.device = device
        self.model[0].to(device)
        self.params = {n: p.to(device) for n, p in self.params.items()}
        return self
    def state_dict(self, *args, **kwargs):
        return self.params
    
    def load_state_dict(self, sd, *args, **kwargs):
        for k, v in sd.items():
            if k in self.params and self.params[k].shape == v.shape:
                self.params[k].copy_(v)
        return self
    
    def _update(self, model, update_fn):
        with torch.no_grad():
            for ema_v, model_v in zip(
                    self.params.values(), model.state_dict().values()
            ):
                if self.device is not None:
                    model_v = model_v.to(device=self.device, dtype=ema_v.dtype)
                ema_v.copy_(update_fn(ema_v, model_v))
    
    def update(self, model):
        self._update(
            model, update_fn=lambda e, m: self.decay * e + (1.0 - self.decay) * m
        )
    
    def set(self, model):
        self._update(model, update_fn=lambda e, m: m)
        
class Lion(torch.optim.Optimizer):
    
    def __init__(self, params, lr=1e-4, betas=(0.9, 0.99), weight_decay=0.0):
        
        if not 0.0 <= lr:
            raise ValueError('Invalid learning rate: {}'.format(lr))
        if not 0.0 <= betas[0] < 1.0:
            raise ValueError('Invalid beta parameter at index 0: {}'.format(betas[0]))
        if not 0.0 <= betas[1] < 1.0:
            raise ValueError('Invalid beta parameter at index 1: {}'.format(betas[1]))
        defaults = dict(lr=lr, betas=betas, weight_decay=weight_decay)
        super().__init__(params, defaults)
    
    @torch.no_grad()
    def step(self, closure=None):
        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()
        
        for group in self.param_groups:
            for p in group['params']:
                if p.grad is None:
                    continue
                
                p.data.mul_(1 - group['lr'] * group['weight_decay'])
                
                grad = p.grad
                state = self.state[p]
                if len(state) == 0:
                    state['exp_avg'] = torch.zeros_like(p)
                
                exp_avg = state['exp_avg']
                beta1, beta2 = group['betas']
                
                update = exp_avg * beta1 + grad * (1 - beta1)
                p.add_(torch.sign(update), alpha=-group['lr'])
                exp_avg.mul_(beta2).add_(grad, alpha=1 - beta2)
        
        return loss

def get_orig_module(trainable_module):
    # unwrap from ddp
    if isinstance(trainable_module, torch.torch.nn.parallel.DistributedDataParallel):
        trainable_module = trainable_module.module
    # unwrap from compile
    trainable_module = getattr(trainable_module, "_orig_mod", trainable_module)
    return trainable_module

def expand_as(x, orig):
    diff = len(orig.shape) - len(x.shape)
    return x.view(*x.shape, *(1 for _ in range(diff)))


def fix_state_dict(new_state_dict, model_state_dict):
    
    for k, v in model_state_dict.items():
        if "_orig_mod." in k and k.replace("_orig_mod.", "") in new_state_dict:
            new_k = k.replace("_orig_mod.", "")
            new_state_dict[k] = new_state_dict[new_k]
            del new_state_dict[new_k]
        
    return new_state_dict
    
def list_to_device(lst, device):
    device_list = []
    for item in lst:
        if isinstance(item, torch.Tensor):
            item = item.to(device, non_blocking=True)
        elif isinstance(item, dict):
            item = dict_to_device(item, device)
        elif isinstance(item, list):
            item = list_to_device(item, device)
        device_list.append(item)
    return device_list


def dict_to_device(data_dict, device):
    for value in data_dict:
        if isinstance(data_dict[value], torch.Tensor):
            data_dict[value] = data_dict[value].to(device, non_blocking=True)
        elif isinstance(data_dict[value], dict):
            data_dict[value] = dict_to_device(data_dict[value], device)
        elif isinstance(data_dict[value], list):
            data_dict[value] = list_to_device(data_dict[value], device)
    return data_dict


def to_single_object_list(lst, idx=0, size=1, list_as_tensors=tuple()):
    result_list = []
    for item in lst:
        if isinstance(item, torch.Tensor):
            result_list.append(item[idx : idx + size])
        elif isinstance(item, list):
            result_list.append(to_single_object_list(item, idx, size, list_as_tensors))
        elif isinstance(item, dict):
            result_list.append(to_single_object_dict(item, idx, size, list_as_tensors))
        else:
            result_list.append(item)
    return result_list


def to_single_object_dict(data_dict, idx=0, size=1, list_as_tensors=tuple()):
    for k in data_dict:
        if isinstance(data_dict[k], torch.Tensor):
            data_dict[k] = data_dict[k][idx : idx + size]
        
        if isinstance(data_dict[k], list):
            if k in list_as_tensors:
                data_dict[k] = data_dict[k][idx : idx + size]
            else:
                data_dict[k] = to_single_object_list(data_dict[k], idx, size, list_as_tensors)

        if isinstance(data_dict[k], dict):
            data_dict[k] = to_single_object_dict(data_dict[k], idx, size, list_as_tensors)

    return data_dict
    
    
def all_gather_image(x):

    world_size = torch.distributed.get_world_size()
    local_size = torch.tensor(len(x), device=x.device, dtype=torch.float32)
    all_sizes = [torch.zeros_like(local_size) for _ in range(world_size)]
    torch.distributed.all_gather(all_sizes, local_size)
    max_size = max([size.item() for size in all_sizes])
    diff_size = max_size - len(x)
    if diff_size > 0:
        x = torch.cat((x, torch.zeros(int(diff_size), *x.shape[1:], device=x.device, dtype=x.dtype)))

    if torch.distributed.get_rank() == 0:
        all_tensors_padded = [torch.zeros_like(x) for _ in range(world_size)]
        torch.distributed.gather(
            x, gather_list=all_tensors_padded, group=torch.distributed.group.WORLD
        )
        x_all = torch.cat(
            [xx[: int(size.item())] for xx, size in zip(all_tensors_padded, all_sizes)]
        )
    else:
        torch.distributed.gather(x, dst=0, group=torch.distributed.group.WORLD)
        x_all = None

    return x_all

@torch.no_grad()
def check_ddp_consistency(module):
    for name, param in get_orig_module(module).named_parameters():
        tensor = param.data.detach()
        other = tensor.clone()
        torch.distributed.broadcast(tensor=other, src=0)
        if not (tensor == other).all():
            print(f"RANK {torch.distributed.get_rank()}: FOUND WEIGHTS DDP INCONSISTENT IN {name}")
            param.data.copy_(other)
            # assert (tensor == other).all(), f"{name}_{torch.distributed.get_rank()}"