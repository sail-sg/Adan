import math
from copy import deepcopy

import torch
from torch.optim.optimizer import Optimizer

from torchvision.models import resnet18

from adan import Adan

# copy-paste of original implementation. could be removed before merging the PR
class AdanOriginal(Optimizer):
    """
    Implements a pytorch variant of Adan
    Adan was proposed in
    Adan: Adaptive Nesterov Momentum Algorithm for Faster Optimizing Deep Models[J]. arXiv preprint arXiv:2208.06677, 2022.
    https://arxiv.org/abs/2208.06677
    Arguments:
        params (iterable): iterable of parameters to optimize or dicts defining parameter groups.
        lr (float, optional): learning rate. (default: 1e-3)
        betas (Tuple[float, float, flot], optional): coefficients used for computing 
            running averages of gradient and its norm. (default: (0.98, 0.92, 0.99))
        eps (float, optional): term added to the denominator to improve 
            numerical stability. (default: 1e-8)
        weight_decay (float, optional): decoupled weight decay (L2 penalty) (default: 0)
        max_grad_norm (float, optional): value used to clip 
            global grad norm (default: 0.0 no clip)
        no_prox (bool): how to perform the decoupled weight decay (default: False)
    """

    def __init__(self, params, lr=1e-3, betas=(0.98, 0.92, 0.99), eps=1e-8,
                 weight_decay=0.0, max_grad_norm=0.0, no_prox=False):
        if not 0.0 <= max_grad_norm:
            raise ValueError("Invalid Max grad norm: {}".format(max_grad_norm))
        if not 0.0 <= lr:
            raise ValueError("Invalid learning rate: {}".format(lr))
        if not 0.0 <= eps:
            raise ValueError("Invalid epsilon value: {}".format(eps))
        if not 0.0 <= betas[0] < 1.0:
            raise ValueError("Invalid beta parameter at index 0: {}".format(betas[0]))
        if not 0.0 <= betas[1] < 1.0:
            raise ValueError("Invalid beta parameter at index 1: {}".format(betas[1]))
        if not 0.0 <= betas[2] < 1.0:
            raise ValueError("Invalid beta parameter at index 2: {}".format(betas[2]))
        defaults = dict(lr=lr, betas=betas, eps=eps,
                        weight_decay=weight_decay,
                        max_grad_norm=max_grad_norm, no_prox=no_prox)
        super().__init__(params, defaults)

    def __setstate__(self, state):
        super(Adan, self).__setstate__(state)
        for group in self.param_groups:
            group.setdefault('no_prox', False)

    @torch.no_grad()
    def restart_opt(self):
        for group in self.param_groups:
            group['step'] = 0
            for p in group['params']:
                if p.requires_grad:
                    state = self.state[p]
                    # State initialization

                    # Exponential moving average of gradient values
                    state['exp_avg'] = torch.zeros_like(p)
                    # Exponential moving average of squared gradient values
                    state['exp_avg_sq'] = torch.zeros_like(p)
                    # Exponential moving average of gradient difference
                    state['exp_avg_diff'] = torch.zeros_like(p)

    @torch.no_grad()
    def step(self):
        """
            Performs a single optimization step.
        """
        if self.defaults['max_grad_norm'] > 0:
            device = self.param_groups[0]['params'][0].device
            global_grad_norm = torch.zeros(1, device=device)

            max_grad_norm = torch.tensor(self.defaults['max_grad_norm'], device=device)
            for group in self.param_groups:

                for p in group['params']:
                    if p.grad is not None:
                        grad = p.grad
                        global_grad_norm.add_(grad.pow(2).sum())

            global_grad_norm = torch.sqrt(global_grad_norm)

            clip_global_grad_norm = torch.clamp(max_grad_norm / (global_grad_norm + group['eps']), max=1.0)
        else:
            clip_global_grad_norm = 1.0

        for group in self.param_groups:
            beta1, beta2, beta3 = group['betas']
            # assume same step across group now to simplify things
            # per parameter step can be easily support by making it tensor, or pass list into kernel
            if 'step' in group:
                group['step'] += 1
            else:
                group['step'] = 1

            bias_correction1 = 1.0 - beta1 ** group['step']

            bias_correction2 = 1.0 - beta2 ** group['step']

            bias_correction3 = 1.0 - beta3 ** group['step']

            for p in group['params']:
                if p.grad is None:
                    continue

                state = self.state[p]
                if len(state) == 0:
                    state['exp_avg'] = torch.zeros_like(p)
                    state['exp_avg_sq'] = torch.zeros_like(p)
                    state['exp_avg_diff'] = torch.zeros_like(p)

                grad = p.grad.mul_(clip_global_grad_norm)
                if 'pre_grad' not in state or group['step'] == 1:
                    state['pre_grad'] = grad

                copy_grad = grad.clone()

                exp_avg, exp_avg_sq, exp_avg_diff = state['exp_avg'], state['exp_avg_sq'], state['exp_avg_diff']
                diff = grad - state['pre_grad']

                update = grad + beta2 * diff
                exp_avg.mul_(beta1).add_(grad, alpha=1 - beta1)  # m_t
                exp_avg_diff.mul_(beta2).add_(diff, alpha=1 - beta2)  # diff_t
                exp_avg_sq.mul_(beta3).addcmul_(update, update, value=1 - beta3)  # n_t

                denom = ((exp_avg_sq).sqrt() / math.sqrt(bias_correction3)).add_(group['eps'])
                update = ((exp_avg / bias_correction1 + beta2 * exp_avg_diff / bias_correction2)).div_(denom)

                if group['no_prox']:
                    p.data.mul_(1 - group['lr'] * group['weight_decay'])
                    p.add_(update, alpha=-group['lr'])
                else:
                    p.add_(update, alpha=-group['lr'])
                    p.data.div_(1 + group['lr'] * group['weight_decay'])

                state['pre_grad'] = copy_grad

N_ITER = 10

def test_original_vs_new():
    model = resnet18()
    model_new = deepcopy(model)
    model_original = deepcopy(model)
    opt_original = AdanOriginal(model_original.parameters())
    opt_new = Adan(model_new.parameters())
    
    for _ in range(N_ITER):
        inp = torch.randn(2, 3, 64, 64)

        opt_original.zero_grad()
        model_original(inp).sum().backward()
        opt_original.step()
        
        opt_new.zero_grad()
        model_new(inp).sum().backward()
        opt_new.step()

    # results of both optimizers should be identical
    zipped_params = zip(model_new.parameters(), model_original.parameters())
    assert all([torch.allclose(p1, p2) for p1, p2 in zipped_params])

    # verify that we actually changed the weights of the model 
    zipped_params = zip(model.parameters(), model_original.parameters())
    assert not any([torch.allclose(p1, p2) for p1, p2 in zipped_params])

def test_single_vs_multi_tensor():
    model = resnet18()
    model_single = deepcopy(model)
    model_multi = deepcopy(model)
    opt_single = Adan(model_single.parameters())
    opt_multi = Adan(model_multi.parameters(), foreach=True)
    
    for _ in range(N_ITER):
        inp = torch.randn(2, 3, 64, 64)

        opt_single.zero_grad()
        model_single(inp).sum().backward()
        opt_single.step()
        
        opt_multi.zero_grad()
        model_multi(inp).sum().backward()
        opt_multi.step()

    # results of both optimizers should be identical
    zipped_params = zip(model_single.parameters(), model_multi.parameters())
    assert all([torch.allclose(p1, p2) for p1, p2 in zipped_params])

def test_single_vs_multi_tensor_no_proxy():
    model = resnet18()
    model_single = deepcopy(model)
    model_multi = deepcopy(model)
    opt_single = Adan(model_single.parameters(), no_prox=True)
    opt_multi = Adan(model_multi.parameters(), no_prox=True, foreach=True)
    
    for _ in range(N_ITER):
        inp = torch.randn(2, 3, 64, 64)

        opt_single.zero_grad()
        model_single(inp).sum().backward()
        opt_single.step()
        
        opt_multi.zero_grad()
        model_multi(inp).sum().backward()
        opt_multi.step()

    # results of both optimizers should be identical
    zipped_params = zip(model_single.parameters(), model_multi.parameters())
    assert all([torch.allclose(p1, p2) for p1, p2 in zipped_params])


if __name__ == "__main__":
    from torch.utils.benchmark import Compare, Timer

    model = resnet18().cuda().half()
    inp = torch.randn(8, 3, 256, 256).cuda().half()
    res = []

    single_tensor_model = deepcopy(model)
    opt_single = Adan(single_tensor_model.parameters())
    opt_single.zero_grad()
    single_tensor_model(inp).mean().backward()
    opt_single.step()
    with torch.cuda.amp.autocast():
        res.append(Timer("opt.step()", globals=dict(opt=opt_single), label="label", sub_label="single", description="adan").blocked_autorange())

    multi_tensor_model = deepcopy(model)
    opt_multi = Adan(multi_tensor_model.parameters(), foreach=True)
    opt_multi.zero_grad()
    multi_tensor_model(inp).mean().backward()
    opt_multi.step()
    with torch.cuda.amp.autocast():
        res.append(Timer("opt.step()", globals=dict(opt=opt_multi), label="label", sub_label="multi", description="adan").blocked_autorange())

    single_tensor_adam_model = deepcopy(model)
    opt_single_adam = torch.optim.Adam(single_tensor_adam_model.parameters(), lr=1e-5)
    opt_single_adam.zero_grad()
    single_tensor_adam_model(inp).mean().backward()
    opt_single_adam.step()
    with torch.cuda.amp.autocast():
        res.append(Timer("opt.step()", globals=dict(opt=opt_single_adam), label="label", sub_label="single", description="adam").blocked_autorange())

    multi_tensor_adam_model = deepcopy(model)
    opt_multi_adam = torch.optim.Adam(multi_tensor_adam_model.parameters(), lr=1e-5, foreach=True)
    opt_multi_adam.zero_grad()
    multi_tensor_adam_model(inp).mean().backward()
    opt_multi_adam.step()
    with torch.cuda.amp.autocast():
        res.append(Timer("opt.step()", globals=dict(opt=opt_multi_adam), label="label", sub_label="multi", description="adam").blocked_autorange())

    Compare(res).print()