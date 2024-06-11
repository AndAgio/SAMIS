import torch
from .base import FlatMetric


class EpsilonFlatness(FlatMetric):
    def __init__(self, model: torch.nn.Module, loss_function: torch.nn.Module, device: str = 'cpu', rho: float = 0.05, adaptive: bool = True):
        super().__init__(model=model, loss_function=loss_function, device=device, mode='epsilon')
        assert rho >= 0.0, f"Invalid rho, should be non-negative: {rho}"
        self.rho = rho
        self.adaptive = adaptive
        self.equalizer = 1e-12

    def compute(self, inputs, targets):
        inputs = inputs.to(self.device)
        targets = targets.to(self.device)
        inputs.requires_grad_()
        loss_z = self.loss_function(self.model.eval()(inputs), targets)                
        loss_z.backward(torch.ones(targets.size()).to(self.device))
        grad_norm = self._grad_norm()
        scale = self.rho / (grad_norm + 1e-12)
        epsilons = []
        for p in self.model.parameters():
            if p.grad is None: continue
            epsilon = (torch.pow(p, 2) if self.adaptive else 1.0) * p.grad * scale.to(p)
            epsilons.append(epsilon.detach().cpu())
        epsilon_norm = torch.norm(torch.stack([ep.norm(p=2) for ep in epsilons]), p=2).detach().cpu().item()
        # print('epsilon_norm: {}'.format(epsilon_norm))
        self.model.zero_grad()
        return epsilon_norm

    def _grad_norm(self):
        norm = torch.norm(
                    torch.stack([
                        ((torch.abs(p) if self.adaptive else 1.0) * p.grad).norm(p=2).to(self.device)
                        for p in self.model.parameters() if p.grad is not None
                    ]),
                    p=2
                )
        return norm
