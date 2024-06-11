import torch
from .base import FlatMetric


class HessianTraceCurvature(FlatMetric):
    def __init__(self, model, loss_function, device='cpu', h = 3.):
        super().__init__(model=model, loss_function=loss_function, device=device, mode='hessian')
        self.h = h

    def _find_z(self, inputs, targets):
        '''
        Finding the direction in the regularizer
        '''
        inputs.requires_grad_()
        loss_z = self.loss_function(self.model.eval()(inputs), targets)
        loss_z.backward(torch.ones(targets.size()).to(self.device))
        grad = inputs.grad.data + 0.0
        z = torch.sign(grad).detach() + 0.
        z = (z+1e-7) / (z.reshape(z.size(0), -1).norm(dim=1)[:, None, None, None]+1e-7)
        HessianTraceCurvature.zero_gradients(inputs)
        self.model.zero_grad()
        return z
    
    def compute(self, inputs, targets):
        '''
        Code taken from the regularizer term in CURE
        '''
        # Compute z from input and targets
        z = self._find_z(inputs, targets)
        # Compute curvature from z
        inputs.requires_grad_()
        outputs_pos = self.model.eval()(inputs + 1.*(self.h) * z)
        outputs_orig = self.model.eval()(inputs)
        loss_pos = self.loss_function(outputs_pos, targets)
        loss_orig = self.loss_function(outputs_orig, targets)
        grad_diff = torch.autograd.grad((loss_pos-loss_orig), inputs, grad_outputs=torch.ones(targets.size()).to(self.device),
                                        create_graph=True)[0]
        gamma = grad_diff.reshape(grad_diff.size(0), -1).norm(dim=1).detach().cpu().item()
        self.model.zero_grad()
        return gamma