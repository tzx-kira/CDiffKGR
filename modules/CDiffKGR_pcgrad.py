import random
import torch


class PCGrad:
    """
    内存友好的PCGrad优化器包装器。

    该实现使用torch.autograd.grad分别取得各任务梯度；当两个任务在同一参数上的
    梯度点积为负时，对两侧梯度同时进行正交投影，然后按reduction聚合。
    与原实现相比，不再只单向投影第一个任务，因此不会系统性偏向某一个损失。
    """

    def __init__(self, optimizer, reduction='mean'):
        if reduction not in {'mean', 'sum'}:
            raise ValueError("reduction must be either 'mean' or 'sum'.")
        self._optim = optimizer
        self._reduction = reduction

    @property
    def optimizer(self):
        return self._optim

    def zero_grad(self):
        return self._optim.zero_grad(set_to_none=True)

    def step(self):
        return self._optim.step()

    def state_dict(self):
        return self._optim.state_dict()

    def load_state_dict(self, state_dict):
        return self._optim.load_state_dict(state_dict)

    def _trainable_parameters(self):
        parameters = []
        for group in self._optim.param_groups:
            for parameter in group['params']:
                if parameter.requires_grad:
                    parameters.append(parameter)
        return parameters

    @staticmethod
    def _clone_or_zero(gradient, parameter):
        if gradient is None:
            return torch.zeros_like(parameter)
        return gradient.detach().clone()

    def pc_backward(self, objectives):
        """
        对多个目标执行梯度冲突投影，并把最终梯度写入parameter.grad。

        当前CDiffKGR中推荐损失与生成KG去噪损失拥有独立前向图、共享叶参数，
        因而每个目标可以独立调用autograd.grad，不需要保留整张计算图。
        """
        valid_objectives = [
            objective for objective in objectives
            if isinstance(objective, torch.Tensor) and objective.requires_grad
        ]
        if len(valid_objectives) == 0:
            self.zero_grad()
            return

        parameters = self._trainable_parameters()
        if len(parameters) == 0:
            return

        task_grads = []
        for objective in valid_objectives:
            gradients = torch.autograd.grad(
                objective,
                parameters,
                retain_graph=False,
                create_graph=False,
                allow_unused=True
            )
            task_grads.append([
                self._clone_or_zero(gradient, parameter)
                for gradient, parameter in zip(gradients, parameters)
            ])

        projected_grads = self._project_conflicting_per_param(task_grads)
        self.zero_grad()
        for parameter, gradient in zip(parameters, projected_grads):
            parameter.grad = gradient

        del task_grads
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    def _aggregate(self, gradients):
        final_gradient = torch.zeros_like(gradients[0])
        for gradient in gradients:
            final_gradient.add_(gradient)
        if self._reduction == 'mean':
            final_gradient.div_(len(gradients))
        return final_gradient

    def _project_two_tasks(self, grad_a, grad_b):
        dot = torch.sum(grad_a * grad_b)
        if dot >= 0:
            return self._aggregate([grad_a, grad_b])

        norm_a_sq = torch.sum(grad_a * grad_a)
        norm_b_sq = torch.sum(grad_b * grad_b)
        eps = torch.finfo(grad_a.dtype).eps if grad_a.is_floating_point() else 1e-12

        if norm_a_sq <= eps or norm_b_sq <= eps:
            return self._aggregate([grad_a, grad_b])

        # 对称投影：两个任务都移除指向对方冲突方向的分量。
        grad_a_projected = grad_a - dot / (norm_b_sq + eps) * grad_b
        grad_b_projected = grad_b - dot / (norm_a_sq + eps) * grad_a
        return self._aggregate([grad_a_projected, grad_b_projected])

    def _project_many_tasks(self, gradients):
        projected = [gradient.clone() for gradient in gradients]
        for task_index in range(len(projected)):
            other_indices = list(range(len(projected)))
            random.shuffle(other_indices)
            for other_index in other_indices:
                if task_index == other_index:
                    continue
                current = projected[task_index]
                reference = gradients[other_index]
                dot = torch.sum(current * reference)
                if dot < 0:
                    norm_sq = torch.sum(reference * reference)
                    eps = (
                        torch.finfo(current.dtype).eps
                        if current.is_floating_point() else 1e-12
                    )
                    if norm_sq > eps:
                        projected[task_index] = (
                            current - dot / (norm_sq + eps) * reference
                        )
        return self._aggregate(projected)

    def _project_conflicting_per_param(self, task_grads):
        num_tasks = len(task_grads)
        num_params = len(task_grads[0])
        final_grads = []

        for param_index in range(num_params):
            gradients = [
                task_grads[task_index][param_index]
                for task_index in range(num_tasks)
            ]
            if num_tasks == 1:
                final_gradient = gradients[0]
            elif num_tasks == 2:
                final_gradient = self._project_two_tasks(
                    gradients[0], gradients[1]
                )
            else:
                final_gradient = self._project_many_tasks(gradients)
            final_grads.append(final_gradient)

        return final_grads