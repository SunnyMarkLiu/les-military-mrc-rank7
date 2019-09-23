"""
contains some common network modules.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class Highway(nn.Module):
    """
    Implements Highway Networks(https://arxiv.org/pdf/1505.00387.pdf)
    y = g * x + (1 - g) * f(A(x)) where `A` is a linear transformation, `f` is an element-wise
    non-linearity, `g` is an element-wise gate computed as `sigmoid(B(x))`.
    """

    def __init__(self,
                 input_dim,
                 num_layers=1,
                 activation=F.relu):
        super(Highway, self).__init__()
        self.activation = activation
        self.layers = nn.ModuleList([nn.Linear(input_dim, input_dim * 2) for _ in range(num_layers)])
        for layer in self.layers:
            # We should bias the highway layer to just carry its input forward.  We do that by
            # setting the bias on `B(x)` to be positive, because that means `g` will be biased to
            # be high, so we will carry the input forward.  The bias on `B(x)` is the second half
            # of the bias vector in each Linear layer.
            layer.bias[input_dim:].data.fill_(1)

    def forward(self, inputs):
        current_input = inputs
        for layer in self.layers:
            projected_input = layer(current_input)
            linear_part = current_input
            nonlinear_part, gate = projected_input.chunk(2, dim=-1)
            nonlinear_part = self.activation(nonlinear_part)
            gate = torch.sigmoid(gate)
            current_input = gate * linear_part + (1 - gate) * nonlinear_part
        return current_input
