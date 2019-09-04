import torch.nn as nn


class VariationalDropout(nn.Module):
    """Variational Dropout presented in https://arxiv.org/pdf/1512.05287.pdf"""

    def __init__(self, p, batch_first=True):
        super().__init__()
        self.batch_first = batch_first
        self.dropout = nn.Dropout(p)

    def forward(self, x):
        if not self.training:
            return x
        if self.batch_first:
            mask = x.new_ones(x.size(0), 1, x.size(2), requires_grad=False)
        else:
            mask = x.new_ones(1, x.size(1), x.size(2), requires_grad=False)
        return self.dropout(mask) * x
