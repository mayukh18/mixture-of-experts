import torch
import torch.nn as nn
from torch.distributions.normal import Normal
import numpy as np

from moe import MLP

class ParallelExperts(nn.Module):
    def __init__(self, input_size, output_size, num_experts, hidden_size):
        super(ParallelExperts, self).__init__()
        self.num_experts = num_experts
        self.output_size = output_size
        self.input_size = input_size
        self.hidden_size = hidden_size

        # instantiate experts
        self.experts = nn.ModuleList([MLP(self.input_size, self.output_size, self.hidden_size) for i in range(self.num_experts)])

        self.final = nn.Linear(self.output_size, self.output_size)

    def forward(self, x):
        outputs = [self.experts[i](x) for i in range(self.num_experts)]
        outputs = torch.stack([self.final(out) for out in outputs], dim=2)
        return torch.mean(outputs, dim=-1), 0


class CascadedExperts(nn.Module):
    def __init__(self, input_size, output_size, num_experts, hidden_size):
        super(CascadedExperts, self).__init__()
        self.num_experts = num_experts
        self.output_size = output_size
        self.input_size = input_size
        self.hidden_size = hidden_size

        # instantiate experts
        self.experts = nn.Sequential(
            MLP(self.input_size, self.hidden_size, self.hidden_size), \
            nn.ModuleList([MLP(self.hidden_size, self.output_size, self.hidden_size) for i in range(self.num_experts-1)
                           ]))

        self.final = nn.Linear(self.output_size, self.output_size)

    def forward(self, x):
        outputs = self.experts(x)
        return self.final(outputs), 0


