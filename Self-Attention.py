import math
import torch
import torch.nn as nn

import warnings

class SelfAttention(nn.Module):
    def __init__(self, dim) -> None:
        super().__init__()
        self.dim = dim

        self.query_proj = nn.Linear(dim, dim)
        self.key_proj = nn.Linear(dim, dim)
        self.value_proj = nn.Linear(dim, dim)

        self.att_drop = nn.Dropout(0.1)

        self.output_proj = nn.Linear(dim, dim)

    
    def forward(self, X, attention_mask=None):
        Q = self.query_proj(X) # shape = (3, 4, 2)
        K = self.key_proj(X)
        V = self.value_proj(X)

        att_weight = Q @ K.transpose(-1, -2) / math.sqrt(self.dim) # shape = (3, 4, 4)
        if attention_mask is not None:
            att_weight = att_weight.masked_fill(attention_mask == 0, float("-1e20"))

        att_weight = torch.softmax(att_weight, dim=-1)
        print(att_weight)

        att_weight = self.att_drop(att_weight)

        output = att_weight @ V
        return self.output_proj(output)
    
    
def main():
    X = torch.rand(3, 4, 2)
    b = torch.tensor(
        [
            [1, 1, 1, 0],
            [1, 1, 0, 0],
            [1, 0, 0, 0]
        ]
    ) # shape = (3, 4)

    print(b.shape)
    mask = b.unsqueeze(dim=1).repeat(1, 4, 1) # shape from (3, 4) => (3, 1, 4) = > (3, 4, 4)
    print(mask.shape)
    net = SelfAttention(2)
    print(net(X, mask).shape)

if __name__ == "__main__":
    main()