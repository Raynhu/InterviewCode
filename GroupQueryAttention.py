import torch
import torch.nn as nn
import math

class GroupQueryAttention(nn.Module):
    def __init__(self, hidden_dim, nums_head, nums_key_value_head):
        super().__init__()
        assert hidden_dim % nums_head == 0
        assert nums_head % nums_key_value_head == 0

        self.hidden_dim = hidden_dim
        self.nums_head = nums_head
        self.nums_key_value_head = nums_key_value_head
        self.head_dim = hidden_dim // nums_head

        self.q_proj = nn.Linear(hidden_dim, nums_head * self.head_dim)
        self.k_proj = nn.Linear(hidden_dim, self.nums_key_value_head * self.head_dim)
        self.v_proj = nn.Linear(hidden_dim, self.nums_key_value_head * self.head_dim)
        self.o_proj = nn.Linear(hidden_dim, hidden_dim)

    def forward(self, X, attention_mask=None):
        batch_size, seq, _ = X.size()

        # qkv projection
        q = self.q_proj(X) # shape = (batch_size, seq, hidden_dim)
        k = self.k_proj(X)
        v = self.v_proj(X)

        # attention weight 目标shape是 (batch, num_head, seq, head_dim)
        q = q.view(batch_size, seq, self.nums_head, self.head_dim) # shape = (batch_size, seq, nums_head, head_dim)
        k = k.view(batch_size, seq, self.nums_key_value_head, self.head_dim) # shape = (batch_size, seq, nums_key_value_head, head_dim)
        v = v.view(batch_size, seq, self.nums_key_value_head, self.head_dim) # shape = (batch_size, seq, nums_key_value_head, head_dim)

        q = q.transpose(1, 2) # shape = (batch_size, nums_head, seq, head_dim)
        k = k.transpose(1, 2) # shape = (batch_size, nums_key_value_head, seq, head_dim)
        v = v.transpose(1, 2) # shape = (batch_size, nums_key_value_head, seq, head_dim)

        # k, v 进行广播
        k = k.repeat_interleave(self.nums_head // self.nums_key_value_head, dim=1) # shape = (batch_size, nums_head, seq, head_dim)
        v = v.repeat_interleave(self.nums_head // self.nums_key_value_head, dim=1) # shape = (batch_size, nums_head, seq, head_dim)

        attention_score = (q @ k.transpose(2, 3)) // math.sqrt(self.head_dim)
        attention_weight = torch.softmax(attention_score, dim=-1)

        output = attention_weight @ v # shape = (batch_size, nums_head, seq, head_dim)
        output = output.transpose(1, 2).contiguous()

        final_output = self.o_proj(output.view(batch_size, seq, -1))# shape = (batch_size, seq, hidden_dim)
        return final_output
    

def main():
    x = torch.rand(3, 2, 128)
    net = GroupQueryAttention(128, 8, 4)
    print(net(x).shape)
    return

if __name__ == "__main__":
    main()



        
        
