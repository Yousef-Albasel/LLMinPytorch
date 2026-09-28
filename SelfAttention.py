import torch

class SelfAttention(torch.nn.Module):
    def __init__(self, d_in, d_out, qkv_bias):
        super().__init__()
        self.Wq = torch.nn.Linear(d_in,d_out, bias = qkv_bias)
        self.Wk = torch.nn.Linear(d_in,d_out, bias = qkv_bias)
        self.Wv = torch.nn.Linear(d_in,d_out, bias = qkv_bias)

    def forward(self, x):
        Q = x @ self.Wq
        K = x @ self.Wk
        V = x @ self.Wv
        attn_scores = Q @ K.T
        normalized_attn_scores = torch.softmax(attn_scores/K.shape[-1]**0.5,dim=-1)
        context_length = normalized_attn_scores.shape[0]
        mask_simple = torch.tril(torch.ones(context_length, context_length))
        attn_weights = mask_simple * normalized_attn_scores
        row_sums = attn_weights.sum(dim=-1, keepdim=True)
        masked_simple_norm = attn_weights / row_sums
        context_vector = masked_simple_norm @ V

        return context_vector
