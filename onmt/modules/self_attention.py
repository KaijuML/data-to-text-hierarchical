"""
Custom reimplementation of torch.nn.MultiHeadAttention

It's actually the same module, with more or less flewibility at times,
and a more flexible use of the mask (different mask per element of the batch)
"""
from torch._jit_internal import weak_module, weak_script_method
from torch.nn.init import constant_
from torch.nn.parameter import Parameter
from torch.nn.init import xavier_uniform_
from torch.nn import functional as F
from onmt.utils.misc import tile
from onmt.modules import GatedLinear
import torch


@weak_module
class MultiHeadSelfAttention(torch.nn.Module):
    """
    if glu_depth is not zero, we use GatedLinear layers instead of regular layers.
    """
    def __init__(self, embed_dim, num_heads, dropout=0., glu_depth=0, bias=True):
        super().__init__()
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.dropout = dropout
        self.head_dim = embed_dim // num_heads
        msg = "embed_dim must be divisible by num_heads, got {} and {}"
        assert self.head_dim * num_heads == self.embed_dim, msg.format(embed_dim, num_heads)
        self.scaling = self.head_dim ** -0.5

        self.in_proj_weight = Parameter(torch.empty(3 * embed_dim, embed_dim))
        if bias:
            self.in_proj_bias = Parameter(torch.empty(3 * embed_dim))
        else:
            self.register_parameter('in_proj_bias', None)
        self.out_proj = torch.nn.Linear(embed_dim, embed_dim, bias=bias)

        # Gated Linear Unit
        self._use_glu = isinstance(glu_depth, int) and glu_depth > 0
        if self._use_glu:
            if not self.head_dim % pow(2, glu_depth) == 0:
                raise ValueError('When using GLU you need to use a head_dim that is '
                                 'a multiple of two to the power glu_depth. '
                                 f'Got {self.head_dim} % 2^{glu_depth} != 0')
            glu_out_dim = self.head_dim // pow(2, glu_depth)
            self.key_glu = GatedLinear(self.head_dim, glu_out_dim, glu_depth)
            self.query_glu = GatedLinear(self.head_dim, glu_out_dim, glu_depth)
        
        self._reset_parameters()

    def _reset_parameters(self):
        xavier_uniform_(self.in_proj_weight[:self.embed_dim, :])
        xavier_uniform_(self.in_proj_weight[self.embed_dim:(self.embed_dim * 2), :])
        xavier_uniform_(self.in_proj_weight[(self.embed_dim * 2):, :])

        xavier_uniform_(self.out_proj.weight)
        if self.in_proj_bias is not None:
            constant_(self.in_proj_bias, 0.)
            constant_(self.out_proj.bias, 0.)

    @weak_script_method
    def forward(self, input, attn_mask=None):
        """
        Inputs of forward function
            input: [target length, batch size, embed dim]
            attn_mask [(batch size), sequence_length, sequence_length]

        Outputs of forward function
            attn_output: [target length, batch size, embed dim]
            attn_output_weights: [batch size, target length, sequence length]
        """

        seq_len, bsz, embed_dim = input.size()
        assert embed_dim == self.embed_dim

        # self-attention
        q, k, v = F.linear(input, self.in_proj_weight, self.in_proj_bias).chunk(3, dim=-1)
        q *= self.scaling
        
        # Cut q, k, v in num_heads part
        q = q.contiguous().view(seq_len, bsz * self.num_heads, self.head_dim).transpose(0, 1)
        k = k.contiguous().view(-1, bsz * self.num_heads, self.head_dim).transpose(0, 1)
        v = v.contiguous().view(-1, bsz * self.num_heads, self.head_dim).transpose(0, 1)
        
        #  Gated Linear Unit
        if self._use_glu:
            q = self.query_glu(q)
            k = self.key_glu(k)
        
        # batch matrix multply query against key
        # attn_output_weights is [bsz * num_heads, seq_len, seq_len]
        attn_output_weights = torch.bmm(q, k.transpose(1, 2))
        
        assert list(attn_output_weights.size()) == [bsz * self.num_heads, seq_len, seq_len]

        if attn_mask is not None:
            if attn_mask.dim() == 2: 
                # We use the same mask for each item in the batch
                attn_mask = attn_mask.unsqueeze(0)
            elif attn_mask.dim() == 3:
                # Each item in the batch has its own mask.
                # We need to inflate the mask to go with all heads
                attn_mask = tile(attn_mask, count=self.num_heads, dim=0)
            else:
                # Don't known what we would be doing here...
                raise RuntimeError(f'Wrong mask dim: {attn_mask.dim()}')
            
            # The mask should be either 0 of -inf to go with softmax
            attn_output_weights += attn_mask

        attn_output_weights = F.softmax(
            attn_output_weights.float(), dim=-1,
            dtype=torch.float32 if attn_output_weights.dtype == torch.float16 else attn_output_weights.dtype)
        attn_output_weights = F.dropout(attn_output_weights, p=self.dropout, training=self.training)

        attn_output = torch.bmm(attn_output_weights, v)
        assert list(attn_output.size()) == [bsz * self.num_heads, seq_len, self.head_dim]
        attn_output = attn_output.transpose(0, 1).contiguous().view(seq_len, bsz, embed_dim)
        attn_output = self.out_proj(attn_output)

        # average attention weights over heads
        attn_output_weights = attn_output_weights.view(bsz, self.num_heads, seq_len, seq_len)
        attn_output_weights = attn_output_weights.sum(dim=1) / self.num_heads

        return attn_output, attn_output_weights