from ..utils.misc import aeq
from .sparse_activations import sparsemax
from torch.nn.utils.rnn import pad_sequence

import torch
import onmt



class ContainsNaN(Exception):
    pass


def _check_for_nan(tensor, msg=''):
    if (tensor!=tensor).any():
        raise ContainsNaN(msg)
        
def _check_sizes(tensor, *sizes):
    for dim, (s, _s) in enumerate(zip(tensor.shape, sizes)):
        assert s == _s, f'dim {dim} are not of equal sizes'


class AttentionScorer(torch.nn.Module):
    """
    dim_query is dim of the decoder
    dim_key is dim of the encoder output
    """
    def __init__(self, dim, attn_type):
        super().__init__()
        
        if isinstance(dim, tuple):
            assert len(dim) == 2
            assert isinstance(dim[0], int)
            assert isinstance(dim[1], int)
            assert attn_type != 'dot'
            self.dim_query = dim[0]
            self.dim_key = dim[1]
        elif isinstance(dim, int):
            self.dim_query = dim
            self.dim_key = dim
        else:
            raise ValueError('dim should a one or two ints')
            
            
        self.attn_type = attn_type
        
        if self.attn_type == "general":
            self.linear_in = torch.nn.Linear(self.dim_query,
                                             self.dim_key,
                                             bias=False)
        elif self.attn_type == "mlp":
            self.linear_context = torch.nn.Linear(self.dim_key,
                                                  self.dim_key,
                                                  bias=False)
            self.linear_query = torch.nn.Linear(self.dim_query,
                                                self.dim_key,
                                                bias=True)
            self.v = torch.nn.Linear(self.dim_key, 1, bias=False)
        
    def forward(self, h_t, h_s):
        """
        Args:
          h_t (FloatTensor): sequence of queries ``(batch, tgt_len, dim)``
          h_s (FloatTensor): sequence of sources ``(batch, src_len, dim``

        Returns:
          FloatTensor: raw attention scores (unnormalized) for each src index
            ``(batch, tgt_len, src_len)``
        """

        # Check input sizes
        src_batch, src_len, src_dim = h_s.size()
        tgt_batch, tgt_len, tgt_dim = h_t.size()
        
        aeq(src_batch, tgt_batch)
        aeq(src_dim, self.dim_key)
        aeq(tgt_dim, self.dim_query)
        
        if self.attn_type in ["general", "dot"]:
            if self.attn_type == "general":
                h_t_ = h_t.view(tgt_batch * tgt_len, tgt_dim)
                h_t_ = self.linear_in(h_t_)
                h_t = h_t_.view(tgt_batch, tgt_len, src_dim)
            h_s_ = h_s.transpose(1, 2)
            # (batch, t_len, d) x (batch, d, s_len) --> (batch, t_len, s_len)
            # where d is self.dim_key
            return torch.bmm(h_t, h_s_)
        else:
            wq = self.linear_query(h_t.view(-1, tgt_dim))
            wq = wq.view(tgt_batch, tgt_len, 1, src_dim)
            wq = wq.expand(tgt_batch, tgt_len, src_len, src_dim)

            uh = self.linear_context(h_s.contiguous().view(-1, src_dim))
            uh = uh.view(src_batch, 1, src_len, src_dim)
            uh = uh.expand(src_batch, tgt_len, src_len, src_dim)

            # (batch, t_len, s_len, d)
            wquh = torch.tanh(wq + uh)

            return self.v(wquh.view(-1, dim)).view(tgt_batch, tgt_len, src_len)

        
class HierarchicalAttention(torch.nn.Module):
    def __init__(self, dim, coverage=False, attn_type="dot",
                 attn_func="softmax", use_pos=True):
        super().__init__()
        assert not coverage
        
        self.ent_size = onmt.ENT_SIZE
        
        self.use_pos = use_pos

        # dims shenanigans. memory_bank should be dim[0]
        if isinstance(dim, tuple):
            assert len(dim) == 2
            assert isinstance(dim[0], int)
            assert isinstance(dim[1], int)
            self.chunks_dim = dim[0]
            self.units_dim = dim[1]
        elif isinstance(dim, int):
            self.chunks_dim = dim
            self.units_dim = dim
        else:
            raise ValueError('dim should be one or two ints')
        
        if attn_func == 'softmax':
            self.attn_func = torch.nn.functional.softmax
        elif attn_func == 'sparsemax':
            self.attn_func = sparsemax
        else:
            raise ValueError("Please select a valid attention function.")
            
        assert attn_type in ["dot", "general", "mlp"], (
            "Please select a valid attention type (got {:s}).".format(
                attn_type))
        self.attn_type = attn_type
        
        self.unit_scorer = AttentionScorer((self.chunks_dim, self.units_dim), 
                                           attn_type)
        self.chunk_scorer = AttentionScorer(self.chunks_dim, attn_type)
        
        # mlp wants it with bias, others no
        self.linear_out = torch.nn.Linear(self.chunks_dim * 2, 
                                          self.chunks_dim,
                                          bias=(attn_type=="mlp"))
        
    def forward(self, source, memory_bank):
        """

        Args:
          source (FloatTensor): query vectors ``(batch, tgt_len, dim)``
          memory_bank (FloatTensor): source vectors ``(batch, src_len, dim)``

        Returns:
          (FloatTensor, FloatTensor):

          * Computed vector ``(tgt_len, batch, dim)``
          * Attention distribtutions for each query
            ``(tgt_len, batch, src_len)``
            
        In this setup, tgt_len will always be equal to one, due to inputfeeding
        """
        
        # assert one step input
        assert source.dim() == 2
        
        source = source.unsqueeze(1)

        # Unpacking memory_bank (we reassign memory_bank to optimize memory
        # and minimize errors when copy/paste exisiting code)
        # we transpose the batch_dim for the scoring compute
        chunks, memory_bank, pos_embs, units_mask, chunk_mask = memory_bank  
        chunks = chunks.transpose(0, 1)
        memory_bank = memory_bank.transpose(0, 1)
        pos_embs = pos_embs.transpose(0, 1)
        units_mask = units_mask.transpose(0, 1)
        chunk_mask = chunk_mask.transpose(0, 1)
        
#         _check_for_nan(chunks)
#         _check_for_nan(memory_bank)
#         _check_for_nan(pos_embs)
        
        # Checks and balances
        batch_size, source_l, dim = memory_bank.size()
        batch_, target_l, dim_ = source.size()
        aeq(batch_size, batch_)
        aeq(dim, dim_)
        aeq(self.chunks_dim, dim)

        # compute attention scores, as in Luong et al.
        # align_units is [batch_size, src_len]
        # align_chunks is [batch_size, 1, n_ents]
        if self.use_pos:
            align_units = self.unit_scorer(source, pos_embs).squeeze(1)
        else:
            align_units = self.unit_scorer(source, memory_bank).squeeze(1)
            
        align_chunks = self.chunk_scorer(source, chunks)
        
        # we compute the softmax first on the unit level
        #   - we reshape so that each row is an entity
        #   - we mask the padding and the <ent> token
        #   - we softmax
        #   - we flatten the scores again
        _check_for_nan(align_units, 'align units scores')  # sanity check (1)
        align_units = align_units.view(batch_size, -1, self.ent_size)
        align_units = align_units.masked_fill(units_mask, float('-inf'))
        _check_for_nan(align_units, 'align units scores filled with -inf')  # sanity check (2)
        
        # tricky block
        # we softmax on the last dim, ie: separatly on each entity
        # However, some entity might be full <pad>, meaning full -inf
        # giving NaN when softmax is computed (dividing by zero)
        # We find those nan and remove them
        align_units = self.attn_func(align_units, -1)  # softmax
        nan_mask = (align_units != align_units).sum(dim=2).ne(0)  # nan != nan
        if nan_mask.sum().item():
            align_units = align_units.masked_fill(nan_mask.unsqueeze(-1), 0)
        _check_for_nan(align_units, 'align units after attn_func')  # sanity check (3)
        
        # we flatten the scores again
        align_units = align_units.view(batch_size, 1, -1)
        
        # Now the second level of attention, on the <ent> tokens
        align_chunks.masked_fill_(chunk_mask, -1)
        align_chunks = self.attn_func(align_chunks, -1)
#         align_chunks = sparsemax(align_chunks, -1)
        _check_for_nan(align_chunks, 'align_chunks after attn_func')

        # To compute the final scores, we weight the unit scores by the chunk
        # score from the chunk to witch they belong. We inflate the chunk scores
        # and simply elementwise multiply.
        # It's easy to see that it remains a proba distribution (ie, sums to 1)
        align_chunks_inflated = align_chunks.repeat_interleave(repeats=self.ent_size, dim=-1)
        align_vectors = align_chunks_inflated * align_units
        
        #print(align_vectors.sum())
        
        # each context vector c_t is the weighted average
        # over all the source hidden states
        c = torch.bmm(align_vectors, memory_bank)

        # concatenate
        concat_c = torch.cat([c, source], 2).view(batch_size*target_l, dim*2)
        attn_h = self.linear_out(concat_c).view(batch_size, target_l, dim)
        if self.attn_type in ["general", "dot"]:
            attn_h = torch.tanh(attn_h)


        attn_h = attn_h.squeeze(1)
        align_vectors = align_vectors.squeeze(1)

        # Check output sizes
        batch_, dim_ = attn_h.size()
        aeq(batch_size, batch_)
        aeq(dim, dim_)
        batch_, source_l_ = align_vectors.size()
        aeq(batch_size, batch_)
        aeq(source_l, source_l_)
    
        ret = {
            '': align_vectors,
            '_align_chunks': align_chunks.squeeze(1),
            '_align_units':align_units.squeeze(1)
        }

        return attn_h, ret

    
