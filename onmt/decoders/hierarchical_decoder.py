"""Same as normal RNNDecoder but using hierarchical attention"""

import torch
from .decoder import RNNDecoderBase
from ..modules import HierarchicalAttention
from ..models.stacked_rnn import StackedLSTM, StackedGRU
from ..utils.rnn_factory import rnn_factory
from ..utils.misc import aeq, nwise, sequence_mask
from torch.nn.utils.rnn import pad_sequence
import onmt


class ContainsNaN(Exception):
    pass


def _check_for_nan(tensor):
    if (tensor!=tensor).any():
        raise ContainsNaN

class HierarchicalRNNDecoder(RNNDecoderBase):
    """Input feeding based decoder.

    See :class:`~onmt.decoders.decoder.RNNDecoderBase` for options.

    Based around the input feeding approach from
    "Effective Approaches to Attention-based Neural Machine Translation"
    :cite:`Luong2015`


    .. mermaid::

       graph BT
          A[Input n-1]
          AB[Input n]
          subgraph RNN
            E[Pos n-1]
            F[Pos n]
            E --> F
          end
          G[Encoder]
          H[memory_bank n-1]
          A --> E
          AB --> F
          E --> H
          G --> H
    """
    def __init__(self, rnn_type, bidirectional_encoder, num_layers,
                 hidden_size, attn_type="general", attn_func="softmax",
                 coverage_attn=False, context_gate=None,
                 copy_attn=False, dropout=0.0, embeddings=None,
                 reuse_copy_attn=False, copy_attn_type="general", use_pos=True):
        super(RNNDecoderBase, self).__init__(
            attentional=attn_type != "none" and attn_type is not None)

        assert not coverage_attn
        self.ent_size = onmt.ENT_SIZE
        
        self.bidirectional_encoder = bidirectional_encoder
        self.num_layers = num_layers
        
        if isinstance(hidden_size, tuple):
            self.hidden_size = hidden_size[0]
            self.units_size = hidden_size[1]
        elif isinstance(hidden_size, int):
            self.hidden_size = hidden_size
            self.units_size = hidden_size
        else:
            raise ValueError('hidden_size should be one or two ints')
        self.embeddings = embeddings
        self.dropout = torch.nn.Dropout(dropout)
        
        # Decoder state
        self.state = {}

        # Build the RNN.
        self.rnn_type = rnn_type
        self.num_layers = num_layers
        self.rnn = self._build_rnn(rnn_type=self.rnn_type,
                                   input_size=self._input_size,
                                   hidden_size=self.hidden_size,
                                   num_layers=self.num_layers,
                                   dropout=dropout)

        # Set up the context gate.
        self.context_gate = None
        if context_gate is not None:
            self.context_gate = context_gate_factory(
                context_gate, self._input_size,
                self.hidden_size, self.hidden_size, self.hidden_size
            )

        # Set up the standard attention.
        self._coverage = coverage_attn
        if not self.attentional:
            if self._coverage:
                raise ValueError("Cannot use coverage term with no attention.")
            self.attn = None
        else:
            self.attn = HierarchicalAttention(
                (self.hidden_size, self.units_size), 
                coverage=coverage_attn, use_pos=use_pos,
                attn_type=attn_type, attn_func=attn_func)

        if copy_attn and not reuse_copy_attn:
            if copy_attn_type == "none" or copy_attn_type is None:
                raise ValueError(
                    "Cannot use copy_attn with copy_attn_type none")
            self.copy_attn = HierarchicalAttention(
                (self.hidden_size, self.units_size),
                attn_type=copy_attn_type, attn_func=attn_func,
                use_pos=use_pos)
        else:
            self.copy_attn = None

        self._reuse_copy_attn = reuse_copy_attn and copy_attn
        if self._reuse_copy_attn and not self.attentional:
            raise ValueError("Cannot reuse copy attention with no attention.")
            
    def init_state(self, src, memory_bank, encoder_final):
        """
        Here we initialize the hidden state of the hierarchical_decoder
        This function only works with the hierarchical_transformer.

        encoder_final is [1, bsz, dim]. We need to:
            - convert it to a tuple if decoder_rnn is LSTM
            - Duplicate it to mimick a multi-layer encoder
        """

        hidden = encoder_final.repeat(self.num_layers, 1, 1)

        self.state["hidden"] = (hidden, hidden) if self.rnn_type == "LSTM" else (hidden, )

        # Init the input feed.
        batch_size = self.state["hidden"][0].size(1)
        h_size = (batch_size, self.hidden_size)
        self.state["input_feed"] = \
            self.state["hidden"][0].data.new(*h_size).zero_().unsqueeze(0)
        self.state["coverage"] = None        
        
#        super().init_state(src, memory_bank, encoder_final)
    
#         num_dirs = 2 if self.bidirectional_encoder else 1
#         def f(hidden):
#             # The encoder hidden is  (layers*directions) x batch x dim
#             tmp_dim, bsz, dim = hidden.shape
#             hidden = hidden.view(-1, num_dirs, bsz, dim)
#             num_layers = hidden.size(0)
#             delta = num_layers - self.num_layers
#             if delta > 0:
#                 return hidden[delta:, ...].view(-1, bsz, dim)
#             elif delta < 0:
#                 for _ in range(delta):
#                     hidden = torch.cat((hidden, hidden[-1].unsqueeze(0)), dim=0)
#                 return hidden.view(-1, bsz, dim)
#             return hidden.view(-1, bsz, dim)
        
#         if isinstance(encoder_final, tuple):
#             hidden = tuple(f(h) for h in encoder_final)
#         else:
#             hidden = f(encoder_final)
        
        
            
    @classmethod
    def from_opt(cls, opt, embeddings, dims=None):
        """Alternate constructor."""
        """
        dims are the dimention of the table embeddings
        It is a tuple of size two (dim_value, dim_pos)
        """
        if dims is None:
            dims = opt.dec_rnn_size
            
        return cls(
            rnn_type=opt.rnn_type,
            bidirectional_encoder=opt.brnn,
            num_layers=opt.dec_layers,
            hidden_size=dims,
            attn_type=opt.global_attention,
            attn_func=opt.global_attention_function,
            coverage_attn=opt.coverage_attn,
            context_gate=opt.context_gate,
            copy_attn=opt.copy_attn,
            dropout=opt.dropout[0] if type(opt.dropout) is list
            else opt.dropout,
            embeddings=embeddings,
            reuse_copy_attn=opt.reuse_copy_attn,
            copy_attn_type=opt.copy_attn_type,
            use_pos=opt.use_pos)

    
    
    def _run_forward_pass(self, tgt, memory_bank, memory_lengths=None):
        """
        memory_bank is a tuple (chunks, units, pos_embs, unit_mask, chunk_mask)
        """
        # Additional args check.
        input_feed = self.state["input_feed"].squeeze(0)
        input_feed_batch, _ = input_feed.size()
        tgt_len, tgt_batch, _ = tgt.size()
        aeq(tgt_batch, input_feed_batch)
        # END Additional args check.

        dec_outs = []
        attns = dict()

        emb = self.embeddings(tgt)
        assert emb.dim() == 3  # len x batch x embedding_dim

        dec_state = self.state["hidden"]
        coverage = self.state["coverage"].squeeze(0) \
            if self.state["coverage"] is not None else None

        # Input feed concatenates hidden state with
        # input at every time step.
        for emb_t in emb.split(1):
            decoder_input = torch.cat([emb_t.squeeze(0), input_feed], 1)
            rnn_output, dec_state = self.rnn(decoder_input, dec_state)
            
            # If the RNN has several layers, we only use the last one to compute
            # the attention scores. In pytorch, the outs of the rnn are:
            #     - rnn_output [seq_len, bsz, n-directions * hidden_size]
            #     - dec_state [n-layers * n-directions, bsz, hidden_size] * 2
            # We unpack the rnn_output on dim 2 and keep the last layer
            
            if self.attentional:
                decoder_output, ret = self.attn(
                    rnn_output,
                    memory_bank)
                
                for postfix, tensor in ret.items():
                    key = 'std' + postfix
                    attns.setdefault(key, list())
                    attns[key].append(tensor)
            else:
                decoder_output = rnn_output
            if self.context_gate is not None:
                # TODO: context gate should be employed
                # instead of second RNN transform.
                decoder_output = self.context_gate(
                    decoder_input, rnn_output, decoder_output
                )
            decoder_output = self.dropout(decoder_output)
            input_feed = decoder_output

            dec_outs += [decoder_output]

            # Update the coverage attention.
            if self._coverage:
                coverage = p_attn if coverage is None else p_attn + coverage
                attns.setdefault('coverage', list())
                attns['coverage'].append(coverage)

            if self.copy_attn is not None:
                _, copy_attn = self.copy_attn(
                    decoder_output, memory_bank)
                for postfix, tensor in copy_attn.items():
                    key = 'copy' + postfix
                    attns.setdefault(key, list())
                    attns[key].append(tensor)
                
        
        # this trick should save memory because torch.stack creates a new
        # object.
        for key in list(attns):
            if key.startswith('std'):
                attns[key] = torch.stack(attns[key])
                if self._reuse_copy_attn:
                    attns[key.replace('std', 'copy')] = attns[key]

        return dec_state, dec_outs, attns

    def _build_rnn(self, rnn_type, input_size,
                   hidden_size, num_layers, dropout):
        assert rnn_type != "SRU", "SRU doesn't support input feed! " \
            "Please set -input_feed 0!"
        stacked_cell = StackedLSTM if rnn_type == "LSTM" else StackedGRU
        return stacked_cell(num_layers, input_size, hidden_size, dropout)

    @property
    def _input_size(self):
        """Using input feed by concatenating input with attention vectors."""
        return self.embeddings.embedding_size + self.hidden_size

    def update_dropout(self, dropout):
        self.dropout.p = dropout
        self.rnn.dropout.p = dropout
        self.embeddings.update_dropout(dropout)