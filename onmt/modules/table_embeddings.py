import torch


class TableEmbeddings(torch.nn.Module):
    """
    Now that I think about it, we can do more efficiently than rewritting the
    onmt module. I will in the future but for now this code works as is,
    so I won't chance breaking it!
    
    These embeddings follow the table structure: a table is an unordered set
    of tuple (pos, value) where pos can be viewed as column name. As
    such, TableEmbeddings' forward returns embeddings for pos and value.
    Furthermore, the value embedding can be merged with the pos embedding.
    
    Most argument names are not very fitting but stay the same 
    as onmt.modules.Embeddings
    """

    def __init__(self,
                 word_vec_size,  # dim of the value embeddings
                 word_vocab_size,  # size of the value vocabulary
                 word_padding_idx,  # idx of <pad>
                 feat_vec_size,  # dim of the pos embeddings
                 feat_vec_exponent, # instead of feat_vec_size
                 feat_vocab_size,  # size of the pos vocabulary
                 feat_padding_idx,  # idx of <pad>
                 merge="concat",  # decide to merge the pos and value
                 merge_activation='ReLU',  # used if merge is mlp
                 dropout=0,
                 ent_idx=None):
        
        super().__init__()
        
        assert ent_idx is not None
        self.ent_idx = ent_idx
        
        self.word_padding_idx = word_padding_idx
        self.word_vec_size = word_vec_size
        
        if feat_vec_size < 0:
            if not 0 < feat_vec_exponent <= 1:
                raise ValueError('feat_vec_exponent should be between 0 and 1')
            feat_vec_size = int(feat_vocab_size ** feat_vec_exponent)
        
        self.value_embeddings = torch.nn.Embedding(word_vocab_size,
                                   word_vec_size, padding_idx=word_padding_idx)
        self.pos_embeddings = torch.nn.Embedding(feat_vocab_size,
                                   feat_vec_size, padding_idx=feat_padding_idx)
        
        self._merge = merge
        if merge is None:
            self.embedding_size = self.word_vec_size
        elif merge == 'concat':
            self.embedding_size = self.word_vec_size + self.feat_vec_size
        elif merge == 'sum':
            assert self.word_vec_size == self.feat_vec_size
            self.embedding_size = self.word_vec_size
        elif merge == 'mlp':
            self.embedding_size = self.word_vec_size
            val_dim = self.value_embeddings.embedding_dim
            pos_dim = self.pos_embeddings.embedding_dim
            in_dim = val_dim + pos_dim
            self.merge = torch.nn.Linear(in_dim, val_dim)
            
            if merge_activation is None:
                self.activation = None
            elif merge_activation == 'ReLU':
                self.activation = torch.nn.ReLU()
            elif merge_activation == 'Tanh':
                self.activation = torch.nn.Tanh()
            else:
                raise ValueError(f'Unknown activation {merge_activation}')
        else:
            raise ValueError('merge should be one of [None|concat|sum|mlp]')
            
        
    @property
    def word_lut(self):
        """Word look-up table.""" 
        return self.value_embeddings

    def load_pretrained_vectors(self, emb_file):
        """
        place holder for onmt compatibility
        """
        if emb_file:
            raise NotImplementedError
    
    def forward(self, inputs):
        # unpack the inputs as cell values and pos (column name)
        values, pos = [item.squeeze(2) for item in inputs.split(1, dim=2)]
        
        # embed them separatly and maybe merge them
        values = self.value_embeddings(values)
        pos = self.pos_embeddings(pos)
        
        if self._merge is None:
            return values, pos
        if self._merge == 'sum':
            values = values + pos
            return values, pos
        
        values = torch.cat((values, pos), 2)
        if self._merge == 'concat':
            return values, pos
        if self._merge == 'mlp':
            values = self.merge(values)
            if self.activation:
                values = self.activation(values)
            return values, pos
