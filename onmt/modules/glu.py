"""Comes directly from fairseq"""
import torch, math


class Downsample(torch.nn.Module):
    """
    Selects every nth element along the last dim, where n is the index
    """
    def __init__(self, in_dim, step):
        super().__init__()
        self._step = step
        self._in_dim = in_dim
        
        if in_dim % step != 0:
            raise ValueError('in_dim should be a multiple of step. '
                             f'Got {in_dim} and {step}.')
        self.index = torch.LongTensor(range(0, in_dim, step))

    def forward(self, input):
        return input.index_select(dim=-1, index=self.index.to(input.device))
    
    def extra_repr(self):
        return f'{self._in_dim}, {self._in_dim//self._step}'


def Linear(in_features, out_features, dropout=0., bias=True):
    """Weight-normalized Linear layer (input: B x T x C)"""
    m = torch.nn.Linear(in_features, out_features, bias=bias)
    m.weight.data.normal_(mean=0, std=math.sqrt((1 - dropout) / in_features))
    m.bias.data.zero_()
    return torch.nn.utils.weight_norm(m)


class GatedLinear(torch.nn.Module):
    def __init__(self, in_features, out_features, depth=2, 
                 downsample=0, dropout=0., bias=True):
        """
        Weight-normalized Linear layer (input: B x T x C) with interspersed GLU units.
        GLU units split the input in half to use one as values and one as gates:
                glu([a; b]) = a * sigmoid(b)
        """
        super().__init__()
        
        self._num_layers = depth
        self._bias = bias
        self._dropout = dropout
        self._downsample = isinstance(downsample, int) and downsample > 0
        self.glu = torch.nn.GLU(dim=-1)
        
        # In order to halve the dims at each step and end on out_features
        # we need to start with out_feature * 2^depth and decrease the power
        # of 2 at each depth.
        if self._downsample:
            self.linear_in = torch.nn.Sequential(
                Downsample(in_features, downsample),
                Linear(in_features//downsample, out_features * pow(2, depth), dropout, bias)
            )
        else:
            if in_features != out_features * pow(2, depth):
                raise ValueError('When not using downsampling, in_features should be '
                                 'equal to out_feature * 2^depth. '
                                 f'Got {in_features} != {out_features} * 2^{depth}')
            
        self.linear_layers = torch.nn.ModuleList([
            Linear(out_features * pow(2, depth - k),
                   out_features * pow(2, depth - k),
                   dropout, bias)
            for k in range(1, depth+1)
        ])
          
    def forward(self, input):
        output = self.linear_in(input) if self._downsample else input
        for linear in self.linear_layers:            
            output = linear(self.glu(output))
        return output