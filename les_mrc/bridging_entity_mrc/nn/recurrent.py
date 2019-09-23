"""
This module is the wrapper of recurrent neural networks(RNNs) in Pytorch.
1. we implement BaseRNN which is the wrapper of ``pack_padded_sequence``,
    ``pad_packed_sequence`` and standard RNNs
2. we implement BaseMultiLayerRNN which uses the VariationalDropout at each RNN layers input,
    rather than use standard Dropout between RNN layers.
3. we inherit the classes described above and implement some common RNNs class, e.g. BiGRU, MultiLayerBiGRU.
"""

import torch
import torch.nn as nn
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence

from .dropout import VariationalDropout


class BaseRNN(nn.Module):
    """Base RNN Module, which has been packed using `pack_padded_sequence` and unpacked using `pad_packed_sequence`"""

    def __init__(self, rnn_type, input_size, hidden_size, batch_first=True,
                 num_layers=1, bidirectional=False, drop_prob=0.0):
        super(BaseRNN, self).__init__()
        self.batch_first = batch_first
        self.rnn_type = rnn_type.lower()
        if self.rnn_type == 'rnn':
            rnn_cls = nn.RNN
        elif self.rnn_type == 'gru':
            rnn_cls = nn.GRU
        elif self.rnn_type == 'lstm':
            rnn_cls = nn.LSTM
        else:
            raise NotImplementedError('rnn_type must be one of `RNN/rnn`, `GRU/gru`, `LSTM/lstm`')

        self.rnn = rnn_cls(input_size, hidden_size, batch_first=True, num_layers=num_layers,
                           bidirectional=bidirectional, dropout=(0 if num_layers == 1 else drop_prob))

    def forward(self, inputs, lengths=None, initial_state=None):
        """
        Args:
            inputs(Tensor): tensor containing the features of the input sequence.
                If `batch_first=True(default)`, tensor shape is `(batch_size, seq_len, input_size)`,
                else `(seq_len, batch_size, input_size)`.
            lengths(Tensor): tensor containing the real length of each sequence.
            initial_state(Tensor or tuple): tensor containing the initial hidden state for each element in the batch.
                if rnn_type is not `lstm`, it means h_0, shape is `(num_layers * num_directions, batch, hidden_size)`,
                else it means a tuple of (h_0, c_0) for lstm, the shape of both h_0 and c_0 are `(num_layers * num_directions, batch, hidden_size)`
        Retures:
            a tuple of (outputs, last_state).
            outputs: the shape of outputs is `(batch_size, seq_len, num_directions * hidden_size)` if `batch_first=True(default)`,
                else `(seq_len, batch_size, num_directions * hidden_size)`.
            last_state: containing the hidden state for `t = seq_len`. The shape of last_state
                is `(num_layers * num_directions, batch_size, hidden_size)`.
        """
        # Ensure inputs is batch_first
        if not self.batch_first:
            inputs.transpose_(0, 1)

        if lengths is None:
            outputs, last_state = self.rnn(inputs, initial_state)
        else:
            orig_len = inputs.size(1)
            # Sort and Pack
            lengths, sort_idx = lengths.sort(dim=0, descending=True)
            inputs = inputs[sort_idx]
            inputs = pack_padded_sequence(inputs, lengths, batch_first=True)
            # Apply RNNs
            outputs, last_state = self.rnn(inputs, initial_state)
            # Unpack and Unsort
            outputs, _ = pad_packed_sequence(outputs, batch_first=True, total_length=orig_len)
            _, unsort_idx = sort_idx.sort(dim=0)
            outputs = outputs[unsort_idx]
            if self.rnn_type == 'lstm':
                last_state = (last_state[0][:, unsort_idx, :], last_state[1][:, unsort_idx, :])
            else:
                last_state = last_state[:, unsort_idx, :]

        # Restored outputs shape
        if not self.batch_first:
            outputs.transpose_(0, 1)

        return outputs, last_state


class LSTM(BaseRNN):
    """Unidirectional LSTM"""

    def __init__(self, input_size, hidden_size, num_layers=1, drop_prob=0.0, batch_first=True):
        super(LSTM, self).__init__('LSTM', input_size, hidden_size,
                                   num_layers=num_layers, drop_prob=drop_prob,
                                   batch_first=batch_first, bidirectional=False)


class GRU(BaseRNN):
    """Unidirectional GRU"""

    def __init__(self, input_size, hidden_size, num_layers=1, drop_prob=0.0, batch_first=True):
        super(GRU, self).__init__('GRU', input_size, hidden_size,
                                  num_layers=num_layers, drop_prob=drop_prob,
                                  batch_first=batch_first, bidirectional=False)


class BiLSTM(BaseRNN):
    """Bidirectional LSTM"""

    def __init__(self, input_size, hidden_size, num_layers=1, drop_prob=0.0, batch_first=True):
        super(BiLSTM, self).__init__('LSTM', input_size, hidden_size,
                                     num_layers=num_layers, drop_prob=drop_prob,
                                     batch_first=batch_first, bidirectional=True)


class BiGRU(BaseRNN):
    """Bidirectional GRU"""

    def __init__(self, input_size, hidden_size, num_layers=1, drop_prob=0.0, batch_first=True):
        super(BiGRU, self).__init__('GRU', input_size, hidden_size,
                                    num_layers=num_layers, drop_prob=drop_prob,
                                    batch_first=batch_first, bidirectional=True)


class BaseMultiLayerRNN(nn.Module):
    """Multi-Layer RNNs Base Model. In particular, the input of each RNN layer uses `Variational Dropout`"""

    def __init__(self, rnn_type, input_size, hidden_size, num_layers,
                 batch_first=True, bidirectional=False, input_drop_prob=0.0):
        super(BaseMultiLayerRNN, self).__init__()
        self.rnn_type = rnn_type.lower()
        self.batch_first = batch_first

        self.rnn_list = nn.ModuleList(
            [BaseRNN(self.rnn_type, input_size, hidden_size, batch_first=True, bidirectional=bidirectional)])

        input_size_ = 2 * hidden_size if bidirectional else hidden_size
        for _ in range(num_layers - 1):
            self.rnn_list.append(BaseRNN(self.rnn_type, input_size_, hidden_size,
                                         batch_first=True, bidirectional=bidirectional))

        self.dropout = VariationalDropout(p=input_drop_prob, batch_first=True)

    def forward(self, inputs, lengths=None, initial_state=None, concat_layers=True):
        """
        Args:
            concat_layers(bool): whether concat all layers outputs when `num_layers > 1`
        Returns:
            a tuple of (outputs, last_state).
            outputs(Tensor): If `concat_layers=True`, will return all layers outputs,
                the last dim shape is num_directions * hidden_size * num_layers.
            last_state(Tensor or tuple): if rnn_type is not lstm return a Tensor which means h_n, else return a tuple (h_n, c_n)
                the tensor shape is `(num_layers * num_directions, batch_size, hidden_size)`.
        """
        # Ensure inputs is batch_first
        if not self.batch_first:
            inputs.transpose_(0, 1)

        # Apply RNNs
        outputs_list, last_state_list = [], []
        for rnn in self.rnn_list:
            outputs, last_state = rnn(self.dropout(inputs), lengths, initial_state)
            outputs_list.append(outputs)
            last_state_list.append(last_state)
            inputs = outputs

        # Prepare the return values
        outputs, last_state = None, None
        if concat_layers:
            outputs = torch.cat(outputs_list, dim=-1)
        else:
            outputs = outputs_list[-1]
        if self.rnn_type == 'lstm':
            hn_state = torch.cat([layer_state[0] for layer_state in last_state_list], dim=0)
            cn_state = torch.cat([layer_state[1] for layer_state in last_state_list], dim=0)
            last_state = (hn_state, cn_state)
        else:
            last_state = torch.cat(last_state_list, dim=0)

        # Restored outputs shape
        if not self.batch_first:
            outputs.transpose_(0, 1)

        return outputs, last_state


class MultiLayerBiGRU(BaseMultiLayerRNN):
    def __init__(self, input_size, hidden_size, num_layers, input_drop_prob=0.0, batch_first=True):
        super(MultiLayerBiGRU, self).__init__('GRU', input_size, hidden_size,
                                              num_layers=num_layers,
                                              input_drop_prob=input_drop_prob,
                                              batch_first=batch_first,
                                              bidirectional=True)


class MultiLayerBiLSTM(BaseMultiLayerRNN):
    def __init__(self, input_size, hidden_size, num_layers, input_drop_prob=0.0, batch_first=True):
        super(MultiLayerBiLSTM, self).__init__('LSTM', input_size, hidden_size,
                                               num_layers=num_layers,
                                               input_drop_prob=input_drop_prob,
                                               batch_first=batch_first,
                                               bidirectional=True)
