from torch import nn, torch

from config import DEVICE
from modules.nn.attention import SelfAttention
from modules.nn.modules import Embed, RNNEncoder

class ModelHelper:
    @staticmethod
    def _sort_by(lengths):
        """
        Sort batch data and labels by length.
        Useful for variable length inputs, for utilizing PackedSequences
        Args:
            lengths (nn.Tensor): tensor containing the lengths for the data

        Returns:
            - sorted lengths Tensor
            - sort (callable) which will sort a given iterable
                according to lengths
            - unsort (callable) which will revert a given iterable to its
                original order

        """
        batch_size = lengths.size(0)

        sorted_lengths, sorted_idx = lengths.sort()
        _, original_idx = sorted_idx.sort(0, descending=True)
        reverse_idx = torch.linspace(batch_size - 1, 0, batch_size).long()

        reverse_idx = reverse_idx.to(DEVICE)

        sorted_lengths = sorted_lengths[reverse_idx]

        def sort(iterable):
            if len(iterable.shape) > 1:
                return iterable[sorted_idx.data][reverse_idx]
            else:
                return iterable

        def unsort(iterable):
            if len(iterable.shape) > 1:
                return iterable[reverse_idx][original_idx][reverse_idx]
            else:
                return iterable

        return sorted_lengths, sort, unsort


class ModelWrapperBert(nn.Module, ModelHelper):
    def __init__(self, bertModel=None, out_size=1,
                 pretrained=None, finetune=None, **kwargs):
        """
        Define the layer of the model and perform the initializations
        of the layers (wherever it is necessary)
        Args:
            embeddings (numpy.ndarray): the 2D ndarray with the word vectors
            out_size ():
        """
        super(ModelWrapperBert, self).__init__()

        self.bertModel = bertModel

#        self.feature_size = 768 * 93

        self.linear = nn.Linear(in_features=768, out_features=out_size)

    def forward(self, x, mask, segment):
        """
        Defines how the data passes through the network.
        Args:
            x (): the input data (the sentences)
            lengths (): the lengths of each sentence

        Returns: the logits for each class

        """
        encoded_layers, _ = self.bertModel(x.type('torch.cuda.LongTensor'), segment.type('torch.cuda.LongTensor'), mask.type('torch.cuda.LongTensor'))
        
        representations = encoded_layers[int(10)]
        representations = torch.mean(representations,1)
#        representations = torch.mean(representations,1,keepdim=True)

        logits = self.linear(representations)#self.linear(representations.view(segment.shape[0], -1))

        return logits
