from torch import nn, torch

from config import DEVICE
from modules.nn.attention import SelfAttention
from modules.nn.modules import Embed, RNNEncoder
from modules.nn.selfAttention import FeedForward, MultiHeadAttention, Norm, DecoderLayer, PositionalEncoder

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


class FeatureExtractor(nn.Module):
    def __init__(self, embeddings=None, num_embeddings=0, **kwargs):
        """
        Define the layer of the model and perform the initializations
        of the layers (wherever it is necessary)
        Args:
            embeddings (numpy.ndarray): the 2D ndarray with the word vectors
            out_size ():
        """
        super(FeatureExtractor, self).__init__()
        embed_dim = kwargs.get("embed_dim", 300)
        embed_finetune = kwargs.get("embed_finetune", False)
        embed_noise = kwargs.get("embed_noise", 0.)
        embed_dropout = kwargs.get("embed_dropout", 0.)

        encoder_size = kwargs.get("encoder_size", 128)
        encoder_layers = kwargs.get("encoder_layers", 1)
        encoder_dropout = kwargs.get("encoder_dropout", 0.)
        bidirectional = kwargs.get("encoder_bidirectional", False)

        attention = kwargs.get("attention", False)
        attention_layers = kwargs.get("attention_layers", 1)
        attention_dropout = kwargs.get("attention_dropout", 0.)
        attention_activation = kwargs.get("attention_activation", "tanh")
        self.attention_context = kwargs.get("attention_context", False)
        
        
        self.attention_heads = kwargs.get("attention_heads", 10)
        self.recurrent_steps = kwargs.get("recurrent_steps", 2)

        ########################################################

        # define the embedding layer, with the corresponding dimensions
        if embeddings is not None:
            self.embedding = Embed(
                num_embeddings=embeddings.shape[0],
                embedding_dim=embeddings.shape[1],
                embeddings=embeddings,
                noise=embed_noise,
                dropout=embed_dropout,
                trainable=embed_finetune)
        else:
            if num_embeddings == 0:
                raise ValueError("if an embedding matrix is not passed, "
                                 "`num_embeddings` cant be zero.")
            self.embedding = Embed(
                num_embeddings=num_embeddings,
                embedding_dim=embed_dim,
                noise=embed_noise,
                dropout=embed_dropout,
                trainable=True)
            

        if embeddings is not None:
            encoder_input_size = embeddings.shape[1]
        else:
            encoder_input_size = embed_dim


        self.pe = PositionalEncoder(encoder_input_size)


        #################################################################
        # Encoders
        #################################################################
        self.encoder = RNNEncoder(input_size=encoder_input_size,
                                  rnn_size=encoder_size,
                                  num_layers=encoder_layers,
                                  bidirectional=bidirectional,
                                  dropout=encoder_dropout)

        feat_size = self.encoder.feature_size

        self.feature_size = feat_size


        self.norm_1 = Norm(encoder_input_size)
        self.norm_2 = Norm(encoder_size*2)
        self.attentionDecoder = DecoderLayer(d_model=encoder_input_size, d_model2=encoder_size*2, heads=self.attention_heads, dropout=encoder_dropout)




        if attention:
            att_size = feat_size
            if self.attention_context:
                context_size = self.encoder.feature_size
                att_size += context_size

            self.attention = SelfAttention(att_size,
                                           layers=attention_layers,
                                           dropout=attention_dropout,
                                           non_linearity=attention_activation,
                                           batch_first=True)

    def init_embeddings(self, weights, trainable):
        self.embedding.weight = nn.Parameter(torch.from_numpy(weights),
                                             requires_grad=trainable)

    @staticmethod
    def _mean_pooling(x, lengths):
        sums = torch.sum(x, dim=1)
        _lens = lengths.view(-1, 1).expand(sums.size(0), sums.size(1))
        means = sums / _lens.float()
        return means

    def forward(self, x, lengths):
        embeddings = self.embedding(x.type('torch.cuda.LongTensor'))
#        embeddings = self.pe(embeddings)
#        embeddings = self.norm_1(embeddings)
    
        attentions = None
        encoder_outputs, last_output = self.encoder(embeddings, lengths)
        encoder_outputs = self.norm_2(encoder_outputs)

        reps = []
        atts = []

        if hasattr(self, 'attention'):
            outputs = encoder_outputs
            if self.attention_context:
                context = self._mean_pooling(encoder_outputs, lengths)
                context = context.unsqueeze(1).expand(-1, encoder_outputs.size(1), -1)
                outputs = torch.cat([encoder_outputs, context], -1)

            representations, attentions = self.attention(outputs, lengths)

            if self.attention_context:
                representations = representations[:, :context.size(-1)]
        else:
            representations = last_output
            
        reps.append(representations)
        atts.append(attentions)
        

        for i in range(self.recurrent_steps):
            new_input = self.attentionDecoder(embeddings, encoder_outputs, lengths)
#            new_input = self.norm_1(new_input)

            encoder_outputs, last_output = self.encoder(new_input, lengths)
            encoder_outputs = self.norm_2(encoder_outputs)
    
            if hasattr(self, 'attention'):
                outputs = encoder_outputs
                if self.attention_context:
                    context = self._mean_pooling(encoder_outputs, lengths)
                    context = context.unsqueeze(1).expand(-1, encoder_outputs.size(1), -1)
                    outputs = torch.cat([encoder_outputs, context], -1)
    
                representations, attentions = self.attention(outputs, lengths)
    
                if self.attention_context:
                    representations = representations[:, :context.size(-1)]
            else:
                representations = last_output
                
            reps.append(representations)
            atts.append(attentions)


        return reps, atts


class ModelWrapper(nn.Module, ModelHelper):
    def __init__(self, embeddings=None, out_size=1, num_embeddings=0,
                 pretrained=None, finetune=None, **kwargs):
        """
        Define the layer of the model and perform the initializations
        of the layers (wherever it is necessary)
        Args:
            embeddings (numpy.ndarray): the 2D ndarray with the word vectors
            out_size ():
        """
        super(ModelWrapper, self).__init__()

        if pretrained is not None:
            self.feature_extractor = pretrained.feature_extractor
            if not finetune:
                for param in self.feature_extractor.parameters():
                    param.requires_grad = False
        else:
            self.feature_extractor = FeatureExtractor(embeddings,
                                                      num_embeddings,
                                                      **kwargs)

        self.feature_size = self.feature_extractor.feature_size
        self.recurrent_steps = self.feature_extractor.recurrent_steps
        self.linear = nn.Linear(in_features=self.feature_size, out_features=out_size)

    def forward(self, x, lengths):
        """
        Defines how the data passes through the network.
        Args:
            x (): the input data (the sentences)
            lengths (): the lengths of each sentence

        Returns: the logits for each class

        """
        # sort
        lengths, sort, unsort = self._sort_by(lengths)
        x = sort(x)

        representations, attentions = self.feature_extractor(x, lengths)

        # unsort
        logits=[[]]*(self.recurrent_steps+1)
        
        for i in range(self.recurrent_steps+1):
            representations[i] = unsort(representations[i])
        
            if attentions is not None: 
                attentions[i] = unsort(attentions[i])

            logits[i] = self.linear(representations[i])

        return logits, attentions
