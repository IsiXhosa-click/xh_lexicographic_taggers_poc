import torch
import torch.nn as nn
import torch.nn.functional as function
from torch.nn import Embedding
from torch.nn.utils.rnn import unpad_sequence

from dataset import AnnotatedCorpusDataset, SEQ_PAD_IX


class BiLSTMTagger(nn.Module):
    """
    A `BiLSTMTagger` uses a bidirectional long short-term memory model to classify a given sequence of morphemes
    with their corresponding grammatical tags
    """

    def __init__(self, config, trainset: AnnotatedCorpusDataset):
        super(BiLSTMTagger, self).__init__()
        self.dev = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
        self.num_tags = trainset.num_tags

        self.hidden_dim = config["hidden_dim"]
        self.char_drop = config["char_dropout"]
        self.embed = Embedding(trainset.input_vocab_size, config["embed_dim"], device=self.dev)

        # The LSTM takes morpheme embeddings as inputs, and outputs hidden states
        # with dimensionality hidden_dim.
        self.lstm = nn.LSTM(config["embed_dim"], self.hidden_dim, batch_first=True, bidirectional=True, device=self.dev, num_layers=config["lstm_layers"], dropout=config["dropout"])

        # The linear layer that maps from hidden state space to tag space
        self.hidden2tag = nn.Linear(self.hidden_dim * 2, trainset.num_tags, device=self.dev)
        self.drop = nn.Dropout(config["dropout"])
        self.loss_fn = nn.NLLLoss()

    def loss(self, chars, expected):
        return self.loss_fn(self.forward(chars), expected)

    def forward_tags_only(self, xs):
        return torch.argmax(self.forward(xs), dim=1)

    def forward(self, chars):
        lengths = []
        for batch in torch.unbind(chars):
            lengths.append(len(torch.where(batch != SEQ_PAD_IX)[0]))
        lengths = torch.tensor(lengths, device=self.dev)

        # long dropout with no scale - https://discuss.pytorch.org/t/dropout-for-long-tensor/50914
        # perfect to make sure we can still handle unknown chars
        chars = torch.empty_like(chars).bernoulli_(1 - self.char_drop) * chars
        embeds = self.embed(chars)

        lstm_out, hidden = self.lstm(embeds)
        lstm_out = self.drop(lstm_out)
        tag_space = self.hidden2tag(lstm_out)

        scores = function.log_softmax(tag_space, dim=2)
        scores = unpad_sequence(scores, lengths, batch_first=True)
        scores = [score[-1,:] for score in scores]
        scores = torch.stack(scores)
        return scores
