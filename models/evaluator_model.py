import torch.nn as nn
from mtmt_main.other.config import *


class SiameseEncoder(nn.Module):
    def __init__(self, HIDDEN_DIM):
        super(SiameseEncoder, self).__init__()
        self.bert = BertModel.from_pretrained(BERT_PREMODEL, output_hidden_states=True)
        self.dropout = nn.Dropout(DROPOUT_RATIO)
        self.hidden_dim = HIDDEN_DIM

    def forward(self, x, word_idx):
        encoded_layers = self.bert(x).hidden_states
        output = encoded_layers[-1]
        output = self.dropout(output)
        embeds = output[range(len(word_idx)), [w_i+1 for w_i in word_idx]]
        return embeds



class EvaluatorLayer(nn.Module):
    def __init__(self, HIDDEN_DIM):
        super(EvaluatorLayer, self).__init__()
        self.hidden = HIDDEN_DIM
        self.cossim = nn.CosineSimilarity(dim=1, eps=1e-6)
    def forward(self, embed_1, embed_2):

        y_hat = self.cossim(embed_1, embed_2)
        y_hat = torch.sigmoid(y_hat)

        return y_hat


class RecognizerLayer(nn.Module):
    def __init__(self, tag_num, hidden_dim, feature_dim):
        super(RecognizerLayer, self).__init__()
        self.tag_num = tag_num
        self.hidden_dim = hidden_dim
        self.linear = nn.Linear(feature_dim, self.tag_num)

    def forward(self, embeds):
        logits = self.linear(embeds)
        tag_pred = logits.argmax(-1)

        return logits, tag_pred


class EvaluatorModel(nn.Module):
    def __init__(self, batch_size, tag_num, hidden_dim, feature_dim):
        super(EvaluatorModel, self).__init__()
        self.batch_size = batch_size
        self.hidden_dim = hidden_dim
        self.tag_num = tag_num
        self.feature_dim = feature_dim
        self.siamese_encoder = SiameseEncoder(self.hidden_dim)
        self.evaluator_layer = EvaluatorLayer(self.hidden_dim)
        self.recognizer_layer = RecognizerLayer(self.tag_num, self.hidden_dim, self.feature_dim)


    def forward(self, x_1_2, word_idx_1_2):
        x_1, x_2 = x_1_2[0], x_1_2[1]
        word_idx_1, word_idx_2 = word_idx_1_2[0], word_idx_1_2[1]

        embeds_1 = self.siamese_encoder(x_1, word_idx_1)
        embeds_2 = self.siamese_encoder(x_2, word_idx_2)

        sim = self.evaluator_layer(embeds_1, embeds_2)

        logits_1, tag_pred_1 = self.recognizer_layer(embeds_1)
        logits_2, tag_pred_2 = self.recognizer_layer(embeds_2)

        return sim, logits_1, tag_pred_1, logits_2, tag_pred_2, embeds_1, embeds_2
