import torch.nn as nn
from mtmt_main.other.config import *

class RecognizerModel(nn.Module):
    def __init__(self, tag_num, hidden_dim):
        super(RecognizerModel, self).__init__()

        self.tag_num = tag_num
        self.hidden_dim = hidden_dim

        self.bert = BertModel.from_pretrained(BERT_PREMODEL, output_hidden_states=True)
        self.dropout = nn.Dropout(DROPOUT_RATIO)

        self.linear = nn.Linear(self.hidden_dim, self.tag_num)


    def forward(self, x, att_mask, lengths):

        output = self.bert(x).last_hidden_state

        output = self.dropout(output)

        logits = self.linear(output)

        prediction = logits.argmax(-1)

        return logits, prediction, output

