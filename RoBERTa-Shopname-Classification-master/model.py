import torch
import torch.nn as nn
from transformers import AutoModel

class BidirectionalLSTM(nn.Module):
    def __init__(self, input_size, hidden_size, output_size, use_dropout=True, dropout=0.1):
        super(BidirectionalLSTM, self).__init__()
        self.rnn = nn.LSTM(input_size, hidden_size, bidirectional=True, batch_first=True)
        self.dropout = nn.Dropout(dropout)
        self.linear = nn.Linear(hidden_size * 2, output_size)
        self.softmax = nn.Softmax(dim=1)
        self.use_dropout = use_dropout
    def forward(self, input):
        """
        input : visual feature [batch_size x T x input_size]
        output : contextual feature [batch_size x T x output_size]
        """
        output, (hidden_last,cn_last) = self.rnn(input)  # batch_size x T x input_size -> batch_size x T x (2*hidden_size)
        hidden_last_L = hidden_last[-2]
        hidden_last_R = hidden_last[-1]
        hidden_last_out = torch.cat([hidden_last_L, hidden_last_R], dim=-1)
        if self.use_dropout:
            output = self.dropout(hidden_last_out)
        else:
            output = hidden_last_out
        output = self.linear(output)
        # output = self.softmax(output)
        return output


class BertBiLSTMModel(nn.Module):
    # 初始化类
    def __init__(self, class_size, pretrained_name='bert-base-chinese', use_finetune=True, use_dropout=True,dropout = 0.1):
        super(BertBiLSTMModel, self).__init__()
        self.bert = AutoModel.from_pretrained(pretrained_name)
        for c in self.parameters():
            c.requires_grad = use_finetune
        # self.lstm_model = BidirectionalLSTM(768, 300, class_size,use_dropout)
        # self.transition = nn.Parameter(torch.ones(ner_labels, ner_labels) * 1 / ner_labels)
        self.classifier = nn.Linear(768, class_size)
        self.dropout = nn.Dropout(dropout)
        self.softmax = nn.Softmax(dim=1)

    def forward(self, inputs, use_crf=True):
        # input_ids, input_tyi, input_attn_mask = inputs['input_ids'], inputs[
        #     'token_type_ids'], inputs['attention_mask']
        output = self.bert(**inputs)
        # categories_numberic = self.lstm_model(output.pooler_output)
        categories_numberic = self.classifier(self.dropout(output.pooler_output))
        # categories_numberic = self.softmax(categories_numberic)
        # batch_size, seq_len, ner_class_num = categories_numberic.shape
        return categories_numberic