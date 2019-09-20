import torch
import torch.nn as nn
from torch.nn import CrossEntropyLoss
from pytorch_transformers import BertPreTrainedModel, BertModel

from nn.layers import Highway
from nn.recurrent import BiGRU
from nn.bert_modules.transformer import TransformerBlock

VERY_NEGATIVE_NUMBER = -1e29


class BertMultiAnswer(BertPreTrainedModel):
    def __init__(self, config):
        super(BertMultiAnswer, self).__init__(config)
        self.num_labels = config.num_labels


class BertForLes(BertPreTrainedModel):
    def __init__(self, config):
        super(BertForLes, self).__init__(config)
        self.num_labels = config.num_labels

        self.bert = BertModel(config)
        self.qa_outputs = nn.Linear(config.hidden_size, config.num_labels)

        self.apply(self.init_weights)

    def forward(self, input_ids, token_type_ids=None, attention_mask=None, start_positions=None,
                end_positions=None, position_ids=None, head_mask=None,
                input_span_mask=None, doc_position=None):
        outputs = self.bert(input_ids, position_ids=position_ids, token_type_ids=token_type_ids,
                            attention_mask=attention_mask, head_mask=head_mask)
        sequence_output = outputs[0]

        logits = self.qa_outputs(sequence_output)
        start_logits, end_logits = logits.split(1, dim=-1)
        start_logits = start_logits.squeeze(-1)
        end_logits = end_logits.squeeze(-1)

        # 增加input_span_mask, 这里为None时会报错(防止特征没有load)
        adder = (1.0 - input_span_mask.float()) * VERY_NEGATIVE_NUMBER
        start_logits += adder
        end_logits += adder

        outputs = (start_logits, end_logits,) + outputs[2:]
        if start_positions is not None and end_positions is not None:
            # If we are on multi-GPU, split add a dimension
            if len(start_positions.size()) > 1:
                start_positions = start_positions.squeeze(-1)
            if len(end_positions.size()) > 1:
                end_positions = end_positions.squeeze(-1)
            # sometimes the start/end positions are outside our model inputs, we ignore these terms
            ignored_index = start_logits.size(1)
            start_positions.clamp_(0, ignored_index)
            end_positions.clamp_(0, ignored_index)

            loss_fct = CrossEntropyLoss(ignore_index=ignored_index)
            start_loss = loss_fct(start_logits, start_positions)
            end_loss = loss_fct(end_logits, end_positions)
            total_loss = (start_loss + end_loss) / 2
            outputs = (total_loss,) + outputs

        return outputs  # (loss), start_logits, end_logits, (hidden_states), (attentions)


class BertConcatBiGRU(BertPreTrainedModel):
    """
    bert encoder 输出结合 char2vec + bigru 输出
    """

    def __init__(self, config, bigru_hidden_size, bigru_dropout_prob):
        super(BertConcatBiGRU, self).__init__(config)

        self.num_labels = config.num_labels
        self.bert = BertModel(config)

        # 复用 bert embedding layer
        self.char_embeddings = self.bert.embeddings
        # bigru layer
        self.birnn_encoder = BiGRU(input_size=config.hidden_size, hidden_size=bigru_hidden_size, num_layers=2,
                                   drop_prob=bigru_dropout_prob)

        self.hidden_size = config.hidden_size + bigru_hidden_size * 2

        self.highway = Highway(input_dim=self.hidden_size, num_layers=2)
        self.qa_outputs = nn.Linear(self.hidden_size, config.num_labels)

        self.apply(self.init_weights)

    def forward(self, input_ids, token_type_ids=None, attention_mask=None, start_positions=None,
                end_positions=None, position_ids=None, head_mask=None,
                input_span_mask=None, doc_position=None):
        bert_outputs = self.bert(input_ids, position_ids=position_ids, token_type_ids=token_type_ids,
                                 attention_mask=attention_mask, head_mask=head_mask)
        bert_reprs = bert_outputs[0]

        char_embed = self.char_embeddings(input_ids, position_ids=position_ids, token_type_ids=token_type_ids)
        birnn_reprs, _ = self.birnn_encoder(inputs=char_embed, lengths=None)
        # 拼接 bert 输出和 bigru 编码输出，再 highway 组合
        sequence_output = torch.cat([bert_reprs, birnn_reprs], dim=-1)
        sequence_output = self.highway(sequence_output)

        logits = self.qa_outputs(sequence_output)
        start_logits, end_logits = logits.split(1, dim=-1)
        start_logits = start_logits.squeeze(-1)
        end_logits = end_logits.squeeze(-1)

        # 增加input_span_mask, 这里为None时会报错(防止特征没有load)
        adder = (1.0 - input_span_mask.float()) * VERY_NEGATIVE_NUMBER
        start_logits += adder
        end_logits += adder

        outputs = (start_logits, end_logits,) + bert_outputs[2:]
        if start_positions is not None and end_positions is not None:
            # If we are on multi-GPU, split add a dimension
            if len(start_positions.size()) > 1:
                start_positions = start_positions.squeeze(-1)
            if len(end_positions.size()) > 1:
                end_positions = end_positions.squeeze(-1)
            # sometimes the start/end positions are outside our model inputs, we ignore these terms
            ignored_index = start_logits.size(1)
            start_positions.clamp_(0, ignored_index)
            end_positions.clamp_(0, ignored_index)

            loss_fct = CrossEntropyLoss(ignore_index=ignored_index)
            start_loss = loss_fct(start_logits, start_positions)
            end_loss = loss_fct(end_logits, end_positions)
            total_loss = (start_loss + end_loss) / 2
            outputs = (total_loss,) + outputs

        return outputs  # (loss), start_logits, end_logits, (hidden_states), (attentions)


class BertConcatTransformer(BertPreTrainedModel):
    """
    bert encoder 输出结合 transformer 输出
    """

    def __init__(self, config):
        super(BertConcatTransformer, self).__init__(config)

        self.num_labels = config.num_labels
        self.bert = BertModel(config)

        # 复用 bert embedding layer
        self.char_embeddings = self.bert.embeddings
        # bigru layer
        self.transformer = TransformerBlock(config.hidden_size, 8, config.hidden_size * 4, 0.1)

        self.hidden_size = config.hidden_size * 2

        self.highway = Highway(input_dim=self.hidden_size, num_layers=2)
        self.qa_outputs = nn.Linear(self.hidden_size, config.num_labels)

        self.apply(self.init_weights)

    def forward(self, input_ids, token_type_ids=None, attention_mask=None, start_positions=None,
                end_positions=None, position_ids=None, head_mask=None,
                input_span_mask=None, doc_position=None):
        bert_outputs = self.bert(input_ids, position_ids=position_ids, token_type_ids=token_type_ids,
                                 attention_mask=attention_mask, head_mask=head_mask)
        bert_reprs = bert_outputs[0]

        input_mask = (input_ids > 0).unsqueeze(1).repeat(1, input_ids.size(1), 1).unsqueeze(1)
        char_embed = self.char_embeddings(input_ids, position_ids=position_ids, token_type_ids=token_type_ids)
        trans_reprs = self.transformer.forward(char_embed, input_mask)

        # 拼接 bert 输出和 bigru 编码输出，再 highway 组合
        sequence_output = torch.cat([bert_reprs, trans_reprs], dim=-1)
        sequence_output = self.highway(sequence_output)

        logits = self.qa_outputs(sequence_output)
        start_logits, end_logits = logits.split(1, dim=-1)
        start_logits = start_logits.squeeze(-1)
        end_logits = end_logits.squeeze(-1)

        # 增加input_span_mask, 这里为None时会报错(防止特征没有load)
        adder = (1.0 - input_span_mask.float()) * VERY_NEGATIVE_NUMBER
        start_logits += adder
        end_logits += adder

        outputs = (start_logits, end_logits,) + bert_outputs[2:]
        if start_positions is not None and end_positions is not None:
            # If we are on multi-GPU, split add a dimension
            if len(start_positions.size()) > 1:
                start_positions = start_positions.squeeze(-1)
            if len(end_positions.size()) > 1:
                end_positions = end_positions.squeeze(-1)
            # sometimes the start/end positions are outside our model inputs, we ignore these terms
            ignored_index = start_logits.size(1)
            start_positions.clamp_(0, ignored_index)
            end_positions.clamp_(0, ignored_index)

            loss_fct = CrossEntropyLoss(ignore_index=ignored_index)
            start_loss = loss_fct(start_logits, start_positions)
            end_loss = loss_fct(end_logits, end_positions)
            total_loss = (start_loss + end_loss) / 2
            outputs = (total_loss,) + outputs

        return outputs  # (loss), start_logits, end_logits, (hidden_states), (attentions)
