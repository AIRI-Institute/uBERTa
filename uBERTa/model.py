from torch import nn
from transformers import BertPreTrainedModel, BertModel


class uBERTa(BertPreTrainedModel):
    def __init__(self, config):
        super().__init__(config)
        self.num_labels = config.num_labels

        self.bert = BertModel(config)
        self.dropout = nn.Dropout(config.hidden_dropout_prob)
        self.classifier = nn.Linear(config.hidden_size, 1)
        self.sigmoid = nn.Sigmoid()
        self.init_weights()

    def forward(
            self,
            input_ids=None,
            attention_mask=None,
            token_type_ids=None,
            position_ids=None,
            head_mask=None,
            inputs_embeds=None,
            labels=None,
            sample_weight=None  # <--- sample weights (!)
    ):
        outputs = self.bert(
            input_ids,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids,
            position_ids=position_ids,
            head_mask=head_mask,
            inputs_embeds=inputs_embeds,
        )

        # return outputs

        pooled_output = outputs[1]

        pooled_output = self.dropout(pooled_output)
        logits = self.classifier(pooled_output)
        probs = self.sigmoid(logits)

        loss = nn.BCELoss(weight=sample_weight)(probs, labels)

        return loss, probs, logits, pooled_output


class BertCentralPooler(nn.Module):

    def __init__(self, config):
        super().__init__()

        self.dense = nn.Linear(config.hidden_size, config.hidden_size)
        self.activation = nn.Tanh()
        self.central_position = config.central_position

    def forward(self, hidden_states):
        # We "pool" the model by simply taking the hidden state corresponding
        # to the first token.
        central_token_tensor = hidden_states[:, self.central_position]
        pooled_output = self.dense(central_token_tensor)
        pooled_output = self.activation(pooled_output)
        return pooled_output


if __name__ == '__main__':
    raise RuntimeError
