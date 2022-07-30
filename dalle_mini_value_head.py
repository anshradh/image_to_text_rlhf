from transformers import eBart
from transformers import top_k_top_p_filtering
from torch import nn
from torch.nn import Identity
import torch.nn.functional as F
import torch


class DalleMiniValueHead(nn.Module):
    """The DalleMiniValueHead class implements a head for a model that returns a scalar for each output."""

    def __init__(self, config):
        super().__init__()
        self.summary_type = (
            config.summary_type if hasattr(config, "summary_type") else "last"
        )
        if self.summary_type == "attn":
            raise NotImplementedError

        self.summary = Identity()
        if hasattr(config, "summary_use_proj") and config.summary_use_proj:
            if (
                hasattr(config, "summary_proj_to_labels")
                and config.summary_proj_to_labels
                and config.num_labels > 0
            ):
                num_classes = config.num_labels
            else:
                num_classes = config.hidden_size
            self.summary = nn.Linear(config.hidden_size, num_classes)

        self.activation = Identity()
        if (
            hasattr(config, "summary_activation")
            and config.summary_activation == "tanh"
        ):
            self.activation = nn.Tanh()

        self.first_dropout = Identity()
        if (
            hasattr(config, "summary_first_dropout")
            and config.summary_first_dropout > 0
        ):
            self.first_dropout = nn.Dropout(config.summary_first_dropout)

        self.last_dropout = Identity()
        if hasattr(config, "summary_last_dropout") and config.summary_last_dropout > 0:
            self.last_dropout = nn.Dropout(config.summary_last_dropout)

        self.flatten = nn.Flatten()

    def forward(self, hidden_states, cls_index=None):
        output = hidden_states
        output = self.first_dropout(output)
        output = self.summary(output)
        output = self.activation(output)
        output = self.last_dropout(output)

        return output


class DalleMiniWithValueHeadModel(eBart):
    """The DalleMiniWithValueHeadModel class implements a Dalle Mini model with a secondary, scalar head."""

    def __init__(self, config):
        super().__init__(config)
        config.num_labels = 1
        self.transformer = AutoModel.from_config(config)
        self.lm_head = nn.Linear(config.n_embd, config.vocab_size, bias=False)
        self.v_head = ValueHead(config)

        self.init_weights()

    def get_output_embeddings(self):
        return self.lm_head

    def forward(
        self,
        input_ids=None,
        past_key_values=None,
        attention_mask=None,
        token_type_ids=None,
        position_ids=None,
        head_mask=None,
        inputs_embeds=None,
        mc_token_ids=None,
        lm_labels=None,
        mc_labels=None,
    ):

        transformer_outputs = self.transformer(
            input_ids,
            past_key_values=past_key_values,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids,
            position_ids=position_ids,
            head_mask=head_mask,
            inputs_embeds=inputs_embeds,
        )

        hidden_states = transformer_outputs[0]

        lm_logits = self.lm_head(hidden_states)
        value = self.v_head(hidden_states).squeeze(-1)

        outputs = (lm_logits,) + transformer_outputs[1:] + (value,)

        return outputs