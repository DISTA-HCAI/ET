import torch.nn as nn
import torch
from typing import List, Optional, Tuple, Union


class CustomLlamaMLP(nn.Module):
    def __init__(self, original_llama_mlp, low_rank_dimension, kwargs):
        super().__init__()
        self.intervention_strategy = kwargs['defence_strategy']
        self.hidden_size = original_llama_mlp.hidden_size
        self.intermediate_size = original_llama_mlp.intermediate_size
        self.gate_proj = original_llama_mlp.gate_proj
        self.up_proj = original_llama_mlp.up_proj
        self.down_proj = original_llama_mlp.down_proj
        self.act_fn = original_llama_mlp.act_fn
        self.low_rank_dimension = low_rank_dimension

        if 'UP' in self.intervention_strategy:
            self.up_A = nn.Linear(
                in_features=self.up_proj.in_features, 
                out_features=self.low_rank_dimension,
                bias=False,
                dtype=torch.bfloat16,
                device=kwargs['device']
            )
            self.up_B = nn.Linear(
                in_features=self.low_rank_dimension,
                out_features=self.up_proj.out_features,
                bias=False,
                dtype=torch.bfloat16,
                device=kwargs['device']
            )

        if 'GATE' in self.intervention_strategy:
            self.gate_A = nn.Linear(
                in_features=self.gate_proj.in_features, 
                out_features=self.low_rank_dimension,
                bias=False,
                dtype=torch.bfloat16,
                device=kwargs['device']
            )
            self.gate_B = nn.Linear(
                in_features=self.low_rank_dimension,
                out_features=self.gate_proj.out_features,
                bias=False,
                dtype=torch.bfloat16,
                device=kwargs['device']
            )

        if 'DOWN' in self.intervention_strategy:
            self.down_A = nn.Linear(
                in_features=self.down_proj.in_features, 
                out_features=self.low_rank_dimension,
                bias=False,
                dtype=torch.bfloat16,
                device=kwargs['device']
            )
            self.down_B = nn.Linear(
                in_features=self.low_rank_dimension,
                out_features=self.down_proj.out_features,
                bias=False,
                dtype=torch.bfloat16,
                device=kwargs['device']
            )

        for param in list(self.gate_proj.parameters()) + \
                        list(self.up_proj.parameters()) + \
                        list(self.down_proj.parameters()):
            param.requires_grad = False


    def forward(self, x):
        down_proj = self.down_proj(self.act_fn(self.gate_proj(x)) * self.up_proj(x))
        return down_proj


    def intervened_forward(self, x):

        up_projected = self.up_proj(x)
        gate_projected = self.gate_proj(x)

        if 'UP' in self.intervention_strategy:
            up_projected = up_projected + self.up_B(self.up_A(x))
        if 'GATE' in self.intervention_strategy:
            gate_projected = gate_projected + self.gate_B(self.gate_A(x))

        input_to_down_projection = self.act_fn(gate_projected) * up_projected
        down_projected = self.down_proj(input_to_down_projection)

        if 'DOWN' in self.intervention_strategy:
            down_projected = down_projected + self.down_B(self.down_A(input_to_down_projection))

        return down_projected



class LlamaBlockDefendor(nn.Module):
    
    def __init__(self, originalBlock, low_rank_dimension, kwargs):  # originalBlock is a LlamaDecoderLayer
        super().__init__()

        self.hidden_size = originalBlock.hidden_size
        self.self_attn = originalBlock.self_attn

        self.mlp = CustomLlamaMLP(originalBlock.mlp, low_rank_dimension, kwargs)

        self.input_layernorm = originalBlock.input_layernorm
        self.post_attention_layernorm = originalBlock.post_attention_layernorm

        for param in list(self.self_attn.parameters()) + \
                                list(self.input_layernorm.parameters()) + \
                                list(self.post_attention_layernorm.parameters()):
            param.requires_grad = False

    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_value = None,
        output_attentions: Optional[bool] = False,
        use_cache: Optional[bool] = False,
        cache_position: Optional[torch.LongTensor] = None,
    ) -> Tuple[torch.FloatTensor, Optional[Tuple[torch.FloatTensor, torch.FloatTensor]]]:
        """
        This is the original forward method of the LlamaDecoder pytorch implementation from huggingface copied from:
        https://github.com/huggingface/transformers/blob/main/src/transformers/models/llama/modeling_llama.py

        Args:
            hidden_states (`torch.FloatTensor`): input to the layer of shape `(batch, seq_len, embed_dim)`
            attention_mask (`torch.FloatTensor`, *optional*):
                attention mask of size `(batch_size, sequence_length)` if flash attention is used or `(batch_size, 1,
                query_sequence_length, key_sequence_length)` if default attention is used.
            output_attentions (`bool`, *optional*):
                Whether or not to return the attentions tensors of all attention layers. See `attentions` under
                returned tensors for more detail.
            use_cache (`bool`, *optional*):
                If set to `True`, `past_key_values` key value states are returned and can be used to speed up decoding
                (see `past_key_values`).
            past_key_value (`Tuple(torch.FloatTensor)`, *optional*): cached past key and value projection states
        """
        residual = hidden_states

        hidden_states = self.input_layernorm(hidden_states)

        # Self Attention
        hidden_states, self_attn_weights, present_key_value = self.self_attn(
            hidden_states=hidden_states,
            attention_mask=attention_mask,
            position_ids=position_ids,
            past_key_value=past_key_value,
            output_attentions=output_attentions,
            use_cache=use_cache,
            cache_position=cache_position,
        )
        hidden_states = residual + hidden_states

        # Fully Connected
        residual = hidden_states
        hidden_states = self.post_attention_layernorm(hidden_states)
        hidden_states = self.mlp(hidden_states)
        hidden_states = residual + hidden_states

        outputs = (hidden_states,)

        if output_attentions:
            outputs += (self_attn_weights,)

        if use_cache:
            outputs += (present_key_value,)

        return outputs


    def intervened_forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_value = None,
        output_attentions: Optional[bool] = False,
        use_cache: Optional[bool] = False,
        cache_position: Optional[torch.LongTensor] = None,
    ) -> Tuple[torch.FloatTensor, Optional[Tuple[torch.FloatTensor, torch.FloatTensor]]]:

        residual = hidden_states

        hidden_states = self.input_layernorm(hidden_states)

        # Self Attention
        hidden_states, self_attn_weights, present_key_value = self.self_attn(
            hidden_states=hidden_states,
            attention_mask=attention_mask,
            position_ids=position_ids,
            past_key_value=past_key_value,
            output_attentions=output_attentions,
            use_cache=use_cache,
            cache_position=cache_position,
        )
        hidden_states = residual + hidden_states

        # Fully Connected
        residual = hidden_states
        hidden_states = self.post_attention_layernorm(hidden_states)
        hidden_states = self.mlp.intervened_forward(hidden_states)
        hidden_states = residual + hidden_states

        outputs = (hidden_states,)

        if output_attentions:
            outputs += (self_attn_weights,)

        if use_cache:
            outputs += (present_key_value,)

        return outputs