import torch.nn as nn
import torch
from typing import List, Optional, Tuple, Union, Callable
from transformers.modeling_flash_attention_utils import FlashAttentionKwargs
from transformers.models.llama.modeling_llama import LlamaAttention, apply_rotary_pos_emb, eager_attention_forward, LlamaModel, LlamaPreTrainedModel
from transformers.models.llama.configuration_llama import LlamaConfig
from transformers.cache_utils import Cache, DynamicCache
from transformers.processing_utils import Unpack
from transformers.modeling_utils import ALL_ATTENTION_FUNCTIONS


class LoRALayer(nn.Module):
    def __init__(self, original_layer, rank=4, alpha=1.0):
        super().__init__()
        self.original_layer = original_layer
        self.rank = rank
        self.alpha = alpha
        self.scale = alpha / rank

        # Decompose ΔW into A and B
        self.lora_A = nn.Parameter(torch.zeros(size=(original_layer.out_features, rank), device=original_layer.weight.device, dtype=original_layer.weight.dtype))
        self.lora_B = nn.Parameter(torch.zeros(size=(rank, original_layer.in_features), device=original_layer.weight.device,  dtype=original_layer.weight.dtype))
        nn.init.normal_(self.lora_A, std=0.02)
        nn.init.normal_(self.lora_B, std=0.02)

    def forward(self, x):
        # Original projection + LoRA adaptation
        delta_w = self.lora_A @ self.lora_B  # Compute low-rank update
        return self.original_layer(x) + (x @ delta_w.T) * self.scale


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
        self.config = original_llama_mlp.config

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


class CustomLlamaAttention(nn.Module):
    """Multi-headed attention from 'Attention Is All You Need' paper
    Taken from https://github.com/huggingface/transformers/blob/main/src/transformers/models/llama/modeling_llama.py"""

    def __init__(self, originalmodule: LlamaAttention, kwargs):
        super().__init__()
        config = originalmodule.config
        self.config = originalmodule.config
        self.layer_idx = originalmodule.layer_idx
        self.head_dim = originalmodule.head_dim
        self.num_key_value_groups = originalmodule.num_key_value_groups
        self.scaling = originalmodule.scaling
        self.attention_dropout = originalmodule.attention_dropout
        self.is_causal = originalmodule.is_causal

        self.q_proj = originalmodule.q_proj
        self.k_proj = originalmodule.k_proj
        self.v_proj = originalmodule.v_proj
        self.o_proj = originalmodule.o_proj
        
        self.interveened_q_proj = LoRALayer(originalmodule.q_proj, kwargs['lora_attn_rank'], kwargs['lora_attn_alpha'])
        self.interveened_k_proj = LoRALayer(originalmodule.k_proj, kwargs['lora_attn_rank'], kwargs['lora_attn_alpha'])
        self.interveened_v_proj = LoRALayer(originalmodule.v_proj, kwargs['lora_attn_rank'], kwargs['lora_attn_alpha'])
        self.interveened_o_proj = LoRALayer(originalmodule.o_proj, kwargs['lora_attn_rank'], kwargs['lora_attn_alpha'])

        for param in list(self.q_proj.parameters()) + \
                        list(self.k_proj.parameters()) + \
                        list(self.v_proj.parameters()) + \
                        list(self.o_proj.parameters()):
            param.requires_grad = False

    def forward(
        self,
        hidden_states: torch.Tensor,
        position_embeddings: Tuple[torch.Tensor, torch.Tensor],
        attention_mask: Optional[torch.Tensor],
        past_key_value: Optional[Cache] = None,
        cache_position: Optional[torch.LongTensor] = None,
        **kwargs: Unpack[FlashAttentionKwargs],
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor], Optional[Tuple[torch.Tensor]]]:
        input_shape = hidden_states.shape[:-1]
        hidden_shape = (*input_shape, -1, self.head_dim)

        query_states = self.q_proj(hidden_states).view(hidden_shape).transpose(1, 2)
        key_states = self.k_proj(hidden_states).view(hidden_shape).transpose(1, 2)
        value_states = self.v_proj(hidden_states).view(hidden_shape).transpose(1, 2)

        cos, sin = position_embeddings
        query_states, key_states = apply_rotary_pos_emb(query_states, key_states, cos, sin)

        if past_key_value is not None:
            # sin and cos are specific to RoPE models; cache_position needed for the static cache
            cache_kwargs = {"sin": sin, "cos": cos, "cache_position": cache_position}
            key_states, value_states = past_key_value.update(key_states, value_states, self.layer_idx, cache_kwargs)

        attention_interface: Callable = eager_attention_forward
        if self.config._attn_implementation != "eager":
            if self.config._attn_implementation == "sdpa" and kwargs.get("output_attentions", False):
                logger.warning_once(
                    "`torch.nn.functional.scaled_dot_product_attention` does not support `output_attentions=True`. Falling back to "
                    'eager attention. This warning can be removed using the argument `attn_implementation="eager"` when loading the model.'
                )
            else:
                attention_interface = ALL_ATTENTION_FUNCTIONS[self.config._attn_implementation]

        attn_output, attn_weights = attention_interface(
            self,
            query_states,
            key_states,
            value_states,
            attention_mask,
            dropout=0.0 if not self.training else self.attention_dropout,
            scaling=self.scaling,
            **kwargs,
        )

        attn_output = attn_output.reshape(*input_shape, -1).contiguous()
        attn_output = self.o_proj(attn_output)
        return attn_output, attn_weights


    def interveened_forward(
        self,
        hidden_states: torch.Tensor,
        position_embeddings: Tuple[torch.Tensor, torch.Tensor],
        attention_mask: Optional[torch.Tensor],
        past_key_value: Optional[Cache] = None,
        cache_position: Optional[torch.LongTensor] = None,
        **kwargs: Unpack[FlashAttentionKwargs],
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor], Optional[Tuple[torch.Tensor]]]:
        input_shape = hidden_states.shape[:-1]
        hidden_shape = (*input_shape, -1, self.head_dim)

        query_states = self.interveened_q_proj(hidden_states).view(hidden_shape).transpose(1, 2)
        key_states = self.interveened_k_proj(hidden_states).view(hidden_shape).transpose(1, 2)
        value_states = self.interveened_v_proj(hidden_states).view(hidden_shape).transpose(1, 2)

        cos, sin = position_embeddings
        query_states, key_states = apply_rotary_pos_emb(query_states, key_states, cos, sin)

        """
        # The past_key_value has already been updated in the normal forward pass, and the key_states and value_states were not 
        # updated in the normal forward pass
        if past_key_value is not None:
            # sin and cos are specific to RoPE models; cache_position needed for the static cache
            cache_kwargs = {"sin": sin, "cos": cos, "cache_position": cache_position}
            key_states, value_states = past_key_value.update(key_states, value_states, self.layer_idx, cache_kwargs)
        """

        attention_interface: Callable = eager_attention_forward
        if self.config._attn_implementation != "eager":
            if self.config._attn_implementation == "sdpa" and kwargs.get("output_attentions", False):
                logger.warning_once(
                    "`torch.nn.functional.scaled_dot_product_attention` does not support `output_attentions=True`. Falling back to "
                    'eager attention. This warning can be removed using the argument `attn_implementation="eager"` when loading the model.'
                )
            else:
                attention_interface = ALL_ATTENTION_FUNCTIONS[self.config._attn_implementation]

        attn_output, attn_weights = attention_interface(
            self,
            query_states,
            key_states,
            value_states,
            attention_mask,
            dropout=0.0 if not self.training else self.attention_dropout,
            scaling=self.scaling,
            **kwargs,
        )

        attn_output = attn_output.reshape(*input_shape, -1).contiguous()
        attn_output = self.interveened_o_proj(attn_output)
        return attn_output, attn_weights


class LlamaBlockDefendor(nn.Module):
    
    def __init__(self, originalBlock, low_rank_dimension, kwargs):  # originalBlock is a LlamaDecoderLayer
        super().__init__()

        self.hidden_size = originalBlock.hidden_size
        # self.self_attn = originalBlock.self_attn
        self.self_attn = CustomLlamaAttention(originalBlock.self_attn, kwargs)
        self.mlp = CustomLlamaMLP(originalBlock.mlp, low_rank_dimension, kwargs)

        self.input_layernorm = originalBlock.input_layernorm
        self.post_attention_layernorm = originalBlock.post_attention_layernorm

        for param in list(self.input_layernorm.parameters()) + \
                     list(self.post_attention_layernorm.parameters()):
            param.requires_grad = False

        self.pre_mlp_res_cache = None
        self.mlp_output_cache = None

    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_value: Optional[Cache] = None,
        output_attentions: Optional[bool] = False,
        use_cache: Optional[bool] = False,
        cache_position: Optional[torch.LongTensor] = None,
        position_embeddings: Optional[Tuple[torch.Tensor, torch.Tensor]] = None,  # necessary, but kept here for BC
        **kwargs: Unpack[FlashAttentionKwargs],
    ) -> Tuple[torch.FloatTensor, Optional[Tuple[torch.FloatTensor, torch.FloatTensor]]]:
        """
        This is the original forward method of the LlamaDecoderLayer pytorch implementation from huggingface copied from:
        https://github.com/huggingface/transformers/blob/v4.48.0/src/transformers/models/llama/modeling_llama.py#L82
        """
        residual = hidden_states

        hidden_states = self.input_layernorm(hidden_states)

        # Self Attention
        hidden_states, self_attn_weights = self.self_attn(
            hidden_states=hidden_states,
            attention_mask=attention_mask,
            position_ids=position_ids,
            past_key_value=past_key_value,
            output_attentions=output_attentions,
            use_cache=use_cache,
            cache_position=cache_position,
            position_embeddings=position_embeddings,
            **kwargs,
        )

        hidden_states = residual + hidden_states

        # Fully Connected
        residual = hidden_states

        self.pre_mlp_res_cache = residual
        
        hidden_states = self.post_attention_layernorm(hidden_states)
        mlp_output = self.mlp(hidden_states)

        self.mlp_output_cache = mlp_output

        hidden_states = residual + mlp_output

        outputs = (hidden_states,)

        if output_attentions:
            outputs += (self_attn_weights,)

        return outputs


    def interveened_forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_value: Optional[Cache] = None,
        output_attentions: Optional[bool] = False,
        use_cache: Optional[bool] = False,
        cache_position: Optional[torch.LongTensor] = None,
        position_embeddings: Optional[Tuple[torch.Tensor, torch.Tensor]] = None,  # necessary, but kept here for BC
        **kwargs: Unpack[FlashAttentionKwargs],
    ) -> Tuple[torch.FloatTensor, Optional[Tuple[torch.FloatTensor, torch.FloatTensor]]]:
          
        residual = hidden_states

        hidden_states = self.input_layernorm(hidden_states)

        # Self Attention
        hidden_states, self_attn_weights = self.self_attn.interveened_forward(
            hidden_states=hidden_states,
            attention_mask=attention_mask,
            position_ids=position_ids,
            past_key_value=past_key_value,
            output_attentions=output_attentions,
            use_cache=use_cache,
            cache_position=cache_position,
            position_embeddings=position_embeddings,
            **kwargs,
        )
        
        hidden_states = residual + hidden_states

        # Fully Connected
        residual = hidden_states

        self.pre_mlp_res_cache = residual

        hidden_states = self.post_attention_layernorm(hidden_states)

        mlp_output = self.mlp.intervened_forward(hidden_states)

        self.mlp_output_cache = mlp_output  # we do not detach here, as these gradients will drive the learning...

        hidden_states = residual + mlp_output

        outputs = (hidden_states,)

        if output_attentions:
            outputs += (self_attn_weights,)

        return outputs



class LlamaInputsCatcher():

    def __init__(self, llama_model):
        self.llama_model = llama_model


    def get_inputs_dict(
        self,
        input_ids: torch.LongTensor = None,
        attention_mask: Optional[torch.Tensor] = None
    ) -> dict:

        position_ids = None
        past_key_values = None
        inputs_embeds = None
        use_cache = None
        output_attentions = None
        output_hidden_states = None
        return_dict = None
        cache_position = None


        output_attentions = output_attentions if output_attentions is not None else self.llama_model.config.output_attentions
        output_hidden_states = (
            output_hidden_states if output_hidden_states is not None else self.llama_model.config.output_hidden_states
        )
        use_cache = use_cache if use_cache is not None else self.llama_model.config.use_cache
        return_dict = return_dict if return_dict is not None else self.llama_model.config.use_return_dict

        if (input_ids is None) ^ (inputs_embeds is not None):
            raise ValueError("You must specify exactly one of input_ids or inputs_embeds")

        if self.llama_model.gradient_checkpointing and self.llama_model.training and use_cache:
            logger.warning_once(
                "`use_cache=True` is incompatible with gradient checkpointing. Setting `use_cache=False`."
            )
            use_cache = False

        if inputs_embeds is None:
            inputs_embeds = self.llama_model.embed_tokens(input_ids)

        if use_cache and past_key_values is None:
            past_key_values = DynamicCache()

        if cache_position is None:
            past_seen_tokens = past_key_values.get_seq_length() if past_key_values is not None else 0
            cache_position = torch.arange(
                past_seen_tokens, past_seen_tokens + inputs_embeds.shape[1], device=inputs_embeds.device
            )

        if position_ids is None:
            position_ids = cache_position.unsqueeze(0)

        causal_mask = self.llama_model._update_causal_mask(
            attention_mask, inputs_embeds, cache_position, past_key_values, output_attentions
        )

        hidden_states = inputs_embeds

        # create position embeddings to be shared across the decoder layers
        position_embeddings = self.llama_model.rotary_emb(hidden_states, position_ids)

        return {
            "position_ids": position_ids,
            "past_key_values": past_key_values,
            "inputs_embeds": inputs_embeds,
            "use_cache": use_cache,
            "output_attentions": output_attentions,
            "output_hidden_states": output_hidden_states,
            "return_dict": return_dict,
            "cache_position": cache_position,
            "causal_mask": causal_mask,
            "position_embeddings": position_embeddings
        }


    def _update_causal_mask(
        self,
        attention_mask: torch.Tensor,
        input_tensor: torch.Tensor,
        cache_position: torch.Tensor,
        past_key_values: Cache,
        output_attentions: bool,
    ):
        if self.llama_model.config._attn_implementation == "flash_attention_2":
            if attention_mask is not None and (attention_mask == 0.0).any():
                return attention_mask
            return None

        # For SDPA, when possible, we will rely on its `is_causal` argument instead of its `attn_mask` argument, in
        # order to dispatch on Flash Attention 2. This feature is not compatible with static cache, as SDPA will fail
        # to infer the attention mask.
        past_seen_tokens = past_key_values.get_seq_length() if past_key_values is not None else 0
        using_static_cache = isinstance(past_key_values, StaticCache)

        # When output attentions is True, sdpa implementation's forward method calls the eager implementation's forward
        if self.llama_model.config._attn_implementation == "sdpa" and not using_static_cache and not output_attentions:
            if AttentionMaskConverter._ignore_causal_mask_sdpa(
                attention_mask,
                inputs_embeds=input_tensor,
                past_key_values_length=past_seen_tokens,
                is_training=self.llama_model.training,
            ):
                return None

        dtype, device = input_tensor.dtype, input_tensor.device
        sequence_length = input_tensor.shape[1]
        if using_static_cache:
            target_length = past_key_values.get_max_cache_shape()
        else:
            target_length = (
                attention_mask.shape[-1]
                if isinstance(attention_mask, torch.Tensor)
                else past_seen_tokens + sequence_length + 1
            )

        # In case the provided `attention` mask is 2D, we generate a causal mask here (4D).
        causal_mask = self.llama_model._prepare_4d_causal_attention_mask_with_cache_position(
            attention_mask,
            sequence_length=sequence_length,
            target_length=target_length,
            dtype=dtype,
            device=device,
            cache_position=cache_position,
            batch_size=input_tensor.shape[0],
        )

        if (
            self.llama_model.config._attn_implementation == "sdpa"
            and attention_mask is not None
            and attention_mask.device.type == "cuda"
            and not output_attentions
        ):
            # Attend to all tokens in fully masked rows in the causal_mask, for example the relevant first rows when
            # using left padding. This is required by F.scaled_dot_product_attention memory-efficient attention path.
            # Details: https://github.com/pytorch/pytorch/issues/110213
            min_dtype = torch.finfo(dtype).min
            causal_mask = AttentionMaskConverter._unmask_unattended(causal_mask, min_dtype)

        return causal_mask

    @staticmethod
    def _prepare_4d_causal_attention_mask_with_cache_position(
        attention_mask: torch.Tensor,
        sequence_length: int,
        target_length: int,
        dtype: torch.dtype,
        device: torch.device,
        cache_position: torch.Tensor,
        batch_size: int,
        **kwargs,
    ):
        """
        Creates a causal 4D mask of shape `(batch_size, 1, query_length, key_value_length)` from a 2D mask of shape
        `(batch_size, key_value_length)`, or if the input `attention_mask` is already 4D, do nothing.

        Args:
            attention_mask (`torch.Tensor`):
                A 2D attention mask of shape `(batch_size, key_value_length)` or a 4D attention mask of shape
                `(batch_size, 1, query_length, key_value_length)`.
            sequence_length (`int`):
                The sequence length being processed.
            target_length (`int`):
                The target length: when generating with static cache, the mask should be as long as the static cache,
                to account for the 0 padding, the part of the cache that is not filled yet.
            dtype (`torch.dtype`):
                The dtype to use for the 4D attention mask.
            device (`torch.device`):
                The device to plcae the 4D attention mask on.
            cache_position (`torch.Tensor`):
                Indices depicting the position of the input sequence tokens in the sequence.
            batch_size (`torch.Tensor`):
                Batch size.
        """
        if attention_mask is not None and attention_mask.dim() == 4:
            # In this case we assume that the mask comes already in inverted form and requires no inversion or slicing.
            causal_mask = attention_mask
        else:
            min_dtype = torch.finfo(dtype).min
            causal_mask = torch.full(
                (sequence_length, target_length), fill_value=min_dtype, dtype=dtype, device=device
            )
            if sequence_length != 1:
                causal_mask = torch.triu(causal_mask, diagonal=1)
            causal_mask *= torch.arange(target_length, device=device) > cache_position.reshape(-1, 1)
            causal_mask = causal_mask[None, None, :, :].expand(batch_size, 1, -1, -1)
            if attention_mask is not None:
                causal_mask = causal_mask.clone()  # copy to contiguous memory for in-place edit
                mask_length = attention_mask.shape[-1]
                padding_mask = causal_mask[:, :, :, :mask_length] + attention_mask[:, None, None, :]
                padding_mask = padding_mask == 0
                causal_mask[:, :, :, :mask_length] = causal_mask[:, :, :, :mask_length].masked_fill(
                    padding_mask, min_dtype
                )

        return causal_mask
