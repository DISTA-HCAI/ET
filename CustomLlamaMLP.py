import torch.nn as nn
import torch

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


    def intervene_forward(self, x):

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