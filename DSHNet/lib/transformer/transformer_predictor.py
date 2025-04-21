import fvcore.nn.weight_init as weight_init
from torch import nn
from torch.nn import functional as F

from .position_encoding import PositionEmbeddingSine
from .transformer import Transformer


class TransformerPredictor(nn.Module):
    def __init__(
            self,
            in_channels,
            hidden_dim: int,
            num_queries: int,
            nheads: int,
            dropout: float,
            dim_feedforward: int,
            enc_layers: int,
            dec_layers: int,
            pre_norm: bool,
            deep_supervision: bool,
            mask_dim: int,
            enforce_input_project: bool,
            base_c: int,
            num_stage:int=5):
        super().__init__()

        # positional encoding
        N_steps = hidden_dim // 2
        self.pe_layer = PositionEmbeddingSine(N_steps, normalize=True)
        self.base_C = base_c

        transformer = Transformer(
            d_model=hidden_dim,
            dropout=dropout,
            nhead=nheads,
            dim_feedforward=dim_feedforward,
            num_encoder_layers=enc_layers,
            num_decoder_layers=dec_layers,
            normalize_before=pre_norm,
            return_intermediate_dec=deep_supervision,
        )

        self.num_queries = num_queries*4
        self.transformer = transformer
        hidden_dim = transformer.d_model
        self.num_stage=num_stage
        if in_channels != hidden_dim or enforce_input_project:
            self.input_proj = nn.Conv2d(in_channels, hidden_dim, kernel_size=1)
            weight_init.c2_xavier_fill(self.input_proj)
        else:
            self.input_proj = nn.Sequential()
        self.aux_loss = deep_supervision

        self.kernel_embed = MLP(hidden_dim, hidden_dim * 2, self.base_C * self.base_C * 9, 2)

        self.layer_embed = MLP(9, 9 ** 2, 9 * num_stage, 2)

        self.mask_dim = mask_dim

    def forward(self, x, query):
        b, _, _, _ = x.size()
        pos = self.pe_layer(x)
        src = x
        mask = None
        hs, memory = self.transformer(self.input_proj(src), mask, query, pos)

        kernel_pred = self.kernel_embed(hs[-1])
        
        kernel_pred = kernel_pred.view(b, self.num_queries, self.base_C, self.base_C, 9)
        layer_kernel_pred = self.layer_embed(kernel_pred).view(b, self.num_queries, self.base_C, self.base_C, 3, 3, self.num_stage)

        region_layer_kernel = [layer_kernel_pred[:, :, :, :, :, :, i] for i in range(self.num_stage)]

        return region_layer_kernel
    

class MLP(nn.Module):

    def __init__(self, input_dim, hidden_dim, output_dim, num_layers):
        super().__init__()
        self.num_layers = num_layers
        h = [hidden_dim] * (num_layers - 1)
        self.layers = nn.ModuleList(
            nn.Linear(n, k) for n, k in zip([input_dim] + h, h + [output_dim])
        )

    def forward(self, x):
        for i, layer in enumerate(self.layers):
            x = F.relu(layer(x)) if i < self.num_layers - 1 else layer(x)
        return x

