# Copyright (c) Meta Platforms, Inc. and affiliates.
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.


from collections import OrderedDict
from dataclasses import dataclass
from typing import Tuple, Union

import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F
from timm.models.layers import to_2tuple

from torch.utils.checkpoint import checkpoint

from diht.build import MODELS_REGISTRY


class LayerNorm(nn.LayerNorm):
    """Subclass torch's LayerNorm to handle fp16."""

    def forward(self, x):
        orig_type = x.dtype
        x = F.layer_norm(x, self.normalized_shape, self.weight, self.bias, self.eps)
        return x.to(orig_type)


class QuickGELU(nn.Module):
    # NOTE This is slower than nn.GELU or nn.SiLU and uses more GPU memory
    def forward(self, x):
        return x * torch.sigmoid(1.702 * x)


class ResidualAttentionBlock(nn.Module):
    def __init__(
        self,
        d_model,
        n_head,
        mlp_ratio=4.0,
        act_layer=nn.GELU,
    ):
        super().__init__()

        self.attn = nn.MultiheadAttention(d_model, n_head)
        self.ln_1 = LayerNorm(d_model)
        mlp_width = int(d_model * mlp_ratio)
        self.mlp = nn.Sequential(
            OrderedDict(
                [
                    ("c_fc", nn.Linear(d_model, mlp_width)),
                    ("gelu", act_layer()),
                    ("c_proj", nn.Linear(mlp_width, d_model)),
                ]
            )
        )
        self.ln_2 = LayerNorm(d_model)

    def attention(
        self,
        x,
        attn_mask=None,
        key_padding_mask=None,
    ):
        return self.attn(
            x,
            x,
            x,
            need_weights=False,
            attn_mask=attn_mask,
            key_padding_mask=key_padding_mask,
        )[0]

    def forward(
        self,
        x,
        attn_mask=None,
        key_padding_mask=None,
    ):
        x = x + self.attention(
            self.ln_1(x), attn_mask=attn_mask, key_padding_mask=key_padding_mask
        )
        x = x + self.mlp(self.ln_2(x))
        return x


class Transformer(nn.Module):
    def __init__(
        self,
        width,
        layers,
        heads,
        mlp_ratio=4.0,
        act_layer=nn.GELU,
        log_last_actvnorm=False,
    ):
        super().__init__()
        self.width = width
        self.layers = layers
        self.grad_checkpointing = False

        self.resblocks = nn.ModuleList(
            [
                ResidualAttentionBlock(width, heads, mlp_ratio, act_layer=act_layer)
                for _ in range(layers)
            ]
        )

        self._log_last_actvnorm = log_last_actvnorm
        self._last_actvnorm = None

    @property
    def last_actvnorm(self):
        return self._last_actvnorm

    def forward(
        self,
        x,
        attn_mask=None,
        key_padding_mask=None,
    ):
        for r in self.resblocks:
            if self.grad_checkpointing and not torch.jit.is_scripting():
                x = checkpoint(r, x, attn_mask, key_padding_mask)
            else:
                x = r(x, attn_mask=attn_mask, key_padding_mask=key_padding_mask)

        if self._log_last_actvnorm is True:
            self._last_actvnorm = x.detach().data.norm(2)

        return x


class VisualTransformer(nn.Module):
    def __init__(
        self,
        image_size,
        patch_size,
        width,
        layers,
        heads,
        mlp_ratio,
        output_dim,
        act_layer=nn.GELU,
        log_last_actvnorm=False,
    ):
        super().__init__()
        self.image_size = to_2tuple(image_size)
        self.patch_size = to_2tuple(patch_size)
        self.grid_size = (
            self.image_size[0] // self.patch_size[0],
            self.image_size[1] // self.patch_size[1],
        )
        self.output_dim = output_dim
        self.conv1 = nn.Conv2d(
            in_channels=3,
            out_channels=width,
            kernel_size=patch_size,
            stride=patch_size,
            bias=False,
        )

        scale = width**-0.5
        self.class_embedding = nn.Parameter(scale * torch.randn(width))
        self.positional_embedding = nn.Parameter(
            scale * torch.randn(self.grid_size[0] * self.grid_size[1] + 1, width)
        )
        self.ln_pre = LayerNorm(width)

        self.transformer = Transformer(
            width, layers, heads, mlp_ratio, act_layer=act_layer
        )

        self.ln_post = LayerNorm(width)
        self.proj = nn.Parameter(scale * torch.randn(width, output_dim))

        self._log_last_actvnorm = log_last_actvnorm
        self._last_actvnorm = None

    def lock(self, unlocked_groups=0, freeze_bn_stats=False):
        assert (
            unlocked_groups == 0
        ), "partial locking not currently supported for this model"
        for param in self.parameters():
            param.requires_grad = False

    @torch.jit.ignore
    def set_grad_checkpointing(self, enable=True):
        self.transformer.grad_checkpointing = enable

    @property
    def last_actvnorm(self):
        return self._last_actvnorm

    def forward(self, x):
        x = self.conv1(x)  # shape = [*, width, grid, grid]
        x = x.reshape(x.shape[0], x.shape[1], -1)  # shape = [*, width, grid ** 2]
        x = x.permute(0, 2, 1)  # shape = [*, grid ** 2, width]
        x = torch.cat(
            [
                self.class_embedding.to(x.dtype)
                + torch.zeros(
                    x.shape[0], 1, x.shape[-1], dtype=x.dtype, device=x.device
                ),
                x,
            ],
            dim=1,
        )  # shape = [*, grid ** 2 + 1, width]
        x = x + self.positional_embedding.to(x.dtype)
        x = self.ln_pre(x)

        x = x.permute(1, 0, 2)  # NLD -> LND
        x = self.transformer(x)
        x = x.permute(1, 0, 2)  # LND -> NLD

        if self._log_last_actvnorm is True:
            self._last_actvnorm = x.detach().data.norm(2)

        x = self.ln_post(x[:, 0, :])

        if self.proj is not None:
            x = x @ self.proj

        return x


@dataclass
class DiHTVisionCfg:
    layers: Union[Tuple[int, int, int, int], int] = 12
    width: int = 768
    head_width: int = 64
    mlp_ratio: float = 4.0
    patch_size: int = 16
    image_size: Union[Tuple[int, int], int] = 224
    image_resize_res: Union[Tuple[int, int], int] = None
    project_vision_fts: bool = False


@dataclass
class DiHTTextCfg:
    context_length: int = 77
    vocab_size: int = 49408
    width: int = 512
    heads: int = 8
    layers: int = 12
    causal_mask: bool = True
    cls_token_id: int = None
    pad_token_id: int = 0
    pooling: str = "max_token_id"


@MODELS_REGISTRY.register()
class DiHT(nn.Module):
    def __init__(
        self,
        embed_dim,
        vision_cfg,
        text_cfg,
        quick_gelu=False,
    ):
        super().__init__()

        if isinstance(vision_cfg, dict):
            vision_cfg = DiHTVisionCfg(**vision_cfg)
        if isinstance(text_cfg, dict):
            text_cfg = DiHTTextCfg(**text_cfg)

        self.context_length = text_cfg.context_length

        act_layer = QuickGELU if quick_gelu else nn.GELU

        self.project_vision = None

        vision_heads = vision_cfg.width // vision_cfg.head_width
        self.visual = VisualTransformer(
            image_size=vision_cfg.image_size,
            patch_size=vision_cfg.patch_size,
            width=vision_cfg.width,
            layers=vision_cfg.layers,
            heads=vision_heads,
            mlp_ratio=vision_cfg.mlp_ratio,
            output_dim=embed_dim,
            act_layer=act_layer,
        )

        self.transformer = Transformer(
            width=text_cfg.width,
            layers=text_cfg.layers,
            heads=text_cfg.heads,
            act_layer=act_layer,
        )

        self.vocab_size = text_cfg.vocab_size
        self.token_embedding = nn.Embedding(text_cfg.vocab_size, text_cfg.width)
        self.positional_embedding = nn.Parameter(
            torch.empty(self.context_length, text_cfg.width)
        )
        self.ln_final = LayerNorm(text_cfg.width)

        self.causal_mask = text_cfg.causal_mask
        self.cls_token_id = text_cfg.cls_token_id
        self.pad_token_id = text_cfg.pad_token_id
        self.text_pooling = text_cfg.pooling

        self.text_projection = nn.Parameter(torch.empty(text_cfg.width, embed_dim))
        self.logit_scale = nn.Parameter(torch.ones([]) * np.log(1 / 0.07))
        self.register_buffer("attn_mask", self.build_attention_mask(), persistent=False)

        self.init_parameters()

    def init_parameters(self):
        nn.init.normal_(self.token_embedding.weight, std=0.02)
        nn.init.normal_(self.positional_embedding, std=0.01)
        nn.init.constant_(self.logit_scale, np.log(1 / 0.07))

        if hasattr(self.visual, "init_parameters"):
            self.visual.init_parameters()

        proj_std = (self.transformer.width**-0.5) * (
            (2 * self.transformer.layers) ** -0.5
        )
        attn_std = self.transformer.width**-0.5
        fc_std = (2 * self.transformer.width) ** -0.5
        for block in self.transformer.resblocks:
            nn.init.normal_(block.attn.in_proj_weight, std=attn_std)
            nn.init.normal_(block.attn.out_proj.weight, std=proj_std)
            nn.init.normal_(block.mlp.c_fc.weight, std=fc_std)
            nn.init.normal_(block.mlp.c_proj.weight, std=proj_std)

        if self.text_projection is not None:
            nn.init.normal_(self.text_projection, std=self.transformer.width**-0.5)

    def build_attention_mask(self):
        if self.causal_mask:
            # lazily create causal attention mask, with full attention between the vision
            # tokens; pytorch uses additive attention mask; fill with -inf
            mask = torch.empty(self.context_length, self.context_length)
            mask.fill_(float("-inf"))
            mask.triu_(1)  # zero out the lower diagonal
        else:
            # we dont want causal mask, we want bidirectional transformer
            # mask = torch.ones(self.context_length, self.context_length)
            mask = None
        return mask

    @torch.jit.ignore
    def set_grad_checkpointing(self, enable=True):
        self.visual.set_grad_checkpointing(enable)
        self.transformer.grad_checkpointing = enable

    def encode_image(self, image):
        out = self.visual(image)
        if self.project_vision is not None:
            out = self.project_vision(out)
        return out

    def pool_from_encoded_text(self, encoded_text, text):
        if self.text_pooling == "before_max_token_id":
            # use the token before the token with highest id
            single_token_pool_ids = text.argmax(dim=-1) - 1
        else:
            # max_token_id -- default clip pooling
            if self.causal_mask:
                if self.cls_token_id is None:
                    # take features from the <end_of_text> embedding
                    # (eot_token is the highest number in each sequence)
                    # this is the default clip
                    single_token_pool_ids = text.argmax(dim=-1)
                else:
                    # take features from the cls_token_id
                    # which is provided via config
                    single_token_pool_ids = (text == self.cls_token_id).nonzero(
                        as_tuple=True
                    )[1]
            else:
                # take features from the <start_of_text> embedding
                # now that we have bidirectional transformer encoder
                # (sot_token in the first token in each sequence)
                single_token_pool_ids = 0

        encoded_text = encoded_text[
            torch.arange(encoded_text.shape[0]), single_token_pool_ids
        ]

        return encoded_text

    def encode_text_tokens(self, text):
        x = self.token_embedding(text)  # [batch_size, n_ctx, d_model]

        x = x + self.positional_embedding
        x = x.permute(1, 0, 2)  # NLD -> LND
        if self.causal_mask:
            x = self.transformer(x, attn_mask=self.attn_mask)
        else:
            key_padding_mask = text == self.pad_token_id
            x = self.transformer(
                x, attn_mask=self.attn_mask, key_padding_mask=key_padding_mask
            )
        x = x.permute(1, 0, 2)  # LND -> NLD
        x = self.ln_final(x)

        return x

    def encode_text_with_tokens(self, text) -> Tuple[torch.Tensor, torch.Tensor]:
        # [batch_size, n_ctx, transformer.width]
        x_tokens = self.encode_text_tokens(text)

        # pool the representation represntation
        x_out = self.pool_from_encoded_text(x_tokens, text)
        x_out = x_out @ self.text_projection

        return x_out, x_tokens

    def encode_text(self, text, return_all=False):
        x = self.encode_text_tokens(text)

        # x.shape = [batch_size, n_ctx, transformer.width]
        if return_all:
            return x @ self.text_projection

        # pool the representation represntation
        x = self.pool_from_encoded_text(x, text)
        x = x @ self.text_projection

        return x

    def forward(self, image, text, return_text_tokens=False):
        if isinstance(text, list) is True:
            text = torch.stack(text)
        if image is None:
            return self.encode_text(text)
        elif text is None:
            return self.encode_image(image)

        image_features = self.encode_image(image)
        image_features = F.normalize(image_features, dim=-1)

        if not return_text_tokens:
            text_features = self.encode_text(text)
            text_features = F.normalize(text_features, dim=-1)
            return image_features, text_features, self.logit_scale.exp()
        else:
            text_features, text_token_features = self.encode_text_with_tokens(text)
            text_features = F.normalize(text_features, dim=-1)
            text_token_features = F.normalize(text_token_features, dim=-1)

            return (
                image_features,
                text_features,
                self.logit_scale.exp(),
                text_token_features,
            )
