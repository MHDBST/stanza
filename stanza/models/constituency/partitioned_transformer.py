"""
Transformer with partitioned content and position features.

See section 3 of https://arxiv.org/pdf/1805.01052.pdf
"""

import copy
import math

import torch
import torch.nn as nn
import torch.nn.functional as F

from stanza.models.constituency.positional_encoding import ConcatSinusoidalEncoding

LAYER_MIX_BIAS = 0.1

class FeatureDropoutFunction(torch.autograd.function.InplaceFunction):
    @staticmethod
    def forward(ctx, input, p=0.5, train=False, inplace=False):
        if p < 0 or p > 1:
            raise ValueError(
                "dropout probability has to be between 0 and 1, but got {}".format(p)
            )

        ctx.p = p
        ctx.train = train
        ctx.inplace = inplace

        if ctx.inplace:
            ctx.mark_dirty(input)
            output = input
        else:
            output = input.clone()

        if ctx.p > 0 and ctx.train:
            ctx.noise = torch.empty(
                (input.size(0), input.size(-1)),
                dtype=input.dtype,
                layout=input.layout,
                device=input.device,
            )
            if ctx.p == 1:
                ctx.noise.fill_(0)
            else:
                ctx.noise.bernoulli_(1 - ctx.p).div_(1 - ctx.p)
            ctx.noise = ctx.noise[:, None, :]
            output.mul_(ctx.noise)

        return output

    @staticmethod
    def backward(ctx, grad_output):
        if ctx.p > 0 and ctx.train:
            return grad_output.mul(ctx.noise), None, None, None
        else:
            return grad_output, None, None, None


class FeatureDropout(nn.Dropout):
    """
    Feature-level dropout: takes an input of size len x num_features and drops
    each feature with probabibility p. A feature is dropped across the full
    portion of the input that corresponds to a single batch element.
    """

    def forward(self, x):
        if isinstance(x, tuple):
            x_c, x_p = x
            x_c = FeatureDropoutFunction.apply(x_c, self.p, self.training, self.inplace)
            x_p = FeatureDropoutFunction.apply(x_p, self.p, self.training, self.inplace)
            return x_c, x_p
        else:
            return FeatureDropoutFunction.apply(x, self.p, self.training, self.inplace)


# TODO: this module apparently is not treated the same the built-in
# nonlinearity modules, as multiple uses of the same relu on different
# tensors winds up mixing the gradients See if there is a way to
# resolve that other than creating a new nonlinearity for each layer
class PartitionedReLU(nn.ReLU):
    def forward(self, x):
        if isinstance(x, tuple):
            x_c, x_p = x
        else:
            x_c, x_p = torch.chunk(x, 2, dim=-1)
        return super().forward(x_c), super().forward(x_p)


class PartitionedLinear(nn.Module):
    def __init__(self, in_features, out_features, bias=True):
        super().__init__()
        self.linear_c = nn.Linear(in_features // 2, out_features // 2, bias)
        self.linear_p = nn.Linear(in_features // 2, out_features // 2, bias)

    def forward(self, x):
        if isinstance(x, tuple):
            x_c, x_p = x
        else:
            x_c, x_p = torch.chunk(x, 2, dim=-1)

        out_c = self.linear_c(x_c)
        out_p = self.linear_p(x_p)
        return out_c, out_p


class PartitionedMultiHeadAttention(nn.Module):
    def __init__(
        self, d_model, n_head, d_qkv, attention_dropout=0.1, initializer_range=0.02
    ):
        super().__init__()

        self.w_qkv_c = nn.Parameter(torch.Tensor(n_head, d_model // 2, 3, d_qkv // 2))
        self.w_qkv_p = nn.Parameter(torch.Tensor(n_head, d_model // 2, 3, d_qkv // 2))
        self.w_o_c = nn.Parameter(torch.Tensor(n_head, d_qkv // 2, d_model // 2))
        self.w_o_p = nn.Parameter(torch.Tensor(n_head, d_qkv // 2, d_model // 2))

        bound = math.sqrt(3.0) * initializer_range
        for param in [self.w_qkv_c, self.w_qkv_p, self.w_o_c, self.w_o_p]:
            nn.init.uniform_(param, -bound, bound)
        self.scaling_factor = 1 / d_qkv ** 0.5

        self.dropout = nn.Dropout(attention_dropout)

    def forward(self, x, mask=None):
        if isinstance(x, tuple):
            x_c, x_p = x
        else:
            x_c, x_p = torch.chunk(x, 2, dim=-1)
        qkv_c = torch.einsum("btf,hfca->bhtca", x_c, self.w_qkv_c)
        qkv_p = torch.einsum("btf,hfca->bhtca", x_p, self.w_qkv_p)
        q_c, k_c, v_c = [c.squeeze(dim=3) for c in torch.chunk(qkv_c, 3, dim=3)]
        q_p, k_p, v_p = [c.squeeze(dim=3) for c in torch.chunk(qkv_p, 3, dim=3)]
        q = torch.cat([q_c, q_p], dim=-1) * self.scaling_factor
        k = torch.cat([k_c, k_p], dim=-1)
        v = torch.cat([v_c, v_p], dim=-1)
        dots = torch.einsum("bhqa,bhka->bhqk", q, k)
        if mask is not None:
            dots.data.masked_fill_(~mask[:, None, None, :], -float("inf"))
        probs = F.softmax(dots, dim=-1)
        probs = self.dropout(probs)
        o = torch.einsum("bhqk,bhka->bhqa", probs, v)
        o_c, o_p = torch.chunk(o, 2, dim=-1)
        out_c = torch.einsum("bhta,haf->btf", o_c, self.w_o_c)
        out_p = torch.einsum("bhta,haf->btf", o_p, self.w_o_p)
        return out_c, out_p


class PartitionedTransformerEncoderLayer(nn.Module):
    def __init__(self,
                 d_model,
                 n_head,
                 d_qkv,
                 d_ff,
                 ff_dropout,
                 residual_dropout,
                 attention_dropout,
                 activation=PartitionedReLU(),
    ):
        super().__init__()
        self.self_attn = PartitionedMultiHeadAttention(
            d_model, n_head, d_qkv, attention_dropout=attention_dropout
        )
        self.linear1 = PartitionedLinear(d_model, d_ff)
        self.ff_dropout = FeatureDropout(ff_dropout)
        self.linear2 = PartitionedLinear(d_ff, d_model)

        self.norm_attn = nn.LayerNorm(d_model)
        self.norm_ff = nn.LayerNorm(d_model)
        self.residual_dropout_attn = FeatureDropout(residual_dropout)
        self.residual_dropout_ff = FeatureDropout(residual_dropout)

        self.activation = activation

    def forward(self, x, mask=None):
        residual = self.self_attn(x, mask=mask)
        residual = torch.cat(residual, dim=-1)
        residual = self.residual_dropout_attn(residual)
        x = self.norm_attn(x + residual)
        residual = self.linear2(self.ff_dropout(self.activation(self.linear1(x))))
        residual = torch.cat(residual, dim=-1)
        residual = self.residual_dropout_ff(residual)
        x = self.norm_ff(x + residual)
        return x


class PartitionedTransformerEncoder(nn.Module):
    def __init__(self,
                 n_layers,
                 d_model,
                 n_head,
                 d_qkv,
                 d_ff,
                 ff_dropout,
                 residual_dropout,
                 attention_dropout,
                 activation=PartitionedReLU,
    ):
        super().__init__()
        self.layer_mix_linear = nn.Linear(n_layers, 1, False)
        self.layers = nn.ModuleList([PartitionedTransformerEncoderLayer(d_model=d_model,
                                                                        n_head=n_head,
                                                                        d_qkv=d_qkv,
                                                                        d_ff=d_ff,
                                                                        ff_dropout=ff_dropout,
                                                                        residual_dropout=residual_dropout,
                                                                        attention_dropout=attention_dropout,
                                                                        activation=activation())
                                     for i in range(n_layers)])

    def forward(self, x, mask=None):
        intermediates = []
        for layer in self.layers:
            x = layer(x, mask=mask)
            intermediates.append(x)
        intermediate = torch.stack(intermediates, axis=3)
        # the bias is to fight against regularization
        x = self.layer_mix_linear(intermediate).squeeze(dim=3) + torch.sum(torch.stack(intermediates, axis=3), dim=3) * LAYER_MIX_BIAS
        return x


class ConcatPositionalEncoding(nn.Module):
    """
    Learns a position embedding
    """
    def __init__(self, d_model=256, max_len=512):
        super().__init__()
        self.timing_table = nn.Parameter(torch.FloatTensor(max_len, d_model))
        nn.init.normal_(self.timing_table)

    def forward(self, x):
        timing = self.timing_table[:x.shape[1], :]
        timing = timing.expand(x.shape[0], -1, -1)
        out = torch.cat([x, timing], dim=-1)
        return out

#
class PartitionedTransformerModule(nn.Module):
    def __init__(self,
                 n_layers,
                 d_model,
                 n_head,
                 d_qkv,
                 d_ff,
                 ff_dropout,
                 residual_dropout,
                 attention_dropout,
                 word_input_size,
                 bias,
                 morpho_emb_dropout,
                 timing,
                 encoder_max_len,
                 activation=PartitionedReLU()
    ):
        super().__init__()
        self.project_pretrained = nn.Linear(
            word_input_size, d_model // 2, bias=bias
        )

        self.pattention_morpho_emb_dropout = FeatureDropout(morpho_emb_dropout)
        if timing == 'sin':
            self.add_timing = ConcatSinusoidalEncoding(d_model=d_model // 2, max_len=encoder_max_len)
        elif timing == 'learned':
            self.add_timing = ConcatPositionalEncoding(d_model=d_model // 2, max_len=encoder_max_len)
        else:
            raise ValueError("Unhandled timing type: %s" % timing)
        self.transformer_input_norm = nn.LayerNorm(d_model)
        self.pattn_encoder = PartitionedTransformerEncoder(
            n_layers,
            d_model=d_model,
            n_head=n_head,
            d_qkv=d_qkv,
            d_ff=d_ff,
            ff_dropout=ff_dropout,
            residual_dropout=residual_dropout,
            attention_dropout=attention_dropout,
        )


    #
    def forward(self, attention_mask, bert_embeddings):
        # Prepares attention mask for feeding into the self-attention
        device = bert_embeddings[0].device
        if attention_mask:
            valid_token_mask = attention_mask
        else:
            valids = []
            for sent in bert_embeddings:
                valids.append(torch.ones(len(sent), device=device))

            padded_data = torch.nn.utils.rnn.pad_sequence(
                valids,
                batch_first=True,
                padding_value=-100
            )

            valid_token_mask = padded_data != -100

        valid_token_mask = valid_token_mask.to(device=device)
        padded_embeddings = torch.nn.utils.rnn.pad_sequence(
            bert_embeddings,
            batch_first=True,
            padding_value=0
        )

        # Project the pretrained embedding onto the desired dimension
        extra_content_annotations = self.project_pretrained(padded_embeddings)

        # Add positional information through the table
        encoder_in = self.add_timing(self.pattention_morpho_emb_dropout(extra_content_annotations))
        encoder_in = self.transformer_input_norm(encoder_in)
        # Put the partitioned input through the partitioned attention
        annotations = self.pattn_encoder(encoder_in, valid_token_mask)

        return annotations

