import torch
import torch.nn as nn


class MergeAdapter(nn.Module):
    def __init__(
        self, ds_factor, hidden_dim, ln_after=False, ln_before=False, dropout=0.1, num_models=1
    ):
        super().__init__()
        assert not hidden_dim % ds_factor
        self.down = Merge_Linear(hidden_dim, hidden_dim // ds_factor, num_models=num_models)
        self.act = nn.ReLU()
        self.up = Merge_Linear(hidden_dim // ds_factor, hidden_dim, num_models=num_models)
        self.apply(self.init_weights)
        self.ln_after = ln_after
        self.ln_before = ln_before
        self.dropout = dropout
        if ln_after or ln_before:
            self.ln = nn.LayerNorm(hidden_dim)
        if dropout:
            self.dropout = nn.Dropout(dropout)

    def init_weights(self, m: nn.Module, std=1e-3):
        if isinstance(m, nn.Linear):
            torch.nn.init.normal_(m.weight, std=std)
            torch.nn.init.normal_(m.bias, std=std)
            m.weight.data = torch.clamp(m.weight.data, min=-2 * std, max=2 * std)
            m.bias.data = torch.clamp(m.bias.data, min=-2 * std, max=2 * std)
        elif isinstance(m, nn.LayerNorm):
            m.bias.data.zero_()
            m.weight.data.fill_(1.0)

    def forward(self, hidden_states, prob):
        if self.ln_before:
            residual = self.ln(hidden_states)
            residual = self.down(residual, prob)
        else:
            residual = self.down(hidden_states, prob)
        residual = self.act(residual)
        if self.dropout:
            residual = self.dropout(residual)
        residual = self.up(residual, prob)
        if self.ln_after:
            residual = self.ln(hidden_states)
        return hidden_states + residual


class Merge_Linear(nn.Module):
    """Linear layer for SMEAR.
    Applies a linear transformation to the incoming data: :math:`y = xA^T + b`.
    `A` and `b` are weighted average of `num_experts` experts, each of which has it own
        `A` and `b`.

    Args:
        in_features (int): size of each input sample.
        out_features (int): size of each output sample.
        num_experts (int): The number of experts.
        bias (bool): If set to ``False``, the layer will not learn an additive bias.
            Default: ``True``
    """

    def __init__(
        self,
        in_features: int,
        out_features: int,
        num_models: int,
        bias: bool = True,
    ):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features

        self.weight = nn.ParameterList(
            [
                nn.parameter.Parameter(torch.empty(out_features, in_features))
                for _ in range(num_models)
            ]
        )
        if bias:
            self.bias = nn.ParameterList(
                [nn.parameter.Parameter(torch.empty(out_features)) for _ in range(num_models)]
            )
        else:
            self.register_parameter("bias", None)

    def forward(self, x: torch.Tensor, prob: torch.Tensor):
        """For each input instance, form a new Linear and apply to it.

        Args:
            x (torch.Tensor): The input features. Shape: `(N, *, H_{in})`.
            prob (torch.Tensor): The normalized probability for each experts.
                Shape: `(N, num_models)`.

        Returns: TODO

        """
        k = prob.shape[1]  # num_models
        weight = torch.stack(list(self.weight[:k]))
        weight = torch.sum(weight * prob[:, :, None, None], dim=1)  # (N, H_out, H_in)
        if self.bias is not None:
            bias = torch.stack(list(self.bias[:k]))
            bias = torch.sum(bias * prob[:, :, None], dim=1)  # (N, H_out)
        else:
            bias = None

        assert x.ndim == 3, "not implemented for x.ndim != 3"
        x = torch.matmul(x, weight.transpose(1, 2))  # (N, *, H_out)
        if bias is not None:
            x = x + bias.view(bias.shape[0], *([1] * (x.ndim - 2)), bias.shape[1])
        return x
