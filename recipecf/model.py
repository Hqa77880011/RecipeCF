import math

import torch
from torch import nn
from torch.nn import functional as F


def network(inputs, outputs, width):
    return nn.Sequential(nn.Linear(inputs, width), nn.SiLU(),
                         nn.Linear(width, width), nn.SiLU(),
                         nn.Linear(width, outputs))


class ActionFlow(nn.Module):
    def __init__(self, actions, context, width, layers=6):
        super().__init__()
        self.transforms = nn.ModuleList([
            network(actions + context, 2 * actions, width)
            for _ in range(layers)
        ])
        masks = [(torch.arange(actions) + i) % 2 for i in range(layers)]
        self.register_buffer("masks", torch.stack(masks).float())

    def log_prob(self, action, context):
        a = action.clamp(1e-6, 1 - 1e-6)
        z = torch.logit(a)
        determinant = -(a.log() + torch.log1p(-a)).sum(-1)
        for transform, mask in zip(self.transforms, self.masks):
            scale, shift = transform(torch.cat([z * mask, context], -1)).chunk(2, -1)
            scale = 1.5 * torch.tanh(scale) * (1 - mask)
            shift = shift * (1 - mask)
            z = z * mask + (1 - mask) * (z - shift) * torch.exp(-scale)
            determinant = determinant - scale.sum(-1)
        return -0.5 * (z.square() + math.log(2 * math.pi)).sum(-1) + determinant


class CubicBasis(nn.Module):
    def __init__(self):
        super().__init__()
        self.register_buffer("knots", torch.tensor(
            [0., 0., 0., 0., .25, .5, .75, 1., 1., 1., 1.]))

    def forward(self, action):
        a = action.clamp(0, 1 - 1e-6).unsqueeze(-1)
        k = self.knots
        b = ((a >= k[:-1]) & (a < k[1:])).to(a.dtype)
        for degree in range(1, 4):
            count = len(k) - degree - 1
            d1 = k[degree:degree + count] - k[:count]
            d2 = k[degree + 1:degree + count + 1] - k[1:count + 1]
            left = (a - k[:count]) / d1.clamp_min(1e-8)
            right = (k[degree + 1:degree + count + 1] - a) / d2.clamp_min(1e-8)
            b = left * b[..., :count] * (d1 > 0) + right * b[..., 1:count + 1] * (d2 > 0)
        return torch.cat([action, b.flatten(1)], -1)


class RecipeCF(nn.Module):
    def __init__(self, conditions, actions, width=128, rank=8, variant="full"):
        super().__init__()
        self.variant = variant
        self.encoder = network(conditions, width, width)
        self.basis = CubicBasis()
        self.conditional = ActionFlow(actions, conditions, width)
        self.marginal = ActionFlow(actions, 0, width)
        if variant in {"mlp", "concat"}:
            self.concat = network(width + actions * 8, 2, width)
        else:
            self.b = nn.Linear(width, 1)
            self.g = network(actions * 8, 1, width)
            self.u = nn.Linear(width, rank, bias=False)
            self.v = network(actions * 8, rank, width)
            self.scale = network(width + actions * 8, 1, width)

    def response(self, x, a):
        h, p = self.encoder(x), self.basis(a)
        joint = torch.cat([h, p], -1)
        if self.variant in {"mlp", "concat"}:
            mean, raw_scale = self.concat(joint).unbind(-1)
        else:
            mean = self.b(h).squeeze(-1) + self.g(p).squeeze(-1)
            mean = mean + (self.u(h) * self.v(p)).sum(-1)
            raw_scale = self.scale(joint).squeeze(-1)
        return mean, F.softplus(raw_scale) + 1e-3


def balance(model, x, a, weights, anchors, bandwidth, tempered=True):
    h = model.encoder(x).detach()
    kernel = torch.exp(-torch.cdist(a, anchors).square() / (2 * bandwidth ** 2))
    wk = weights[:, None] * kernel
    rho = wk / wk.sum(0).clamp_min(1e-8)
    hm, am = rho.T @ h, rho.T @ a
    covariance = torch.einsum("nl,nlh,nla->lha", rho,
                              h[:, None] - hm, a[:, None] - am)
    ess = wk.sum(0).square() / wk.square().sum(0).clamp_min(1e-8)
    gate = torch.sigmoid((ess - 128) / 24) if tempered else torch.ones_like(ess)
    return (gate * covariance.square().sum((1, 2))).sum() / gate.sum().clamp_min(1e-8)


def weighted_quantile(residuals, weights, query_weight, coverage=.9):
    order = torch.argsort(residuals)
    values, w = residuals[order], weights[order]
    target = coverage * (w.sum() + query_weight)
    indices = torch.searchsorted(w.cumsum(0), target)
    return torch.where(indices < len(w), values[indices.clamp_max(len(w) - 1)],
                       torch.full_like(query_weight, torch.inf))


def action_jacobian(model, x, a, create_graph=False):
    a = a.detach().requires_grad_(True)
    mean, scale = model.response(x, a)
    jac = torch.autograd.grad(mean.sum(), a, create_graph=create_graph)[0]
    if create_graph:
        return mean, scale, jac
    return mean.detach(), scale.detach(), jac.detach()
