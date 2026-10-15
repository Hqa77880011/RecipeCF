import argparse
import csv
import importlib.util
import json
import time
from pathlib import Path

import numpy as np
import torch
from torch.nn import functional as F

from .model import RecipeCF, action_jacobian, balance, weighted_quantile


VARIANTS = ["full", "mlp", "no_balance", "fixed_balance", "no_jacobian",
            "no_diff", "no_gate", "concat", "full_rank"]


def tensors(path):
    with np.load(path, allow_pickle=False) as data:
        return {k: torch.from_numpy(data[k]).long() if k == "group" else
                torch.from_numpy(data[k]).float() for k in data.files}


def save_json(path, value):
    Path(path).write_text(json.dumps(value, indent=2, allow_nan=False), encoding="utf-8")


def surface(x, a):
    target = .2 + .6 * torch.sigmoid(x[..., :a.shape[-1]])
    response = .2 * x[..., 0] + (a - target).square().sum(-1)
    response = response + .2 * a.prod(-1)
    response = response + .1 * (a[..., 1:] * a[..., :-1]).sum(-1)
    return response


def generate(args):
    torch.manual_seed(args.seed)
    x = torch.randn(args.samples, max(args.conditions, args.actions))
    v = torch.randn(args.actions, x.shape[1]) / x.shape[1] ** .5
    interactions = .2 * torch.randn(args.actions, args.actions)
    a = torch.zeros(args.samples, args.actions)
    for k in range(args.actions):
        eta = x @ v[k] + (a[:, :k] * interactions[:k, k]).sum(-1)
        first = 1 + args.confounding * F.softplus(eta)
        second = 1 + args.confounding * F.softplus(-eta)
        a[:, k] = torch.distributions.Beta(first, second).sample().clamp(1e-5, 1 - 1e-5)
    a.requires_grad_(True)
    truth = surface(x, a)
    jacobian = torch.autograd.grad(truth.sum(), a)[0]
    y = truth.detach() + args.noise * torch.randn(args.samples)
    grid = torch.rand(args.samples, args.grid_size, args.actions)
    grid_truth = surface(x[:, None, :], grid)
    sign = torch.where((a.detach() > .85), torch.ones_like(a), torch.zeros_like(a))
    flip = torch.rand_like(sign) < args.sign_corruption
    sign = torch.where(flip, -sign, sign)
    target = Path(args.output)
    target.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(target, x=x.numpy(), a=a.detach().numpy(), y=y.numpy(),
                        group=np.arange(args.samples), sign=sign.detach().numpy(),
                        jacobian=jacobian.numpy(), grid_a=grid.numpy(), grid_mu=grid_truth.numpy())


def prepare(args):
    source = tensors(args.input)
    x, a, y = source["x"], source["a"], source["y"].flatten()
    if x.ndim != 2 or a.ndim != 2 or len(x) != len(a) or len(x) != len(y):
        raise ValueError("x[N,Dx], a[N,Da], y[N] must have matching sample counts")
    if not torch.isfinite(a).all() or not ((a > 0) & (a < 1)).all():
        raise ValueError("Actions must be finite and strictly inside (0,1)")
    if not torch.isfinite(y).all():
        raise ValueError("Outcomes must be finite")
    groups = source.get("group", torch.arange(len(x)))
    unique = torch.unique(groups, sorted=True)
    if len(unique) < 10:
        raise ValueError("At least ten independent groups are required")
    if args.shuffle:
        generator = torch.Generator().manual_seed(args.seed)
        unique = unique[torch.randperm(len(unique), generator=generator)]
    n1, n2 = int(.7 * len(unique)), int(.85 * len(unique))
    splits = [torch.isin(groups, g) for g in (unique[:n1], unique[n1:n2], unique[n2:])]
    train = x[splits[0]]
    median = torch.nanmedian(torch.where(torch.isfinite(train), train, torch.nan), 0).values
    if not torch.isfinite(median).all():
        raise ValueError("A training condition column has no finite observations")
    x = torch.where(torch.isfinite(x), x, median)
    mean, std = x[splits[0]].mean(0), x[splits[0]].std(0).clamp_min(1e-6)
    out = Path(args.output)
    out.mkdir(parents=True, exist_ok=True)
    for name, mask in zip(["train", "calibration", "test"], splits):
        values = {k: v[mask].numpy() for k, v in source.items() if len(v) == len(x)}
        values.update(x=((x[mask] - mean) / std).numpy(), x_raw=x[mask].numpy(),
                      a=a[mask].numpy(), y=y[mask].numpy())
        np.savez_compressed(out / f"{name}.npz", **values)
    save_json(out / "preprocessing.json", {"median": median.tolist(), "mean": mean.tolist(),
                                           "std": std.tolist(), "outcome_scale": 1.0})


def density_weights(model, x, a, clip):
    conditional = model.conditional.log_prob(a, x)
    marginal = model.marginal.log_prob(a, a[:, :0])
    ratio = (marginal - conditional).clamp(-30, 30).exp()
    clipped = ratio.clamp(clip[0], clip[1])
    return conditional, clipped / clipped.mean(), clipped


def train(args):
    torch.manual_seed(args.seed)
    data = tensors(Path(args.data) / "train.npz")
    x, a, y = data["x"], data["a"], data["y"]
    width, rank = args.width, (args.width if args.variant == "full_rank" else args.rank)
    model = RecipeCF(x.shape[1], a.shape[1], width, rank, args.variant)
    opt = torch.optim.Adam(model.parameters(), lr=args.lr)
    start = time.perf_counter()
    for _ in range(args.density_epochs):
        for ix in torch.randperm(len(x)).split(args.batch_size):
            loss = -model.conditional.log_prob(a[ix], x[ix]).mean()
            loss = loss - model.marginal.log_prob(a[ix], a[ix, :0]).mean()
            opt.zero_grad()
            loss.backward()
            opt.step()
    with torch.no_grad():
        ratio = (model.marginal.log_prob(a, a[:, :0]) - model.conditional.log_prob(a, x)).clamp(-30, 30).exp()
        clip = torch.quantile(ratio, torch.tensor([.01, .99])).tolist()
        initial_support = torch.quantile(model.conditional.log_prob(a, x), .01).item()
    for parameter in model.marginal.parameters():
        parameter.requires_grad_(False)
    anchors = a[torch.randperm(len(a))[:args.anchors]]
    history = []
    for epoch in range(args.epochs):
        total = 0.
        for ix in torch.randperm(len(x)).split(args.batch_size):
            xb, ab, yb = x[ix], a[ix].detach().requires_grad_(True), y[ix]
            q, weights, _ = density_weights(model, xb, ab, clip)
            if args.variant == "mlp":
                weights = torch.ones_like(weights)
            mu, sigma = model.response(xb, ab)
            outcome = (weights.detach() * ((yb - mu).square() / (2 * sigma.square()) + sigma.log())).mean()
            local = balance(model, xb, ab, weights, anchors, args.bandwidth,
                            args.variant != "fixed_balance")
            assignment = -q.mean()
            if args.variant not in {"no_balance", "mlp"}:
                assignment = assignment + args.balance * local
            derivative = mu.sum() * 0
            if args.variant not in {"mlp", "no_jacobian"}:
                jac = torch.autograd.grad(mu.sum(), ab, create_graph=True)[0]
                supported = q.detach() >= initial_support
                if "sign" in data:
                    sign = data["sign"][ix]
                    active = (sign != 0) & supported[:, None]
                    if active.any():
                        derivative = F.relu(-sign * jac).square()[active].mean()
                delta = torch.randn_like(ab)
                delta = delta / delta.norm(dim=-1, keepdim=True).clamp_min(1e-8)
                delta = delta * torch.rand(len(ab), 1).pow(1 / a.shape[1]) * .02
                ap = (ab + delta).clamp(1e-5, 1 - 1e-5)
                delta = ap - ab
                mp, _ = model.response(xb, ap)
                jp = torch.autograd.grad(mp.sum(), ap, create_graph=True)[0]
                scope = supported & (model.conditional.log_prob(ap, xb).detach() >= initial_support)
                norm2 = delta.square().sum(-1)
                if scope.any():
                    smooth = (jp - jac).square().sum(-1) / (norm2 + 1e-8)
                    remainder = (mp - mu - (jac * delta).sum(-1)).square() / (norm2.square() + 1e-8)
                    derivative = derivative + .5 * smooth[scope].mean()
                    if args.variant != "no_diff":
                        derivative = derivative + .25 * remainder[scope].mean()
            regularizer = mu.sum() * 0
            if hasattr(model, "u"):
                uv = model.u.weight.norm(dim=1) * model.v[-1].weight.norm(dim=1)
                regularizer = 1e-4 * uv.sum()
            loss = outcome + .2 * assignment + .1 * derivative + regularizer
            opt.zero_grad()
            loss.backward()
            opt.step()
            total += loss.detach().item() * len(ix)
        history.append({"epoch": epoch + 1, "loss": total / len(x)})
    for parameter in model.parameters():
        parameter.requires_grad_(False)
    model.eval()
    cal = tensors(Path(args.data) / "calibration.npz")
    with torch.no_grad():
        cq, _, cw = density_weights(model, cal["x"], cal["a"], clip)
        cm, cs = model.response(cal["x"], cal["a"])
        residual = (cal["y"] - cm).abs() / cs
        thresholds = torch.unique(torch.quantile(cq, torch.linspace(0, .2, 21)), sorted=True)
        threshold = None
        for candidate in thresholds:
            keep = cq >= candidate
            covered = residual <= weighted_quantile(residual, cw, cw)
            coverage = (cw[keep] * covered[keep]).sum() / cw[keep].sum()
            if keep.float().mean() >= .8 and coverage >= .9:
                threshold = candidate.item()
                break
        if threshold is None:
            raise ValueError("No calibration support threshold satisfies coverage >= .9 and retention >= .8")
    out = Path(args.output)
    out.mkdir(parents=True, exist_ok=True)
    torch.save({"model": model.state_dict(), "conditions": x.shape[1], "actions": a.shape[1],
                "width": width, "rank": rank, "variant": args.variant, "clip": clip,
                "threshold": threshold, "residual": residual, "cal_weights": cw,
                "train_seconds": time.perf_counter() - start}, out / "model.pt")
    save_json(out / "history.json", history)


def load_model(path):
    state = torch.load(path, map_location="cpu", weights_only=True)
    model = RecipeCF(state["conditions"], state["actions"], state["width"], state["rank"], state["variant"])
    model.load_state_dict(state["model"])
    model.eval()
    for parameter in model.parameters():
        parameter.requires_grad_(False)
    return model, state


def predict(model, state, x, a, gate=True):
    mu, scale, jac = action_jacobian(model, x, a)
    with torch.no_grad():
        q, _, weights = density_weights(model, x, a, state["clip"])
        quantile = weighted_quantile(state["residual"], state["cal_weights"], weights)
        keep = q >= state["threshold"] if gate else torch.ones(len(a), dtype=torch.bool)
        return mu, scale * quantile, jac, keep, q


def search(model, state, x, args):
    generator = torch.Generator().manual_seed(args.seed)
    candidate = torch.rand(len(x), args.restarts, state["actions"], generator=generator).clamp(.001, .999)
    flat_x = x[:, None].expand(-1, args.restarts, -1).reshape(-1, x.shape[1])
    current = candidate.flatten(0, 1)
    gate = state["variant"] not in {"mlp", "no_gate"}
    for _ in range(args.search_steps):
        _, _, jac = action_jacobian(model, flat_x, current)
        proposed = (current - args.search_lr * jac).clamp(1e-5, 1 - 1e-5)
        with torch.no_grad():
            supported = model.conditional.log_prob(proposed, flat_x) >= state["threshold"]
            current = torch.where(supported[:, None], proposed, current) if gate else proposed
    with torch.no_grad():
        score = model.response(flat_x, current)[0].reshape(len(x), -1)
        keep = (model.conditional.log_prob(current, flat_x) >= state["threshold"]).reshape(len(x), -1)
        if gate:
            score = score.masked_fill(~keep, torch.inf)
        best = score.argmin(-1)
        chosen = current.reshape_as(candidate)[torch.arange(len(x)), best]
        valid = torch.isfinite(score.min(-1).values)
    return chosen, valid


def evaluate(args):
    model, state = load_model(args.model)
    data = tensors(args.input)
    x, a, y = data["x"], data["a"], data["y"]
    gate = state["variant"] not in {"mlp", "no_gate"}
    start = time.perf_counter()
    mu, radius, jac, keep, q = predict(model, state, x, a, gate)
    batch_seconds = time.perf_counter() - start
    metrics = {"factual_rmse": (mu - y).square().mean().sqrt().item(),
               "retention": keep.float().mean().item(), "abstention_rate": (~keep).float().mean().item(),
               "parameters": sum(p.numel() for p in model.parameters()),
               "train_seconds": state["train_seconds"], "batch_seconds": batch_seconds}
    if keep.any():
        metrics["interval_coverage"] = ((mu - y).abs()[keep] <= radius[keep]).float().mean().item()
        finite_width = torch.isfinite(radius[keep]).all().item()
        metrics["interval_width"] = (2 * radius[keep]).mean().item() if finite_width else None
        metrics["infinite_interval_fraction"] = (~torch.isfinite(radius[keep])).float().mean().item()
    if "jacobian" in data:
        active = (data["jacobian"].abs() > 1e-8) & keep[:, None]
        if active.any():
            metrics["gradient_sign_accuracy"] = (jac.sign() == data["jacobian"].sign())[active].float().mean().item()
            metrics["jacobian_magnitude_error"] = (jac.norm(dim=-1) - data["jacobian"].norm(dim=-1)).abs()[keep].mean().item()
    with torch.no_grad():
        _, weights, _ = density_weights(model, x, a, state["clip"])
        metrics["local_balance_error"] = balance(model, x, a, weights,
            a[:min(32, len(a))], .2, tempered=False).item()
    losses = None
    if {"grid_a", "grid_mu"} <= data.keys():
        ga, truth = data["grid_a"], data["grid_mu"]
        gx = x[:, None].expand(-1, ga.shape[1], -1).reshape(-1, x.shape[1])
        with torch.no_grad():
            pred = model.response(gx, ga.flatten(0, 1))[0].reshape_as(truth)
            supported = (model.conditional.log_prob(ga.flatten(0, 1), gx) >= state["threshold"]).reshape_as(truth)
            count = supported.sum(-1)
            grid_valid = count > 1
            mean = (truth * supported).sum(-1) / count.clamp_min(1)
            variance = ((truth - mean[:, None]).square() * supported).sum(-1) / count.clamp_min(1)
            losses = ((pred - truth).square() * supported).sum(-1) / count.clamp_min(1) / (variance + 1e-8)
            metrics["grid_entities_without_support"] = (~grid_valid).sum().item()
            if grid_valid.any():
                metrics["normalized_mise"] = losses[grid_valid].mean().item()
                selected = pred.masked_fill(~supported, torch.inf).argmin(-1)
                oracle = truth.masked_fill(~supported, torch.inf).min(-1).values
                regret = truth[torch.arange(len(x)), selected] - oracle
                metrics["supported_grid_regret"] = regret[grid_valid].mean().item()
    chosen, valid_search = search(model, state, x, args)
    metrics["search_admission_rate"] = valid_search.float().mean().item()
    oracle = None
    if args.oracle == "process":
        oracle = surface
    elif args.oracle:
        spec = importlib.util.spec_from_file_location("response_oracle", args.oracle)
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        oracle = module.response
    if oracle is not None:
        if "x_raw" not in data:
            raise ValueError("Oracle evaluation requires x_raw from the prepare command")
        target, target_valid = search_oracle(model, state, data["x_raw"], x, oracle, args)
        valid = target_valid & valid_search
        if valid.any():
            with torch.no_grad():
                regret = oracle(data["x_raw"], chosen) - oracle(data["x_raw"], target)
                metrics["supported_search_regret"] = regret[valid].mean().item()
        metrics["oracle_search_admission_rate"] = target_valid.float().mean().item()
    repeats = 20
    for _ in range(3):
        predict(model, state, x[:1], a[:1], gate)
    start = time.perf_counter()
    for _ in range(repeats):
        predict(model, state, x[:1], a[:1], gate)
    metrics["batch_one_latency_ms"] = 1000 * (time.perf_counter() - start) / repeats
    out = Path(args.output)
    out.mkdir(parents=True, exist_ok=True)
    save_json(out / "metrics.json", metrics)
    arrays = {"prediction": mu.numpy(), "radius": radius.numpy(), "jacobian": jac.numpy(),
              "admitted": keep.numpy(), "support": q.numpy(), "action": chosen.numpy(),
              "search_admitted": valid_search.numpy()}
    if losses is not None:
        arrays.update(mise=losses.numpy(), mise_valid=grid_valid.numpy(), grid_prediction=pred.numpy(),
                      grid_truth=truth.numpy(), grid_supported=supported.numpy(), grid_a=ga.numpy())
    np.savez_compressed(out / "predictions.npz", **arrays)
    rows = []
    for retention in np.linspace(.5, 1, 11):
        threshold = torch.quantile(q, 1 - float(retention))
        select = q >= threshold
        rows.append({"retention": select.float().mean().item(),
                     "rmse": (mu[select] - y[select]).square().mean().sqrt().item(),
                     "coverage": ((mu - y).abs()[select] <= radius[select]).float().mean().item()})
    save_json(out / "retention.json", rows)


def search_oracle(model, state, raw_x, x, oracle, args):
    generator = torch.Generator().manual_seed(args.seed)
    actions = torch.rand(len(x), args.restarts, state["actions"], generator=generator)
    actions = actions.clamp(1e-5, 1 - 1e-5).flatten(0, 1)
    normalized = x[:, None].expand(-1, args.restarts, -1).flatten(0, 1)
    conditions = raw_x[:, None].expand(-1, args.restarts, -1).flatten(0, 1)
    for _ in range(args.search_steps):
        actions = actions.detach().requires_grad_(True)
        response = oracle(conditions, actions)
        derivative = torch.autograd.grad(response.sum(), actions)[0]
        proposed = (actions - args.search_lr * derivative).clamp(1e-5, 1 - 1e-5)
        with torch.no_grad():
            keep = model.conditional.log_prob(proposed, normalized) >= state["threshold"]
            actions = torch.where(keep[:, None], proposed, actions.detach())
    with torch.no_grad():
        score = oracle(conditions, actions).reshape(len(x), -1)
        keep = model.conditional.log_prob(actions, normalized).reshape(len(x), -1) >= state["threshold"]
        score = score.masked_fill(~keep, torch.inf)
        best = score.argmin(-1)
        chosen = actions.reshape(len(x), args.restarts, -1)[torch.arange(len(x)), best]
        return chosen, torch.isfinite(score.min(-1).values)


def plot(args):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    paths = list(Path(args.input).rglob("metrics.json"))
    if not paths:
        raise ValueError("No evaluated metrics.json files found")
    records = [json.loads(p.read_text()) for p in paths]
    labels = [str(p.parent.relative_to(args.input)) for p in paths]
    keys = sorted(set.intersection(*(set(r) for r in records)))
    keys = [k for k in keys if all(isinstance(r[k], (float, int)) for r in records)]
    out = Path(args.output)
    out.mkdir(parents=True, exist_ok=True)
    with (out / "comparison.csv").open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=["run"] + keys)
        writer.writeheader()
        writer.writerows({"run": label, **{k: r[k] for k in keys}} for label, r in zip(labels, records))
    for key in keys:
        fig, ax = plt.subplots(figsize=(max(6, len(labels) * .8), 4))
        ax.bar(labels, [r[key] for r in records], color="#4b809b")
        ax.set_ylabel(key)
        ax.tick_params(axis="x", labelrotation=35)
        fig.tight_layout()
        fig.savefig(out / f"{key}.png", dpi=180)
        plt.close(fig)


def compare(args):
    a, b = tensors(args.first), tensors(args.second)
    if a["grid_a"].shape != b["grid_a"].shape or not torch.equal(a["grid_a"], b["grid_a"]):
        raise ValueError("Paired comparison requires identical entity ordering and action grids")
    if not torch.equal(a["grid_truth"], b["grid_truth"]):
        raise ValueError("Paired comparison requires identical response truth")
    support = a["grid_supported"].bool() & b["grid_supported"].bool()
    count = support.sum(-1)
    mask = count > 1
    truth = a["grid_truth"]
    mean = (truth * support).sum(-1) / count.clamp_min(1)
    variance = ((truth - mean[:, None]).square() * support).sum(-1) / count.clamp_min(1)
    errors = []
    for run in [a, b]:
        error = ((run["grid_prediction"] - truth).square() * support).sum(-1)
        errors.append(error / count.clamp_min(1) / (variance + 1e-8))
    difference = (errors[0] - errors[1])[mask].numpy()
    if len(difference) < 2:
        raise ValueError("At least two jointly supported test entities required")
    rng = np.random.default_rng(args.seed)
    draws = np.array([rng.choice(difference, len(difference), replace=True).mean()
                      for _ in range(args.resamples)])
    centered = difference - difference.mean()
    null = np.array([rng.choice(centered, len(centered), replace=True).mean()
                     for _ in range(args.resamples)])
    p = (1 + (np.abs(null) >= abs(difference.mean())).sum()) / (args.resamples + 1)
    save_json(args.output, {"difference": float(difference.mean()),
                           "ci95": np.quantile(draws, [.025, .975]).tolist(), "p": float(p)})


def main():
    parser = argparse.ArgumentParser(prog="recipecf")
    commands = parser.add_subparsers(dest="command", required=True)
    p = commands.add_parser("generate")
    p.add_argument("--output", required=True)
    for key, default in [("samples", 24000), ("conditions", 30), ("actions", 4),
                         ("grid-size", 128), ("seed", 0)]:
        p.add_argument(f"--{key}", type=int, default=default)
    for key, default in [("confounding", .5), ("noise", .1), ("sign-corruption", 0.)]:
        p.add_argument(f"--{key}", type=float, default=default)
    p = commands.add_parser("prepare")
    p.add_argument("--input", required=True)
    p.add_argument("--output", required=True)
    p.add_argument("--shuffle", action="store_true")
    p.add_argument("--seed", type=int, default=0)
    p = commands.add_parser("train")
    p.add_argument("--data", required=True)
    p.add_argument("--output", required=True)
    p.add_argument("--variant", choices=VARIANTS, default="full")
    for key, default in [("epochs", 200), ("density-epochs", 30), ("batch-size", 512),
                         ("width", 128), ("rank", 8), ("anchors", 32), ("seed", 0)]:
        p.add_argument(f"--{key}", type=int, default=default)
    for key, default in [("lr", .001), ("balance", 1.), ("bandwidth", .2)]:
        p.add_argument(f"--{key}", type=float, default=default)
    p = commands.add_parser("evaluate")
    p.add_argument("--model", required=True)
    p.add_argument("--input", required=True)
    p.add_argument("--output", required=True)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--restarts", type=int, default=8)
    p.add_argument("--search-steps", type=int, default=100)
    p.add_argument("--search-lr", type=float, default=.01)
    p.add_argument("--oracle", help="process or a Python file exporting response(raw_x, actions)")
    p = commands.add_parser("plot")
    p.add_argument("--input", required=True)
    p.add_argument("--output", required=True)
    p = commands.add_parser("compare")
    p.add_argument("--first", required=True)
    p.add_argument("--second", required=True)
    p.add_argument("--output", required=True)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--resamples", type=int, default=10000)
    args = parser.parse_args()
    globals()[args.command](args)


if __name__ == "__main__":
    main()
