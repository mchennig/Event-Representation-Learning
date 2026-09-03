# Using Alternative Self-Supervised Objectives

Both `ContrastiveProcessSetTransformer` and `ContrastiveProcessTabTransformer` are
self-supervised encoders: each `train_step`/`test_step` creates two augmented views
of the input, embeds both, and passes the two projection tensors
(`projections_1`, `projections_2`) into `self.default_loss(...)`.

All three losses share the exact same call signature —
`loss(projections_1, projections_2) -> scalar` — so any of them can be dropped
into either encoder via the `loss` constructor argument, no other code changes
are required.

## 1. The three losses at a glance

| Loss | Needs Negative Pairs? | Batch-size Sensitivity | Key Hyperparameters |
|---|---|---|---|
| `BarlowTwinsLoss` | No | Low — works fine with small batches | `lambda_param` (off-diagonal weight, default `5e-3`) |
| `VICRegLoss` | No | Low — works fine with small batches | `sim_coeff`, `var_coeff`, `cov_coeff` (defaults `25/25/1`) |
| `NTXentLoss` | Yes — uses in-batch negatives | High — larger batches give more/better negatives | `temperature` (default `0.1`) |

Because Barlow Twins and VICReg only need positive pairs (they regularize the
correlation/covariance structure instead of contrasting against negatives),
they're the safer default when batch size is constrained by memory or dataset
size — which is why both encoders default to `BarlowTwinsLoss(lambda_param=1e-4)`.
`NTXentLoss` tends to need larger batches (as in SimCLR) to have enough negatives
per sample to be effective.

## 2. Basic usage

### `ContrastiveProcessSetTransformer`

```python
model = ContrastiveProcessSetTransformer(
    input_encoder=input_encoder,
    embedding_model=set_transformer,
    augmentation_1=augmentation,
    loss=BarlowTwinsLoss(lambda_param=1e-4),  # default
)
```

### `ContrastiveProcessTabTransformer`

```python
model = ContrastiveProcessTabTransformer(
    input_encoder=input_encoder,
    embedding_model=tab_transformer,
    augmentation_1=tabular_augmentation,
    loss=BarlowTwinsLoss(lambda_param=1e-4),  # default
)
```

## 3. Swapping in a different loss

Just pass a different loss instance — the rest of `train_step`/`test_step` is
loss-agnostic:

```python
# VICReg
model = ContrastiveProcessSetTransformer(
    input_encoder=input_encoder,
    embedding_model=set_transformer,
    augmentation_1=augmentation,
    loss=VICRegLoss(sim_coeff=25.0, var_coeff=25.0, cov_coeff=1.0),
)

# NT-Xent (SimCLR)
model = ContrastiveProcessTabTransformer(
    input_encoder=input_encoder,
    embedding_model=tab_transformer,
    augmentation_1=tabular_augmentation,
    loss=NTXentLoss(temperature=0.07),
)
```

Tuning notes:
- **`BarlowTwinsLoss.lambda_param`**: raise it (toward `1e-2`) to push harder on
  feature decorrelation; lower it (toward `1e-5`) if training is unstable early on.
- **`VICRegLoss`**: `sim_coeff` controls how tightly the two views are pulled
  together, `var_coeff` prevents collapse (keeps per-dimension std ≥ 1), and
  `cov_coeff` decorrelates features. The large default `sim_coeff`/`var_coeff`
  values (25) relative to `cov_coeff` (1) reflect the original paper's balance.
- **`NTXentLoss.temperature`**: lower (`0.05`–`0.1`) sharpens the distribution and
  focuses on hard negatives; higher (`0.5`–`1.0`) smooths it. Also increase
  effective batch size if using this loss, since positives are contrasted
  against every other in-batch sample.
