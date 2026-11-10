"""Neural controls: CNN, LSTM and Transformer on the window (B3, B4), and the bounded CNN residual on the anchor (V1)."""
import numpy as np
import torch
import torch.nn as nn

from .anchor import anchor
from .model import folds, group

VARIANTS = ("cnn", "lstm", "transformer", "cnn_residual")
EPOCHS, BATCH_CYCLES, LR = 80, 256, 2e-3
CURVE_POINTS = 32


class Net(nn.Module):
    def __init__(self, variant, channels, scalars):
        super().__init__()
        if variant == "lstm":
            self.rnn = nn.LSTM(channels, 64, num_layers=2, batch_first=True)
            self.encode = lambda x: self.rnn(x.transpose(1, 2))[0][:, -1]
        elif variant == "transformer":
            self.inp = nn.Linear(channels, 64)
            self.pos = nn.Parameter(torch.zeros(1, CURVE_POINTS, 64))
            self.enc = nn.TransformerEncoder(nn.TransformerEncoderLayer(64, 4, 128, 0.1, batch_first=True), 2)
            self.encode = lambda x: self.enc(self.inp(x.transpose(1, 2)) + self.pos).mean(1)
        else:
            self.cnn = nn.Sequential(nn.Conv1d(channels, 32, 5, padding=2), nn.GELU(), nn.Conv1d(32, 64, 5, padding=2, stride=2), nn.GELU(),
                                     nn.Conv1d(64, 64, 3, padding=1, stride=2), nn.GELU(), nn.AdaptiveAvgPool1d(1))
            self.encode = lambda x: self.cnn(x).flatten(1)
        self.head = nn.Sequential(nn.Linear(64 + scalars, 128), nn.GELU(), nn.Linear(128, 64), nn.GELU(), nn.Linear(64, 1))

    def forward(self, x, z):
        return self.head(torch.cat([self.encode(x), z], 1)).squeeze(1)


def run_fold(D, variant, fold, seed, device="cuda", memory_fraction=None):
    """Train on the other folds, choose the epoch on one of them, predict the fold. Returns (test indices, predictions, val MAE)."""
    torch.manual_seed(seed)
    np.random.seed(seed)
    if memory_fraction and device == "cuda":
        torch.cuda.set_per_process_memory_fraction(memory_fraction)
    seq, sca, y, cid, cells = D["seq"], D["sca"], D["y"].astype(np.float32), D["cyc_id"], D["cells"]
    residual = variant == "cnn_residual"
    if not residual:                                         # window only: no anchor channels or scalars
        seq, sca = seq[:, [0, 3, 5]], sca[:, 3:]
    fl = folds(cells)
    groups = np.array([group(c) for c in cells])
    train, test = fl[fold]
    val_groups = list(set(groups[fl[(fold + 1) % 5][1]]))
    val = train[np.isin(groups[train], val_groups)]
    train = train[~np.isin(groups[train], val_groups)]
    mu, sd = sca[train].mean(0), sca[train].std(0)
    const = sd < 1e-3                                        # an input constant in training carries nothing and would explode elsewhere
    sd = np.where(const, 1.0, sd)
    z = (sca - mu) / sd
    z[:, const] = 0.0
    base = anchor(D["s"]).astype(np.float32) if residual else np.ones_like(y)
    t = lambda a: torch.tensor(a, device=device)
    X, Z, Y, B = t(seq), t(z.astype(np.float32)), t(y), t(base)
    net = Net("cnn" if residual else variant, X.shape[1], Z.shape[1]).to(device)
    opt = torch.optim.AdamW(net.parameters(), LR, weight_decay=1e-4)
    sched = torch.optim.lr_scheduler.OneCycleLR(opt, LR, total_steps=EPOCHS * (len(np.unique(cid[train])) // BATCH_CYCLES + 1))

    def output(j):
        f = net(X[j], Z[j])
        return B[j] + 0.2 * torch.tanh(f) if residual else f + 1.0

    def predict(idx):
        net.eval()
        with torch.no_grad():
            return torch.cat([output(t(idx[i:i + 2048])) for i in range(0, len(idx), 2048)]).cpu().numpy()

    # training batches are whole cycles, as in the paper
    order = train[np.argsort(cid[train], kind="stable")]
    cycles, start = np.unique(cid[order], return_index=True)
    bounds = np.r_[start, len(order)]
    best, best_state = np.inf, None
    for _ in range(EPOCHS):
        net.train()
        perm = np.random.permutation(len(cycles))
        for b in range(0, len(perm), BATCH_CYCLES):
            idx = np.concatenate([order[bounds[i]:bounds[i + 1]] for i in perm[b:b + BATCH_CYCLES]])
            j = t(idx)
            loss = ((output(j) - Y[j]) ** 2).mean()
            opt.zero_grad()
            loss.backward()
            opt.step()
            sched.step()
        v = float(np.mean(np.abs(predict(val) - y[val])))
        if device == "cuda":
            torch.cuda.empty_cache()
        if v < best:
            best, best_state = v, {k: x.clone() for k, x in net.state_dict().items()}
    net.load_state_dict(best_state)
    return test, predict(test), best
