"""
OpenVaccine mRNA Degradation Predictor

Agents: modify this file to improve the MCRMSE score.
Do NOT modify eval/ or prepare.sh.

Experiment: BPPS graph convolution + SNR-weighted loss + early stopping
"""

import json
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import random

# ======================= CONFIG =======================
SEED = 42          # keep in sync with eval/score.py
VAL_SPLIT = 0.2    # keep in sync with eval/score.py
BATCH_SIZE = 32
EPOCHS = 100
LR = 1e-3
HIDDEN_SIZE = 256
NUM_LAYERS = 3
DROPOUT = 0.3
PATIENCE = 15
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

TARGETS = ["reactivity", "deg_Mg_pH10", "deg_pH10", "deg_Mg_50C", "deg_50C"]
SCORED_TARGETS = ["reactivity", "deg_Mg_pH10", "deg_Mg_50C"]
SEQ_SCORED = 68
# ======================================================

SEQ_VOCAB    = {"A": 0, "G": 1, "C": 2, "U": 3}
STRUCT_VOCAB = {".": 0, "(": 1, ")": 2}
LOOP_VOCAB   = {"S": 0, "M": 1, "I": 2, "B": 3, "H": 4, "E": 5, "X": 6}


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def load_json(path):
    with open(path) as f:
        content = f.read().strip()
    if content.startswith("["):
        return json.loads(content)
    return [json.loads(line) for line in content.splitlines() if line.strip()]


def parse_bpps(bpps_raw, seq_len):
    """Return (seq_len, seq_len) BPPS matrix, zero-padded if needed."""
    flat = np.array(bpps_raw, dtype=np.float32) if bpps_raw else np.array([], dtype=np.float32)
    mat = np.zeros((seq_len, seq_len), dtype=np.float32)
    if flat.size > 0:
        L = int(round(flat.size ** 0.5))
        if L * L == flat.size:
            L = min(L, seq_len)
            mat[:L, :L] = flat.reshape(int(flat.size**0.5), int(flat.size**0.5))[:L, :L]
    return mat


class RNADataset(Dataset):
    def __init__(self, data):
        self.data = data

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        row = self.data[idx]
        seq_len = len(row["sequence"])

        seq_t    = torch.tensor([SEQ_VOCAB.get(c, 0)    for c in row["sequence"]],            dtype=torch.long)
        struct_t = torch.tensor([STRUCT_VOCAB.get(c, 0) for c in row["structure"]],           dtype=torch.long)
        loop_t   = torch.tensor([LOOP_VOCAB.get(c, 0)   for c in row["predicted_loop_type"]], dtype=torch.long)

        bpps_mat = parse_bpps(row.get("bpps", []), seq_len)
        bpps_t = torch.tensor(bpps_mat, dtype=torch.float32)  # (L, L)

        labels = np.zeros((SEQ_SCORED, len(TARGETS)), dtype=np.float32)
        for i, t in enumerate(TARGETS):
            if t in row and row[t]:
                vals = row[t][:SEQ_SCORED]
                labels[:len(vals), i] = vals

        snr = float(row.get("signal_to_noise", 0.0))
        return seq_t, struct_t, loop_t, bpps_t, torch.tensor(labels), snr, row["id"]


def collate_fn(batch):
    seqs, structs, loops, bpps_list, labels, snrs, ids = zip(*batch)
    max_len = max(s.shape[0] for s in seqs)
    B = len(seqs)
    seq_p    = torch.zeros(B, max_len, dtype=torch.long)
    struct_p = torch.zeros(B, max_len, dtype=torch.long)
    loop_p   = torch.zeros(B, max_len, dtype=torch.long)
    bpps_p   = torch.zeros(B, max_len, max_len, dtype=torch.float32)
    for i, (s, st, l, bp) in enumerate(zip(seqs, structs, loops, bpps_list)):
        L = s.shape[0]
        seq_p[i, :L]      = s
        struct_p[i, :L]   = st
        loop_p[i, :L]     = l
        bpps_p[i, :L, :L] = bp
    return seq_p, struct_p, loop_p, bpps_p, torch.stack(labels), torch.tensor(snrs, dtype=torch.float32), list(ids)


class GRUGraphModel(nn.Module):
    """biGRU with a single BPPS graph convolution step before the GRU."""
    def __init__(self):
        super().__init__()
        self.seq_emb    = nn.Embedding(len(SEQ_VOCAB) + 1,    32)
        self.struct_emb = nn.Embedding(len(STRUCT_VOCAB) + 1, 16)
        self.loop_emb   = nn.Embedding(len(LOOP_VOCAB) + 1,   16)
        # After embedding: 64 dims
        # After graph conv: 64 + 64 = 128 dims concatenated with original
        self.gcn_proj = nn.Linear(64, 64)
        self.gru = nn.GRU(
            128, HIDDEN_SIZE,
            num_layers=NUM_LAYERS,
            batch_first=True,
            bidirectional=True,
            dropout=DROPOUT if NUM_LAYERS > 1 else 0.0,
        )
        self.drop = nn.Dropout(DROPOUT)
        self.head = nn.Linear(HIDDEN_SIZE * 2, len(TARGETS))

    def forward(self, seq, struct, loop, bpps):
        # bpps: (B, L, L) — BPPS adjacency matrix
        emb = torch.cat([self.seq_emb(seq), self.struct_emb(struct), self.loop_emb(loop)], dim=-1)
        # Row-normalize BPPS for graph convolution
        row_sum = bpps.sum(dim=-1, keepdim=True).clamp(min=1e-8)
        bpps_norm = bpps / row_sum  # (B, L, L)
        # Graph convolution: each node aggregates neighbor embeddings weighted by BPPS
        gcn_out = torch.bmm(bpps_norm, self.gcn_proj(emb))  # (B, L, 64)
        x = torch.cat([emb, gcn_out], dim=-1)               # (B, L, 128)
        out, _ = self.gru(x)
        return self.head(self.drop(out))  # (B, L, n_targets)


def snr_weighted_loss(preds, labels, snr):
    scored_idx = [TARGETS.index(t) for t in SCORED_TARGETS]
    diff = (preds[:, :, scored_idx] - labels[:, :, scored_idx]) ** 2
    mse_per_sample = diff.mean(dim=(1, 2))
    weights = torch.sigmoid(snr).to(preds.device)
    return (weights * mse_per_sample).mean()


def mcrmse(preds, labels):
    scored_idx = [TARGETS.index(t) for t in SCORED_TARGETS]
    return torch.sqrt(((preds[:, :, scored_idx] - labels[:, :, scored_idx]) ** 2).mean(dim=(0, 1))).mean().item()


def main():
    set_seed(SEED)
    print(f"Device: {DEVICE}")

    all_data = load_json("data/train.json")

    np.random.seed(SEED)
    idx = np.random.permutation(len(all_data))
    val_size = int(len(all_data) * VAL_SPLIT)
    val_idx = set(idx[:val_size].tolist())
    train_split = [d for i, d in enumerate(all_data) if i not in val_idx]
    val_split   = [d for i, d in enumerate(all_data) if i in val_idx]

    print(f"Train: {len(train_split)} | Val: {len(val_split)}")

    train_loader = DataLoader(RNADataset(train_split), batch_size=BATCH_SIZE, shuffle=True,  collate_fn=collate_fn)
    val_loader   = DataLoader(RNADataset(val_split),   batch_size=BATCH_SIZE, shuffle=False, collate_fn=collate_fn)

    model     = GRUGraphModel().to(DEVICE)
    optimizer = torch.optim.Adam(model.parameters(), lr=LR)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=EPOCHS, eta_min=1e-5)

    best_score = float("inf")
    no_improve = 0

    for epoch in range(1, EPOCHS + 1):
        model.train()
        train_loss = 0.0
        for seq, struct, loop, bpps, labels, snr, _ in train_loader:
            seq, struct, loop, bpps, labels = (
                seq.to(DEVICE), struct.to(DEVICE), loop.to(DEVICE),
                bpps.to(DEVICE), labels.to(DEVICE),
            )
            preds = model(seq, struct, loop, bpps)[:, :SEQ_SCORED, :]
            loss  = snr_weighted_loss(preds, labels, snr)
            optimizer.zero_grad()
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            train_loss += loss.item()

        scheduler.step()

        model.eval()
        all_preds, all_labels = [], []
        with torch.no_grad():
            for seq, struct, loop, bpps, labels, _, _ in val_loader:
                seq, struct, loop, bpps = seq.to(DEVICE), struct.to(DEVICE), loop.to(DEVICE), bpps.to(DEVICE)
                all_preds.append(model(seq, struct, loop, bpps)[:, :SEQ_SCORED, :].cpu())
                all_labels.append(labels)

        val_score = mcrmse(torch.cat(all_preds), torch.cat(all_labels))
        if val_score < best_score:
            best_score = val_score
            no_improve = 0
            torch.save(model.state_dict(), "best_model.pt")
        else:
            no_improve += 1

        if epoch % 5 == 0 or epoch == 1:
            print(f"Epoch {epoch:3d}/{EPOCHS} | loss: {train_loss/len(train_loader):.4f} | val MCRMSE: {val_score:.4f} | best: {best_score:.4f}")

        if no_improve >= PATIENCE:
            print(f"Early stopping at epoch {epoch}")
            break

    # Save predictions with best model
    model.load_state_dict(torch.load("best_model.pt", map_location=DEVICE))
    model.eval()
    rows = []
    with torch.no_grad():
        for seq, struct, loop, bpps, _, _, ids in val_loader:
            seq, struct, loop, bpps = seq.to(DEVICE), struct.to(DEVICE), loop.to(DEVICE), bpps.to(DEVICE)
            preds = model(seq, struct, loop, bpps)[:, :SEQ_SCORED, :].cpu().numpy()
            for b, sid in enumerate(ids):
                for pos in range(SEQ_SCORED):
                    row = {"id_seqpos": f"{sid}_{pos}"}
                    row.update({t: float(preds[b, pos, k]) for k, t in enumerate(TARGETS)})
                    rows.append(row)

    pd.DataFrame(rows).to_csv("predictions.csv", index=False)
    print(f"\nBest val MCRMSE: {best_score:.4f}")
    print("Saved predictions.csv")


if __name__ == "__main__":
    main()
