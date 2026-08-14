"""
models.py
=========
Standalone, torch-only (no tkinter) extraction of the BiLSTM architecture
from train_context_aware_bilstm_gui.py, for use by the edge-deployment
retraining/export tooling in context-aware-bilstm-edge.

WHY A NEW MODULE INSTEAD OF IMPORTING THE GUI SCRIPT'S CLASSES DIRECTLY:
  train_context_aware_bilstm_gui.py defines Attention/BiLSTM/etc. as local
  classes NESTED inside its _pipeline() method, closing over local
  variables (HIDDEN, N_LAYERS, DROPOUT, in_dim, n_out) rather than taking
  them as constructor arguments -- they are not importable as-is. This
  repo's own CLAUDE.md already documents the same reasoning for why the
  standalone analysis scripts don't import the GUI as a library ("analysis
  scripts shouldn't need to import a tkinter+torch GUI application") --
  applying that symmetrically here rather than retrofitting the GUI's
  already-working, delicate _pipeline() method.

  The architecture below is a byte-for-byte faithful reproduction of the
  GUI's BiLSTM/Attention classes (same layer types, same per-target head
  logic, same PM-gets-ReLU/gas-gets-GELU split) with closure variables
  turned into explicit constructor parameters -- nothing about the
  architecture itself has changed. If train_context_aware_bilstm_gui.py's
  BiLSTM class is ever intentionally modified, this file needs the same
  change applied by hand (same caveat this repo already lives with for
  phase boundaries -- see top-level CLAUDE.md).

ONLY BiLSTM IS EXTRACTED, not the other 7 architectures the GUI supports
-- this repo's own analysis (src/analysis/ct_significance_testing,
forecast_horizon_and_early_warning) found BiLSTM was the one architecture
showing reproducible early-warning behavior; that's the one being taken
to the edge, not a general-purpose model zoo.
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


class Attention(nn.Module):
    """Additive attention over time-steps -- focuses on the trigger moment
    within the lookback window. Identical to the GUI's Attention class."""

    def __init__(self, hidden_dim: int):
        super().__init__()
        self.W = nn.Linear(hidden_dim, 1, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (Batch, Time, Hidden)
        scores = self.W(x).squeeze(-1)             # (B, T)
        weights = F.softmax(scores, dim=1)          # (B, T)
        ctx = (x * weights.unsqueeze(-1)).sum(1)    # (B, H)
        return ctx


class BiLSTM(nn.Module):
    """
    Bidirectional LSTM + attention + per-target heads.

    Per-target head split (index into the 5 ALL_TARGETS = pm1, pm2_5,
    pm10, co2, voc): PM targets (indices 0-2) get a deeper ReLU head
    (concentrations are non-negative); gas targets (indices 3-4) get a
    GELU head (smoother gradient for large-range values). This ordering
    is load-bearing -- it must match feature_engineering.ALL_TARGETS'
    order exactly, or the wrong head activation gets applied to the wrong
    physical quantity.
    """

    def __init__(self, in_dim: int, hidden: int, n_layers: int,
                 dropout: float, n_out: int = 5):
        super().__init__()
        self.lstm = nn.LSTM(
            in_dim, hidden, n_layers, batch_first=True, bidirectional=True,
            dropout=dropout if n_layers > 1 else 0.0,
        )
        self.attn = Attention(hidden * 2)
        self.drop = nn.Dropout(dropout)

        pm_idx = {0, 1, 2}   # pm1, pm2_5, pm10
        gas_idx = {3, 4}     # co2, voc
        self.heads = nn.ModuleList()
        for i in range(n_out):
            if i in pm_idx:
                self.heads.append(nn.Sequential(
                    nn.Linear(hidden * 2, hidden), nn.ReLU(),
                    nn.Dropout(dropout * 0.5), nn.Linear(hidden, 1),
                ))
            elif i in gas_idx:
                self.heads.append(nn.Sequential(
                    nn.Linear(hidden * 2, hidden), nn.GELU(),
                    nn.Linear(hidden, 1),
                ))
            else:
                self.heads.append(nn.Linear(hidden * 2, 1))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        o, _ = self.lstm(x)                  # (B, T, hidden*2)
        ctx = self.drop(self.attn(o))        # (B, hidden*2)
        return torch.cat([hd(ctx) for hd in self.heads], 1)  # (B, n_out)
