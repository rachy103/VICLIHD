# your_project/rk_diffuser_mod/context_dataset.py
from __future__ import annotations
from typing import Iterable, Callable, Dict, Any
import numpy as np
import torch
from torch.utils.data import Dataset

def aggregate_context(arr: np.ndarray, mode: str | Callable) -> np.ndarray:
    """(T,N,D) 또는 (T,D) → (T,D)."""
    if arr.ndim == 2:
        return arr
    if callable(mode):
        return mode(arr)
    if mode == "mean":
        return arr.mean(axis=1)
    if mode == "max":
        return arr.max(axis=1)
    if mode == "sum":
        return arr.sum(axis=1)
    raise ValueError(f"Unknown agg mode: {mode}")

class ContextRLBenchDataset(Dataset):
    """
    기존 RLBenchDataset을 감싸서 sample에 'context'를 추가.
    base_dataset[i]가 dict를 반환한다고 가정: {obs, action, ...}
    """
    def __init__(self,
                 base_dataset: Dataset,
                 context_provider: Iterable[np.ndarray],
                 agg_fn: str | Callable = "mean") -> None:
        self.base = base_dataset
        self.ctx_iter = iter(context_provider)
        self.agg_fn = agg_fn
        # 에피소드 단위 컨텍스트를 미리 리스트로 모아 인덱싱 가능하게.
        self.ctx_bank = []
        for ctx in context_provider:
            self.ctx_bank.append(ctx)
        # 다시 iterator 복구
        self.ctx_iter = iter(self.ctx_bank)

    def __len__(self) -> int:
        return len(self.base)

    def __getitem__(self, i: int) -> Dict[str, Any]:
        sample = self.base[i]                 # {obs:(H,do), action:(H,da), ...}
        T = sample["action"].shape[0]
        epi_ctx = self.ctx_bank[min(i, len(self.ctx_bank)-1)]
        epi_ctx = aggregate_context(epi_ctx, self.agg_fn)  # (T,D)
        if epi_ctx.shape[0] != T:
            # 길이 안 맞으면 앞/뒤를 잘라 맞춤(간단 전략)
            m = min(T, epi_ctx.shape[0])
            epi_ctx = epi_ctx[:m]
            if m < T:
                pad = np.repeat(epi_ctx[-1:], T - m, axis=0)
                epi_ctx = np.concatenate([epi_ctx, pad], axis=0)
        sample["context"] = torch.from_numpy(epi_ctx).float()
        return sample

def context_collate_fn(batch: list[Dict[str, Any]]) -> Dict[str, Any]:
    out: Dict[str, Any] = {}
    for k in batch[0].keys():
        if k == "context":
            out[k] = torch.stack([b[k] for b in batch], dim=0)  # (B,H,D)
        elif isinstance(batch[0][k], torch.Tensor):
            out[k] = torch.stack([b[k] for b in batch], dim=0)
        else:
            out[k] = [b[k] for b in batch]
    return out
