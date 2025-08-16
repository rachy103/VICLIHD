# VICLIHD/rk_diffuser_mod/context_aware_diffusion.py
from typing import Dict, Any, Optional, Tuple
import torch
import torch.nn as nn

class ContextAwareGaussianDynDiffusion(nn.Module):
    """
    Wraps an existing GaussianDynDiffusion and injects an extra
    conditioning vector 'context' of shape (B,H,D), (B,D), or (H,D).

    중요: HDP의 apply_conditioning(x, cond, ...)는 cond의 키를 타임스텝 인덱스(int)로 가정한다.
    따라서 컨텍스트는 cond에 넣지 말고, **kwargs로만 전달한다.
    또한 cond에 들어온 비정수 키는 모두 제거한다.
    """
    def __init__(self, base_model: nn.Module, context_dim: int, hidden_dim: int = 256):
        super().__init__()
        self.base = base_model
        self.ctx_proj = nn.Linear(context_dim, hidden_dim)
        self.fuse = nn.Linear(hidden_dim, hidden_dim)

    def _ensure_device_dtype(self, x: torch.Tensor) -> torch.Tensor:
        p = next(self.base.parameters(), None)
        if p is None:
            return x
        return x.to(device=p.device, dtype=p.dtype)

    def _normalize_context_shape(
        self,
        ctx: torch.Tensor,
        B: Optional[int] = None,
        H: Optional[int] = None,
    ) -> torch.Tensor:
        """(B,H,D) / (B,D) / (H,D) -> (B,H,D)"""
        if ctx.dim() == 3:
            return ctx  # (B,H,D)
        if ctx.dim() == 2:
            if B is not None:
                ctx = ctx.unsqueeze(1)  # (B,1,D)
                if H is not None and H > 1 and ctx.size(1) != H:
                    ctx = ctx.repeat(1, H, 1)
                return ctx
            if H is not None:
                return ctx.unsqueeze(0)  # (1,H,D)
            return ctx.unsqueeze(0).unsqueeze(1)  # (1,1,D)
        raise ValueError(f"Unsupported context shape: {tuple(ctx.shape)}")

    def _build_context_kwargs(
        self,
        cond: Optional[Dict[str, Any]],
        obs: Optional[torch.Tensor] = None,
    ) -> Tuple[Dict[str, Any], Dict[str, Any]]:
        """
        반환:
          - cond_passthrough: 정수 키만 남긴 cond (문자열/그 외 키 제거)
          - extra_kwargs: context_seq/context_fused (kwargs로만 전달)
        """
        cond_in = {} if cond is None else dict(cond)

        # 1) 원본 cond에서 비정수 키 제거 (특히 'context' 제거)
        cond_out: Dict[int, Any] = {}
        for k, v in cond_in.items():
            if isinstance(k, int):
                cond_out[k] = v
            # else: 문자열/기타 키는 버림 (e.g., 'context')

        # 2) kwargs로 전달할 컨텍스트 임베딩 구성
        ctx = cond_in.get("context", None)
        if ctx is None:
            return cond_out, {}

        # obs로부터 (B,H) 추정
        B = H = None
        if isinstance(obs, torch.Tensor) and obs.dim() >= 2:
            B, H = obs.shape[:2]

        # 텐서화 + 디바이스/타입 정렬
        if not isinstance(ctx, torch.Tensor):
            ctx = torch.as_tensor(ctx)
        ctx = self._ensure_device_dtype(ctx)

        # (B,H,D) 정규화
        ctx_bhd = self._normalize_context_shape(ctx, B=B, H=H)

        # per-timestep 임베딩
        ctx_seq = self.ctx_proj(ctx_bhd)             # (B,H,hidden)
        # 에피소드 임베딩
        ctx_fused = self.fuse(ctx_seq.mean(dim=1))   # (B,hidden)

        extra_kwargs = {
            "context_seq": ctx_seq,
            "context_fused": ctx_fused,
        }
        # 베이스 샘플러가 cond[0] 길이로 배치 크기를 추정하는 경우가 있어, 비어있으면 더미 cond를 넣어줌
        if len(cond_out) == 0:
            Bsize = ctx_bhd.shape[0]
            cond_out = {0: torch.zeros(Bsize, device=ctx_bhd.device)}
        return cond_out, extra_kwargs

    def loss(
        self,
        obs: torch.Tensor,
        action: torch.Tensor,
        cond: Optional[Dict[str, Any]] = None,
        **kw,
    ) -> torch.Tensor:
        """
        GaussianDynDiffusion.loss expects: loss(x, cond, **kwargs)
        여기서 diffusion_var='gripper_poses' → x=action
        """
        cond_passthrough, extra = self._build_context_kwargs(cond, obs=obs)
        return self.base.loss(x=action, cond=cond_passthrough, **extra, **kw)

    @torch.no_grad()
    def conditional_sample(
        self,
        cond: Optional[Dict[str, Any]] = None,
        horizon: Optional[int] = None,
        **kw,
    ):
        """
        GaussianDynDiffusion.conditional_sample(cond, horizon=None, **kwargs)
        """
        cond_passthrough, extra = self._build_context_kwargs(cond, obs=None)
        return self.base.conditional_sample(cond=cond_passthrough, horizon=horizon, **extra, **kw)

    def forward(self, *args, **kwargs):
        # 편의상 forward는 conditional_sample을 위임
        return self.conditional_sample(*args, **kwargs)
