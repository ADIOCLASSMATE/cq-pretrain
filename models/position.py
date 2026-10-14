import math

import torch


def build_row_col_position_ids(
    token_types: torch.Tensor,
    image_tokens_per_img: int,
) -> torch.Tensor:
    """Build two-axis multimodal coordinates with exact pure-text 1D compatibility.

    Text, special, and padding tokens receive ``(p, p)``. Each contiguous image
    span is anchored at the current running cursor and laid out on a square grid.
    After an image span, the cursor advances by the canonical spatial extent
    instead of the flattened image-token count.
    """

    if token_types.ndim != 2:
        raise ValueError(f"token_types must be [B,L], got {tuple(token_types.shape)}")
    image_tokens_per_img = int(image_tokens_per_img)
    side = int(math.isqrt(image_tokens_per_img))
    if side * side != image_tokens_per_img:
        raise ValueError(
            f"row/column image positions require a square grid, got {image_tokens_per_img} tokens"
        )
    canonical_spatial_extent = side

    batch_size, seq_len = token_types.shape
    image_mask = token_types == 1
    previous_is_image = torch.cat(
        [
            torch.zeros(batch_size, 1, device=token_types.device, dtype=torch.bool),
            image_mask[:, :-1],
        ],
        dim=1,
    )
    span_starts = image_mask & ~previous_is_image
    span_count = span_starts.long().cumsum(dim=1)

    non_image = (~image_mask).long()
    non_image_before = non_image.cumsum(dim=1) - non_image
    sequence_positions = torch.arange(
        seq_len,
        device=token_types.device,
        dtype=torch.long,
    ).unsqueeze(0).expand(batch_size, -1)
    start_locations = torch.where(
        span_starts,
        sequence_positions,
        torch.zeros_like(sequence_positions),
    )
    latest_start = torch.cummax(start_locations, dim=1).values
    local = sequence_positions - latest_start

    image_anchor = non_image_before + (
        (span_count - 1).clamp_min(0) * canonical_spatial_extent
    )
    text_position = non_image_before + span_count * canonical_spatial_extent
    rows = image_anchor + local.div(side, rounding_mode="floor")
    cols = image_anchor + (local % side)
    row_ids = torch.where(image_mask, rows, text_position)
    col_ids = torch.where(image_mask, cols, text_position)
    return torch.stack([row_ids, col_ids], dim=0)


def _interleaved_coordinate_axes(axis_dims: tuple[int, int], device) -> torch.Tensor:
    remaining = [int(axis_dims[0]) // 2, int(axis_dims[1]) // 2]
    axes = []
    while remaining[0] or remaining[1]:
        for axis in (0, 1):
            if remaining[axis]:
                axes.append(axis)
                remaining[axis] -= 1
    return torch.tensor(axes, device=device, dtype=torch.long)


def build_local_row_col_rope(
    positions: torch.Tensor,
    *,
    image_tokens_per_img: int,
    head_dim: int,
    axis_dims: tuple[int, int],
    rope_theta: float = 10000.0,
    dtype: torch.dtype | None = None,
    validate_positions: bool = False,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Build interleaved local-grid row/column RoPE cos/sin in FP32."""

    if positions.ndim != 2:
        raise ValueError(f"positions must be [B,L], got {tuple(positions.shape)}")
    image_tokens_per_img = int(image_tokens_per_img)
    side = int(math.isqrt(image_tokens_per_img))
    if side * side != image_tokens_per_img:
        raise ValueError(
            "Flow-head row/column RoPE requires a square image-token grid, "
            f"got {image_tokens_per_img}."
        )
    head_dim = int(head_dim)
    axis_dims = tuple(int(item) for item in axis_dims)
    if len(axis_dims) != 2 or sum(axis_dims) != head_dim:
        raise ValueError(
            f"axis_dims={axis_dims} must contain two entries summing to {head_dim}."
        )
    if any(dim <= 0 or dim % 2 for dim in axis_dims):
        raise ValueError(f"axis_dims must be positive and even, got {axis_dims}.")
    positions = positions.to(dtype=torch.long)
    if validate_positions and positions.numel():
        min_position = int(positions.min().item())
        max_position = int(positions.max().item())
        if min_position < 0 or max_position >= image_tokens_per_img:
            raise ValueError(
                f"positions must be in [0,{image_tokens_per_img}), "
                f"got min={min_position}, max={max_position}."
            )

    rows = positions.div(side, rounding_mode="floor")
    cols = positions % side
    coordinates = torch.stack([rows, cols], dim=-1).float()
    coordinate_axes = _interleaved_coordinate_axes(axis_dims, positions.device)
    selected_coordinates = coordinates.index_select(-1, coordinate_axes)
    inv_freq = 1.0 / (
        float(rope_theta)
        ** (
            torch.arange(
                0,
                head_dim,
                2,
                device=positions.device,
                dtype=torch.float32,
            )
            / float(head_dim)
        )
    )
    frequencies = selected_coordinates.float() * inv_freq
    phases = torch.cat([frequencies, frequencies], dim=-1)
    cos = phases.cos()
    sin = phases.sin()
    if dtype is not None:
        cos = cos.to(dtype=dtype)
        sin = sin.to(dtype=dtype)
    return cos, sin


def rotate_half(x: torch.Tensor) -> torch.Tensor:
    first, second = x.chunk(2, dim=-1)
    return torch.cat([-second, first], dim=-1)


def apply_local_row_col_rope(
    x: torch.Tensor,
    positions: torch.Tensor,
    *,
    image_tokens_per_img: int,
    axis_dims: tuple[int, int],
    rope_theta: float = 10000.0,
) -> torch.Tensor:
    """Apply local-grid RoPE to ``[B,H,L,D]`` Q or K tensors."""

    if x.ndim != 4:
        raise ValueError(f"x must be [B,H,L,D], got {tuple(x.shape)}")
    if positions.shape != (x.shape[0], x.shape[2]):
        raise ValueError(
            f"positions must have shape {(x.shape[0], x.shape[2])}, "
            f"got {tuple(positions.shape)}."
        )
    cos, sin = build_local_row_col_rope(
        positions.to(device=x.device),
        image_tokens_per_img=image_tokens_per_img,
        head_dim=x.shape[-1],
        axis_dims=axis_dims,
        rope_theta=rope_theta,
        dtype=x.dtype,
    )
    cos = cos.unsqueeze(1)
    sin = sin.unsqueeze(1)
    return x * cos + rotate_half(x) * sin
