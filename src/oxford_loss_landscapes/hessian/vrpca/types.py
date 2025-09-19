"""Type definitions shared across VR-PCA modules."""

from __future__ import annotations

from typing import Callable, Iterable, List, Sequence

import torch

TensorSequence = Sequence[torch.Tensor]
TensorIterable = Iterable[torch.Tensor]
TensorList = List[torch.Tensor]

HVPFn = Callable[[TensorSequence], TensorList]
MinibatchHVPFn = Callable[[TensorSequence, Sequence[int]], TensorList]
