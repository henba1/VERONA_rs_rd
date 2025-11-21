from dataclasses import dataclass
from pathlib import Path

import numpy as np


@dataclass
class VNNLibProperty:
    """Dataclass for a VNNLib property."""

    name: str
    content: str
    path: Path = None
    epsilon: float | None = None
    image: np.ndarray | None = None
    image_class: int | None = None
