import numpy as np
from typing import Dict, List, Tuple

Points = Dict[int, List[Tuple[int, int, int]]]
SegmentationResult = Dict[int, Dict[int, np.ndarray]]  # {slice_index: {object_id: corresponding segmentation_mask}}