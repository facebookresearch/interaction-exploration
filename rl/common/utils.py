# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved.

#!/usr/bin/env python3

# Copyright (c) Facebook, Inc. and its affiliates.
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

from collections import defaultdict
from typing import Any, Dict, List, Optional
import numpy as np
import torch
import logging
from torch.utils.tensorboard import SummaryWriter


# [!!] Remove habitat imports
# from habitat import logger
# from habitat.utils.visualizations.utils import images_to_video
# from habitat_baselines.common.tensorboard_utils import TensorboardWriter

# [!!] from habitat.core.logging
class HabitatLogger(logging.Logger):
    def __init__(
        self,
        name,
        level,
        filename=None,
        filemode="a",
        stream=None,
        format=None,
        dateformat=None,
        style="%",
    ):
        super().__init__(name, level)
        if filename is not None:
            handler = logging.FileHandler(filename, filemode)
        else:
            handler = logging.StreamHandler(stream)
        self._formatter = logging.Formatter(format, dateformat, style)
        handler.setFormatter(self._formatter)
        super().addHandler(handler)

    def add_filehandler(self, log_filename):
        filehandler = logging.FileHandler(log_filename)
        filehandler.setFormatter(self._formatter)
        self.addHandler(filehandler)


logger = HabitatLogger(
    name="habitat", level=logging.INFO, format="%(asctime)-15s %(message)s"
)

# [!!] from habitat_baselines.common.tensorboard_utils
class TensorboardWriter:
    def __init__(self, log_dir: str, *args: Any, **kwargs: Any):
        r"""A Wrapper for tensorboard SummaryWriter. It creates a dummy writer
        when log_dir is empty string or None. It also has functionality that
        generates tb video directly from numpy images.

        Args:
            log_dir: Save directory location. Will not write to disk if
            log_dir is an empty string.
            *args: Additional positional args for SummaryWriter
            **kwargs: Additional keyword args for SummaryWriter
        """
        self.writer = None
        if log_dir is not None and len(log_dir) > 0:
            self.writer = SummaryWriter(log_dir, *args, **kwargs)

    def __getattr__(self, item):
        if self.writer:
            return self.writer.__getattribute__(item)
        else:
            return lambda *args, **kwargs: None

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        if self.writer:
            self.writer.close()


def linear_decay(epoch: int, total_num_updates: int) -> float:
    r"""Returns a multiplicative factor for linear value decay

    Args:
        epoch: current epoch number
        total_num_updates: total number of

    Returns:
        multiplicative factor that decreases param value linearly
    """
    return 1 - (epoch / float(total_num_updates))


def _to_tensor(v) -> torch.Tensor:
    if torch.is_tensor(v):
        return v
    elif isinstance(v, np.ndarray):
        return torch.from_numpy(v)
    else:
        return torch.tensor(v, dtype=torch.float)


def batch_obs(
    observations: List[Dict], device: Optional[torch.device] = None
) -> Dict[str, torch.Tensor]:
    r"""Transpose a batch of observation dicts to a dict of batched
    observations.

    Args:
        observations:  list of dicts of observations.
        device: The torch.device to put the resulting tensors on.
            Will not move the tensors if None

    Returns:
        transposed dict of lists of observations.
    """
    batch = defaultdict(list)

    for obs in observations:
        for sensor in obs:
            batch[sensor].append(_to_tensor(obs[sensor]))

    for sensor in batch:
        batch[sensor] = (
            torch.stack(batch[sensor], dim=0)
            .to(device=device)
            .to(dtype=torch.float)
        )

    return batch