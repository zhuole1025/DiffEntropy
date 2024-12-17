# Copyright (c) OpenMMLab. All rights reserved.
import json
import os
import random
import logging
from copy import deepcopy
from random import choice, shuffle
from typing import Sequence

from torch.utils.data import BatchSampler, Dataset, Sampler

logger = logging.getLogger(__name__)

class AspectRatioBatchSampler(BatchSampler):
    """A sampler wrapper for grouping images with similar aspect ratio into a same batch.

    Args:
        sampler (Sampler): Base sampler.
        dataset (Dataset): Dataset providing data information.
        batch_size (int): Size of mini-batch.
        drop_last (bool): If ``True``, the sampler will drop the last batch if
            its size would be less than ``batch_size``.
        aspect_ratios (dict): The predefined aspect ratios.
    """

    def __init__(
        self,
        sampler: Sampler,
        dataset: Dataset,
        batch_size: int,
        aspect_ratios: dict,
        drop_last: bool = False,
    ) -> None:
        self.sampler = sampler
        self.dataset = dataset
        self.batch_size = batch_size
        self.aspect_ratios = aspect_ratios
        self.drop_last = drop_last
        self._aspect_ratio_buckets = {ratio: [] for ratio in aspect_ratios.keys()}

    def __iter__(self) -> Sequence[int]:
        for idx in self.sampler:
            data_info, closest_ratio = self._get_data_info_and_ratio(idx)
            if not data_info:
                continue

            bucket = self._aspect_ratio_buckets[closest_ratio]
            bucket.append(idx)
            # yield a batch of indices in the same aspect ratio group
            if len(bucket) == self.batch_size:
                yield bucket[:]
                del bucket[:]

        for bucket in self._aspect_ratio_buckets.values():
            while bucket:
                if not self.drop_last or len(bucket) == self.batch_size:
                    yield bucket[:]
                del bucket[:]

    def _get_data_info_and_ratio(self, idx):
        data_info = self.dataset.get_data_info(int(idx))
        if data_info is None:
            return None, None
        closest_ratio = data_info["closest_ratio_1024"]
        return data_info, closest_ratio