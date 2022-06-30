from .base import CLASSES, PALETTE

import typing as ty
import os.path as osp

from mmseg.datasets.builder import DATASETS
from mmseg.datasets.custom import CustomDataset

@DATASETS.register_module()
class MaSTr1325(CustomDataset):
  CLASSES = CLASSES
  PALETTE = PALETTE
  def __init__(self, split: str, **kwargs: ty.Dict[str, ty.Any]):
    super().__init__(img_suffix='.jpg', seg_map_suffix='.png', 
                     split=split, **kwargs)
    assert osp.exists(self.img_dir) and self.split is not None



@DATASETS.register_module()
class MODD2(CustomDataset):
  CLASSES = CLASSES
  PALETTE = PALETTE
  def __init__(self, split: str, **kwargs: ty.Dict[str, ty.Any]):
    super().__init__(img_suffix='.jpg', seg_map_suffix='.png', 
                     split=split, **kwargs)
    assert osp.exists(self.img_dir) and self.split is not None, "MODD2 dataset not found"



@DATASETS.register_module()
class SMD(CustomDataset):
  CLASSES = CLASSES
  PALETTE = PALETTE
  def __init__(self, split: str, **kwargs: ty.Dict[str, ty.Any]):
    super().__init__(img_suffix='.jpg', seg_map_suffix='.png', 
                     split=split, **kwargs)
    assert osp.exists(self.img_dir) and self.split is not None


@DATASETS.register_module()
class MID(CustomDataset):
  CLASSES = CLASSES
  PALETTE = PALETTE
  def __init__(self, split: str, **kwargs: ty.Dict[str, ty.Any]):
    super().__init__(img_suffix='.jpg', seg_map_suffix='.png', 
                     split=split, **kwargs)
    assert osp.exists(self.img_dir) and self.split is not None
