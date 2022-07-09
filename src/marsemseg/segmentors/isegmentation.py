#import all your necessary libs
from ..datasets.base import CLASSES, PALETTE

import typing as ty
from pathlib import Path

import torch
import numpy as np

import mmcv
from mmcv.runner import load_checkpoint
from mmcv.parallel import collate, scatter

from mmseg.datasets.pipelines import Compose
from mmseg.models import BaseSegmentor, build_segmentor
from mmseg.apis.inference import LoadImage

InputSignature = ty.Union[
	np.ndarray, 
	str,
	ty.Union[
		ty.List[str],
		ty.List[np.ndarray]
	]
]

OutputSignature = ty.Union[
	np.ndarray,
	ty.List[np.ndarray]
]

class ISegmentation:
	"""
	Segmentation interface.
	
	segmentor: BaseSegmentor the segmentation model

	"""
	def __init__(
		self, 
		cfg_file: ty.Union[Path, str], 
		input_size: ty.Tuple[int, int],
		device: ty.Optional[str] = None,
		half: bool = False,
	) -> None:
		# your machine learning model
		self.segmentor: ty.Optional[BaseSegmentor] = None
		self.half = half
		
		
		if isinstance(cfg_file, str):
			cfg_file = Path(cfg_file)

		if not cfg_file.exists():
			raise FileNotFoundError(f"Config file {self.cfg_file} not found")
			
		self.cfg_file = cfg_file
		self.config: mmcv.Config = mmcv.Config.fromfile(self.cfg_file)

		# device, cpu, gpu
		if device is not None:
			assert any(device.startswith(d) for d in ['cuda', 'cpu']), "Invalid device {device}"
			self.device = device
		else:
			gpu_available = torch.cuda.is_available()
			self.device = "cpu" if gpu_available else "cuda"
		
		self.input_size = input_size
	
		 
	def initialize(self, ckpt_file: ty.Union[Path, str]) -> None:
		if isinstance(ckpt_file, str):
			ckpt_file = Path(ckpt_file)

		if not ckpt_file.exists():
			raise FileNotFoundError(f"Checkpoint file {ckpt_file} not found")
		
		# Initialize your model here
		self.config.model.pretrained = None
		self.config.model.train_cfg = None

		self.model: BaseSegmentor = build_segmentor(
			self.config.model,
			test_cfg=self.config.get('test_cfg')
		)
		
		if self.half:
			self.model = self.model.half()  # to FP16
		load_checkpoint(self.model, str(ckpt_file))
		self.model.CLASSES = CLASSES
		self.model.PALETTE = PALETTE
		self.model.cfg = self.config
		self.model.to(self.device)
		self.model.eval()

		# warmup your model and test once, please print error information if the programming cannot initialize the model
		_ = self.segmentation(np.zeros(list(self.input_size) + [3]))
		print("Load model done")	
		
	def segmentation(self, img: InputSignature) -> OutputSignature:
		cfg = self.model.cfg
		# build the data pipeline
		test_pipeline: ty.List[ty.Dict[str, ty.Any]] = [LoadImage()] + cfg.data.test.pipeline[1:]
		test_pipeline = Compose(test_pipeline)
    	# prepare data
		data = dict(img=img)
		data = test_pipeline(data)
		data = collate([data], samples_per_gpu=1)
		if next(self.model.parameters()).is_cuda:
			# scatter to specified GPU
			data = scatter(data, [self.device])[0]
		else:
			data['img_metas'] = [i.data[0] for i in data['img_metas']]
		
		if self.half:
			for i in range(len(data['img'])):
				data['img'][i] = data['img'][i].half()
				data['img'][i].requires_grad = False
		
		# forward the model
		with torch.no_grad():
			result: OutputSignature = self.model(return_loss=False, rescale=True, **data)
		return result
