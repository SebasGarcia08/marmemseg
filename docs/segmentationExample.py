#import all your necessary libs
import cv2
import torch
import torch.backends.cudnn as cudnn
from numpy import random
import numpy as np

class iSegmentation:
	def __init__(self):
		# your machine learning model
		self.model = None
		# device, cpu, gpu
		self.device = select_device('0')
		# the processed imgae size in your model
		self.imgsz = 640
		# define other parameters if you have
		
	def initialize(self, xi_weights, device):
		# Initialize your model here
		self.device = select_device(device)
		self.half = self.device.type != 'cpu'  # half precision only supported on CUDA
		# resize the input image "xi_weights" for your model
		self.model = attempt_load(xi_weights, map_location=self.device)  # load FP32 model
		self.imgsz = check_img_size(self.imgsz, s=self.model.stride.max())  # check img_size
		if self.half:
			self.model.half()  # to FP16
		# warmup your model and test once, please print error information if the programming cannot initialize the model
		img = torch.zeros((1, 3, self.imgsz, self.imgsz), device=self.device)
		_ = self.model(img.half() if self.half else img) if self.device.type != 'cpu' else None  # run once
		print("Load model done")	
		
	def segmentation(self, imgInput):
		# the main semantic segmentation code/function
		# resize the input image and siwth to CHW (channel-height-width) with RGB color 
		img = resizeToModelInput(imgInput,imgsz)
		img = np.moveaxis(img, -1, 0)
		# set data device
		img = torch.from_numpy(img).to(self.device)
		img = img.half() if self.half else img.float()  # uint8 to fp16/32
		img /= 255.0  # 0 - 255 to 0.0 - 1.0
		if img.ndimension() == 3:
			img = img.unsqueeze(0)

		# call model processing inference
		output = self.model(img, augment=False)[0]
		
		# resize the output imaget, the output image should have the same size as the input image "imgInput".
		# Format: Output the segmentation image with the same size as the input image. Using different pixel value to define the pixel category, such as, “0” means background, “1” means person,… 
		# Category List: Please send the category-index matching file to tell us the meaning of the pixel index in your segmentation image. Please also prefer to use some common category list, e.g. COCO category ID.
		imgOutput = resizeToInputImage(output,imgInput)
		return imgOutput
