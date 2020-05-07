import os
import numpy as np
import torch
from PIL import Image
from torch.utils.data.dataset import Dataset
#from scipy.misc import imread
from torch import Tensor
from scipy import io
import torchvision.transforms as transforms

from scipy.io import loadmat
from datetime import datetime
"""
Loads the train/test set. 
Every image in the dataset is 28x28 pixels and the labels are numbered from 0-9
for A-J respectively.

Set root to point to the Train/Test folders.
"""

# https://pytorch.org/tutorials/beginner/data_loading_tutorial.html

IMG_EXTENSIONS = [
    '.jpg', '.JPG', '.jpeg', '.JPEG',
    '.png', '.PNG', '.ppm', '.PPM', '.bmp', '.BMP',
    '.tif', '.TIF', '.tiff', '.TIFF',
]

class ClassifyDataset(Dataset):
	def __init__(self, root_dir, transform = None):
		self.root_dir = root_dir
		self.paths = sorted(self.make_dataset(self.root_dir,8000))
		self.transform = transforms.Compose([
			transforms.Resize([224, 224]),
			transforms.ToTensor(),
			transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
		])

	def __getitem__(self, index):

		path = self.paths[index]
		img = Image.open(path).convert('RGB')

		img = self.transform(img)

		return img, path

	def __len__(self):
		return len(self.paths)

	def make_dataset(self, dir, max_dataset_size=float("inf")):
		images = []
		assert os.path.isdir(dir), '%s is not a valid directory' % dir

		for root, _, fnames in sorted(os.walk(dir)):
			for fname in fnames:
				if self.is_image_file(fname):
					path = os.path.join(root, fname)
					images.append(path)
		return images[:min(max_dataset_size, len(images))]

	def is_image_file(self, filename):
		return any(filename.endswith(extension) for extension in IMG_EXTENSIONS)

class IMDBWIKIDataset(Dataset):
	def __init__(self, root_dir, phase, attr, transform = None):
		self.phase = phase
		self.attr = attr
		self.root_dir = root_dir + "/" + self.phase
		self.mat_file = io.loadmat('./imdb_mat/imdb_' + self.attr + '_' + self.phase + '.mat')

		if phase=='train':
			self.transform = transforms.Compose([
				transforms.Resize([256, 256]),
				transforms.RandomResizedCrop(224),
				transforms.RandomHorizontalFlip(),
				transforms.ToTensor(),
				transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
			])
		elif phase == 'test' or phase == 'classify':
			self.transform = transforms.Compose([
				transforms.Resize([224, 224]),
				transforms.ToTensor(),
				transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
			])


	def __getitem__(self, index):
		#image
		img_name = self.mat_file['image'][index]
		img = Image.open(self.root_dir + '/' + img_name).convert('RGB')
		img = self.transform(img)

		#gender
		if self.attr == 'gender':
			gender_mat = [0,0]
			gender = self.mat_file[self.attr][0][index]
			gender_mat[gender] = 1

			return img, gender

		elif self.attr == 'age':
			age = self.mat_file[self.attr][0][index]

			# This should be consistent to 'misc_imdb_preprocessing.py'
			if age < 25:
				age_mat = torch.tensor([0])
			elif age>35 and age<45:
				age_mat = torch.tensor([1])
			elif age>60:
				age_mat = torch.tensor([2])
			else:
				age_mat = torch.tensor([-1])

			return img, age_mat

	def __len__(self):
		return self.mat_file['image'].size