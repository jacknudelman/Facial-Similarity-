
# -*- coding: utf-8 -*-
import cv2
import torch
import torchvision
import torchvision.datasets as dset
from torch.utils.data import Dataset, DataLoader
import torchvision.transforms as transforms
import torch.optim as optim
import torch.nn as nn
import torch.nn.functional as F
import matplotlib.pyplot as plt
import numpy as np
from torch.autograd import Variable
# import pandas as pd
import os
from skimage import io
from PIL import Image
import random


def splitIntoMatrix(file):

	separated = open(file, 'r').read().split()
	final_matrix = list()

	for i in range(0, len(separated), 3):
		first = separated[i]
		second = separated[i + 1]
		third = separated[i + 2]
		temp = [first, second, third]
		final_matrix.append(temp)

	return final_matrix

def apply_transformations(img):
	# print 'begin ', img.shape
	if random.uniform(0.0, 1.0) > 0.5:
		img = np.flip(img, axis=0).copy()
		# print '0', img.shape
	if random.uniform(0.0, 1.0) > 0.5:
		rot = random.randint(-30, 30)
		matrix = cv2.getRotationMatrix2D((img.shape[1]/2, img.shape[0]/2), rot, 1.0)
		img = cv2.warpAffine(img, matrix, (img.shape[1], img.shape[0]))
		# print '1', img.shape
	if random.uniform(0.0, 1.0) > 0.5:
		deltax = random.randint(-10, 10)
		deltay = random.randint(-10, 10)
		matrix = np.float32([[1, 0, deltax], [0, 1, deltay]])
		img = cv2.warpAffine(img, matrix, (img.shape[1], img.shape[0]))
		# print '2', img.shape
	# if random.uniform(0.0, 1.0) > 0.5:
	# scale = random.uniform(0.7, 1.3)
	# scale = 0.7
	# img = cv2.resize(img, (0,0), fx=scale, fy=scale)
	# print '3', img.shape
	# print 'final ', img.shape
	return img
class FaceDataset(Dataset):
	def __init__(self, csv_file, root_dir, transform):

		self.faces_with_output = splitIntoMatrix(csv_file)
		self.root_dir = root_dir
		self.transform = transform
		self.possible_trans = list()

	def __len__(self):
		return len(self.faces_with_output)

	def __getitem__(self, idx):
		img1_name = self.root_dir + self.faces_with_output[idx][0]
		img2_name = self.root_dir + self.faces_with_output[idx][1]

		image1 = Image.open(img1_name).convert('RGB')
		image2 = Image.open(img2_name).convert('RGB')

		# if random.uniform(0.0, 1.0) > 0.3:
		# 	image1 = image1.rotate(random.randint(-30,30), expand=1 ), transforms.Scale((128, 128))
		# 	image2 = image2.rotate(random.randint(-30,30), expand=1 ), transforms.Scale((128, 128))
		#
		image1_transformed = self.transform(image1)
		image2_transformed = self.transform(image2)

		label = self.faces_with_output[idx][2]

		return image1_transformed, image2_transformed, label


class RandFaceDataset(Dataset):
	def __init__(self, csv_file, root_dir, transform):

		self.faces_with_output = splitIntoMatrix(csv_file)
		self.root_dir = root_dir
		self.transform = transform
		self.possible_trans = list()

	def __len__(self):
		return len(self.faces_with_output)

	def __getitem__(self, idx):
		img1_name = self.root_dir + self.faces_with_output[idx][0]
		img2_name = self.root_dir + self.faces_with_output[idx][1]

		image1 = Image.open(img1_name).convert('RGB')
		image2 = Image.open(img2_name).convert('RGB')
		# print '####', image1.size
		# scale_transform = transforms.Compose([transforms.Scale((128, 128))])
		poss = list()
		poss.append(transforms.Scale((128, 128)))
		if random.uniform(0.0, 1.0) > 0.5:
			poss.append(transforms.CenterCrop(np.floor(128 * random.uniform(0.7, 1.3))))
			poss.append(transforms.Scale((128, 128)))
			# print 'got scaled'
		scale_transform = transforms.Compose(poss)
		image1 = scale_transform(image1)
		image2 = scale_transform(image2)
		# print 'start ', image1.size


		if random.uniform(0.0, 1.0) > 0.3:
			image1 = np.asarray(image1, dtype=np.float32) / 255
			image2 = np.asarray(image2, dtype=np.float32) / 255

			image1 = apply_transformations(image1).transpose(2, 0, 1)
			image2 = apply_transformations(image2).transpose(2, 0, 1)
			image1 = torch.from_numpy(image1)
			image2 = torch.from_numpy(image2)
			# print 'got transformed', image1.size()
 		# 	image1 = image1.rotate(random.randint(-30,30), expand=1 ), transforms.Scale((128, 128))
		# 	image2 = image2.rotate(random.randint(-30,30), expand=1 ), transforms.Scale((128, 128))
		#
		else:
			# transforms.Scale((128, 128))
			scale_transform = transforms.Compose([transforms.ToTensor()])
			image1 = scale_transform(image1)
			image2 = scale_transform(image2)
			# print 'else size', image1.size()
			# print type(image1)

		image1 = image1.type(torch.FloatTensor)
		image2 = image2.type(torch.FloatTensor)

		# print type(image1)
		# print type(image2)

		label = self.faces_with_output[idx][2]

		return image1, image2, label
