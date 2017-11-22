
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


class FaceDataset(Dataset):
	def __init__(self, csv_file, root_dir, transform):

		self.faces_with_output = splitIntoMatrix(csv_file)
		self.root_dir = root_dir
		self.transform = transform

	def __len__(self):
		return len(self.faces_with_output)

	def __getitem__(self, idx):
		img1_name = self.root_dir + self.faces_with_output[idx][0]
		img2_name = self.root_dir + self.faces_with_output[idx][1]

		image1 = Image.open(img1_name).convert('RGB')
		image2 = Image.open(img2_name).convert('RGB')

		label = self.faces_with_output[idx][2]

		# print 'about to apply transform'
		image1_transformed = self.transform(image1)
		image2_transformed = self.transform(image2)
		# print 'tranform applied'
		# label_tranform = transforms.Compose([transforms.ToTensor()])
		# label_tran = label_tranform(label)
		label_tran = float(label)
		row = {'image1': image1_transformed, 'image2': image2_transformed, 'label': label_tran}

		return row
