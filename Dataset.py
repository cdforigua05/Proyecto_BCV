import os 
import torch
from torch.utils.data import Dataset
from torchvision import transforms
import skimage.io as io
import numpy as np
import scipy.io as sio

class MelanomaDataset(Dataset):
	def __init__(self,data_path,distribution,transform=None):
		super(myDataset,self).__init__()
		self.distribution = distribution
		self.data_path = data_path
		self.list_IDs = self.get_IDs(distribution)
		self.transform =  transform
		self.labels = self.get_labels()

	def get_IDs(self, distribution):
		'Denotes the IDs of the image acording to the input distribution'
		'0:train, 1:val, 2:test'
		IDs=[]
		if self.distribution==0: 
			pth = "train"
		elif self.distribution==1:
			pth = "val"
		else self.distribution==2:
			pth = "test"
		names = os.listdir(os.path.join(self.data_path,pth))
		for image in names:
			IDs.append(image)	
		return IDs
	def get_labels(self): 
		labels={}
		annots=sio.loadmat(os.path.join(self.data_path,"Anotaciones.mat"))
		if self.distribution==0: 
			annots = annots["train"][0]
		elif self.distribution==1:
			annots = annots["val"][0]
		else self.distribution==2:
			annots = annots["test"][0]
		for ID,label in zip(self.list_IDS,annots):
			labels[ID] = label

		return labels

	def __len__(self): 
		return len(self.list_IDs)
	def __getitem__(self,index):
		if self.distribution==0: 
			pth = "train"
		elif self.distribution==1:
			pth = "val"
		else self.distribution==2:
			pth = "test"
		
		ID = self.list_IDs[index]
		label= self.labels[ID]
		image = io.imread(os.path.join(self.data_path,pth,ID))
		
		if self.transform:
			image = self.transform(image)
		return image, label