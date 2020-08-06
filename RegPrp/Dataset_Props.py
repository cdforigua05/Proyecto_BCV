import os 
import torch
from torch.utils.data import Dataset
from torchvision import transforms
import skimage.io as io
import numpy as np
import scipy.io as sio
from PIL import Image
import matplotlib.pyplot as plt
import matplotlib
from Region_props import getProps

class PropsDataset(Dataset):
	def __init__(self,data_path=None,pesos_path=None, seg_path=None, distribution=0,cuda=True):
		super(PropsDataset,self).__init__()
		self.data_path = data_path
		self.pesos_path = pesos_path
		self.seg_path = seg_path
		self.distribution = distribution
		self.cuda = cuda
		self.list_IDs = self.get_IDs(distribution)
		self.labels = self.get_labels()

	def get_IDs(self, distribution):
		'Denotes the IDs of the image acording to the input distribution'
		'0:train, 1:val, 2:test'
		IDs=[]
		if self.distribution==0: 
			pth = "train"
		elif self.distribution==1:
			pth = "val"
		elif self.distribution==2:
			pth = "test"
		names = os.listdir(os.path.join(self.data_path,pth))
		names = sorted(names)
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
		elif self.distribution==2:
			annots = annots["test"][0]
		for ID,label in zip(self.list_IDs,annots):
			labels[ID] = label
		return labels

	def __len__(self): 
		return len(self.list_IDs)

	def __getitem__(self,index):
		if self.distribution==0: 
			pth = "train"
		elif self.distribution==1:
			pth = "val"
		elif self.distribution==2:
			pth = "test"
		ID = self.list_IDs[index]
		label= self.labels[ID]
		seg = Image.open(os.path.join(self.seg_path,pth,ID)).convert('L')
		bin = np.array(seg,dtype=np.uint8)
		bin[bin<100]=0
		bin[bin>=100]=1
		region_props = getProps(bin)
		return region_props, label
