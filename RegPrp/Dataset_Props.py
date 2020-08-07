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
from RegionProps import getProps
import custom_transforms as tr

class PropsDataset(Dataset):
	def __init__(self,data_path=None,pesos_path=None, seg_path=None, distribution=0,cuda=True,input_size=None):
		super(PropsDataset,self).__init__()
		self.data_path = data_path
		self.pesos_path = pesos_path
		self.seg_path = seg_path
		self.distribution = distribution
		self.cuda = cuda
		self.list_IDs = self.get_IDs(distribution)
		self.labels = self.get_labels()
		self.input_size = input_size

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
		image=Image.open(os.path.join(self.data_path,pth,ID)).convert('RGB')
		bin = np.array(seg,dtype=np.uint8)
		bin[bin<100]=0
		bin[bin>=100]=1
		seg = Image.fromarray(bin)
		sample = {'image':image, 'label':seg}
		if self.distribution == 0:
			sample = self.transforms_train_esp(sample)
		elif self.distribution!=0: 
			sample = self.transforms_val(sample)
		img = sample["image"]
		mask = sample["label"]
		bin = mask.numpy()
		bin = np.uint8(bin)
		region_props = getProps(bin)
		return img,region_props,label

	def transforms_val(self,sample): 
		composed_transforms = transforms.Compose([
			tr.FixedResize(size=self.input_size),
            tr.Normalize(mean=[x/255.0 for x in [125.3, 123.0, 113.9]],
                                     std=[x/255.0 for x in [63.0, 62.1, 66.7]]),
            tr.ToTensor()
			])
		return composed_transforms(sample)

	def transforms_train_esp(self,sample):
		composed_transforms = transforms.Compose([
			tr.RandomVerticalFlip(),
            tr.RandomHorizontalFlip(),
            tr.RandomAffine(degrees=40, scale=(.9, 1.1), shear=30),
            tr.ColorJitter(brightness=0.5, contrast=0.5, saturation=0.5),
            tr.FixedResize(size=self.input_size),
            tr.Normalize(mean=[x/255.0 for x in [125.3, 123.0, 113.9]],
                                     std=[x/255.0 for x in [63.0, 62.1, 66.7]]),
            tr.ToTensor()            
            ]) 
		return composed_transforms(sample)
