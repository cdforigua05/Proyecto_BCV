from modeling.deeplab import *
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
from dataloaders import custom_transforms as tr
class MelanomaDataset(Dataset):
	# distribution: Train (0), Val(1) y test(2)
	#data_path: ruta a donde están los datos. la carpeta debe contener la carpeta train, val, test y el Anotaciones.mat
	#dejar_quieto num_classes,backbone,cuda,out_stride,freeze_bn
	#el sync_bn representa si se van a usar más de una GPU. Dejalo en True, yo lo corrí en un GPU y no afecto en nada. 
	#path_seg: es la ruta hacía el archivo de los pesos. Pon en la ruta el .pth.tar.
		#ejemplo: "/home/cdforigua/Proyecto_BCV/deeplab/deeplab-resnet.pth.tar"
	# el input size esta por default en 224, que creo que es el tamaño de entrada de DenseNet. 

	def __init__(self,data_path,distribution=0,transform=None,num_classes=2, backbone='resnet',cuda=True, 
					out_stride=16,sync_bn=True,freeze_bn=False,path_seg="deeplab-resnet.pth.tar",input_size=224):
		super(MelanomaDataset,self).__init__()
		self.distribution = distribution
		self.input_size = input_size
		self.data_path = data_path
		self.list_IDs = self.get_IDs(distribution)
		self.transform =  transform
		self.labels = self.get_labels()
		self.model = DeepLab(num_classes=num_classes,
                        backbone=backbone,
                        output_stride=out_stride,
                        sync_bn=sync_bn,
                        freeze_bn=freeze_bn)
		pesos = torch.load(path_seg)
		self.cuda=cuda
		if self.cuda: 
			self.model=self.model.cuda()
		self.model.load_state_dict(pesos['state_dict'])
		self.model.eval()

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
		image=Image.open(os.path.join(self.data_path,pth,ID)).convert('RGB')
		image_tensor=transforms.functional.to_tensor(image)
		image_tensor=image_tensor.reshape(1,image_tensor.shape[0],image_tensor.shape[1],image_tensor.shape[2])
		output =  self.model(image_tensor.cuda())
		m=nn.Softmax(dim=1)
		seg = m(output)
		seg = output.data.cpu().numpy()
		seg = np.argmax(seg, axis=1)
		seg = np.uint8(seg[0,:,:])
		seg = Image.fromarray(seg)
		sample = {'image':image, 'label':seg}
		if self.distribution == 0:
			sample = self.transforms_train_esp(sample)
			img = sample["image"]
			mask = sample["label"].unsqueeze(0)
			fusion = torch.cat((img,mask),dim=0)
		elif self.distribution!=0: 
			sample = self.transforms_val(sample)
			img = sample["image"]
			mask = sample["label"].unsqueeze(0)
			fusion = torch.cat((img,mask),dim=0)
		return fusion, label

	def transforms_val(self,sample): 
		composed_transforms = transforms.Compose([
			tr.FixedResize(size=self.input_size),
            tr.Normalize(mean=[0.485, 0.456, 0.406],std=[0.229, 0.224, 0.225]),
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
            tr.Normalize(mean=[0.485, 0.456, 0.406],std=[0.229, 0.224, 0.225]),
            tr.ToTensor()            
            ]) 
		return composed_transforms(sample)

#a = MelanomaDataset('../data1',distribution=0,path_seg="/home/cdforigua/Proyecto_BCV/deeplab/run/isic/deeplab-drn/experiment_0/deeplab-resnet.pth.tar")
