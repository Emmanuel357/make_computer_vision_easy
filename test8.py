import os
import numpy as np

from torchvision.utils import save_image

import torch.nn as nn
import torch
from mods import *
from torchvision import transforms

n_imgs = 12400
img_shape = 28*28
img_size = (28, 28)

class Classifier(nn.Module):
	def __init__(self):
		super(Classifier, self).__init__()
		
		self.lin_pred_model = nn.Sequential(
			nn.LayerNorm(img_shape),
			nn.Linear(img_shape, 32, bias = False),
			nn.Sigmoid(),
			
			nn.LayerNorm(32),
			nn.Linear(32, 3, bias = False),
			nn.Sigmoid()
		)
		
		self.cfe_model = nn.Sequential(
			nn.GroupNorm(1, 1),
			nn.Conv2d(1, 16, 4, stride=2, padding=0),
			nn.GroupNorm(1, 16), 
			nn.LeakyReLU(0.2, inplace=True),
			nn.Conv2d(16, 4, 3, stride=1, padding=0),
			nn.GroupNorm(1, 4), 
			nn.LeakyReLU(0.2, inplace=True),
			
		)
		
		self.conv_pred_model = nn.Sequential(
			nn.Conv2d(4, 3, 11, stride=1, padding=0),
			nn.Sigmoid()
		)
		
	def forward(self, img):
		preds = self.lin_pred_model(img)
		'''
		img = img.reshape(img.shape[0], 1, *img_size)
		self.features = self.cfe_model(img)
		
		preds = self.conv_pred_model(self.features)
		preds = preds.reshape(preds.shape[0],3)
		'''
		return preds

class randomTransforms:
	def __init__(self):
		self.image_transforms = [
			transforms.RandomPerspective(),
			transforms.RandomRotation(360),
			transforms.RandomVerticalFlip(p=0.5),
			transforms.RandomErasing(value = "random")
		]
	
	def transform(self, img):
		ct = np.random.choice(len(self.image_transforms))
		if ct != 3:
			timg = transforms.ToPILImage()(img)
		timg = self.image_transforms[ct](timg)
		timg = transforms.ToTensor()(timg)
		return timg
		
# Initialize generator and discriminator
classifier = Classifier()

dogs = torch.tensor(np.load("d_gs.npy")).float().reshape(12500, img_shape)[:n_imgs] + 1
cats = torch.tensor(np.load("c_gs.npy")).float().reshape(12500, img_shape)[:n_imgs] + 1
cars = torch.tensor(np.load("non_cd_imgs.npy")).float().reshape(354565, img_shape)[:n_imgs] + 1


rt = randomTransforms()
timg = rt.transform(dogs[0].reshape(1,28,28))

print(timg.shape)

save_image(timg[:25], "images/timg.png", nrow=5, normalize=True)







