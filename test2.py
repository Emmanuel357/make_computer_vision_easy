import os
import numpy as np

from torchvision.utils import save_image

import torch.nn as nn
import torch
from mods import *

n_imgs = 12400
img_shape = 28*28
img_size = (28, 28)

class Classifier(nn.Module):
	def __init__(self):
		super(Classifier, self).__init__()
		

		
		self.cfe_model1 = nn.Sequential(
			nn.GroupNorm(1, 1),
			nn.Conv2d(1, 16, 4, stride=2, padding=0),
			nn.GroupNorm(1, 16), 
			nn.LeakyReLU(0.2, inplace=True),
			nn.Conv2d(16, 8, 3, stride=1, padding=0),
			nn.GroupNorm(1, 8), 
			nn.LeakyReLU(0.2, inplace=True),
			nn.Conv2d(8, 1, 11, stride=1, padding=0),
			nn.Tanh()
		)
		
		self.cfe_model2 = nn.Sequential(
			nn.GroupNorm(1, 1),
			nn.Conv2d(1, 16, 4, stride=2, padding=0),
			nn.GroupNorm(1, 16), 
			nn.LeakyReLU(0.2, inplace=True),
			nn.Conv2d(16, 8, 3, stride=1, padding=0),
			nn.GroupNorm(1, 8), 
			nn.LeakyReLU(0.2, inplace=True),
			nn.Conv2d(8, 1, 11, stride=1, padding=0),
			nn.Tanh()
		)
		
		self.cfe_model3 = nn.Sequential(
			nn.GroupNorm(1, 1),
			nn.Conv2d(1, 16, 4, stride=2, padding=0),
			nn.GroupNorm(1, 16), 
			nn.LeakyReLU(0.2, inplace=True),
			nn.Conv2d(16, 8, 3, stride=1, padding=0),
			nn.GroupNorm(1, 8), 
			nn.LeakyReLU(0.2, inplace=True),
			nn.Conv2d(8, 1, 11, stride=1, padding=0),
			nn.Tanh()
		)
		
		self.cfe_model4 = nn.Sequential(
			nn.GroupNorm(1, 1),
			nn.Conv2d(1, 16, 4, stride=2, padding=0),
			nn.GroupNorm(1, 16), 
			nn.LeakyReLU(0.2, inplace=True),
			nn.Conv2d(16, 8, 3, stride=1, padding=0),
			nn.GroupNorm(1, 8), 
			nn.LeakyReLU(0.2, inplace=True),
			nn.Conv2d(8, 1, 11, stride=1, padding=0),
			nn.Tanh()
		)
		
		self.lin_pred_model = nn.Sequential(
			nn.LayerNorm(784),
			cLinear(784, 784),
			nn.Sigmoid(),
		)
	
	def test_forward(self, img):
		img = self.lin_pred_model(img)
		self.img = img
		
		img = img.reshape(img.shape[0], 1, *img_size)
		
		preds = self.cfe_model1(img)
		return preds
		
	def forward(self, img):
		img = self.lin_pred_model(img)
		self.img = img
		
		img = img.reshape(img.shape[0], 1, *img_size)
		
		preds1 = self.cfe_model1(img)
		preds2 = self.cfe_model2(img)
		preds3 = self.cfe_model3(img)
		preds4 = self.cfe_model4(img)
		
		preds = torch.cat((preds1, preds2, preds3, preds4))
		preds = preds.reshape(preds.shape[0], 1)
		#print(preds.shape); quit()

		return preds


# Initialize generator and discriminator
classifier = Classifier()

dogs = torch.tensor(np.load("d_gs.npy")).float().reshape(12500, img_shape)[:n_imgs] #+ 1
dogs = normalize_pattern(dogs) #torch.cat((normalize_pattern(dogs), dogs))

cats = torch.tensor(np.load("c_gs.npy")).float().reshape(12500, img_shape)[:n_imgs] #+ 1
cats = normalize_pattern(cats) #torch.cat((normalize_pattern(cats), cats))

idc = DataContainer2([dogs, cats])

dogs_labels = torch.ones((len(dogs),1))
cats_labels = -1*torch.ones((len(cats),1))

ldc = DataContainer2([dogs_labels, cats_labels])

# Optimizers
optim_c = torch.optim.Adam(classifier.parameters(), lr = 0.0001)

loss_func = nn.MSELoss()

tdogs = torch.tensor(np.load("d_gs.npy")).float().reshape(12500, img_shape)[n_imgs:n_imgs+100] #+ 1
tdogs = normalize_pattern(tdogs)

tcats = torch.tensor(np.load("c_gs.npy")).float().reshape(12500, img_shape)[n_imgs:n_imgs+100] #+ 1
tcats = normalize_pattern(tcats)

test_samples = torch.cat((tdogs, tcats))

# ----------
#  Training
# ----------

htsa = 0
for epoch in range(100000000):
	dog_, cat_ = idc.get_data(64)
	timgs = torch.cat((dog_, cat_)) 
	
	dl, cl = ldc.get_data(64) 
	
	#tlabels = torch.cat((dl, cl))
	
	optim_c.zero_grad()
	
	preds = classifier(timgs)

	dogs_labels = torch.ones((len(preds)//8,1))
	cats_labels = -1*torch.ones((len(preds)//8,1))
	tlabels = torch.cat((dogs_labels, cats_labels, dogs_labels, cats_labels, dogs_labels, cats_labels, dogs_labels, cats_labels))

	loss = loss_func(preds, tlabels)
	loss.backward()
	
	optim_c.step()

	
	with torch.no_grad():
		tpreds = classifier.test_forward(test_samples)
		
	tst_acc = get_acc(tpreds)
	if tst_acc > htsa:
		htsa = tst_acc
	
	if epoch % 500 == 0:
		imgs_preds = classifier.test_forward(torch.cat((dog_, cat_)))
		ta = get_acc(imgs_preds)
		Printer(f"{epoch = }, {loss.item() = }, {ta = }, {tst_acc = }, {htsa = }")
		
		save_image(dog_[:25].reshape(dog_[:25].shape[0], 1, *img_size), "images/dog1.png", nrow=5, normalize=True)
		save_image(cat_[:25].reshape(cat_[:25].shape[0], 1, *img_size), "images/cat.png", nrow=5, normalize=True)
		
		#save_image(classifier.features[0].unsqueeze(1), "images/features.png", nrow=5, normalize=True)

		save_image(classifier.img.reshape(classifier.img.shape[0],1,28,28), "images/attn_img.png", nrow=5, normalize=True)

		#print(tpreds)


