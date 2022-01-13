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
		
		#self.attn_l = aLinear(img_size, 1, ks = (20,20), axis = (4,4))
		
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
			nn.Conv2d(4, 1, 11, stride=1, padding=0),
			nn.Tanh()
		)
		
	def forward(self, img):
		#self.attn_img = self.attn_l(img)
		img = img.reshape(img.shape[0], 1, *img_size)
		self.features = self.cfe_model(img)

		preds = self.conv_pred_model(self.features)
		preds = preds.reshape(preds.shape[0],1)
		
		#print(preds.shape); quit()
		
		return preds


# Initialize generator and discriminator
classifier = Classifier()

dogs = torch.tensor(np.load("d_gs.npy")).float().reshape(12500, img_shape)[:n_imgs] #+ 1
cats = torch.tensor(np.load("c_gs.npy")).float().reshape(12500, img_shape)[:n_imgs] #+ 1

idc = DataContainer2([dogs, cats])

dogs_labels = torch.ones((len(dogs),1))
cats_labels = -1*torch.ones((len(cats),1))

ldc = DataContainer2([dogs_labels, cats_labels])

# Optimizers
optim_c = torch.optim.Adam(classifier.parameters(), lr = 0.0001)

loss_func = nn.MSELoss()

tdogs = torch.tensor(np.load("d_gs.npy")).float().reshape(12500, img_shape)[n_imgs:n_imgs+100] #+ 1
tcats = torch.tensor(np.load("c_gs.npy")).float().reshape(12500, img_shape)[n_imgs:n_imgs+100] #+ 1

test_samples = torch.cat((tdogs, tcats))

# ----------
#  Training
# ----------

htsa = 0
for epoch in range(100000000):
	dog_, cat_ = idc.get_data(64)
	timgs = torch.cat((dog_, cat_)) 
	
	dl, cl = ldc.get_data(64) 
	tlabels = torch.cat((dl, cl))
	
	optim_c.zero_grad()
	
	preds = classifier(timgs)

	loss = loss_func(preds, tlabels)
	loss.backward()
	
	optim_c.step()

	
	with torch.no_grad():
		tpreds = classifier(test_samples)
		
	tst_acc = get_acc(tpreds)
	if tst_acc > htsa:
		htsa = tst_acc
	
	if epoch % 500 == 0:
		imgs_preds = classifier(torch.cat((dog_, cat_)))
		ta = get_acc(imgs_preds)
		Printer(f"{epoch = }, {loss.item() = }, {ta = }, {tst_acc = }, {htsa = }")
		
		save_image(dog_[:25].reshape(dog_[:25].shape[0], 1, *img_size), "images/dog1.png", nrow=5, normalize=True)
		save_image(cat_[:25].reshape(cat_[:25].shape[0], 1, *img_size), "images/cat.png", nrow=5, normalize=True)
		
		save_image(classifier.features[0].unsqueeze(1), "images/features.png", nrow=5, normalize=True)

		#save_image(classifier.attn_img.reshape(classifier.attn_img.shape[0],1,28,28), "images/attn_img.png", nrow=5, normalize=True)




