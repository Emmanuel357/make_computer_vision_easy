import os
import numpy as np

from torchvision.utils import save_image

import torch.nn as nn
import torch
from mods import *

n_imgs = 12400
img_shape = 28*28
img_size = (28, 28)

ld = 144
class Attacker(nn.Module):
	def __init__(self):
		super(Attacker, self).__init__()
		
		self.lin_pred_model = nn.Sequential(
			nn.LayerNorm(ld),
			nn.Linear(ld, 512, bias = False),
			#nn.LeakyReLU(0.2, inplace=True)
			nn.Sigmoid(),
		)
		
		self.syn_model = nn.Sequential(
			nn.LayerNorm(512),
			gLinear(512, 784),
			#nn.Sigmoid()
		)
		
	def forward(self, z):

		img = self.lin_pred_model(z)
		img = self.syn_model(img)
		
		#preds = preds.reshape(preds.shape[0], 1)
		#print(preds.shape); quit()

		return img
		
class Classifier(nn.Module):
	def __init__(self):
		super(Classifier, self).__init__()
		
		self.cfe_model1 = nn.Sequential(
			nn.GroupNorm(1, 3),
			nn.Conv2d(3, 16, 4, stride=2, padding=0),
			nn.GroupNorm(1, 16), 
			nn.LeakyReLU(0.2, inplace=True),
			nn.Conv2d(16, 8, 3, stride=1, padding=0),
			nn.GroupNorm(1, 8), 
			nn.LeakyReLU(0.2, inplace=True),
			nn.Conv2d(8, 1, 8, stride=1, padding=0),
			nn.Tanh()
		)
		
		self.lin_pred_model = nn.Sequential(
			nn.LayerNorm(784),
			nn.Linear(784, 32, bias = False),
			nn.LeakyReLU(0.2, inplace=True),
			
			nn.LayerNorm(32),
			nn.Linear(32, 1, bias = False),
			nn.Tanh()
		)
		
	def forward(self, img):

		preds = self.lin_pred_model(img)
		
		#preds = preds.reshape(preds.shape[0], 1)
		#print(preds.shape); quit()

		return preds


# Initialize generator and discriminator
attacker = Attacker()
classifier = Classifier()

dogs = torch.tensor(np.load("d_gs.npy")).float().reshape(12500, img_shape)[:n_imgs] #+ 1
#dogs = normalize_pattern(dogs) #torch.cat((normalize_pattern(dogs), dogs)) #

cats = torch.tensor(np.load("c_gs.npy")).float().reshape(12500, img_shape)[:n_imgs] #+ 1
#cats = normalize_pattern(cats) #torch.cat((normalize_pattern(cats), cats)) #

idc = DataContainer2([dogs, cats])

# Optimizers
optim_a = torch.optim.Adam(attacker.parameters(), lr = 0.0001)
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
	
	'''
	Train Attacker
	'''
	optim_a.zero_grad()
	
	zs = torch.tensor(np.random.normal(0,1,(len(timgs), ld))).float()
	
	advs_img = attacker(zs)
	
	apreds = classifier(advs_img)
	
	dogs_labels = -1*torch.ones((len(apreds)//2,1))
	cats_labels = torch.ones((len(apreds)//2,1))
	alabels = torch.cat((dogs_labels, cats_labels))
	
	aloss = loss_func(apreds, alabels)
	aloss.backward()
	
	optim_a.step()
	
	'''
	Train Classifier
	'''
	
	optim_c.zero_grad()
	
	preds1 = classifier(advs_img.detach())
	preds2 = classifier(timgs)

	dogs_labels = torch.ones((len(preds2)//2,1))
	cats_labels = -1*torch.ones((len(preds2)//2,1))
	tlabels = torch.cat((dogs_labels, cats_labels))

	loss = (loss_func(preds1, 0.5*torch.ones((len(preds1),1))) + loss_func(preds2, tlabels) )/2
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
		Printer(f"{epoch = }, {loss.item() = }, {aloss.item() = }, {ta = }, {tst_acc = }, {htsa = }")
		
		save_image(dog_[:25].reshape(dog_[:25].shape[0], 1, *img_size), "images/dog1.png", nrow=5, normalize=True)
		save_image(cat_[:25].reshape(cat_[:25].shape[0], 1, *img_size), "images/cat.png", nrow=5, normalize=True)
		
		#save_image(classifier.features[0].unsqueeze(1), "images/features.png", nrow=5, normalize=True)

		save_image(advs_img[:25].reshape(advs_img[:25].shape[0],1,28,28), "images/advs_img.png", nrow=5, normalize=True)

		#print(tpreds)


