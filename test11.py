import os
import numpy as np

from torchvision.utils import save_image

import torch.nn as nn
import torch
from mods import *

n_imgs = 12400
img_shape = 64*64
img_size = (64, 64)

class Classifier(nn.Module):
	def __init__(self):
		super(Classifier, self).__init__()
		
		self.lin_pred_model = nn.Sequential(
			nn.LayerNorm(64*64),
			nn.Linear(64*64, 12*12, bias = False),
			nn.LeakyReLU(0.2, inplace=True),
		
			nn.LayerNorm(12*12),
			nn.Linear(12*12, 12*12, bias = False),
			nn.LeakyReLU(0.2, inplace=True)
		)
		
		
		self.predictor = nn.Sequential(
			nn.LayerNorm(12*12),
			nn.Linear(12*12, 1, bias = False),
			nn.Tanh()
		)
		
	def forward(self, img):
		
		pred_vector = self.lin_pred_model(img)
		
		class1_v = pred_vector[:len(pred_vector)//2]
		class2_v = pred_vector[len(pred_vector)//2:]
		
		return class1_v, class2_v

csm = nn.CosineSimilarity(dim=1)

# Initialize generator and discriminator
classifier1 = Classifier()
classifier2 = Classifier()
classifier3 = Classifier()

dogs = torch.tensor(np.load("d_gs64.npy")).float().reshape(12500, img_shape)[:n_imgs] + 1
cats = torch.tensor(np.load("c_gs64.npy")).float().reshape(12500, img_shape)[:n_imgs] + 1
cars = torch.tensor(np.load("hf64_gs.npy")).float().reshape(16001, img_shape)[:n_imgs] + 1

idc1 = DataContainer2([dogs, cats])
idc2 = DataContainer2([cats, cars])
idc3 = DataContainer2([dogs, cars])

# Optimizers
optim_c1 = torch.optim.Adam(classifier1.parameters(), lr = 0.0001)
optim_c2 = torch.optim.Adam(classifier2.parameters(), lr = 0.0001)
optim_c3 = torch.optim.Adam(classifier3.parameters(), lr = 0.0001)

loss_func = nn.MSELoss()

tdogs = torch.tensor(np.load("d_gs64.npy")).float().reshape(12500, img_shape)[n_imgs:n_imgs+100] + 1
tcats = torch.tensor(np.load("c_gs64.npy")).float().reshape(12500, img_shape)[n_imgs:n_imgs+100] + 1

test_samples = torch.cat((tdogs, tcats))

# ----------
#  Training
# ----------

similar_label = torch.ones((64,1)).float()
difference_label = -1*torch.ones((64,1)).float()

for epoch in range(100000000):
	dog1_, cat1_ = idc1.get_data(64, random = True)
	timgs1 = torch.cat((dog1_, cat1_)) 
	
	cat2_, car1_ = idc2.get_data(64, random = True)
	timgs2 = torch.cat((cat2_, car1_)) 

	dog2_, car2_ = idc3.get_data(64, random = True)
	timgs3 = torch.cat((dog2_, car2_)) 
	
	optim_c1.zero_grad()
	optim_c2.zero_grad()
	optim_c3.zero_grad()
	
	dog1_v, cat1_v = classifier1(timgs1)
	cat2_v, car1_v = classifier2(timgs2)
	dog2_v, car2_v = classifier3(timgs3)
	
	pair1 = csm(dog1_v, cat1_v).unsqueeze(1) # -1
	pair2 = csm(cat1_v, cat2_v).unsqueeze(1) #  1
	pair3 = csm(cat2_v, car1_v).unsqueeze(1) # -1
	pair4 = csm(car1_v, car2_v).unsqueeze(1) #  1
	pair5 = csm(car2_v, dog2_v).unsqueeze(1) # -1
	pair6 = csm(dog1_v, dog2_v).unsqueeze(1) #  1
	
	loss1 = loss_func(pair1, difference_label)
	loss2 = loss_func(pair2, similar_label)
	loss3 = loss_func(pair3, difference_label)
	loss4 = loss_func(pair4, similar_label)
	loss5 = loss_func(pair5, difference_label)
	loss6 = loss_func(pair6, similar_label)
	
	loss = (loss1 + loss2 + loss3 + loss4 + loss5 + loss6)/6
	loss.backward()
	
	optim_c1.step()
	optim_c2.step()
	optim_c3.step()
	
	if epoch % 500 == 0:
		with torch.no_grad():
			dog_v, cat_v = classifier1(test_samples)
			
		test_csd = csm(dog_v, cat_v)
		
		Printer(f"{epoch = },  {loss.item() = }, {len(pair1[pair1<0]) = }, {len(pair2[pair2>0]) = }, {len(test_csd[test_csd<0]) = }")
		
		
		save_image(dog1_[:25].reshape(dog1_[:25].shape[0], 1, *img_size), "images/dog1.png", nrow=5, normalize=True)
		save_image(cat1_[:25].reshape(cat1_[:25].shape[0], 1, *img_size), "images/cat.png", nrow=5, normalize=True)






