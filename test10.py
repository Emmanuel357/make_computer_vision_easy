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
		
		self.csm = nn.CosineSimilarity(dim=1)
	
	def test_forward(self, img):
		pred_vector = self.lin_pred_model(img)
		
		self.cs = nn.CosineSimilarity(dim=1)(pred_vector[:len(pred_vector)//2], pred_vector[len(pred_vector)//2:])
		
		preds = self.predictor(pred_vector)
		preds = torch.sum(preds, 1).unsqueeze(1)
		return preds
		
	def forward(self, img):
		
		pred_vector = self.lin_pred_model(img)
		
		dogs = pred_vector[:len(pred_vector)//2]
		cats = pred_vector[len(pred_vector)//2:]
		
		csd = self.csm(dogs, cats).unsqueeze(1)
		
		css1 = self.csm(dogs[:len(dogs)//2], dogs[len(dogs)//2:])
		css2 = self.csm(cats[:len(cats)//2], cats[len(cats)//2:])
		css = torch.cat([css1, css2]).unsqueeze(1)
		
		return css, csd


# Initialize generator and discriminator
classifier = Classifier()

dogs = torch.tensor(np.load("d_gs64.npy")).float().reshape(12500, img_shape)[:n_imgs] + 1
cats = torch.tensor(np.load("c_gs64.npy")).float().reshape(12500, img_shape)[:n_imgs] + 1

idc = DataContainer2([dogs, cats])

# Optimizers
optim_c = torch.optim.Adam(classifier.parameters(), lr = 0.0001)

loss_func = nn.MSELoss()

tdogs = torch.tensor(np.load("d_gs64.npy")).float().reshape(12500, img_shape)[n_imgs:n_imgs+100] + 1
tcats = torch.tensor(np.load("c_gs64.npy")).float().reshape(12500, img_shape)[n_imgs:n_imgs+100] + 1

test_samples = torch.cat((tdogs, tcats))

# ----------
#  Training
# ----------

for epoch in range(100000000):
	dog_, cat_ = idc.get_data(64, random = True)
	timgs = torch.cat((dog_, cat_)) 

	optim_c.zero_grad()
	
	css, csd = classifier(timgs)
	
	cs_loss = (loss_func(css, torch.ones((len(css),1))) + loss_func(csd, -1*torch.ones((len(csd),1))))/2
	cs_loss.backward()
	
	optim_c.step()
	
	if epoch % 500 == 0:
		with torch.no_grad():
			test_css, test_csd = classifier(test_samples)
		
		Printer(f"{epoch = },  {cs_loss.item() = }, {len(test_css[test_css>0]) = }, {len(test_csd[test_csd<0]) = }, {len(css[css>0]) = }, {len(csd[csd<0]) = }")
		
		
		save_image(dog_[:25].reshape(dog_[:25].shape[0], 1, *img_size), "images/dog1.png", nrow=5, normalize=True)
		save_image(cat_[:25].reshape(cat_[:25].shape[0], 1, *img_size), "images/cat.png", nrow=5, normalize=True)






