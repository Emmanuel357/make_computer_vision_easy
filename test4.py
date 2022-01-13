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
		
		self.cfe_model = nn.Sequential(
			nn.GroupNorm(1, 1),
			nn.Conv2d(1, 16, 3, stride=1, padding=0),
			nn.GroupNorm(1, 16), 
			nn.LeakyReLU(0.2, inplace=True),
			nn.Conv2d(16, 1, 3, stride=2, padding=0),
			nn.GroupNorm(1, 1), 
			nn.LeakyReLU(0.2, inplace=True),
			
		)
		
		self.lin_pred_model = nn.Sequential(
			nn.LayerNorm(30*30),
			nn.Linear(30*30, 6*6, bias = False),
			nn.LeakyReLU(0.2, inplace=True)
		)
		
		
		self.predictor = nn.Sequential(
			nn.LayerNorm(6*6),
			nn.Linear(6*6, 1, bias = False),
			#nn.Tanh()
		)
			
			
	def forward(self, img):
		#print(len(img))
		img = img.reshape(img.shape[0], 1, *img_size)
		self.features = self.cfe_model(img)
		
		features = self.features.reshape(self.features.shape[0], self.features.shape[2]*self.features.shape[3]) #self.features.shape[1],
		
		pred_vector = self.lin_pred_model(features)
		
		self.cs = nn.CosineSimilarity(dim=1)(pred_vector[:len(pred_vector)//2], pred_vector[len(pred_vector)//2:])
		#print(len(pred_vector[:len(pred_vector)//2]), pred_vector[:len(pred_vector)//2].shape, self.cs.shape)
		
		preds = self.predictor(pred_vector)
		preds = torch.sum(preds, 1).unsqueeze(1)
		
		return preds


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

htsa = 0
for epoch in range(100000000):
	dog_, cat_ = idc.get_data(64)
	timgs = torch.cat((dog_, cat_)) 
	
	optim_c.zero_grad()
	
	preds = classifier(timgs)

	dogs_labels = 1*torch.ones((len(preds)//2,1))
	cats_labels = -1*torch.ones((len(preds)//2,1))
	tlabels = torch.cat((dogs_labels, cats_labels))

	loss = loss_func(preds, tlabels)
	loss.backward()
	
	optim_c.step()


	optim_c.zero_grad()
	
	classifier(timgs)

	#dogs_labels = 50*torch.ones((len(preds)//2,1))
	#cats_labels = -50*torch.ones((len(preds)//2,1))
	#tlabels = torch.cat((dogs_labels, cats_labels))
	
	
	cs_loss = loss_func(classifier.cs.unsqueeze(1), -1*torch.ones((len(classifier.cs),1)))
	cs_loss.backward()
	
	optim_c.step()
	
	tcs = classifier.cs.detach().numpy()
	
	with torch.no_grad():
		tpreds = classifier(test_samples)
		
	tst_acc = get_acc(tpreds)
	if tst_acc > htsa:
		htsa = tst_acc
	
	if epoch % 250 == 0:
		ta = get_acc(preds)
		
		Printer(f"{epoch = }, {loss.item() = }, {cs_loss.item() = }, {ta = }, {tst_acc = }, {htsa = }, {len(classifier.cs[classifier.cs<0]) = }, {len(tcs[tcs<0]) = }")
		
		
		save_image(dog_[:25].reshape(dog_[:25].shape[0], 1, *img_size), "images/dog1.png", nrow=5, normalize=True)
		save_image(cat_[:25].reshape(cat_[:25].shape[0], 1, *img_size), "images/cat.png", nrow=5, normalize=True)
		
		save_image(classifier.features[0].unsqueeze(1), "images/features.png", nrow=5, normalize=True)






