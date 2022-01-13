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
		
		self.feature_extractor = nn.Sequential(
			nn.LayerNorm(img_shape),
			nn.Linear(img_shape, img_shape, bias = False),
			nn.Sigmoid(),
			
		)
		
		self.pred_model = nn.Sequential(
			nn.LayerNorm(img_shape),
			nn.Linear(img_shape, 1, bias = False),
			#sigmoid(1)
		)
		
	def forward(self, img):
		x = self.feature_extractor(img)
		preds = self.pred_model(x)
		return preds


# Initialize generator and discriminator
classifier = Classifier()

dogs = torch.tensor(np.load("d_gs.npy")).float().reshape(12500, img_shape)[:n_imgs] + 1
cats = torch.tensor(np.load("c_gs.npy")).float().reshape(12500, img_shape)[:n_imgs] + 1
cars = torch.tensor(np.load("non_cd_imgs.npy")).float().reshape(354565, img_shape)[:n_imgs] + 1
faces = torch.tensor(np.load("hf_gs.npy")).float().reshape(202599, img_shape)[:n_imgs] + 1

idc = DataContainer2([dogs, cats, cars, faces])

dogs_labels = torch.ones((len(dogs),1))
cats_labels = torch.zeros((len(cats),1))
cars_labels = 0.4*torch.ones((len(cars),1))
faces_labels = 0.6*torch.ones((len(faces),1))

ldc = DataContainer2([dogs_labels, cats_labels, cars_labels, faces_labels])

# Optimizers
optim_c = torch.optim.Adam(classifier.parameters(), lr = 0.0001)

loss_func = nn.MSELoss()

tdogs = torch.tensor(np.load("d_gs.npy")).float().reshape(12500, img_shape)[n_imgs:n_imgs+100] + 1
tcats = torch.tensor(np.load("c_gs.npy")).float().reshape(12500, img_shape)[n_imgs:n_imgs+100] + 1

test_samples = torch.cat((tdogs, tcats))

# ----------
#  Training
# ----------

htsa = 0
for epoch in range(100000000):
	dog_, cat_, cars_, faces_ = idc.get_data(64)
	timgs = torch.cat((dog_, cat_, cars_, faces_))
	
	dl, cl, cal, fl = ldc.get_data(64)
	tlabels = torch.cat((dl, cl, cal, fl))
	
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
	
	if epoch % 100 == 0:
		imgs_preds = classifier(torch.cat((dog_, cat_)))
		ta = get_acc(imgs_preds)
		Printer(f"{epoch = }, {loss.item() = }, {ta = }, {tst_acc = }, {htsa = }")

		save_image(dog_[:25].reshape(dog_[:25].shape[0], 1, *img_size), "images/dog.png", nrow=5, normalize=True)
		save_image(cat_[:25].reshape(cat_[:25].shape[0], 1, *img_size), "images/cat.png", nrow=5, normalize=True)
		save_image(cars_[:25].reshape(cars_[:25].shape[0], 1, *img_size), "images/dimgs.png", nrow=5, normalize=True)
		save_image(faces_[:25].reshape(faces_[:25].shape[0], 1, *img_size), "images/faces.png", nrow=5, normalize=True)



