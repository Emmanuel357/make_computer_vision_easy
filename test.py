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
			nn.Linear(img_shape, img_shape, bias = False), #fLinear((28, 28), img_shape, ks = (13,13), axis = (5,6)),
			nn.Sigmoid(),
			
			#nn.LayerNorm(img_shape),
			#nn.Linear(img_shape, img_shape, bias = False),
			#nn.Sigmoid() 
		)
		
		self.pred_model = nn.Sequential(
			nn.LayerNorm(img_shape),
			nn.Linear(img_shape, 1, bias = False), #mono_linear(img_shape, 1), #
			nn.Sigmoid()
		)
		
	def forward(self, img):
		x = self.feature_extractor(img)
		preds = self.pred_model(x)
		return preds


# Initialize generator and discriminator
classifier = Classifier()

dogs = torch.tensor(np.load("d_gs.npy")).float().reshape(12500, img_shape)[:n_imgs] + 1
cats = torch.tensor(np.load("c_gs.npy")).float().reshape(12500, img_shape)[:n_imgs] + 1
dimgs = torch.tensor(np.load("non_cd_imgs.npy")).float().reshape(354565, img_shape) + 1 

train_imgs = torch.cat((dogs, cats, dimgs))
tidc = DataContainer(train_imgs)

dogs_labels = torch.ones((len(dogs),1))
cats_labels = torch.zeros((len(cats),1))
dimgs_labels = 0.5*torch.ones((len(dimgs),1))
t_labels = torch.cat((dogs_labels, cats_labels, dimgs_labels))

ldc = DataContainer(t_labels)

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

for r in range(100000000):
	for epoch in range(5928):
		timgs = tidc.get_data(64)
		tlabels = ldc.get_data(64)
		
		optim_c.zero_grad()
		
		imgs_preds = classifier(timgs)

		loss = loss_func(imgs_preds, tlabels)
		loss.backward()
		
		optim_c.step()

		
		with torch.no_grad():
			tpreds = classifier(test_samples)
			
		tst_acc = get_acc(tpreds)
		if tst_acc > htsa:
			htsa = tst_acc
		
		if epoch % 500 == 0:
			imgs_preds = classifier(torch.cat((dogs[:64], cats[:64])))
			ta = get_acc(imgs_preds)
			Printer(f"{r = }, {epoch = }, {loss.item() = }, {ta = }, {tst_acc = }, {htsa = }, {tidc.offset}")

			save_image(timgs[:25].reshape(timgs[:25].shape[0], 1, *img_size), "images/timgs.png", nrow=5, normalize=True)
			#save_image(cat_[:25].reshape(cat_[:25].shape[0], 1, *img_size), "images/cat.png", nrow=5, normalize=True)
			#save_image(dimgs_[:25].reshape(dimgs_[:25].shape[0], 1, *img_size), "images/dimgs.png", nrow=5, normalize=True)













