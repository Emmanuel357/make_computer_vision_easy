import os
import numpy as np

from torchvision.utils import save_image

import torch.nn as nn
import torch
from mods import *

n_imgs = 1024
img_shape = 28*28
img_size = (28, 28)

ld = 11
class Generator(nn.Module):
	def __init__(self):
		super(Generator, self).__init__()

		self.model = nn.Sequential(
			nn.LayerNorm(ld),
			nn.Linear(ld, 1500),
			nn.Sigmoid(), 
			
			nn.LayerNorm(1500),
			nn.Linear(1500, 1500),
			nn.Sigmoid(),
			
			nn.LayerNorm(1500),
			nn.Linear(1500, img_shape),
			nn.Sigmoid()
		)
		
	def forward(self, ld):
		img = self.model(ld)
		return img

# Initialize generator and discriminator
generator = Generator()
dogs = torch.tensor(np.load("d_gs.npy")).float().reshape(12500, img_shape)[:n_imgs]
cats = torch.tensor(np.load("c_gs.npy")).float().reshape(12500, img_shape)[:n_imgs-1]

gc = GenDataContainer(torch.cat((dogs, cats)), ld, code = 2)

# Optimizers
optim_g = torch.optim.Adam(generator.parameters(), lr = 0.0001)

loss_func = nn.MSELoss()

# ----------
#  Training
# ----------

htsa = 0
for epoch in range(100000000):
	optim_g.zero_grad()
	
	zs, imgs = gc.get_data(64)
	
	recr_imgs = generator(zs)

	rloss = loss_func(recr_imgs, imgs)
	rloss.backward()
	
	optim_g.step()
	
	if epoch % 100 == 0:
		Printer(f"{epoch = }, {rloss.item() = }")


		save_image(imgs.reshape(imgs.shape[0], 1, *img_size), "images/real_imgs.png", nrow=5, normalize=True)
		
		save_image(recr_imgs.reshape(recr_imgs.shape[0],1,28,28), "images/recr_imgs.png", nrow=5, normalize=True)
		
		zs = torch.tensor(np.random.normal(0, 1, (25, ld))).float()
		gen_imgs = generator(zs)
		save_image(gen_imgs.reshape(25,1,28,28), "images/gen_imgs.png", nrow=5, normalize=True)

		zs = torch.tensor(np.random.randint(0, 2, (25, ld))).float()
		gen_imgs = generator(zs)
		save_image(gen_imgs.reshape(25,1,28,28), "images/gen_imgs2.png", nrow=5, normalize=True)




