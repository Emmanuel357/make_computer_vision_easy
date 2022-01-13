import os
import numpy as np

import torch.nn as nn
import torch
from mods import *

class Regressor(nn.Module):
	def __init__(self):
		super(Regressor, self).__init__()
		
		self.l = nn.Sequential(
			nn.LayerNorm(2),
			
			nn.Linear(2, 2),
			nn.LeakyReLU(0.2, inplace=True),
			
			nn.LayerNorm(2),
			
			nn.Linear(2, 1),
			
			#nn.Tanh()
		)
		
	def forward(self, x):
		#nn.LayerNorm(2)(x)
		#print(x.shape)
		y = self.l(x) 
		#print(y)
		y = y ** torch.tensor([[1],[1],[2]]).float()
		#print(y); quit()
		return y

def c_loss(pred, target):
	loss = abs(target - pred)
	loss = torch.sum(loss)
	return loss
	
# Initialize generator and discriminator
regressor = Regressor()

# Optimizers
optim_r = torch.optim.Adam(regressor.parameters(), lr = 0.0001)

loss_func = nn.MSELoss()

inputs = torch.tensor([[0,1],[1,0],[1,1]]).float()#/6 #torch.tensor(np.random.choice(list(range(1,4)),(3,1), replace = False)).float()
outputs = torch.tensor([3, 5, 2]).float().unsqueeze(1)/7 #torch.tensor(np.random.choice(list(range(4,7)),(3,1), replace = False)).float()

print(inputs)
print(outputs)

for epoch in range(100000000):
	
	optim_r.zero_grad()
	
	preds = regressor(inputs)

	loss = c_loss(preds, outputs)
	loss.backward()
	
	optim_r.step()

	Printer(f"{epoch = }, {loss.item() = }, ")	
	
	
	
	
