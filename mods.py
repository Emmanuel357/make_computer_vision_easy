import sys
import torch
from torch import nn
import torch.nn.functional as F
import math
import numpy as np
import itertools
from torchvision.utils import save_image

class Printer():
    """Print things to stdout on one line dynamically"""
    def __init__(self,data):
        sys.stdout.write("\r\x1b[K"+data.__str__())
        sys.stdout.flush()
        
def get_acc(preds, th = 0):
	dp = preds[:len(preds)//2]
	da = len(dp[dp > th])
	
	cp = preds[len(preds)//2:]
	ca = len(cp[cp < th])
	
	acc = da + ca
	
	return (acc * 100) / len(preds)

def get_acc2(preds):
	preds = preds[:,:2]
	
	ht = preds[:len(preds)//2]
	_, indices = torch.max(ht, 1)
	
	da = len(indices[indices == 0])
	
	ht = preds[len(preds)//2:]
	_, indices = torch.max(ht, 1)
	
	ca = len(indices[indices == 1])
	
	acc = da + ca
	return (acc * 100) / len(preds)

class DataContainer2:
	def __init__(self, datasets):
		self.datasets = datasets
		self.offset = 0
		
	def get_data(self, bs, random = False):
		if not random:
			cds = []
			for i in range(len(self.datasets)):
				cd = self.datasets[i][self.offset:self.offset+bs]
				cds.append(cd)
				
			if self.offset + bs >= len(self.datasets[0]):
				self.offset = 0
			
			else:
				self.offset += bs
		
		elif random:
			cds = []
			ci = np.random.choice(len(self.datasets[0]), bs)
			for i in range(len(self.datasets)):
				cd = self.datasets[i][ci]
				cds.append(cd)
				
		return cds
		
class DataContainer:
	def __init__(self, dataset):
		self.dataset = dataset
		self.offset = 0
		
	def get_data(self, bs, random = False):
		if not random:
			cd = self.dataset[self.offset:self.offset+bs]
	
			if self.offset + bs >= len(self.dataset):
				self.offset = 0
			
			else:
				self.offset += bs
		
		elif random:
			ci = np.random.choice(len(self.dataset), bs)
			cd = self.dataset[ci]
		
		return cd

def place_ones(size, count):
	for positions in itertools.combinations(range(size), count):
		p = [0] * size

		for i in positions:
			p[i] = 1

		yield p
        
class GenDataContainer:
	def __init__(self, dataset, z_dim, code = 1, nm = 5):
		self.dataset = dataset
		self.offset = 0
		
		zs = []
		
		for i in range(1, z_dim+1):
			zs += list(place_ones(z_dim, i))
		
		self.zs = torch.tensor(zs).float()
		
		if len(self.zs) > len(self.dataset):
			nri = len(self.zs) - len(self.dataset)
			adtn_imgs = []
			
			if code == 1: # generate additional data by merging data in dataset
				for i in range(nri):
					ci = np.random.choice(len(self.dataset), nm)
					new_img = torch.sum(self.dataset[ci], 0)/nm
					adtn_imgs.append(new_img.unsqueeze(0))
				
				adtn_imgs = torch.cat(adtn_imgs)
				self.dataset = torch.cat((self.dataset, adtn_imgs))
				
			elif code == 2: # get additional data by duplicating data in dataset
				ci = np.random.choice(len(self.dataset), nri)
				self.dataset = torch.cat((self.dataset, self.dataset[ci]))
				
		print(f"done preparing data. length of zs: {len(self.zs)}, length of dataset: {len(self.dataset)}")
			
	def get_data(self, bs):
		czs = self.zs[self.offset:self.offset+bs]
		cd = self.dataset[self.offset:self.offset+bs]
	
		if self.offset + bs >= len(self.dataset):
			self.offset = 0
		
		else:
			self.offset += bs
		
		return czs, cd

class mono_linear(nn.Module):
	def __init__(self, in_features, n_weights, last_sum = True):
		super(mono_linear, self).__init__()
		
		self.in_features = in_features
		self.n_weights = n_weights
		self.last_sum = last_sum
		self.weight = nn.Parameter(torch.tensor(np.random.uniform(0, 1, (n_weights))).float())
		self.exp_weight = self.weight.expand(self.in_features, self.n_weights).T

	def forward(self, img):
		out = F.linear(img, self.exp_weight)
		if self.last_sum:
			out = torch.sum(out, 1).unsqueeze(1)
		
		return out

class cLinear(nn.Module):
	def __init__(self, in_features, out_features):
		super(cLinear, self).__init__()
		
		self.in_features = in_features
		self.out_features = out_features
		
		self.weight = 0.05*torch.ones((out_features, in_features)).float()
		ind = np.diag_indices(self.weight.shape[0])
		self.weight[ind[0], ind[1]] = torch.ones(self.weight.shape[0])
		self.weight = nn.Parameter(self.weight)
		
		#stdv = 1. / math.sqrt(self.weight.size(1))
		#self.weight.data.uniform_(-stdv, stdv)

	def forward(self, img):
		return F.linear(img, self.weight)

class gLinear(nn.Module):
	def __init__(self, in_features, out_features):
		super(gLinear, self).__init__()
		
		self.in_features = in_features
		self.out_features = out_features
		
		dogs = torch.tensor(np.load("d_gs.npy")).float().reshape(12500, out_features)[:in_features]
		
		dogs = dogs.T
		
		self.weight = dogs

		self.weight = nn.Parameter(self.weight)
		
		#stdv = 1. / math.sqrt(self.weight.size(1))
		#self.weight.data.uniform_(-stdv, stdv)

	def forward(self, img):
		return F.linear(img, self.weight)
		

class nSigmoid(nn.Module):
	def __init__(self, in_features):
		super(cSigmoid, self).__init__()
		
		self.sigmoid = nn.Sigmoid()
		self.p1 = nn.Parameter(torch.ones(in_features).float())
		self.p2 = nn.Parameter(0.4*torch.ones(in_features).float())

	def forward(self, x):
		
		x = self.p2 + (x * self.p1)
		
		x = self.sigmoid(x)
		
		return x
		
class cSigmoid(nn.Module):
	def __init__(self, in_features):
		super(cSigmoid, self).__init__()
		
		self.sigmoid = nn.Sigmoid()
		self.p1 = 1 #nn.Parameter(torch.ones(in_features).float())
		self.p2 = 0 #nn.Parameter(0.4*torch.ones(in_features).float())
		self.p3 = 1 #nn.Parameter(torch.ones(in_features).float())
		self.p4 = 1 #nn.Parameter(0.01*torch.ones(in_features).float())

	def forward(self, x):
		
		x = self.p2 + (x * self.p1)
		
		x = self.p3*self.sigmoid(x) + self.p4
		
		return x

class sigmoid(nn.Module):
	def __init__(self, k = 1):
		super(sigmoid, self).__init__()
		
		self.k = k

	def forward(self, x):
		x = self.k/(1 + torch.exp(-1*x))
		return x


class fLinear(nn.Module):
	def __init__(self, in_size, out_features, ks = None, axis = (0,0)):
		super(fLinear, self).__init__()
		
		self.in_size = in_size
		self.out_features = out_features
		
		self.weight = torch.zeros((out_features, *in_size)).float()
		
		if ks != None:
			stdv = 1.0/math.sqrt((self.weight.size(1)))
			self.weight[:,axis[1]:axis[1]+ks[1], axis[0]:axis[0]+ks[0]] = torch.tensor(np.random.uniform(-stdv,stdv,ks)).float()
			self.weight = nn.Parameter(self.weight.reshape(out_features, np.product(in_size)))
			
		save_image(torch.abs(self.weight[0].clone()).reshape(1, 1, *in_size), "images/vs_img.png", nrow=5, normalize=True)
		
		
	def forward(self, img):
		return F.linear(img, self.weight)

class aLinear(nn.Module):
	def __init__(self, in_size, out_features, ks = None, axis = (0,0)):
		super(aLinear, self).__init__()
		
		self.in_size = in_size
		self.out_features = out_features
		
		self.weight = 0.5*torch.ones((out_features, *in_size)).float()
		
		if ks != None:
			self.weight[:,axis[1]:axis[1]+ks[1], axis[0]:axis[0]+ks[0]] = 1
			self.weight = self.weight.reshape(out_features, np.product(in_size)) ##nn.Parameter()
		
		#print(self.weight[0])
		save_image(self.weight[0].reshape(1, 1, *in_size), "images/vs_img.png", nrow=5, normalize=True)
		#quit()
		
	def forward(self, img):
		
		return img*self.weight

def normalize_pattern(arrays):
	new_arrays = []
	
	for array in arrays:
		sarray = np.sort(array)
		
		sorter = np.argsort(sarray)
		oarray = sorter[np.searchsorted(sarray, array, sorter=sorter)] / 783
		
		new_arrays.append(oarray)
	
	new_arrays = torch.tensor(new_arrays).float()
	
	save_image(arrays[:25].reshape(25,1,28,28), "images/array.png", nrow=5, normalize=True)
	save_image(new_arrays[:25].reshape(25,1,28,28), "images/oarray.png", nrow=5, normalize=True) #; quit()

	return new_arrays
	
	


	
