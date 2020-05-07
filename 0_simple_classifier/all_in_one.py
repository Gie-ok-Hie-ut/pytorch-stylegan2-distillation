# This code is implemented based on the codes below
#
# Reference 1: https://pytorch.org/tutorials/beginner/blitz/cifar10_tutorial.html
# Reference 2: https://pytorch.org/tutorials/beginner/finetuning_torchvision_models_tutorial.html
# Reference 3: https://gist.github.com/santi-pdp/d0e9002afe74db04aa5bbff6d076e8fe

import os
import numpy as np
import argparse
import torch
from torch.utils.data import DataLoader
from dataloader import IMDBWIKIDataset
from torchvision import transforms, models, utils
import torch.nn as nn
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm
import scipy.io
import matplotlib.pyplot as plt
from scipy.io import loadmat
from datetime import datetime


class AllInOneTrainer(object):
	def __init__(self,args):
		self.lr = args.lr
		self.epoch = args.epoch
		self.db_root = args.db_root
		self.batch_size = 8
		self.attr = args.attribute

		# Network
		if self.attr == "age":	 
			# 0: young(~25) 1: middle(30~40) 2:old(50~)
			self.num_classes = 3
		elif self.attr == "gender": 
			# 0: Female 1: Male
			self.num_classes = 2

		self.net = models.vgg16(pretrained=True).cuda()
		self.net.fc = nn.Linear(1000, self.num_classes).cuda() # 4096
		self.requires_grad(self.net, True)
		self.phase = args.phase

		print("Network Architecture...")
		print(self.net)

		# Dataset # Dataloader
		if self.phase == 'train':
			self.train_dataset = IMDBWIKIDataset(os.path.join(self.db_root), self.phase, self.attr)
			self.train_loader = DataLoader(dataset=self.train_dataset, batch_size=self.batch_size, shuffle=True)
		elif self.phase == 'test':
			self.test_dataset = IMDBWIKIDataset(os.path.join(self.db_root), self.phase, self.attr)
			self.test_loader = DataLoader(dataset=self.test_dataset, batch_size=1, shuffle=False)

		# Oths
		if self.num_classes == 2:
			self.criterion = torch.nn.BCELoss()
		else:
			self.criterion = torch.nn.CrossEntropyLoss()
		#self.criterion = torch.nn.BCEWithLogitsLoss() # Sigmoid + BCE Loss // Sigmiod (0~1)

		self.optimizer_all = torch.optim.Adam(self.net.parameters(),lr= self.lr, betas = (0.9, 0.999))
		self.optimizer_fc = torch.optim.Adam(self.net.fc.parameters(), lr=self.lr, betas = (0.9, 0.999))
		self.writer = SummaryWriter('./loss')

	def requires_grad(self, model, flag=True):
		for p in model.parameters():
			p.requires_grad = flag



	def train(self):
		# =================================================================================== #
		#                                   Train Classifier                                  #
		# =================================================================================== #

		for epoch in range(0, self.epoch):
			print("Epoch={}".format(epoch))

			if epoch<0:
				print("Train fc only")
				self.requires_grad(self.net, False)
				self.requires_grad(self.net.fc,True)
			else:
				print("Train whole net")
				self.requires_grad(self.net,True)

			for step,data in enumerate(tqdm(self.train_loader), 0):

				train_x, train_y = data

				# img
				y_hat1 = self.net.forward(train_x.cuda())
				y_hat2 = self.net.fc.forward(y_hat1)
				sm = nn.Softmax(dim=1)
				y_hat3 = sm(y_hat2)

				# label
				if self.attr == 'gender':
					gt = torch.FloatTensor(torch.zeros(train_y.size(0),2)).cuda()
					train_y0 = (train_y==0).nonzero()
					train_y1 = (train_y==1).nonzero()
					gt[train_y0] = torch.FloatTensor([1,0]).cuda()
					gt[train_y1] = torch.FloatTensor([0,1]).cuda()
				elif self.attr == 'age':
					gt = train_y.long().cuda().squeeze(1)

				# loss
				loss = self.criterion(y_hat3, gt)

				# Backprob
				if epoch<2:
					self.optimizer_fc.zero_grad()
					loss.backward()
					self.optimizer_fc.step()
				else:
					self.optimizer_all.zero_grad()
					loss.backward()
					self.optimizer_all.step()

				# Visualizer
				if step % 100 == 0 and step != 0:
					print(loss.data)
					self.writer.add_scalar('Loss/train',loss.data, step)

				if step % 5000 == 0 and step != 0:
					torch.save(self.net, "./checkpoint/ckpt_" + self.attr + '_' + str(epoch) + '.pt')
					print("Saved model...")

			# Saving the model
			torch.save(self.net, "./checkpoint/ckpt_" + self.attr + '_final.pt')
			print("Saved model...")


	def test(self):
		# =================================================================================== #
		#                                    Test Classifier                                  #
		# =================================================================================== #
		self.requires_grad(self.net, False)
		self.net = torch.load("./checkpoint/ckpt_"+self.attr+"_final.pt").eval()
		conf_low, conf_60, conf_70, conf_80, conf_90 = 0, 0, 0, 0, 0
		gender0, gender1, gender0_hit, gender1_hit = 0, 0, 0, 0
		total_hit = 0

		print("Test Model")

		class_num = []
		for i in range(0,self.num_classes):
			class_num.append(0)

		arng = np.arange(20)/20
		conf_all = []

		for step, data in enumerate(tqdm(self.test_loader), 0):
			test_x, test_y = data

			# img
			y_hat1 = self.net.forward(test_x.cuda())
			y_hat2 = self.net.fc.forward(y_hat1)
			sm = nn.Softmax(dim=1)
			y_hat3 = sm(y_hat2)

			y_final = np.argmax(y_hat3.cpu().data)

			if test_y[0].long() == 0 :
				class_num[0] +=1
			elif test_y[0].long() == 1:
				class_num[1] +=1
			else:
				class_num[2] += 1


			if test_y[0].long() == y_final.long():
				total_hit += 1
				conf = y_hat3[0][test_y[0].long()]

				if conf <0.6:
					conf_low +=1
				elif conf >= 0.6 and conf <0.7:
					conf_60 +=1
				elif conf >=0.7 and conf <0.8:
					conf_70 +=1
				elif conf >=0.8 and conf <0.9:
					conf_80 +=1
				elif conf>=0.9 and conf <=1.0:
					conf_90 +=1


		print("Checkpoint={}".format(checkpoint_name))
		print("Accuracy_Final={}".format(total_hit / len(self.test_dataset)))
		print("conf_low={}".format(conf_low))
		print("conf_60={}".format(conf_60))
		print("conf_70={}".format(conf_70))
		print("conf_80={}".format(conf_80))
		print("conf_90={}".format(conf_90))
		for i in range(0,self.num_classes):
			print("class{}={}".format(i,class_num[i]))


def main():
	parser = argparse.ArgumentParser(description='Simple Classification')

	parser.add_argument('--phase', type=str, default='test', choices=['train', 'test', 'classify', ], help='mode', )
	parser.add_argument('--attribute', type=str, default='age', choices=['gender', 'age', ],
						help='mode', )
	parser.add_argument('--db_root', type=str, default='./imdb_dataset', help='mode', )
	parser.add_argument('--size', default=256, type=int, help='initial image size')
	parser.add_argument('--lr', default=0.00001, type=float, help='learning rate')
	parser.add_argument('--epoch', default=30, type=int, help='the number of epoch')
	args = parser.parse_args()

	all_in_one_trainer = AllInOneTrainer(args)

	if args.phase == "train":
		print("[Train Phase]")
		all_in_one_trainer.train()
	elif args.phase == "test":
		print("[Test Phase]")
		all_in_one_trainer.test()
	return

if __name__ == '__main__':
    main()