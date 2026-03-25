import torch
torch.backends.cudnn.benchmark=True
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.autograd import Variable
import numpy as np
from PIL import Image
from tqdm import tqdm
import time
import copy

import torchvision.models as models
import torchvision.transforms as transforms
from hybrid_hsi_lite import hybrid_hsi_lite

def MultiClassCrossEntropy(logits, labels, T):
	# Ld = -1/N * sum(N) sum(C) softmax(label) * log(softmax(logit))
	device = logits.device
	labels = Variable(labels.data, requires_grad=False).to(device)
	outputs = torch.log_softmax(logits/T, dim=1)   # compute the log of softmax values
	labels = torch.softmax(labels/T, dim=1)
	# print('outputs: ', outputs)
	# print('labels: ', labels.shape)
	outputs = torch.sum(outputs * labels, dim=1, keepdim=False)
	outputs = -torch.mean(outputs, dim=0, keepdim=False)
	# print('OUT: ', outputs)
	return Variable(outputs.data, requires_grad=True).to(device)

def kaiming_normal_init(m):
	if isinstance(m, nn.Conv2d):
		nn.init.kaiming_normal_(m.weight, nonlinearity='relu')
	elif isinstance(m, nn.Linear):
		nn.init.kaiming_normal_(m.weight, nonlinearity='sigmoid')

class Model(nn.Module):
	def __init__(self, classes, classes_map, args):
		# Hyper Parameters
		self.init_lr = args.init_lr
		self.num_epochs = args.num_epochs
		self.inc_epochs = int(getattr(args, "inc_epochs", self.num_epochs))
		self.batch_size = args.batch_size
		self.lower_rate_epoch = [int(0.7 * self.num_epochs), int(0.9 * self.num_epochs)] #hardcoded decay schedule
		self.lr_dec_factor = 10
		
		self.pretrained = False
		self.device = torch.device(getattr(args, "device", "cuda:0") if torch.cuda.is_available() and getattr(args, "device", "cuda:0") != "cpu" else "cpu")
		self.momentum = 0.9
		self.weight_decay = 0.0001
		# Constant to provide numerical stability while normalizing
		self.epsilon = 1e-16

		# Network architecture
		super(Model, self).__init__()
		self.backbone_name = getattr(args, "backbone_name", "resnet34")
		in_channels = int(getattr(args, "input_channels", 3))
		if self.backbone_name == "hybrid_hsi_lite":
			self.feature_extractor = hybrid_hsi_lite(
				pretrained=self.pretrained,
				mode="normal",
				in_channels=in_channels,
				feature_dim=512,
			)
			num_features = self.feature_extractor.feature_dim
		else:
			self.model = models.resnet34(pretrained=self.pretrained)
			if in_channels != 3:
				self.model.conv1 = nn.Conv2d(
					in_channels,
					64,
					kernel_size=7,
					stride=2,
					padding=3,
					bias=False,
				)
			self.model.apply(kaiming_normal_init)
			num_features = self.model.fc.in_features
			self.feature_extractor = nn.Sequential(*list(self.model.children())[:-1])
			if self.device.type == "cuda" and torch.cuda.device_count() > 1:
				self.feature_extractor = nn.DataParallel(self.feature_extractor)
		self.fc = nn.Linear(num_features, classes, bias=False)


		# n_classes is incremented before processing new data in an iteration
		# n_known is set to n_classes after all data for an iteration has been processed
		self.n_classes = int(classes)
		self.n_known = 0
		self.classes_map = classes_map

	def forward(self, x):
		x = self.feature_extractor(x)
		x = x.view(x.size(0), -1)
		x = self.fc(x)
		return x

	def increment_classes(self, new_classes):
		"""Add n classes in the final fc layer"""
		n = len(new_classes)
		print('new classes: ', n)
		in_features = self.fc.in_features
		out_features = self.fc.out_features
		weight = self.fc.weight.data

		if self.n_known == 0:
			new_out_features = n
		else:
			new_out_features = out_features + n
		print('new out features: ', new_out_features)
		self.fc = nn.Linear(in_features, new_out_features, bias=False)
		
		kaiming_normal_init(self.fc.weight)
		self.fc.weight.data[:out_features] = weight
		self.n_classes += n

	def classify(self, images):
		"""Classify images by softmax

		Args:
			x: input image batch
		Returns:
			preds: Tensor of size (batch_size,)
		"""
		_, preds = torch.max(torch.softmax(self.forward(images), dim=1), dim=1, keepdim=False)

		return preds

	def update(self, dataset, class_map, args):

		self.compute_means = True

		# Save a copy to compute distillation outputs
		prev_model = copy.deepcopy(self)
		prev_model.to(self.device)

		classes = list(set(dataset.train_labels))
		#print("Classes: ", classes)
		print('Known: ', self.n_known)
		if self.n_known == 0 and self.fc.out_features >= len(classes):
			new_classes = []
		elif self.n_classes == 1 and self.n_known == 0:
			new_classes = [classes[i] for i in range(1,len(classes))]
		else:
			new_classes = [cl for cl in classes if class_map[cl] >= self.n_known]

		if len(new_classes) > 0:
			self.increment_classes(new_classes)
			self.to(self.device)

		loader = torch.utils.data.DataLoader(dataset, batch_size=self.batch_size,
											   shuffle=True, num_workers=0)

		print("Batch Size (for n_classes classes) : ", len(dataset))
		freeze_on_increment = bool(getattr(args, "freeze_feature_extractor_on_increment", False))
		if self.n_known > 0 and freeze_on_increment:
			self.feature_extractor.eval()
			for param in self.feature_extractor.parameters():
				param.requires_grad = False
			for param in self.fc.parameters():
				param.requires_grad = True
			current_lr = float(getattr(args, "incremental_lr", self.init_lr))
			optimizer = optim.SGD(self.fc.parameters(), lr=current_lr, momentum=self.momentum, weight_decay=self.weight_decay)
			print("Incremental stage: feature extractor frozen, classifier-only update")
		else:
			for param in self.feature_extractor.parameters():
				param.requires_grad = True
			current_lr = self.init_lr
			optimizer = optim.SGD(self.parameters(), lr=current_lr, momentum = self.momentum, weight_decay=self.weight_decay)

		current_epochs = self.inc_epochs if (self.n_known > 0 and freeze_on_increment) else self.num_epochs
		with tqdm(total=current_epochs) as pbar:
			for epoch in range(current_epochs):
				epoch_loss = 0.0
				num_batches = 0
				
				# Modify learning rate
				# if (epoch+1) in lower_rate_epoch:
				# 	self.lr = self.lr * 1.0/lr_dec_factor
				# 	for param_group in optimizer.param_groups:
				# 		param_group['lr'] = self.lr

				
				for i, (indices, images, labels) in enumerate(loader):
					seen_labels = []
					images = Variable(torch.FloatTensor(images)).to(self.device)
					seen_labels = torch.LongTensor([class_map[label] for label in labels.numpy()])
					labels = Variable(seen_labels).to(self.device)
					# indices = indices.cuda()

					optimizer.zero_grad()
					logits = self.forward(images)
					cls_loss = nn.CrossEntropyLoss()(logits, labels)
					if len(new_classes) > 0 and self.n_classes // len(new_classes) > 1:
						dist_target = prev_model.forward(images)
						logits_dist = logits[:,:-(self.n_classes-self.n_known)]
						dist_loss = MultiClassCrossEntropy(logits_dist, dist_target, 2)
						loss = dist_loss+cls_loss
					else:
						loss = cls_loss

					loss.backward()
					optimizer.step()
					epoch_loss += float(loss.item())
					num_batches += 1

				avg_loss = epoch_loss / max(num_batches, 1)
				if (epoch + 1) == 1 or (epoch + 1) == current_epochs or (epoch + 1) % 10 == 0:
					tqdm.write(
						"Epoch [%d/%d] Avg Loss: %.4f"
						% (epoch + 1, current_epochs, avg_loss)
					)
				pbar.update(1)
