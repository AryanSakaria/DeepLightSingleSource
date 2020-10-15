import torch
import torch.nn as nn
import torch.nn.functional as F 

class DeepLightModel(nn.Module):

	def __init__(self):
		super(DeepLightModel,self).__init__()
		self.conv1 = nn.Conv2d(4, 64, 7, 2)

		self.conv1_up = nn.Conv2d(64, 256, 1)
		self.conv2_up = nn.Conv2d(256, 512, 1)
		self.conv3_up = nn.Conv2d(512, 1024, 1)
		self.conv4_up = nn.Conv2d(1024, 2048, 1)




		#green blocks
		self.conv1_1 = nn.Conv2d(64, 64, 1)
		self.conv1_2 = nn.Conv2d(64, 64, 3)
		self.conv1_3 = nn.Conv2d(64, 256, 1)

		self.conv2_1 = nn.Conv2d(256, 64, 1)
		self.conv2_2 = nn.Conv2d(64, 64, 3)
		self.conv2_3 = nn.Conv2d(64, 256, 1)


		self.conv3_1 = nn.Conv2d(256, 64, 1)
		self.conv3_2 = nn.Conv2d(64, 64, 3)
		self.conv3_3 = nn.Conv2d(64, 256, 1)

		#red blocks
		self.conv4_1 = nn.Conv2d(256, 128, 1)
		self.conv4_2 = nn.Conv2d(128, 128, 3)
		self.conv4_3 = nn.Conv2d(128, 512, 1)

		self.conv5_1 = nn.Conv2d(512, 128, 1)
		self.conv5_2 = nn.Conv2d(128, 128, 3)
		self.conv5_3 = nn.Conv2d(128, 512, 1)

		self.conv6_1 = nn.Conv2d(512, 128, 1)
		self.conv6_2 = nn.Conv2d(128, 128, 3)
		self.conv6_3 = nn.Conv2d(128, 512, 1)

		self.conv7_1 = nn.Conv2d(512, 128, 1)
		self.conv7_2 = nn.Conv2d(128, 128, 3)
		self.conv7_3 = nn.Conv2d(128, 512, 1)


		#purple block

		self.conv8_1 = nn.Conv2d(512, 256, 1)
		self.conv8_2 = nn.Conv2d(256, 256, 3)
		self.conv8_3 = nn.Conv2d(256, 1024, 1)

		self.conv9_1 = nn.Conv2d(1024, 256, 1)
		self.conv9_2 = nn.Conv2d(256, 256, 3)
		self.conv9_3 = nn.Conv2d(256, 1024, 1)

		self.conv10_1 = nn.Conv2d(1024, 256, 1)
		self.conv10_2 = nn.Conv2d(256, 256, 3)
		self.conv10_3 = nn.Conv2d(256, 1024, 1)

		self.conv11_1 = nn.Conv2d(1024, 256, 1)
		self.conv11_2 = nn.Conv2d(256, 256, 3)
		self.conv11_3 = nn.Conv2d(256, 1024, 1)

		self.conv12_1 = nn.Conv2d(1024, 256, 1)
		self.conv12_2 = nn.Conv2d(256, 256, 3)
		self.conv12_3 = nn.Conv2d(256, 1024, 1)

		self.conv13_1 = nn.Conv2d(1024, 256, 1)
		self.conv13_2 = nn.Conv2d(256, 256, 3)
		self.conv13_3 = nn.Conv2d(256, 1024, 1)

		#yellow block

		self.conv14_1 = nn.Conv2d(1024, 512, 1)
		self.conv14_2 = nn.Conv2d( 512, 512, 3)
		self.conv14_3 = nn.Conv2d( 512, 2048, 1)

		self.conv15_1 = nn.Conv2d(2048, 512 , 1)
		self.conv15_2 = nn.Conv2d( 512, 512 , 3)
		self.conv15_3 = nn.Conv2d( 512, 2048, 1)	

		self.conv16_1 = nn.Conv2d(2048, 512 , 1)
		self.conv16_2 = nn.Conv2d( 512, 512 , 3)
		self.conv16_3 = nn.Conv2d( 512, 2048, 1)	




		# an affine operation: y = Wx + b
		self.fc256 = nn.Linear(544768, 256)  # 6*6 from image dimension
		self.fc64 = nn.Linear(256, 64)  # 6*6 from image dimension
		self.fc64_2 = nn.Linear(64, 64)  # 6*6 from image dimension
		self.fc2 = nn.Linear(64, 2)  # 6*6 from image dimension
		
		self.weight = nn.Parameter(torch.randn(64, 64))
		self.weight2 = nn.Parameter(torch.randn(2, 2))

		# self.fc2 = nn.Linear(256, 64)
		# self.fc3 = nn.Linear(64, 64)
		# self.fc3 = nn.Linear(64, 2)


	def forward(self, x):
		# Max pooling over a (2, 2) window
		x = F.max_pool2d(F.relu(self.conv1(x)), (2, 2))
		residual = x
		p2d = (1, 1, 1, 1)

		#green block 1
		x = F.relu(self.conv1_1(x))
		x = F.relu(self.conv1_2(x))
		x = F.relu(self.conv1_3(x))

		x = F.pad(x, p2d, "constant", 0)

		residual = self.conv1_up(residual)
		# residual = F.relu(self.conv1_up(residual))
		x += residual
		residual = x

		#green block 2

		x = F.relu(self.conv2_1(x))
		x = F.relu(self.conv2_2(x))
		x = F.relu(self.conv2_3(x))

		x = F.pad(x, p2d, "constant", 0)

		x += residual

		#green block 3

		residual = x
		x = F.relu(self.conv3_1(x))
		x = F.relu(self.conv3_2(x))
		x = F.relu(self.conv3_3(x))

		x = F.pad(x, p2d, "constant", 0)

		x += residual
		residual = x

		#red block 1
		x = F.relu(self.conv4_1(x))
		x = F.relu(self.conv4_2(x))
		x = F.relu(self.conv4_3(x))

		x = F.pad(x, p2d, "constant", 0)
		residual = self.conv2_up(residual)

		x += residual

		residual = x
		#red block 2

		x = F.relu(self.conv5_1(x))
		x = F.relu(self.conv5_2(x))
		x = F.relu(self.conv5_3(x))

		x = F.pad(x, p2d, "constant", 0)
		x += residual
		residual = x

		#red block 3
		x = F.relu(self.conv6_1(x))
		x = F.relu(self.conv6_2(x))
		x = F.relu(self.conv6_3(x))

		x = F.pad(x, p2d, "constant", 0)
		x += residual
		residual = x

		#red block 4

		x = F.relu(self.conv7_1(x))
		x = F.relu(self.conv7_2(x))
		x = F.relu(self.conv7_3(x))

		x = F.pad(x, p2d, "constant", 0)
		x += residual
		residual = x

		#purple block 1
		x = F.relu(self.conv8_1(x))
		x = F.relu(self.conv8_2(x))
		x = F.relu(self.conv8_3(x))

		x = F.pad(x, p2d, "constant", 0)

		residual = self.conv3_up(residual)
		x += residual
		residual = x

		#purple block 2
		x = F.relu(self.conv9_1(x))
		x = F.relu(self.conv9_2(x))
		x = F.relu(self.conv9_3(x))

		x = F.pad(x, p2d, "constant", 0)

		# residual = self.conv3_up(residual)
		x += residual
		residual = x

		#purple block 3
		x = F.relu(self.conv10_1(x))
		x = F.relu(self.conv10_2(x))
		x = F.relu(self.conv10_3(x))

		x = F.pad(x, p2d, "constant", 0)

		# residual = self.conv3_up(residual)
		x += residual
		residual = x

		#purple block 4
		x = F.relu(self.conv11_1(x))
		x = F.relu(self.conv11_2(x))
		x = F.relu(self.conv11_3(x))

		x = F.pad(x, p2d, "constant", 0)

		# residual = self.conv3_up(residual)
		x += residual
		residual = x

		#purple block 5
		x = F.relu(self.conv12_1(x))
		x = F.relu(self.conv12_2(x))
		x = F.relu(self.conv12_3(x))

		x = F.pad(x, p2d, "constant", 0)

		# residual = self.conv3_up(residual)
		x += residual
		residual = x

		#purple block 6
		x = F.relu(self.conv13_1(x))
		x = F.relu(self.conv13_2(x))
		x = F.relu(self.conv13_3(x))

		x = F.pad(x, p2d, "constant", 0)
		x += residual
		residual = x

		#yellow block 1
		x = F.relu(self.conv14_1(x))
		x = F.relu(self.conv14_2(x))
		x = F.relu(self.conv14_3(x))

		x = F.pad(x, p2d, "constant", 0)

		residual = self.conv4_up(residual)
		x += residual
		residual = x

		#yellow block 2
		x = F.relu(self.conv15_1(x))
		x = F.relu(self.conv15_2(x))
		x = F.relu(self.conv15_3(x))

		x = F.pad(x, p2d, "constant", 0)
		x += residual
		residual = x

		#yellow block 3
		x = F.relu(self.conv16_1(x))
		x = F.relu(self.conv16_2(x))
		x = F.relu(self.conv16_3(x))

		x = F.pad(x, p2d, "constant", 0)
		x += residual

		print("x shape")
		x = F.avg_pool2d(x, (2,2))
		x = x.view(-1, self.num_flat_features(x))
		# print(x.shape)
		x = F.relu(self.fc256(x))
		x = F.relu(self.fc64(x))
		x = F.linear(self.fc64_2(x),self.weight)
		x = F.linear(self.fc2(x),self.weight2)

		return x

	def num_flat_features(self, x):
		size = x.size()[1:]  # all dimensions except the batch dimension
		num_features = 1
		for s in size:
			num_features *= s
		return num_features


model = DeepLightModel()
x = torch.randn(4, 4, 160, 120)
out = model(x)