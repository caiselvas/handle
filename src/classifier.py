import torch
import torch.nn as nn



class Classifier(nn.Module):
	def __init__(self, interaction_embedding_dim, output_dims):
		super(Classifier, self).__init__()
		
		self.l1 = nn.Linear(interaction_embedding_dim, 512)
		self.l2 = nn.Linear(512, 256)
		
		self.output_layers = nn.ModuleList([nn.Linear(256, output_dim) for output_dim in output_dims])
		
	def forward(self, x):
		x = torch.relu(self.l1(x))
		x = torch.relu(self.l2(x))
		
		outputs = [torch.softmax(layer(x), dim=1) for layer in self.output_layers]
		return outputs
		