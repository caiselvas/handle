from typing import List

import torch
import torch.nn as nn

class Classifier(nn.Module):
	def __init__(self, 
			interaction_embedding_dim: int, 
			num_categories_list: List[int]
			) -> None:
		"""
		Args:
			interaction_embedding_dim (int): The dimension of the output vector from the multimodal interaction module
			num_categories_list (List[int]): A list of integers representing the number of categories for each categorical feature
		"""
		super(Classifier, self).__init__()
		
		self.output_layers = nn.ModuleList(
			[nn.Linear(interaction_embedding_dim, num_classes) for num_classes in num_categories_list]
		)

	def forward(self, x): 
		outputs = [nn.functional.softmax(layer(x), dim=1) for layer in self.output_layers]
		return outputs


