from typing import List

import torch
import torch.nn as nn
from src.image_encoder import ImageEncoder
from src.tabular_encoder import TabularEncoder
from src.multimodal_interaction import MultimodalInteraction
from src.classifier import Classifier

class Handler(nn.Module):
	def __init__(self, x_num_categories_list: List[int], y_num_categories_list: List[int]) -> None:
		"""
		Multimodal model handler that integrates image and tabular data for classification tasks.

		Args:
			num_categories_list (List[int]): A list of integers representing the number of categories for each categorical feature to predict.
		"""
		super(Handler, self).__init__()

		self.embedding_dim = 512
		self.num_interaction_blocks = 2
		self.num_tabular_blocks = 2
		self.x_num_categories_list = x_num_categories_list
		self.y_num_categories_list = y_num_categories_list
		
		self.image_encoder = ImageEncoder()
		self.tabular_encoder = TabularEncoder(embedding_dim=self.embedding_dim, num_categories_list=self.x_num_categories_list, num_blocks=self.num_tabular_blocks)
		self.multimodal_interaction = MultimodalInteraction(embedding_dim=self.embedding_dim, num_blocks=self.num_interaction_blocks)
		self.classifier = Classifier(interaction_embedding_dim=self.embedding_dim, num_categories_list=self.y_num_categories_list)

	def forward(self, img, tab):
		image_features = self.image_encoder(img)
		tabular_features = self.tabular_encoder(tab)
		interaction_features = self.multimodal_interaction(image_features, tabular_features)
		predictions = self.classifier(interaction_features)

		return predictions