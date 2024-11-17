from typing import Dict

import torch
import torch.nn as nn
from torch.utils.data import Dataset
from torchvision import transforms

import pandas as pd
from sklearn.preprocessing import LabelEncoder
from PIL import Image

class CustomInferenceDataset(Dataset):
	def __init__(self, data, label_encoders: Dict, onehot_encoders: Dict, image_folder_path: str, filename_col: str='des_filename', test_id_col: str='test_id', attribute_name_col: str='attribute_name'):
		"""
		Args:
			label_encoders (Dict): Dictionary containing the label encoders used in the x_train dataset.
			onehot_encoders (Dict): Dictionary containing the one-hot encoders used in the y_train dataset.
			x_test_path (str): Path to the CSV file containing the tabular data (including the image names).
			image_folder_path (str): Path to the folder containing the images.
			filename_col (str): Name of the column containing the image names.
			test_id_col (str): Name of the column containing the test IDs.
			attribute_name_col (str): Name of the column containing the attribute names.
		"""
		self.image_folder_path = image_folder_path
		self.label_encoders = label_encoders
		self.onehot_encoders = onehot_encoders
		self.data = data
		self.filename_col = filename_col
		self.test_id_col = test_id_col
		self.attribute_name_col = attribute_name_col
		self.transform = transforms.ToTensor()
		self.image_shape = None

		# Process and encode non-numeric columns in the tabular data
		self.tabular_data = self.data.drop(columns=[self.filename_col, self.test_id_col, self.attribute_name_col])
		for col in self.tabular_data.columns:
			if self.tabular_data[col].dtype == 'object':  # Non-numeric columns
				le = self.label_encoders[col]
				self.tabular_data[col] = le.transform(self.tabular_data[col])

		self.tabular_data = torch.tensor(self.tabular_data.values, dtype=torch.long)

		self.image_names = self.data[self.filename_col].values

		self.image_prefixes = [''.join(img_name.split('_')[:2]) for img_name in self.image_names]

		self.attribute_categories = self.data[self.attribute_name_col].values
		self.test_ids = self.data[self.test_id_col].values

	def __len__(self):
		return len(self.data)
	
	def __getitem__(self, idx):
		try:
			img_path = f"{self.image_folder_path}/{self.image_names[idx]}"
			image = Image.open(img_path).convert("RGB")
			image = self.transform(image)

			if self.image_shape is None:
				self.image_shape = image.shape

		except Exception as e:
			print(f"Error loading image at index {idx}: {e}")
			image = torch.zeros(self.image_shape, dtype=torch.float32) if self.image_shape else torch.zeros((3, 224, 224), dtype=torch.float32)

		# Get tabular data and label for the current index
		tabular_data = self.tabular_data[idx].clone().detach().to(torch.long)

		image_prefix = self.image_prefixes[idx]

		return image, tabular_data, image_prefix
	
	def value_from_label_encoder(self, col: int, label: int) -> str:
		"""
		Converts a label back to its original value using the label encoder.

		Args:
			col (int): Column index.
			label (int): Label to convert.

		Returns:
			str: Original value of the label.
		"""
		return self.label_encoders[col].inverse_transform([label])[0]
	
	def value_from_onehot_encoder(self, col: int, onehot: torch.Tensor) -> str:
		"""
		Converts a one-hot encoded vector back to its original value using the one-hot encoder.

		Args:
			col (int): Column index.
			onehot (torch.Tensor): One-hot encoded vector to convert.

		Returns:
			str: Original value of the one-hot encoded vector.
		"""
		return self.onehot_encoders[col].inverse_transform([onehot])[0]