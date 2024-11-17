from typing import List
import pandas as pd
import torch
import numpy as np
from torch.utils.data import Dataset
from torchvision import transforms
from sklearn.preprocessing import LabelEncoder, OneHotEncoder
from PIL import Image

class CustomDataset(Dataset):
	def __init__(self, x_path: str, y_path: str, image_folder_path: str, transform: object = None, filename_col: str = 'des_filename'):
		"""
		Args:
			x_path (str): Path to the CSV file containing the tabular data (including the image names).
			image_folder_path (str): Path to the folder containing the images.
			y_path (str): Path to the CSV file containing the target labels (matrix format for multi-label softmax).
			transform (object): Function to apply to the images.
			filename_col (str): Name of the column containing the image names.
		"""
		self.data = pd.read_csv(x_path)
		self.image_folder_path = image_folder_path
		self.filename_col = filename_col
		self.transform = transform if transform else transforms.ToTensor()
		
		# Load the labels and apply one-hot encoding
		self.labels_data = pd.read_csv(y_path)
		self.onehot_encoders = {}
		onehot_encoded_labels = []
		
		# Apply OneHotEncoder separately to each column and concatenate results
		for col in self.labels_data.columns:
			encoder = OneHotEncoder(sparse_output=False)
			onehot_encoded_col = encoder.fit_transform(self.labels_data[[col]])
			onehot_encoded_labels.append(onehot_encoded_col)
			self.onehot_encoders[col] = encoder  # Store encoder for each column if needed for inverse transform

		# Concatenate one-hot encoded columns
		self.labels = torch.tensor(np.concatenate(onehot_encoded_labels, axis=1), dtype=torch.float)
	
		# Process and encode non-numeric columns in the tabular data
		self.tabular_data = self.data.drop(columns=[self.filename_col])
		self.label_encoders = {}
		for col in self.tabular_data.columns:
			if self.tabular_data[col].dtype == 'object':  # Non-numeric columns
				le = LabelEncoder()
				self.tabular_data[col] = le.fit_transform(self.tabular_data[col])
				self.label_encoders[col] = le

		# Convert tabular data to tensor
		self.tabular_data = torch.tensor(self.tabular_data.values, dtype=torch.long)
		
		# Save image names
		self.image_names = self.data[self.filename_col].values

	def __len__(self):
		return len(self.data)

	def __getitem__(self, idx):
		img_path = f"{self.image_folder_path}/{self.image_names[idx]}"
		image = Image.open(img_path).convert("RGB")
		
		if self.transform is not None:
			image = self.transform(image)
		
		# Get tabular data and label for the current index
		tabular_data = self.tabular_data[idx].clone().detach().to(torch.long)

		label = torch.tensor(self.labels[idx], dtype=torch.float)

		return image, tabular_data, label

	def get_y_num_categories_list(self) -> List[int]:
		"""
		Gets the number of unique categories for each column in the y_path CSV file.

		Returns:
			List[int]: A list of integers representing the number of unique categories for each
		"""
		num_categories_list = [len(self.labels_data[col].unique()) for col in self.labels_data.columns]
		
		return num_categories_list
	
	def get_x_num_categories_list(self) -> List[int]:
		"""
		Gets the number of unique categories for each column in the x_path CSV file.

		Returns:
			List[int]: A list of integers representing the number of unique categories for each
		"""
		num_categories_list = [len(self.data[col].unique()) for col in self.data.columns if col != self.filename_col]
		
		return num_categories_list
	
	def get_label_encoders(self) -> dict:
		"""
		Gets the label encoders for each non-numeric column in the x_path CSV file.

		Returns:
			dict: A dictionary mapping column names to their corresponding LabelEncoders.
		"""
		return self.label_encoders
	
	def get_onehot_encoders(self) -> dict:
		"""
		Gets the one-hot encoders for each column in the y_path CSV file.

		Returns:
			dict: A dictionary mapping column names to their corresponding OneHotEncoders.
		"""
		return self.onehot_encoders
