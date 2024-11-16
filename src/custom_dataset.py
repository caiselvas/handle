from typing import List

import pandas as pd
import torch
from torch.utils.data import Dataset
from PIL import Image

class CustomDataset(Dataset):
	def __init__(self, csv_file_path: str, image_folder_path: str, transform: object=None, filename_col: str='des_filename'):
		""""
		Args:
			csv_file_path (str): Path to the CSV file containing the tabular data (including the image names).
			image_folder_path (str): Path to the folder containing the images.
			transform (object): Function to apply to the images.
			filename_col (str): Name of the column containing the image names.
		"""
		self.data = pd.read_csv(csv_file_path)
		self.image_folder_path = image_folder_path
		self.transform = transform
		self.filename_col = filename_col
		
		# Removes the filename column and stores the tabular data as a tensor
		self.tabular_data = self.data.drop(columns=[self.filename_col]).values
		# Saves the image names
		self.image_names = self.data[self.filename_col].values
	
	def __len__(self):
		return len(self.data)
	
	def __getitem__(self, idx):
		img_path = f"{self.image_folder_path}/{self.image_names[idx]}"
		image = Image.open(img_path).convert("RGB")
		
		if self.transform is not None:
			image = self.transform(image)
		
		tabular_data = torch.tensor(self.tabular_data[idx], dtype=torch.float)
		
		return image, tabular_data
	
	def get_num_categories_list(self) -> List[int]:
		"""
		Gets the number of unique categories for each column in the tabular data.

		Returns:
			List[int]: List of the number of unique categories for each column in the tabular data.
		"""
		return [len(self.data[col].unique()) for col in self.data.columns if col != self.filename_col]
