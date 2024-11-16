import torch
import torch.nn as nn
import torch.optim as optim

from src.handler import Handler

images_data_path = './data/archive/images'
attributes_data_path = './data/attributes.csv'

handler = Handler(num_categories_list=