from torch.utils.data import Dataset
import torch
from PIL import Image
import h5py

class _CustomDataset(Dataset):
	def __init__(self, dataframe, transform=None):
		self.dataframe = dataframe
		self.transform = transform

	def __len__(self):
		return len(self.dataframe)
	
	def __get_dataframe__(self):
		return self.dataframe
	
	def __getitem__(self, idx):
		img_path = self.dataframe.iloc[idx, 1]
		image = Image.open(img_path).convert('RGB')
		image = image.crop((295, 0, 295+1018, image.size[1]))
	
		labels = torch.tensor(self.dataframe.iloc[idx, -9:].values.astype('float32'), dtype=torch.float32)

		if self.transform:
			image = self.transform(image)

		return image, labels


class CustomDataset(Dataset):
	"""
	HDF5-backed dataset with the same external interface as the original.

	Each instance is defined by:
	  - hdf5_path: path to the HDF5 file
	  - domain: domain string (e.g. 'Home')
	  - split: 'train' or 'test'
	  - indices: list/range of integer indices into {domain}/{split}/labels

	__getitem__ returns (image_tensor, labels_tensor).
	"""

	def __init__(self, hdf5_path, domain, split, indices, image_variant="image_path",
				 transform=None):
		self.hdf5_path = hdf5_path
		self.domain = domain
		self.split = split
		self.indices = list(indices)  # ensure indexable / len()
		self.image_variant = image_variant
		self.transform = transform

		self._f = None  # lazy open per worker

	def __len__(self):
		return len(self.indices)

	def __get_indices__(self):
		return self.indices

	def _ensure_open(self):
		if self._f is None:
			self._f = h5py.File(self.hdf5_path, "r")

	def __getitem__(self, idx):
		self._ensure_open()

		h5_idx = int(self.indices[idx])
		dom_split = f"{self.domain}/{self.split}"
		img_grp = self._f[f"{dom_split}/images"]
		labels_ds = self._f[f"{dom_split}/labels"]

		img_np = img_grp[self.image_variant][h5_idx]
		image = torch.from_numpy(img_np).float()

		labels = torch.tensor(labels_ds[h5_idx], dtype=torch.float32)

		if self.transform is not None:
			image = self.transform(image)

		return image, labels
