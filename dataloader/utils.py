import pandas as pd
import os
import torchvision.transforms as transforms
from .imageloader import CustomDataset
from torch.utils.data import random_split, DataLoader, ConcatDataset
import torch
import numpy as np
import sys
from sklearn.utils import shuffle
import h5py

sys.path.append('..')
from utils import predict_gen, train, predict_gen_distil


def load_universal(path):
	data = pd.read_csv(path + "/all_data.csv")
	df = pd.DataFrame(columns=data.columns)
	for i in range(250, 1000):
		df = pd.concat([df, (data.loc[data['Stamp'] == i].mean()).to_frame().T], ignore_index=True)
	# df=df.append(data.loc[data['Stamp'] == i].mean(), ignore_index=True)
	return df


def load_augmented(df):
	data_new = pd.DataFrame(columns=df.columns)
	for c in range(10):
		for i in range(df.shape[0]):
			data_new = pd.concat([data_new, df.iloc[i].to_frame().T])
	return data_new


def load_images(path):
	import re
	def sort_nicely(l):
		""" Sort the given list in the way that humans expect.
		"""

		convert = lambda text: int(text) if text.isdigit() else text
		alphanum_key = lambda key: [convert(c) for c in re.split('([0-9]+)', key)]
		l.sort(key=alphanum_key)
		return l
	
	
	df = load_universal(path)
	path = path + "/images"
	li = sort_nicely(os.listdir(path))
	data_images = pd.DataFrame(columns=['Stamp', 'path'] + list(df.columns[1:3]) + list(df.columns[-8:]))

	for i in range(len(li)):
		try:
			entry = [float(li[i][:3]),
					 f'{path}/' + li[i],
					 int(df[df['Stamp'] == float(li[i][:3])]['Using circle']),
					 int(df[df['Stamp'] == float(li[i][:3])]['Using arrow']),
					 float(df[df['Stamp'] == float(li[i][:3])]['Vacuum cleaning']),
					 float(df[df['Stamp'] == float(li[i][:3])]['Mopping the floor']),
					 float(df[df['Stamp'] == float(li[i][:3])]['Carry warm food']),
					 float(df[df['Stamp'] == float(li[i][:3])]['Carry cold food']),
					 float(df[df['Stamp'] == float(li[i][:3])]['Carry drinks']),
					 float(df[df['Stamp'] == float(li[i][:3])]['Carry small objects (plates, toys)']),
					 float(df[df['Stamp'] == float(li[i][:3])]['Carry big objects (tables, chairs)']),
					 float(df[df['Stamp'] == float(li[i][:3])]['Cleaning (Picking up stuff) / Starting conversation'])
					 ]
			new_df = pd.DataFrame([entry], columns=data_images.columns)
			data_images = pd.concat([data_images, new_df], ignore_index=True)
		except:
			print('Error Loading FileName; Continuing to next.')

	return data_images


def load_datasets(num_clients, path, aug, batch_size=16, out='', DEVICE=torch.device("cpu"), data_permutation=None,
				  distil=False, teacher_model=None):
	data = load_images(path)
	if data_permutation is None:
		data_permutation = np.random.permutation(len(data))
	data = data.iloc[data_permutation]
	# Splitting 75% Train and 25% Test Data
	data_images_train = data.iloc[:int(data.shape[0] * 0.75)]
	data_images_train = data_images_train.sample(frac=1).reset_index(drop=True)
	
	data_images_test = data.iloc[int(data.shape[0] * 0.75):]
	data_images_test = data_images_test.sample(frac=1).reset_index(drop=True)
	
	if aug:
		train_transform = transforms.Compose([
			transforms.Resize((128, 128)),
			transforms.RandomHorizontalFlip(),
			transforms.RandomRotation(10),
			# transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1),
			# transforms.RandomResizedCrop(32, scale=(0.8, 1.0)),
			transforms.ToTensor(),
			transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
		])
		data_images_train = load_augmented(data_images_train)
		data_images_train = data_images_train.sample(frac=1).reset_index(drop=True)

	else:
		train_transform = transforms.Compose([
			transforms.Resize((128, 128)),  # Adjust the size according to your model requirements
			transforms.ToTensor(),
			transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
		])
	test_transform = transforms.Compose([
		transforms.Resize((128, 128)),  # Adjust the size according to your model requirements
		transforms.ToTensor(),
		transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
	])
	trainset = CustomDataset(dataframe=data_images_train, transform=train_transform)
	testset = CustomDataset(dataframe=data_images_test, transform=test_transform)
	
	if distil:
		if os.path.exists(f'{out}/teacher_model.pt'):
			teacher_model.to(DEVICE)
			teacher_model.load_state_dict(torch.load(f'{out}/teacher_model.pt'))
		elif os.path.exists(f'{out}/teacher_model_cuda.pt'):
			teacher_model.to(DEVICE)
			teacher_model.load_state_dict(torch.load(f'{out}/teacher_model_cuda.pt'))
		else:
			teacher_model.to(DEVICE)
			# trainset_distil = CustomDataset(dataframe=data_images_train, transform=train_transform)
			train_loader_distil = DataLoader(trainset, batch_size=batch_size, shuffle=False, drop_last=True)
			print("Training Teacher Model")
			train(model=teacher_model, train_loader=train_loader_distil, epochs=10, DEVICE=DEVICE)
		if DEVICE.type == 'cuda':
			torch.save(teacher_model.state_dict(), f'{out}/teacher_model_cuda.pt')
		else:
			torch.save(teacher_model.state_dict(), f'{out}/teacher_model.pt')
	
		outputs = pd.DataFrame(predict_gen_distil(teacher_model, train_loader_distil, DEVICE))
		trainset = CustomDataset(dataframe=pd.concat([data_images_train.iloc[:len(outputs), :4], outputs], axis=1, ignore_index=True),
								transform=train_transform)
	

	# Split training set into `num_clients` partitions to simulate different local datasets
	partition_size = len(trainset) // num_clients

	lengths = [partition_size] * num_clients
	# trim trainset to partition_size*num_clients without iloc
	# trainset = torch.utils.data.Subset(trainset, range(partition_size * num_clients))
	datasets = random_split(torch.utils.data.Subset(trainset, range(partition_size * num_clients)), lengths, torch.Generator().manual_seed(42))
	# datasets = random_split(testset[:sum(lengths)], lengths, torch.Generator().manual_seed(42))
	y_labels = data.columns[-8:]
	# Split each partition into train/val and create DataLoader
	trainloaders = []
	# valloaders = []
	
	for ds in datasets:
		# if distil:
		# 	outputs = pd.DataFrame(predict_gen_distil(teacher_model, DataLoader(ds, batch_size=batch_size, shuffle=True, drop_last=True), DEVICE))
		# 	inputs = ds.__get_dataframe__().iloc[:, :4]
		# 	ds_teacher = pd.concat([inputs, outputs], axis=1)
		# 	trainloaders.append(DataLoader(ds_teacher, batch_size=batch_size, shuffle=True, drop_last=True))
		# 	# trainloaders.append(
		# 	# 	predict_gen_distil(teacher_model, DataLoader(ds, batch_size=batch_size, shuffle=True, drop_last=True), DEVICE))
		# else:
		trainloaders.append(DataLoader(ds, batch_size=batch_size, shuffle=True, drop_last=True))
	if distil:
		teacher_model = None
	# valloaders.append(DataLoader(ds_val, batch_size=batch_size))
	testloader = DataLoader(testset, batch_size=batch_size)
	# return trainloaders, valloaders, testloader, y_labels
	return trainloaders, testloader, y_labels, data_permutation


def load_datasets_pretrain(num_clients, path, split, aug=True, batch_size=16, out='', DEVICE=torch.device("cpu"),
						   data_permutation=None):
	data = load_images(path)
	if data_permutation is None:
		data_permutation = np.random.permutation(len(data))
	data = data.iloc[data_permutation]
	# Splitting 75% Train and 25% Test Data
	# shuffle data dataframe
	data = shuffle(data)
	# reset index after shuffle
	data.reset_index(inplace=True, drop=True)
	# data_images_train = data.iloc[:int(data.shape[0] * (1-split))]
	data_images_test = data.iloc[int(data.shape[0] * (1 - split)):]

	test_transform = transforms.Compose([
		transforms.Resize((128, 128)),  # Adjust the size according to your model requirements
		transforms.RandomHorizontalFlip(),
		transforms.RandomRotation(10),
		transforms.ToTensor(),
		transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
	])
	testset = CustomDataset(dataframe=data_images_test, transform=test_transform)
	# Split each partition into train/val and create DataLoader
	trainloaders = []
	# valloaders = []
	testloader = DataLoader(testset, batch_size=batch_size)
	# return trainloaders, valloaders, testloader, y_labels
	return testloader



def ac_split(path):
	data = load_images(path)
	arrow = data[data['Using arrow'] == 1]
	circle = data[data['Using circle'] == 1]
	return arrow, circle


def task_splitter_circle_arrow(path, n_clients, aug, batch_size=16):
	if aug:
		train_transform = transforms.Compose([
			transforms.Resize((128, 128)),
			transforms.RandomHorizontalFlip(),
			transforms.RandomRotation(10),
			# transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1),
			# transforms.RandomResizedCrop(32, scale=(0.8, 1.0)),
			transforms.ToTensor(),
			transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
		])
	else:
		train_transform = transforms.Compose([
			transforms.Resize((128, 128)),  # Adjust the size according to your model requirements
			transforms.ToTensor(),
			transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
		])
	test_transform = transforms.Compose([
		transforms.Resize((128, 128)),  # Adjust the size according to your model requirements
		transforms.ToTensor(),
		transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
	])

	arrow, circle = ac_split(path)
	train_arrow_cl = []
	train_circle_cl = []

	# Label columns
	y_labels = arrow.columns[-8:]
	
	arrow = arrow.sample(frac=1).reset_index(drop=True)
	circle = circle.sample(frac=1).reset_index(drop=True)
	
	test_circle = circle.iloc[int(circle.shape[0] * 0.75):]
	test_circle = DataLoader(CustomDataset(test_circle, transform=test_transform), batch_size=batch_size, drop_last=True)

	test_arrow = arrow.iloc[int(arrow.shape[0] * 0.75):]
	test_arrow = DataLoader(CustomDataset(test_arrow, transform=test_transform), batch_size=batch_size, drop_last=True)

	test = pd.concat([circle.iloc[int(circle.shape[0] * 0.75):], arrow.iloc[int(arrow.shape[0] * 0.75):]], axis=0)
	test = test.sample(frac=1).reset_index(drop=True)
	test = DataLoader(CustomDataset(test, test_transform), batch_size=batch_size, drop_last=True)

	arrow = arrow.iloc[:int(arrow.shape[0] * 0.75)]
	circle = circle.iloc[:int(circle.shape[0] * 0.75)]
	
	if aug:
		arrow = load_augmented(arrow)
		circle = load_augmented(circle)
		arrow = arrow.sample(frac=1).reset_index(drop=True)
		circle = circle.sample(frac=1).reset_index(drop=True)
		
	size_arrow = int(arrow.shape[0] / n_clients)
	size_circle = int(circle.shape[0] / n_clients)
	for i in range(n_clients):
		train_arrow_cl.append(
			DataLoader(CustomDataset(arrow.iloc[i * size_arrow:(i + 1) * size_arrow], train_transform), batch_size=batch_size,
					   shuffle=True, drop_last=True))
		train_circle_cl.append(
			DataLoader(CustomDataset(circle.iloc[i * size_circle:(i + 1) * size_circle], train_transform), batch_size=batch_size,
					   shuffle=True, drop_last=True))

	return [train_circle_cl, train_arrow_cl], [test_circle, test_arrow], test, y_labels


def load_datasets_hyper(path, aug):
	data = load_images(path, aug)
	data_images_train = data.iloc[:int(data.shape[0] * 0.75)]
	data_images_test = data.iloc[int(data.shape[0] * 0.75):]
	if aug == True:
		transform = transforms.Compose([
			transforms.Resize((128, 128)),
			transforms.RandomHorizontalFlip(),
			transforms.RandomRotation(10),
			# transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1),
			# transforms.RandomResizedCrop(32, scale=(0.8, 1.0)),
			transforms.ToTensor(),
		])
	else:
		transform = transforms.Compose([
			transforms.Resize((128, 128)),  # Adjust the size according to your model requirements
			transforms.ToTensor(),
			#         transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))  # Normalize if required
		])
	trainset = CustomDataset(dataframe=data_images_train, transform=transform)
	testset = CustomDataset(dataframe=data_images_test, transform=transform)
	return trainset, testset

def hdf5_task_splitter(path, n_clients, aug, batch_size=16):
	"""
	HDF5-based tasksplitter.

	- Uses ../data/mean_data_pepper.hdf5 (or `path`).
	- Tasks = domains: ['Home', 'BigOffice-2', 'BigOffice-3', 'Hallway', 'MeetingRoom', 'SmallOffice'].
	- For each domain:
		* train: {domain}/train/labels
		* test:  {domain}/test/labels
	- Returns:
		* traintaskcl: list[tasks] of list[n_clients] of DataLoaders over domain-specific train splits.
		* valloaders:  list[tasks] of DataLoaders over per-domain test splits (not cumulative).
		* testloader:  DataLoader over joint test set (all domains concatenated).
		* ylabels:     list of 9 action names (for metrics only).
	"""

	hdf5_path = path or "../data/mean_data_pepper.hdf5"

	# 9 regression outputs in fixed order
	ylabels = [
		"Vaccum Cleaning", "Mopping the Floor", "Carry Warm Food",
		"Carry Cold Food", "Carry Drinks", "Carry Small Objects",
		"Carry Large Objects", "Cleaning", "Starting a conversation",
	]

	# HDF5 images are already resized/normalized; keep transforms None
	traintransform = None
	testtransform = None

	# Discover domains and sample counts
	with h5py.File(hdf5_path, "r") as f:
		domains = ['Home', 'BigOffice-2', 'BigOffice-3', 'Hallway', 'MeetingRoom', 'SmallOffice']

		# Prepare index info per domain
		train_counts = {}
		test_counts = {}
		for domain in domains:
			dom_grp = f[domain]

			# TRAIN
			n_train = dom_grp["train/labels"].shape[0]
			train_counts[domain] = n_train

			# TEST
			n_test = dom_grp["test/labels"].shape[0]
			test_counts[domain] = n_test

	# --- Build train loaders per task (domain), per client ---
	traintaskcl = []
	for domain in domains:
		n_train = train_counts[domain]

		# Shuffle indices per domain
		all_indices = list(range(n_train))

		# Simple deterministic shuffling, or you can use random.Random(seed).shuffle(...)
		import random
		random.Random(42).shuffle(all_indices)

		samples_per_client = n_train // n_clients
		client_loaders = []
		for c in range(n_clients):
			start = c * samples_per_client
			end = (c + 1) * samples_per_client if c < n_clients - 1 else n_train
			client_indices = all_indices[start:end]

			ds = CustomDataset(
				hdf5_path=hdf5_path,
				domain=domain,
				split="train",
				indices=client_indices,
				image_variant="image_path",
				transform=traintransform,
			)
			loader = DataLoader(ds, batch_size=batch_size, shuffle=True, drop_last=False)
			client_loaders.append(loader)

		traintaskcl.append(client_loaders)

	# --- Build per-domain validation loaders (no cumulative) ---
	valloaders = []
	for domain in domains:
		n_test = test_counts[domain]
		indices = list(range(n_test))
		ds = CustomDataset(
			hdf5_path=hdf5_path,
			domain=domain,
			split="test",
			indices=indices,
			image_variant="image_path",
			transform=testtransform,
		)
		loader = DataLoader(ds, batch_size=batch_size, shuffle=False, drop_last=False)

		valloaders.append(loader)

	# --- Build joint testloader over all domains ---
	test_datasets = []
	for domain in domains:
		n_test = test_counts[domain]

		indices = list(range(n_test))
		ds = CustomDataset(
			hdf5_path=hdf5_path,
			domain=domain,
			split="test",
			indices=indices,
			image_variant="image_path",
			transform=testtransform,
		)
		test_datasets.append(ds)

	joint_dataset = ConcatDataset(test_datasets)

	testloader = DataLoader(joint_dataset, batch_size=batch_size, shuffle=False, drop_last=False)

	return traintaskcl, valloaders, testloader, ylabels