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

	for i in range(len(df)):
		try:
			entry = [float(li[i][:3]),
					 f'{path}/' + li[i],
					 int(df[df['Stamp'] == float(li[i].split('_')[0])]['Using circle']),
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
		except Exception as e:
			print(f"Error Loading FileName {li[i].split('_')[0]}: {e}; Continuing to next.")

	return data_images

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
	
	#shuffle order
	arrow = arrow.sample(frac=1).reset_index(drop=True)
	circle = circle.sample(frac=1).reset_index(drop=True)
	
	#last 25% test for each dataset + combined dataset > dataset > loaders
	test_circle = circle.iloc[int(circle.shape[0] * 0.75):]
	test_circle = DataLoader(CustomDataset(test_circle, transform=test_transform), batch_size=batch_size, drop_last=True)

	test_arrow = arrow.iloc[int(arrow.shape[0] * 0.75):]
	test_arrow = DataLoader(CustomDataset(test_arrow, transform=test_transform), batch_size=batch_size, drop_last=True)

	test = pd.concat([circle.iloc[int(circle.shape[0] * 0.75):], arrow.iloc[int(arrow.shape[0] * 0.75):]], axis=0)
	test = test.sample(frac=1).reset_index(drop=True)
	test = DataLoader(CustomDataset(test, test_transform), batch_size=batch_size, drop_last=True)

	#take first 75% (for training)
	arrow = arrow.iloc[:int(arrow.shape[0] * 0.75)]
	circle = circle.iloc[:int(circle.shape[0] * 0.75)]
	
	#duplicate the dataset if needed and shuffle
	if aug:
		arrow = load_augmented(arrow)
		circle = load_augmented(circle)
		arrow = arrow.sample(frac=1).reset_index(drop=True)
		circle = circle.sample(frac=1).reset_index(drop=True)
		
	#calc size of each dataset - divide total dataset into n_clients	
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


def task_splitter(path, n_clients, aug, batch_size=16):
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

	train = pd.read_csv(path + "/pepper_data_train.csv")
	test = pd.read_csv(path + "/pepper_data_test_equal.csv")
	train["path"] = train["path"].str.replace("../data/", "data/", regex=False)
	test["path"] = test["path"].str.replace("../data/", "data/", regex=False)
	
	y_labels = train.columns[-9:]
	tasks = train['domain'].unique()

	# --- Setup test dataloaders (per task) ---
    # For each task, create dataloader over just that task
	test_sets = [test[test['domain'] == task] for task in tasks]

    # --- Setup incremental test sets (cumulative) ---
	incremental_test_loaders = []
	cum_df = pd.DataFrame()
	for i, task in enumerate(tasks):
		cum_df = pd.concat([cum_df, test_sets[i]], ignore_index=True)
		cum_loader = DataLoader(CustomDataset(cum_df.copy(), transform=test_transform), batch_size=batch_size, drop_last=False)
		incremental_test_loaders.append(cum_loader)


	# --- Setup train dataloaders per task, per client ---
	train_task_cl = []
	for task in tasks:
		task_data = train[train['domain'] == task].copy()
		if aug:
			task_data = load_augmented(task_data)
			task_data = task_data.sample(frac=1).reset_index(drop=True)

		n_samples = len(task_data)
		samples_per_client = n_samples // n_clients
		task_client_loaders = []
		for i in range(n_clients):
			start = i * samples_per_client
			end = (i + 1) * samples_per_client if i < n_clients - 1 else n_samples
			subset = task_data.iloc[start:end].reset_index(drop=True)
			loader = DataLoader(CustomDataset(subset, train_transform), batch_size=batch_size, shuffle=True, drop_last=False)
			task_client_loaders.append(loader)
		train_task_cl.append(task_client_loaders)
		# train_task_cl: list of [list of client of DataLoader objects]  shape: tasks x clients

	# Make DataLoader over all test data (optional, for convenience)
	all_test_loader = DataLoader(CustomDataset(test, transform=test_transform), batch_size=batch_size, drop_last=False)

	# Output structure:
	# train_task_cl: list(t) of list(n_clients) of DataLoaders
	# incremental_test_loaders: list(t) of DataLoaders
	# all_test_loader: DataLoader
	# y_labels: column names
	return train_task_cl, incremental_test_loaders, all_test_loader, y_labels


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

