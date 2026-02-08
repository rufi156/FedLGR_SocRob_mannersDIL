# import all libraries in the code below
import flwr as fl
import sys

sys.path.append('..')
from metrics.computation import RAMU
from utils import get_parameters, set_parameters, train, test, predict, predict_gen
import pickle
import os
import copy
from CL.default import Memory
from typing import Any, Callable, Dict, List, Optional, Tuple
from collections import OrderedDict
import torch
import numpy as np


class FlowerClient_Root(fl.client.NumPyClient):
	def __init__(self, cid, net, trainloader, testloader, epochs, y_labels, num_clients, DEVICE, path):
		self.cid = cid
		self.net = net
		self.trainloader = trainloader
		self.testloader = testloader
		self.epochs = epochs
		self.y_labels = y_labels
		self.path = path
		self.num_clients = num_clients
		self.DEVICE = DEVICE
	
	def get_parameters(self, config):
		print(f"[Client {self.cid}] get_parameters")
		self.net.conv_module.train()
		return [val.cpu().numpy() for _, val in self.net.conv_module.state_dict().items()]
	
	def get_parameters_fc(self, config):
		print(f"[Client {self.cid}] get_parameters")
		self.net.fc_module.train()
		return [val.cpu().numpy() for _, val in self.net.fc_module.state_dict().items()]
	
	def set_parameters(self, parameters: List[np.ndarray]) -> None:
		self.net.conv_module.train()
		params_dict = zip(self.net.conv_module.state_dict().keys(), parameters)
		state_dict = OrderedDict({k: torch.tensor(v) for k, v in params_dict})
		self.net.conv_module.load_state_dict(state_dict, strict=True)
	
	def set_parameters_fc(self, parameters: List[np.ndarray]) -> None:
		self.net.fc_module.train()
		params_dict = zip(self.net.fc_module.state_dict().keys(), parameters)
		state_dict = OrderedDict({k: torch.tensor(v) for k, v in params_dict})
		self.net.fc_module.load_state_dict(state_dict, strict=True)
	
	def fit(self, parameters, config):
		print(f"[Client {self.cid}] fit, config: {config}")
		ramu=RAMU()
		init_ram=ramu.compute("TRAINING")

		self.set_parameters(parameters)
		if config['server_round'] == 1:
			peak_ram=train(self.net, self.trainloader, epochs=self.epochs, DEVICE=self.DEVICE)
		
		if config['server_round'] > 1:
			with open(f'{self.path}/mod{self.cid}.pkl', 'rb') as f:
				state_dict = pickle.load(f)
			self.net.fc_module.load_state_dict(state_dict, strict=True)
			peak_ram=train(self.net, self.trainloader, epochs=self.epochs, DEVICE=self.DEVICE)
		state_dict = self.net.fc_module.state_dict()
		with open(f'{self.path}/mod{self.cid}.pkl', 'wb') as f:
			pickle.dump(state_dict, f)
		if not os.path.exists(f'{self.path}/clientwise'):
			os.makedirs(f'{self.path}/clientwise')
		with open(f'{self.path}/clientwise/ram{int(self.cid)}.csv', 'a+') as f:
			f.write(f'{config["server_round"]},{init_ram},{peak_ram},{peak_ram-init_ram}\n')
		return self.get_parameters(config={}), len(self.trainloader), {}
	
	def evaluate(self, parameters, config):
		if not os.path.exists(f'{self.path}/clientwise'):
			os.makedirs(f'{self.path}/clientwise')
		print(f"[Client {self.cid}] evaluate, config: {config}")
		self.set_parameters(parameters)
		try:
			with open(f'{self.path}/mod{self.cid}.pkl', 'rb') as f:
				state_dict = pickle.load(f)
			self.net.fc_module.load_state_dict(state_dict, strict=True)
		except:
			print('')
		loss, avg_pearson, avg_rmse = test(self.net, self.testloader, self.y_labels, self.DEVICE)
		with open(f'{self.path}/clientwise/results{int(self.cid)}.txt', 'a+') as f:  # Python 3: open(..., 'wb')
			f.write(f'{config["server_round"]},{loss},{avg_pearson},{avg_rmse}\n')
		return float(loss), len(self.testloader), {"avg_pearson_score": avg_pearson, "avg_rmse": avg_rmse}


class FlowerClientCL_Root(fl.client.NumPyClient):
	def __init__(self, cid, net, trainloader, valloader, testloader, epochs, y_labels, cl_strategy, agent_config, nrounds, path, DEVICE, num_clients,
				 strat_name, params):
		self.cid = cid
		# self.net = net
		self.trainloaders = trainloader
		self.valloader = valloader
		self.testloader = testloader
		self.epochs = epochs
		self.y_labels = y_labels
		self.strat = cl_strategy(agent_config, net, params, fedroot=True)
		self.nrounds = nrounds
		self.path = path
		self.DEVICE = DEVICE
		self.num_clients = num_clients
		self.strat_name = strat_name
	
	def get_parameters_all(self, config):
		print(f"[Client {self.cid}] get_parameters")
		# self.net.train()
		self.strat.model.train()
		return [val.cpu().numpy() for _, val in self.strat.model.state_dict().items()]
	

	def get_parameters(self, config):
		print(f"[Client {self.cid}] get_parameters")
		self.strat.model.conv_module.train()
		# self.net.conv_module.train()
		return [val.cpu().numpy() for _, val in self.strat.model.conv_module.state_dict().items()]
	
	def get_parameters_fc(self, config):
		print(f"[Client {self.cid}] get_parameters FC")
		self.strat.model.fc_module.train()
		# self.net.fc_module.train()
		return [val.cpu().numpy() for _, val in self.strat.model.fc_module.state_dict().items()]
	
	def set_parameters(self, parameters: List[np.ndarray]) -> None:
		print(f"[Client {self.cid}] set_parameters")
		self.strat.model.conv_module.train()
		# self.net.conv_module.train()
		params_dict = zip(self.strat.model.conv_module.state_dict().keys(), parameters)
		state_dict = OrderedDict({k: torch.tensor(v) for k, v in params_dict})
		self.strat.model.conv_module.load_state_dict(state_dict, strict=True)
	
	def set_parameters_fc(self, parameters: List[np.ndarray]) -> None:
		print(f"[Client {self.cid}] set_parameters FC")
		# self.net.fc_module.train()
		self.strat.model.fc_module.train()
		params_dict = zip(self.strat.model.fc_module.state_dict().keys(), parameters)
		state_dict = OrderedDict({k: torch.tensor(v) for k, v in params_dict})
		self.strat.model.fc_module.load_state_dict(state_dict, strict=True)
	
	def fit(self, parameters, config):
		print(f"[Client {self.cid}] fit, config: {config}")
		# set_parameters(self.net.conv_module, parameters)
		init_ram=RAMU().compute("TRAINING")

		self.set_parameters(parameters)
		try:
			with open(f'{self.path}/mod{self.cid}.pkl', 'rb') as f:
				state_dict = pickle.load(f)
			self.strat.model.fc_module.load_state_dict(state_dict, strict=True)

			# set_parameters(self.net.fc_module, state_dict)
		except:
			print('')
		# self.strat.load_model(self.get_parameters_all({}))
		
		try:
			with open(f'{self.path}/task{int(self.cid)}.txt', 'r') as f:  # Python 3: open(..., 'rb')
				task_count = int(f.readline())
		except:
			task_count = 0
		try:
			with open(f'{self.path}/reg{int(self.cid)}.pkl', 'rb') as f:  # Python 3: open(..., 'rb')
				reg_term = pickle.load(f)
		except:
			reg_term = {}
		if config["server_round"] < int(self.nrounds / 2):
			# self.net = copy.deepcopy(self.strat.learn_batch(task_count, reg_term, self.trainloaders[0][int(self.cid)], learn_ewc=False))
			peak_ram=self.strat.learn_batch(task_count, reg_term, self.trainloaders[0][int(self.cid)], learn_ewc=False)
			reg_term = self.strat.regularization_terms
		
		elif config["server_round"] == int(self.nrounds / 2):
			task_count += 1
			# self.net = copy.deepcopy(self.strat.learn_batch(task_count, reg_term, self.trainloaders[0][int(self.cid)], learn_ewc=True))
			peak_ram = self.strat.learn_batch(task_count, reg_term, self.trainloaders[0][int(self.cid)], learn_ewc=True)
			reg_term = self.strat.regularization_terms
			
		else:
			# self.net = copy.deepcopy(self.strat.learn_batch(task_count, reg_term, self.trainloaders[1][int(self.cid)], learn_ewc=False))
			peak_ram = self.strat.learn_batch(task_count, reg_term, self.trainloaders[1][int(self.cid)], learn_ewc=False)
			reg_term = self.strat.regularization_terms
		
		with open(f'{self.path}/reg{int(self.cid)}.pkl', 'wb') as f:  # Python 3: open(..., 'wb')
				pickle.dump(reg_term, f)
		with open(f'{self.path}/task{int(self.cid)}.txt', 'w+') as f:  # Python 3: open(..., 'wb')
			f.write(f'{task_count}')
		# state_dict = get_parameters(self.net.fc_module)
		if not os.path.exists(f'{self.path}/clientwise'):
			os.makedirs(f'{self.path}/clientwise')
		with open(f'{self.path}/clientwise/ram{int(self.cid)}.csv', 'a+') as f:
			f.write(f'{config["server_round"]},{init_ram},{peak_ram},{peak_ram-init_ram}\n')
		state_dict = self.get_parameters_fc(config)

		with open(f'{self.path}/mod{self.cid}.pkl', 'wb') as f:
			pickle.dump(state_dict, f)
		if config['server_round'] <= int(self.nrounds / 2):
			return self.get_parameters(config={}), len(self.trainloaders[0][int(self.cid)]), {}
		return self.get_parameters(config={}), len(self.trainloaders[1][int(self.cid)]), {}
	
	def evaluate(self, parameters, config):
		if not os.path.exists(f'{self.path}/clientwise'):
			os.makedirs(f'{self.path}/clientwise')
		print(f"[Client {self.cid}] evaluate, config: {config}")
		# set_parameters(self.net.conv_module, parameters)
		self.set_parameters(parameters)
		try:
			with open(f'{self.path}/mod{self.cid}.pkl', 'rb') as f:
				state_dict = pickle.load(f)
			# set_parameters(self.net.fc_module, state_dict)
			self.set_parameters_fc(state_dict)
		except:
			print('')
		if config["server_round"] <= int(self.nrounds / 2):
			loss, avg_pearson, avg_rmse = test(self.strat.model, self.valloader[0], self.y_labels, self.DEVICE)
		else:
			loss, avg_pearson, avg_rmse = test(self.strat.model, self.testloader, self.y_labels, self.DEVICE)
		# append the results to a file
		with open(f'{self.path}/clientwise/results{int(self.cid)}.txt', 'a+') as f:  # Python 3: open(..., 'wb')
			f.write(f'{config["server_round"]},{loss},{avg_pearson},{avg_rmse}\n')
		if config['server_round'] <= int(self.nrounds / 2):
			return float(loss), len(self.valloader[0]), {"avg_pearson_score": avg_pearson, "avg_rmse": avg_rmse}
		return float(loss), len(self.testloader), {"avg_pearson_score": avg_pearson, "avg_rmse": avg_rmse}


class FlowerClient_NR_Root(fl.client.NumPyClient):
	def __init__(self, cid, net, trainloader, valloader, testloader, epochs, y_labels, cl_strategy, agent_config, nrounds, path, DEVICE, num_clients,
				 strat_name, params):
		self.cid = cid
		# self.net = net
		self.trainloaders = trainloader
		self.valloader = valloader
		self.testloader = testloader
		self.epochs = epochs
		self.y_labels = y_labels
		self.strat = cl_strategy(agent_config, net, params, fedroot=True)
		self.nrounds = nrounds
		self.path = path
		self.DEVICE = DEVICE
		self.num_clients = num_clients
		self.strat_name = strat_name
	
	def get_parameters(self, config):
		print(f"[Client {self.cid}] get_parameters")
		self.strat.model.conv_module.train()
		return [val.cpu().numpy() for _, val in self.strat.model.conv_module.state_dict().items()]
	
	def get_parameters_fc(self, config):
		print(f"[Client {self.cid}] get_parameters")
		self.strat.model.fc_module.train()
		return [val.cpu().numpy() for _, val in self.strat.model.fc_module.state_dict().items()]
	
	def set_parameters(self, parameters: List[np.ndarray]) -> None:
		self.strat.model.conv_module.train()
		params_dict = zip(self.strat.model.conv_module.state_dict().keys(), parameters)
		state_dict = OrderedDict({k: torch.tensor(v) for k, v in params_dict})
		self.strat.model.conv_module.load_state_dict(state_dict, strict=True)
	
	def set_parameters_fc(self, parameters: List[np.ndarray]) -> None:
		self.strat.model.fc_module.train()
		params_dict = zip(self.strat.model.fc_module.state_dict().keys(), parameters)
		state_dict = OrderedDict({k: torch.tensor(v) for k, v in params_dict})
		self.strat.model.fc_module.load_state_dict(state_dict, strict=True)
	
	def fit(self, parameters, config):
		init_ram=RAMU().compute("TRAINING")

		# set_parameters(self.net, parameters)
		print(f"[Client {self.cid}] fit, config: {config}")
		# set_parameters(self.net.conv_module, parameters)
		self.set_parameters(parameters)
		try:
			with open(f'{self.path}/mod{self.cid}.pkl', 'rb') as f:
				state_dict = pickle.load(f)
			# set_parameters(self.net.fc_module, state_dict)
			self.set_parameters_fc(state_dict)
		except:
			print('')
		# self.strat.model = copy.deepcopy(self.net)
		# print(config.keys())
		try:
			with open(f'{self.path}/task{int(self.cid)}.txt', 'r') as f:  # Python 3: open(..., 'rb')
				task_count = int(f.readline())
		except:
			task_count = 0
		
		if config["server_round"] < int(self.nrounds / 2):
			# self.net = copy.deepcopy(self.strat.learn_batch(task_count, {}, self.trainloaders[0][int(self.cid)], False))
			peak_ram=self.strat.learn_batch(task_count, {}, self.trainloaders[0][int(self.cid)], False)
			memory = self.strat.task_memory
		elif config["server_round"] == int(self.nrounds / 2):
			print("here\n")
			task_count += 1
			# self.net = copy.deepcopy(self.strat.learn_batch(task_count, {}, self.trainloaders[0][int(self.cid)], True))
			peak_ram=self.strat.learn_batch(task_count, {}, self.trainloaders[0][int(self.cid)], True)
			
			with open(f'{self.path}/reg{int(self.cid)}.pkl', 'wb') as f:  # Python 3: open(..., 'wb')
				pickle.dump(self.strat.task_memory[task_count].storage, f)
		else:
			with open(f'{self.path}/reg{int(self.cid)}.pkl', 'rb') as f:  # Python 3: open(..., 'rb')
				stor = pickle.load(f)
			memory = {task_count: Memory()}
			memory[task_count].update(stor)
			# self.net = copy.deepcopy(self.strat.learn_batch(task_count, memory, self.trainloaders[1][int(self.cid)], False))
			peak_ram=self.strat.learn_batch(task_count, memory, self.trainloaders[1][int(self.cid)], False)
			memory = self.strat.task_memory
			if config["server_round"] == int(self.nrounds):
				if os.path.exists(f'{self.path}/reg{int(self.cid)}.pkl'):
					os.remove(f'{self.path}/reg{int(self.cid)}.pkl')
		
		with open(f'{self.path}/task{int(self.cid)}.txt', 'w+') as f:  # Python 3: open(..., 'wb')
			f.write(f'{task_count}')
		# state_dict = get_parameters(self.net.fc_module)
		state_dict = self.get_parameters_fc(config)
		
		with open(f'{self.path}/mod{self.cid}.pkl', 'wb') as f:
			pickle.dump(state_dict, f)
		if not os.path.exists(f'{self.path}/clientwise'):
			os.makedirs(f'{self.path}/clientwise')
		with open(f'{self.path}/clientwise/ram{int(self.cid)}.csv', 'a+') as f:
			f.write(f'{config["server_round"]},{init_ram},{peak_ram},{peak_ram-init_ram}\n')
		if config['server_round'] <= int(self.nrounds / 2):
			return self.get_parameters(config={}), len(self.trainloaders[0][int(self.cid)]), {}
		return self.get_parameters(config={}), len(self.trainloaders[1][int(self.cid)]), {}
	
	def evaluate(self, parameters, config):
		# check if {self.path}/{self.num_clients} exists, if not create it
		if not os.path.exists(f'{self.path}/clientwise'):
			os.makedirs(f'{self.path}/clientwise')
		print(f"[Client {self.cid}] evaluate, config: {config}")
		# set_parameters(self.net.conv_module, parameters)
		self.set_parameters(parameters)
		try:
			with open(f'{self.path}/mod{self.cid}.pkl', 'rb') as f:
				state_dict = pickle.load(f)
			# set_parameters(self.net.fc_module, state_dict)
			self.set_parameters_fc(state_dict)
		except:
			print('')
		if config["server_round"] <= int(self.nrounds / 2):
			loss, avg_pearson, avg_rmse = test(self.strat.model, self.valloader[0], self.y_labels, self.DEVICE)
		else:
			loss, avg_pearson, avg_rmse = test(self.strat.model, self.testloader, self.y_labels, self.DEVICE)
		# append the results to a file
		with open(f'{self.path}/clientwise/results{int(self.cid)}.txt', 'a+') as f:  # Python 3: open(..., 'wb')
			f.write(f'{config["server_round"]},{loss},{avg_pearson},{avg_rmse}\n')
		if config['server_round'] <= int(self.nrounds / 2):
			return float(loss), len(self.valloader[0]), {"avg_pearson_score": avg_pearson, "avg_rmse": avg_rmse}
		return float(loss), len(self.testloader), {"avg_pearson_score": avg_pearson, "avg_rmse": avg_rmse}


class FlowerClient_LGR(fl.client.NumPyClient):
	def __init__(self, cid, net, trainloader, valloader, testloader, epochs, y_labels, cl_strategy, agent_config, nrounds, path, DEVICE, num_clients,
				 strat_name, params, gr):
		self.cid = cid
		# self.net = net
		self.trainloaders = trainloader
		self.valloader = valloader
		self.testloader = testloader
		self.epochs = epochs
		self.y_labels = y_labels
		self.strat = cl_strategy(agent_config, net, params, gr, path, cid)
		self.gr=gr
		self.nrounds = nrounds
		self.path = path
		self.DEVICE = DEVICE
		self.num_clients = num_clients
		self.strat_name = strat_name
	
	def get_parameters(self, config):
		print(f"[Client {self.cid}] get_parameters")
		self.strat.model.conv_module.train()
		return [val.cpu().numpy() for _, val in self.strat.model.conv_module.state_dict().items()]
	
	def get_parameters_fc(self, config):
		print(f"[Client {self.cid}] get_parameters")
		self.strat.model.fc_module.train()
		return [val.cpu().numpy() for _, val in self.strat.model.fc_module.state_dict().items()]
	
	def set_parameters(self, parameters: List[np.ndarray]) -> None:
		self.strat.model.conv_module.train()
		params_dict = zip(self.strat.model.conv_module.state_dict().keys(), parameters)
		state_dict = OrderedDict({k: torch.tensor(v) for k, v in params_dict})
		self.strat.model.conv_module.load_state_dict(state_dict, strict=True)
	
	def set_parameters_fc(self, state_dict) -> None:
		self.strat.model.fc_module.train()
		# params_dict = zip(self.net.fc_module.state_dict().keys(), parameters)
		# state_dict = OrderedDict({k: torch.tensor(v) for k, v in params_dict})
		self.strat.model.fc_module.load_state_dict(state_dict, strict=True)
	
	def fit(self, parameters, config):
		print(f"[Client {self.cid}] fit, config: {config}")
		init_ram=RAMU().compute("TRAINING")

		# set_parameters(self.net.conv_module, parameters)
		self.set_parameters(parameters)
		try:
			with open(f'{self.path}/mod{self.cid}.pkl', 'rb') as f:
				state_dict = pickle.load(f)
			self.set_parameters_fc(state_dict)

			# set_parameters(self.net.fc_module, state_dict)
		except:
			print('')
		# self.strat.model = copy.deepcopy(self.net)
		
		try:
			with open(f'{self.path}/task{int(self.cid)}.txt', 'r') as f:  # Python 3: open(..., 'rb')
				task_count = int(f.readline())
			print("----------task counted---------")
		except:
			task_count = 0
		try:
			with open(f'{self.path}/gen{int(self.cid)}.pkl', 'rb') as f:  # Python 3: open(..., 'rb')
				reg_term = pickle.load(f)
			self.gr.load_state_dict(state_dict, strict=True)
		except:
			reg_term=self.gr.state_dict()

		num_tasks = 6
		rounds_per_task = int(self.nrounds / num_tasks)  # 30 rounds / 6 tasks = 5 rounds/task
		# Current task index (0-indexed)
		current_task = (config["server_round"] - 1) // rounds_per_task
		# Round number *within* the current task (0 to rounds_per_task-1)
		round_within_task = (config["server_round"] - 1) % rounds_per_task

		if current_task == 0:
			if round_within_task < (rounds_per_task - 1):
				peak_ram = self.strat.learn_batch(task_count, reg_term, self.trainloaders[0][int(self.cid)], learn_gen=False)
				reg_term = self.strat.get_generator_weights()
			else:
				# last task round - train gen and increment task
				peak_ram = self.strat.learn_batch(task_count, reg_term, self.trainloaders[0][int(self.cid)], learn_gen=True)
				reg_term = self.strat.get_generator_weights()
				task_count += 1
		elif 0 < current_task < num_tasks-1:
			if round_within_task < (rounds_per_task - 1):
				peak_ram = self.strat.learn_batch(task_count, reg_term, self.trainloaders[current_task][int(self.cid)], learn_gen=False)
				reg_term = self.strat.get_generator_weights()
			else:
				#last task round - train gen and increment task
				peak_ram = self.strat.learn_batch(task_count, reg_term, self.trainloaders[current_task][int(self.cid)], learn_gen=True)
				reg_term = self.strat.get_generator_weights()
				task_count += 1
		elif current_task == num_tasks-1:
			# last task no gen train
			peak_ram = self.strat.learn_batch(task_count, reg_term, self.trainloaders[num_tasks-1][int(self.cid)], learn_gen=False)

		with open(f'{self.path}/gen{int(self.cid)}.pkl', 'wb') as f:  # Python 3: open(..., 'wb')
				pickle.dump(reg_term, f)
		with open(f'{self.path}/task{int(self.cid)}.txt', 'w+') as f:  # Python 3: open(..., 'wb')
			f.write(f'{task_count}')
		# state_dict = get_parameters(self.net.fc_module)
		state_dict = self.get_parameters_fc(config)
		if not os.path.exists(f'{self.path}/clientwise'):
			os.makedirs(f'{self.path}/clientwise')
		with open(f'{self.path}/clientwise/ram{int(self.cid)}.csv', 'a+') as f:
			f.write(f'{config["server_round"]},{init_ram},{peak_ram},{peak_ram-init_ram}\n')
		with open(f'{self.path}/mod{self.cid}.pkl', 'wb') as f:
			pickle.dump(state_dict, f)

		return self.get_parameters(config={}), len(self.trainloaders[current_task][int(self.cid)]), {}
	
	def evaluate(self, parameters, config):
		if not os.path.exists(f'{self.path}/clientwise'):
			os.makedirs(f'{self.path}/clientwise')
		print(f"[Client {self.cid}] evaluate, config: {config}")
		# set_parameters(self.net.conv_module, parameters)
		self.set_parameters(parameters)
		try:
			with open(f'{self.path}/mod{self.cid}.pkl', 'rb') as f:
				state_dict = pickle.load(f)
			# set_parameters(self.net.fc_module, state_dict)
			self.set_parameters_fc(state_dict)
		except:
			print('')

		torch.save(self.strat.model.state_dict(), f'{self.path}/fullmod{self.cid}.pth')
		num_tasks = 6
		rounds_per_task = int(self.nrounds / num_tasks)
		current_task = (config["server_round"] - 1) // rounds_per_task
		loss, avg_pearson, avg_rmse = test(self.strat.model, self.valloader[current_task], self.y_labels, self.DEVICE)

		# append the results to a file
		with open(f'{self.path}/clientwise/results{int(self.cid)}.txt', 'a+') as f:  # Python 3: open(..., 'wb')
			f.write(f'{config["server_round"]},{loss},{avg_pearson},{avg_rmse}\n')
		return float(loss), len(self.valloader[current_task]), {"avg_pearson_score": avg_pearson, "avg_rmse": avg_rmse}

