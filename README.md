# Setup
1. conda create -n lgr python=3.10
2. pip install "flwr[simulation]==1.7.0"
3. pip install "tensorflow==2.15.0"
4. pip install -r tf_requirements.txt
5. pip install -r base_requirements.txt
6. pip install "protobuf>=4.25.2,<5.0.0" (in case TensorFlow pulled protobuf==4.23.x)
7. pip install ipykernel
8. python -m ipykernel install --user --name lgr --display-name "lgr (py310)"





# Feature Aggregation with Latent Generative Replay for Federated Continual Learning of Socially Appropriate Robot Behaviours

## Overview
This is a PyTorch-based code implementation accompanying the IEEE RA-L submission [Feature Aggregation with Latent Generative Replay for Federated Continual Learning of Socially Appropriate Robot Behaviours](https://arxiv.org/abs/2405.15773). 

## Installation

Ensure you have the necessary dependencies installed. Run the following command to set up the environment:

```bash
pip install -r requirements.txt
```

## Dataset

Access to the Manners-DB data files can be requested here: https://github.com/jonastjoms/MANNERS-DB.
The csv file with the labels, once acquired, should be placed under ```Data/all_data.csv```. Currently, a dummy file is included for reference.
All images should be placed under ```Data/images/```. Currently, an ```empty.file``` is placed there for reference, please remove that before running your code.

## Training

### Federated Learning
FL strategies are implemented as python packages. To execute the code on MANNERS-DB run the following:

```bash
bash run_FL_local.sh
```
Make sure all necessary paths are provided correctly. 

### Federated Continual Learning
FCL strategies are alsoimplemented as python packages. To execute the code on MANNERS-DB run the following:

```bash
bash run_CL_local.sh
```
Make sure all necessary paths are provided correctly. 


## Citation

```
@misc{Churamani2024Feature,  
  author        = {N. {Churamani} and S. {Checker} and F.I. {Dogan} and H.T.L {Chiang} and H. {Gunes}},  
  title         = {{Feature Aggregation with Latent Generative Replay for Federated Continual Learning of Socially Appropriate Robot Behaviours}},   
  year          = {2025},  
  eprint        = {},
  archivePrefix = {arXiv},
  primaryClass  = {cs.LG}
 }
```

