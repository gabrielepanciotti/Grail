# **Base**
import os
import time
import numpy as np
import pandas as pd

# **Visualizzazione**
import matplotlib.pyplot as plt
from matplotlib.colors import Normalize
from matplotlib import cm

# **Preprocessing**
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
from sklearn.neighbors import kneighbors_graph
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report, precision_score, recall_score, f1_score

# **PyTorch**
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset

# **PyTorch Geometric**
import torch_geometric
from torch_geometric.data import Data
from torch_geometric.loader import DataLoader as GraphDataLoader
from torch_geometric.nn import GCNConv, global_mean_pool

# **Funzioni di utilità PyTorch**
import torch.nn.functional as F

from contextlib import nullcontext
from sklearn.preprocessing import MinMaxScaler
import networkx as nx

