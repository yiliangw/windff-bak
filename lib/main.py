from .mtgnn import MTGNN
from .wpf_dataset import WPFDataset
import dgl
from dgl import DGLGraph
import torch
import torch.functional as F


def train(g: DGLGraph, model):
  optimizer = torch.optim.Adam(model.parameters(), lr=0.01)

  for epoch in range(5):
    

if __name__ == '__main__':
  dataset = WPFDataset('data/WFP')
  model = MTGNN(dataset, 1, 1, 1, 1, 1, 1)
  model.train(1000)
