import os
import numpy as np
import pandas as pd
import dgl
from dgl.data import DGLDataset
import torch
from logging import getLogger


# Write a function to generate the above dictionary
def _generate_time_dict():
  time_dict = {}
  for i in range(24):
    for j in range(6):
      time_dict[f"{i:02d}:{j * 10:02d}"] = i * 6 + j
  return time_dict


time_dict = _generate_time_dict()


class WPFDataset(DGLDataset):
  def __init__(
      self,
      filepath,
      flag='train',
      size=None,
      capacity=134,
      day_len=24 * 6,
      train_days=214,  # 15 days
      val_days=16,  # 3 days
      test_days=15,  # 6 days
      total_days=245,  # 30 days
      theta=0.9,
      random=False,
      only_useful=False,
      graph_type='sem',
      weight_adj_epsilon=0.8,
      dtw_topk=5,
      binary=True,
  ):

    super().__init__('WPF Dataset')

    self.filepath = filepath
    self.unit_size = day_len
    self.train_days = train_days
    self.points_per_hour = day_len // 24
    self.random = random
    self.only_useful = only_useful
    self.dtw_topk = dtw_topk
    self.binary = binary
    if size is None:
      self.input_len = self.unit_size
      self.output_len = self.unit_size
    else:
      self.input_len = size[0]
      self.output_len = size[1]

    self.start_col = 0
    self.capacity = capacity
    self.theta = theta

    # initialization
    assert flag in ['train', 'test', 'val']
    type_map = {'train': 0, 'val': 1, 'test': 2}
    self.set_type = type_map[flag]
    self.flag = flag
    self.graph_type = graph_type
    self.weight_adj_epsilon = weight_adj_epsilon
    self._logger = getLogger()

    self.total_size = self.unit_size * total_days
    self.train_size = train_days * self.unit_size
    self.val_size = val_days * self.unit_size
    self.test_size = test_days * self.unit_size

    self.graph = None
    self.data = None

    self._read_data()

  def _read_data(self):

    df_raw = pd.read_csv(self.filepath)
    cols = [c for c in df_raw.columns if 'Day' not in c and 'Tmstamp' not in c]
    df = df_raw[cols]
    # Add a time column by assigning
    df.insert(0, 'time', df_raw['Tmstamp'].apply(lambda x: time_dict[x]))
    # df['Time'] = df_raw['Tmstamp'].apply(lambda x: time_dict[x])
    df = df.sort_values('time')
    df = df.replace(to_replace=np.nan, value=0)

    # Get all TurbID as integers
    turb_ids = df['TurbID'].unique()
    turb_ids = sorted(turb_ids)
    self.turb_ids = turb_ids

    feat_cols = [
        c for c in df.columns if 'Patv' not in c and 'time' not in c]
    label_cols = ['Patv']

    # Get the data for each TurbID
    assert df.shape[0] % len(turb_ids) == 0
    num_time = df.shape[0] // len(turb_ids)
    feat = torch.zeros(len(turb_ids), num_time,
                       len(feat_cols), dtype=torch.float32)
    label = torch.zeros(len(turb_ids), num_time,
                        len(label_cols), dtype=torch.float32)
    for i, turb_id in enumerate(turb_ids):
      df_turb = df[df['TurbID'] == turb_id]
      feat[i] = torch.tensor(df_turb[feat_cols].values)
      label[i] = torch.tensor(df_turb[label_cols].values)

    self.feat = feat
    self.label = label

    u_list = []
    v_list = []
    for u in range(len(turb_ids)):
      for v in range(len(turb_ids)):
        if u != v:
          u_list.append(u)
          v_list.append(v)

    self.graph = dgl.graph((u_list, v_list), num_nodes=len(turb_ids))
    # Currently, use a fully connected graph, with dummy features
    self.graph.edata['feat'] = torch.ones(self.graph.number_of_edges(), 1)

    self.graph.ndata['feat'] = feat
    self.graph.ndata['label'] = label

  def __getitem__(self, idx):
    # construct a dgl graph based on the edges of self.graph (dgl.graph)
    graph = dgl.graph(self.graph.edges())

    feat_start = idx
    feat_end = idx + self.input_len
    target_start = idx + self.input_len
    target_end = idx + self.input_len + self.output_len

    graph.ndata['x_feat'] = self.graph.ndata['feat'][:, feat_start:feat_end]
    graph.ndata['x_target'] = self.graph.ndata['target'][:, feat_start:feat_end]

    graph.ndata['y_feat'] = self.graph.ndata['feat'][:,
                                                     target_start:target_end]
    graph.ndata['y_target'] = self.graph.ndata['target'][:,
                                                         target_start:target_end]

    return graph

  def __len__(self):
    return self.graph.ndata['x'].shape[1] - self.input_len - self.output_len + 1
