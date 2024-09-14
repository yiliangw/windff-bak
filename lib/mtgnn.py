import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import dgl
from dgl.nn import GraphConv
import logging

from easydict import EasyDict as edict


class GCN(nn.Module):
  def __init__(self, in_feats, h_feats, num_classes):
    super(GCN, self).__init__()
    self.conv1 = GraphConv(in_feats, h_feats)
    self.conv2 = GraphConv(h_feats, num_classes)

  def forward(self, g, in_feat):
    h = self.conv1(g, in_feat)
    h = F.relu(h)
    h = self.conv2(g, h)
    return h


class Linear(nn.Module):
  def __init__(self, c_in, c_out, bias=True):
    super(Linear, self).__init__()
    self.mlp = torch.nn.Conv2d(c_in, c_out, kernel_size=(
        1, 1), padding=(0, 0), stride=(1, 1), bias=bias)

  def forward(self, x):
    return self.mlp(x)


class Prop(nn.Module):
  def __init__(self, c_in, c_out):
    super(Prop, self).__init__()

  def forward(self)


class MTGNN(nn.Module):
  def __init__(self, config: edict):
    super(MTGNN, self).__init__()

    self.logger = logging.getLogger()

    self.feat_dim = config.feat_dim
    self.label_dim = config.label_dim
    self.input_dim = self.feat_dim + self.label_dim
    self.output_dim = self.label_dim

    self.data_diff = config.get("data_diff", True)
    self.input_len = config.input_len
    self.output_len = config.output_len

    # Hyperparameters
    self.layers = config.get("layers", 3)
    self.dropout_rate = config.get("dropout_rate", 0.3)

    self.kernel_len = config.get("kernel_len", 7)
    self.dialation_exponential = config.get("dialation_exponential", 2)
    if self.dialation_exponential > 1:
      self.receptive_len = int(
          self.output_dim +
          (self.kernel_len - 1) *
          (self.dialation_exponential ** self.layers - 1) /
          (self.dialation_exponential - 1)
      )
    else:
      self.receptive_len = self.layers * \
          (self.kernel_len - 1) + self.output_dim

    self.logger.info("receptive_len: ", self.receptive_len)

    self.conv_channels = config.get("conv_channels", 32)
    self.residual_channels = config.get("residual_channels", 32)
    self.skip_channels = config.get("skip_channels", 64)
    self.end_channels = config.get("end_channels", 128)

    # Layers
    self.l_conv_start = nn.Conv2d(
        in_channels=self.input_dim,
        out_channels=self.residual_channels,
        kernel_size=(1, 1)
    )

    if self.input_len > self.receptive_len:
      self.l_skip_start = nn.Conv2d(
          in_channels=self.input_dim,
          out_channels=self.skip_channels,
          kernel_size=(1, self.input_len),
          bias=True
      )
      self.l_skip_end = nn.Conv2d(
          in_channels=self.residual_channels,
          out_channels=self.skip_channels,
          kernel_size=(1, self.input_len - self.receptive_len + 1),
          bias=True
      )
    else:
      self.l_skip_start = nn.Conv2d(
          in_channels=self.input_dim,
          out_channels=self.skip_channels,
          kernel_size=(1, self.receptive_len),
          bias=True
      )
      self.l_skip_end = nn.Conv2d(
          in_channels=self.residual_channels,
          out_channels=self.skip_channels,
          kernel_size=(1, 1),
          bias=True
      )

    self.l_filter_convs = nn.ModuleList()
    self.l_gate_convs = nn.ModuleList()
    self.l_skip_convs = nn.ModuleList()
    self.l_graph_convs_1 = nn.ModuleList()
    self.l_graph_convs_2 = nn.ModuleList()
    self.l_norms = nn.ModuleList()

    self.l_conv_end_1 = nn.Conv2d(
        in_channels=self.skip_channels,
        out_channels=self.end_channels,
        kernel_size=(1, 1),
        bias=True
    )
    self.l_conv_end_2 = nn.Conv2d(
        in_channels=self.end_channels,
        out_channels=self.output_len,
        kernel_size=(1, 1),
        bias=True
    )

  def forward(self, g: dgl.DGLGraph, x_feat: torch.Tensor, x_label: torch.Tensor, feat_mean: torch.Tensor, label_mean: torch.Tensor, feat_scale: torch.Tensor, label_scale: torch.Tensor):
    B_, N_, T_, F_ = x_feat.size()
    B1_, N1_, T1_, L_ = x_label.size()

    # Check input shape
    assert B_ == B1_
    assert N_ == N1_
    assert T_ == T1_ and T_ == self.input_len
    assert F_ == self.feat_dim
    assert L_ == self.label_dim

    '''
    x: (B, N, T, D) (batch_sz, node_num, input_len, dim(features and labels))
    mean: (D)
    scale: (D)
    '''
    x = torch.cat([x_feat, x_label], dim=-1)

    # Normalize input
    x = (x - torch.cat(label_mean, feat_mean)
         [None, None, None, :]) / torch.cat(label_scale, feat_scale)[None, None, None, :]

    x = x.permute((0, 3, 1, 2))  # (B, D, N, T)

    skip = self.l_skip_start(
        F.dropout(x, self.dropout_rate, training=self.training))

    x = self.l_conv_start(x)  # (B, residual_channels, N, T)
    for i in range(self.layers):
      r = x
      filter = self.l_filter_convs[i](x)
      filter = torch.tanh(filter)
      gate = self.l_gate_convs[i](x)
      gate = torch.sigmoid(gate)
      x = filter * gate
      x = F.dropout(x, self.dropout_rate, training=self.training)
      s = self.l_skip_convs[i](x)
      skip = s + skip
      x = self.l_graph_convs_1[i](g, x) + self.l_graph_convs_2[i](g, x)

      x = x + r
      x = self.l_norms[i](x)

    skip = self.l_skip_end(x) + skip
    x = F.relu(skip)
    x = self.l_conv_end_1(x)
    x = F.relu(x)
    x = self.l_conv_end_2(x)  # (B, output_len, N, D)
    x = x[..., :self.label_dim].permute((0, 2, 1))  # (B, N, output_len)

    return x
