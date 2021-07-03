import torch


class NumpyDataset(torch.utils.data.Dataset):
  def __init__(self, xs, ys):
    self.xs = xs
    self.ys = ys.astype(int)

  def __len__(self):
    return self.ys.shape[0]

  def __getitem__(self, i):
    return self.xs[i, :], self.ys[i]
