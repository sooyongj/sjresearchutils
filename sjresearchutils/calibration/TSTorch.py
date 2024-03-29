import torch

from scipy.special import softmax
from math import sqrt


class TemperatureScaling(torch.nn.Module):
  def __init__(self, device):
    super(TemperatureScaling, self).__init__()
    self.device = device
    self.temperature = torch.nn.Parameter(torch.ones(1, device=device))

  def forward(self, logits):
    return self._temerature_scale(logits)

  def temperature_scale(self, logits):
    temperature = self.temperature.unsqueeze(1).expand(logits.size(0), logits.size(1))
    return logits / temperature

  def train(self, val_loader, epochs, weight=None):
    nll_criterion = torch.nn.CrossEntropyLoss(weight=weight).to(device=self.device)
    lr = 0.1
    optimizer = torch.optim.SGD([self.temperature], lr=lr)
    # optimizer = torch.optim.Adam([self.temperature], lr=lr)

    def scheduler(initial_lr, epoch):
      return initial_lr * (0.1 ** (epoch // int(epochs * 0.5))) * (0.1 ** (epoch // int(epochs * 0.75)))

    lr_scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=lambda e: scheduler(lr, e))

    for epoch in range(epochs):
      total_loss = 0
      for logits, ys in val_loader:
        logits, ys = logits.to(self.device), ys.to(self.device)

        optimizer.zero_grad()
        loss = nll_criterion(self.temperature_scale(logits), ys)
        loss.backward()
        optimizer.step()

        total_loss += loss

      print('Epoch: {} - Loss: {:.4f}'.format(epoch, total_loss))
      lr_scheduler.step()
    print("Final Temperature: {:.4f}".format(self.temperature.item()))

    return self


def tstorch_calibrate(val_logits, val_ys, logits, batch_size=100, epochs=100, device=None, balance=False):
  """
  Temperature scaling
  :param val_logits: (Numpy array) validation logits
  :param val_ys: (Numpy array) validation true labels
  :param logits: (Numpy array) test logits
  :param batch_size: (int) batch size
  :param epochs: (int) number of epochs
  :param device: (torch.device) whether GPU or CPU
  :param balance: (boolean) True if calibration considering the occurence of each class, False otherwise
  :return: (Numpy array) calibrated confidence distribution over labels
  """
  val_dataset = torch.utils.data.TensorDataset(torch.from_numpy(val_logits).float(),
                                               torch.from_numpy(val_ys).long()
                                               )
  val_loader = torch.utils.data.DataLoader(val_dataset,
                                           batch_size=batch_size,
                                           shuffle=False)

  model = TemperatureScaling(device)

  if balance:
    n_class = torch.unique(torch.from_numpy(val_ys)).size()[0]
    weight = torch.ones(n_class)
    for c in range(n_class):
      cnt = (val_ys == c).sum().item()
      weight[c] = 1 / sqrt(cnt)
    print("class weight:", weight)
  else:
    weight = None
  model.train(val_loader, epochs=epochs, weight=weight)

  with torch.no_grad():
    logits = torch.from_numpy(logits)
    return softmax(model.temperature_scale(logits).cpu().numpy(), axis=1)
