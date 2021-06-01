import numpy as np


class ECE:
  def __init__(self, n_bin, dynamic=False):
    self.n_bin = n_bin
    self.dynamic = dynamic
    if dynamic:
      self.list = []
      self.bin_lowers = np.zeros(n_bin)
      self.bin_uppers = np.zeros(n_bin)
    else:
      bin_boundaries = np.linspace(0, 1, n_bin+1)
      self.bin_lowers = bin_boundaries[:-1]
      self.bin_uppers = bin_boundaries[1:]

    self.ece_mat = None
  
  def compute_acc_conf(self, y_pred, y_true, conf):
    acc = np.equal(y_true, y_pred)

    acc_conf = np.zeros((self.n_bin, 3))
    for i, (bin_lower, bin_upper) in enumerate(zip(self.bin_lowers, self.bin_uppers)):
        in_bin = (conf > bin_lower.item()) & (conf <= bin_upper.item())
        if i == 0:
          in_bin |= (conf == 0)
        acc_conf[i, 0] = in_bin.astype(float).sum()
        if acc_conf[i, 0] > 0:
            acc_conf[i, 1] = acc[in_bin].astype(float).sum()
            acc_conf[i, 2] = conf[in_bin].astype(float).sum()

    n_total_data = np.sum(acc_conf, axis=0)[0]
    assert n_total_data == y_true.shape[0]

    return acc_conf

  def split_dynamic_bin(self):
    acc_conf = np.zeros((self.n_bin, 3))
    samples_in_bin = len(self.list) // self.n_bin
    rest_samples = len(self.list) % self.n_bin
    end_idx = 0
    for i in range(self.n_bin):
      start_idx = end_idx
      end_idx = start_idx + samples_in_bin + (1 if i < rest_samples else 0)

      temp_arr = np.array(self.list[start_idx: end_idx])
      correct = (temp_arr[:, 0] == temp_arr[:, 1]).sum()
      conf = temp_arr[:, 2].sum()
      acc_conf[i, 0] = end_idx - start_idx
      acc_conf[i, 1] = correct
      acc_conf[i, 2] = conf

      self.bin_lowers[i] = temp_arr[:, 2].min()  # TODO: need to minus epsilon?
      self.bin_uppers[i] = temp_arr[:, 2].max()

    return acc_conf

  def get_ECE_mat(self):
    res_mat = np.copy(self.ece_mat)
    ind = res_mat[:, 0] > 0
    res_mat[ind, 1] = np.divide(res_mat[ind, 1], res_mat[ind, 0])
    res_mat[ind, 2] = np.divide(res_mat[ind, 2], res_mat[ind, 0])
    res_mat[:, 0] = np.divide(res_mat[:, 0], np.sum(res_mat[:, 0]))
    return res_mat

  def compute_ECE(self):
    res_mat = np.copy(self.ece_mat)
    ind = res_mat[:, 0] > 0
    res_mat[ind, 1] = np.divide(res_mat[ind, 1], res_mat[ind, 0])
    res_mat[ind, 2] = np.divide(res_mat[ind, 2], res_mat[ind, 0])
    res_mat[:, 0] = np.divide(res_mat[:, 0], np.sum(res_mat[:, 0]))
    res = np.sum(np.multiply(res_mat[:,0], np.absolute(res_mat[:,1]-res_mat[:,2])))
    return res

  def compute_MCE(self):
    res_mat = np.copy(self.ece_mat)
    ind = res_mat[:, 0] > 0
    res_mat[ind, 1] = np.divide(res_mat[ind, 1], res_mat[ind, 0])
    res_mat[ind, 2] = np.divide(res_mat[ind, 2], res_mat[ind, 0])
    res_mat[:, 0] = np.divide(res_mat[:, 0], np.sum(res_mat[:, 0]))
    res = np.max(np.absolute(res_mat[:, 1] - res_mat[:, 2]))
    return res

  def compute_VCE(self):
    ece = self.compute_ECE()
    res_mat = np.copy(self.ece_mat)
    ind = res_mat[:, 0] > 0
    res_mat[ind, 1] = np.divide(res_mat[ind, 1], res_mat[ind, 0])
    res_mat[ind, 2] = np.divide(res_mat[ind, 2], res_mat[ind, 0])
    res_mat[:, 0] = np.divide(res_mat[:, 0], np.sum(res_mat[:, 0]))
    res = np.sum(np.multiply(res_mat[:,0], np.square(np.absolute(res_mat[:,1]-res_mat[:,2]) - ece)))
    return res

  def add_data(self, y_pred, y_true, conf):
    if self.dynamic:
      cur_list = [(p, t, c,) for (p, t, c) in zip(y_pred, y_true, conf)]
      self.list = self.list + cur_list

      self.list = sorted(self.list, key=lambda x: x[2])
      self.ece_mat = self.split_dynamic_bin()
    else:
      temp_mat = self.compute_acc_conf(y_pred, y_true, conf)
      if self.ece_mat is None:
        self.ece_mat = temp_mat
      else:
        self.ece_mat = self.ece_mat + temp_mat


if __name__ == '__main__':
  e = ECE(15)

  y_pred = np.array([0,1,2,3,4,5,1,2])
  y_true = np.array([0,1,2,2,2,3,1,2])
  conf = np.array([0.4,0.2,0.3,0.5,0.3,0.7,0.8,0.3])

  e.add_data(y_pred,y_true, conf)
  print(e.ece_mat)
  c = e.compute_ECE()
  print(c)
