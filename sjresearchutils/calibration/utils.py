import matplotlib.pyplot as plt
import numpy as np


def plot_reliability_diagram(ece_obj):
  mat = ece_obj.get_ECE_mat()
  lb = ece_obj.bin_lowers
  ub = ece_obj.bin_uppers
  cb = (lb + ub) / 2

  fig, ax1 = plt.subplots()

  ax2 = ax1.twinx()

  ax1.bar(cb, mat[:, 1], width=ub[:] - lb[:], edgecolor='k')
  diff = mat[:, 2] - mat[:, 1]
  diff[mat[:, 0] == 0] = 0

  ax1.bar(cb, diff, width=ub[:] - lb[:], bottom=mat[:, 1], edgecolor='r', color=(1, 0, 0, 0.2))
  plt.legend(['Output', 'Gap'])
  plt.plot([0, 1], [0, 1], linestyle='--')
  plt.xlabel('Confidence')
  ax1.set_ylabel('Accuracy')

  ax2.set_ylim([0, 1.0])
  ax2.set_ylabel('Example Percentage')
  ax2.scatter(cb, mat[:, 0], color='r')


def plot_confidence_dist(confs, n_bins=15):
  bin_boundaries = np.linspace(0, 1, n_bins + 1)
  cnt, bin_edges = np.histogram(confs, bins=bin_boundaries)

  plt.figure()
  plt.bar(bin_edges[:-1], height=cnt / confs.shape[0], width=bin_edges[1:] - bin_edges[:-1])
  plt.xlim([0.0, 1.0])
  plt.xlabel('Confidence')
  plt.ylabel('Percentages')


if __name__ == '__main__':
  from sjresearchutils.calibration.Ece import ECE
  import numpy as np
  e = ECE(15)

  y_pred = np.array([0, 1, 2, 3, 4, 5, 1, 2])
  y_true = np.array([0, 1, 2, 2, 2, 3, 1, 2])
  conf = np.array([0.4, 0.2, 0.3, 0.5, 0.3, 0.7, 0.8, 0.3])

  e.add_data(y_pred, y_true, conf)
  c = e.compute_ECE()
  print(c)

  plot_reliability_diagram(e)

  plot_confidence_dist(conf)

  plt.show()
