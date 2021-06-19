import numpy as np
from scipy.special import softmax


def calibrate(val_logits, val_y, logits):
  """Temperature scaling using the Newton's method
  :parameter
  val_logits: logits for the validation set
  val_y: true lable for the validation set
  logits: logits for the test set
  :return
  calibrated confidence for the given test set
  """
  val_confs = softmax(val_logits, axis=1)
  t = _calibrate(val_confs, val_y)

  test_confs = softmax(logits, axis=1)
  cal_logits = _temp_scale(test_confs, t)

  return softmax(cal_logits, axis=1)


def _calibrate(conf, true_idx, tol=1e-6, max_iter=30, num_guess=100):
  conf = conf.transpose()
  conf = np.maximum(conf, 1e-16)
  x = np.log(conf)

  xt = x[true_idx, np.arange(x.shape[1])]
  xt = np.expand_dims(xt, axis=0)

  cal = np.linspace(start=0.1, stop=10, num=num_guess)

  for j in range(len(cal)):
    for n in range(max_iter):
      f1 = np.sum(xt - np.divide(np.sum(np.multiply(x, np.exp(cal[j] * x)), 0), np.sum(np.exp(cal[j] * x), 0)))
      f2 = np.sum(np.divide(-np.sum(np.multiply(np.square(x), np.exp(cal[j] * x)), 0),
                            np.sum(np.exp(cal[j] * x), 0))
                  + np.divide(np.square(np.sum(np.multiply(x, np.exp(cal[j] * x)), 0)),
                              np.square(np.sum(np.exp(cal[j] * x), 0))))

      cal[j] = cal[j] - f1 / f2
      if np.isnan(f1) or np.isnan(f2):
        break
      if np.abs(f1 / f2) < tol:
        break
  cal = np.append([1], cal, axis=0)
  f0 = np.zeros(cal.shape)
  for j in range(len(cal)):
    f0[j] = np.sum(-np.log(np.divide(np.exp(cal[j] * xt), np.sum(np.exp(cal[j] * x), 0))), 1)

  n = np.nanargmin(f0)
  cal = cal[n]
  if n == 0:
    print("calibration failed")

  return cal


def _temp_scale(old_confs, t):
  # take old confidence as logits and return new logits
  temp = np.ones_like(old_confs) * t
  conf = old_confs
  conf = np.maximum(conf, 1e-16)
  return np.log(conf) * temp
