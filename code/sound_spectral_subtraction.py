import numpy as np
from scipy import signal


def get_sound_scaled_to_one(x_in: np.array):
  x = x_in

  maxsx = np.max(x)
  minsx = np.min(x)

  if maxsx != 1.0 or minsx != -1.0:
    rangesx = maxsx - minsx
    x = 2 * x / rangesx
    newmax = np.max(x)
    offset = newmax - 1.0
    x = x - offset

  return x


def get_soud_spec_sub (x: np.array):
  _, _, st = signal.stft(x)

  stmags = np.multiply(np.abs(st), np.abs(st))
  stangles = np.angle(st)

  hold_col = np.quantile(stmags, 0.5, axis=1)

  for q in range(stmags.shape[1]):
    stmags[:, q] -= hold_col
    stmags[:, q] = np.maximum(stmags[:,q], 0.0)

  stmags = np.sqrt(stmags)
  newst = np.multiply(stmags, 1j * stangles)

  _, new_x = signal.istft(newst)

  new_x = get_sound_scaled_to_one(new_x)

  return new_x
