import numpy as np
from scipy import signal


# This function scale and center sound to the range [-1, 1]
def get_sound_scaled_to_one(sound: np.array):
  maxv = np.max(sound)
  minv = np.min(sound)

  if maxv != 1.0 or minv != -1.0:
    rangev = maxv - minv
    sound = 2 * sound / rangev
    newmax = np.max(sound)
    offset = newmax - 1.0
    sound -= offset

  return sound


# Function to improve sound using spectral subtraction. Adapted
# from the original work of Myers Abraham Davis (Abe Davis), MIT
def get_soud_spec_sub(sound: np.array, qtl=0.5):
  _, _, st = signal.stft(sound)

  st_mags = np.multiply(np.abs(st), np.abs(st))
  st_angles = np.angle(st)

  noise_floor = np.quantile(st_mags, qtl, axis=-1)

  for q in range(st_mags.shape[-1]):
    st_mags[:, q] -= noise_floor
    st_mags[:, q] = np.maximum(st_mags[:,q], 0.0)

  st_mags = np.sqrt(st_mags)
  newst = np.multiply(st_mags, 1j * st_angles)

  _, new_sound = signal.istft(newst)

  new_sound = get_sound_scaled_to_one(new_sound)

  return new_sound
