import cv2 as cv
import math
import numpy as np
import pyrtools as pt
from scipy import signal

from .sound_spectral_subtraction import get_scaled_sound


# This function allign v1 and v2 vectors: it is the formula (4) of paper
def align_vectors(v1: np.array, v2: np.array):
  acorb = np.convolve(v1, np.flip(v2))

  maxind = np.argmax(acorb)

  shift = v2.size - maxind
  out = np.roll(v1, shift)

  return out


def sound_from_video(video: cv.VideoCapture, nscale, norientation, downsample_factor=1, nframes=None, sampling_rate=None):
  if sampling_rate is None:
    sampling_rate = video.get(cv.CAP_PROP_FPS)

  ret, frame = video.read()

  if downsample_factor < 1:
    colorframe = cv.resize(frame, (0, 0), fx=downsample_factor, fy=downsample_factor)
  else:
    colorframe = frame

  # Converting the first frame to gray
  grayframe = cv.cvtColor(colorframe, cv.COLOR_BGR2GRAY)
  norm_frame = cv.normalize(grayframe.astype('float'), None, 0.0, 1.0, cv.NORM_MINMAX)

  first_frame = norm_frame

  if nframes is None:
    nframes = int(video.get(cv.CAP_PROP_FRAME_COUNT))

  # Creating StreerablePyramid of the first frame
  first_pyramid = pt.pyramids.SteerablePyramidFreq(first_frame, nscale, norientation - 1, is_complex=True)
  first_piramid = first_pyramid.pyr_coeffs

  # Creating an empty copy of pyramid bands
  signals = {b: list() for b in first_piramid.keys()}

  # Iteration over frames
  while ret:
    if downsample_factor < 1:
      frame = cv.resize(frame, (0,0), fx=downsample_factor, fy=downsample_factor)

    # Creating a grey frame
    grayframe = cv.cvtColor(frame, cv.COLOR_BGR2GRAY)
    norm_frame = cv.normalize(grayframe.astype('float'), None, 0.0, 1.0, cv.NORM_MINMAX)

    # Creating StreerablePyramid of the frame
    pyramid = pt.pyramids.SteerablePyramidFreq(norm_frame, nscale, norientation - 1, is_complex=True)

    # Can be replaced with this
    # pyramid = pt.pyramids.SteerablePyramidSpace(norm_frame, nscale, norientation - 1)

    pyramid = pyramid.pyr_coeffs

    # Making all bands positive to build the pyramide amplitude
    amp_pyramid = dict()
    for band, coeffs in pyramid.items():
      amp_pyramid[band] = np.abs(coeffs)

    # We have that np.angle return for each complex number the angle (in radiant) of the vector that the complex number form over (Real, i) space.
    # We calculate the differences of the angle of the bands between the current frame and the first frame of the image.
    # Formula (2) of the paper
    dphase_pyramid = dict()
    for band, coeffs in pyramid.items():
      first_coeffs = first_piramid[band]
      dphase_pyramid[band] = np.mod(math.pi + np.angle(coeffs) - np.angle(first_coeffs), 2 * math.pi) - math.pi

    for band in pyramid.keys():
      amp = amp_pyramid[band]
      phase = dphase_pyramid[band]

      # Here we have the formula (3) of the paper where we compute a sigle motion signal
      sms = np.multiply(phase, np.multiply(amp, amp))

      # Here we do the mean
      total_amp = np.sum(amp.flatten())
      signals[band].append(np.mean(sms.flatten()) / total_amp)

    ret, frame = video.read()

  # Here we do the formula (4) and (5) of the paper where we allign the signals and after that we do the sum
  sound = np.zeros(nframes)
  for sig in signals.values():
    sig_aligned = align_vectors(np.array(sig), np.array(signals[(0, 0)]))  # With "residual_lowpass" same result

    sound += sig_aligned

  # b, a = signal.butter(3, 0.05, btype='highpass')
  # x = signal.lfilter(b, a, sound)

  # More stable filter
  sos = signal.butter(3, 0.05, btype='highpass', output='sos')
  filtered_sound = signal.sosfilt(sos, sound)

  filtered_sound = get_scaled_sound(filtered_sound)

  return filtered_sound
