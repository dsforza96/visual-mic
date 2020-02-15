import cv2 as cv
import math
import numpy as np
import pyrtools as pt
from scipy import signal

from code.sound_spectral_subtraction import get_sound_scaled_to_one


def align_A2B(ax: np.array, bx: np.array):
  acorb = np.convolve(ax, np.flip(bx))

  maxind = np.argmax(acorb)

  shiftam = bx.size - maxind
  ax_out = np.roll(ax, shiftam)

  return ax_out, shiftam


def sound_from_video(v_hsandle: cv.VideoCapture, nscale, norientation, downsample_factor=1, nframes=None, sampling_rate=None):
  if sampling_rate is None:
    sampling_rate = v_hsandle.get(cv.CAP_PROP_FPS)

  ret, vframein = v_hsandle.read()

  if downsample_factor < 1:
    colorframe = cv.resize(vframein, (0,0), fx=downsample_factor, fy=downsample_factor)
  else:
    colorframe = vframein

  # Converting the first frame to Gray
  grayframe = cv.cvtColor(colorframe, cv.COLOR_BGR2GRAY)
  full_frame = cv.normalize(grayframe.astype('float'), None, 0.0, 1.0, cv.NORM_MINMAX)

  ref_frame = full_frame

  if nframes is None:
    nframes = int(v_hsandle.get(cv.CAP_PROP_FRAME_COUNT))

  # Creating StreerablePyramid of the first frame 
  pyr = pt.pyramids.SteerablePyramidFreq(ref_frame, nscale, norientation - 1, is_complex=True)
  pyr_ref = pyr.pyr_coeffs

  # Creating a zeros copy of pyramid bands
  signalffs = {b: list() for b in pyr_ref.keys()}

  # iteration over the frames
  while ret:
    if downsample_factor < 1:
      vframein = cv.resize(vframein, (0,0), fx=downsample_factor, fy=downsample_factor)

    # Create a grey frame
    grayframe = cv.cvtColor(vframein, cv.COLOR_BGR2GRAY)
    full_frame = cv.normalize(grayframe.astype('float'), None, 0.0, 1.0, cv.NORM_MINMAX)

    # Creating StreerablePyramid of the frame
    pyr = pt.pyramids.SteerablePyramidFreq(full_frame, nscale, norientation - 1, is_complex=True)
    pyr = pyr.pyr_coeffs

    # Make all bands positive to build the pyramide amplitude
    pyr_amp = dict()
    for band, matrix in pyr.items():
      pyr_amp[band] = np.abs(matrix)

    # We have that np.angle return for each complex number the angle (in radiant) of the vector that the complex number form over (Real, i) space. 
    # We calculate the differences of the angle of the bands between the current frame and the first frame of the image
    # Formula (2) of the paper
    pyr_delta_phase = dict()
    for band, matrix in pyr.items():
      matrix_ref = pyr_ref[band]
      pyr_delta_phase[band] = np.mod(math.pi + np.angle(matrix) - np.angle(matrix_ref), 2 * math.pi) - math.pi

    
    for band in pyr.keys():
      amp = pyr_amp[band]
      phase = pyr_delta_phase[band]

      # Here we have the formula (3) of the paper where we compute a sigle motion signal 
      phasew = np.multiply(phase, np.multiply(np.abs(amp), np.abs(amp)))
      sumamp = np.sum(np.abs(amp.flatten()))

      # Here we do the mean 
      signalffs[band].append(np.mean(phasew.flatten()) / sumamp)

    ret, vframein = v_hsandle.read()


  # Here we do the formula (4) and (5) of the paper where we allign the signals and after that we do the sum
  sigout = np.zeros(nframes)
  for sig in signalffs.values():
    sig_aligned , _ = align_A2B(np.array(sig), np.array(signalffs["residual_highpass"]))  # With "residual_lowpass" same result

    sigout = sigout + sig_aligned

  # b, a = signal.butter(3, 0.05, btype='highpass')
  # x = signal.lfilter(b, a, sigout)

  # More stable filter
  sos = signal.butter(3, 0.05, btype='highpass', output='sos')
  x = signal.sosfilt(sos, sigout)

  # TODO
  # S.x(1:10)=mean(S.x);

  x = get_sound_scaled_to_one(x)

  # TODO
  # S.averageNoAlignment = mean(reshape(double(signalffs),nScales*nOrients,nF)).';

  return x, sigout
