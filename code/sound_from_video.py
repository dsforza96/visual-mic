import cv2 as cv
import math
import numpy as np
import pyrtools as pt
from scipy import signal


def sound_from_video(v_hsandle: cv.VideoCapture, nscale, norientation, downsample_factor=1, nframes=None, sampling_rate=None):
  if sampling_rate is None:
    sampling_rate = v_hsandle.get(cv.CAP_PROP_FPS)

  ret, colorframe =  v_hsandle.read()
  vframein = colorframe

  if downsample_factor < 1:
    colorframe = cv.resize(colorframe, (0,0), fx=downsample_factor, fy=downsample_factor)
  
  grayframe = cv.cvtColor(colorframe, cv.COLOR_BGR2GRAY)
  full_frame = cv.normalize(grayframe.astype('float'), None, 0.0, 1.0, cv.NORM_MINMAX)
  
  ref_frame = full_frame
  h, w = ref_frame.shape

  if nframes is None:
    nframes = int(v_hsandle.get(cv.CAP_PROP_FRAME_COUNT))

  pyr = pt.pyramids.SteerablePyramidFreq(ref_frame, nscale, norientation - 1, is_complex=True)
  pyr_ref = pyr.pyr_coeffs
  pind = pyr.pyr_size

  signalffs = {b: list() for b in pyr_ref.keys()}

  while (ret):
    if downsample_factor < 1:
      vframein = cv.resize(vframein, (0,0), fx=downsample_factor, fy=downsample_factor)
    
    grayframe = cv.cvtColor(vframein, cv.COLOR_BGR2GRAY)
    full_frame = cv.normalize(grayframe.astype('float'), None, 0.0, 1.0, cv.NORM_MINMAX)

    pyr = pt.pyramids.SteerablePyramidFreq(full_frame, nscale, norientation - 1, is_complex=True)
    pyr = pyr.pyr_coeffs

    pyr_amp = dict()
    for band, matrix in pyr.items():
      pyr_amp[band] = np.abs(matrix)

    pyr_delta_phase = dict()
    for band, matrix in pyr.items():
      matrix_ref = pyr_ref[band]
      pyr_delta_phase[band] = np.mod(math.pi + np.angle(matrix) - np.angle(matrix_ref) , 2 * math.pi) - math.pi
      
    for band in pyr.keys():
      amp = pyr_amp[band]
      phase = pyr_delta_phase[band]

      phasew = np.multiply(phase, np.multiply(np.abs(amp), np.abs(amp)))

      sumamp = np.sum(np.abs(amp.flatten()))
      
      signalffs[band].append(np.mean(phasew.flatten()) / sumamp)
    
    ret, vframein = v_hsandle.read()

  sigout = np.zeros(nframes)
  
  for sig in signalffs.values():
    sig_aligned , _ = align_A2B(np.array(sig), np.array(signalffs["residual_highpass"]))
    sigout = sigout + sig_aligned

  # b, a = signal.butter(3, 0.05, btyte='highpass')
  # x = signal.ifilter(b, a, sigout)
  
  # More stable filter
  sos = signal.butter(3, 0.05, btype='highpass', output='sos')
  x = signal.sosfilt(sos, sigout)

  # TODO
  # S.x(1:10)=mean(S.x);

  maxsx = np.max(x)
  minsx = np.min(x)

  if maxsx != 1.0 or minsx != -1.0:
    rangesx = maxsx - minsx
    x = 2 * x / rangesx
    newmax = np.max(x)
    offset = newmax - 1.0
    x = x - offset

  # TODO
  # S.averageNoAlignment = mean(reshape(double(signalffs),nScales*nOrients,nF)).';

  return x, sigout
  
def align_A2B(ax: np.array, bx: np.array):
  acorb = np.convolve(ax, np.flip(bx))

  maxval = np.max(acorb)
  maxind = np.argmax(acorb)

  shiftam = bx.size - maxind
  ax_out = np.roll(ax, shiftam)

  return ax_out, shiftam
  
# Not used functions
def stft_forward(x, sz, hp, pd, w, ll):
  pass

def stft_resynth(x, sz, hp, pd, w, ll):
  pass

def computer_STFT(S, widowsize, hopsize):
  pass

def computer_spec_sub(sstft, qtl1, qtl2):
  pass
