import cv2 as cv
import math
import numpy as np
import pyrtools as pt


def sound_from_video(v_hsandle: cv.VideoCapture, nscalesin, norientationsin, downsample_factor=1, nframes=None, sampling_rate=None):
  if sampling_rate is None:
    sampling_rate = v_hsandle.get(cv.CAP_PROP_FPS)

  _, colorframe =  v_hsandle.read()

  if downsample_factor < 1:
    colorframe = cv.resize(colorframe, (0,0), fx=downsample_factor, fy=downsample_factor)
  
  grayframe = cv.cvtColor(colorframe, cv.COLOR_BGR2GRAY)
  full_frame = cv.normalize(grayframe.astype('float'), None, 0.0, 1.0, cv.NORM_MINMAX)
  
  ref_frame = full_frame
  h, w = ref_frame.shape

  if nframes is None:
    nframes = int(v_hsandle.get(cv.CAP_PROP_FRAME_COUNT))

  pyr = pt.pyramids.SteerablePyramidFreq(ref_frame, nscalesin, norientationsin - 1, is_complex=True)
  pyr_ref = pyr.pyr_coeffs
  pind = pyr.pyr_size
  print(pyr_ref.keys())

  totalsigs = nscalesin * norientationsin  
  signalffs = np.zeros((nscalesin, norientationsin, nframes))
  ampsigs = np.zeros((nscalesin, norientationsin, nframes))

  for q in range(nframes):
    vframein = v_hsandle.read()
    
    if downsample_factor < 1:
      vframein = cv.resize(vframein, (0,0), fx=downsample_factor, fy=downsample_factor)
    
    grayframe = cv.cvtColor(vframein, cv.COLOR_BGR2GRAY)
    full_frame = cv.normalize(grayframe.astype('float'), None, 0.0, 1.0, cv.NORM_MINMAX)

    pyr = pt.pyramids.SteerablePyramidFreq(full_frame, nscalesin, norientationsin - 1, is_complex=True)
    pyr = pyr.pyr_coeffs

    pyr_amp = np.abs(pyr)
    pyr_delta_phase = (math.pi + np.angle(pyr) - np.angle(pyr_ref)) % (2  *math.pi)

    for j in range(nscalesin):
      band_idx = (j - 1) * norientationsin + 2

      cur_h = pind(band_idx, 1)
      cur_w = pind(band_idx, 2)

      for k in range(norientationsin):
        band_idx = (j - 1) * norientationsin + k + 1
        

def align_A2B(ax, bx):
  pass


  
# Not used functions
def stft_forward(x, sz, hp, pd, w, ll):
  pass

def stft_resynth(x, sz, hp, pd, w, ll):
  pass

def computer_STFT(S, widowsize, hopsize):
  pass

def computer_spec_sub(sstft, qtl1, qtl2):
  pass
