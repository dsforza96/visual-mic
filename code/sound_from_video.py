import cv2 as cv
import numpy as np
import pyrtools as pyr

def sound_from_video(v_hsandle: cv.VideoCapture, nscalesin, norientationsin, varargin, downsample_factor=1, nframes=0, sampling_rate=None):
  if sampling_rate is None:
    sampling_rate = v_hsandle.get(cv.CAP_PROP_FPS)


def align_A2B(Ax, Bx):

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
