from argparse import ArgumentParser
import cv2 as cv
import matplotlib.pyplot as plt
from os import path
from scipy.io import wavfile

from code.sound_from_video import *
from code.sound_spectral_subtraction import *


def parse_args():
  parser = ArgumentParser()
  parser.add_argument('input_video', help='The path of the input video')
  parser.add_argument('-o', '--output', help='The path of the output file', default='recoveredsound.wav')
  parser.add_argument('-s', '--sampling-rate', help='The video sampling rate', type=int, default=None)

  return parser.parse_args()


if __name__ == '__main__':
  args = parse_args()

  vr = cv.VideoCapture(args.input_video)
  sr = round(vr.get(cv.CAP_PROP_FPS)) if args.sampling_rate is None else args.sampling_rate

  x, _ = sound_from_video(vr, 1, 2, downsample_factor=0.1, sampling_rate=sr)

  plt.figure()
  plt.specgram(x, Fs=sr)
  plt.colorbar()
  plt.show()

  wavfile.write(args.output, sr, x)

  x_specsub = get_soud_spec_sub(x)

  plt.figure()
  plt.specgram(x_specsub, Fs=sr)
  plt.colorbar()
  plt.show()

  dir, file = path.split(args.output)
  f_name, f_extension = path.splitext(file)

  wavfile.write(path.join(dir, f_name + '_specsub' + f_extension), sr, x_specsub)
