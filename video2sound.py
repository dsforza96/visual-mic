from argparse import ArgumentParser
import cv2 as cv
import matplotlib.pyplot as plt
from os import path
from scipy.io import wavfile

from visualmic.sound_from_video import sound_from_video
from visualmic.sound_spectral_subtraction import get_soud_spec_sub


def parse_args():
  parser = ArgumentParser()
  parser.add_argument('input_video', help='The path of the input video')
  parser.add_argument('-o', '--output', help='The path of the output file', default='recoveredsound.wav')
  parser.add_argument('-s', '--sampling-rate', help='The video sampling rate', type=int, default=None)

  return parser.parse_args()


def plot_specgram(x, sampling_rate):
  plt.figure()
  plt.specgram(x, Fs=sampling_rate, cmap=plt.get_cmap('jet'))
  plt.xlabel('Time (sec)')
  plt.ylabel('Frequency (Hz)')
  plt.colorbar().set_label('PSD (dB)')
  plt.show()


if __name__ == '__main__':
  args = parse_args()

  vr = cv.VideoCapture(args.input_video)
  sr = round(vr.get(cv.CAP_PROP_FPS)) if args.sampling_rate is None else args.sampling_rate

  x = sound_from_video(vr, 1, 2, downsample_factor=0.1, sampling_rate=sr)

  plot_specgram(x, sr)
  wavfile.write(args.output, sr, x)

  x_specsub = get_soud_spec_sub(x)

  plot_specgram(x_specsub, sr)

  dir, file = path.split(args.output)
  f_name, f_extension = path.splitext(file)

  wavfile.write(path.join(dir, f_name + '_specsub' + f_extension), sr, x_specsub)
