from argparse import ArgumentParser
import cv2 as cv
import matplotlib.pyplot as plt
from os import path
from scipy.io import wavfile

from video2sound.sound_from_video import sound_from_video
from video2sound.sound_spectral_subtraction import get_soud_spec_sub


def parse_args():
  parser = ArgumentParser()
  parser.add_argument('input_video', help='The path of the input video')
  parser.add_argument('-o', '--output', help='The path of the output file', default='recoveredsound.wav')
  parser.add_argument('-s', '--sampling-rate', help='The video sampling rate', type=int, default=None)

  return parser.parse_args()


def plot_specgram(x: np.array):
  plt.figure()
  plt.specgram(x, Fs=sr, cmap=plt.get_cmap('jet'))
  plt.xlabel('Time (sec)')
  plt.ylabel('Frequency (Hz)')
  plt.colorbar().set_label('PSD (dB)')
  plt.show()


def save_audio(x: np.array, sr, output_file, file_suffix=''):
  dir, file = path.split(output_file)
  f_name, f_extension = path.splitext(file)

  wavfile.write(path.join(dir, f_name + file_suffix + f_extension), sr, x)


if __name__ == '__main__':
  args = parse_args()

  vr = cv.VideoCapture(args.input_video)
  sr = round(vr.get(cv.CAP_PROP_FPS)) if args.sampling_rate is None else args.sampling_rate

  x, x_nofilt, x_noalig = sound_from_video(vr, 1, 2, downsample_factor=0.1, sampling_rate=sr)

  plot_specgram(x)
  save_audio(x, sr, args.output)

  # plot_specgram(x_nofilt)
  # save_audio(x_nofilt, sr, args.output, '_nofilt')

  # plot_specgram(x_noalig)
  # save_audio(x_noalig, sr, args.output, '_noalig')

  x_specsub = get_soud_spec_sub(x)

  plot_specgram(x_specsub)
  save_audio(x_specsub, sr, args.output, '_specsub')
