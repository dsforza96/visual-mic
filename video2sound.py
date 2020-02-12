import cv2 as cv
from argparse import ArgumentParser
from code.sound_from_video import *

def parse_args():
    parser = ArgumentParser()
    parser.add_argument('input_video', help='The path of the input video')
    parser.add_argument('-o', '--output', help='The path of the output file', default='RecoveredSound.wav')

    return parser.parse_args()


if __name__ == '__main__':
    args = parse_args()
    vr = cv.VideoCapture(args.input_video)
    sound_from_video(vr, 1, 2)