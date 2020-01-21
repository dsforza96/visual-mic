from argparse import ArgumentParser

def parse_args():
    parser = ArgumentParser()
    parser.add_argument('input_video', help='The path of the input video')
    parser.add_argument('-o', '--output', help='The path of the output file', default='RecoveredSound.wav')

    return parser.parse_args()


if __name__ == '__main__':
    args = parse_args()