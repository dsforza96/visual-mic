# The Visual Microphone: Passive Recovery of Sound from Video

A Phyton implementation of MIT's Visual Microphone [1].

Our code is based on the original MATLAB version and depends on OpenCV for handling videos, SciPy for signal processing and writing audio files and [pyrtools](https://github.com/LabForComputationalVision/pyrtools) [2] to compute complex steerable pyramids. Other requirements are NumPy and Matplotlib.


## Usage

```
python video2sound.py [-h] [-o OUTPUT] [-s SAMPLING_RATE] input_video
```

```
positional arguments:
  input_video           The path of the input video

optional arguments:
  -h, --help            Show this help message and exit
  -o OUTPUT, --output OUTPUT
                        The path of the output file
  -s SAMPLING_RATE, --sampling-rate SAMPLING_RATE
                        The video frame rate
```


## Authors

[Antonio Musolino](https://github.com/antoniomuso/) and [Davide Sforza](https://github.com/dsforza96).


## Referencies

[1] DAVIS, Abe, et al. The visual microphone: Passive recovery of sound from video. 2014.

[2] PORTILLA, Javier; SIMONCELLI, Eero P. A parametric texture model based on joint statistics of complex wavelet coefficients. International journal of computer vision, 2000, 40.1: 49-70.
