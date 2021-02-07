import numpy as np
from scipy.io.wavfile import write
import sounddevice as sd
# Samples per second
sps = 16000

# Frequency / pitch of the sine wave
freq_hz = 440.0

# Duration
duration_s = 5.0

# NumpPy magic
each_sample_number = np.arange(duration_s * sps)
waveform = np.sin(2 * np.pi * each_sample_number * freq_hz / sps)
waveform_quiet = waveform * 0.3
waveform_integers = np.int16(waveform_quiet * 32767)

sd.play(data, sps)
# Write the .wav file
# write('first_sine_wave.wav', sps, waveform_integers)
