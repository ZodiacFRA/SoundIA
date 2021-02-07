#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import warnings
# Ignore a bunch of deprecation warnings
warnings.filterwarnings("ignore")

import ddsp
import ddsp.training
import sounddevice as sd
import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf

SAMPLE_RATE = 16000
N_FRAMES = 1000
HOP_SIZE = 64
N_SAMPLES = N_FRAMES * HOP_SIZE

""" https://colab.research.google.com/github/magenta/ddsp/blob/master/ddsp/colab/tutorials/0_processor.ipynb

DDSP Processor Demo
This notebook provides an introduction to the signal `Processor()` object. The main object type in the DDSP library,
it is the base class used for Synthesizers and Effects, which share the methods:

* `get_controls()`: inputs -> controls.
* `get_signal()`: controls -> signal.
* `__call__()`: inputs -> signal. (i.e. `get_signal(**get_controls())`)

Where:
* `inputs` is a variable number of tensor arguments (depending on processor). Often the outputs of a neural network.
* `controls` is a dictionary of tensors scaled and constrained specifically for the processor
* `signal` is an output tensor (usually audio or control signal for another processor)

Let's see why this is a helpful approach by looking at the specific example of the `Harmonic()` synthesizer processor.
The harmonic synthesizer models a sound as a linear combination of harmonic sinusoids.
Amplitude envelopes are generated with 50% overlapping hann windows.
The final audio is cropped to N_SAMPLES.

All member variables are initialized in the constructor,
which makes it easy to change them as hyperparameters using the
[gin](https://github.com/google/gin-config) dependency injection library.
All processors also have a `name` that is used by `ProcessorGroup()`.
"""

harmonic_synth = ddsp.synths.Harmonic(n_samples=N_SAMPLES, sample_rate=SAMPLE_RATE)

"""
`get_controls()`

The outputs of a neural network are often not properly scaled and constrained.
The `get_controls` method gives a dictionary of valid control parameters based on neural network outputs.

**3 inputs (amps, hd, f0)**
* `amplitude`: Amplitude envelope of the synthesizer output.
* `harmonic_distribution`: Normalized amplitudes of each harmonic.
* `fundamental_frequency`: Frequency in Hz of base oscillator
"""

# Generate some arbitrary inputs.
# Amplitude [batch, N_FRAMES, 1].
# Make amplitude linearly decay over time.
amps = np.linspace(1.0, -3.0, N_FRAMES)
amps = amps[np.newaxis, :, np.newaxis]
# Harmonic Distribution [batch, N_FRAMES, n_harmonics].
# Make harmonics decrease linearly with frequency.
n_harmonics = 30
harmonic_distribution = (np.linspace(-2.0, 2.0, N_FRAMES)[:, np.newaxis] +
                         np.linspace(3.0, -3.0, n_harmonics)[np.newaxis, :])
harmonic_distribution = harmonic_distribution[np.newaxis, :, :]
# Fundamental frequency in Hz [batch, N_FRAMES, 1].
f0_hz = 440.0 * np.ones([1, N_FRAMES, 1], dtype=np.float32)

"""Consider the values above as outputs of a neural network.
These outputs violate the synthesizer's expectations:
* Amplitude is not >= 0 (avoids phase shifts)
* Harmonic distribution is not normalized (factorizes timbre and amplitude)
* Fundamental frequency * n_harmonics > nyquist frequency (440 * 20 > 8000),
which will lead to [aliasing](https://en.wikipedia.org/wiki/Aliasing).
"""
controls = harmonic_synth.get_controls(amps, harmonic_distribution, f0_hz)
print(controls.keys())

""" The get_controls() function corrects the inputs:
* Amplitudes are now all positive
* The harmonic distribution sums to 1.0
* All harmonics that are above the Nyquist frequency now have an amplitude of 0.

The amplitudes and harmonic distribution are scaled by an "exponentiated sigmoid" function
(`ddsp.core.exp_sigmoid`). There is nothing particularly special about this function
(other functions can be specified as `scale_fn=` during construction), but it has several nice properties:
* Output scales logarithmically with input (as does human perception of loudness).
* Centered at 0, with max and min in reasonable range for normalized neural network outputs.
* Max value of 2.0 to prevent signal getting too loud.
* Threshold value of 1e-7 for numerical stability during training.

Now use get_signal() To Synthesizes audio from controls. """
audio = harmonic_synth.get_signal(**controls)

"""
Otherwise we can use __call__() to directly
Synthesizes audio from the raw inputs.
`get_controls()` is called internally to turn them into valid control parameters.
"""

audio = harmonic_synth(amps, harmonic_distribution, f0_hz)

sd.play(np.asarray(audio).flatten(), SAMPLE_RATE)
input()

"""
Example just for fun...
Let's run another example where we tweak some of the controls...
"""

## Some weird control envelopes...

# Amplitude [batch, N_FRAMES, 1].
amps = np.ones([N_FRAMES]) * -5.0
amps[:50] +=  np.linspace(0, 7.0, 50)
amps[50:200] += 7.0
amps[200:900] += (7.0 - np.linspace(0.0, 7.0, 700))
amps *= np.abs(np.cos(np.linspace(0, 2*np.pi * 10.0, N_FRAMES)))
amps = amps[np.newaxis, :, np.newaxis]

# Harmonic Distribution [batch, N_FRAMES, n_harmonics].
n_harmonics = 20
harmonic_distribution = np.ones([N_FRAMES, 1]) * np.linspace(1.0, -1.0, n_harmonics)[np.newaxis, :]
for i in range(n_harmonics):
  harmonic_distribution[:, i] = 1.0 - np.linspace(i * 0.09, 2.0, 1000)
  harmonic_distribution[:, i] *= 5.0 * np.abs(np.cos(np.linspace(0, 2*np.pi * 0.1 * i, N_FRAMES)))
  if i % 2 != 0:
    harmonic_distribution[:, i] = -3
harmonic_distribution = harmonic_distribution[np.newaxis, :, :]

# Fundamental frequency in Hz [batch, N_FRAMES, 1].
f0_hz = np.ones([N_FRAMES]) * 200.0
f0_hz[:100] *= np.linspace(2, 1, 100)**2
f0_hz[200:1000] += 20 * np.sin(np.linspace(0, 8.0, 800) * 2 * np.pi * np.linspace(0, 1.0, 800))  * np.linspace(0, 1.0, 800)
f0_hz = f0_hz[np.newaxis, :, np.newaxis]

# Get valid controls
controls = harmonic_synth.get_controls(amps, harmonic_distribution, f0_hz)

# Plot!
time = np.linspace(0, N_SAMPLES / SAMPLE_RATE, N_FRAMES)

plt.figure(figsize=(18, 4))
plt.subplot(131)
plt.plot(time, controls['amplitudes'][0, :, 0])
plt.xticks([0, 1, 2, 3, 4])
plt.title('Amplitude')

plt.subplot(132)
plt.plot(time, controls['harmonic_distribution'][0, :, :])
plt.xticks([0, 1, 2, 3, 4])
plt.title('Harmonic Distribution')

plt.subplot(133)
plt.plot(time, controls['f0_hz'][0, :, 0])
plt.xticks([0, 1, 2, 3, 4])
_ = plt.title('Fundamental Frequency')

audio = harmonic_synth.get_signal(**controls)

sd.play(np.asarray(audio).flatten(), SAMPLE_RATE)
input()
