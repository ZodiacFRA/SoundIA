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

from utils import *


# Create synth
harmonic_synth = ddsp.synths.Harmonic(n_samples=N_SAMPLES, sample_rate=SAMPLE_RATE)
# Generate inputs for the synth
amps, harmonic_distribution, f0_hz = get_basic_inputs()
# Now either get the corrected controls then ask for the signal
controls = harmonic_synth.get_controls(amps, harmonic_distribution, f0_hz)
audio = harmonic_synth.get_signal(**controls)
# OR call the object directly, get_controls() is called internally
audio = harmonic_synth(amps, harmonic_distribution, f0_hz)

print("Playing basic inputs")
sd.play(np.asarray(audio).flatten(), SAMPLE_RATE)
input()

# Just for fun with some weird control envelopes...
amps, harmonic_distribution, f0_hz = get_fun_inputs()
audio = harmonic_synth(amps, harmonic_distribution, f0_hz)

print("Playing fun inputs")
sd.play(np.asarray(audio).flatten(), SAMPLE_RATE)
input()
