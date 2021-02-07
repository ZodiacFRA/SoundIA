import numpy as np
import sounddevice as sd
from scipy.io import wavfile
import tensorflow as tf


# SAMPLE_RATE = 16000
SAMPLE_RATE = 48000
N_FRAMES = 1000
HOP_SIZE = 64
N_SAMPLES = N_FRAMES * HOP_SIZE


def play(audio, name):
    print("Playing", name, audio[:10])
    sd.play(np.asarray(audio).flatten(), SAMPLE_RATE)
    input()


def load_audio_signal(filepath, amp=1):
    samplerate, data = wavfile.read(filepath)
    if samplerate != SAMPLE_RATE:
        print("WARNING: Sample rate mismatch for", filepath[filepath.rfind('/'):])
    data = data * amp
    return data[np.newaxis, :]


def sin_phase(mod_rate, audio):
  """Helper function."""
  n_samples = audio.size
  n_seconds = n_samples / SAMPLE_RATE
  phase = tf.sin(tf.linspace(0.0, mod_rate * n_seconds * 2.0 * np.pi, n_samples))
  return phase[tf.newaxis, :, tf.newaxis]


def get_basic_inputs(n_harmonics=30):
    """ Generate some arbitrary inputs """
    # Amplitude [batch, N_FRAMES, 1].
    # Make amplitude linearly decay over time.
    amps = np.linspace(1.0, -3.0, N_FRAMES)
    amps = amps[np.newaxis, :, np.newaxis]
    # Harmonic Distribution [batch, N_FRAMES, n_harmonics].
    # Make harmonics decrease linearly with frequency.
    harmonic_distribution = (np.linspace(-2.0, 2.0, N_FRAMES)[:, np.newaxis] +
                             np.linspace(3.0, -3.0, n_harmonics)[np.newaxis, :])
    harmonic_distribution = harmonic_distribution[np.newaxis, :, :]
    # Fundamental frequency in Hz [batch, N_FRAMES, 1].
    f0_hz = 440.0 * np.ones([1, N_FRAMES, 1], dtype=np.float32)
    return amps, harmonic_distribution, f0_hz


def get_fun_inputs(n_harmonics=20):
    # Amplitude [batch, N_FRAMES, 1].
    amps = np.ones([N_FRAMES]) * -5.0
    amps[:50] +=  np.linspace(0, 7.0, 50)
    amps[50:200] += 7.0
    amps[200:900] += (7.0 - np.linspace(0.0, 7.0, 700))
    amps *= np.abs(np.cos(np.linspace(0, 2*np.pi * 10.0, N_FRAMES)))
    amps = amps[np.newaxis, :, np.newaxis]

    # Harmonic Distribution [batch, N_FRAMES, n_harmonics].
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
    return amps, harmonic_distribution, f0_hz
