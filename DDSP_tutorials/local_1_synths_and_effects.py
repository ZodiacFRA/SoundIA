#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import warnings
# Ignore a bunch of deprecation warnings
warnings.filterwarnings("ignore")

import ddsp
import ddsp.training
import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf
import tensorflow_datasets as tfds

from utils import *


def gen_harmonic():
    """ Harmonic
    The harmonic synthesizer models a sound as a linear combination of harmonic sinusoids.
    Amplitude envelopes are generated with 50% overlapping hann windows.
    The final audio is cropped to `n_samples`.
    Inputs:
    * `amplitudes`: Amplitude envelope of the synthesizer output.
    * `harmonic_distribution`: Normalized amplitudes of each harmonic.
    * `frequencies`: Frequency in Hz of base oscillator.
    """
    # Generate inputs for the synth
    amps, harmonic_distribution, f0_hz = get_basic_inputs()
    # Create synthesizer object.
    harmonic_synth = ddsp.synths.Harmonic(
                                    n_samples=N_SAMPLES,
                                    # Default scale, not really needed
                                    scale_fn=ddsp.core.exp_sigmoid,
                                    sample_rate=SAMPLE_RATE
                                )
    audio = harmonic_synth(amps, harmonic_distribution, f0_hz)
    return audio


def gen_filtered_noise():
    """
    The filtered noise synthesizer is a subtractive synthesizer that shapes white noise
    with a series of time-varying filter banks.
    Inputs:
    * `magnitudes`: Amplitude envelope of each filter bank
    (linearly spaced from 0Hz to the Nyquist frequency).
    """
    n_frames = 250
    n_frequencies = 1000
    n_samples = 64000
    # Bandpass filters, [n_batch, n_frames, n_frequencies].
    magnitudes = [tf.sin(tf.linspace(0.0, w, n_frequencies)) for w in np.linspace(8.0, 80.0, n_frames)]
    magnitudes = 0.5 * tf.stack(magnitudes)**4.0
    magnitudes = magnitudes[tf.newaxis, :, :]
    # Create synthesizer object.
    filtered_noise_synth = ddsp.synths.FilteredNoise(n_samples=n_samples, scale_fn=None)
    # Generate some audio.
    audio = filtered_noise_synth(magnitudes)
    return audio


def gen_wavetable():
    """ Wavetable
    The wavetable synthesizer generates audio through interpolative lookup from small chunks of waveforms
    (wavetables) provided by the network.
    In principle, it is very similar to the `Harmonic` synth,
    but with a parameterization in the waveform domain and generation using linear interpolation
    vs. cumulative summation of sinusoid phases.

    Inputs:
    * `amplitudes`: Amplitude envelope of the synthesizer output.
    * `wavetables`: A series of wavetables that are interpolated to cover n_samples.
    * `frequencies`: Frequency in Hz of base oscillator.
    """
    n_wavetable = 2048
    n_frames = 100
    # Amplitude [batch, n_frames, 1].
    amps = tf.linspace(0.5, 1e-3, n_frames)[tf.newaxis, :, tf.newaxis]
    # Fundamental frequency in Hz [batch, n_frames, 1].
    f0_hz = 110 * tf.linspace(1.5, 1, n_frames)[tf.newaxis, :, tf.newaxis]
    # Wavetables [batch, n_frames, n_wavetable].
    # Sin wave
    wavetable_sin = tf.sin(tf.linspace(0.0, 2.0 * np.pi, n_wavetable))
    wavetable_sin = wavetable_sin[tf.newaxis, tf.newaxis, :]
    # Square wave
    wavetable_square = tf.cast(wavetable_sin > 0.0, tf.float32) * 2.0 - 1.0
    # Combine them and upsample to n_frames.
    wavetables = tf.concat([wavetable_square, wavetable_sin], axis=1)
    wavetables = ddsp.core.resample(wavetables, n_frames)
    # Create synthesizer object.
    wavetable_synth = ddsp.synths.Wavetable(n_samples=N_SAMPLES,
                                            sample_rate=SAMPLE_RATE,
                                            scale_fn=None)
    # Generate some audio.
    audio = wavetable_synth(amps, wavetables, f0_hz)
    return audio


def add_reverb(audio):
    """ Reverb
    There are several types of reverberation processors in ddsp.
    * Reverb
    * ExpDecayReverb
    * FilteredNoiseReverb
    Unlike other processors, reverbs also have the option to treat the impulse response as a 'trainable' variable, and not require it from network outputs. This is helpful for instance if the room environment is the same for the whole dataset. To make the reverb trainable, just pass the kwarg `trainable=True` to the constructor
    """
    # Let's just do a simple exponential decay reverb.
    reverb = ddsp.effects.ExpDecayReverb(reverb_length=48000)
    gain = [[-2.0]]
    decay = [[2.0]]
    # gain: Linear gain of impulse response. Scaled by self._gain_scale_fn.
    # decay: Exponential decay coefficient. The final impulse response is
    #          exp(-(2 + exp(decay)) * time) where time goes from 0 to 1.0 over the
    #          reverb_length samples.
    audio_out = reverb(audio, gain, decay)
    return audio_out
    print("Playing reverb")
    sd.play(np.asarray(audio).flatten(), SAMPLE_RATE)
    input()
    # Just the filtered noise reverb can be quite expressive.
    reverb = ddsp.effects.FilteredNoiseReverb(reverb_length=48000,
                                              scale_fn=None)
    # Rising gaussian filtered band pass.
    n_frames = 1000
    n_frequencies = 100

    frequencies = np.linspace(0, SAMPLE_RATE / 2.0, n_frequencies)
    center_frequency = 4000.0 * np.linspace(0, 1.0, n_frames)
    width = 500.0
    gauss = lambda x, mu: 2.0 * np.pi * width**-2.0 * np.exp(- ((x - mu) / width)**2.0)

    # Actually make the magnitudes.
    magnitudes = np.array([gauss(frequencies, cf) for cf in center_frequency])
    magnitudes = magnitudes[np.newaxis, ...]
    magnitudes /= magnitudes.sum(axis=-1, keepdims=True) * 5
    # Apply the reverb.
    audio_out = reverb(audio, magnitudes)
    print("Playing reverb again")
    sd.play(np.asarray(audio).flatten(), SAMPLE_RATE)
    input()


def add_fir_filter(audio):
    """ FIR Filter
    Linear time-varying finite impulse response (LTV-FIR) filters are a broad class of filters that can vary over time """
    # Let's make an oscillating gaussian bandpass filter.
    fir_filter = ddsp.effects.FIRFilter(scale_fn=None)

    # Make up some oscillating gaussians.
    n_seconds = audio.size / SAMPLE_RATE
    frame_rate = 100  # Hz
    n_frames = int(n_seconds * frame_rate)
    n_samples = int(n_frames * SAMPLE_RATE / frame_rate)
    audio_trimmed = audio[:, :n_samples]

    n_frequencies = 1000
    frequencies = np.linspace(0, SAMPLE_RATE / 2.0, n_frequencies)

    lfo_rate = 0.5  # Hz
    n_cycles = n_seconds * lfo_rate
    center_frequency = 1000 + 500 * np.sin(np.linspace(0, 2.0*np.pi*n_cycles, n_frames))
    width = 500.0
    gauss = lambda x, mu: 2.0 * np.pi * width**-2.0 * np.exp(- ((x - mu) / width)**2.0)

    # Actually make the magnitudes.
    magnitudes = np.array([gauss(frequencies, cf) for cf in center_frequency])
    magnitudes = magnitudes[np.newaxis, ...]
    magnitudes /= magnitudes.max(axis=-1, keepdims=True)
    # Filter.
    audio_out = fir_filter(audio_trimmed, magnitudes)
    return audio_out


def modulate_audio(audio, center_ms, depth_ms, mod_rate):
    """ Variable length delay lines create an instantaneous pitch shift
    that can be useful in a variety of time modulation effects such as
    [vibrato](https://en.wikipedia.org/wiki/Vibrato)
    [chorus](https://en.wikipedia.org/wiki/Chorus_effect)
    [flanging](https://en.wikipedia.org/wiki/Flanging) """
    mod_delay = ddsp.effects.ModDelay(center_ms=center_ms,
                                    depth_ms=depth_ms,
                                    gain_scale_fn=None,
                                    phase_scale_fn=None
                            )
    phase = sin_phase(mod_rate, audio)  # Hz
    gain = 1.0 * np.ones_like(audio)[..., np.newaxis]
    audio_out = 0.5 * mod_delay(audio, gain, phase)
    return audio_out


if __name__ == '__main__':
    # Test the 3 generators
    # play(gen_harmonic(), "harmonic")
    # play(gen_filtered_noise(), "filtered noise")
    # play(gen_wavetable(), "wavetable")

    # Now load audio
    audio_filepath = "../audio_files/Risers_20.wav"
    audio = load_audio_signal(audio_filepath, 3)
    play(audio, "initial audio")
    # Effects, located in `ddsp.effects` are different in that they
    # take network outputs to transform a given audio signal.
    # Some effects, such as Reverb, optionally have trainable parameters of their own
    play(add_reverb(audio), audio_filepath + " + reverb")
    play(add_fir_filter(audio), audio_filepath + " + FIR filter")
    # Three different modulation effects.
    play(modulate_audio(audio, center_ms=0.75, depth_ms=0.75, mod_rate=0.25), "flanger")
    play(modulate_audio(audio, center_ms=25.0, depth_ms=1.0, mod_rate=2.0), "chorus")
    play(modulate_audio(audio, center_ms=25.0, depth_ms=12.5, mod_rate=5.0), "vibrato")
