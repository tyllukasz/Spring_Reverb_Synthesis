import numpy as np
import scipy.signal
import scipy.io.wavfile as wavfile

import matplotlib.pyplot as plt


class SpringReverbIR ():

    def __init__(self, sample_rate=48000, length=5, sweeps=5, f_max=2000, decay_coef=4):

        self.sample_rate = sample_rate # sample rate we are working with [Hz]
        self.length = length # length of the window [s]
        self.sweeps = sweeps # number of sine sweeps in the window
        self.n_samples = int(self.sample_rate * self.length) # length of the window [number of samples]
        self.sweep_size = self.n_samples // self.sweeps # max length of one sweep [number of samples]
        self.f_max = f_max # max frequency of sine sweep [Hz] sweep from 0 to f_max
        self.decay_coef = decay_coef

    def generate(self, sample_rate, length, sweeps, f_max, decay_coef):

        self.sample_rate = sample_rate # sample rate we are working with [Hz]
        self.length = length # length of the window [s]
        self.sweeps = sweeps # number of sine sweeps in the window
        self.n_samples = int(self.sample_rate * self.length) # length of the window [number of samples]
        self.sweep_size = self.n_samples // self.sweeps # max length of one sweep [number of samples]
        self.f_max = f_max # max frequency of sine sweep [Hz] sweep from 0 to f_max
        self.decay_coef = decay_coef
        
        full_ir = []
        n_samples_array = np.arange(0 , self.sweep_size)
        frequency_change = (self.f_max / self.sweep_size) * n_samples_array # f(n) = A*n

        for i in range(0 , self.sweeps):

            sine_sweep = np.sin(2 * np.pi * frequency_change * n_samples_array / self.sample_rate)
            full_ir.append(sine_sweep) 

        full_ir = np.array(full_ir).flatten()

        my_length = int(len(n_samples_array) * self.sweeps)
        total_n_samples_array = np.arange(0,my_length)
        envelope = np.exp(-total_n_samples_array / (my_length / self.decay_coef))

        full_ir = envelope * full_ir

        return full_ir
