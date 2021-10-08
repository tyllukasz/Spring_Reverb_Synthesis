import numpy as np
import scipy.signal
import scipy.io.wavfile as wavfile
import matplotlib.pyplot as plt

import reverb_ir_definition

from functions import to_dB, from_dB


#file loading to convolve with
file_name = 'RGuitar_(bridge)_dry.wav'
sample_rate, dry_wave = wavfile.read(file_name)
dry_wave = dry_wave / np.max(np.abs(dry_wave))

#======================================================================================================

#IR initialization
length = 5 #length of IR window [s]
sweeps = 20 #number of echoes in IR window
max_freq = 1500 #sweep target frequency [Hz] 0 --> max_freq
decay_ratio = 4 #amplitude decay speed --> the smaller ratio the longer sustain

my_ir = reverb_ir_definition.SpringReverbIR()
my_ir = my_ir.generate(sample_rate=sample_rate, length=length, sweeps=sweeps, f_max=max_freq, decay_coef=decay_ratio)
wavfile.write('my_ir.wav', sample_rate, my_ir)

#======================================================================================================

#wet signal calculation (convolution)
#wet_wave = np.convolve(dry_wave , self.generate(), mode='same')
wet_wave = scipy.signal.fftconvolve(dry_wave , my_ir) #faster method than numpy
wet_wave = wet_wave / np.max(np.abs(wet_wave)) #normalizing <-1,1>
wavfile.write('wet_signal.wav', sample_rate, (wet_wave*np.iinfo(np.int16).max).astype(np.int16))

#======================================================================================================

#wet/dry signal mix

wet_delta_dB = -6
mix_coef = from_dB(wet_delta_dB)

dry_wave = np.concatenate((dry_wave, np.zeros(len(wet_wave) - len(dry_wave))))

dry_wet_mix = dry_wave + mix_coef * wet_wave
dry_wet_mix = dry_wet_mix / np.max(np.abs(dry_wet_mix)) #normalizing <-1,1>
wavfile.write('dry_wet_mix_signal.wav', sample_rate, (dry_wet_mix*np.iinfo(np.int16).max).astype(np.int16))

print(min(dry_wave))