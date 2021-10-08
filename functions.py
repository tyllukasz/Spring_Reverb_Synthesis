import numpy as np

#magnitude to dB conversion
to_dB = lambda x: 20 * np.log10(x)

#dB to magnitude conversion
from_dB = lambda x: 10**(x/20)