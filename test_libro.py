# -*- coding: utf-8 -*-
"""
Created on Thu Jun 27 15:20:35 2019

@author: Luofeng
"""
import librosa
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import librosa.display


sound = pd.read_csv("AAPL_2019-05-15_.csv") # within a single window (390<512)
#sound = pd.read_csv("USDJPY_2015-12-24_.csv") # 3 windows (2 * 512 < 14XX < 3 * 512)
#sound1 = sound["0"]
#sound2 = sound["1"]
sound1 = sound["close"]
sound2 = sound["volume"]
sound1 = np.array(sound1)
sound2 = np.array(sound2)

# FEATURE 1: MFCC: 20X3 (nums of MFCCs and nums of fracs)
mfccs = librosa.feature.mfcc(sound1)
# VISUALIZATION  
plt.figure(figsize=(10, 4))
librosa.display.specshow(mfccs, x_axis='time')
plt.colorbar()
plt.title('MFCC')
plt.tight_layout()


# FEATURE 2: ZERO-CROSS-RATE 1X3 (nums of fracs)
zrate = librosa.feature.zero_crossing_rate(sound1 - np.mean(sound1))


# FEATURE 3: Chromagram: 12X3 (nums of chroma and nums of fracs)

chromagram = librosa.feature.chroma_stft(sound1, n_chroma = 12)

# VISUALIZATION  
plt.figure(figsize=(10, 4))
librosa.display.specshow(chromagram, x_axis='time')
plt.colorbar()
plt.title('CHROMAGRAM')
plt.tight_layout()

# FEATURE 4: CQT(constant-Q chromagram): 12X3 (nums of chroma and nums of fracs)

cqt = librosa.feature.chroma_cqt(sound1, n_chroma = 12)

# VISUALIZATION  
plt.figure(figsize=(10, 4))
librosa.display.specshow(cqt, x_axis='time')
plt.colorbar()
plt.title('CQT')
plt.tight_layout()


# FEATURE 5: CENS: 12X3 (...)
# Computes the chroma variant “Chroma Energy Normalized” (CENS), following [1].
# Meinard Müller and Sebastian Ewert “Chroma Toolbox: MATLAB implementations for extracting variants of chroma-based audio features” In Proceedings of the International Conference on Music Information Retrieval (ISMIR), 2011.

cens = librosa.feature.chroma_cens(sound1, n_chroma = 12)

# VISUALIZATION  
plt.figure(figsize=(10, 4))
librosa.display.specshow(cens, x_axis='time')
plt.colorbar()
plt.title('CENS')
plt.tight_layout()

# FEATURE 6: MEL: 12X3
mel = librosa.feature.melspectrogram(sound1 - np.mean(sound1), n_mels = 12)
plt.figure(figsize=(10, 4))
librosa.display.specshow(mel, x_axis='time')
plt.colorbar()
plt.title('MEL')
plt.tight_layout()

# FEATURE 7: RMS: 1X3
rms = librosa.feature.rms(sound1)
plt.figure(figsize=(10, 4))
librosa.display.specshow(rms, x_axis='time')
plt.colorbar()
plt.title('RMS')
plt.tight_layout()


# FEATURE 8: SPECTRAL CENTROID: 1X3
sc = librosa.feature.spectral_centroid(sound1)
plt.figure(figsize=(10, 4))
librosa.display.specshow(sc, x_axis='time')
plt.colorbar()
plt.title('SC')
plt.tight_layout()


# FEATURE 9: BANDWIDTH: 1X3Xp (p is the order of bandwidth, default 2)
bw1 = librosa.feature.spectral_bandwidth(sound1, p = 1)
plt.figure(figsize=(10, 4))
librosa.display.specshow(bw1, x_axis='time')
plt.colorbar()
plt.title('BW1')
plt.tight_layout()
bw2 = librosa.feature.spectral_bandwidth(sound1, p = 2)
plt.figure(figsize=(10, 4))
librosa.display.specshow(bw2, x_axis='time')
plt.colorbar()
plt.title('BW2')
plt.tight_layout()

# FEATURE 10: SPECTRAL CONTRAST: (nbands+1)X3, nbands default 6
scon = librosa.feature.spectral_contrast(sound1)
plt.figure(figsize=(10, 4))
librosa.display.specshow(scon, x_axis='time')
plt.colorbar()
plt.title('SCON')
plt.tight_layout()

# FEATURE 11: SPECTRAL FLATNESS: 1X3
sf = librosa.feature.spectral_flatness(sound1)
plt.figure(figsize=(10, 4))
librosa.display.specshow(sf, x_axis='time')
plt.colorbar()
plt.title('SF')
plt.tight_layout()


# FEATURE 12: Compute roll-off frequency: 1X3
ro = librosa.feature.spectral_rolloff(sound1 - np.mean(sound1))
plt.figure(figsize=(10, 4))
librosa.display.specshow(ro, x_axis='time')
plt.colorbar()
plt.title('RO')
plt.tight_layout()

# FEATURE 13: POLY-FITTING: (order + 1)X3
pf1 = librosa.feature.poly_features(sound1 - np.mean(sound1), order = 1)
plt.figure(figsize=(10, 4))
librosa.display.specshow(pf1, x_axis='time')
plt.colorbar()
plt.title('PF1')
plt.tight_layout()

pf2 = librosa.feature.poly_features(sound1 - np.mean(sound1), order = 2)
plt.figure(figsize=(10, 4))
librosa.display.specshow(pf2, x_axis='time')
plt.colorbar()
plt.title('PF2')
plt.tight_layout()

#pf3 = librosa.feature.poly_features(sound1 - np.mean(sound1), order = 3)
#plt.figure(figsize=(10, 4))
#librosa.display.specshow(pf3, x_axis='time')
#plt.colorbar()
#plt.title('PF3')
#plt.tight_layout()

# FEATURE 14: Computes the tonal centroid features (tonnetz), following the method of [1].
# Harte, C., Sandler, M., & Gasser, M. (2006). “Detecting Harmonic Change in Musical Audio.” In Proceedings of the 1st ACM Workshop on Audio and Music Computing Multimedia (pp. 21-26). Santa Barbara, CA, USA: ACM Press. doi:10.1145/1178723.1178727.
ton = librosa.feature.tonnetz(sound1)
plt.figure(figsize=(10, 4))
librosa.display.specshow(ton, x_axis='time')
plt.colorbar()
plt.title('TON')
plt.tight_layout()

