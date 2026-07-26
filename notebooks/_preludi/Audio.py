import numpy as np
import soundfile as sf

# Il file audio che la pagina dà per esistente. Mezzo secondo di parlato finto:
# una portante a 220 Hz modulata in ampiezza, che basta a far uscire uno
# spettrogramma con una struttura visibile invece di rumore piatto.
sr = 16_000
t = np.linspace(0, 0.5, int(0.5 * sr), endpoint=False)
inviluppo = 0.5 * (1 + np.sin(2 * np.pi * 4 * t))
onda = inviluppo * (np.sin(2 * np.pi * 220 * t) + 0.3 * np.sin(2 * np.pi * 660 * t))
sf.write("frase.wav", (0.4 * onda).astype(np.float32), sr)
