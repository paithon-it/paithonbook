# Dal suono alle feature

Quando pronunciamo una parola non facciamo altro che spingere aria. Le corde
vocali vibrano, l'aria si comprime e si rarefà, e un'onda di pressione viaggia
fino al timpano di chi ascolta — o alla membrana di un microfono. Negli anni
Quaranta, ai Bell Labs, un gruppo guidato da Ralph Potter costruì il *sound
spectrograph*, una macchina che trasformava quest'onda in un'immagine e la
chiamò *visible speech*, "parola visibile". È esattamente il percorso che
compie oggi *qualsiasi* sistema che lavora sull'audio — riconoscere una voce,
un canto, un allarme — **prima** ancora di provare a capire *cosa* quel suono
significhi: trasformare un'onda in numeri, e i numeri in un'immagine su cui un
modello sa lavorare. Questa sezione, in apertura del capitolo, costruisce quel
percorso: sono le fondamenta comuni a tutto ciò che segue — qui, e nel capitolo
sullo Speech Recognition.

## Il suono come numeri

Un microfono è, in fondo, un orecchio semplificato: una membrana che si muove
avanti e indietro seguendo la pressione dell'aria. Per portare quel movimento
dentro un computer dobbiamo misurarlo.

`````{tab} Elementare

Immagina di annotare la posizione della membrana migliaia di volte al secondo,
come un sismografo che disegna una linea tremolante mentre la terra si muove.
Il risultato è una lunghissima lista di numeri: quanto era "in avanti" o "in
indietro" la membrana in ogni istante. Un suono, per il computer, è tutto qui —
una sequenza di numeri che sale e scende nel tempo. Numeri grandi (in positivo
o in negativo) quando il suono è forte, vicini allo zero quando c'è silenzio.

`````

`````{tab} Superiore

Il suono è un segnale continuo $x(t)$: l'ampiezza dell'onda di pressione in
funzione del tempo. La digitalizzazione compie due operazioni. Il
**campionamento** discretizza il tempo, misurando $x$ a intervalli regolari
$T_s$ e ottenendo la sequenza $x[n] = x(nT_s)$. La **quantizzazione**
discretizza l'ampiezza su un numero finito di livelli: con la codifica PCM a
16 bit ogni campione è un intero su $2^{16} = 65536$ valori. Un secondo di
audio "CD" è quindi, per ogni canale, un vettore di $44\,100$ interi.

`````

## Quanti campioni al secondo? Il teorema di Nyquist

La domanda cruciale è: quante volte al secondo dobbiamo misurare per non
perdere informazione? Troppo poche e il suono si deforma; troppe e sprechiamo
memoria.

`````{tab} Elementare

Pensa a una ruota che gira ripresa da una telecamera. Se scatti troppe poche
foto, nel video la ruota sembra girare al contrario, o ferma: è l'effetto che
vediamo sulle ruote delle auto nei film. Con il suono succede lo stesso. La
regola è semplice: bisogna misurare **almeno il doppio** di volte rispetto alla
vibrazione più rapida che vogliamo catturare. L'orecchio umano arriva a circa
$20\,000$ oscillazioni al secondo (20 kHz): per questo i CD campionano a
$44\,100$ volte al secondo. Per la voce al telefono bastano $8\,000$: la voce
"vive" quasi tutta sotto i 4 kHz.

`````

`````{tab} Superiore

È il **teorema del campionamento** di Nyquist–Shannon: per ricostruire senza
perdita un segnale la cui frequenza massima è $f_{\max}$, la frequenza di
campionamento deve soddisfare

$$
f_s > 2\,f_{\max}.
$$

La soglia $f_s/2$ è la **frequenza di Nyquist**. Se il segnale contiene
componenti oltre questa soglia, esse si "ripiegano" su frequenze più basse
generando l'**aliasing**: artefatti irreversibili. Per questo si applica un
filtro anti-aliasing (passa-basso) *prima* di campionare. La voce viene tipicamente
trattata a $f_s = 16\,\text{kHz}$, un buon compromesso tra fedeltà e peso.

`````

## Dal tempo alla frequenza: la trasformata di Fourier

La forma d'onda ci dice *quando* il suono è forte o debole, ma non ci dice di
quali "note" è fatto. Eppure è proprio quella composizione a distinguere una
"a" da una "i", o la voce di una persona da un'altra. Serve un cambio di punto
di vista.

`````{tab} Elementare

Ascolta un accordo al pianoforte: senti più note insieme, ma il tuo orecchio
riesce a dire quali sono. La **trasformata di Fourier** fa esattamente questo
con un suono qualsiasi: prende l'onda ingarbugliata e la scompone nelle sue
frequenze pure, dicendoti *quanto* di ciascuna è presente. È come un prisma che
separa la luce bianca nei colori dell'arcobaleno: la luce sembrava una sola,
invece era una somma. Passiamo così dal "come cambia nel tempo" al "di quali
frequenze è fatto".

`````

`````{tab} Superiore

L'idea, dovuta a Joseph Fourier (1822), è che un segnale possa essere scritto
come somma di sinusoidi di frequenze diverse. Per un segnale campionato di $N$
punti si usa la **trasformata di Fourier discreta** (DFT), calcolata in
pratica con l'algoritmo **FFT** in tempo $O(N\log N)$:

$$
X[k] = \sum_{n=0}^{N-1} x[n]\, e^{-\,i\,2\pi kn/N}.
$$

Qui $x[n]$ sono i campioni nel tempo, $X[k]$ è il coefficiente (complesso)
associato alla frequenza $k$, e $|X[k]|$ ne misura l'ampiezza. Non deriviamo la
formula: ci basta l'interpretazione. La DFT trasforma $N$ numeri "nel tempo" in
$N$ numeri "in frequenza", senza perdere informazione.

`````

## Lo spettrogramma: l'immagine del suono

C'è un problema: la trasformata di Fourier di un intero file dice *quali*
frequenze ci sono, ma non *quando*. Una frase è fatta di suoni che cambiano di
continuo. La soluzione è tagliare l'audio in tante finestrelle brevi (20–40
millisecondi), calcolare la trasformata di Fourier di ciascuna, e affiancare i
risultati. Questa è la **Short-Time Fourier Transform** (STFT). Il risultato,
disposto in una tabella, è lo **spettrogramma**: un'immagine con il tempo
sull'asse orizzontale, la frequenza su quello verticale, e l'intensità di
ciascuna frequenza resa dal colore ({numref}`fig-onda-spettrogramma`).

```{figure} ../figures/onda-spettrogramma.svg
:name: fig-onda-spettrogramma
:alt: A sinistra una forma d'onda audio come barre verticali nel dominio del tempo; una freccia la trasforma in uno spettrogramma a destra, una griglia con il tempo in orizzontale, la frequenza in verticale e l'intensità resa da tonalità della palette.
:width: 90%

Dalla forma d'onda allo spettrogramma. La trasformata di Fourier a finestre
(STFT) trasforma il segnale nel tempo in una mappa tempo–frequenza: ogni cella
dice quanta energia c'è a una certa frequenza in un certo istante.
```

È la stessa "parola visibile" di Potter: le bande orizzontali più intense sono
le *formanti*, le risonanze del tratto vocale che danno a ogni vocale il suo
timbro. Un modello di riconoscimento vocale può ora trattare l'audio come tratta
un'immagine — e sui dati a griglia sappiamo far lavorare bene le reti.

## MFCC e la scala mel: ascoltare come un orecchio

Lo spettrogramma è ancora ricco e ridondante. Possiamo comprimerlo imitando il
modo in cui l'orecchio umano percepisce davvero i suoni, ottenendo **feature**
— caratteristiche riassuntive del suono — più compatte e robuste: i
**coefficienti cepstrali in scala mel** (MFCC).

`````{tab} Elementare

Il nostro orecchio non è un righello: distingue benissimo due note gravi vicine,
ma fatica con due note acute altrettanto vicine. In basso è "preciso", in alto è
"approssimativo". La **scala mel** riscrive le frequenze proprio così, come le
sente una persona. Gli MFCC prendono lo spettrogramma, lo rileggono con questa
scala e lo riassumono in una manciata di numeri per finestrella — di solito
13 — che catturano la "forma" del suono buttando via i dettagli inutili.
Meno numeri, ma quelli che contano davvero per capire il parlato.

`````

`````{tab} Superiore

La conversione da hertz a mel comprime le alte frequenze in modo logaritmico:

$$
m = 2595\,\log_{10}\!\Big(1 + \frac{f}{700}\Big).
$$

La pipeline degli MFCC {cite}`davis1980comparison` è: (1) spettro di potenza
dalla STFT; (2) banco di filtri triangolari spaziati sulla scala mel; (3)
logaritmo delle energie di banda, che imita la percezione dell'intensità; (4)
**trasformata coseno discreta** (DCT), che decorrela le bande e concentra
l'informazione nei primi coefficienti. Si tengono i primi $\sim 13$: sono le
feature classiche dei sistemi pre-deep-learning basati su modelli di Markov
nascosti (HMM).

`````

## Perché queste feature aiutano il modello

La forma d'onda grezza è enorme (decine di migliaia di numeri al secondo) e
piena di variazioni irrilevanti: il volume, il rumore di fondo, uno spostamento
di pochi millisecondi. Spettrogramma mel e MFCC concentrano l'informazione
*linguisticamente utile* — quali frequenze, quando — scartando gran parte del
resto. Il compito del modello diventa così più facile: parte da una
rappresentazione più piccola, più stabile e già "orientata" verso ciò che
distingue un suono dall'altro.

Onestà d'obbligo: i sistemi più recenti *end-to-end*, come wav2vec 2.0 (Baevski
et al., 2020) o Whisper (Radford et al., 2022), tendono a imparare le feature
direttamente dai dati. Ma non partono dal nulla: Whisper, per esempio, riceve in
input proprio uno **spettrogramma log-mel**. La scala mel, ispirata al nostro
orecchio, resta il punto di partenza più diffuso anche nell'era del deep
learning.

## In pratica

Con la libreria `librosa` l'intera catena — dal file audio allo spettrogramma
mel — è una manciata di righe.

```python
import librosa

# carica l'audio a 16 kHz (frequenza tipica per la voce)
y, sr = librosa.load("frase.wav", sr=16000)

# spettrogramma mel: 40 bande, finestre da 25 ms ogni 10 ms
S = librosa.feature.melspectrogram(
    y=y, sr=sr, n_fft=400, hop_length=160, n_mels=40
)
S_db = librosa.power_to_db(S, ref=S.max())  # in decibel, come "vede" l'orecchio

# 13 coefficienti MFCC per ogni finestra temporale
mfcc = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=13)

print(S_db.shape, mfcc.shape)  # (bande, tempo) e (coefficienti, tempo)
```

```{admonition} Da ricordare
:class: important
- Un suono digitale è una sequenza di numeri (**ampiezza nel tempo**), ottenuta
  campionando la membrana del microfono.
- Il **teorema di Nyquist** impone $f_s > 2 f_{\max}$: sotto quella soglia
  compare l'aliasing.
- La **trasformata di Fourier** passa dal tempo alle frequenze; applicata a
  finestre brevi (STFT) produce lo **spettrogramma**, l'immagine del suono.
- **Scala mel** e **MFCC** comprimono lo spettro imitando l'orecchio: feature
  più piccole e robuste che rendono più facile il compito del modello.
```
