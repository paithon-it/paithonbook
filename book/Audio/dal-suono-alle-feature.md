# Dal suono alle feature

Quando pronunciamo una parola non facciamo altro che spingere aria. Le corde
vocali vibrano, l'aria si comprime e si dirada (si *rarefà*, dicono i fisici), e
un'onda di pressione viaggia
fino al timpano di chi ascolta, o alla membrana di un microfono. Negli anni
Quaranta, ai Bell Labs, un gruppo guidato da Ralph Potter costruì il *sound
spectrograph*, una macchina che trasformava quest'onda in un'immagine e la
chiamò *visible speech*, "parola visibile". È esattamente il percorso che
compie oggi *qualsiasi* sistema che lavora sull'audio (riconoscere una voce,
un canto, un allarme) **prima** ancora di provare a capire *cosa* quel suono
significhi: trasformare un'onda in numeri, e i numeri in un'immagine su cui un
modello sa lavorare. Questa sezione, in apertura del capitolo, costruisce quel
percorso: sono le fondamenta comuni a tutto ciò che segue (qui, e nel capitolo
sullo Speech Recognition).

Nel titolo c'è la parola **feature**, che in questo libro indica i numeri con
cui descriviamo un esempio: qui, i numeri con cui descriviamo un suono. La
domanda della sezione è quali scegliere, e perché proprio quelli.

## Il suono come numeri

Un microfono è, in fondo, un orecchio semplificato: una membrana che si muove
avanti e indietro seguendo la pressione dell'aria. Per portare quel movimento
dentro un computer dobbiamo misurarlo.

Prima però conviene fissare una parola, perché regge tutto il resto del
capitolo. Quel movimento è un'oscillazione, e **quante volte al secondo la
membrana va avanti e indietro si chiama frequenza**: si misura in hertz (Hz), e
mille hertz fanno un kilohertz (kHz). La traduzione da tenere a mente è
semplice: **frequenza alta vuol dire suono acuto, frequenza bassa vuol dire
suono grave**. Il la del diapason oscilla 440 volte al secondo, cioè 440 Hz; la
stessa nota un'ottava sopra ne fa il doppio, 880.

`````{tab} Elementare

Immagina di annotare la posizione della membrana migliaia di volte al secondo,
come un sismografo che disegna una linea tremolante mentre la terra si muove.
Il risultato è una lunghissima lista di numeri: quanto era "in avanti" o "in
indietro" la membrana in ogni istante. Un suono, per il computer, è tutto qui:
una sequenza di numeri che sale e scende nel tempo. Numeri grandi (in positivo
o in negativo), quando il suono è forte, vicini allo zero quando c'è silenzio.

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

Attenzione a un tranello di vocabolario, perché da qui in avanti la parola
«frequenza» fa due mestieri. Quante volte al secondo *misuriamo* si chiama
**frequenza di campionamento**, e si scrive in hertz esattamente come la
frequenza di un suono: la prima dice quanto è fitto il nostro metro, la seconda
quanto è acuto ciò che stiamo misurando. Sono numeri diversi, e quando più
avanti un disegno avrà «16 kHz» in una didascalia e «3 kHz» su un asse non è una
contraddizione: sono le due grandezze, non la stessa due volte.

`````{tab} Elementare

Pensa a una ruota che gira ripresa da una telecamera. Se scatti troppe poche
foto, nel video la ruota sembra girare al contrario, o ferma: è l'effetto che
vediamo sulle ruote delle auto nei film. Con il suono succede lo stesso. La
regola è semplice: bisogna misurare **più del doppio** delle volte rispetto alla
vibrazione più rapida che vogliamo catturare. L'orecchio umano arriva a circa
$20\,000$ oscillazioni al secondo (20 kHz), quindi servono più di $40\,000$
misure al secondo: i CD ne fanno $44\,100$, che stanno larghi apposta. Per la voce al telefono bastano $8\,000$, che è esattamente il doppio
dei 4 kHz sotto i quali la voce "vive" quasi tutta. E c'è una via di mezzo,
$16\,000$ misure al secondo, che è quella scelta dalla maggior parte dei
sistemi di questo capitolo: prende tutta la voce senza sprecare memoria, ed è il
numero che ritroverai nelle figure e nel codice.

`````

`````{tab} Superiore

È il **teorema del campionamento** di Nyquist–Shannon: per ricostruire senza
perdita un segnale la cui frequenza massima è $f_{\max}$, la frequenza di
campionamento deve soddisfare

$$
f_s > 2\,f_{\max}.
$$

dove $f_{\max}$ è la frequenza oltre la quale lo **spettro** del segnale (che
costruiamo nella sezione seguente, con la trasformata di Fourier) è nullo: per
un segnale che non sia una sinusoide pura, «la frequenza più alta» non è
definibile senza quella decomposizione, ed è la ragione per cui l'enunciato di
Nyquist si appoggia a uno strumento che il capitolo introduce fra poco.

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

Quella mappa non nasce tutta insieme: si riempie una colonna alla volta, e
{numref}`fig-finestra-spettrogramma` mostra il gesto. La finestra scorre sul
segnale a passi regolari, e ogni sua posizione lascia dietro di sé una colonna.

```{figure} ../figures/finestra-spettrogramma.svg
:name: fig-finestra-spettrogramma
:alt: In alto la forma d'onda di tre note che salgono una dopo l'altra, con la finestra larga 25 millisecondi ferma sull'ultima posizione; una freccia scende verso lo spettrogramma sottostante, dove ogni posizione della finestra ha lasciato una colonna e la banda scura sale a gradini, di nota in nota.
:width: 92%

Il segnale sono tre note che salgono, una dopo l'altra. La finestra ci scorre
sopra un passo alla volta e a ogni posizione lascia una colonna dello
spettrogramma: è lunga 25 millesimi di secondo e avanza di 10, quindi due
finestre vicine si sovrappongono ma le loro colonne no. Le note che salgono si
vedono come gradini, ed è esattamente l'informazione che la trasformata del file
intero non saprebbe dare.
```

Le finestre a cavallo fra una nota e l'altra mostrano **entrambe** le frequenze,
ed è la cosa più istruttiva del disegno: non è una sbavatura, è il compromesso su
cui la trasformata a finestre è costruita. Una finestra lunga distingue bene le
frequenze e male gli istanti, perché dentro ci finiscono due note; una corta fa
il contrario. Non esiste una lunghezza che vinca su tutti e due i fronti, ed è
per questo che sceglierla è una decisione e non un dettaglio.

Una parola sulla **forma** della finestra, perché nel disegno non è il rettangolo
che «finestrella» lascia immaginare: è gonfia in mezzo e va a zero ai bordi.
Tagliare di netto un pezzo di onda creerebbe due gradini artificiali agli
estremi, e la trasformata leggerebbe quei gradini come frequenze che nel suono
non ci sono. Smussando i bordi il taglio diventa una dissolvenza e il difetto
quasi sparisce, al prezzo di dare meno peso a ciò che capita ai margini. La
curva più usata per farlo si chiama **finestra di Hann**, dal nome del
meteorologo austriaco Julius von Hann, ed è quella disegnata in
{numref}`fig-finestra-spettrogramma`.

`````{tab} Elementare

Vale la pena insistere sul compromesso, perché non è un consiglio pratico: è un
limite, e nessuna astuzia lo aggira. Le due domande («quando è successo?» e «che
nota era?») si contendono la stessa finestra, perché per riconoscere una nota
bisogna sentirla oscillare un po' di volte, e quel po' di tempo è esattamente
l'istante che si perde.

I numeri della finestra da 25 millesimi di secondo, che è quella usata qui e nel
resto del libro: colloca un suono nel tempo con un'incertezza di circa tre
millesimi di secondo, e distingue due frequenze che distano una ventina di
hertz. Allungala a 100 millesimi e la seconda cifra migliora di quattro volte
(scende a sei hertz), ma la prima peggiora esattamente di quattro (sale a
quattordici millesimi di secondo). Il prodotto delle due non cambia mai, e sotto
una certa soglia non si va: quello che si guadagna da una parte si paga, tutto,
dall'altra. Scegliere la lunghezza della finestra vuol dire scegliere quale
delle due precisioni comprare.

`````

`````{tab} Superiore

Formalmente la STFT di un segnale $\mathbf{x}$, i cui campioni sono gli $x[n]$, è

$$
X[m,k] = \sum_{n} x[n]\, w[n - mH]\, e^{-\,i\,2\pi kn/N},
$$

dove $\mathbf{w}$ è la **finestra** (lunga $N$ campioni), $H$ il **passo** (*hop*) di cui
essa avanza da una colonna alla successiva, $m$ l'indice della colonna e $k$
quello del bin di frequenza. Con frequenza di campionamento $f_s$, i bin sono
spaziati di $f_s/N$ hertz e le colonne di $H/f_s$ secondi: sono le due
risoluzioni, e si muovono in senso opposto. Per il parlato la scelta standard è
$N$ pari a 25 ms e $H$ a 10 ms (a 16 kHz: `n_fft=400`, `hop_length=160`, gli
stessi valori del codice più sotto), perché i fonemi durano decine di
millisecondi e una finestra più lunga ne mescolerebbe due.

Il compromesso non è pratico ma **teorico**: è la relazione di indeterminazione
di Gabor (1946), l'analogo per l'analisi di Fourier del principio di
indeterminazione di Heisenberg,

$$
\sigma_t \cdot \sigma_f \ \geq\ \frac{1}{4\pi},
$$

dove $\sigma_t$ e $\sigma_f$ sono le deviazioni standard dell'energia della
finestra nel tempo e in frequenza. L'uguaglianza vale per la finestra
gaussiana; per la finestra di Hann il prodotto vale $0{,}0817$ contro il limite
$1/(4\pi) = 0{,}0796$, ed è **costante al variare della lunghezza**: la Hann da
25 ms ha $\sigma_t \approx 3{,}5$ ms e $\sigma_f \approx 23$ Hz, quella da 100 ms
$\sigma_t \approx 14{,}1$ ms e $\sigma_f \approx 5{,}8$ Hz. Quadruplicare la
finestra quadruplica $\sigma_t$ e divide $\sigma_f$ per quattro, senza sconti.
Non esiste una finestra furba: esiste un cambio fisso, e sceglierne la lunghezza
significa decidere quale delle due risoluzioni si vuole comprare.

`````

È la stessa "parola visibile" di Potter. Nello spettrogramma di una vocale
alcune bande orizzontali sono più intense delle altre: sono le frequenze che la
bocca e la gola, facendo da cassa di risonanza come il corpo di una chitarra,
rinforzano più delle vicine. Cambiano a seconda di come teniamo lingua e labbra,
ed è per questo che distinguono una "a" da una "i". Si chiamano **formanti**, e
sono il primo esempio di una cosa che si vede nell'immagine del suono e non si
vedeva nell'onda. Un modello di riconoscimento vocale può ora trattare l'audio
come tratta un'immagine, e sui dati a griglia sappiamo far lavorare bene le
reti.

## MFCC e la scala mel: ascoltare come un orecchio

Lo spettrogramma è ancora ricco e ridondante. Possiamo comprimerlo imitando il
modo in cui l'orecchio umano percepisce davvero i suoni, ottenendo feature più
compatte e robuste: i **coefficienti cepstrali in scala mel** (MFCC).

«Cepstrale» non è un refuso per «spettrale»: è un gioco di parole degli
ingegneri che negli anni Sessanta inventarono il metodo, che rovesciarono le
prime lettere di *spectrum* per dire che si applica una seconda trasformata allo
spettro, e il termine è rimasto.

`````{tab} Elementare

Il nostro orecchio non è un righello: distingue benissimo due note gravi
vicine, ma fatica con due note acute altrettanto vicine. In basso è "preciso",
in alto è "approssimativo". La **scala mel** riscrive le frequenze proprio
così, come le sente una persona. Gli MFCC prendono lo spettrogramma, lo
rileggono con questa scala e lo riassumono in una manciata di numeri per
finestrella (di solito 13) che catturano la "forma" del suono buttando via i
dettagli inutili. Meno numeri, ma quelli che contano davvero per capire il
parlato.

Perché proprio 13? Nessuna legge di natura: è il numero che negli anni Ottanta
funzionava bene sul parlato, ed è rimasto per abitudine e per poter confrontare
un sistema con l'altro. Sono numeri che qualcuno ha deciso, e infatti la parte
finale del capitolo racconta come le macchine abbiano poi imparato a
sceglierseli da sole.

`````

`````{tab} Superiore

La conversione da hertz a mel comprime le alte frequenze in modo logaritmico.
Le formule in circolazione però sono **più d'una**, perché la scala mel è
un'interpolazione di dati sperimentali di ascolto e non una legge fisica: non
esiste *la* conversione. La più diffusa, dovuta a O'Shaughnessy (1987) e
adottata da HTK, è

$$
f_{\text{mel}} = 2595\,\log_{10}\!\Big(1 + \frac{f}{700}\Big),
$$

mentre l'implementazione di Slaney, che `librosa` usa per impostazione
predefinita, tiene la scala lineare sotto 1 kHz e logaritmica sopra. Le due non
danno le stesse bande: con i parametri del codice qui sotto ($f_s = 16$ kHz, 40
bande) la decima banda è centrata a 636 Hz con la formula HTK e a 773 Hz con
quella di Slaney, il 22 % più in alto. È una differenza che va dichiarata quando
si confrontano due sistemi, ed è il motivo per cui il codice della prossima
pagina passa `htk=True`: perché faccia davvero il conto scritto qui sopra.

La pipeline degli MFCC {cite}`davis1980comparison` è: (1) spettro di potenza
dalla STFT; (2) banco di filtri triangolari spaziati sulla scala mel; (3)
logaritmo delle energie di banda, che imita la percezione dell'intensità; (4)
**trasformata coseno discreta** (DCT), che decorrela le bande e concentra
l'informazione nei primi coefficienti. Si tengono i primi $\sim 13$: sono le
feature classiche dei sistemi pre-deep-learning basati su modelli di Markov
nascosti (HMM).

Il punto (4) merita una spiegazione che di solito manca, perché senza di essa la
DCT sembra un accorgimento di buon senso valido sempre. Non lo è: serviva a una
cosa precisa e datata. I sistemi GMM-HMM modellavano ogni stato con gaussiane a
**covarianza diagonale**, per costo e per scarsità di dati, e una covarianza
diagonale su feature correlate è un modello sbagliato; le bande mel si
sovrappongono, quindi correlate lo sono parecchio. Decorrelare le rendeva
lecite. Caduta l'ipotesi, è caduta la ragione: una rete non chiede feature
scorrelate, e la DCT le fa pagare un prezzo, perché mescolando tutte le bande in
ogni coefficiente **distrugge la località in frequenza** su cui una convoluzione
lavora. Per questo dagli anni Dieci si è tornati allo **spettrogramma log-mel
grezzo**, che è ciò che ricevono in ingresso tutti i modelli del resto di questo
capitolo (l'AST prende 128 bande log-mel, Whisper 80), mentre i 13 MFCC restano
una feature d'archivio, ancora comoda dove serve un vettore piccolo (HuBERT li
usa proprio per il primo raggruppamento).

`````

## Perché queste feature aiutano il modello

La forma d'onda grezza è enorme (decine di migliaia di numeri al secondo) e
piena di variazioni irrilevanti: il volume, il rumore di fondo, uno
spostamento di pochi millisecondi. Spettrogramma mel e MFCC concentrano
l'informazione *linguisticamente utile* (quali frequenze, quando) scartando
gran parte del resto. Il compito del modello diventa così più facile: parte da
una rappresentazione più piccola, più stabile e già "orientata" verso ciò che
distingue un suono dall'altro.

Onestà d'obbligo: i sistemi più recenti *end-to-end*, come wav2vec 2.0
{cite}`baevski2020wav2vec` o Whisper {cite}`radford2022robust`, tendono a
imparare le feature
direttamente dai dati. Ma non partono dal nulla: Whisper, per esempio, riceve in
input proprio uno **spettrogramma log-mel**. La scala mel, ispirata al nostro
orecchio, resta il punto di partenza più diffuso anche nell'era del deep
learning.

Vale la pena fermarsi su quella parola, perché è qui che cambia padrone. Fin dal
capitolo sul machine learning le feature erano *scelte da noi*: qualcuno decideva
quali numeri estrarre da ogni esempio, e quella decisione era metà del lavoro.
Da questa pagina in poi saranno quasi sempre *imparate dalla rete*, che si
costruisce da sé i numeri che le servono. La parola indica lo stesso oggetto (i
numeri che descrivono un esempio) e cambia solo chi li sceglie; ma è il
passaggio che divide questa sezione da tutte quelle che seguono, e conviene
saperlo prima di incontrarlo scritto come se fosse ovvio.

## In pratica

Con la libreria `librosa` l'intera catena (dal file audio allo spettrogramma
mel) è una manciata di righe.

```python
import librosa

# carica l'audio a 16 kHz (frequenza tipica per la voce)
y, sr = librosa.load("frase.wav", sr=16000)

# spettrogramma mel: 40 bande, finestre da 25 ms ogni 10 ms
# htk=True: la scala mel della formula qui sopra (il default di librosa e' quella di Slaney)
S = librosa.feature.melspectrogram(
    y=y, sr=sr, n_fft=400, hop_length=160, n_mels=40, htk=True
)
S_db = librosa.power_to_db(S, ref=S.max())  # in decibel, come "vede" l'orecchio

# 13 coefficienti MFCC per ogni finestra temporale.
# Si calcolano DALLA S_db appena ottenuta: gli MFCC sono la DCT delle log-mel,
# e chiamando mfcc(y=...) librosa rifarebbe lo spettrogramma con i propri
# default (n_fft=2048, hop_length=512), su un asse temporale diverso.
mfcc = librosa.feature.mfcc(S=S_db, n_mfcc=13)

print(S_db.shape, mfcc.shape)  # (bande, tempo) e (coefficienti, tempo)
```

Le due matrici escono con lo stesso numero di colonne, e non è un dettaglio:
sono allineate finestra per finestra, quindi si possono affiancare e dare al
modello insieme. Sarebbe bastato chiamare `mfcc(y=...)` invece che `mfcc(S=...)`
per ritrovarsi due assi temporali diversi e un errore di dimensione
incomprensibile.

`````{tab} Elementare

```{admonition} Da ricordare
:class: important
- Per un computer un suono è una lunghissima lista di numeri: la posizione
  della membrana del microfono, annotata migliaia di volte al secondo come fa
  un sismografo.
- **Frequenza** vuol dire quante volte al secondo qualcosa oscilla, e si misura
  in hertz: tanta frequenza è un suono acuto, poca è un suono grave. Da non
  confondere con la **frequenza di campionamento**, che è quante volte al
  secondo *noi* misuriamo.
- Bisogna misurare più del **doppio** delle volte rispetto alla vibrazione più
  rapida che vogliamo catturare: sotto quella soglia il suono si deforma, come
  la ruota che nei film sembra girare al contrario.
- La **trasformata di Fourier** è il prisma che scompone l'onda nelle sue
  frequenze pure; applicata a tante finestrelle brevi una dopo l'altra dà lo
  **spettrogramma**, l'immagine del suono (tempo in orizzontale, frequenze in
  verticale).
- Le finestrelle non si possono avere insieme corte e precise sulle note:
  allungarle fa guadagnare sulle note e perdere sugli istanti, e il cambio è
  fisso, non migliorabile. Sceglierne la lunghezza vuol dire decidere quale
  delle due precisioni si compra.
- La **scala mel** e gli **MFCC** rileggono quell'immagine come la sente un
  orecchio (preciso sui suoni gravi, approssimativo sugli acuti) e la
  riassumono in pochi numeri per finestrella: meno dati, ma quelli che contano
  davvero, e il modello ha un compito più facile.
```

`````

`````{tab} Superiore

```{admonition} Da ricordare
:class: important
- Un suono digitale è una sequenza di numeri (**ampiezza nel tempo**),
  $x[n] = x(nT_s)$: campionamento del tempo e quantizzazione dell'ampiezza (PCM
  a 16 bit).
- Il **teorema di Nyquist** impone $f_s > 2 f_{\max}$: sotto quella soglia
  compare l'aliasing, e serve un filtro passa-basso *prima* di campionare.
- La **trasformata di Fourier** passa dal tempo alle frequenze (DFT, calcolata
  con la FFT in $O(N\log N)$); applicata a finestre brevi (STFT) produce lo
  **spettrogramma**, l'immagine del suono.
- La finestra impone un **limite**, non un compromesso negoziabile:
  $\sigma_t \cdot \sigma_f \ge 1/(4\pi)$ (Gabor, 1946). Per una finestra di Hann
  il prodotto vale $0{,}0817$ a **ogni** lunghezza: raddoppiare la finestra
  dimezza $\sigma_f$ e raddoppia $\sigma_t$, senza sconti.
- La **scala mel** è un adattamento a dati percettivi, non una legge: di formule
  ne esiste più d'una (HTK contro Slaney) e danno bande diverse, quindi la
  scelta va dichiarata.
- Gli **MFCC** (banco di filtri, logaritmo, DCT, primi $\sim 13$ coefficienti)
  sono una feature **d'archivio**: la DCT serviva a rendere lecita la covarianza
  diagonale delle GMM, e con le reti profonde quell'ipotesi non c'è più. I
  modelli di questo capitolo mangiano **spettrogramma log-mel grezzo**.
```

`````
