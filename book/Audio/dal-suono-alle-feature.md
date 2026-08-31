# Dal suono alle feature

Quando pronunciamo una parola non facciamo altro che spingere aria. Le corde
vocali vibrano, l'aria si comprime e si dirada (si *rarefà*, dicono i fisici), e
un'onda di pressione viaggia
fino al timpano di chi ascolta, o alla membrana di un microfono. Negli anni
Quaranta, ai Bell Labs, un gruppo guidato da Ralph Potter costruì il *sound
spectrograph*, una macchina che trasformava quest'onda in un'immagine e la
chiamò *visible speech*, "parola visibile". È esattamente il percorso che
compie oggi *qualsiasi* sistema che lavora sull'audio (riconoscere una voce,
un canto, un allarme) prima ancora di provare a capire *cosa* quel suono
significhi: trasformare un'onda in numeri, e i numeri in un'immagine su cui un
modello sa lavorare.

**Feature** sono i numeri con cui descriviamo una cosa da dare in pasto a un
modello: qui, i numeri con cui descriviamo un suono. Quali scegliere, e perché
proprio quelli, è tutto il resto.

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

Un sismografo disegna la scossa con un pennino che sale e scende su un rullo di
carta. Una macchina fa lo stesso mestiere con il suono, solo che invece di una
linea scrive numeri: annota dov'è la membrana del microfono, migliaia di volte
al secondo. Il risultato è una lunghissima lista: quanto era "in avanti" o "in
indietro" la membrana in ogni istante. Un suono, per il computer, è tutto qui:
una sequenza di numeri che sale e scende nel tempo. Numeri grandi (in positivo
o in negativo), quando il suono è forte, vicini allo zero quando c'è silenzio.

Due decisioni restano da prendere: ogni quanto guardare la membrana, e con
quanta finezza scrivere quello che si è visto. Il pennino può fermarsi ovunque,
il numero no: ogni misura viene arrotondata alla tacca più vicina di un
righello, e le tacche sono in numero fisso, deciso una volta per tutte.

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

Ognuna di quelle annotazioni si chiama **campione**: una singola misura della
posizione della membrana, presa in un istante preciso. Un suono digitale è una
fila di campioni, e le due domande che vengono subito dopo sono quanti
prenderne al secondo e quanto precisa debba essere ciascuna misura. La prima ha
la sezione qui sotto; alla seconda rispondiamo in due righe alla fine.

## Quanti campioni al secondo? Il teorema di Nyquist

La domanda è: quante volte al secondo dobbiamo misurare per non perdere
informazione? Troppo poche e il suono si deforma; troppe e sprechiamo memoria.

Attenzione, però, a un tranello di vocabolario, perché da qui in avanti la
parola «frequenza» fa due mestieri diversi. Uno lo conosciamo già: **la
frequenza di un suono**, quante volte al secondo oscilla l'aria, quella che dice
se è acuto o grave. L'altro è nuovo: **la frequenza di campionamento**, quante
volte al secondo *noi* andiamo a guardare dov'è la membrana. Il suono oscilla da
sé, e non gliene importa niente di noi; noi decidiamo ogni quanto guardare. Sono
due cose distinte, ma si scrivono tutte e due in hertz, ed è lì che ci si
confonde.

Un modo per non sbagliare mai: la frequenza di campionamento è una nostra
scelta, quella del suono no.

`````{tab} Elementare

Nei film le ruote delle auto ogni tanto sembrano girare al contrario, o stare
ferme: capita perché la telecamera scatta troppe poche foto al secondo per
tenere il passo del giro. Con il suono succede la stessa cosa, e la battuta
finale è altrettanto strana. Un fischio troppo acuto per il numero di
misure che stiamo prendendo non sparisce: si **traveste**, e torna indietro più
grave di quanto era, come una nota che nessuno ha mai suonato.

Dalla pellicola, poi, non si torna indietro: quella ruota che sembra girare al
rovescio non si distingue più da una ruota che girava davvero al rovescio.
Ecco perché il fischio troppo acuto si spegne prima di cominciare a misurare, e
non dopo.

La regola che tiene lontano il travestimento è semplice: misurare **più del
doppio** delle volte rispetto alla vibrazione più rapida che vogliamo
catturare. L'orecchio
umano arriva a circa $20\,000$ oscillazioni al secondo (20 kHz), quindi servono
più di $40\,000$ misure al secondo: i CD ne fanno $44\,100$, che stanno larghi
apposta. Per la voce al telefono bastano $8\,000$ misure al secondo, perché la
linea butta via, prima di misurare, tutte le oscillazioni sopra le $3\,400$ al
secondo, e sotto quella soglia la voce "vive" quasi tutta. (Il doppio di 3.400
sarebbe 6.800: gli 8.000 lasciano un margine, perché nessun filtro taglia di
netto.)

E c'è una via di mezzo, $16\,000$ misure al secondo, che copre le oscillazioni
fino a poco meno di $8\,000$ (la regola dice «più del doppio», quindi $8\,000$
tondi resterebbero fuori di un soffio): è la scelta abituale dei sistemi che
lavorano sulla voce. Per la musica non basta, e chi la tratta sale a 24.000,
32.000 o 48.000 misure al secondo: il brillare di un piatto della batteria, o
il fischio acutissimo di certi uccelli, vivono lassù, dove la voce non arriva.

`````

`````{tab} Superiore

È il **teorema del campionamento** di Nyquist–Shannon: per ricostruire senza
perdita un segnale la cui frequenza massima è $f_{\max}$, la frequenza di
campionamento deve soddisfare

$$
f_s > 2\,f_{\max}.
$$

dove $f_{\max}$ è la frequenza oltre la quale lo **spettro** del segnale (la
sua decomposizione in frequenze pure, prodotta dalla trasformata di Fourier) è
nullo: per un segnale che non sia una sinusoide pura, «la frequenza più alta»
non è definibile senza quella decomposizione, ed è la ragione per cui
l'enunciato di Nyquist ha bisogno di quello strumento.

La soglia $f_s/2$ è la **frequenza di Nyquist**. Se il segnale contiene
componenti oltre questa soglia, esse si "ripiegano" su frequenze più basse
generando l’**aliasing**: artefatti irreversibili. Per questo si applica un
filtro anti-aliasing (passa-basso) *prima* di campionare. La voce viene tipicamente
trattata a $f_s = 16\,\text{kHz}$, un buon compromesso tra fedeltà e peso.

`````

Resta in sospeso la seconda domanda: quanto precisa deve essere ogni singola
misura? La risposta abituale è che ogni campione si scrive con 16 risposte
sì/no, e siccome ogni risposta raddoppia i casi possibili, sedici ne fanno
65.536. Sono abbastanza da non farsi sentire: fra un valore e il successivo la
differenza è troppo piccola perché l'orecchio la colga. Ci torneremo
nell'ultima sezione del capitolo, dove un modello famoso si accontenta di 256
valori e deve inventarsi un trucco per farli bastare.

## Dal tempo alla frequenza: la trasformata di Fourier

La lista di campioni che abbiamo in mano ha un nome, **forma d'onda**, ed è il
disegno che farebbe un sismografo: quanto era in avanti o indietro la membrana,
istante per istante. Ci dice *quando* il suono è forte o debole, ma non ci dice
di quali "note" è fatto. Eppure è proprio quella composizione a distinguere una
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

Per strada non si perde niente. I colori rimessi insieme ridanno la luce
bianca, e le frequenze rimesse insieme ridanno l'onda esatta di partenza: sono
gli stessi numeri, tanti quanti erano, scritti in un altro alfabeto.

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
risultati. Questa è la trasformata di Fourier a finestre brevi, che in
letteratura si trova sempre con la sigla inglese: **STFT**, *Short-Time Fourier
Transform*. Il risultato,
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
spettrogramma: è lunga 25 millesimi di secondo e avanza di 10 alla volta, quindi
due finestre vicine coprono in parte lo stesso pezzo di suono, mentre le colonne
che ne escono restano una accanto all'altra. Le note che salgono si vedono come
gradini, ed è esattamente l'informazione che la trasformata del file intero non
saprebbe dare.
```

Le finestre a cavallo fra una nota e l'altra mostrano **entrambe** le frequenze,
ed è la cosa più istruttiva del disegno: è il compromesso su cui la trasformata
a finestre è costruita, non una sbavatura. Una finestra lunga distingue bene le
frequenze e male gli istanti, perché dentro ci finiscono due note; una corta fa
il contrario. Non esiste una lunghezza che vinca su tutti e due i fronti, ed è
per questo che sceglierla è una decisione e non un dettaglio.

Una parola sulla **forma** della finestra, perché nel disegno è gonfia in mezzo
e va a zero ai bordi, invece del rettangolo che «finestrella» lascia
immaginare.
Tagliare di netto un pezzo di onda creerebbe due gradini artificiali agli
estremi, e la trasformata leggerebbe quei gradini come frequenze che nel suono
non ci sono. Smussando i bordi il taglio diventa una dissolvenza e il difetto
quasi sparisce, al prezzo di dare meno peso a ciò che capita ai margini. La
curva più usata per farlo si chiama **finestra di Hann**, dal nome del
meteorologo austriaco Julius von Hann, ed è quella disegnata in
{numref}`fig-finestra-spettrogramma`.

`````{tab} Elementare

Chi accorda un pianoforte non guarda niente, ascolta. Tiene il diapason vicino
alla corda del la e aspetta che il suono ondeggi: due note quasi uguali si
rinforzano e si smorzano a turno, e quelle ondate si contano. Più sono vicine,
più le ondate vengono lente, e più bisogna aspettare per sentirne una. Chi ha
fretta sente una nota sola e se ne va convinto.

La finestra ascolta allo stesso modo, e paga lo stesso prezzo: per dire che
nota era deve sentirla oscillare un po’ di volte, e quel po’ di tempo è
esattamente l'istante che si perde.

I due lati del baratto hanno dei numeri. Con la finestra da 25 millesimi di
secondo il quando si mette a fuoco entro 3,5 millesimi, la nota entro 23
oscillazioni al secondo. Il primo è più piccolo della finestra perché non ne
misura la lunghezza, ma quanto il suo peso sta stretto attorno al centro: la
finestra è gonfia in mezzo e va a zero ai bordi.

Il secondo va preso con cautela: 23 è quanto una nota sola si spalma, non la
distanza minima fra due note che si riescano ancora a separare. Quella è più
larga, una sessantina di oscillazioni al secondo: due note così distanti
ondeggiano una volta e mezza dentro i 25 millesimi, appena abbastanza per
accorgersene. Due tasti vicini attorno al la ne distano 26, e di ondate non ne
fanno nemmeno una: restano una macchia sola.

Allunghiamo la finestra a 100 millesimi, quattro volte tanto. Adesso le ondate
ci stanno, e i due tasti si separano appena: fra i due picchi si apre un
avvallamento, ancora poco profondo. La precisione sulle note migliora di
quattro (23 diviso 4 fa 5,75) e quella sugli istanti peggiora di quattro (3,5
per 4 fa 14 millesimi). Moltiplichiamole, prima e dopo:
$3{,}5 \times 23 = 80{,}5$ e $14 \times 5{,}75 = 80{,}5$. Identico.

Quel prodotto resta lo stesso comunque si allunghi o accorci la finestra.
Cambiandole la *forma* si scende un pochino, poi ci si ferma: sotto una certa
soglia non ci va nessuna curva. Nemmeno guardare più spesso aiuta: colonne più
fitte sono la stessa macchia ridisegnata più larga.

`````

`````{tab} Superiore

Formalmente la STFT di un segnale $\mathbf{x}$, i cui campioni sono gli $x[n]$, è

$$
X[m,k] = \sum_{n} x[n]\, w[n - mH]\, e^{-\,i\,2\pi kn/N},
$$

dove $\mathbf{w}$ è la **finestra** (lunga $N$ campioni), $H$ il **passo** (*hop*) di cui
essa avanza da una colonna alla successiva, $m$ l'indice della colonna e $k$
quello del bin di frequenza. Con frequenza di campionamento $f_s$ la trasformata restituisce bin spaziati
di $f_s/N$ hertz e una colonna ogni $H/f_s$ secondi. Sono i passi con cui
*campioniamo* il piano tempo-frequenza, e non vanno confusi con la
risoluzione: infittirli (zero-padding in frequenza, $H$ più piccolo nel tempo)
non aggiunge informazione, la interpola soltanto. La risoluzione vera dipende
dalla lunghezza della finestra (a parità di forma, solo da quella), e ha un
limite che nessuna scelta di parametri aggira. Per il parlato la scelta
standard è una finestra di 25 ms e un passo di 10 ms (a 16 kHz sono
$N = 400$ campioni e $H = 160$, cioè
`n_fft=400` e `hop_length=160` per `librosa`), perché i fonemi durano decine di
millisecondi e una finestra più lunga ne mescolerebbe due.

Il compromesso è **teorico** e non pratico: viene dalla relazione di
indeterminazione di Gabor (1946), l'analogo per l'analisi di Fourier del
principio di indeterminazione di Heisenberg,

$$
\sigma_t \cdot \sigma_f \ \geq\ \frac{1}{4\pi},
$$

dove $\sigma_t$ e $\sigma_f$ sono le deviazioni standard dell'energia della
finestra nel tempo e in frequenza. L'uguaglianza vale per la finestra
gaussiana; per la finestra di Hann il prodotto vale $0{,}0817$ contro il limite
$1/(4\pi) = 0{,}0796$, ed è **costante al variare della lunghezza**: la Hann da
25 ms ha $\sigma_t \approx 3{,}54$ ms e $\sigma_f \approx 23{,}1$ Hz, quella da
100 ms $\sigma_t \approx 14{,}1$ ms e $\sigma_f \approx 5{,}77$ Hz. Quadruplicare la
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

Lo spettrogramma dice tutto, e proprio per questo dice troppo: righe vicine si
somigliano parecchio, e molti dei suoi numeri non fanno che ripetere quello che
c'è accanto. Possiamo riassumerlo, e conviene farlo imitando il modo in cui
l'orecchio percepisce davvero i suoni: quello che l'orecchio butta via possiamo
buttarlo via anche noi, senza rimpianti. Il riassunto più famoso porta la sigla
inglese **MFCC**, da *mel-frequency cepstral coefficients*, cioè «coefficienti
cepstrali in scala mel»: quattro parole difficili, e le prossime righe le
sciolgono una a una.

«Cepstrale» non è un refuso per «spettrale». È un gioco di parole degli
ingegneri che negli anni Sessanta inventarono il metodo: rovesciarono le prime
lettere di *spectrum* (*spec* diventa *ceps*) per dire che allo spettro si
applica una **seconda** trasformata, dopo quella di Fourier. La prima scompone
il suono nelle sue frequenze; la seconda prende quella scomposizione e ne ricava
un riassunto ancora più corto. Il termine è rimasto.

`````{tab} Elementare

Il nostro orecchio non è un righello: distingue benissimo due note gravi
vicine, ma fatica con due note acute altrettanto vicine. La **scala mel**
riscrive le frequenze proprio così, come le sente una persona. Persone vere,
per la precisione: la scala esce da gente messa ad ascoltare suoni in
laboratorio, e non tutti hanno risposto allo stesso modo. Di scale mel ne
circola più d'una, le bande che ne escono non coincidono, e quale si è usata va
detto.

Gli MFCC prendono lo spettrogramma, lo rileggono con questa scala e lo
riassumono in una manciata di numeri per finestrella (di solito 13) che
catturano la "forma" del suono buttando via i dettagli inutili. Perché proprio
13? Nessuna legge di natura: negli anni Ottanta funzionavano bene sul parlato, e
sono rimasti.

Quel riassunto serviva a macchine con pochissima memoria e pochissimo calcolo, e
serviva a un'altra cosa ancora: quelle macchine davano per buono che i 13 numeri
raccontassero ciascuno un fatto suo, indipendente dagli altri. Righe vicine
dell'immagine invece si somigliano parecchio, e l'ultimo passaggio del riassunto
rimescola tutto apposta per togliere di mezzo quella somiglianza. Le reti di
oggi non chiedono niente del genere e si fermano **un passo prima**: prendono
l'immagine riletta a orecchio e saltano il riassunto finale. Anzi, quel
riassunto le danneggerebbe, perché mescolando frequenze lontane cancella la
vicinanza fra righe che stanno accanto, che è proprio ciò su cui una rete
lavora.

Resta il nome che ricorre dappertutto: **log-mel**.
La parte «mel» è la riscrittura delle frequenze a orecchio. La parte «log» è la
stessa idea applicata alle *intensità*: fra un sussurro e un concerto ci passa
un fattore diecimila, e sulla stessa immagine il sussurro sparirebbe sotto
l'altro. Allora si schiacciano, in modo che passare da 1 a 10 conti quanto
passare da 10 a 100 e da 100 a 1.000. È il trucco dei **decibel**, la scala con
cui si misurano i rumori: quel fattore diecimila diventa la distanza fra 30 (una
biblioteca) e 110 (un concerto), due numeri vicini per due mondi lontanissimi.
Uno spettrogramma log-mel è l'immagine del suono con le frequenze riscritte a
orecchio e le intensità schiacciate allo stesso modo.

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
danno le stesse bande: con i parametri dell'esempio in `librosa` ($f_s = 16$
kHz, 40 bande) la decima banda è centrata a 594 Hz con la formula HTK e a
736 Hz con quella di Slaney, il 24 % più in alto. I centri si leggono da
`librosa.mel_frequencies(n_mels=42, fmax=8000, htk=…)[1:-1]`, perché per 40
bande triangolari servono 42 frequenze e le due estreme sono bordi, non centri.
È una differenza che va dichiarata quando
si confrontano due sistemi, ed è il motivo per cui l'esempio passa `htk=True`:
perché faccia davvero il conto di O'Shaughnessy.

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
grezzo**, mentre i 13 MFCC restano una feature d'archivio, ancora comoda dove
serve un vettore piccolo (HuBERT li usa proprio per il primo raggruppamento).

Da qui in poi le strade si dividono, e la divisione attraversa tutto il resto
del capitolo. Chi in uscita ha un'etichetta o del testo parte dal log-mel:
l'AST della prossima sezione ne prende 128 bande, Whisper, che è del capitolo
dopo, ne prende 80, e 128 nelle versioni più recenti. Chi in uscita ha del
*suono* parte invece dai campioni, perché il log-mel butta via la fase e non si
torna indietro: sono i codec e i generatori delle ultime due sezioni. E chi
vuole imparare tutto dai dati sceglie i campioni comunque, e il banco di filtri
che qui abbiamo disegnato a mano se lo costruisce da sé.

`````

## Perché queste feature aiutano il modello

La forma d'onda grezza è enorme (decine di migliaia di numeri al secondo) e
piena di variazioni che al modello non servono: quanto forte è stata registrata
la stessa frase, il rumore di fondo della stanza, il fatto che il suono cominci
due millesimi di secondo prima o dopo. Sono differenze vere, ma non cambiano
*che suono è*, ed è quello che vogliamo sapere. Lo spettrogramma mel (e, nei
sistemi di una volta, gli MFCC) concentra l'informazione che serve a
riconoscerlo, cioè quali frequenze e quando, scartando gran parte del resto.
Il compito del modello diventa così più facile:
parte da una descrizione più piccola, più stabile e già "orientata" verso ciò
che distingue un suono dall'altro.

I sistemi più recenti, come wav2vec 2.0 {cite}`baevski2020wav2vec` o Whisper
{cite}`radford2022robust`, tendono a imparare le feature direttamente dai dati.
Ma non partono dal nulla: Whisper, per esempio, riceve in input proprio uno
**spettrogramma log-mel**. La scala mel, ispirata al nostro orecchio, resta il
punto di partenza più diffuso anche nell'era del deep learning.

La parola **feature**, quella del titolo, qui cambia padrone. Fin dal
{doc}`capitolo sul machine learning </MachineLearning/overview>` le feature
erano *scelte da noi*: qualcuno decideva quali numeri estrarre da ogni esempio,
e quella decisione era metà del lavoro. Da questa pagina in poi saranno quasi
sempre *imparate dalla rete*, che si costruisce da sé i numeri che le servono.
La parola indica lo stesso oggetto (i numeri che descrivono un esempio) e
cambia solo chi li sceglie; ma è il passaggio che divide questa sezione da
tutte quelle che seguono, e conviene saperlo prima di incontrarlo scritto come
se fosse ovvio.

## In pratica

Con la libreria `librosa` l'intera catena (dal file audio allo spettrogramma
mel) è una manciata di righe.

```python
import librosa

# carica l'audio prendendo 16.000 misure al secondo (lo standard per la voce)
y, sr = librosa.load("frase.wav", sr=16000)

# spettrogramma mel: 40 bande, finestre da 25 ms ogni 10 ms
# htk=True sceglie una delle due scale mel in circolazione: danno bande
# diverse, quindi qual e' delle due va sempre dichiarato
S = librosa.feature.melspectrogram(
    y=y, sr=sr, n_fft=400, hop_length=160, n_mels=40, htk=True
)
# intensita' schiacciate (la parte "log" del log-mel): i suoni deboli
# tornano visibili accanto a quelli forti
S_db = librosa.power_to_db(S, ref=S.max())

# 13 coefficienti MFCC per ogni finestra temporale: il riassunto piu' corto.
# Si calcolano DALLA S_db appena ottenuta. Chiamando invece mfcc(y=...) librosa
# rifarebbe lo spettrogramma da capo con i propri default (n_fft=2048,
# hop_length=512), quindi su un asse dei tempi diverso dal nostro.
mfcc = librosa.feature.mfcc(S=S_db, n_mfcc=13)

print(S_db.shape, mfcc.shape)  # (bande, tempo) e (coefficienti, tempo)
```

Le due tabelle escono con lo stesso numero di colonne: sono allineate finestra
per finestra, quindi si possono affiancare e dare al modello insieme. Sarebbe
bastato chiamare `mfcc(y=...)` invece che `mfcc(S=...)` per ritrovarsi due
assi dei tempi diversi e un errore di dimensione incomprensibile.

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
  allungarle di quattro volte fa guadagnare quattro sulle note e perdere quattro
  sugli istanti, e il prodotto delle due precisioni resta lo stesso. Sceglierne
  la lunghezza vuol dire decidere quale delle due si compra.
- La **scala mel** e gli **MFCC** rileggono quell'immagine come la sente un
  orecchio (preciso sui suoni gravi, approssimativo sugli acuti) e la
  riassumono in pochi numeri per finestrella: meno dati, ma quelli che contano
  davvero, e il modello ha un compito più facile.
- I modelli di oggi però si fermano **un passo prima**: prendono l'immagine
  riletta a orecchio (lo spettrogramma **log-mel**, con anche le intensità
  schiacciate) e saltano il riassunto finale. Quel riassunto serviva a macchine
  con molta meno memoria, che per giunta pretendevano numeri indipendenti l'uno
  dall'altro; a una rete non serve, e anzi le cancella la vicinanza fra righe
  accanto, che è ciò su cui lavora. È il primo esempio di una cosa che si
  ripeterà: un accorgimento intelligente smette di servire quando cambia chi lo
  usa.
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
  il prodotto vale $0{,}0817$ a ogni lunghezza: raddoppiare la finestra
  dimezza $\sigma_f$ e raddoppia $\sigma_t$, senza sconti.
- La **scala mel** è un adattamento a dati percettivi, non una legge: di formule
  ne esiste più d'una (HTK contro Slaney) e danno bande diverse, quindi la
  scelta va dichiarata.
- Gli **MFCC** (banco di filtri, logaritmo, DCT, primi $\sim 13$ coefficienti)
  sono una feature **d'archivio**: la DCT serviva a rendere lecita la covarianza
  diagonale delle GMM, e con le reti profonde quell'ipotesi non c'è più. Chi in
  uscita ha un'etichetta o del testo mangia **log-mel grezzo**; chi produce
  suono parte dai **campioni**, perché il log-mel non si inverte.
```

`````
