# Speech Recognition: dalla voce al testo

```{image} ../figures/aperture/speech-recognition.png
:class: pt-apertura only-light
:width: 100%
:alt: Un microfono d'epoca sull'asta, con le onde della voce che diventano una fila di punti.
```

```{image} ../figures/aperture/speech-recognition-scura.png
:class: pt-apertura only-dark
:width: 100%
:alt: Un microfono d'epoca sull'asta, con le onde della voce che diventano una fila di punti.
```

Nel 1952, nei laboratori della Bell Telephone, un apparecchio ingombrante fatto
di valvole e relè (l'elettronica di allora, con i suoi bulbi di vetro
incandescenti e gli interruttori che scattano da soli) imparò a riconoscere le
cifre da zero a nove pronunciate ad alta voce. Lo chiamarono poi **Audrey**, e
aveva un limite quasi comico: funzionava con un solo parlante, che poteva
parlare a velocità normale ma doveva staccare nettamente una cifra dall'altra,
con una pausa in mezzo. Se erano attaccate, non le riconosceva più. Settant'anni
dopo diciamo "metti la sveglia alle sette" al telefono mentre carichiamo la
lavastoviglie, e la frase diventa testo (e poi azione) in una frazione di
secondo. In mezzo c'è la storia dell’ASR (*Automatic Speech Recognition*), il
riconoscimento vocale automatico.

Gli strumenti per trattare il suono li ha già costruiti il {doc}`capitolo
sull'audio </Audio/overview>`; qui servono sulla voce. In questo capitolo
definiamo il compito, vediamo perché è difficile e ripercorriamo la storia da
Audrey a Whisper; descriviamo poi le tappe che il suono attraversa dal
microfono al testo e chiudiamo con i luoghi in cui il riconoscimento vocale si
incontra.

## Che cosa fa, in fondo

Il compito è facile da enunciare: prendere un segnale audio e restituire le
parole che contiene.

`````{tab} Elementare

Ascolti una registrazione e la trascrivi a mano: ricevi un flusso continuo di
suono (alti, bassi, pause) e produci una riga di parole scritte. Dieci secondi
di registrazione diventano una ventina di parole, e [nessuno ti dice quanti
centesimi di secondo tocchino a ciascuna]{.pt-matita}. Dove il parlato è
impastato torni indietro, riascolti, provi due letture possibili e scrivi
quella che ti convince di più. Il riconoscitore vocale fa lo stesso: in
ingresso l'onda sonora catturata dal microfono, in uscita del testo. Del
significato non capisce niente, e si ferma alle lettere giuste.

`````

`````{tab} Superiore

Formalmente l'ASR è un problema di **trasduzione di sequenze**: da una sequenza
di vettori acustici $\mathbf{X} = (\mathbf{x}_1, \dots, \mathbf{x}_T)$
vogliamo la sequenza di parole $\hat{W}$ più probabile,

$$
\hat{W} = \arg\max_{W} P(W \mid \mathbf{X}).
$$

Si noti che $\mathbf{X}$ è tipicamente lunga centinaia o migliaia di passi (uno
ogni dieci millisecondi circa), mentre $W$ conta poche decine di parole: input
e output hanno lunghezze molto diverse e non allineate una-a-una. [Gestire
questo disallineamento temporale è il cuore tecnico del problema]{.pt-matita}.

`````

## Perché è difficile

Nessuno pronuncia due volte la stessa parola allo stesso modo. La voce cambia
con la persona (timbro, accento, velocità), con lo stato d'animo, con il
raffreddore. Poi c'è il rumore di fondo: traffico, musica, altre voci
sovrapposte.

A questo si aggiunge la **coarticolazione**, il fenomeno per cui i suoni si
contaminano a vicenda. Ad esempio, la "n" di *un cane* non è la stessa "n" di
*un pane*, perché la bocca si sta già preparando al suono che segue. E ci sono
gli **omofoni**: *l'ago* e *lago* suonano identici, e solo il contesto della
frase dice quale sia quello giusto.

Infine, un microfono non "vede" gli spazi tra le parole: il parlato è un flusso
continuo, e trovare dove finisce una parola e inizia l'altra è già metà del
lavoro.

## Da Audrey a Whisper: una breve storia

I primi sistemi, fino agli anni Settanta, lavoravano per somiglianza:
tenevano in memoria una registrazione di ogni parola del loro piccolo
vocabolario e cercavano quella più simile al suono appena arrivato,
allungandolo o comprimendolo nel tempo per farlo combaciare. L'operazione ha
un nome che ogni tanto si incontra ancora, *dynamic time warping*, «deformare
il tempo». Andava bene per pochissime parole e un solo parlante, perché
aggiungerne una voleva dire registrarla.

Il salto di qualità, preparato negli anni Settanta all'IBM e a Carnegie Mellon
e diventato la norma nel decennio seguente, [arrivò da due idee, non da
macchine più potenti]{.pt-matita}. La prima fu smettere di ragionare per parole
intere e ragionare per **fonemi**, i suoni minimi che distinguono una parola
dall'altra (*pane* e *cane* differiscono per uno solo). Infatti le parole di
una lingua sono centinaia di migliaia, i fonemi qualche decina, e una parola
mai sentita prima è comunque fatta di fonemi già sentiti. La seconda fu
dividere il lavoro in due pareri.

`````{tab} Elementare

Un pezzo del sistema giudica *quanto questo suono somiglia alla parola
«cane»*; un altro pezzo, che ha letto montagne di testo italiano, giudica
*quanto è plausibile la frase «il cane abbaia» rispetto a «il cane
abbaglia»*. La trascrizione finale è quella che supera il giudizio di entrambi;
se uno dei due la boccia, è fuori, per quanto l'altro la lodi. Ecco perché il
contesto risolve gli omofoni: il suono non basta, serve sapere quali frasi
hanno senso.

`````

`````{tab} Superiore

Applicando il teorema di Bayes, il problema si scompone nel classico modello a
*canale rumoroso*:

$$
\hat{W} = \arg\max_{W}\
\underbrace{P(\mathbf{X} \mid W)}_{\text{modello acustico}}\
\underbrace{P(W)}_{\text{modello di linguaggio}}.
$$

(Il denominatore $P(\mathbf{X})$ di Bayes è sparito legittimamente: non dipende
da $W$, quindi non altera l'argmax.) Il modello acustico $P(\mathbf{X} \mid W)$
misura quanto i suoni osservati siano
compatibili con una data sequenza di parole, e storicamente si scompone ancora
in due: un **dizionario di pronuncia** traduce $W$ in fonemi, e ogni fonema,
preso nel contesto del precedente e del successivo (il *trifone*), è un
*Hidden Markov Model* sinistra-destra di tre stati. Gli HMM dei trifoni, messi
in fila nell'ordine delle parole di $W$, formano un HMM composto; con
$Q = (q_1, \dots, q_T)$ una sequenza dei suoi stati e $q_0$ il suo stato
iniziale,

$$
P(\mathbf{X} \mid W) = \sum_{Q} \prod_{t=1}^{T} P(q_t \mid q_{t-1})\, p(\mathbf{x}_t \mid q_t),
\qquad
p(\mathbf{x} \mid q = j) = \sum_{m=1}^{M} \pi_{jm}\,\mathcal{N}\!\big(\mathbf{x};\, \boldsymbol{\mu}_{jm},\, \boldsymbol{\Sigma}_{jm}\big),
$$

dove l'emissione di ogni stato è una mistura di $M$ gaussiane (GMM) con
covarianze $\boldsymbol{\Sigma}_{jm}$ diagonali. Gli stati dei trifoni sono
decine di migliaia, troppi per stimarli uno per uno: per questo si raggruppano
con alberi di decisione fonetici e si addestrano con l'algoritmo EM di
Baum-Welch.

In decodifica la somma su $Q$ diventa il massimo di Viterbi, e in questa fase
il modello di linguaggio entra elevato a un peso, $P(W)^{\lambda}$ con
$\lambda$ di solito fra $5$ e $15$. Il peso serve perché il modello acustico
moltiplica le densità di centinaia di frame come se fossero indipendenti,
mentre frame vicini si somigliano molto; di conseguenza i suoi punteggi escono
fuori scala rispetto a quelli del linguaggio, e $\lambda$ li riequilibra. Il
modello di linguaggio $P(W)$ assegna una probabilità a priori alle frasi
($n$-grammi, oggi reti neurali) e disambigua gli omofoni.

Dal 2012 le reti neurali profonde sostituiscono le GMM nel modello acustico
{cite}`hinton2012deep`. Ad esempio, su una prova di conversazioni telefoniche
(Switchboard, RT03S), con le stesse trecento ore circa di addestramento,
l'errore sulle parole scende dal $27{,}4\%$ del sistema a GMM al $18{,}5\%$, un
terzo in meno.

`````

Il parere sul suono spetta al **modello acustico**, quello sulla plausibilità
della frase come italiano al **modello di linguaggio**.

Su questa divisione del lavoro il riconoscimento vocale ha poggiato per
trent'anni, dagli anni Ottanta al 2010 circa, e la sigla che lo designa nei
testi di quel periodo è **HMM-GMM**: due modelli già incontrati, che qui
lavorano in coppia.

Il primo, l'HMM («modello di Markov nascosto»), descrive una parola come
una fila di stati che non si vedono. Ogni fonema ne ha tre, il suo attacco, il
suo centro e la sua coda, e la voce passa dall'uno all'altro senza che nessuno
lo osservi.

Si osservano soltanto i *frame*, le fettine di suono lunghe pochi centesimi di
secondo in cui il segnale viene tagliato (ne comincia una ogni dieci
millisecondi), e da quelli si risale agli stati. È la stessa macchina di
{doc}`POS tagging ed entità </NaturalLanguageProcessing/etichettare-sequenze>`,
dove gli stati nascosti erano le categorie grammaticali e quello che si
osservava erano le parole.

Il secondo, la GMM (mistura di gaussiane), dice che aspetto hanno i frame
che ciascuno stato produce. Ogni frame è ridotto a un vettore di poche decine
di misure, e la «a» di mille persone diverse non dà mai lo stesso vettore: dà
una nuvola di vettori vicini, fitta attorno a un centro e sempre più rada man
mano che ci si allontana.

Una gaussiana è la curva che descrive una nuvola così, e dove la nuvola è
storta se ne sommano parecchie. È lo stesso strumento di
{doc}`Riduzione e clustering </MachineLearning/riduzione-clustering>`, dove
serviva a trovare gruppi nei dati; qui ogni stato dell'HMM ha la sua.

L'ultimo salto è l'approccio **end-to-end** ("da un capo all'altro"): una sola
rete neurale che impara direttamente il passaggio dall'audio al testo, senza
scomporre a mano acustica e linguaggio.

Tecniche come la CTC (*Connectionist Temporal Classification*
{cite}`graves2006connectionist`) e i modelli con attenzione (una parte della
rete ascolta tutto, l'altra scrive, e mentre scrive torna a guardare il punto
dell'audio che le serve: quel tornare a guardare è l'attenzione) hanno reso
possibile addestrare l'intero sistema da coppie (audio, testo). Il trasduttore è
nato dalla CTC per aggiungerle ciò che le manca, la memoria del testo già
scritto, senza toglierle l'andatura frame per frame che permette di trascrivere
mentre si parla. Le tratta una per una la sezione {doc}`I modelli di
riconoscimento </SpeechRecognition/modelli-asr>`.

L'esempio più noto è Whisper di OpenAI (2022): con un unico modello trascrive
quasi cento lingue, italiano compreso, e traduce in inglese il parlato delle
altre. È stato addestrato su circa 680.000 ore di audio multilingue: per farsi
un'idea, sono quasi settantotto anni di parlato ininterrotto, giorno e notte,
senza una pausa.

Il modello alla sua base è un Transformer, l'architettura del {doc}`capitolo
sui Transformer </Transformers/overview>`, montata in due metà: un *encoder*,
che ascolta l'audio e lo riassume, e un *decoder*, che da quel riassunto scrive
il testo.

## La catena di montaggio, passo per passo

Dal microfono al testo il suono passa per alcune tappe, sempre le stesse
({numref}`fig-asr-pipeline`).

Nei sistemi end-to-end di oggi il modello acustico e il modello di linguaggio
sono finiti dentro un'unica rete e non si vedono più dall'esterno, mentre
l'estrazione delle feature resta un calcolo fisso, fatto prima che il suono
entri nella rete; i nomi delle tappe però sono rimasti in uso. La catena nel
suo insieme, in inglese, è la *pipeline* del riconoscimento vocale.

```{figure} ../figures/asr-pipeline.svg
:name: fig-asr-pipeline
:alt: "Diagramma di flusso a cinque stadi: audio, estrazione delle feature, modello acustico, modello di linguaggio, testo."
:width: 90%

Le cinque tappe del riconoscimento vocale. Dall'onda sonora si ricavano poche
misure per ogni frammento (le feature), il modello acustico giudica a quali
suoni somigliano, il modello di linguaggio sceglie fra le frasi che quei suoni
consentono, e in fondo esce il testo.
```

La prima tappa riduce il suono all'essenziale, e si chiama **estrazione delle
feature** («caratteristiche», in inglese): al posto dell'onda grezza, poche
misure per ogni frammento di suono. È il passaggio che rende trattabile tutto
il resto.

`````{tab} Elementare

Il segnale grezzo è troppo minuto e dettagliato per lavorarci direttamente.
Allora lo si taglia in fettine di pochi centesimi di secondo e, per ognuna, si
misura "quanta energia c'è a ciascuna altezza sonora": un po’ come le barrette
colorate che ballano nelle app della musica, dove le più a sinistra si alzano
sui suoni gravi e le più a destra sugli acuti. Le fettine si prendono
accavallate, così una consonante a cavallo di un taglio non va persa. Questa
sequenza di istantanee sonore è lo spettrogramma costruito passo per
passo in {doc}`Dal suono alle feature </Audio/dal-suono-alle-feature>`, e il
[modello acustico ascolta questo al posto dell'onda]{.pt-matita}.

Un tempo lo riceveva anche più spremuto, una dozzina di numeri per fettina,
perché lavorava bene solo con poche misure che non si ripetessero fra loro, e
le bande vicine dello spettrogramma salgono e scendono quasi insieme; oggi le
reti se lo prendono così com'è.

`````

`````{tab} Superiore

Il segnale continuo viene diviso in frame sovrapposti e trasformato nello
spettrogramma log-mel costruito in {doc}`Dal suono alle
feature </Audio/dal-suono-alle-feature>`: [è la matrice $\mathbf{X}$ che entra
nel modello acustico]{.pt-matita}. I sistemi classici applicavano un ultimo
passaggio, la trasformata coseno discreta (DCT), che decorrelava i coefficienti
e riduceva ogni frame a una dozzina di MFCC (*Mel-Frequency Cepstral
Coefficients*, una quarantina con le derivate prima e seconda). Serviva alle
covarianze diagonali delle GMM, mentre i sistemi neurali end-to-end, Whisper
compreso, non lo fanno più.

`````

## Dove lo incontriamo

Il riconoscimento vocale è ormai ovunque: gli assistenti vocali (Siri,
Alexa, Google Assistant) che aspettano un comando; la sottotitolazione
automatica di YouTube e delle videochiamate, preziosa per l'accessibilità;
la dettatura che trasforma la voce in documenti, dai referti medici ai
messaggi scritti mentre si guida. È la porta d'ingresso di quasi ogni sistema
che "capisce" il parlato: una volta che la voce è diventata testo, tocca agli
strumenti del linguaggio naturale interpretarne il significato. Ma quel primo,
difficile passo (dall'onda alla parola) comincia sempre qui.

Resta un passaggio in cui si sbaglia ancora, ogni giorno, sotto i nostri
occhi: i sottotitoli automatici che scrivono una parola per un'altra, o che
riempiono di frasi inventate un pezzo di video in cui nessuno parla, sono la
parte visibile di limiti precisi, che {doc}`I modelli di riconoscimento
</SpeechRecognition/modelli-asr>` racconta uno per uno.

Il viaggio ha anche un ritorno: {doc}`La voce sintetica
</SpeechRecognition/sintesi-vocale>` percorre la strada opposta, dal testo
all'onda sonora, e chiude il cerchio tra ascoltare e parlare.
