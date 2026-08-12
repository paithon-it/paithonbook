# Speech Recognition: dalla voce al testo

Nel 1952, nei laboratori della Bell Telephone, un apparecchio ingombrante
fatto di valvole e relè (l'elettronica di allora: bulbi di vetro
incandescenti e interruttori che scattano da soli) imparò a riconoscere le
cifre da zero a nove
pronunciate ad alta voce. Si chiamava **Audrey** e aveva un limite quasi
comico: funzionava con un solo parlante, che doveva scandire i numeri
lentamente e fare una pausa netta fra l'uno e l'altro. Settant'anni dopo
diciamo "metti la sveglia alle sette" al telefono mentre carichiamo la
lavastoviglie, e la frase diventa testo (e poi azione) in una frazione di
secondo. In mezzo c'è la storia dell'**ASR** (*Automatic Speech Recognition*),
il riconoscimento vocale automatico.

## Che cosa fa, in fondo

Il compito è facile da enunciare: prendere un segnale audio e restituire le
parole che contiene.

`````{tab} Elementare

Pensa a una persona che ascolta una registrazione e la trascrive a mano.
Riceve un flusso continuo di suono (alti, bassi, pause) e produce una riga di
parole scritte. Il riconoscitore vocale fa esattamente questo: in ingresso
l'onda sonora catturata dal microfono, in uscita del testo. Attenzione:
nessuna *comprensione* del significato, per ora. Solo il passaggio dal suono
alle lettere giuste.

`````

`````{tab} Superiore

Formalmente l'ASR è un problema di **trasduzione di sequenze**: da una sequenza
di vettori acustici $X = (x_1, \dots, x_T)$ vogliamo la sequenza di parole
$\hat{W}$ più probabile,

$$
\hat{W} = \arg\max_{W} P(W \mid X).
$$

Qui $X$ è tipicamente lunga centinaia o migliaia di passi (uno ogni dieci
millisecondi circa), mentre $W$ conta poche decine di parole: input e output
hanno lunghezze molto diverse e non allineate una-a-una. Gestire questo
disallineamento temporale è il cuore tecnico del problema.

`````

## Perché è difficile

Nessuno pronuncia due volte la stessa parola allo stesso modo. La voce cambia
con la persona (timbro, accento, velocità), con lo stato d'animo, con il
raffreddore. C'è il **rumore di fondo**: traffico, musica, altre voci
sovrapposte. C'è la **coarticolazione**, il fenomeno per cui i suoni si
contaminano a vicenda: la "n" di *un cane* non è la stessa "n" di *un pane*,
perché la bocca si sta già preparando al suono che segue. E ci sono gli
**omofoni**: *l'ago* e *lago* suonano identici, e solo il contesto della frase
dice quale sia quello giusto. Un microfono non "vede" gli spazi tra le parole:
il parlato è un flusso continuo, e trovare dove finisce una parola e inizia
l'altra è già metà del lavoro.

## Da Audrey a Whisper: una breve storia

I primi sistemi, fino agli anni Settanta, funzionavano per **template
matching**: memorizzavano un modello sonoro di ogni parola e cercavano quello
più simile al suono in arrivo, stirandolo o comprimendolo nel tempo per farlo
combaciare (una tecnica detta *dynamic time warping*). Andava bene per
vocabolari minuscoli e un unico parlante. Il salto di qualità arrivò con una
famiglia di modelli statistici nota con la sigla **HMM-GMM**: HMM sta per
«modello di Markov nascosto» (il parlato descritto come una catena di stati
che non si vedono, uno dietro l'altro) e GMM per «mistura di gaussiane» (ogni
stato descritto non da un suono solo, ma da una nuvola di suoni possibili).
Fu l'impianto dominante dagli anni Ottanta fino al 2010 circa.

`````{tab} Elementare

L'idea vincente fu dividere il lavoro in due pareri. Un pezzo del sistema
giudica *quanto questo suono somiglia alla parola «cane»*; un altro pezzo, che
ha letto montagne di testo italiano, giudica *quanto è plausibile la frase «il
cane abbaia» rispetto a «il cane abbaglia»*. La trascrizione finale è il
miglior compromesso fra i due giudizi. Ecco perché il contesto risolve gli
omofoni: da solo il suono non basta, serve sapere che frasi hanno senso.

`````

`````{tab} Superiore

Applicando il teorema di Bayes, il problema si scompone nel classico modello a
*canale rumoroso*:

$$
\hat{W} = \arg\max_{W}\ \underbrace{P(X \mid W)}_{\text{modello acustico}}\
\underbrace{P(W)}_{\text{modello di linguaggio}}.
$$

(Il denominatore $P(X)$ di Bayes è sparito legittimamente: non dipende da $W$,
quindi non altera l'argmax.) Il **modello acustico** $P(X \mid W)$ misura
quanto i suoni osservati siano compatibili con una data sequenza di parole;
storicamente era un *Hidden Markov Model* con emissioni modellate da misture
di gaussiane (GMM). Il **modello di
linguaggio** $P(W)$ assegna una probabilità a priori alle frasi ($n$-grammi,
oggi reti neurali) ed è quello che disambigua gli omofoni. Dal 2012 le reti
neurali profonde sostituiscono le GMM nel modello acustico
{cite}`hinton2012deep`, tagliando in modo netto il tasso di errore.

`````

I due pareri hanno un nome, e conviene impararlo qui perché torna in tutto il
capitolo: il primo, quello che giudica il suono, si chiama **modello
acustico**; il secondo, quello che giudica se la frase è italiano plausibile,
si chiama **modello di linguaggio**. Li ritroveremo anche nell'ultima sezione,
dove il viaggio si fa al contrario e il modello acustico, invece di ascoltare
suoni, decide quali produrre.

L'ultimo salto è l'approccio **end-to-end** ("da un capo all'altro"): una sola
rete neurale che impara direttamente il passaggio dall'audio al testo, senza
scomporre a mano acustica e linguaggio. Tecniche come la **CTC**
(*Connectionist Temporal Classification* {cite}`graves2006connectionist`) e i
modelli encoder-decoder con attenzione (una parte della rete ascolta tutto,
l'altra scrive) hanno reso possibile addestrare l'intero sistema da coppie
(audio, testo): le vedremo una per una nella prossima sezione, qui bastano i
nomi. **Whisper** di OpenAI (2022), un Transformer encoder-decoder
allenato su circa 680.000 ore di audio multilingue, è l'esempio più noto:
trascrive e traduce decine di lingue, italiano compreso, con un unico modello.
Per farsi un'idea di quelle ore: sono settantasette anni di parlato
ininterrotto, giorno e notte, senza una pausa.

## La pipeline, passo per passo

Anche nei sistemi end-to-end, dove i confini si sfumano dentro un'unica rete, è
utile riconoscere le fasi concettuali che il segnale attraversa
({numref}`fig-asr-pipeline`). La prima si chiama **estrazione delle feature**,
che è l'inglese per «caratteristiche»: al posto dell'onda grezza, poche misure
per ogni frammento di suono.

```{figure} ../figures/asr-pipeline.svg
:name: fig-asr-pipeline
:alt: "Diagramma di flusso a cinque stadi: audio, estrazione delle feature, modello acustico, modello di linguaggio, testo."
:width: 90%

La pipeline dell'ASR: dall'onda sonora grezza si estraggono le feature, il
modello acustico le mappa in suoni, il modello di linguaggio ricostruisce le
parole plausibili, e in fondo esce il testo.
```

Quel primo passaggio merita un dettaglio: è qui che l'onda grezza diventa
qualcosa di trattabile.

`````{tab} Elementare

Il segnale grezzo è troppo minuto e dettagliato per lavorarci direttamente.
Allora lo si taglia in fettine di pochi centesimi di secondo e, per ognuna, si
misura "quanta energia c'è a ciascuna altezza sonora": un po' come le barre di
un equalizzatore che salgono e scendono a ritmo di musica. Questa sequenza di
istantanee sonore ha un nome, e lo useremo per tutto il capitolo: si chiama
**spettrogramma**. È molto più facile da dare in pasto a un modello del segnale
originale.

`````

`````{tab} Superiore

Il segnale continuo viene diviso in **frame** sovrapposti (finestre di circa
25 ms, una ogni 10 ms). Su ogni frame si calcola lo spettro con la trasformata
di Fourier a tempo breve e lo si filtra su scala **Mel** (che imita la
sensibilità non lineare dell'orecchio umano); il logaritmo delle energie di
banda dà il *log-mel*, e una trasformata coseno discreta (DCT), che decorrela
i coefficienti, lo comprime nei **MFCC** (*Mel-Frequency Cepstral
Coefficients*): un vettore $x_t \in \mathbb{R}^{d}$ di una dozzina di
componenti per frame (una quarantina se si aggiungono le derivate prima e
seconda). I sistemi neurali end-to-end, Whisper compreso, si
fermano di solito al log-mel (la decorrelazione serviva alle covarianze
diagonali delle GMM); in entrambi i casi è la matrice $X$ così ottenuta a
entrare nel modello acustico.

`````

## Dove lo incontriamo

Il riconoscimento vocale è ormai ovunque: gli **assistenti vocali** (Siri,
Alexa, Google Assistant) che aspettano un comando; la **sottotitolazione
automatica** di YouTube e delle videochiamate, preziosa per l'accessibilità;
la **dettatura** che trasforma la voce in documenti, dai referti medici ai
messaggi scritti mentre si guida. È la porta d'ingresso di quasi ogni sistema
che "capisce" il parlato: una volta che la voce è diventata testo, tocca agli
strumenti del linguaggio naturale interpretarne il significato. Ma quel primo,
difficile passo (dall'onda alla parola) comincia sempre qui.

Ed è un passo che sbaglia ancora, ogni giorno, sotto i nostri occhi: i
sottotitoli automatici che scrivono una parola per un'altra, o che riempiono
di frasi inventate un pezzo di video in cui nessuno parla, sono la parte
visibile di limiti precisi, che le prossime sezioni raccontano uno per uno.

E il viaggio, in questo capitolo, ha anche un ritorno: l'ultima sezione
percorre la strada opposta (dal testo all'onda sonora, la **sintesi vocale**)
chiudendo il cerchio tra ascoltare e parlare.
