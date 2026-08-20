# Speech Recognition: dalla voce al testo

Nel 1952, nei laboratori della Bell Telephone, un apparecchio ingombrante
fatto di valvole e relè (l'elettronica di allora: bulbi di vetro
incandescenti e interruttori che scattano da soli) imparò a riconoscere le
cifre da zero a nove
pronunciate ad alta voce. Lo chiamarono poi **Audrey**, e aveva un limite
quasi comico: funzionava con un solo parlante, che poteva parlare a velocità
normale ma doveva staccare nettamente una cifra dall'altra, con una pausa in
mezzo. Attaccate, non le riconosceva più. Settant'anni dopo
diciamo "metti la sveglia alle sette" al telefono mentre carichiamo la
lavastoviglie, e la frase diventa testo (e poi azione) in una frazione di
secondo. In mezzo c'è la storia dell’**ASR** (*Automatic Speech Recognition*),
il riconoscimento vocale automatico.

## Che cosa fa, in fondo

Il compito è facile da enunciare: prendere un segnale audio e restituire le
parole che contiene.

`````{tab} Elementare

Ascolti una registrazione e la trascrivi a mano: ricevi un flusso continuo di
suono (alti, bassi, pause) e produci una riga di parole scritte. Il
riconoscitore vocale fa esattamente questo: in ingresso l'onda sonora
catturata dal microfono, in uscita del testo. Attenzione: nessuna
*comprensione* del significato, per ora. Solo il passaggio dal suono
alle lettere giuste.

`````

`````{tab} Superiore

Formalmente l'ASR è un problema di **trasduzione di sequenze**: da una sequenza
di vettori acustici $\mathbf{X} = (\mathbf{x}_1, \dots, \mathbf{x}_T)$
vogliamo la sequenza di parole $\hat{W}$ più probabile,

$$
\hat{W} = \arg\max_{W} P(W \mid \mathbf{X}).
$$

Qui $\mathbf{X}$ è tipicamente lunga centinaia o migliaia di passi (uno ogni
dieci millisecondi circa), mentre $W$ conta poche decine di parole: input e
output hanno lunghezze molto diverse e non allineate una-a-una. Gestire questo
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

I primi sistemi, fino agli anni Settanta, lavoravano per somiglianza:
tenevano in memoria una registrazione di ogni parola del loro piccolo
vocabolario e cercavano quella più simile al suono appena arrivato,
allungandolo o comprimendolo nel tempo per farlo combaciare. L'operazione ha
un nome che ogni tanto si incontra ancora, *dynamic time warping*, «deformare
il tempo». Andava bene per pochissime parole e un solo parlante: aggiungerne
una voleva dire registrarla.

Il salto di qualità, negli anni Ottanta, non venne da macchine più potenti.
Venne da due idee. La prima: smettere di ragionare per parole intere e
ragionare per suoni, perché le parole di una lingua sono centinaia di
migliaia mentre i suoni sono qualche decina, e una parola mai sentita prima è
comunque fatta di suoni già sentiti. La seconda: dividere il lavoro in due
pareri.

`````{tab} Elementare

Un pezzo del sistema giudica *quanto questo suono somiglia alla parola
«cane»*; un altro pezzo, che
ha letto montagne di testo italiano, giudica *quanto è plausibile la frase «il
cane abbaia» rispetto a «il cane abbaglia»*. La trascrizione finale è il
miglior compromesso fra i due giudizi. Ecco perché il contesto risolve gli
omofoni: da solo il suono non basta, serve sapere che frasi hanno senso.

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
da $W$, quindi non altera l'argmax.) Il **modello acustico**
$P(\mathbf{X} \mid W)$ misura
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

Su questa divisione del lavoro il riconoscimento vocale si è retto per
trent'anni, dagli anni Ottanta al 2010 circa, e chi legge qualsiasi cosa
scritta in quel periodo trova sempre la stessa sigla poco amichevole,
**HMM-GMM**. Sono due macchine già incontrate, e qui lavorano insieme.

La prima descrive il parlato come una fila di suoni che si susseguono e che
nessuno vede direttamente: è una recita dietro la tenda, dove si sentono le
battute ma non si vede chi le dice, e dagli anni Ottanta è il modo standard di
raccontare una cosa che si svolge nel tempo e che si può solo intuire da fuori.
Si chiama «modello di Markov nascosto», HMM, e il libro l'ha già raccontata nel
capitolo sul linguaggio naturale, dove gli attori dietro la tenda erano le
categorie grammaticali invece dei suoni.

La seconda dice che uno stesso suono non è mai due volte identico. Dopo
l'estrazione delle feature ogni frammento di suono è ridotto a una manciata di
numeri, cioè, se ti aiuta immaginarlo, a un puntino su una mappa; e la «a» di
mille persone diverse non cade tutte le volte sullo stesso puntino, cade in una
nuvola di puntini vicini. Descrivere quella nuvola invece del suo centro è
esattamente quello che fanno le misture di gaussiane (GMM) incontrate nel
{doc}`capitolo di Machine Learning </MachineLearning/overview>`, dove servivano a trovare gruppi nei dati:
«gaussiana» è il nome della forma di nuvola più comune in natura, quella fitta
al centro e sempre più rada man mano che ci si allontana.

L'ultimo salto è l'approccio **end-to-end** ("da un capo all'altro"): una sola
rete neurale che impara direttamente il passaggio dall'audio al testo, senza
scomporre a mano acustica e linguaggio. Tecniche come la **CTC**
(*Connectionist Temporal Classification* {cite}`graves2006connectionist`) e i
modelli con **attenzione** (una parte della rete ascolta tutto, l'altra
scrive, e mentre scrive torna a guardare il punto dell'audio che le serve:
quel tornare a guardare è l'attenzione) hanno reso possibile addestrare
l'intero sistema da coppie (audio, testo). Le vedremo una per una nella
prossima sezione: qui bastano i nomi.

L'esempio più noto è **Whisper** di OpenAI (2022): trascrive e traduce
decine di lingue, italiano compreso, con un unico modello, allenato su circa
680.000 ore di audio multilingue. Per farsi un'idea di quelle ore: sono
settantasette anni di parlato ininterrotto, giorno e notte, senza una pausa.
La rete che ci sta dentro è un **Transformer**, l'architettura a cui il libro
ha già dedicato un capitolo suo, montata anche lei come encoder e decoder.

## La catena di montaggio, passo per passo

Dal microfono al testo il suono passa per alcune tappe, sempre le stesse
({numref}`fig-asr-pipeline`). Conviene conoscerle anche adesso che sono finite
tutte dentro un'unica rete e non si vedono più dall'esterno, perché i nomi
sono rimasti quelli e li useremo per tutto il capitolo. La catena nel suo
insieme si chiama **pipeline**, che in inglese è la conduttura, e qui vale
quello che da noi si chiamerebbe catena di montaggio.

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
il resto, e merita un dettaglio.

`````{tab} Elementare

Il segnale grezzo è troppo minuto e dettagliato per lavorarci direttamente.
Allora lo si taglia in fettine di pochi centesimi di secondo e, per ognuna, si
misura "quanta energia c'è a ciascuna altezza sonora": un po’ come le barrette
colorate che ballano nelle app della musica, dove le più a sinistra si alzano
sui suoni gravi e le più a destra sugli acuti. Questa sequenza di
istantanee sonore ha un nome, e lo useremo per tutto il capitolo: si chiama
**spettrogramma**. Per un modello lavorare su queste istantanee è molto più
facile che lavorare sull'onda di partenza.

`````

`````{tab} Superiore

Il segnale continuo viene diviso in **frame** sovrapposti e trasformato nello
spettrogramma **log-mel** costruito in
{doc}`Dal suono alle feature </Audio/dal-suono-alle-feature>`: è la matrice
$\mathbf{X}$ che entra nel modello acustico. I sistemi classici applicavano un
ultimo passaggio, la trasformata coseno discreta (DCT), che decorrelava i
coefficienti e riduceva ogni frame a una dozzina di **MFCC** (*Mel-Frequency
Cepstral Coefficients*, una quarantina con le derivate prima e seconda):
serviva alle covarianze diagonali delle GMM, e i sistemi neurali end-to-end,
Whisper compreso, non lo fanno più.

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
