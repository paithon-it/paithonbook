# Il suono come token: i codec neurali

Una promessa era rimasta in sospeso: se si riuscisse a trasformare un suono in
una sequenza di simboli discreti (un «alfabeto sonoro»
finito) potremmo generarne di nuovo esattamente come un modello di linguaggio
genera testo, un simbolo alla volta. Ma il testo quell'alfabeto ce l'ha già,
regalato dalla lingua: ventuno lettere e via. L'audio no. È un'onda continua,
e un alfabeto per il suono in natura non esiste: va **costruito**. Questa
sezione racconta chi lo costruisce (i **codec neurali**) e come. È la chiave
di volta della generazione audio moderna: senza un buon alfabeto, non c'è
nulla su cui scrivere.

## Comprimere imparando

La parola *codec* non è nuova. Ogni volta che ascoltate un brano in streaming
o salvate un vocale, un codec ha ridotto l'audio a una frazione della sua
dimensione. Il più famoso, l’**MP3** (il cui progetto fu completato nel 1992 e
pubblicato come standard ISO l'anno dopo), comprime
buttando via ciò che l'orecchio non sente: si appoggia a un modello
psicoacustico (un insieme di **regole fisse**, scritte a mano da ingegneri)
che decide quali frequenze sono coperte da altre e quindi eliminabili. È un
ottimo mestiere artigianale, ma è *congelato*: quelle regole non cambiano, non
imparano, non si adattano ai dati.

Un codec **neurale** ribalta l'approccio. Invece di scrivere le regole, le fa
**imparare** a una rete. La struttura ha un nome, **autoencoder**, e una forma
da guardare: un **encoder** che stringe quello che entra fino a farlo
diventare un pugno di numeri, e un **decoder** che da quel pugno di numeri
cerca di ritirare fuori l'originale. I due si addestrano *insieme*, con
un'unica regola: quello che esce deve somigliare a quello che è entrato.
Questa forma non è del suono, è di qualunque cosa si voglia comprimere, ed è
qui che il libro la monta: il {doc}`capitolo sui modelli latenti </ModelliLatenti/overview>` la riprenderà per
le immagini, e le aggiungerà l'unica cosa che le manca per servire anche a
*generare*.

```{figure} ../figures/autoencoder-comprimere-per-capire.svg
:name: fig-autoencoder-clessidra
:alt: "Schema a clessidra: l'ingresso attraversa l'encoder, che lo restringe progressivamente fino a uno spazio latente molto più piccolo; da lì il decoder lo riespande fino a ricostruire un'uscita della stessa forma dell'ingresso. La strozzatura centrale è il punto più stretto della figura."
:width: 88%

La strozzatura è il compito. Se l'uscita deve somigliare all'ingresso ma in
mezzo c'è un collo molto più stretto, la rete è costretta a tenere solo ciò
che serve a ricostruire.
```

La forma di {numref}`fig-autoencoder-clessidra` è quella di ogni compressione
imparata, e il resto della sezione non fa che stringere e disciplinare quel
collo centrale. Nel disegno le parole sono in inglese, come si trovano nel
codice: l’*input* è ciò che entra, l’*encoder* la parte che stringe, il
*bottleneck* la strozzatura, il *decoder* la parte che riapre, l’*output* ciò
che esce. Il pugno di numeri che sopravvive nella strozzatura si chiama
**latente**, ed è una parola che da qui in poi torna in ogni pagina: latente
perché quei numeri non li ha scelti nessuno e non dicono niente a guardarli, ma
dentro c'è tutto ciò che serve per rifare il suono. E siccome un pugno di numeri
si può sempre immaginare come un punto, l'insieme di tutti i latenti possibili
prende il nome di **spazio latente**: il magazzino dove la rete tiene i suoi
riassunti. È un nome che nel libro tornerà ogni volta che un modello preferisce
lavorare sulla versione compressa dei dati invece che sui dati. Le lettere sono
le abbreviazioni consuete ($\mathbf{x}$ l'ingresso, $\hat{\mathbf{x}}$ la sua
ricostruzione, $\mathbf{z}$ il latente, *loss* la distanza fra i primi due, cioè
quanto la rete ha sbagliato).

I numeri $784 \to 128 \to 32$ sono solo un esempio, e vengono dalle immagini
perché è lì che questo schema si vede meglio: una cifra scritta a mano di
$28 \times 28$ pixel, cioè 784 numeri, ridotta a 32 e poi rifatta. Con l'audio è
uguale, con questi ordini di grandezza: entra un secondo di suono misurato
24.000 volte, cioè 24.000 numeri, ed escono dalla strozzatura 75 latenti, uno
ogni 320 misure. Sono i valori di EnCodec, il codec che accompagnerà tutta la
sezione, e li ritroveremo.

`````{tab} Elementare

Una valigia si può rimpicciolire in due modi. Il primo è una lista di regole
stampata sul coperchio: «togli sempre il beauty-case, arrotola le magliette,
lascia a casa il terzo paio di scarpe». Vale per tutti, non cambia
mai: è l'MP3. Il secondo modo è imparare *facendo*, viaggio dopo viaggio: provi
a chiudere la valigia, vedi cosa si è sgualcito all'arrivo, e la prossima volta
sistemi meglio proprio quelle cose. Dopo mille viaggi hai un tuo metodo, cucito
sul tuo bagaglio, che nessuno ti ha dettato. Il codec neurale è il secondo
viaggiatore: nessuno gli dice *cosa* buttare, lo scopre da solo cercando di far
tornare a casa la valigia il più intatta possibile.

La vera sorpresa, però, non è la compressione in sé: l'MP3 già comprime bene.
È che quel riassunto compatto, imparato dalla rete, possiamo poi
**arrotondarlo** a un piccolo insieme di valori-tipo. E un valore-tipo è un
simbolo: un numero intero. È il ponte che stavamo cercando, dall'onda continua
all'alfabeto.

`````

`````{tab} Superiore

Un codec neurale è un autoencoder addestrato per la ricostruzione. L'encoder
$E$ mappa la forma d'onda $\mathbf{x}$ in una sequenza di vettori latenti
$\mathbf{Z} = E(\mathbf{x})$ a **frequenza di frame** molto più bassa del tasso di campionamento
(un vettore ogni poche centinaia di campioni); il decoder $D$ ricostruisce
$\hat{\mathbf{x}} = D(\mathbf{Z})$. L'obiettivo minimizza una perdita di ricostruzione, spesso
combinando errore nel dominio del tempo e nello spettro (multi-scala
tempo–frequenza), ed è tipicamente affiancato da un **discriminatore** in stile
GAN (il capitolo dedicato, più avanti, ne racconta il meccanismo per intero)
che spinge $\hat{\mathbf{x}}$ a suonare realistico, non solo a minimizzare
l'errore medio.

Fin qui è compressione con rappresentazione **continua**: ogni $\mathbf{z}$ è un vettore di
numeri reali. La novità che ci interessa è renderla **discreta**: sostituire
ogni vettore latente con un simbolo preso da un insieme finito. È il passaggio
che trasforma un compressore in un *tokenizzatore* del suono, e apre la porta
ai modelli di linguaggio sull'audio. Il come è il tema delle due parti che
seguono.

`````

## Vector quantization: dal continuo al discreto

Il latente che esce dall'encoder è ancora fatto di numeri che possono valere
qualunque cosa, e a noi serve un elenco finito di simboli, come le lettere:
serve, dicono i matematici, passare dal **continuo** al **discreto**. Lo
strumento che fa quel passaggio si chiama, all'inglese, **vector
quantization** (VQ). L'hanno portato nelle reti neurali van den Oord, Vinyals
e Kavukcuoglu nel 2017, con il **VQ-VAE** {cite}`oord2017neural`. L'idea è
sorprendentemente semplice, e conviene vederla prima con un'immagine e poi con
i numeri.

`````{tab} Elementare

Sedici colori bastano per una fotografia che ne aveva milioni. Prendi una
tavolozza fissa di sedici e, per ogni pixel della foto, scegli il colore della
tavolozza che gli somiglia di più e lo sostituisci: la foto diventa un po’ più
«a blocchi», ma la riconosci ancora. E adesso il colpo di genio: invece di
salvare per ogni pixel i suoi tre numeri di colore, salvi **un solo numero**
(la *posizione* nella tavolozza, da 0 a 15). La tavolozza la conosciamo già,
ci basta l'indice.

La *vector quantization* fa esattamente questo, ma invece dei colori dei pixel
tratta i **pezzetti di suono** così come escono dall'encoder: ognuno è un
gruppetto di numeri, come un colore è un gruppetto di tre numeri. La
«tavolozza» si chiama **codebook**: un elenco di pezzetti-tipo, i
*prototipi*. Ogni pezzetto di audio, dopo l'encoder, viene avvicinato al
prototipo più simile, e di lui si tiene solo il numero di posizione nell'elenco.
Quel numero è il **token**: il nostro simbolo dell'alfabeto sonoro. E l'operazione
che abbiamo appena fatto, sostituire una cosa qualsiasi con la più vicina di un
elenco prestabilito, si chiama **quantizzare**: è la parola che tornerà per
tutta la sezione, e vuol dire arrotondare, né più né meno.

`````

`````{tab} Superiore

Sia $\mathcal{C} = \{\mathbf{e}_1, \dots, \mathbf{e}_K\}$ un **codebook** di $K$
vettori-prototipo, appresi
durante l'addestramento. Dato un vettore latente $\mathbf{z}$ prodotto dall'encoder, la
quantizzazione sceglie il prototipo più vicino (in norma euclidea) e ne
restituisce l’**indice**:

$$
k^\star = \arg\min_{k \in \{1,\dots,K\}} \lVert \mathbf{z} - \mathbf{e}_k \rVert^2,
\qquad q(\mathbf{z}) = \mathbf{e}_{k^\star},
$$

dove $q(\mathbf{z})$ è il vettore quantizzato e $k^\star$ è il token: un intero in
$\{1, \dots, K\}$. L'audio non è più una sequenza di vettori reali ma una
sequenza di interi, esattamente come un testo tokenizzato.

Un dettaglio importante: l'operazione $\arg\min$ non è differenziabile, quindi
il gradiente non attraverserebbe la quantizzazione. Il VQ-VAE
{cite}`oord2017neural` lo aggira con lo **straight-through estimator** (il
gradiente del decoder viene copiato tal quale sull'uscita dell'encoder, come se
$q$ fosse l'identità) e con due termini quadratici: il *codebook loss*
$\lVert \mathrm{sg}[\mathbf{z}] - \mathbf{e}_{k^\star} \rVert^2$, che tira il
prototipo scelto verso i latenti che l'hanno scelto, ed è l'unica cosa che fa
imparare il codebook visto che l'$\arg\min$ non lascia passare gradiente, e la
*commitment loss*
$\beta \lVert \mathbf{z} - \mathrm{sg}[\mathbf{e}_{k^\star}] \rVert^2$, che
tira i latenti verso i prototipi ($\mathrm{sg}$ è lo *stop-gradient*, e il
verso della freccia sta tutto in quale dei due membri lo porta). Molte
implementazioni sostituiscono il primo con una media mobile esponenziale, che
è la stessa idea scritta in modo più stabile: è la regola alla k-means di cui
si dice più avanti. Resta il compromesso di fondo: un codebook
grande ($K$ alto) ricostruisce meglio ma costa più bit per token; uno piccolo
comprime di più ma perde fedeltà.

`````

Conviene fare i conti a mano su un esempio minuscolo, perché il meccanismo è
tutto qui. Prendiamo un codebook di appena **quattro** prototipi e, per
poterli scrivere su una riga, immaginiamo che ogni pezzetto di suono sia
descritto da due soli numeri invece che da centinaia. Una avvertenza prima di
guardarli: dentro le parentesi tonde troverai virgole di due tipi, quelle che
separano le due caselle e quelle dei decimali. Ogni parentesi contiene sempre
**due** numeri, mai quattro.

$$
\mathbf{e}_1 = (0,0),\quad \mathbf{e}_2 = (1,0),\quad
\mathbf{e}_3 = (0,1),\quad \mathbf{e}_4 = (1,1).
$$

Vogliamo quantizzare il latente $\mathbf{z} = (0{,}8,\ 0{,}1)$, cioè il pezzetto
che ha $0{,}8$ nella prima casella e $0{,}1$ nella seconda.

Calcoliamo, per ciascun prototipo, la distanza quadratica. Le due
sbarrette con il quadratino, $\lVert\,\cdot\,\rVert^2$, non chiedono niente di
più di questo: per ognuna delle due caselle fai la differenza, elevala al
quadrato e somma. Per $\mathbf{e}_1 = (0,0)$ viene
$(0{,}8-0)^2 + (0{,}1-0)^2 = 0{,}64 + 0{,}01 = 0{,}65$; per
$\mathbf{e}_2 = (1,0)$ viene invece
$(0{,}8-1)^2 + (0{,}1-0)^2 = 0{,}04 + 0{,}01 = 0{,}05$, molto meno. Gli altri due
si fanno allo stesso modo, con carta e penna:

$$
\lVert \mathbf{z} - \mathbf{e}_1\rVert^2 = 0{,}65,\quad
\lVert \mathbf{z} - \mathbf{e}_2\rVert^2 = 0{,}05,\quad
\lVert \mathbf{z} - \mathbf{e}_3\rVert^2 = 1{,}45,\quad
\lVert \mathbf{z} - \mathbf{e}_4\rVert^2 = 0{,}85.
$$

Il più vicino è $\mathbf{e}_2$: il token è **2**, e il pezzetto arrotondato è
$(1,0)$. Facciamo lo stesso con $\mathbf{u} = (0{,}2,\ 0{,}9)$: le distanze sono
$0{,}85$, $1{,}45$, $0{,}05$, $0{,}65$, il più vicino è $\mathbf{e}_3$, token
**3**. Abbiamo sostituito due pezzetti fatti di numeri qualsiasi con due soli
numeri interi, `2` e `3`.

Guadagnare, si guadagna, anche se qui non si vede a occhio: un numero
«qualsiasi» un computer lo scrive con una trentina di risposte sì/no, mentre per
dire «il secondo di quattro» ne bastano due. Il pezzetto costava una sessantina
di quelle risposte e adesso ne costa due. Questo è tutto ciò che serve per
scrivere l'audio in un alfabeto.

## Residual vector quantization: strati di precisione

C'è un problema, e lo si vede proprio nell'esempio. Sostituire
$(0{,}8,\ 0{,}1)$ con $(1,0)$ è comodo ma **grossolano**: ci siamo persi lo
scarto, cioè $0{,}2$ nella prima casella e $0{,}1$ nella seconda. Per l'audio,
uno scarto del genere è la differenza tra una voce naturale e una voce
metallica da citofono. La soluzione ovvia sarebbe allargare il codebook,
mettendo più prototipi per avvicinarci di più. Ma allargare costa, e conviene
guardare da vicino *quanto*, perché è tutta la ragione di quello che viene
dopo.

Serve prima la parola con cui si misura il costo. Un **bit** è una risposta
sì/no. Con 3 bit, cioè tre risposte sì/no in fila, si distinguono
$2 \times 2 \times 2 = 8$ casi; con 10 bit se ne distinguono 1024. Per dire a
quale prototipo si riferisce, un token deve spendere tanti bit quanti bastano a
distinguere le voci dell'elenco: quindi **raddoppiare** l'elenco costa una
risposta in più, non il doppio.

Sembra poco, ed è il punto. Perché quel bit in più lo paga *ogni* token, per
sempre, e il guadagno che porta è minuscolo: per dimezzare l'errore non basta
raddoppiare i prototipi, ne servirebbero tantissimi di più. Si spende in
proporzione al numero di volte che si raddoppia e si guadagna molto meno. Non
regge.

Ultima parola sulle unità: quanti bit al secondo servano in tutto a un codec si
chiama **bitrate**, e si misura in kbps, migliaia di bit al secondo. Per farsi
un'idea degli ordini di grandezza: un CD non compresso viaggia sui 1.400 kbps,
un MP3 di buona qualità sui 128, e i codec neurali di questa sezione scendono
sotto i 10. E attenzione al verso, perché è il contrario di quasi tutti gli
altri numeri del libro: qui **più è basso, meglio è**, perché vuol dire meno
roba da trasmettere a parità di suono.

La soluzione, elegante, è la **residual vector quantization** (RVQ), introdotta
per i codec neurali da **SoundStream** {cite}`zeghidour2021soundstream`, di
Google, e poi da **EnCodec** {cite}`defossez2023high`, di Meta: invece di un solo
codebook enorme, si mettono in **cascata** più codebook piccoli, ciascuno che
corregge l'errore lasciato dal precedente.

`````{tab} Elementare

Hai presente quando devi dare un resto di 87 centesimi con le monete? Non
cerchi una moneta magica da 87: prendi prima la più grossa che ci sta (50), ti
restano 37; poi la più grossa che ci sta nei 37 (20), restano 17; poi 10,
restano 7; poi 5, poi 2. Ogni moneta si occupa di ciò che è avanzato dalla
precedente, e passo dopo passo ti avvicini alla cifra esatta.

La RVQ fa la stessa cosa con i vettori del suono. Il primo codebook dà
l'approssimazione grossolana: la moneta da 50. Poi calcola quanto ha sbagliato
(il **residuo**, il resto da coprire) e chiede a un secondo codebook di
approssimare *quel residuo*. Il secondo lascia a sua volta un residuo più
piccolo, che un terzo codebook rifinisce ancora, e così via. Alla fine ogni
pezzetto di audio non è più un solo token, ma una **pila** di token (uno per
codebook) che insieme lo descrivono con la precisione che serve, spendendo
pochissimi bit.

`````

`````{tab} Superiore

La RVQ applica $N$ quantizzatori in cascata sul **residuo**. Posto
$\mathbf{r}_0 = \mathbf{z}$, al
livello $i$ si quantizza il residuo corrente con il codebook $\mathcal{C}^{(i)}$ e si
aggiorna il residuo:

$$
k_i^\star = \arg\min_{k} \big\lVert \mathbf{r}_{i-1} - \mathbf{e}_k^{(i)} \big\rVert^2,
\qquad
\mathbf{r}_i = \mathbf{r}_{i-1} - \mathbf{e}_{k_i^\star}^{(i)}.
$$

La ricostruzione finale è la somma dei prototipi scelti,
$q(\mathbf{z}) = \sum_{i=1}^{N} \mathbf{e}_{k_i^\star}^{(i)}$, e il token di quel frame diventa la
tupla di indici $(k_1^\star, \dots, k_N^\star)$: **$N$ flussi paralleli** di
interi. Ogni stadio quantizza ciò che è avanzato, ma questo da solo non basta
a garantire un miglioramento: l'errore non può crescere con $N$ se ogni
codebook contiene il vettore nullo, perché scegliere lo zero equivale a non
correggere (è il motivo per cui, nell'esempio in NumPy più avanti, il secondo
codebook lo include). In pratica, con codebook appresi sui dati, l'errore
decresce a ogni stadio.

Il conto del **bitrate** è pulito. Con $N$ quantizzatori, codebook di $K$ voci
ciascuno e frequenza di frame $f_r$:

$$
\text{bitrate} = N \cdot \log_2 K \cdot f_r,
$$

dove $\log_2 K$ sono i bit per indice. EnCodec a $24$ kHz usa codebook di
$K = 1024$ voci ($10$ bit) a $f_r = 75$ frame al secondo: con $N = 8$
quantizzatori si ottengono $8 \cdot 10 \cdot 75 = 6000$ bit/s, cioè **6 kbps**.
Variando $N$ si sceglie il compromesso: da $1{,}5$ kbps ($N=2$) fino a $24$ kbps
($N=32$).

Due cautele sui numeri, perché è facile ricordarseli storti. La prima riguarda
il paragone con l'MP3, che si legge dappertutto: la parità a 64 kbps è del
**gemello a 48 kHz stereo**, non di questo modello a 24 kHz monofonico, i cui
termini di confronto nel paper sono Opus, EVS e Lyra-v2. E quel gemello arriva
ai 6 kbps per un'altra strada: a 48 kHz l'encoder produce 150 passi latenti al
secondo invece di 75, quindi sono $4 \cdot 10 \cdot 150$, non gli
$8 \cdot 10 \cdot 75$ appena calcolati. Nelle prove d'ascolto
MUSHRA prende $82{,}9$ a 6 kbps contro $82{,}7$ di un MP3 a 64:
qualità percepita indistinguibile con un decimo dei bit, però con il riferimento
non compresso a $95{,}1$, quindi *entrambi* si distinguono dall'originale.

La seconda: il bitrate non è una manopola monotona. Nella stessa tabella EnCodec
a 12 kbps prende $88{,}0$ e a 24 kbps $87{,}5$, cioè sono indistinguibili entro
l'incertezza dichiarata; e a parità di bit due codec diversi danno risultati
lontanissimi (a 6 kbps, $82{,}9$ contro $17{,}7$ di Opus). Raddoppiare $N$
aspettandosi un guadagno proporzionale è il modo più comune di sprecare bit: il
bitrate dice quanto costa, non quanto suona bene.

`````

Il risultato è una tabella con due direzioni. Lungo il **tempo**, il suono viene
tagliato a fettine, e ogni fettina si chiama **frame**: dura una manciata di
millesimi di secondo. Lungo la **profondità**, ogni frame porta non un token ma
la pila di token della cascata, uno per codebook.
{numref}`fig-audio-codec-rvq` mostra la seconda direzione, che è quella nuova:
un frame solo, la pila di token che ne esce, l'encoder che li produce e il
decoder che li rilegge.

Il conto, con i valori che usa EnCodec sull'audio misurato 24.000 volte al
secondo, viene così: 75 frame in ogni secondo di audio, 8 token per ogni frame,
cioè **600 simboli al secondo** al posto di 24.000 misure. È il salto che rende
possibile tutto il resto del capitolo.

Da quei 600 numeri il decoder tira fuori un suono che *suona* come l'originale,
e la parola «ricostruire» va presa con le pinze. Quel decoder non impara solo a
sbagliare poco. Accanto a lui, durante l'addestramento, lavora un
**discriminatore**: una seconda rete il cui unico mestiere è smascherare l'audio
finto, e che quindi lo costringe a produrre qualcosa che *suoni* vero, non
soltanto qualcosa di numericamente vicino all'originale (è il meccanismo delle
GAN, a cui il libro dedica un capitolo più avanti).

Sotto quella pressione il decoder diventa a tutti gli effetti un piccolo
generatore, guidato dai token che riceve. A bitrate bassi il dettaglio più fine
(la grana, le code di riverbero, le frequenze più alte) non viene recuperato:
viene **reinventato** in modo credibile. Ecco perché la fedeltà misurata
campione per campione crolla mentre la qualità che si sente regge. Ed ecco anche
perché, nella prossima sezione, questo stesso decoder potrà fare da generatore
senza cambiare una riga: a quel punto la differenza fra un codec e un modello
che inventa suono sta soltanto in da dove arrivano i token. Ed è proprio quei
token che daremo in pasto a un modello di linguaggio.

```{figure} ../figures/audio-codec-rvq.svg
:name: fig-audio-codec-rvq
:alt: "Pipeline di un codec neurale: l'onda audio entra in un encoder convoluzionale che la comprime nel latente z; z attraversa il blocco RVQ, tre codebook in cascata dove ognuno quantizza il residuo del precedente ed emette un token intero; i tre flussi di token alimentano un decoder che ricostruisce l'onda."
:width: 100%

Un codec audio neurale. L'encoder comprime l'onda in vettori latenti; la RVQ li
trasforma in una pila di token (uno per codebook, ciascuno sul residuo del
precedente); il decoder ripercorre la strada al contrario e ricostruisce
l'audio.
```

## Un RVQ in miniatura

Tolte le reti neurali, la RVQ è quattro operazioni in croce e si scrive in poche
righe: un elenco di prototipi, la ricerca del più vicino, il calcolo di quello
che è avanzato, e un secondo elenco che rifinisce l'avanzo. Il codice qui sotto
lo fa su sei pezzetti finti da due numeri ciascuno, presi a caso, e misura
quanto si sbaglia usando un solo elenco e poi due.

Due avvertenze prima di leggerlo, per non inciampare sui numeri. Qui i prototipi
sono numerati a partire da **zero**, come conta Python, mentre nella formula
partivano da uno: è solo un modo di contare. E gli elenchi qui sotto non sono
quelli dell'esempio a mano di poco fa, quindi i token che ne escono non devono
coincidere con il `2` e il `3` di prima.

```python
import numpy as np

rng = np.random.default_rng(0)

# Sei pezzetti da due numeri ciascuno: quelli che uscirebbero dall'encoder
Z = rng.uniform(-1, 1, size=(6, 2)).round(2)

# Primo codebook: 4 prototipi grossolani (K = 4)
C1 = np.array([[-0.5, -0.5],
               [ 0.5, -0.5],
               [-0.5,  0.5],
               [ 0.5,  0.5]])

# Secondo codebook: 4 aggiustamenti fini per il residuo (lo zero e' incluso)
C2 = np.array([[ 0.00,  0.00],
               [ 0.30,  0.00],
               [ 0.00,  0.30],
               [-0.30, -0.30]])


def quantizza(V, C):
    """Per ogni riga di V trova il prototipo piu' vicino nell'elenco C."""
    # distanze quadratiche fra ogni pezzetto e ogni prototipo
    d = ((V[:, None, :] - C[None, :, :]) ** 2).sum(axis=2)
    idx = d.argmin(axis=1)      # posizione del prototipo piu' vicino: il "token"
    return idx, C[idx]          # le posizioni e i pezzetti arrotondati


def mse(A, B):
    """MSE, errore quadratico medio: di quanto sbaglia in media la ricostruzione."""
    return ((A - B) ** 2).mean()


# --- Stadio 1: arrotondo il pezzetto al prototipo piu' vicino ---
idx1, q1 = quantizza(Z, C1)
ric1 = q1                       # ricostruzione con 1 solo stadio

# --- Stadio 2: quantizzo il RESIDUO ---
residuo = Z - q1
idx2, q2 = quantizza(residuo, C2)
ric2 = q1 + q2                  # ricostruzione con 2 stadi

print("vettori da quantizzare:\n", Z)
print("token stadio 1:", idx1.tolist())
print("token stadio 2:", idx2.tolist())
print(f"MSE con 1 quantizzatore: {mse(Z, ric1):.4f}")
print(f"MSE con 2 quantizzatori: {mse(Z, ric2):.4f}")
```

L'output mostra i due «flussi» di token e, soprattutto, l'errore che cala
aggiungendo il secondo stadio:

```text
vettori da quantizzare:
 [[ 0.27 -0.46]
 [-0.92 -0.97]
 [ 0.63  0.83]
 [ 0.21  0.46]
 [ 0.09  0.87]
 [ 0.63 -0.99]]
token stadio 1: [1, 0, 3, 3, 3, 1]
token stadio 2: [0, 3, 2, 3, 2, 3]
MSE con 1 quantizzatore: 0.1021
MSE con 2 quantizzatori: 0.0481
```

Quel numero, l'MSE, è la media di quanto ogni pezzetto ricostruito si discosta
da quello vero, e più è piccolo meglio va. Aggiungendo il secondo elenco più che
si dimezza, da $0{,}1021$ a $0{,}0481$, e ogni pezzetto adesso è descritto da due
numeri interi invece che da due numeri qualsiasi. È l'intera idea della RVQ, in
scala di laboratorio: nei codec veri i pezzetti hanno centinaia di numeri, gli
elenchi migliaia di voci e gli stadi sono otto o più, ma la meccanica è
precisamente questa, ed è quella che il codice qui sopra esegue.

Gli elenchi qui sopra li abbiamo scritti noi; nei codec veri i prototipi si
**imparano** insieme all'encoder e al decoder. La regola con cui si imparano è semplice: ogni prototipo viene
spostato ogni tanto nel mezzo dei pezzetti che l'hanno scelto, così da
rappresentarli meglio (è lo stesso meccanismo del **k-means**, l'algoritmo di
raggruppamento del {doc}`capitolo sul machine learning </MachineLearning/overview>`).

E quella regola porta con sé il guasto caratteristico di tutta la famiglia. Una
voce che nessun pezzetto sceglie non viene mai spostata, quindi resta dov'è e
continua a non essere scelta: è morta, e non risuscita. L'elenco che si usa
davvero si riduce in silenzio a una frazione di quello dichiarato, mentre il
bitrate resta quello di prima, calcolato sull'elenco intero. Si pagano tutti i
bit e se ne usa una parte, e si chiama **codebook collapse**. Quanto morda
dipende da dove si parte: se i prototipi nascono sparsi molto più larghi dei
pezzetti che dovranno descrivere, ne sopravvivono pochissimi, perché tutti i
pezzetti finiscono addosso agli stessi due o tre. I rimedi sono di ingegneria e
stanno nei codec citati qui sopra: EnCodec {cite}`defossez2023high` sostituisce
le voci mai usate con pezzetti presi dal mucchietto che sta processando in quel
momento, dandogli così un posto dove sono utili (si chiama *restart*); altri
arrotondano in uno spazio più piccolo e riportano prototipi e pezzetti alla
stessa scala. Un codebook va sempre misurato per quante voci **usa davvero**,
non per quante ne dichiara.

Sulla misura della qualità serve poi una distinzione che il gergo tende a
cancellare, e conviene dirla in ordine. **Primo**: l'errore quadratico medio sui
campioni non è il criterio giusto nemmeno per addestrare, perché non ha
orecchio, e i codec veri usano invece perdite calcolate sullo spettro, più il
discriminatore di cui abbiamo parlato, che premiano ciò che *suona* bene.
**Secondo**, ed è il punto: quelli sono obiettivi di addestramento. Dicono al
modello dove andare, non dicono a noi dove è arrivato, e un discriminatore che
promuove il proprio generatore è metà di una partita, non un verdetto.

**Misurare** la qualità è un problema diverso, e ancora aperto. Esistono voti
che una macchina può dare da sola (PESQ, STOI, ViSQOL, FAD sono i nomi che si
incontrano), ma sono tutti approssimazioni, ognuna tarata su un tipo di difetto
e nessuna affidabile fuori dal suo. I primi due, per dire, nascono per il
parlato rovinato dal rumore, e si comportano male davanti a un decoder che il
segnale se lo reinventa. Per questo i lavori del settore continuano a chiudere
con prove d'ascolto fatte da persone, secondo un protocollo che si chiama
**MUSHRA**: a chi ascolta si fanno sentire, mescolati e senza dire quale è
quale, il suono da giudicare, l'originale intatto e una versione volutamente
rovinata, così che nessuno sappia mai cosa sta votando. Quando un lavoro riporta
un solo voto automatico, quel voto è un indizio, non la qualità.

Il principio della cascata sul residuo, però, è esattamente quello che hai
appena visto girare.

`````{tab} Elementare

```{admonition} Da ricordare
:class: important
- Un **codec neurale** non segue una lista di regole scritte da qualcuno, come
  fa l'MP3: impara a comprimere **provando**, come il viaggiatore che a ogni
  viaggio chiude meglio la valigia. Nessuno gli dice cosa buttare: glielo impone
  la strettoia in mezzo.
- Il passo che serve a noi è la **tavolozza**: si tiene un elenco di
  pezzetti-tipo e di ogni pezzetto di suono si salva solo il *numero di
  posizione* nell'elenco. Quel numero è il **token**, cioè la lettera
  dell'alfabeto sonoro.
- Una tavolozza sola è troppo grossolana, e per raffinarla servirebbero
  tantissimi colori. Meglio fare come con il **resto in monete**: una prima
  tavolozza dà l'approssimazione grossa, una seconda copre quel che è avanzato,
  una terza quel che avanza ancora. Ogni pezzetto di suono diventa così una
  **pila** di token invece di uno solo.
- Il risultato è che un secondo di musica si scrive con qualche centinaio di
  numeri invece che con decine di migliaia di misure. Quanti bit al secondo
  servono si chiama **bitrate**, e qui più è basso meglio è.
- Attenzione a due parole. Il decoder non «ricostruisce» l'originale: a bitrate
  bassi il dettaglio più fine se lo **reinventa** in modo credibile. E la
  qualità, alla fine, la decidono ancora delle **persone che ascoltano**: i
  numeri automatici sono indizi, non verdetti.
```

`````

`````{tab} Superiore

```{admonition} Da ricordare
:class: important
- Un **codec neurale** non applica regole fisse come l'MP3: è un autoencoder che
  **impara** a comprimere l'audio (encoder → latente → decoder), addestrato sulla
  ricostruzione.
- La **vector quantization** {cite}`oord2017neural` rende la rappresentazione
  *discreta*: un **codebook** di prototipi, ogni latente sostituito dal più
  vicino, e il suo **indice** diventa il **token** (l'alfabeto sonoro).
- Un solo codebook è troppo grossolano. La **residual vector quantization**
  {cite}`zeghidour2021soundstream` {cite}`defossez2023high` mette più codebook in
  **cascata**: ognuno quantizza il **residuo** del precedente, come dare il resto
  con monete via via più piccole.
- L'audio diventa così una **griglia di token** (tempo × profondità della
  cascata): con EnCodec a 24 kHz, 600 simboli per secondo a 6 kbps. Il bitrate è
  $N \cdot \log_2 K \cdot f_r$, ma **non** è una manopola monotona: nelle prove
  MUSHRA 12 e 24 kbps sono indistinguibili, e a parità di bit codec diversi
  distano decine di punti.
- Il decoder non ricostruisce, **risintetizza**: addestrato con un
  discriminatore è un generatore condizionato, e a bitrate bassi inventa il
  dettaglio fine in modo plausibile.
- Due trappole di misura. Il **codebook collapse**: una voce mai scelta non
  viene più aggiornata e muore, quindi il codebook effettivo si riduce mentre il
  bitrate nominale resta (rimedio: il *restart* delle voci morte). E perdite
  spettrali e discriminatori sono **obiettivi di addestramento**, non metriche:
  gli indicatori oggettivi (PESQ, STOI, ViSQOL, FAD) sono surrogati d'ambito, e
  il giudizio resta MUSHRA.
- Con due soli stadi, nell'esempio in NumPy, l'errore di ricostruzione più che si
  dimezza: è l'intera meccanica della RVQ in scala di laboratorio.
- Ottenuto l'alfabeto, l'audio *è* una sequenza di simboli: tutto
  l'armamentario dei Transformer diventa applicabile, ed è ciò che vedremo
  nella sezione sulla generazione.
```

`````
