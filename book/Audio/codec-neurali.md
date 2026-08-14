# Il suono come token: i codec neurali

Nell'overview del capitolo abbiamo lanciato una promessa: se riuscissimo a
trasformare un suono in una sequenza di simboli discreti (un «alfabeto sonoro»
finito) potremmo generarne di nuovo esattamente come un modello di linguaggio
genera testo, un simbolo alla volta. Ma il testo quell'alfabeto ce l'ha già,
regalato dalla lingua: ventuno lettere e via. L'audio no. È un'onda continua,
e un alfabeto per il suono in natura non esiste: va **costruito**. Questa
sezione racconta chi lo costruisce (i **codec neurali**) e come. È la chiave
di volta della generazione audio moderna: senza un buon alfabeto, non c'è
nulla su cui scrivere.

## Comprimere imparando

```{figure} ../figures/autoencoder-comprimere-per-capire.svg
:name: fig-autoencoder-clessidra
:alt: "Schema a clessidra: l'ingresso attraversa l'encoder, che lo restringe progressivamente fino a uno spazio latente molto più piccolo; da lì il decoder lo riespande fino a ricostruire un'uscita della stessa forma dell'ingresso. La strozzatura centrale è il punto più stretto della figura."
:width: 88%

La strozzatura è il compito. Se l'uscita deve somigliare all'ingresso ma in
mezzo c'è un collo molto più stretto, la rete è costretta a tenere solo ciò
che serve a ricostruire.
```

La forma di {numref}`fig-autoencoder-clessidra` è quella di ogni codice
imparato, e il resto della sezione non fa che stringere e disciplinare quel
collo centrale. Nel disegno le parole sono in inglese, come si trovano nel
codice: l'*input* è ciò che entra, l'*encoder* la parte che stringe, il
*bottleneck* la strozzatura, il *decoder* la parte che riapre, l'*output* ciò
che esce. Le lettere sono le abbreviazioni consuete ($\mathbf{x}$ l'ingresso,
$\hat{\mathbf{x}}$ la sua ricostruzione, $\mathbf{z}$ quel poco che sopravvive nella strozzatura,
*loss* la distanza fra i primi due, cioè quanto la rete ha sbagliato); i numeri
$784 \to 128 \to 32$ sono solo un esempio, quello di una cifra scritta a mano di
$28 \times 28$ pixel, cioè 784 numeri, ridotta a 32. Vale la pena notare che
nessuno dice alla rete *cosa* buttare: glielo impone la larghezza della
strozzatura, e il resto lo decide lei.

La parola *codec* non è nuova. Ogni volta che ascoltate un brano in streaming
o salvate un vocale, un codec ha ridotto l'audio a una frazione della sua
dimensione. Il più famoso, l'**MP3** (specifica chiusa nel 1992, pubblicata
come standard ISO l'anno dopo), comprime
buttando via ciò che l'orecchio non sente: si appoggia a un modello
psicoacustico (un insieme di **regole fisse**, scritte a mano da ingegneri)
che decide quali frequenze sono mascherate da altre e quindi eliminabili. È un
ottimo mestiere artigianale, ma è *congelato*: quelle regole non cambiano, non
imparano, non si adattano ai dati.

Un codec **neurale** ribalta l'approccio. Invece di scrivere le regole, le fa
**imparare** a una rete. La struttura è quella di un **autoencoder** (la stessa
idea di compressione appresa che ritroveremo, per le immagini, nella diffusione
latente): un **encoder** che comprime l'input in una rappresentazione
compatta, e un **decoder** che da quella rappresentazione cerca di ricostruire
l'originale. I due si addestrano *insieme*, con un'unica regola: la
ricostruzione deve somigliare all'ingresso.

`````{tab} Elementare

Immagina due modi di rimpicciolire una valigia. Il primo è avere una lista di
regole stampata sul coperchio: «togli sempre il beauty-case, arrotola le
magliette, lascia a casa il terzo paio di scarpe». Vale per tutti, non cambia
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

Lo strumento che rende discreta la rappresentazione ha un nome e una data:
**vector quantization** (VQ), portata nel deep learning dal **VQ-VAE** di van
den Oord, Vinyals e Kavukcuoglu nel 2017 {cite}`oord2017neural`. L'idea è
sorprendentemente semplice, e vale la pena vederla prima con un'immagine e poi
con i numeri.

`````{tab} Elementare

Pensa a una fotografia con milioni di sfumature di colore e a una tavolozza
fissa di, diciamo, 16 colori. Per ogni pixel della foto scegli il colore della
tavolozza che gli somiglia di più e lo sostituisci: la foto diventa un po' più
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
Quel numero è il **token**: il nostro simbolo dell'alfabeto sonoro.

`````

`````{tab} Superiore

Sia $\mathcal{C} = \{\mathbf{e}_1, \dots, \mathbf{e}_K\}$ un **codebook** di $K$
vettori-prototipo, appresi
durante l'addestramento. Dato un vettore latente $\mathbf{z}$ prodotto dall'encoder, la
quantizzazione sceglie il prototipo più vicino (in norma euclidea) e ne
restituisce l'**indice**:

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
$q$ fosse l'identità) e con una *commitment loss* $\beta \lVert \mathbf{z} -
\mathrm{sg}[\mathbf{e}_{k^\star}] \rVert^2$ che tiene i latenti vicini ai prototipi
($\mathrm{sg}$ è lo *stop-gradient*). Resta il compromesso di fondo: un codebook
grande ($K$ alto) ricostruisce meglio ma costa più bit per token; uno piccolo
comprime di più ma perde fedeltà.

`````

Vale la pena fare i conti a mano su un esempio minuscolo, perché il meccanismo è
tutto qui. Prendiamo un codebook di appena **quattro** prototipi in due
dimensioni:

$$
\mathbf{e}_1 = (0,0),\quad \mathbf{e}_2 = (1,0),\quad
\mathbf{e}_3 = (0,1),\quad \mathbf{e}_4 = (1,1).
$$

Vogliamo quantizzare il vettore latente $\mathbf{z} = (0{,}8,\ 0{,}1)$. Calcoliamo la
distanza quadratica da ciascun prototipo. Le due sbarrette con il quadratino,
$\lVert\,\cdot\,\rVert^2$, non chiedono niente di più di questo: per ognuno dei
due numeri fai la differenza, elevala al quadrato e somma. Per $\mathbf{e}_1 = (0,0)$
viene $(0{,}8-0)^2 + (0{,}1-0)^2 = 0{,}64 + 0{,}01 = 0{,}65$, e gli altri tre si
fanno allo stesso modo, con carta e penna:

$$
\lVert \mathbf{z} - \mathbf{e}_1\rVert^2 = 0{,}65,\quad
\lVert \mathbf{z} - \mathbf{e}_2\rVert^2 = 0{,}05,\quad
\lVert \mathbf{z} - \mathbf{e}_3\rVert^2 = 1{,}45,\quad
\lVert \mathbf{z} - \mathbf{e}_4\rVert^2 = 0{,}85.
$$

Il più vicino è $\mathbf{e}_2$: il token è **2**, e il vettore quantizzato è $(1,0)$.
Facciamo lo stesso con $\mathbf{u} = (0{,}2,\ 0{,}9)$: le distanze sono $0{,}85$,
$1{,}45$, $0{,}05$, $0{,}65$, il più vicino è $\mathbf{e}_3$, token **3**. Abbiamo
sostituito due vettori di numeri reali con due interi, `2` e `3`. Questo è tutto
ciò che serve per scrivere l'audio in un alfabeto.

Una nota di lettura per il codice che segue: qui i prototipi sono numerati da 1,
come nella formula, mentre Python conta da zero, quindi lo stesso prototipo si
chiamerà 1 invece che 2. È solo un modo di contare, e i due elenchi di token che
troverai più avanti vengono comunque da codebook diversi da questo: non devono
coincidere.

## Residual vector quantization: strati di precisione

C'è un problema, e lo si vede proprio nell'esempio. Sostituire
$(0{,}8,\ 0{,}1)$ con $(1,0)$ è comodo ma **grossolano**: ci siamo persi lo
scarto $(-0{,}2,\ 0{,}1)$. Per l'audio, questo scarto è la differenza tra una
voce naturale e una voce metallica da citofono. Potremmo allargare il codebook
(più prototipi, approssimazione più fine) ma per dimezzare l'errore
servirebbero *tantissimi* prototipi, e ogni raddoppio costa un bit in più per
token. Non regge.

Due parole su quelle unità di misura, perché tornano fino alla fine della
sezione. Un **bit** è una risposta sì/no, e con $n$ bit si distinguono $2^n$
casi: raddoppiare le voci del codebook vuol dire aggiungere esattamente una di
quelle risposte. Quanti bit al secondo serva in tutto un codec si chiama
**bitrate** e si misura in kbps, migliaia di bit al secondo. Attenzione al verso:
qui **più è basso, meglio è**, perché vuol dire meno roba da trasmettere a
parità di suono, ed è il contrario di quasi tutti gli altri numeri del libro.

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

Il risultato è una rappresentazione a due assi: lungo il **tempo**, un frame
ogni manciata di millisecondi; lungo la **profondità**, gli $N$ token della
cascata RVQ per ciascun frame. {numref}`fig-audio-codec-rvq` mostra il secondo
asse, che è quello nuovo: un frame solo, e la pila di token che ne esce, con
l'encoder che lo produce e il decoder che lo rilegge. Un secondo di audio
diventa così una piccola griglia di interi: con i parametri di sopra, 8 token
per ciascuno dei 75 frame, cioè **600 simboli** al posto di 24.000 campioni.

Da quei 600 numeri il decoder tira fuori un suono che *suona* come l'originale,
e la parola «ricostruire» va presa con le pinze. Quel decoder non impara solo a
sbagliare poco: accanto a lui, durante l'addestramento, c'è un
**discriminatore**, una seconda rete il cui unico mestiere è smascherare l'audio
finto (è il meccanismo delle GAN, a cui il libro dedica un capitolo più avanti).
Sotto quella pressione diventa a tutti gli effetti un generatore condizionato: a
bitrate bassi il dettaglio fine (la grana, le code di riverbero, le frequenze
più alte) non viene recuperato, viene **risintetizzato** in modo plausibile. Per
questo la fedeltà campione per campione crolla mentre la qualità percepita
regge, ed è anche il motivo per cui nella prossima sezione lo stesso decoder
potrà fare da generatore senza cambiare una riga: la differenza fra un codec e
un modello generativo, a quel punto, è solo da dove arrivano i token. Ed è
questo l'oggetto che daremo in pasto a un modello di linguaggio.

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

La RVQ, spogliata delle reti neurali, è pochissime righe di NumPy: un
codebook, la ricerca del prototipo più vicino, il residuo, e un secondo
codebook che lo rifinisce. Il codice seguente quantizza sei vettori 2D (un
«batch» giocattolo di latenti) e misura l'errore di ricostruzione con uno e
con due stadi.

```python
import numpy as np

rng = np.random.default_rng(0)

# Un piccolo "batch" di vettori latenti 2D da quantizzare (le z dell'encoder)
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
    """Per ogni riga di V trova il codice piu' vicino di C (nearest-neighbor)."""
    # distanze quadratiche fra ogni vettore e ogni prototipo
    d = ((V[:, None, :] - C[None, :, :]) ** 2).sum(axis=2)
    idx = d.argmin(axis=1)      # indice del prototipo piu' vicino: il "token"
    return idx, C[idx]          # indici e vettori ricostruiti


def mse(A, B):
    return ((A - B) ** 2).mean()


# --- Stadio 1: quantizzo il vettore ---
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

L'errore più che si dimezza, da $0{,}1021$ a $0{,}0481$, semplicemente
correggendo il residuo con un secondo codebook, e ogni vettore è ora descritto
da due interi invece che da due numeri reali. È l'intera idea della RVQ, in
scala di laboratorio: nei codec veri i vettori hanno centinaia di dimensioni,
i codebook migliaia di voci e gli stadi sono otto o più, ma la meccanica è
precisamente questa.

Un'onestà d'obbligo, prima di chiudere. Questo giocattolo usa codebook *fissati
a mano*; nei codec reali i prototipi si **apprendono** insieme all'encoder e al
decoder, di solito aggiornandoli come centroidi, in stile *k-means*, sui latenti
che li scelgono.

Quella regola porta con sé il guasto caratteristico di tutta la famiglia: una
voce che nessun latente sceglie **non viene più aggiornata**, quindi resta dov'è
e continua a non essere scelta. Il codebook effettivo si riduce in silenzio a
una frazione di $K$, mentre il bitrate nominale $N \log_2 K \cdot f_r$ resta
quello di prima: si pagano tutti i bit e se ne usa una parte. Si chiama
**codebook collapse**, e quanto morda dipende da come si parte: inizializzando i
prototipi con valori piccoli e casuali, in una simulazione a 64 voci ne restano
vive fra un terzo e la metà, e il conto cambia a ogni seme. I rimedi sono di
ingegneria e stanno nei codec citati qui sopra:
EnCodec {cite}`defossez2023high` sostituisce le voci inutilizzate con campioni
presi dal batch corrente (il *restart*, che nella stessa simulazione le riporta
tutte e 64 in gioco); altri quantizzano in uno spazio di dimensione più bassa e
normalizzano prototipi e latenti. Un codebook va sempre misurato per quante voci
**usa davvero**, non per quante ne dichiara.

Sulla misura della qualità serve poi una distinzione che il gergo tende a
cancellare. L'MSE sui campioni non è il criterio giusto nemmeno per
**addestrare**, perché è percettivamente cieco, e i codec veri usano perdite
spettrali multi-scala più un discriminatore avversario, che premiano ciò che
*suona* bene. Ma quelli sono **obiettivi di addestramento**: dicono al modello
dove andare, non dicono a noi dove è arrivato, e un discriminatore che promuove
il proprio generatore è metà di una partita, non un giudizio. **Misurare** è un
problema diverso e ancora aperto. Gli indicatori oggettivi esistono (PESQ e STOI
per il parlato, ViSQOL per la qualità generale, FAD per l'audio generato) ma
sono tutti surrogati, ciascuno tarato su un tipo di degrado e nessuno affidabile
fuori dal proprio ambito: PESQ e STOI, per dire, nascono per il parlato rovinato
da rumore o da codec a forma d'onda e si comportano male sull'uscita di un
decoder che il segnale se lo reinventa. Per questo i lavori del settore
continuano a chiudere con prove d'ascolto umane, il protocollo **MUSHRA**, con un
riferimento nascosto e un'ancora di bassa qualità mescolati agli altri stimoli,
così che chi ascolta non sappia mai cosa sta giudicando. Quando un lavoro
riporta un solo numero oggettivo, quel numero è un indizio, non la qualità.

Il principio della cascata sul residuo, però, è esattamente quello che hai
appena eseguito.

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
