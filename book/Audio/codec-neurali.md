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

La parola *codec* non è nuova. Ogni volta che ascoltate un brano in streaming
o salvate un vocale, un codec ha ridotto l'audio a una frazione della sua
dimensione. Il più famoso, l'**MP3** (standardizzato nel 1993), comprime
buttando via ciò che l'orecchio non sente: si appoggia a un modello
psicoacustico (un insieme di **regole fisse**, scritte a mano da ingegneri)
che decide quali frequenze sono mascherate da altre e quindi eliminabili. È un
ottimo mestiere artigianale, ma è *congelato*: quelle regole non cambiano, non
imparano, non si adattano ai dati.

Un codec **neurale** ribalta l'approccio. Invece di scrivere le regole, le fa
**imparare** a una rete. La struttura è quella di un **autoencoder** (la stessa
idea di compressione appresa che ritroveremo, per le immagini, nella diffusione
latente): un **encoder** che comprime l'input in una rappresentazione compatta,
e un **decoder** che da quella rappresentazione cerca di ricostruire l'originale. I due si addestrano *insieme*, con un'unica
regola: la ricostruzione deve somigliare all'ingresso.

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
È che questa rappresentazione compatta, imparata, possiamo poi
**arrotondarla** a un piccolo insieme di valori-tipo. E un valore-tipo è un
simbolo: un numero intero. È il ponte che stavamo cercando, dall'onda continua
all'alfabeto.

`````

`````{tab} Superiore

Un codec neurale è un autoencoder addestrato per la ricostruzione. L'encoder
$E$ mappa la forma d'onda $x$ in una sequenza di vettori latenti
$z = E(x)$ a **frequenza di frame** molto più bassa del tasso di campionamento
(un vettore ogni poche centinaia di campioni); il decoder $D$ ricostruisce
$\hat{x} = D(z)$. L'obiettivo minimizza una perdita di ricostruzione, spesso
combinando errore nel dominio del tempo e nello spettro (multi-scala
tempo–frequenza), ed è tipicamente affiancato da un **discriminatore** in stile
GAN che spinge $\hat{x}$ a suonare realistico, non solo a minimizzare l'errore
medio.

Fin qui è compressione con rappresentazione **continua**: $z$ è un vettore di
numeri reali. La novità che ci interessa è renderla **discreta**: sostituire
ogni vettore latente con un simbolo preso da un insieme finito. È il passaggio
che trasforma un compressore in un *tokenizzatore* del suono, e apre la porta
ai modelli di linguaggio sull'audio. Il come è il tema delle prossime due
sezioni.

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

La *vector quantization* fa esattamente questo, ma sui vettori latenti del
suono invece che sui colori dei pixel. La «tavolozza» si chiama **codebook**: un
elenco di vettori-prototipo. Ogni pezzetto di audio, dopo l'encoder, viene
avvicinato al prototipo più simile, e di lui si tiene solo il numero di
posizione. Quel numero è il **token**: il nostro simbolo dell'alfabeto sonoro.

`````

`````{tab} Superiore

Sia $C = \{e_1, \dots, e_K\}$ un **codebook** di $K$ vettori-prototipo, appresi
durante l'addestramento. Dato un vettore latente $z$ prodotto dall'encoder, la
quantizzazione sceglie il prototipo più vicino (in norma euclidea) e ne
restituisce l'**indice**:

$$
k^\star = \arg\min_{k \in \{1,\dots,K\}} \lVert z - e_k \rVert^2,
\qquad q(z) = e_{k^\star},
$$

dove $q(z)$ è il vettore quantizzato e $k^\star$ è il token: un intero in
$\{1, \dots, K\}$. L'audio non è più una sequenza di vettori reali ma una
sequenza di interi, esattamente come un testo tokenizzato.

Un dettaglio importante: l'operazione $\arg\min$ non è differenziabile, quindi
il gradiente non attraverserebbe la quantizzazione. Il VQ-VAE
{cite}`oord2017neural` lo aggira con lo **straight-through estimator** (il
gradiente del decoder viene copiato tal quale sull'uscita dell'encoder, come se
$q$ fosse l'identità) e con una *commitment loss* $\beta \lVert z -
\mathrm{sg}[e_{k^\star}] \rVert^2$ che tiene i latenti vicini ai prototipi
($\mathrm{sg}$ è lo *stop-gradient*). Resta il compromesso di fondo: un codebook
grande ($K$ alto) ricostruisce meglio ma costa più bit per token; uno piccolo
comprime di più ma perde fedeltà.

`````

Vale la pena fare i conti a mano su un esempio minuscolo, perché il meccanismo è
tutto qui. Prendiamo un codebook di appena **quattro** prototipi in due
dimensioni:

$$
e_1 = (0,0),\quad e_2 = (1,0),\quad e_3 = (0,1),\quad e_4 = (1,1).
$$

Vogliamo quantizzare il vettore latente $z = (0{,}8,\ 0{,}1)$. Calcoliamo la
distanza quadratica da ciascun prototipo:

$$
\lVert z - e_1\rVert^2 = 0{,}65,\quad
\lVert z - e_2\rVert^2 = 0{,}05,\quad
\lVert z - e_3\rVert^2 = 1{,}45,\quad
\lVert z - e_4\rVert^2 = 0{,}85.
$$

Il più vicino è $e_2$: il token è **2**, e il vettore quantizzato è $(1,0)$.
Facciamo lo stesso con $u = (0{,}2,\ 0{,}9)$: le distanze sono $0{,}85$,
$1{,}45$, $0{,}05$, $0{,}65$, il più vicino è $e_3$, token **3**. Abbiamo
sostituito due vettori di numeri reali con due interi, `2` e `3`. Questo è tutto
ciò che serve per scrivere l'audio in un alfabeto.

## Residual vector quantization: strati di precisione

C'è un problema, e lo si vede proprio nell'esempio. Sostituire
$(0{,}8,\ 0{,}1)$ con $(1,0)$ è comodo ma **grossolano**: ci siamo persi lo
scarto $(-0{,}2,\ 0{,}1)$. Per l'audio, questo scarto è la differenza tra una
voce naturale e una voce metallica da citofono. Potremmo allargare il codebook
(più prototipi, approssimazione più fine) ma per dimezzare l'errore
servirebbero *tantissimi* prototipi, e ogni raddoppio costa un bit in più per
token. Non regge.

La soluzione, elegante, è la **residual vector quantization** (RVQ), introdotta
per i codec neurali da **SoundStream** {cite}`zeghidour2021soundstream` (Google,
2021) e poi da **EnCodec** {cite}`defossez2023high` (Meta): invece di un solo
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

La RVQ applica $N$ quantizzatori in cascata sul **residuo**. Posto $r_0 = z$, al
livello $i$ si quantizza il residuo corrente con il codebook $C^{(i)}$ e si
aggiorna il residuo:

$$
k_i^\star = \arg\min_{k} \big\lVert r_{i-1} - e_k^{(i)} \big\rVert^2,
\qquad
r_i = r_{i-1} - e_{k_i^\star}^{(i)}.
$$

La ricostruzione finale è la somma dei prototipi scelti,
$q(z) = \sum_{i=1}^{N} e_{k_i^\star}^{(i)}$, e il token di quel frame diventa la
tupla di indici $(k_1^\star, \dots, k_N^\star)$: **$N$ flussi paralleli** di
interi. Poiché ogni stadio quantizza ciò che è avanzato, l'errore di
ricostruzione decresce in modo monotòno con $N$.

Il conto del **bitrate** è pulito. Con $N$ quantizzatori, codebook di $K$ voci
ciascuno e frequenza di frame $f_r$:

$$
\text{bitrate} = N \cdot \log_2 K \cdot f_r,
$$

dove $\log_2 K$ sono i bit per indice. EnCodec a $24$ kHz usa codebook di
$K = 1024$ voci ($10$ bit) a $f_r = 75$ frame al secondo: con $N = 8$
quantizzatori si ottengono $8 \cdot 10 \cdot 75 = 6000$ bit/s, cioè **6
kbps**, contro i 128 kbps e più di un MP3 di buona qualità, a fedeltà
comparabile. Variando $N$ si sceglie il compromesso: da $1{,}5$ kbps ($N=2$)
fino a $24$ kbps ($N=32$).

`````

Il risultato è una rappresentazione a due assi
({numref}`fig-audio-codec-rvq`): lungo il **tempo**, un frame ogni manciata di
millisecondi; lungo la **profondità**, gli $N$ token della cascata RVQ per
ciascun frame. Un secondo di audio diventa così una piccola griglia di interi
(poche migliaia di simboli) da cui il decoder sa ricostruire un suono ad alta
fedeltà. È questo l'oggetto che, nella prossima sezione, daremo in pasto a un
modello di linguaggio.

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

# Un piccolo "batch" di vettori latenti 2D da quantizzare
X = rng.uniform(-1, 1, size=(6, 2)).round(2)

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
idx1, q1 = quantizza(X, C1)
ric1 = q1                       # ricostruzione con 1 solo stadio

# --- Stadio 2: quantizzo il RESIDUO ---
residuo = X - q1
idx2, q2 = quantizza(residuo, C2)
ric2 = q1 + q2                  # ricostruzione con 2 stadi

print("vettori da quantizzare:\n", X)
print("token stadio 1:", idx1.tolist())
print("token stadio 2:", idx2.tolist())
print(f"MSE con 1 quantizzatore: {mse(X, ric1):.4f}")
print(f"MSE con 2 quantizzatori: {mse(X, ric2):.4f}")
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
decoder (di solito aggiornandoli come centroidi, in stile *k-means*, sui latenti
che li scelgono). E la fedeltà vera non si misura con l'MSE sui campioni, che è
percettivamente cieco, ma con perdite spettrali e discriminatori avversari che
premiano ciò che *suona* bene. Il principio della cascata sul residuo, però, è
esattamente quello che hai appena eseguito.

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
  cascata), ad alta fedeltà e bitrate bassissimo: poche kbps contro le decine
  di un MP3. Il bitrate è $N \cdot \log_2 K \cdot f_r$.
- Con due soli stadi, nell'esempio in NumPy, l'errore di ricostruzione più che si
  dimezza: è l'intera meccanica della RVQ in scala di laboratorio.
- Ottenuto l'alfabeto, l'audio *è* una sequenza di simboli: tutto
  l'armamentario dei Transformer diventa applicabile, ed è ciò che vedremo
  nella sezione sulla generazione.
```
