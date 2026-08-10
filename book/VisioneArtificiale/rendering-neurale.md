# Scene che si addestrano: NeRF e splatting

Nel 1999, per la scena in cui Neo schiva i proiettili piegandosi all'indietro
mentre la telecamera gli gira attorno, i fratelli Wachowski non avevano nessun
trucco software: avevano un anello di macchine fotografiche vere, montate su
una struttura, che scattavano in sequenza. Se la telecamera doveva passare per
un punto, in quel punto ci doveva essere una macchina fotografica. Il
*bullet time* costò una sala di posa e un impianto costruito apposta, ed è la
risposta di forza bruta a una domanda che la visione artificiale si pone da
sempre: **come si ottiene l'immagine da un punto di vista in cui nessuno è
mai stato?**

La domanda si chiama *sintesi di nuove viste*, ed è il rovescio esatto di
quella della sezione precedente. Là partivamo dalle immagini per ricavare la
geometria; qui vogliamo tornare alle immagini, da posizioni nuove. Per
trent'anni la strada è stata una sola: ricostruire un modello tridimensionale
(una superficie fatta di triangoli, con le fotografie incollate sopra come
texture) e poi renderizzarlo con la grafica tradizionale. Funziona, e fallisce
esattamente dove il mondo non è fatto di superfici nette: capelli, foglie,
fumo, vetro, riflessi.

Nel 2020 un articolo di sei autori di Berkeley, San Diego e Google propose di
smettere di ricostruire l'oggetto e di **addestrare una funzione**
{cite}`mildenhall2020nerf`. L'idea è tanto semplice da sembrare ingenua, e ha
riscritto un campo intero in un paio d'anni.

## La scena non è un oggetto, è una funzione

`````{tab} Elementare

Fermati un momento sull'idea di "modello 3D". Di solito è un elenco di cose:
questo triangolo sta qui, quest'altro là, sopra ci va questa immagine. È un
archivio, e come tutti gli archivi ha una risoluzione: più triangoli, più
dettaglio, più memoria.

Un **campo di radianza** è un'altra cosa. Non è un elenco, è una **risposta a
una domanda**. La domanda è: «se mi metto in questo punto dello spazio e
guardo in questa direzione, che colore vedo, e c'è qualcosa di solido qui?».
La scena diventa un oggetto che sa rispondere a quella domanda in ogni punto
e per ogni direzione, e la risposta la dà una piccola rete neurale.

Due conseguenze, che è il caso di sentire come strane prima di trovarle
normali. La prima: **la scena non ha una risoluzione**. Puoi chiedere il
colore in un punto qualsiasi, non c'è una griglia sotto. La seconda: la rete
non è addestrata su altre scene e non "sa" cosa siano gli alberi o le sedie.
**Viene addestrata su questa scena e su nient'altro**, a partire dalle foto
che le hai dato, e quando hai finito quella rete *è* quella scena. Non è un
modello di come sono fatte le stanze: è quella stanza lì, scritta in forma di
pesi.

Il colore dipende anche dalla direzione, e non è un dettaglio: è ciò che
permette a un riflesso di spostarsi mentre giri attorno a un tavolo lucido,
cosa che un colore incollato su un triangolo non sa fare.

`````

`````{tab} Superiore

Un **campo di radianza neurale** è una funzione

$$
F_\theta : (\mathbf{x}, \mathbf{d}) \;\longmapsto\; (\mathbf{c}, \sigma),
\qquad \mathbf{x} \in \mathbb{R}^3, \;\; \mathbf{d} \in \mathbb{S}^2,
$$

realizzata da un percettrone multistrato: in ingresso una posizione nello
spazio e una direzione di vista, in uscita un colore RGB $\mathbf{c}$ e una
**densità volumetrica** $\sigma \geq 0$, che si interpreta come probabilità
differenziale che un raggio venga fermato in quel punto.

Due scelte architetturali del lavoro originale sono cariche di significato.
La prima è che $\sigma$ dipende **solo** da $\mathbf{x}$, mentre $\mathbf{c}$
dipende da entrambi: è un vincolo imposto a mano che impedisce alla rete di
inventare geometria diversa per ogni punto di vista, ed è ciò che costringe la
forma a essere coerente. La seconda è che la direzione entra **tardi**, negli
ultimi strati, così che il grosso della capacità sia speso sulla struttura e
non sull'aspetto.

La rappresentazione è **implicita** e **continua**: non esiste una griglia,
non esiste una risoluzione, e la memoria occupata è quella dei pesi (nel
lavoro originale una manciata di megabyte per scena, contro i gigabyte di una
griglia voxel di pari qualità). In cambio, $F_\theta$ è ottimizzata **per una
singola scena**: non è un modello che generalizza, è una compressione con
perdita di quel particolare insieme di fotografie, in una forma che si può
interrogare da punti di vista nuovi.

`````

## Il rendering volumetrico, e perché è differenziabile

Avere una funzione che risponde punto per punto non basta: bisogna trasformare
quelle risposte in un'immagine. Il passaggio è la parte più importante di
tutto il meccanismo, e non è una rete: è fisica ottocentesca.

```{figure} ../figures/nerf-campo-di-radianza.svg
:name: fig-nerf-rendering
:alt: "Da una fotocamera parte un raggio che attraversa un volume tratteggiato contenente un oggetto. Lungo il raggio sono segnati sette punti di campionamento: quelli nell'aria sono piccoli e vuoti, quelli sulla superficie dell'oggetto sono grandi e pieni. Ogni campione, insieme alla direzione di vista, entra in una piccola rete che restituisce un colore e una densità. I campioni vengono composti in ordine di profondità e producono il colore di un singolo pixel, confrontato con il pixel della fotografia vera."
:width: 96%

Il ciclo che addestra un campo di radianza. Per ogni pixel si lancia un raggio,
si campionano dei punti, si interroga la rete, si compone il risultato in
ordine di profondità e si confronta con la foto vera. Ogni passaggio è
derivabile, quindi il gradiente rifà la strada all'indietro.
```

`````{tab} Elementare

Segui {numref}`fig-nerf-rendering`. Per calcolare il colore di **un solo
pixel**, si parte dalla fotocamera e si lancia un raggio nella scena, come se
si tendesse un filo. Lungo il filo si scelgono alcune decine di punti, e a
ognuno si chiede alla rete: che colore, e quanto sei solido?

Poi si sommano i colori, ma non in parti uguali. Si va dal più vicino al più
lontano tenendo il conto di quanta luce è già stata fermata: un punto conta
poco se è molto trasparente, e conta poco anche se è opaco ma sta **dietro** a
qualcosa di opaco, perché quel qualcosa lo nasconde. È lo stesso ragionamento
di una vetrata di più strati sovrapposti, o della nebbia: il primo strato
conta pieno, il secondo conta per quel che passa del primo, e così via.

Ecco il punto decisivo: **tutta questa procedura è fatta di somme e
moltiplicazioni**. Non c'è nessun passaggio brusco, nessuna decisione del tipo
«qui c'è una superficie, quindi mi fermo». E una catena di somme e
moltiplicazioni si può derivare: se il pixel calcolato è troppo scuro, si può
risalire all'indietro e capire quali punti dovevano essere più chiari, o meno
densi, e correggere la rete di conseguenza.

L'addestramento è quindi banale da descrivere: rendi un pixel, confrontalo con
la foto vera, misura la differenza, correggi. Ripeti per milioni di pixel
presi a caso da tutte le foto. Nessuno ha mai detto alla rete dove stanno le
superfici. La geometria compare da sola, perché è l'unica configurazione che
mette d'accordo tutte le fotografie insieme.

`````

`````{tab} Superiore

Il colore di un raggio $\mathbf{r}(t) = \mathbf{o} + t\mathbf{d}$ fra i limiti
$t_n$ e $t_f$ è l'integrale del rendering volumetrico:

$$
C(\mathbf{r}) = \int_{t_n}^{t_f} T(t)\, \sigma(\mathbf{r}(t))\,
\mathbf{c}(\mathbf{r}(t), \mathbf{d})\; \mathrm{d}t,
\qquad
T(t) = \exp\!\left(-\int_{t_n}^{t} \sigma(\mathbf{r}(s))\,\mathrm{d}s\right).
$$

$T(t)$ è la **trasmittanza**: la frazione di luce che arriva fino a $t$ senza
essere stata assorbita. È la legge di Beer-Lambert, e il suo effetto è che un
punto contribuisce in proporzione a quanto è denso *e* a quanto è libera la
strada davanti a lui.

In pratica l'integrale si valuta per quadratura su $N$ campioni con passo
$\delta_i = t_{i+1} - t_i$:

$$
\hat{C}(\mathbf{r}) = \sum_{i=1}^{N} T_i\,\alpha_i\, \mathbf{c}_i,
\qquad
\alpha_i = 1 - e^{-\sigma_i \delta_i},
\qquad
T_i = \prod_{j<i} (1 - \alpha_j).
$$

Chi conosce la grafica riconoscerà l'*alpha compositing* classico: la forma
discreta è esattamente il "sopra" di Porter e Duff, con l'opacità ricavata
dalla densità. I pesi $w_i = T_i \alpha_i$ formano una distribuzione lungo il
raggio, e la loro massa $\sum_i w_i$ è l'opacità totale, mentre $\sum_i w_i
t_i$ è la profondità attesa: **una mappa di profondità si ottiene gratis**,
senza averla mai addestrata.

La loss è la più elementare possibile, l'errore quadratico sui pixel resi
rispetto a quelli osservati, sommato sui raggi di un batch:

$$
\mathcal{L} = \sum_{\mathbf{r} \in \mathcal{R}}
\big\| \hat{C}(\mathbf{r}) - C_{\text{vera}}(\mathbf{r}) \big\|_2^2 .
$$

Ogni operazione della catena (interrogazione della rete, esponenziali,
prodotti cumulati, somma pesata) è derivabile, quindi $\nabla_\theta
\mathcal{L}$ si ottiene per differenziazione automatica come per qualunque
altra rete del libro. **Il rendering differenziabile è tutto il trucco**: la
supervisione arriva solo dalle immagini, e la struttura tridimensionale emerge
come unica spiegazione coerente con tutte insieme.

Restano due accorgimenti pratici del lavoro originale. I campioni si prendono
**stratificati** e casuali dentro ogni intervallo, non a posizioni fisse,
altrimenti la rete viene valutata sempre negli stessi punti e la
rappresentazione discretizza. E si campiona in due fasi (*hierarchical
sampling*): una rete grossolana individua dove stanno i pesi, una fine mette i
campioni lì, perché spendere calcolo nell'aria vuota è sprecarlo.

`````

## Perché serve la codifica posizionale

C'è un dettaglio che, tolto, fa collassare il metodo in un'immagine sfocata. È
lo stesso fenomeno che il capitolo sulle PINN descrive come limite delle reti
sui fronti ripidi, e qui torna con la soluzione in mano.

`````{tab} Elementare

Una rete a strati densi, alimentata direttamente con le coordinate $(x, y,
z)$, impara facilmente le cose che cambiano lentamente nello spazio e con
enorme fatica quelle che cambiano in fretta. Un muro uniforme lo prende
subito; il bordo netto fra il muro e la finestra, o la trama del legno, quasi
mai. Il risultato è una scena giusta ma smarrita nella nebbia.

Il rimedio è sorprendente: invece di dare alla rete le coordinate, le si danno
**molte onde di quelle coordinate**, con frequenze via via più alte, seni e
coseni raddoppiati a ogni passo. Due punti vicinissimi, che come numeri
grezzi si somigliano quasi del tutto, sulle onde ad alta frequenza diventano
subito diversi, e la rete può finalmente distinguerli.

È esattamente lo stesso trucco della codifica posizionale dei Transformer, che
il libro ha già incontrato: là serviva a dare un'identità a posizioni in una
frase, qui a dare un'identità a punti nello spazio. Stesso problema, stessa
soluzione, due campi che non si parlavano.

`````

`````{tab} Superiore

Il fenomeno è lo **spectral bias**: una rete densa apprende le componenti di
Fourier a bassa frequenza in poche iterazioni e quelle ad alta frequenza in un
numero molto maggiore {cite}`rahaman2019spectral`. Per un campo di radianza è
letale, perché il dettaglio visivo *è* alta frequenza.

La soluzione è mappare l'ingresso in uno spazio di dimensione maggiore prima
di darlo alla rete:

$$
\gamma(p) = \big(\sin(2^0 \pi p),\, \cos(2^0 \pi p),\, \dots,\,
\sin(2^{L-1} \pi p),\, \cos(2^{L-1} \pi p)\big),
$$

applicata a ciascuna delle tre coordinate (con $L = 10$ nel lavoro originale)
e alle componenti della direzione (con $L = 4$). Non è un espediente: Tancik e
colleghi mostrano che le *Fourier features* trasformano il **neural tangent
kernel** dell'MLP in un kernel stazionario di banda regolabile, e che senza di
esse una rete densa non può, in teoria prima ancora che in pratica, apprendere
le alte frequenze in domini di bassa dimensione {cite}`tancik2020fourier`.

Il legame con i Transformer non è un'analogia vaga: la forma è la stessa,
sinusoidi a frequenze geometricamente scalate, e il ruolo è lo stesso, rendere
distinguibili ingressi vicini. La differenza sta nel dominio (posizioni intere
in una sequenza contro coordinate reali in $\mathbb{R}^3$) e nella scelta
delle frequenze, che qui si fa in base alla risoluzione più fine che si vuole
rappresentare.

`````

## Il costo, e come è crollato

Il NeRF originale era splendido e proibitivo: **uno o due giorni** di
addestramento su una GPU per una scena sola, e decine di secondi per rendere
un fotogramma. Con quei numeri il metodo è un articolo, non una tecnologia. In
due anni sono diventati secondi e millisecondi, e la ragione per cui è
successo è istruttiva.

`````{tab} Elementare

Il conto è impietoso: per ogni pixel servono decine di interrogazioni della
rete, e un'immagine ha un milione di pixel. Se la rete è grande, non si
finisce più. L'idea che ha sbloccato tutto è stata smettere di chiedere alla
rete di ricordare **anche dove stanno le cose**, e darle un aiuto.

Invece di una rete grande che deve contenere in sé tutta la scena, si tiene
accanto una tabella di appunti indicizzata per posizione, a più livelli di
dettaglio, e alla rete si passa ciò che c'è scritto negli appunti vicini al
punto richiesto. Gli appunti si imparano insieme ai pesi. Così la rete può
essere minuscola, perché non deve più memorizzare: deve solo interpretare.

È lo stesso baratto che si incontra ovunque nell'informatica: memoria contro
calcolo. Qui la memoria costa poco e il calcolo costava tantissimo, e
spostare il peso da una parte all'altra ha accorciato l'addestramento di
diversi ordini di grandezza.

`````

`````{tab} Superiore

**Instant-NGP** sostituisce la codifica sinusoidale con una **codifica hash
multirisoluzione**: $L$ livelli di griglia a risoluzioni geometricamente
crescenti, ciascuno con una tabella di vettori di feature addestrabili
indicizzata da una funzione hash spaziale. Per un punto si interpolano
trilinearmente i vettori degli otto vertici di ogni livello, si concatenano, e
si dà il risultato a un MLP **molto** piccolo (due strati da 64 unità)
{cite}`muller2022instant`.

La parte controintuitiva è che le collisioni della tabella hash **non si
risolvono**: si lasciano. Il gradiente medio di due punti che collidono è
dominato da quello dove c'è densità, perché la regione vuota contribuisce
poco alla loss, e i livelli a risoluzione diversa collidono in modi diversi,
quindi l'ambiguità di un livello viene sciolta dagli altri. Il risultato è un
addestramento di ordini di grandezza più veloce, con qualità paragonabile, e
rendering a decine di millisecondi per fotogramma in alta definizione.

Vale la pena leggerlo per quello che è: una parte sostanziale della
rappresentazione si è spostata dai **pesi** a una **struttura dati esplicita e
addestrabile**. Il campo continuo resta, ma non è più tutto dentro l'MLP.

`````

## Splatting: dai raggi ai granelli

Nel 2023 lo stesso obiettivo è stato raggiunto da un'altra direzione, e il
risultato ha cambiato di nuovo cosa si intende per stato dell'arte.

`````{tab} Elementare

Il rendering per raggi ha un costo strutturale: bisogna *cercare* dove sta la
materia lungo ogni raggio, e la maggior parte dei campioni cade nel vuoto. E
se invece di cercare la materia partendo dall'occhio, si partisse dalla
materia e la si proiettasse sullo schermo?

È l'idea dello **splatting**: la scena si rappresenta come qualche milione di
granelli sfumati, ciascuno con la sua posizione, la sua forma (schiacciata,
allungata, orientata come serve), il suo colore e la sua trasparenza. Per fare
un'immagine, si proietta ogni granello sullo schermo, si ordinano per
profondità e si sovrappongono. Nessuna ricerca, nessun campionamento a vuoto,
e le schede grafiche fanno questo tipo di lavoro da trent'anni: è il loro
mestiere.

Il risultato è che la scena si guarda in tempo reale, muovendosi liberamente,
con la stessa qualità di prima. E l'addestramento resta quello di sempre:
confronta con le foto, correggi. Solo che qui a essere corretti non sono i
pesi di una rete, sono **posizione, forma, colore e trasparenza dei granelli**,
e ogni tanto si aggiungono granelli dove il dettaglio manca e si tolgono dove
sono inutili.

`````

`````{tab} Superiore

Il **3D Gaussian Splatting** rappresenta la scena con un insieme di gaussiane
tridimensionali anisotrope, ciascuna definita da un centro $\boldsymbol{\mu}$,
una covarianza $\boldsymbol{\Sigma}$ (parametrizzata come $\mathbf{R}
\mathbf{S} \mathbf{S}^\top \mathbf{R}^\top$ con scala e rotazione separate, per
mantenerla semidefinita positiva durante l'ottimizzazione), un'opacità e dei
coefficienti di armoniche sferiche per il colore dipendente dalla direzione
{cite}`kerbl20233d`.

La proiezione di una gaussiana 3D sul piano immagine è ancora una gaussiana
2D, il che rende il rendering una **rasterizzazione** invece di un *ray
marching*: si ordina per profondità, si compone con la stessa formula di
$\alpha$-blending vista sopra, e si sfrutta appieno l'hardware grafico. Gli
autori riportano sintesi di nuove viste in tempo reale ($\geq$ 30 fotogrammi
al secondo) a risoluzione 1080p, con qualità allo stato dell'arte e tempi di
addestramento competitivi.

L'ottimizzazione alterna discesa del gradiente sui parametri e un
**controllo adattivo della densità**: le gaussiane con gradiente di posizione
grande vengono clonate (se piccole, la scena è sotto-ricostruita) o divise (se
grandi, la scena è sotto-rappresentata), e quelle quasi trasparenti vengono
rimosse. È una rappresentazione **esplicita** che si comporta come una
continua, e chiude il cerchio: il pendolo torna verso le primitive
geometriche, ma con la loss differenziabile del rendering neurale.

`````

## Cosa questo cambia, e cosa resta difficile

Conviene dire con precisione che cosa è stato risolto, perché intorno a questi
metodi la retorica è abbondante.

**Cosa funziona.** Data una manciata di decine di fotografie di una scena
statica, con pose note, si ottiene una rappresentazione che permette di
guardarla da punti di vista nuovi con realismo fotografico, comprese le
trasparenze e i riflessi, in tempo reale, con qualche minuto di calcolo. Dieci
anni fa era fantascienza.

**Cosa serve, e viene dalla sezione precedente.** Le **pose** delle
fotocamere. Praticamente ogni pipeline le ottiene da una ricostruzione
*structure from motion*, e quando quella sbaglia il campo di radianza non
sbaglia un po': produce una nuvola incoerente. La geometria classica non è
stata sostituita, è diventata l'infrastruttura su cui il metodo poggia.

**Cosa resta aperto.** Tre cose, e conviene distinguerle.

- **Si addestra una scena alla volta.** Non c'è nessun transfer: il modello di
  ieri non aiuta la scena di oggi. I lavori che generalizzano da poche viste,
  o addirittura da una sola, esistono, ma pagano in qualità e sono un altro
  problema, più vicino ai modelli generativi che alla ricostruzione.
- **Le scene sono statiche.** Estendere al tempo (persone che si muovono,
  foglie che oscillano) è possibile ed è materia di ricerca attiva, ma
  aggiunge una dimensione a un problema già mal posto.
- **Modificare è difficile.** Una mesh si modifica: si sposta un vertice, si
  cambia una texture. Un campo di radianza è una funzione appresa, e "sposta
  quella sedia" non è un'operazione che abbia un senso ovvio. Lo splatting,
  essendo esplicito, sta un po' meglio, ed è una delle ragioni della sua
  fortuna.

Su una cosa vale la pena non lasciarsi trascinare: questi metodi **non
capiscono** la scena. Non sanno che c'è una sedia, non sanno che il tavolo
continua dietro il vaso, non sanno cosa succederebbe spingendolo. Sono
un'interpolazione straordinariamente buona fra le fotografie che hanno visto.
La differenza fra saper rigenerare le apparenze di un mondo e averne un
modello è precisamente il tema del capitolo sui world model, e questi sistemi
stanno tutti dalla parte delle apparenze.

## In pratica: la composizione lungo un raggio

Il cuore del metodo, la composizione volumetrica, sono cinque righe di NumPy e
si può guardare da vicino senza addestrare niente. Costruiamo un raggio che
attraversa sei metri di vuoto con una superficie opaca a quattro metri, e
vediamo che cosa fanno i pesi.

```python
import numpy as np

def rendi_raggio(sigma, colori, delta):
    """Composizione volumetrica lungo un raggio.
    sigma: densità per campione; colori: (N,3); delta: passo fra i campioni."""
    alpha = 1.0 - np.exp(-sigma * delta)                 # quanto ogni campione occlude
    trasmittanza = np.cumprod(np.concatenate([[1.0], 1.0 - alpha[:-1]]))
    pesi = trasmittanza * alpha                          # quanto ogni campione conta
    return (pesi[:, None] * colori).sum(axis=0), pesi

N, lunghezza = 60, 6.0
delta = lunghezza / N                                    # 10 cm fra un campione e l'altro
t = np.arange(N) * delta                                 # 0.0, 0.1, ... 5.9 metri

# vuoto, e a quattro metri una superficie opaca
sigma = np.where(np.isclose(t, 4.0), 60.0, 0.0)
colori = np.tile(np.array([0.71, 0.33, 0.17]), (N, 1))   # terracotta

C, pesi = rendi_raggio(sigma, colori, delta)
print("colore reso       :", np.round(C, 3))
print("massa dei pesi    :", round(float(pesi.sum()), 4))
print("profondità attesa :", round(float((pesi * t).sum()), 3), "m")

# la stessa scena riempita di nebbia: nessuna superficie, i pesi si spalmano
C2, pesi2 = rendi_raggio(np.full(N, 0.45), colori, delta)
print("massa con la nebbia:", round(float(pesi2.sum()), 4),
      "| il picco dei pesi vale", round(float(pesi2.max()), 4),
      "contro", round(float(pesi.max()), 4), "della superficie")
```

Tre numeri da leggere con attenzione. La **massa dei pesi** vale $0{,}9975$:
quasi tutta la luce viene fermata, e il $0{,}25\%$ che passa non è un errore
numerico, è $e^{-\sigma \delta} = e^{-6}$, cioè quanto resta davvero di un
raggio dopo aver attraversato quello spessore. La **profondità attesa** vale
$3{,}99$ m senza che nessuno abbia mai calcolato una profondità: è la media
delle distanze pesata dai $w_i$, ed è il modo in cui da un campo di radianza si
estrae una mappa di profondità. Nel caso della nebbia, infine, il picco dei
pesi vale $0{,}044$ contro $0{,}9975$: i pesi si spalmano lungo tutto il
raggio invece di concentrarsi, che è esattamente la firma numerica di
"nessuna superficie qui", e la ragione per cui questi metodi rendono bene il
fumo e la foschia, dove una mesh non saprebbe che pesci pigliare.

```{admonition} Da ricordare
:class: important
- Un **campo di radianza** rappresenta una scena come una *funzione*
  $(\mathbf{x}, \mathbf{d}) \mapsto (\mathbf{c}, \sigma)$, non come un elenco
  di triangoli: continua, senza risoluzione, e addestrata **su una scena sola**.
- Il colore di un pixel si ottiene per **composizione volumetrica** lungo un
  raggio, $\hat{C} = \sum_i T_i \alpha_i \mathbf{c}_i$: legge di
  Beer-Lambert, cioè l'$\alpha$-blending della grafica.
- Tutta la catena è **differenziabile**, quindi basta confrontare i pixel resi
  con le foto vere: la geometria emerge da sola, come unica spiegazione
  coerente con tutte le immagini insieme. Nessuno la supervisiona.
- Senza **codifica posizionale** il metodo produce nebbia: è lo *spectral
  bias*, lo stesso limite che il capitolo sulle PINN descrive, e la soluzione
  è la stessa forma sinusoidale della codifica posizionale dei Transformer.
- Il costo è crollato spostando la rappresentazione dai pesi a una struttura
  dati addestrabile (**Instant-NGP**, codifica hash multirisoluzione) e poi
  passando dai raggi ai granelli (**3D Gaussian Splatting**), che rasterizza
  invece di marciare e rende in tempo reale.
- Le **pose** delle fotocamere restano un ingresso obbligatorio, e vengono
  dalla *structure from motion*: la geometria classica non è stata sostituita,
  è diventata l'infrastruttura.
- Questi metodi **non capiscono** la scena: la sanno rigenerare. È
  un'interpolazione eccellente fra le viste osservate, non un modello del
  mondo.
```
