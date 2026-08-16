# Neural style transfer: la tua foto dipinta da van Gogh

Nell'estate del 2015 tre ricercatori dell'Università di Tubinga (Leon Gatys,
Alexander Ecker e Matthias Bethge) misero online una manciata di immagini
destinate a fare il giro del mondo: la Neckarfront, la fila di case sul fiume
che è la cartolina della loro città, ridipinta nello stile della *Notte
stellata* di van Gogh, dell’*Urlo* di Munch, di una composizione di Kandinsky
{cite}`gatys2016image`. Nessun pittore e nessun filtro fotografico: solo una
rete convoluzionale e la solita correzione a piccoli passi. Il metodo aveva un
difetto
(minuti di calcolo per una singola immagine) ma l'idea era irresistibile, e
l'anno dopo scappò dal laboratorio: nel giugno 2016 l'app **Prisma** la portò
in tasca a milioni di persone, con dieci milioni di download nelle prime
settimane e settanta milioni in quattro mesi. Per un'estate i social si
riempirono di gatti dipinti alla van Gogh.

Dietro il giocattolo c'è una domanda seria: che cosa sono, per una rete
neurale, il *contenuto* di un'immagine e il suo *stile*? E si possono
separare? Lo schema della risposta è in {numref}`fig-style-transfer`.

```{figure} ../figures/style-transfer.svg
:name: fig-style-transfer
:alt: "Tre riquadri affiancati: una scena semplice con casa e albero (il contenuto), un pannello di pennellate a spirale (lo stile) e, dopo una freccia, la stessa scena ridisegnata con quelle pennellate (il risultato)."
:width: 95%

Il neural style transfer combina il *contenuto* di una foto (cosa c'è) con lo
*stile* di un quadro (come è dipinto) in un'unica immagine nuova.
```

## L'idea capovolta: si ottimizza l'immagine, non la rete

In tutto il libro, finora, "addestrare" ha voluto dire una cosa sola: si
correggono a piccoli passi i **pesi** della rete, cioè i numeri interni che
decidono come si comporta, finché le sue risposte non migliorano. Il neural
style transfer capovolge lo schema. La rete, una **VGG** (il nome viene dal
laboratorio di Oxford che la costruì) già addestrata a riconoscere oggetti sul
grande archivio di foto etichettate ImageNet {cite}`simonyan2015very`, non
impara nulla: i suoi pesi restano **congelati**. A muoversi, un piccolo passo
alla volta, sono i **pixel dell'immagine**.

`````{tab} Elementare

Immagina un critico d'arte dal giudizio infallibile ma immobile: non cambia
mai idea, sa solo valutare. Gli mostri una tela e lui ti dice due cose: quanto
la scena somiglia ancora alla tua foto, e quanto la pennellata somiglia a
quella del quadro che vuoi imitare. Tu ritocchi la tela un pochino, gliela
rimostri, ritocchi ancora: centinaia di volte, sempre nella direzione che
migliora i suoi due giudizi. Alla fine la tela è la tua foto, ma dipinta.

Una domanda che viene naturale: da che cosa si parte, la prima volta? Da quello
che si vuole, ed è una scelta che conta. Si può partire dalla foto stessa, e
allora il critico ha già metà del lavoro fatto e si arriva prima. Oppure si può
partire da una tela di puntini a caso, quello che si chiama **rumore**, come
uno schermo televisivo senza segnale: ci vuole più pazienza, ma siccome i
puntini a caso sono ogni volta diversi, ogni volta esce un quadro diverso.

Il critico è la rete convoluzionale: ha già imparato a "vedere" su milioni di
immagini e qui non deve imparare altro. Ciò che cambia, ritocco dopo ritocco,
è soltanto l'immagine.

`````

`````{tab} Superiore

Formalmente è lo stesso problema di ottimizzazione, con le variabili
scambiate. L'addestramento classico cerca i parametri migliori a dati fissati,

$$
\hat{\theta} = \arg\min_{\theta} \; \mathcal{L}(\theta;\, \mathbf{X}) ;
$$

qui cerchiamo l’**immagine** migliore a parametri fissati:

$$
\hat{\mathbf{X}} = \arg\min_{\mathbf{X}} \; \mathcal{L}(\mathbf{X};\, \theta),
$$

dove $\mathbf{X}$ è il tensore-immagine che stiamo generando e $\theta$ sono i
pesi (congelati) della VGG. Per autograd non fa differenza: basta dichiarare
$\mathbf{X}$
come foglia con `requires_grad_(True)` e la backpropagation restituisce
$\partial \mathcal{L} / \partial \mathbf{X}$, il gradiente della loss
**rispetto ai pixel**. È lo stesso meccanismo che rende possibili gli *esempi avversari*
(immagini ritoccate in modo impercettibile apposta per ingannare una rete) qui
usato a fin di bene.

`````

Perché proprio una rete già addestrata? Perché, come abbiamo visto nel capitolo
sul Deep Learning e ritrovato nella sezione sul transfer learning, i suoi strati
formano una **gerarchia**. Ogni strato è fatto di rilevatori che si accendono
quando trovano quello che cercano, e più si va in profondità più quello che
cercano è grande: i primi si accendono su bordi, colori e piccole trame, quelli
profondi su parti di oggetti e su oggetti interi
{cite}`zeiler2014visualizing`. Serve proprio questo, perché una pennellata e un
campanile stanno a due scale diversissime e qui vanno giudicati tutti e due,
dalla stessa rete, nello stesso momento: ai primi strati si guarda la
pennellata, agli ultimi il campanile.

## Contenuto e stile: cosa c'è, come è dipinto

La scoperta di Gatys e colleghi è che dentro questa gerarchia contenuto e
stile vivono in posti diversi, e quindi si possono **separare** e ricombinare.

`````{tab} Elementare

Pensa a un quadro come a due cose sovrapposte: il **soggetto** (una notte, un
paese, un cipresso) e la **mano del pittore** (la tavolozza dei colori, lo
spessore e la direzione delle pennellate, il ritmo delle texture). Riconosci
uno van Gogh da tre centimetri quadrati di cielo, senza sapere cosa
rappresenta il quadro: quella è la mano, non il soggetto. È come la grafia di
un amico: la riconosci su qualunque parola, perché non dipende da *cosa*
scrive ma da *come* scrive.

Nella rete succede lo stesso. Il "cosa c'è" abita negli strati profondi, quelli
che si accendono sugli oggetti e sulla loro disposizione.

Il "come è dipinto" abita invece in una domanda diversa, e conviene arrivarci
per gradi. Prendi uno dei primi strati: dentro ci sono qualche decina o
centinaio di rilevatori, e ciascuno si accende su una cosa diversa, uno sulle
righe oblique, uno sul giallo acceso, uno sulle curve strette, uno sul blu
scuro. Quelli sono i **motivi elementari**: non li ha scelti nessuno, se li è
costruiti la rete addestrandosi su ImageNet, e sono gli stessi qualunque quadro
le si metta davanti.

Adesso la domanda: *quali di questi rilevatori si accendono insieme, negli
stessi punti del quadro?* Nella *Notte stellata* «curva stretta» e «blu scuro»
si accendono quasi sempre nello stesso posto, perché van Gogh disegna le
spirali col blu; «riga obliqua» e «giallo» pure. Si prendono allora tutte le
coppie possibili di rilevatori e si conta, per ciascuna, quanto spesso i due si
accendono insieme, **senza segnarsi dove**. Quella tabella di conteggi è la
carta d'identità della mano del pittore: dice quali ingredienti vanno assieme e
non dice niente su dove stiano, ed è esattamente per questo che si può
appiccicare a un'altra scena.

`````

`````{tab} Superiore

Il **contenuto** è codificato dalle attivazioni di uno strato profondo (nel
paper, `conv4_2` della VGG-19): due immagini con attivazioni profonde simili
mostrano gli stessi oggetti nella stessa disposizione, anche se differiscono
pixel per pixel.

Lo **stile** è codificato dalle correlazioni tra i canali di uno strato. Allo
strato $l$ la rete produce $N_l$ mappe di attivazione di
$M_l = h_l \times w_l$ posizioni ciascuna; srotolando ogni mappa in una riga
si ottiene la matrice $\mathbf{F}^{(l)} \in \mathbb{R}^{N_l \times M_l}$. La
**matrice di Gram** è

$$
\mathbf{G}^{(l)} = \mathbf{F}^{(l)} \left( \mathbf{F}^{(l)} \right)^{\top}
\in \mathbb{R}^{N_l \times N_l},
$$

dove l'elemento $G^{(l)}_{ij}$ è il prodotto scalare tra il canale $i$ e il
canale $j$: misura quanto i due filtri si attivano *insieme*, sommando su
tutte le posizioni spaziali. In quella somma la geometria della scena
sparisce: resta solo la statistica delle co-occorrenze di texture e colori,
cioè lo stile. È per questo che il risultato conserva la disposizione della
foto ma non copia i cipressi di van Gogh: della *Notte stellata* sopravvivono
solo le correlazioni.

`````

## La loss composita: due giudizi in un voto solo

Per fondere le due cose serve un numero che dica quanto la tela è ancora
sbagliata, e che sommi i due giudizi del critico. Quel numero, in tutto il
libro, si chiama **loss** (alla lettera «perdita»): più è alto, più c'è da
correggere, e il gradiente serve appunto ad abbassarlo. Qui la loss, scritta
$\mathcal{L}$, è la somma di due voci, una per giudice:

$$
\mathcal{L} = \alpha \, \mathcal{L}_{\text{contenuto}} + \beta \, \mathcal{L}_{\text{stile}},
$$

dove $\mathcal{L}_{\text{contenuto}}$ misura quanto la tela si è allontanata
dalla foto di partenza, $\mathcal{L}_{\text{stile}}$ quanto la pennellata è
ancora diversa da quella del quadro, e $\alpha$ e $\beta$ sono i due pesi che
decidono a quale delle due voci dare più importanza.

`````{tab} Elementare

$\alpha$ e $\beta$ sono due manopole. Con $\alpha$ alto comanda il giudice del
contenuto: la foto resta quasi intatta, con una leggera patina pittorica. Con
$\beta$ alto comanda il giudice dello stile: le pennellate prendono il
sopravvento e la scena scivola verso l'astratto.

In pratica allo stile si dà molto più peso, per esempio $\alpha = 1$ contro
$\beta = 1000$, e la ragione non è che lo stile sia più importante: è che i due
giudizi si misurano in unità diverse. Uno confronta attivazioni, l'altro
conteggi di coppie, e i loro numeri nascono di taglia diversa, come confrontare
metri e chilometri. I due pesi servono prima di tutto a rimetterli sulla stessa
scala. Trovare poi l'equilibrio giusto è questione di gusto, letteralmente: si
prova e si guarda il risultato.

Attenzione però a un tranello: quel mille non è un numero universale. Se si
cambia il modo di fare i conti dei due giudizi, cambia anche il rapporto che
li mette in pari, e il codice della prossima sezione li conta in un altro modo,
per cui lì lo stesso equilibrio si ottiene con un $\beta$ molto più grande. Un
numero del genere va sempre riletto insieme alla ricetta che lo accompagna, e
mai copiato da solo.

`````

`````{tab} Superiore

Il termine di contenuto confronta le attivazioni dello strato scelto $l$ tra
immagine generata e foto:

$$
\mathcal{L}_{\text{contenuto}} = \frac{1}{2} \sum_{i,j} \left( F^{(l)}_{ij} - P^{(l)}_{ij} \right)^2 ,
$$

dove $\mathbf{F}^{(l)}$ e $\mathbf{P}^{(l)}$ sono le mappe di attivazione
(canali × posizioni)
dell'immagine generata e della foto di contenuto. Il termine di stile
confronta le matrici di Gram su più strati (nel paper, il primo strato di
ogni blocco: `conv1_1`, `conv2_1`, `conv3_1`, `conv4_1`, `conv5_1`):

$$
\mathcal{L}_{\text{stile}} = \sum_{l} \frac{w_l}{4 N_l^2 M_l^2} \sum_{i,j} \left( G^{(l)}_{ij} - A^{(l)}_{ij} \right)^2 ,
$$

dove $\mathbf{G}^{(l)}$ e $\mathbf{A}^{(l)}$ sono le Gram dell'immagine
generata e del quadro di
stile allo strato $l$, $w_l$ è il peso dello strato e il fattore
$1/(4 N_l^2 M_l^2)$ normalizza rispetto a numero di canali e posizioni. Usare
più strati cattura lo stile a più scale: dai granelli di colore alle volute
larghe. Nel paper il rapporto $\alpha/\beta$ è dell'ordine di $10^{-3}$–$10^{-4}$,
ma quel numero è solidale con **questa** normalizzazione: cambiandola cambia
il rapporto utile, ed è il motivo per cui il codice della prossima sezione, che
normalizza in un altro modo, usa un $\beta$ di tutt'altra taglia.

`````

## In pratica, con PyTorch

Bastano sorprendentemente poche righe: una VGG-19 congelata da cui leggere le
attivazioni, un'immagine dichiarata "ottimizzabile" e il solito loop, con
l'ottimizzatore che riceve i pixel al posto dei pesi.

```{code-block} python
:class: pt-non-eseguibile

import torch
from torch import nn, optim
from torchvision import models

device = "cuda" if torch.cuda.is_available() else "cpu"

# 1. VGG-19 pre-addestrata: solo la parte convoluzionale, congelata
vgg = models.vgg19(weights=models.VGG19_Weights.IMAGENET1K_V1)
vgg = vgg.features.to(device).eval()
for p in vgg.parameters():
    p.requires_grad_(False)

STRATI_STILE = [0, 5, 10, 19, 28]   # conv1_1 ... conv5_1
STRATO_CONTENUTO = 21               # conv4_2

def attivazioni(x):
    stile, contenuto = [], None
    for i, strato in enumerate(vgg):
        x = strato(x)
        if i in STRATI_STILE:
            stile.append(x)
        elif i == STRATO_CONTENUTO:
            contenuto = x
        if i == STRATI_STILE[-1]:
            break                    # oltre conv5_1 non serve
    return stile, contenuto

def gram(f):
    _, c, h, w = f.shape             # f: (1, c, h, w)
    F = f.view(c, h * w)
    return F @ F.T / (c * h * w)     # Gram (c, c), normalizzata a modo nostro:
                                     # non e' la 1/(4 N^2 M^2) del paper, quindi
                                     # il beta qui sotto non e' quello del paper

# img_contenuto, img_stile: tensori (1, 3, H, W) già ridimensionati
# e normalizzati con media e deviazione standard di ImageNet
with torch.no_grad():
    stile_rif, _ = attivazioni(img_stile)
    _, contenuto_rif = attivazioni(img_contenuto)
    gram_rif = [gram(f) for f in stile_rif]

# 2. Si ottimizza l'IMMAGINE: parte dalla foto, il gradiente scende sui pixel
img = img_contenuto.clone().requires_grad_(True)
opt = optim.Adam([img], lr=0.02)
alpha, beta = 1.0, 1e5           # taglia solidale con la gram() qui sopra:
                                 # il 1000 della sezione precedente vale per
                                 # la normalizzazione del paper, non per questa

for passo in range(300):
    opt.zero_grad()
    stile_gen, contenuto_gen = attivazioni(img)
    l_contenuto = nn.functional.mse_loss(contenuto_gen, contenuto_rif)
    l_stile = sum(nn.functional.mse_loss(gram(f), g)
                  for f, g in zip(stile_gen, gram_rif))
    loss = alpha * l_contenuto + beta * l_stile
    loss.backward()
    opt.step()
```

Tre dettagli pratici, e sono anche le tre differenze rispetto all'articolo
originale di Gatys.

**Da dove si parte.** Qui si parte dalla foto, mentre nell'articolo si partiva
dai puntini a caso. Partire dalla foto fa arrivare al risultato in meno passi e
piega un po’ l'esito verso la struttura della foto, ma non lo rende «più
fedele» in generale: gli autori osservano che il punto di partenza incide poco
sull'esito finale. Quello a cui si rinuncia è la varietà, perché da una
partenza sempre uguale esce sempre la stessa immagine, mentre dai puntini a
caso se ne possono generare quante se ne vuole.

**Chi decide i passi.** Il ritocco della tela è affidato a un ottimizzatore,
cioè al pezzo di codice che, saputo di quanto si è sbagliato, decide come
muovere i pixel. Gli autori usavano L-BFGS, che su un problema come questo
arriva in meno passi ma va richiamato in un modo tutto suo; noi usiamo Adam,
che è lo stesso del ciclo di addestramento visto nel capitolo su PyTorch e
funziona benissimo.

**Un ritocco alla rete.** Gli autori, dove la VGG tiene solo il valore più
grande di ogni quadratino, preferivano tenerne la media
(`MaxPool2d` sostituito da `nn.AvgPool2d(2, 2)`), che a loro dire dà risultati
leggermente più gradevoli, e le immagini famose sono fatte così. Qui usiamo la
`vgg19` di torchvision com'è, come fa anche il tutorial ufficiale di PyTorch.

## L'eredità: da minuti a millisecondi

Il limite del metodo di Gatys è strutturale: ogni immagine è un problema di
ottimizzazione a sé, centinaia di passi di gradiente ogni volta. Johnson,
Alahi e Fei-Fei {cite}`johnson2016perceptual` lo aggirarono con una mossa
elegante: usare la loss di Gatys non per generare un'immagine, ma per
**addestrare una rete** feed-forward che trasforma qualunque foto in un dato
stile. L'ottimizzazione costosa si paga una volta sola, in fase di
addestramento; dopo, applicare lo stile è una singola passata in avanti: circa
mille volte più veloce (tre ordini di grandezza), abbastanza per un video in
tempo reale. È la
famiglia di tecniche che ha reso possibili app come Prisma, con il compromesso
di una rete da addestrare *per ciascuno stile*.

La storia poi è proseguita altrove. Per insegnare a un programma a tradurre
una foto in un quadro il modo ovvio sarebbe mostrargli tante coppie, la
stessa identica scena fotografata e dipinta, e nessuno le ha: Monet è morto e
non torna a dipingere su commissione. CycleGAN {cite}`zhu2017unpaired` ha
risolto il problema imparando **senza coppie**, da due mucchi separati e non
corrispondenti, tante foto da una parte e tanti Monet dall'altra. E oggi il
trasferimento di stile è una delle tante abilità dei **modelli di diffusione**,
che con un'istruzione scritta ridipingono un'immagine in qualunque maniera
{cite}`rombach2022high`. Di tutti e due parla il capitolo sulle GAN (sta per
*generative adversarial network*, le «reti generative avversarie»), nella
sezione sulle evoluzioni. Ma l'idea di fondo, contenuto e stile come due
conteggi diversi dentro una stessa rete, nasce qui, da una passeggiata sul
Neckar.

`````{tab} Elementare

```{admonition} Da ricordare
:class: important
- Qui **non si addestra la rete**: il critico d'arte (una rete che ha già
  imparato a vedere) non cambia mai idea, e a essere ritoccata centinaia di
  volte è la **tela**, cioè l'immagine stessa.
- Il **contenuto** (*cosa* c'è: la casa, il cipresso) si legge negli strati
  profondi della rete; lo **stile** (*come* è dipinto) sta nel conteggio di
  quali motivi elementari compaiono insieme, la carta d'identità della mano del
  pittore, che non dipende da dove quei motivi si trovino nell'immagine.
- Il giudizio da migliorare somma due voci, fedeltà al soggetto e fedeltà alla
  pennellata, pesate da due manopole: alzando quella dello stile le pennellate
  prendono il sopravvento, alzando quella del contenuto la foto resta quasi
  intatta.
- Chi ha fretta fa la fatica **una volta sola**: invece di ritoccare la tela
  per ogni foto, addestra una rete apposta per un solo stile, e da quel momento
  dipingere una foto qualunque è questione di un istante, abbastanza da stare
  dietro anche a un video. Il prezzo è che per un altro stile serve un altro
  addestramento.
```

`````

`````{tab} Superiore

```{admonition} Da ricordare
:class: important
- Nel neural style transfer **non si addestra la rete**: i pesi della VGG
  restano congelati e il gradiente scende **sui pixel** dell'immagine.
- Il **contenuto** vive nelle attivazioni degli strati profondi (*cosa* c'è);
  lo **stile** nelle correlazioni tra canali, riassunte dalla **matrice di
  Gram** $\mathbf{G} = \mathbf{F}\mathbf{F}^{\top}$ (*come* è dipinto).
- La loss è composita:
  $\mathcal{L} = \alpha\,\mathcal{L}_{\text{contenuto}} + \beta\,\mathcal{L}_{\text{stile}}$,
  con $\alpha$ e $\beta$ a bilanciare fedeltà e pennellata.
- Il **fast style transfer** sposta il costo nell'addestramento di una rete
  feed-forward: stile applicato in una sola passata, in tempo reale.
```

`````
