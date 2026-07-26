# Neural style transfer: la tua foto dipinta da van Gogh

Nell'estate del 2015 tre ricercatori dell'Università di Tubinga — Leon Gatys,
Alexander Ecker e Matthias Bethge — misero online una manciata di immagini
destinate a fare il giro del mondo: la Neckarfront, la fila di case sul fiume
che è la cartolina della loro città, ridipinta nello stile della *Notte
stellata* di van Gogh, dell'*Urlo* di Munch, di una composizione di Kandinsky
{cite}`gatys2016image`. Nessun pittore e nessun filtro fotografico: solo una
rete convoluzionale e una discesa del gradiente. Il metodo aveva un difetto —
minuti di calcolo per una singola immagine — ma l'idea era irresistibile, e
l'anno dopo scappò dal laboratorio: nel giugno 2016 l'app **Prisma** la portò
in tasca a milioni di persone, con oltre dieci milioni di download nelle prime
settimane e più di settanta milioni entro fine anno. Per un'estate i social si
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

In tutto il libro, finora, "addestrare" ha voluto dire una cosa sola: il
gradiente scende sui **pesi** della rete finché le predizioni migliorano. Il
neural style transfer capovolge lo schema. La rete — una VGG pre-addestrata su
ImageNet {cite}`simonyan2015very` — non impara nulla: i suoi pesi restano
**congelati**. A muoversi, un passo di gradiente alla volta, sono i **pixel
dell'immagine**.

`````{tab} Elementare

Immagina un critico d'arte dal giudizio infallibile ma immobile: non cambia
mai idea, sa solo valutare. Gli mostri una tela e lui ti dice due cose: quanto
la scena somiglia ancora alla tua foto, e quanto la pennellata somiglia a
quella del quadro che vuoi imitare. Tu ritocchi la tela un pochino, gliela
rimostri, ritocchi ancora — centinaia di volte, sempre nella direzione che
migliora i suoi due giudizi. Alla fine la tela è la tua foto, ma dipinta.

Il critico è la rete convoluzionale: ha già imparato a "vedere" su milioni di
immagini e qui non deve imparare altro. Ciò che cambia, ritocco dopo ritocco,
è soltanto l'immagine.

`````

`````{tab} Superiore

Formalmente è lo stesso problema di ottimizzazione, con le variabili
scambiate. L'addestramento classico cerca i parametri migliori a dati fissati,

$$
\hat{\theta} = \arg\min_{\theta} \; \mathcal{L}(\theta;\, X) ;
$$

qui cerchiamo l'**immagine** migliore a parametri fissati:

$$
\hat{X} = \arg\min_{X} \; \mathcal{L}(X;\, \theta),
$$

dove $X$ è il tensore-immagine che stiamo generando e $\theta$ sono i pesi
(congelati) della VGG. Per autograd non fa differenza: basta dichiarare $X$
come foglia con `requires_grad_(True)` e la backpropagation restituisce
$\partial \mathcal{L} / \partial X$, il gradiente della loss **rispetto ai
pixel**. È lo stesso meccanismo che rende possibili gli *esempi avversari* —
immagini ritoccate in modo impercettibile apposta per ingannare una rete —
qui usato a fin di bene.

`````

Perché proprio una CNN pre-addestrata? Perché, come abbiamo visto nel
capitolo sul Deep Learning e ritrovato nel transfer learning, i suoi strati
formano una **gerarchia**: i primi rispondono a bordi, colori e texture,
quelli profondi a parti di oggetti e a oggetti interi
{cite}`zeiler2014visualizing`. È esattamente la scala che serve per parlare,
con la stessa rete, sia di pennellate sia di case e campanili.

## Contenuto e stile: cosa c'è, come è dipinto

La scoperta di Gatys e colleghi è che dentro questa gerarchia contenuto e
stile vivono in posti diversi — e quindi si possono **separare** e
ricombinare.

`````{tab} Elementare

Pensa a un quadro come a due cose sovrapposte: il **soggetto** (una notte, un
paese, un cipresso) e la **mano del pittore** — la tavolozza dei colori, lo
spessore e la direzione delle pennellate, il ritmo delle texture. Riconosci
uno van Gogh da tre centimetri quadrati di cielo, senza sapere cosa
rappresenta il quadro: quella è la mano, non il soggetto. È come la grafia di
un amico: la riconosci su qualunque parola, perché non dipende da *cosa*
scrive ma da *come* scrive.

Nella rete succede lo stesso. Il "cosa c'è" abita negli strati profondi, che
rispondono agli oggetti e alla loro disposizione. Il "come è dipinto" abita
in una domanda diversa: *quali coppie di motivi elementari compaiono insieme?*
Pennellata spessa insieme a giallo acceso, tratto a spirale insieme a blu
notte. Contando queste coppie in tutta l'immagine — senza badare a *dove*
compaiono — si ottiene una specie di carta d'identità della mano del pittore.

`````

`````{tab} Superiore

Il **contenuto** è codificato dalle attivazioni di uno strato profondo (nel
paper, `conv4_2` della VGG-19): due immagini con attivazioni profonde simili
mostrano gli stessi oggetti nella stessa disposizione, anche se differiscono
pixel per pixel.

Lo **stile** è codificato dalle correlazioni tra i canali di uno strato. Allo
strato $l$ la rete produce $N_l$ mappe di attivazione di
$M_l = h_l \times w_l$ posizioni ciascuna; srotolando ogni mappa in una riga
si ottiene la matrice $F^{(l)} \in \mathbb{R}^{N_l \times M_l}$. La **matrice di Gram** è

$$
G^{(l)} = F^{(l)} \left( F^{(l)} \right)^{\top} \in \mathbb{R}^{N_l \times N_l},
$$

dove l'elemento $G^{(l)}_{ij}$ è il prodotto scalare tra il canale $i$ e il
canale $j$: misura quanto i due filtri si attivano *insieme*, sommando su
tutte le posizioni spaziali. In quella somma la geometria della scena sparisce
— resta solo la statistica delle co-occorrenze di texture e colori, cioè lo
stile. È per questo che il risultato conserva la disposizione della foto ma
non copia i cipressi di van Gogh: della *Notte stellata* sopravvivono solo le
correlazioni.

`````

## La loss composita

Per fondere le due cose serve una funzione costo con due termini, uno per
giudice:

$$
\mathcal{L} = \alpha \, \mathcal{L}_{\text{contenuto}} + \beta \, \mathcal{L}_{\text{stile}},
$$

dove $\alpha$ e $\beta$ sono i pesi che bilanciano fedeltà al soggetto e
fedeltà alla pennellata.

`````{tab} Elementare

$\alpha$ e $\beta$ sono due manopole. Con $\alpha$ alto comanda il giudice del
contenuto: la foto resta quasi intatta, con una leggera patina pittorica. Con
$\beta$ alto comanda il giudice dello stile: le pennellate prendono il
sopravvento e la scena scivola verso l'astratto. In pratica lo stile parte
svantaggiato — è più "difficile da accontentare" — e gli si dà molto più
peso: con $\alpha = 1$ e $\beta = 1000$, un errore di stile conta mille volte
un pari errore di contenuto. Trovare l'equilibrio giusto è questione di gusto,
letteralmente: si prova e si guarda il risultato.

`````

`````{tab} Superiore

Il termine di contenuto confronta le attivazioni dello strato scelto $l$ tra
immagine generata e foto:

$$
\mathcal{L}_{\text{contenuto}} = \frac{1}{2} \sum_{i,j} \left( F^{(l)}_{ij} - P^{(l)}_{ij} \right)^2 ,
$$

dove $F^{(l)}$ e $P^{(l)}$ sono le mappe di attivazione (canali × posizioni)
dell'immagine generata e della foto di contenuto. Il termine di stile
confronta le matrici di Gram su più strati (nel paper, il primo strato di
ogni blocco: `conv1_1`, `conv2_1`, `conv3_1`, `conv4_1`, `conv5_1`):

$$
\mathcal{L}_{\text{stile}} = \sum_{l} \frac{w_l}{4 N_l^2 M_l^2} \sum_{i,j} \left( G^{(l)}_{ij} - A^{(l)}_{ij} \right)^2 ,
$$

dove $G^{(l)}$ e $A^{(l)}$ sono le Gram dell'immagine generata e del quadro di
stile allo strato $l$, $w_l$ è il peso dello strato e il fattore
$1/(4 N_l^2 M_l^2)$ normalizza rispetto a numero di canali e posizioni. Usare
più strati cattura lo stile a più scale: dai granelli di colore alle volute
larghe. Nel paper il rapporto $\alpha/\beta$ è dell'ordine di $10^{-3}$–$10^{-4}$.

`````

## In pratica, con PyTorch

Bastano sorprendentemente poche righe: una VGG-19 congelata da cui leggere le
attivazioni, un'immagine dichiarata "ottimizzabile" e il solito loop — con
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
    return F @ F.T / (c * h * w)     # Gram (c, c) normalizzata

# img_contenuto, img_stile: tensori (1, 3, H, W) già ridimensionati
# e normalizzati con media e deviazione standard di ImageNet
with torch.no_grad():
    stile_rif, _ = attivazioni(img_stile)
    _, contenuto_rif = attivazioni(img_contenuto)
    gram_rif = [gram(f) for f in stile_rif]

# 2. Si ottimizza l'IMMAGINE: parte dalla foto, il gradiente scende sui pixel
img = img_contenuto.clone().requires_grad_(True)
opt = optim.Adam([img], lr=0.02)
alpha, beta = 1.0, 1e5

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

Due dettagli pratici. Primo: partire dalla *foto* invece che dal rumore rende
la convergenza più rapida e il risultato più fedele. Secondo: il paper
originale usava **L-BFGS**, che su questo problema converge in meno passi; in
PyTorch è `optim.LBFGS([img])` con una *closure*, al prezzo di un po' di
codice in più. Adam funziona bene ed è il gemello del training loop che già
conosciamo.

## L'eredità: da minuti a millisecondi

Il limite del metodo di Gatys è strutturale: ogni immagine è un problema di
ottimizzazione a sé, centinaia di passi di gradiente ogni volta. Johnson,
Alahi e Fei-Fei {cite}`johnson2016perceptual` lo aggirarono con una mossa
elegante: usare la loss di Gatys non per generare un'immagine, ma per
**addestrare una rete** feed-forward che trasforma qualunque foto in un dato
stile. L'ottimizzazione costosa si paga una volta sola, in fase di
addestramento; dopo, applicare lo stile è una singola passata in avanti —
tre ordini di grandezza più veloce, abbastanza per un video in tempo reale. È
la famiglia di tecniche che ha reso possibili app come Prisma, con il
compromesso di una rete da addestrare *per ciascuno stile*.

La storia poi è proseguita altrove. CycleGAN {cite}`zhu2017unpaired` ha
imparato a "tradurre" foto in quadri (e viceversa) con reti avversarie, senza
nemmeno bisogno di coppie di esempi; e oggi il trasferimento di stile è una
delle tante abilità dei **modelli di diffusione**, che con un'istruzione
testuale ridipingono un'immagine in qualunque maniera
{cite}`rombach2022high`. Ne parleremo nel capitolo sulle GAN, nella sezione
sulle evoluzioni. Ma l'idea di fondo — contenuto e stile come statistiche
diverse dentro una stessa rete — nasce qui, da una passeggiata sul Neckar.

```{admonition} Da ricordare
:class: important
- Nel neural style transfer **non si addestra la rete**: i pesi della VGG
  restano congelati e il gradiente scende **sui pixel** dell'immagine.
- Il **contenuto** vive nelle attivazioni degli strati profondi (*cosa* c'è);
  lo **stile** nelle correlazioni tra canali, riassunte dalla **matrice di
  Gram** $G = F F^{\top}$ (*come* è dipinto).
- La loss è composita:
  $\mathcal{L} = \alpha\,\mathcal{L}_{\text{contenuto}} + \beta\,\mathcal{L}_{\text{stile}}$,
  con $\alpha$ e $\beta$ a bilanciare fedeltà e pennellata.
- Il **fast style transfer** sposta il costo nell'addestramento di una rete
  feed-forward: stile applicato in una sola passata, in tempo reale.
```
