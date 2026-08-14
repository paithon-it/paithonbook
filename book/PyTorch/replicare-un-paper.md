# Replicare un paper

Un articolo scientifico si può leggere in tre modi. Il primo è scorrerlo:
mezz'ora, si ricava l'idea generale e si dimentica in una settimana. Il
secondo è studiarlo: si seguono le derivazioni, si capisce l'argomento. Il
terzo è **farlo girare**: trasformare le equazioni in `nn.Module`, mandare
avanti un tensore e guardare se esce quello che deve uscire. È l'unico modo
che non consente di autoingannarsi, perché il codice non accetta i passaggi
vaghi: dove il testo dice "si proietta linearmente" bisogna decidere una
matrice, e la matrice ha una forma precisa.

Quello che questa sezione insegna è un **metodo**, ed è la cosa più
trasferibile che si possa imparare qui dentro. Il modello su cui lo mettiamo
alla prova usa due strati che il libro spiegherà più avanti: la
**convoluzione**, nel capitolo sul deep learning, e l'**attenzione
multi-testa**, in quello sui Transformer.

Non serve sapere che cosa facciano dentro. Contano come scatole di cui si
conosce solo che forma entra e che forma esce, ed è precisamente il punto:
quello che si controlla sono le **proprietà che devono valere comunque**, prima
e a prescindere da qualunque addestramento (nel gergo si chiamano
*invarianti*), e quelle si verificano dal di fuori, senza aprire le scatole.
Per questo il metodo funziona anche su un articolo di cui non si è capito
tutto.

## Il metodo, in quattro mosse

`````{tab} Elementare
Replicare un paper somiglia a montare un mobile a partire da una fotografia
invece che dalle istruzioni. Si procede così.

**Uno: fai l'inventario dei pezzi.** Quasi ogni articolo ha una figura
dell'architettura e una tabella di numeri (quanti strati, quanto sono larghi).
Quelle due cose insieme sono la distinta dei materiali.

**Due: traduci un'equazione alla volta.** Le formule di un paper sono
tipicamente tre o quattro, e ognuna diventa poche righe di codice. Si va in
ordine, e non si passa alla successiva finché la precedente non gira.

**Tre: controlla le misure a ogni passo.** Dopo ogni pezzo, si manda dentro un
tensore finto e si guarda che forma esce. È l'equivalente del metro da
falegname: se una misura non torna, l'errore è lì, non tre pezzi più avanti.

**Quattro: conta i pezzi alla fine.** Se il paper dice che il modello ha 86
milioni di parametri e il tuo ne ha 40, hai saltato qualcosa. È la verifica più
potente di tutte, e non richiede di addestrare nulla.
`````

`````{tab} Superiore
Formalizzato, il procedimento è una verifica incrementale su tre invarianti,
tutti controllabili **senza addestrare**:

1. **Invariante di forma.** Ogni modulo definisce una mappa
   $f: \mathbb{R}^{d_{\text{in}}} \to \mathbb{R}^{d_{\text{out}}}$; se ne
   verifica il tipo
   con un tensore casuale della forma dichiarata nel paper. Un `assert` sulla
   shape in uscita è un test unitario a costo zero.
2. **Invariante di conteggio.** Il numero di parametri è una funzione chiusa
   degli iperparametri architetturali, e i paper lo dichiarano. Coincidere a
   meno dell'1% significa che la struttura è quella; discostarsi del 30%
   significa che manca un blocco o che una dimensione è sbagliata.
3. **Invariante di gradiente.** Un `backward()` su una loss finta deve
   produrre `p.grad is not None` per **ogni** parametro. Un tensore creato con
   `torch.tensor(...)` invece che `nn.Parameter`, o un ramo staccato per
   sbaglio con `detach()`, si scopre qui e non dopo tre giorni di
   addestramento che non converge.

Solo dopo che i tre invarianti sono soddisfatti ha senso parlare di risultati
numerici, ed è a quel punto che comincia la parte difficile.
`````

## Il caso: il Vision Transformer

Prendiamo un articolo che il libro incontrerà più volte più avanti: *An Image
is Worth 16x16 Words* {cite}`dosovitskiy2021image`, che nel 2021 ha portato
l'architettura Transformer dentro la visione artificiale. È un ottimo caso di
studio perché l'architettura è breve (quattro equazioni) e perché i numeri da
verificare sono pubblicati. Che sia un articolo di cui non abbiamo ancora
letto la teoria è, per una volta, un vantaggio: mostra che il metodo non
richiede di aver capito prima il modello, e che si può montare qualcosa
correttamente pur non sapendo ancora perché funzioni.

La prima equazione del paper costruisce la sequenza di ingresso, e in parole
dice questo: l'immagine viene tagliata in quadratini (in inglese *patch*, ed è
la parola che si troverà nel codice), ognuno viene trasformato in una fila di
numeri, si mette davanti a tutti un quadratino in più che immagine non è, e
infine si dice al modello in che ordine stavano. Quest'ultimo passo serve
perché, una volta tagliata, l'immagine è diventata un mucchio di pezzi
sciolti: senza qualcosa che lo dica, il modello non saprebbe più quale stava in
alto a sinistra.

Eccola come appare sull'articolo. Non serve decifrarla simbolo per simbolo:
dice in notazione compatta esattamente la frase appena letta, con le parentesi
quadre che mettono i quadratini in fila, $\mathbf{E}$ che li trasforma e la
somma finale che aggiunge le posizioni. I simboli sono sciolti qui sotto per
chi li vuole, ma il conto che conta viene dopo.

$$
\mathbf{z}_0 = [\, \mathbf{x}_{\text{class}} ;\;
\mathbf{x}_p^{1}\mathbf{E} ;\; \mathbf{x}_p^{2}\mathbf{E} ;\; \dots ;\;
\mathbf{x}_p^{N}\mathbf{E} \,] + \mathbf{E}_{\text{pos}}
$$

dove $\mathbf{x}_p^{i}$ è la $i$-esima patch appiattita, $\mathbf{E} \in
\mathbb{R}^{(P^2 \cdot C) \times D}$ è la proiezione lineare condivisa,
$\mathbf{x}_{\text{class}}$ è un vettore imparabile premesso alla sequenza e
$\mathbf{E}_{\text{pos}} \in \mathbb{R}^{(N+1) \times D}$ sono le codifiche di
posizione. Con immagini $224 \times 224$, patch $P = 16$ e $C = 3$ canali (i
tre colori, rosso verde e blu, di cui è fatta ogni foto) si ottengono
$N = (224/16)^2 = 196$ patch, ciascuna di $16 \cdot 16 \cdot 3 = 768$ numeri.
Il quadrato viene da lì: $224/16 = 14$ è il numero di quadratini che stanno su
**una** riga, e l'immagine è una griglia, quindi le righe sono altrettante e i
quadratini in tutto sono $14 \times 14$.

Una coincidenza da segnalare, perché altrimenti confonde: il $768$ appena
calcolato ($16 \cdot 16 \cdot 3$, quanti numeri contiene una patch) e il $768$
che comparirà fra poco nel codice come `d_modello` **sono due cose diverse**.
Il primo è quanto entra nella proiezione, il secondo quanto ne esce, ed è una
scelta degli autori del paper. Che coincidano vuol dire soltanto che la
proiezione, in questo caso, non cambia il numero di numeri; con patch da $32$
pixel il primo diventerebbe $3072$ e il secondo resterebbe $768$.

Ecco la stessa equazione, in PyTorch:

```python
import torch
from torch import nn

class IncorporazionePatch(nn.Module):
    """Equazione 1 del paper: da immagine a sequenza di token."""

    def __init__(self, canali=3, patch=16, d_modello=768, immagine=224):
        super().__init__()
        n_patch = (immagine // patch) ** 2                    # 196

        # Il trucco: una convoluzione con kernel = stride = patch È la
        # proiezione lineare delle patch appiattite, calcolata tutta insieme.
        self.proiezione = nn.Conv2d(canali, d_modello,
                                    kernel_size=patch, stride=patch)

        self.token_classe = nn.Parameter(torch.randn(1, 1, d_modello))
        self.posizioni = nn.Parameter(torch.randn(1, n_patch + 1, d_modello))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B = x.shape[0]
        x = self.proiezione(x)                    # (B, 3, 224, 224) -> (B, 768, 14, 14)
        x = x.flatten(2).transpose(1, 2)          #                  -> (B, 196, 768)
        cls = self.token_classe.expand(B, -1, -1) #                     (B, 1, 768)
        x = torch.cat([cls, x], dim=1)            #                  -> (B, 197, 768)
        return x + self.posizioni                 # broadcast su tutto il batch
```

La colonna di commenti a destra del `forward` è la terza mossa in atto, il
metro da falegname, e vale la pena leggerla ad alta voce. `B` è il numero di
immagini nel vassoio, e resta uguale per tutto il percorso. Si entra con
$(B, 3, 224, 224)$, cioè immagini a tre colori da 224 pixel di lato. La
convoluzione dà $(B, 768, 14, 14)$: la griglia si è ridotta a 14 per 14, che
sono i quadratini, e per ciascuno ci sono ora 768 numeri. La riga dopo
appiattisce quella griglia e scambia due assi, e ottiene $(B, 196, 768)$, cioè
196 quadratini in fila ($14 \times 14$), 768 numeri ciascuno. Poi si mette in
testa il quadratino in più e diventano $(B, 197, 768)$: è la forma che il
modello si porterà dietro identica per tutti e dodici i blocchi che seguono.

Nel codice le righe da guardare sono due: la convoluzione e il token di classe.

`````{tab} Elementare
**La convoluzione.** Del pezzo di codice `nn.Conv2d` basta sapere, per ora,
questo: fa scorrere una finestrella sull'immagine e a ogni posizione produce un
pugno di numeri a partire dai pixel che ci stanno dentro. Due manopole ne
governano il movimento: quanto è grande la finestra (`kernel_size`) e di quanto
si sposta a ogni scatto (`stride`). Il capitolo sul deep learning spiegherà
perché questa operazione sia il modo giusto di guardare un'immagine; qui serve
solo il movimento.

Il paper dice "si taglia l'immagine in quadratini e si proietta ciascuno". La
traduzione letterale sarebbe: taglia, impila, applica uno strato lineare (tre
operazioni). Ma una convoluzione con la finestra grande esattamente quanto il
passo fa già questo: scorre di sedici pixel alla volta con una finestra di
sedici, quindi guarda ogni quadratino una volta e nessun pixel due volte. Una
riga invece di tre, e più veloce. Accorgersi che
due descrizioni diverse sono la stessa operazione è metà del mestiere di chi
replica un paper.

**Il token di classe.** È un vettore in più, premesso ai 196 quadratini, che
non contiene nessun pezzo di immagine: è un foglio bianco. Attraversando la
rete, quel foglio raccoglie informazione da tutti gli altri, e alla fine è lì
che si va a leggere la risposta: un po' come il verbale di una riunione, che
non è uno dei partecipanti ma è dove finisce il senso di quello che si sono
detti.
`````

`````{tab} Superiore
**L'equivalenza.** Proiettare le patch appiattite significa calcolare
$\mathbf{x}_p^{i}\mathbf{E}$ con $\mathbf{E} \in \mathbb{R}^{(P^2C) \times D}$
per ogni $i$. Una `Conv2d` con `kernel_size = stride = P` calcola, per ogni
posizione non sovrapposta, il prodotto scalare tra la finestra e ciascuno dei
$D$ filtri: gli stessi $P^2C \cdot D$ moltiplicatori, riorganizzati. È
identica anche nei parametri (basta un `reshape` per passare da una forma
all'altra), ma delegata a un kernel ottimizzato invece che a `unfold` seguito
da `nn.Linear`, che materializza un tensore intermedio molto più grande.

**Token di classe e posizioni.** Entrambi sono `nn.Parameter`, cioè imparati:
$\mathbf{x}_{\text{class}}$ è la sonda da cui l'equazione 4 legge l'uscita, e
le codifiche di posizione sono *apprese*, non sinusoidali come nel Transformer
originale {cite}`vaswani2017attention` (dove gli autori avevano verificato che
i due tipi danno risultati quasi identici); l'ablazione del ViT confronta
invece varianti tutte apprese (1-D, 2-D, relative) e trova differenze
trascurabili. Due conseguenze pratiche. La prima: la lunghezza
di $\mathbf{E}_{\text{pos}}$ è legata alla risoluzione, quindi cambiare la
dimensione dell'immagine richiede di **interpolare** le codifiche, non basta
riallocarle. La seconda: l'alternativa al token di classe è il *global average
pooling* sui token delle patch, che funziona altrettanto bene ma richiede un
learning rate diverso; dettaglio che il paper riporta in appendice, ed
esattamente il tipo di nota che fa fallire una replica.
`````

Le equazioni 2 e 3 descrivono il blocco che poi si ripete dodici volte, e in
esse compaiono tre sigle e due parole che il libro spiegherà per esteso nel
capitolo sui Transformer. Qui bastano una riga a testa. **MSA** è l'attenzione
multi-testa, cioè la scatola in cui i quadratini si guardano fra loro e ognuno
raccoglie qualcosa dagli altri. **MLP** è una coppia di strati come quelli già
visti, che lavora su ogni posizione per conto suo. **LN** è la
*LayerNorm*, che rimette i numeri su una scala comoda prima di darli in pasto
alle altre due. *Pre-norm* vuol dire soltanto che quella rimessa in scala
avviene **prima** delle scatole e non dopo. E la **connessione residua** è il
`+ z` in fondo a ciascuna riga: quello che la scatola ha prodotto non
sostituisce l'ingresso, gli si somma, così il segnale originale ha sempre una
strada libera per arrivare in fondo.

$$
\begin{aligned}
\mathbf{z}'_{\ell} &= \text{MSA}\big(\text{LN}(\mathbf{z}_{\ell-1})\big) + \mathbf{z}_{\ell-1}, \\
\mathbf{z}_{\ell}  &= \text{MLP}\big(\text{LN}(\mathbf{z}'_{\ell})\big) + \mathbf{z}'_{\ell},
\end{aligned}
\qquad \ell = 1 \dots L
$$

e l'equazione 4 legge la risposta dal solo token di classe, dopo un'ultima
normalizzazione: $\mathbf{y} = \text{LN}(\mathbf{z}_L^0)$.

```python
class BloccoTransformer(nn.Module):
    """Equazioni 2 e 3: attenzione multi-testa e MLP, entrambe pre-norm."""

    def __init__(self, d_modello=768, teste=12, d_mlp=3072, dropout=0.1):
        super().__init__()
        self.norm1 = nn.LayerNorm(d_modello)
        self.attenzione = nn.MultiheadAttention(d_modello, teste,
                                                dropout=dropout, batch_first=True)
        self.norm2 = nn.LayerNorm(d_modello)
        self.mlp = nn.Sequential(
            nn.Linear(d_modello, d_mlp),
            nn.GELU(),                       # il paper usa GELU, non ReLU
            nn.Dropout(dropout),
            nn.Linear(d_mlp, d_modello),
            nn.Dropout(dropout),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h = self.norm1(x)
        x = x + self.attenzione(h, h, h, need_weights=False)[0]   # residuo 1
        x = x + self.mlp(self.norm2(x))                           # residuo 2
        return x
```

Due trappole in dieci righe, ed è normale. `nn.MultiheadAttention` restituisce
**due** cose insieme, il risultato e i pesi dell'attenzione: dimenticare il
`[0]` che tiene solo la prima produce un errore di tipo poco comprensibile. E
`batch_first=True` non è il default: senza, il modulo si aspetta i tre assi
nell'ordine (posizione, esempio, numeri) e non (esempio, posizione, numeri).
È un errore che *non* solleva eccezioni quando gli esempi del vassoio sono
tanti quante le posizioni della sequenza, e che quindi va messo lì e
dimenticato.

## Verificare senza addestrare

Ora arriva la parte che fa la differenza. La tabella 1 del paper dichiara, per
la variante **ViT-Base**: $12$ strati, dimensione nascosta $768$, dimensione
dell'MLP $3072$, $12$ teste di attenzione, **86 milioni** di parametri. I primi
quattro numeri li abbiamo copiati nel codice, e sono quindi ciò che abbiamo
dichiarato; il quinto no, il quinto **discende** dagli altri quattro, ed è per
questo che è l'unico che verifica davvero qualcosa. Si controlla in trenta
secondi, senza una GPU e senza dati.

```python
modello = nn.Sequential(
    IncorporazionePatch(),
    *[BloccoTransformer() for _ in range(12)],
)

finto = torch.randn(2, 3, 224, 224)               # due immagini finte
uscita = modello(finto)
print(uscita.shape)                                # torch.Size([2, 197, 768])

n_parametri = sum(p.numel() for p in modello.parameters() if p.requires_grad)
print(f"{n_parametri:,}")                          # 85,797,120
```

Il conto torna, e vale la pena rifarlo a mano una volta, perché è il tipo di
verifica che smaschera qualunque svista:

Nella prima riga i due $768$ sono i due numeri diversi di cui si diceva poco
fa: quello di sinistra è quanto entra nella proiezione ($16 \cdot 16 \cdot 3$,
i valori di una patch), quello di destra quanto ne esce (`d_modello`, la
scelta degli autori). Che siano uguali resta una coincidenza.

| Pezzo | Formula | Parametri |
|---|---|---|
| Proiezione delle patch | $768 \cdot 768 + 768$ | $590\,592$ |
| Token di classe | $768$ | $768$ |
| Codifiche di posizione | $197 \cdot 768$ | $151\,296$ |
| Attenzione, per blocco | $4 \cdot (768^2 + 768)$ | $2\,362\,368$ |
| MLP, per blocco | $2 \cdot 768 \cdot 3072 + 3072 + 768$ | $4\,722\,432$ |
| Due LayerNorm, per blocco | $2 \cdot 2 \cdot 768$ | $3\,072$ |
| **Un blocco intero** | $2\,362\,368 + 4\,722\,432 + 3\,072$ | $7\,087\,872$ |
| **12 blocchi** | $12 \cdot 7\,087\,872$ | $85\,054\,464$ |
| **Totale** | $590\,592 + 768 + 151\,296 + 85\,054\,464$ | $\mathbf{85\,797\,120}$ |

Due righe hanno un fattore che sembra piovere dall'alto, e vale la pena
scioglierlo perché il testo invita a rifare il conto a mano. Il **4**
dell'attenzione conta quattro proiezioni della stessa forma
$768 \times 768$ più bias: tre servono a produrre le tre versioni di ogni
elemento della sequenza che l'attenzione mette in gioco, la quarta a
ricomporre il risultato. Il **2** della LayerNorm è perché una
normalizzazione, dopo aver riportato i numeri su una scala standard, li
riscala di nuovo con due parametri imparati per canale, un moltiplicatore e
uno spostamento: due numeri per ciascuno dei $768$ canali, e i normalizzatori
per blocco sono due, da cui $2 \cdot 2 \cdot 768$.

Poco meno di $86$ milioni: è il numero che dichiara il paper. Il nostro conto,
però, si è fermato prima di due pezzi finali, e sono quelli che ci mancano per
arrivare a un modello completo. Il primo è la LayerNorm dell'equazione 4, che
vale $2 \cdot 768 = 1\,536$ con lo stesso conto di prima. Il secondo è la
**testa di classificazione**, cioè lo strato che dai 768 numeri del token di
classe ricava un punteggio per ciascuna delle $K$ classi: sono $768 \cdot K$
pesi, uno per ogni coppia canale-classe, più $K$ bias, uno per classe.

La verifica si può quindi portare fino in fondo su
`torchvision.models.vit_b_16`, che è lo stesso modello con $K = 1000$ classi:

$$
85\,797\,120 + 1\,536 + (768 \cdot 1000 + 1000) = 86\,567\,656,
$$

ed è esattamente il numero che quel modello riporta, fino all'ultima cifra.
Vale la pena notare che il $+K$ in coda non è un dettaglio decorativo:
scordarsi i mille bias della testa farebbe chiudere il conto mille parametri
sotto, e in una verifica che si vanta di essere esatta all'unità mille
parametri si vedono.

Se invece il nostro conteggio fosse uscito attorno ai $43$ milioni sapremmo,
senza dover fare ipotesi, di aver usato sei blocchi invece di dodici; se fosse
uscito $170$ milioni, di aver raddoppiato qualcosa. È una verifica che costa
niente e che quasi nessuno fa.

Lo stesso controllo, strato per strato, lo dà `torchinfo`:

```python
from torchinfo import summary
summary(modello, input_size=(1, 3, 224, 224),
        col_names=["input_size", "output_size", "num_params"])
```

Quello che stampa è una tabella, una riga per strato, con la forma in ingresso,
la forma in uscita e quanti parametri quel pezzo si porta dietro; in fondo, la
somma. Serve per due cose: quando la catena si spezza, per vedere in quale riga
la forma smette di combaciare, e quando il totale non torna, per capire su
quale blocco è andato perso. Sul nostro modello l'ultima riga dice
`Total params: 85,797,120`, cioè lo stesso numero di due righe fa, ma stavolta
con davanti il dettaglio di dove sta ciascun pezzo.

## Quando i numeri non tornano

Architettura verificata, e poi? Qui comincia il territorio onesto. Riprodurre
la *struttura* di un paper è alla portata di chiunque; riprodurne i **risultati**
spesso non lo è, e non per colpa di chi ci prova.

Il ViT è un caso esemplare proprio in questo. La tesi dell'articolo è che
l'architettura raggiunge o supera le reti convoluzionali (le **CNN**, la
famiglia di modelli per immagini del capitolo sul deep learning) **solo dopo**
essere stata addestrata una prima volta su quantità di dati enormi, e solo
allora rifinita sul compito che interessa: è quello che si chiama
*pre-addestramento*. Nel paper quelle quantità sono ImageNet-21k, o il
JFT-300M interno a Google, trecento milioni di immagini mai rese pubbliche.
Addestrato da zero sul solo ImageNet-1k, lo stesso identico codice dà risultati
mediocri, e questo è un *risultato* del paper, non un fallimento della replica.

Sapere in anticipo che la riproduzione completa è impossibile cambia
l'obiettivo, e in meglio: si replica l'architettura, la si verifica scaricando
i pesi che gli autori hanno pubblicato, e si addestra su un problema alla
propria portata partendo da quei pesi invece che da zero. Quest'ultima mossa si
chiama *transfer learning*, ed è l'argomento del capitolo sulla visione
artificiale.

Quando invece i numeri dovrebbero tornare e non tornano, la lista dei sospetti
è quasi sempre questa, in ordine di frequenza:

1. **I dati e le trasformazioni.** Ritaglio, risoluzione, statistiche di
   normalizzazione, augmentation: sono la prima causa, e spesso descritte in
   una riga di appendice.
2. **Il programma del learning rate.** Warmup lineare, decadimento a coseno,
   valore di picco che dipende dalla dimensione del batch. Un paper che dice
   solo "lr $= 10^{-3}$" ne sta omettendo metà.
3. **La dimensione del batch e l'accumulo.** Chi ha 8 GPU e chi ne ha una non
   stanno addestrando lo stesso modello, a meno di accumulare i gradienti.
4. **I freni**, cioè tutto quello che si mette apposta per rendere la vita più
   difficile al modello mentre impara. Ce n'è una famiglia intera, con nomi che
   il libro incontrerà più avanti (weight decay, dropout, *label smoothing*),
   e un paper che ne omette uno solo è già un altro esperimento.
5. **L'inizializzazione**, cioè da quali numeri partono i pesi, quando non è
   quella che la libreria mette di suo.
6. **Il protocollo di valutazione**: su quale porzione di dati si misura, in
   quanti modi si ritaglia ogni immagine di prova, e se il numero riportato è
   il migliore ottenuto o l'ultimo.
7. **Il caso**: il seme, e quanti semi sono stati provati.

`````{tab} Elementare
Di questa lista, i primi due punti spiegano da soli la maggior parte dei casi.
Il primo perché la preparazione dei dati è quasi sempre raccontata di fretta,
in mezza riga d'appendice, mentre cambia il risultato molto più di
un'architettura. Il secondo perché il learning rate quasi mai è un numero
fisso: sale piano all'inizio (il *riscaldamento*) e poi scende lungo
l'addestramento, e chi legge solo il valore di picco sta copiando un terzo
dell'informazione.

C'è poi una regola che salva molte replicazioni: **se hai una GPU sola e il
paper ne usava otto, non stai addestrando lo stesso modello**. Il numero di
esempi visti a ogni aggiornamento è diverso, e con esso il rumore del
gradiente. Si rimedia accumulando i gradienti per più batch prima di
aggiornare: un modo di fingere un batch grande su una macchina piccola.
`````

`````{tab} Superiore
Sui due punti principali, in dettaglio.

**Programma del learning rate.** La forma quasi universale è warmup lineare
per $T_w$ passi seguito da decadimento a coseno fino a zero. Il valore di
picco non è trasferibile tra batch di dimensione diversa: la *linear scaling
rule* prescrive $\eta \propto B$ (con warmup, per evitare l'instabilità
iniziale) per SGD; con Adam e AdamW la dipendenza empirica è più vicina a
$\eta \propto \sqrt{B}$. Un paper che riporta solo $\eta$ senza $B$, warmup e
schedule non è replicabile alla lettera.

**Accumulo dei gradienti.** Il batch efficace è
$B_{\text{eff}} = B_{\text{micro}} \times k \times n_{\text{GPU}}$, dove $k$
sono i passi di accumulo: si eseguono $k$ `backward()` e un solo
`optimizer.step()`, ricordando di dividere la loss per $k$ se la riduzione è
la media. Non è del tutto equivalente a un batch grande vero: le statistiche
della batch normalization restano calcolate sul micro-batch, ragione per cui i
lavori che scalano molto preferiscono LayerNorm o GroupNorm.

**Weight decay.** Va tipicamente escluso da bias e parametri di
normalizzazione: applicarlo a tutto è un errore silenzioso che costa qualche
punto. In PyTorch si realizza passando a AdamW due *parameter group* distinti,
uno con `weight_decay=0`.

**Protocollo di valutazione.** Se il paper usa una media esponenziale dei pesi
(EMA), o *test-time augmentation*, o riporta la metrica migliore sulla
validazione invece dell'ultima, confrontarsi con l'addestramento nudo dà una
differenza sistematica che non ha nulla a che vedere con l'architettura.
`````

La disciplina è la stessa della sezione sul
[flusso di lavoro](flusso-di-lavoro.md): si cambia **un sospetto alla volta** e
si registra. Ed è utile sapere che il problema è riconosciuto e studiato: la
comunità ha risposto con i *reproducibility checklist* adottati dalle grandi
conferenze, che chiedono agli autori di dichiarare esattamente questi punti
{cite}`pineau2021improving`.

### Il diario, che è la metà che nessuno scrive

«Si registra» merita più di due parole, perché è la pratica che distingue tre
giorni di lavoro da tre giorni di lavoro buttati. La forma minima è una riga
per esperimento, scritta **prima** di lanciarlo:

> *Esperimento 7. Ipotesi: la differenza viene dal warmup, che nel paper è di
> 10k passi e nel mio di 500. Cambio solo quello. Mi aspetto che la loss
> iniziale smetta di impennarsi. Esito: …*

Tre proprietà rendono utile questo rito, e tolta una delle tre non funziona più.
**Una cosa alla volta**, altrimenti il risultato non attribuisce il merito a
nessuno dei due cambi. **L'ipotesi prima del risultato**, perché scritta dopo si
adatta sempre a ciò che è successo, e si finisce per credere di aver capito. E
soprattutto **si annota anche quello che non ha funzionato**: è la metà che
nessuno scrive, ed è l'unica che impedisce di riprovare fra due settimane la
stessa cosa senza ricordarsene.

Gli strumenti che tracciano gli esperimenti (nel capitolo su MLOps) rendono
tutto questo cercabile e condivisibile, e non c'è ragione di non usarli. Ma
registrano bene i **parametri** e i **numeri**, e non registrano l'unica cosa
che non si può ricostruire dopo: **perché** si era provato. Quella va scritta a
mano.

```{admonition} Onestà intellettuale
:class: note
"Non sono riuscito a riprodurre il risultato" è un esito legittimo, e vale la
pena scriverlo: quanto ci si è avvicinati, che cosa si è provato, che cosa
mancava. Una replica fallita e documentata è informazione utile per tutti; una
replica dichiarata riuscita senza esserlo, no. Vale anche per il proprio
lavoro: se un risultato dipende da un seme fortunato, non è un risultato.
```

Il metodo vale più del caso su cui l'abbiamo provato: si applica identico a
qualunque articolo che dichiari un'architettura e dei numeri.

`````{tab} Elementare
```{admonition} Da ricordare
:class: important
- Replicare un articolo è il modo più affidabile di capirlo: il codice non
  accetta i passaggi vaghi. Dove il testo dice «si proietta», il codice deve
  dire con che cosa e di che misura.
- Le mosse sono quattro: **inventario** dei pezzi, **una formula alla volta**,
  **metro da falegname** a ogni passo (mandi dentro un dato finto e guardi che
  forma esce), **conteggio dei pezzi** alla fine.
- Nessuna delle quattro richiede di addestrare niente, e nessuna richiede di
  aver capito che cosa fanno i pezzi dentro: bastano le misure in entrata e in
  uscita.
- Il conteggio dei pezzi è la verifica più potente e costa trenta secondi: se
  l'articolo dice 86 milioni e a te ne escono 43, ne hai montata metà.
- Riprodurre **il montaggio** è quasi sempre possibile; riprodurre **i
  risultati** spesso no, perché mancano i dati o metà delle istruzioni. Dirlo
  è parte del lavoro, non un'ammissione di sconfitta.
```
`````

`````{tab} Superiore
```{admonition} Da ricordare
:class: important
- Replicare un paper è il modo più affidabile di capirlo: il codice non tollera
  i passaggi vaghi.
- Il metodo ha quattro mosse: **inventario** dei pezzi, **un'equazione alla
  volta**, **controllo delle forme** a ogni passo, **conteggio dei parametri**
  alla fine.
- I tre **invarianti** (cioè le proprietà che devono valere comunque, prima di
  qualunque addestramento) si verificano a costo zero: forma in uscita, numero
  di parametri, presenza di gradiente su ogni parametro dopo un `backward()`.
- Nel ViT, una `Conv2d` con `kernel_size = stride = patch` *è* la proiezione
  lineare delle patch: riconoscere queste equivalenze fa parte del mestiere.
- ViT-Base ha $85\,797\,120$ parametri senza LayerNorm finale né testa (la
  testa ne aggiunge $768 \cdot K + K$): il conto si rifà a mano e smaschera
  qualunque svista strutturale.
- Riprodurre l'**architettura** è quasi sempre possibile; riprodurre i
  **risultati** spesso no: dati non pubblici, iperparametri omessi, hardware
  diverso. Dirlo è parte del lavoro.
```
`````
