# Replicare un paper

Un articolo scientifico si può leggere in tre modi. Il primo è scorrerlo:
mezz'ora, si ricava l'idea generale e si dimentica in una settimana. Il secondo
è studiarlo: si seguono le derivazioni, si capisce l'argomento. Il terzo è
**farlo girare** — trasformare le equazioni in `nn.Module`, mandare avanti un
tensore e guardare se esce quello che deve uscire. È l'unico modo che non
consente di autoingannarsi, perché il codice non accetta i passaggi vaghi: dove
il testo dice "si proietta linearmente" bisogna decidere una matrice, e la
matrice ha una forma precisa.

Questa sezione non introduce nulla di nuovo di PyTorch: usa quello che il
capitolo ha già costruito. Introduce un **metodo**, che è la cosa più
trasferibile che si possa imparare qui dentro.

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
   $f: \mathbb{R}^{d_{in}} \to \mathbb{R}^{d_{out}}$; se ne verifica il tipo
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
numerici — ed è a quel punto che comincia la parte difficile.
`````

## Il caso: il Vision Transformer

Prendiamo un articolo che il libro ha già incontrato più volte: *An Image is
Worth 16x16 Words* {cite}`dosovitskiy2021image`, che nel 2021 ha portato
l'architettura Transformer dentro la visione artificiale. È un ottimo caso di
studio perché l'architettura è breve — quattro equazioni — e perché i numeri da
verificare sono pubblicati.

La prima equazione del paper costruisce la sequenza di ingresso: l'immagine
viene tagliata in quadratini, ognuno viene proiettato in un vettore, si
aggiunge un token speciale e le posizioni.

$$
\mathbf{z}_0 = [\, \mathbf{x}_{\text{class}} ;\;
\mathbf{x}_p^{1}\mathbf{E} ;\; \mathbf{x}_p^{2}\mathbf{E} ;\; \dots ;\;
\mathbf{x}_p^{N}\mathbf{E} \,] + \mathbf{E}_{\text{pos}}
$$

dove $\mathbf{x}_p^{i}$ è la $i$-esima patch appiattita, $\mathbf{E} \in
\mathbb{R}^{(P^2 \cdot C) \times D}$ è la proiezione lineare condivisa,
$\mathbf{x}_{\text{class}}$ è un vettore imparabile premesso alla sequenza e
$\mathbf{E}_{\text{pos}} \in \mathbb{R}^{(N+1) \times D}$ sono le codifiche di
posizione. Con immagini $224 \times 224$, patch $P = 16$ e $C = 3$ canali si
ottengono $N = (224/16)^2 = 196$ patch, ciascuna di $16 \cdot 16 \cdot 3 = 768$
numeri.

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

Le righe da guardare sono due: la convoluzione e il token di classe.

`````{tab} Elementare
**La convoluzione.** Il paper dice "si taglia l'immagine in quadratini e si
proietta ciascuno". La traduzione letterale sarebbe: taglia, impila, applica
uno strato lineare — tre operazioni. Ma una convoluzione con la finestra grande
esattamente quanto il passo fa già questo: scorre di sedici pixel alla volta
con una finestra di sedici, quindi guarda ogni quadratino una volta e nessun
pixel due volte. Una riga invece di tre, e più veloce. Accorgersi che due
descrizioni diverse sono la stessa operazione è metà del mestiere di chi
replica un paper.

**Il token di classe.** È un vettore in più, premesso ai 196 quadratini, che
non contiene nessun pezzo di immagine: è un foglio bianco. Attraversando la
rete, quel foglio raccoglie informazione da tutti gli altri, e alla fine è lì
che si va a leggere la risposta — un po' come il verbale di una riunione, che
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
le codifiche di posizione sono *apprese* (non sinusoidali come nel Transformer
originale {cite}`vaswani2017attention`) — il ViT verifica sperimentalmente che
la differenza è trascurabile. Due conseguenze pratiche. La prima: la lunghezza
di $\mathbf{E}_{\text{pos}}$ è legata alla risoluzione, quindi cambiare la
dimensione dell'immagine richiede di **interpolare** le codifiche, non basta
riallocarle. La seconda: l'alternativa al token di classe è il *global average
pooling* sui token delle patch, che funziona altrettanto bene ma richiede un
learning rate diverso — dettaglio che il paper riporta in appendice, ed
esattamente il tipo di nota che fa fallire una replica.
`````

Le equazioni 2 e 3 sono il blocco Transformer, con la normalizzazione **prima**
dei sottoblocchi (*pre-norm*) e due connessioni residue:

$$
\begin{align*}
\mathbf{z}'_{\ell} &= \text{MSA}\big(\text{LN}(\mathbf{z}_{\ell-1})\big) + \mathbf{z}_{\ell-1}, \\
\mathbf{z}_{\ell}  &= \text{MLP}\big(\text{LN}(\mathbf{z}'_{\ell})\big) + \mathbf{z}'_{\ell},
\end{align*}
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
una **tupla** (output, pesi): dimenticare `[0]` produce un errore di tipo poco
comprensibile. E `batch_first=True` non è il default: senza, il modulo si
aspetta tensori $(L, B, D)$ e non $(B, L, D)$ — un errore che *non* solleva
eccezioni se $B$ e $L$ per caso coincidono, e che quindi va messo lì e
dimenticato.

## Verificare senza addestrare

Ora arriva la parte che fa la differenza. La tabella 1 del paper dichiara, per
la variante **ViT-Base**: $L = 12$ strati, dimensione nascosta $D = 768$,
dimensione dell'MLP $3072$, $12$ teste di attenzione, **86 milioni** di
parametri. Tutti e cinque i numeri si controllano in trenta secondi, senza una
GPU e senza dati.

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

| Pezzo | Formula | Parametri |
|---|---|---|
| Proiezione delle patch | $768 \cdot 768 + 768$ | $590\,592$ |
| Token di classe | $768$ | $768$ |
| Codifiche di posizione | $197 \cdot 768$ | $151\,296$ |
| Attenzione, per blocco | $4 \cdot (768^2 + 768)$ | $2\,362\,368$ |
| MLP, per blocco | $2 \cdot 768 \cdot 3072 + 3072 + 768$ | $4\,722\,432$ |
| Due LayerNorm, per blocco | $2 \cdot 2 \cdot 768$ | $3\,072$ |
| **12 blocchi** | $12 \cdot 7\,087\,872$ | $85\,054\,464$ |
| **Totale** | | $\mathbf{85\,797\,120}$ |

Poco meno di $86$ milioni: il numero dichiarato dal paper, che qui non
comprende né la LayerNorm finale dell'equazione 4 né la testa di
classificazione (altri $768 \cdot K$ parametri, con $K$ le classi). Se il
nostro conteggio fosse uscito attorno ai $43$ milioni sapremmo, senza ipotesi,
di aver usato sei blocchi invece di dodici; se fosse uscito $170$ milioni, di
aver raddoppiato qualcosa. È una verifica che costa niente e che
quasi nessuno fa.

Lo stesso controllo, strato per strato, lo dà `torchinfo`:

```python
from torchinfo import summary
summary(modello, input_size=(1, 3, 224, 224),
        col_names=["input_size", "output_size", "num_params"])
```

## Quando i numeri non tornano

Architettura verificata, e poi? Qui comincia il territorio onesto. Riprodurre
la *struttura* di un paper è alla portata di chiunque; riprodurne i **risultati**
spesso non lo è, e non per colpa di chi ci prova.

Il ViT è un caso esemplare proprio in questo. La tesi dell'articolo è che
l'architettura raggiunge o supera le CNN **solo dopo** un pre-addestramento su
grandi quantità di dati: ImageNet-21k, o il JFT-300M interno a Google — 300
milioni di immagini, mai reso pubblico. Addestrato da zero sul solo ImageNet-1k,
lo stesso identico codice dà risultati mediocri, e questo è un *risultato* del
paper, non un fallimento della replica. Sapere in anticipo che la
riproduzione completa è impossibile cambia l'obiettivo: si replica
l'architettura, si verifica sui pesi pubblicati, e si addestra su un problema
alla propria portata usando il *transfer learning*.

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
4. **Regolarizzazione**: weight decay (e su quali parametri — di solito non su
   bias e LayerNorm), dropout, *stochastic depth*, *label smoothing*, clipping.
5. **L'inizializzazione**, quando non è quella di default.
6. **Il protocollo di valutazione**: quale split, quante *crop* in test, se la
   metrica riportata è la migliore o l'ultima.
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
aggiornare — un modo di fingere un batch grande su una macchina piccola.
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
la media. Non è del tutto equivalente a un batch grande vero — le statistiche
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

```{admonition} Onestà intellettuale
:class: note
"Non sono riuscito a riprodurre il risultato" è un esito legittimo, e vale la
pena scriverlo: quanto ci si è avvicinati, che cosa si è provato, che cosa
mancava. Una replica fallita e documentata è informazione utile per tutti; una
replica dichiarata riuscita senza esserlo, no. Vale anche per il proprio
lavoro: se un risultato dipende da un seme fortunato, non è un risultato.
```

```{admonition} Da ricordare
:class: important
- Replicare un paper è il modo più affidabile di capirlo: il codice non tollera
  i passaggi vaghi.
- Il metodo ha quattro mosse: **inventario** dei pezzi, **un'equazione alla
  volta**, **controllo delle forme** a ogni passo, **conteggio dei parametri**
  alla fine.
- I tre invarianti si verificano **senza addestrare**: forma in uscita, numero
  di parametri, presenza di gradiente su ogni parametro.
- Nel ViT, una `Conv2d` con `kernel_size = stride = patch` *è* la proiezione
  lineare delle patch: riconoscere queste equivalenze fa parte del mestiere.
- ViT-Base ha $85\,797\,120$ parametri senza LayerNorm finale né testa: il
  conto si rifà a mano e smaschera qualunque svista strutturale.
- Riprodurre l'**architettura** è quasi sempre possibile; riprodurre i
  **risultati** spesso no — dati non pubblici, iperparametri omessi, hardware
  diverso. Dirlo è parte del lavoro.
```
