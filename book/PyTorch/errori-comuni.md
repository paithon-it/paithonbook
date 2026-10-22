# I tre errori più comuni

Esiste un rito d'iniziazione che accomuna chiunque abbia scritto codice
PyTorch, dal primo esercizio al laboratorio di ricerca: un `RuntimeError`
lungo venti righe, la sensazione che il messaggio sia scritto in una lingua
straniera, e mezz'ora persa a spostare `.to(device)` da una riga all'altra
sperando che smetta. La buona notizia è che quel messaggio quasi mai è nuovo.
Sotto la varietà apparente, gran parte dei messaggi riguarda tre cose:
la forma dei dati, il loro tipo, il dispositivo su cui stanno.

La ragione è che, quando due tensori si incontrano, PyTorch controlla tre loro
attributi: la forma (`shape`), il tipo dei numeri (`dtype`) e il dispositivo
(`device`). Dove una regola lo permette li adatta (allarga una forma col
*broadcasting*, porta due tipi a uno comune); dove non può si ferma. Quasi
sempre succede perché una di queste tre regole non è rispettata, e per questo
chi impara a riconoscerle a colpo d'occhio smette di cercare l'errore per
tentativi. Restano fuori gli errori sui valori, come un'etichetta che indica
una classe che non esiste (`IndexError: Target 10 is out of bounds`), e quelli
di risorse, come la memoria della scheda grafica che finisce (`CUDA out of
memory`).

```python
import torch

x = torch.randn(32, 3, 224, 224)
print(x.shape)    # torch.Size([32, 3, 224, 224])  -> la forma
print(x.dtype)    # torch.float32                   -> il tipo
print(x.device)   # cpu                             -> dove abita
```

Questa terna è la prima cosa da stampare quando qualcosa non va.

## 1. La forma non torna

```text
RuntimeError: mat1 and mat2 shapes cannot be multiplied (32x784 and 128x10)
```

Lo si incontra presto, e il messaggio (una volta imparato a leggerlo) dice già
tutto: PyTorch ha provato a moltiplicare una matrice $32 \times 784$
per una $128 \times 10$, e non si può, perché il numero di colonne della prima
($784$) non coincide con il numero di righe della seconda ($128$). È la regola
del prodotto fra matrici vista nella {doc}`sezione di algebra
lineare </Matematica/algebra-lineare>`, e il motivo per cui esiste è che ogni
riga della prima matrice viene accoppiata a una colonna della seconda, numero
per numero, e se le due file non hanno la stessa lunghezza
gli accoppiamenti non si chiudono.

I numeri dicono anche *dove* è successo. Il $32$ è il numero di esempi nel
mini-batch (nel codice, *batch*); il $784$ sono i pixel di un'immagine MNIST
srotolata, $28 \times 28$. Il $128$ e il $10$ sono la seconda matrice, cioè lo
strato d'uscita del nostro percettrone, che prende 128 numeri e ne produce 10,
uno per cifra. In altri termini, sono arrivate 32 immagini appiattite
direttamente allo strato finale, perché fra i due manca lo strato nascosto, che
avrebbe dovuto portarle da $784$ a $128$.

`````{tab} Elementare
Un tensore porta scritte addosso le sue dimensioni, che dicono quante cose
contiene e come sono disposte. Ad esempio, $(3, 224, 224)$ è un'immagine: tre
tavole di $224 \times 224$ numeri, una per colore, rosso verde e blu. Ogni
numero è un lato del blocco, e due pezzi si agganciano quando le dimensioni
combaciano sul lato giusto, come due tubi che si avvitano quando hanno lo
stesso diametro. Con una differenza che conta: due tubi sbagliati semplicemente
non si avvitano, mentre due dimensioni sbagliate a volte si agganciano lo
stesso, senza avvisare, e danno un numero privo di significato. Le cause
ricorrenti hanno un nome ciascuna.

**Manca l'appiattimento.** Uno strato `nn.Linear` legge un numero solo delle
dimensioni, l'ultimo, e degli altri non si occupa. Davanti a
$(3, 224, 224)$ guarda il $224$ finale. Se ne aspettava un altro si ferma; se
per caso aspettava $224$ va perfino peggio, perché lo strato calcola e nessuno
protesta: tratta ogni riga di pixel di ogni colore come un esempio a sé, e di
un'immagine sola ne fa $3 \times 224 = 672$ risposte scollegate invece di una.

Serve un `nn.Flatten()` in mezzo, che srotola ogni immagine in un'unica fila di
$3 \times 224 \times 224 = 150\,528$ numeri e lascia stare il mazzetto,
perché srotola le immagini una per una senza impastarle fra loro. Lo strato che
viene dopo va allora costruito per accettarne $150\,528$.

**Due strati non si parlano.** L'uscita di uno strato deve essere l'ingresso
del successivo; se il primo produce $128$ numeri, il secondo deve aspettarsene
$128$, non $256$. Quando quel numero l'abbiamo scritto noi si tratta di un
refuso, e si corregge guardando la riga sopra. Spesso però non lo si conosce
affatto, perché esce da una catena di passaggi che lo cambiano ognuno un po’.
In quel caso c'è `nn.LazyLinear`, che misura il diametro del tubo alla prima
chiamata, quando gli passa qualcosa, e su quella misura costruisce il pezzo.
È comodo mentre si prova, ma nella versione definitiva quel numero è meglio
scriverlo.

**Manca il lato del mazzetto.** Il modello si aspetta un mazzetto di esempi,
anche quando l'esempio è uno solo. Un'immagine singola va quindi portata da
$(3, 224, 224)$ a $(1, 3, 224, 224)$, con `x.unsqueeze(dim=0)`, cioè "un
mazzetto che contiene un'immagine". È l'inciampo classico del momento in cui
si prova il modello su una foto scaricata al volo.

L'operazione inversa ha una trappola: `x.squeeze()`, senza dire quale lato,
toglie tutti i lati che valgono uno, e su un mazzetto da un esempio solo porta
via anche quello, riportando l'immagine com'era prima. Il lato si indica
sempre, con `x.squeeze(dim=...)`, così quello del mazzetto resta dov'è.

Per trovare il punto in cui la fila si spezza non serve indovinare: si fanno
stampare le dimensioni a ogni passaggio, e la prima coppia che non combacia è
quella da riparare.
`````

`````{tab} Superiore
Bisogna tracciare le dimensioni lungo la rete, e le tre cause si
formalizzano così.

**Appiattimento.** `nn.Linear(d_in, d_out)` opera sull’ultima dimensione e
lascia intatte le precedenti: applicato a $(B, C, H, W)$ fallisce a meno che
$W = d_{\text{in}}$. Per un MLP su immagini serve `nn.Flatten()` (che per
default appiattisce da `start_dim=1`, preservando il batch) e
$d_{\text{in}} = C \cdot H \cdot W$.

**Composizione.** In una `nn.Sequential`, `out_features` di uno strato deve
uguagliare `in_features` del successivo. La dimensione, però, a volte dipende
da calcoli, come negli strati che fanno scorrere un filtro sull'immagine,
dove `stride` e `padding` cambiano la forma dell'uscita (si veda la
{doc}`sezione sulle reti convoluzionali
</DeepLearning/reti-convoluzionali>`). Per un filtro di lato $k$
(`kernel_size`), con passo $s$ (`stride`) e riempimento $p$ (`padding`), la
lunghezza d'uscita lungo un asse di lunghezza $n$ è
$\lfloor (n + 2p - k)/s \rfloor + 1$, con la dilatazione di default
(`dilation=1`); ad esempio, con $n = 10$, $k = 3$, $s = 2$, $p = 1$ viene $5$.
Il calcolo si può fare a mano, oppure si lascia fare: `nn.LazyLinear(d_out)`
deduce la dimensione dal primo tensore che riceve, materializzando i pesi alla
prima chiamata. È comodo in fase esplorativa, ma nel codice definitivo è meglio
fissare il numero.

**Dimensione di batch.** Quasi tutti i moduli assumono la convenzione
*batch-first* $(B, \dots)$, con un'eccezione insidiosa perché non solleva
eccezioni: `nn.RNN`, `nn.LSTM`, `nn.GRU`, `nn.MultiheadAttention` e i moduli
`nn.Transformer*` hanno `batch_first=False` di default, cioè si aspettano
$(L, B, \dots)$, con $L$ la lunghezza della sequenza. Per aggiungere la
dimensione di batch a un esempio singolo si usa `unsqueeze(0)` nel caso comune
e `unsqueeze(1)` in quei moduli, che però accettano anche l'ingresso senza
batch, $(L, d_{\text{in}})$. Quanto all'operazione inversa, `squeeze()` senza
argomento elimina tutte le dimensioni unitarie e, su un batch da un elemento,
cancella anche quella del batch; per questo si passa sempre `dim` esplicito.

Lo strumento di diagnosi è il pacchetto `torchinfo`, da installare a parte
(`pip install torchinfo`). Dopo `from torchinfo import summary`, la chiamata
`summary(modello, input_size=(32, 3, 224, 224), col_names=("input_size",
"output_size", "num_params"))` stampa forma in ingresso, forma in uscita e
parametri di ogni strato (senza `col_names` la forma in ingresso non c'è), e
mostra a colpo d'occhio dove la catena si spezza.

Senza dipendenze esterne, la catena delle forme si prova sul dispositivo
`meta`, dove i tensori hanno forma e tipo ma nessun numero, e quindi non
occupano memoria:

```python
from torch import nn

with torch.device("meta"):                   # forme e tipi, nessun numero
    rete = nn.Sequential(nn.Flatten(), nn.Linear(28 * 28, 128), nn.ReLU(),
                         nn.Linear(128, 10))
    print(rete(torch.empty(32, 1, 28, 28)).shape)
```

```text
torch.Size([32, 10])
```

In alternativa basta un `print(x.shape)` in ogni riga del `forward`, poco
elegante ma sempre efficace.
`````

## 2. Il tipo non torna

```text
RuntimeError: mat1 and mat2 must have the same dtype, but got Byte and Float
RuntimeError: expected target dtype to be Long or Byte, but got Float
```

Il secondo errore riguarda il `dtype`, cioè il tipo di numero con cui il
tensore è scritto: interi (il $3$) oppure decimali (il $3{,}0$). Quando somma o
moltiplica due tensori numero per numero, PyTorch li converte a un tipo
comune (un intero e un decimale, per esempio, diventano due decimali, anche
a costo di qualche cifra: `int64` e `float32` danno `float32`). Ma nel
prodotto fra matrici e dentro gli strati `nn` (i casi dei due messaggi) la
libreria richiede un tipo preciso, e preferisce fermarsi piuttosto che
indovinare quale volevamo.

I due messaggi vanno letti in modi diversi. Nel primo, i due tipi sono i due
operandi nell'ordine, cioè il tensore che abbiamo passato e i pesi del modello,
e non "atteso" e "trovato". `Byte` è il nome interno di `uint8`,
cioè interi da $0$ a $255$, il tipo di un'immagine appena letta da un file, e
`Float` è il tipo dei pesi. In altre parole, la frase dice che un operando
intero e uno decimale si sono incontrati in una moltiplicazione fra matrici.

Il secondo messaggio riguarda le etichette, e lì i nomi sono quelli dei tipi
ammessi (`Long`, cioè `int64`, interi lunghi) contro quello ricevuto.

La stessa situazione produce frasi diverse a seconda
dell'operazione. Uno strato lineare dice `mat1 and mat2 must have the same
dtype`. Uno strato convoluzionale, che fa scorrere un piccolo filtro
sull'immagine (spiegato nella {doc}`sezione sulle reti convoluzionali
</DeepLearning/reti-convoluzionali>`), dice `Input type (unsigned char) and
bias type (float) should be the same` se ha il bias, e `expected scalar type
Byte but found Float` se non ce l'ha. Sono tre modi di dire la stessa cosa, e
riconoscerne il tema è più utile che impararli a memoria.

`````{tab} Elementare
Un'immagine appena letta da un file è fatta di numeri interi da $0$ a $255$: è
il tipo `uint8`, il "Byte" del messaggio. Una rete neurale lavora invece con
numeri decimali, e li vuole in una scala piccola, di solito fra $0$ e $1$ in
ingresso. In una somma PyTorch fa la conversione da
sé, ma la rete no: uno strato si aspetta il tipo preciso, e la
conversione è a carico nostro.

La fa la trasformazione `ToTensor()`, che converte *e* divide per $255$
portando i valori nell'intervallo giusto. Se la si fa a mano servono tutti e
due i passaggi, `.float() / 255`, e il secondo si dimentica
spesso, con l'effetto di dare alla rete numeri $255$ volte più
grandi di quelli che si aspetta.

C'è una seconda coppia che non si incontra, e stavolta sono due decimali. I
numeri con la virgola si scrivono in due misure, una lunga e una corta, cioè
con più cifre dopo la virgola o con meno, e strumenti diversi non scelgono la
stessa. Ad esempio, un foglio di dati letto con NumPy, la libreria di Python
per il calcolo numerico, arriva nella misura lunga, mentre i pesi della rete
sono scritti in quella corta. Dentro uno strato le due misure non si incontrano
e la libreria si ferma, anche se i numeri sono giusti e sono decimali tutti e
due.

Il messaggio nomina le due misure con i nomi che hanno in inglese, `Double` per
la lunga e `Float` per la corta. È il caso che fa perdere più tempo, perché la
riga da guardare sta fuori da PyTorch, molte righe più su, dove i dati sono
stati letti. Il rimedio è lo stesso comando di prima, `.float()`, scritto subito
dopo la lettura.

Il secondo messaggio è il caso opposto e riguarda le etichette. Alla
`CrossEntropyLoss` le classi vere si danno come numeri interi (la classe $3$,
non $3{,}0$), perché sono nomi, non quantità. Passare $3{,}0$ produce
quell'errore. Alla `BCEWithLogitsLoss`, invece, servono decimali,
perché lì l'etichetta è un numero fra zero e uno: quasi sempre è uno dei due
estremi, «no» oppure «sì», ma la porta è aperta anche ai valori in mezzo, che
servono quando l'etichetta è incerta.

La regola pratica è una sola conversione, il più presto possibile, cioè quando
i dati entrano e non a metà del training loop.
`````

`````{tab} Superiore
Il default di PyTorch è `float32`, mentre quello di NumPy è `float64`. Un
`torch.from_numpy(array)` conserva il `float64` e produce un tensore che non
può essere moltiplicato per i pesi `float32` del modello (`mat1 and mat2 must
have the same dtype, but got Double and Float`, dove `Double` è `float64`). È
la sorgente più insidiosa di errori di tipo, perché nasce fuori da PyTorch.
La conversione esplicita `torch.from_numpy(a).float()` (o `.to(torch.float32)`)
va quindi fatta al confine.

Sul *type promotion* elemento per elemento la formula che si legge di solito
(«le stesse regole di NumPy») è vera solo in parte. Le due librerie hanno in
comune otto tipi numerici di uso quotidiano (`uint8`, i quattro interi con
segno e i tre decimali, escluso il `bfloat16` che in NumPy non esiste), per un
totale di $8^2 = 64$ coppie possibili. Se si confronta `torch.result_type` con
`np.result_type` su ciascuna di esse, dieci divergono, e sono tutte del tipo
intero per decimale: a differenza di NumPy, in PyTorch un intero non spinge mai
un decimale a una precisione più alta.

Ad esempio, `int64 + float32` dà `float32` in torch
e `float64` in NumPy; `torch.tensor([1,2,3]) * 2.5` dà `float32`, mentre
`np.array([1,2,3]) * 2.5` dà `float64`. Questo rende il `float64` in arrivo da
NumPy ancora più insidioso, perché le due librerie adottano regole diverse.

Il quadro completo dei tipi che si incontrano:

| Contenuto | dtype atteso | Note |
|---|---|---|
| Immagini, feature, attivazioni | `torch.float32` | il default dei pesi |
| Etichette per `CrossEntropyLoss` | `torch.int64` (`long`) | indici di classe, shape $(B,)$; oppure `float32` di shape $(B,K)$, cioè probabilità |
| Etichette per `BCEWithLogitsLoss` | `torch.float32` | shape uguale ai logit |
| Maschere booleane | `torch.bool` | per `masked_fill` e le maschere di attenzione |
| Immagini appena lette | `torch.uint8` | da convertire, e da scalare in $[0,1]$ |

La riga della `CrossEntropyLoss` spiega anche perché il suo messaggio parla di
«target dtype» e non di «scalar type». Da PyTorch 1.10, infatti, quella loss
accetta *anche* target `float`, ma solo con la stessa shape dei logit,
interpretandoli come probabilità di classe (è la forma che serve per il *mixup*
e per la distillazione). Un target `float` di shape $(B,)$ è quindi un tipo
sbagliato, non una forma sbagliata: per questo l'errore parla di tipi.

Si noti che `float16` e `bfloat16` non fanno eccezione a queste regole: la
precisione mista, trattata in [prestazioni](prestazioni.md), non si ottiene
convertendo i tensori a mano ma lasciando gestire le conversioni a
`torch.autocast`, che sa quali operazioni si possono degradare e quali no.
`````

## 3. Il dispositivo non torna

```text
RuntimeError: Expected all tensors to be on the same device,
but found at least two devices, cuda:0 and cpu!
```

Il terzo errore è il più banale nella diagnosi e il più fastidioso da
prevenire: due tensori che devono incontrarsi si trovano in due memorie diverse.

`````{tab} Elementare
La CPU ha la sua memoria e la GPU la sua: sono due stanze separate, e da una
non si vede che cosa c'è nell'altra. Un'operazione richiede che entrambi i pezzi
siano nella stessa stanza, e il trasloco va chiesto esplicitamente con
`.to(device)`.

Il caso classico è aver spostato il modello e dimenticato i dati:

```{code-block} python
:class: pt-non-eseguibile

device = "cuda" if torch.cuda.is_available() else "cpu"
modello = modello.to(device)               # il modello trasloca...

for X, y in loader:
    X, y = X.to(device), y.to(device)      # ...e i dati devono seguirlo
    ...
```

C'è una differenza sottile nel segno `=` delle due righe. Nella prima, quella
del modello, il `=` si potrebbe anche togliere, perché `modello.to(device)`
senza assegnazione sposta il modello sul posto, e dopo quella riga il modello
sta di là. Nella seconda no.

La chiamata `X.to(device)`, infatti, non tocca `X`: fabbrica una copia di `X`
che sta di là e la restituisce (a meno che `X` non sia già di là, e allora
restituisce `X` stesso), e se non si salva questa copia, va persa. Ecco perché
nella riga dei dati c'è `X = X.to(device)`. Senza il `=` è una riga che sembra
funzionare e non fa nulla, e non dà nessun errore finché il tensore non arriva
al modello.

Un altro caso, che sfugge quasi sempre, riguarda un pezzo che nasce
dentro il modello mentre lavora. Una riga come `torch.ones(...)` lo fabbrica
nella stanza della CPU, perché nessuno le ha detto dove farlo, e il modello che
sta di là se lo trova davanti come qualcuno entrato dalla porta sbagliata. Il
rimedio è guardare dove sta un dato che si ha già in mano, invece di scrivere
il nome della stanza dentro il modello: `torch.ones(..., device=x.device)`
fabbrica il pezzo dove serve, qualunque stanza sia.

Le cose che appartengono al modello (una tabella di riferimento, una media da
sottrarre) vanno dichiarate sue con `self.register_buffer("nome", tensore)`.
Così traslocano insieme a lui e, quando il modello si salva su disco, partono
con lui. Un tensore lasciato lì come una variabile qualsiasi resta indietro in
tutti e due i casi, e l'errore salta fuori il giorno in cui si passa alla GPU.

Anche il viaggio di ritorno vuole il suo trasloco. Un risultato calcolato di là
si porta dietro la catena delle operazioni che l'hanno prodotto, che la
libreria tiene da parte perché le servirà per correggere i pesi; per questo,
così com'è, non lo si consegna agli strumenti che fanno i grafici. Prima lo si
stacca da quella catena, poi lo si riporta nella stanza della CPU, e a quel
punto è un numero come gli altri. In codice sono i tre passaggi di
`tensore.detach().cpu().numpy()`.
`````

`````{tab} Superiore
`nn.Module.to()` opera in-place su parametri e *buffer* registrati e
restituisce `self`; `Tensor.to()` è una funzione pura che restituisce un nuovo
tensore (o lo stesso oggetto, se è già sul dispositivo giusto). Da qui la
diversa disciplina d'uso.

Il caso che sfugge quasi sempre è un tensore creato dentro il `forward`:

```{code-block} python
:class: pt-non-eseguibile

def forward(self, x):
    maschera = torch.ones(x.shape[-1])            # nasce su CPU: errore
    maschera = torch.ones(x.shape[-1], device=x.device)   # corretto
```

Dentro un modulo il dispositivo si deduce da un tensore che si ha già in mano
(`x.device`) o dai propri parametri (`next(self.parameters()).device`), mai da
una variabile globale `device`.

Per le costanti che appartengono al modello (una tabella di codifiche
posizionali, una media di normalizzazione), la soluzione corretta è
`self.register_buffer("nome", tensore)`: i **buffer** non sono
parametri (non ricevono gradiente) ma seguono il modulo in `.to()`, finiscono
nello `state_dict` e si salvano con lui.

Infine, un tensore ancora agganciato al grafo autograd non si passa a NumPy, e
la sequenza completa è `tensore.detach().cpu().numpy()`: `detach` stacca dal
grafo, `cpu` lo riporta nella memoria della CPU, `numpy` converte.
`````

## Un metodo, non un rimedio

I tre errori si risolvono in trenta secondi *se* si legge il messaggio. Il
metodo con cui li si risolve va reso esplicito, perché funziona anche sugli
errori che nessun elenco contiene, compresi quelli che
nasceranno il mese prossimo.

1. **Leggi il traceback dal basso verso l'alto.** Quando un programma Python si
   ferma non stampa una frase, stampa un elenco: è il *traceback*, la catena di
   chiamate che ha portato all'errore, dalla prima riga lanciata (in cima) fino
   al punto in cui è esploso (in fondo). L'ultima riga dice *che cosa* è
   successo. Risalendo si trova la prima riga di codice *tuo*, che è il
   punto da guardare, non le venti righe interne di PyTorch che stanno sotto.
2. **Stampa la terna.** `print(x.shape, x.dtype, x.device)` prima della riga
   che esplode, e la stessa cosa per l'altro operando. Quasi sempre, a quel
   punto, l'errore diventa evidente.
3. **Riduci il problema.** Un batch solo, un esempio solo, un modello di due
   strati. Un errore che sopravvive alla riduzione si trova in un minuto.
4. **Fai una prova a vuoto prima di addestrare.** Un `forward` su un tensore
   finto della forma giusta (`torch.randn(2, 3, 224, 224)`) verifica in un
   istante tutta la catena, senza aspettare che il `DataLoader` scaldi i
   motori.

## Gli altri classici

I tre grandi errori si annunciano con un messaggio. Gli altri sono peggiori,
perché non danno nessun errore: il codice gira, le curve della loss dicono poco
o dicono il falso, e non c'è niente di rosso da leggere.

```{figure} ../figures/overfitting-memoria.svg
:name: fig-curva-nervosa
:alt: "Una decina di punti disposti lungo una tendenza crescente, attraversati da due curve. La prima è quasi una retta e coglie la tendenza generale, lasciando i punti un po’ sopra e un po’ sotto. La seconda è nervosa, sale e scende fra un punto e l'altro e passa esattamente per ognuno."
:width: 92%

La curva che passa per tutti i punti non ha capito meglio: ha memorizzato. Sul
training set il suo errore è zero, e questo deve insospettire.
```

{numref}`fig-curva-nervosa` ritrae un errore silenzioso frequente, e ne fa
vedere la causa più che il sintomo: una curva che passa per ogni punto invece
di seguire la tendenza. Il sintomo, sulle due loss, è che quella di
addestramento scende benissimo mentre quella di validazione risale. Il codice
funziona, non c'è niente da correggere in PyTorch, e per questo lo si scopre
tardi. È l'overfitting, già descritto nella {doc}`sezione su
overfitting e validazione </MachineLearning/overfitting-validazione>`, e il
paragrafo «Quando fermarsi» della {doc}`sezione sul training loop
<addestramento>` mostra come ci si ferma in tempo, con l'arresto anticipato.

- **`optimizer.zero_grad()` dimenticato.** Senza quella riga i gradienti si
  sommano invece di sostituirsi, e all'iterazione $t$ l'aggiornamento usa la
  somma dei $t$ gradienti calcolati fin lì, cioè circa $t$ volte la correzione
  media. Che cosa si vede dipende dall'ottimizzatore e dal passo, e non nel
  verso che ci si aspetta: con SGD a passo corto la loss all'inizio può perfino
  scendere più in fretta, a passo lungo esplode, e con Adam al passo di default
  l'esito dipende dal problema. In nessun caso arriva un messaggio, e la curva
  della loss non denuncia il bug.
- **Predizione e target di forma diversa.** Se le predizioni sono una colonna
  di otto numeri e le etichette una riga di otto, `nn.MSELoss` e `nn.L1Loss`
  non si fermano: allineano le due forme allargandole (è il *broadcasting* già
  visto sui tensori) e confrontano ogni predizione con ogni etichetta, cioè
  sessantaquattro coppie invece di otto. Il numero che esce è una media di
  errori incrociati senza significato, e non ha un verso, perché secondo gli
  esempi può risultare più basso del valore giusto o più alto; da quel numero,
  quindi, non ci si accorge di nulla. Un avviso nomina il problema (*«Using a
  target size that is different to the input size»*), ma è un `UserWarning`,
  che Python stampa una volta per messaggio distinto: compare alla prima
  chiamata della loss e poi tace (se l'ultimo batch è più corto i messaggi
  diventano due). Il modello passa la notte a ottimizzare l'obiettivo
  sbagliato mentre l'unica traccia è scorsa via un'ora prima.

  La `BCEWithLogitsLoss` fa eccezione, perché si rifiuta e lo dice, con un
  `ValueError` invece del solito `RuntimeError` (`Target size (...) must be the
  same as input size (...)`, con le due forme fra parentesi). Il rimedio è una
  riga sola, da scrivere prima delle loss che confrontano due tensori della
  stessa forma (`MSELoss`, `L1Loss`, `BCEWithLogitsLoss`):
  `assert pred.shape == target.shape`. Con `CrossEntropyLoss` e le classi date
  come numeri interi le forme sono diverse per costruzione, un punteggio per
  classe contro una classe per esempio, e l'asserzione fallirebbe sempre.
- **`optimizer.step()` dimenticato.** I gradienti si calcolano ma nessuno
  aggiorna i pesi: la loss resta piatta, identica, epoca dopo epoca.
- **Softmax applicata due volte.** Se l'ultimo strato del modello ha già una
  `nn.Softmax` e si usa `nn.CrossEntropyLoss`, la trasformazione avviene due
  volte: il modello impara comunque qualcosa, ma molto peggio. La loss,
  infatti, vuole i logit, e il paragrafo sulle funzioni di perdita della
  sezione sui {doc}`moduli <moduli>` ne mostra il perché, insieme al pavimento
  sotto cui, con una softmax di troppo, la loss non scende.
- **`model.eval()` dimenticato in valutazione.** Il modello resta in modalità
  di addestramento, cioè si comporta come quando impara: il dropout continua a
  spegnere neuroni a caso, e la batch norm continua a normalizzare con le
  statistiche del batch che ha davanti invece che con quelle raccolte durante
  l'addestramento (sono gli strati a doppia personalità della sezione su
  [studiare e dare l'esame](addestramento.md)). Le metriche ne escono diverse a
  ogni esecuzione, e non per forza peggiori: con la sola batch norm possono
  perfino uscire migliori. Resta poi un danno: in modalità di addestramento la
  batch norm aggiorna le proprie statistiche sui dati di validazione, quindi il
  modello esce cambiato dall'essere stato valutato male.
- **Memoria che cresce a ogni epoca.** Il tensore `perdita` è un numero *più*
  la catena di operazioni che l'ha prodotto, e accumulare `totale += perdita`
  tiene in vita la catena di ogni batch, una sopra l'altra. In addestramento la
  `backward()` ha già liberato i valori intermedi e restano soltanto i nodi
  della catena, per cui la memoria cresce piano, ma senza fermarsi. In
  valutazione senza `torch.no_grad()`, invece, nessuna `backward()` libera
  niente e per ogni batch restano anche i valori intermedi, così dopo qualche
  centinaio di iterazioni la memoria finisce (*out of memory*). In tutti e due
  i casi il rimedio è `perdita.item()`, che estrae il numero e lascia andare la
  catena.
- **`shuffle=True` sul `DataLoader` di test.** Non è un errore di per sé, ma
  rende impossibile confrontare le predizioni con le etichette in un ordine
  stabile. Non cambia invece l'accuratezza: con `model.eval()`
  è identica cifra per cifra con il mescolamento acceso
  e con quello spento, perché sono gli stessi esempi in un altro ordine.
  L'unico numero che cambia è la media delle medie per batch, che però è un
  errore numerico a sé, spiegato in [dal notebook agli
  script](dal-notebook-agli-script.md), e si sistema lì.
- **La loss diventa `nan`.** La sigla `nan` sta per *not a number* ed è il
  valore che i computer usano per dire "questo calcolo non ha un risultato". Lo
  si ottiene dividendo zero per zero, o sottraendo fra loro due infiniti (il
  valore assunto da un numero cresciuto oltre il limite rappresentabile).
  Quando compare al posto della loss, il primo sospettato è un learning rate
  troppo alto, con i pesi schizzati via; altrimenti c'è un denominatore che si
  annulla da qualche parte, e il posto dove si nasconde più spesso è la
  divisione per una deviazione standard nulla, cioè per un gruppo di numeri
  tutti uguali. Il secondo sospettato è il logaritmo di una probabilità esatta
  $0$, che non è una divisione per zero: vale meno infinito, e diventa un `nan`
  appena quell'infinito incontra uno zero o un altro infinito.

  I rimedi sono altrettanto meccanici: dividere il learning rate per tre o per
  dieci, e sommare al denominatore un numero minuscolo (si scrive
  $\varepsilon$, epsilon, e vale $10^{-5}$ negli strati che normalizzano,
  $10^{-8}$ dentro Adam) perché non possa mai annullarsi.

Per sapere *dove* nasce un `nan` c'è il **rilevatore di anomalie** di autograd,
`torch.autograd.set_detect_anomaly(True)`. Quando è acceso, fa fermare la
`backward()` sulla prima operazione che restituisce un `nan` e stampa la riga
della passata in avanti che l'ha creata. Rallenta molto, quindi si accende solo
per cercare. Vediamolo su un logaritmo di zero moltiplicato per zero:

```python
w = torch.zeros(3, requires_grad=True)
with torch.autograd.set_detect_anomaly(True):
    try:
        (torch.log(w) * 0).sum().backward()   # log(0) = -inf; -inf * 0 = nan
    except RuntimeError as e:
        print(e)
```

```text
Function 'LogBackward0' returned nan values in its 0th output.
```

Se invece i gradienti esplodono prima di arrivare al `nan`,
`torch.nn.utils.clip_grad_norm_(modello.parameters(), 1.0)`, scritta fra
`backward()` e `step()`, ne limita la norma.

Lo `zero_grad` dimenticato ha dietro un meccanismo preciso, che spiega perché
la diagnosi cambia con l'ottimizzatore.

`````{tab} Elementare
È il foglio delle correzioni che non si toglie mai dal tavolo. Al primo giro
c'è scritta una correzione, al secondo due, al decimo dieci, e ogni volta ci
si corregge con l'intero foglio. All'inizio sembra andare meglio, perché
le correzioni spingono tutte dalla stessa parte e si corre verso la risposta
giusta più in fretta di chi cambia il foglio.

Quando ci si arriva, però, non ci si ferma, perché sul foglio c'è ancora la
spinta di tutti i giri prima, e così si scavalca la risposta, si torna indietro,
la si riscavalca. Con passi piccoli si oscilla attorno a lei per sempre; con
passi grandi le oscillazioni crescono a ogni giro, e l'errore esplode.

Adam, l'ottimizzatore che si prova per primo, ogni volta divide la spinta per
quanto è grande di solito. In questo modo al passo normale evita
l'esplosione, ma la spinta dei giri prima resta, e secondo il problema arriva
lo stesso alla risposta giusta, o si pianta a metà strada. È il caso peggiore,
perché la curva dell'errore non mostra niente di strano; e se il passo è
grande, esplode anche lui.
`````

`````{tab} Superiore
Con SGD il bug ha un nome. Se `p.grad` non si svuota, l'aggiornamento
diventa $\theta_{t+1} = \theta_t - \eta \sum_{s \le t} \mathbf{g}_s$, cioè la
discesa col momento della {doc}`sezione sull'ottimizzazione
</Matematica/analisi-ottimizzazione>` con coefficiente $1$ e nessuno
smorzamento, e percorre cifra per cifra lo stesso cammino di
`optim.SGD(params, lr=eta, momentum=1.0)` con il ciclo corretto. Un momento
senza attrito all'inizio accelera, e per qualche centinaio di passi la loss può
stare perfino *sotto* quella del ciclo corretto, ma non si ferma mai.

Consideriamo una conca quadratica di curvatura $\lambda$, su cui l'iterazione
diventa $\theta_{t+1} = (2 - \eta\lambda)\,\theta_t - \theta_{t-1}$, le cui
radici hanno modulo $1$ per $0 < \eta\lambda < 4$. In quell'intervallo il
parametro oscilla attorno al minimo con ampiezza costante, e oltre quella
soglia l'ampiezza cresce a ogni iterazione e la loss esplode.

Con Adam, al passo che si usa di default, la loss non esplode, e la ragione è
strutturale: Adam non usa il gradiente così com'è, ma lo divide per
una misura di quanto quel gradiente è grande di solito, quindi moltiplicare il
gradiente per un fattore *costante* si semplifica e il passo resta lungo come
sempre. Qui però il fattore non è costante, perché cresce con $t$. Le due medie
di Adam (quella dei gradienti, al numeratore, e quella dei loro quadrati, al
denominatore, come le scrive la {doc}`sezione sul training loop
<addestramento>`) hanno memorie diverse, la prima corta e pronta a inseguire,
la seconda lunga e in ritardo, e il passo effettivo cresce lentamente.

A passo grande quella crescita basta a far divergere anche Adam, mentre al
passo di default l'esito dipende dal problema, e questa incertezza è il
pericolo. La normalizzazione toglie la scala ma non la memoria: la direzione
del passo resta quella della somma di tutti i gradienti visti, cioè un momento
senza attrito, che vicino al minimo lo scavalca invece di fermarcisi. Su un
batch fatto apposta per essere memorizzato la loss col bug può arrivare a zero
come quella del ciclo corretto, mentre su una regressione con una rete piccola
può restare bloccata molto più in alto. In nessuno dei due casi arriva una
divergenza o un messaggio: la curva non denuncia il bug, e il caso peggiore è
quindi l'ottimizzatore che si prova per primo.
`````

## Il grafico è il messaggio d'errore

Per gli errori rumorosi c'è il traceback. Per quelli silenziosi c'è un
grafico, lo strumento diagnostico che si usa più spesso: le due curve
della loss, addestramento e validazione, epoca per epoca.

Quel grafico va guardato dall'epoca uno, non alla fine. Aspettare il termine
dell'addestramento per tracciare le curve significa, su un modello
serio, scoprire dopo sei ore un difetto già leggibile dopo due epoche. Il costo
di stampare due numeri a fine epoca e disegnarli è nullo; il costo di non farlo
è una notte di GPU.

```{figure} ../figures/curve-di-addestramento.svg
:name: fig-curve-diagnosi
:alt: "Quattro grafici della loss per epoca, ciascuno con la curva di addestramento e quella di validazione: nel primo restano entrambe alte e piatte, nel secondo scendono insieme con un divario stabile, nel terzo quella di addestramento continua a scendere mentre quella di validazione risale dopo un minimo segnato, nel quarto oscillano entrambe senza una tendenza chiara."
:width: 95%

Quattro forme e quattro diagnosi. La forma della coppia di curve dice più del
valore finale della loss, e per questo va disegnata mentre l'addestramento
gira.
```

Le quattro forme di {numref}`fig-curve-diagnosi` coprono i casi più
frequenti.

**Non impara.** Le curve restano alte e piatte, ma occorre distinguere. Se la
loss è identica epoca dopo epoca, il problema non è di
apprendimento ma un bug, e i sospetti sono pochi (manca `optimizer.step()`, il
learning rate è zero, i parametri sono congelati da un `requires_grad=False` di
troppo, oppure la loss che si retropropaga non è collegata all'uscita del
modello). Se invece scende pochissimo ma scende, è *underfitting*: il modello
non ha abbastanza capacità, o il passo è troppo corto, o le feature non
contengono l'informazione richiesta.

**Sano.** Entrambe scendono, la validazione sta un po’ sopra, e il divario fra
le due resta più o meno costante. Un divario c'è quasi sempre e non indica un
problema: quello che si sorveglia è se si allarga.

**Overfitting.** L'addestramento continua a scendere, la validazione tocca un
minimo e risale. Il rimedio, l'arresto anticipato, è descritto in [quando
fermarsi](addestramento.md). Soprattutto nei modelli con molti più parametri
che esempi da imparare, però, la risalita può non essere definitiva, e
continuando ad addestrare la validazione può tornare a scendere una seconda
volta. È il fenomeno della *doppia discesa*, che la {doc}`sezione su
overfitting e validazione
</MachineLearning/overfitting-validazione>` racconta in tutte e due le forme,
quella che dipende dai parametri e quella che dipende dalle epoche.

In pratica l'arresto anticipato resta la regola, ma chi ha tempo da spendere
può continuare oltre il primo minimo, tenendo da parte la copia migliore, e
guardare se la seconda discesa arriva.

**Passo troppo lungo.** Le curve oscillano vistosamente, magari con una
tendenza generale al ribasso ma con salti che la coprono. In questo caso
l'ottimizzatore sta scavalcando il minimo a ogni passo, oppure il gradiente è
talmente rumoroso da non indicare più una direzione stabile.

`````{tab} Elementare

La regoletta operativa, una forma di curva alla volta.

Se la curva è piatta come un tavolo, c'è un errore nel codice, e va cercato nel
ciclo di addestramento prima di toccare qualunque manopola. Se scende appena,
allunga il passo (moltiplicandolo per tre, non alzandolo del dieci per cento,
perché la misura giusta può stare mille volte più in là) e, quando non basta,
ingrandisci il modello.

Se cala in fretta all'inizio e poi si pianta su un valore mediocre, il passo
resta troppo lungo per infilarsi nel punto più basso, e va accorciato. Se il
divario fra le due si allarga servono i freni, cioè qualcosa che renda la
vita più difficile al modello mentre studia (deformare un po’ gli esempi a ogni
giro, spegnergli a caso qualche neurone) oppure, più semplicemente, fermarsi
prima.

La curva che salta, invece, ha due cause che si somigliano e non si curano allo
stesso modo. O il passo è troppo lungo e il modello scavalca il punto più basso
a ogni giro, oppure i mazzetti sono troppo piccoli, e con pochi esempi alla
volta la direzione che il modello ricava è una media presa male, che traballa da
un giro all'altro.

Per distinguerle basta una prova: quadruplica il mazzetto, lasciando il passo
com'è, e riguarda le curve. Se il tremolio si calma era la media presa male;
se invece le curve saltano identiche la colpa è del passo, e allora ingrandire
i mazzetti non serve a niente, perché costa più calcolo senza dare niente in
cambio. Si quadruplica perché il tremolio si dimezza ogni volta che gli esempi
a giro si quadruplicano, mentre raddoppiando se ne toglie meno di un terzo, e a
occhio non si vede.

Il collaudo dei dieci esempi, quello del {doc}`flusso di lavoro
<flusso-di-lavoro>`, si fa prima di lanciare l'addestramento vero: un solo
mazzetto di esempi, una decina, nessun freno, un passo normale, e si addestra
finché la loss non arriva quasi a zero. Se la rete, che ha più pesi che
esempi, non ci riesce, quasi sempre si è rotto qualcosa lungo la catena, e i
posti dove guardare sono pochi. Le etichette non corrispondono più
agli esempi a cui erano attaccate; una trasformazione ha rovinato i dati per
strada; la perdita si calcola su una grandezza sbagliata; oppure la correzione
non arriva fino a tutti i pesi.

`````

`````{tab} Superiore

Il learning rate lascia una firma riconoscibile, e conviene imparare a
leggerla perché è l'iperparametro a cui l'addestramento è più sensibile
{cite}`goodfellow2016deep`. Troppo alto: la loss esplode o resta alta e
irregolare (nel caso estremo diventa `nan`). Alto: cala in fretta nelle prime
iterazioni e poi si assesta su un valore mediocre, perché il passo è troppo
lungo per entrare nella conca. Troppo basso: scende in modo quasi lineare e
lentissimo, senza mai piegare. Giusto: cala rapidamente e continua a
migliorare. Poiché la scala giusta cambia di ordini di grandezza fra un
problema e l'altro, la si esplora moltiplicando o dividendo per tre, non
aggiustandola di percentuali.

Il rumore delle curve ha due sorgenti diverse, che si riconoscono da come si
comporta l'oscillazione. Il passo troppo lungo produce oscillazioni che non si
riducono aumentando la dimensione del batch; il rumore del gradiente scala
invece come $1/\sqrt{B}$ con la dimensione $B$ del batch, quindi quadruplicare
il batch dimezza l'ampiezza. La prova va fatta quadruplicando e non
raddoppiando, perché raddoppiare toglie solo il 29%, che su curve rumorose a
occhio non si distingue. Se quadruplicare $B$ calma visibilmente le curve, era
rumore di campionamento; se non cambia nulla, è il passo. La prova si fa a
learning rate invariato, in modo che il passo medio resti lo stesso e cambi
soltanto il rumore.

Il collaudo canonico prima di ogni addestramento serio è sovradattare di
proposito un batch solo: si prendono otto o dieci esempi, si disattiva ogni
regolarizzazione e si addestra su quelli per qualche centinaio di iterazioni.
Un modello sano arriva a una loss praticamente nulla, perché memorizzare dieci
esempi è alla portata di qualunque rete con più parametri che esempi (una
regressione lineare con sei parametri no, e lì il fallimento è atteso).

Se il modello non ci riesce con un learning rate ordinario (per Adam,
$10^{-3}$) e senza regolarizzazione, il difetto è quasi sempre nella catena:
etichette disallineate rispetto agli ingressi, una trasformazione che distrugge
il segnale, una loss calcolata su una grandezza sbagliata, un gradiente che non
arriva a tutti i parametri.

`````

### Quando la validazione va meglio dell'addestramento

C'è una forma che confonde chiunque la incontri la prima volta: la curva di
validazione sta sotto quella di addestramento. Sembra impossibile, perché
il modello ha visto i dati di addestramento e non quelli di validazione.
Quasi sempre, però, non c'è nulla di rotto, e le spiegazioni sono quattro.

1. Si stanno confrontando due misure prese in momenti diversi. La loss di
   addestramento che si stampa è di solito la media su tutti i batch
   dell'epoca, calcolata *mentre* i pesi cambiavano, e quindi include anche i
   pesi peggiori di inizio epoca. Quella di validazione è calcolata alla fine,
   con i pesi migliori. Nelle prime epoche, quando il miglioramento dentro una
   singola epoca è grande, questo basta a invertire l'ordine.
2. La regolarizzazione è attiva solo in addestramento. Il dropout spegne
   neuroni e la data augmentation deforma gli esempi durante il training e non
   in valutazione, per cui la rete che produce la loss di addestramento è una
   rete azzoppata, mentre quella che produce la validazione no.
3. Il set di validazione è più facile. Può capitare per caso con split
   piccoli, o sistematicamente se la separazione non è stata casuale (i casi
   difficili concentrati da una parte sola).
4. Si è addestrato poco. Nelle prime epoche il modello non ha ancora
   memorizzato niente, quindi non ha alcun vantaggio sui dati visti.

Le prime due, in particolare, si verificano in un minuto: si ricalcola la loss
di addestramento a fine epoca con `model.eval()`, sugli stessi pesi con cui si
misura la validazione. Se il paradosso sparisce, non c'era.

```{code-block} python
:class: pt-non-eseguibile

# Registrare la storia costa due liste, e senza storia non c'è diagnosi.
storia = {"train": [], "val": []}

for epoca in range(n_epoche):
    modello.train()
    somma, n = 0.0, 0
    for X, y in loader_train:
        ...                                   # i cinque passi soliti
        somma += perdita.item() * X.size(0)   # .item(): niente grafo in memoria
        n += X.size(0)
    storia["train"].append(somma / n)

    modello.eval()                            # niente dropout, niente augmentation
    with torch.no_grad():
        somma, n = 0.0, 0
        for X, y in loader_val:
            somma += loss_fn(modello(X), y).item() * X.size(0)
            n += X.size(0)
    storia["val"].append(somma / n)

    # stampare a ogni epoca, non alla fine: è tutto il punto
    print(f"epoca {epoca:3d}  train {storia['train'][-1]:.4f}  val {storia['val'][-1]:.4f}")

import matplotlib.pyplot as plt
plt.plot(storia["train"], label="addestramento")
plt.plot(storia["val"], label="validazione", linestyle="--")
plt.xlabel("epoche"); plt.ylabel("loss"); plt.legend()
```

Tre errori che si annunciano, otto che tacciono, più l'overfitting che si
legge nelle curve. Il metodo vale per tutti e dodici: leggere il messaggio dal
basso, stampare la terna, ridurre il problema, fare una prova a vuoto, e per gli
errori silenziosi disegnare le due curve dalla prima epoca.

`````{tab} Elementare
```{admonition} Da ricordare
:class: important
- Un tensore ha forma, tipo e dispositivo: quasi ogni errore rosso di
  PyTorch riguarda uno di questi tre, e la prima cosa da fare è stamparli tutti
  e tre, per il tuo dato e per l'altro operando.
- Forma: manca lo strato che srotola l'immagine, due strati non combaciano,
  oppure manca il mazzetto attorno all'esempio singolo.
- Tipo: numeri decimali per i dati, numeri interi per le etichette quando
  la risposta è una categoria fra tante. Le immagini appena lette sono fatte di
  interi da 0 a 255 e vanno convertite una volta sola, appena entrano.
- Dispositivo: `modello.to(device)` sposta il modello sul posto; spostare un dato
  restituisce una *copia*, e se non la riassegni non hai fatto niente.
- Il metodo vale più dei rimedi: leggi l'errore dall'ultima riga in su,
  stampa forma-tipo-dispositivo, riduci il problema a un esempio solo, e prova
  la catena con un dato finto prima di lanciare l'addestramento vero.
- Gli errori peggiori sono quelli silenziosi, quelli senza niente di rosso: le
  correzioni del giro prima non azzerate, la softmax applicata due volte, la
  modalità studio lasciata accesa durante l'esame, la perdita accumulata
  intera invece che ridotta al suo numero, e la predizione confrontata con
  un'etichetta di forma diversa, che si aggancia lo stesso e dà un numero senza
  senso.
- Per gli errori silenziosi il messaggio d'errore è la coppia di curve, e
  va disegnata dalla prima epoca. Piatta come un tavolo: è un errore nel
  codice. Scende appena: passo corto o modello piccolo. Salta: passo lungo o
  mazzetti troppo piccoli. Distanza fra le due che si allarga: servono i
  freni.
- Far imparare a memoria dieci esempi, con un passo normale e senza freni, è
  il collaudo più economico che esista: se il modello non ci riesce, quasi
  sempre il difetto è nel codice.
- La validazione sotto l'addestramento di solito è normale: il numero
  dell'addestramento è una media presa mentre il modello stava ancora
  cambiando, e i trucchi che si usano solo mentre si studia penalizzano
  soltanto lui.
```
`````

`````{tab} Superiore
```{admonition} Da ricordare
:class: important
- Un tensore ha forma, tipo e dispositivo: quasi ogni `RuntimeError` di
  PyTorch riguarda uno di questi tre.
- Forma: manca `nn.Flatten()`, due strati non combaciano, o manca la
  dimensione del batch (`unsqueeze(0)`). `torchinfo.summary` la mostra strato
  per strato.
- Tipo: `float32` per i dati, `int64` per le etichette della
  `CrossEntropyLoss`, `float32` per quelle della `BCEWithLogitsLoss`. NumPy
  produce `float64`: convertire al confine.
- Dispositivo: `modello.to(device)` modifica sul posto, `x.to(device)`
  restituisce una copia da riassegnare. I tensori creati nel `forward` vanno
  creati con `device=x.device`; le costanti del modello con
  `register_buffer`.
- Il metodo vale più dei rimedi: traceback dal basso, stampa della terna,
  problema ridotto, prova a vuoto con un tensore finto.
- Gli errori peggiori sono quelli silenziosi: `zero_grad` mancante (con SGD
  e passo corto la loss all'inizio può perfino scendere più in fretta, con
  passo lungo esplode; con Adam al passo di default l'esito dipende dal
  problema e la curva non denuncia il bug, caso peggiore; a passo grande
  diverge anche Adam), softmax doppia, `eval()` dimenticato,
  `.item()` dimenticato nell'accumulo (e in valutazione senza
  `torch.no_grad()` la memoria finisce), e forme di predizione e target che si
  allineano per broadcasting (`assert pred.shape == target.shape` prima delle
  loss che confrontano forme uguali: MSE, L1, BCE).
- Per gli errori silenziosi il messaggio d'errore è la coppia di curve, e
  va disegnata dall'epoca uno. Piatta come un tavolo: è un bug. Scende appena:
  passo corto o modello piccolo. Salta: passo lungo o batch piccolo. Divario
  che si allarga: servono i freni.
- Sovradattare di proposito un batch da dieci esempi è il collaudo più
  economico che esista: se il modello non ci riesce con un learning rate
  ordinario e senza regolarizzazione, il difetto è quasi sempre nella
  catena.
- La validazione sotto l'addestramento di solito non è un errore: la loss
  di training è una media presa mentre i pesi cambiavano, e dropout e
  augmentation penalizzano solo il training.
```
`````

Il codice visto finora è codice da notebook, che per esplorare è lo strumento
giusto. Quando lo stesso codice deve sopravvivere più di un pomeriggio, ed
essere rilanciato da altri, o da te fra un mese, la forma cambia: è il
passaggio {doc}`dal notebook agli script <dal-notebook-agli-script>`.
