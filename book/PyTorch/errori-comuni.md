# I tre errori più comuni

Esiste un rito d'iniziazione che accomuna chiunque abbia scritto codice
PyTorch, dal primo esercizio al laboratorio di ricerca: un `RuntimeError`
lungo venti righe, la sensazione che il messaggio sia scritto in una lingua
straniera, e mezz'ora persa a spostare `.to(device)` da una riga all'altra
sperando che smetta. La buona notizia è che quel messaggio non è mai davvero
nuovo. Sotto la varietà apparente ci sono **tre** errori, e sono sempre gli
stessi tre: la forma dei dati, il loro tipo, il dispositivo su cui abitano.

Un tensore ha esattamente queste tre proprietà oltre ai numeri che contiene
(`shape`, `dtype`, `device`) ed è per questo che sbagliare significa quasi
sempre sbagliare una delle tre. Chi impara a riconoscerle a colpo d'occhio
smette di debuggare per tentativi.

```python
x = torch.randn(32, 3, 224, 224)
print(x.shape)    # torch.Size([32, 3, 224, 224])  -> la forma
print(x.dtype)    # torch.float32                   -> il tipo
print(x.device)   # cpu                             -> dove abita
```

Questa terna è la prima cosa da stampare quando qualcosa non va. Il resto
della sezione la percorre un vertice alla volta.

## 1. La forma non torna

```text
RuntimeError: mat1 and mat2 shapes cannot be multiplied (32x784 and 128x10)
```

È il più frequente in assoluto, e il messaggio (una volta imparato a leggerlo)
dice già tutto: PyTorch ha provato a moltiplicare una matrice $32 \times 784$
per una $128 \times 10$, e non si può, perché il numero di colonne della prima
($784$) non coincide con il numero di righe della seconda ($128$). È la regola
del prodotto fra matrici vista nel capitolo di algebra lineare, e il motivo per
cui esiste è che ogni riga della prima matrice viene accoppiata a una colonna
della seconda, numero per numero: se le due file non hanno la stessa lunghezza,
gli accoppiamenti non si chiudono.

`````{tab} Elementare
Pensa ai tensori come a scatole con un'etichetta che dice quante cose
contengono e come sono disposte. Un'operazione tra due scatole richiede che le
etichette combacino su un certo lato, esattamente come due tubi che si
avvitano solo se hanno lo stesso diametro. Le cause sono sempre di tre tipi.

**Manca l'appiattimento.** Un'immagine è una scatola $(3, 224, 224)$; uno
strato `nn.Linear` vuole una lista piatta di numeri. Serve un `nn.Flatten()`
in mezzo, che srotola l'immagine in un unico vettore lungo.

**Due strati non si parlano.** L'uscita di uno strato deve essere l'ingresso
del successivo: se il primo produce $128$ numeri, il secondo deve aspettarsene
$128$, non $256$. È un errore di battitura più che di concetto, ma capita ogni
volta che si cambia una dimensione e ci si dimentica di aggiornare la riga
sotto.

**Manca la dimensione del gruppo.** Il modello si aspetta *un mucchietto* di
esempi, anche quando l'esempio è uno solo. Un'immagine singola va da
$(3, 224, 224)$ a $(1, 3, 224, 224)$, con `x.unsqueeze(dim=0)`, cioè "un
mucchietto che contiene un'immagine". È l'inciampo classico del momento in cui
si prova il modello su una foto scaricata al volo.
`````

`````{tab} Superiore
Il conto da tenere è quello delle dimensioni lungo la rete, e le tre cause si
formalizzano così.

**Appiattimento.** `nn.Linear(d_in, d_out)` opera sull'**ultima** dimensione e
lascia intatte le precedenti: applicato a $(B, C, H, W)$ fallisce a meno che
$W = d_{\text{in}}$. Per un MLP su immagini serve `nn.Flatten()` (che per
default appiattisce da `start_dim=1`, preservando il batch) e
$d_{\text{in}} = C \cdot H \cdot W$.

**Composizione.** In una `nn.Sequential`, `out_features` di uno strato deve
uguagliare `in_features` del successivo. Quando la dimensione dipende da
calcoli (l'uscita di uno stack convoluzionale, per esempio, dove ogni `stride`
e ogni `padding` la modificano), conviene non calcolarla a mano:
`nn.LazyLinear(d_out)` la deduce dal primo tensore che riceve, materializzando
i pesi alla prima chiamata. È comodo in fase esplorativa; nel codice
definitivo, meglio fissare il numero.

**Dimensione di batch.** Quasi tutti i moduli assumono la convenzione
*batch-first* $(B, \dots)$. In inferenza su un singolo esempio si aggiunge con
`unsqueeze(0)`. Attenzione all'inverso: `squeeze()` senza argomento elimina
**tutte** le dimensioni unitarie, e su un batch da un elemento cancella anche
quella del batch; si passi sempre `dim` esplicito.

Lo strumento di diagnosi è `torchinfo`:
`summary(modello, input_size=(32, 3, 224, 224))` stampa la forma in ingresso e
in uscita di ogni strato, ed è il modo più rapido per vedere dove la catena si
spezza. In alternativa, un `print(x.shape)` in ogni riga del `forward`: poco
elegante, sempre efficace.
`````

## 2. Il tipo non torna

```text
RuntimeError: mat1 and mat2 must have the same dtype, but got Byte and Float
RuntimeError: expected target dtype to be Long or Byte, but got Float
```

Il secondo errore riguarda il `dtype`, cioè il tipo di numero con cui il
tensore è scritto: interi (il $3$) oppure decimali (il $3{,}0$). Quando somma
o moltiplica due tensori numero per numero, PyTorch converte da sé quello
"più stretto" nel tipo dell'altro; ma nel prodotto fra matrici e dentro gli
strati `nn` (i casi dei due messaggi qui sopra) il tipo se lo aspetta preciso,
e preferisce fermarsi piuttosto che indovinare quale volevamo.

I due messaggi vanno letti in modi diversi, e conviene saperlo perché è proprio
la varietà delle formulazioni a disorientare. Nel **primo** i due tipi non sono
"atteso" e "trovato": sono i **due operandi**, il tensore che abbiamo passato e
i pesi del modello, nell'ordine. `Byte` è il nome interno di `uint8`, il tipo
di un'immagine appena letta da un file, e `Float` è il tipo dei pesi: la frase
dice che una cosa a interi e una a decimali si sono incontrate in una
moltiplicazione fra matrici. Il **secondo** riguarda le etichette, e lì i nomi
sono quelli dei tipi ammessi (`Long`, cioè `int64`) contro quello ricevuto.

Una nota sulla varietà: la stessa situazione produce frasi diverse a seconda
dell'operazione. Uno strato lineare dice `mat1 and mat2 must have the same
dtype`; una convoluzione con bias dice `Input type (unsigned char) and bias
type (float) should be the same`; una convoluzione senza bias dice
`expected scalar type Byte but found Float`. Sono tre modi di dire la stessa
cosa, e riconoscerne il tema è più utile che impararli a memoria.

`````{tab} Elementare
Un'immagine appena letta da un file è fatta di numeri interi da $0$ a $255$: è
il tipo `uint8`, il "Byte" del messaggio. Una rete neurale lavora invece con
numeri decimali fra $0$ e $1$. In una somma PyTorch la conversione la fa da
sé, ma **la rete no**: dentro uno strato il tipo se lo aspetta preciso, e la
conversione è a carico nostro. La fa la trasformazione `ToTensor()`, che
converte *e* divide per $255$ portando i valori nell'intervallo giusto; a mano
servono tutti e due i gesti, `.float() / 255`, e il secondo si dimentica
spesso, con l'effetto di dare alla rete numeri fino a $255$ volte più grandi di
quelli che si aspetta.

Il secondo messaggio è il caso opposto e riguarda le **etichette**. Alla
`CrossEntropyLoss` le classi vere si danno come numeri interi (la classe $3$,
non $3{,}0$), perché sono nomi, non quantità. Passare $3{,}0$ produce
quell'errore. Alla `BCEWithLogitsLoss`, invece, servono proprio decimali,
perché lì l'etichetta è una probabilità ($0{,}0$ oppure $1{,}0$).

La regola pratica: **una sola conversione, il più presto possibile**. Si
converte quando i dati entrano, non a metà del training loop.
`````

`````{tab} Superiore
Il default di PyTorch è `float32`; il default di NumPy è `float64`. Un
`torch.from_numpy(array)` conserva il `float64` e produce un tensore che non
può essere moltiplicato per i pesi `float32` del modello (`mat1 and mat2 must
have the same dtype, but got Double and Float`, dove `Double` è `float64`): è
la sorgente più insidiosa di errori di tipo, perché nasce fuori da PyTorch. La
conversione esplicita `torch.from_numpy(a).float()` (o `.to(torch.float32)`) va
fatta al confine.

Sul *type promotion* elemento per elemento vale la pena essere precisi, perché
la formula che si legge di solito («le stesse regole di NumPy») è vera solo in
parte. Confrontando `torch.result_type` con `np.result_type` su tutte le
coppie dei tipi che le due librerie hanno in comune (il `bfloat16` in NumPy
non esiste, quindi i tipi confrontabili sono otto e le coppie $8^2 = 64$),
**dieci divergono**, e sono tutte del tipo intero
per decimale: a differenza di NumPy, in PyTorch un intero non spinge mai un
decimale a una precisione più alta. `int64 + float32` dà `float32` in torch e
`float64` in NumPy; `torch.tensor([1,2,3]) * 2.5` dà `float32`, mentre
`np.array([1,2,3]) * 2.5` dà `float64`. Il che rafforza il paragrafo qui sopra
invece di indebolirlo: il `float64` che arriva da NumPy è insidioso proprio
perché le due librerie non la pensano allo stesso modo.

Il quadro completo dei tipi che si incontrano:

| Contenuto | dtype atteso | Note |
|---|---|---|
| Immagini, feature, attivazioni | `torch.float32` | il default dei pesi |
| Etichette per `CrossEntropyLoss` | `torch.int64` (`long`) | indici di classe, shape $(N,)$; oppure `float32` di shape $(N,K)$, cioè probabilità |
| Etichette per `BCEWithLogitsLoss` | `torch.float32` | shape uguale ai logit |
| Maschere booleane | `torch.bool` | per `masked_fill` e le maschere di attenzione |
| Immagini appena lette | `torch.uint8` | da convertire, e da scalare in $[0,1]$ |

La riga della `CrossEntropyLoss` spiega anche perché il suo messaggio parla di
«target dtype» e non di «scalar type»: da PyTorch 1.10 quella loss accetta
*anche* target `float`, ma solo con la stessa shape dei logit, interpretandoli
come probabilità di classe (è la forma che serve per il *label smoothing* e per
la distillazione). Un target `float` di shape $(N,)$ non è quindi una forma
sbagliata, è un **tipo** sbagliato, ed è per questo che l'errore parla di tipi.

Nota che `float16` e `bfloat16` non fanno eccezione a queste regole: la
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
prevenire: due tensori che devono incontrarsi abitano in due memorie diverse.

`````{tab} Elementare
La CPU e la GPU hanno ciascuna la propria memoria, e non si vedono tra loro:
sono due stanze separate. Un'operazione richiede che entrambi i pezzi siano
nella stessa stanza, e il trasloco va chiesto esplicitamente con `.to(device)`.

Il caso classico è aver spostato il modello e dimenticato i dati:

```python
device = "cuda" if torch.cuda.is_available() else "cpu"
modello = modello.to(device)               # il modello trasloca...

for X, y in loader:
    X, y = X.to(device), y.to(device)      # ...e i dati devono seguirlo
    ...
```

Attenzione a una differenza sottile: `modello.to(device)` sposta il modello
*sul posto*, mentre `X.to(device)` restituisce una **copia** e non modifica
`X`. Scrivere `X.to(device)` senza riassegnare è una riga che sembra
funzionare e non fa nulla.
`````

`````{tab} Superiore
`nn.Module.to()` opera in-place su parametri e *buffer* registrati e
restituisce `self`; `Tensor.to()` è una funzione pura che restituisce un nuovo
tensore (o lo stesso oggetto, se è già sul dispositivo giusto). Da qui la
diversa disciplina d'uso.

Il caso che sfugge quasi sempre è un tensore **creato dentro il `forward`**:

```python
def forward(self, x):
    maschera = torch.ones(x.shape[-1])            # nasce su CPU: errore
    maschera = torch.ones(x.shape[-1], device=x.device)   # corretto
```

La regola generale è non leggere mai una variabile globale `device` dentro un
modulo, ma dedurlo da un tensore che si ha già in mano (`x.device`) o dai
propri parametri (`next(self.parameters()).device`). Per le costanti che
appartengono al modello (una tabella di codifiche posizionali, una media di
normalizzazione), la soluzione corretta è
`self.register_buffer("nome", tensore)`: i buffer non sono parametri (non
ricevono gradiente) ma seguono il modulo in `.to()`, finiscono nello
`state_dict` e si salvano con lui.

Ultimo dettaglio: un tensore ancora agganciato al grafo autograd non si passa
a NumPy. `tensore.detach().cpu().numpy()` è la sequenza completa: `detach`
stacca dal grafo, `cpu` fa il trasloco, `numpy` converte.
`````

## Un metodo, non un rimedio

I tre errori sopra si risolvono in trenta secondi *se* si legge il messaggio.
Vale la pena rendere esplicito il metodo, perché funziona anche per il
quarto errore, quello che non è in nessun elenco.

1. **Leggi il traceback dal basso verso l'alto.** Quando un programma Python si
   ferma non stampa una frase, stampa un elenco: è il *traceback*, la catena di
   chiamate che ha portato all'errore, dalla prima riga lanciata (in cima) fino
   al punto esatto in cui è esploso (in fondo). L'ultima riga dice *che cosa* è
   successo; risalendo si trova la prima riga di codice *tuo*: quella è il
   punto da guardare, non le venti righe interne di PyTorch che stanno sotto.
2. **Stampa la terna.** `print(x.shape, x.dtype, x.device)` prima della riga
   che esplode, e la stessa cosa per l'altro operando. Nove volte su dieci
   l'errore diventa evidente.
3. **Riduci il problema.** Un batch solo, un esempio solo, un modello di due
   strati. Un errore che sopravvive alla riduzione si trova in un minuto.
4. **Fai un giro a vuoto prima di addestrare.** Un `forward` su un tensore
   finto della forma giusta (`torch.randn(2, 3, 224, 224)`) verifica in un
   istante tutta la catena, senza aspettare che il `DataLoader` scaldi i
   motori.

## Gli altri classici

I tre grandi errori si annunciano con un messaggio. I prossimi sono peggiori,
perché **non danno nessun errore**: il codice gira, la loss scende poco o non
scende affatto, e non c'è niente di rosso da leggere.

```{figure} ../figures/overfitting-memoria.svg
:name: fig-curva-nervosa
:alt: "Una nube di punti attraversata da due curve. La prima è semplice e coglie la tendenza generale, lasciando i punti sparsi attorno a sé. La seconda è nervosa e passa esattamente per ogni punto, comprese le oscillazioni che sono soltanto rumore."
:width: 92%

La curva che passa per tutti i punti non ha capito meglio: ha memorizzato. Sul
training set il suo errore è zero, ed è proprio questo a doverci insospettire.
```

{numref}`fig-curva-nervosa` è la forma grafica dell'errore silenzioso più
comune di tutti, quello che chiude questa lista: una loss di addestramento che
scende benissimo mentre quella di validazione risale. Il codice funziona, non
c'è niente da correggere in PyTorch, e proprio per questo lo si scopre tardi.

- **`optimizer.zero_grad()` dimenticato.** I gradienti si accumulano, quindi
  al passo $t$ quello che si usa è la somma dei $t$ gradienti precedenti,
  all'incirca $t\,\bar{g}$, dove $\bar{g}$ è il gradiente medio visto finora.
  Che cosa si veda dipende dall'ottimizzatore, e conviene saperlo perché la
  polarità è l'opposto di quella che ci si aspetta. Con **SGD** il passo vale
  $\eta \, t \, \bar{g}$ e cresce insieme alla somma: con un learning rate
  piccolo la loss resta *migliore* di quella del ciclo corretto per tutta la
  corsa ($0{,}043$ contro $0{,}690$, misurando), con uno grande diverge fino a
  $4{,}5 \cdot 10^{3}$ (su MNIST è arrivata a $177$, dove una loss sana sta a
  $2{,}3$, e il modello ha finito per tirare a indovinare). Con **Adam** non
  esplode mai, in nessuna prova, e la ragione è strutturale:
  l'aggiornamento vale $\hat{m}/(\sqrt{\hat{v}} + \varepsilon)$, cioè è
  invariante di scala, quindi un gradiente che cresce come $t\,\bar{g}$ si
  semplifica fra numeratore e denominatore e il passo resta quello di sempre.
  A $\eta = 10^{-3}$ il bug è perfino *più bravo* del ciclo corretto. Il caso
  peggiore è quindi Adam, cioè proprio l'ottimizzatore che il capitolo
  raccomanda come default: lì il bug si traveste da successo e resta
  invisibile per sempre, perché non c'è nessuna divergenza a denunciarlo.
- **La predizione e il target non hanno la stessa forma.** `nn.MSELoss` e
  `nn.L1Loss` non protestano: applicano il broadcasting e mediano su una
  matrice $N \times N$ di residui incrociati, cioè confrontano ogni predizione
  con *ogni* etichetta. Con predizioni $(8, 1)$ e target $(8,)$ la MSE
  restituisce $1{,}8507$ invece di $2{,}0946$; con la L1 lo scarto è dello
  $0{,}3\%$, indistinguibile dal rumore, e il modello passa la notte a
  ottimizzare la cosa sbagliata. La `BCEWithLogitsLoss` è l'eccezione gentile,
  perché si rifiuta e lo dice (`Target size must be the same as input size`).
  Il rimedio è una riga sola, e va scritta prima di ogni loss:
  `assert pred.shape == target.shape`.
- **`optimizer.step()` dimenticato.** I gradienti si calcolano ma nessuno
  aggiorna i pesi: la loss resta piatta, identica, epoca dopo epoca.
- **Softmax applicata due volte.** Se l'ultimo strato del modello ha già una
  `nn.Softmax` e si usa `nn.CrossEntropyLoss`, la trasformazione avviene due
  volte: il modello impara comunque qualcosa, ma molto peggio. La loss vuole i
  **logit** (si veda il [flusso di lavoro](flusso-di-lavoro.md)).
- **`model.eval()` dimenticato in valutazione.** Con dropout e batch norm
  attivi, le metriche di test risultano peggiori e (cosa più insidiosa)
  diverse a ogni esecuzione.
- **Memoria che cresce a ogni epoca.** `perdita` non è un numero: è un numero
  *più* tutta la catena di operazioni che l'ha prodotto, che PyTorch conserva
  perché servirà al calcolo della derivata. Accumulare `totale += perdita`
  tiene quindi in vita l'intera catena di ogni batch, una sopra l'altra;
  `perdita.item()` estrae il numero e basta, e la catena può essere buttata.
  Senza quel `.item()`, dopo qualche centinaio di iterazioni la memoria finisce
  (*out of memory*).
- **La loss diventa `nan`.** `nan` sta per *not a number* ed è il valore che i
  computer usano per dire "questo conto non ha un risultato": lo si ottiene
  dividendo zero per zero, o da un numero cresciuto oltre il rappresentabile.
  Quando compare al posto della loss, quasi sempre il learning rate è troppo
  alto e i pesi sono schizzati via; altrimenti c'è una divisione per zero
  nascosta da qualche parte, e i due posti dove si nasconde più spesso sono il
  logaritmo di una probabilità esatta $0$ (che vale meno infinito) e la
  divisione per una deviazione standard nulla, cioè per un gruppo di numeri
  tutti uguali. I rimedi sono altrettanto meccanici: dimezzare il learning
  rate, e sommare al denominatore un numero minuscolo (si scrive
  $\varepsilon$, epsilon, e vale tipicamente $10^{-8}$) perché non possa mai
  essere zero esatto.
- **`shuffle=True` sul `DataLoader` di test.** Non è un errore di per sé, ma
  rende impossibile confrontare le predizioni con le etichette in un ordine
  stabile. Quello che invece **non** cambia è il voto: con `model.eval()`
  l'accuratezza misurata è identica cifra per cifra con il mescolamento acceso
  e con quello spento, perché sono gli stessi esempi in un altro ordine.
  L'unico numero che balla è la media delle medie per batch, che però è un
  errore di conto per conto suo, quello spiegato in [dal notebook agli
  script](dal-notebook-agli-script.md), e si sistema lì.

## Il grafico è il messaggio d'errore

Per gli errori rumorosi c'è il traceback. Per quelli silenziosi c'è un
grafico, ed è l'unico strumento diagnostico che si userà tutti i giorni per
il resto della carriera: le due curve della loss, addestramento e validazione,
epoca per epoca.

Una premessa che sembra banale e non lo è: **quel grafico va guardato
dall'epoca uno, non alla fine**. Aspettare il termine dell'addestramento per
tracciare le curve significa, su un modello serio, scoprire dopo sei ore una
cosa che era leggibile dopo due epoche. Il costo di stampare due numeri a fine
epoca e disegnarli è nullo; il costo di non farlo è una notte di GPU.

```{figure} ../figures/curve-di-addestramento.svg
:name: fig-curve-diagnosi
:alt: "Quattro grafici della loss per epoca, ciascuno con la curva di addestramento e quella di validazione: nel primo restano entrambe alte e piatte, nel secondo scendono insieme con un divario stabile, nel terzo quella di addestramento continua a scendere mentre quella di validazione risale dopo un minimo segnato, nel quarto oscillano entrambe senza una tendenza chiara."
:width: 95%

Quattro forme e quattro diagnosi. La forma della coppia di curve dice più del
valore finale della loss, ed è il motivo per cui va disegnata mentre
l'addestramento gira.
```

Le quattro forme di {numref}`fig-curve-diagnosi` coprono quasi tutto ciò che
capita.

**Non impara.** Le curve restano alte e piatte. Qui conviene guardare da
vicino *quanto* piatte, perché la distinzione è diagnostica. Se la loss è
**esattamente** identica epoca dopo epoca, non è un problema di
apprendimento: è un bug, e i sospetti sono pochi (manca `optimizer.step()`,
il learning rate è zero, i parametri sono congelati da un
`requires_grad=False` di troppo, oppure la loss che si retropropaga non è
collegata all'uscita del modello). Se invece scende pochissimo ma scende, è
**sottoadattamento**: il modello non ha abbastanza capacità, o il passo è
troppo corto, o le feature non contengono l'informazione richiesta.

**Sano.** Entrambe scendono, la validazione sta un po' sopra, e il divario fra
le due resta più o meno costante. Un divario c'è quasi sempre e non è una
malattia: quello che si sorveglia è se **si allarga**.

**Sovradattamento.** L'addestramento continua a scendere, la validazione tocca
un minimo e risale. È la forma già incontrata parlando di
[quando fermarsi](addestramento.md), con l'arresto anticipato come rimedio, e
non la ripetiamo qui. Vale però un'avvertenza: quando il modello ha molti più
parametri che esempi da imparare, quella risalita può non essere la fine della
storia, e continuando ad addestrare la validazione può tornare a scendere una
seconda volta. È il fenomeno della *doppia discesa*, raccontato nel capitolo di
machine learning.

**Passo troppo lungo.** Le curve oscillano vistosamente, magari con una
tendenza generale al ribasso ma con salti che la coprono. L'ottimizzatore sta
scavalcando il minimo a ogni passo, oppure il gradiente è talmente rumoroso da
non indicare più una direzione stabile.

`````{tab} Elementare

La regoletta operativa, in ordine di ciò che conviene provare.

Curva piatta come un tavolo: è un bug, cercalo nel ciclo di addestramento
prima di toccare qualsiasi iperparametro. Curva che scende appena: alza il
learning rate (di un fattore tre, non del dieci per cento) e, se non basta,
ingrandisci il modello. Curva che salta: abbassa il learning rate (di un
fattore tre) o aumenta la dimensione del batch, che è l'altro modo di rendere
meno rumoroso il gradiente. Divario che si allarga: servono i freni, cioè
regolarizzazione, data augmentation o semplicemente fermarsi prima.

E un collaudo che vale i cinque minuti che costa, da fare **prima** di lanciare
l'addestramento vero: prendi un solo mucchietto di esempi, una decina, e
addestra su quello finché la loss non arriva quasi a zero. Se un modello non
riesce a imparare a memoria dieci esempi, non c'è iperparametro che lo salverà
su cinquantamila: c'è un errore da qualche parte, e lo stai cercando nel posto
sbagliato.

`````

`````{tab} Superiore

Il learning rate lascia una firma riconoscibile, e conviene imparare a
leggerla perché è l'iperparametro che si tocca per primo. Troppo alto: la loss
esplode o resta alta e irregolare (nel caso estremo diventa `nan`). Alto: cala
in fretta nelle prime iterazioni e poi si assesta su un valore mediocre,
perché il passo è troppo lungo per entrare nella conca. Troppo basso: scende
in modo quasi lineare e lentissimo, senza mai piegare. Giusto: cala rapidamente
e continua a migliorare. Poiché la scala giusta cambia di ordini di grandezza
fra un problema e l'altro, la si esplora moltiplicando o dividendo per tre,
non aggiustandola di percentuali.

Il rumore delle curve ha due sorgenti distinte, che si distinguono da come si
comporta l'oscillazione. Il **passo** troppo lungo produce oscillazioni che non
si riducono aumentando la dimensione del batch; il **rumore del gradiente**
scala invece come $1/\sqrt{B}$ con la dimensione $B$ del batch, quindi
quadruplicare il batch dimezza l'ampiezza. Se raddoppiare $B$ calma
visibilmente le curve, era rumore di campionamento; se non cambia nulla, è il
passo.

Il collaudo canonico prima di ogni addestramento serio è **sovradattare di
proposito un batch solo**: si prendono otto o dieci esempi, si disattiva ogni
regolarizzazione e si addestra su quelli per qualche centinaio di iterazioni.
Un modello sano arriva a una loss praticamente nulla, perché memorizzare dieci
esempi è alla portata di qualunque rete. Se non ci riesce, il difetto non è nei
dati né negli iperparametri ma nella catena: etichette disallineate rispetto
agli ingressi, una trasformazione che distrugge il segnale, una loss calcolata
sulla cosa sbagliata, un gradiente che non arriva a tutti i parametri. È il
test più economico del mestiere e quasi nessuno lo fa.

`````

### Quando la validazione va meglio dell'addestramento

C'è una forma che confonde chiunque la incontri la prima volta: la curva di
validazione sta **sotto** quella di addestramento. Sembra impossibile, perché
il modello i dati di addestramento li ha visti e quelli di validazione no.
Quasi sempre non c'è nulla di rotto, e le spiegazioni sono quattro.

1. **Si stanno confrontando due misure prese in momenti diversi.** La loss di
   addestramento che si stampa è di solito la media su tutti i batch
   dell'epoca, calcolata *mentre* i pesi cambiavano, e quindi include anche i
   pesi peggiori di inizio epoca. Quella di validazione è calcolata alla fine,
   con i pesi migliori. Nelle prime epoche, quando il miglioramento dentro una
   singola epoca è grande, questo basta da solo a invertire l'ordine.
2. **La regolarizzazione è attiva solo in addestramento.** Il dropout spegne
   neuroni e la data augmentation deforma gli esempi durante il training e non
   in valutazione: la rete che produce la loss di addestramento è una rete
   handicappata, quella che produce la validazione no.
3. **Il set di validazione è più facile.** Può capitare per caso con split
   piccoli, o sistematicamente se la separazione non è stata casuale (i casi
   difficili concentrati da una parte sola).
4. **Si è addestrato poco.** Nelle prime epoche il modello non ha ancora
   memorizzato niente, quindi non ha alcun vantaggio sui dati visti.

Le prime due si verificano in un minuto: si ricalcola la loss di addestramento
a fine epoca con `model.eval()`, sugli stessi pesi con cui si misura la
validazione. Se il paradosso sparisce, non c'era.

```{code-block} python
:class: pt-non-eseguibile

# Registrare la storia costa due liste, e senza storia non c'e' diagnosi.
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

    # stampare a ogni epoca, non alla fine: e' tutto il punto
    print(f"epoca {epoca:3d}  train {storia['train'][-1]:.4f}  val {storia['val'][-1]:.4f}")

import matplotlib.pyplot as plt
plt.plot(storia["train"], label="addestramento")
plt.plot(storia["val"], label="validazione", linestyle="--")
plt.xlabel("epoche"); plt.ylabel("loss"); plt.legend()
```

Tre errori con un messaggio, sette senza, e un metodo che vale per tutti e
dieci. È la sezione a cui si torna, e vale la pena rileggerne il riassunto la
prima volta che qualcosa si rompe.

`````{tab} Elementare
```{admonition} Da ricordare
:class: important
- Un tensore ha **forma, tipo e dispositivo**: quasi ogni errore rosso di
  PyTorch riguarda uno di questi tre, e la prima cosa da fare è stamparli tutti
  e tre, per il tuo dato e per quello che gli sta di fronte.
- **Forma**: manca lo strato che srotola l'immagine, due strati non combaciano,
  oppure manca il "mucchietto" attorno all'esempio singolo.
- **Tipo**: numeri decimali per i dati, numeri interi per le etichette quando
  la risposta è una categoria fra tante. Le immagini appena lette sono fatte di
  interi da 0 a 255 e vanno convertite una volta sola, appena entrano.
- **Dispositivo**: spostare il modello lo sposta davvero; spostare un dato
  restituisce una *copia*, e se non la riassegni non hai fatto niente.
- Il metodo vale più dei rimedi: **leggi l'errore dall'ultima riga in su**,
  stampa forma-tipo-dispositivo, riduci il problema a un esempio solo, e prova
  la catena con un dato finto prima di lanciare l'addestramento vero.
- Gli errori peggiori sono quelli **silenziosi**, quelli senza niente di rosso:
  gli appunti del giro prima non buttati, la trasformazione in probabilità
  fatta due volte, la modalità esame dimenticata, il numero estratto dalla
  perdita dimenticato nell'accumulo.
- Per gli errori silenziosi il messaggio d'errore è **la coppia di curve**, e
  va disegnata dalla prima epoca. Piatta come un tavolo: è un errore nel
  codice. Scende appena: passo corto o modello piccolo. Salta: passo lungo o
  mucchietti troppo piccoli. Distanza fra le due che si allarga: servono i
  freni.
- **Far imparare a memoria dieci esempi** è il collaudo più economico che
  esista: se il modello non ci riesce, il difetto è nel codice, non nelle
  manopole.
- La validazione **sotto** l'addestramento di solito non è un guasto: il numero
  dell'addestramento è una media presa mentre il modello stava ancora
  cambiando, e i trucchi che si usano solo mentre si studia penalizzano
  soltanto lui.
```
`````

`````{tab} Superiore
```{admonition} Da ricordare
:class: important
- Un tensore ha **forma, tipo e dispositivo**: quasi ogni `RuntimeError` di
  PyTorch riguarda uno di questi tre.
- **Forma**: manca `nn.Flatten()`, due strati non combaciano, o manca la
  dimensione del batch (`unsqueeze(0)`). `torchinfo.summary` la mostra strato
  per strato.
- **Tipo**: `float32` per i dati, `int64` per le etichette della
  `CrossEntropyLoss`, `float32` per quelle della `BCEWithLogitsLoss`. NumPy
  produce `float64`: convertire al confine.
- **Dispositivo**: `modello.to(device)` modifica sul posto, `x.to(device)`
  restituisce una copia da riassegnare. I tensori creati nel `forward` vanno
  creati con `device=x.device`; le costanti del modello con
  `register_buffer`.
- Il metodo vale più dei rimedi: **traceback dal basso**, stampa della terna,
  problema ridotto, giro a vuoto con un tensore finto.
- Gli errori peggiori sono quelli **silenziosi**: `zero_grad` mancante (che con
  SGD può far *scendere* la loss più in fretta, non divergere), softmax
  doppia, `eval()` dimenticato, `.item()` dimenticato nell'accumulo.
- Per gli errori silenziosi il messaggio d'errore è **la coppia di curve**, e
  va disegnata dall'epoca uno. Piatta come un tavolo: è un bug. Scende appena:
  passo corto o modello piccolo. Salta: passo lungo o batch piccolo. Divario
  che si allarga: servono i freni.
- **Sovradattare di proposito un batch da dieci esempi** è il collaudo più
  economico che esista: se il modello non ci riesce, il difetto è nella
  catena, non negli iperparametri.
- La validazione **sotto** l'addestramento di solito non è un guasto: la loss
  di training è una media presa mentre i pesi cambiavano, e dropout e
  augmentation penalizzano solo il training.
```
`````
