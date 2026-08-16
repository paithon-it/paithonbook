# Moduli: costruire il modello

Verso la metà degli anni Novanta, Yann LeCun e i suoi colleghi ai Bell Labs
presero decine di migliaia di cifre scritte a mano (raccolte su formulari
cartacei da impiegati dell'ufficio del censimento statunitense e da studenti
delle scuole superiori) le normalizzarono e le centrarono in quadratini di
$28\times 28$ pixel. Ne nacque **MNIST** (*Modified NIST*): 70.000 immagini in
scala di grigi, ciascuna una cifra da $0$ a $9$, divise in 60.000 esempi di
addestramento e 10.000 di test {cite}`lecun1998gradient`. Da allora MNIST è il
"*Hello, world!*" del deep learning: piccolo abbastanza da addestrarsi in
pochi secondi, ricco abbastanza da mostrare tutto il ciclo di vita di un
modello. In questa sezione costruiamo il modello che leggerà quelle cifre;
nella prossima lo addestreremo.

## `nn.Module`: il mattone di ogni rete

In PyTorch qualunque pezzo di rete (un singolo strato, un blocco, il modello
intero) è un **modulo**, cioè una classe che **eredita** da `nn.Module`. (Il
`nn` che si incontrerà in ogni riga di questo capitolo sta per *neural
networks*: è la parte di PyTorch che contiene i pezzi con cui si montano le
reti.) È la scelta di design più caratteristica della libreria: il modello non
si descrive in un elenco a parte da consegnare alla libreria, si *scrive* come
una normale classe Python.

Le classi le abbiamo viste nel capitolo su Python con l'immagine dello stampo
per biscotti: una classe è lo stampo, l'oggetto è il biscotto. *Ereditare*
vuol dire partire da uno stampo che esiste già e aggiungergli qualcosa invece
di intagliarne uno da zero: il nuovo stampo sa fare tutto quello che sapeva
fare il vecchio, più ciò che gli abbiamo aggiunto. Nel codice l'eredità si
scrive mettendo il nome dello stampo di partenza fra parentesi,
`class MLP(nn.Module):`. E la prima riga di `__init__` (che è il metodo
eseguito quando l'oggetto viene creato, quello che lo mette insieme: si chiama
**costruttore**) è `super().__init__()`, la chiamata con cui lo stampo vecchio
si prepara prima che noi ci aggiungiamo il nostro. Va scritta sempre, ed è la
ragione per cui la
si ritroverà, identica, in ogni modello del capitolo. Ciò che si eredita da
`nn.Module` è molto: tenere il conto di tutti i pesi sparsi nella rete,
spostarli tutti insieme sulla scheda grafica, salvarli su un file,
e accendere su ciascuno il registratore di autograd. Sono tutte cose che
nessuno ha voglia di riscrivere ogni volta.

Ecco il modello per MNIST, intero; le righe che contano sono i due metodi, e
li smontiamo subito sotto.

```python
import torch
from torch import nn

class MLP(nn.Module):
    def __init__(self):
        super().__init__()
        self.flatten = nn.Flatten()          # da griglia 28x28 a vettore 784
        self.hidden = nn.Linear(28 * 28, 128)  # ogni ingresso collegato a ogni neurone
        self.out = nn.Linear(128, 10)        # 10 uscite: una per cifra 0-9

    def forward(self, x):
        x = self.flatten(x)
        x = torch.relu(self.hidden(x))       # ReLU: i numeri negativi diventano zero
        return self.out(x)                   # punteggi grezzi, non probabilita'

model = MLP()
print(model)          # elenca i pezzi che compongono il modello
# MLP(
#   (flatten): Flatten(start_dim=1, end_dim=-1)
#   (hidden): Linear(in_features=784, out_features=128, bias=True)
#   (out): Linear(in_features=128, out_features=10, bias=True)
# )
```

`````{tab} Elementare
Due metodi, due domande. In `__init__` rispondi a "**di quali pezzi è fatta**
la rete?": qui uno strato che srotola l'immagine, uno nascosto da 128 neuroni
e uno d'uscita da 10. I 784 in ingresso sono obbligati ($28 \times 28$, i pixel
dell'immagine) e i 10 in uscita pure (le cifre da 0 a 9); il 128 nel mezzo no,
l'abbiamo scelto noi. Un valore più grande dà una rete più capiente e più
lenta, uno più piccolo il contrario: si prova, e come si sceglie è l'argomento
della sezione sul [flusso di lavoro](flusso-di-lavoro.md). In `forward`
rispondi a "**che strada fanno i dati**?": entra l'immagine, viene srotolata,
passa per lo strato nascosto e poi per la ReLU (che è un filtro semplicissimo:
lascia passare i numeri positivi e schiaccia a zero i negativi), ed esce come
10 punteggi, uno per cifra. Tutto qui: il resto (tenere traccia dei pesi,
calcolare i gradienti) lo fa `nn.Module` per conto tuo. E siccome `forward` è
normale Python, puoi metterci un `print` per sbirciare, o un `if` per cambiare
strada: il modello è codice che gira, non una descrizione da consegnare a
qualcun altro.
`````

`````{tab} Superiore
`nn.Module` fornisce la contabilità dei **parametri**: ogni attributo che sia
a sua volta un modulo (o un `nn.Parameter`) viene registrato automaticamente,
e `model.parameters()` restituisce l'iteratore su tutti i tensori addestrabili
(quello che passeremo all'ottimizzatore). `nn.Linear(d, u)` realizza la
trasformazione affine

$$
\mathbf{h} = \mathbf{W}\mathbf{x} + \mathbf{b},
$$

con $\mathbf{W} \in \mathbb{R}^{u \times d}$ e $\mathbf{b} \in \mathbb{R}^{u}$
creati con `requires_grad=True`: autograd li traccia senza che si debba fare
nulla.
Si noti che l'attivazione non è "dentro" lo strato, come accade in altre
librerie: è una funzione (`torch.relu`) o un modulo (`nn.ReLU`) applicato
esplicitamente in `forward`; coerente con la filosofia "il modello è il
codice". La chiamata `model(x)` invoca `forward` attraverso `__call__`, che
aggiunge gli *hook* di libreria: per questo non si chiama mai
`model.forward(x)` direttamente.
`````

## La scorciatoia: `nn.Sequential`

Quando la rete è una semplice catena (l'uscita di uno strato entra nel
successivo, senza rami), la classe si può evitare del tutto:

```python
model = nn.Sequential(
    nn.Flatten(),
    nn.Linear(28 * 28, 128),
    nn.ReLU(),
    nn.Linear(128, 10),
)
```

`````{tab} Elementare
Immagina una catena di montaggio: ogni stazione riceve il pezzo dalla
precedente, ci lavora sopra e lo passa alla successiva. `nn.Sequential`
descrive la rete esattamente così: elenchi le stazioni nell'ordine in cui il
dato le attraversa, e i collegamenti si fanno da soli. Nota che qui anche la
ReLU è una "stazione" della catena (`nn.ReLU()`): nella catena di montaggio
ogni passaggio, filtri compresi, deve avere il suo posto in fila.

Quale delle due scritture usare? Questa, finché la rete è una fila. Si torna
alla classe il giorno in cui la fila non basta più, cioè quando il dato deve
prendere due strade e ricongiungersi dopo, o saltare qualche stazione: cose
che in una lista non si scrivono, e in `forward` sì, perché lì sono normali
variabili Python.
`````

`````{tab} Superiore
`nn.Sequential` modella una funzione composta
$f = f_L \circ \dots \circ f_2 \circ f_1$, dove la lista ne fissa l'ordine di
composizione. È adatta a topologie *lineari* (un ingresso, un'uscita, nessuna
ramificazione); per più input, skip connection o rami paralleli (come le
ResNet che incontreremo nel capitolo sul deep learning) si torna a `nn.Module`
con un `forward` esplicito, dove le ramificazioni sono semplici variabili
Python. È la differenza chiave rispetto alle API dichiarative: non serve
un’"API funzionale" separata, perché la composizione arbitraria è già Python.
Per MNIST la pila lineare basta e avanza.
`````

## Quanti parametri ha questa rete?

I **parametri** sono i numeri che il modello impara, quelli che l'addestramento
regolerà: pesi e bias tutti insieme, cioè le manopole di cui si è parlato
nella sezione sui tensori. Contarli è il primo controllo da fare su qualunque
modello, prima ancora di addestrarlo: se il numero non è quello che ci si
aspetta, la rete montata non è quella che si aveva in mente.

`````{tab} Elementare
Ogni collegamento tra un ingresso e un neurone ha il suo peso, più un piccolo
termine di aggiustamento (il *bias*) per neurone. Lo strato nascosto collega
784 ingressi a 128 neuroni: $784 \times 128 + 128 = 100\,480$ numeri da
imparare. Lo strato d'uscita: $128 \times 10 + 10 = 1\,290$. In tutto
$101\,770$ manopole che l'addestramento dovrà regolare, tante, ma una rete
moderna ne ha miliardi: MNIST è davvero una palestra in miniatura.
`````

`````{tab} Superiore
Per `nn.Linear(d, u)` i parametri sono $u \cdot d + u$. Verifichiamolo:

```python
n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
print(n_params)   # 101770
```

`p.numel()` conta gli elementi di ciascun tensore; il filtro su
`requires_grad` esclude eventuali parti congelate (tornerà utile nel
*transfer learning*). La pila per MNIST è riassunta in
{numref}`fig-mlp-mnist`: $784 \cdot 128 + 128 = 100\,480$ per lo strato
nascosto, $128 \cdot 10 + 10 = 1\,290$ per l'uscita.
`````

```{figure} ../figures/mlp-mnist.svg
:name: fig-mlp-mnist
:alt: "Pila verticale di cinque blocchi collegati da frecce: l'immagine 28 per 28 in ingresso, Flatten verso 784, Linear 784-128 con ReLU, Linear 128-10 che produce i logit, e in fondo la softmax. Una graffa laterale marca come nn.Sequential soltanto i tre blocchi centrali: l'ingresso e la softmax restano fuori dal modello."
:width: 70%

Il percettrone multistrato per MNIST: l'immagine viene srotolata in 784
numeri, compressa a 128, infine proiettata su 10 punteggi grezzi, uno per
cifra. La trasformazione di quei punteggi in probabilità (si chiama *softmax*)
non sta nel modello: dove stia lo dicono le prossime righe.
```

## Misurare l'errore: le funzioni di perdita

Il modello ora esiste, ma è ignorante: i pesi sono numeri casuali. Per
addestrarlo serve prima di tutto un modo di misurare *quanto sbaglia*: la
funzione di perdita, o **loss**. `torch.nn` le offre come moduli pronti, e le
due che useremo più spesso coprono i due grandi casi: quando la risposta
giusta è un numero, e quando è una scelta fra categorie.

```python
loss_regressione = nn.MSELoss()            # per predire numeri continui
loss_classi = nn.CrossEntropyLoss()        # per scegliere tra classi

# esempio: 2 immagini finte date in pasto al modello ancora ignorante.
# (2, 1, 28, 28) = 2 immagini, 1 canale (MNIST e' in scala di grigi), 28x28 pixel
logits = model(torch.randn(2, 1, 28, 28))  # shape (2, 10): 10 punteggi per immagine
target = torch.tensor([3, 7])              # le cifre vere sono un 3 e un 7
errore = loss_classi(logits, target)       # un numero solo: la loss media
print(errore.item())                       # circa 2,3 (con due sole immagini balla)
```

Quel $2{,}3$ non è un numero qualunque, ed è il metro con cui leggeremo tutte
le loss di questo capitolo: è quanto vale la cross-entropy per un modello che
tira a indovinare fra dieci cifre, cioè che dà a ciascuna una probabilità su
dieci. Un addestramento che funziona parte da lì e scende; uno che resta a
$2{,}3$ non ha imparato niente. (Il valore esatto dipende dai pesi casuali di
partenza, e su due sole immagini oscilla fra $2$ e $2{,}5$: è la media su
tante immagini che si assesta.)

`````{tab} Elementare
La **MSE** (errore quadratico medio) serve quando la risposta è un numero, il
prezzo di una casa, la temperatura di domani: misura la distanza tra
predizione e verità, ed eleva al quadrato per punire di più gli errori grossi.
Con i numeri: sbagliare di $2$ costa $4$, sbagliare di $10$ costa $100$. Un
errore cinque volte più grande non ne vale cinque, ne vale venticinque, e il
modello lo sente.

La **cross-entropy** serve quando la risposta è una scelta, quale cifra, quale
animale: guarda quanta fiducia il modello ha dato alla risposta giusta, e lo
punisce tanto più quanto era sicuro di quella sbagliata. Anche qui con i
numeri: se alla cifra giusta ha dato il 90% di fiducia paga $0{,}11$, se le ha
dato il 10% (cioè ha tirato a indovinare) paga $2{,}3$, e se le ha dato l'1%
paga $4{,}6$. La penalità non cresce in proporzione: precipita verso l'alto man
mano che il modello esclude la risposta vera.

Un dettaglio pratico che sorprende chiunque inizi: alla `CrossEntropyLoss` di
PyTorch si danno i punteggi grezzi (i *logit*, che è il nome tecnico di quei
numeri prima che diventino probabilità), **non** le probabilità; la
trasformazione la fa lei, al suo interno. Il motivo, in breve, è che fare i due
conti insieme è più preciso che farli uno dopo l'altro: con probabilità
piccolissime il secondo passaggio perderebbe cifre per strada. Conseguenza da
tenere a mente: nel modello la softmax **non ci va**, e metterla lì è uno degli
errori silenziosi più comuni.
`````

`````{tab} Superiore
Per la regressione, `nn.MSELoss` calcola

$$
\mathcal{L} = \frac{1}{N D} \sum_{i=1}^{N} \sum_{k=1}^{D}
              (\hat{y}_{ik} - y_{ik})^2,
$$

dove $i$ scorre gli $N$ esempi del batch e $k$ le $D$ uscite di ciascun
esempio: la media è su **tutti gli elementi** del tensore, non sugli esempi.
Quando l'uscita è una sola, come qui, le due letture coincidono e la
distinzione non si vede; in regressione multi-uscita no. Chi somma i quadrati
di un esempio e poi media sugli esempi ottiene un numero $D$ volte più grande
di quello che restituisce il modulo: misurato su forme $(4, 3)$, $6{,}7676$
contro $2{,}2559$. Il punto di minimo è lo stesso, la scala del gradiente no,
e con essa il learning rate che serviva. Per la classificazione a $K$ classi,
`nn.CrossEntropyLoss` combina in un solo modulo `LogSoftmax` e `NLLLoss`: dati
i logit $z_1, \dots, z_K$ e la classe vera $c$ di un singolo esempio,

$$
\mathcal{L} = -\log \hat{y}_c,
\qquad
\hat{y}_k = \frac{e^{z_k}}{\sum_{j=1}^{K} e^{z_j}} .
$$

dove qui $k$ e $j$ scorrono le $K$ classi, non gli esempi. Sul batch il modulo
restituisce la **media** di questi termini sugli $N$ esempi
(`reduction='mean'`, il default): è il "numero solo" del codice qui sopra, e
qui la media è davvero per esempio, perché di termini ce n'è uno per esempio. Applicarla ai logit, e non a probabilità già normalizzate, non è un
capriccio: il calcolo congiunto del logaritmo e della softmax è numericamente
più stabile (evita underflow con il *log-sum-exp trick*), e per questo
l'ultimo strato del modello **non** deve avere la softmax. Se servono le
probabilità (per leggere l'output, non per addestrare), si applica
`torch.softmax(logits, dim=1)` a valle. Con etichette intere il target ha
shape $(N,)$ e dtype `int64`, non serve il one-hot.
`````

Le due penalità che abbiamo appena visto in cifre hanno anche una forma, e
metterle una accanto all'altra dice in un colpo d'occhio quello che i numeri
dicono uno alla volta ({numref}`fig-mse-vs-crossentropy`). Attenzione a come si
legge: **sono due disegni distinti, con due cose diverse sull'asse
orizzontale**, e non due curve sovrapposte. A sinistra scorre l'errore, cioè di
quanto la predizione ha mancato il valore vero; a destra scorre la fiducia che
il modello ha dato alla risposta giusta, da zero (l'ha esclusa) a uno (ne era
certo). In verticale, in tutti e due, la penalità.

```{figure} ../figures/loss-function-cosa-ottimizziamo.svg
:name: fig-mse-vs-crossentropy
:alt: "Due grafici affiancati. A sinistra, la MSE: una parabola con il minimo nello zero, sull'asse orizzontale l'errore fra predizione e valore vero. A destra, la cross-entropy: una curva che scende da valori altissimi vicino allo zero fino a zero in uno, sull'asse orizzontale la probabilità assegnata alla classe vera."
:width: 96%

Due disegni, due assi orizzontali diversi, due caratteri. La parabola perdona
gli errori piccoli; la cross-entropy non perdona la sicurezza sbagliata, e
cresce senza limite man mano che il modello esclude la risposta giusta.
```

È il comportamento agli estremi, e non altro, la ragione per cui in
classificazione si sceglie la seconda. Lì ciò che deve fare male non è
sbagliare di poco, è essere convinti del contrario: la cross-entropy è
costruita esattamente per questo.

Il modello esiste e sa dire quanto sbaglia. Manca chi usa quel numero per
correggerlo, ed è l'argomento della sezione seguente.

`````{tab} Elementare
```{admonition} Da ricordare
:class: important
- Ogni pezzo di rete è un **modulo**: in `__init__` si elencano i componenti,
  in `forward` si dice che strada fanno i dati. È normale codice Python, quindi
  ci si può mettere un `print` per sbirciare.
- **`nn.Sequential`** è la scorciatoia quando la rete è una catena di
  montaggio; se ci sono rami o scorciatoie, si torna a scrivere `forward` a
  mano.
- Uno strato che collega $d$ ingressi a $u$ neuroni ha $u \cdot d + u$ numeri
  da imparare: un peso per collegamento, più un aggiustamento per neurone.
  Contarli è il primo controllo da fare su qualunque modello, e costa una
  moltiplicazione per strato.
- La **funzione di perdita** misura quanto il modello sbaglia: `nn.MSELoss`
  quando la risposta è un numero, `nn.CrossEntropyLoss` quando è una scelta fra
  categorie. A quest'ultima si danno i punteggi grezzi, non le probabilità: la
  trasformazione la fa lei.
```
`````

`````{tab} Superiore
```{admonition} Da ricordare
:class: important
- Ogni pezzo di rete è un **`nn.Module`**: in `__init__` i componenti, in
  `forward` la strada dei dati (normale Python, ispezionabile riga per riga).
- **`nn.Sequential`** è la scorciatoia per le catene semplici; per topologie
  con rami si scrive il `forward` a mano.
- `nn.Linear(d, u)` calcola $\mathbf{W}\mathbf{x}+\mathbf{b}$ e ha
  $u \cdot d + u$ parametri; `model.parameters()` li consegna
  all'ottimizzatore, il componente che nella prossima sezione applicherà le
  correzioni.
- Le loss sono moduli: `nn.MSELoss` per la regressione, `nn.CrossEntropyLoss`
  per la classificazione; quest'ultima **vuole i logit**, la softmax ce l'ha
  dentro.
```
`````
