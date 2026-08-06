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
intero) è un **modulo**, cioè una classe che eredita da `nn.Module`. È la
scelta di design più caratteristica della libreria: il modello non si
"dichiara", si *scrive* come una normale classe Python.

```python
import torch
from torch import nn

class MLP(nn.Module):
    def __init__(self):
        super().__init__()
        self.flatten = nn.Flatten()          # da griglia 28x28 a vettore 784
        self.hidden = nn.Linear(28 * 28, 128)  # strato completamente connesso
        self.out = nn.Linear(128, 10)        # 10 uscite: una per cifra 0-9

    def forward(self, x):
        x = self.flatten(x)
        x = torch.relu(self.hidden(x))       # attivazione ReLU
        return self.out(x)                   # logit grezzi, senza softmax

model = MLP()
print(model)          # stampa la struttura dei sottomoduli
```

`````{tab} Elementare
Due metodi, due domande. In `__init__` rispondi a "**di quali pezzi è fatta**
la rete?": qui uno strato che srotola l'immagine, uno nascosto da 128 neuroni
e uno d'uscita da 10. In `forward` rispondi a "**che strada fanno i dati**?":
entra l'immagine, viene srotolata, passa per lo strato nascosto col suo filtro
ReLU, esce come 10 punteggi (uno per cifra). Tutto qui: il resto (tenere
traccia dei pesi, calcolare i gradienti) lo fa `nn.Module` per conto tuo. E
siccome `forward` è normale Python, puoi metterci un `print` per sbirciare, o
un `if` per cambiare strada: il modello è codice, non un modulo da compilare.
`````

`````{tab} Superiore
`nn.Module` fornisce la contabilità dei **parametri**: ogni attributo che sia
a sua volta un modulo (o un `nn.Parameter`) viene registrato automaticamente,
e `model.parameters()` restituisce l'iteratore su tutti i tensori addestrabili
(quello che passeremo all'ottimizzatore). `nn.Linear(d, u)` realizza la
trasformazione affine

$$
\mathbf{h} = W\mathbf{x} + \mathbf{b},
$$

con $W \in \mathbb{R}^{u \times d}$ e $\mathbf{b} \in \mathbb{R}^{u}$ creati
con `requires_grad=True`: autograd li traccia senza che si debba fare nulla.
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
`````

`````{tab} Superiore
`nn.Sequential` modella una funzione composta
$f = f_L \circ \dots \circ f_2 \circ f_1$, dove la lista ne fissa l'ordine di
composizione. È adatta a topologie *lineari* (un ingresso, un'uscita, nessuna
ramificazione); per più input, skip connection o rami paralleli (come le
ResNet che incontreremo nel capitolo sul deep learning) si torna a `nn.Module`
con un `forward` esplicito, dove le ramificazioni sono semplici variabili
Python. È la differenza chiave rispetto alle API dichiarative: non serve
un'"API funzionale" separata, perché la composizione arbitraria è già Python.
Per MNIST la pila lineare basta e avanza.
`````

## Quanti parametri ha questa rete?

Contare i parametri è il primo controllo di sanità su qualunque modello.

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
:alt: "Pila verticale di cinque blocchi: immagine 28 per 28, Flatten verso 784, Linear 128 con ReLU, Linear 10, output a 10 punteggi trasformati in probabilità."
:width: 70%

Il percettrone multistrato per MNIST: l'immagine viene srotolata in 784
numeri, compressa a 128 attivazioni, infine proiettata su 10 punteggi (i
*logit*), che la softmax (dentro la loss) trasforma in probabilità.
```

## Misurare l'errore: le funzioni di perdita

Il modello ora esiste, ma è ignorante: i pesi sono numeri casuali. Per
addestrarlo serve prima di tutto un modo di misurare *quanto sbaglia*: la
funzione di perdita, o **loss**. `torch.nn` le offre come moduli pronti; le
due che useremo più spesso coprono i due grandi casi.

```{figure} ../figures/loss-function-cosa-ottimizziamo.svg
:name: fig-mse-vs-crossentropy
:alt: "Due curve a confronto. L'errore quadratico medio cresce come una parabola all'allontanarsi della predizione dal valore vero, penalizzando poco gli errori piccoli. La cross-entropy invece diverge: quando il modello assegna probabilità quasi nulla alla classe giusta, la penalità tende all'infinito."
:width: 96%

Due forme, due caratteri. La parabola perdona gli errori piccoli; la
cross-entropy non perdona la sicurezza sbagliata, e cresce senza limite quando
il modello esclude la risposta giusta.
```

Il comportamento agli estremi mostrato in {numref}`fig-mse-vs-crossentropy` è
la ragione della scelta, più della matematica che le distingue. In
classificazione ciò che deve fare male non è sbagliare di poco, è essere
convinti del contrario: la cross-entropy è costruita esattamente per questo.

```python
loss_regressione = nn.MSELoss()            # per predire numeri continui
loss_classi = nn.CrossEntropyLoss()        # per scegliere tra classi

# esempio: 2 immagini, 10 logit ciascuna, etichette vere 3 e 7
logits = model(torch.randn(2, 1, 28, 28))  # shape (2, 10)
target = torch.tensor([3, 7])
errore = loss_classi(logits, target)       # un numero solo: la loss media
```

`````{tab} Elementare
La **MSE** (errore quadratico medio) serve quando la risposta è un numero, il
prezzo di una casa, la temperatura di domani: misura la distanza tra
predizione e verità, ed eleva al quadrato per punire di più gli errori grossi.
La **cross-entropy** serve quando la risposta è una scelta, quale cifra, quale
animale: guarda quanta fiducia il modello ha dato alla risposta giusta, e lo
punisce tanto più quanto era sicuro di quella sbagliata. Un dettaglio pratico
che sorprende chiunque inizi: alla `CrossEntropyLoss` di PyTorch si danno i
punteggi grezzi (i *logit*), **non** le probabilità; la trasformazione in
probabilità (la softmax) la fa lei, al suo interno.
`````

`````{tab} Superiore
Per la regressione, `nn.MSELoss` calcola
$\mathcal{L} = \frac{1}{N}\sum_{i=1}^{N} (\hat{y}_i - y_i)^2$.
Per la classificazione a $K$ classi, `nn.CrossEntropyLoss` combina in un solo
modulo `LogSoftmax` e `NLLLoss`: dati i logit $z_i$ e la classe vera $c$,

$$
\mathcal{L} = -\log \hat{y}_c,
\qquad
\hat{y}_i = \frac{e^{z_i}}{\sum_{j=1}^{K} e^{z_j}} .
$$

Applicarla ai logit, e non a probabilità già normalizzate, non è un capriccio:
il calcolo congiunto del logaritmo e della softmax è numericamente più stabile
(evita underflow con il *log-sum-exp trick*) e per questo l'ultimo strato del
modello **non** deve avere la softmax. Se servono le probabilità (per leggere
l'output, non per addestrare), si applica `torch.softmax(logits, dim=1)` a
valle. Con etichette intere il target ha shape $(N,)$ e dtype `int64`, non
serve il one-hot.
`````

```{admonition} Da ricordare
:class: important
- Ogni pezzo di rete è un **`nn.Module`**: in `__init__` i componenti, in
  `forward` la strada dei dati (normale Python, ispezionabile riga per riga).
- **`nn.Sequential`** è la scorciatoia per le catene semplici; per topologie
  con rami si scrive il `forward` a mano.
- `nn.Linear(d, u)` calcola $W\mathbf{x}+\mathbf{b}$ e ha $u \cdot d + u$
  parametri; `model.parameters()` li consegna all'ottimizzatore.
- Le loss sono moduli: `nn.MSELoss` per la regressione, `nn.CrossEntropyLoss`
  per la classificazione; quest'ultima **vuole i logit**, la softmax ce l'ha
  dentro.
```
