# Il training loop: addestrare un modello

Chi programma in PyTorch riconosce a colpo d'occhio cinque righe che tornano,
identiche, in ogni progetto — dal tutorial per principianti al codice che
addestra i grandi modelli linguistici. Sono il **training loop**, e il fatto
che si scrivano *a mano* non è una dimenticanza della libreria: è una presa di
posizione. Dove altri framework nascondono l'addestramento dietro un unico
comando, PyTorch preferisce che ogni passo — previsione, errore, gradiente,
correzione — resti visibile e modificabile. È più codice, ma è *tuo*: quando
vorrai cambiare qualcosa nel modo di apprendere, saprai esattamente dove
mettere le mani.

## Il rito: i cinque passi

Eccole, le cinque righe. Tutto il resto della sezione non fa che spiegarle e
metterle al lavoro su un problema vero.

```{code-block} python
:class: pt-non-eseguibile

for X_batch, y_batch in dataloader:
    y_pred = model(X_batch)             # 1. forward: la previsione
    loss = criterion(y_pred, y_batch)   # 2. loss: quanto abbiamo sbagliato
    optimizer.zero_grad()               # 3. azzera i gradienti vecchi
    loss.backward()                     # 4. backward: calcola i gradienti
    optimizer.step()                    # 5. aggiorna i pesi
```

`````{tab} Elementare
È il metodo con cui si impara con le flashcard, le carte per memorizzare.
Guardi la domanda e provi a rispondere (passo 1, la previsione). Giri la
carta e confronti con la risposta giusta: quanto eri lontano? (passo 2,
l'errore). Butti via gli appunti del giro precedente (passo 3), capisci *in
che direzione* hai sbagliato — troppo alto? troppo basso? (passo 4) — e
aggiusti di conseguenza il tuo modo di rispondere, un poco alla volta (passo
5). Poi passi al mazzetto successivo — e quando hai ripassato l'intero mazzo
una volta, hai completato quella che si chiama un'**epoca**. Ripetuto per
migliaia di carte ed epoche, questo giro è tutto ciò che serve a una rete per
imparare.
`````

`````{tab} Superiore
Il loop realizza un passo di discesa del gradiente su mini-batch. Con
$\mathcal{L}$ la loss media sul batch e $\theta$ i parametri:

$$
\theta \leftarrow \theta - \eta \, \nabla_{\theta} \mathcal{L},
$$

dove $\eta$ è il *learning rate*. `loss.backward()` calcola
$\nabla_{\theta}\mathcal{L}$ via autograd e lo deposita in `p.grad` per ogni
parametro; `optimizer.step()` applica l'aggiornamento — la formula esatta
dipende dall'ottimizzatore: la discesa semplice per `optim.SGD`, stime
adattive dei momenti per `optim.Adam` {cite}`kingma2015adam`, il default
robusto di quasi ogni progetto. `zero_grad()` è necessario perché autograd
**accumula** i gradienti a ogni `backward()`: senza azzeramento, ogni passo
userebbe la somma di tutti i gradienti precedenti. L'ordine dei passi 3–5 è
l'unica liturgia da rispettare; tutto il resto è normale Python, e infatti
qui si innestano senza attrito *gradient clipping*, *scheduler* del learning
rate, *mixed precision*.
`````

## `Dataset` e `DataLoader`: la catena di rifornimento

Le reti non mangiano il dataset intero in un boccone, né un esempio alla
volta: mangiano **mini-batch**, pacchetti di qualche decina di esempi. A
prepararli ci pensano due classi di `torch.utils.data`.

```python
from torch.utils.data import DataLoader
from torchvision import datasets, transforms

# MNIST scaricato e trasformato in tensori con valori in [0, 1]
train_data = datasets.MNIST(root="data", train=True, download=True,
                            transform=transforms.ToTensor())
test_data = datasets.MNIST(root="data", train=False, download=True,
                           transform=transforms.ToTensor())

train_loader = DataLoader(train_data, batch_size=64, shuffle=True)
test_loader = DataLoader(test_data, batch_size=256)
```

`````{tab} Elementare
Il `Dataset` è la dispensa: sa quanti esempi ci sono e sa consegnarti
l'esempio numero $i$ quando glielo chiedi. Il `DataLoader` è il cameriere che
apparecchia: pesca dalla dispensa, **mescola** l'ordine a ogni giro (così la
rete non impara la sequenza a memoria, come uno studente che ripassa sempre
le carte nello stesso ordine) e porta in tavola vassoi da 64 esempi alla
volta. Perché proprio a vassoi? Un esempio alla volta è uno spreco — la GPU
resta ferma ad aspettare; tutti insieme non entrano in memoria. Il
mini-batch è la via di mezzo che tiene la cucina sempre piena.
`````

`````{tab} Superiore
`Dataset` (variante *map-style*) è un protocollo minimo: `__len__` e
`__getitem__`. Qualunque classe che li implementi — un file CSV, una
cartella di immagini, un database — diventa una sorgente per il
`DataLoader`, che aggiunge campionamento (`shuffle=True` rimescola gli
indici a ogni epoca), *batching* (impila gli esempi lungo il primo asse: qui
tensori $(64, 1, 28, 28)$), e caricamento parallelo (`num_workers`) con
trasferimento asincrono verso la GPU (`pin_memory=True`). La `transform`
`ToTensor()` converte le immagini PIL in tensori `float32` con valori in
$[0, 1]$ e layout channels-first $(C, H, W)$; per MNIST si può aggiungere
`transforms.Normalize((0.1307,), (0.3081,))` — media e deviazione standard
del dataset — per centrare gli input, come visto nel capitolo sulle reti
neurali. Statisticamente, il gradiente su un mini-batch è una stima non
distorta ma rumorosa del gradiente vero: il rumore è il prezzo (e in parte
il segreto) della discesa *stocastica*.
`````

## MNIST da cima a fondo

Mettiamo insieme tutto quello che il capitolo ha costruito: tensori, modello,
loss, dati. Questo è un programma completo che scarica MNIST, addestra il
percettrone multistrato della sezione precedente e lo valuta su immagini mai
viste.

```{code-block} python
:class: pt-lento

import torch
from torch import nn, optim
from torch.utils.data import DataLoader
from torchvision import datasets, transforms

device = "cuda" if torch.cuda.is_available() else "cpu"

# --- dati ---
train_data = datasets.MNIST(root="data", train=True, download=True,
                            transform=transforms.ToTensor())
test_data = datasets.MNIST(root="data", train=False, download=True,
                           transform=transforms.ToTensor())
train_loader = DataLoader(train_data, batch_size=64, shuffle=True)
test_loader = DataLoader(test_data, batch_size=256)

# --- modello, loss, ottimizzatore ---
model = nn.Sequential(
    nn.Flatten(),
    nn.Linear(28 * 28, 128),
    nn.ReLU(),
    nn.Linear(128, 10),
).to(device)

criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=1e-3)

# --- addestramento ---
for epoca in range(5):
    model.train()                        # modalità addestramento
    for X, y in train_loader:
        X, y = X.to(device), y.to(device)
        y_pred = model(X)
        loss = criterion(y_pred, y)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

    # --- valutazione a fine epoca ---
    model.eval()                         # modalità valutazione
    corretti = 0
    with torch.no_grad():                # niente gradienti: solo lettura
        for X, y in test_loader:
            X, y = X.to(device), y.to(device)
            y_pred = model(X)
            corretti += (y_pred.argmax(dim=1) == y).sum().item()

    print(f"epoca {epoca + 1}: accuratezza sul test {corretti / len(test_data):.3f}")
```

Cinque epoche, meno di un minuto su un computer qualunque, e l'accuratezza
sul test arriva attorno al **97–98%**: novantasette cifre su cento lette
correttamente da $101\,770$ numeri che un'ora fa erano casuali.

## Studiare e dare l'esame: `train()` ed `eval()`

Nel programma compaiono due chiamate su cui vale la pena fermarsi:
`model.train()` e `model.eval()`, con il blocco `torch.no_grad()`.

`````{tab} Elementare
La rete ha due modalità, come uno studente. Quando **studia**
(`model.train()`) può usare trucchi che servono solo a imparare meglio — per
esempio coprirsi a caso qualche appunto per non adagiarsi (il *dropout* che
vedremo nel prossimo capitolo). Quando **dà l'esame** (`model.eval()`) i
trucchi si spengono: risponde e basta, al meglio di quel che sa. E
`torch.no_grad()` dice al registratore dei gradienti di spegnersi: durante
l'esame non si prende appunti per migliorare, si risponde soltanto — e senza
il registratore acceso tutto è più veloce e leggero.
`````

`````{tab} Superiore
`train()`/`eval()` commutano un flag che cambia il comportamento dei moduli
"a doppia personalità": `nn.Dropout` (attivo solo in training) e
`nn.BatchNorm` (statistiche del batch in training, medie mobili in
valutazione) sono i due casi principali. `torch.no_grad()` è un context
manager che sospende la costruzione del grafo autograd: dimezza circa la
memoria e accelera l'inferenza, perché non vengono salvati i valori
intermedi per un `backward()` che non arriverà mai. Sono due meccanismi
indipendenti e servono entrambi: `eval()` senza `no_grad()` dà predizioni
corrette ma spreca memoria; `no_grad()` senza `eval()` lascia il dropout
acceso e falsa le predizioni. Il nostro MLP non ha né dropout né batch norm,
quindi qui `eval()` è tecnicamente superfluo — ma scriverlo sempre è
un'abitudine che evita bug sottili appena il modello cresce.
`````

## Quando fermarsi: la validazione

Il numero stampato a fine epoca merita rispetto: è la differenza tra
*imparare* e *imparare a memoria*.

```{figure} ../figures/curve-overfitting-validazione.svg
:name: fig-curve-overfitting
:alt: Due curve di perdita in funzione delle epoche. La curva di addestramento scende con continuità; quella di validazione scende, tocca un minimo e poi risale. Una linea tratteggiata verticale segna il punto di arresto anticipato in corrispondenza del minimo della validazione.
:width: 85%

La perdita di addestramento scende sempre; quella di validazione tocca un
minimo e poi risale. Da lì in poi il modello memorizza il rumore: la linea
dell'arresto anticipato marca il momento giusto per fermarsi.
```

`````{tab} Elementare
Guarda le due curve in {numref}`fig-curve-overfitting`. Quella
dell'addestramento è come i voti nei compiti fatti a casa: migliorano sempre,
perché il modello rivede gli stessi esercizi. Quella della validazione è la
simulazione d'esame con domande nuove. All'inizio migliorano insieme — buon
segno. Poi la validazione si ferma e comincia a peggiorare, mentre
l'addestramento continua a salire: da lì in avanti il modello sta imparando a
memoria, è l'**overfitting** che abbiamo incontrato nel capitolo sul machine
learning. La mossa giusta è fermarsi al punto migliore della validazione — e
tenere da parte la copia del modello salvata in quel momento.
`````

`````{tab} Superiore
Nel loop esplicito la diagnosi si scrive da sé: si ritaglia un set di
validazione (ad esempio con `torch.utils.data.random_split(train_data,
[55000, 5000])`), a fine epoca si misura $\mathcal{L}_{\text{val}}$, e
l'*early stopping* è un `if`: se la validazione non migliora da $k$ epoche
(la *patience*), si esce dal ciclo e si ricaricano i pesi dell'epoca
migliore, salvati via via con `torch.save`. Ciò che Keras offriva come
callback preconfezionate, in PyTorch sono sei righe di controllo di flusso —
in cambio, nessun limite: fermarsi su una metrica composta, salvare solo a
condizioni particolari, riprendere da checkpoint sono varianti banali dello
stesso `if`. Il divario $\mathcal{L}_{\text{val}} -
\mathcal{L}_{\text{train}}$ resta la bussola: se si allarga, servono i freni
(regolarizzazione L2 via `weight_decay` dell'ottimizzatore, `nn.Dropout`) che
approfondiremo nel capitolo sul deep learning.
`````

## Salvare il lavoro: lo `state_dict`

Un modello addestrato va messo al sicuro. In PyTorch non si salva l'oggetto
modello: si salva il suo **`state_dict`**, il dizionario di tutti i pesi.

```python
torch.save(model.state_dict(), "mnist_mlp.pt")     # salva i pesi

model2 = nn.Sequential(                            # stessa architettura...
    nn.Flatten(), nn.Linear(784, 128), nn.ReLU(), nn.Linear(128, 10)
)
model2.load_state_dict(torch.load("mnist_mlp.pt")) # ...pesi ricaricati
model2.eval()                                      # pronto per l'uso
```

Il codice che definisce l'architettura resta la fonte di verità; il file
`.pt` contiene solo i numeri. È una divisione dei compiti coerente con tutto
il capitolo — il modello è codice, i pesi sono dati — ed è il formato in cui
circolano i modelli pre-addestrati che riutilizzeremo nel capitolo sulla
visione artificiale, quando un modello nato per un compito verrà rifinito
(*fine-tuning*) su un altro.

```{admonition} Da ricordare
:class: important
- Il **training loop** ha cinque passi fissi: forward → loss →
  `zero_grad()` → `backward()` → `step()`. L'azzeramento serve perché i
  gradienti si accumulano.
- `Dataset` consegna gli esempi, `DataLoader` li rimescola e li impila in
  **mini-batch**: il gradiente sul batch è una stima rumorosa ma economica di
  quello vero.
- In valutazione: `model.eval()` spegne dropout e batch norm,
  `torch.no_grad()` spegne autograd — servono entrambi.
- Le curve di training e validazione diagnosticano l'**overfitting**;
  l'early stopping in PyTorch è un semplice `if` nel loop.
- Si salva lo **`state_dict`** (`torch.save`/`load_state_dict`), non
  l'oggetto: il modello è codice, i pesi sono dati.
```
