# Il training loop: addestrare un modello

Chi programma in PyTorch riconosce a colpo d'occhio cinque righe che tornano,
identiche, in ogni progetto: dal tutorial per principianti al codice che
addestra i grandi modelli linguistici. Sono il **training loop**, e il fatto
che si scrivano *a mano* non è una dimenticanza della libreria: è una presa di
posizione. Dove altri framework nascondono l'addestramento dietro un unico
comando, PyTorch preferisce che ogni passo (previsione, errore, gradiente,
correzione) resti visibile e modificabile. È più codice, ma è *tuo*: quando
vorrai cambiare qualcosa nel modo di apprendere, saprai esattamente dove
mettere le mani.

## Il rito: i cinque passi

Eccole, le cinque righe. Tutto il resto della sezione non fa che spiegarle e
metterle al lavoro su un problema vero. Prima, due parole che nel codice
compaiono senza presentazioni. L'**ottimizzatore** è il pezzo che a ogni giro
corregge i pesi del modello; il **learning rate** è la sua manopola
principale, e decide quanto è grande ogni correzione: un passo corto impara
piano ma non sbaglia strada, un passo lungo va veloce ma rischia di scavalcare
il punto buono. L'ottimizzatore più semplice si chiama **SGD** (dall'inglese
*stochastic gradient descent*, discesa del gradiente a caso: «a caso» perché
guarda ogni volta un pacchetto di esempi presi a sorte invece di tutti) e usa
lo stesso passo per tutti i pesi; **Adam** è quello che si prova per primo in
quasi ogni progetto, per la ragione che mostra
{numref}`fig-adam-passo-per-peso`.

```{figure} ../figures/adam-ottimizzatore.svg
:name: fig-adam-passo-per-peso
:alt: "Confronto fra due ottimizzatori sugli stessi pesi. Con SGD tutti i parametri si muovono con lo stesso passo, grande o piccolo che sia stata finora la loro correzione. Con Adam ogni peso ha il passo suo, calcolato da quanto quel peso si è mosso nelle correzioni precedenti: chi si è mosso molto rallenta, chi si è mosso poco accelera."
:width: 96%

Un learning rate per ciascuno. Adam non sceglie una velocità migliore: ne
sceglie una diversa per ogni parametro, in base a quanto quel parametro si è
mosso finora.
```

La differenza mostrata in {numref}`fig-adam-passo-per-peso` è il motivo per
cui `Adam` è quasi sempre il primo ottimizzatore da provare. Con SGD il
learning rate va tarato bene perché è uno solo per tutti; con Adam il valore
di partenza è meno critico, perché ciascun peso lo riscala per conto proprio.

Nel codice le cose hanno il nome inglese: `criterion` è la funzione di perdita,
`dataloader` è il cameriere che porta i pacchetti di esempi (la prossima
sezione lo costruisce), `optimizer` è l'ottimizzatore appena presentato.

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
Guardi la domanda e provi a rispondere (passo 1, la previsione). Giri la carta
e confronti con la risposta giusta: quanto eri lontano? (passo 2, l'errore).
Butti via gli appunti del giro precedente (passo 3), capisci *in che
direzione* hai sbagliato, troppo alto? troppo basso? (passo 4), e aggiusti di
conseguenza il tuo modo di rispondere, un poco alla volta (passo 5). Poi passi
al mazzetto successivo, e quando hai ripassato l'intero mazzo una volta, hai
completato quella che si chiama un'**epoca**. Ripetuto per migliaia di carte
ed epoche, questo giro è tutto ciò che serve a una rete per imparare.
`````

`````{tab} Superiore
Il loop realizza un passo di discesa del gradiente su mini-batch. Con
$\mathcal{L}$ la loss media sul batch e $\theta$ i parametri:

$$
\theta \leftarrow \theta - \eta \, \nabla_{\theta} \mathcal{L},
$$

dove $\eta$ è il *learning rate*. `loss.backward()` calcola
$\nabla_{\theta}\mathcal{L}$ via autograd e lo deposita in `p.grad` per ogni
parametro; `optimizer.step()` applica l'aggiornamento, la formula esatta
dipende dall'ottimizzatore: la discesa semplice per `optim.SGD`, stime
adattive dei momenti per `optim.Adam` {cite}`kingma2015adam`, il default
robusto di quasi ogni progetto. `zero_grad()` è necessario perché autograd
**accumula** i gradienti a ogni `backward()`: senza azzeramento, ogni passo
userebbe la somma di tutti i gradienti precedenti. L'ordine dei passi 3–5 è
l'unica liturgia da rispettare; tutto il resto è normale Python, e infatti qui
si innestano senza attrito *gradient clipping*, *scheduler* del learning rate,
*mixed precision*.
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
rete non impara la sequenza a memoria, come uno studente che ripassa sempre le
carte nello stesso ordine) e porta in tavola vassoi da 64 esempi alla volta.
Perché proprio a vassoi? Un esempio alla volta è uno spreco, la GPU resta
ferma ad aspettare; tutti insieme non entrano in memoria. Il mini-batch è la
via di mezzo che tiene la cucina sempre piena.
`````

`````{tab} Superiore
`Dataset` (variante *map-style*) è un protocollo minimo: `__len__` e
`__getitem__`. Qualunque classe che li implementi (un file CSV, una cartella
di immagini, un database) diventa una sorgente per il `DataLoader`, che
aggiunge campionamento (`shuffle=True` rimescola gli indici a ogni epoca),
*batching* (impila gli esempi lungo il primo asse: qui tensori
$(64, 1, 28, 28)$), e caricamento parallelo (`num_workers`) con memoria
*page-locked* (`pin_memory=True`), che è la **premessa** del trasferimento
asincrono verso la GPU, non l'asincronia: quella richiede anche
`non_blocking=True` nel `.to()`, come si vedrà in
[prestazioni](prestazioni.md). La `transform` `ToTensor()`
converte le immagini PIL in tensori `float32` con valori in $[0, 1]$ e layout
channels-first $(C, H, W)$; per MNIST si può aggiungere
`transforms.Normalize((0.1307,), (0.3081,))` (media e deviazione standard del
dataset) per centrare gli input, come visto nel capitolo sulle reti neurali.
Statisticamente, il gradiente su un mini-batch è una stima non distorta ma
rumorosa del gradiente vero: il rumore è il prezzo (e in parte il segreto)
della discesa *stocastica*.
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
optimizer = optim.Adam(model.parameters(), lr=1e-3)   # 1e-3 e' 0,001

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
            # per ogni immagine prendi il punteggio piu' alto -> la cifra scelta;
            # confrontala con quella vera; conta i sì
            corretti += (y_pred.argmax(dim=1) == y).sum().item()

    print(f"epoca {epoca + 1}: accuratezza sul test {corretti / len(test_data):.3f}")
```

La riga che conta le risposte giuste merita di essere sciolta, perché è quella
che produce il numero di cui il modello si vanta, e sta tutta in una riga sola.
`y_pred` è una tabella con una riga per immagine e dieci punteggi per riga;
`argmax(dim=1)` scorre ciascuna riga e restituisce la **posizione** del
punteggio più alto, cioè la cifra che il modello ha scelto. Il confronto `== y`
mette a fianco la risposta vera e produce una colonna di sì e no; `.sum()` conta
i sì (che valgono uno) e `.item()` estrae quel conteggio come numero Python
normale, da poter sommare al totale. Quattro gesti, quattro parole, e sono gli
stessi quattro che torneranno in ogni programma del capitolo.

Cinque epoche, e l'accuratezza sul test arriva attorno al **97–98%**:
novantasette cifre su cento lette correttamente da $101\,770$ numeri che
un'ora fa erano casuali. Il tempo dipende molto dalla macchina, e vale la pena
dirlo per non lasciare aspettative sbagliate: su una GPU sono decine di
secondi, su una CPU normale (quattro core, nient'altro di speciale) siamo
nell'ordine dei minuti, sette in una misura fatta su questa macchina, che
chiudeva a $97{,}5\%$.

## Studiare e dare l'esame: `train()` ed `eval()`

Nel programma compaiono due chiamate su cui vale la pena fermarsi:
`model.train()` e `model.eval()`, con il blocco `torch.no_grad()`.

`````{tab} Elementare
La rete ha due modalità, come uno studente. Quando **studia**
(`model.train()`) può usare trucchi che servono solo a imparare meglio, per
esempio coprirsi a caso qualche appunto per non adagiarsi (il *dropout* che
vedremo nel prossimo capitolo). Quando **dà l'esame** (`model.eval()`) i
trucchi si spengono: risponde e basta, al meglio di quel che sa. E
`torch.no_grad()` dice al registratore dei gradienti di spegnersi: durante
l'esame non si prende appunti per migliorare, si risponde soltanto, e senza il
registratore acceso tutto è più veloce e leggero.
`````

`````{tab} Superiore
`train()`/`eval()` commutano un flag che cambia il comportamento dei moduli "a
doppia personalità": `nn.Dropout` (attivo solo in training) e le
`nn.BatchNorm1d/2d/3d` (statistiche del batch in training, medie mobili in
valutazione) sono i due casi principali. Attenzione a come si dice, perché la
formulazione sbrigativa («`eval()` spegne dropout e batch norm») è falsa per la
seconda: in `eval()` il dropout diventa davvero l'identità, la batch norm
invece continua a normalizzare, solo che usa le medie mobili accumulate invece
delle statistiche del batch corrente. Misurato su torch 2.13: un
`nn.BatchNorm1d(3)` allenato su dati centrati attorno a $10$ e poi messo in
`eval()` restituisce uscite di media $\approx 0$ a fronte di ingressi di media
$\approx 10$. La differenza non è terminologica: chi crede che `eval()`
disattivi la batch norm non capisce perché un modello valutato con un batch da
un solo esempio funzioni benissimo in `eval()` (le medie mobili non dipendono
dal batch) e produca `nan` in `train()` (la varianza di un esempio solo è
zero). `torch.no_grad()` è un context manager che sospende la
costruzione del grafo autograd: non vengono salvati i valori intermedi per un
`backward()` che non arriverà mai, con un risparmio di memoria che cresce con
la profondità della rete, e l'inferenza accelera. Sono due meccanismi
indipendenti e servono entrambi: `eval()` senza `no_grad()` dà predizioni
corrette ma spreca memoria; `no_grad()` senza `eval()` lascia il dropout
acceso e falsa le predizioni. Il nostro MLP non ha né dropout né batch norm,
quindi qui `eval()` è tecnicamente superfluo, ma scriverlo sempre è
un'abitudine che evita bug sottili appena il modello cresce. Quando si fa
soltanto inferenza esiste una forma più stretta di `no_grad()`,
`torch.inference_mode()`, che oltre a non registrare rinuncia anche al
*version counter* e al tracciamento delle viste: è leggermente più veloce, al
prezzo che i tensori che produce non possono poi rientrare in un grafo
autograd.
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
simulazione d'esame con domande nuove. All'inizio migliorano insieme: buon
segno. Poi la validazione si ferma e comincia a peggiorare, mentre
l'addestramento continua a salire: da lì in avanti il modello sta imparando a
memoria, è l'**overfitting** che abbiamo incontrato nel capitolo sul machine
learning. La mossa giusta è fermarsi al punto migliore della validazione, e
tenere da parte la copia del modello salvata in quel momento.
`````

`````{tab} Superiore
Nel loop esplicito la diagnosi si scrive da sé: si ritaglia un set di
validazione (ad esempio con
`torch.utils.data.random_split(train_data, [55000, 5000])`), a fine epoca si
misura $\mathcal{L}_{\text{val}}$, e l'*early stopping* è un `if`: se la
validazione non migliora da $k$ epoche (la *patience*), si esce dal ciclo e si
ricaricano i pesi dell'epoca migliore, salvati via via con `torch.save`. Ciò
che Keras offriva come callback preconfezionate, in PyTorch sono sei righe di
controllo di flusso, in cambio, nessun limite: fermarsi su una metrica
composta o salvare solo a condizioni particolari sono varianti banali dello
stesso `if`. Riprendere da checkpoint no, ed è la trappola della sezione
seguente: vuole anche lo stato dell'ottimizzatore. Il divario
$\mathcal{L}_{\text{val}} - \mathcal{L}_{\text{train}}$ resta la bussola: se
si allarga, servono i freni (regolarizzazione L2 via `weight_decay`
dell'ottimizzatore, `nn.Dropout`) che approfondiremo nel capitolo sul deep
learning.
`````

## Salvare il lavoro: lo `state_dict`

Un modello addestrato va messo al sicuro. In PyTorch non si salva l'oggetto
modello: si salva il suo **`state_dict`**, il dizionario che associa a ogni
tensore del modello il suo nome. Dentro non ci sono solo i pesi imparati: ci
sono anche i *buffer*, cioè i numeri che il modello si tiene da parte senza
impararli (le medie mobili della batch norm, per esempio).

Il motivo per cui non si salva l'oggetto intero è pratico. Salvare l'oggetto
significherebbe mettere nel file anche la classe Python che lo descrive, e
quella classe sta nel tuo codice, che intanto cambia: fra sei mesi il file
proverebbe a ricostruire una classe che non esiste più con quel nome, o che
esiste con un `forward` diverso. Salvando solo i numeri, il file resta leggibile
finché sai ricostruire l'architettura, e l'architettura è scritta nel codice,
dove si può leggere e correggere.

```python
torch.save(model.state_dict(), "mnist_mlp.pt")     # salva i numeri

model2 = nn.Sequential(                            # stessa architettura...
    nn.Flatten(), nn.Linear(28 * 28, 128), nn.ReLU(), nn.Linear(128, 10)
)
model2.load_state_dict(torch.load("mnist_mlp.pt")) # ...numeri ricaricati
model2.eval()                                      # pronto per l'uso
```

Il codice che definisce l'architettura resta la fonte di verità; il file `.pt`
contiene solo i numeri. È una divisione dei compiti coerente con tutto il
capitolo (il modello è codice, i pesi sono dati) ed è il formato in cui
circolano i modelli pre-addestrati che riutilizzeremo nel capitolo sulla
visione artificiale, quando un modello nato per un compito verrà rifinito
(*fine-tuning*) su un altro.

C'è però una distinzione da fare subito, perché costa una riga farla e giorni
scoprirla dopo: **salvare per usare** e **salvare per riprendere** non sono la
stessa cosa.

`````{tab} Elementare
Il file con i soli pesi serve a **usare** il modello: lo ricarichi, gli dai
un'immagine, ti risponde. Non serve a **riprendere** l'addestramento dal punto
in cui l'avevi interrotto.

La ragione è che l'ottimizzatore, mentre corregge i pesi, si costruisce una
memoria di come si sono mossi finora: è proprio quella memoria che gli permette
di dare a ciascun peso il passo giusto. Ricaricare i pesi e ripartire con un
ottimizzatore appena creato è come rimettere in corsa un ciclista da fermo, nel
punto esatto in cui l'avevi lasciato: la posizione è quella giusta, ma lo
slancio è perduto, e la prima pedalata è sbagliata. Misurato: il primo passo
dopo una ripresa fatta così è quasi quattro volte più lungo di quello che
sarebbe stato senza interruzione.

Il rimedio costa una riga: nel file si mette anche lo stato
dell'ottimizzatore, e al ritorno lo si ricarica.
`````

`````{tab} Superiore
Con SGD nudo la questione è marginale; con `optim.Adam`, che è il default
raccomandato all'inizio della sezione, i momenti $m$ e $v$ *sono* stato, e
ripartire
senza di essi non riprende la stessa traiettoria: la correzione bias del primo
passo si ricalcola da $t = 1$, quindi il primo aggiornamento vale
$\approx \eta$ pieno. Verificato su torch 2.13 (venti passi di Adam con
$\eta = 0{,}1$, poi ripresa): il passo successivo misura $0{,}100$ senza lo
stato dell'ottimizzatore contro $0{,}0256$ ricaricandolo, ed è esattamente il
valore che si sarebbe avuto senza interruzione. Un fattore $3{,}9$, e nessun
messaggio d'errore.

Lo stesso vale per tutto ciò che ha uno `state_dict` e che il ciclo tocca:
lo *scheduler* del learning rate, il `GradScaler` della precisione mista, il
`sampler` distribuito. Un checkpoint completo è un dizionario, non un tensore.
`````

Ecco la forma minima, quella che si scrive una volta e si copia in ogni
progetto:

```python
from torch import optim

ottimizzatore = optim.Adam(model.parameters(), lr=1e-3)

# checkpoint per RIPRENDERE: i pesi da soli non bastano
torch.save({"epoca": 5,
            "modello": model.state_dict(),
            "ottimizzatore": ottimizzatore.state_dict()}, "checkpoint.pt")

stato = torch.load("checkpoint.pt")
model.load_state_dict(stato["modello"])
ottimizzatore.load_state_dict(stato["ottimizzatore"])   # la riga che si dimentica
print(f"ripresa dall'epoca {stato['epoca']}; lo stato dell'ottimizzatore "
      f"ha le chiavi {list(stato['ottimizzatore'])}")
```

Il capitolo [dal notebook agli script](dal-notebook-agli-script.md) riprende
questo file e ci aggiunge le altre due cose che servono a rileggerlo fra sei
mesi: i nomi delle classi e la configurazione dell'esperimento.

`````{tab} Elementare
```{admonition} Da ricordare
:class: important
- Il **giro di addestramento** ha cinque passi fissi, sempre nello stesso
  ordine: prevedi, misura l'errore, butta gli appunti del giro prima, capisci
  in che direzione hai sbagliato, correggi. Il terzo passo serve perché
  altrimenti gli appunti si sommano.
- Gli esempi arrivano in **mucchietti** (i mini-batch): la dispensa
  (`Dataset`) sa consegnarli uno per uno, il cameriere (`DataLoader`) li
  mescola e li porta in tavola a vassoi.
- Quando il modello **dà l'esame** si aziona `model.eval()`, e si aggiunge
  `torch.no_grad()` per non prendere appunti inutili. Sono due interruttori
  diversi e servono tutti e due.
- Le due curve, quella dell'addestramento e quella della simulazione d'esame,
  dicono quando è ora di **fermarsi**: quando la seconda smette di migliorare.
- Del modello si salvano **i numeri**, non l'oggetto: l'architettura sta nel
  codice. E per riprendere l'addestramento dove si era interrotto serve anche
  la memoria dell'ottimizzatore, non solo i pesi.
```
`````

`````{tab} Superiore
```{admonition} Da ricordare
:class: important
- Il **training loop** ha cinque passi fissi: forward → loss →
  `zero_grad()` → `backward()` → `step()`. L'azzeramento serve perché i
  gradienti si accumulano.
- `Dataset` consegna gli esempi, `DataLoader` li rimescola e li impila in
  **mini-batch**: il gradiente sul batch è una stima rumorosa ma economica di
  quello vero.
- In valutazione: `model.eval()` spegne il dropout e passa la batch norm alle
  **medie mobili** (non la spegne: continua a normalizzare); `torch.no_grad()`
  sospende autograd. Servono entrambi.
- Le curve di training e validazione diagnosticano l'**overfitting**;
  l'early stopping in PyTorch è un semplice `if` nel loop.
- Si salva lo **`state_dict`** (`torch.save`/`load_state_dict`), non
  l'oggetto: contiene parametri **e** buffer. Per *riprendere* servono anche
  `ottimizzatore.state_dict()` e l'epoca; senza, con Adam il primo passo dopo
  la ripresa è quello di un ottimizzatore appena nato.
```
`````
