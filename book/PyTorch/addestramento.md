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
metterle al lavoro su un problema vero. Prima, però, due parole che nel codice
compaiono senza presentazioni.

L’**ottimizzatore** è il pezzo che a ogni giro corregge i pesi del modello. Il
**learning rate** (in italiano si dice anche, più brevemente, il **passo**: le
due parole in questo libro vogliono dire la stessa cosa) è la sua manopola
principale, e decide quanto è grande ogni correzione: un passo corto impara
piano ma non sbaglia strada, un passo lungo va veloce ma rischia di scavalcare
il punto buono.

Di ottimizzatori ce ne sono parecchi, e per adesso ne bastano due. Il più
semplice si chiama **SGD**, dall'inglese *stochastic gradient descent*, discesa
del gradiente a caso; «a caso» perché a ogni giro guarda un pacchetto di
esempi sorteggiati invece di tutti quanti, ed è il modo in cui gli esempi
arrivano a una rete, come vedremo fra poco. SGD usa lo stesso passo per tutti i
pesi. **Adam** invece dà a ciascun peso il passo suo, ed è la ragione per cui
si prova per primo in quasi ogni progetto ({numref}`fig-adam-passo-per-peso`).
Il nome non è di persona: sta per *adaptive moment estimation*.

```{figure} ../figures/adam-ottimizzatore.svg
:name: fig-adam-passo-per-peso
:alt: "Confronto fra due ottimizzatori sugli stessi pesi. Con SGD il passo di un parametro è proporzionale al suo gradiente, quindi chi ha il gradiente grande fa un salto grande. Con Adam ogni peso ha il passo suo, tarato sul rapporto fra la direzione media dei suoi gradienti e la loro grandezza tipica: chi viene spinto sempre dalla stessa parte avanza a passo pieno, chi sbanda avanti e indietro rallenta."
:width: 96%

Un learning rate per ciascuno. Adam non sceglie una velocità migliore: ne
sceglie una diversa per ogni parametro, in base a quanto è costante la
direzione in cui quel parametro viene spinto.
```

La conseguenza pratica: con SGD il learning rate va tarato bene, perché è uno
solo per tutti; con Adam il valore di partenza è meno critico, perché ciascun
peso lo riscala per conto proprio.

Nel codice le cose hanno il nome inglese: `criterion` è la funzione di perdita
(sì, la stessa che il capitolo chiama *loss* e che qui a volte si chiama
«criterio»: sono tre nomi per un oggetto solo), `dataloader` è un cameriere che
porta i pacchetti di esempi e che costruiremo nel prossimo paragrafo,
`optimizer` è l'ottimizzatore appena presentato.

```{code-block} python
:class: pt-non-eseguibile

for X_batch, y_batch in dataloader:
    y_pred = model(X_batch)             # 1. forward: la previsione
    loss = criterion(y_pred, y_batch)   # 2. loss: quanto abbiamo sbagliato
    optimizer.zero_grad()               # 3. via i gradienti del giro prima
    loss.backward()                     # 4. backward: calcola i gradienti
    optimizer.step()                    # 5. aggiorna i pesi
```

Il terzo passo è quello che a prima vista sembra fuori posto: si butta via una
cosa prima di averla calcolata. La ragione è che PyTorch, quando calcola i
gradienti, non li scrive sopra ai vecchi: li **somma** a quelli che trova.
Disastroso qui, perché al secondo giro il modello si correggerebbe usando anche
l'errore del primo; comodo in un caso solo, che è quello di chi ha una macchina
piccola e vuole far finta di avere pacchetti grandi, e che la sezione su
[replicare un paper](replicare-un-paper.md) riprende. Fuori da quel caso, si fa
piazza pulita prima, ed è per questo che il gesto sta *prima* del calcolo e non
dopo l'uso.

Quelle cinque righe dicono l'ordine, non il movimento:
{numref}`fig-ciclo-addestramento` le fa girare tre volte, cioè su tre pacchetti
di esempi in fila (il nome tecnico è *mini-batch*, e il prossimo paragrafo
racconta perché gli esempi arrivino a pacchetti), e mostra a ogni passo che
cosa cambia dentro il modello.

```{figure} ../figures/ciclo-addestramento.svg
:name: fig-ciclo-addestramento
:alt: "I cinque passi del ciclo di addestramento in fila, con una freccia che dal quinto torna al primo. Sotto, lo stato del modello: le barre dei gradienti di sei pesi e la posizione di quei sei pesi rispetto al valore di partenza. Il quarto passo, backward, riempie le barre dei gradienti; il quinto, step, sposta i pesi; le barre restano piene fino allo zero_grad del giro successivo, che le riporta a zero. A destra la loss dei tre giri, che scende: 2,35 poi 2,29 poi 2,25."
:width: 96%

Tre giri dello stesso ciclo, su una rete piccola presa a esempio. I gradienti
compaiono quando `backward()` li calcola, restano finché lo `zero_grad()` del
giro dopo non li toglie di mezzo, e intanto `step()` sposta i pesi. A destra la
loss dei tre giri: parte da $2{,}35$, poco sopra il $2{,}30$ di chi tira a
indovinare fra dieci cifre, e a ogni giro scende un poco.
```

`````{tab} Elementare
È il metodo con cui si impara con le flashcard, le carte per memorizzare.
Guardi la domanda e provi a rispondere (passo 1, la previsione). Giri la carta
e confronti con la risposta giusta: quanto eri lontano? (passo 2, l'errore).
Butti via gli appunti del giro precedente (passo 3), capisci *in che
direzione* hai sbagliato, troppo alto? troppo basso? (passo 4), e aggiusti di
conseguenza il tuo modo di rispondere, un poco alla volta (passo 5). Poi passi
al mazzetto successivo, e quando hai ripassato l'intero mazzo una volta, hai
completato quella che si chiama un’**epoca**. Ripetuto per migliaia di carte
ed epoche, questo giro è tutto ciò che serve a una rete per imparare.

Gli appunti del giro prima non si cancellano riga per riga: si toglie il foglio
e se ne prende uno pulito. Chi lo va a cercare dopo si aspetta di trovarlo
bianco, e sul tavolo non c'è più niente, quindi il momento buono per leggerlo è
subito dopo averlo scritto.

Il foglio si cambia a ogni giro, con un'eccezione sola. Se sul tavolo ci stanno
otto carte per volta e la correzione la vuoi decidere su trentadue, fai quattro
mazzetti da otto e scrivi gli appunti di tutti e quattro sullo stesso foglio,
uno sotto l'altro; correggi una volta, alla fine, e solo allora cambi foglio.
Chi cambia foglio a ogni mazzetto si corregge su otto carte credendo di averne
guardate trentadue, il ripasso fila liscio uguale e non arriva nessun avviso.
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
**accumula** i gradienti a ogni `backward()`: senza, ogni passo userebbe la
somma di tutti i gradienti precedenti.

Il nome però dice meno di quello che il metodo fa. Da PyTorch 2.0 il default è
`set_to_none=True`, quindi `p.grad` non diventa un tensore di zeri: diventa
`None`. È un risparmio (nessuna memoria tenuta occupata da gradienti che non
ci sono, e un'operazione in meno per parametro) ed è una trappola per chi va a
controllare i gradienti nel posto sbagliato: un `assert p.grad is not None`
scritto dopo l'azzeramento fallisce su tutti i parametri, e quel controllo va
messo subito dopo il `backward()`.

L'ordine dei passi 3–5 è l'unica liturgia da rispettare; tutto il resto è
normale Python, e infatti qui si innestano senza attrito *gradient clipping*,
*scheduler* del learning rate, *mixed precision*. L'eccezione alla liturgia è
una sola, ed è l’**accumulo dei gradienti**, il modo di simulare un batch
grande su una macchina piccola che vedremo in
[replicare un paper](replicare-un-paper.md): lì si eseguono $k$ `backward()` e
un solo `step()`, quindi l'azzeramento esce dal giro e si fa una volta ogni
$k$ micro-batch. Rispettare la liturgia lì è l'errore: il codice gira
identico, e la matematica no. L'accumulo fatto bene coincide con il batch
grande vero a meno di $1{,}5 \cdot 10^{-8}$, quello con lo `zero_grad()`
a ogni micro-batch sbaglia di $8{,}4 \cdot 10^{-2}$ senza una riga di errore.
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
test_loader = DataLoader(test_data, batch_size=256)   # niente shuffle, pacchetti piu' grandi
```

Le due righe finali non sono uguali per caso, e la ragione è la stessa per tutte
e due: in esame il modello non impara. Il mescolamento serve a non fargli
imparare a memoria l'ordine delle carte, quindi in esame non serve. E la
dimensione del pacchetto, in addestramento, decide anche ogni quanti esempi il
modello si corregge: 64 vuol dire una correzione ogni 64 immagini. In esame non
si corregge niente, quindi si può prendere il pacchetto più capiente che la
memoria regge, e si guadagna solo velocità.

`````{tab} Elementare
Il `Dataset` è la dispensa: sa quanti esempi ci sono e sa consegnarti
l'esempio numero $i$ quando glielo chiedi. Altro non gli si chiede, e per
questo la dispensa può essere quasi qualunque cosa, una cartella di
fotografie, un foglio di calcolo, un archivio su un altro computer. Il
`DataLoader` è il cameriere che apparecchia: pesca dalla dispensa,
**mescola** l'ordine a ogni giro (così la rete non impara la sequenza a
memoria, come uno studente che ripassa sempre le carte nello stesso ordine) e
porta in tavola vassoi da 64 esempi alla volta. Se il servizio non tiene il
passo della cucina, si mettono più camerieri.

Perché proprio a vassoi? Un esempio alla volta è uno spreco, la GPU resta
ferma ad aspettare; tutti insieme non entrano in memoria. Il mini-batch è la
via di mezzo che tiene la cucina sempre piena. E un vassoio dice quasi quello
che direbbe il dataset intero. Assaggiare un cucchiaio dice quanto sale c'è in
tutta la pentola, con la risposta giusta in media e un po’ di scarto da un
cucchiaio all'altro. Allo stesso modo sessantaquattro esempi indicano la
direzione in cui correggersi quasi come la indicherebbero tutti e sessantamila,
e sbandano un poco a ogni giro. Lo sbandamento costa in precisione, e in cambio
scuote la discesa quel tanto che basta a non farla fermare al primo
avvallamento.
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
viste. Ci sono dentro tre chiamate che non abbiamo ancora presentato
(`model.train()`, `model.eval()` e il blocco `torch.no_grad()`): per ora si
possono leggere come «adesso studia» e «adesso rispondi e basta», e la sezione
subito dopo il codice se ne occupa per esteso.

```python
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
punteggio più alto, cioè la cifra che il modello ha scelto (`dim` sta per
*dimension*, cioè quale asse percorrere: `dim=1` è il secondo, quello dei dieci
punteggi, perché il primo, `dim=0`, è quello delle immagini). Il confronto
`== y` mette a fianco la risposta vera e produce una colonna di sì e no;
`.sum()` conta i sì (che valgono uno) e `.item()` estrae quel conteggio come
numero Python normale, da poter sommare al totale. Quattro gesti, quattro
parole, e sono gli stessi quattro che torneranno in ogni programma del
capitolo. Il rapporto fra i sì e il totale è l’**accuratezza**: la quota di
risposte giuste, e basta.

Cinque epoche, e l'accuratezza sul test arriva attorno al **97–98%**:
novantasette cifre su cento lette correttamente da $101\,770$ numeri che prima
di partire erano casuali. Quanto ci vuole dipende molto dalla macchina, e
conviene dirlo per non lasciare aspettative sbagliate: su una GPU sono decine
di secondi, su una CPU normale si va sui minuti, e su quattro core sono sette.

## Studiare e dare l'esame: `train()` ed `eval()`

Nel programma compaiono due chiamate su cui conviene fermarsi: `model.train()`
e `model.eval()`, con il blocco `torch.no_grad()`.

`````{tab} Elementare
La rete ha due modalità, come uno studente. Quando **studia**
(`model.train()`) può usare trucchi che servono solo a imparare meglio, per
esempio coprirsi a caso qualche appunto per non adagiarsi (il *dropout*, che
vedremo nel [capitolo sul deep
learning](../DeepLearning/ottimizzazione-regolarizzazione.md)). Quando **dà
l'esame** (`model.eval()`) quel trucco si spegne: risponde e basta, al meglio
di quel che sa.

Non tutto si spegne, però. Certi pezzi hanno bisogno di sapere quanto sono
grandi di solito i numeri che ricevono, per rimetterli in scala prima di
passarli avanti (è la *batch norm*, di cui parla lo stesso capitolo).
Studiando prendono quella misura sul mazzetto che hanno davanti;
all'esame usano la media che si sono annotati durante il ripasso. Rimettere in
scala lo fanno in tutti e due i casi, e a cambiare è soltanto da dove viene il
metro.

Da qui una stranezza che altrimenti non si spiegherebbe. All'esame una carta
sola basta, perché il metro è già annotato. Durante lo studio quella stessa
carta sola blocca tutto, perché da una misura sola non si capisce quanto le
cose varino, e il programma si ferma e lo dice invece di tirare a indovinare.

E `torch.no_grad()` dice al registratore dei gradienti di spegnersi: durante
l'esame non si prende appunti per migliorare, si risponde soltanto, e senza il
registratore acceso tutto è più veloce e leggero. I due gesti sono distinti, e
nessuno dei due sostituisce l'altro: spegnere il registratore lascia accesi i
trucchi dello studio, e dichiarare l'esame lascia acceso il registratore. Chi
ne fa uno solo o riempie fogli che nessuno leggerà, o dà l'esame con qualche
appunto ancora coperto.
`````

`````{tab} Superiore
`train()`/`eval()` commutano un flag che cambia il comportamento dei moduli "a
doppia personalità": `nn.Dropout` (attivo solo in training) e le
`nn.BatchNorm1d/2d/3d` (statistiche del batch in training, medie mobili in
valutazione) sono i due casi principali. Di che cosa facciano davvero, e
perché aiutino, si occupa il capitolo sul deep learning in [ottimizzazione e
regolarizzazione](../DeepLearning/ottimizzazione-regolarizzazione.md); qui
serve solo sapere che hanno due comportamenti e che l'interruttore è questo.
Attenzione a come si dice, perché la
formulazione sbrigativa («`eval()` spegne dropout e batch norm») è falsa per la
seconda: in `eval()` il dropout diventa davvero l'identità, la batch norm
invece continua a normalizzare, solo che usa le medie mobili accumulate invece
delle statistiche del batch corrente. Su torch 2.13 un
`nn.BatchNorm1d(3)` allenato su dati centrati attorno a $10$ e poi messo in
`eval()` restituisce uscite di media $\approx 0$ a fronte di ingressi di media
$\approx 10$. La differenza non è terminologica: chi crede che `eval()`
disattivi la batch norm non capisce perché un modello valutato con un batch da
un solo esempio funzioni benissimo in `eval()` (le medie mobili non dipendono
dal batch) e in `train()` invece non parta affatto. Non con un `nan`, come si
legge spesso: con un'eccezione esplicita, `ValueError: Expected more than 1
value per channel when training`, che una `nn.BatchNorm1d(3)` solleva su un
ingresso di forma $(1, 3)$. Il messaggio è più utile della leggenda, perché
chi lo incontra riconosce il caso senza doverlo dedurre: con un esempio solo
la varianza per canale non è stimabile, e la libreria preferisce fermarsi
piuttosto che normalizzare per qualcosa che non ha calcolato. Dove invece i
valori per canale sono più di uno il conto si fa e `nan` non ne esce: un
ingresso $(1, 3, 5)$, o l'immagine $(1, 3, 4, 4)$ di una `nn.BatchNorm2d`,
passano senza storie, e su un ingresso costante l'uscita è zero, o un residuo
minuscolo di arrotondamento, e non `nan`, perché l’$\varepsilon$ che si somma al
denominatore impedisce che si divida zero per zero.
`torch.no_grad()` è un context manager che sospende la
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

Prima però va sistemata una parola, perché da qui in avanti ne compaiono due
dove finora ce n'era una. Nel programma qui sopra i mucchi di dati sono due,
addestramento e **test**, e a fine epoca abbiamo guardato il test. Facendo
così, però, il test smette di essere quello che deve essere: se lo guardo a
ogni epoca e in base a quel numero decido quando fermarmi o che cosa cambiare,
allora quelle immagini hanno partecipato alle mie decisioni, e il voto che mi
danno non è più il voto di uno che non le aveva mai viste.

Per questo i mucchi in un progetto serio sono **tre**. L'addestramento è quello
su cui il modello impara. La **validazione** è quello che si guarda spesso, a
ogni epoca, per decidere: è la simulazione d'esame, e la si può consumare senza
danno perché serve proprio a quello. Il **test** è quello che si tocca una
volta sola, alla fine, e che dà il voto vero. Nel programma qui sopra ne
abbiamo usati due per non appesantire il codice, ed è una scorciatoia comune
negli esempi: fuori dagli esempi, il terzo mucchio si ritaglia.

```{figure} ../figures/curve-overfitting-validazione.svg
:name: fig-curve-overfitting
:alt: Due curve di perdita in funzione delle epoche. La curva di addestramento scende con continuità; quella di validazione scende, tocca un minimo e poi risale. Una linea tratteggiata verticale segna il punto di arresto anticipato in corrispondenza del minimo della validazione.
:width: 85%

La perdita di addestramento scende sempre; quella di validazione tocca un
minimo e poi risale. Da lì in poi il modello memorizza il rumore: la linea
dell'arresto anticipato marca il momento giusto per fermarsi.
```

`````{tab} Elementare
Guarda le due curve in {numref}`fig-curve-overfitting`. Attenzione al verso:
qui in verticale c'è l’**errore**, quindi *scendere* è migliorare. La curva
dell'addestramento è come i compiti fatti a casa: l'errore cala sempre, perché
il modello rivede gli stessi esercizi. Quella della validazione è la
simulazione d'esame con domande nuove. All'inizio scendono insieme, ed è buon
segno. Poi quella della validazione tocca il fondo e ricomincia a salire,
mentre quella dell'addestramento continua a scendere: da lì in avanti il
modello non sta più imparando, sta imparando **a memoria**, ed è
l’**overfitting** incontrato nel capitolo sul machine learning. La mossa giusta
è fermarsi nel punto più basso della validazione, e tenere da parte la copia
del modello salvata in quel momento. La distanza fra le due curve è la spia da
guardare: finché resta stretta il ripasso serve a qualcosa, e quando si allarga
il modello sta lavorando per i compiti a casa e non per l'esame. Fermarsi è il
rimedio più immediato. L'altro è rendergli lo studio un po’ più difficile
mentre impara, e lo racconta il capitolo sul deep learning.

Nel programma di poco fa niente di tutto questo c'è: cinque epoche e via,
perché su MNIST cinque epoche non bastano a mandare a memoria sessantamila
immagini. Aggiungerlo però costa poco, ed è un `if`: a ogni epoca si guarda il
numero della validazione, se è il migliore finora si salva una copia del
modello, e se non migliora per un po’ di epoche di fila si esce dal ciclo.
Salvare quella copia è una riga sola, e conta molto che cosa ci si mette dentro.
`````

`````{tab} Superiore
Nel loop esplicito la diagnosi si scrive da sé: si ritaglia un set di
validazione (ad esempio con
`torch.utils.data.random_split(train_data, [55000, 5000])`), a fine epoca si
misura $\mathcal{L}_{\text{val}}$, e l’*early stopping* è un `if`: se la
validazione non migliora per un numero fissato di epoche (la *patience*), si
esce dal ciclo e si ricaricano i pesi dell'epoca migliore, salvati via via con
`torch.save`. Ciò che Keras offriva come callback preconfezionate, in PyTorch
sono sei righe di controllo di flusso; in cambio, nessun limite: fermarsi su
una metrica composta o salvare solo a condizioni particolari sono varianti
banali dello stesso `if`. Riprendere da checkpoint no, ed è la trappola del
salvataggio: vuole anche lo stato dell'ottimizzatore. Il divario
$\mathcal{L}_{\text{val}} - \mathcal{L}_{\text{train}}$ resta la bussola: se
si allarga, servono i freni (regolarizzazione L2 via `weight_decay`
dell'ottimizzatore, `nn.Dropout`) che approfondiremo nel capitolo sul deep
learning.
`````

## Salvare il lavoro: lo `state_dict`

Un modello addestrato va messo al sicuro. In PyTorch non si salva l'oggetto
modello: si salva il suo **`state_dict`**, cioè l'elenco di tutti i suoi numeri
con accanto il nome del pezzo a cui appartengono (in Python un elenco fatto
così, dove a ogni nome corrisponde una cosa, si chiama *dizionario*). Dentro
non ci sono solo i pesi imparati: ci sono anche i *buffer*, cioè i numeri che
il modello si tiene da parte senza impararli, perché li ha misurati sui dati
invece di ricavarli dall'errore. Ne incontreremo un caso nella {doc}`sezione
sulla normalizzazione </DeepLearning/ottimizzazione-regolarizzazione>`; qui
basta sapere che nel file finiscono anche quelli.

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

La ragione è che Adam, mentre corregge i pesi, si costruisce una memoria di
come si sono mossi finora, ed è quella memoria che gli permette di dare a
ciascun peso il passo giusto, e in particolare di accorciarlo man mano che ci
si avvicina. SGD, il più spartano dei due, quella memoria non ce l'ha, e con
lui la ripresa cambia poco. Ricaricare i pesi e ripartire con un ottimizzatore
appena creato è come rimettere qualcuno alla guida nel punto esatto in cui lo
avevi lasciato, ma senza dirgli che sta arrivando in curva: la posizione è
giusta, la velocità no, ed è troppa.

La direzione sorprende: dopo una ripresa fatta così il primo passo è il più
lungo che quella manopola consenta, perché un ottimizzatore appena nato non ha
ancora nessun motivo per moderarsi. La corsa non interrotta, alla stessa
altezza, ne avrebbe fatto uno molto più corto. Di quanto più corto dipende dal
problema; che sia più corto, sempre.

Il rimedio costa una riga: nel file si mette anche lo stato
dell'ottimizzatore, e al ritorno lo si ricarica. Lo stesso vale per qualunque
altro pezzo del programma che tenga il conto di quello che è successo finora,
perché un file per riprendere è un fascicolo, con dentro più di una cosa.
`````

`````{tab} Superiore
Con SGD nudo la questione è marginale; con `optim.Adam`, quello del programma
su MNIST, i momenti $m$ e $v$ *sono* stato, e
ripartire senza di essi non riprende la stessa traiettoria. La parte
strutturale, quella che vale su qualunque problema, è questa: la correzione
del bias riparte da $t = 1$, e a $t = 1$ il rapporto
$\hat{m}/(\sqrt{\hat{v}} + \varepsilon)$ vale $\pm 1$ per costruzione, quindi
il primo aggiornamento è **$\eta$ pieno**, il passo più lungo che quella
manopola consenta. Qui $\hat{m}$ e $\hat{v}$ sono i due momenti corretti per il
bias ($m$ e $v$ divisi per $1-\beta_1^t$ e $1-\beta_2^t$) ed $\varepsilon$ è il
termine minuscolo che evita la divisione per zero: a $t = 1$ quelle correzioni
danno $\hat{m} = g$ e $\hat{v} = g^2$, con $g$ il gradiente, da cui il rapporto
$\pm 1$. Ricaricando lo stato, invece, il passo coincide
esattamente con quello della traiettoria mai interrotta.

Di quanto sia più lungo dipende dal problema, e quindi va detto su quale è
misurato e come. Il problema: una quadratica
$\mathcal{L}(\theta) = \frac{1}{2}\|\theta\|^2$ con cento parametri
inizializzati da una normale standard, venti passi di Adam con $\eta = 0{,}1$,
poi la ripresa. La misura: la media quadratica dello spostamento sui cento
parametri, al primo passo dopo la ripresa. Viene $0{,}100$ senza lo stato
dell'ottimizzatore (cioè esattamente $\eta$, come previsto) contro $0{,}0294$
ricaricandolo (torch 2.13). Un fattore $3{,}4$ qui, un fattore $2{,}4$ sulla
stessa quadratica con un parametro solo, e nessun messaggio d'errore in nessuno
dei due casi.

Lo stesso vale per tutto ciò che ha uno `state_dict` e che il ciclo tocca:
lo *scheduler* del learning rate, il `GradScaler` della precisione mista, il
`sampler` distribuito. Un checkpoint completo è un dizionario, non un tensore.
`````

Ecco la forma minima, quella che si scrive una volta e si copia in ogni
progetto:

```python
from torch import optim

optimizer = optim.Adam(model.parameters(), lr=1e-3)

# checkpoint per RIPRENDERE: i pesi da soli non bastano
torch.save({"epoca": 5,
            "modello": model.state_dict(),
            "ottimizzatore": optimizer.state_dict()}, "checkpoint.pt")

stato = torch.load("checkpoint.pt")
model.load_state_dict(stato["modello"])
optimizer.load_state_dict(stato["ottimizzatore"])   # la riga che si dimentica
print(f"ripresa dall'epoca {stato['epoca']}; lo stato dell'ottimizzatore "
      f"ha le chiavi {list(stato['ottimizzatore'])}")
# ripresa dall'epoca 5; lo stato dell'ottimizzatore ha le chiavi ['state', 'param_groups']
```

Le due chiavi stampate dicono che cosa c'era da salvare: in `state` c'è la
memoria accumulata su ciascun peso, in `param_groups` ci sono le impostazioni
dell'ottimizzatore, il learning rate per primo. Senza `state`, la ripresa
riparte senza memoria.

La sezione [dal notebook agli script](dal-notebook-agli-script.md) riprende
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
  `zero_grad()` → `backward()` → `step()`. Il terzo serve perché i gradienti
  si accumulano, e non li azzera: li toglie (`p.grad` torna a `None`).
  Nell'accumulo dei gradienti si fa una volta ogni $k$ micro-batch, ed è
  l'unica eccezione all'ordine.
- `Dataset` consegna gli esempi, `DataLoader` li rimescola e li impila in
  **mini-batch**: il gradiente sul batch è una stima rumorosa ma economica di
  quello vero.
- In valutazione: `model.eval()` spegne il dropout e passa la batch norm alle
  **medie mobili** (non la spegne: continua a normalizzare); `torch.no_grad()`
  sospende autograd. Servono entrambi.
- Le curve di training e validazione diagnosticano l’**overfitting**;
  l'early stopping in PyTorch è un semplice `if` nel loop.
- Si salva lo **`state_dict`** (`torch.save`/`load_state_dict`), non
  l'oggetto: contiene parametri **e** buffer. Per *riprendere* servono anche
  `ottimizzatore.state_dict()` e l'epoca; senza, con Adam il primo passo dopo
  la ripresa è quello di un ottimizzatore appena nato.
```
`````
