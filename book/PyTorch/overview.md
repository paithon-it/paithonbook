# PyTorch: costruire reti in pratica

C'è stato un periodo, tra il 2011 e il 2016, in cui per fare deep learning
all'avanguardia conveniva imparare **Lua**, un linguaggio di scripting nato in
Brasile e famoso soprattutto per gli *addon* di World of Warcraft. Il motivo
si chiamava **Torch**: una libreria di calcolo scientifico potente e veloce,
usata dal gruppo di Yann LeCun alla New York University e nei primi anni di
DeepMind, ma scritta appunto in Lua. Nel 2016, nei laboratori di Facebook AI
Research (oggi Meta AI), un piccolo gruppo (tra cui lo stagista Adam Paszke,
Sam Gross e Soumith Chintala) decise di rifare l'interfaccia da zero in
Python, conservando il motore di calcolo. Il risultato, rilasciato in beta
pubblica a inizio 2017, si chiama **PyTorch** {cite}`paszke2019pytorch`. In
pochi anni è diventato lo strumento standard della ricerca mondiale
sull'intelligenza artificiale: la quasi totalità dei modelli pubblicati su
Hugging Face (il grande archivio pubblico dove la comunità condivide i propri
modelli) e gran parte degli articoli scientifici recenti sono scritti così.

## La filosofia: il grafo si costruisce mentre giri

La scelta di fondo di PyTorch (quella che l'ha fatto vincere) riguarda
*quando* i conti vengono eseguiti.

`````{tab} Elementare
Due modi di seguire un percorso in auto. Il primo: stampi l'itinerario completo
prima di partire e lo esegui alla lettera; se una strada è chiusa, devi tornare
a casa e ristampare tutto. Il secondo: usi il navigatore, che ricalcola strada
facendo e a ogni incrocio sa dove sei. I primi framework di deep learning
funzionavano nel primo modo: prima descrivevi *tutta* la rete, poi la
consegnavi al motore ed eseguivi, e se qualcosa andava storto capirlo era
un'impresa. PyTorch funziona come il navigatore: ogni riga di codice viene
eseguita **subito**, puoi fermarti a guardare i numeri in qualunque punto, e
correggere è facile come in qualsiasi programma Python.
`````

`````{tab} Superiore
È il paradigma **define-by-run** (reso popolare dal framework giapponese
Chainer nel 2015): il grafo delle operazioni non viene dichiarato in anticipo
ma **costruito dinamicamente** durante l'esecuzione, registrando le operazioni
man mano che avvengono. Il contrario del *define-and-run* di TensorFlow 1.x,
dove si compilava un grafo statico da eseguire in una `Session`. Le
conseguenze pratiche: il *control flow* è normale Python (`if`, `for`,
ricorsione) anche dentro il modello; il debugging usa gli strumenti ordinari
(`print`, `pdb`); reti a struttura variabile (sequenze di lunghezza diversa,
alberi) si scrivono in modo naturale. Il costo storico era la minore
ottimizzazione rispetto a un grafo compilato; da PyTorch 2.0 (2023)
`torch.compile` recupera il divario compilando *just-in-time* il codice Python
in kernel ottimizzati, senza cambiarne una riga. TensorFlow stesso, con la
versione 2.0 del 2019, è passato all'esecuzione *eager* di default: su questo
punto la storia ha dato ragione a PyTorch.
`````

## Perché ha vinto nella ricerca

Nel 2017 TensorFlow dominava e PyTorch era l'ultimo arrivato; cinque anni dopo
i rapporti si erano invertiti, almeno nei laboratori. Secondo i dati di Papers
with Code, già nel 2022 circa tre articoli su quattro tra quelli che
dichiarano il framework usavano PyTorch. Le ragioni sono meno misteriose di
quanto sembri: un ricercatore passa le giornate a *provare idee strane*, e uno
strumento che si lascia ispezionare riga per riga, che non chiede di imparare
un secondo linguaggio mentale oltre a Python, fa risparmiare esattamente il
tempo che conta. Attorno a questa comodità è cresciuto un circolo virtuoso: i
paper pubblicano codice PyTorch, chi vuole riprodurli usa PyTorch, le librerie
di alto livello (Hugging Face Transformers, PyTorch Lightning, torchvision)
nascono PyTorch-first. Nel 2022 Meta ha ceduto il progetto alla neonata
**PyTorch Foundation** sotto la Linux Foundation: da progetto aziendale a bene
comune dell'ecosistema, com'era successo a Linux stesso.

Onestà impone di dire che non ha vinto *ovunque*: TensorFlow resta diffuso in
produzione industriale e su mobile, e i concetti (tensori, strati, gradienti,
ottimizzatori) sono identici nei due mondi. Imparato uno, l'altro si legge
senza fatica.

## Lo stack: Python sopra, C++ sotto

La comodità di PyTorch potrebbe far pensare a uno strumento lento, visto che
Python non è famoso per la velocità. Il trucco è che Python è solo la
superficie ({numref}`fig-stack-pytorch`).

```{figure} ../figures/stack-pytorch.svg
:name: fig-stack-pytorch
:alt: "Diagramma a strati: alla base l'hardware con CPU, GPU e MPS; sopra il motore C++ con ATen e autograd; sopra ancora l'API Python con torch, torch.nn e torch.optim; in cima il tuo modello."
:width: 85%

Lo stack: il tuo modello è normale codice Python, ma ogni operazione scende
nel motore C++ (ATen e autograd) e da lì sull'hardware disponibile.
```

`````{tab} Elementare
Pensa a un ristorante. La sala (il menu, il cameriere che prende l'ordine) è
Python: accogliente, flessibile, parla la tua lingua. La cucina è scritta in
C++: quando ordini "moltiplica queste due matrici", il piatto viene preparato
da cuochi velocissimi (routine di calcolo compilate) che sfruttano tutti i
fuochi disponibili, dalla CPU alla scheda grafica. Tu non entri mai in cucina:
ordini in Python, e la velocità è quella della cucina, non della sala.
`````

`````{tab} Superiore
L'API Python (`torch`, `torch.nn`, `torch.optim`, `torch.utils.data`) è un
guscio sottile sopra **ATen**, la libreria C++ dei tensori, e sopra il motore
**autograd** che registra le operazioni e calcola i gradienti. Un *dispatcher*
smista ogni operazione al kernel giusto per il dispositivo del tensore: BLAS
e simili su CPU, kernel CUDA/cuDNN su GPU NVIDIA, Metal Performance Shaders
(MPS) su Apple Silicon. Per il passaggio in produzione: `torch.compile`
(fusione e compilazione JIT dei kernel), l'esportazione in **ONNX** verso
runtime esterni, ed **ExecuTorch** per mobile ed embedded. La divisione del
lavoro è netta: Python decide *cosa* calcolare, il motore C++ decide *come*.
`````

## Installazione e primo contatto

PyTorch si installa come qualunque pacchetto Python; sul sito ufficiale
(`pytorch.org`) un selettore genera il comando adatto a sistema operativo e
GPU. La versione CPU, sufficiente per tutto questo capitolo, è una riga:

```bash
pip install torch torchvision
```

E questo è il primo contatto, un assaggio delle due cose che vedremo nelle
prossime sezioni, i tensori e i gradienti automatici:

```python
import torch

print(torch.__version__)          # versione installata
print(torch.cuda.is_available())  # True se c'è una GPU NVIDIA utilizzabile

x = torch.tensor(3.0, requires_grad=True)  # un tensore "osservato"
y = x**2 + 2*x                             # y = x² + 2x, calcolato subito
y.backward()                               # gradiente automatico
print(x.grad)                              # dy/dx = 2x + 2 -> tensor(8.)
```

Nessuna sessione da aprire, nessun grafo da compilare: tre righe di algebra e
una **derivata** calcolata da sola. Se il termine non ti dice niente, basta
questo: la derivata misura quanto cambia il risultato quando l'ingresso si
sposta di un soffio, e qui dice che attorno a $x = 3$ il valore di $y$ cresce
$8$ volte più in fretta di $x$. È il numero che il commento nel codice chiama
**gradiente** e che PyTorch deposita in `x.grad`; chi ricorda le regole del
capitolo di richiami matematici può verificarlo (la derivata di $x^2 + 2x$ è
$2x + 2$, che in $x = 3$ vale $8$). In miniatura, è il meccanismo che addestra
ogni rete neurale di questo libro.

## Come è organizzato il capitolo

Due movimenti, dal mattone al mestiere.

Il primo mette in mano gli attrezzi. I **tensori**: le scatole di numeri su
cui tutto si appoggia, con le loro operazioni e il meccanismo **autograd** che
calcola i gradienti da solo. I **moduli**: come si costruisce un modello con
`nn.Module` e `nn.Sequential`, e come si misura l'errore con le funzioni di
perdita. L'**addestramento**: il training loop scritto per esteso (è la firma
stilistica di PyTorch) e un esempio completo sulle cifre scritte a mano di
MNIST.

Il secondo insegna a usarli su un problema che non è un esercizio. Il **flusso
di lavoro**, cioè l'ordine delle mosse che si ripete in ogni progetto e il
ciclo con cui un modello si migliora. I **dati su misura**: come si porta
dentro la rete una cartella di file propri, con `Dataset`, `DataLoader` e
trasformazioni. I **tre errori più comuni** (forma, tipo, dispositivo) che
valgono da soli la metà del tempo perso da chi comincia. Il passaggio **dal
notebook agli script**, quando un esperimento va reso ripetibile. E infine
**replicare un paper**: il metodo per trasformare quattro equazioni in codice
che gira, verificato sul Vision Transformer. Chiude il capitolo una sezione
sulle **prestazioni**, per quando il modello funziona ma è troppo lento.

L'obiettivo è che a fine capitolo tu sappia leggere, e scrivere, il codice con
cui oggi si fa ricerca in deep learning.

```{admonition} Se vuoi lo stesso percorso in forma di corso
:class: seealso
Il libro spiega *come funziona*; per esercitarsi con i notebook alla mano, il
corso gratuito [Learn PyTorch for Deep
Learning](https://www.learnpytorch.io/) di Daniel Bourke copre lo stesso
terreno in inglese, con codice eseguibile e video. È una buona palestra
parallela alla lettura di questo capitolo.
```

```{admonition} Da ricordare
:class: important
- **PyTorch** nasce nel 2016 a Facebook AI Research (oggi Meta AI) come
  interfaccia Python del vecchio motore **Torch** (Lua); beta pubblica a
  inizio 2017, dal 2022 governato dalla **PyTorch Foundation**.
- Filosofia **define-by-run**: ogni operazione è eseguita subito e il grafo
  dei calcoli si costruisce dinamicamente; debugging e control flow sono
  normale Python.
- Python è la superficie: sotto lavorano **ATen** e **autograd** in C++, con
  kernel dedicati per CPU, GPU (CUDA) e Apple Silicon (MPS);
  `torch.compile` (PyTorch 2.0) aggiunge la compilazione JIT.
- È lo standard *de facto* della ricerca; TensorFlow resta un onesto
  vicino di casa, e i concetti si trasferiscono senza attrito.
```
