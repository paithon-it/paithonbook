# PyTorch: costruire reti in pratica

C'è stato un periodo, tra il 2011 e il 2016, in cui per fare deep learning
all'avanguardia conveniva imparare **Lua**, un linguaggio di scripting nato in
Brasile e famoso soprattutto per gli *addon* di World of Warcraft. Il motivo
si chiamava **Torch**, una libreria di calcolo scientifico potente e veloce. Il
termine tornerà spesso: una *libreria*, in informatica, è una cassetta di
attrezzi già pronti che qualcun altro ha scritto e che tu chiami dal tuo
programma invece di rifarli da capo. Quella cassetta la usavano il gruppo di
Yann LeCun alla New York University e i primi anni di DeepMind, ed era scritta
appunto in Lua. Nel 2016, nei laboratori di Facebook AI Research (oggi Meta
AI), un piccolo gruppo (tra cui lo stagista Adam Paszke, Sam Gross e Soumith
Chintala) decise di rifare da zero la parte con cui ci si parla, il manico
degli attrezzi, riscrivendola in Python e lasciando intatto il motore di
calcolo sotto. Il risultato, uscito in versione di prova a inizio 2017, si
chiama **PyTorch** {cite}`paszke2019pytorch`. In
pochi anni è diventato lo strumento standard della ricerca mondiale
sull'intelligenza artificiale: la quasi totalità dei modelli pubblicati su
Hugging Face (il grande archivio pubblico dove la comunità condivide i propri
modelli) e gran parte degli articoli scientifici recenti sono scritti così.

## La filosofia: il grafo si costruisce mentre giri

La scelta di fondo di PyTorch (quella che l'ha fatto vincere) riguarda
*quando* i conti vengono eseguiti. Il **grafo** del titolo è la parola tecnica
per una cosa semplice: l'elenco delle operazioni da fare e delle frecce che le
collegano, cioè quale risultato serve a quale conto successivo. È la ricetta,
prima che qualcuno la esegua; e la domanda è se vada scritta tutta in anticipo
o se possa nascere mentre si cucina.

`````{tab} Elementare
Due modi di seguire un percorso in auto. Il primo: stampi l'itinerario completo
prima di partire e lo esegui alla lettera; se una strada è chiusa, devi tornare
a casa e ristampare tutto. Il secondo: usi il navigatore, che ricalcola strada
facendo e a ogni incrocio sa dove sei. I primi **framework** di deep learning
(un framework è una libreria abbastanza grande da dettare anche il modo in cui
si scrive il programma, non solo da offrire funzioni pronte) funzionavano nel
primo modo: prima descrivevi *tutta* la rete, cioè scrivevi tutto il grafo, poi
la consegnavi al motore ed eseguivi, e se qualcosa andava storto capirlo era
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

Nel 2017 il posto di PyTorch era già occupato. Lo teneva **TensorFlow**, la
libreria di Google, uscita nel 2015 e allora usata praticamente da tutti: era
lei lo strumento con cui si faceva deep learning, e PyTorch era l'ultimo
arrivato. Cinque anni dopo i rapporti si erano invertiti, almeno nei
laboratori, dove ormai la maggior parte degli articoli scientifici dichiara di
aver usato PyTorch.

Le ragioni sono meno misteriose di quanto sembri. Un ricercatore passa le
giornate a *provare idee strane*, e quasi sempre a sbagliarle: gli serve uno
strumento che si lasci aprire e guardare dentro riga per riga, mentre gira. Con
un grafo da scrivere tutto in anticipo, invece, si finisce per pensare in due
lingue insieme, Python e quella del grafo, e il tempo che si perde è quello
buono. Attorno a questa comodità è cresciuto un circolo virtuoso: i
paper pubblicano codice PyTorch, chi vuole riprodurli usa PyTorch, le librerie
di alto livello (Hugging Face Transformers, PyTorch Lightning, torchvision)
nascono PyTorch-first. Nel 2022 Meta ha ceduto il progetto alla neonata
**PyTorch Foundation** sotto la Linux Foundation: da progetto aziendale a bene
comune dell'ecosistema, com'era successo a Linux stesso.

Onestà impone di dire che non ha vinto *ovunque*: TensorFlow resta diffuso
dove il modello non si studia più ma si usa per lavoro, dentro i sistemi di
un'azienda o dentro un'app del telefono. E i concetti sono identici nei due
mondi: imparato uno, l'altro si legge senza fatica.

## Lo stack: Python sopra, C++ sotto

La comodità di PyTorch potrebbe far pensare a uno strumento lento, visto che
Python non è famoso per la velocità. Il trucco è che Python è solo la
superficie ({numref}`fig-stack-pytorch`).

```{figure} ../figures/stack-pytorch.svg
:name: fig-stack-pytorch
:alt: "Diagramma a strati: alla base l'hardware con CPU, GPU e MPS; sopra il motore C++ con ATen e autograd; sopra ancora l'API Python con torch, torch.nn e torch.optim; in cima il tuo modello."
:width: 85%

Lo stack: il tuo modello è normale codice Python, ma ogni operazione scende
nel motore di calcolo scritto in C++ e da lì sull'hardware disponibile.
```

`````{tab} Elementare
In un ristorante la sala (il menu, il cameriere che prende l'ordine) è Python:
accogliente, flessibile, parla la tua lingua. La cucina è scritta in C++:
quando ordini "moltiplica queste due matrici", il piatto viene preparato da
cuochi velocissimi (routine di calcolo compilate) che sfruttano tutti i fuochi
disponibili, dalla CPU alla scheda grafica. Tu non entri mai in cucina: ordini
in Python, e la velocità è quella della cucina, non della sala.
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

Due parole prima del comando, perché compaiono subito e conviene averle. La
**GPU** è la scheda grafica, il chip nato per i videogiochi che si è rivelato
bravissimo a fare tanti conti identici tutti insieme; **CUDA** è il nome che
NVIDIA, che quelle schede le costruisce, dà al modo in cui i programmi le
parlano (per questo nel codice il dispositivo si chiamerà `"cuda"` e non
`"gpu"`). La prossima sezione riprende entrambe con calma.

PyTorch si installa come qualunque pacchetto Python, cioè scrivendo una riga
nel **terminale**, la finestra in cui si danno comandi scritti al computer; sul
sito ufficiale (`pytorch.org`) un selettore genera la riga adatta al proprio
sistema operativo e alla propria scheda. Per tutto questo capitolo basta la
versione senza GPU:

```bash
pip install torch torchvision --index-url https://download.pytorch.org/whl/cpu
```

I pacchetti sono due: `torch` è PyTorch, `torchvision` è la sua cassetta di
attrezzi per le immagini (dataset pronti, trasformazioni, modelli
pre-addestrati), e serve dalla sezione sull'addestramento in poi.

L'indirizzo in coda conviene non saltarlo. Su Linux il pacchetto che `pip`
(il programma che scarica e installa le librerie di Python) prende di sua
iniziativa si porta dentro le librerie CUDA: qualche gigabyte che su una
macchina senza scheda grafica non serve a niente, e che paghi in banda e in
disco senza accorgertene. Su Windows e su macOS il pacchetto predefinito è già
quello per la sola CPU, e la riga si accorcia a
`pip install torch torchvision`.

Ed ecco il primo contatto, un assaggio delle due cose che vedremo nelle
prossime sezioni: i **tensori** (le scatole in cui PyTorch tiene i numeri) e i
gradienti automatici.

```python
import torch

print(torch.__version__)          # versione installata
print(torch.cuda.is_available())  # True se c'è una GPU NVIDIA utilizzabile

x = torch.tensor(3.0, requires_grad=True)  # un tensore "osservato"
y = x**2 + 2*x                             # y = x² + 2x, calcolato subito
y.backward()                               # gradiente automatico
print(x.grad)                              # la derivata di y in x=3 -> tensor(8.)
```

Niente da dichiarare in anticipo e niente da preparare prima di partire: si
scrivono i conti come si scriverebbero su un foglio, e la **derivata** esce da
sola. Se il termine non ti dice niente, basta questo: la derivata misura quanto
cambia il risultato quando l'ingresso si sposta di un soffio. Qui vale $8$, e
$8$ vuol dire questo: sposta $x$ da $3$ a $3{,}01$, cioè di un centesimo, e $y$
cresce di circa otto centesimi, otto volte tanto. Il conto si può rifare a
mano: $y$ vale $15$ in $x = 3$ e $15{,}0801$ in $x = 3{,}01$. Quel numero, nel
mestiere, si chiama **gradiente**, ed è quello che la riga `y.backward()`
calcola e che PyTorch deposita in `x.grad`;
chi ricorda le regole del capitolo di richiami matematici riconoscerà che la
derivata di $x^2 + 2x$ è $2x + 2$, che in $x = 3$ vale appunto $8$. In
miniatura, è il meccanismo che addestra ogni rete neurale di questo libro.

## Che cosa si impara qui

Due movimenti, dal mattone al mestiere.

Il primo mette in mano gli attrezzi, e sono tre. I **tensori**, le scatole di
numeri su cui tutto si appoggia, insieme al meccanismo che calcola le derivate
da solo. I **moduli**, cioè come si mette insieme un modello pezzo per pezzo, e
come si misura quanto sbaglia. L’**addestramento**, cioè il giro di cinque
mosse che in PyTorch si scrive a mano invece di chiederlo a un comando, e che
qui si vede all'opera su un problema vero: leggere cifre scritte a mano.

Il secondo insegna a usarli su un problema che non è un esercizio. Il **flusso
di lavoro**, cioè l'ordine delle mosse che si ripete in ogni progetto e il
ciclo con cui un modello si migliora. I **dati su misura**: come si porta
dentro la rete una cartella di file propri, con `Dataset`, `DataLoader` e
trasformazioni. I **tre errori più comuni** (forma, tipo, dispositivo), che da
soli si prendono metà del tempo perso da chi comincia. Il passaggio **dal
notebook agli script**, quando un esperimento va reso ripetibile. E infine
**replicare un paper**, cioè un articolo scientifico: il metodo per
trasformare quattro equazioni in codice che gira. Chiude il capitolo una
sezione sulle **prestazioni**, per quando il modello funziona ma è troppo
lento.

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

Un ripasso della pagina, per chi legge di seguito e per chi torna a
consultarla.

`````{tab} Elementare
```{admonition} Da ricordare
:class: important
- **PyTorch** nasce nel 2016 nei laboratori di Facebook (oggi Meta) come
  versione in Python di un vecchio strumento chiamato **Torch**, che si usava
  con un altro linguaggio; oggi non appartiene più a un'azienda sola.
- Ogni riga viene **eseguita subito**, come su una calcolatrice: puoi fermarti
  a guardare i numeri in qualunque punto, e correggere un errore è come
  correggerlo in un normale programma Python.
- Python è la sala del ristorante; la cucina è scritta in un linguaggio più
  veloce e sta sotto, invisibile. Tu ordini in Python, la velocità è quella
  della cucina.
- È lo strumento con cui oggi si fa quasi tutta la ricerca. Non è l'unico:
  imparato questo, gli altri si leggono senza fatica, perché le idee (numeri
  in scatole, strati, gradienti) sono le stesse.
```
`````

`````{tab} Superiore
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
`````
