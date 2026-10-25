---
titolo_ricerca: "Python: il linguaggio dell'intelligenza artificiale"
descrizione: "Perché Python è il linguaggio dell'AI: com'è nato, come si legge,
  quali librerie scientifiche lo accompagnano e come preparare l'ambiente di
  lavoro."
---
# Python: il linguaggio dell'intelligenza artificiale

```{image} ../figures/aperture/python.png
:class: pt-apertura only-light
:width: 100%
:alt: Un pitone arrotolato su un quaderno aperto, sotto una piccola stella di Natale.
```

```{image} ../figures/aperture/python-scura.png
:class: pt-apertura only-dark
:width: 100%
:alt: Un pitone arrotolato su un quaderno aperto, sotto una piccola stella di Natale.
```

Nel Natale del 1989 un programmatore olandese, Guido van Rossum, si annoia. Gli
uffici del centro di ricerca dove lavora ad Amsterdam sono chiusi per le feste,
e lui riempie il tempo scrivendo, per hobby, un nuovo linguaggio di
programmazione. Lo chiama **Python**: non per il serpente, ma per i Monty
Python, il gruppo comico inglese di cui è fan. La prima versione pubblica esce
nel 1991.

Oggi quel passatempo è il linguaggio in cui si scrivono i principali framework
di deep learning e gran parte del codice che accompagna la ricerca sul machine
learning, dalla rivoluzione del 2012 ai grandi modelli linguistici attuali. Ci
è arrivato grazie a una divisione del lavoro: programmi scritti in linguaggi più
veloci e più faticosi da usare eseguono i calcoli pesanti, e Python li comanda.

Questa divisione c'era già in AlexNet, la rete che nel 2012 aprì la stagione
del deep learning. Il calcolo era scritto in C++ e in CUDA, l'estensione di C++
con cui si programmano le schede grafiche NVIDIA, e sopra c'era uno strato di
Python per configurarla e lanciarla.

Che quel livello superiore fosse Python non era scontato. Negli anni seguenti,
infatti, diversi laboratori di punta, fra cui Facebook AI Research, scrivevano
in Lua con Torch, mentre Python aveva già Theano, nato in ambito accademico
prima del 2012. In definitiva, la partita si è chiusa con librerie che si
scrivono in Python e calcolano in C++. La storia di PyTorch, cioè Torch rifatto
per Python, apre il {doc}`capitolo che porta il suo nome </PyTorch/overview>`.

Python non è il linguaggio più veloce né il più elegante in senso accademico,
e ha prevalso per altre ragioni. In questo capitolo vediamo innanzitutto quali
sono, e passiamo poi agli strumenti per scrivere ed eseguire il codice
(l'interprete e i notebook) e alla preparazione dell'ambiente di lavoro.
Seguono le basi del linguaggio e tre librerie: NumPy, Pandas e Matplotlib.

## Perché proprio Python

`````{tab} Elementare

Un **programma** è un elenco di istruzioni scritte in un file di testo, che il
computer esegue una dopo l'altra, dall'alto in basso: un foglio di ordini in
fila. Python si legge quasi come un inglese scarno. Ad esempio, una riga come
`if voto >= 18: print("promosso")` si indovina anche senza averla mai scritta,
perché è come la diresti a voce: «se il voto è almeno diciotto, scrivi
*promosso*» (all'università i voti vanno in trentesimi, e il diciotto è la
sufficienza).

Le regole di scrittura di un linguaggio (dove vanno i due punti, gli a capo,
le parentesi) si chiamano la sua **sintassi**. Quella di Python è breve e si
impara in fretta, e così il pensiero resta su *che cosa* dire invece che su
*come* scriverlo.

Python è anche gratuito e ha una comunità enorme. Per quasi ogni problema
(leggere un file, disegnare un grafico, addestrare una rete neurale) qualcuno
ha già scritto una libreria, un pacchetto di istruzioni confezionate da altri,
che ti risparmia il lavoro. Basta importarla, cioè scrivere una riga in
cima al programma, `import numpy`, per dire a Python «da qui in poi voglio
usare anche questa», e da quel momento tutti gli strumenti della libreria sono
a disposizione.

Con Python arriva già pronta una **libreria standard** (`math` per le
funzioni matematiche, `time` per misurare il tempo, e decine di altre), mentre
le librerie scientifiche vanno installate, una volta sola, con un programma
apposta che si chiama `pip`. La riga di importazione ha due varianti che si
incontrano subito: `import numpy as np` dà alla libreria un nome più corto, e
`from math import sqrt` porta dentro un solo nome, da usare senza prefisso.

Nella ricerca sul machine learning, inoltre, un lavoro nuovo esce molto spesso
con il suo codice allegato, e quasi sempre in Python. Così chi legge può rifare
l'esperimento invece di crederci sulla parola, e chi lo rifà parte da dove
l'altro è arrivato. A rendere possibile questo scambio è un linguaggio comune a
tutti.

[Sui calcoli pesanti Python, da solo, è lento, ma quei calcoli non li fa quasi
mai lui]{.pt-matita}. Le librerie importanti, infatti, sono scritte in
linguaggi più veloci e più faticosi da usare, e Python fa il capocantiere:
legge il progetto e dà gli ordini alle squadre. Tu scrivi la riga chiara, e il
muro lo tira su chi ha gli attrezzi per farlo in fretta.

`````

`````{tab} Superiore

CPython, l'implementazione di riferimento, compila il sorgente in *bytecode* e
lo esegue su una macchina virtuale a pila. Poiché i tipi sono dinamici, ogni
operazione va risolta a runtime sul tipo degli operandi. L'interprete adattivo
della PEP 659 (Python 3.11) specializza il bytecode sui tipi che osserva, e il
JIT della PEP 744 (sperimentale dal 3.13) prova ad andare oltre, ma il costo per
operazione resta comunque molto sopra quello del codice compilato. In altri
termini, si cede velocità di esecuzione in cambio di velocità di sviluppo.

Il calcolo scientifico in Python convive con il *problema dei due linguaggi*
(prototipare in un linguaggio comodo e riscrivere in uno veloce, il problema da
cui è nato Julia) e lo risolve ripartendolo per ruoli: chi sviluppa le librerie
scrive nel secondo linguaggio, e chi le usa resta nel primo. Di conseguenza [le
operazioni pesanti non girano affatto in Python puro]{.pt-matita}. NumPy, ad
esempio, è scritto in C, PyTorch e TensorFlow in C++ e CUDA. Python fa da
**collante** (*glue language*), orchestrando componenti compilate che sfruttano
BLAS, SIMD e GPU.

Attorno al linguaggio è cresciuto un ecosistema di centinaia di migliaia di
pacchetti (distribuiti via PyPI e `pip`) e una prassi di ricerca in cui i
paper arrivano spesso con il codice allegato. In questo modo riprodurre un
risultato diventa praticabile, e Python è il denominatore comune.

`````

## Un linguaggio pensato per essere letto

La filosofia di Python è così esplicita da essere scritta dentro il linguaggio
stesso, e basta un'istruzione per farsela stampare:

```python
import this   # stampa lo "Zen of Python"
```

Si noti che metà di quella riga non è un'istruzione. Il cancelletto `#`,
infatti, apre un **commento**, e tutto il testo che lo segue sulla stessa riga
è scritto per chi legge, non per il computer, che lo salta. I commenti dicono
che cosa fa la riga accanto, e la freccia `->` dentro un commento significa «e
viene fuori questo».

Quel «viene fuori» segue una convenzione. Nell'interprete, la finestra in cui
le istruzioni si scrivono una per volta, un'espressione digitata e confermata
con Invio mostra il proprio valore; ad esempio, scrivendo `2 + 3` compare `5`.
In un programma salvato in un file, invece, la stessa riga calcola e non
mostra niente, e per vedere il risultato bisogna stamparlo con `print`. Gli
esempi che seguono sono scritti come li si digiterebbe nell'interprete, perciò
spesso una riga mostra un valore senza `print` accanto.

Eseguendo `import this` compaiono i diciannove aforismi che Tim Peters codificò
nel 1999, fra i quali *"Explicit is better than implicit"* («meglio esplicito
che implicito»), *"Simple is better than complex"* («meglio semplice che
complicato») e *"Readability counts"* («la leggibilità conta»). [Il codice,
infatti, si legge molte più volte di quante lo si scriva]{.pt-matita}, e in un
progetto condiviso ciò che non si capisce non si può nemmeno verificare. Questa
chiarezza rende Python adatto a insegnare e a comunicare l'AI, ma per lavorarci
serve anche altro, e lo fornisce l'ecosistema.

## L'ecosistema scientifico

Il linguaggio da solo non basta: la forza di Python nell'AI sta nella torre di
librerie costruite l'una sull'altra ({numref}`fig-stack-python`; in inglese
quella torre si chiama *stack*).

- NumPy: il fondamento. Introduce l’*array* N-dimensionale, una griglia di
  numeri con quante dimensioni si vuole (una fila, una tabella, una pila di
  tabelle) su cui i calcoli si fanno tutti insieme. Rende veloce l’algebra
  lineare, cioè la matematica delle tabelle di numeri, e quasi tutto il resto
  poggia su di lui.
- Pandas: dati tabellari. Il `DataFrame` è un foglio di calcolo
  programmabile, con cui si caricano, si puliscono e si trasformano i dati
  prima di passarli a un modello.
- Matplotlib: visualizzazione. I grafici con cui esplori i dati e racconti
  i risultati.
- scikit-learn: la raccolta di strumenti del machine learning *classico*, quello
  che viene prima delle reti neurali. Contiene decine di modelli diversi, e li
  si comanda tutti con le stesse due parole: `fit` (impara da questi dati) e
  `predict` (adesso prevedi). Dopo averne usato uno, l'interfaccia degli altri è
  già nota, e per questo l'API si dice uniforme (l’**API** di una libreria è
  l'insieme dei comandi con cui le si parla).
- PyTorch (Facebook AI Research, oggi Meta), deep learning: costruisce reti
  neurali e le addestra, calcolando da sé le derivate che servono a correggerne
  i parametri (la *differenziazione automatica*), anche sulla GPU, la scheda
  grafica. Non sta sopra NumPy, perché esegue il calcolo con un motore proprio,
  scritto in C++, e con NumPy scambia soltanto i dati. In particolare, sulla CPU
  `torch.from_numpy` e `Tensor.numpy()` condividono la stessa memoria invece di
  ricopiarli. Il suo concorrente storico è TensorFlow (Google, 2015), che si
  affianca alla torre nello stesso modo.

```{figure} ../figures/stack-scientifico-python.svg
:name: fig-stack-python
:alt: "Diagramma a strati: alla base Python, sopra NumPy, poi Pandas Matplotlib e scikit-learn, in cima PyTorch e TensorFlow."
:width: 85%

La torre delle librerie scientifiche di Python. Le due frecce sono i due
«poggia su» veri: NumPy sul linguaggio, e Pandas, Matplotlib e scikit-learn su
NumPy, senza il quale non funzionano. Sopra il secondo strato la
freccia manca: PyTorch e TensorFlow stanno accanto alla torre
più che sopra, perché fanno i calcoli con un motore
tutto loro, e con NumPy si limitano a scambiarsi i dati.
```

Dire che Pandas o scikit-learn sono «costruiti su NumPy» vuol dire che tengono i
numeri in array di NumPy e calcolano con le sue operazioni, invece di
riscriverle. Pandas, ad esempio, aggiunge le etichette, le colonne e il
raggruppamento delle righe, ma quando c'è da sommare un milione di valori, la
somma tocca a NumPy.

Il modo in cui NumPy esegue quella somma prende il nome di **vettorizzazione**.
Invece di dire al computer che cosa fare a un numero e poi ripeterglielo un
milione di volte, glielo si dice una volta sola sul blocco intero. [Quasi tutto
il guadagno sta nel lavoro che l'interprete svolge attorno a ogni operazione, e
non nell'aritmetica]{.pt-matita}.

`````{tab} Elementare

Dieci centesimi di pedaggio sono poca cosa; un milione di volte fanno centomila
euro.

Con il solo Python, raddoppiare un milione di numeri vuol dire percorrerli uno
per uno in un ciclo (un'istruzione che si ripete tante volte, spiegata nelle
{doc}`basi del linguaggio </Python/basi>`). A ogni giro Python paga il suo
pedaggio: apre la casella, guarda che cosa c'è dentro, si ricorda che quello è
un numero, cerca come si moltiplicano i numeri, e solo alla fine moltiplica. La
moltiplicazione dura un istante; il pedaggio è tutto il resto, e si paga a ogni
giro.

Con NumPy si scrive `2 * x`, dove `x` è il blocco intero dei numeri, e il
pedaggio si paga una volta sola, all'ingresso. Da lì in poi Python resta fuori:
dentro lavora un programma già tradotto in linguaggio macchina, che percorre il
blocco senza fermarsi a chiedersi che cosa contiene, e prende i numeri a
manciate invece che uno per volta, perché il processore sa moltiplicarne
parecchi con un'unica istruzione.

Perché il processore possa farlo, però, i numeri devono stare in fila ed essere
tutti dello stesso tipo, una condizione che NumPy impone e che una lista di
Python non rispetta. Per i calcoli fra intere tabelle di numeri, inoltre,
NumPy affida il lavoro a librerie specializzate che qualcuno ha passato
decenni a limare, le stesse che girano nei centri di calcolo.

In definitiva, fra il ciclo e la riga sola c'è un fattore cento, a volte mille.
Per questo le librerie scientifiche si sono messe d'accordo tutte sullo stesso
blocco di numeri, e così chi lo produce e chi lo consuma non hanno niente da
tradursi.

`````

`````{tab} Superiore

La vettorizzazione è una singola operazione dichiarata su un intero array e
delegata in blocco a codice compilato, invece di riattraversare il ciclo di
valutazione dell'interprete a ogni elemento. Un `ndarray` di NumPy è una vista
tipizzata su un blocco di memoria, contiguo nel caso più comune, e questa
tipizzazione rende possibile la delega, poiché il codice chiamato sa in
anticipo quanti byte è largo ogni elemento e come si combinano, mentre su una
lista di oggetti Python dovrebbe scoprirlo caso per caso.

Il costo che sparisce è dunque l’*overhead* dell'interprete su ogni iterazione,
e la sua scomparsa vale due o tre ordini di grandezza sul ciclo Python
equivalente. In particolare, le operazioni elemento-per-elemento vanno a cicli
in C ottimizzati e spesso a istruzioni **SIMD**, che fanno operare il processore
su più numeri con una sola istruzione; i prodotti fra matrici passano invece per
librerie **BLAS** dedicate, codice specializzato di terzi. Da qui la scelta
dell'array come struttura dati comune a tutto l'ecosistema: è insieme il formato
di scambio fra librerie e l'unità su cui il calcolo è veloce.

`````

## L'interprete, i notebook e il cloud

Python si può usare in tre modi, di crescente comodità per chi sperimenta.

`````{tab} Elementare

Il modo più diretto è l’**interprete interattivo**: apri una finestra, scrivi
un'istruzione, premi Invio e vedi subito il risultato, come una calcolatrice
con cui puoi conversare.

Un passo avanti è il **notebook Jupyter**, cioè un quaderno digitale fatto di
celle, dove testo, codice, grafici e formule convivono nella stessa pagina.
Scrivi una cella, la esegui, guardi il grafico che appare lì sotto. È il modo
in cui gran parte del machine learning viene insegnato e praticato.

Le celle condividono la memoria: quello che una ha calcolato resta a
disposizione delle altre. Questa comodità, però, ha un tranello. Se esegui le
celle in un ordine diverso da come stanno sulla pagina, [il quaderno segue
l'ordine dei tuoi clic, non quello delle righe]{.pt-matita}, e può mostrare un
risultato che, rileggendo la pagina, non torna. Il rimedio, quando succede, è
rieseguire tutto da capo, dall'alto in basso.

Infine, se non vuoi installare nulla, **Google Colab** ti dà un notebook nel
browser, gratuito e spesso con una scheda grafica in prestito: abbastanza per
addestrare un modello senza possedere un computer potente.

`````

`````{tab} Superiore

L'interprete interattivo è un **REPL** (*read–eval–print loop*); la versione
arricchita `IPython` ne è il motore. Un notebook Jupyter è un documento
`.ipynb` (JSON di celle) eseguito da un **kernel** che mantiene lo stato tra
una cella e l'altra. Da qui la potenza (e i tranelli) dell'esecuzione fuori
ordine. Poiché lo stato vive nel kernel e non nel file, [un notebook è
riproducibile solo se, riavviato il kernel e rieseguito dall'alto, produce gli
stessi risultati]{.pt-matita}.

**Google Colab** è un ambiente Jupyter gestito, con acceleratori hardware
(GPU/TPU) accessibili gratuitamente entro certi limiti, ideale per
prototipazione e didattica riproducibile.

`````

## Preparare l'ambiente: la prima riga eseguita davvero

Per provare gli esempi ci sono due strade: nel browser, senza installare
niente, oppure sul proprio computer.

### Senza installare niente

Quasi ogni capitolo esiste anche come notebook su Google Colab, con tutte le
sue celle in ordine e già pronte. Il notebook gira nel browser, le librerie
sono già installate, e per eseguire una cella si preme il triangolino che ha
accanto. Non serve altro che un browser e un account Google. È il modo più
rapido per provare gli esempi mentre si legge.

:::{only} html
Il collegamento, dove c'è, sta in testa al capitolo («Esegui il codice»).
:::

:::{only} latex
Il codice di ogni capitolo si esegue online, su book.paithon.it.
:::

### Sul proprio computer

Serve un **terminale**, cioè la finestra in cui si scrivono comandi al computer
invece di cliccare. Si chiama *Terminale* su macOS e Linux, *Prompt dei comandi*
(o *PowerShell*) su Windows. Serve anche Python 3.11 o successivo, perché le
versioni delle librerie usate negli esempi (NumPy 2, Pandas 3, Matplotlib 3.11)
non girano su un Python più vecchio.

Su Linux un Python di solito c'è già, ma una distribuzione non recente ne ha
uno più datato, quindi va controllato. Su macOS `/usr/bin/python3` è un
segnaposto: al primo uso propone di installare gli strumenti da sviluppatore di
Apple e contiene una versione vecchia, perciò anche lì Python si scarica da
`python.org`, come su Windows. Su Windows, durante l'installazione, si spunta
«Add Python to PATH», la casella che dice al terminale dove si trova Python;
senza di essa, il terminale risponde che Python non lo trova.

Fatto questo, i gesti sono quattro (su Windows il comando si chiama `python`,
oppure `py`, al posto di `python3`):

```text
python3 --version        # c'è? risponde con il numero, per esempio "Python 3.12.3"
python3                  # apre l'interprete: compaiono tre maggiori, >>>
>>> import this          # si scrive un'istruzione, si preme Invio
>>> exit()               # si esce
```

Quei tre maggiori si chiamano **prompt** e sono l'invito a scrivere, il modo in
cui l'interprete dice «tocca a te». Finché li vedi, sei dentro l'interprete e
non nel terminale.

Le istruzioni scritte all'interprete si perdono quando lo si chiude. Per
conservare un programma, quindi, lo si scrive in un file di testo con
estensione `.py` (per esempio `primo.py`, con dentro `print("ciao")`) e lo si
esegue con `python3 primo.py`.

Il file si può scrivere con qualunque editor di testo, ma conviene usarne uno
che conosca Python e segnali gli errori mentre scrivi; i due più diffusi sono
Visual Studio Code e PyCharm, gratuiti entrambi. Se un programma non finisce
più (capita: basta un ciclo scritto male) si ferma premendo `Ctrl+C`.

### Le librerie, e l'ambiente in cui metterle

Le librerie scientifiche non arrivano con Python: si installano con `pip`, il
programma che va a prenderle in rete (da PyPI, l'archivio pubblico dei
pacchetti Python) e le mette al posto giusto. Prima di installarle, però,
serve un posto dove metterle, per una ragione che di solito si impara a proprie
spese.

Hai un lavoro che gira, non lo tocchi da mesi, e nel frattempo per un esercizio
nuovo aggiorni una libreria. La versione nuova ha cambiato il nome di un
comando, il lavoro di prima smette di funzionare, e nessuno ti dice che è stato
l'aggiornamento: te ne accorgi settimane dopo, quando riapri quel progetto e
non parte più.

Il rimedio è un **ambiente virtuale**, cioè una cartella dentro il progetto con
un proprio interprete e le proprie librerie, separate da quelle di tutti gli
altri progetti. Si crea una volta, si attiva, e da lì `pip` installa dentro
l'ambiente:

```text
python3 -m venv .venv           # crea l'ambiente dentro la cartella del progetto
source .venv/bin/activate       # lo attiva (su Windows: .venv\Scripts\activate)
pip install numpy pandas matplotlib scikit-learn
```

Su Debian e su Ubuntu il primo dei tre comandi può fermarsi lamentando che
manca `ensurepip`. Quelle due distribuzioni, infatti, consegnano Python senza
il modulo che crea gli ambienti, e il modulo va chiesto a parte, una volta
sola, con `sudo apt install python3-venv`. La parola `sudo` davanti a un
comando significa «questo lo faccio da amministratore», e il terminale chiederà
la password: serve perché stai aggiungendo qualcosa al computer intero, e non
al singolo progetto.

Il `-m` vuol dire «esegui il modulo che si chiama così», dove un **modulo** è un
file di Python che si può tanto eseguire quanto importare. È il modo di lanciare
uno strumento incluso in Python invece che un file scritto da te, e `venv` è
quello strumento.

Il secondo comando, `source`, esegue le istruzioni contenute in un file senza
aprire una finestra nuova, e serve proprio perché l'attivazione deve valere per
il terminale che hai davanti. Si vede subito se ha funzionato: all'inizio della
riga del terminale compare `(.venv)`, e resta lì finché l'ambiente è attivo.

Finito di lavorare, dall'ambiente si esce con `deactivate`, e quel `(.venv)`
sparisce. Per ritrovare le stesse librerie su un'altra macchina,
`pip freeze > requirements.txt` ne scrive l'elenco con le versioni in un file,
e `pip install -r requirements.txt` le reinstalla.

Ci sono anche altri strumenti per gli ambienti, come `conda` o `uv`, che puoi
incontrare in una guida e che fanno lo stesso lavoro con altri comandi. Per gli
esempi, comunque, `venv` e `pip` bastano.

```{figure} ../figures/preparare-ambiente-python.svg
:name: fig-ambiente-python
:alt: "Anatomia di un ambiente di lavoro: in basso il sistema operativo, con l'installazione di Python valida per tutta la macchina; sopra, la cartella del progetto, che contiene i file di codice e un ambiente virtuale isolato con il proprio interprete e le proprie librerie; a lato l'editor, collegato da una freccia all'interprete dell'ambiente virtuale."
:width: 92%

L'ambiente virtuale è una scatola dentro il progetto. Le librerie che installa
non escono da lì, e due progetti sulla stessa macchina possono usare versioni
diverse della stessa libreria senza incontrarsi.
```

La separazione di {numref}`fig-ambiente-python` evita il progetto che non parte
più, poiché l'aggiornamento resta dentro l'ambiente del progetto che lo ha
chiesto. Il Python di sistema resta intoccato, e cestinare un progetto significa
cestinare anche il suo ambiente.

Infine, per usare i notebook sulla propria macchina, `pip install jupyterlab` e
poi `jupyter lab` li aprono nel browser, come su Colab.

## Dagli strumenti al linguaggio

A questo punto manca il linguaggio: la sintassi di base e i contenitori che
Python offre di suo (liste, tuple, dizionari, insiemi), poi le tre librerie con
cui si lavora sui numeri, sulle tabelle e sui grafici. Sono gli strumenti con
cui il {doc}`capitolo di matematica </Matematica/overview>` fa i suoi
calcoli e quello sul {doc}`machine learning </MachineLearning/overview>`
costruisce i primi modelli.

`````{tab} Elementare

```{admonition} Da ricordare
:class: important
- Python domina l'AI perché si legge, perché per quasi tutto c'è già una
  libreria pronta e perché lo usa moltissima gente. Veloce non lo è, ma
  affida i calcoli pesanti a librerie scritte in linguaggi più vicini alla
  macchina, e si limita a dare gli ordini.
- Le librerie sono una torre: NumPy alla base (i numeri), sopra Pandas (le
  tabelle), Matplotlib (i grafici) e scikit-learn. PyTorch, con cui si
  costruiscono le reti neurali, sta accanto alla torre più che sopra: fa i
  calcoli con un motore proprio, e con NumPy si limita a scambiarsi i dati.
- Il guadagno di scrivere il calcolo sul blocco intero invece che numero per
  numero sta nel pedaggio che Python paga a ogni giro di un ciclo, e che così
  si paga una volta sola.
- Il codice si prova in tre posti: l’interprete (scrivi una riga, risponde
  subito), i notebook (quaderni fatti di celle) e Colab, che dà
  notebook e schede grafiche gratis nel browser.
- Nei notebook le celle condividono la memoria, e c'è un tranello: eseguirle in
  disordine può far comparire un risultato che, rileggendo la pagina, non
  torna. Si rimedia rieseguendo tutto dall'alto in basso.
- Sul proprio computer le librerie di ogni progetto vanno in un ambiente
  virtuale (`python3 -m venv`), una cartella-scatola che le tiene separate da
  quelle di tutti gli altri progetti.
```

`````

`````{tab} Superiore

```{admonition} Da ricordare
:class: important
- Python domina l'AI per leggibilità, ecosistema e comunità più che
  per velocità bruta: fa da collante a librerie compilate in C/C++/CUDA.
- Lo stack scientifico è a strati: NumPy alla base, poi Pandas, Matplotlib
  e scikit-learn, che poggiano su di lui. PyTorch e TensorFlow stanno
  invece a fianco: hanno un motore di calcolo proprio, e con NumPy si limitano
  a scambiare i dati.
- Il guadagno della vettorizzazione viene dall’*overhead* dell'interprete
  che sparisce a ogni iterazione: due o tre ordini di grandezza sul ciclo
  Python equivalente. I prodotti fra matrici passano per BLAS.
- Si lavora nell’interprete, nei notebook Jupyter (un *kernel* che
  mantiene lo stato, da cui i tranelli dell'esecuzione fuori ordine) e su
  Colab, che offre GPU gratuite nel browser; in locale, un ambiente
  virtuale per progetto (`python3 -m venv`) tiene separate le dipendenze.
```

`````

Con l'ambiente pronto si comincia dal linguaggio: poche cose ripetute molte
volte, cioè dare un nome a un valore, tenerne insieme tanti, decidere,
ripetere, racchiudere un pezzo di lavoro in una funzione che risponde a un
nome. Sono le basi, e bastano già a scrivere un primo programma
utile.
