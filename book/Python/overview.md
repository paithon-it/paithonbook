# Python: il linguaggio dell'intelligenza artificiale

Nel Natale del 1989 un programmatore olandese, Guido van Rossum, si annoia.
Gli uffici del centro di ricerca dove lavora ad Amsterdam sono chiusi per le
feste, e lui riempie il tempo scrivendo, per hobby, un nuovo linguaggio di
programmazione. Lo chiama **Python**: non per il serpente, ma per i Monty
Python, il gruppo comico inglese di cui è fan. La prima versione pubblica esce
nel 1991. Nessuno, allora, poteva immaginare che trent'anni dopo quel
passatempo sarebbe diventato la lingua franca dell'intelligenza artificiale:
dalla rivoluzione del deep learning del 2012 fino ai grandi modelli
linguistici di oggi, quasi ogni svolta recente dell'AI è nata come esperimento
in Python, in Python è stata messa a punto e in Python è stata pubblicata.

Com'è successo? Python non è il linguaggio più veloce, né il più elegante in
senso accademico. Ha vinto per altre ragioni.

## Perché proprio Python

`````{tab} Elementare

Python si legge quasi come inglese scarno. Una riga come
`if voto >= 18: print("promosso")` si indovina prima ancora di sapere che cosa
sia un programma: è scritta come la diresti a voce. Le regole di scrittura di
un linguaggio (dove vanno i due punti, gli a capo, le parentesi) si chiamano la
sua **sintassi**, e quella di Python è breve e leggera: sta in un pomeriggio, e
poi il tempo lo passi a pensare a *cosa* dire, non a *come* scriverlo. Per chi
arriva dalla matematica, dalla fisica o dalla biologia e vuole esprimere un
ragionamento, è una liberazione.

A questo si aggiungono due cose decisive: è **gratuito** e ha una **comunità**
enorme. Per quasi ogni problema (leggere un file, disegnare un grafico,
addestrare una rete neurale), qualcuno ha già scritto una **libreria** pronta:
un pacchetto di istruzioni già confezionate da altri, che ti risparmia il
lavoro. Non devi reinventare la ruota: la **importi** e la usi. Importare vuol
dire scrivere una riga in cima al programma, `import numpy`, per dire a Python
«da qui in poi voglio usare anche questa»: da quel momento tutti gli strumenti
della libreria sono a disposizione. La libreria deve però essere già presente
sul computer, e installarla è un gesto a parte, che si fa una volta sola: come
si fa lo vediamo poco più avanti, in questa stessa pagina.

`````

`````{tab} Superiore

Python è un linguaggio **interpretato** e **dinamicamente tipizzato**: paghi
in velocità di esecuzione ciò che guadagni in velocità di sviluppo. La chiave
del suo successo scientifico è però che aggira il *problema dei due
linguaggi*. Le operazioni pesanti non girano affatto in Python puro: NumPy è
scritto in C, PyTorch e TensorFlow in C++ e CUDA. Python fa da **collante**
(*glue language*), orchestrando componenti compilate che sfruttano BLAS, SIMD
e GPU. Scrivi codice ad alto livello, leggibile; sotto, gira codice nativo
alla massima velocità.

Attorno al linguaggio è cresciuto un ecosistema di centinaia di migliaia di
pacchetti (distribuiti via PyPI e `pip`) e una prassi di ricerca in cui i
paper arrivano con il codice allegato: la riproducibilità diventa la norma, e
Python il denominatore comune.

`````

## Un linguaggio pensato per essere letto

La filosofia di Python è così esplicita da essere scritta dentro il linguaggio
stesso. Digita in un interprete (la finestra in cui scrivi un'istruzione e
Python risponde subito, la vedremo tra poco):

```python
import this   # stampa lo "Zen of Python"
```

Metà di quella riga non è un'istruzione: il **cancelletto `#`** apre un
*commento*, e tutto ciò che lo segue sulla stessa riga è scritto per chi legge,
non per il computer, che lo salta. In questo libro i commenti dicono che cosa
fa la riga accanto, e la freccia `->` dentro un commento significa «e viene
fuori questo».

Eseguendola compaiono i diciannove aforismi che Tim Peters codificò nel 1999.
Tra questi: *"Explicit is better than implicit"* («meglio esplicito che
implicito»), *"Simple is better than complex"* («meglio semplice che
complicato»), *"Readability counts"* («la leggibilità conta»). Non è poesia
gratuita: il codice si legge molte più
volte di quante lo si scriva, e in un progetto di ricerca condiviso la
leggibilità vale quanto la correttezza. È questa attenzione alla chiarezza,
prima ancora delle librerie, a rendere Python la scelta naturale per insegnare
e comunicare l'AI.

## L'ecosistema scientifico

La forza di Python nell'AI non è il linguaggio da solo, ma la torre di
librerie costruite l'una sull'altra ({numref}`fig-stack-python`; in inglese si
dice *stack*, e il nome torna spesso). Ognuna fa
una cosa e la fa bene. Non serve ancora capire ogni termine di questo elenco:
è la mappa del viaggio, e ogni territorio ha il suo capitolo più avanti.

- **NumPy**: il fondamento. Introduce l'*array* N-dimensionale e rende
  l'algebra lineare veloce; quasi tutto il resto poggia su di lui.
- **Pandas**: dati tabellari. Il `DataFrame` è un foglio di calcolo
  programmabile: caricare, pulire e trasformare i dati prima di darli a un
  modello.
- **Matplotlib**: visualizzazione. I grafici con cui esplori i dati e racconti
  i risultati.
- **scikit-learn**, machine learning *classico*: regressione, alberi, SVM,
  clustering, metriche, tutto dietro le stesse due parole d'ordine, `fit`
  (impara dai dati) e `predict` (prevedi): un modo di chiedere le cose che non
  cambia da un modello all'altro, ed è quel che si intende con **API uniforme**
  (l'API di una libreria è l'insieme dei comandi con cui le si parla).
- **PyTorch** (Facebook AI Research, 2016; oggi Meta), deep learning: reti
  neurali, differenziazione automatica, addestramento su **GPU** (la scheda
  grafica, che sa fare moltissimi conti tutti insieme). È la libreria attorno
  a cui è costruito il codice di questo libro; il suo concorrente storico è
  **TensorFlow** (Google, 2015).

```{figure} ../figures/stack-scientifico-python.svg
:name: fig-stack-python
:alt: "Diagramma a strati: alla base Python, sopra NumPy, poi Pandas Matplotlib e scikit-learn, in cima PyTorch e TensorFlow."
:width: 85%

La torre delle librerie scientifiche di Python: ogni strato poggia su quello
sotto. NumPy è la base numerica su cui sono costruite le librerie di analisi e
di deep learning.
```

`````{tab} Elementare

Cosa vuol dire che una libreria è "costruita su NumPy"? Che non rifà da capo il
lavoro di tenere insieme tanti numeri e di farci i conti sopra: quello lo
chiede a NumPy, e si concentra sul proprio mestiere. Pandas sa che cos'è una
colonna di una tabella e come si raggruppano le righe, ma quando c'è da sommare
un milione di valori passa la palla a chi lo fa meglio.

Il guadagno è anche di velocità. Con il solo Python, raddoppiare un milione di
numeri vuol dire percorrerli uno per uno in un **ciclo** (un'istruzione che si
ripete tante volte, la vediamo nella prossima sezione). Con NumPy si scrive
`2 * x`, dove `x` è il blocco intero dei numeri, e il raddoppio avviene su
tutti in un colpo solo: più corto da scrivere, e molto più veloce da eseguire.

`````

`````{tab} Superiore

La differenza si chiama **vettorizzazione**. Un `ndarray` di NumPy è una vista
tipizzata su un blocco di memoria, contiguo nel caso più comune: le operazioni
elemento-per-elemento sono delegate a cicli in C ottimizzati (spesso con
istruzioni SIMD), evitando
l'*overhead* dell'interprete Python su ogni iterazione; i prodotti tra
matrici passano invece per librerie BLAS dedicate. Il risultato tipico è un
codice più conciso e due o tre ordini di grandezza più veloce del ciclo
Python equivalente: la ragione per cui l'intero ecosistema adotta l'array
come struttura dati comune.

`````

## L'interprete, i notebook e il cloud

Python si può usare in tre modi, di crescente comodità per chi sperimenta.

`````{tab} Elementare

Il modo più diretto è l'**interprete interattivo**: apri una finestra, scrivi
un'istruzione, premi Invio e vedi subito il risultato, come una calcolatrice
con cui puoi conversare. Un passo avanti è il **notebook Jupyter**: un
quaderno digitale fatto di celle, dove testo, codice, grafici e formule
convivono nella stessa pagina. Scrivi una cella, la esegui, guardi il grafico
che appare lì sotto. È il modo in cui gran parte del machine learning viene
davvero insegnato e praticato, ed è il formato di questo libro.

E se non vuoi installare nulla? **Google Colab** ti dà un notebook nel
browser, gratuito, con GPU incluse: perfetto per addestrare un modello senza
possedere un computer potente.

`````

`````{tab} Superiore

L'interprete interattivo è un **REPL** (*read–eval–print loop*); la versione
arricchita `IPython` ne è il motore. Un notebook Jupyter è un documento
`.ipynb` (JSON di celle) eseguito da un *kernel* che mantiene lo stato tra una
cella e l'altra: da qui la potenza (e i tranelli) dell'esecuzione fuori
ordine. **Google Colab** è un ambiente Jupyter gestito, con acceleratori
hardware (GPU/TPU) accessibili gratuitamente entro certi limiti, ideale per
prototipazione e didattica riproducibile.

`````

## Preparare l'ambiente: la prima riga eseguita davvero

Tutto il codice di questo libro si può leggere, ma è fatto per essere provato.
Ecco come, in concreto.

**La via senza installare niente.** In alto in ogni pagina che contiene del
codice c'è il pulsante **«Esegui il codice»**: apre su Google Colab un notebook
con tutte le celle del capitolo, in ordine, già pronte. Si preme il triangolino
accanto a una cella e la si esegue; le librerie sono già installate. È il modo
più rapido per provare gli esempi mentre si legge, e non richiede altro che un
browser e un account Google.

**Sul proprio computer.** Serve un **terminale**, cioè la finestra in cui si
scrivono comandi al computer invece di cliccare: si chiama *Terminale* su macOS
e Linux, *Prompt dei comandi* (o *PowerShell*) su Windows. Su Linux e macOS
Python c'è già; su Windows si scarica da `python.org`, ricordando di spuntare
«Add Python to PATH» durante l'installazione. Poi, quattro gesti:

```text
python3 --version        # c'è? (su Windows il comando è "python")
python3                  # apre l'interprete: compare il prompt >>>
>>> import this          # si scrive un'istruzione, si preme Invio
>>> exit()               # si esce
```

Le istruzioni scritte all'interprete si perdono quando lo si chiude. Per tenere
un programma lo si scrive in un file di testo con estensione `.py` (per esempio
`primo.py`, con dentro `print("ciao")`) e lo si esegue con `python3 primo.py`.
Se un programma non finisce più (capita: basta un ciclo scritto male) si ferma
premendo `Ctrl+C`.

**Le librerie** non arrivano con Python: si installano una volta, con `pip`, e
conviene farlo dentro un **ambiente virtuale**, cioè una cartella-scatola che
tiene le librerie di *questo* progetto separate da quelle di tutti gli altri:

```text
python3 -m venv .venv           # crea la scatola dentro la cartella del progetto
source .venv/bin/activate       # la apre (su Windows: .venv\Scripts\activate)
pip install numpy pandas matplotlib scikit-learn
```

```{figure} ../figures/preparare-ambiente-python.svg
:name: fig-ambiente-python
:alt: "Anatomia di un ambiente di lavoro: il sistema operativo con la sua installazione di Python; dentro, la cartella del progetto, che contiene un ambiente virtuale isolato con le proprie librerie e la propria versione degli strumenti, separato dal Python di sistema."
:width: 92%

L'ambiente virtuale è una scatola dentro il progetto. Le librerie che installa
non escono da lì, e due progetti sulla stessa macchina possono usare versioni
diverse della stessa libreria senza incontrarsi.
```

La separazione di {numref}`fig-ambiente-python` risparmia la classe di
problemi più frustrante per chi comincia: un aggiornamento fatto per un
progetto che rompe silenziosamente un altro. Il Python di sistema resta
intoccato, e cestinare un progetto significa cestinare anche il suo ambiente.
Per lavorare come si lavora davvero, `pip install jupyterlab` e poi
`jupyter lab` aprono nel browser i notebook di cui sopra, questa volta sulla
propria macchina.

## Come è organizzato il capitolo

Nelle sezioni che seguono passiamo dalla teoria alla tastiera. L'ambiente è
pronto: prendiamo confidenza con la sintassi di base
(variabili, tipi, controllo di flusso, funzioni) e con le strutture dati
native (liste, dizionari) che useremo ovunque. Poi affrontiamo i tre pilastri
del calcolo scientifico: **NumPy** per gli array e l'algebra lineare,
**Pandas** per manipolare i dati reali, **Matplotlib** per visualizzarli. Alla
fine avrai gli strumenti per prendere un problema, tradurlo in codice e
arrivare a un primo modello: il ponte tra la matematica dei capitoli vicini e
il machine learning dei capitoli successivi.

`````{tab} Elementare

```{admonition} Da ricordare
:class: important
- Python domina l'AI perché **si legge**, perché ha **una libreria già pronta
  per quasi tutto** e perché ha **tanta gente** che lo usa. Non perché sia
  veloce: i conti pesanti li fa fare a librerie scritte in linguaggi più
  vicini alla macchina, e si limita a dare gli ordini.
- Le librerie sono **una torre**: NumPy alla base (i numeri), sopra Pandas
  (le tabelle), Matplotlib (i grafici) e scikit-learn, in cima PyTorch, quella
  con cui si costruiscono le reti neurali ed è usata in questo libro.
- Il codice si prova in tre posti: l'**interprete** (scrivi una riga, risponde
  subito), i **notebook** (quaderni fatti di celle) e **Colab**, che dà
  notebook e schede grafiche gratis nel browser. Il pulsante «Esegui il
  codice», in cima a ogni pagina che contiene codice, porta lì.
```

`````

`````{tab} Superiore

```{admonition} Da ricordare
:class: important
- Python domina l'AI per **leggibilità**, **ecosistema** e **comunità**, non
  per velocità bruta: fa da collante a librerie compilate in C/C++/CUDA.
- Lo **stack scientifico** è a strati: NumPy alla base, poi Pandas, Matplotlib
  e scikit-learn, in cima PyTorch (il framework di questo libro) e TensorFlow.
- Si lavora nell'**interprete**, nei **notebook Jupyter** e su **Colab**, che
  offre GPU gratuite nel browser; in locale, un **ambiente virtuale** per
  progetto (`python3 -m venv`) tiene separate le dipendenze.
```

`````
