# Python: il linguaggio dell'intelligenza artificiale

Nel Natale del 1989 un programmatore olandese, Guido van Rossum, si annoia.
Gli uffici del centro di ricerca dove lavora ad Amsterdam sono chiusi per le
feste, e lui riempie il tempo scrivendo, per hobby, un nuovo linguaggio di
programmazione. Lo chiama **Python**: non per il serpente, ma per i Monty
Python, il gruppo comico inglese di cui è fan. La prima versione pubblica esce
nel 1991. Nessuno, allora, poteva immaginare che trent'anni dopo quel
passatempo sarebbe diventato la lingua franca dell'intelligenza artificiale:
dalla rivoluzione del deep learning del 2012 fino ai grandi modelli
linguistici di oggi, la quasi totalità della ricerca recente si scrive, si
mette a punto e si pubblica in Python. Non è sempre stato così: la rete che
nel 2012 aprì la stagione del deep learning era scritta in C++ e CUDA, e il
framework dei laboratori di punta, negli anni subito dopo, si programmava in
Lua. Python però aveva già Theano, nato in ambito accademico prima di quella
rivoluzione, e vince quando accanto a Theano arrivano Caffe e poi TensorFlow;
la partita si chiude nel 2016
con PyTorch.

Com'è successo? Python non è il linguaggio più veloce, né il più elegante in
senso accademico. Ha vinto per altre ragioni.

## Perché proprio Python

`````{tab} Elementare

Un **programma** è un elenco di istruzioni scritte in un file di testo, che il
computer esegue una dopo l'altra dall'alto in basso. Nient'altro: niente di
misterioso, un foglio di ordini in fila. E Python si legge quasi come inglese
scarno. Una riga come `if voto >= 18: print("promosso")` si indovina prima
ancora di aver scritto il primo programma: è come la diresti a voce, «se il
voto è almeno diciotto, scrivi *promosso*». Le regole di scrittura di
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
si fa lo vediamo fra poco.

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

Su quel «viene fuori» c'è una cosa da chiarire subito, perché altrimenti
confonde per tutto il capitolo. Nell'interprete, se scrivi una riga che *vale*
qualcosa (un conto, il nome di una variabile) e premi Invio, lui te ne mostra
il risultato di sua iniziativa, senza che tu glielo abbia chiesto: è fatto per
conversare. In un programma salvato in un file, invece, quella stessa riga
calcola e non dice niente, e per vedere il risultato bisogna chiederlo con
`print`. Gli esempi che seguono sono scritti come li si digiterebbe nell'interprete, ed
è per questo che spesso una riga mostra un valore senza `print` accanto.

Eseguendo `import this` compaiono i diciannove aforismi che Tim Peters
codificò nel 1999.
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
una cosa e la fa bene. 

- **NumPy**: il fondamento. Introduce l’*array* N-dimensionale e rende
  l'algebra lineare veloce; quasi tutto il resto poggia su di lui.
- **Pandas**: dati tabellari. Il `DataFrame` è un foglio di calcolo
  programmabile: caricare, pulire e trasformare i dati prima di darli a un
  **modello**, che è il nome che in questo campo si dà a un programma che
  invece di seguire regole scritte da noi le ricava dagli esempi che gli
  diamo.
- **Matplotlib**: visualizzazione. I grafici con cui esplori i dati e racconti
  i risultati.
- **scikit-learn**: la cassetta degli attrezzi del machine learning *classico*,
  quello che viene prima delle reti neurali. Dentro ci sono decine di modelli
  diversi, e li si comanda tutti con le stesse due parole: `fit` (impara da
  questi dati) e `predict` (adesso prevedi). Imparato a usarne uno, li sai
  usare tutti, ed è quel che si intende con **API uniforme** (l'API di una
  libreria è l'insieme dei comandi con cui le si parla).
- **PyTorch** (Facebook AI Research, 2016; oggi Meta), deep learning:
  costruisce reti neurali e le addestra, calcolando da sé le correzioni da
  fare ai numeri interni della rete ogni volta che sbaglia; lavora sulla
  **GPU** (la scheda grafica, che sa fare moltissimi conti tutti insieme). Qui la torre cambia
  natura: PyTorch non è costruito su NumPy, ha un proprio motore di calcolo in
  C++, e con NumPy si scambia i dati senza copiarli. È la libreria attorno a
  cui è costruito il codice di questo libro; il suo concorrente storico è
  **TensorFlow** (Google, 2015).

```{figure} ../figures/stack-scientifico-python.svg
:name: fig-stack-python
:alt: "Diagramma a strati: alla base Python, sopra NumPy, poi Pandas Matplotlib e scikit-learn, in cima PyTorch e TensorFlow."
:width: 85%

La torre delle librerie scientifiche di Python. Pandas, Matplotlib e
scikit-learn poggiano davvero su NumPy: per funzionare hanno bisogno che sia
installato, e lo dichiarano. PyTorch e TensorFlow stanno in cima in un altro
senso: i conti se li fanno per conto proprio, con un motore tutto loro, e con
NumPy si limitano a scambiarsi i dati.
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
elemento-per-elemento sono delegate a cicli in C ottimizzati, e spesso a
**istruzioni SIMD** (*single instruction, multiple data*: una sola istruzione
del processore che opera su più numeri insieme, tanti quanti ne entrano nei
suoi registri vettoriali), evitando
l’*overhead* dell'interprete Python su ogni iterazione; i prodotti tra
matrici passano invece per librerie BLAS dedicate. Il risultato tipico è un
codice più conciso e due o tre ordini di grandezza più veloce del ciclo
Python equivalente: la ragione per cui l'intero ecosistema adotta l'array
come struttura dati comune.

`````

## L'interprete, i notebook e il cloud

Python si può usare in tre modi, di crescente comodità per chi sperimenta.

`````{tab} Elementare

Il modo più diretto è l’**interprete interattivo**: apri una finestra, scrivi
un'istruzione, premi Invio e vedi subito il risultato, come una calcolatrice
con cui puoi conversare. Un passo avanti è il **notebook Jupyter**: un
quaderno digitale fatto di celle, dove testo, codice, grafici e formule
convivono nella stessa pagina. Scrivi una cella, la esegui, guardi il grafico
che appare lì sotto. È il modo in cui gran parte del machine learning viene
davvero insegnato e praticato.

E se non vuoi installare nulla? **Google Colab** ti dà un notebook nel
browser, gratuito, e spesso con una scheda grafica in prestito: abbastanza per
addestrare un modello senza
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
e Linux, *Prompt dei comandi* (o *PowerShell*) su Windows. Su Linux Python c'è già. Su macOS no: `/usr/bin/python3` è un segnaposto che
al primo uso propone di installare gli strumenti da sviluppatore di Xcode, e
conviene accettare, oppure scaricare Python da `python.org` come su Windows.
Su Windows si scarica da `python.org`, ricordando di spuntare
«Add Python to PATH» durante l'installazione: è la casella che dice al
terminale dove Python è stato messo, e senza di essa il comando qui sotto
risponderà che non lo trova. Poi, quattro gesti:

```text
python3 --version        # c'è? risponde con il numero, per esempio "Python 3.12.3"
python3                  # apre l'interprete: compaiono tre maggiori, >>>
>>> import this          # si scrive un'istruzione, si preme Invio
>>> exit()               # si esce
```

Quei tre maggiori si chiamano **prompt**: sono l'invito a scrivere, il modo in
cui l'interprete dice «tocca a te». Finché li vedi, sei dentro l'interprete e
non nel terminale.

Le istruzioni scritte all'interprete si perdono quando lo si chiude. Per tenere
un programma lo si scrive in un file di testo con estensione `.py` (per esempio
`primo.py`, con dentro `print("ciao")`) e lo si esegue con `python3 primo.py`.
Il file lo si può scrivere con qualunque editor di testo, ma conviene
usarne uno che conosca Python e segnali gli errori mentre scrivi: i due più
diffusi sono **Visual Studio Code** e **PyCharm**, gratuiti entrambi. Se un
programma non finisce più (capita: basta un ciclo scritto male) si ferma
premendo `Ctrl+C`.

**Le librerie** non arrivano con Python: si installano una volta con **`pip`**,
il programma che va a prenderle in rete e le mette al posto giusto (dentro
l'ambiente virtuale di cui fra un attimo, `pip` c'è sempre). E conviene
installarle dentro
un **ambiente virtuale**, cioè una cartella-scatola che tiene le librerie di
*questo* progetto separate da quelle di tutti gli altri:

```text
python3 -m venv .venv           # crea la scatola dentro la cartella del progetto
source .venv/bin/activate       # la apre (su Windows: .venv\Scripts\activate)
pip install numpy pandas matplotlib scikit-learn
```

Se il primo comando si ferma lamentando che manca `ensurepip`, sei quasi
certamente su Debian o Ubuntu, dove il Python di sistema viaggia senza il
pezzo che crea le scatole: si rimedia una volta per tutte con
`sudo apt install python3-venv`. Quando hai finito di lavorare nella scatola
si esce con `deactivate`. E se in un tutorial incontri altri gestori di
ambienti (conda, uv), nessuna sorpresa: fanno lo stesso mestiere, e qui si usa
la coppia `venv` più `pip` perché basta per tutto il percorso.

Due parole sui comandi. Il `-m` vuol dire «esegui il modulo che si chiama
così», ed è il modo di lanciare uno strumento che viaggia dentro Python invece
che un file scritto da te; `venv` è quello strumento. `source` esegue le
istruzioni contenute in un file senza aprire una finestra nuova, e serve
proprio perché l'apertura della scatola deve valere per il terminale che hai
davanti. Che abbia funzionato lo vedi subito: all'inizio della riga del
terminale compare `(.venv)`, e resta lì finché la scatola è aperta.

```{figure} ../figures/preparare-ambiente-python.svg
:name: fig-ambiente-python
:alt: "Anatomia di un ambiente di lavoro: in basso il sistema operativo, con l'installazione di Python valida per tutta la macchina; sopra, la cartella del progetto, che contiene i file di codice e un ambiente virtuale isolato con il proprio interprete e le proprie librerie; a lato l'editor, collegato da una freccia all'interprete dell'ambiente virtuale."
:width: 92%

L'ambiente virtuale è una scatola dentro il progetto. Le librerie che installa
non escono da lì, e due progetti sulla stessa macchina possono usare versioni
diverse della stessa libreria senza incontrarsi.
```

La separazione di {numref}`fig-ambiente-python` risparmia la classe di
problemi più frustrante per chi comincia. Un esempio di quelli che capitano
davvero: hai un lavoro che gira, non lo tocchi da mesi, e nel frattempo per un
esercizio nuovo aggiorni una libreria. La versione nuova ha cambiato il nome di
un comando, il lavoro di prima smette di funzionare, e nessuno ti dice che è
stato l'aggiornamento: te ne accorgi settimane dopo, quando riapri quel
progetto e non parte più. Con una scatola per progetto non succede, perché
l'aggiornamento resta dentro la sua. Il Python di sistema resta
intoccato, e cestinare un progetto significa cestinare anche il suo ambiente.
Per lavorare come si lavora davvero, `pip install jupyterlab` e poi
`jupyter lab` aprono nel browser i notebook di cui sopra, questa volta sulla
propria macchina.

## Che cosa si impara qui

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
- Il codice si prova in tre posti: l’**interprete** (scrivi una riga, risponde
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
- Si lavora nell’**interprete**, nei **notebook Jupyter** e su **Colab**, che
  offre GPU gratuite nel browser; in locale, un **ambiente virtuale** per
  progetto (`python3 -m venv`) tiene separate le dipendenze.
```

`````
