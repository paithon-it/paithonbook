# Python: il linguaggio dell'intelligenza artificiale

Nel Natale del 1989 un programmatore olandese, Guido van Rossum, si annoia.
Gli uffici del centro di ricerca dove lavora ad Amsterdam sono chiusi per le
feste, e lui riempie il tempo scrivendo, per hobby, un nuovo linguaggio di
programmazione. Lo chiama **Python**: non per il serpente, ma per i Monty
Python, il gruppo comico inglese di cui è fan. La prima versione pubblica esce
nel 1991. Nessuno, allora, poteva immaginare che trent'anni dopo quel
passatempo sarebbe diventato la lingua franca dell'intelligenza artificiale:
dalla rivoluzione del deep learning del 2012 fino ai grandi modelli
linguistici di oggi, quasi ogni svolta recente dell'AI è stata prototipata,
addestrata o rilasciata in Python.

Com'è successo? Python non è il linguaggio più veloce, né il più elegante in
senso accademico. Ha vinto per altre ragioni.

## Perché proprio Python

`````{tab} Elementare

Python si legge quasi come inglese scarno. Dove altri linguaggi ti chiedono
parentesi graffe, punti e virgola e altre formalità, Python ti lascia scrivere
l'idea e basta, come quegli appunti in cui descrivi i passaggi di una ricetta
a parole tue, solo che questi appunti il computer li esegue davvero. Per chi
arriva dalla matematica, dalla fisica o dalla biologia e vuole *esprimere un
ragionamento*, non litigare con la sintassi, è una liberazione.

A questo si aggiungono due cose decisive: è **gratuito** e ha una **comunità**
enorme. Per quasi ogni problema (leggere un file, disegnare un grafico,
addestrare una rete neurale), qualcuno ha già scritto una **libreria** pronta:
un pacchetto di istruzioni già confezionate da altri, che ti risparmia il
lavoro. Non devi reinventare la ruota: la importi e la usi.

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

e compaiono i diciannove aforismi che Tim Peters codificò nel 1999. Tra questi:
*"Explicit is better than implicit"*, *"Simple is better than complex"*,
*"Readability counts"*. Non è poesia gratuita: il codice si legge molte più
volte di quante lo si scriva, e in un progetto di ricerca condiviso la
leggibilità vale quanto la correttezza. È questa attenzione alla chiarezza,
prima ancora delle librerie, a rendere Python la scelta naturale per insegnare
e comunicare l'AI.

## L'ecosistema scientifico

La forza di Python nell'AI non è il linguaggio da solo, ma la torre di
librerie costruite l'una sull'altra ({numref}`fig-stack-python`). Ognuna fa
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
  clustering, metriche, tutto dietro un'API uniforme `fit` / `predict`.
- **PyTorch** (Meta, 2016), deep learning: reti neurali, differenziazione
  automatica, addestramento su GPU. È il framework di questo libro; il suo
  concorrente storico è **TensorFlow** (Google, 2015).

```{figure} ../figures/stack-scientifico-python.svg
:name: fig-stack-python
:alt: "Diagramma a strati: alla base Python, sopra NumPy, poi Pandas Matplotlib e scikit-learn, in cima PyTorch e TensorFlow."
:width: 85%

Lo stack scientifico di Python: ogni strato poggia su quello sotto. NumPy è
la base numerica su cui sono costruite le librerie di analisi e di deep
learning.
```

`````{tab} Elementare

Cosa vuol dire che una libreria è "costruita su NumPy"? Immagina di dover
raddoppiare un milione di numeri. In Python puro scriveresti un ciclo che li
scorre uno per uno. Con NumPy scrivi `2 * x` e l'operazione avviene sull'intero
blocco in un colpo solo: è più corto da scrivere e molto più veloce da
eseguire.

`````

`````{tab} Superiore

La differenza si chiama **vettorizzazione**. Un `ndarray` di NumPy è un blocco
di memoria contigua e tipizzata: le operazioni elemento-per-elemento sono
delegate a cicli in C ottimizzati (spesso con librerie BLAS e istruzioni
SIMD), evitando l'*overhead* dell'interprete Python su ogni iterazione. Il
risultato tipico è un codice più conciso e uno o due ordini di grandezza più
veloce del ciclo Python equivalente: la ragione per cui l'intero ecosistema
adotta l'array come struttura dati comune.

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

## Come è organizzato il capitolo

Nelle sezioni che seguono passiamo dalla teoria alla tastiera. Prima
prepariamo l'ambiente e prendiamo confidenza con la sintassi di base
(variabili, tipi, controllo di flusso, funzioni) e con le strutture dati
native (liste, dizionari) che useremo ovunque. Poi affrontiamo i tre pilastri
del calcolo scientifico: **NumPy** per gli array e l'algebra lineare,
**Pandas** per manipolare i dati reali, **Matplotlib** per visualizzarli. Alla
fine avrai gli strumenti per prendere un problema, tradurlo in codice e
arrivare a un primo modello: il ponte tra la matematica dei capitoli vicini e
il machine learning dei capitoli successivi.

```{admonition} Da ricordare
:class: important
- Python domina l'AI per **leggibilità**, **ecosistema** e **comunità**, non
  per velocità bruta: fa da collante a librerie compilate in C/C++/CUDA.
- Lo **stack scientifico** è a strati: NumPy alla base, poi Pandas, Matplotlib
  e scikit-learn, in cima PyTorch (il framework di questo libro) e TensorFlow.
- Si lavora nell'**interprete**, nei **notebook Jupyter** e su **Colab**, che
  offre GPU gratuite nel browser.
```
