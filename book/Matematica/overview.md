# Richiami di Algebra, Statistica e Analisi Numerica

```{epigraph}
La matematica è il linguaggio con cui Dio ha scritto l'universo.

<p class="attribution">attribuito a Galileo Galilei</p>
```

C'è un equivoco tenace da sciogliere prima di cominciare: per usare il machine
learning non serve essere matematici, ma per *capirlo* serve riconoscere le
poche idee matematiche che vi ritornano di continuo. Sono sorprendentemente
poche. Un modello di deep learning con miliardi di parametri (le "manopole"
interne che l'addestramento regola da solo), quando lo si smonta, parla tre
lingue soltanto: **algebra lineare** (per rappresentare i dati e i **pesi**,
cioè i numeri per cui il modello moltiplica ciò che riceve), **analisi** (il
ramo della matematica che studia come cambia una quantità quando se ne muove
un'altra: serve a capire in che direzione migliorare) e **probabilità** (per
convivere con l'incertezza). A quelle tre se ne aggiungono due, che entrano in
gioco quando i conti passano dalla carta al calcolatore: la **teoria
dell'informazione**, che dà il modo di misurare quanto un modello sbaglia, e
l'**analisi numerica**, che si occupa di ciò che succede ai numeri dentro una
macchina che ne può scrivere solo un numero finito di cifre.

Il titolo dice «richiami» per tradizione editoriale, non perché tu debba già
saperle: ogni idea qui dentro è ripresa da capo, e chi non le ha mai viste è
esattamente il lettore per cui questo capitolo è scritto.

Questo capitolo non è un corso di matematica: è una cassetta degli attrezzi.
Prendiamo solo gli strumenti che useremo davvero nei capitoli successivi, e li
prendiamo due volte (una in modo intuitivo, una in modo formale), così che tu
possa fermarti al livello che ti serve.

## Perché proprio questi tre pilastri

Immagina il compito più classico dell'AI: riconoscere se una foto contiene un
gatto. Per un calcolatore quella foto non è un gatto, è una griglia di numeri
(l'intensità di ogni pixel). Metterli in fila ordinata è **algebra lineare**.
Trasformare quei numeri, strato dopo strato, finché non emerge la risposta
"gatto / non gatto" è ancora algebra lineare, alternata a semplici funzioni
non lineari. Misurare *quanto* il modello sbaglia e decidere come correggere
ogni singolo parametro per sbagliare un po' meno la prossima volta è
**analisi** (nello specifico: derivate e gradienti). E poiché nessun modello è
mai certo (è "abbastanza sicuro" che sia un gatto), il modo naturale di
esprimere quella sicurezza è la **probabilità**.

Le tre lingue, insieme, coprono l'intero ciclo di vita di un modello:
*rappresentare* i dati, *ottimizzare* i parametri, *quantificare* la fiducia
nel risultato. Le altre due voci servono al momento in cui quel ciclo si
mette in moto davvero: la teoria dell'informazione per dire *quanto* il
modello sbaglia con un numero solo, l'analisi numerica perché quel numero lo
calcola una macchina con le cifre contate.

## Come è organizzato il capitolo

- **Algebra lineare**: vettori, matrici, i prodotti che contano e le norme con
  cui misuriamo lunghezze ed errori.
- **Analisi e ottimizzazione**, derivate, gradiente, regola della catena: la
  bussola che indica dove migliorare, e la discesa del gradiente che la segue.
- **Probabilità e statistica**: variabili aleatorie, valore atteso e varianza,
  le distribuzioni ricorrenti, il teorema di Bayes.
- **Teoria dell'informazione**: entropia, cross-entropia e divergenza KL,
  cioè da dove viene la funzione di costo con cui si addestra quasi ogni
  classificatore.
- **Analisi numerica**, cosa cambia quando i numeri hanno una precisione
  finita: stabilità, overflow, il trucco del *log-sum-exp*.
- **La matematica di un modello linguistico**: le cinque voci rimesse
  insieme. Un modello linguistico (in inglese *large language model*, da cui
  la sigla **LLM** che si incontra ovunque) smontato con i soli attrezzi di
  questo capitolo, che sono poi gli unici che servono.

Ogni sezione segue la regola del libro: una spiegazione **Elementare** con
un'analogia concreta, e una **Superiore** con la notazione e le formule per
chi vuole i dettagli. Puoi leggere tutto a un livello solo, oppure alternarli:
spesso è proprio il passaggio dall'intuizione alla formula (e ritorno) a far
scattare la comprensione.

```{tip}
Se un simbolo ti blocca, non saltarlo: cercalo nella tab *Elementare* della
sezione corrispondente. Quasi sempre dietro una formula intimidatoria si
nasconde un'idea che sapresti spiegare a voce.
```
