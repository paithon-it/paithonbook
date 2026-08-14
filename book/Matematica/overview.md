# Richiami di Algebra, Statistica e Analisi Numerica

```{epigraph}
La matematica è il linguaggio con cui Dio ha scritto l'universo.

<p class="attribution">attribuito a Galileo Galilei</p>
```

C'è un equivoco tenace da sciogliere prima di cominciare: per usare il machine
learning non serve essere matematici, ma per *capirlo* serve riconoscere le
poche idee matematiche che vi ritornano di continuo. Sono sorprendentemente
poche.

Prima però conviene fissare tre parole che in questo capitolo torneranno a ogni
pagina. Un **modello** è un programma che riceve dei numeri e ne restituisce
altri: una foto in ingresso, la risposta «gatto» in uscita. Non è scritto a
mano come un programma normale, perché dentro ha migliaia (o miliardi) di
**parametri**, che sono manopole regolabili: cambiando le manopole cambia la
risposta. **Addestrarlo** è girare quelle manopole finché le risposte non sono
quelle giuste, e a girarle non è una persona ma una procedura automatica, che
guarda gli esempi e corregge. È questo, tutto insieme, il **machine learning**:
programmi che si regolano da soli sugli esempi invece di essere istruiti riga
per riga. Il **deep learning** è la sua versione con i modelli più grandi, fatti
di molti passaggi in fila.

Un modello del genere, quando lo si smonta, parla tre lingue soltanto:
**algebra lineare** (per rappresentare i dati e i **pesi**, cioè i numeri per
cui il modello moltiplica ciò che riceve), **analisi** (il ramo della
matematica che studia come cambia una quantità quando se ne muove un'altra:
serve a capire in che direzione migliorare) e **probabilità** (per convivere
con l'incertezza). A quelle tre se ne aggiungono due, che entrano in gioco
quando i conti passano dalla carta al calcolatore: la **teoria
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
Poi quei numeri vengono trasformati più volte di seguito, e ogni passaggio è
uno **strato**: prende la lista di numeri che gli arriva e ne produce un'altra,
fino a che dall'ultimo esce la risposta "gatto / non gatto". Anche questo è
algebra lineare, con in mezzo qualche piccola funzione che *piega* i numeri
(perché senza una piega, come vedremo, cento passaggi non fanno più di uno
solo). Misurare *quanto* il modello sbaglia e decidere come correggere ogni
singolo parametro per sbagliare un po' meno la prossima volta è **analisi**
(nello specifico: derivate e gradienti). E poiché nessun modello è mai certo (è
"abbastanza sicuro" che sia un gatto), il modo naturale di esprimere quella
sicurezza è la **probabilità**.

Le tre lingue, insieme, coprono l'intera vita di un modello: *rappresentare* i
dati, *regolare* i parametri, *quantificare* la fiducia nel risultato. Le altre
due voci servono nel momento in cui tutto questo smette di stare sulla carta e
lo si fa girare su una macchina vera: la teoria dell'informazione per dire
*quanto* il modello sbaglia con un numero solo, l'analisi numerica perché quel
numero lo calcola un computer, che di cifre ne ha un numero finito.

## Come è organizzato il capitolo

Sei sezioni, e ciascuna risponde a una domanda che si può fare a voce.

- **Algebra lineare**: come si mettono i numeri in fila, e come si trasformano
  tutti insieme (vettori, matrici, prodotti e norme).
- **Analisi e ottimizzazione**: come si capisce da che parte migliorare, e come
  ci si arriva un passo alla volta (derivate, gradiente, discesa del gradiente).
- **Probabilità e statistica**: come si convive con l'incertezza, e come si
  aggiorna un'opinione quando arrivano dati nuovi (fino al teorema di Bayes).
- **Teoria dell'informazione**: come si misura la sorpresa con un numero solo.
  È da lì che viene il punteggio d'errore con cui si addestra quasi ogni
  modello che deve scegliere fra alternative (entropia e cross-entropia).
- **Analisi numerica**: che cosa cambia quando i conti li fa una macchina che
  scrive solo poche cifre per numero, e come si evita che il conto vada fuori
  strada.
- **La matematica di un modello linguistico**: le cinque voci rimesse insieme.
  Un modello linguistico (in inglese *large language model*, da cui la sigla
  **LLM** che si incontra ovunque) smontato con i soli attrezzi di questo
  capitolo, che sono poi gli unici che servono.

I nomi fra parentesi sono le etichette tecniche, e nessuna di esse è data per
saputa: ognuna nasce dentro la sua sezione, con un'immagine prima e la formula
dopo.

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
