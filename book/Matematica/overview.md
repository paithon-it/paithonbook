# Richiami di Algebra, Statistica e Analisi Numerica

```{epigraph}
La filosofia è scritta in questo grandissimo libro che continuamente ci sta aperto innanzi a gli occhi (io dico l'universo), ma non si può intendere se prima non s'impara a intender la lingua, e conoscer i caratteri, ne’ quali è scritto. Egli è scritto in lingua matematica, e i caratteri son triangoli, cerchi, ed altre figure geometriche, senza i quali mezi è impossibile a intenderne umanamente parola; senza questi è un aggirarsi vanamente per un oscuro laberinto.

<p class="attribution">Galileo Galilei, <i>Il Saggiatore</i>,&nbsp;1623</p>

:::{only} latex
*Galileo Galilei, «Il Saggiatore», 1623.*
:::
```

% L'attribuzione della citazione qui sopra e’ scritta in HTML perche’ al sito
% serve la classe CSS `attribution`, e in stampa quel pezzo sparisce: senza la
% ripetizione per il solo LaTeX la citazione resterebbe senza autore. Il blocco
% `only` sta DENTRO la recinzione dell'epigrafe di proposito: fuori, la riga
% dell'attribuzione diventerebbe il primo paragrafo della pagina, e
% `scripts/genera-radice.py` la userebbe come descrizione in `llms.txt`.

«Chiamala entropia: nessuno sa davvero che cosa sia, e in una discussione
partirai sempre in vantaggio.» Si racconta che il consiglio venisse da John von
Neumann, e che a riceverlo fosse Claude Shannon, il quale nel 1948 aveva appena
trovato il modo di misurare l'informazione e non sapeva come chiamare la
grandezza che gli era venuta fuori. Aveva pensato a «informazione», parola già
troppo usata, poi a «incertezza». Alla fine seguì il consiglio, e quel nome è
rimasto: lo incontreremo nella sezione sulla teoria dell'informazione.

L'aneddoto è tramandato, non documentato: lo raccontò Shannon stesso, a voce,
nel 1961, e a stampa arrivò dieci anni più tardi, riferito da chi glielo aveva
sentito dire. Va preso con la cautela che meritano le battute riportate. Ma
dice una cosa giusta, ed è la cosa da mettere in chiaro prima di cominciare:
**spaventa più il nome della cosa**. L'entropia, sotto quel nome
greco, è la sorpresa che ti aspetti in media da quello che sta per succedere.
Si spiega in una riga, e in una riga l'abbiamo appena spiegata. Con gradiente,
vettore e verosimiglianza funziona allo stesso modo.

C'è quindi un equivoco tenace da sciogliere. Galileo, qui sopra, dice che il
libro dell'universo è scritto in lingua matematica, e non aveva torto; il
vocabolario che serve a noi, però, è corto. Per usare il machine learning non
serve essere matematici, ma per *capirlo* serve riconoscere le poche idee
matematiche che vi ritornano di continuo. Sono sorprendentemente poche.

Il titolo dice «richiami» per tradizione editoriale, non perché tu debba già
saperle: ogni idea qui dentro è ripresa da capo, e chi non le ha mai viste è
esattamente il lettore per cui questo capitolo è scritto.

## Una foto, e tre lingue

Il compito più classico dell'AI è questo: guardare una foto e dire se contiene
un gatto. Per un calcolatore quella foto non è un gatto, è una griglia di
numeri (l'intensità di ogni pixel). Metterli in fila ordinata è **algebra
lineare**. Poi quei numeri vengono trasformati più volte di seguito, e ogni
passaggio è uno **strato**: prende la lista di numeri che gli arriva e ne
produce un'altra, fino a che dall'ultimo esce la risposta «gatto / non gatto».
Anche questo è algebra lineare, con in mezzo qualche piccola funzione che
*piega* i numeri (perché senza una piega, come vedremo, cento passaggi non
fanno più di uno solo). Misurare *quanto* il programma sbaglia, e capire come
ritoccare i numeri che ha dentro perché sbagli un po’ meno la volta dopo, è
**analisi**. L'analisi è il ramo della matematica che studia come cambia una
quantità quando se ne muove un'altra: qui, di quanto cambia l'errore se si
sposta un numero. E poiché una risposta del genere non è mai una certezza (il programma
è «abbastanza sicuro» che sia un gatto), il modo naturale di esprimere quella
sicurezza è la **probabilità**.

Quel «programma» ha un nome preciso, e ce l'hanno anche i numeri che tiene
dentro. Un **modello** è un programma che riceve dei numeri e ne restituisce
altri: una foto in ingresso, la risposta «gatto» in uscita. Non è scritto a
mano come un programma normale, perché dentro ha migliaia (o miliardi) di
**parametri**, che sono manopole regolabili: cambiando le manopole cambia la
risposta. Quelli per cui il modello moltiplica ciò che riceve si chiamano
**pesi**. **Addestrarlo** è girare quelle manopole finché le risposte non sono
quelle giuste, e a girarle non è una persona ma una procedura automatica, che
guarda gli esempi e corregge. È questo, tutto insieme, il **machine learning**:
programmi che si regolano da soli sugli esempi invece di essere istruiti riga
per riga. Il **deep learning** è la sua versione con i modelli più grandi,
fatti di molti passaggi in fila, cioè di molti strati.

Le tre lingue, insieme, coprono l'intera vita di un modello: *rappresentare* i
dati, *regolare* i parametri, *quantificare* la fiducia nel risultato. Se ne
aggiungono due nel momento in cui tutto questo smette di stare sulla carta e lo
si fa girare su una macchina vera: la **teoria dell'informazione**, che dà il
modo di dire *quanto* il modello sbaglia con un numero solo (ed è la casa
dell'entropia di poco fa), e l’**analisi numerica**, che si occupa di ciò che
succede ai numeri dentro una macchina capace di scrivere, per ogni numero, solo
poche cifre.

Questo capitolo non è un corso di matematica: è una cassetta degli attrezzi.
Prendiamo solo gli strumenti che useremo davvero nei capitoli successivi, e li
prendiamo due volte (una in modo intuitivo, una in modo formale), così che tu
possa fermarti al livello che ti serve.

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

Alla fine delle sei sezioni avrai in mano poche cose, e sono sempre quelle: una
lista di numeri, un modo per capire da che parte migliorare, un modo per dire
quanto sei sicuro. Servono subito, e non fra dieci capitoli: il capitolo sul
machine learning, che viene dopo questo, è già tutto un metterle al lavoro su
un problema vero.
