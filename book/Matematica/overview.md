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

«Chiamala entropia, per due ragioni. La prima è che la tua funzione
d'incertezza in meccanica statistica si chiama già così. La seconda, più
importante, è che nessuno sa davvero che cosa sia l'entropia, quindi in una
discussione
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

C'è quindi un equivoco tenace da sciogliere. Nell'epigrafe Galileo dice che il
libro dell'universo è scritto in lingua matematica, e non aveva torto; il
vocabolario che serve a noi, però, è corto. Per usare il machine learning non
serve essere matematici, ma per *capirlo* serve riconoscere le poche idee
matematiche che vi ritornano di continuo. Sono sorprendentemente poche.

Nessuna di queste idee è data per saputa: ognuna è ripresa da capo, e chi non
le ha mai viste è esattamente il lettore per cui questo capitolo è scritto.

## Una foto, e tre lingue

Il compito più classico dell'AI è questo: guardare una foto e dire se contiene
un gatto. Per un calcolatore quella foto è una griglia di numeri (l'intensità
di ogni pixel), non un gatto. Metterli in fila ordinata è **algebra lineare**.
Poi quei numeri vengono trasformati più volte di seguito, e ogni passaggio è
uno **strato**: prende la lista di numeri che gli arriva e ne produce un'altra,
fino a che dall'ultimo esce la risposta «gatto / non gatto». Anche questo è
algebra lineare, con in mezzo qualche piccola funzione che *piega* i numeri:
sommare e moltiplicare, ripetuto cento volte, resta un sommare e moltiplicare
con altri numeri, e la piega serve a rompere questa regola. Misurare *quanto*
il programma sbaglia, e capire come ritoccare i numeri che ha dentro perché
sbagli un po’ meno la volta dopo, è **analisi**. L'analisi è il ramo della
matematica che studia come cambia una quantità quando se ne muove un'altra:
qui, di quanto cambia l'errore se si sposta un numero. E poiché una risposta
del genere non è mai una certezza (il programma è «abbastanza sicuro» che sia
un gatto), il modo naturale di esprimere quella sicurezza è la **probabilità**.

Quel «programma» ha un nome preciso, e i nomi li ha già dati
l’{doc}`Introduzione </Introduzione/overview>`: un modello è un programma che
riceve numeri e ne restituisce altri, i suoi parametri sono le migliaia (o
miliardi) di manopole regolabili che decidono la risposta, e addestrarlo è
girare quelle manopole finché le risposte non sono quelle giuste. Serve qui
una sola aggiunta, perché è quella su cui l'algebra lineare lavora: i
parametri per cui il modello moltiplica ciò che riceve si chiamano **pesi**.

Le tre lingue, insieme, coprono l'intera vita di un modello: *rappresentare* i
dati, *regolare* i parametri, *quantificare* la fiducia nel risultato. Se ne
aggiungono due nel momento in cui tutto questo smette di stare sulla carta e lo
si fa girare su una macchina vera: la **teoria dell'informazione**, che dà il
modo di dire *quanto* il modello sbaglia con un numero solo (ed è la casa
dell'entropia di poco fa), e l’**analisi numerica**, che si occupa di ciò che
succede ai numeri dentro una macchina capace di scrivere, per ogni numero, solo
poche cifre.

Qui dentro non c'è tutta la matematica: ci sono gli attrezzi che i capitoli
successivi useranno davvero, e nient'altro.

## Cinque attrezzi, una domanda ciascuno

Cinque attrezzi, e ciascuno risponde a una domanda che si può fare a voce.
Alcuni occupano più di una sezione, perché la domanda si articola; una sezione
finale li rimette al lavoro tutti insieme su un oggetto solo.

- **Algebra lineare**: come si mettono i numeri in fila, e come si trasformano
  tutti insieme. Sono quattro sezioni: *vettori, matrici, prodotti e norme*
  per cominciare; i *sistemi lineari*, cioè che cosa succede quando i dati
  impongono dei vincoli e quando quei vincoli non bastano; *ortogonalità e
  proiezioni*, che rispondono quando una risposta esatta non esiste; e il
  *determinante*, che misura di quanto una trasformazione gonfia lo spazio.
- **Analisi e ottimizzazione**: come si capisce da che parte migliorare, e come
  ci si arriva un passo alla volta (derivate, gradiente, discesa del gradiente).
- **Probabilità e statistica**: come si convive con l'incertezza, e come si
  aggiorna un'opinione quando arrivano dati nuovi (fino al teorema di Bayes).
  Seguono due sezioni che ne tirano le conseguenze: *quanto può sbagliare una
  media*, che dice quante prove servono per fidarsi di un numero misurato, e le
  *catene di Markov*, dove la probabilità incontra l'algebra lineare e la
  domanda «dove finisce, andando avanti per sempre?» ha una risposta esatta.
- **Teoria dell'informazione**: come si misura la sorpresa con un numero solo.
  È da lì che viene il punteggio d'errore con cui si addestra quasi ogni
  modello che deve scegliere fra alternative (entropia e cross-entropia).
- **Analisi numerica**: che cosa cambia quando i conti li fa una macchina che
  scrive solo poche cifre per numero, e come si evita che il conto vada fuori
  strada.

L'ultima sezione non aggiunge un sesto attrezzo, rimette al lavoro i cinque
insieme su un oggetto solo: un modello linguistico (in inglese *large language
model*, da cui la sigla **LLM** che si incontra ovunque), smontato con i soli
attrezzi di questo capitolo, che sono poi gli unici che servono.

I nomi fra parentesi sono le etichette tecniche, e nessuna di esse è data per
saputa: ognuna nasce dentro la sua sezione, con un'immagine prima e la formula
dopo.

```{tip}
Se un simbolo ti blocca, non saltarlo: quasi sempre dietro una formula
intimidatoria si nasconde un'idea che sapresti spiegare a voce, ed è scritta
accanto alla formula.
```

Alla fine avrai in mano poche cose, e sono sempre quelle: una lista di numeri
con cui rappresentare qualunque dato, un modo per capire da che parte
migliorare, un modo per dire quanto sei sicuro, un modo per misurare l'errore
con un numero solo, e la consapevolezza che quei conti li fa una macchina
capace di scrivere poche cifre per volta. Le prime tre servono subito, e non
fra dieci capitoli: il {doc}`capitolo sul machine learning
</MachineLearning/overview>`, che viene dopo questo, è già tutto un metterle al
lavoro su un problema vero.
