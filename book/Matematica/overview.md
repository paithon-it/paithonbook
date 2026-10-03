# La matematica del machine learning: algebra lineare, analisi, probabilità

```{image} ../figures/aperture/matematica.png
:class: pt-apertura only-light
:width: 100%
:alt: Una clessidra: in alto le perline sono disposte in ordine, in basso ricadono in disordine.
```

```{image} ../figures/aperture/matematica-scura.png
:class: pt-apertura only-dark
:width: 100%
:alt: Una clessidra: in alto le perline sono disposte in ordine, in basso ricadono in disordine.
```

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

«Chiamala entropia, per due ragioni. La prima è che la tua funzione d'incertezza
in meccanica statistica si chiama già così. La seconda, più importante, è che
nessuno sa davvero che cosa sia l'entropia, quindi in una discussione partirai
sempre in vantaggio.» Si racconta che il consiglio venisse da John von Neumann,
e che a riceverlo fosse Claude Shannon, il quale nel 1948 aveva appena trovato
il modo di misurare l'informazione e non sapeva come chiamare la grandezza che
gli era venuta fuori. Aveva pensato a «informazione», parola già troppo usata,
poi a «incertezza». Alla fine seguì il consiglio, e quel nome è rimasto.

L'aneddoto è tramandato, non documentato: lo raccontò Shannon stesso, a voce,
nel 1961, e a stampa arrivò dieci anni più tardi, riferito da chi glielo aveva
sentito dire. Va preso con la cautela che meritano le battute riportate. Ma dice
una cosa giusta, ed è la cosa da mettere in chiaro prima di cominciare: spaventa
più il nome della cosa. L'entropia, sotto quel nome greco, misura quanto è
imprevedibile, in media, quello che sta per succedere: il lancio di una moneta
onesta è più imprevedibile di quello di una moneta che dà testa nove volte su
dieci, e ha un'entropia più alta. La {doc}`sezione sulla teoria
dell'informazione <teoria-informazione>` la trasforma in un numero. Con
gradiente, vettore e verosimiglianza succede lo stesso: dietro il nome c'è
un'idea che si dice a voce.

C'è quindi un equivoco tenace da sciogliere. Nell'epigrafe Galileo dice che il
libro dell'universo è scritto in lingua matematica, e non aveva torto; il
vocabolario che serve a noi, però, è corto. Per usare il machine learning non
serve essere matematici, ma per *capirlo* serve riconoscere le poche idee
matematiche che vi ritornano di continuo.

## Una foto, e tre lingue

Il compito più classico dell'AI è questo: guardare una foto e dire se contiene
un gatto. Per un calcolatore quella foto è una griglia di numeri (l'intensità di
ogni pixel), non un gatto. Disporre quei numeri in vettori e matrici (liste e
tabelle di numeri), e trasformarli con somme e prodotti per numeri fissi, è il
mestiere dell’**algebra lineare**: le trasformazioni di questo tipo si dicono
lineari. Poi quei numeri vengono trasformati più volte di seguito, e ogni
passaggio è uno strato: prende la lista di numeri che gli arriva e ne produce
un'altra, fino a che dall'ultimo esce la risposta «gatto / non gatto». Anche
questo è algebra lineare, alternata a funzioni non lineari, le funzioni di
attivazione, che *piegano* i numeri. Le pieghe sono indispensabili: sommare e
moltiplicare per numeri fissi, ripetuto cento volte, resta un sommare e
moltiplicare per altri numeri fissi, e senza funzioni di attivazione cento
strati farebbero quello che fa uno solo. Capire come ritoccare i numeri che il
programma ha dentro perché sbagli un po’ meno la volta dopo è **analisi**.
L'analisi è il ramo della matematica che studia come cambia una quantità quando
se ne muove un'altra: qui, di quanto cambia l'errore se si sposta un numero. E
poiché una risposta del genere non è mai una certezza (il programma è
«abbastanza sicuro» che sia un gatto), il modo naturale di esprimere quella
sicurezza è la **probabilità**.

Quel «programma» ha un nome preciso, e i nomi li ha già dati
l’{doc}`Introduzione </Introduzione/overview>`: un modello è un programma che
riceve numeri e ne restituisce altri; i suoi parametri $\theta$ sono le
migliaia (o i miliardi) di numeri regolabili che decidono la risposta; e
addestrarlo vuol dire modificarli finché le risposte non si avvicinano a quelle
giuste. Serve qui
una sola aggiunta, perché è quella su cui l'algebra lineare lavora: i
parametri per cui il modello moltiplica ciò che riceve si chiamano **pesi**.

Le tre lingue, insieme, coprono quello che serve a un modello: *rappresentare* i
dati, *regolare* i parametri, *quantificare* la fiducia nel risultato. Se ne
aggiungono due. La **teoria dell'informazione** dà la misura dell'errore per i
modelli che rispondono con probabilità, un numero solo (ed è la casa
dell'entropia di poco fa); l’**analisi numerica** si occupa di ciò che succede
ai numeri quando i conti li fa una macchina che per ogni numero scrive solo
poche cifre.

Qui dentro non c'è tutta la matematica: ci sono gli attrezzi che i capitoli
successivi useranno davvero, e nient'altro.

## Cinque attrezzi, una domanda ciascuno

Le tre lingue e le due che si aggiungono sono i cinque attrezzi del capitolo, e
ciascuno risponde a una domanda che si può fare a voce. Alcuni occupano più di
una sezione, perché la domanda si articola.

- Algebra lineare: come si mettono i numeri in fila, e come si trasformano tutti
  insieme. Sono quattro sezioni: *vettori, matrici, prodotti e norme* per
  cominciare; i *sistemi lineari*, cioè che cosa succede quando i dati impongono
  dei vincoli e quando quei vincoli non bastano; *ortogonalità e proiezioni*,
  che dicono che cosa fare quando un sistema non ha soluzione esatta; e il
  *determinante*, il fattore per cui una trasformazione moltiplica i volumi.
- Analisi e ottimizzazione: come si capisce da che parte migliorare, e come
  ci si arriva un passo alla volta (derivate, gradiente, discesa del gradiente).
- Probabilità e statistica: come si convive con l'incertezza, e come si
  aggiorna un'opinione quando arrivano dati nuovi (fino al teorema di Bayes).
  Seguono due sezioni che ne tirano le conseguenze: *quanto può sbagliare una
  media*, che dice quante prove servono per fidarsi di un numero misurato, e le
  *catene di Markov*, dove la probabilità incontra l'algebra lineare e la
  domanda «dove finisce, andando avanti per sempre?» ha una risposta esatta.
- Teoria dell'informazione: come si misura la sorpresa con un numero solo.
  È da lì che viene il punteggio d'errore con cui si addestra quasi ogni
  modello che deve scegliere fra alternative (entropia e cross-entropia).
- Analisi numerica: che cosa cambia quando i conti li fa una macchina che
  scrive solo poche cifre per numero, e come si evita che il conto vada fuori
  strada.

L'ultima sezione non aggiunge un sesto attrezzo: rimette al lavoro i cinque su
un oggetto solo, un modello linguistico (in inglese *large language model*, da
cui la sigla LLM che si incontra ovunque), smontato con gli attrezzi appena
elencati.

```{tip}
Se un simbolo ti blocca, non saltarlo: quasi sempre dietro una formula
intimidatoria si nasconde un'idea che sapresti spiegare a voce, ed è scritta
accanto alla formula.
```
