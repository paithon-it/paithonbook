---
titolo_ricerca: "Matematica per il machine learning: cinque strumenti"
descrizione: "Quale matematica serve al machine learning: algebra lineare,
  analisi e probabilità, e la domanda a cui risponde ciascuno dei cinque
  strumenti del capitolo."
---
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
sentito dire. Resta vero che il nome spaventa più della cosa.

L'entropia misura quanto è imprevedibile, in media, ciò che sta per succedere.
Ad esempio, il lancio di una moneta onesta è più imprevedibile di quello di una
moneta che dà testa nove volte su dieci, e ha quindi un'entropia più alta. La
{doc}`sezione sulla teoria dell'informazione <teoria-informazione>` la
trasforma in un numero. Con gradiente, vettore e verosimiglianza succede lo
stesso: dietro il nome c'è un'idea che si dice a voce.

Galileo scrive che il libro dell'universo è in lingua matematica, ma il
vocabolario che serve al machine learning è corto. Per usarlo non serve essere
matematici; per *capirlo* serve riconoscere le poche idee matematiche che vi
ritornano di continuo.

## Una foto, e tre lingue

Il compito più classico dell'AI è guardare una foto e dire se contiene un
gatto. Per un computer, però, quella foto è una griglia di numeri (l'intensità
di ogni pixel), non un gatto. Disporre quei numeri in vettori e matrici (liste
e tabelle di numeri), e trasformarli con somme e prodotti per numeri fissi, è
compito dell’**algebra lineare**, e le trasformazioni di questo tipo si dicono
appunto lineari.

Poi quei numeri vengono trasformati più volte di seguito, e ogni passaggio,
detto strato, prende la lista di numeri che gli arriva e ne produce un'altra,
fino a che dall'ultimo esce la risposta «gatto / non gatto». Anche questo è
algebra lineare, alternata a funzioni non lineari, le funzioni di
attivazione, che *piegano* i numeri. Senza di esse cento strati equivarrebbero
a uno solo, perché sommare e moltiplicare per numeri fissi, ripetuto cento
volte, resta una somma e una moltiplicazione per altri numeri fissi.

Ritoccare i numeri del programma perché sbagli un po’ meno la volta dopo è
compito dell’**analisi**, il ramo della matematica che studia come cambia una
quantità quando se ne muove un'altra: nel nostro caso, di quanto cambia
l'errore se si sposta un numero. Infine, poiché la risposta del modello non è
mai una certezza (il programma è «abbastanza sicuro» che sia un gatto), la
sicurezza si esprime con la **probabilità**.

Quel «programma» ha un nome, e l’{doc}`Introduzione </Introduzione/overview>`
ha già fissato i termini. Ricordiamo che un modello è un programma che riceve
numeri e ne restituisce altri; i suoi parametri $\theta$ sono le migliaia (o i
miliardi) di numeri regolabili da cui dipende la risposta; addestrarlo vuol
dire modificarli finché le risposte si avvicinano a quelle giuste. In
particolare, i parametri per cui il modello moltiplica l'ingresso si chiamano
**pesi**.

In sintesi, [le tre lingue servono a *rappresentare* i dati, a *regolare* i
parametri e a *quantificare* la fiducia nel risultato]{.pt-matita}. A queste se ne
aggiungono altre due. La **teoria dell'informazione** riduce a un numero solo
l'errore dei modelli che rispondono con probabilità (l'entropia di Shannon
nasce lì); l’**analisi numerica** studia che cosa accade ai numeri quando a
calcolare è una macchina che per ogni numero scrive solo poche cifre.

In questo capitolo non raccogliamo tutta la matematica, ma gli strumenti che
i capitoli successivi useranno.

## Cinque strumenti, una domanda ciascuno

Le tre lingue e le due che si aggiungono sono i cinque strumenti del capitolo,
e ciascuno risponde a una domanda che si può fare a voce.

- Algebra lineare: come si mettono i numeri in fila, e come si trasformano tutti
  insieme. Sono quattro sezioni: *vettori, matrici, prodotti e norme* per
  cominciare; i *sistemi lineari*, cioè che cosa succede quando i dati impongono
  dei vincoli e quando quei vincoli non bastano; *ortogonalità e proiezioni*,
  che spiegano che cosa fare quando un sistema non ha soluzione esatta; e il
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
  Da lì viene il punteggio d'errore con cui si addestra quasi ogni
  modello che deve scegliere fra alternative (entropia e cross-entropia).
- Analisi numerica: che cosa cambia quando a calcolare è una macchina che
  scrive solo poche cifre per numero, e come si evita che il risultato vada
  fuori strada.

L'ultima sezione riprende i cinque strumenti e li applica a un oggetto solo, un
modello linguistico (in inglese *large language model*, da cui la sigla LLM che
si incontra ovunque).
