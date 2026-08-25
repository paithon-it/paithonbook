# Conclusione



Ogni tanto arriva una tecnologia che non risolve un problema, ma cambia il
modo in cui si risolvono tutti gli altri: la macchina a vapore, l'elettricità,
il computer, Internet. Non si riconoscono dal mestiere che svolgono, perché non
ne svolgono uno solo; si riconoscono da quanti mestieri diversi finiscono per
attraversare. È la compagnia in cui molti collocano l'intelligenza artificiale,
e la formula più fortunata è di Andrew Ng, che insegna all'Università di
Stanford ed è autore di alcuni fra i corsi di machine learning più seguiti al
mondo:

> L'intelligenza artificiale è la nuova elettricità.

È uno slogan, e come tutti gli slogan funziona perché butta via i distinguo.
Ng lo ha ripetuto in più occasioni; ma nella conversazione del 2017 da cui la
formula è stata raccolta il ragionamento per esteso è più cauto, e più
interessante: «proprio come l'elettricità ha trasformato quasi tutto cento anni
fa, oggi faccio davvero fatica a pensare a un settore che l'AI non trasformerà
nei prossimi anni» {cite}`ng2017electricity`. È una previsione, non un
bilancio, e come tutte le previsioni andrà verificata; ma qualche esempio
concreto, di quelli già successi, c'è.

Il primo riguarda l'elettricità per davvero. I servizi che usiamo ogni giorno
girano dentro capannoni pieni di computer accesi giorno e notte, i **centri di
elaborazione dati**, che scaldano al punto da doverli raffreddare in
continuazione: il condizionatore è una voce enorme della loro bolletta. Già nel
2016 DeepMind, il laboratorio di ricerca sull'intelligenza artificiale di
Google, aveva usato le proprie reti neurali per ridurre fino al 40% l'energia
che serviva a raffreddare quelli di casa propria {cite}`evans2016deepmind`.

Quel 40%, però, riguarda la sola voce del raffreddamento, e conviene guardare
anche l'altro numero dichiarato nello stesso annuncio. Un centro del genere
consuma parecchio oltre ai computer: le ventole, le luci, e l'energia che va
perduta per strada negli impianti elettrici. Su tutto quel contorno messo
insieme, che è una fetta più larga del solo condizionamento, il calo dichiarato
fu del 15%. È il numero meno spettacolare, ed è quello che dice di più.

Il secondo è in medicina. Una rete neurale addestrata su più di novantamila
tracciati, raccolti da oltre cinquantamila pazienti, riconosce le aritmie
cardiache dal solo elettrocardiogramma con un'accuratezza confrontabile con
quella di un cardiologo {cite}`hannun2019cardiologist`. Il confronto si fa
così: si consegnano gli stessi tracciati a un gruppo di cardiologi in carne e
ossa, e si prende come risposta giusta quella su cui il gruppo converge; poi si
guarda quanto spesso ci arriva ciascun cardiologo da solo, e quanto spesso ci
arriva la rete. Il gruppo che l'ha costruita è quello dello stesso Andrew Ng
citato qui sopra.

Il terzo è AlphaFold, sempre di DeepMind, che ha imparato a prevedere la forma
tridimensionale delle proteine {cite}`jumper2021highly`. Conviene dire perché
fosse difficile: una proteina è una catena di amminoacidi, e quella catena si
ripiega su se stessa fino ad assumere una forma tridimensionale precisa, che è
poi ciò che decide la sua funzione. La sequenza si legge in laboratorio con
relativa facilità; la forma in cui si ripiegherà, no, e capire come si
passasse dall'una all'altra era un problema aperto da mezzo secolo. Conoscerla
permette agli scienziati di comprendere il ruolo di una proteina all'interno
del corpo, e di studiare le malattie che si ritiene siano causate da proteine
«mal ripiegate», come l'Alzheimer, il Parkinson e la fibrosi cistica.

Un esempio che invece conviene togliere dall'elenco, perché lo si trova sempre
dentro e non gli appartiene: i robot chirurgici che assistono il medico in sala
operatoria. La precisione di quelle incisioni non viene dall'apprendimento.
Viene dal fatto che la macchina toglie il tremore della mano e rimpicciolisce
il gesto, così che al centimetro percorso dalla mano del chirurgo corrisponda
un millimetro percorso dalla punta dello strumento. È meccanica e controllo,
cioè bella ingegneria, e non c'è nessun modello che abbia imparato qualcosa dai
dati. Saperlo distinguere è già metà del mestiere che questo libro prova a
insegnare.

Distinguere l'apprendimento dalla buona ingegneria, del resto, è anche l'unica
difesa che c'è contro le due reazioni opposte che questa tecnologia raccoglie:
l'entusiasmo che le attribuisce qualunque cosa e il timore che la immagina
fuori controllo. Hanno in comune più di quanto sembri, perché nascono tutte e
due dal non sapere che cosa ci sia dentro, e si curano nello stesso modo:
andando a guardare. Elencare tutte le regole che un modello si è dato, è vero,
non può farlo nessuno; ma si può misurare che cosa sbaglia e su quali casi, e
si può coprire un pezzo alla volta della fotografia che gli si dà da guardare,
per scoprire quale pezzo gli fa cambiare risposta. È un mestiere vero, e ha il suo capitolo:
{doc}`Interpretabilità </Interpretabilita/overview>`. Non serve a concludere
che non c'è niente di cui preoccuparsi (qualche problema è reale, e a quelli è
dedicato il capitolo sull'AI responsabile), ma a sapere quali.

E si comincia dagli attrezzi. Il prossimo capitolo è dedicato a **Python**, il
linguaggio con cui tutto il resto del libro è scritto, e quello dopo alla
manciata di matematica che serve davvero: se hai in mano le frazioni, le
potenze e le percentuali, il resto lo impari qui, strada facendo. Poi si entra
nel merito: il machine learning, cioè il salto raccontato per bene, e da lì le
reti neurali, che sono il modo di farlo che ha vinto. Il libro si allarga da
lì. I Transformer, i programmi con cui oggi si conversa. Il reinforcement
learning, per quando gli esempi giusti non esistono e resta solo un punteggio.
E i capitoli che si occupano di quello che va storto, dall'interpretabilità
all'AI responsabile.

`````{tab} Elementare

```{admonition} Da ricordare
:class: important
- Un **algoritmo** è una ricetta: una lista finita di passi precisi che portano
  a un risultato. Quello di Euclide, per il massimo comune divisore, è fra i
  più antichi che si conoscano e sta in quattro righe di Python.
- Il salto di questo libro è che per moltissimi compiti (riconoscere un gatto,
  tradurre una frase) **la ricetta non la sappiamo scrivere**: allora si
  raccolgono migliaia di **esempi** e si lascia che le regole **emergano dai
  dati**. È questo che significa, qui, dire che un programma *impara*. Gli
  esempi possono portare la risposta scritta accanto da una persona
  (l’**etichetta**), oppure averla già dentro di sé, come una parola coperta in
  mezzo a una frase.
- I tre nomi da tenere distinti: **machine learning** è ricavare le regole dagli
  esempi; **deep learning** è farlo con reti a molti strati; **reinforcement
  learning** è imparare dalle conseguenze delle proprie azioni, con un
  punteggio al posto degli esempi.
- Buona parte di tutto questo funziona così: si sceglie un **punteggio** da far
  salire (o un errore da far scendere) e si lascia che sia la macchina a
  scoprire come. Con l'avvertenza che quel punteggio lo scriviamo noi, e non è
  mai esattamente la cosa che volevamo.
- Non è stata una salita continua: fra il 1956 e oggi ci sono **due inverni**,
  e funziona adesso perché sono arrivati insieme tre ingredienti, i **dati**,
  la **potenza di calcolo** e gli **algoritmi**.
```

`````

`````{tab} Superiore

```{admonition} Da ricordare
:class: important
- La cornice che regge buona parte del libro: si parametrizza il comportamento
  con $\theta$, se ne misura la qualità con $J(\theta) = \mathbb{E}[U \mid
  \theta]$ e si cerca $\theta^\star \in \arg\max_\theta J(\theta)$, cioè
  $\arg\min_\theta \mathcal{L}$ con $\mathcal{L} = -J$.
- Quell'attesa **non è calcolabile**, perché è presa sui casi futuri: in
  pratica si ottimizza la media su un campione già raccolto. La distanza fra le
  due quantità è la differenza fra *ottimizzare* e *imparare*, ed è l'oggetto
  del {doc}`capitolo sul machine learning </MachineLearning/overview>`.
- Le eccezioni sono istruttive: le GAN sostituiscono la minimizzazione con
  l'equilibrio di un gioco fra due reti, i metodi non parametrici (k-NN) non
  hanno parametri da stimare per addestramento, perché al posto dei parametri
  conservano i dati. E la cornice ha una crepa nota, il *reward
  hacking*: $J$ è il punteggio scritto da noi, non l'obiettivo vero.
- Il formalismo del reinforcement learning, ripreso nei due capitoli che gli
  sono dedicati: un agente in uno stato $s_t$ sceglie $a_t$ secondo una policy
  $\pi(a \mid s)$, riceve $r_{t+1}$, e massimizza il ritorno scontato
  $\mathbb{E}[\sum_t \gamma^t r_{t+1}]$.
- Euclide: $\mathrm{MCD}(a,b) = \mathrm{MCD}(b, a \bmod b)$ converge in
  $O(\log \min(a,b))$ **passi**, che non è lo stesso di $O(\log)$ nel tempo
  quando gli interi sono lunghi.
```

`````

Benvenuto in *Paithon Book*.
