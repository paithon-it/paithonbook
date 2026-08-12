# Conclusione

Abbiamo iniziato il capitolo con una citazione e concludiamo allo stesso modo.

Ogni tanto arriva una tecnologia che non risolve un problema, ma cambia il
modo in cui si risolvono tutti gli altri: la macchina a vapore, l’elettricità,
il computer, Internet. Non si riconoscono dal mestiere che svolgono, perché non
ne svolgono uno solo; si riconoscono da quanti mestieri diversi finiscono per
attraversare. È la compagnia in cui molti collocano l’intelligenza artificiale,
e la formula più fortunata è di Andrew Ng, professore all’Università di
Stanford e autore di uno dei corsi più apprezzati sul Deep Learning
(gratuitamente fruibile sulla piattaforma Coursera):

> L'intelligenza artificiale è la nuova elettricità.

È uno slogan, che Ng ha ripetuto in più occasioni; e come tutti gli slogan
regge perché taglia. Nella conversazione del 2017 da cui la formula è stata
raccolta il ragionamento per esteso è più cauto, e più interessante: «proprio
come l'elettricità ha trasformato quasi tutto cento anni fa, oggi faccio
davvero fatica a pensare a un settore che l'AI non trasformerà nei prossimi
anni» {cite}`ng2017electricity`. È una previsione, non un bilancio, e come
tutte le previsioni andrà verificata; ma qualche esempio concreto, di quelli
già successi, c'è.

Già nel 2016 DeepMind, uno dei principali laboratori di ricerca
sull'intelligenza artificiale, aveva usato le proprie reti neurali per ridurre
del 40% l'energia impiegata nel raffreddamento dei centri di elaborazione dati
di Google {cite}`evans2016deepmind`. Nella telemedicina, una rete neurale
addestrata su decine di migliaia di tracciati rileva le aritmie cardiache dal
solo elettrocardiogramma con un'accuratezza confrontabile con quella di un
cardiologo {cite}`hannun2019cardiologist`, e il gruppo che l'ha costruita è
quello dello stesso Andrew Ng citato qui sopra. E ancora: AlphaFold, sempre di
DeepMind, ha imparato a prevedere la forma tridimensionale delle proteine
{cite}`jumper2021highly`; conoscerla permette agli scienziati di comprenderne
il ruolo all'interno del corpo, e di studiare le malattie che si ritiene siano
causate da proteine «mal ripiegate», come l'Alzheimer, il Parkinson e la
fibrosi cistica.

Un esempio che invece conviene togliere dall'elenco, perché lo si trova
sempre dentro e non gli appartiene: i robot chirurgici che assistono il
medico in sala operatoria. La precisione di quelle incisioni non viene
dall'apprendimento, viene dal fatto che la macchina filtra il tremore della
mano e riduce la scala del gesto: è meccanica e controllo, cioè bella
ingegneria, e non c'è nessun modello che abbia imparato qualcosa dai dati.
Saperlo distinguere è già metà del mestiere che questo libro prova a
insegnare.

Ma se da un lato c’è fermento ed eccitazione per questa tecnologia, dall’altro l’intelligenza artificiale viene vista con scetticismo, paura e sgomento, per la possibilità che possa sostituire posti di lavoro o addirittura sfuggire al controllo umano.

Il modo migliore per superare tanto l'entusiasmo cieco quanto la paura è lo stesso: capire e studiare i meccanismi di questa tecnologia. È esattamente lo scopo di questo libro. Solo attraverso la conoscenza e la comprensione possiamo dissipare i timori infondati, riconoscere quelli fondati e sfruttare appieno il potenziale dell'intelligenza artificiale.

E si comincia dagli attrezzi. Il prossimo capitolo è dedicato a **Python**, il
linguaggio con cui tutto il resto del libro è scritto, e quello dopo alla
manciata di matematica che serve davvero (nulla che non si possa imparare qui,
strada facendo). Poi si entra nel merito: il machine learning, cioè il salto
di questa pagina raccontato per bene, e da lì in avanti le reti neurali.

`````{tab} Elementare

```{admonition} Da ricordare
:class: important
- Un **algoritmo** è una ricetta: una lista finita di passi precisi che portano
  a un risultato. Quello di Euclide, per il massimo comune divisore, ne è
  l'esempio più antico e sta in cinque righe di Python.
- Il salto di questo libro è che per moltissimi compiti (riconoscere un gatto,
  tradurre una frase) **la ricetta non la sappiamo scrivere**: allora si
  raccolgono migliaia di esempi già etichettati e si lascia che le regole
  **emergano dai dati**. È questo che significa, qui, dire che un programma
  *impara*.
- I tre nomi del titolo: **machine learning** è ricavare le regole dagli
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
  del capitolo sul machine learning.
- Le eccezioni sono istruttive: le GAN sostituiscono la minimizzazione con
  l'equilibrio di un gioco fra due reti, i metodi non parametrici (k-NN) non
  hanno parametri da stimare. E la cornice ha una crepa nota, il *reward
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

Benvenuto in Paithon Book!
