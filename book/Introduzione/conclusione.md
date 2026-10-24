# Conclusione

Ogni tanto arriva una tecnologia che non risolve un problema, ma cambia il
modo in cui si risolvono tutti gli altri: la macchina a vapore, l'elettricità,
il computer, Internet. Gli economisti le chiamano **tecnologie di uso
generale** (*general-purpose technologies*) {cite}`bresnahan1995general`. Non
fanno una cosa sola, e si riconoscono dal numero di settori che finiscono per
attraversare. Molti collocano l'intelligenza artificiale fra queste tecnologie,
e la formula più fortunata è di Andrew Ng:

> L'intelligenza artificiale è la nuova elettricità.

È uno slogan, e come tutti gli slogan funziona perché elimina le
sfumature. Ng insegna all'Università di Stanford ed è autore di alcuni fra i
corsi di machine learning più seguiti al mondo.

Nell'intervento del 2017 da cui la formula è presa, però, il ragionamento per
esteso dice qualcosa di più preciso: «proprio come l'elettricità ha trasformato
quasi tutto cento anni fa, oggi faccio davvero fatica a pensare a un settore
che l'AI non trasformerà nei prossimi anni» {cite}`ng2017electricity`. È una
previsione, non un bilancio, e come tutte le previsioni andrà verificata; ma
c'è già qualche esempio concreto.

Il primo esempio riguarda l'elettricità vera e propria. I servizi online girano
nei **data center** (centri di elaborazione dati), capannoni pieni di
server accesi giorno e notte. Il calore va smaltito di continuo, per cui il
raffreddamento pesa molto sui consumi di energia. Già nel 2016 DeepMind, il
laboratorio di ricerca sull'intelligenza artificiale di Google, aveva usato le
proprie reti neurali per ridurre fino al 40% l'energia usata per raffreddare i
data center dell'azienda {cite}`evans2016deepmind`.

Quel 40%, però, riguarda la sola voce del raffreddamento, e nello stesso
annuncio c'è un altro numero. Un centro del genere consuma parecchio oltre ai
computer: le ventole, le luci, e l'energia che si disperde negli impianti
elettrici. Chi gestisce questi centri misura il peso di quel contorno con il
**PUE** (*power usage effectiveness*), cioè il rapporto fra l'energia totale
del centro e quella assorbita dai soli server. Un PUE di $1{,}1$, per esempio,
vuol dire che il contorno consuma un decimo di quanto consumano i server.

Su quel contorno, che è una fetta più larga del solo condizionamento, il calo
dichiarato fu del 15%. Si noti che è il 15% del contorno, e non dell'energia
totale: un numero meno spettacolare, ma più indicativo.

Il secondo esempio viene dalla medicina. Una rete neurale addestrata su più di
novantamila tracciati, raccolti da oltre cinquantamila pazienti, riconosce le
aritmie cardiache da un elettrocardiogramma a una sola derivazione registrato
da un monitor portatile. Sul **punteggio F1** (una misura fra $0$ e $1$ che
premia chi trova le aritmie senza dare troppi falsi allarmi), la rete fa meglio
del cardiologo medio ($0{,}837$ contro $0{,}780$, su un insieme di prova di
alcune centinaia di tracciati) {cite}`hannun2019cardiologist`.

Per il confronto, un gruppo di cardiologi discute i tracciati finché non
converge su una risposta, e quella diventa la risposta giusta. Gli stessi
tracciati vanno poi ad altri cardiologi, che li leggono da soli, e si guarda
quante volte ciascuno di loro dà la risposta giusta e quante volte la dà la
rete. Chi fissa la risposta giusta e chi viene misurato non sono le
stesse persone, altrimenti il confronto non direbbe niente. La rete è opera del
gruppo di Andrew Ng, lo stesso della «nuova elettricità».

Il terzo esempio è AlphaFold, sempre di DeepMind, che ha imparato a prevedere
la forma tridimensionale delle proteine {cite}`jumper2021highly`. Una proteina
è una catena di amminoacidi, e quella catena si ripiega su se stessa fino ad
assumere una forma tridimensionale precisa, che ne decide la funzione. La
sequenza si legge in laboratorio con relativa facilità; la forma in cui si
ripiegherà, no, e capire come si passasse dall'una all'altra era un problema
aperto da mezzo secolo.

Conoscere la forma permette agli scienziati di comprendere il ruolo di una
proteina all'interno del corpo, e di studiare le malattie che si ritiene siano
causate da proteine «mal ripiegate», come l'Alzheimer, il Parkinson e la
fibrosi cistica.

Dall'elenco, invece, va tolto un esempio che ci finisce sempre dentro senza
appartenergli: i robot chirurgici che assistono il medico in sala operatoria.
La precisione di quelle incisioni, infatti, non viene dall'apprendimento. La
macchina toglie il tremore della mano e riduce il gesto in scala, così che a un
centimetro percorso dalla mano del chirurgo corrispondano pochi millimetri
percorsi dalla punta dello strumento, in un rapporto che il chirurgo sceglie. È
meccanica e controllo, cioè buona ingegneria, e in questa funzione non c'è
nessun modello che abbia imparato qualcosa dai dati.

Distinguere ciò che una macchina ha imparato dalla buona ingegneria serve anche
contro due reazioni opposte che questa tecnologia suscita: l'entusiasmo che le
attribuisce qualunque cosa e il timore che la immagina fuori controllo.
Nascono tutte e due dal non sapere che cosa ci sia dentro, e per questo [il
rimedio è lo stesso: guardare]{.pt-matita}.

Nessuno può elencare tutte le regole che un modello grande si è dato. Si può
però misurare che cosa sbaglia e su quali casi, e si può coprire un pezzo alla
volta la fotografia che gli si dà da guardare, per scoprire quale pezzo gli fa
cambiare risposta. È una disciplina a sé, con il suo capitolo:
{doc}`Interpretabilità </Interpretabilita/overview>`.

Guardare non porta a concludere che non c'è niente di cui
preoccuparsi (qualche problema è reale, e a quelli è dedicato il
{doc}`capitolo sull'AI responsabile </AIResponsabile/overview>`), ma a sapere
quali.

Si comincia dagli strumenti: {doc}`Python </Python/overview>`, il linguaggio
in cui è scritto il codice, e la {doc}`matematica </Matematica/overview>` che
serve davvero (chi ha in mano frazioni, potenze e percentuali impara il resto
strada facendo).

Poi si entra nel merito con il
{doc}`machine learning </MachineLearning/overview>`, cioè l'apprendimento dagli
esempi, e con le {doc}`reti neurali </RetiNeurali/overview>`, che oggi si usano
per immagini, testo e suono. Da lì il percorso si allarga: il
{doc}`reinforcement learning </ReinforcementLearning/overview>`, per quando gli
esempi giusti non esistono e resta solo un punteggio; i
{doc}`Transformer </Transformers/overview>`, l'architettura dei modelli con cui
oggi si conversa; e i capitoli su ciò che va storto, dall'interpretabilità
all'AI responsabile.

`````{tab} Elementare

```{admonition} Da ricordare
:class: important
- Un algoritmo è una ricetta: una lista finita di passi precisi che portano
  a un risultato. Quello di Euclide, per il massimo comune divisore, è fra i
  più antichi che si conoscano e sta in quattro righe di Python.
- Per moltissimi compiti (riconoscere un gatto, tradurre una frase) nessuno sa
  scrivere una ricetta che funzioni nel mondo vero. Allora si raccolgono
  migliaia di esempi e si lascia che le regole emergano dai dati: in questo
  senso un programma *impara*. Gli
  esempi possono portare la risposta scritta accanto da una persona
  (l’etichetta), oppure averla già dentro di sé, come la parola che in una
  frase viene dopo.
- Dentro il programma c'è un elenco di numeri, i parametri, e sono l'unica
  cosa che l'addestramento cambia: il comportamento viene dietro.
- I tre nomi da tenere distinti: machine learning è ricavare le regole dagli
  esempi; deep learning è farlo con reti a molti strati; reinforcement
  learning è imparare dalle conseguenze delle proprie azioni, con un
  punteggio al posto degli esempi. Il deep learning è un modo di fare machine
  learning; il reinforcement learning è un altro tipo di machine learning, che
  può usare le reti profonde o no; e l'intelligenza artificiale è più larga di
  tutti e tre, perché comprende anche i programmi che ragionano su regole
  scritte a mano.
- Buona parte di tutto questo funziona così: si sceglie un punteggio da far
  salire (o un errore da far scendere) e si lascia che sia la macchina a
  scoprire come. Con l'avvertenza che quel punteggio lo scriviamo noi, e non è
  mai esattamente la cosa che volevamo.
- Quando gli esempi giusti non ci sono, il vocabolario è questo: l’agente
  decide, l’ambiente risponde con una nuova situazione (lo stato) e con
  un punteggio (la ricompensa), e la policy è la regola con cui l’agente
  sceglie, cioè quello che deve imparare. Si allena dentro una simulazione, e
  portare sul robot vero quello che ha imparato lì resta un problema aperto.
- Non tutto si può calcolare: nessun programma sa dire, per ogni programma,
  se si fermerà (Turing), e in ogni sistema di regole abbastanza ricco ci sono
  verità che le regole non dimostrano (Gödel). I programmi che imparano
  restano programmi: nessun metodo generale sa dire che cosa farà ciascuno
  di loro, e un modello preso da solo si può controllare, ma a un costo che
  cresce in fretta con la sua taglia.
- Non è stata una salita continua: fra il 1956 e oggi ci sono due inverni,
  e funziona adesso perché sono arrivati insieme tre ingredienti, i dati,
  la potenza di calcolo e gli algoritmi.
```

`````

`````{tab} Superiore

```{admonition} Da ricordare
:class: important
- La cornice in cui stanno quasi tutti i metodi che seguono: si parametrizza
  il comportamento
  con $\theta$, se ne misura la qualità con
  $J(\theta) = \mathbb{E}_{\xi \sim p_\theta}[U(\theta, \xi)]$ e si cerca
  $\theta^\star \in \arg\max_\theta J(\theta)$, cioè
  $\arg\min_\theta \mathcal{L}$ con $\mathcal{L} = -J$.
- Quel valore atteso non si può calcolare, perché è preso sui casi futuri: in
  pratica si ottimizza la media su un campione già raccolto. La distanza fra le
  due quantità è la differenza fra *ottimizzare* e *imparare*, oggetto
  del {doc}`capitolo sul machine learning </MachineLearning/overview>`.
- Le eccezioni sono istruttive: le GAN sostituiscono la minimizzazione con
  l'equilibrio di un gioco fra due reti, i metodi non parametrici (k-NN) non
  hanno parametri da stimare per addestramento, perché al posto dei
  parametri conservano i dati (restano però da scegliere degli iperparametri,
  come il numero di vicini). E la cornice ha una crepa nota, il
  *reward hacking*: $J$ è il punteggio scritto da noi, non l'obiettivo vero.
- Il formalismo del reinforcement learning: un agente in uno stato $s_t$
  sceglie $a_t$ secondo una policy $\pi(a \mid s)$, riceve $r_{t+1}$, e
  massimizza il ritorno scontato $\mathbb{E}_{\pi}[\sum_t \gamma^t r_{t+1}]$,
  finito perché le ricompense sono limitate e $0 \le \gamma < 1$.
- Euclide: $\mathrm{MCD}(a,b) = \mathrm{MCD}(b, a \bmod b)$ converge in
  $O(\log \min(a,b))$ passi, cioè $O(n)$ passi su numeri di $n$ cifre; il
  tempo totale è $O(n^2)$, perché ogni cifra di quoziente costa $O(n)$ e le
  cifre dei quozienti sono in tutto $O(n)$.
- I limiti del calcolabile: la macchina di Turing dà all'algoritmo una
  definizione precisa, $\mathrm{HALT}$ è indecidibile (e con esso, per il
  teorema di Rice, ogni proprietà semantica non banale dei programmi), e ogni
  teoria coerente e ricorsivamente enumerabile che interpreti l'aritmetica è
  incompleta (Gödel). Valgono per ogni formalismo Turing-completo, quindi per
  i programmi che imparano presi come classe: non esiste un verificatore
  generale, anche se un modello fissato su ingressi di lunghezza limitata si
  può controllare, a un costo che cresce in fretta (verificare una rete ReLU è
  NP-completo).
```

`````

Di questo capitolo il libro riprende tre idee: le regole che emergono dagli
esempi invece di essere scritte a mano, il punteggio che si fa salire nel
frattempo, e il confine fra ciò che una macchina ha imparato e la buona
ingegneria che non ha imparato niente. Per usarle serve un linguaggio, e si
parte da Python.

Benvenuto in *Paithon Book*.
