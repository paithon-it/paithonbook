# Visione artificiale: far vedere le macchine

Nel 1966, al MIT, Seymour Papert affidò a un gruppo di studenti un compito per
l'estate: collegare una telecamera a un computer e insegnargli a descrivere
quello che vedeva. Il progetto si chiamava, con ottimismo, *Summer Vision
Project*. L'idea di fondo era che un problema tanto naturale (noi vediamo
senza sforzo, di continuo) si potesse sistemare in una manciata di settimane.
Mezzo secolo dopo, la visione artificiale è ancora un campo di ricerca vivo e
aperto. Quella sottovalutazione racconta una verità profonda: vedere ci
*sembra* facile solo perché il nostro cervello lo fa per noi, in silenzio e in
pochi millisecondi.

Questo capitolo parte proprio da lì: da che cosa significhi, per una macchina,
"vedere". E dalla scoperta (arrivata sul serio solo negli anni Dieci di questo
secolo) che il modo migliore per insegnarglielo non è scrivere regole, ma
mostrarle milioni di esempi.

## Un'immagine è una griglia di numeri

Per un computer non esistono "gatti", "cieli" o "volti": esistono numeri. Prima
di qualunque ragionamento, un'immagine dev'essere tradotta in qualcosa che una
macchina possa manipolare, e quel qualcosa è una griglia.

`````{tab} Elementare

Immagina una foto come un enorme foglio a quadretti. Ogni quadretto è un
**pixel**, e dentro ci sta un numero che dice quanto quel puntino è chiaro o
scuro: $0$ è nero pieno, $255$ è bianco pieno, e i valori in mezzo sono le
sfumature di grigio. Una foto in bianco e nero, per il computer, è tutta qui:
una tabella di numeri fra $0$ e $255$.

Se la foto è a colori, ogni quadretto non ha più un numero solo ma tre, quanto
rosso, quanto verde, quanto blu (il famoso **RGB**), che mescolati ricreano
ogni tinta. Un'immagine, insomma, è una griglia di numeri. Tutto il lavoro
della visione artificiale consiste nel trovare, dentro quella griglia, delle
regolarità che corrispondano a ciò che noi chiamiamo "un gatto".

`````

`````{tab} Superiore

Un'immagine è un **tensore** $X \in \mathbb{R}^{C \times H \times W}$,
nell'ordine *channels-first* di PyTorch: canali, altezza, larghezza (dove
$C=1$ in scala di grigi, $C=3$ per RGB). L'elemento $X_{c,i,j}$ è l'intensità
del canale $c$ nel pixel di riga $i$ e colonna $j$, tipicamente un intero in
$\{0,\dots,255\}$ che in fase di addestramento si normalizza in $[0,1]$ o si
standardizza a media nulla e varianza unitaria.

Le dimensioni crescono in fretta: una modesta immagine
$3 \times 224 \times 224$ (la taglia d'ingresso classica delle reti addestrate
su ImageNet) è un vettore di $150\,528$ numeri. Trattarla come un vettore
piatto, ignorando che i pixel vicini sono correlati, è proprio l'errore che le
reti convoluzionali (protagoniste dei prossimi paragrafi) evitano per
costruzione.

`````

## Perché è difficile: la distanza fra pixel e significato

Fra la griglia di numeri e la parola «gatto» c'è un salto che vale la pena
misurare prima di provare a colmarlo, perché ogni tecnica dei prossimi
capitoli è la risposta a una voce precisa di questo elenco.

`````{tab} Elementare

Il problema non è che i numeri siano tanti. È che **lo stesso gatto produce
griglie di numeri completamente diverse**, e due gatti diversi possono
produrne di molto simili. Sette modi in cui questo accade.

**Punto di vista.** Lo stesso animale visto di fronte, di lato o dall'alto non
ha un solo pixel in comune con sé stesso.

**Scala.** Vicino occupa tutta l'immagine, lontano una manciata di pixel.

**Deformazione.** Un gatto è un oggetto morbido: accucciato, in salto o
disteso è la stessa cosa con una forma diversa.

**Occlusione.** Metà del gatto è dietro il divano, e la rete deve rispondere
lo stesso.

**Illuminazione.** In controluce i valori dei pixel si ribaltano; al tramonto
tutta la scena vira all'arancione.

**Sfondo confuso.** Un gatto tigrato su un tappeto a righe: i confini che noi
vediamo senza pensarci il computer li deve inferire.

**Variazione dentro la classe.** Un siamese e un persiano condividono
l'etichetta e quasi nient'altro.

Tenete a mente l'elenco, perché tornerà voce per voce. La convoluzione, che
scorre lo stesso filtro dappertutto, è la risposta strutturale a una di queste
(un gatto è un gatto ovunque stia nell'immagine). La *data augmentation* del
capitolo seguente è, letteralmente, l'elenco riletto come ricettario: ruotare
contro il punto di vista, ritagliare contro la scala, cancellare rettangoli
contro l'occlusione, alterare la luminosità contro l'illuminazione. Non è un
insieme di trucchi: è un modo di dire alla rete quali cambiamenti **non**
devono cambiare la risposta.

`````

`````{tab} Superiore

Il salto si chiama **divario semantico**: fra la rappresentazione numerica
$X \in \mathbb{R}^{C\times H\times W}$ e la categoria semantica non c'è nessuna
relazione semplice, e in particolare nessuna relazione che si possa scrivere
guardando i valori dei pixel uno per uno.

Formulato con precisione, il compito è imparare una funzione
$f: \mathbb{R}^{C\times H\times W} \to \{1,\dots,K\}$ che sia **invariante** a
una famiglia di trasformazioni di *nuisance* (traslazione, scala, rotazione
limitata, cambi fotometrici, occlusioni parziali) e allo stesso tempo
**discriminativa** rispetto alle differenze fra classi, che sono spesso
molto più piccole, nella metrica dei pixel, delle variazioni da ignorare. Due
immagini della stessa classe possono avere distanza euclidea maggiore di due
immagini di classi diverse: è il motivo per cui un classificatore a vicini più
prossimi sui pixel grezzi funziona male, e la sezione sui filtri disegnati a
mano racconta il primo tentativo di rimediare.

Le invarianze si ottengono in tre modi, che il resto del capitolo percorre
tutti. **Per architettura**: la condivisione dei pesi della convoluzione dà
l'equivarianza alla traslazione, e il pooling una tolleranza locale alle
piccole traslazioni. **Per dati**: la *data augmentation* espone il modello
alle trasformazioni che deve ignorare, ed è un modo di iniettare
un'invarianza senza cablarla nell'architettura. **Per addestramento**:
l'apprendimento auto-supervisionato costruisce il compito proprio a partire
dalla scelta di quali trasformazioni debbano lasciare invariata la
rappresentazione, e lì la scelta delle trasformazioni **è** la definizione del
problema.

`````

## I compiti della visione

Avere i numeri è solo l'inizio. La domanda vera è: *che cosa chiediamo al modello
di produrre?* Da qui nascono i quattro compiti fondamentali, che si possono
leggere come una scala di ambizione crescente ({numref}`fig-compiti-visione`).

```{figure} ../figures/compiti-visione.svg
:name: fig-compiti-visione
:alt: Quattro pannelli mostrano la stessa scena con due gatti e una palla. Classificazione assegna una sola etichetta all'intera immagine; il rilevamento disegna un riquadro attorno a ogni oggetto; la segmentazione semantica colora i pixel per categoria, con i due gatti dello stesso colore; la segmentazione di istanza dà a ciascun oggetto un colore diverso.
:width: 90%

I quattro compiti classici della visione a confronto sulla stessa scena. Si va
dall'etichetta unica per l'immagine (classificazione) fino a distinguere ogni
singolo oggetto pixel per pixel (segmentazione di istanza).
```

`````{tab} Elementare

- **Classificazione**: "che cosa c'è in questa foto?". Il modello risponde con
  una sola parola per l'intera immagine (*gatto*) senza dire dove si trovi.
- **Rilevamento (detection)**: "che cosa c'è, e *dove*?". Il modello disegna un
  riquadro attorno a ogni oggetto e lo etichetta: due riquadri "gatto" e uno
  "palla".
- **Segmentazione semantica**: "a quale categoria appartiene *ogni singolo
  pixel*?". Si colora l'immagine come una cartina: tutti i pixel-gatto di un
  colore, i pixel-palla di un altro. I due gatti finiscono nella stessa tinta,
  perché sono la stessa *categoria*.
- **Segmentazione di istanza**: come sopra, ma i due gatti diventano due
  oggetti distinti, con colori diversi. È il compito più fine: separa non solo
  le categorie, ma i singoli individui.

`````

`````{tab} Superiore

Formalmente, i quattro compiti differiscono per la forma dell'output.

- **Classificazione**: $\hat{y} = \arg\max_{k \in \{1,\dots,K\}} f_k(X)$, una
  sola etichetta su $K$ classi per l'intera immagine.
- **Rilevamento**: l'output è un insieme di coppie
  $\{(\hat{c}_i,\ \mathbf{b}_i)\}_{i=1}^{N}$, dove $\hat{c}_i$ è la classe e
  $\mathbf{b}_i = (x, y, w, h)$ il *bounding box*. La qualità si misura con la
  *Intersection over Union* $\mathrm{IoU} = \frac{|A \cap B|}{|A \cup B|}$ tra
  box predetto e reale, aggregata nella *mean Average Precision* (mAP).
- **Segmentazione semantica**: una predizione per ogni pixel,
  $\hat{y}_{i,j} \in \{1,\dots,K\}$. Due istanze della stessa classe condividono
  l'etichetta.
- **Segmentazione di istanza**: a ogni pixel si associa una classe *e*
  un'identità di istanza, distinguendo gatto-1 da gatto-2.

Il costo di annotazione cresce nello stesso ordine: etichettare un'immagine è
questione di secondi, tracciare una maschera pixel-perfetta richiede minuti.

`````

Questi quattro compiti hanno una cosa in comune: rispondono tutti alla domanda
«che cosa», e lo fanno **dentro il piano dell'immagine**. Il riquadro di un
rilevatore dice dove sta un oggetto in pixel, non a quanti metri. Esiste però
una seconda famiglia di domande, che il capitolo affronta più avanti e che
richiede strumenti diversi: **dove sono le cose nello spazio**, quanto sono
lontane, come si muovono, che forma hanno. Lì la risposta non si trova
guardando meglio una fotografia, perché la profondità non è nascosta
nell'immagine: è andata perduta nel momento dello scatto, e va ricostruita da
più viste o indovinata con un modello di come è fatto il mondo.

## Dai filtri disegnati a mano alle feature imparate

Per decenni la strategia fu ovvia quanto faticosa: se vuoi trovare un bordo,
scrivi tu la regola per trovarlo.

`````{tab} Elementare

L'idea era che un esperto progettasse a mano dei "rilevatori": una formula per
scovare i bordi (dove il colore cambia bruscamente è probabile ci sia un
contorno), un'altra per le forme, un'altra per gli angoli. Funzionava, ma solo
fino a un certo punto: ogni nuovo problema richiedeva nuove regole cucite a
mano, e la realtà (luci, ombre, angolazioni) è troppo varia per essere
ingabbiata in istruzioni fisse.

La svolta è stata capovolgere il ragionamento: invece di dire alla macchina
*come* riconoscere un gatto, le mostriamo migliaia di gatti e lasciamo che sia
lei a costruirsi i rilevatori giusti. Le "regole" non le scrive più l'ingegnere:
emergono dai dati.

`````

`````{tab} Superiore

L'era delle *feature* ingegnerizzate ci ha lasciato strumenti tuttora eleganti:
il rilevatore di bordi di **Canny** (1986), i descrittori **SIFT** di Lowe
(2004), invarianti a scala e rotazione, e l'istogramma dei gradienti orientati
**HOG** di Dalal e Triggs {cite}`dalal2005histograms`, a lungo lo standard
per il rilevamento di
pedoni. Erano feature *fisse*, seguite da un classificatore addestrabile (spesso
una SVM).

La rottura arriva con le reti convoluzionali. LeCun e colleghi mostrano già nel
1998, con **LeNet-5**, che una CNN può imparare da sola i filtri leggendo cifre
scritte a mano. Ma è il 2012 lo spartiacque: **AlexNet** (Krizhevsky, Sutskever,
Hinton) vince la competizione ImageNet con un *top-5 error* del $15{,}3\%$
contro il $26{,}2\%$ del secondo classificato. Da allora le feature non si
disegnano più: si *imparano*, strato dopo strato, direttamente dai pixel.

`````

## Il carburante: i grandi dataset

Le CNN non avrebbero spiccato il volo senza qualcosa su cui volare. Il progetto
**ImageNet**, guidato da Fei-Fei Li e presentato nel 2009, mette insieme oltre
quattordici milioni di immagini etichettate; la sua sfida annuale (ILSVRC) usa un
sottoinsieme di mille categorie ed è la palestra su cui, nel 2012, AlexNet cambia
la storia. Poco dopo arriva **COCO** (*Common Objects in Context*, 2014), con
centinaia di migliaia di immagini annotate non solo con etichette, ma con box e
maschere per circa ottanta categorie di oggetti comuni: il banco di prova
naturale per rilevamento e segmentazione. La lezione è netta e vale per tutto il
deep learning: buoni dati, in grande quantità, contano quanto la buona
architettura.

## Come è organizzato il capitolo

Nei prossimi paragrafi partiamo dal mattone fondamentale: la **convoluzione**,
l'operazione che permette a una rete di "guardare" un'immagine rispettandone
la struttura spaziale. Da lì costruiamo le **reti convoluzionali** e ne
ripercorriamo le architetture che hanno fatto scuola. Passeremo poi ai compiti
più ambiziosi, il rilevamento e la segmentazione, e a tecniche pratiche come
il *transfer learning*, che consente di riusare reti già addestrate su
ImageNet per i nostri problemi con pochi dati. L'obiettivo non è solo capire
come funzionano: è metterle al lavoro, con PyTorch, sulle nostre immagini.

Le ultime sezioni cambiano domanda e passano dal «che cosa» al «dove»: la
**geometria** che lega una fotografia alla scena da cui viene, e che permette
di ricavare la profondità da due viste o dal movimento, e i **campi di
radianza**, con cui una scena si rappresenta addestrando una funzione invece
di ricostruire una superficie. Sono la parte della visione artificiale che le
reti non hanno sostituito ma su cui hanno costruito, ed è anche la più antica:
il modello di fotocamera che useremo lo dimostrò Brunelleschi nel Quattrocento
con una tavoletta forata.

`````{tab} Elementare

```{admonition} Da ricordare
:class: important
- Per un computer un'immagine è un **foglio a quadretti pieno di numeri**: ogni
  quadretto (un **pixel**) dice quanto quel puntino è chiaro o scuro, oppure
  quanto rosso, verde e blu contiene. Dentro non ci sono gatti né cieli: solo
  numeri, e il mestiere della visione artificiale è trovarci delle regolarità.
- I quattro compiti classici (**classificazione, rilevamento, segmentazione
  semantica e di istanza**) chiedono risposte via via più precise: una parola
  per tutta la foto, un riquadro attorno a ogni oggetto, un colore per ogni
  pixel secondo la categoria, fino a distinguere un gatto dall'altro.
- La grande transizione: prima erano gli esperti a scrivere a mano le regole
  per trovare bordi, angoli e forme; poi si è lasciato che fosse la rete a
  costruirsi da sola i propri rilevatori, guardando milioni di esempi. Il
  momento simbolo è la vittoria di AlexNet alla gara di ImageNet nel 2012.
- Senza le grandi raccolte di immagini già etichettate (**ImageNet**, **COCO**)
  niente di tutto questo sarebbe stato possibile: contano quanto una buona
  architettura.
- A quei compiti se ne aggiunge una famiglia diversa, che chiede **dove** sono
  le cose e non solo che cosa sono. Lì la difficoltà è di natura opposta: la
  distanza non è nascosta nella foto, è stata cancellata dallo scatto, e si
  recupera confrontando più immagini o affidandosi a ciò che il modello ha
  imparato su come è fatto il mondo.
```

`````

`````{tab} Superiore

```{admonition} Da ricordare
:class: important
- Per un computer un'immagine è un **tensore**
  $X \in \mathbb{R}^{C \times H \times W}$: una griglia di numeri, non
  di oggetti.
- I quattro compiti classici (**classificazione, rilevamento, segmentazione
  semantica e di istanza**) differiscono per la forma dell'output,
  dall'etichetta unica alla maschera per singolo oggetto.
- La grande transizione è dalle **feature disegnate a mano** (Canny, SIFT, HOG)
  alle **feature imparate** dalle CNN: la svolta è AlexNet su ImageNet (2012).
- Senza i grandi dataset (**ImageNet**, **COCO**) niente di tutto questo sarebbe
  stato possibile.
- Accanto ai quattro compiti c'è la **visione geometrica**, che stima posizioni
  e distanze invece di categorie. La proiezione prospettica non è invertibile
  (un pixel determina una direzione, non un punto), quindi la profondità si
  ricostruisce da più viste con vincoli esatti, oppure si stima da una vista
  sola con un prior appreso.
```

`````
