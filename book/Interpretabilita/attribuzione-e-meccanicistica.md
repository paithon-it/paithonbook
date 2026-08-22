# Dentro le reti profonde: attribuzione e interpretabilità meccanicistica

Torniamo un'ultima volta al modello che credeva di riconoscere i lupi e in
realtà riconosceva la neve, quello con cui si è aperto il capitolo. Le macchie
colorate che lo smascherarono le aveva disegnate LIME, il metodo della sezione
precedente: trattava il modello come una scatola chiusa, gli passava l'immagine
con alcune porzioni spente e guardava come cambiava la risposta, senza mai
aprire niente.

E doveva andare così, per come quel modello era fatto. Non era una rete
addestrata da capo sulle venti foto: era una rete già pronta, presa da altri e
lasciata intatta, con appiccicato sopra un modello a somma di quelli della
prima sezione, e ad addestrarsi sulle venti foto truccate era stato soltanto
quest'ultimo. La scorciatoia («c'è neve → lupo») l'aveva imparata lui, non la
rete sotto.

Ma un modello, quando ce l'abbiamo in mano, aprirlo si può. Se dentro c'è una
rete neurale con tutti i suoi numeri, si può smettere di bussare da fuori e
andare a guardare che cosa succede lì dentro. È quello che fa questa sezione.
Qualche attrezzo che guarda dentro l'abbiamo già usato (l'importanza da
impurità legge i tagli di un albero, TreeSHAP ne sfrutta la forma), ma erano
alberi, e un albero si legge. Qui la cosa da aprire è una rete, e cambia
tutto.

Nella prima sezione abbiamo visto modelli che si spiegano da soli: un modello a
somma consegna un numero per ogni colonna, e quel numero *è* la spiegazione.
Una rete neurale no. Ha milioni di numeri intrecciati, e nessuno di essi, preso
da solo, dice qualcosa di sensato. Bisogna cambiare domanda. Non «quanto pesa
questa colonna in generale?», ma «quanto ha contribuito *questo* pezzo
dell'ingresso a *questa* risposta?». La quota di merito che si assegna a
ciascun pezzo dell'ingresso si chiama **attribuzione**, ed è la parola che
intitola questa sezione. La risposta, sorprendentemente, viene da uno strumento
che il libro ha già usato per tutt'altro mestiere: il **gradiente**.

## Mappe di salienza: il gradiente come misura di importanza

Conviene ricordare in due righe che cos'è, il gradiente, perché tutto quel che
segue ci si appoggia. Immagina un apparecchio pieno di manopole e con un solo
indicatore. La domanda «di quanto si sposta l'indicatore se giro questa manopola
di un nulla?» ha una risposta tecnica che si chiama **derivata**; il
**gradiente** è la stessa cosa fatta per tutte le manopole insieme, cioè
l'elenco completo di quelle risposte, una per manopola.

Nell'addestramento di una rete le manopole erano i numeri interni della rete
(che si chiamano **pesi**) e l'indicatore era l'errore: girare le manopole nel
verso che abbassa l'errore *è* l'addestramento, come si è visto nel capitolo
sulle reti neurali.

Adesso puntiamo lo stesso strumento altrove. Teniamo ferme le manopole dei pesi
e chiamiamo manopola ogni **pixel dell'immagine in ingresso**; l'indicatore da
guardare non è più l'errore, ma quanto la rete è convinta della risposta che ha
dato. Se la rete sceglie fra mille risposte possibili (cane, gatto, camion: si
chiamano le **classi**), l'indicatore è il punteggio che ha assegnato a quella
che ha scelto. Il risultato è una mappa che dice quanto ciascun pixel conta, e si chiama mappa
di **salienza**, cioè di ciò che «salta all'occhio» (in inglese *saliency map*,
ed è così che la si trova nei programmi): l'hanno proposta Simonyan, Vedaldi e Zisserman nel 2014
{cite}`simonyan2014deep`.

`````{tab} Elementare

Quali pixel, toccati appena, cambierebbero il verdetto della rete? Sposta di un
nulla il pixel del muso e la fiducia in «cane» crolla: quel pixel è
*importante*. Tocca un pixel dello sfondo e non succede niente: quello non
conta. La mappa di salienza è questa, un'immagine in bianco e nero delle stesse
dimensioni della foto, che si accende dove un piccolo ritocco farebbe la
differenza più grande.

È come cercare i punti fragili di un castello di carte: dai un colpetto qua e là
e guardi che cosa fa tremare tutta la struttura. Il difetto più visibile è che
due pixel vicini, che a occhio nostro fanno parte della stessa cosa, possono
rispondere in modo molto diverso: uno fa tremare tutto, quello accanto niente.
La mappa risulta allora piena di puntini sparsi (si dice che è **rumorosa**,
come una radio male sintonizzata), e la forma dell'oggetto si intravede appena.

Il secondo difetto è meno visibile. Il colpetto dice quanto è fragile *questo*
castello, così com'è messo adesso: sposta una carta e i punti fragili non sono
più gli stessi. La mappa vale per la foto che hai davanti, non per le foto di
cane in generale.

`````

`````{tab} Superiore

Sia $S_c(\mathbf{X})$ il punteggio (il logit, prima della softmax) che la rete
assegna alla classe $c$ per l'immagine $\mathbf{X}$. La saliency map è il modulo
del gradiente del punteggio rispetto all'ingresso:

$$
\mathbf{M} = \left| \frac{\partial S_c}{\partial \mathbf{X}} \right|,
$$

calcolato con una singola *backpropagation* fino allo strato di input anziché
fermarsi ai pesi. L'idea è una **linearizzazione locale**: nell'intorno di
$\mathbf{X}$,
$S_c(\mathbf{X} + \boldsymbol{\Delta}) \approx S_c(\mathbf{X}) +
\sum_{i,j} \big(\partial S_c/\partial X_{ij}\big)\, \Delta_{ij}$,
quindi le componenti del gradiente di modulo maggiore individuano i pixel la cui
piccola variazione altera di più il punteggio. Per un'immagine a colori si
prende in genere il massimo del modulo sui tre canali RGB.

Il limite è duplice. Primo, il gradiente è **locale**: coglie la pendenza solo
nel punto $\mathbf{X}$, e le reti profonde sono tutt'altro che lineari. Secondo, è
**rumoroso**, perché la superficie $S_c$ ha derivate che oscillano rapidamente.
Le mappe risultano granulose, e le tecniche successive nascono quasi tutte per
domare questo rumore.

`````

## Grad-CAM: dove guarda la rete

La salienza lavora sui pixel e paga in rumore. Un'alternativa più stabile
rinuncia alla risoluzione fine e chiede una cosa più grossolana ma più
robusta: in quale *regione* dell'immagine la rete ha trovato le prove della sua
decisione? È l'idea di **Grad-CAM** (*Gradient-weighted Class Activation
Mapping*), di Selvaraju e colleghi nel 2017 {cite}`selvaraju2017grad`.

`````{tab} Elementare

Serve prima una parola. Quando un'immagine attraversa una rete, ogni pezzo
della rete produce dei numeri, e quei numeri sono la fotografia di ciò che la
rete «ha in mente» a quel punto del percorso: si chiamano le **attivazioni** di
quello strato.

Come abbiamo visto nel capitolo sulla Visione Artificiale, in una rete fatta per
le immagini le attivazioni cambiano natura salendo di strato in strato: in basso
sono puntini e bordi, in alto sono zone che «assomigliano a un muso»,
«assomigliano a una ruota». Nell'ultimo strato di quel tipo (si chiama
convoluzionale) ce ne sono centinaia, una per ciascuna forma ricorrente che la
rete ha imparato a riconoscere, e ognuna è accesa nei punti dell'immagine in cui
quella forma compare. Ognuna, insomma, è come un faretto puntato su una parte
della foto.

E adesso il passaggio: il gradiente si può puntare anche su quei faretti, non
solo sui pixel. La domanda diventa «se questo faretto si accendesse un po’ di
più, di quanto salirebbe la fiducia nella risposta?». Grad-CAM chiede questo, e
poi accende ciascun faretto in proporzione alla risposta che ha avuto. Se
stiamo spiegando la risposta «cane», i faretti sul muso e sulle orecchie pesano
tanto, quelli sull'erba pesano zero. Sovrapposti alla foto, danno una macchia
calda (una *heatmap*) che dice, letteralmente, *dove* la rete ha guardato per
dire «cane». Dove invece i faretti tirano dall'altra parte, verso una risposta
diversa, la macchia resta fredda: tiene le prove a favore, non quelle contro. È
grossolana (la risoluzione è quella dell'ultimo strato, non dei pixel), ma è
pulita e onesta: nel caso dell'husky, la macchia calda finirebbe proprio sulla
neve, smascherando l'inganno.

`````

`````{tab} Superiore

Sia $\mathbf{A}^k \in \mathbb{R}^{u \times v}$ la $k$-esima *feature map*
dell'ultimo strato convoluzionale e $S_c$ il punteggio della classe $c$.
Grad-CAM procede in due mosse. Prima calcola un peso per ogni mappa, mediando
spazialmente il gradiente della classe rispetto a quella mappa:

$$
\alpha_k^c = \frac{1}{Z} \sum_{i}\sum_{j}
   \frac{\partial S_c}{\partial A^k_{ij}},
$$

dove $A^k_{ij}$ è il valore della mappa nella posizione $(i,j)$ e $Z = u\,v$ è
il numero di posizioni (un *global average pooling* del gradiente). Poi combina
le mappe pesate e tiene solo il contributo positivo:

$$
\mathbf{L}^c_{\text{Grad-CAM}} =
   \mathrm{ReLU}\!\left( \sum_k \alpha_k^c\, \mathbf{A}^k \right).
$$

Il peso $\alpha_k^c$ misura quanto la mappa $k$ conta per la classe $c$; la
$\mathrm{ReLU}$ scarta le regioni che *abbassano* il punteggio, tenendo solo
quelle che lo sostengono. La heatmap $\mathbf{L}^c$ ha la bassa risoluzione dello strato
convoluzionale ($7\times 7$ in una ResNet su input $224\times 224$) e va
sovracampionata alle dimensioni dell'immagine per la sovrapposizione. A
differenza della saliency, non risale ai pixel: guadagna in robustezza al rumore
ciò che perde in dettaglio spaziale, e localizza in modo affidabile l'oggetto
che ha guidato la decisione.

`````

## Non un punto solo, ma tutto il cammino: gli Integrated Gradients

Sia la salienza sia Grad-CAM misurano il gradiente in **un solo punto**,
l'immagine così com'è, e qui si nasconde un problema.

Le funzioni che una rete usa per decidere non salgono all'infinito: crescono
ripide finché la rete è incerta, e poi si appiattiscono, perché una fiducia non
può superare il suo massimo. La curva a esse che si comporta così, ripida in
mezzo e piatta alle due estremità, è la **sigmoide** vista nel capitolo sulle
reti neurali; e di un neurone arrivato sul tratto piatto si dice che è
**saturo**. Su un tratto piatto, però, il gradiente è quasi zero: toccare
l'ingresso non cambia più niente, perché la rete è già convinta. Il paradosso è
che il gradiente dichiara «questo pixel non conta» proprio quando quel pixel è
la ragione per cui la rete è così sicura.

Nel 2017 Sundararajan, Taly e Yan hanno affrontato la questione in un modo
diverso dal solito {cite}`sundararajan2017axiomatic`. Il metodo che ne è uscito
si chiama **Integrated Gradients**, cioè «gradienti integrati»: integrare, qui,
vuol dire raccogliere lungo tutta una strada invece che in un punto solo, ed è
esattamente quello che sta per succedere. Invece di inventare un
metodo e poi guardare se le mappe venivano belle, hanno scritto prima due
**assiomi**, cioè due proprietà che una buona spiegazione *deve* avere, e poi
hanno cercato il metodo che le rispetta.

Prima di enunciarli, però, serve un oggetto che avrà un ruolo grosso in tutta la
sezione: un ingresso «vuoto», da cui partire, che per un'immagine sarà
tipicamente un rettangolo tutto nero. È la situazione in cui il modello non ha
davanti niente, e si chiama **baseline**, che è l'inglese per «linea di
partenza».

Il primo assioma è la **sensibilità**, e va enunciato con la sua condizione,
altrimenti dice il falso. Prendiamo un ingresso e la baseline, e
supponiamo che differiscano per **una cosa sola**: un pixel, e nient'altro. Se
su quei due il modello risponde in modo diverso, allora quel pixel deve ricevere
una quota di merito diversa da zero. Non si può dire «non conta niente» di
quello che è l'unica differenza fra i due casi.

Il secondo assioma è l’**invarianza all'implementazione**: due reti costruite in
modo diverso, ma che a conti fatti calcolano esattamente la stessa cosa, devono
ricevere le stesse attribuzioni. La spiegazione riguarda *cosa* la rete calcola,
non *come* è scritta.

Ed è proprio il primo assioma che il gradiente misurato nel solo punto d'arrivo
viola, nei casi di saturazione: lì risponde zero anche quando quel pixel è
l'unica differenza che c'è fra l'immagine e la baseline.

`````{tab} Elementare

Invece di misurare la pendenza solo nel punto di arrivo, percorri la strada che
va dalla baseline, l'immagine tutta nera, fino all'immagine vera, mescolandole
a poco a poco. La strada si divide in tappe uguali, mettiamo otto, e ogni tappa
**copre un pezzetto di strada**, un ottavo per la precisione: è una cosa da
tenere a mente. A ogni tappa ti chiedi di quanto cambierebbe la fiducia della
rete se toccassi appena quel pixel, e alla fine fai la media delle otto
risposte. Così, anche se all'arrivo la rete è satura e non reagisce più, hai
comunque registrato la sua reazione lungo tutta la salita, quando reagiva
eccome.

Resta un ultimo passo, e c'è una ragione per farlo. Quella media dice quanto la
rete reagisce a *un ritocco* di quel pixel; a noi serve quanto ha contato il
pixel per intero, cioè tutta la strada che ha percorso dal nero al suo valore
vero. Quindi si moltiplica la media per quella strada: un pixel che è passato da
nero a bianco pieno l'ha fatta tutta e si prende tutto; uno che è rimasto quasi
nero ne ha fatta pochissima e si prende quasi niente.

Fare la media di otto numeri e poi moltiplicarla per tutta la strada è la stessa
identica cosa che **sommare** otto contributi, ciascuno la pendenza di una tappa
moltiplicata per il suo pezzetto di strada. Media o somma, dunque, sono due modi
di dire lo stesso conto.

E il conto torna, che è la proprietà bella di questo metodo: se sommi
le attribuzioni di tutti i pixel ottieni esattamente *quanto* la fiducia della
rete è cambiata fra l'immagine nera e quella vera. Niente si perde e niente si
inventa. Quella proprietà ha un nome: si chiama **completezza**. È come dividere
il conto di una cena tra i commensali in modo che la somma delle quote faccia, al
centesimo, il totale sullo scontrino.

Sull'analogia della cena, però, c'è una domanda scomoda da fare subito: il
conto torna, ma torna **a partire da dove?** Il punto di partenza, quell'immagine
nera, non è un fatto di natura: è una scelta, e la scelta cambia le risposte. La
più chiara delle conseguenze è questa: di ciò che era **già nero** in partenza
non si può misurare nessun contributo, perché fra partenza e arrivo non è
cambiato. Se la ragione della decisione fosse proprio una zona buia della foto
(un'ombra, il cielo notturno, il nero di una radiografia), quel metodo le
darebbe zero, e la somma tornerebbe lo stesso. Che il conto torni non dice che
si è partiti dal punto giusto.

`````

`````{tab} Superiore

Sia $\mathbf{x}'$ la **baseline**, il punto di riferimento «neutro» rispetto a cui si
misura il contributo (per un'immagine, tipicamente il nero,
$\mathbf{x}' = 0$), e $\mathbf{x}$ l'ingresso da spiegare. In forma precisa,
l'assioma di **sensibilità** chiede che se $\mathbf{x}$ e $\mathbf{x}'$
differiscono in una sola componente e
$f(\mathbf{x}) \neq f(\mathbf{x}')$,
quella componente riceva attribuzione non nulla: è proprio ciò che il gradiente
valutato nel solo punto $\mathbf{x}$ non garantisce, perché in regime di saturazione è
quasi zero anche quando quella componente è la ragione dell'uscita.

Gli *Integrated Gradients* integrano il gradiente lungo
il segmento rettilineo da $\mathbf{x}'$ a $\mathbf{x}$:

$$
\mathrm{IG}_i(\mathbf{x}) = (x_i - x'_i)\,
   \int_0^1 \frac{\partial
     f\big(\mathbf{x}' + \alpha\,(\mathbf{x} - \mathbf{x}')\big)}{\partial x_i}\,
   \mathrm{d}\alpha,
$$

dove $f$ è l'uscita della rete per la classe d'interesse, $\alpha \in [0,1]$
parametrizza il cammino e $\mathrm{IG}_i$ è l'attribuzione della $i$-esima
componente d'ingresso. In pratica l'integrale si approssima con una somma di
Riemann su $m$ passi. Integrando lungo il cammino, il metodo cattura anche i
gradienti *prima* della saturazione, dove il segnale è vivo, risolvendo la
cecità del gradiente locale.

La proprietà che ne fa uno strumento affidabile è la **completezza**: le
attribuzioni si sommano esattamente alla differenza di uscita tra input e
baseline,

$$
\sum_i \mathrm{IG}_i(\mathbf{x}) = f(\mathbf{x}) - f(\mathbf{x}').
$$

La completezza implica la sensibilità e conferisce alle attribuzioni un
significato preciso: ciascuna è la *quota* di quel salto di punteggio imputabile
a quella componente.

C'è però un limite, ed è il punto in cui il metodo si presta a essere letto per
più di quel che promette: la completezza vale per **qualunque** baseline. Gli
assiomi vincolano come si ripartisce il salto $f(\mathbf{x}) - f(\mathbf{x}')$,
non da dove il salto parte, e la scelta di $\mathbf{x}'$ resta il grado di
libertà principale del metodo. Due conseguenze concrete. La prima è che il
fattore $(x_i - x'_i)$ davanti all'integrale **azzera per costruzione**
l'attribuzione di ogni componente che coincide con la baseline: con la baseline
nera ogni pixel nero riceve esattamente zero, per definizione e non per misura,
mentre in una radiografia o in una foto notturna il nero non è affatto assenza
di informazione. La seconda è che cambiando baseline le attribuzioni cambiano
di grandezza e perfino di **segno**, mentre la somma continua a tornare: sulla
funzione giocattolo $f(\mathbf{x}) = \tanh(\mathbf{w}^\top \mathbf{x})$ con
$\mathbf{w} = (2,-1)$ e $\mathbf{x} = (2,1)$, spostando la baseline da $(0,0)$
a $(2,0)$ l'attribuzione della componente con il peso maggiore passa da
$1{,}327$ a esattamente $0$, e con la baseline $(-1,2)$ la seconda componente
prende segno positivo invece che negativo; in tutti e tre i casi la completezza
è verificata al quarto decimale. Gli autori chiedono infatti che la baseline
sia scelta e **verificata**: deve rappresentare un'assenza di segnale, e su di
essa il punteggio della classe dev'essere quasi nullo. Le alternative d'uso
comune (rumore gaussiano, immagine sfocata, media del dataset, media su più
baseline) danno attribuzioni diverse, e nessun assioma le ordina
{cite}`sturmfels2020baselines`.

`````

Questo metodo ha una particolarità da rendere esplicita: **non si vede in un
fotogramma**, perché il fotogramma è proprio il punto in cui il gradiente non
dice niente. In {numref}`fig-gradienti-integrati` c'è il cammino, percorso a
passi.

```{figure} ../figures/gradienti-integrati.svg
:name: fig-gradienti-integrati
:alt: "A sinistra la curva dell'uscita della rete lungo il segmento che va dalla baseline all'ingresso: parte ripida e si appiattisce. Un pallino la percorre a passi, e a ogni passo un segmento mostra la pendenza in quel punto, che all'inizio è grande e alla fine quasi nulla. A destra una barra accumula la somma delle pendenze e si ferma esattamente sulla riga che segna la differenza fra l'uscita sull'ingresso e quella sulla baseline."
:width: 92%

Il cammino dall'immagine neutra a quella vera, percorso a otto tappe. A
sinistra la fiducia della rete lungo la strada, e a ogni tappa un trattino che
mostra quanto è ripida lì: molto all'inizio, quasi niente alla fine. A destra la
somma che si accumula tappa per tappa, e la riga tratteggiata che segna dove
deve arrivare. La curva del disegno è quella di un neurone che satura, la stessa
forma della sigmoide, e i numeri sono calcolati su di essa.
```

Il disegno di {numref}`fig-gradienti-integrati` mostra due cose che il conto,
da solo, non fa vedere.

La prima è **quanto** la saturazione morda. Sulla curva della figura la
pendenza (cioè di quanto salirebbe la fiducia della rete a spingere un pochino
sull'ingresso) vale $3{,}76$ alla prima delle otto tappe e $0{,}0088$
all'ultima: oltre quattrocento volte meno. Il disegno arrotonda a $0{,}01$ e
sembra zero, e infatti è quello il numero su cui lavora un metodo che guardi
solo il punto d'arrivo. Da lì l'errore: dichiara «non conta niente» un pixel che
è tutta la spiegazione.

La seconda è che la barra di destra si ferma esattamente sulla riga, e lì
conviene separare due cose che sembrano una sola.

Che la riga sia il posto giusto è un **teorema**: se si misura la pendenza in
*ogni* punto del cammino, e non solo in otto, la somma fa esattamente il salto
di fiducia fra la partenza e l'arrivo. Sempre, senza eccezioni. È la
completezza.

Che ci arrivi anche una somma di **otto sole tappe** è invece un'altra
faccenda, e non è garantita da niente. Qui torna utile la cosa da tenere a
mente: ogni tappa copre un pezzetto di strada, e lungo quel pezzetto la pendenza
non è costante, cala. Bisogna quindi decidere in che punto del pezzetto
misurarla, e la decisione cambia il risultato.

Nel disegno la si misura **in mezzo** al pezzetto, ed è la scelta buona: il
totale che ne esce, $0{,}99936$, differisce dal valore vero, $0{,}99933$, di
appena $0{,}000027$, e a occhio la barra tocca la riga. Se invece la si
misurasse **all'inizio** di ciascun pezzetto, cioè dove la salita è ancora la
più ripida di tutto il tratto, si sopravvaluterebbe ogni volta: il totale
verrebbe $1{,}249$ invece di $0{,}999$, cioè il $25\%$ di troppo, e la barra
sforerebbe la riga di un quarto della sua altezza. Il conto torna per teorema;
l'approssimazione del conto, no.

## Le mappe dicono dove, non che cosa

Tre metodi, tre mappe sempre più pulite. Prima di andare avanti conviene
fermarsi e chiedersi una cosa che le mappe, per come sono fatte, non
suggeriscono da sole: **una mappa di importanza è una spiegazione?**

Ci sono due obiezioni, e sono di natura diversa. La prima riguarda cosa una
mappa può dire in linea di principio; la seconda, più grave, riguarda se stia
davvero parlando del modello.

`````{tab} Elementare

La prima obiezione la mette bene Cynthia Rudin: sapere **dove** la rete
guarda dentro l'immagine non dice **che cosa** stia facendo con quella parte.
Una mappa che si accende sul muso del cane è compatibile con «la rete
riconosce la forma di un muso», ma anche con «la rete ha imparato che in
quella zona di solito c'è del pelo, e riconosce quello», o con «quel pezzo di
immagine è semplicemente il più contrastato». La salienza dice ciò che la rete *vede*,
non ciò che la rete *pensa*.

La seconda obiezione è più radicale ed è arrivata da un esperimento tanto
semplice quanto crudele. Prendi una rete addestrata, produci la sua mappa, poi
**cancella quello che ha imparato**: randomizza i pesi, strato per strato, e
rifai la mappa. Oppure la riaddestri su etichette mescolate a caso, così che
non le resti da imparare altro che rumore. Se la mappa fosse una spiegazione
del modello, dovrebbe disintegrarsi, perché il modello non c'è più. Per diversi
metodi popolari, la mappa cambia pochissimo, e resta riconoscibile come una
sagoma dell'oggetto.

Il test però non li boccia tutti. I colpetti pixel per pixel e i faretti di
Grad-CAM reagiscono, e il test lo superano. A fallirlo del tutto sono due
metodi non ancora incontrati, *Guided BackProp* e *Guided Grad-CAM*: sono due
raffinamenti costruiti sopra i primi due, con qualche accorgimento in più per
avere mappe visivamente più nitide, e proprio quegli accorgimenti sono ciò che
li rende ciechi al modello. Le loro mappe restano riconoscibili anche su una
rete a cui è stato cancellato tutto.

Gli **Integrated Gradients** stanno in mezzo, ed è il caso più insidioso. La
mappa cambia davvero, e cambia parecchio: un pixel che prima spingeva verso la
risposta può ritrovarsi a spingere contro. Però resta visibile la sagoma
dell'oggetto fotografato, e il motivo è nel metodo stesso. Ricordiamo l'ultimo
passo: la reazione media si moltiplica per quanto il pixel è cambiato dal nero.
Ma quel «quanto è cambiato», messo insieme per tutti i pixel, *è l'immagine*.
Quindi quando la reazione media si riduce a un guazzabuglio senza senso, a
restare in piedi nel prodotto è soprattutto la foto. A occhio si continua a
«vedere il cane» anche quando la rete non sa più niente.

La conclusione è spiacevole: per i metodi bocciati e per quelli in mezzo, la
mappa stava in buona parte descrivendo l’**immagine** e non la rete. Somigliava
al risultato di un programmino che segna i bordi degli oggetti in una foto, uno
di quelli che esistono da cinquant'anni e non hanno bisogno di imparare niente;
e siccome i bordi di una foto di cane disegnano un cane, la mappa sembrava
sensata.

`````

`````{tab} Superiore

L'obiezione di Rudin {cite}`rudin2019stop` è che una mappa di salienza è
compatibile con troppe spiegazioni diverse dello stesso comportamento: è
un'informazione sulla *posizione* dell'evidenza, non sul calcolo che la usa.
Nei contesti ad alto rischio, sostiene, questo la rende inadatta a sostituire
un modello intrinsecamente interpretabile.

La seconda obiezione è empirica e ha una forma metodologica importante: sono i
**controlli di sanità** di Adebayo e colleghi {cite}`adebayo2018sanity`. Il
ragionamento è che un metodo di attribuzione, per essere utile, deve almeno
essere **sensibile** alle cose da cui la predizione dipende. Da qui due test.

Nel *model parameter randomization test* si randomizzano progressivamente i
pesi del modello, dall'ultimo strato verso il primo, confrontando ogni volta la
mappa con quella originale. Nel *data randomization test* si riaddestra il
modello su etichette permutate a caso, cosicché abbia necessariamente
memorizzato rumore, e si confronta la mappa con quella del modello addestrato
sulle etichette vere.

Un metodo che superi i test deve cambiare drasticamente in entrambi i casi.
Diversi metodi molto usati non lo fanno, e le loro mappe restano visivamente
simili all'originale: si comportano, per usare l'espressione degli autori, come
un rilevatore di bordi indipendente dal modello. Quali siano è l'informazione
operativa. A **fallire** sono **Guided BackProp** e **Guided Grad-CAM**,
invarianti ai pesi degli strati alti e quindi riconoscibili anche a rete
randomizzata. A **passare** sono il gradiente semplice e Grad-CAM, quest'ultimo
con la precisazione che gli autori mettono per esteso: è sensibile ai pesi
quando la randomizzazione è *a valle* dell'ultimo strato convoluzionale, cioè
su quella parte della rete che Grad-CAM attraversa per calcolare i suoi
gradienti.

In mezzo stanno i metodi che moltiplicano il gradiente per l'ingresso, cioè
**gradient $\odot$ input** e gli **Integrated Gradients**. Qui gli autori
osservano che le mappe cambiano, e cambiano perfino di segno, ma che la
struttura dell'ingresso resta chiaramente prevalente nelle maschere: chi
moltiplica per l'ingresso, quando il gradiente si fa rumoroso, finisce per
restituire soprattutto l'ingresso. È il caso più insidioso, perché il metodo
*è* sensibile al modello e nondimeno l'occhio continua a riconoscere l'oggetto.
Il punto metodologico da portarsi via è questo: **la plausibilità visiva di una
spiegazione non è una prova della sua fedeltà**, e un occhio umano non
distingue le due cose. Un metodo di attribuzione va sottoposto a un test che
possa farlo fallire, esattamente come un modello.

`````

Il che non rende inutili le mappe: le ricolloca. Servono a **esplorare** (dove
guardare, quale ipotesi farsi, quale scorciatoia sospettare in una raccolta di
dati) e non a certificare che un modello funzioni. E lo stesso dubbio, tale e
quale, si è poi posto per un altro oggetto, che a differenza delle mappe non bisogna
nemmeno costruire, perché nel modello c'è già.

## L'attenzione è una spiegazione?

Quell'oggetto è l’**attenzione** dei Transformer, e c'è una tentazione naturale
a usarlo così {cite}`vaswani2017attention`. Richiamiamo in due righe di che si
tratta: per decidere che cosa fare di una parola, il modello distribuisce una
specie di sguardo sulle altre parole della frase, dando a ciascuna una quota. Le
quote sommano sempre a uno, come le fette di una torta divisa fra tutte le
parole, e sembrano dire su che cosa il modello si è concentrato. (Si chiamano
anch'esse «pesi», con lo stesso nome dei numeri interni della rete: sono
un'altra cosa, e qui, per non confonderli, le chiameremo quote.) Sono già lì,
non costa niente guardarle: perché non usarle come spiegazione, gratis?

La risposta breve è: con molta cautela. Nel 2019 Jain e Wallace
{cite}`jain2019attention` hanno mostrato che spesso si possono costruire quote
di attenzione **molto diverse** che portano il modello alla **stessa** risposta.
E il ragionamento che ne segue è pulito: se più modi di distribuire lo sguardo
danno lo stesso verdetto, nessuno di essi può essere *la* ragione del verdetto.
Il loro articolo si intitolava, senza giri di parole, *«Attention is not
Explanation»*. Altri hanno ribattuto, con un articolo intitolato *«Attention is
not not Explanation»* {cite}`wiegreffe2019attention`, che dipende da che cosa si
pretende, e che con richieste più modeste qualcosa quelle quote lo dicono. La
morale pratica è quella: sono un indizio, non una prova, e vanno lette come una
traccia, non come una confessione.

Va però messa un'avvertenza accanto al risultato, perché quel titolo è più
largo dell'esperimento che lo sostiene. Jain e Wallace guardano modelli con **un
solo strato di attenzione**, appoggiato per lo più sopra un tipo di rete che
legge la frase parola per parola nei due sensi, e che non è un Transformer. Un
Transformer vero, con le sue decine di strati sovrapposti, nel loro articolo non
compare mai: né come esperimento né come parola. E nel lavoro stesso il risultato **cambia col modello**: sul più semplice fra
quelli provati, gli stessi criteri assolvono l'attenzione invece di
condannarla. È quindi un monito metodologico solido, non un teorema sui
Transformer.

C'è però una domanda tecnica che precede quella filosofica, e che di solito
viene saltata: **l'attenzione di quale strato?** Un Transformer ne ha decine,
impilati, e guardarne uno solo è come giudicare una catena di montaggio da una
sola stazione.

`````{tab} Elementare

Il problema è che, a ogni piano della pila, una parte dell'informazione non
passa affatto dall'attenzione: prende una **scorciatoia** e scivola dritta al
piano di sopra (sono le connessioni residuali della
{doc}`sezione sulla struttura del Transformer </Transformers/architettura>`).
Quindi le quote di un singolo strato raccontano solo un pezzo del viaggio: per
sapere quanto ogni parola d'ingresso ha influenzato il risultato in cima
bisogna seguire l'intero percorso, scorciatoie comprese, piano dopo piano. Gli
strumenti che fanno questo conto si chiamano **attention rollout** e
**attention flow**, e restituiscono una mappa sulle parole di partenza, spesso
più sensata di quella del singolo strato.

C'è infine un attrezzo complementare, il **probing** (sondaggio): per scoprire
se a un certo piano della rete è scritta una data informazione (per esempio, se
una parola è un nome o un verbo), si prova a leggerla da lì con lo strumento
più semplice che c'è, un piccolo classificatore addestrato apposta. Se ci
riesce, l'informazione a quel piano c'è; se fallisce, non c'è, o non è scritta
in modo semplice. Con un'avvertenza: uno strumento di lettura troppo bravo
rischia di indovinare da sé ciò che doveva soltanto leggere. Per accorgersene
gli si dà da leggere un'informazione che non c'è, cioè etichette tirate a caso:
se riesce lo stesso, non stava leggendo, stava indovinando.

`````

`````{tab} Superiore

Abnar e Zuidema {cite}`abnar2020quantifying` mostrano che la composizione fra
strati non è affatto banale, per una ragione che il capitolo sui Transformer
ha già messo in evidenza: le **connessioni residuali**. A ogni blocco il
valore di un token non viene sostituito da ciò che l'attenzione gli porta, ma
sommato ad esso; quindi una parte dell'informazione che arriva allo strato
$l+1$ non è passata dall'attenzione di quel livello, ma è scivolata lungo la
scorciatoia.

Il rimedio proposto è di tenerne conto e poi comporre. Si corregge la matrice
di attenzione di ogni strato mescolandola con l'identità, che rappresenta
appunto il passaggio diretto,

$$
\mathbf{R}^{(l)} = \tfrac{1}{2} \mathbf{W}^{(l)}_{\text{att}}
   + \tfrac{1}{2} \mathbf{I} ,
$$

dove $\mathbf{W}^{(l)}_{\text{att}}$ è la matrice di attenzione grezza dello
strato $l$, $\mathbf{I}$ l'identità e $\mathbf{R}^{(l)}$ la matrice corretta.
Si moltiplicano poi le $\mathbf{R}^{(l)}$ fra loro per ottenere quanto di ogni
token di ingresso è finito in ogni posizione all'altezza voluta. È l’**attention
rollout**. La variante *attention flow* tratta la stessa struttura come un
grafo orientato aciclico e calcola il flusso massimo dal token di ingresso a
quello di arrivo, che è più costoso e tiene conto dei colli di bottiglia lungo
il cammino. In entrambi i casi il risultato è una mappa sui **token
d'ingresso**, cioè finalmente confrontabile con le attribuzioni delle sezioni
precedenti, e visibilmente diversa (spesso più sensata) della matrice del
singolo strato che si è tentati di visualizzare.

Un approccio complementare, più controllato, è il **probing**. L'idea: se una
rappresentazione interna «sa» qualcosa (poniamo, la parte del discorso di una
parola), allora un classificatore *lineare* addestrato su quella
rappresentazione dovrebbe saperlo prevedere. Si congela la rete, si estraggono
le attivazioni di uno strato e ci si allena sopra una semplice regressione
logistica per una proprietà a scelta. Se il probe riesce, l'informazione è
presente e linearmente accessibile in quello strato; se fallisce, non lo è. È
un modo economico per mappare *dove*, nella pila di strati, emergono le varie
proprietà. Il probe lineare è la proposta di Alain e Bengio
{cite}`alain2017understanding`, che lo motivano proprio con la separabilità
lineare e con la convessità del problema di addestramento; l'avvertenza che ne
delimita l'uso è invece di Hewitt e Liang {cite}`hewitt2019control`, e va
attribuita a loro: un probe troppo potente rischia di *imparare* lui la
proprietà invece di limitarsi a leggerla. La loro proposta per accorgersene
sono i *control task*, cioè rifare lo stesso addestramento su un'etichettatura
casuale, e misurare la **selectivity**, lo scarto fra quanto il probe riesce
sulla proprietà vera e quanto riesce sul casuale. Un probe che va bene su
entrambe non stava leggendo: stava risolvendo.

`````

## Interpretabilità meccanicistica: fare reverse-engineering dei circuiti

L'attribuzione dice *cosa* pesa, il sondaggio degli strati (il *probing* di
poco fa) dice *dove* sta l'informazione; nessuno dei due dice *come* la rete la
calcola. La frontiera, giovane e ambiziosa e ancora molto aperta, punta più in
alto: **fare reverse-engineering** dei calcoli interni, come si smonta un
circuito elettronico per capire cosa fa ciascun componente. È
l’**interpretabilità meccanicistica**.

`````{tab} Elementare

Finora abbiamo trattato la rete come una scatola su cui bussare da fuori: le
mostri un ingresso, guardi l'uscita, misuri le reazioni. L'interpretabilità
meccanicistica apre la scatola e prova a leggere il circuito dentro.
L'obiettivo è ricostruire i **circuiti**: piccoli gruppi di neuroni collegati
che, insieme, svolgono un compito riconoscibile (un rilevatore di curve, un
pezzo che tiene il conto delle parentesi aperte in un testo).

C'è però un ostacolo curioso. La rete ha meno neuroni dei concetti che deve
rappresentare, e allora fa come chi ha poche scatole e troppa roba: mette più
concetti nella stessa scatola. Il risultato è che un singolo neurone si accende
per cose scollegate fra loro, un po’ per i gatti, un po’ per le automobili, un
po’ per il colore verde, e diventa illeggibile.

Il nome tecnico di questa faccenda è **sovrapposizione**, e viene dal modo in
cui la si disegna: non come scatole, ma come frecce su un foglio. Immagina un
foglio in cui **ogni asse è un neurone**: il primo asse è quanto si accende il
neurone 1, il secondo quanto si accende il neurone 2. Un concetto diventa
allora una freccia, e la sua direzione dice in quale mescolanza dei due neuroni
quel concetto è scritto. Se «gatto» stesse tutto e solo nel neurone 1, la sua
freccia punterebbe dritta lungo il primo asse, in perfetto accordo con quel
neurone e senza niente in comune con l'altro. Quando invece i concetti sono più
dei neuroni, ciascuno deve prendersi una freccia obliqua, che si sovrappone in
parte a quelle degli altri: da lì il nome. È l'immagine di
{numref}`fig-superposizione`.

Una tecnica recente prova a ri-sistemare gli scatoloni. Prende le
**attivazioni** di uno strato (la fotografia di ciò che la rete ha in mente lì
dentro, quei numeri che si erano incontrati parlando di Grad-CAM) e le riscrive
usando **molte più caselle** di quanti erano i neuroni, chiedendo però che a
ogni esempio se ne accendano pochissime. Uno strumento che riscrive qualcosa
tenendone acceso poco si chiama, con un nome che in italiano non si traduce,
*sparse autoencoder*: «sparso» perché accende poco, «autoencoder» perché deve
riscrivere ciò che riceve in modo da poterlo poi ricostruire uguale.

Perché quelle due regole insieme dovrebbero produrre caselle leggibili? Ecco il
ragionamento. Riscrivere tutto potendo accendere pochissime caselle è una
richiesta severa, e il ragionamento sta tutto nel prezzo di una casella
confusa. Una casella non è solo un interruttore: quando si accende, aggiunge
alla ricostruzione un contributo sempre uguale, la sua impronta. Se una casella
si occupasse di tre cose diverse, quella unica impronta dovrebbe andare bene
per tutte e tre, e non può: per rimettere a posto la ricostruzione bisognerebbe
accendere altre caselle di correzione, cioè spendere di più proprio dove il
conto va tenuto basso. Una casella che si occupa di una cosa sola, invece,
quella cosa la ricostruisce da sé. Il posto abbondante serve appunto perché ce
ne sia una per ogni cosa, senza doverle mescolare, e la rarità serve perché
così, su ogni esempio, se ne accendono davvero poche. Questa è la stessa idea
della rete, rovesciata: la rete sovrappone i concetti perché ha poco spazio e
li può sovrapporre proprio perché sono rari; lo sparse autoencoder li separa
dando spazio in abbondanza e sfruttando la stessa rarità. Che poi il risultato
sia davvero una casella per concetto, però, è un'altra questione.

Il campo è giovane e va raccontato per quello che è. La tecnica ha retto la
prova della scala, e si applica ormai a modelli linguistici veri, non a
giocattoli. Quello che non è dimostrato è il punto d'arrivo: che ogni casella
contenga davvero una cosa sola resta un giudizio dato guardandole a campione, e
si sono trovati casi in cui una casella che sembrava pulita non si accende
proprio dove dovrebbe. Nessuno, comunque, ha ancora letto una rete grande per
intero.

`````

`````{tab} Superiore

Il programma dei **circuiti** è stato articolato da Olah e colleghi su
*Distill* nel 2020 {cite}`olah2020zoom`: studiare una rete come un oggetto
scientifico, individuando *feature* (direzioni nello spazio delle attivazioni
che codificano un concetto) e i *circuiti* che le collegano; sottografi di
neuroni e pesi che implementano un calcolo interpretabile, come i rilevatori
di curve nelle prime reti di visione.

L'ostacolo teorico è la **sovrapposizione** (*superposition*), studiata a
fondo da Elhage e colleghi nei *Toy Models of Superposition* (Anthropic, 2022)
{cite}`elhage2022toy`: una rete con $n$ neuroni può rappresentare molte più di
$n$ feature sfruttando direzioni quasi ortogonali in $\mathbb{R}^n$, purché
ciascuna feature sia rara. La
conseguenza pratica è la **polisemanticità**: un singolo neurone risponde a
stimoli non correlati, e diventa illeggibile. Bricken e colleghi, in *Towards
Monosemanticity* (Anthropic, 2023), affrontano il problema con uno **sparse
autoencoder** {cite}`bricken2023monosemanticity`: le attivazioni di uno strato
vengono ricodificate in un dizionario **sovracompleto** (molte più unità dei
neuroni originali) sotto un vincolo di **sparsità**, che spinge poche unità
attive per esempio. Le feature così estratte risultano in buona parte
**leggibili**, e conviene prendere «in buona parte» e «leggibili» per quel che
sono: il giudizio di valutatori umani (o di un modello usato come valutatore)
su un campione di feature, non una proprietà dimostrata.

Il campo è nascente e va preso con l'onestà che si deve alle frontiere. La
frontiera, però, oggi non passa dove si tende a metterla. La prova della
**scala** la tecnica l'ha superata: Templeton e colleghi
{cite}`templeton2024scaling` hanno addestrato autoencoder sparsi fino a decine
di milioni di feature sulle attivazioni interne di un modello linguistico di
produzione, non di un giocattolo di laboratorio. Quella che non ha superato è
la prova dell’**affidabilità**: Chanin e colleghi, nel 2024, hanno documentato
modi sistematici in cui un latente apparentemente monosemantico non si accende
proprio dove dovrebbe, perché un latente più specifico ne ha assorbito i casi
(lo chiamano *feature absorption*), e che il fenomeno non si risolve cambiando
la dimensione del dizionario o il grado di sparsità
{cite}`chanin2024absorption`. La sovrapposizione resta una spiegazione teorica
solida del *perché* i neuroni siano illeggibili; che gli sparse autoencoder
siano *la* cura è ancora un programma di ricerca, non un risultato acquisito. E
nessuno, in ogni caso, ha ancora «letto» un modello di grande scala per intero.
La posta in gioco, però, è alta.

`````

```{figure} ../figures/toy-models-superposition.svg
:name: fig-superposizione
:alt: "Due piani a due dimensioni. Nel primo, con feature dense, i due assi interni ospitano due sole feature, ad angolo retto fra loro, una per direzione. Nel secondo, con feature sparse, gli stessi due assi ospitano cinque feature disposte a raggiera: non sono ad angolo retto, si sovrappongono, ma poiché raramente sono attive insieme il modello riesce comunque a distinguerle."
:width: 92%

La sovrapposizione, disegnata su un foglio. Ogni freccia è un concetto, e la
sua direzione dice in quale mescolanza di neuroni quel concetto è scritto. A
sinistra i concetti sono due quanti i neuroni, e ognuno si prende una direzione
tutta sua, perpendicolare all'altra: leggere quel neurone vuol dire leggere quel
concetto. A destra i concetti sono cinque e i neuroni sempre due, e allora le
frecce si dispongono a raggiera, oblique, ciascuna un po’ addosso alle altre. Il
prezzo è che nessuna ha più una direzione pulita, e infatti nessun neurone,
guardato da solo, corrisponde più a un concetto.
```

Come faccia la rete a cavarsela lo stesso, con cinque frecce e due soli assi,
lo dice la condizione che tiene in piedi tutto: che ciascun concetto si
accenda di rado, e quasi mai insieme agli altri. Conviene vedere perché, con
dei numeri inventati. Se è acceso il solo concetto A, i due neuroni segnano
$0{,}9$ e $0{,}4$, e quella coppia di numeri appartiene ad A e a nessun altro:
il concetto si riconosce. Ma se A e B si accendono insieme, i loro contributi
si sommano, e i due neuroni possono segnare $1{,}2$ e $1{,}1$, che è per
esempio esattamente quello che segnerebbe il concetto C da solo. Chi legge non
ha modo di distinguere i due casi. Quando le frecce sono ad angolo retto
questo non succede, perché ciascuna muove un asse e lascia fermo l'altro;
quando sono oblique succede, e l'unica difesa è che capiti di rado. Ecco
perché smontare una rete è difficile più del previsto: la speranza naturale,
un neurone un concetto, è vera solo nella metà sinistra della figura, e le
reti vere stanno nella metà destra.

```{figure} ../figures/interpretabilita-scatola-nera.svg
:name: fig-sparse-autoencoder
:alt: "A sinistra quattro neuroni disegnati come cerchi, n1, n2, n3 e n4, da cui partono linee che si incrociano: ogni neurone tiene dentro pezzi di concetti diversi. Le linee entrano tutte in un riquadro al centro, lo sparse autoencoder. A destra ne escono quattro caselle separate, ciascuna col nome di un concetto leggibile: ponte Golden Gate, sintassi Python, tono adulatorio, sequenze di DNA. In basso la scritta «un groviglio di neuroni entra, migliaia di feature nitide escono»."
:width: 96%

La mossa che scioglie il groviglio. A sinistra i neuroni, con le linee che si
incrociano perché ciascuno tiene dentro pezzi di cose diverse; a destra quello
che ne esce, una casella per concetto leggibile. Il disegno ne mette quattro e
quattro per stare in pagina, ma il numero è proprio il punto, ed è scritto in
basso: le caselle in uscita sono **migliaia**, molte di più dei neuroni in
entrata. La seconda regola, che a ogni esempio se ne accendano pochissime, il
disegno non può mostrarla, e va tenuta a mente lo stesso: senza di essa il posto
in più non servirebbe a niente.
```

Che in {numref}`fig-sparse-autoencoder` il posto si faccia **più largo**,
invece di stringerlo, può sembrare il contrario di quello che ci si aspetta: di
solito, per capire meglio, si riassume. Ma qui il problema di partenza era
proprio l'opposto, troppe cose in troppo poco spazio, come nella metà destra di
{numref}`fig-superposizione`. Non si sta riassumendo: si sta **disfando** una
sovrapposizione, e per disfarla serve appunto lo spazio che alla rete mancava.

Perché tutto questo conta, e non è solo un esercizio di curiosità? Per la
**sicurezza**. Un modello linguistico di grandi dimensioni può imparare
comportamenti che non vogliamo (raccontare frottole, prendere scorciatoie,
trattare in modo diverso persone che andrebbero trattate uguale) senza che
nulla, dall'esterno, lo tradisca. Poter leggere i circuiti interni significherebbe
accorgersene *prima* che si manifestino: è il ponte, che riprenderemo nel
capitolo sull'AI responsabile, tra l'interpretabilità come curiosità
scientifica e l'interpretabilità come strumento di controllo.

Il quadro concettuale finisce qui. Restano i due conti veri, gli Integrated
Gradients e Grad-CAM, rifatti con i numeri e con poche righe di codice.

## Integrated Gradients coi numeri: un esempio eseguibile

Vale più di mille formule vedere la completezza tornare al centesimo. La
funzione che segue non è quella disegnata nella figura di poco fa, è un secondo
esempio, costruito per lo stesso scopo ma con due variabili invece di una, così
che si possano guardare due attribuzioni separate. Prendiamo dunque una funzione
giocattolo di due variabili costruita apposta per **saturare**,
cioè per riprodurre il caso in cui il gradiente locale mente. È la solita somma
pesata di un neurone (due volte la prima variabile, meno una volta la seconda)
passata dentro la **tangente iperbolica** $\tanh$, una funzione che schiaccia:
sale ripida vicino allo zero e si appiattisce man mano che l'uscita si avvicina
a 1, come la sigmoide di poco fa. In formule,
$f(\mathbf{x}) = \tanh(\mathbf{w}^\top \mathbf{x})$ con
$\mathbf{w} = (2, -1)$: la scrittura $\mathbf{w}^\top \mathbf{x}$ è il modo
compatto di dire «moltiplica ogni peso per la variabile che gli corrisponde e
somma tutto», cioè appunto la somma
pesata. Nel punto $\mathbf{x} = (2, 1)$ essa vale $2 \cdot 2 - 1 \cdot 1 = 3$, e
$\tanh(3) \approx 0{,}995$: siamo sulla parte piatta della curva, dove la
pendenza è quasi nulla. Un solo gradiente direbbe «qui non conta niente»; gli
Integrated Gradients, integrando dalla baseline tutta a zero, recuperano
l'intero contributo. Nel codice, il commento «regola della catena» segnala il
modo standard di derivare una funzione dentro un'altra: si deriva prima quella
esterna, la $\tanh$, poi quella interna, la somma pesata, e si moltiplicano i
due risultati.

```python
import numpy as np

# funzione giocattolo che satura: f(x) = tanh(w . x)
w = np.array([2.0, -1.0])

def f(x):
    return np.tanh(w @ x)

def grad_f(x):
    z = w @ x
    return (1.0 - np.tanh(z) ** 2) * w   # regola della catena

x = np.array([2.0, 1.0])     # input da spiegare
baseline = np.zeros(2)        # baseline neutra (lo "zero")

# gradiente grezzo nel solo punto x: saturo, quasi nullo -> saliency cieca
print("gradiente in x :", np.round(grad_f(x), 4))       # [ 0.0197 -0.0099]

# Integrated Gradients: media dei gradienti lungo il cammino baseline -> x
m = 200
alphas = (np.arange(1, m + 1) - 0.5) / m   # punti medi delle m tappe
grad_medio = np.zeros(2)
for a in alphas:
    grad_medio += grad_f(baseline + a * (x - baseline))
grad_medio /= m
ig = (x - baseline) * grad_medio
print("attribuzioni IG:", np.round(ig, 4))              # [ 1.3267 -0.3317]

# assioma di completezza: la somma delle attribuzioni = f(x) - f(baseline)
print("somma IG       :", round(ig.sum(), 4))           # 0.9951
print("f(x) - f(base) :", round(f(x) - f(baseline), 4)) # 0.9951
```

Il gradiente nel punto è $(0{,}0197,\, -0{,}0099)$: minuscolo, come previsto per
un neurone saturo. Ma le attribuzioni integrate valgono $(1{,}327,\, -0{,}332)$
e la loro **somma**, $0{,}9951$, coincide al quarto decimale con
$f(x) - f(\text{baseline}) = 0{,}9951$: la completezza è verificata. Notate anche
il segno: la prima variabile ($w_1 = 2 > 0$) spinge il punteggio in alto, la
seconda ($w_2 = -1 < 0$) lo tira giù, esattamente come ci si aspetta.

## Uno sketch di Grad-CAM in PyTorch

Su una rete vera, Grad-CAM si costruisce agganciando due *hook* allo stadio
convoluzionale finale: uno cattura le attivazioni in avanti, l'altro i
gradienti all'indietro. In una ResNet il punto giusto è l’**uscita dell'ultimo
blocco** di `layer4`, dopo la somma residuale: è lì che agganciano le
implementazioni di riferimento, perché fermarsi a una convoluzione interna al
blocco ignorerebbe il contributo della scorciatoia. Ecco lo scheletro su una
ResNet-18 di `torchvision`, con l'API reale.

```python
import torch
import torch.nn.functional as F
from torchvision import models

model = models.resnet18(weights=models.ResNet18_Weights.IMAGENET1K_V1).eval()
target = model.layer4[-1]                # ultimo blocco: uscita post-residuo

att, grad = {}, {}
target.register_forward_hook(lambda m, i, o: att.__setitem__("v", o.detach()))
target.register_full_backward_hook(
    lambda m, gi, go: grad.__setitem__("v", go[0].detach())
)

x = torch.randn(1, 3, 224, 224)          # immagine gia pre-processata
logit = model(x)                          # (1, 1000)
classe = logit.argmax(dim=1)              # classe predetta
model.zero_grad()
logit[0, classe].backward()               # gradiente della sola classe scelta

A = att["v"]                              # attivazioni  (1, C, h, w)
dY = grad["v"]                            # gradienti    (1, C, h, w)
alpha = dY.mean(dim=(2, 3), keepdim=True)  # peso per canale (global avg pool)
heatmap = F.relu((alpha * A).sum(dim=1))   # (1, h, w), solo contributi positivi
heatmap = heatmap / (heatmap.max() + 1e-8) # normalizzata in [0, 1]
# heatmap va poi sovracampionata a 224x224 e sovrapposta all'immagine

print("attivazioni:", tuple(A.shape))
print("gradienti  :", tuple(dY.shape))
print("heatmap    :", tuple(heatmap.shape))
```

Le tre righe stampate dicono che cosa è uscito:

```text
attivazioni: (1, 512, 7, 7)
gradienti  : (1, 512, 7, 7)
heatmap    : (1, 7, 7)
```

cioè 512 mappe di attivazione da $7 \times 7$ ciascuna, altrettanti gradienti,
e una sola heatmap $7 \times 7$ che le riassume. Quel $7 \times 7$ è la
risoluzione a cui la rete è arrivata dopo aver rimpicciolito più volte
l'immagine di partenza, che era $224 \times 224$: è per questo che una heatmap
Grad-CAM è per forza grossolana, e va poi ingrandita per essere sovrapposta alla
foto.

Il cuore del calcolo è tutto nelle ultime tre righe. `alpha` è il peso di
ciascuna delle 512 mappe, cioè il suo gradiente mediato su tutte le posizioni;
la riga dopo somma le mappe con quei pesi e con `relu` butta via i contributi
negativi, tenendo solo le zone che *sostengono* la risposta; l'ultima porta i
valori fra $0$ e $1$ per poterli disegnare (il $10^{-8}$ evita una divisione per
zero nel caso, raro ma possibile su una classe non predetta, in cui `relu`
azzeri tutta la mappa).
Su un'immagine di cane la macchia calda cadrebbe sul muso; su
un husky delle venti foto truccate, sulla neve, ed è precisamente questo che
volevamo poter vedere.

`````{tab} Elementare

```{admonition} Da ricordare
:class: important
- Una rete profonda non ha numeri leggibili come i coefficienti di un modello
  lineare. Per capirla si cambia domanda: quanto ha pesato *questo* pezzo
  dell'ingresso su *questa* decisione? Si misura di quanto cambierebbe la
  risposta toccandolo appena.
- Le mappe di **salienza** (Simonyan e colleghi, 2014) danno questi colpetti
  pixel per pixel: informative, ma rumorose (due pixel vicini possono rispondere
  in modo molto diverso) e valide solo attorno a quella foto lì, perché la
  pendenza è misurata in quel punto e non altrove.
  **Grad-CAM** (Selvaraju e colleghi, 2017) accende invece i «faretti»
  dell'ultimo strato convoluzionale in proporzione a quanto contano per la
  classe: più grossolana, ma pulita e affidabile nel dire *dove* la rete ha
  guardato.
- Gli **Integrated Gradients** (Sundararajan e colleghi, 2017) partono da
  un'immagine neutra e arrivano a quella vera per tappe, registrando la
  reazione lungo tutta la salita: così vedono anche i contributi che
  all'arrivo, ormai satura, la rete non segnala più. La somma delle quote fa
  esattamente il salto di fiducia fra le due immagini, come un conto di cena
  che torna al centesimo. Con un'avvertenza: il conto torna **da qualunque
  punto si parta**, quindi che torni non dimostra che il punto di partenza sia
  quello giusto; e di ciò che era già nero in partenza non si misura nessun
  contributo, nemmeno se era la ragione della decisione.
- Una mappa dice **dove**, non **che cosa** (Cynthia Rudin): sapere quale zona
  la rete guarda non dice che cosa ci trovi. E i **controlli di sanità**
  (Adebayo e colleghi, 2018) mostrano che, cancellando ciò che il modello ha
  imparato, per parecchi metodi la mappa resta quasi identica: descriveva
  l'immagine, non la rete. I colpetti pixel per pixel e Grad-CAM il test lo
  superano; a fallirlo sono due varianti più elaborate, *Guided BackProp* e
  *Guided Grad-CAM*; e gli Integrated Gradients stanno in mezzo, perché la mappa
  cambia ma la sagoma della foto resta lì a farla sembrare sensata. Una
  spiegazione che sembra sensata non è per questo fedele.
- Le **quote di attenzione** dei Transformer sono un indizio, non una prova:
  quote molto diverse possono portare alla stessa risposta. E guardare un solo strato non basta,
  perché una parte dell'informazione salta l'attenzione e prende la scorciatoia
  verso il piano di sopra; **attention rollout** e *attention flow* rifanno il
  conto lungo tutta la pila. Il **probing** risponde a un'altra domanda: a
  quale piano è scritta una certa informazione.
- L’**interpretabilità meccanicistica** apre la scatola e prova a ricostruire i
  circuiti con cui la rete calcola, sciogliendo la **sovrapposizione** (troppi
  concetti nella stessa scatola, un neurone che si accende per cose scollegate)
  con gli *sparse autoencoder*, che riscrivono le attivazioni su molte più
  caselle chiedendo di accenderne pochissime per volta. La tecnica si applica ormai a modelli veri e
  grandi; che le caselle che ne escono contengano davvero **una cosa sola**
  resta però un giudizio dato guardandole, non una cosa dimostrata. Campo
  giovane, ma centrale per la sicurezza dei modelli grandi.
```

`````

`````{tab} Superiore

```{admonition} Da ricordare
:class: important
- Le reti profonde non hanno coefficienti leggibili: l’**attribuzione** usa il
  **gradiente dell'uscita rispetto all'ingresso** per stimare quanto ogni parte
  dell'input ha pesato su una singola decisione.
- Le **saliency maps** {cite}`simonyan2014deep` sono il gradiente sui pixel:
  informative ma rumorose e locali. **Grad-CAM** {cite}`selvaraju2017grad` pesa
  le mappe dell'ultimo strato convoluzionale coi gradienti della classe e
  localizza in modo robusto *dove* guarda la CNN.
- Gli **Integrated Gradients** {cite}`sundararajan2017axiomatic` integrano il
  gradiente lungo il cammino dalla baseline all'input: fondati su assiomi
  (sensibilità **a input e baseline che differiscono in una sola componente**,
  invarianza all'implementazione), risolvono la saturazione e
  soddisfano la **completezza**,
  $\sum_i \mathrm{IG}_i = f(\mathbf{x}) - f(\mathbf{x}')$. La completezza vale
  però per **qualunque** baseline: $\mathbf{x}'$ è il grado di libertà che gli
  assiomi non vincolano, e ogni componente uguale alla baseline riceve
  attribuzione nulla per costruzione {cite}`sturmfels2020baselines`.
- Una mappa dice **dove**, non **che cosa** {cite}`rudin2019stop`, e i
  **controlli di sanità** {cite}`adebayo2018sanity` mostrano che per diversi
  metodi popolari la mappa cambia pochissimo randomizzando i pesi del modello:
  descriveva l'immagine, non la rete. Nel paper i bocciati sono **Guided
  BackProp** e **Guided Grad-CAM**; gradiente semplice e Grad-CAM passano; i
  metodi che moltiplicano per l'ingresso (gradient $\odot$ input e gli
  **Integrated Gradients**) cambiano, anche di segno, ma conservano visibile la
  struttura dell'ingresso. La plausibilità visiva di una spiegazione
  non è una prova della sua fedeltà.
- I **pesi di attenzione** non sono di per sé una spiegazione affidabile
  (dibattito *«Attention is not Explanation»*, {cite}`jain2019attention` e
  {cite}`wiegreffe2019attention`, con l'avvertenza che quegli esperimenti sono
  su un singolo strato di attenzione sopra una BiLSTM, non su un Transformer).
  E prima ancora c'è un
  problema tecnico: comporre gli strati richiede di tener conto delle
  connessioni residuali, che è ciò che fanno **attention rollout** e
  *attention flow* {cite}`abnar2020quantifying`. Il **probing** con
  classificatori lineari {cite}`alain2017understanding` mappa invece dove sta
  l'informazione negli strati interni, con i *control task* di Hewitt e Liang
  {cite}`hewitt2019control` a misurare che stia leggendo e non risolvendo.
- L’**interpretabilità meccanicistica** (circuiti {cite}`olah2020zoom` e
  sparse autoencoder {cite}`bricken2023monosemanticity`) punta a fare
  reverse-engineering dei calcoli interni. La scala non è più il limite
  {cite}`templeton2024scaling`; la **monosemanticità** delle feature estratte
  sì, ed è tuttora contesa. Campo giovane, ma centrale per la sicurezza degli
  LLM.
```

`````

Una spiegazione convincente non è per questo vera, ed è la cosa più scomoda di
questo capitolo: una spiegazione va messa alla prova esattamente come si mette
alla prova un modello. Nel capitolo sull'AI responsabile la stessa pretesa si
sposta sulle conseguenze, perché quando un sistema decide di una persona sapere
che cosa guarda non basta, bisogna stabilire se sia giusto e chi risponde
quando sbaglia.
