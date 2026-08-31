# Object detection e segmentazione

Un'auto a guida autonoma si avvicina a un incrocio. Una rete di
classificazione, di quelle viste finora, sa dirle una cosa sola:
*nell'immagine c'è un pedone*. Vero, ma inutile. Per frenare in tempo l'auto
deve sapere **dove** si trova quel pedone, se è uno o sono tre, se quello a
destra è un ciclista, e (al limite) quale sagoma esatta occupa sull'asfalto.
La classificazione risponde alla domanda «cosa»; qui impariamo a rispondere
anche a «dove» e «quali contorni».

## Dalla classificazione al riquadro

Salire dalla classificazione alla localizzazione è come passare da «in questa
foto c'è un gatto» a «il gatto sta in *quel* rettangolo». Il compito si chiama
**object detection**: per ogni oggetto presente, la rete deve produrre insieme
un riquadro e un'etichetta.

```{figure} ../figures/classification-detection-segmentation.svg
:name: fig-tre-uscite
:alt: "La stessa scena, un cane e una palla su un prato, trattata da tre compiti diversi. Nella classificazione l'uscita è una sola etichetta per l'intera immagine, «cane». Nella detection sono due riquadri, uno attorno al cane e uno attorno alla palla, ciascuno con la sua etichetta e la sua confidenza. Nella segmentazione ogni pixel prende un colore secondo la categoria a cui appartiene, sfondo compreso: il cane in terracotta, la palla in ocra, il prato in teal."
:width: 100%

Stessa foto, tre uscite. Salendo da sinistra a destra cresce la precisione
della risposta e, con essa, il costo di annotare i dati per addestrarla.
```

La progressione di {numref}`fig-tre-uscite` va letta anche al contrario, cioè
dal lato dei dati. Un'etichetta per foto la scrive chiunque in un secondo; un
riquadro richiede di trascinare il mouse; una maschera pixel per pixel costa
minuti a immagine. È spesso questo, e non l'architettura, a decidere quale dei
tre compiti si può davvero affrontare.

`````{tab} Elementare

Prendi un pennarello, cerchia in una foto ogni oggetto interessante e scrivigli
accanto un nome: "auto", "cane", "semaforo". Ogni cerchio è in realtà un
rettangolo (lo chiamiamo **bounding box**, la "cornice") e ogni nome è la
classe. Un rilevatore fa esattamente questo, ma in automatico e per molti
oggetti insieme. Per ciascuno indovina tre cose alla volta: *dove* tracciare la
cornice, *cosa* c'è dentro e *quanto ci crede*, cioè un numero fra 0 e 1 con
cui dichiara la sua sicurezza che lì dentro qualcosa ci sia davvero.

Il terzo numero serve perché nessuno gli ha detto quante cornici disegnare. In
una foto di strada gli oggetti possono essere due o quaranta, e prima di
guardare non lo sa. Allora tiene pronte tantissime cornici sparse su tutta la
foto e le riempie tutte, sapendo che per la stragrande maggioranza la risposta
giusta è "qui non c'è niente".

Chi lo corregge, mentre impara, somma tre penalità: la cornice storta, il nome
sbagliato, la sicurezza fuori posto. E non pesano uguale, perché il rapporto si
sceglie prima di cominciare: in uno dei primi rilevatori era di dieci a uno fra
una cornice storta attorno a un oggetto vero e un pizzico di sicurezza
dichiarato su un pezzo di asfalto vuoto. I pezzi di asfalto vuoto sono
migliaia, e se contassero quanto gli altri il rilevatore imparerebbe la
scorciatoia più comoda del mondo, cioè rispondere "niente" dappertutto e avere
quasi sempre ragione.

`````

`````{tab} Superiore

Per ogni oggetto la rete predice un vettore
$(x, y, w, h, c, p_{\text{obj}})$: le coordinate del centro e le dimensioni del
riquadro, la classe $c$ e una confidenza $p_{\text{obj}} \in [0,1]$ che stima
se nel riquadro c'è davvero un oggetto. L'addestramento minimizza una loss
composita che somma un
termine di **localizzazione** (errore sulle coordinate, tipicamente
*smooth L1* o una IoU-loss), un termine di **classificazione** (cross-entropy
sulla classe) e un termine di **objectness** che supervisiona la confidenza,
spingendola verso l'alto dove un oggetto c'è e verso zero sullo sfondo (in
YOLOv1, e ancora in YOLOv2, il bersaglio non è uno ma la IoU stessa fra
riquadro predetto e riquadro vero, così che il punteggio incorpori già la
qualità della localizzazione; da YOLOv3 in poi, semplicemente uno):

$$
\mathcal{L} = \mathcal{L}_{\text{obj}} + \mathcal{L}_{\text{cls}} + \lambda \,\mathcal{L}_{\text{box}} .
$$

Il coefficiente $\lambda$ bilancia localizzazione e riconoscimento; YOLO ne usa
in realtà due, e il secondo ($\lambda_{\text{noobj}} = 0{,}5$ contro
$\lambda_{\text{coord}} = 5$) serve proprio a smorzare il contributo delle
moltissime celle vuote, cioè è un primo rimedio a quello squilibrio
oggetto/sfondo su cui torneremo con la *focal loss*. Questa è
la parametrizzazione di YOLO; nella famiglia Faster R-CNN il termine di
objectness non sparisce, si sposta. Nel **primo** stadio, la *Region Proposal
Network* di cui parla la prossima sezione, è proprio una objectness: ogni ancora
viene classificata in binario come oggetto o sfondo, e il termine di regressione
è attivo solo sulle ancore positive. Manca invece dalla testa del **secondo**
stadio, dove la confidenza coincide con il punteggio softmax della classe perché
lì lo sfondo è trattato come una classe in più. Un'immagine può contenere un
numero variabile di oggetti: gestire questa cardinalità ignota è il vero nodo
architetturale della detection.

`````

## Due stadi contro uno stadio

Storicamente i rilevatori si dividono in due famiglie, e la differenza è un
classico compromesso tra **accuratezza e velocità**. Semplificando: gli uni
guardano l'immagine due volte, prima per capire dove conviene guardare e poi
per guardarci davvero; gli altri una volta sola.

```{figure} ../figures/yolo-2016.svg
:name: fig-yolo
:alt: "Un'immagine coperta da una griglia sette per sette entra una sola volta in una rete convoluzionale, che produce direttamente i riquadri degli oggetti con le rispettive classi. Non c'è nessuna fase separata di proposta delle regioni: ogni cella della griglia è responsabile degli oggetti il cui centro le cade dentro."
:width: 92%

L'approccio a stadio singolo, cioè una sola passata sull'immagine. La griglia
è una divisione di responsabilità decisa in anticipo, non una ricerca.
```

Guardiamo la seconda famiglia, quella della passata unica, perché
{numref}`fig-yolo` rende evidente cosa si guadagna e cosa si perde. Guardare
l'immagine una volta sola è quello che rende possibile il tempo reale; il prezzo
lo si legge nella griglia disegnata sopra la foto. Quella griglia è una
divisione del lavoro decisa prima di guardare, in cui ogni
casella (una **cella**) si prende la responsabilità degli oggetti che le cadono
dentro. E siccome a ogni cella si concede in partenza un numero fisso di
riquadri, di solito due, oggetti piccoli e ammassati nella stessa cella se li
contendono: il terzo passerotto dello stormo non ha una cornice a
disposizione.

`````{tab} Elementare

I rilevatori **a due stadi** lavorano come un revisore scrupoloso: prima
propongono un po’ di zone "sospette" dove *potrebbe* esserci qualcosa, poi
guardano con calma dentro ognuna per decidere cosa sia e correggere la cornice.
Più lenti, e per anni i più precisi.

I rilevatori **a uno stadio** fanno tutto in un colpo solo: un'unica passata
sull'immagine sputa fuori direttamente cornici ed etichette. A lungo sono stati
meno precisi sui casi difficili, ma abbastanza rapidi da lavorare in tempo
reale su un video, ed è per questo che si chiama YOLO, *You Only Look Once*.

La ragione di quella minore precisione sta in un conto. Chi guarda una volta
sola deve dare una risposta per ogni casella dell'immagine, e le caselle
con dentro un oggetto sono una manciata contro decine di migliaia di asfalto,
cielo e muro. Alla correzione arrivano così diecimila risposte quasi tutte
uguali e quasi tutte facili: sommate, seppelliscono le poche difficili, e il
rilevatore impara benissimo a dire "niente" e molto peggio tutto il resto.
Abbassare in blocco il peso di tutte le risposte vuote, come si faceva
all'inizio, aiuta e non basta: fra quelle vuote ce ne sono migliaia di ovvie e
qualcuna insidiosa, e un peso unico le tratta allo stesso modo. Il rimedio,
trovato nel 2017, cambia il modo di correggere invece dell'architettura: una
risposta facile, di quelle su cui il rilevatore ha già ragione ed è pure sicuro,
conta quasi zero, e a decidere la lezione restano i pochi casi su cui sta
ancora sbagliando. Da lì lo svantaggio in precisione si è in gran parte chiuso,
e fra le due famiglie è rimasta soprattutto la differenza di velocità.

`````

`````{tab} Superiore

La famiglia a **due stadi** nasce con R-CNN {cite}`girshick2014rich` e matura
con Faster R-CNN {cite}`ren2015faster`, che introduce la *Region Proposal
Network*: uno
stadio propone regioni candidate, il secondo le classifica e ne raffina i
riquadri. Accuratezza elevata, ma latenza maggiore.

La famiglia a **uno stadio** (YOLO {cite}`redmon2016you` e SSD
{cite}`liu2016ssd`) elimina la fase di proposta: una sola rete convoluzionale
predice simultaneamente riquadri e classi su una griglia dell'immagine. Il
prezzo storico è stato lo squilibrio tra i pochi riquadri con oggetto e i
moltissimi di sfondo; la *focal loss* di RetinaNet {cite}`lin2017focal` lo ha
in gran parte sanato, avvicinando le due famiglie in accuratezza.

`````

## Le ancore: non partire da zero

Le due famiglie condividono un problema, e la soluzione. Nello stesso punto
dell'immagine possono trovarsi oggetti dalle forme opposte: un pedone alto e
stretto, un'auto bassa e larga, un pallone quasi quadrato. Chiedere alla rete
di disegnare il riquadro giusto partendo dal nulla è chiederle molto. La
scorciatoia si chiama **anchor box**, l’"ancora": un riquadro di partenza già
pronto, da correggere invece che da inventare.

```{figure} ../figures/anchor-boxes.svg
:name: fig-anchor-boxes
:alt: Griglia sovrapposta a un'immagine con un pedone stilizzato; su un punto della griglia sono centrate tre ancore, una alta e stretta in terracotta che ricalca il pedone, una bassa e larga in teal e una quadrata in ocra, entrambe tratteggiate.
:width: 95%

In ogni punto della griglia la rete parte da più ancore di forma diversa: per
il pedone, quella alta e stretta (terracotta) è già quasi giusta.
```

`````{tab} Elementare

Un corniciaio non costruisce una cornice su misura per ogni quadro che entra in
bottega: tiene pronti alcuni formati standard (verticale per i ritratti,
orizzontale per i paesaggi, quadrato) e poi ritocca quello che ci va più
vicino. Le ancore funzionano allo stesso modo. In ogni punto dell'immagine la
rete ha a disposizione qualche cornice predefinita, di taglie e proporzioni
diverse ({numref}`fig-anchor-boxes`). Davanti a un pedone non deve inventarsi
il rettangolo: prende la cornice verticale, che gli somiglia già, e la aggiusta.
Correggere una cornice quasi giusta è molto più facile che disegnarne una dal
nulla, e la rete impara proprio questo: piccole correzioni, più il nome di ciò
che c'è dentro.

Le correzioni si dicono in frazioni della cornice stessa, mai in centimetri:
"spostati a destra di un decimo della tua larghezza, allungati di un quinto".
Detta così, la stessa istruzione va bene per la cornicetta del passerotto in
fondo al prato e per quella del camion in primo piano. Detta in centimetri, tre
centimetri manderebbero la cornice del passerotto lontano dal passerotto e
quella del camion appena appena, e la rete dovrebbe imparare un ritocco diverso
per ogni taglia.

Il confronto con il quadro, però, il corniciaio può farlo solo in bottega, sui
quadri di prova di cui conosce già la cornice giusta: è lì che impara quale
formato scegliere e di quanto ritoccarlo. Davanti a un quadro nuovo la cornice
giusta non la conosce nessuno, e infatti la rete ritocca tutte le sue cornici,
dicendo per ciascuna quanto è sicura che lì dentro ci sia qualcosa: alla fine
restano solo le più convinte.

`````

`````{tab} Superiore

Su ogni cella della mappa di feature si centrano $k$ riquadri predefiniti (le
ancore) a più **scale** e **proporzioni**. La rete non predice coordinate
assolute: per ciascuna ancora produce i punteggi di classe e quattro
**offset** che la deformano verso il riquadro vero,

$$
t_x = \frac{x - x_a}{w_a}, \qquad
t_y = \frac{y - y_a}{h_a}, \qquad
t_w = \log\frac{w}{w_a}, \qquad
t_h = \log\frac{h}{h_a},
$$

dove $(x_a, y_a, w_a, h_a)$ sono centro e dimensioni dell'ancora e
$(x, y, w, h)$ quelli del riquadro da raggiungere; in addestramento ogni
oggetto è assegnato alle ancore che meglio lo ricoprono (la sovrapposizione
si misura con la IoU, protagonista della prossima sezione). Quel confronto
appartiene al solo addestramento, quando i riquadri veri ci sono: le ancore
che ricoprono bene un oggetto diventano positive e imparano classe e offset
verso di lui, le altre fanno da sfondo. In inferenza non c'è nessun riquadro
vero da ricoprire: la rete produce punteggi e offset per tutte le ancore, ogni
ancora corretta diventa un candidato, e la IoU ricompare solo alla fine,
misurata fra i candidati stessi, per sfoltire i doppioni sullo stesso oggetto.
È il meccanismo
della *Region Proposal Network* di Faster R-CNN {cite}`ren2015faster`, che
usa $k=9$ ancore per posizione (3 scale × 3 proporzioni), e di SSD
{cite}`liu2016ssd`, che le chiama *default boxes* e le distribuisce su mappe
di feature a più risoluzioni, per coprire oggetti piccoli e grandi. Esistono
anche rilevatori *anchor-free*, che predicono direttamente centri e distanze
dai bordi senza riquadri di partenza; ma le ancore restano il modo più chiaro
per capire come una griglia fissa possa produrre riquadri di ogni forma.

`````

## Quanto è buona una predizione? IoU e mAP

Un riquadro predetto non è mai esattamente sovrapposto a quello vero. Serve una
misura numerica di "quanto ci ha preso". Quella misura è l’**Intersection over
Union**.

```{figure} ../figures/iou.svg
:name: fig-iou
:alt: Due riquadri sovrapposti, reale in teal e predetto in terracotta; a sinistra evidenziata in ocra l'area di intersezione, a destra in tinta teal l'area di unione.
:width: 90%

L'Intersection over Union confronta il riquadro predetto (terracotta) con quello
reale (teal): è l'area di **intersezione** divisa per l'area di **unione**.
```

`````{tab} Elementare

Chi corregge appoggia sulla foto del cane un foglio trasparente con la cornice
giusta, e lo confronta con la cornice che il rilevatore ha tracciato. Non
combaciano mai. La foto è stampata su carta a quadretti, e allora la somiglianza
si misura contando: i quadretti coperti da tutte e due, divisi per quelli
coperti da almeno una ({numref}`fig-iou`). Trenta in comune, sessanta in tutto:
$30/60 = 0{,}5$. Il conto si chiama IoU e dà sempre un numero fra 0 e 1, dove 1
vuol dire cornici sovrapposte quadretto per quadretto e 0 cornici che non si
toccano nemmeno.

Il correttore deve stabilire a che punto una cornice vale come buona. La regola
solita è la metà: da 0,5 in su passa. Quella metà l'ha fissata una convenzione,
e sposta i verdetti. Portandola a 0,7 o a 0,9 la classifica fra due rilevatori
può ribaltarsi, perché chi azzecca sempre il nome ma disegna cornici
approssimative crolla molto più in fretta di chi le disegna precise. Nelle gare
serie il correttore rifà quindi il conto con dieci soglie, da 0,5 a 0,95, e fa
la media dei dieci verdetti: nessuno si presenta col metro tagliato su misura.

Nella foto accanto i cani sono tre, e il rilevatore consegna cinque cornici in
fila, dalla più sicura alla meno sicura. Il correttore le prende in
quest'ordine. Una cornice passa se copre abbastanza un cane che nessuna cornice
precedente si è già presa, e da quel momento quel cane è suo: una seconda
cornice sullo stesso animale, per quanto ben piazzata, va nella pila degli
errori. Dopo ogni cornice giusta il correttore si ferma e conta quante ne ha
viste giuste su quante ne ha guardate. Quella frazione è la **precisione**.

Sono giuste la prima e le ultime due, quindi le fermate sono tre: 1 su 1, cioè
1; poi 2 su 4, cioè 0,50; poi 3 su 5, cioè 0,60. I tre numeri ballano, perché
ogni cornice sbagliata li tira giù e la giusta che viene dopo li rialza, su e
giù come i denti di una sega. Prima di sommarli il correttore li appiattisce, e
la regola sta in una riga: al posto del numero di quel momento si prende il più
alto fra quello e tutti quelli che vengono dopo. I tre diventano 1, 0,60 e
0,60. Un altro correttore, in un'altra stanza, arriva agli stessi tre, ed è per
questo che si appiattisce.

La somma fa 2,20, e si divide per tre, cioè per i cani che c'erano davvero, non
per le cinque cornici disegnate: 0,73. Il cane che nessuna cornice ha cerchiato
non aggiunge niente alla somma e resta comunque nel divisore. Chi disegna una
cornice sola, la più sicura di tutte, tiene la precisione altissima e porta a
casa un voto basso; chi ne disegna mille li cerchia tutti e tre e paga a ogni
fermata gli errori che ha lasciato dietro. Quel 0,73 è il voto sui cani.

Poi si rifà tutto sulle auto, sui semafori, sulle biciclette, e si fa la media
di quei voti. È la seconda media, quella che dà la «m» di *mean* alla **mAP**.
Ne esce un numero fra 0 e 1: più è alto, più spesso il rilevatore azzecca
insieme la cornice e il nome, ed è il numero con cui due rilevatori si
confrontano.

`````

`````{tab} Superiore

Dati il riquadro predetto $A$ e quello reale $B$, la IoU è

$$
\text{IoU} = \frac{|A \cap B|}{|A \cup B|} \in [0,1] .
$$

Fissata una soglia (ad esempio $\text{IoU} \ge 0{,}5$), le predizioni si
scorrono in ordine di confidenza decrescente: una predizione è un **vero
positivo** se supera la soglia con un oggetto reale non ancora assegnato, e
quell'oggetto viene «consumato». Ogni oggetto reale si accoppia cioè a **una
sola** predizione, e i duplicati, per quanto ben sovrapposti, contano come
**falsi positivi**. Da qui si costruisce la curva
*precision–recall* per ciascuna classe: l'area sotto la sua **interpolata**
(l'inviluppo monotono decrescente, campionato a 11 punti di recall nel VOC fino
al 2009, a tutti i cambi di recall dal 2010, a 101 punti in COCO) è l’**Average
Precision** (AP). La curva grezza è a denti di sega e nessuno la integra: è la
scelta dell'interpolazione a rendere i numeri riproducibili fra
implementazioni. La **mean Average Precision** (mAP) ne fa la media sulle
classi. Il benchmark COCO irrigidisce la metrica mediando la mAP su dieci soglie
di IoU, da $0{,}5$ a $0{,}95$ a passi di $0{,}05$: premia i modelli che
localizzano con precisione, non solo che indovinano la classe.

`````

Prima del verdetto, però, serve un passaggio di pulizia. La rete non produce un
riquadro per oggetto: ne produce migliaia. In ogni punto della griglia tiene
pronte le sue cornici di partenza, e i punti della griglia sono a loro volta
migliaia. Il risultato è che attorno a ogni oggetto se ne accumulano decine
quasi identiche, ciascuna con la sua **confidenza**, cioè il numero da 0 a 1
con cui la rete dichiara quanto è sicura che lì dentro un oggetto ci sia
davvero.

La **non-maximum suppression** (alla lettera «soppressione di ciò che non è il
massimo», e in genere la si chiama NMS) le sfoltisce con una regola semplice:
si ordinano i riquadri per confidenza, si tiene il più sicuro e si scartano
tutti quelli che gli si sovrappongono troppo, poi si ripete sui rimasti finché
non c'è più niente da esaminare. È il passo finale, quasi mai disegnato negli
schemi, di praticamente ogni rilevatore delle due famiglie: senza, ogni oggetto
arriverebbe alla valutazione con un grappolo di doppioni, e tutti tranne uno
conterebbero come errori.

C'è però una terza via, che il problema lo toglie invece di risolverlo. Una
famiglia di rilevatori inaugurata nel 2020 da DETR {cite}`carion2020end` (sta
per *detection transformer*) chiede alla rete un numero fisso di risposte, per
esempio cento, e durante l'addestramento le abbina agli oggetti veri **una a
una**: ogni oggetto vero viene assegnato a una sola delle cento risposte, e le
altre novantotto sono premiate per dire «qui non c'è niente». Chi produce un
doppione viene quindi punito mentre impara, non ripulito dopo, e alla fine
dell'addestramento i doppioni non li produce più. Spariscono così sia le
cornici di partenza sia la fase di pulizia. Il prezzo, storicamente, è stato
che l'addestramento impiega molto più tempo a stabilizzarsi su un risultato
buono.

## La famiglia YOLO: le impalcature tolte una alla volta

Dieci anni di rilevamento si possono ripassare seguendo una famiglia sola. Il
primo YOLO {cite}`redmon2016you` aveva la griglia e nient'altro: niente ancore,
una passata sola, riquadri grossolani e gli oggetti piccoli persi quando si
affollano nella stessa cella. Le versioni successive adottano, uno per uno, gli
attrezzi del mestiere. YOLOv2 {cite}`redmon2017yolo9000` porta dentro le
ancore, e invece di disegnarle a mano le ricava dai dati, raggruppando i
riquadri veri del dataset per trovare le forme che ricorrono davvero. YOLOv3
{cite}`redmon2018yolov3` predice su tre griglie a risoluzione diversa, così
anche l'oggetto piccolo trova una griglia abbastanza fitta da vederlo, e al
posto della softmax mette tanti classificatori indipendenti, uno per classe,
perché la stessa figura può essere insieme «persona» e «pedone».

Poi la famiglia cambia natura. Dal 2020 le versioni nuove sono software,
mantenuto dalla società Ultralytics {cite}`jocher2026ultralytics`, e la storia
si legge nei registri delle versioni invece che negli articoli. YOLOv5 arriva
così, libreria PyTorch senza paper; YOLOv8, nel 2023, toglie le ancore,
predicendo direttamente centro e distanze dai bordi come i rilevatori
*anchor-free* incontrati a proposito delle ancore; e YOLO26, all'inizio del
2026, toglie anche la NMS, punendo i doppioni durante l'addestramento con la
stessa idea dell'abbinamento uno a uno di DETR, così quello che la rete produce
è già il risultato finale. Le due impalcature del mestiere, le ancore e la
pulizia dei doppioni, la famiglia le ha prima usate e poi tolte tutte e due;
restano la griglia, la passata unica e il nome.

Provare l'ultima versione costa cinque righe. La libreria si installa con
`pip install ultralytics`, scarica i pesi alla prima esecuzione e porta con sé
una foto di prova: un minibus elettrico e, sul marciapiede, quattro figure, due
intere e due tagliate dai bordi della foto.

```{code-block} python
:class: pt-lento

# pip install ultralytics; alla prima esecuzione scarica 5 MB di pesi.
from ultralytics import YOLO, ASSETS

modello = YOLO("yolo26n.pt")            # "n" come nano, la taglia più piccola
[esito] = modello(ASSETS / "bus.jpg", verbose=False)
for riquadro in esito.boxes:
    nome = esito.names[int(riquadro.cls)]
    print(f"{nome:10s} confidenza {float(riquadro.conf):.2f}")
```

```text
bus        confidenza 0.92
person     confidenza 0.91
person     confidenza 0.91
person     confidenza 0.87
person     confidenza 0.53
```

Cinque oggetti, nessuna pulizia dopo: i riquadri escono così dalla rete. Le due
persone intere valgono 0,91; quella di schiena sul bordo destro 0,87; e la
0,53 è la figura di cui la foto mostra soltanto una spalla, sul bordo sinistro:
mezza persona, mezzo sì. I pesi stanno sul server di chi pubblica il modello, e
se un giorno li riaddestrano le cifre esatte possono cambiare, mentre la
lettura resta la stessa, con la confidenza che cala dove calerebbe la nostra.

## Segmentare: dal riquadro alla sagoma

Il riquadro è comodo ma grossolano: attorno a un pedone c'è sempre un rettangolo
pieno di sfondo. Quando serve il contorno esatto, pixel per pixel, si passa alla
**segmentazione**. Qui vanno distinti due sapori.

Prima però conviene guardare la forma che quasi tutte le reti di segmentazione
hanno preso, e che si chiama **U-Net** perché sullo schema disegna una U. Il
problema da risolvere è questo: per capire *che cosa* c'è in una zona bisogna
allontanarsi dai pixel e guardare largo, mentre per disegnarne il contorno
esatto bisogna starci attaccati. Sono due esigenze opposte, e la U-Net non
sceglie. Prima scende, rimpicciolendo l'immagine e capendo sempre meglio che
cosa c'è; poi risale, tornando alla risoluzione di partenza per dire dove; e a
ogni gradino della risalita si fa ripassare quello che aveva visto allo stesso
gradino durante la discesa. Sono le connessioni orizzontali della figura, le
**skip connection**: senza di quelle, il contorno tornerebbe su sfocato.

```{figure} ../figures/u-net-clessidra-con-skip.svg
:name: fig-unet
:alt: "Schema a forma di U: il braccio discendente di sinistra riduce progressivamente la risoluzione, dall'immagine intera a metà a un quarto; in fondo c'è il collo di bottiglia; il braccio ascendente di destra la recupera per gradi fino alla mappa a piena risoluzione. Fra livelli corrispondenti dei due bracci corrono connessioni orizzontali tratteggiate, le skip connection, che copiano e concatenano."
:width: 88%

La clessidra della U-Net. Scendendo si capisce *cosa* c'è nell'immagine e si
perde *dove*; le connessioni orizzontali riportano il «dove» dal ramo di
discesa a quello di risalita.
```

Le linee tratteggiate di {numref}`fig-unet` sono esattamente quel rammendo: la
rete fa le due cose incompatibili su due rami, e li ricuce a ogni livello.

`````{tab} Elementare

La segmentazione **semantica** colora ogni pixel dell'immagine con la sua
categoria: tutti i pixel di "strada" di un colore, quelli di "cielo" di un
altro, quelli di "persona" di un terzo. Non distingue però i singoli individui:
due pedoni vicini diventano un'unica macchia "persona".

La segmentazione **di istanza** fa un passo in più: separa anche gli individui.
Pedone-1 e pedone-2 ricevono maschere distinte. È come colorare dentro le linee,
ma tenendo ogni personaggio con la sua tinta.

Il modo più diffuso di ottenerla riusa il rilevatore. Prima si cerchia ogni
pedone con la sua cornice, come per il rilevamento; poi, dentro ogni cornice e
soltanto lì, si decide pixel per pixel chi è pedone e chi è marciapiede rimasto
dentro il rettangolo. Le cornici tengono separati gli individui, la
colorazione rifinisce la sagoma, e siccome ogni cornice arriva già con il suo
nome, di ciascun pedone escono insieme cornice, nome e contorno.

`````

`````{tab} Superiore

La segmentazione **semantica** assegna a ogni pixel una classe. La svolta è la
*Fully Convolutional Network* {cite}`long2015fully`, che sostituisce
gli strati densi finali con convoluzioni e *upsampling* per produrre una mappa
di classi a piena risoluzione, e che già introduce le *skip connections*: la
sua seconda metà si intitola «Combining what and where» e fonde la predizione
grossolana con gli strati a stride più fine (sono le varianti FCN-16s e FCN-8s,
che dal modello base si distinguono solo per quante skip hanno). **U-Net**
{cite}`ronneberger2015u`, nata per l'imaging biomedico e dello stesso anno, ne
generalizza la forma: un decoder simmetrico che a ogni livello **concatena**
l'intera mappa di feature dell'encoder, invece di sommare due mappe di
punteggi di classe a due soli livelli.

La segmentazione **di istanza** unisce detection e maschere: **Mask R-CNN**
{cite}`he2017mask` estende Faster R-CNN con un terzo ramo che, per ciascuna
regione, predice una maschera binaria. Ottiene così, insieme, riquadro, classe e
sagoma di ogni singola istanza.

`````

## Risalire di risoluzione: la convoluzione trasposta

Nelle reti di segmentazione c'è un passaggio rimasto nell'ombra. Le
convoluzioni e il **pooling** (il passaggio che riassume ogni quadratino di
griglia in un numero solo) *riducono* le **mappe**, cioè le griglie di numeri
che ogni strato consegna al successivo: dopo il ramo discendente della U,
quello che comprime, un'immagine 512×512 può essersi ristretta a 16×16. Quel
ramo ha un nome, **encoder**, e vuol dire «la parte che riassume»: è lo stesso
mestiere che nella sezione sull'apprendimento senza etichette riassumeva
un'immagine in una lista di numeri, solo che qui il riassunto è una griglia
piccola. Ma il verdetto della segmentazione va dato pixel per pixel, alla
risoluzione di partenza: serve una seconda metà che riespanda, e quella si
chiama **decoder**. Come si risale?
L'operazione che la *Fully Convolutional Network* {cite}`long2015fully` ha
reso standard è la **convoluzione trasposta**: una convoluzione che invece di
rimpicciolire ingrandisce, con numeri che la rete impara.

```{figure} ../figures/convoluzione-trasposta.svg
:name: fig-convoluzione-trasposta
:alt: Una mappa due per due con i valori 1, 2, 3 e 4, ciascuno in un colore della palette, viene espansa da un kernel due per due con stride 2 in una mappa quattro per quattro in cui ogni valore diventa un blocco due per due dello stesso colore.
:width: 95%

Convoluzione trasposta con un kernel $2 \times 2$ di tutti 1, applicato ogni
due caselle: ogni valore della mappa piccola "timbra" un blocco
$2 \times 2$ della mappa grande.
```

`````{tab} Elementare

Una mappa piccola, $2 \times 2$, con quattro numeri:

$$
\begin{pmatrix} 1 & 2 \\ 3 & 4 \end{pmatrix}
$$

e un timbro, anch'esso $2 \times 2$, con il disegno più semplice possibile:
tutti 1. La convoluzione trasposta è una timbratura: ogni numero della mappa
piccola dà un colpo di timbro su una tela più grande, e il valore del numero
regola la forza della pressione. Quanto si sposta il timbro fra un colpo e
l'altro lo decidiamo noi, e si chiama **passo**. Con passo 2, cioè spostandolo
di due caselle ogni volta, i quattro colpi cadono uno accanto all'altro senza
sovrapporsi, e la tela diventa $4 \times 4$
({numref}`fig-convoluzione-trasposta`):

$$
\begin{pmatrix}
1 & 1 & 2 & 2 \\
1 & 1 & 2 & 2 \\
3 & 3 & 4 & 4 \\
3 & 3 & 4 & 4
\end{pmatrix}
$$

L'1 ha stampato un blocchetto di 1, il 4 un blocchetto di 4. Con passo 1,
invece, i colpi si sarebbero sovrapposti e nelle caselle condivise i valori si
sarebbero sommati. Le sovrapposizioni disuguali lasciano il segno: certe caselle
ricevono due colpi e le loro vicine uno solo, e nell'immagine finale compare una
trama regolare a quadretti che nella scena non c'era.

C'è poi una cosa che la timbratura non può sapere. La mappa $2 \times 2$ di
partenza l'ha prodotta la discesa, che aveva riassunto una tela più grande
prendendone le caselle a due a due. Ma una tela di quattro caselle di lato e una
di cinque si riducono tutte e due a quella stessa mappa: della quinta casella,
spaiata, la discesa non sa che fare e la lascia fuori. Il riassunto non ricorda
da quale delle due veniva, e timbrando si risale sempre alla più piccola. Chi
mette in fila una discesa e una risalita si ritrova per questo, ogni tanto, le
due metà della clessidra sfalsate di una casella, e deve dire a mano quale
delle due taglie voleva.

Con questo timbro banale abbiamo solo ingrandito la mappa; il punto è che i
numeri sul timbro non sono fissi, la rete li **impara**. Può scoprire timbri che
sfumano i bordi e ricostruiscono i dettagli molto meglio di un semplice zoom.
Può però anche fare il contrario, e imparare timbri che la trama a quadretti la
disegnano da soli, anche quando i colpi cadono ordinati e ogni casella ne riceve
lo stesso numero. Sistemare il passo, quindi, non mette al riparo, e molte reti
recenti scelgono una via più prudente: ingrandire la mappa in un modo fisso,
ricopiando ogni casella o sfumando fra una casella e la vicina, e passarci sopra
una convoluzione ordinaria, di quelle che non ingrandiscono.

`````

`````{tab} Superiore

La forma dell'output segue una formula chiusa:

$$
o = s\,(i - 1) + k - 2p ,
$$

dove $i$ è il lato della mappa in ingresso, $k$ quello del kernel, $s$ lo
stride e $p$ il padding. Nell'esempio della figura $i=2$, $k=2$, $s=2$,
$p=0$, quindi $o = 2 \cdot 1 + 2 - 0 = 4$. La formula assume dilatazione 1 e
`output_padding` nullo, e quel parametro esiste per una ragione precisa: la
convoluzione diretta manda taglie di ingresso diverse nella stessa uscita (con
$k=3$, $s=2$, $p=1$ sia un $7$ sia un $8$ diventano $4$), quindi la risalita
non è univoca e la formula dà solo la più piccola delle partenze possibili. Chi
la applica a una U-Net vera e trova le due metà della clessidra disallineate
di un pixel ha trovato questo. Il nome viene dall'algebra: se si
srotola l'input in un vettore, una convoluzione è la moltiplicazione per una
matrice sparsa $\mathbf{C}$; la trasposta moltiplica per $\mathbf{C}^\top$, che
riporta il
vettore alla dimensione di partenza. È la stessa operazione con cui la
backpropagation propaga il gradiente attraverso uno strato convoluzionale: il
*forward* della trasposta è il *backward* della convoluzione. Per questo il
vecchio nome "deconvoluzione" è fuorviante: non inverte la convoluzione, ne
inverte solo la geometria. In PyTorch è `nn.ConvTranspose2d`.

Una nota onesta: quando $k$ non è multiplo di $s$, i colpi di timbro si
sovrappongono in modo disomogeneo e l'output mostra i tipici **artefatti a
scacchiera** {cite}`odena2016deconvolution`. Verrebbe da concludere che basti
scegliere $k$ multiplo di $s$, ed è proprio la conclusione che la fonte citata
smentisce: la scelta toglie la disomogeneità geometrica ma non gli artefatti,
perché una rete impara volentieri kernel che li producono anche a
sovrapposizione uniforme. È per questo, e non per la sola divisibilità, che
molte architetture
recenti preferiscono un
upsampling fisso (bilineare o *nearest-neighbor*) seguito da una convoluzione
ordinaria.

`````

## Dove serve davvero

Questi strumenti non sono un esercizio accademico. Nella **guida autonoma**,
detection e segmentazione insieme dicono al veicolo dove sono i pedoni e dove
finisce la carreggiata. Nell’**imaging medico**, U-Net e derivati delimitano un
nodulo o un organo su una TAC, misurandone il volume con una precisione che a
occhio si perderebbe. Nell’**industria**, un rilevatore su una linea di
produzione individua il graffio o il pezzo mal assemblato prima che arrivi al
cliente.

Nessuno di questi sistemi è infallibile, e conviene dire con precisione dove
sta il margine, perché la sovrapposizione non è un tasso di errore. Una
maschera con IoU $0{,}9$ è una maschera buona, e vuol dire che di dieci
quadretti coperti in tutto, nove sono in comune e uno no: un decimo dell'area è
sbagliato, per eccesso o per difetto. Su un nodulo, un decimo di volume è una
differenza clinica. E soprattutto restano gli oggetti mancati del tutto, che in
quel numero non compaiono affatto: la IoU si calcola confrontando due cornici,
quindi esiste solo dove il sistema una cornice l'ha disegnata. Un pedone che il
rilevatore non ha visto non ha nessuna cornice, quindi nessuna IoU, quindi non
peggiora la media di niente. In medicina o alla guida sono le due cose insieme,
il contorno approssimato e l'oggetto non visto, a chiedere un occhio umano. Ma
la traiettoria è chiara: dal *cosa*, al *dove*, fino al contorno esatto.

`````{tab} Elementare

```{admonition} Da ricordare
:class: important
- Il **rilevamento** non dice solo che cosa c'è in una foto: per ogni oggetto
  disegna una cornice e ci scrive accanto il nome.
- Due modi di farlo: **in due tempi** (prima si segnano le zone sospette, poi
  si guarda con calma dentro ognuna) o **in un colpo solo** (una passata sola
  sull'immagine sputa fuori cornici e nomi). Il primo è nato più preciso, il
  secondo abbastanza rapido da stare dietro a un video; poi è cambiato il modo
  di correggere e in precisione si sono quasi raggiunti, così che a separarli
  resta soprattutto la velocità.
- Nessuno disegna le cornici dal nulla: come un corniciaio, la rete tiene
  pronti alcuni **formati standard** in ogni punto dell'immagine e si limita a
  ritoccare quello che ci va più vicino.
- Per dire quanto una cornice ci ha preso si guarda **quanto si sovrappone** a
  quella giusta: area in comune divisa per area totale, da 0 a 1. Il voto
  complessivo di un rilevatore si ottiene calcolando un voto per ogni categoria
  e poi facendone la media.
- Attorno a ogni oggetto la rete produce decine di cornici quasi uguali: prima
  del verdetto si tiene la più sicura e si buttano quelle che le si
  sovrappongono troppo, altrimenti i doppioni conterebbero tutti come errori.
- Quando la cornice non basta e serve la sagoma esatta si passa alla
  **segmentazione**: colorare ogni pixel con la sua categoria, o addirittura
  distinguere un pedone dall'altro. La forma tipica è la **U**: si scende per
  capire che cosa c'è, si risale per dire dove, e a ogni gradino della risalita
  si ripassa quello che si era visto scendendo.
- Nessuno di questi sistemi è infallibile, e il margine è di due tipi: contorni
  approssimati, e oggetti mancati del tutto (che nel punteggio della
  sovrapposizione non compaiono nemmeno).
```

`````

`````{tab} Superiore

```{admonition} Da ricordare
:class: important
- L’**object detection** predice per ogni oggetto un **riquadro** e una
  **classe** insieme.
- Due famiglie: **due stadi** (R-CNN, Faster R-CNN) storicamente più accurate,
  **uno stadio** (YOLO, SSD) più veloci; la *focal loss* ha in gran parte
  sanato lo squilibrio oggetto/sfondo che costava quel divario, lasciando la
  velocità come differenza principale. E una terza via, i rilevatori a
  **predizione di insieme** (DETR), che con l'abbinamento bipartito eliminano
  per costruzione sia le ancore sia l'NMS.
- Le **anchor box** danno alla rete riquadri di partenza a più scale e
  proporzioni: si predicono piccoli **offset**, non riquadri dal nulla.
- La **IoU** misura la sovrapposizione riquadro-realtà, non un tasso di errore;
  la **mAP** (*mean Average Precision*) riassume in un numero solo la qualità
  complessiva del rilevatore, e nel farlo accoppia ogni oggetto a una sola
  predizione: i doppioni contano come errori.
- Prima della valutazione la **non-maximum suppression** sfoltisce i doppioni:
  si tiene il riquadro più confidente e si scartano quelli troppo sovrapposti.
- **Semantica** (FCN, U-Net) etichetta ogni pixel; **istanza** (Mask R-CNN)
  separa anche i singoli oggetti. Le *skip connection* fra encoder e decoder
  nascono con la FCN; la U-Net ne generalizza la forma concatenando a ogni
  livello.
- La **convoluzione trasposta** riporta le mappe a piena risoluzione con un
  ingrandimento *appreso*: occhio agli **artefatti a scacchiera** (le griglie
  regolari che compaiono nell'uscita), che la sola divisibilità fra kernel e
  stride non basta a evitare.
```

`````
