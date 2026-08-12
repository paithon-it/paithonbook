# Object detection e segmentazione

Un'auto a guida autonoma si avvicina a un incrocio. Una rete di
classificazione, di quelle viste finora, sa dirle una cosa sola:
*nell'immagine c'è un pedone*. Vero, ma inutile. Per frenare in tempo l'auto
deve sapere **dove** si trova quel pedone, se è uno o sono tre, se quello a
destra è un ciclista, e (al limite), quale sagoma esatta occupa sull'asfalto.
La classificazione risponde alla domanda "cosa"; qui impariamo a rispondere
anche a "dove" e "quali contorni".

## Dalla classificazione al riquadro

Salire dalla classificazione alla localizzazione è come passare da "in questa
foto c'è un gatto" a "il gatto sta in *quel* rettangolo". Il compito si chiama
**object detection**: per ogni oggetto presente, la rete deve produrre insieme
un riquadro e un'etichetta.

```{figure} ../figures/classification-detection-segmentation.svg
:name: fig-tre-uscite
:alt: "La stessa fotografia trattata da tre compiti diversi. Nella classificazione l'uscita è una sola etichetta per l'intera immagine. Nella detection sono uno o più riquadri, ciascuno con la sua etichetta. Nella segmentazione è una maschera che segue il contorno esatto di ogni oggetto, pixel per pixel."
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

Immagina di cerchiare con un pennarello ogni oggetto interessante in una foto
e di scrivergli accanto un nome: "auto", "cane", "semaforo". Ogni cerchio è in
realtà un rettangolo (lo chiamiamo **bounding box**, la "cornice") e ogni nome
è la classe. Un rilevatore fa esattamente questo, ma in automatico e per molti
oggetti insieme. Per ciascuno deve indovinare due cose alla volta: *dove*
tracciare la cornice e *cosa* c'è dentro.

`````

`````{tab} Superiore

Per ogni oggetto la rete predice un vettore
$(x, y, w, h, c, p)$: le coordinate del centro e le dimensioni del riquadro,
la classe $c$ e una confidenza $p \in [0,1]$ che stima se nel riquadro c'è
davvero un oggetto. L'addestramento minimizza una loss composita che somma un
termine di **localizzazione** (errore sulle coordinate, tipicamente
*smooth L1* o una IoU-loss), un termine di **classificazione** (cross-entropy
sulla classe) e un termine di **objectness** che supervisiona la confidenza,
spingendola verso l'alto dove un oggetto c'è e verso zero sullo sfondo (in
YOLOv1 il bersaglio non è uno ma la IoU stessa fra riquadro predetto e
riquadro vero, così che il punteggio incorpori già la qualità della
localizzazione; dai successori in poi, semplicemente uno):

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
classico compromesso tra **accuratezza e velocità**.

```{figure} ../figures/yolo-2016.svg
:name: fig-yolo
:alt: "Un'immagine coperta da una griglia sette per sette entra una sola volta in una rete convoluzionale, che produce direttamente i riquadri degli oggetti con le rispettive classi. Non c'è nessuna fase separata di proposta delle regioni: ogni cella della griglia è responsabile degli oggetti il cui centro le cade dentro."
:width: 92%

L'approccio a stadio singolo. La griglia non è una ricerca: è una divisione di
responsabilità decisa in anticipo, e la rete la attraversa una volta sola.
```

Ciò che {numref}`fig-yolo` rende evidente è cosa si guadagna e cosa si perde.
Guardare l'immagine una volta sola è quello che rende possibile il tempo
reale; il prezzo è che ogni cella ha un numero fisso di riquadri da offrire, e
oggetti piccoli e ammassati nella stessa cella se li contendono.

`````{tab} Elementare

I rilevatori **a due stadi** lavorano come un revisore scrupoloso: prima
propongono un po' di zone "sospette" dove *potrebbe* esserci qualcosa, poi
guardano con calma dentro ognuna per decidere cosa sia e correggere la cornice.
Più precisi, ma più lenti.

I rilevatori **a uno stadio** fanno tutto in un colpo solo: un'unica passata
sull'immagine sputa fuori direttamente cornici ed etichette. Meno precisi sui
casi difficili, ma abbastanza rapidi da lavorare in tempo reale su un video,
ed è per questo che si chiama YOLO, *You Only Look Once*.

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
scorciatoia si chiama **anchor box**, l'"ancora": un riquadro di partenza già
pronto, da correggere invece che da inventare.

```{figure} ../figures/anchor-boxes.svg
:name: fig-anchor-boxes
:alt: Griglia sovrapposta a un'immagine con un pedone stilizzato; su un punto della griglia sono centrate tre ancore, una alta e stretta in terracotta che ricalca il pedone, una bassa e larga in teal e una quadrata in ocra, entrambe tratteggiate.
:width: 95%

In ogni punto della griglia la rete parte da più ancore di forma diversa: per
il pedone, quella alta e stretta (terracotta) è già quasi giusta.
```

`````{tab} Elementare

Pensa a un corniciaio. Non costruisce una cornice su misura per ogni quadro
che entra in bottega: tiene pronti alcuni formati standard (verticale per i
ritratti, orizzontale per i paesaggi, quadrato) e poi ritocca quello che ci va
più vicino. Le ancore funzionano allo stesso modo. In ogni punto dell'immagine
la rete ha a disposizione qualche cornice predefinita, di taglie e proporzioni
diverse ({numref}`fig-anchor-boxes`). Davanti a un pedone non deve inventarsi
il rettangolo: prende la cornice verticale, che gli somiglia già, e la
aggiusta ("spostati un po' a sinistra, allungati un pelo"). Correggere una
cornice quasi giusta è molto più facile che disegnarne una dal nulla, e la
rete impara proprio questo: piccole correzioni, più il nome di ciò che c'è
dentro.

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
si misura con la IoU, protagonista della prossima sezione). È il meccanismo
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
misura numerica di "quanto ci ha preso". Quella misura è l'**Intersection over
Union**.

```{figure} ../figures/iou.svg
:name: fig-iou
:alt: Due riquadri sovrapposti, reale in teal e predetto in terracotta; a sinistra evidenziata in ocra l'area di intersezione, a destra in tinta teal l'area di unione.
:width: 90%

L'Intersection over Union confronta il riquadro predetto (terracotta) con quello
reale (teal): è l'area di **intersezione** divisa per l'area di **unione**.
```

`````{tab} Elementare

Prendi la cornice predetta e quella giusta e chiediti: *quanto si
sovrappongono?* La IoU misura proprio questo. Si guarda l'area in comune (la
sovrapposizione) e la si divide per l'area totale coperta dalle due cornici
messe insieme ({numref}`fig-iou`). Il risultato va da 0 a 1: **1** significa
cornici perfettamente combacianti, **0** che non si toccano nemmeno. Un
esempio con i quadretti: se le due cornici ne condividono 30 e, messe insieme,
ne coprono 60, la IoU è $30/60 = 0{,}5$. Di solito si dice che una predizione
è "giusta" proprio se la IoU supera **0,5**: metà o più di sovrapposizione.
Quel mezzo però non è una legge di natura, è una convenzione, e alzarla a 0,7
o a 0,9 può cambiare la classifica fra due rilevatori: uno che indovina sempre
la categoria ma disegna cornici approssimative peggiora molto più in fretta di
uno preciso. Per questo i confronti seri non usano una soglia sola.

Da qui si ricava il voto complessivo di un rilevatore, la **mAP**. La sigla sta
per *mean Average Precision*, e le medie sono due, una dentro l'altra: prima si
calcola un voto per **ogni categoria** (quanto bene il rilevatore se la cava
sui gatti, quanto sulle biciclette), poi si fa la media di quei voti **sulle
categorie**. È un numero fra 0 e 1 che riassume in un colpo solo quanto spesso
il rilevatore azzecca insieme la cornice e il nome. Più è alto, meglio è, ed è
il numero con cui due rilevatori si confrontano senza doverli guardare foto per
foto.

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
al 2009, a tutti i cambi di recall dal 2010, a 101 punti in COCO) è l'**Average
Precision** (AP). La curva grezza è a denti di sega e nessuno la integra: è la
scelta dell'interpolazione a rendere i numeri riproducibili fra
implementazioni. La **mean Average Precision** (mAP) ne fa la media sulle
classi. Il benchmark COCO irrigidisce la metrica mediando la mAP su dieci soglie
di IoU, da $0{,}5$ a $0{,}95$ a passi di $0{,}05$: premia i modelli che
localizzano con precisione, non solo che indovinano la classe.

`````

Prima del verdetto, però, serve un passaggio di pulizia. La rete non produce
un riquadro per oggetto: ne produce migliaia (Faster R-CNN, per esempio, tiene
pronte nove cornici di partenza in ogni punto della griglia, tre taglie per tre
proporzioni, e i punti della griglia sono migliaia), e attorno a ogni oggetto
se ne accumulano decine quasi
identici, ciascuno con la sua confidenza. La **non-maximum suppression** (NMS)
li sfoltisce con una regola semplice: si ordinano i riquadri per confidenza,
si tiene il più sicuro e si scartano tutti quelli che gli si sovrappongono
troppo (IoU sopra una soglia), poi si ripete sui rimasti finché non c'è più
niente da esaminare. È il passo finale, quasi mai disegnato negli schemi, di
praticamente ogni rilevatore, da Faster R-CNN a YOLO e SSD: senza, ogni
oggetto arriverebbe alla valutazione con un grappolo di doppioni, e tutti
tranne uno conterebbero come errori.

C'è però una terza via, che il problema lo toglie invece di risolverlo. Una
famiglia di rilevatori inaugurata da DETR {cite}`carion2020end` chiede alla
rete un numero fisso di risposte e, durante l'addestramento, le abbina agli
oggetti veri **una a una**, come si formano le coppie in un ballo: ogni oggetto
ha un solo inseguitore, quindi i doppioni non nascono affatto, e spariscono
insieme sia le cornici di partenza sia la fase di pulizia. Il prezzo,
storicamente, è stato un addestramento molto più lento ad assestarsi.

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
:alt: "Schema a forma di U: il braccio discendente di sinistra riduce progressivamente la risoluzione aumentando il numero di canali, in fondo c'è il collo di bottiglia, e il braccio ascendente di destra recupera la risoluzione. Fra livelli corrispondenti dei due bracci corrono connessioni orizzontali tratteggiate, le skip connection."
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

Nelle reti di segmentazione c'è un passaggio rimasto nell'ombra. Convoluzioni
e pooling *riducono* le **mappe**, cioè le griglie di numeri che ogni strato
consegna al successivo: dopo il ramo discendente della U, quello che comprime,
un'immagine 512×512 può essersi ristretta a 16×16. (È lo stesso ramo che nella
sezione precedente abbiamo chiamato **encoder**: la parola vuol dire sempre «la
parte che riassume», e qui il riassunto è una griglia piccola invece di una
lista di numeri.) Ma il
verdetto della segmentazione va dato pixel per pixel, alla risoluzione di
partenza: serve una metà che riespanda, il **decoder**. Come si risale?
L'operazione che la *Fully Convolutional Network* {cite}`long2015fully` ha
reso standard è la **convoluzione trasposta**: una convoluzione che invece di
rimpicciolire ingrandisce, con numeri che la rete impara.

```{figure} ../figures/convoluzione-trasposta.svg
:name: fig-convoluzione-trasposta
:alt: Una mappa due per due con i valori 1, 2, 3 e 4, ciascuno in un colore della palette, viene espansa da un kernel due per due con stride 2 in una mappa quattro per quattro in cui ogni valore diventa un blocco due per due dello stesso colore.
:width: 95%

Convoluzione trasposta con kernel $2 \times 2$ di tutti 1 e stride 2: ogni
valore della mappa piccola "timbra" un blocco $2 \times 2$ della mappa grande.
```

`````{tab} Elementare

Immagina una mappa piccola, $2 \times 2$, con quattro numeri:

$$
\begin{pmatrix} 1 & 2 \\ 3 & 4 \end{pmatrix}
$$

e un timbro, anch'esso $2 \times 2$, con il disegno più semplice possibile:
tutti 1. La convoluzione trasposta è una timbratura: ogni numero della mappa
piccola dà un colpo di timbro su una tela più grande, e il valore del numero
regola la forza della pressione. Con passo 2, i quattro colpi cadono uno
accanto all'altro senza sovrapporsi, e la tela diventa $4 \times 4$
({numref}`fig-convoluzione-trasposta`):

$$
\begin{pmatrix}
1 & 1 & 2 & 2 \\
1 & 1 & 2 & 2 \\
3 & 3 & 4 & 4 \\
3 & 3 & 4 & 4
\end{pmatrix}
$$

L'1 ha stampato un blocchetto di 1, il 4 un blocchetto di 4. Con questo
timbro banale abbiamo solo ingrandito la mappa; il punto è che i numeri sul
timbro non sono fissi, la rete li **impara**. Può scoprire timbri che sfumano
i bordi e ricostruiscono i dettagli molto meglio di un semplice zoom.

`````

`````{tab} Superiore

La forma dell'output segue una formula chiusa:

$$
o = s\,(i - 1) + k - 2p ,
$$

dove $i$ è il lato della mappa in ingresso, $k$ quello del kernel, $s$ lo
stride e $p$ il padding. Nell'esempio della figura $i=2$, $k=2$, $s=2$,
$p=0$, quindi $o = 2 \cdot 1 + 2 - 0 = 4$. La formula assume dilatazione 1 e
`output_padding` nullo, e vale la pena capire perché quel parametro esista: la
convoluzione diretta manda taglie di ingresso diverse nella stessa uscita (con
$k=3$, $s=2$, $p=1$ sia un $7$ sia un $8$ diventano $4$), quindi la risalita
non è univoca e la formula dà solo la più piccola delle partenze possibili. Chi
la applica a una U-Net vera e trova le due metà della clessidra disallineate
di un pixel ha trovato questo. Il nome viene dall'algebra: se si
srotola l'input in un vettore, una convoluzione è la moltiplicazione per una
matrice sparsa $C$; la trasposta moltiplica per $C^\top$, che riporta il
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
finisce la carreggiata. Nell'**imaging medico**, U-Net e derivati delimitano un
nodulo o un organo su una TAC, misurandone il volume con una precisione che a
occhio si perderebbe. Nell'**industria**, un rilevatore su una linea di
produzione individua il graffio o il pezzo mal assemblato prima che arrivi al
cliente.

Nessuno di questi sistemi è infallibile, e conviene dire con precisione dove
sta il margine, perché la IoU non è un tasso di errore. Una maschera con IoU
$0{,}9$ è una maschera buona, e lascia comunque fuori (o dentro) un decimo
dell'area: su un nodulo, un decimo di volume è una differenza clinica. E
soprattutto restano gli oggetti mancati del tutto, che in quel numero non
compaiono affatto, perché la IoU si calcola solo sulle coppie che il sistema ha
prodotto. In medicina o alla guida sono le due cose insieme, il contorno
approssimato e l'oggetto non visto, a chiedere un occhio umano. Ma la
traiettoria è chiara: dal *cosa*, al *dove*, fino al contorno esatto.

`````{tab} Elementare

```{admonition} Da ricordare
:class: important
- Il **rilevamento** non dice solo che cosa c'è in una foto: per ogni oggetto
  disegna una cornice e ci scrive accanto il nome.
- Due modi di farlo: **in due tempi** (prima si segnano le zone sospette, poi
  si guarda con calma dentro ognuna) o **in un colpo solo** (una passata sola
  sull'immagine sputa fuori cornici e nomi). Il primo è più preciso, il secondo
  abbastanza rapido da stare dietro a un video.
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
- L'**object detection** predice per ogni oggetto un **riquadro** e una
  **classe** insieme.
- Due famiglie: **due stadi** (R-CNN, Faster R-CNN) più accurate, **uno
  stadio** (YOLO, SSD) più veloci (un compromesso accuratezza/velocità); e una
  terza via, i rilevatori a **predizione di insieme** (DETR), che con
  l'abbinamento bipartito eliminano per costruzione sia le ancore sia l'NMS.
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
