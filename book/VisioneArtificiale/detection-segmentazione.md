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
la classe $c$ e una confidenza $p \in [0,1]$. L'addestramento minimizza una
loss composita che somma un termine di **localizzazione** (errore sulle
coordinate, tipicamente *smooth L1* o una IoU-loss) e un termine di
**classificazione** (cross-entropy sulla classe):

$$
\mathcal{L} = \mathcal{L}_{\text{cls}} + \lambda \,\mathcal{L}_{\text{box}} .
$$

Il coefficiente $\lambda$ bilancia i due obiettivi. Un'immagine può contenere un
numero variabile di oggetti: gestire questa cardinalità ignota è il vero nodo
architetturale della detection.

`````

## Due stadi contro uno stadio

Storicamente i rilevatori si dividono in due famiglie, e la differenza è un
classico compromesso tra **accuratezza e velocità**.

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

`````

`````{tab} Superiore

Dati il riquadro predetto $A$ e quello reale $B$, la IoU è

$$
\text{IoU} = \frac{|A \cap B|}{|A \cup B|} \in [0,1] .
$$

Fissata una soglia (ad esempio $\text{IoU} \ge 0{,}5$), ogni predizione diventa
un **vero positivo** o un **falso positivo**. Da qui si costruisce la curva
*precision–recall* per ciascuna classe: la sua area sotto la curva è l'**Average
Precision** (AP). La **mean Average Precision** (mAP) ne fa la media sulle
classi. Il benchmark COCO irrigidisce la metrica mediando la mAP su dieci soglie
di IoU, da $0{,}5$ a $0{,}95$ a passi di $0{,}05$: premia i modelli che
localizzano con precisione, non solo che indovinano la classe.

`````

## Segmentare: dal riquadro alla sagoma

Il riquadro è comodo ma grossolano: attorno a un pedone c'è sempre un rettangolo
pieno di sfondo. Quando serve il contorno esatto, pixel per pixel, si passa alla
**segmentazione**. Qui vanno distinti due sapori.

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
di classi a piena risoluzione. **U-Net** {cite}`ronneberger2015u`, nata per
l'imaging biomedico, aggiunge le *skip connections* tra encoder e decoder,
recuperando i dettagli fini persi nel *downsampling*.

La segmentazione **di istanza** unisce detection e maschere: **Mask R-CNN**
{cite}`he2017mask` estende Faster R-CNN con un terzo ramo che, per ciascuna
regione, predice una maschera binaria. Ottiene così, insieme, riquadro, classe e
sagoma di ogni singola istanza.

`````

## Risalire di risoluzione: la convoluzione trasposta

Nelle reti di segmentazione c'è un passaggio rimasto nell'ombra. Convoluzioni
e pooling *riducono* le mappe: dopo la metà della rete che comprime
(l'**encoder**) un'immagine 512×512 può essersi ristretta a 16×16. Ma il
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
$p=0$, quindi $o = 2 \cdot 1 + 2 - 0 = 4$. Il nome viene dall'algebra: se si
srotola l'input in un vettore, una convoluzione è la moltiplicazione per una
matrice sparsa $C$; la trasposta moltiplica per $C^\top$, che riporta il
vettore alla dimensione di partenza. È la stessa operazione con cui la
backpropagation propaga il gradiente attraverso uno strato convoluzionale: il
*forward* della trasposta è il *backward* della convoluzione. Per questo il
vecchio nome "deconvoluzione" è fuorviante: non inverte la convoluzione, ne
inverte solo la geometria. In PyTorch è `nn.ConvTranspose2d`.

Una nota onesta: quando $k$ non è multiplo di $s$, i colpi di timbro si
sovrappongono in modo disomogeneo e l'output mostra i tipici **artefatti a
scacchiera** {cite}`odena2016deconvolution`. Per questo molte architetture
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

Nessuno di questi sistemi è infallibile: una IoU del 90% resta un 10% di
errore, e in medicina o alla guida quel margine va sempre soppesato con un
occhio umano. Ma la traiettoria è chiara: dal *cosa*, al *dove*, fino al
contorno esatto.

```{admonition} Da ricordare
:class: important
- L'**object detection** predice per ogni oggetto un **riquadro** e una
  **classe** insieme.
- Due famiglie: **due stadi** (R-CNN, Faster R-CNN) più accurate, **uno
  stadio** (YOLO, SSD) più veloci (un compromesso accuratezza/velocità).
- Le **anchor box** danno alla rete riquadri di partenza a più scale e
  proporzioni: si predicono piccoli **offset**, non riquadri dal nulla.
- La **IoU** misura la sovrapposizione riquadro-realtà; la **mAP** riassume la
  qualità complessiva del rilevatore.
- **Semantica** (FCN, U-Net) etichetta ogni pixel; **istanza** (Mask R-CNN)
  separa anche i singoli oggetti.
- La **convoluzione trasposta** riporta le mappe a piena risoluzione con un
  ingrandimento *appreso*: occhio agli artefatti a scacchiera.
```
