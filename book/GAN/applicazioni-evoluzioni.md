# Evoluzioni e applicazioni

Le prime immagini generate da una GAN, nel 2014, erano cifre sgranate e
faccine indistinte, appena qualche decina di pixel di lato. Ian Goodfellow le
mostrava con orgoglio, ma nessuno le avrebbe scambiate per fotografie. Cinque
anni dopo, il sito già incontrato in apertura di capitolo,
*thispersondoesnotexist.com*, sforna a ripetizione volti fotorealistici di
persone che non esistono. In mezzo c'è la storia di questa sezione: una
sequenza di idee (quasi una all'anno) che ha trasformato un'intuizione fragile
in una delle famiglie di modelli generativi più influenti del decennio.
Ripercorriamola, seguendo il filo delle idee più che il calendario.

I due personaggi restano quelli: un falsario che dipinge e un esperto che
giudica. Quello che cambia, di variante in variante, è come sono fatti dentro
(DCGAN), che cosa gli si mette in mano oltre al rumore (conditional GAN), in
che momento del lavoro gli si danno le istruzioni (StyleGAN), che cosa gli si
chiede di conservare mentre dipinge (pix2pix e CycleGAN) e, all'ultima tappa,
perfino a che cosa serva il duello (VQ-GAN). Ogni variante, insomma, si
legge bene come una modifica al regolamento di quel duello.

## DCGAN: dare occhi alla rete

La prima GAN girava soprattutto su **strati densi** (*fully-connected*), cioè
strati in cui ogni numero in uscita dipende da tutti i numeri in entrata, senza
che la loro posizione conti niente: un'immagine, per uno strato così, è una
lista disordinata di numeri, e il fatto che due pixel (i puntini di cui si è
parlato finora) siano vicini non gli dice niente. Ma in un'immagine la posizione conta, e pixel vicini quasi sempre si
somigliano: un pezzo di cielo è azzurro tutto intorno.

Una variante che ne teneva conto c'era già nel paper del 2014, e addestrarla
era un terno al lotto: quel che mancava era una ricetta che stesse in piedi, non
l'idea. Arriva a fine 2015 con la **DCGAN** (*Deep Convolutional GAN*)
di Radford, Metz e Chintala {cite}`radford2016unsupervised`.

`````{tab} Elementare
L'idea è dare alla rete lo strumento giusto per "vedere": far scorrere sopra
l'immagine, casella per casella, una piccola griglia di numeri che reagisce a
una certa forma (un bordo verticale, una macchia chiara su fondo scuro, la
grana di un tessuto). Quella griglietta si chiama **filtro**, e l'operazione è
la **convoluzione**; quali numeri mettere nei filtri è precisamente ciò che la
rete impara. È il discriminatore ad analizzare l'immagine così, con gli stessi
filtri di un classificatore qualsiasi; il generatore fa il percorso inverso, e
parte da un pugno di numeri casuali per "gonfiarli" fino a un'immagine intera.

Gonfiare come, se i numeri di partenza sono un centinaio e i puntini d'arrivo
qualche migliaio? Il primo passaggio dispone quel centinaio di numeri in una
griglia minuscola, $4\times4$, ma spessa: in ogni casella non un valore solo,
bensì una pila di valori. Da lì in poi ogni passaggio raddoppia il lato: da
$4\times4$ a $8\times8$, poi $16\times16$, fino ai $64\times64$ che erano la
taglia della DCGAN, mentre la pila si assottiglia. I puntini in più sono
*decisi* e non copiati da nessuna parte: dove il vecchio puntino era uno solo,
il passaggio successivo ne mette quattro, e quanto ciascuno debba essere chiaro
o scuro lo dicono i filtri. Con questa architettura le immagini smettono di
essere macchie e cominciano ad avere bordi netti e coerenza.

E su quel pugno di numeri di partenza si possono fare i conti. Si prendono i
numeri che hanno fatto tre uomini con gli occhiali, si tolgono quelli di tre
uomini senza, si aggiungono quelli di tre donne: esce una donna con gli
occhiali. Con un esemplare solo per gruppo il conto salta, perché «occhiali»
sta in una zona di quei numeri e non in un punto preciso.
`````

`````{tab} Superiore
La DCGAN codifica una serie di scelte architetturali diventate standard: il generatore $G$ usa convoluzioni trasposte (*strided transposed convolutions*) per l'upsampling, il discriminatore $D$ usa convoluzioni con *stride*; nessun pooling; *batch normalization* in entrambe le reti, con due eccezioni che il paper stesso impone (niente batchnorm sullo strato di uscita di $G$ e su quello d'ingresso di $D$, dove gli autori riportano oscillazioni dei campioni e instabilità); attivazioni ReLU nel generatore (tranne l'output con $\tanh$) e LeakyReLU nel discriminatore.

Il paper mostra anche che lo spazio latente $\mathcal{Z}$ è **semanticamente strutturato**: aritmetica vettoriale come "uomo con occhiali $-$ uomo $+$ donna" produce, decodificata da $G$, il volto di una donna con occhiali. Con una cautela che il paper stesso dichiara, e che pesa: sui singoli vettori l'operazione è instabile, e il risultato regge mediando i $\mathbf{z}$ di tre esemplari per concetto. La struttura c'è, ma è una proprietà di regioni dello spazio, non di punti singoli; ed è comunque un indizio precoce del fatto che la rete apprende una rappresentazione, non una tabella di memorizzazione.
`````

## Conditional GAN: prendere il controllo

Una GAN base genera "qualcosa di plausibile", ma non possiamo chiederle *cosa*. La **conditional GAN** {cite}`mirza2014conditional`, proposta già a fine 2014, pochi mesi dopo il paper originale, aggiunge il timone: si passa alla rete anche un'etichetta, e la generazione la rispetta.

`````{tab} Elementare
Insieme ai numeri casuali forniamo un'informazione in più: "voglio un 7",
oppure "un gatto". Il primo esempio viene da MNIST, la raccolta di cifre
scritte a mano che da decenni si usa per le prove: settantamila immaginette
minuscole, 28 pixel per lato, ciascuna con accanto la cifra che rappresenta.

La richiesta si consegna alla rete come tutto il resto, cioè sotto forma di
numeri. Una cifra fra zero e nove è già un numero; per un'etichetta a parole si
decide una volta per tutte un numero per ciascuna parola possibile, e "voglio
un gatto" diventa una manciata di numeri accodata al rumore. Sia il generatore
sia il discriminatore ricevono questa etichetta: il primo la usa come
istruzione, il secondo per giudicare non solo *se* l'immagine è verosimile, ma
se corrisponde davvero all'etichetta richiesta. Così smettiamo di pescare a
caso e cominciamo a ordinare su misura.
`````

`````{tab} Superiore
Si condiziona il gioco minimax su una variabile ausiliaria $y$ (la classe, o un vettore qualsiasi):

$$
\min_{G}\max_{D}\;\mathbb{E}_{(\mathbf{x},y)\sim p_{\text{dati}}(\mathbf{x},y)}\big[\log D(\mathbf{x},y)\big]
+\mathbb{E}_{y\sim p_y,\;\mathbf{z}\sim p_z}\big[\log\big(1-D(G(\mathbf{z},y),\,y)\big)\big].
$$

Qui $p_{\text{dati}}(\mathbf{x},y)$ è la distribuzione **congiunta** di dato e
condizione, scritta con gli argomenti espliciti proprio per distinguerla dalla
marginale sui soli dati che nel resto del capitolo abbiamo indicato con
$p_{\text{dati}}$; $p_y$ è la distribuzione delle condizioni e $\mathbf{z}$ il rumore
latente. Entrambe le reti ricevono $y$ come ingresso aggiuntivo, e per questo
lo scriviamo come secondo argomento: $G(\mathbf{z},y)$ è il campione generato coerente
con $y$, e $D(\mathbf{x},y)$ il giudizio del discriminatore sulla coppia. Il paper di
Mirza e Osindero usa la barra ($D(\mathbf{x}\mid y)$), ma quella barra non denota una
probabilità condizionata: è solo un ingresso in più, e la virgola lo dice senza
ambiguità.

Il condizionamento è il seme concettuale di quasi tutto ciò che segue: la
traduzione immagine-a-immagine, e in ultima analisi il *text-to-image*, non
sono altro che GAN (o modelli generativi) condizionati su un input sempre più
ricco (da un'etichetta discreta a un'intera frase).
`````

## StyleGAN: il fotorealismo

È la tecnologia dietro i volti impossibili da smascherare a occhio, ed è quella
che ha reso famose le GAN presso chi non le ha mai studiate. Ma il cambiamento
che StyleGAN {cite}`karras2019style` porta sta nel **governo** di ciò che si
ottiene, più che nella qualità delle immagini. Il generatore di NVIDIA cambia
impianto per farsi guidare; poi disegna anche meglio, ma quello viene in
aggiunta.

`````{tab} Elementare
StyleGAN non "disegna" il volto tutto in una volta: lo costruisce a livelli,
dal grossolano al fine. Gli strati iniziali decidono posa e forma del viso,
quelli intermedi i lineamenti, quelli finali dettagli come lentiggini e
ciocche di capelli. A ogni livello consegna uno "stile", che è una fila di
manopole: una manciata di numeri che invece di stare tutta all'ingresso arriva
al singolo livello e decide come quel livello lavorerà. Le manopole non escono
grezze dai numeri casuali di partenza: fra i due c'è una piccola rete che li
traduce, e serve a sbrogliarli, perché nel mazzo grezzo posa, età e taglio di
capelli sono aggrovigliati, e girando una manopola se ne muovono tre.

Cambiando le manopole dei primi livelli cambia la posa; cambiando quelle degli
ultimi cambiano le lentiggini. Per questo si mescolano tratti di volti diversi:
le manopole dei primi livelli da un volto, quelle degli ultimi da un altro, e
viene fuori un fotomontaggio impossibile e perfettamente coerente. Un pizzico
di casualità, poi, non passa dalle manopole: entra dritto a ogni livello e
decide i dettagli che nessuno sceglie, dove cada una ciocca e come si posi la
grana della pelle. Stesse manopole e casualità diversa danno la stessa persona
appena spettinata.

Una cosa StyleGAN non l'ha inventata, ed è quella che si nota per prima: le
immagini grandi come una fotografia vera erano state conquistate l'anno prima
dal modello da cui parte, la **Progressive GAN**, che faceva crescere le due
reti un pezzo alla volta. StyleGAN eredita quella scala e ci aggiunge il
controllo.

Che poi il risultato sia una fotografia e non un disegno non dipende dai
livelli: dipende dall'esperto, allenato su fotografie vere, che boccia tutto
ciò che non lo sembra. I livelli danno il *controllo*, il duello dà il realismo.
`````

`````{tab} Superiore
L'innovazione è architetturale: una rete di *mapping* trasforma $\mathbf{z}$ in uno spazio latente intermedio $\mathcal{W}$, più disaccoppiato; i vettori di stile $\mathbf{w}$ modulano ogni strato del generatore via *adaptive instance normalization* (AdaIN); rumore stocastico separato controlla i dettagli ad alta frequenza. Il risultato è il controllo *scale-specific*.

La risoluzione $1024\times1024$, invece, StyleGAN la eredita: viene dalla
**Progressive GAN** dello stesso gruppo {cite}`karras2018progressive`, che
l'anno prima aveva imparato a salire fino a lì facendo crescere le due reti un
livello per volta, e che StyleGAN dichiara come propria configurazione di base,
«da cui ereditiamo le reti e tutti gli iperparametri, salvo dove indicato». Là
si era imparato a *salire* la scala; qui si impara a decidere che cosa succede
a ciascun gradino. StyleGAN2 (2020) elimina poi i caratteristici artefatti "a
goccia" ridisegnando la normalizzazione: con quella revisione la ricetta si
assesta, ed è la forma in cui la famiglia è entrata nell'uso corrente.
`````

## Sotto il cofano: crescere, modulare, e la goccia

Tre meccanismi sono stati nominati e non aperti: come si
fanno crescere due reti, che cosa vuol dire «consegnare una manopola a un
livello», e che cos'era la macchia a goccia che StyleGAN2 ha fatto sparire.
Sono meccanismi e non risultati, e ciascuno vale al di là di StyleGAN. La
crescita per gradini è l'esempio più limpido di un'idea che torna ogni volta
che un addestramento è troppo grosso per essere affrontato tutto insieme, e
poco più avanti verrà superata proprio da chi l'aveva inventata. La
modulazione degli strati è finita dentro i generatori di immagini di oggi, e
si ritrova sotto un altro nome nella {doc}`sezione sul Diffusion Transformer
</ModelliDiffusione/diffusion-transformer>`. La storia della goccia è la
migliore lezione di metodo del capitolo.

### Crescere per gradini

Un'immagine di $4 \times 4$ pixel è fatta di sedici puntini. Su una cosa così
il falsario e l'esperto hanno poco su cui litigare: dov'è il chiaro, dov'è lo
scuro, e basta. È una lite che si chiude in fretta. Su un'immagine di
$1024 \times 1024$ le cose su cui litigare sono un milione, e il duello, che
già di suo è capriccioso, non arriva da nessuna parte. L'idea della
**Progressive GAN** {cite}`karras2018progressive` è ovvia a dirsi: si comincia
dai sedici puntini, si aspetta che le due reti vadano d'accordo, poi si aggiunge
un gradino e si raddoppia il lato. Sedici puntini, poi sessantaquattro, poi
duecentocinquantasei, e via fino al milione.

Il punto delicato è il gradino, ed è lì che sta il mestiere. Aggiungere di
colpo uno strato nuovo, coi suoi numeri ancora casuali, davanti a due reti che
avevano appena trovato l'equilibrio vuol dire buttare all'aria l'equilibrio. La
soluzione è una **dissolvenza**: per un po’ l'immagine che esce è una miscela
fra quella del gradino vecchio (semplicemente ingrandita) e quella del gradino
nuovo, e il peso della miscela scivola da zero
a uno nel corso dell'addestramento. All'inizio comanda il vecchio, alla fine il
nuovo, e in mezzo non c'è nessun salto. Lo stesso, specularmente, dalla parte
dell'esperto. Nessuno dei due si sveglia una mattina in un mondo diverso.

`````{tab} Elementare

Un maestro che insegna a copiare un volto non fa cominciare dalle ciglia: fa
tracciare l'ovale, poi la posizione degli occhi, poi i lineamenti, e i dettagli
per ultimi. Se si comincia dalle ciglia si sbaglia tutto, perché non c'è ancora
una faccia su cui metterle.

La crescita per gradini fa esattamente questo con due allievi che si
controllano a vicenda. E la dissolvenza è la cortesia di non cambiare foglio di
colpo: per un po’ il disegno che si consegna è mezzo quello vecchio ingrandito
e mezzo quello nuovo, e la proporzione si sposta piano. Chi giudica non se ne
accorge, e continua a giudicare.

Nel paper ci sono altri tre accorgimenti, e li nominiamo perché sono il genere
di cosa che non si trova sui manuali e fa la differenza fra un addestramento
che regge e uno che no. Il primo: si fa in modo che tutti i pesi della
rete rispondano con la stessa prontezza, perché altrimenti alcuni si aggiustano
in fretta e altri restano indietro, e chi resta indietro rallenta tutti. Il
secondo: dentro il falsario, dopo ogni passaggio, i numeri vengono riportati a
una taglia standard, perché in una gara a chi urla più forte tendono a
gonfiarsi da soli. Il terzo: si mostra all'esperto **quanto sono diverse fra
loro** le immagini di un gruppo, che è il modo più diretto di smascherare un
falsario che dipinge sempre lo stesso quadro.

`````

`````{tab} Superiore

La dissolvenza si applica agli strati $1\times1$ che convertono le mappe di
attivazione in tre canali RGB (*toRGB*) e viceversa (*fromRGB*). Passando da
risoluzione $R$ a $2R$, per un certo numero di iterazioni l'uscita è

$$
\mathbf{x} = (1 - \alpha)\, \mathrm{up}\big(\mathrm{toRGB}_R(\mathbf{h}_R)\big)
\;+\; \alpha\, \mathrm{toRGB}_{2R}(\mathbf{h}_{2R}),
\qquad \alpha: 0 \to 1,
$$

con $\mathrm{up}$ un semplice raddoppio per interpolazione: il vecchio ramo
resta in funzione e cede il passo con continuità. Il discriminatore fa il
percorso simmetrico sui *fromRGB*.

Gli altri tre contributi del lavoro:

- **Equalized learning rate.** I pesi si inizializzano da $\mathcal{N}(0,1)$ e
  si riscalano *a tempo di esecuzione* con la costante per-strato
  dell'inizializzatore di He, invece di applicarla all'inizializzazione. Il
  motivo è che ottimizzatori adattivi come
  Adam normalizzano l'aggiornamento peso per peso, quindi il tempo che un peso
  impiega ad adattarsi dipende dalla sua scala: con l'inizializzazione classica
  gli strati con pochi ingressi hanno pesi grandi e si muovono più lentamente
  degli altri. Scalando a runtime, tutti i pesi hanno lo stesso raggio d'azione
  e lo stesso passo effettivo.
- **Pixelwise feature normalization** nel generatore: dopo ogni convoluzione il
  vettore di attivazioni di ciascun pixel è riscalato perché la media dei suoi
  quadrati sui canali valga uno. Non
  ha parametri appresi e serve a impedire che le magnitudini scappino via
  durante l'escalation del duello.
- **Minibatch standard deviation**: al discriminatore si aggiunge un canale
  costante che riporta la deviazione standard delle attivazioni *attraverso il
  batch*. È una misura diretta della varietà del gruppo, e rende il *mode
  collapse* visibile a chi giudica invece che invisibile.

`````

### Modulare invece di ordinare

Detto come cresce, resta da dire che cosa StyleGAN aggiunge, e la risposta sta
in una sigla: **AdaIN**, *adaptive instance normalization*, cioè «taratura
adattiva, una corsia alla volta». Le corsie sono la parte da spiegare.

Il gesto ha due tempi. Primo tempo, si azzera: dentro il falsario, a ogni
livello, il segnale viaggia in tante corsie parallele (le stesse pile di valori
della DCGAN, che in gergo si chiamano *canali*), e di ciascuna corsia si prende il livello medio e l'ampiezza
delle sue oscillazioni e li si riporta a zero e a uno. Tutte le corsie escono
da lì con la stessa taratura, come un mixer con tutti i cursori rimessi in
posizione neutra. Secondo tempo, si riassegna: a ciascuna corsia si rimette un
livello medio e un'ampiezza, e questa volta a dirli è lo **stile**, cioè la
manciata di numeri consegnata a quel livello. Non c'è nessun ordine impartito
all'immagine, c'è una taratura del mixer: ed è per questo che cambiando lo
stile dei primi livelli cambia la posa e cambiando quello degli ultimi cambiano
le lentiggini, perché nei primi livelli le corsie decidono cose grosse e negli
ultimi cose fini.

Il meccanismo non nasce qui. Viene dal trasferimento di stile fra immagini,
dove Xun Huang e Serge Belongie {cite}`huang2017arbitrary` l'avevano proposto
proprio per prendere il «carattere» di un quadro e appiccicarlo a una
fotografia: allineare media e ampiezza delle corsie del contenuto a quelle
dello stile. StyleGAN se lo porta dentro il generatore e lo usa livello per
livello, ed è il motivo per cui il controllo esce separato per scala.

### La goccia

Le immagini di StyleGAN avevano un difetto che si vedeva a occhio nudo, e nella
maggior parte dei casi anche senza cercarlo: una macchia a forma di goccia
d'acqua, sempre uguale, da qualche parte nel quadro. Quando nel quadro finito
non si vedeva c'era comunque nei passaggi intermedi dentro il falsario, e lì
c'era praticamente sempre. Per un anno è stata una di quelle cose che si vedono
e non si spiegano, e a renderla un enigma c'era anche il fatto che l'esperto,
che sta lì apposta per accorgersi di ciò che non torna, se la lasciava passare.

La spiegazione, quando arriva {cite}`karras2020analyzing`, è la parte
istruttiva. La goccia non è un errore del falsario: è il falsario che **aggira
il proprio impianto**. Il primo tempo di AdaIN, quello che rimette tutte le
corsie alla stessa taratura, butta via un'informazione che al falsario serve,
cioè quanto una corsia è forte rispetto alle altre. E allora il falsario si
inventa un trucco da contrabbandiere: fabbrica in un punto qualunque
dell'immagine un picco enorme, così enorme da dominare da solo la statistica
della corsia; quando la normalizzazione divide per quell'ampiezza gonfiata,
tutto il resto della corsia esce schiacciato della quantità che serviva. Il
picco è la goccia. Il costo di una macchia in un angolo, per lui, è minore del
costo di perdere quel controllo.

Il rimedio, che arriva con StyleGAN2, è elegante e vale come regola generale:
invece di tarare le
**attivazioni**, si tarano i **pesi**. Lo stile scala i pesi della convoluzione
(la *modulazione*), e subito dopo i pesi vengono rinormalizzati (la
*demodulazione*) in modo che l'uscita torni ad ampiezza unitaria. Il risultato
sulla carta è quasi lo stesso, ma la seconda operazione non guarda mai il
contenuto dell'immagine: lavora su una tabella di numeri, prima che l'immagine
esista. E un contrabbandiere non può nascondere niente dentro un controllo che
non lo guarda. Le gocce spariscono.

```{admonition} Come si legge questa storia
:class: tip
Un difetto visibile in quasi ogni immagine è rimasto senza spiegazione per un
anno, e quando è arrivata la spiegazione non diceva «c'è un errore»: diceva che
il modello stava **facendo il suo lavoro nel modo che l'impianto gli
permetteva**. È la differenza fra il debug di un programma, dove qualcosa è
scritto storto, e la diagnosi di un modello, dove di solito non c'è niente di
storto e c'è invece un obiettivo che premia una strada che non avevamo
previsto. Quando una rete fa una cosa strana e ripetuta, la domanda che paga è
che cosa ci stia guadagnando, più che dove sia il guasto; il
{doc}`capitolo sull'interpretabilità </Interpretabilita/overview>` ne fa un
mestiere.
```

Quel rimedio, e le altre due revisioni che lo stesso lavoro si porta dietro,
meritano di essere guardati da vicino.

`````{tab} Elementare

Le altre due revisioni sono queste. La prima è che **la crescita per gradini
viene messa da parte**. Serviva a
tenere in piedi un addestramento che nel frattempo si era imparato a tenere in
piedi in altri modi, e in cambio faceva un danno: inchiodava i dettagli a
posizioni fisse sul foglio. Nei volti che si muovono si vedeva benissimo,
perché i denti restavano orientati verso l'obiettivo invece di seguire la
testa. Al posto dei gradini, dentro ciascuna rete si aprono delle scorciatoie
che portano il segnale da un livello all'uscita senza passare per tutti gli
altri, e le due reti non prendono lo stesso genere di scorciatoia.

La seconda è una regola nuova che chiede al falsario di **camminare a passo
costante**: spostare le manopole di un tanto deve cambiare l'immagine di un
tanto, né di più né di meno, dovunque ci si trovi.
Serve a fare immagini migliori, e regala un mestiere in più: un falsario che
cammina a passo costante è molto più facile da percorrere **al contrario**,
cioè da usare per scoprire quali manopole produrrebbero una fotografia che
abbiamo già in mano, e quindi per dire se un volto l'ha fatto lui.

`````

`````{tab} Superiore

L'operazione che sparisce è AdaIN, che in StyleGAN {cite}`karras2019style`
agisce separatamente su ogni mappa di attivazioni $\mathbf{x}_i$:

$$
\mathrm{AdaIN}(\mathbf{x}_i, \mathbf{y}) =
y_{s,i}\, \frac{\mathbf{x}_i - \mu(\mathbf{x}_i)}{\sigma(\mathbf{x}_i)}
+ y_{b,i},
$$

dove $\mathbf{x}_i$ è l’$i$-esima mappa di attivazioni, $\mu$ e $\sigma$ sono
media e deviazione standard **della mappa stessa**, e la coppia di vettori
$(\mathbf{y}_s, \mathbf{y}_b)$ è lo stile, ricavato da $\mathbf{w}$ con una
trasformazione affine appresa; $y_{s,i}$ e $y_{b,i}$, tondi perché sono due
numeri e non due vettori, sono le loro componenti sul canale $i$. Il varco del
contrabbandiere è quella divisione per $\sigma(\mathbf{x}_i)$, il punto in cui il
calcolo si lascia dominare dai valori prodotti.

Al suo posto la modulazione scala i pesi della convoluzione per lo stile, e la
demodulazione li rinormalizza sotto l'ipotesi che gli ingressi siano
indipendenti e a varianza unitaria:

$$
w'_{ijk} = s_i \, w_{ijk},
\qquad
w''_{ijk} = \frac{w'_{ijk}}{\sqrt{\sum_{i',k'} \big(w'_{i'jk'}\big)^2 + \epsilon}},
$$

dove $i$ indicizza i canali d'ingresso, $j$ quelli d'uscita, $k$ le posizioni
spaziali del filtro, $s_i$ è la scala dettata dallo stile ed $\epsilon$ evita la
divisione per zero. Questi $w$ sono i **pesi della convoluzione**, e non hanno
niente a che vedere con il $\mathbf{w}$ di poche righe sopra, che è il vettore
dello spazio latente intermedio: sono due notazioni consolidate dello stesso
paper. Gli apici al denominatore non sono decorativi: la somma
corre su tutti i canali d'ingresso e su tutte le posizioni **a canale d'uscita
$j$ fissato**, cioè è la norma dell'intero filtro che produce il canale $j$, e
$i'$ e $k'$ scorrono mentre l’$i$ e il $k$ del numeratore restano fermi.
L'ipotesi statistica è il punto: la demodulazione non misura le attivazioni
vere, le assume, e per questo non offre nessun canale in
cui nascondere segnale. Gli autori la dichiarano per quello che è, cioè **più
debole** dell'instance normalization proprio perché poggia su ipotesi sul
segnale invece che sul contenuto effettivo delle mappe: il controllo
sull'ampiezza vale in media e non su ogni singolo esempio. Ed è quel «in media»
a chiudere il varco.

Lo stesso lavoro rivede la crescita progressiva, che a quel punto risolveva un
problema di stabilità già risolto altrove e in cambio dava ai dettagli una
preferenza per le posizioni fisse (i denti che restano allineati alla macchina
fotografica invece di seguire la posa). Al suo posto va una coppia
**asimmetrica**, e l'asimmetria è il risultato: fra le nove combinazioni
provate, il generatore vuole connessioni *skip* e il discriminatore
connessioni **residue**, mentre un generatore residuo peggiora le cose.

Arriva poi la **path length regularization**. L'ideale che insegue è che un
passo di ampiezza fissa in $\mathcal{W}$ produca nell'immagine un cambiamento
di ampiezza fissa, quale che sia il punto di partenza e quale che sia la
direzione; lo scarto da quell'ideale si misura sui gradienti rispetto a
$\mathbf{w}$ di una proiezione casuale dell'immagine, e si penalizza la loro
distanza da una costante:

$$
\mathbb{E}_{\mathbf{w},\, \mathbf{u} \sim \mathcal{N}(\mathbf{0}, \mathbf{I})}
\Big( \big\lVert \mathbf{J}_{\mathbf{w}}^{\top} \mathbf{u} \big\rVert_2 - a \Big)^2,
\qquad
\mathbf{J}_{\mathbf{w}} = \frac{\partial g(\mathbf{w})}{\partial \mathbf{w}},
$$

dove $g$ è il generatore, $\mathbf{u}$ è l'immagine di rumore gaussiano su cui
si proietta (il paper la chiama $\mathbf{y}$, lettera che qui è già lo stile) e
$a$ non è un iperparametro ma una media mobile
esponenziale delle lunghezze osservate, cioè un bersaglio che il termine si
sceglie da solo strada facendo. Lo jacobiano non si calcola mai per esteso:
basta l'identità $\mathbf{J}_{\mathbf{w}}^{\top}\mathbf{u} =
\nabla_{\mathbf{w}}\big(g(\mathbf{w}) \cdot \mathbf{u}\big)$, che è una normale
retropropagazione. È un vincolo di buon condizionamento della mappa
latente-immagine, e ha un effetto collaterale utile dichiarato dagli autori: i
generatori così regolarizzati sono molto più facili da **invertire**, cioè da
usare al contrario per trovare il $\mathbf{w}$ che produce una data fotografia,
e questo permette di attribuire un'immagine generata alla rete che l'ha fatta.

`````

## pix2pix e CycleGAN: tradurre le immagini

Se condizioniamo una GAN non su un'etichetta ma su un’*intera immagine*,
otteniamo un traduttore visivo: schizzo → foto, mappa → satellite, giorno →
notte. È **pix2pix** {cite}`isola2017image`. Ha però un vincolo: servono
coppie allineate, cioè lo stesso soggetto ripreso nei due mondi che si vogliono
tradurre l'uno nell'altro (in gergo, i due **domini**), e coppie così sono
difficili da procurare. **CycleGAN** {cite}`zhu2017unpaired` rimuove il
vincolo, con la regola di {numref}`fig-cyclegan`.

```{figure} ../figures/cyclegan-ciclo.svg
:name: fig-cyclegan
:alt: "Diagramma del ciclo di CycleGAN: una foto x viene tradotta dal generatore G in stile Monet, poi il generatore F la riporta indietro, ottenendo una ricostruzione che deve coincidere con la x di partenza."
:width: 90%

Il vincolo di andata e ritorno (in inglese *cycle-consistency*): tradurre e poi
ritradurre deve riportare al punto di partenza.
```

`````{tab} Elementare
CycleGAN impara a trasformare foto in quadri di Monet (e viceversa) senza mai
vedere una foto e il suo quadro corrispondente: gli bastano due mucchi
separati, tante foto e tanti Monet. Il trucco è il vincolo di andata e ritorno:
se prendo una foto, la converto in "stile Monet" e poi la riconverto in foto,
devo ritrovare la foto di partenza.

Detta così, la regola sembra avere una scappatoia grande come una casa: al traduttore converrebbe **non cambiare niente**, restituire la foto tale e quale e vincere senza fatica. Non gli conviene, perché il vincolo di andata e ritorno non gioca da solo: dall'altra parte c'è sempre un esperto, uno per dominio, addestrato a distinguere i veri Monet dai finti Monet. Chi non dipinge viene smascherato da lui. Le due regole si tengono a vicenda: l'esperto obbliga a cambiare stile, il ciclo obbliga a non stravolgere il contenuto.

Resta una scappatoia più fine, e i falsari la trovano. Si può dipingere un Monet vero e nasconderci dentro la foto, in una trama di puntini troppo debole perché l'occhio la veda: al ritorno il compagno rilegge la trama e ricostruisce la foto identica. L'esperto è contento e il giro si chiude, ma nessuno ha promesso che nel quadro ci siano gli stessi alberi della foto. Un andata e ritorno perfetto non dimostra che il soggetto abbia fatto il viaggio.
`````

`````{tab} Superiore
Si addestrano due generatori, $G:\mathcal{X}\to\mathcal{Y}$ e $F:\mathcal{Y}\to\mathcal{X}$, con due discriminatori. Oltre alle due perdite avversarie si aggiunge la **cycle-consistency loss**:

$$
\mathcal{L}_{\text{cyc}}=\mathbb{E}_{\mathbf{x}}\big\lVert F(G(\mathbf{x}))-\mathbf{x}\big\rVert_1
+\mathbb{E}_{\mathbf{y}}\big\lVert G(F(\mathbf{y}))-\mathbf{y}\big\rVert_1 .
$$

I termini misurano, in norma $\ell_1$, quanto la doppia traduzione si discosta dall'originale: $G(\mathbf{x})$ porta $\mathbf{x}$ nel dominio $\mathcal{Y}$, $F$ lo riporta indietro, e il risultato deve coincidere con $\mathbf{x}$. Nell'obiettivo completo la $\mathcal{L}_{\text{cyc}}$ non compare da sola ma pesata da un coefficiente $\lambda$ (nel paper, $\lambda = 10$) accanto alle due perdite avversarie: il rapporto fra le due spinte è un iperparametro, e non dei più innocui.

Questo vincolo rende superfluo l'allineamento a coppie. Che poi «ancori il contenuto» è vero solo in parte, ed è un limite noto: Chu, Zhmoginov e Sandler {cite}`chu2017cyclegan` hanno mostrato che CycleGAN impara a soddisfare il ciclo **nascondendo** l'immagine di partenza dentro quella tradotta, come un segnale ad alta frequenza quasi invisibile a occhio, che $F$ poi rilegge per ricostruire l'originale. Il ciclo si chiude, ma per steganografia invece che per conservazione del soggetto: una $\mathcal{L}_{\text{cyc}}$ bassa non è di per sé una garanzia di fedeltà.
`````

## Applicazioni

L'onda applicativa è stata vasta. La **super-risoluzione**: SRGAN
{cite}`ledig2017photo` ricostruisce dettagli plausibili in immagini a bassa
risoluzione, usata poi nel restauro fotografico e nell’*upscaling*
(l'ingrandimento di un'immagine senza che diventi sgranata). La parola
"plausibili" va presa alla lettera, ed è il limite dello strumento: i dettagli
che nella foto piccola non c'erano la rete non li recupera, li **inventa** in
modo verosimile. Su una foto di famiglia è una scelta estetica; su una lastra
medica o su un fotogramma di sorveglianza è una trappola, perché il risultato
ha l'aria di un'informazione e non lo è. La **generazione
di dati sintetici**: volti, lastre mediche, scene stradali per addestrare
altri modelli quando i dati reali sono scarsi o sensibili, con l'avvertenza
che un dato sintetico eredita le distorsioni di chi l'ha generato (i *bias*: se
il generatore ha visto soltanto volti chiari, soltanto quelli saprà fare).

E l’**arte**. Nel 2018 il ritratto *Edmond de Belamy*, prodotto con una GAN dal
collettivo francese Obvious, fu battuto da Christie's per 432.500 dollari, con
una stima di partenza di 7.000–10.000: la casa d'aste lo presentava come il
primo ritratto generato da un algoritmo mai arrivato all'asta. Il dibattito
sull'autorialità dell'arte generativa si aprì proprio lì, e con un'ironia
utile: buona parte del codice e del lavoro sui dati era di Robbie Barrat, un
altro artista, cosa che Obvious riconobbe pubblicamente dopo l'asta. La domanda
"di chi è l'opera" nasce già con due risposte possibili prima ancora di
arrivare alla macchina.

## VQ-GAN: il duello che fabbrica un alfabeto

Tutte le varianti viste finora cambiano il regolamento del duello lasciandone
intatto lo scopo: alla fine esce un'immagine, e a farla è il falsario.
L'ultima che raccontiamo cambia lo scopo. Il duello non serve più a fare
immagini: serve a fabbricare un **alfabeto** con cui scriverle. E chi poi le
scrive è una macchina che il lettore conosce da tempo, quella che indovina il
simbolo successivo.

Il ragionamento parte da un desiderio. Un Transformer sa continuare qualunque
cosa gli si dia in fila, purché sia una fila corta di simboli presi da un
elenco finito: è così che scrive testo, ed è così che, nella
{doc}`sezione sulla generazione di suono e musica </Audio/generazione-audio>`,
ha scritto musica. Un'immagine non è né l'una né l'altra cosa. Non è un elenco
finito, perché i suoi puntini sono numeri che variano con continuità; e non è
corta, perché di puntini ce ne sono centinaia di migliaia. Servono due
riduzioni, e per tutte e due il libro ha già gli attrezzi: un **autoencoder**
per accorciare, e un **codebook** per rendere finito, cioè la tavolozza di
pezzetti-tipo dei codec audio, quella del VQ-VAE {cite}`oord2017neural`.

Il guaio è che le due riduzioni si mordono la coda. Un Transformer paga
l'attenzione col quadrato della lunghezza della fila (fila doppia, conto
quadruplo), quindi la fila va accorciata tanto; ma comprimere tanto, con una
rete addestrata a somigliare all'originale puntino per puntino, produce
immagini molli, perché la strada più sicura per somigliare a tutto è fare la
media di tutto. È qui che entra il duello. Patrick Esser, Robin Rombach e
Björn Ommer {cite}`esser2021taming` cambiano due cose al compressore. Primo: al
posto del conto puntino per puntino mettono un giudizio **percettivo**, che
misura la somiglianza come la valuterebbe un occhio (due fili d'erba
diversi nello stesso prato sono la stessa cosa, una scritta storta no).
Secondo: gli mettono contro un esperto. A quel punto il compressore non può più
cavarsela con la media, perché una media l'esperto la riconosce, e con lo stesso
accorciamento le immagini tornano nitide.

I numeri dicono quanto pesa il trucco. Con un fattore di riduzione $f = 16$
un'immagine di $256 \times 256$ diventa una griglia di $16 \times 16$, cioè
**256 simboli** presi da un catalogo di 1.024 voci; il VQ-VAE-2, uscito un anno
e mezzo prima, a parità di fedeltà nel rimettere insieme l'immagine ne chiedeva
5.120. Duecentocinquantasei
simboli sono una fila che un Transformer digerisce senza fatica, e da lì in poi
generare un'immagine è, alla lettera, scrivere una frase di 256 parole.

`````{tab} Elementare

Un quadro si può dettare al telefono, se chi ascolta ha lo stesso catalogo che
hai tu: mille tessere di mosaico numerate. Il quadro lo copri con una griglia di
sedici caselle per lato, per ogni casella scegli la tessera del catalogo che le
somiglia di più, e detti i numeri: duecentocinquantasei numeri, e dall'altra
parte qualcuno rimonta il quadro. Il catalogo è l'alfabeto, i numeri sono la
frase.

Chi rimonta il quadro non ha bisogno che i numeri vengano da un quadro vero: se
qualcuno gliene inventa una fila plausibile, lui la rimonta lo stesso. E
inventare file plausibili di simboli è il mestiere della macchina che completa
le frasi. Un'immagine nuova diventa una frase nuova.

Il catalogo però va fatto bene, e qui torna il duello. Se le tessere si scelgono
soltanto col criterio «somiglia», il catalogo si riempie di tessere sbiadite:
davanti a un dubbio la scelta più prudente è sempre il grigio medio, che
somiglia un po’ a tutto e non è niente. Mettendo un esperto a bocciare le
ricostruzioni molli, le tessere restano nette. È il mestiere di sempre, ma
stavolta il prodotto dell'esperto è un alfabeto e non un'immagine.

Un'avvertenza: il catalogo è **un soffitto**. Ciò che nessuna delle mille
tessere sa dire, nessuno lo recupera più a valle, per quanto bravo sia chi
scrive le frasi. E le tessere devono restare grosse: se ciascuna coprisse un
pezzetto minuscolo, quei duecentocinquantasei numeri basterebbero per un angolo
del quadro, e chi rimonta si ritroverebbe con due mezze facce che non si
guardano. Per i quadri grandi si detta a finestra, un riquadro per volta:
funziona finché il quadro è fatto di roba che si somiglia dappertutto, un prato
o una città; su una figura al centro bisogna dire anche in che punto si sta
lavorando.

`````

`````{tab} Superiore

L'encoder $E$ produce $\hat{\mathbf{z}} = E(\mathbf{x}) \in \mathbb{R}^{h
\times w \times d}$; ogni vettore spaziale è quantizzato al più vicino elemento
del codebook $\mathcal{C} = \{\mathbf{e}_1, \dots, \mathbf{e}_K\}$ con la
regola e lo *straight-through estimator* già visti per il VQ-VAE
{cite}`oord2017neural`. La differenza sta nella loss del primo stadio, e sono
due mosse distinte: la ricostruzione $\ell_2$ sui pixel viene **sostituita** da
una **loss percettiva**, e sopra si **aggiunge** una **loss avversaria** con un
discriminatore *patch-based*, quello di pix2pix {cite}`isola2017image`, che
giudica riquadri invece dell'immagine intera e quindi valuta la resa locale più
che il contenuto globale. I due termini che tengono agganciati codebook ed
encoder (la perdita sul codebook e la *commitment loss*) restano quelli del
VQ-VAE.

L'effetto dichiarato è sul **fattore di compressione ammissibile**: con
$f = 16$ e $K = 1024$ un'immagine $256 \times 256$ si riduce a $16 \times 16 =
256$ indici, contro i $5\,120 = 32^2 + 64^2$ della gerarchia a due livelli di
VQ-VAE-2, a fedeltà di ricostruzione dello stesso ordine (il modello ImageNet
di punta del lavoro tiene $f = 16$ e sale a $K = 16\,384$).

Sul valore di $f$ conviene riportare esattamente ciò che il paper misura,
perché le due metà del compromesso vengono da due esperimenti diversi. Verso
l'alto: gli autori osservano che oltre un certo numero di dimezzamenti la
qualità della ricostruzione degrada, e la soglia dipende dal dataset. Verso il
basso: tenendo **fissa** la lunghezza della sequenza a $16 \times 16 = 256$ e
variando $f \in \{1, 2, 8, 16\}$, la porzione d'immagine che quei 256 simboli
coprono si stringe al calare di $f$, e sotto $f = 16$ le strutture globali non
reggono più (a $f = 8$ escono facce mezze barbute, con punti di vista
incoerenti da una zona all'altra). Il vincolo dal basso, quindi, non è che la
fila si allunghi: a fila costante è il campo visivo che si restringe.

Il secondo stadio è un Transformer autoregressivo sugli indici, addestrato a
massimizzare $\sum_i \log p(s_i \mid s_{<i})$ sulla sequenza serializzata in
ordine di scansione. Per risoluzioni oltre quella di addestramento il
campionamento avviene **a finestra scorrevole**, condizionando ogni blocco sul
contesto che ricade nella finestra: è ciò che permette immagini di dimensione
arbitraria a lunghezza di contesto fissa. Gli autori dichiarano anche a quale
condizione funziona, ed è la parte che conviene non perdere: il contesto
disponibile basta finché le statistiche del dataset sono all'incirca invarianti
per traslazione, oppure finché c'è un condizionamento spaziale che regge la
coerenza. Dove la condizione cade (sintesi non condizionata su dati allineati)
si rimedia condizionando sulle coordinate.

Due fili vanno tirati, perché il libro li riprende. Il primo: la stessa ricetta
di perdite (ricostruzione percettiva, avversaria di patch, più un termine che
disciplina il latente) è quella con cui è addestrato l'autoencoder di **Stable
Diffusion**, e non per caso, dato che Esser, Rombach e Ommer sono tutti e tre
fra i suoi autori; là però il latente resta **continuo**, perché il termine che
disciplina è una KL leggera invece di una quantizzazione, e a comporre non
pensa un Transformer autoregressivo ma la diffusione. VQ-GAN è la fucina in cui
quella ricetta è stata forgiata. Il secondo: l'idea di trattare un'immagine
come una sequenza di simboli non è nuova per chi legge, perché la {doc}`sezione
sulla fusione precoce </VisioneLinguaggio/fusione-precoce-tardiva>` l'ha già
usata per i generatori multimodali; qui se ne vede l'officina, cioè da dove
arriva l'alfabeto e a che prezzo lo si fabbrica.

`````

## Il passaggio di testimone ai modelli di diffusione

Verso il 2021 il primato cambia mano, e a dare il nome al sorpasso è il paper
*Diffusion Models Beat GANs on Image Synthesis* {cite}`dhariwal2021diffusion`,
che lo argomenta sulla qualità delle immagini. Ma la ragione per cui il
passaggio è stato così rapido sta altrove, e sono i due difetti che questo
capitolo ha già raccontato: i **modelli di diffusione** si addestrano con la
stessa tranquillità di una rete a cui si mostra la risposta giusta, e non
conoscono il *mode collapse*. L'idea è opposta a quella avversaria: si insegna
alla rete a ripulire un'immagine sporcata da una grana casuale, partendo da
un'immagine fatta di sola grana. («Rumore» è la stessa parola usata finora, ma
qui indica una cosa diversa: non i numeri casuali in ingresso, bensì la
sporcizia sparsa sopra un'immagine.) Su questa base nascono **Stable Diffusion**
{cite}`rombach2022high` e **DALL·E 2** (OpenAI, 2022): a entrambi si descrive a
parole quello che si vuole ("un gatto nero seduto su un muro al tramonto") e
loro lo disegnano, ed è così che la generazione di immagini su richiesta è
arrivata al grande pubblico. Stable Diffusion in più fa il lavoro di
ripulitura non sull'immagine a grandezza naturale ma su una sua versione
ridotta e compatta, che occupa molta meno memoria: è la ragione per cui gira
anche su un computer di casa. Del meccanismo parla il
{doc}`capitolo sui modelli di diffusione </ModelliDiffusione/overview>`.

Le GAN non sono scomparse, e il motivo è la velocità: una GAN produce
l'immagine in un colpo solo, un unico passaggio attraverso il generatore,
mentre la diffusione parte dal rumore e lo ripulisce un po’ per volta,
ripetendo l'operazione decine di volte. Il vantaggio era così evidente che la
ricerca sulla diffusione ha passato anni a rincorrerlo, imparando a ottenere lo
stesso risultato in pochi passi invece che in molti; e per riuscirci ha spesso
rimesso in gioco un discriminatore, cioè proprio l'idea avversaria di questo
capitolo.

C'è di più, ed è la ragione migliore per aver letto questo capitolo anche
volendo usare soltanto la diffusione. La sezione su VQ-GAN l'ha anticipato, e
adesso conviene dirlo per esteso: un discriminatore è servito a costruire
**un pezzo di** Stable Diffusion. Alla fine di tutto il lavoro c'è una parte
che riporta quella versione ridotta e compatta ai pixel veri e propri, e la si
chiama decodificatore: ecco, è stata addestrata anche con una loss avversaria,
cioè con un esperto contro. Finito l'addestramento l'esperto se ne va, come nel
duello di questo capitolo, ed
è quella parte a tenere nitide le ricostruzioni. Il duello, insomma, è passato
dal centro della scena a un ruolo di manutenzione, invece di finire in
soffitta. Ma il centro di gravità si è spostato.

```{admonition} Nota etica: i deepfake
:class: warning
La stessa tecnologia che genera volti fotorealistici genera **deepfake**:
video e audio falsi ma credibili di persone reali. Le implicazioni sono serie:
disinformazione, frodi, abusi non consensuali. Chi lavora con questi modelli
ha una responsabilità concreta: verificare le fonti, sostenere i sistemi che
marchiano un contenuto generato con una filigrana invisibile (il
*watermarking*) e ne registrano la provenienza, e ricordare che "verosimile"
non significa "vero". La potenza generativa e la vigilanza critica devono
crescere insieme.
```

Le varianti che abbiamo visto sono altrettante modifiche al regolamento del
duello, e conviene ripassarle così.

`````{tab} Elementare

```{admonition} Da ricordare
:class: important
- La **DCGAN** cambia il modo in cui le due reti guardano le immagini, dando
  loro dei filtri che scorrono sull'immagine invece di trattarla come una lista
  di numeri: è la variante che fa smettere alle GAN di produrre macchie.
- La **conditional GAN** aggiunge il timone: insieme al rumore si consegna
  un'etichetta ("voglio un 7"), e la ricevono tutti e due, perché anche
  l'esperto deve poter dire "sarà pure un bel disegno, ma non è un 7".
- **StyleGAN** costruisce il volto a livelli e consegna a ciascun livello la
  propria manopola: da lì il controllo separato di posa, lineamenti e
  lentiggini. La risoluzione da fotografia, invece, era già stata conquistata
  dal modello che l'ha preceduta, la Progressive GAN.
- Sotto il cofano, tre meccanismi. Si **cresce per gradini**, da sedici puntini
  a un milione, e a ogni gradino si passa in **dissolvenza** invece che di
  colpo. La manopola non impartisce ordini: **rimette tutte le corsie del
  segnale alla stessa taratura e poi le ritara** secondo lo stile, come un
  mixer. E la famosa macchia a goccia delle immagini di StyleGAN era il
  falsario che si fabbricava un picco enorme per contrabbandare,
  attraverso quella taratura, un'informazione che la taratura gli toglieva.
  StyleGAN2 la fa sparire tarando i **pesi** invece del segnale, cioè con un
  controllo che l'immagine non la guarda; e con la stessa revisione mette da
  parte anche la crescita per gradini, sostituita da scorciatoie che portano il
  segnale da un livello all'uscita.
- **pix2pix** e **CycleGAN** traducono un'immagine in un'altra (schizzo in
  foto, foto in Monet); la seconda ci riesce senza coppie di immagini
  corrispondenti, grazie alla regola dell'andata e ritorno, che però va tenuta
  insieme all'esperto: da sola si lascia aggirare. E nemmeno insieme garantisce
  il soggetto, perché il falsario può nascondere la foto dentro il quadro, in una
  trama che l'occhio non vede, e ritrovarsela al ritorno.
- **VQ-GAN** cambia lo scopo del duello: l'esperto non serve più a fare
  immagini, serve a fabbricare un **alfabeto**. Un'immagine diventa una fila di
  256 simboli presi da un catalogo di mille, e da lì generarne una nuova è
  scrivere una frase nuova, con la stessa macchina che completa il testo. Senza
  l'esperto il catalogo si riempirebbe di tessere sbiadite, perché per
  «somigliare» conviene sempre il grigio medio.
- Dal 2021 il testimone passa ai **modelli di diffusione** (Stable Diffusion,
  DALL·E 2). Le GAN restano dove conta la velocità (un colpo solo contro decine
  di passaggi), e l'idea avversaria continua a servire per **costruire** gli
  strumenti di oggi, anche quando poi nel prodotto finito l'esperto non c'è
  più.
```

`````

`````{tab} Superiore

```{admonition} Da ricordare
:class: important
- La **DCGAN** porta la convoluzione nelle GAN e ne fissa la ricetta di
  addestramento (convoluzioni con *stride* al posto del pooling, batchnorm
  salvo lo strato d'uscita di $G$ e quello d'ingresso di $D$, $\tanh$ in uscita);
  la **conditional GAN** aggiunge il controllo tramite una variabile ausiliaria
  $y$ passata a entrambe le reti.
- **StyleGAN** introduce lo spazio latente intermedio $\mathcal{W}$, la
  modulazione per strato via AdaIN {cite}`huang2017arbitrary` e il rumore
  per-livello: il suo contributo è il controllo *scale-specific*, mentre la
  scala di risoluzione è ereditata dalla Progressive GAN
  {cite}`karras2018progressive`, che la ottiene con la **dissolvenza** sui
  *toRGB*/*fromRGB* più equalized learning rate, normalizzazione pixelwise e
  minibatch standard deviation.
- **StyleGAN2** {cite}`karras2020analyzing` diagnostica gli artefatti a goccia
  come contrabbando di segnale attraverso l'instance normalization, e li elimina
  con **modulazione e demodulazione dei pesi**: la rinormalizzazione agisce sui
  filtri sotto ipotesi statistiche, mai sulle attivazioni vere, quindi non offre
  un canale nascosto. Con essa cade anche la crescita progressiva, sostituita
  da una coppia asimmetrica (*skip* nel generatore, connessioni **residue** nel
  discriminatore), e arriva la *path length regularization*, che spinge un
  passo di ampiezza fissa in $\mathcal{W}$ a produrre un cambiamento di
  ampiezza fissa nell'immagine: migliora il condizionamento della mappa
  $\mathcal{W} \to$ immagine e rende il generatore molto più facile da
  invertire.
- **pix2pix** e **CycleGAN** fanno traduzione immagine-a-immagine (la seconda
  senza coppie, grazie alla *cycle-consistency* pesata da un $\lambda$); il
  ciclo però si può chiudere per steganografia, quindi non certifica la fedeltà
  al soggetto.
- **VQ-GAN** {cite}`esser2021taming` sposta l'obiettivo del duello sul
  **vocabolario**: encoder e codebook come nel VQ-VAE, ma primo stadio in cui
  la loss percettiva sostituisce l’$\ell_2$ e un discriminatore *patch-based*
  si aggiunge, il che alza il
  fattore di compressione ammissibile ($f=16$: $256\times256 \to 16\times16 =
  256$ indici su $K = 1024$, contro i 5.120 di VQ-VAE-2). Sopra, un Transformer
  autoregressivo sugli indici, a finestra scorrevole oltre la risoluzione di
  addestramento, e regge finché le statistiche del dataset sono all'incirca
  invarianti per traslazione o c'è un condizionamento spaziale. È la ricetta di
  perdite poi riusata nell'autoencoder di Stable
  Diffusion, con latente continuo invece che quantizzato.
- Dal 2021 i **modelli di diffusione** (Stable Diffusion, DALL·E 2) raccolgono
  il testimone della generazione di immagini; le GAN restano rilevanti per
  velocità di campionamento e come componente ibrida, decodificatori dei modelli
  latenti compresi.
```

`````

Il duello esce da questo capitolo ridimensionato ma non archiviato, ed è quello
il lascito: non una famiglia di architetture, ma un modo di addestrare che
sopravvive dentro sistemi che non si chiamano più GAN. La domanda però resta
intera, fabbricare dati nuovi e plausibili senza un originale con cui
confrontarsi, e i capitoli che seguono sono altrettante risposte diverse alla
stessa domanda. La prima è «Modelli di diffusione», che al posto di due reti
che si sfidano mette un dato ridotto a rumore e una rete che rifà la strada
all'indietro.
