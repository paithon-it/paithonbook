# Evoluzioni e applicazioni

Le prime immagini generate da una GAN, nel 2014, erano cifre sgranate e
faccine indistinte, appena qualche decina di pixel di lato. Ian Goodfellow le
mostrava con orgoglio, ma nessuno le avrebbe scambiate per fotografie. Cinque
anni dopo, il sito già incontrato in apertura di capitolo,
*thispersondoesnotexist.com*, serve a ripetizione volti fotorealistici di
persone che non esistono. In mezzo c'è la storia di questa sezione: una
sequenza di idee (quasi una all'anno) che ha trasformato un'intuizione fragile
in una delle famiglie di modelli generativi più influenti del decennio.
Ripercorriamola, seguendo il filo delle idee più che il calendario.

I due personaggi restano quelli: un falsario che dipinge e un esperto che
giudica. Quello che cambia, di variante in variante, è come sono fatti dentro
(DCGAN), che cosa gli si mette in mano oltre al rumore (conditional GAN), da
dove il falsario prende le proprie decisioni (StyleGAN) e che cosa gli si
chiede di conservare mentre dipinge (CycleGAN). Ogni variante, insomma, si
legge bene come una modifica al regolamento di quel duello.

## DCGAN: dare occhi alla rete

La prima GAN girava soprattutto su **strati densi** (*fully-connected*), cioè
strati in cui ogni numero in uscita dipende da tutti i numeri in entrata,
tutti allo stesso modo: un'immagine, per uno strato così, è una lista
disordinata di numeri, e il fatto che due pixel siano vicini non gli dice
niente. Ma un'immagine ha una struttura spaziale: pixel vicini sono correlati.
Una variante convoluzionale c'era già nel paper del 2014, e addestrarla era un
terno al lotto: quel che mancava non era l'idea, era una ricetta che stesse in
piedi. Arriva a fine 2015 con la **DCGAN** (*Deep Convolutional GAN*) di
Radford, Metz e Chintala {cite}`radford2016unsupervised`.

`````{tab} Elementare
L'idea è dare alla rete lo strumento giusto per "vedere": la convoluzione,
cioè far scorrere piccoli filtri sull'immagine per riconoscere contorni,
texture, forme. È il discriminatore ad analizzarla così, con gli stessi filtri
di un classificatore qualsiasi; il generatore fa il percorso inverso, e parte
da un pugno di numeri casuali per "gonfiarli" fino a un'immagine intera.

Gonfiare come, se i numeri di partenza sono cento e i puntini d'arrivo un
milione? A ogni passaggio la rete prende la griglia che ha in mano e ne
restituisce una più larga, raddoppiando il lato: da $4\times4$ a $8\times8$, poi
$16\times16$, e così via. I puntini in più non sono copiati da nessuna parte,
sono *decisi*: dove il vecchio puntino era uno solo, il passaggio successivo ne
mette quattro, e quanto ciascuno dei quattro debba essere chiaro o scuro lo
stabiliscono i filtri, che è esattamente ciò che la rete impara. Con questa
architettura le immagini smettono di essere macchie e cominciano ad avere
bordi netti e coerenza.
`````

`````{tab} Superiore
La DCGAN codifica una serie di scelte architetturali diventate standard: il generatore $G$ usa convoluzioni trasposte (*strided transposed convolutions*) per l'upsampling, il discriminatore $D$ usa convoluzioni con *stride*; nessun pooling; *batch normalization* in entrambe le reti, con due eccezioni che il paper stesso impone (niente batchnorm sullo strato di uscita di $G$ e su quello d'ingresso di $D$, dove gli autori riportano oscillazioni dei campioni e instabilità); attivazioni ReLU nel generatore (tranne l'output con $\tanh$) e LeakyReLU nel discriminatore.

Il paper mostra anche che lo spazio latente $\mathcal{Z}$ è **semanticamente strutturato**: aritmetica vettoriale come "uomo con occhiali $-$ uomo $+$ donna" produce, decodificata da $G$, il volto di una donna con occhiali. Con una cautela che il paper stesso dichiara, e che pesa: sui singoli vettori l'operazione è instabile, e il risultato regge mediando i $\mathbf{z}$ di tre esemplari per concetto. La struttura c'è, ma è una proprietà di regioni dello spazio, non di punti singoli; ed è comunque un indizio precoce del fatto che la rete apprende una rappresentazione, non una tabella di memorizzazione.
`````

## Conditional GAN: prendere il controllo

Una GAN base genera "qualcosa di plausibile", ma non possiamo chiederle *cosa*. La **conditional GAN** {cite}`mirza2014conditional`, proposta già a fine 2014, pochi mesi dopo il paper originale, aggiunge il timone: si passa alla rete anche un'etichetta, e la generazione la rispetta.

`````{tab} Elementare
Insieme al rumore casuale forniamo un'informazione in più, per esempio "voglio
un 7" su MNIST (la raccolta di cifre scritte a mano che da decenni si usa per
le prove: settantamila immaginette minuscole, 28 pixel per lato, ciascuna con
accanto la cifra che rappresenta), oppure "un gatto". La richiesta si consegna
alla rete come tutto il resto, cioè sotto forma di numeri: una cifra fra zero e
nove è già un numero, e per un'etichetta a parole basta assegnare a ciascuna
parola il suo, così che "voglio un 7" diventi una manciata di numeri accodata
al rumore. Sia il generatore sia il discriminatore
ricevono questa etichetta: il primo la usa come istruzione, il secondo per
giudicare non solo *se* l'immagine è verosimile, ma se corrisponde davvero
all'etichetta richiesta. Così smettiamo di pescare a caso e cominciamo a
ordinare su misura.
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

Con **StyleGAN** (NVIDIA {cite}`karras2019style`) e il successivo StyleGAN2 il generatore cambia impianto, e non per generare *meglio* ma per generare in modo governabile. È la tecnologia dietro i volti impossibili da smascherare a occhio.

`````{tab} Elementare
StyleGAN non "disegna" il volto tutto in una volta: lo costruisce a livelli,
dal grossolano al fine. Gli strati iniziali decidono posa e forma del viso,
quelli intermedi i lineamenti, quelli finali dettagli come lentiggini e
ciocche di capelli. A ogni livello inietta uno "stile", e la parola merita di
essere sciolta: uno stile, qui, è una manciata di numeri, una specie di
manopola con molte tacche che invece di essere fissata una volta per tutte
all'ingresso viene consegnata al singolo livello e decide come quel livello
lavorerà. Cambiando i numeri consegnati ai primi livelli cambia la posa;
cambiando quelli consegnati agli ultimi cambiano le lentiggini. Per questo si
possono mescolare tratti di volti diversi (la struttura di uno, il colore di
pelle di un altro) come un fotomontaggio impossibile ma perfettamente
coerente: basta prendere le manopole dei primi livelli da un volto e quelle
degli ultimi da un altro.

Che poi il risultato sia una fotografia e non un disegno non dipende dai
livelli: dipende, come sempre in questo capitolo, dall'esperto, che è stato
allenato su fotografie vere e boccia tutto ciò che non lo sembra. I livelli
danno il *controllo*, il duello dà il realismo.
`````

`````{tab} Superiore
L'innovazione è architetturale: una rete di *mapping* trasforma $\mathbf{z}$ in uno spazio latente intermedio $\mathcal{W}$, più disaccoppiato; i vettori di stile $\mathbf{w}$ modulano ogni strato del generatore via *adaptive instance normalization* (AdaIN); rumore stocastico separato controlla i dettagli ad alta frequenza. Il risultato è il controllo *scale-specific*.

La risoluzione $1024\times1024$, invece, StyleGAN la eredita: viene dalla **Progressive GAN** dello stesso gruppo {cite}`karras2018progressive`, che l'anno prima aveva imparato a salire fino a lì facendo crescere le due reti un livello per volta, e che StyleGAN dichiara come propria configurazione di base, «da cui ereditiamo le reti e tutti gli iperparametri». Là si era imparato a *salire* la scala; qui si impara a decidere che cosa succede a ciascun gradino. StyleGAN2 (2020) elimina poi i caratteristici artefatti "a goccia" ridisegnando la normalizzazione: con quella revisione la ricetta si assesta, ed è la forma in cui la famiglia è entrata nell'uso corrente.
`````

## pix2pix e CycleGAN: tradurre le immagini

Se condizioniamo una GAN non su un'etichetta ma su un'*intera immagine*, otteniamo un traduttore visivo: schizzo → foto, mappa → satellite, giorno → notte. È **pix2pix** {cite}`isola2017image`. Ha però un vincolo: servono coppie allineate (lo stesso soggetto nei due domini), difficili da procurare. **CycleGAN** {cite}`zhu2017unpaired` rimuove il vincolo.

```{figure} ../figures/cyclegan-ciclo.svg
:name: fig-cyclegan
:alt: "Diagramma del ciclo di CycleGAN: una foto x viene tradotta dal generatore G in stile Monet, poi il generatore F la riporta indietro, ottenendo una ricostruzione che deve coincidere con la x di partenza."
:width: 90%

Il vincolo di *cycle-consistency*: tradurre e poi ritradurre deve riportare al punto di partenza.
```

`````{tab} Elementare
CycleGAN impara a trasformare foto in quadri di Monet (e viceversa) senza mai vedere una foto e il suo quadro corrispondente: gli bastano due mucchi separati, tante foto e tanti Monet. Il trucco è il vincolo di andata e ritorno ({numref}`fig-cyclegan`): se prendo una foto, la converto in "stile Monet" e poi la riconverto in foto, devo ritrovare la foto di partenza.

Detta così, la regola sembra avere una scappatoia grande come una casa: al traduttore converrebbe **non cambiare niente**, restituire la foto tale e quale e vincere senza fatica. Non gli conviene, perché il vincolo di andata e ritorno non gioca da solo: dall'altra parte c'è sempre un esperto, uno per dominio, addestrato a distinguere i veri Monet dai finti Monet. Chi non dipinge viene smascherato da lui. Le due regole si tengono a vicenda: l'esperto obbliga a cambiare stile, il ciclo obbliga a non stravolgere il contenuto.
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
risoluzione, usata poi nel restauro fotografico e nell'*upscaling*
(l'ingrandimento di un'immagine senza che diventi sgranata). La parola
"plausibili" va presa alla lettera, ed è il limite dello strumento: i dettagli
che nella foto piccola non c'erano la rete non li recupera, li **inventa** in
modo verosimile. Su una foto di famiglia è una scelta estetica; su una lastra
medica o su un fotogramma di sorveglianza è una trappola, perché il risultato
ha l'aria di un'informazione e non lo è. La **generazione
di dati sintetici**: volti, lastre mediche, scene stradali per addestrare
altri modelli quando i dati reali sono scarsi o sensibili, con l'avvertenza
che un dato sintetico eredita i *bias* di chi l'ha generato. E l'**arte**: nel
2018 il ritratto *Edmond de Belamy*, prodotto con una GAN dal collettivo
francese Obvious, fu battuto da Christie's per 432.500 dollari, presentato
dalla casa d'aste come la prima opera generata da un algoritmo mai passata
sotto il martello di una grande casa (la stima di partenza era di
7.000–10.000 dollari). Il dibattito sull'autorialità dell'arte
generativa si aprì proprio lì, e con un'ironia utile: buona parte del codice e
del lavoro sui dati era di Robbie Barrat, un altro artista, cosa che Obvious
riconobbe pubblicamente dopo l'asta. La domanda "di chi è l'opera" nasce già
con due risposte possibili prima ancora di arrivare alla macchina.

## Il passaggio di testimone ai modelli di diffusione

Verso il 2021 il primato cambia mano, e a dare il nome al sorpasso è il paper
*Diffusion Models Beat GANs on Image Synthesis* {cite}`dhariwal2021diffusion`,
che lo argomenta sulla qualità delle immagini. Ma la ragione per cui il
passaggio è stato così rapido sta altrove, e sono i due difetti che questo
capitolo ha già raccontato: i **modelli di diffusione** si addestrano con la
stessa stabilità di un qualunque modello supervisionato, e non conoscono il
*mode collapse*. L'idea è opposta a quella avversaria: si insegna
alla rete a rimuovere gradualmente il rumore da un'immagine, partendo dal
rumore puro fino all'immagine. Su questa base nascono **Stable Diffusion**
{cite}`rombach2022high` e **DALL·E 2** (OpenAI, 2022): a entrambi si descrive a
parole quello che si vuole ("un gatto nero seduto su un muro al tramonto") e
loro lo disegnano, ed è così che la generazione di immagini su richiesta è
arrivata al grande pubblico. Stable Diffusion in più fa il lavoro di
ripulitura non sull'immagine a grandezza naturale ma su una sua versione
ridotta e compatta, che occupa molta meno memoria: è la ragione per cui gira
anche su un computer di casa. Del meccanismo parleremo nel capitolo dedicato
alla diffusione.

Le GAN non sono scomparse, e il motivo è la velocità: una GAN produce
l'immagine in un colpo solo, un unico passaggio attraverso il generatore,
mentre la diffusione parte dal rumore e lo ripulisce un po' per volta,
ripetendo l'operazione decine di volte. Il vantaggio era così evidente che la
ricerca sulla diffusione ha passato anni a rincorrerlo, imparando a ottenere lo
stesso risultato in pochi passi invece che in molti; e per riuscirci ha spesso
rimesso in gioco un discriminatore, cioè proprio l'idea avversaria di questo
capitolo.

C'è di più, ed è la ragione migliore per aver letto questo capitolo anche
volendo usare soltanto la diffusione: un discriminatore lavora
**dentro** Stable Diffusion. Il pezzo che riporta la versione ridotta e
compatta ai pixel veri e propri è addestrato anche con una loss avversaria, ed
è quella parte a tenere nitide le ricostruzioni. Il duello, insomma, non è
finito in soffitta: è passato dal centro della scena a un ruolo di
manutenzione. Ma il centro di gravità si è spostato.

```{admonition} Nota etica: i deepfake
:class: warning
La stessa tecnologia che genera volti fotorealistici genera **deepfake**:
video e audio falsi ma credibili di persone reali. Le implicazioni sono serie:
disinformazione, frodi, abusi non consensuali. Chi lavora con questi modelli
ha una responsabilità concreta: verificare le fonti, sostenere sistemi di
*watermarking* e provenienza dei contenuti, e ricordare che "verosimile" non
significa "vero". La potenza generativa e la vigilanza critica devono crescere
insieme.
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
- **pix2pix** e **CycleGAN** traducono un'immagine in un'altra (schizzo in
  foto, foto in Monet); la seconda ci riesce senza coppie di immagini
  corrispondenti, grazie alla regola dell'andata e ritorno, che però va tenuta
  insieme all'esperto: da sola si lascia aggirare.
- Dal 2021 il testimone passa ai **modelli di diffusione** (Stable Diffusion,
  DALL·E 2). Le GAN restano dove conta la velocità (un colpo solo contro decine
  di passaggi), e un discriminatore continua a lavorare dentro gli strumenti di
  oggi.
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
  modulazione per strato via AdaIN e il rumore per-livello: il suo contributo è
  il controllo *scale-specific*, mentre la scala di risoluzione è ereditata
  dalla Progressive GAN {cite}`karras2018progressive`.
- **pix2pix** e **CycleGAN** fanno traduzione immagine-a-immagine (la seconda
  senza coppie, grazie alla *cycle-consistency* pesata da un $\lambda$); il
  ciclo però si può chiudere per steganografia, quindi non certifica la fedeltà
  al soggetto.
- Dal 2021 i **modelli di diffusione** (Stable Diffusion, DALL·E 2) raccolgono
  il testimone della generazione di immagini; le GAN restano rilevanti per
  velocità di campionamento e come componente ibrida, decodificatori dei modelli
  latenti compresi.
```

`````
