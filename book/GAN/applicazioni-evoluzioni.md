# Evoluzioni e applicazioni

Le prime immagini generate da una GAN, nel 2014, erano cifre sgranate e
faccine indistinte, appena qualche decina di pixel di lato. Ian Goodfellow le
mostrava con orgoglio, ma un occhio distratto le scambiava per rumore. Dieci
anni dopo, siti come *thispersondoesnotexist.com* servono a ripetizione volti
fotorealistici di persone che non esistono. In mezzo c'è la storia di questo
capitolo: una sequenza di idee (quasi una all'anno) che ha trasformato
un'intuizione fragile in una delle famiglie di modelli generativi più
influenti del decennio. Ripercorriamola, seguendo il filo delle idee più che
il calendario.

## DCGAN: dare occhi alla rete

La prima GAN usava strati densi (*fully-connected*), gli stessi che tratterebbero un'immagine come una lista disordinata di numeri. Ma un'immagine ha una struttura spaziale: pixel vicini sono correlati. La svolta arriva a fine 2015 con la **DCGAN** (*Deep Convolutional GAN*) di Radford, Metz e Chintala {cite}`radford2016unsupervised`.

`````{tab} Elementare
L'idea è dare alla rete lo strumento giusto per "vedere": la convoluzione,
cioè far scorrere piccoli filtri sull'immagine per riconoscere contorni,
texture, forme. Il generatore fa il percorso inverso (parte da un pugno di
numeri casuali e li "gonfia" progressivamente fino a diventare un'immagine),
mentre il discriminatore la analizza con gli stessi filtri. Con questa
architettura le immagini smettono di essere macchie e cominciano ad avere
bordi netti e coerenza.
`````

`````{tab} Superiore
La DCGAN codifica una serie di scelte architetturali diventate standard: il generatore $G$ usa convoluzioni trasposte (*strided transposed convolutions*) per l'upsampling, il discriminatore $D$ usa convoluzioni con *stride*; nessun pooling; *batch normalization* in entrambe le reti; attivazioni ReLU nel generatore (tranne l'output con $\tanh$) e LeakyReLU nel discriminatore. Il paper mostra anche che lo spazio latente $z$ è **semanticamente strutturato**: aritmetica vettoriale come "uomo con occhiali $-$ uomo $+$ donna" produce, decodificata da $G$, il volto di una donna con occhiali. Un indizio precoce del fatto che la rete apprende una rappresentazione, non una tabella di memorizzazione.
`````

## Conditional GAN: prendere il controllo

Una GAN base genera "qualcosa di plausibile", ma non possiamo chiederle *cosa*. La **conditional GAN** {cite}`mirza2014conditional`, proposta già a fine 2014, pochi mesi dopo il paper originale, aggiunge il timone: si passa alla rete anche un'etichetta, e la generazione la rispetta.

`````{tab} Elementare
Insieme al rumore casuale forniamo un'informazione in più, per esempio "voglio
un 7" su MNIST, oppure "un gatto". Sia il generatore sia il discriminatore
ricevono questa etichetta: il primo la usa come istruzione, il secondo per
giudicare non solo *se* l'immagine è verosimile, ma se corrisponde davvero
all'etichetta richiesta. Così smettiamo di pescare a caso e cominciamo a
ordinare su misura.
`````

`````{tab} Superiore
Si condiziona il gioco minimax su una variabile ausiliaria $y$ (la classe, o un vettore qualsiasi):

$$
\min_{G}\max_{D}\;\mathbb{E}_{(x,y)\sim p_{\text{dati}}(x,y)}\big[\log D(x,y)\big]
+\mathbb{E}_{y\sim p_y,\;z\sim p_z}\big[\log\big(1-D(G(z,y),\,y)\big)\big].
$$

Qui $p_{\text{dati}}(x,y)$ è la distribuzione **congiunta** di dato e
condizione, scritta con gli argomenti espliciti proprio per distinguerla dalla
marginale sui soli dati che nel resto del capitolo abbiamo indicato con
$p_{\text{dati}}$; $p_y$ è la distribuzione delle condizioni e $z$ il rumore
latente. Entrambe le reti ricevono $y$ come ingresso aggiuntivo, e per questo
lo scriviamo come secondo argomento: $G(z,y)$ è il campione generato coerente
con $y$, e $D(x,y)$ il giudizio del discriminatore sulla coppia. Il paper di
Mirza e Osindero usa la barra ($D(x\mid y)$), ma quella barra non denota una
probabilità condizionata: è solo un ingresso in più, e la virgola lo dice senza
ambiguità.

Il condizionamento è il seme concettuale di quasi tutto ciò che segue: la
traduzione immagine-a-immagine, e in ultima analisi il *text-to-image*, non
sono altro che GAN (o modelli generativi) condizionati su un input sempre più
ricco (da un'etichetta discreta a un'intera frase).
`````

## StyleGAN: il fotorealismo

Il salto di qualità visivo più clamoroso arriva con **StyleGAN** (NVIDIA {cite}`karras2019style`) e il successivo StyleGAN2. È la tecnologia dietro i volti impossibili da smascherare a occhio.

`````{tab} Elementare
StyleGAN non "disegna" il volto tutto in una volta: lo costruisce a livelli,
dal grossolano al fine. Gli strati iniziali decidono posa e forma del viso,
quelli intermedi i lineamenti, quelli finali dettagli come lentiggini e
ciocche di capelli. A ogni livello inietta uno "stile", e per questo si
possono mescolare tratti di volti diversi (la struttura di uno, il colore di
pelle di un altro) come un fotomontaggio impossibile ma perfettamente
coerente.
`````

`````{tab} Superiore
L'innovazione è architetturale: una rete di *mapping* trasforma $z$ in uno spazio latente intermedio $\mathcal{W}$, più disaccoppiato; i vettori di stile $\mathbf{w}$ modulano ogni strato del generatore via *adaptive instance normalization* (AdaIN); rumore stocastico separato controlla i dettagli ad alta frequenza. Il risultato è il controllo *scale-specific* e un fotorealismo a 1024×1024 senza precedenti. StyleGAN2 (2020) elimina i caratteristici artefatti "a goccia" ridisegnando la normalizzazione. È qui che le GAN toccano il loro apice qualitativo sui volti.
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
CycleGAN impara a trasformare foto in quadri di Monet (e viceversa) senza mai vedere una foto e il suo quadro corrispondente: gli bastano due mucchi separati, tante foto e tanti Monet. Il trucco è il vincolo di andata e ritorno ({numref}`fig-cyclegan`): se prendo una foto, la converto in "stile Monet" e poi la riconverto in foto, devo ritrovare la foto di partenza. Questo obbliga la traduzione a cambiare lo stile senza stravolgere il contenuto.
`````

`````{tab} Superiore
Si addestrano due generatori, $G:X\to Y$ e $F:Y\to X$, con due discriminatori. Oltre alle due perdite avversarie si aggiunge la **cycle-consistency loss**:

$$
\mathcal{L}_{\text{cyc}}=\mathbb{E}_{x}\big\lVert F(G(x))-x\big\rVert_1
+\mathbb{E}_{y}\big\lVert G(F(y))-y\big\rVert_1 .
$$

I termini misurano, in norma $\ell_1$, quanto la doppia traduzione si discosta dall'originale: $G(x)$ porta $x$ nel dominio $Y$, $F$ lo riporta indietro, e il risultato deve coincidere con $x$. Questo vincolo rende superfluo l'allineamento a coppie e mantiene il contenuto ancorato durante il cambio di stile.
`````

## Applicazioni

L'onda applicativa è stata vasta. La **super-risoluzione**: SRGAN
{cite}`ledig2017photo` ricostruisce dettagli plausibili in immagini a bassa
risoluzione, usata poi nel restauro fotografico e nell'*upscaling*
(l'ingrandimento di un'immagine senza che diventi sgranata). La **generazione
di dati sintetici**: volti, lastre mediche, scene stradali per addestrare
altri modelli quando i dati reali sono scarsi o sensibili, con l'avvertenza
che un dato sintetico eredita i *bias* di chi l'ha generato. E l'**arte**: nel
2018 il ritratto *Edmond de Belamy*, prodotto da una GAN, fu venduto da
Christie's per 432.500 dollari, aprendo il dibattito sull'autorialità
dell'arte generativa.

## Il passaggio di testimone ai modelli di diffusione

Verso il 2021 il primato cambia mano. Il paper *Diffusion Models Beat GANs on
Image Synthesis* {cite}`dhariwal2021diffusion` segna il sorpasso: i **modelli
di diffusione** superano le GAN in qualità e, soprattutto, in stabilità di
addestramento e diversità dei campioni; le GAN soffrivano di training
instabile e *mode collapse*. L'idea è opposta a quella avversaria: si insegna
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
capitolo. Ma il centro di gravità si è spostato.

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

```{admonition} Da ricordare
:class: important
- La **DCGAN** porta la convoluzione nelle GAN; la **conditional GAN** aggiunge il controllo tramite un'etichetta $y$.
- **StyleGAN** raggiunge il fotorealismo dei volti; **pix2pix** e **CycleGAN** fanno traduzione immagine-a-immagine (la seconda senza coppie, grazie alla *cycle-consistency*).
- Dal 2021 i **modelli di diffusione** (Stable Diffusion, DALL·E 2) raccolgono il testimone della generazione di immagini; le GAN restano rilevanti per velocità e ibridi.
```
