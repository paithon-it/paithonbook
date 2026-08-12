# Modelli di Diffusione

Versa una goccia d'inchiostro in un bicchiere d'acqua e guarda. Prima un
filamento scuro, poi volute sempre più sottili, e in pochi minuti l'acqua è
tutta di un azzurro pallido e uniforme. Succede da solo, ogni volta, senza che
nessuno mescoli. Il contrario non lo ha mai visto nessuno: l'inchiostro che
spontaneamente si ritira dalle volute e si riconcentra in goccia. La fisica ha
un nome per questa asimmetria (è il secondo principio della termodinamica) e
un verdetto onesto: disfare è facile; rifare non è *vietato*, è solo così
improbabile che non basterebbe l'età dell'universo per vederlo accadere per
caso.

Ed è esattamente qui che si nasconde il trucco. Nel 2015 Jascha Sohl-Dickstein
e colleghi, tra Stanford e Berkeley, pubblicano un paper dal titolo che sembra
uscito da un dipartimento di fisica: *Deep Unsupervised Learning using
Nonequilibrium Thermodynamics* {cite}`sohl2015deep`. L'ispirazione viene
davvero dalla termodinamica di non equilibrio, e l'idea è di un'eleganza
spudorata: se distruggere è facile e costruire da zero non lo sa fare nessuno,
filmiamo la distruzione e insegniamo a una rete a **proiettare il film al
contrario**. Prendi una fotografia e aggiungile rumore, un pizzico alla volta,
finché non resta che pulviscolo: questo è il verso facile, la goccia che
si disperde. Poi addestra una rete a percorrere la pellicola all'indietro, un
fotogramma per volta. Se impara bene, potrà partire da rumore puro (un pulviscolo nuovo,
mai vista) e riavvolgerla fino a un'immagine che non è mai esistita.

## Il film proiettato al contrario

Un modello di diffusione vive dunque di due processi speculari. L'**andata**
non si impara: è una ricetta fissa che corrompe i dati aggiungendo rumore
casuale in tanti piccoli passi. Il **ritorno** è l'unica cosa che si
apprende, e quello che si impara è una cosa sola: dato un fotogramma
rumoroso, *che aspetto ha il rumore che c'è dentro*. Tutta la magia sta nella
modestia del compito. Come si passi da quella risposta a un'immagine è una
faccenda a parte, meno intuitiva di quanto sembri, ed è il centro della
prossima sezione.

`````{tab} Elementare

Immagina di rovinare una fotografia in mille passi: al passo 10 si nota
appena una grana fine, al passo 500 le forme si indovinano a fatica, al
passo 1.000 è rumore puro, come un televisore senza segnale. Questo è il verso
facile: lo fa un dado, non serve intelligenza.

Ora la parte furba. Non chiediamo alla rete l'impossibile («da questo pulviscolo
tira fuori una foto») ma una cosa umile: «ecco il fotogramma 500: dimmi che
aspetto aveva il disturbo aggiunto, così lo tolgo e torno al 499». È il
mestiere di un restauratore paziente, che non ridipinge il quadro ma sa dire,
strato per strato, dov'è lo sporco. Ed è un compito facile da imparare, perché
durante l'addestramento la risposta esatta la conosciamo: il disturbo
l'abbiamo aggiunto noi, sappiamo com'era fatto.

Per **generare** un'immagine nuova si parte dalla fine: si tira a caso una
manciata di pulviscolo appena estratto e si ripete mille volte il giro di domanda e
risposta, dal passo 1.000 al passo 1. A ogni passo emerge qualcosa (una massa
scura, una sagoma, un gatto) e all'ultimo fotogramma c'è un'immagine che non
esisteva da nessuna parte: la rete ha imparato che aspetto ha il mondo, e la
rumore iniziale, sempre diverso, decide quale immagine del mondo verrà fuori.

Attenzione a una cosa, però, perché è il punto in cui quasi tutti i racconti
di questa storia sbagliano: non è che a ogni giro si tolga un velo di sporco.
A ogni giro se ne toglie una scheggia e se ne rimette una manciata più grossa,
e l'immagine emerge per un motivo più sottile. Nella prossima sezione lo
vedremo con i numeri alla mano.

`````

`````{tab} Superiore

Il processo diretto è una catena di Markov che corrompe il dato $\mathbf{x}_0$ in $T$
passi (in DDPM, $T = 1000$):

$$
q(\mathbf{x}_t \mid \mathbf{x}_{t-1}) = \mathcal{N}\!\left(\mathbf{x}_t;\ \sqrt{1-\beta_t}\,\mathbf{x}_{t-1},\
\beta_t \mathbf{I}\right),
$$

dove $\mathbf{x}_t$ è il dato al passo $t$, $\beta_t$ una piccola varianza fissata
da uno *schedule* crescente e $\mathbf{I}$ l'identità: a ogni passo si attenua un
po' il segnale e si inietta rumore gaussiano. La catena ammette una
scorciatoia in forma chiusa:

$$
\mathbf{x}_t = \sqrt{\bar{\alpha}_t}\,\mathbf{x}_0 + \sqrt{1-\bar{\alpha}_t}\,\boldsymbol{\epsilon},
\qquad \boldsymbol{\epsilon} \sim \mathcal{N}(0, \mathbf{I}),
$$

dove $\bar{\alpha}_t = \prod_{s=1}^{t}(1-\beta_s)$ misura quanto segnale
originale sopravvive al passo $t$: si campiona $\mathbf{x}_t$ direttamente da $\mathbf{x}_0$,
senza percorrere la catena; per $t \to T$, $\bar{\alpha}_t \to 0$ e $\mathbf{x}_T$ è
rumore puro.

Il risultato che rende tutto possibile viene dall'analisi dei processi di
diffusione: se i passi $\beta_t$ sono piccoli, anche il processo inverso
$q(\mathbf{x}_{t-1} \mid \mathbf{x}_t)$ è approssimativamente gaussiano. Non è un risultato del
deep learning ed è molto più vecchio di tutto ciò di cui parla questo
capitolo: lo si deve a William Feller, che lo pubblica nel 1949
{cite}`feller1949theory`; Sohl-Dickstein e colleghi lo riprendono e ci
costruiscono sopra il modello {cite}`sohl2015deep`.
Ha quindi senso modellare il processo inverso con una gaussiana
parametrizzata da una rete,
$p_\theta(\mathbf{x}_{t-1} \mid \mathbf{x}_t)$, con parametri appresi $\theta$. Il contributo
di DDPM {cite}`ho2020denoising` è una parametrizzazione che riduce ogni cosa
a una regressione: la rete $\boldsymbol{\epsilon}_\theta(\mathbf{x}_t, t)$ predice il rumore
$\boldsymbol{\epsilon}$ iniettato, con la loss

$$
\mathcal{L} = \mathbb{E}_{\mathbf{x}_0,\, \boldsymbol{\epsilon},\, t}\!\left[\,\big\lVert \boldsymbol{\epsilon}
- \boldsymbol{\epsilon}_\theta(\mathbf{x}_t, t) \big\rVert^2\,\right],
$$

dove l'attesa è su un dato reale $\mathbf{x}_0$, un rumore $\boldsymbol{\epsilon}$ e un passo $t$
estratti a caso: un errore quadratico medio. La derivazione completa, dal
limite variazionale a questa forma, è il tema della prossima sezione.

`````

## Una parabola in tre atti

```{figure} ../figures/ddpm-denoising-iterativo.svg
:name: fig-ddpm-passi
:alt: "Catena lunga di stati intermedi fra un'immagine nitida e il rumore puro: procedendo verso destra ogni passo aggiunge un poco di rumore, procedendo verso sinistra la rete ne toglie un poco. Nessuno dei due sensi salta stati: il percorso è fatto di moltissimi passi piccoli."
:width: 100%

L'idea di fondo, quella del 2015, in una figura: tanti passi piccoli invece
di pochi grandi. Ogni passo chiede alla rete un compito facile, e la difficoltà si
distribuisce sull'intera catena.
```

Quello che {numref}`fig-ddpm-passi` rende evidente è anche il costo del
metodo, che la sezione sui Diffusion Transformer riprenderà. Se la generazione
è una catena di centinaia di passi, generare un'immagine significa
attraversare la rete centinaia di volte, non una.

Le idee, in questo mestiere, raramente vincono al primo colpo. Il paper del
2015 dimostra che il meccanismo funziona, ma su immagini piccole e con una
qualità che non impensierisce nessuno: sono gli anni in cui le GAN, nate
l'anno prima, si prendono la scena, e la diffusione resta per cinque anni una
curiosità da addetti ai lavori.

Il secondo atto è del 2020. Jonathan Ho, Ajay Jain e Pieter Abbeel, a
Berkeley, ripuliscono la formulazione e la battezzano **DDPM**, *Denoising
Diffusion Probabilistic Models* {cite}`ho2020denoising`: addestramento ridotto
a «indovina il rumore», architettura U-Net {cite}`ronneberger2015u` in
prestito dalla visione artificiale, e campioni che per la prima volta reggono
il confronto con le migliori GAN sul metro con cui si misurano questi modelli
(il FID, la *Fréchet Inception Distance* definita nel capitolo sulle GAN, che
misura quanto la nuvola delle immagini generate somiglia a quella delle
immagini vere: più è basso, meglio è). Il terzo atto arriva un anno dopo, e
il titolo dice tutto: *Diffusion Models Beat GANs on Image Synthesis*
{cite}`dhariwal2021diffusion`. La diffusione supera le GAN migliori non solo
in qualità delle singole immagini, ma anche nella **varietà** di quelle che sa
produrre, che è il punto dolente del capitolo precedente.

Poi c'è l'epilogo che non è più storia della ricerca ma storia e basta: il
2022. Nel giro di pochi mesi OpenAI presenta DALL·E 2 (aprile), Google
      risponde con Imagen (maggio, annunciato ma non accessibile al pubblico),
      Midjourney apre la beta a tutti (luglio) e soprattutto, in agosto,
      arriva **Stable Diffusion**: nato dai *latent diffusion models* del
      gruppo di Björn Ommer a Monaco di Baviera {cite}`rombach2022high` e
      rilasciato **con i pesi aperti**: i pesi sono i milioni di numeri che
      una rete si ritrova dentro dopo l'addestramento, cioè tutto quello che
      ha imparato, e rilasciarli vuol dire mettere in rete un file che
      chiunque può scaricare e far girare a casa propria. Per la prima volta
      si può scrivere «un gatto nero che salta sul muro, in stile acquerello»
      e vedere l'immagine materializzarsi sul proprio computer, non nel data
      center di qualcun altro. La generazione di immagini smette di essere una
      demo e diventa un fenomeno pubblico, con il suo seguito di entusiasmo,
      cause legali sui dati di addestramento e domande aperte sul lavoro
      creativo, su cui torneremo.

## Il falsario e il restauratore

```{figure} ../figures/modelli-diffusione.svg
:name: fig-dal-rumore-all-immagine
:alt: "Quattro riquadri in fila, da rumore puro a immagine nitida. La freccia in alto, verso destra, marca il processo di generazione che toglie rumore un passo alla volta; la freccia in basso, verso sinistra, marca il processo di corruzione che lo aggiunge e che non si impara."
:width: 100%

Le due frecce del capitolo. Quella in basso è una ricetta fissa e si può
calcolare; quella in alto è l'unica cosa che una rete deve imparare.
```

L'asimmetria di {numref}`fig-dal-rumore-all-immagine` è la ragione per cui il
metodo funziona. Poiché l'andata è nota, per ogni immagine del dataset si può
fabbricare gratis un numero illimitato di esempi di allenamento («ecco questa
immagine con un tot di rumore: dimmi qual era»), e il ritorno diventa un
problema come quelli del capitolo sul machine learning: domande di cui si
conosce già la risposta giusta, tante quante ne servono.

Nel capitolo precedente avevamo lasciato le GAN con un annuncio: verso il 2021
il primato generativo cambia mano. Quella promessa la manteniamo qui, e vale
la pena dire subito *perché* il testimone è passato, e che cosa la diffusione
paga in cambio.

`````{tab} Elementare

La GAN è un duello: un falsario e un detective che si allenano ostacolandosi
a vicenda. Quando funziona è spettacolare, ma tenere in equilibrio due
avversari è un mestiere da domatori: se uno dei due prende il sopravvento
l'allenamento si incarta, e il falsario può scoprire un solo quadro che
inganna sempre il detective e mettersi a rifare quello per sempre (il *mode
collapse* del capitolo scorso, la fine della varietà).

Il modello di diffusione, invece, è un artigiano solitario con un compito
umile: togli un velo di disturbo, mille volte. Niente avversario, niente
equilibri delicati: imparare è come studiare da un libro di esercizi con le
soluzioni in fondo, perché il disturbo da togliere lo conosciamo sempre. E
siccome deve saper ripulire *ogni* foto dell'archivio, non può rifugiarsi
in un unico quadro vincente: la varietà è di serie.

Il conto da pagare è la lentezza. Il falsario, una volta allenato, dipinge in
una pennellata sola: un passaggio, un'immagine. Il restauratore deve ripetere
la pulitura centinaia o migliaia di volte per ogni immagine: la stabilità si
paga in tempo d'attesa, e vedremo che accorciarlo è diventato un filone di
ricerca a sé.

`````

`````{tab} Superiore

Il confronto si gioca su tre assi.

**Stabilità.** L'addestramento GAN cerca l'equilibrio di un gioco minimax
$\min_G \max_D V(D,G)$: una dinamica a due giocatori che può oscillare,
divergere o collassare, come visto nel capitolo precedente. La diffusione
minimizza una singola loss di regressione,
$\mathcal{L} = \mathbb{E}\lVert\boldsymbol{\epsilon} - \boldsymbol{\epsilon}_\theta(\mathbf{x}_t, t)\rVert^2$:
un obiettivo stazionario che scende con la discesa del gradiente come
qualunque problema supervisionato, senza equilibri da inseguire.

**Diversità.** Il minimo della loss quadratica è la media condizionata
$\mathbb{E}[\boldsymbol{\epsilon} \mid \mathbf{x}_t]$, che è definita per *ogni* esempio del
dataset a *ogni* livello di rumore: un modello che coprisse solo alcuni modi
della distribuzione pagherebbe su tutti gli altri, e non ha modo di compensare
la perdita. Il *mode collapse*, patologia strutturale del gioco avversario,
qui non ha un meccanismo con cui manifestarsi. Dhariwal e Nichol
{cite}`dhariwal2021diffusion` lo confermano misurando, oltre alla qualità,
anche la copertura della distribuzione dei dati. (Attenzione a non appoggiare
questo argomento al limite variazionale: la loss che si usa davvero non è un
bound, come vedremo nella prossima sezione. Regge da sé.)

**Costo di campionamento.** Una GAN genera con **una** valutazione della rete;
un DDPM ne richiede $T$ (mille, in origine), perché il campionamento percorre
l'intera catena inversa. È il rovescio della medaglia, e ha aperto un filone
di ricerca sui campionatori accelerati (a partire da DDIM, che riduce i passi
a poche decine) che incontreremo più avanti nel capitolo.

`````

Nessun vincitore assoluto, dunque, e conviene dirlo con i meccanismi invece
che con una classifica, perché i meccanismi non invecchiano: una GAN genera
con una sola valutazione della rete, un DDPM con molte, e la diffusione paga
in tempo ciò che guadagna in stabilità dell'addestramento e in copertura della
varietà. Da lì in poi, buona parte del lavoro sulla diffusione è servito ad
accorciare quel conto, e nel farlo ha spesso rimesso in gioco un
discriminatore, cioè proprio l'idea del capitolo precedente. Ma dal 2022 in
poi, quando leggete «immagine generata dall'AI», nella stragrande maggioranza
dei casi dietro c'è una diffusione.

## Come è organizzato il capitolo

Tre tappe. Prima **come funziona davvero**: i due processi di DDPM
{cite}`ho2020denoising` visti da vicino, da dove esce il voto che si dà alla
rete, che rete sia (una vecchia conoscenza della visione artificiale) e tutto
il meccanismo in miniatura, funzionante, in poche righe di Python. Poi il
**salto di scala**: come si può far lavorare la rete non sulla fotografia ma
su una sua versione compressa decine di volte, che è il segreto per cui Stable
Diffusion {cite}`rombach2022high` gira su un computer di casa, e come si fa a
ordinargli che cosa disegnare scrivendolo a parole. Infine **l'incontro con i
Transformer** {cite}`vaswani2017attention`, quelli del capitolo sul
linguaggio: si può buttare via la rete di visione e mettere al suo posto uno
di loro {cite}`peebles2023scalable`, e la risposta ha cambiato il modo in cui
si costruiscono questi modelli, video compresi.

`````{tab} Elementare

```{admonition} Da ricordare
:class: important
- Un **modello di diffusione** impara a proiettare al contrario il film di una
  rovina: l'andata (aggiungere un pizzico di disturbo mille volte, finché non
  resta che pulviscolo) è una ricetta fissa che sa eseguire un dado; il
  ritorno è l'unica cosa che si apprende, e quello che si impara è una domanda
  sola, «che aspetto ha il disturbo che c'è qui dentro». L'idea viene da
  un'osservazione di fisica: disfare è facile, rifare per caso è così
  improbabile da non accadere mai.
- **DDPM** rende l'addestramento facile perché non chiede il capolavoro ma una
  cosa umile: «dimmi che aspetto aveva il disturbo che ho aggiunto». La
  risposta esatta la conosciamo sempre, visto che il disturbo l'abbiamo messo
  noi: è come studiare su un libro di esercizi con le soluzioni in fondo.
- Dal 2021 la diffusione **supera le GAN** in qualità e in varietà dei
  risultati; nel 2022, con DALL·E 2, Imagen, Midjourney e soprattutto Stable
  Diffusion, scaricabile da chiunque, diventa un fenomeno di massa.
- Rispetto alle GAN: niente duello fra falsario e detective, quindi niente
  allenamenti che si incartano e nessun rifugio in un unico quadro vincente.
  Il conto si paga in attesa: il falsario dipinge in una pennellata sola, il
  restauratore ripete il suo giro centinaia di volte. Non è una classifica, è
  un baratto: tempo in cambio di stabilità e varietà.
- Nel resto del capitolo: DDPM in dettaglio, la diffusione **latente** di
  Stable Diffusion, i **diffusion Transformer** e i modelli video.
```

`````

`````{tab} Superiore

```{admonition} Da ricordare
:class: important
- Un **modello di diffusione** impara a invertire una degradazione:
  l'andata (aggiungere rumore in tanti piccoli passi) è fissa e banale, il
  ritorno è l'unica cosa che si apprende, ed è la **stima** del rumore
  presente, non la sua rimozione. L'idea nasce dalla termodinamica di non
  equilibrio {cite}`sohl2015deep`; il risultato che la sostiene (per passi
  piccoli l'inverso è approssimativamente gaussiano) è di Feller
  {cite}`feller1949theory`.
- **DDPM** {cite}`ho2020denoising` riduce l'addestramento a una regressione,
  predire il rumore iniettato, loss
  $\mathbb{E}\lVert\boldsymbol{\epsilon} - \boldsymbol{\epsilon}_\theta(\mathbf{x}_t,t)\rVert^2$,
  stabile come un problema supervisionato.
- Nel 2021 la diffusione **supera le GAN** in qualità e copertura dei modi
  {cite}`dhariwal2021diffusion`; nel 2022, con DALL·E 2, Imagen, Midjourney
  e lo Stable Diffusion open source {cite}`rombach2022high`, diventa un
  fenomeno pubblico.
- Rispetto alle GAN: niente duello, niente *mode collapse*, addestramento
  stabile, ma il campionamento costa molti passi di rete invece di uno.
- Nel resto del capitolo: DDPM in dettaglio, la diffusione **latente** di
  Stable Diffusion, i **diffusion Transformer** (DiT) e i modelli video.
```

`````
