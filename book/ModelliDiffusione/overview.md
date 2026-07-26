# Modelli di Diffusione

Versa una goccia d'inchiostro in un bicchiere d'acqua e guarda. Prima un
filamento scuro, poi volute sempre più sottili, e in pochi minuti l'acqua è
tutta di un azzurro pallido e uniforme. Succede da solo, ogni volta, senza
che nessuno mescoli. Il contrario non lo ha mai visto nessuno: l'inchiostro
che spontaneamente si ritira dalle volute e si riconcentra in goccia. La
fisica ha un nome per questa asimmetria — è il secondo principio della
termodinamica — e un verdetto onesto: disfare è facile; rifare non è
*vietato*, è solo così improbabile che non basterebbe l'età dell'universo
per vederlo accadere per caso.

Ed è esattamente qui che si nasconde il trucco. Nel 2015 Jascha
Sohl-Dickstein e colleghi, tra Stanford e Berkeley, pubblicano un paper dal
titolo che sembra uscito da un dipartimento di fisica: *Deep Unsupervised
Learning using Nonequilibrium Thermodynamics* {cite}`sohl2015deep`.
L'ispirazione viene davvero dalla termodinamica di non equilibrio, e l'idea
è di un'eleganza spudorata: se distruggere è facile e costruire da zero non
lo sa fare nessuno, filmiamo la distruzione e insegniamo a una rete a
**proiettare il film al contrario**. Prendi una fotografia e aggiungile
rumore, un pizzico alla volta, finché non resta che neve televisiva: questo
è il verso facile, la goccia che si disperde. Poi addestra una rete a
percorrere la pellicola all'indietro, un fotogramma per volta. Se impara
bene, potrà partire da neve pura — neve nuova, mai vista — e riavvolgerla
fino a un'immagine che non è mai esistita.

## Il film proiettato al contrario

Un modello di diffusione vive dunque di due processi speculari. L'**andata**
non si impara: è una ricetta fissa che corrompe i dati aggiungendo rumore
casuale in tanti piccoli passi. Il **ritorno** è l'unica cosa che si
apprende: una rete che, dato un fotogramma rumoroso, stima come togliere
*un po'* di quel rumore. Tutta la magia sta nella modestia del compito.

`````{tab} Elementare

Immagina di rovinare una fotografia in mille passi: al passo 10 si nota
appena una grana fine, al passo 500 le forme si indovinano a fatica, al
passo 1.000 è neve pura, come un televisore senza segnale. Questo è il verso
facile: lo fa un dado, non serve intelligenza.

Ora la parte furba. Non chiediamo alla rete l'impossibile — «da questa neve
tira fuori una foto» — ma una cosa umile: «ecco il fotogramma 500: dimmi
che aspetto aveva il disturbo aggiunto, così lo tolgo e torno al 499». È il
mestiere di un restauratore paziente, che non ridipinge il quadro ma
solleva una velatura di sporco alla volta. Ed è un compito facile da
imparare, perché durante l'addestramento la risposta esatta la conosciamo:
il disturbo l'abbiamo aggiunto noi, sappiamo com'era fatto.

Per **generare** un'immagine nuova si parte dalla fine: si tira a caso una
schermata di neve fresca e si applica la pulitura mille volte, dal passo
1.000 al passo 1. A ogni passo emerge qualcosa — una massa scura, una
sagoma, un gatto — e all'ultimo fotogramma c'è un'immagine che non esisteva
da nessuna parte: la rete ha imparato che aspetto ha il mondo, e la neve
iniziale, sempre diversa, decide quale immagine del mondo verrà fuori.

`````

`````{tab} Superiore

Il processo diretto è una catena di Markov che corrompe il dato $x_0$ in $T$
passi (in DDPM, $T = 1000$):

$$
q(x_t \mid x_{t-1}) = \mathcal{N}\!\left(x_t;\ \sqrt{1-\beta_t}\,x_{t-1},\
\beta_t I\right),
$$

dove $x_t$ è il dato al passo $t$, $\beta_t$ una piccola varianza fissata
da uno *schedule* crescente e $I$ l'identità: a ogni passo si attenua un
po' il segnale e si inietta rumore gaussiano. La catena ammette una
scorciatoia in forma chiusa:

$$
x_t = \sqrt{\bar{\alpha}_t}\,x_0 + \sqrt{1-\bar{\alpha}_t}\,\epsilon,
\qquad \epsilon \sim \mathcal{N}(0, I),
$$

dove $\bar{\alpha}_t = \prod_{s=1}^{t}(1-\beta_s)$ misura quanto segnale
originale sopravvive al passo $t$: si campiona $x_t$ direttamente da $x_0$,
senza percorrere la catena; per $t \to T$, $\bar{\alpha}_t \to 0$ e $x_T$ è
rumore puro.

Il risultato che rende tutto possibile viene dall'analisi dei processi di
diffusione: se i passi $\beta_t$ sono piccoli, anche il processo inverso
$q(x_{t-1} \mid x_t)$ è approssimativamente gaussiano {cite}`sohl2015deep`,
e ha quindi senso modellarlo con una gaussiana parametrizzata da una rete,
$p_\theta(x_{t-1} \mid x_t)$, con parametri appresi $\theta$. Il contributo
di DDPM {cite}`ho2020denoising` è una parametrizzazione che riduce ogni cosa
a una regressione: la rete $\epsilon_\theta(x_t, t)$ predice il rumore
$\epsilon$ iniettato, con la loss

$$
\mathcal{L} = \mathbb{E}_{x_0,\, \epsilon,\, t}\!\left[\,\big\lVert \epsilon
- \epsilon_\theta(x_t, t) \big\rVert^2\,\right],
$$

dove l'attesa è su un dato reale $x_0$, un rumore $\epsilon$ e un passo $t$
estratti a caso: un errore quadratico medio. La derivazione completa, dal
limite variazionale a questa forma, è il tema della prossima sezione.

`````

## Una parabola in tre atti

Le idee, in questo mestiere, raramente vincono al primo colpo. Il paper del
2015 dimostra che il meccanismo funziona, ma su immagini piccole e con una
qualità che non impensierisce nessuno: sono gli anni in cui le GAN, nate
l'anno prima, si prendono la scena, e la diffusione resta per un
lustro una curiosità da addetti ai lavori.

Il secondo atto è del 2020. Jonathan Ho, Ajay Jain e Pieter Abbeel, a
Berkeley, ripuliscono la formulazione e la battezzano **DDPM** — *Denoising
Diffusion Probabilistic Models* {cite}`ho2020denoising`: addestramento
ridotto a «indovina il rumore», architettura U-Net {cite}`ronneberger2015u`
in prestito dalla visione artificiale, e campioni che per la prima volta
reggono il confronto con le migliori GAN: su CIFAR-10 un FID di 3,17,
allora lo stato dell'arte (il FID, *Fréchet Inception Distance*, è la
metrica standard della generazione di immagini: misura la distanza
statistica tra immagini generate e immagini reali, e più è basso meglio
è). Il terzo atto arriva un anno dopo, e il titolo
dice tutto: *Diffusion Models Beat GANs on Image Synthesis*
{cite}`dhariwal2021diffusion`. Su ImageNet la diffusione supera le GAN
migliori non solo in qualità, ma anche in **diversità** dei campioni.

Poi c'è l'epilogo che non è più storia della ricerca ma storia e basta: il
2022. Nel giro di pochi mesi OpenAI presenta DALL·E 2 (aprile), Google
risponde con Imagen (maggio, annunciato ma non accessibile al pubblico),
Midjourney apre la beta a tutti (luglio) e soprattutto, in agosto, arriva
**Stable Diffusion**: nato dai *latent diffusion models* del gruppo di Björn Ommer a
Monaco di Baviera {cite}`rombach2022high` e rilasciato con pesi aperti,
scaricabili da chiunque. Per la prima volta si può scrivere «un gatto nero
che salta sul muro, in stile acquerello» e vedere l'immagine
materializzarsi sul proprio computer, non nel data center di qualcun altro.
La generazione di immagini smette di essere una demo e diventa un fenomeno
pubblico — con il suo seguito di entusiasmo, cause legali sui dati di
addestramento e domande aperte sul lavoro creativo, su cui torneremo.

## Il falsario e il restauratore

Nel capitolo precedente avevamo lasciato le GAN con un annuncio: verso il
2021 il primato generativo cambia mano. Questo capitolo è il pagamento di
quella cambiale, e vale la pena dire subito *perché* il testimone è passato
— e che cosa la diffusione paga in cambio.

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

Il conto da pagare è la lentezza. Il falsario, una volta allenato, dipinge
in una pennellata sola: un passaggio, un'immagine. Il restauratore deve
ripetere la pulitura centinaia o migliaia di volte per ogni immagine: la
stabilità si paga in tempo d'attesa — e vedremo che accorciarlo è diventato
un filone di ricerca a sé.

`````

`````{tab} Superiore

Il confronto si gioca su tre assi.

**Stabilità.** L'addestramento GAN cerca l'equilibrio di un gioco minimax
$\min_G \max_D V(D,G)$: una dinamica a due giocatori che può oscillare,
divergere o collassare, come visto nel capitolo precedente. La diffusione
minimizza una singola loss di regressione,
$\mathcal{L} = \mathbb{E}\lVert\epsilon - \epsilon_\theta(x_t, t)\rVert^2$:
un obiettivo stazionario che scende con la discesa del gradiente come
qualunque problema supervisionato, senza equilibri da inseguire.

**Diversità.** La loss discende da un limite variazionale sulla
log-verosimiglianza: ogni esempio del dataset, a ogni livello di rumore,
contribuisce all'obiettivo, e il modello non può azzerarla coprendo solo
alcuni modi della distribuzione — il *mode collapse*, patologia strutturale
del gioco avversario, qui non ha un meccanismo per manifestarsi. Dhariwal e
Nichol {cite}`dhariwal2021diffusion` misurano infatti sia miglior FID sia
miglior copertura (*recall*) rispetto alle GAN di riferimento su ImageNet.

**Costo di campionamento.** Una GAN genera con **una** valutazione della
rete; un DDPM ne richiede $T$ (mille, in origine), perché il campionamento
percorre l'intera catena inversa. È il rovescio della medaglia, e ha aperto
un filone di ricerca sui campionatori accelerati — a partire da DDIM, che
riduce i passi a poche decine — che incontreremo più avanti nel capitolo.

`````

Nessun vincitore assoluto, dunque: le GAN restano imbattibili in velocità
di campionamento, la diffusione domina in stabilità, qualità e diversità.
Ma dal 2022 in poi, quando leggete «immagine generata dall'AI», nella
stragrande maggioranza dei casi dietro c'è una diffusione.

## Come è organizzato il capitolo

Tre tappe. Prima **come funziona davvero**: il processo diretto e quello
inverso di DDPM {cite}`ho2020denoising`, la derivazione della loss, la
U-Net che fa da spina dorsale e il ciclo di addestramento in PyTorch. Poi
il **salto di scala**: l'idea dei *latent diffusion models*
{cite}`rombach2022high` di spostare la diffusione in uno spazio latente
compresso — il segreto che rende possibile Stable Diffusion su una GPU da
videogiochi — e il condizionamento sul testo che trasforma un generatore di
immagini in un illustratore su richiesta. Infine **l'incontro con i
Transformer** {cite}`vaswani2017attention`: l'architettura DiT di Peebles
e Xie {cite}`peebles2023scalable` che sostituisce la U-Net con un
Transformer su patch — la stessa
ricetta che, estesa ai video, sta dietro a generatori come Sora. Il
cerchio, per chi arriva dal capitolo sui Transformer, si chiude.

```{admonition} Da ricordare
:class: important
- Un **modello di diffusione** impara a invertire una degradazione:
  l'andata (aggiungere rumore in tanti piccoli passi) è fissa e banale, il
  ritorno (togliere un po' di rumore alla volta) è l'unica cosa che si
  apprende. L'idea nasce dalla termodinamica di non equilibrio
  {cite}`sohl2015deep`.
- **DDPM** {cite}`ho2020denoising` riduce l'addestramento a una regressione
  — predire il rumore iniettato, loss
  $\mathbb{E}\lVert\epsilon - \epsilon_\theta(x_t,t)\rVert^2$ — stabile
  come un problema supervisionato.
- Nel 2021 la diffusione **supera le GAN** in qualità e diversità
  {cite}`dhariwal2021diffusion`; nel 2022, con DALL·E 2, Imagen, Midjourney
  e lo Stable Diffusion open source {cite}`rombach2022high`, diventa un
  fenomeno pubblico.
- Rispetto alle GAN: niente duello, niente *mode collapse*, addestramento
  stabile — ma il campionamento costa molti passi di rete invece di uno.
- Nel resto del capitolo: DDPM in dettaglio, la diffusione **latente** di
  Stable Diffusion, i **diffusion Transformer** (DiT) e i modelli video.
```
