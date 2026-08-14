# Lo spazio latente: Stable Diffusion

Il 22 agosto 2022 compare online un file da circa quattro gigabyte. Sono i
pesi di **Stable Diffusion**, un modello text-to-image nato dai *latent
diffusion models* del gruppo di Björn Ommer all'Università Ludwig Maximilian
di Monaco {cite}`rombach2022high`, sviluppato con Runway e addestrato con la
potenza di calcolo di Stability AI. La novità non è la qualità delle immagini
(DALL·E 2 e Imagen, usciti pochi mesi prima, erano già impressionanti) ma le
condizioni: quei modelli vivevano nei data center dei loro proprietari,
accessibili con il contagocce dietro liste d'attesa e interfacce controllate.
Stable Diffusion invece si *scarica*. Chiunque, gratis, può metterlo sul
proprio computer, e per farlo girare basta una scheda video da videogiochi con
meno di dieci gigabyte di memoria. Nel giro di poche settimane i forum si
riempiono di immagini, spuntano interfacce grafiche amatoriali, plugin per
Photoshop e Blender, versioni modificate per ogni gusto. La generazione di
immagini smette di essere una demo da guardare e diventa uno strumento da
usare.

La domanda di questa sezione è: che cosa lo rende possibile *tecnicamente*?
Non un modello più grande: al contrario, uno più piccolo. Il segreto è un
trasloco: la diffusione che conosciamo fa le valigie, lascia i pixel e si
trasferisce in uno spazio compresso, decine di volte più piccolo, dove ogni
passo di pulitura costa una frazione. Per capire il trasloco, però, dobbiamo
prima conoscere il traslocatore: una rete che si chiama *variational
autoencoder*. Il libro l'ha finora solo nominata di sfuggita, nel capitolo sul
reinforcement learning profondo, e ne ha usato una variante fra i codec
neurali dell'audio: qui la guardiamo per esteso, perché è lei che porta i
mobili.

## Il prezzo dei pixel

Facciamo due conti. Nella sezione precedente un'immagine era una griglia di
numeri, uno per pixel, perché la guardavamo in bianco e nero. A colori i numeri
per pixel diventano tre, uno per ciascuno dei colori con cui uno schermo compone
tutti gli altri (rosso, verde, blu). Un'immagine a colori di
$512 \times 512$ pixel è fatta quindi di
$512 \times 512 \times 3 = 786\,432$ numeri. Il restauratore della sezione
precedente (la U-Net {cite}`ronneberger2015u` che predice il rumore) deve
elaborarli *tutti*, e deve farlo a ogni passo di pulitura: centinaia o
migliaia di passaggi per una sola immagine. Su un cluster da laboratorio si
può fare; su un computer di casa no.

Ed è uno spreco, per una ragione precisa: la maggior parte di quei 786.432
numeri non descrive *che cosa* c'è nell'immagine, ma dettagli percettivi (la
grana dell'intonaco, il micro-rumore del sensore, la trama del pelo). La
compressione JPEG campa su questo da trent'anni: butta via gran parte dei bit
e l'occhio quasi non se ne accorge. Un modello di diffusione che lavora sui
pixel spende quindi la parte del leone del suo sforzo a modellare dettagli ad
alta frequenza (i dettagli piccolissimi, quelli che cambiano da un pixel al
suo vicino) che contano poco, e solo una frazione a decidere le cose
importanti: dov'è il gatto, dov'è il muro, da che parte arriva la luce.

L'idea di Robin Rombach, Björn Ommer e colleghi {cite}`rombach2022high` è
dividere il lavoro tra due specialisti. Una prima rete impara la
**compressione percettiva**: trasforma i pixel in una rappresentazione
compatta (lo **spazio latente**) che conserva il contenuto e scarta il
dettaglio ricostruibile. La diffusione, poi, impara la **composizione** dentro
quello spazio compatto, dove ogni passo costa decine di volte meno. È la
ricetta dei *latent diffusion models*, e Stable Diffusion ne è il figlio
famoso.

## L'archivista: il variational autoencoder

Il pezzo nuovo della pipeline è una rete a forma di clessidra: un
**autoencoder**, addestrato a ricostruire il proprio input dopo averlo fatto
passare da una strettoia. La variante che ci serve, il **variational
autoencoder** (VAE) di Diederik Kingma e Max Welling {cite}`kingma2014auto`, è
del 2014 (più vecchia della diffusione moderna e persino delle GAN) e tornerà
più avanti nel libro. Qui ci serve l'essenziale: che cosa fa e perché rende lo
spazio latente un posto dove si può lavorare. Per parlarne useremo un'immagine
che ci accompagnerà fino alla fine del capitolo: la rete che comprime è un
**archivista** e la rappresentazione compatta che scrive è una **scheda**.

```{figure} ../figures/vae-auto-encoding-variational-bayes.svg
:name: fig-vae
:alt: "Schema del variational autoencoder: l'immagine entra nell'encoder, che non produce un punto ma una media e una deviazione standard; da quella distribuzione si campiona un punto nello spazio latente; il decoder riceve il punto campionato e ricostruisce l'immagine. La perdita somma il termine di ricostruzione e il termine che tiene la distribuzione vicina al prior."
:width: 100%

La rete che comprime non restituisce una scheda sola, ma una scheda **e un
margine di tolleranza**: «all'incirca questo, più o meno tanto». È quel margine
la trovata, ed è ciò che costringe schede vicine a ridiventare immagini simili.
```

Il sorteggio in mezzo alla {numref}`fig-vae` (si prende la scheda, la si sposta
di un po' a caso dentro il margine di tolleranza, e solo allora la si passa a
chi deve ridipingere) è ciò che distingue questa rete da un compressore
qualunque, e ha una conseguenza pratica precisa: siccome l'addestramento vede
ogni volta una scheda leggermente diversa, chi ridipinge è costretto a
funzionare su tutto un intorno, non su un punto solo. Lo spazio delle schede ne
esce **continuo**, cioè senza buchi: qualunque scheda si peschi, anche una a
metà strada fra due che esistono, corrisponde a un'immagine sensata. Ed è la
premessa perché la diffusione ci si possa muovere dentro, dato che la
diffusione, di suo, passa il tempo a mettere piede in posti a caso.

`````{tab} Elementare

Immagina l'archivista di un museo pieno di quadri enormi. Per ogni quadro
scrive una scheda molto più piccola dell'originale, e il suo mestiere si
giudica con una prova concreta: un collega deve *ridipingere* il quadro
leggendo solo la scheda. Se la copia somiglia all'originale, la scheda
conteneva l'essenziale; se non somiglia, la scheda va scritta meglio. Dopo
milioni di prove su milioni di quadri, l'archivista ha imparato da solo che
cosa annotare (soggetto, composizione, colori dominanti) e che cosa lasciar
perdere, perché il collega sa ricostruirlo da sé: la grana della tela, le
singole pennellate dello sfondo. Nei numeri di Stable Diffusion: il quadro è
fatto di 786.432 valori, la scheda di 16.384, quarantotto volte meno. E quel
16.384 non è un numero magico ma una scelta di progetto: la scheda è una
griglia di 64 caselle per lato invece delle 512 dell'immagine (otto volte meno
per lato) con quattro numeri per casella, e $64 \times 64 \times 4$ fa
appunto 16.384.

Va detta anche l'altra metà, perché il capitolo ci tornerà: **la compressione
distrugge**, e quello che l'archivista non ha annotato non lo recupera più
nessuno, per quanto bravo sia il restauratore che lavora dopo di lui. La scheda
è il soffitto della qualità finale. Quattro numeri per casella bastano per un
gatto su un muro e non bastano per una scritta leggibile, per un volto in
secondo piano o per una mano con cinque dita: è una delle ragioni (non l'unica)
dei difetti tipici della prima generazione di questi modelli, che sono sempre
difetti di dettaglio fine. Alzare quel soffitto è una delle cose che i
successori hanno fatto.

E il «variazionale» del nome? Sta in due regole che tengono l'archivio in
ordine. Primo: la scheda non inchioda il quadro a un punto esatto ma descrive
una *nuvola di possibilità* («un gatto nero più o meno così») cosicché quadri
quasi uguali abbiano schede quasi uguali. Secondo: l'archivio non deve avere
buchi; se peschi una scheda plausibile a caso, anche una mai scritta da
nessuno, il collega deve comunque saperne dipingere un quadro sensato.
Sembrano pignolerie, ma sono esattamente ciò che serve alla diffusione: il
restauratore lavorerà *dentro* questo archivio, muovendosi tra schede piene di
rumore, e ogni casella in cui mette piede deve corrispondere a un'immagine
possibile.

`````

`````{tab} Superiore

Un VAE è una coppia di reti. L'**encoder** mappa il dato $\mathbf{x}$ non in un punto
ma in una distribuzione sul latente,
$q_\phi(\mathbf{z} \mid \mathbf{x}) = \mathcal{N}\big(\mathbf{z};\, \boldsymbol{\mu}_\phi(\mathbf{x}),\, \sigma_\phi^2(\mathbf{x})\, \mathbf{I}\big)$;
il **decoder** definisce $p_\psi(\mathbf{x} \mid \mathbf{z})$, la ricostruzione a partire
dal codice (scriviamo $\psi$ per i suoi parametri perché in questo capitolo
$\theta$ è già impegnato dalla rete di diffusione $\boldsymbol{\epsilon}_\theta$: sono due
reti distinte, addestrate separatamente). Sul latente si
impone un prior semplice, $p(\mathbf{z}) = \mathcal{N}(0, \mathbf{I})$. L'addestramento
massimizza l'**ELBO** (*evidence lower bound*):

$$
\mathrm{ELBO}(\psi, \phi; \mathbf{x}) =
\mathbb{E}_{q_\phi(\mathbf{z} \mid \mathbf{x})}\!\big[\log p_\psi(\mathbf{x} \mid \mathbf{z})\big]
- D_{KL}\!\big(q_\phi(\mathbf{z} \mid \mathbf{x}) \,\|\, p(\mathbf{z})\big),
$$

dove il primo termine premia la fedeltà della ricostruzione e il secondo (la
divergenza di Kullback–Leibler vista nei richiami di matematica) penalizza gli
encoder che si allontanano dal prior. È questo secondo termine a rendere lo
spazio latente **continuo** (input simili, codici vicini) e **campionabile**
(ogni regione con probabilità apprezzabile sotto il prior decodifica in un
dato plausibile). Nella convenzione del libro, dove $\mathcal{L}$ si
minimizza, la loss corrispondente è $\mathcal{L} = -\mathrm{ELBO}$. La
derivazione dell'ELBO come limite inferiore della
log-verosimiglianza è nel paper originale {cite}`kingma2014auto`; qui ci basta
il ruolo funzionale dei due termini.

Il VAE di Stable Diffusion è convoluzionale e riduce ogni lato di un fattore
$f = 8$, con 4 canali latenti: da $512 \times 512 \times 3$ a
$64 \times 64 \times 4$, cioè da $786\,432$ a $16\,384$ valori, un fattore 48.
Rispetto al VAE da manuale è addestrato con un peso KL molto piccolo, più una
loss percettiva e una avversaria (un discriminatore in stile GAN, come nel
capitolo precedente) che tengono nitide le ricostruzioni
{cite}`rombach2022high`.

Sarebbe però un errore liquidarlo come un dettaglio di efficienza: la
compressione **è distruttiva, e il danno è misurabile e definitivo**. Tutto ciò
che il decoder non sa ricostruire è perduto prima che la U-Net veda alcunché,
quindi la capacità di ricostruzione dell'autoencoder è un **limite superiore**
sulla qualità dell'intero sistema, che nessuna quantità di diffusione può
superare. Rombach e colleghi lo mettono fra i limiti dichiarati del metodo, e
la conferma più netta arriva dai loro stessi successori: a parità di $f$,
portare i canali latenti da 4 a 16 migliora nettamente ogni misura di
ricostruzione, ed è una delle scelte di Stable Diffusion 3
{cite}`esser2024scaling`, che vedremo nella prossima sezione. Le due loss
aggiuntive vanno lette nella stessa luce: il decoder non ricostruisce e basta,
**sceglie che cosa è plausibile** ricostruire, il che è un'altra cosa e ha
conseguenze proprie.

`````

## La ricetta in quattro mosse

Ora abbiamo tutti i pezzi, e {numref}`fig-latent-diffusion` li mette in fila:
la rete che comprime, lo spazio delle schede dove avviene la diffusione, il
testo che entra di lato, la rete che riporta ai pixel. Un dettaglio
dell'ordine dei lavori conta più di quanto sembri: l'archivista impara il suo
mestiere *prima*, da solo, e poi **smette di imparare**. Da quel momento in
avanti è uno strumento fisso, e mentre il restauratore si allena nessuno gli
tocca più niente. In gergo si dice che i suoi pesi vengono *congelati*, e la
ragione è che il restauratore deve allenarsi su un archivio che non cambia
sotto i suoi occhi.

```{figure} ../figures/latent-diffusion.svg
:name: fig-latent-diffusion
:alt: "Pipeline di Stable Diffusion: un'immagine 512 per 512 per 3 passa dall'encoder VAE che la comprime in un latente 64 per 64 per 4; nel riquadro dello spazio latente una U-Net toglie il rumore in T passi, mentre il prompt, trasformato dal text encoder CLIP, entra nella U-Net attraverso la cross-attention; il decoder VAE riconverte il latente ripulito in un'immagine."
:width: 100%

La catena di montaggio di Stable Diffusion. La diffusione non tocca mai i
pixel: lavora sulla scheda compressa, e a ogni passo dà un'occhiata alla
richiesta scritta dall'utente, che entra di lato.
```

`````{tab} Elementare

Quattro mosse. **Prima**: l'archivista comprime ogni fotografia dell'archivio
di addestramento nella sua scheda; d'ora in poi si lavora solo su schede.
**Seconda**: il restauratore della sezione precedente fa esattamente il suo
solito mestiere (sporca di rumore, impara a pulire un velo alla volta) ma su
schede da 16.384 numeri invece che su quadri da 786.432: come restaurare
cartoline anziché affreschi, ogni passata costa decine di volte meno (non
esattamente quarantotto: la rete che lavora sulle schede non è quella dei
quadri rimpicciolita, è una rete sua).
**Terza**: mentre pulisce, il restauratore tiene sul tavolo la commissione
scritta dal cliente («un gatto nero che salta sul muro, in acquerello») e a
ogni pennellata le dà un'occhiata, soffermandosi sulle parole che servono in
quel momento: «nero» quando decide i toni, «acquerello» quando decide il
tratto. È la stessa occhiata selettiva dell'interprete della traduzione
automatica, ritrovata poi nei Transformer: lì collegava due lingue, qui
collega parole e immagine. **Quarta**: finita la pulitura, la scheda torna
all'archivista, che ridipinge il quadro a piena risoluzione.

Per **generare** un'immagine nuova si parte, come sempre, dalla fine: una
scheda di puro rumore casuale, mai appartenuta a nessun quadro. Il
restauratore la pulisce passo dopo passo con la commissione sotto gli
occhi, e l'archivista trasforma il risultato in pixel. Il gatto in
acquerello che compare non esisteva da nessuna parte: né il quadro, né la
sua scheda.

`````

`````{tab} Superiore

Formalmente, le quattro componenti sono queste.

**1. Compressione.** L'encoder congelato porta ogni immagine nel latente. Non
è una funzione, ed è la distribuzione definita sopra: il latente di
addestramento si **campiona**, $\mathbf{z} \sim q_\phi(\cdot \mid \mathbf{x})$, con
$\mathbf{z} \in \mathbb{R}^{64 \times 64 \times 4}$ per
$\mathbf{x} \in \mathbb{R}^{512 \times 512 \times 3}$ (scriveremo $\mathcal{E}(\mathbf{x})$
per brevità, ricordando che sotto c'è un campionamento).

**2. Diffusione nel latente.** Il processo diretto e quello inverso hanno la
stessa forma di quelli della sezione precedente, applicati a $\mathbf{z}$ anziché a
$\mathbf{x}$, con un'avvertenza che il paragrafo sul processo diretto aveva già
anticipato: lo schedule *variance-preserving* presuppone dati a **varianza
unitaria**, e il latente del VAE non ce l'ha. LDM lo riscala quindi per la
deviazione standard misurata componente per componente sui latenti (in Stable
Diffusion 1.x la costante vale $0{,}18215$, cioè l'inverso di quella
deviazione standard). Non è una limatura: Rombach e colleghi documentano che
il rapporto segnale/rumore indotto dalla scala del latente incide
sensibilmente sul risultato, e senza quella riscalatura tutta la taratura
dello schedule sarebbe sbagliata. Fatta la riscalatura, la U-Net
$\boldsymbol{\epsilon}_\theta$ è addestrata a predire il rumore con la solita
regressione, ora condizionata anche dal testo $c$:

$$
\mathcal{L}_{\mathrm{LDM}} =
\mathbb{E}_{\mathbf{z},\, c,\, \boldsymbol{\epsilon},\, t}\!\left[\,
\big\lVert \boldsymbol{\epsilon} - \boldsymbol{\epsilon}_\theta\big(\mathbf{z}_t,\, t,\, \tau(c)\big)
\big\rVert^2 \,\right],
$$

dove $\mathbf{z}_t$ è il latente rumoroso al passo $t$, $\boldsymbol{\epsilon}$ il rumore
iniettato e $\tau$ il text encoder. Ogni valutazione della rete lavora su
$16\,384$ valori invece di $786\,432$: è qui che si paga l'affitto ridotto
dello spazio latente.

**3. Condizionamento testuale.** $\tau$ è il text encoder di CLIP
{cite}`radford2021learning`, il modello contrastivo del capitolo su visione e
linguaggio: congelato, trasforma il prompt in una sequenza di 77
embedding da 768 dimensioni. Questi entrano nella U-Net tramite strati di
**cross-attention** inseriti a più risoluzioni, la stessa identica formula del
capitolo sui Transformer:

$$
\mathrm{Attention}(\mathbf{Q}, \mathbf{K}, \mathbf{V}) =
\mathrm{softmax}\!\left(\frac{\mathbf{Q}\mathbf{K}^\top}{\sqrt{d_k}}\right)
\mathbf{V},
\qquad
\mathbf{Q} = h(\mathbf{z}_t)\, \mathbf{W}_Q, \quad
\mathbf{K} = \tau(c)\, \mathbf{W}_K, \quad
\mathbf{V} = \tau(c)\, \mathbf{W}_V,
$$

dove $h(\mathbf{z}_t)$ sono le mappe di attivazione intermedie della U-Net
appiattite in sequenza, $\tau(c)$ gli embedding del prompt e $\mathbf{W}_Q$,
$\mathbf{W}_K$, $\mathbf{W}_V$ proiezioni apprese, che moltiplicano a destra le
sequenze secondo la
convenzione per righe (un token per riga) di quel capitolo. Come nel decoder del Transformer originale, le
query vengono da chi genera e le key/value dalla sorgente da consultare:
solo che qui chi genera è un'immagine e la sorgente è una frase.

**4. Decodifica.** Al termine della catena inversa, il decoder riporta il
latente ripulito ai pixel: $\hat{\mathbf{x}} = \mathcal{D}(\mathbf{z}_0)$.

Gli ordini di grandezza di Stable Diffusion v1: U-Net da circa 860 milioni di
parametri, text encoder da 123 milioni (congelato), addestramento su
sottoinsiemi in lingua inglese, filtrati per qualità estetica, di LAION-5B (un
catalogo aperto di oltre cinque miliardi di coppie immagine–didascalia
raccolte dal web).

`````

Vale la pena fissare l'asimmetria che ne risulta, ed è un'asimmetria di
lavoro, non di listino. *Addestrare* Stable Diffusion è rimasto un mestiere da
data center: la scheda tecnica del modello dichiara centocinquantamila ore di
calcolo su schede grafiche professionali, cioè una macchina sola accesa per
diciassette anni. *Usarlo*, grazie al trasloco nelle schede compresse, costa
quattro gigabyte di memoria video e qualche secondo. È il secondo di questi
due conti, non il primo, ad aver cambiato chi può partecipare.

## Due bussole: quanto dare retta alla richiesta

Manca un ingrediente, quello che decide *quanto* l'immagine obbedisce alla
richiesta. L'occhiata al testo, da sola, è un suggerimento più che un
ordine: lasciato libero, il modello tende a produrre immagini plausibili che
rispettano il testo solo in parte; il gatto c'è, l'acquerello si è perso per
strada. Il correttivo standard, usato da Stable Diffusion e da praticamente
tutti i modelli che disegnano su richiesta, si chiama **classifier-free
guidance** (letteralmente «guida senza classificatore»: fra poco si capirà da
dove viene il nome), è di Jonathan Ho e Tim Salimans {cite}`ho2022classifier`,
ed è di una semplicità che spiazza.

`````{tab} Elementare

Il trucco comincia in addestramento: circa una volta su dieci, al modello
il prompt viene *nascosto*. Così impara due mestieri insieme: disegnare
«un'immagine plausibile qualunque» quando non ha indicazioni, e disegnare
«quello che dice il testo» quando le ha.

In generazione hai quindi due bussole, ed è proprio la bussola della sezione
precedente, quella che indica come ritoccare l'immagine per renderla più
credibile, che si sdoppia. Una punta verso «immagini credibili in generale»,
ed è quella di prima tale e quale; l'altra verso «immagini credibili *che
rispettano la tua richiesta*».
La differenza tra le due direzioni ti dice esattamente da che parte sta il
prompt, e il colpo di genio è camminare *esagerando* quella differenza: non un
passo verso il prompt, ma sette e mezzo (è letteralmente il valore di default
di Stable Diffusion, $w = 7{,}5$). Con $w$ basso, il modello va quasi a
briglia sciolta: immagini varie, prompt preso alla leggera. Con $w$ alto,
aderenza ferrea ma meno fantasia; ed esagerando davvero ($w$ oltre 15–20),
l'immagine viene «sovracotta»: colori saturi, contrasti duri, composizioni
tutte uguali.

I **negative prompt** sono la stessa idea usata al contrario: al posto della
bussola «qualunque cosa» ne metti una che punta verso ciò che *non* vuoi
(«sfocato, deforme, watermark») e cammini allontanandotene.

`````

`````{tab} Superiore

In addestramento il condizionamento viene azzerato con probabilità fissa
(*condition dropout*, circa $0{,}1$ in Stable Diffusion): la stessa rete
apprende sia $\boldsymbol{\epsilon}_\theta(\mathbf{z}_t, c)$ sia il caso non condizionato
$\boldsymbol{\epsilon}_\theta(\mathbf{z}_t, \varnothing)$, dove $\varnothing$ è il prompt vuoto.
In inferenza le due predizioni si combinano per **estrapolazione**:

$$
\tilde{\boldsymbol{\epsilon}}_\theta(\mathbf{z}_t, c) =
\boldsymbol{\epsilon}_\theta(\mathbf{z}_t, \varnothing)
+ w \big( \boldsymbol{\epsilon}_\theta(\mathbf{z}_t, c) - \boldsymbol{\epsilon}_\theta(\mathbf{z}_t, \varnothing) \big),
$$

dove $\tilde{\boldsymbol{\epsilon}}_\theta$ è la predizione di rumore effettivamente usata
dal campionatore, $c$ il prompt e $w$ il **peso di guidance**: con $w = 1$ si
recupera la predizione condizionata, con $w > 1$ ci si spinge *oltre*, nella
direzione che separa il condizionato dal non condizionato (nella
parametrizzazione originale di Ho e Salimans il coefficiente è scritto
$1 + w$; la sostanza non cambia).

**L'ispirazione**, ed è bene chiamarla così e non «l'interpretazione»: la
differenza tra le due predizioni approssima
$-\sqrt{1-\bar{\alpha}_t}\,\nabla_{\mathbf{z}_t} \log p(c \mid \mathbf{z}_t)$, con il fattore
*negativo* che lega $\boldsymbol{\epsilon}$ e score nella sezione sotto il cofano: la
differenza punta nel verso opposto al gradiente, ed è proprio sommandola alla
predizione di rumore (che il campionatore poi sottrae) che si sale su
$\log p(c \mid \mathbf{z}_t)$. È ciò che la *classifier guidance* di
{cite}`dhariwal2021diffusion` otteneva addestrando un classificatore esterno,
ed è da lì che viene il nome «senza classificatore».

Qui però bisogna fermarsi, perché la formula suggerisce una conclusione che
gli autori del metodo **negano esplicitamente**: il classificatore implicito
non c'è. Essendo $\boldsymbol{\epsilon}_\theta$ una rete non vincolata, il campo
$\tilde{\boldsymbol{\epsilon}}_\theta$ non è in generale conservativo, quindi non esiste
alcun potenziale (nessuna log-verosimiglianza di classificatore) di cui sia il
gradiente; sono parole di Ho e Salimans nel paper già citato, che aggiungono
che il passo lungo $\tilde{\boldsymbol{\epsilon}}_\theta$ non può essere letto come un
attacco avversario a un classificatore di immagini. Il «classificatore
implicito» è una guida al ragionamento, non un oggetto che esiste da qualche
parte.

E c'è una seconda conseguenza, che le interfacce non dichiarano mai: per
$w > 1$ il campionatore **non campiona più da $p(\mathbf{x} \mid c)$**, e nemmeno da
$p(\mathbf{x})\,p(c \mid \mathbf{x})^w$, che in generale non è normalizzabile. Non campiona,
cioè, da nessuna distribuzione scritta. Bradley e Nakkiran
{cite}`bradley2024classifier` lo chiudono mostrando che nessuno dei due
campionatori usuali, con la guidance attiva, genera la distribuzione che si
suppone generi, e che la guidance è piuttosto un metodo predittore-correttore
che alterna un passo di denoising e uno di affilatura. Il $w = 7{,}5$ di
default sta quindi in un regime scelto per il **giudizio umano**, ben oltre il
punto in cui la somiglianza statistica con i dati veri comincia a peggiorare:
non è una manopola della qualità, è una manopola della preferenza, e la
distinzione conta ogni volta che si valuta un modello con una metrica invece
che con gli occhi.

Il prezzo è dunque triplice. Computazionale: due valutazioni della U-Net per
ogni passo (in pratica, un batch di due). Statistico: al crescere di $w$
aumenta l'aderenza al prompt ma cala la diversità dei campioni, con
saturazione e artefatti per valori estremi (il trade-off fedeltà–varietà
misurato sistematicamente nel paper {cite}`ho2022classifier`). E concettuale:
si perde la garanzia di star campionando da una distribuzione definita. Il
*negative prompt* delle interfacce comuni è la stessa formula con
$\varnothing$ sostituito da un prompt negativo $c_{\mathrm{neg}}$: si
estrapola allontanandosi da esso.

`````

## Dieci righe di Python

Non reimplementeremo la pipeline: il modo onesto di usare Stable Diffusion è
la libreria `diffusers` di Hugging Face, che gira su PyTorch e impacchetta
VAE, U-Net, text encoder e campionatore in un oggetto solo
(`pip install diffusers transformers accelerate`). Al primo avvio scarica i
pesi (qualche gigabyte) e serve una GPU NVIDIA con circa quattro gigabyte di
memoria video.

```{code-block} python
:class: pt-non-eseguibile

import torch
from diffusers import StableDiffusionPipeline

# carica l'intera pipeline (VAE + U-Net + CLIP + campionatore)
pipe = StableDiffusionPipeline.from_pretrained(
    "stable-diffusion-v1-5/stable-diffusion-v1-5",
    torch_dtype=torch.float16,  # mezza precisione: meno memoria
)
pipe = pipe.to("cuda")          # sposta tutto sulla GPU

immagine = pipe(
    prompt="a black cat jumping on a wall, watercolor",
    negative_prompt="blurry, deformed, watermark",
    guidance_scale=7.5,         # il peso w della guidance
    num_inference_steps=50,     # i passi di denoising nel latente
).images[0]

immagine.save("gatto_acquerello.png")
```

Due note pratiche. Il prompt è in inglese perché i modelli della famiglia
SD v1 sono addestrati su didascalie in inglese: con altre lingue i
risultati peggiorano sensibilmente. E se la memoria non basta,
`pipe.enable_attention_slicing()` scambia un po' di velocità per un
consumo molto più basso.

## L'onda lunga dei pesi aperti

Il rilascio dei pesi ha fatto qualcosa che nessuna API può fare: ha permesso a
chiunque di *modificare* il modello. Nel giro di mesi è nato un ecosistema di
personalizzazioni leggere. Con **LoRA** {cite}`hu2022lora`, una tecnica nata
per i modelli di linguaggio, si specializza il modello su uno stile o un
soggetto senza toccarlo tutto: accanto ai pesi originali, che restano fermi, si
addestrano due tabelle di numeri molto più piccole che ne correggono l'uscita.
Il file da condividere pesa qualche megabyte invece di qualche gigabyte, ed è
la ragione per cui gli stili si sono messi a circolare come circolano le
canzoni. Con **ControlNet** {cite}`zhang2023adding` si vincola invece la
generazione a uno schizzo, una posa o una mappa di profondità forniti
dall'utente. Non li approfondiremo, ma sono il motivo per cui attorno a Stable
Diffusion esiste una comunità e non solo un'utenza.

Le versioni successive raccontano una traiettoria che qui interessa per una
ragione sola: dice dove è andata a finire l'architettura. Le prime rifiniscono
la ricetta senza cambiarla; poi la si ingrandisce, con una rete più capiente,
due lettori di testo invece di uno e una risoluzione nativa doppia; infine, nel
2024, arriva il passo che a questo punto del libro suona inevitabile, cioè
buttare la U-Net e metterci al suo posto un Transformer. Come e perché la
diffusione e i Transformer si siano incontrati è la storia della prossima
sezione.

## Le domande che restano aperte

Chiudiamo con la stessa onestà usata per bias e allucinazioni nel capitolo
sui Transformer, perché i nodi sono paralleli e altrettanto strutturali.

Il primo è il **consenso**: la stessa pipeline che dipinge un gatto in
acquerello genera il volto di una persona reale in una scena mai avvenuta, e i
pesi aperti rendono le contromisure centralizzate (filtri, blocchi nei prompt)
facili da rimuovere. Il secondo sono i **dati**: LAION è raccolto dal web, e
contiene anche opere protette da diritto d'autore e immagini di persone che
non hanno mai acconsentito. Su questo si è aperto un contenzioso vero:
all'inizio del 2023 Getty Images ha citato in giudizio Stability AI, e un
gruppo di artisti ha avviato una class action contro Stability AI, Midjourney
e DeviantArt; mentre scriviamo, i procedimenti hanno prodotto esiti parziali e
diversi da una giurisdizione all'altra, e la questione di fondo (se addestrare
su opere protette sia un uso lecito) non ha ancora una risposta stabile. Il
terzo è la **provenienza**: il codice di rilascio di Stable Diffusion
incorporava di serie una filigrana invisibile nelle immagini generate, e
standard come le *Content Credentials* del consorzio C2PA provano a
certificare l'origine dei contenuti; ma chi ha i pesi può disattivare la
filigrana con una riga di codice, e il riconoscimento a posteriori resta una
rincorsa. Sono problemi aperti nel senso pieno: tecnici solo in parte, e non
risolvibili solo con la tecnica.

`````{tab} Elementare

```{admonition} Da ricordare
:class: important
- Lavorare sui pixel è uno spreco: quasi tutti i 786.432 numeri di una
  fotografia servono a descrivere la grana, non il gatto. L'idea di questa
  sezione è **spostare tutto il lavoro su una versione compressa** della
  fotografia, quarantotto volte più piccola, e tornare ai pixel solo alla fine.
- Chi comprime è l'**archivista**: per ogni quadro scrive una scheda molto più
  piccola, e la prova che la scheda è buona è che un collega, leggendo solo
  quella, sappia ridipingere il quadro. La sua trovata è non scrivere un valore
  esatto ma un valore *con un margine*, così che schede vicine diventino
  immagini simili e l'archivio non abbia buchi.
- Quello che l'archivista non annota è **perso per sempre**: la scheda è il
  soffitto della qualità finale, e nessuna bravura del restauratore lo alza. È
  una delle ragioni per cui i primi modelli di questa famiglia sbagliavano
  scritte, volti piccoli e mani, che sono tutte cose di dettaglio fine.
- La ricetta in quattro mosse: l'archivista comprime, il restauratore fa il suo
  solito mestiere sulle schede invece che sui quadri, tenendo d'occhio la
  commissione scritta dal cliente, poi l'archivista ridipinge. L'archivista
  impara prima e poi smette di imparare.
- Per decidere **quanto dare retta alla richiesta** si usano due bussole (una
  punta alle immagini credibili in generale, l'altra a quelle che rispettano la
  richiesta) e si cammina esagerando la differenza fra le due. Più si esagera,
  più il modello ubbidisce e meno inventa; esagerando troppo, l'immagine viene
  «sovracotta».
- I pesi aperti hanno generato una comunità e non solo un'utenza: sono nate
  tecniche per specializzare il modello con file da pochi megabyte, o per
  costringerlo a seguire uno schizzo.
- Restano aperti i nodi del **consenso**, dei **diritti sui dati** con cui
  questi modelli sono addestrati e della **provenienza** delle immagini che
  producono. Sono i problemi paralleli ai bias e alle allucinazioni dei modelli
  di linguaggio, e altrettanto strutturali: non si risolvono con la sola
  tecnica.
```

`````

`````{tab} Superiore

```{admonition} Da ricordare
:class: important
- Diffondere sui pixel è uno spreco: gran parte dei $786\,432$ numeri di
  un'immagine $512 \times 512$ è dettaglio percettivo. I *latent diffusion
  models* {cite}`rombach2022high` spostano la diffusione in uno spazio
  compresso ($64 \times 64 \times 4$: 48 volte meno).
- Il traslocatore è il **variational autoencoder** {cite}`kingma2014auto`:
  encoder $q_\phi(\mathbf{z} \mid \mathbf{x})$ e decoder $p_\psi(\mathbf{x} \mid \mathbf{z})$ addestrati
  sull'ELBO, che rende il latente continuo e campionabile. In Stable
  Diffusion è addestrato prima e poi congelato, e la sua capacità di
  ricostruzione è un **limite superiore** sulla qualità del sistema.
- Il latente va **riscalato** prima di diffonderci sopra (in SD 1.x per la
  costante $0{,}18215$): lo schedule *variance-preserving* presuppone varianza
  unitaria, che il latente del VAE non ha.
- La ricetta: encoder → diffusione con U-Net nel latente → decoder; il prompt,
  trasformato dal text encoder di CLIP {cite}`radford2021learning`, entra
  nella U-Net via **cross-attention** (la stessa formula dei Transformer, con
  $\mathbf{Q}$ dal latente e $\mathbf{K}$, $\mathbf{V}$ dal testo).
- La **classifier-free guidance** {cite}`ho2022classifier` addestra il
  modello anche senza prompt e in inferenza estrapola tra predizione
  condizionata e non, con peso $w$: più aderenza al testo, meno varietà. Il
  «classificatore implicito» è un'ispirazione, non un oggetto (il campo
  guidato non è conservativo), e per $w > 1$ non si campiona più da
  $p(\mathbf{x} \mid c)$ {cite}`bradley2024classifier`. I *negative prompt* sono la
  stessa formula al contrario.
- I pesi aperti (agosto 2022) hanno generato un ecosistema (LoRA
  {cite}`hu2022lora`, ControlNet {cite}`zhang2023adding`, interfacce di
  comunità) e una traiettoria che finisce col sostituire la U-Net con un
  Transformer (prossima sezione).
- Restano aperti i nodi di consenso, diritti sui dati di addestramento e
  provenienza delle immagini: paralleli ai bias e alle allucinazioni dei
  modelli di linguaggio, e altrettanto strutturali.
```

`````
