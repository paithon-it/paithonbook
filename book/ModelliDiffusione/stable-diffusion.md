# Lo spazio latente: Stable Diffusion

Il 22 agosto 2022 compare online un file da circa quattro gigabyte. Dentro ci
sono i **pesi** di **Stable Diffusion**, un modello che disegna un'immagine a
partire da una frase scritta: i pesi sono i milioni di numeri che una rete si
ritrova dentro dopo l'addestramento, cioè tutto quello che ha imparato, e
averli vuol dire avere il modello. È nato dai *latent
diffusion models* del gruppo di Björn Ommer all'Università Ludwig Maximilian
di Monaco {cite}`rombach2022high`, sviluppato con Runway e addestrato con la
potenza di calcolo di Stability AI. La novità non è la qualità delle immagini
(DALL·E 2 e Imagen, usciti pochi mesi prima, erano già impressionanti) ma le
condizioni: quei modelli vivevano nei data center dei loro proprietari,
accessibili con il contagocce dietro liste d'attesa e interfacce controllate.
Stable Diffusion invece si *scarica*. Chiunque, gratis, può metterlo sul
proprio computer, e per farlo girare basta la **GPU** di un computer da
videogiochi, cioè il processore grafico, quel pezzo che nei giochi disegna le
immagini a schermo e che qui fa i conti del modello. Nel giro di poche
settimane i forum si riempiono di immagini, spuntano interfacce grafiche
amatoriali, plugin per
Photoshop e Blender, versioni modificate per ogni gusto. La generazione di
immagini smette di essere una demo da guardare e diventa uno strumento da
usare.

La domanda di questa sezione è: che cosa lo rende possibile *tecnicamente*?
Non un modello più grande: al contrario, uno più piccolo. Il segreto è un
trasloco: la diffusione che conosciamo fa le valigie, lascia i pixel e si
trasferisce in uno spazio compresso, decine di volte più piccolo, dove ogni
passo di pulitura costa una frazione. Vale la pena dire subito in che moneta si
paga, perché in tutta la sezione parleremo di costi: si paga in **conti da
fare**, cioè in secondi di attesa e in memoria occupata sulla GPU. Meno numeri
da elaborare, meno conti, meno attesa.

Per capire il trasloco, però, dobbiamo prima conoscere il traslocatore, che è
una rete a sé, diversa da quella che toglie il rumore, e si chiama
*variational autoencoder*. Metà di quel nome il libro ce l'ha già: un
**autoencoder** è una rete che impara a comprimere e a ricostruire, e il
capitolo sull'audio ne ha montato uno per i codec neurali, cioè per comprimere
il suono. Nuova è l'altra metà, il *variational*, ed è la sola che guarderemo
per esteso, perché è quella che rende il trasloco possibile.

## Il prezzo dei pixel

Facciamo due conti. Nella sezione precedente un'immagine era una griglia di
numeri, uno per pixel, perché la guardavamo in bianco e nero. A colori i numeri
per pixel diventano tre, uno per ciascuno dei colori con cui uno schermo compone
tutti gli altri (rosso, verde, blu). Un'immagine a colori di
$512 \times 512$ pixel è fatta quindi di
$512 \times 512 \times 3 = 786\,432$ numeri. Il restauratore della sezione
precedente (la U-Net {cite}`ronneberger2015u` che predice il rumore) deve
elaborarli *tutti*, e deve farlo a ogni passo di pulitura: centinaia o
migliaia di passaggi per una sola immagine. Con le centinaia di macchine di un
laboratorio si può fare; su un computer di casa no.

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
compatta che conserva il contenuto e scarta il dettaglio ricostruibile. La
diffusione, poi, impara la **composizione** dentro quello spazio compatto, dove
ogni passo costa decine di volte meno.

Quello spazio compatto ha un nome, e il libro l'ha già battezzato parlando dei
codec neurali nel capitolo sull'audio: si chiama **spazio latente**, cioè
l'insieme dei riassunti che una rete si costruisce da sola, «latenti» perché
nessuno le ha insegnato come farli e a guardarli non dicono niente. Là dentro
c'era del suono, qui ci sono immagini; e cambia soprattutto che cosa ci si fa.
Per un codec quello spazio è un corridoio: ci si entra da una parte per
comprimere e si esce dall'altra. Qui invece ci si va ad abitare, perché è lì che
avverrà tutta la diffusione. La ricetta si chiama infatti dei *latent diffusion
models*, e Stable Diffusion ne è il figlio famoso.

## L'archivista: il variational autoencoder

Il pezzo che porta i mobili è l’**autoencoder**, la rete a clessidra che il
capitolo sull'audio ha montato per i codec neurali: una metà stringe quello che
entra fino a farlo passare per una strettoia dove i numeri sono molti meno,
l'altra metà da quella strettoia prova a ritirare fuori l'originale, e il voto è
uno solo per tutte e due, cioè quanto quello che esce somiglia a quello che è
entrato. (Le due metà hanno i nomi inglesi che si trovano nel codice, *encoder*
e *decoder*.) Quel voto, da solo, insegna a comprimere e nient'altro, e a noi
non basta: in quello spazio bisognerà anche pescare punti a caso e pretendere
che ne esca un'immagine. Ecco perché ci serve la variante,
il **variational
autoencoder** (VAE) di Diederik Kingma e Max Welling {cite}`kingma2014auto`, del
2014 (più vecchio della diffusione moderna e persino delle GAN) e destinato a
tornare più avanti nel libro. Ce ne serve l'essenziale: che cosa aggiunge alla
clessidra, e perché quel poco rende lo spazio latente un posto dove si può
lavorare. Per parlarne useremo una
metafora che ci accompagnerà fino alla fine del capitolo. Le due metà della
clessidra diventano due persone: la rete che comprime è un **archivista** e
la rappresentazione compatta che scrive è la sua **scheda**; la rete che
ricostruisce è un **copista**, che dalla scheda ridipinge il quadro. Da qui
in avanti «scheda» vorrà dire sempre e solo questo.

```{figure} ../figures/vae-auto-encoding-variational-bayes.svg
:name: fig-vae
:alt: "Schema del variational autoencoder: l'immagine entra nell'encoder, che non produce un punto ma una media e una deviazione standard; da quella distribuzione si campiona un punto nello spazio latente; il decoder riceve il punto campionato e ricostruisce l'immagine. La perdita somma il termine di ricostruzione e il termine che tiene la distribuzione vicina al prior."
:width: 100%

La rete che comprime non restituisce una scheda sola, ma una scheda **e un
margine di tolleranza**: «all'incirca questo, più o meno tanto». È quel margine
la trovata, ed è ciò che costringe schede vicine a ridiventare immagini simili.
(Le lettere greche del disegno sono i nomi tecnici delle stesse cose:
$\mu$ il valore scritto sulla scheda, $\sigma$ il margine di tolleranza, e il
pallino nero il punto sorteggiato dentro quel margine. La scheda, quindi, è la
coppia: il valore *e* il margine.)
```

La {numref}`fig-vae` dà per scontata una cosa che vale la pena fissare. Una
scheda è una lista di numeri, e come tale si può immaginare come un **punto su
una mappa**, esattamente la mappa delle immagini
possibili di poche pagine fa: schede simili sono punti vicini, e fra due punti
c'è sempre tutto lo spazio in mezzo. È quello che permette di dire frasi come
«una scheda a metà strada fra due che esistono», che con dei foglietti di carta
non vorrebbero dire niente.

Sulla mappa si vede anche perché un archivista semplice, quello senza il
margine di tolleranza, non serve a inventare niente. Comprime e ricostruisce, e
in quel mestiere è ottimo. Ma proviamo a usarlo al contrario: scegliamo un
punto sulla mappa, diamolo al copista, guardiamo che cosa dipinge. Il guaio è
che non si sa **dove** sceglierlo. Nessuno ha mai chiesto alle schede di stare
in una zona precisa, e quindi si sistemano dove capita: la regione che occupano
non ha una forma nota né un centro noto, i vari soggetti (gatti, ritratti,
paesaggi) se la spartiscono male, chi prendendosi una fetta larga chi restando
schiacciato in una scaglia sottile, e in mezzo restano dei vuoti. Un punto
pescato a caso finisce quasi sempre fuori
dalla regione o dentro uno di quei vuoti, e il copista, che lì non è mai stato,
dipinge una macchia. Non è una svista dell'archivista: nella pagella con cui
l'abbiamo giudicato («la copia somiglia all'originale?») la richiesta di
generare non compariva da nessuna parte, e quello che non si chiede non si
ottiene.

Il sorteggio in mezzo alla figura (si prende la scheda, la si sposta di un po’
a caso dentro il margine di tolleranza, e solo allora la si passa al copista) è
ciò che distingue questa rete da un compressore qualunque. La conseguenza è
precisa: siccome durante l'addestramento il copista vede ogni volta una scheda
leggermente spostata, è costretto a funzionare su tutta una zona e non su un
punto solo. Lo spazio delle schede ne esce **continuo**, cioè senza i vuoti di
prima: un punto a metà strada fra due schede vere non è più terra sconosciuta,
e il copista sa che farsene. Ed è la premessa perché la diffusione ci si
possa muovere dentro, dato che la diffusione, di suo, passa il tempo a mettere
piede in posti sorteggiati a caso.

Il sorteggio, però, è metà del rimedio: aggiusta le vicinanze, una scheda alla
volta, e non dice ancora dove sia la regione in cui pescare. La seconda metà è
una regola che tiene tutte le schede raccolte attorno a uno stesso centro: è il
secondo dei due tratti che danno a questa rete il suo nome, e li vediamo
adesso.

`````{tab} Elementare

Immagina l'archivista di un museo pieno di quadri enormi. È la stessa clessidra
che il capitolo sull'audio usava per comprimere il suono, con due mestieri al
posto delle due metà: per ogni quadro l'archivista scrive una scheda molto più
piccola dell'originale, e il copista deve *ridipingere* il quadro leggendo solo
quella. Se la copia somiglia all'originale la scheda conteneva l'essenziale, e i
due si allenano insieme perché una scheda è buona o cattiva solo rispetto a chi
la deve leggere.

Dopo milioni di prove su milioni di quadri, l'archivista ha imparato da solo
che cosa annotare (soggetto, composizione, colori dominanti) e che cosa
lasciar perdere: la grana della tela, le singole pennellate dello sfondo. Non
perché il copista se le ricordi, ma perché se le **inventa**, e a nessuno
importa che siano proprio quelle: una grana di tela vale l'altra, e nessuno va
a controllarla filo per filo.

Nei numeri di Stable Diffusion: il quadro è fatto di 786.432 valori, la scheda
di 16.384, quarantotto volte meno. E quel 16.384 non è un numero magico ma una
scelta di progetto: la scheda è una griglia di 64 caselle per lato invece delle
512 dell'immagine (otto volte meno per lato) con quattro numeri per casella, e
$64 \times 64 \times 4$ fa appunto 16.384. I quattro numeri, a differenza dei
tre dell'immagine, non hanno un significato leggibile: non sono rosso, verde e
blu, sono quattro coordinate che l'archivista si è scelto da solo e che nessuno
gli ha insegnato. Guardarli non dice niente a un essere umano; al copista sì.

Va detta anche l'altra metà, perché il capitolo ci tornerà: **la compressione
distrugge**, e quello che l'archivista non ha annotato non lo recupera più
nessuno, per quanto bravo sia chi lavora dopo di lui. La scheda è il soffitto
della qualità finale. Quattro numeri per casella bastano per un gatto su un
muro e non bastano per una scritta leggibile, per un volto in secondo piano o
per una mano con cinque dita: se il copista si inventa la grana della tela
nessuno se ne accorge, se si inventa le lettere di un'insegna se ne accorgono
tutti. È una delle ragioni (non l'unica) dei difetti tipici della prima
generazione di questi modelli, che sono sempre difetti di dettaglio fine.
Alzare quel soffitto è una delle cose che i successori hanno fatto.

E il «variational» del nome, che in italiano diremmo «variazionale»? Sta in due
regole che tengono l'archivio in ordine. Primo: la scheda non inchioda il
quadro a un punto esatto ma descrive una *nuvola di possibilità* («un gatto
nero più o meno così»), cosicché quadri quasi uguali abbiano schede quasi
uguali. Secondo: le schede devono stare tutte raccolte attorno a uno stesso
centro, invece di sparpagliarsi dove capita, e questo serve a sapere **dove
pescare**. Se so che l'archivio sta lì attorno posso inventarmi un punto senza
finire fuori dal mondo dei quadri possibili, e il copista deve saperne
dipingere un quadro sensato anche se quel punto non l'ha mai scritto nessuno.
Sembrano
pignolerie, ma sono esattamente ciò che serve alla diffusione: il restauratore
lavorerà *dentro* questo archivio, e ogni punto in cui mette piede (compresi i
mille punti sorteggiati del suo viaggio) deve corrispondere a un'immagine
possibile.

`````

`````{tab} Superiore

Un VAE è una coppia di reti. L’**encoder** mappa il dato $\mathbf{x}$ non in un punto
ma in una distribuzione sul latente,
$q_\phi(\mathbf{z} \mid \mathbf{x}) = \mathcal{N}\big(\mathbf{z};\, \boldsymbol{\mu}_\phi(\mathbf{x}),\, \mathrm{diag}\big(\boldsymbol{\sigma}_\phi^2(\mathbf{x})\big)\big)$
(la covarianza è diagonale, con una varianza propria per componente, non un
unico valore per tutte);
il **decoder** definisce $p_\psi(\mathbf{x} \mid \mathbf{z})$, la ricostruzione a partire
dal codice (scriviamo $\psi$ per i suoi parametri perché in questo capitolo
$\theta$ è già impegnato dalla rete di diffusione $\boldsymbol{\epsilon}_\theta$: sono due
reti distinte, addestrate separatamente). Sul latente si
impone un prior semplice, $p(\mathbf{z}) = \mathcal{N}(0, \mathbf{I})$. L'addestramento
massimizza l’**ELBO** (*evidence lower bound*):

$$
\mathrm{ELBO}(\psi, \phi; \mathbf{x}) =
\mathbb{E}_{q_\phi(\mathbf{z} \mid \mathbf{x})}\!\big[\log p_\psi(\mathbf{x} \mid \mathbf{z})\big]
- D_{KL}\!\big(q_\phi(\mathbf{z} \mid \mathbf{x}) \,\|\, p(\mathbf{z})\big),
$$

dove il primo termine premia la fedeltà della ricostruzione e il secondo (la
divergenza di Kullback–Leibler vista nei richiami di matematica) penalizza gli
encoder che si allontanano dal prior. È questo secondo termine a rendere lo
spazio latente **continuo** (input simili, codici vicini) e **campionabile**,
cioè a fornire una distribuzione da cui pescare $\mathbf{z}$ senza doverla
stimare: l'obiettivo è che ogni regione con probabilità apprezzabile sotto il
prior decodifichi in un dato plausibile, e quanto ci si riesca davvero è la
domanda del paragrafo che segue. Nella convenzione del libro, dove $\mathcal{L}$ si
minimizza, la loss corrispondente è $\mathcal{L} = -\mathrm{ELBO}$. La
derivazione dell'ELBO come limite inferiore della
log-verosimiglianza è nel paper originale {cite}`kingma2014auto`; qui ci basta
il ruolo funzionale dei due termini.

Conviene dire che cosa manca a un autoencoder semplice, perché è esattamente
ciò che il secondo di quei due termini aggiunge. Un autoencoder ottimizza la sola
ricostruzione: nella sua loss non compare nulla che riguardi la distribuzione
dei codici che produce. L'aggregato
$q_\phi(\mathbf{z}) = \mathbb{E}_{p_{\text{dati}}}\!\big[q_\phi(\mathbf{z} \mid \mathbf{x})\big]$
resta quindi ignoto, e generare richiede di conoscerlo: senza, non esiste
alcuna distribuzione da cui pescare $\mathbf{z}$. Il termine KL scioglie il nodo
imponendo il bersaglio invece di stimarlo, ed è il motivo per cui il prior si
sceglie semplice. Con una precisazione che vale la pena registrare: quel
termine agisce **su un esempio alla volta**, quindi vincola ciascuna
$q_\phi(\mathbf{z} \mid \mathbf{x})$ e non direttamente l'aggregato. I due non
coincidono, e lo scarto lascia regioni con massa apprezzabile sotto il prior
che il decoder ha visto poco: campionarle produce immagini deboli. È un limite
noto del VAE puro, non un difetto di implementazione, e ha anche un nome e una
letteratura: lo scarto fra prior e posteriore aggregato
{cite}`hoffman2016elbo,rosca2018distribution`. In Stable Diffusion il
problema si aggira non risolvendolo: al VAE non si chiede affatto di generare,
il peso KL è tenuto molto piccolo (la fedeltà della ricostruzione conta più
della somiglianza al prior) e a decidere che cosa esce dal latente pensa la
diffusione, non il decoder da solo.

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

Manca un pezzo solo, che finora questa sezione ha tenuto in disparte: **il
testo**. Le fotografie con cui questi modelli si
addestrano non arrivano nude, arrivano con una didascalia accanto («un gatto
nero seduto su un muro»), raccolta insieme all'immagine dal sito da cui è stata
presa. È così che il modello impara ad associare le parole alle cose, ed è il
motivo per cui alla fine gli si potrà scrivere che cosa disegnare.

La {numref}`fig-latent-diffusion` mette allora in fila tutto: la rete che
comprime, lo spazio delle schede dove avviene la diffusione, il testo che entra
di lato, la rete che riporta ai pixel. Un dettaglio dell'ordine dei lavori
conta più di quanto sembri: l'archivista impara il suo mestiere *prima*, da
solo, e poi **smette di imparare**. Da quel momento in avanti è uno strumento
fisso, e mentre il restauratore si allena nessuno gli tocca più niente. In
gergo si dice che i suoi pesi vengono *congelati* (i pesi sono i numeri interni
della rete, quelli che decidono le sue risposte, e congelarli vuol dire
smettere di ritoccarli). La ragione è che il restauratore deve allenarsi su un
archivio che non cambia sotto i suoi occhi.

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
solito mestiere (sporca di rumore, impara a indicare il disturbo) ma su schede
da 16.384 numeri invece che su quadri da 786.432, come restaurare cartoline
anziché affreschi. Ogni giro di domanda e risposta costa decine di volte meno.
Non esattamente quarantotto volte meno, e il motivo è che la rete che lavora
sulle schede non è quella dei quadri rimpicciolita, è una rete progettata
apposta e con i suoi numeri.

**Terza**: mentre pulisce, il restauratore tiene sul tavolo la commissione
scritta dal cliente («un gatto nero che salta sul muro, in acquerello») e a
ogni pennellata le dà un'occhiata, soffermandosi sulle parole che servono in
quel momento: «nero» quando decide i toni, «acquerello» quando decide il
tratto. È la stessa occhiata selettiva dell'interprete della traduzione
automatica, ritrovata poi nei Transformer: lì collegava due lingue, qui
collega parole e immagine. (Quella commissione, nel gergo di tutti i giorni,
si chiama **prompt**, ed è la frase che si scrive nella casella di un
generatore di immagini. Da qui in avanti useremo le due parole come sinonimi.)
**Quarta**: finita la pulitura, la scheda passa al copista, che ridipinge il
quadro a piena risoluzione.

Per **generare** un'immagine nuova si parte, come sempre, dalla fine: una
scheda di puro rumore sorteggiato, mai appartenuta a nessun quadro. Il
restauratore la pulisce passo dopo passo con la commissione sotto gli
occhi, e il copista trasforma il risultato in pixel. Il gatto in
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
convenzione per righe (un token per riga) di quel capitolo. Come nel decoder
del Transformer originale, le
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

Vale la pena fissare l'asimmetria che ne risulta, e riguarda il lavoro da
fare, non il prezzo da pagare a qualcuno. *Addestrare* Stable Diffusion è rimasto un mestiere da
data center: la documentazione del modello dichiara centocinquantamila ore di
calcolo su GPU professionali, cioè una macchina sola accesa per diciassette
anni. *Usarlo*, grazie al trasloco nelle schede compresse, chiede alla GPU
quattro gigabyte di memoria e qualche secondo di attesa. (Che siano quattro
come i quattro del file scaricato è quasi un caso: quel file, caricato in
memoria con numeri a metà precisione, di gigabyte ne occupa circa due, e gli
altri due servono ai conti.) È il secondo di questi due conti, non il primo,
ad aver cambiato chi può partecipare.

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

Il trucco comincia in addestramento: circa una volta su dieci, al modello la
commissione viene *nascosta*. Così impara due mestieri insieme: disegnare
«un'immagine plausibile qualunque» quando non ha indicazioni, e disegnare
«quello che dice il testo» quando le ha.

In generazione, allora, a ogni passo puoi porre la domanda due volte e
ottenere due risposte. È il cartello della salita della sezione precedente,
quello che indica in che direzione ritoccare l'immagine per renderla più
credibile, che si sdoppia: un cartello dice «per una figura credibile in
generale, va’ di là», l'altro dice «per una figura credibile *e che rispetta
la richiesta*, va’ di là». Chiamiamole le due bussole, tenendo a mente che non
sono bussole vere: non indicano il nord tutte e due, ognuna indica la sua
direzione, e le due direzioni non coincidono.

Le due direzioni sono quasi uguali, e la piccola differenza fra loro è tutto
quello che il testo ha da dire. Facciamo finta di essere su una cartina: la
prima bussola dice «nord», la seconda dice «nord, un pelo verso est». Quel
«pelo verso est» è il contributo della richiesta. Seguire semplicemente la
seconda bussola si può, ed è quello che si faceva prima: il guaio è che
l'indicazione del testo, nel totale, pesa pochissimo. La rete tira soprattutto
verso «un'immagine credibile», e «acquerello» è una spintarella dentro quella
spinta grossa: si perde per strada, e il gatto viene a olio. Il colpo di genio
è prendere quel pelo verso est e
moltiplicarlo: non un passo, ma sette passi e mezzo verso est, e poi camminare
verso nord-est-est. Sette e mezzo non è una figura retorica, è il numero che
Stable Diffusion usa di serie, e si chiama il **peso della guida**, $w$.

A $w = 1$ non si esagera niente: si cammina nella direzione della seconda
bussola e basta, ed è il caso in cui il gatto viene a olio. Scendendo verso
quel valore il modello va più a briglia sciolta, con immagini varie e richiesta
presa alla leggera; salendo, ubbidisce di più e inventa di meno. Esagerando
davvero, ben oltre il 7 e mezzo, l'immagine viene «sovracotta»: colori saturi,
contrasti duri, composizioni tutte uguali.

Resta da spiegare il nome. Un **classificatore** è una rete che guarda
un'immagine e dice che cosa contiene («questo è un gatto»), e il metodo che
veniva prima faceva proprio così: addestrava un classificatore a parte e lo
usava per tirare la generazione verso la categoria voluta. Costoso, e un pezzo
in più da mantenere. Ho e Salimans ottengono lo stesso effetto senza costruire
nessun classificatore, usando due risposte della rete che c'è già: da qui
*classifier-free*, «senza classificatore».

I **negative prompt** sono la stessa idea usata al contrario: al posto della
bussola «qualunque cosa» ne metti una che punta verso ciò che *non* vuoi
(«sfocato, deforme, con una scritta sopra») e cammini allontanandotene.

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

Non riscriveremo da zero tutta la catena di montaggio: nessuno lo fa, e il modo
in cui la si usa davvero è la libreria `diffusers` di Hugging Face, che gira su
PyTorch e impacchetta in un blocco solo l'archivista, il copista, il
restauratore, la rete che legge la commissione scritta e la procedura che
scende la scala
(`pip install diffusers transformers accelerate`). Al primo avvio scarica i
pesi (qualche gigabyte) e serve una GPU NVIDIA con circa quattro gigabyte di
memoria. Senza una GPU così il blocco non gira, e non è un problema: si legge,
perché quello che c'è da capire sta nei nomi delle opzioni.

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
`pipe.enable_attention_slicing()` scambia un po’ di velocità per un
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
2024, si butta la U-Net e le si mette al posto un Transformer. Chi ha letto il
capitolo sui Transformer se lo aspettava, perché la stessa sostituzione era già
avvenuta nella traduzione e nella visione; ma che funzionasse anche qui non era
affatto scontato, e come e perché sia successo è la storia della prossima
sezione.

## Le domande che restano aperte

Il capitolo sui Transformer si era chiuso elencando i problemi aperti dei
modelli di linguaggio, senza addolcirli. Qui i problemi sono paralleli e
altrettanto strutturali, e sono tre.

Il primo è il **consenso**. La stessa catena di montaggio che dipinge un gatto
in acquerello dipinge il volto di una persona reale in una scena mai avvenuta,
e i pesi aperti rendono facili da rimuovere le contromisure decise da chi
distribuisce il modello (filtri, parole vietate nella richiesta).

Il secondo sono i **dati**. Stable Diffusion è addestrato su LAION, un enorme
elenco pubblico di indirizzi di immagini raccolte dal web con la loro
didascalia: miliardi di voci, prese dove capitava. Dentro ci sono anche opere
protette da diritto d'autore e fotografie di persone che non hanno mai
acconsentito. Su questo si è aperto un contenzioso vero. All'inizio del 2023
Getty Images ha citato in giudizio Stability AI, e un gruppo di artisti ha
fatto causa a Stability AI, Midjourney e DeviantArt. Mentre scriviamo, le
sentenze sono parziali e diverse da un paese all'altro, e la domanda di fondo,
cioè se addestrare un modello su opere protette sia lecito, non ha ancora una
risposta stabile.

Il terzo è la **provenienza**, cioè poter dire se un'immagine è stata generata
o no. Il codice di rilascio di Stable Diffusion incorporava di serie una
filigrana invisibile nelle immagini che produceva, e ci sono standard, come le
*Content Credentials* del consorzio C2PA, che provano a certificare l'origine
dei contenuti. Ma chi possiede i pesi disattiva la filigrana con una riga di
codice, e riconoscere l'origine dopo il fatto resta una rincorsa. Sono problemi
aperti nel senso pieno: tecnici solo in parte, e non risolvibili solo con la
tecnica.

`````{tab} Elementare

```{admonition} Da ricordare
:class: important
- Lavorare sui pixel è uno spreco: quasi tutti i 786.432 numeri di una
  fotografia servono a descrivere la grana, non il gatto. L'idea di questa
  sezione è **spostare tutto il lavoro su una versione compressa** della
  fotografia, quarantotto volte più piccola, e tornare ai pixel solo alla fine.
- Chi comprime è l’**archivista**: per ogni quadro scrive una scheda molto più
  piccola, e chi la legge per ridipingere il quadro è il **copista**. I due si
  allenano insieme, e la prova che la scheda è buona è che la copia somigli
  all'originale. La trovata dell'archivista è non scrivere un valore esatto ma
  un valore *con un margine*, così che schede vicine diventino immagini simili
  e l'archivio non abbia buchi.
- Quello che l'archivista non annota è **perso per sempre**: la scheda è il
  soffitto della qualità finale, e nessuna bravura di chi viene dopo lo alza. È
  una delle ragioni per cui i primi modelli di questa famiglia sbagliavano
  scritte, volti piccoli e mani, che sono tutte cose di dettaglio fine.
- La ricetta in quattro mosse: l'archivista comprime, il restauratore fa il suo
  solito mestiere sulle schede invece che sui quadri, tenendo d'occhio la
  commissione scritta dal cliente, poi il copista ridipinge. L'archivista
  impara prima e poi smette di imparare.
- Per decidere **quanto dare retta alla richiesta** si interroga la rete due
  volte, una senza dirle niente e una dandole la richiesta, e si guarda di
  quanto le due risposte differiscono: quella differenza è il contributo del
  testo, ed è piccola. Si cammina moltiplicandola, di solito per sette e mezzo.
  Più si moltiplica, più il modello ubbidisce e meno inventa; esagerando,
  l'immagine viene «sovracotta».
- I pesi aperti hanno generato una comunità e non solo un'utenza: sono nate
  tecniche per specializzare il modello con file da pochi megabyte, o per
  costringerlo a seguire uno schizzo.
- Restano aperti i problemi del **consenso** di chi finisce ritratto, dei
  **diritti sulle immagini** con cui questi modelli sono addestrati e della
  **provenienza**, cioè del riuscire a dire se un'immagine è stata generata.
  Sono altrettanto strutturali dei difetti dei modelli di linguaggio, e non si
  risolvono con la sola tecnica.
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
