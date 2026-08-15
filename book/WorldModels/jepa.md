# La via di LeCun: predire nello spazio delle idee

Il 27 giugno 2022 Yann LeCun deposita su OpenReview (la piattaforma dove di
solito si caricano gli articoli in attesa di revisione) un documento di 62
pagine intitolato *A Path Towards Autonomous Machine Intelligence*
{cite}`lecun2022path`. Già il sottotitolo è insolito: «versione 0.9.2», come
un software non ancora finito. E insolito è tutto il resto: non è un paper di
risultati, con esperimenti e tabelle, ma un **documento di posizione** (la
visione dell'autore su come costruire macchine intelligenti nei prossimi dieci
anni) messo online apposta perché chiunque potesse commentarlo, criticarlo,
smontarlo pubblicamente. Un premio Turing che espone il proprio programma di
ricerca, in bozza dichiarata, alle obiezioni di tutti: in un'epoca in cui si
tende a mostrare solo ciò che già funziona, è una mossa da notare.

Dentro c'è il disegno di una macchina autonoma, fatta di sei pezzi che si
passano il lavoro. La **percezione** guarda i sensori e ricostruisce com'è
messo il mondo adesso. Il **world model**, che è il cuore del progetto, dice
come quel mondo andrà avanti, anche se l'azione è soltanto immaginata. Il
modulo di **costo** misura quanto la situazione sia sgradita all'agente:
una parte è scritta una volta per tutte da chi progetta e l'esperienza non la
cambia (l'analogo del dolore e del piacere), l'altra si impara ed è un
*critico*, cioè una rete il cui unico mestiere è prevedere quanto costerà il
seguito, così che l'agente sappia se una mossa conviene senza aspettarne le
conseguenze. Poi ci sono l'**attore**, che propone le azioni, la **memoria a
breve termine**, che tiene il filo di quel che è appena successo, e il
**configuratore**, che sovrintende e regola gli altri a seconda del compito.

Un agente così può agire in due modi. Di riflesso, con la percezione che pilota
direttamente l'azione; oppure di testa, usando il world model per provare le
sequenze di azioni e scegliere quella dal costo previsto più basso. È un'eco
della distinzione fra pensiero veloce e pensiero lento resa celebre da Daniel
Kahneman, che LeCun richiama esplicitamente.

Sei pezzi sono tanti, e in gran parte sono ancora sulla carta. Ma tutto il
progetto sta o cade su una domanda sola: **come si addestra il world model?**
La risposta di LeCun è: guardando, come il neonato dell'inizio del capitolo.
Enormi quantità di video senza che nessuno ci abbia scritto sopra niente, e un
solo esercizio, indovinare ciò che viene dopo; la correzione arriva da sé,
perché il futuro arriva. È l'apprendimento **auto-supervisionato** di cui
parlava l'apertura del capitolo, e fin qui non c'è niente di nuovo: anche i
mondi in miniatura di Ha e Schmidhuber facevano qualcosa di simile. La rottura
è nel *dove* si fa la previsione.

## Perché non predire i pixel

Anche nei mondi in miniatura la previsione non avveniva sui puntini dello
schermo: avveniva sui 32 numeri in cui V, la rete che guardava, riassumeva il
fotogramma. Quel
riassunto, però, era stato addestrato a **rimettere insieme i puntini**: era
bravo nella misura in cui il disegno rifatto somigliava all'originale. LeCun
propone di tagliare anche quel cordone. Il futuro, osserva, ha due proprietà
che rendono una pessima idea provare a disegnarlo: è **molteplice** (da uno
stesso presente possono seguire tanti futuri diversi, tutti plausibili) ed è
pieno di **dettagli irrilevanti**. Il suo esempio ricorrente è un albero in un
video: nessun modello potrà mai prevedere la posizione esatta di ogni foglia
mossa dal vento, e soprattutto *non serve a niente* provarci.

`````{tab} Elementare

Un bicchiere è in bilico sul bordo del tavolo. «Cadrà e andrà in pezzi»: lo
prevedi in un decimo di secondo, e questa previsione ti basta per allungare la
mano. Ora prova invece a prevedere la *fotografia esatta* della scena tra due
secondi: dove sarà ogni scheggia, come si rifletterà la luce su ogni
frammento, che forma avrà la macchia d'acqua sul pavimento. Impossibile, e del
tutto inutile: nessuna decisione sensata dipende dalla forma della terza
scheggia. Un modello costretto a prevedere l'immagine pixel per pixel ha
esattamente questo problema, due volte. Primo: spreca quasi tutta la sua
capacità a studiare dettagli che non contano nulla. Secondo: siccome i futuri
possibili sono tanti (le schegge possono disporsi in mille modi) e lui deve
produrre *una* immagine sola, quella che gli costa meno errori è la media di
tutti i futuri; una foto fantasma, sfocata, in cui mille rotture diverse si
sovrappongono. La proposta di LeCun: non prevedere la foto, prevedere il
*succo* («bicchiere in pezzi sul pavimento, acqua sparsa»), cioè prevedere
nello **spazio delle idee**, dove i mille futuri diversi nei dettagli
diventano un futuro solo, quello che conta.

`````

`````{tab} Superiore

Se si addestra un predittore $g$ a minimizzare l'errore quadratico
$\mathbb{E}\,\lVert \mathbf{y} - g(\mathbf{x}) \rVert^2$ su un futuro $\mathbf{y}$ intrinsecamente
stocastico, l'ottimo è la media condizionata $g^*(\mathbf{x}) = \mathbb{E}[\mathbf{y} \mid \mathbf{x}]$:
quando i modi della distribuzione sono molti e distinti, la loro media è
un'immagine sfocata che non corrisponde a *nessun* futuro reale; è la ragione
per cui la predizione video nei pixel produce fantasmi lattiginosi. La
proposta di {cite}`lecun2022path` è la **JEPA** (*Joint-Embedding Predictive
Architecture*): due encoder mappano contesto e target nello spazio delle
rappresentazioni, $\mathbf{s}_x = f_\phi(\mathbf{x})$ e $\mathbf{s}_y = \bar{f}_{\bar{\phi}}(\mathbf{y})$, e
un **predictor** $g_\theta$ opera interamente lì:

$$
E(\mathbf{x}, \mathbf{y}, \mathbf{z}) = \big\lVert\, g_\theta(\mathbf{s}_x, \mathbf{z}) - \mathbf{s}_y \,\big\rVert_2^2,
\qquad
F(\mathbf{x}, \mathbf{y}) = \min_{\mathbf{z}} E(\mathbf{x}, \mathbf{y}, \mathbf{z}),
$$

dove $\mathbf{z}$ è una variabile latente che assorbe la molteplicità dei futuri
(quale dei tanti esiti plausibili si è realizzato) e $\phi$, $\bar{\phi}$,
$\theta$ sono i parametri dei due encoder e del predictor. L'energia della
*coppia* è la seconda quantità, $F$: si sceglie la $\mathbf{z}$ che spiega meglio il
futuro osservato, e quel minimo misura la compatibilità tra $\mathbf{x}$ e $\mathbf{y}$. Il
collegamento con il capitolo precedente è letterale: una JEPA **è** un
modello a energia non normalizzato; la compatibilità tra presente e futuro è
l'errore di predizione nello spazio latente, l'inferenza è la solita
$\arg\min$ (qui, il minimo su $z$), e della funzione di partizione non c'è
alcun bisogno.

Una precisazione, perché altrimenti quel $\min_{\mathbf{z}}$ resta un debito: $\mathbf{z}$ è la
forma **generale** dello schema proposto nel 2022, non la ricetta che poi è
stata implementata. I due sistemi che vedremo in questa sezione (I-JEPA per
le immagini, V-JEPA per i video) istanziano il caso **senza latente**: il
predictor è deterministico, $g_\theta(\mathbf{s}_x)$, quindi $F(\mathbf{x}, \mathbf{y}) = E(\mathbf{x}, \mathbf{y})$ e non
c'è alcun minimo da calcolare, né in addestramento né a inferenza. Quel poco
di «quale futuro» che serve è passato al predictor come informazione
esplicita (i token posizionali che dicono *dove* prevedere), non inferito
come variabile nascosta. Una JEPA con $\mathbf{z}$ vero, capace di produrre più esiti
plausibili invece di uno solo, resta al momento programma di ricerca.

La libertà nuova sta nell'encoder del target: poiché $\mathbf{y}$ non va ricostruito
ma solo *rappresentato*, $\bar{f}$ può legittimamente buttare via
informazione. I gradi di libertà imprevedibili e irrilevanti (le
foglie, i riflessi) possono semplicemente non arrivare nello spazio in cui si
calcola la loss: è una scelta d'architettura, non una speranza.

`````

La {numref}`fig-jepa-architettura` mette i due mondi uno sopra l'altro, e
conviene sciogliere prima le parole che porta scritte dentro, perché da qui in
poi tornano a ogni riga. Il **contesto** è la parte che il modello vede; il
**target** (bersaglio) è la parte nascosta, quella su cui deve indovinare. Un
**encoder** è la rete che guarda qualcosa e ne produce il riassunto; quel
riassunto è una fila di numeri e si chiama **embedding** (in italiano sarebbe
«immersione»: la scena è stata immersa in uno spazio fatto di numeri). Il
**predictor** è la rete che, dal riassunto di quel che si vede, tira fuori il
riassunto di quel che non si vede. Un **decoder**, invece, è la rete che dal
riassunto ridisegna un'immagine intera, puntino per puntino. E la **loss** (in
inglese «perdita») è il voto: il numero che misura quanto la risposta data si
discosta da quella giusta, e che l'addestramento passa il tempo ad abbassare.

Una parola sulle parole, già che ci siamo. Riassunto, embedding,
rappresentazione, «spazio delle idee» e, nel gergo dei paper, *latente*, in
questo capitolo indicano la stessa cosa: la manciata di numeri in cui una rete
ha condensato quello che ha guardato. Cambia il registro, non l'oggetto.

```{figure} ../figures/jepa-architettura.svg
:name: fig-jepa-architettura
:alt: "Confronto a due pannelli. Sopra, l'architettura generativa: dal contesto un decoder disegna ogni pixel del futuro e la loss confronta pixel per pixel la predizione sfocata con il futuro reale, sprecando capacità sui dettagli imprevedibili. Sotto, la JEPA: un encoder in teal trasforma il contesto in un embedding, un encoder target tratteggiato aggiornato per media mobile esponenziale trasforma il target, un predictor in terracotta predice l'embedding del target e la loss confronta i due embedding nello spazio delle rappresentazioni."
:width: 100%

Generativa contro JEPA: la prima predice il futuro nei pixel (e deve
indovinare anche l'irrilevante), la seconda lo predice nello spazio delle
rappresentazioni, dove l'irrilevante non è mai entrato.
```

Adesso la figura si legge da sé. Nel pannello A il decoder deve tornare fino ai
singoli puntini, e la loss lo punisce anche su ogni foglia che trema; nel
pannello B la previsione parte dal contesto e arriva al target senza mai uscire
dallo **spazio delle rappresentazioni**, che è lo spazio delle idee del titolo:
i dettagli irrilevanti restano fuori dalla porta. Quello del pannello B è lo
schema che dà il nome a tutta questa linea di ricerca: **JEPA**,
*Joint-Embedding Predictive Architecture*, cioè «architettura che predice fra
due riassunti»: la parola *joint*, congiunto, dice che i due riassunti vivono
nello stesso spazio, ed è lì che si possono confrontare.

## Il ritorno del collasso

Chi ha letto il capitolo sui modelli a energia sa già dove si nasconde la
trappola, perché è la stessa del buttafuori pigro: quello che, dovendo dare a
ogni coppia un voto di compatibilità (in quel capitolo il voto si chiama
**energia**, e più è basso più le due cose stanno bene insieme), scopre che il
modo più comodo di non sbagliare mai è dire sempre sì.

Qui la scorciatoia è la stessa. Se il voto premia soltanto la vicinanza fra il
riassunto predetto e quello del bersaglio, la strada più comoda non è capire il
mondo, è **appiattirlo**: basta che i due encoder imparino a produrre sempre la
stessa identica fila di numeri, qualunque cosa guardino. Predizione perfetta,
voto pieno, energia zero dappertutto, e rappresentazioni che non distinguono un
gatto da un lampadario. È il **collasso**, e per le JEPA è il pericolo numero
uno, perché qui (a differenza dei modelli generativi, ancorati ai pixel veri)
anche il *bersaglio* è prodotto da una rete che avrebbe tutto l'interesse a
barare. Nel documento del 2022 LeCun indica la famiglia di rimedi che
preferisce, quella già incontrata nel capitolo precedente: invece di
fabbricare risposte sbagliate da bocciare, si toglie al modello la possibilità
stessa di dare a tutto lo stesso riassunto, per esempio obbligandolo a tenerli
diversi fra loro. Ma nei sistemi JEPA costruiti davvero da Meta la difesa
concreta è un'altra, più semplice e più sottile.

`````{tab} Elementare

Immagina un allievo e un insegnante. L'allievo guarda la parte visibile della
foto e prova a *descrivere* che cosa c'è nella parte coperta; l'insegnante,
che vede la foto intera, scrive la descrizione giusta; il voto misura quanto
le due descrizioni combaciano. Se allievo e insegnante potessero mettersi
d'accordo, la truffa sarebbe immediata: rispondere entrambi, sempre, «boh»
(descrizioni identiche, voti perfetti, e nessuno dei due che abbia mai
guardato la foto). Il trucco che rompe la truffa è togliere all'insegnante
ogni voce in capitolo: le lamentele sul voto non lo raggiungono mai. In gergo
si dice che non riceve *gradiente*, cioè quella spinta a correggersi che dopo
ogni voto torna indietro nella rete e le ritocca i numeri. Non potendo
contrattare, l'insegnante non può accordarsi con l'allievo per abbassare
l'asticella, e all'allievo non resta che inseguire le descrizioni dell'altro.

C'è poi una seconda accortezza: come insegnante si usa una **copia lenta
dell'allievo**, non una seconda rete addestrata a parte ma l'allievo stesso
com'era in media negli ultimi tempi, cioè i suoi numeri mescolati un pochino a
ogni passo. Il dosaggio lo scelgono i ricercatori, e nel sistema vero è quattro
parti su mille: se un numero dell'allievo passa da 10 a 20, quello
dell'insegnante non salta a 20, diventa 10,04, cioè copre quattro millesimi
dei dieci di divario. Per raggiungerlo davvero gli servono centinaia di passi,
e nel frattempo il bersaglio cambia idea solo al ritmo a cui l'allievo migliora
*davvero*. Delle due accortezze, quella che impedisce la truffa è la prima; la
lentezza serve a rendere l'esercizio stabile, e alla fine della sezione lo
misuriamo. È una soluzione empirica (perché funzioni così bene è ancora
oggetto di studio) ma funziona.

`````

`````{tab} Superiore

Il rimedio usato dai sistemi JEPA di Meta è **architetturale**: asimmetria
tra i due encoder. L'encoder del target non viene addestrato per
retropropagazione ma mantenuto come **media mobile esponenziale** (EMA,
*exponential moving average*) dei pesi dell'encoder di contesto:

$$
\bar{\phi} \;\leftarrow\; m\, \bar{\phi} + (1 - m)\, \phi,
$$

dove $\phi$ sono i pesi dell'encoder di contesto, $\bar{\phi}$ quelli
dell'encoder target e $m$ un momento vicino a 1 (in BYOL e in parte della
letteratura questo coefficiente si indica con $\tau$, un simbolo che in questo
capitolo è già occupato dalla temperatura del sogno; MoCo, come qui, usa $m$).
In I-JEPA $m$ parte da 0,996, quindi a ogni passo il target si sposta di una
frazione millesimale verso l'encoder corrente, e **cresce linearmente fino a 1**
lungo l'addestramento: verso la fine il bersaglio smette del tutto di muoversi.
All'EMA si accompagna lo **stop-gradient**: la loss non si propaga
mai attraverso il ramo del target, che è puro riferimento. Dei due, il muro
contro il collasso è lo **stop-gradient**: è quello che impedisce la discesa
coordinata dei due encoder verso la costante, perché il bersaglio insegue e non
può contrattare. L'EMA aggiunge lentezza e stabilità al bersaglio, ed è
un'ottima cosa nei sistemi veri, ma non è lei a reggere il muro: nel giocattolo
in fondo a questa sezione, toglierla e tenere il solo stop-gradient non produce
alcun collasso (la varietà delle rappresentazioni, anzi, sale da 1,0 a 1,6). È
comunque la stessa scoperta empirica che aveva sorpreso la comunità con BYOL
nel 2020 {cite}`grill2020bootstrap`: niente coppie negative, niente termini
contrastivi, eppure niente collasso. Una comprensione teorica
completa del *perché* manca ancora, ed è giusto dirlo; il documento del 2022
{cite}`lecun2022path` discute anche l'alternativa esplicitamente regolarizzata
(varianza mantenuta sopra una soglia, covarianze fuori diagonale penalizzate,
alla VICReg), ma I-JEPA e V-JEPA, nei paper, si affidano all'asimmetria EMA.

`````

## I-JEPA: la scommessa alla prova delle immagini

Nel documento del 2022 la JEPA è soprattutto un diagramma. La prima
incarnazione convincente arriva l'anno dopo, dal gruppo di LeCun a Meta AI:
**I-JEPA** (*Image-based JEPA*, la JEPA per le immagini)
{cite}`assran2023self`, presentata alla conferenza CVPR. I pezzi sono quelli di
poco fa. L'encoder è un **Vision Transformer**, la rete che nel capitolo sui
Transformer tagliava l'immagine in tessere e le trattava come le parole di una
frase {cite}`dosovitskiy2021image`. Il compito è un indovinello: dato un solo
blocco di *contesto* dell'immagine, prevedere che cosa c'è in quattro blocchi
*bersaglio* nascosti. La novità è tutta nel **che cosa** si prevede: non i
puntini dei blocchi mancanti, ma i loro riassunti, calcolati dalla copia lenta
di poco fa. Quella copia, da qui in avanti, la chiameremo anche con la sua
sigla, **EMA** (*exponential moving average*, media mobile esponenziale): è il
nome tecnico di quel mescolare, a ogni passo, un pochino dei numeri
dell'allievo in quelli dell'insegnante.

`````{tab} Elementare

È il gioco della cartolina strappata. Ti mostro una cartolina a cui mancano
quattro rettangoli e ti chiedo: che cosa c'era lì? Non ti chiedo di
*ridisegnare* i pezzi mancanti: quello sarebbe il compito generativo, e ti
costringerebbe a inventare dettagli che non puoi sapere. Ti chiedo di
*descriverli*: «lì continua il muso del cane, girato verso destra». A
correggerti è la copia lenta di te stesso, che ha visto la cartolina intera e
ha scritto le sue descrizioni. Due dettagli fanno la differenza. Primo: i
rettangoli nascosti sono *grandi*, per indovinare un pezzo grande devi aver
capito la scena («è un cane, quindi là sotto c'è una zampa»), mentre per un
buchino basta allungare i bordi, senza capire niente. Secondo: al modello non
servono i trucchi artigianali con cui di solito si addestrano questi sistemi
(versioni ritagliate, specchiate, ricolorate della stessa foto, scelte a mano
da chi progetta). Basta l'indovinello. E i risultati danno ragione alla
scommessa: con appena l'1% delle etichette di ImageNet (una dozzina di foto
etichettate per categoria) I-JEPA classifica meglio dei metodi che
ricostruiscono i pixel, e ci arriva con molto meno calcolo. Su quel risparmio
conviene essere precisi, perché è facile capirlo al contrario: non è che ogni
ripasso costi meno (costa anzi un pelo di più, c'è una rete in più da far
girare), è che di ripassi ne servono cinque volte meno.

`````

`````{tab} Superiore

L'encoder di contesto (un ViT) elabora solo le patch visibili del blocco di
contesto; un **predictor** (un ViT più stretto) riceve $\mathbf{s}_x$ e, per ciascuno
dei $M = 4$ blocchi bersaglio, token posizionali che indicano *dove*
prevedere; la loss è la distanza $L_2$ media tra le rappresentazioni predette
e quelle prodotte dall'encoder target:

$$
\mathcal{L} = \frac{1}{M} \sum_{i=1}^{M} \sum_{j \in B_i}
\big\lVert\, \hat{\mathbf{s}}_{y,j} - \mathbf{s}_{y,j} \,\big\rVert_2^2,
$$

dove $B_i$ è l'insieme delle patch del blocco bersaglio $i$, e
$\hat{\mathbf{s}}_{y,j}$ e $\mathbf{s}_{y,j}$ sono le rappresentazioni predette e bersaglio
della singola patch $j$: il confronto avviene patch per patch, non fra due
riassunti di blocco. Un dettaglio architetturale è
decisivo: l'encoder target elabora l'immagine **intera**, e i bersagli si
ottengono mascherando la sua *uscita*, non il suo ingresso; così ogni
rappresentazione-bersaglio incorpora il contesto globale ed è semanticamente
ricca. Niente augmentation artigianali: nessun crop multiplo, nessun jitter di
colore. I numeri del paper {cite}`assran2023self`: su ImageNet-1K con l'**1%
delle etichette**, un ViT-H/14 pre-addestrato con I-JEPA raggiunge il 73,3% di
accuratezza top-1 (77,3% per il ViT-H/16 a risoluzione 448), contro il 71,5%
di MAE (il metodo generativo che ricostruisce i pixel mascherati) e il 69,7%
di iBOT (lì con un ViT-B/16, che è un modello molto più piccolo), e il
pre-addestramento del ViT-H/14 richiede meno di 1200 ore-GPU (meno di 72 ore su
16 A100), oltre dieci volte meno di MAE a parità di architettura.

Il risparmio, però, non viene da dove sembra, e sbagliarne l'attribuzione
significherebbe insegnare al lettore un meccanismo che non c'è. Calcolare i
bersagli nello spazio delle rappresentazioni **aggiunge** costo, perché c'è un
secondo encoder da mandare avanti a ogni passo: il paper misura circa il **7%
in più per iterazione**. Quel che risparmia è il *numero* di iterazioni, di
circa cinque volte (300 epoche di pre-addestramento contro le 1600 di MAE). Il
fattore dieci nasce dal rapporto fra ricette complete, non da un costo unitario
più basso; e la tesi giusta, che è anche la più interessante, suona così:
predire rappresentazioni non rende più economico ogni passo, rende necessari
molti meno passi.

`````

## Dal fotogramma al film: V-JEPA

Le immagini erano il primo collaudo; il progetto di LeCun, però, parla di
*futuro*, e il futuro vive nei video. **V-JEPA** {cite}`bardes2024revisiting`
(2024) trasporta lo schema dalla dimensione spaziale a quella
spazio-temporale: si copre una regione del video (in gergo si dice
**mascherare**) e si prevedono le sue rappresentazioni a partire dal resto.
C'è una finezza che rivela quanto i video siano una bestia diversa: i
fotogrammi vicini sono quasi identici, quindi se la maschera coprisse zone
diverse in fotogrammi diversi il modello potrebbe barare copiando dal
fotogramma accanto. La maschera è perciò un **tubo**: la stessa regione
spaziale, tenuta ferma lungo *tutta* la durata della clip. Ed è una maschera
generosa, perché in media copre circa il 90% del video. Addestrato così su due
milioni di video pubblici, senza etichette, senza testo e senza ricostruzione, V-JEPA
produce rappresentazioni che a quel punto bisogna misurare, e il modo in cui le
si misura conta quanto il risultato.

`````{tab} Elementare

Come si controlla che cosa ha imparato un modello a cui nessuno ha insegnato
niente? Si fa così. Prima si **congela** la rete, cioè si smette di
addestrarla e la si blocca com'è, perché altrimenti non si saprebbe più che
cosa sapeva *prima* dell'esame. Poi le si mette sopra un esaminatore,
addestrato a parte, che riceve soltanto i riassunti prodotti dalla rete
congelata e deve rispondere a una domanda utile: «che cosa sta facendo la
persona in questo video, nuota, suona, versa da bere?». Se l'esaminatore ci
riesce, vuol dire che l'informazione, nei riassunti, c'era.

Le collezioni di video su cui si dà l'esame hanno un nome e sono sempre le
stesse, così che i risultati di gruppi diversi si possano confrontare: una
collezione del genere si chiama **banco di prova** (in inglese *benchmark*).
Qui i banchi sono due. Il primo chiede di riconoscere che cosa succede nella
scena: chi nuota, chi suona, chi taglia le verdure. Il secondo, ed è quello che
conta di più, è costruito apposta per misurare la comprensione del *movimento*
e non dell'aspetto: lì non basta riconoscere gli oggetti, bisogna distinguere
«spingere qualcosa da sinistra a destra» da «spingere qualcosa da destra a
sinistra», che sono la stessa scena al contrario. V-JEPA se la cava bene su
tutti e due: otto risposte giuste su dieci sul primo, sette su dieci sul
secondo.

Un'avvertenza onesta, però, e vale per tutti gli esami fatti così: più
l'esaminatore è bravo, meno si capisce di chi sia il merito. Se è un
programmino, quel che risponde lo ha trovato bell'e pronto nei riassunti; se è
a sua volta una rete capace, una parte del lavoro potrebbe averla fatta lui.
E qui l'esaminatore un programmino non è: è una piccola rete addestrata
apposta, che nella versione successiva del sistema cresce ancora. Quindi quel
«sette su dieci» dice quanto l'informazione sul movimento sia **facile da
tirare fuori** dai riassunti, che non è la stessa cosa che dire che il modello
«ha capito».

`````

`````{tab} Superiore

Con la rete congelata e una sonda addestrata a parte, V-JEPA raggiunge l'81,9%
su Kinetics-400 (riconoscere l'azione: chi nuota, chi suona) e il 72,2% su
Something-Something-v2, un banco di prova che richiede di capire il
*movimento* («spingere qualcosa da sinistra a destra»), non solo l'aspetto.

Su che cosa sia quella sonda conviene essere precisi, perché la formula
corrente («una piccola testa di classificazione») sottostima parecchio. Il
protocollo di V-JEPA usa un *attentive probe*: uno strato di cross-attention
con un token di query appreso, la cui uscita rientra nel token di query per
connessione residua e finisce in un MLP a due strati. Uno strato di
cross-attention non è un classificatore lineare, è un aggregatore *addestrato*
che decide quali token guardare: fra i due estremi «regressione logistica sopra
feature congelate» e «fine-tuning completo» sta molto più vicino al secondo di
quanto la parola «testa» lasci intendere. E in V-JEPA 2 la sonda cresce
ancora: **quattro blocchi transformer**, l'ultimo dei quali sostituisce la
self-attention con una cross-attention a query appresa. Quattro blocchi
transformer sopra un backbone congelato non sono una testa, sono un modello.

Da qui la cautela sulla lettura, ed è la stessa che nell'ultima sezione
applicheremo al probing di Othello-GPT: il protocollo misura quanto le
rappresentazioni congelate rendano **estraibile** l'informazione sul
movimento, non quanto il modello la «capisca», e fra le due ipotesi
(l'informazione c'era nel backbone, oppure a costruirla è stata la sonda) non
distingue. Più la sonda è capace, meno il merito è attribuibile al solo
backbone; il confronto fra metodi resta valido finché la sonda è la stessa per
tutti, ed è per questo che il protocollo va dichiarato insieme al numero.

`````

## V-JEPA 2: il world model tocca il mondo

Nel giugno 2025 arriva il passo successivo, ed è quello che riporta tutta
questa storia al punto di partenza del capitolo: usare il modello per *agire*.
**V-JEPA 2** {cite}`assran2025vjepa` ingrandisce la ricetta, con un modello da
oltre un miliardo di parametri addestrato su più di un milione di ore di video
presi da internet, e i punteggi salgono di conseguenza. Sul banco di prova del
movimento, quello dello «spingere da sinistra a destra», le risposte giuste
passano da sette a quasi otto su dieci (77,3%, sul banco che porta il nome
buffo di Something-Something-v2). Poi
c'è un esame più difficile, l'anticipazione: guardando una cucina ripresa in
soggettiva, indovinare che cosa farà la persona nel secondo che viene. Lì il
modello può proporre cinque risposte e il punteggio conta quante volte quella
giusta è fra le cinque (in gergo *recall@5*): si passa da 27,6 a 39,7 su cento,
cioè da quasi tre volte su dieci a quattro. È un progresso grosso su un
compito che resta largamente irrisolto, il che è già un buon motivo per
diffidare di chi riassume queste cose con «ci riesce».

Ma la parte concettualmente nuova è **V-JEPA 2-AC** (*action-conditioned*,
condizionato sulle azioni), ed è la parte in cui il capitolo arriva finalmente
a un robot vero. Il meccanismo è quello dell'inizio, montato sopra un braccio
meccanico: si dà al robot un'**immagine-obiettivo** (la tazza sopra il piatto),
il modello immagina l'effetto di centinaia di comandi possibili e sceglie
quello il cui esito previsto è più vicino all'obiettivo. Poi lo esegue, guarda
com'è andata e ricomincia da capo, un comando alla volta. Vale la pena notare
la distanza dal progetto del 2022, dove l'agente immaginava intere sequenze di
azioni prima di muoversi: il robot vero, per ora, ne immagina una sola per
volta, e già così ci mette sedici secondi. Immaginare prima, muovere poi: è il
cinema interiore dell'apertura del capitolo, e questa volta muove qualcosa di
fisico.

`````{tab} Elementare

E come impara, questo robot? Non gli si mostra come si fa. Gli si fanno
guardare delle registrazioni di bracci robotici al lavoro: una sessantina
d'ore, prese da una raccolta pubblica che chiunque può scaricare (una raccolta
di dati fatta apposta per addestrare si chiama **dataset**). Le registrazioni
dicono anche come si è mosso il braccio istante per istante, perché è
un'informazione che la macchina scrive da sé mentre lavora. Quel che nessuno ha
annotato è tutto il resto: che compito si stesse svolgendo, se sia riuscito, se
chi guidava fosse bravo. Quei video vengono descritti come «non etichettati», e
vuol dire esattamente questo: manca il giudizio su che cosa si stesse facendo e
su come è andata, non l'informazione sui movimenti.

Poi lo si mette in due laboratori che non aveva mai visto, davanti a oggetti che
non aveva mai visto, senza un solo minuto di pratica lì dentro. Si chiama
**zero-shot**, «a colpo zero»: nemmeno un tentativo di prova.

Qui però serve la cifra, non l'aggettivo, perché «riesce» dice troppo.
Raggiungere un punto gli riesce sempre. Posare un oggetto dove va, circa tre
volte su quattro. Afferrare una tazza, due volte su tre. Afferrare una
scatola, una volta su quattro. E per ogni singolo gesto il robot passa
**sedici secondi** a immaginare le alternative prima di muovere un dito. È un
inizio notevole; non è un maggiordomo.

`````

`````{tab} Superiore

Sopra l'encoder congelato viene addestrato un predictor **condizionato sulle
azioni**, usando meno di 62 ore di video del dataset pubblico DROID. Sulla
parola «non etichettati», che il paper usa e che è facilissimo fraintendere,
conviene essere precisi, perché le azioni ci sono eccome e sono l'ingrediente
su cui poggia tutta la variante AC: il predictor riceve mappe di feature, stato
dell'end-effector (posizione, tre angoli di Eulero, apertura della pinza) e
azioni, interlacciati nel tempo, con l'azione definita come la variazione dello
stato dell'end-effector fra fotogrammi adiacenti. *Unlabeled*, nel paper, vuol
dire un'altra cosa, dichiarata a chiare lettere: nessun meta-dato su
ricompensa, su quale compito fosse in corso, o su se il tentativo sia riuscito.
È una distinzione che tornerà utile nella sezione seguente, perché Genie le
azioni davvero non ce le ha e deve inferirsele.

La pianificazione è controllo predittivo a orizzonte recedente: si ottimizza
una sequenza di azioni su un orizzonte $T$, se ne esegue soltanto la prima, si
osserva il nuovo stato e si ripianifica. A ottimizzare è il *Cross-Entropy
Method*: si campionano 800 candidate da gaussiane, si tengono le dieci
migliori, se ne ricalcolano media e varianza e si ripete per dieci giri. Nei
compiti riportati l'orizzonte è $T = 1$, cioè si ottimizza **una sola azione
per volta**: gli autori lo dichiarano sufficiente perché i compiti considerati
sono ingordi, e osservano che orizzonti più lunghi funzionano anch'essi ma
costano di più. Il costo è **16 secondi di calcolo su GPU per ogni singola
azione**. E i tassi di successo, medi sui due laboratori, dicono a
che punto siamo davvero: *reach* 100%, pick-and-place della tazza 80% e della
scatola 65%, presa della tazza 65%, presa della scatola **25%**. Afferrare una
scatola riesce una volta su quattro. Il sistema regge anche compiti di *video
question answering*, una volta allineato con un modello di linguaggio. È il
punto esatto in cui la via di LeCun smette di essere un diagramma e tocca,
letteralmente, il mondo fisico; non è il punto in cui la partita è vinta.

`````

## Tre famiglie per imparare senza etichette

Vale la pena fermarsi e mettere ordine, perché in questo libro abbiamo ormai
incontrato tutti e tre i grandi modi di imparare senza annotatori umani.

`````{tab} Elementare

Tre studenti, stessi libri, nessun professore. Il primo studia **ricopiando
con i buchi**: cancella pezzi del testo e si allena a riscriverli identici,
parola per parola o pixel per pixel; è il metodo *generativo*, quello di BERT
con le frasi (gli «esercizi a buchi» del capitolo sui Transformer) e di **MAE**
con le foto (*Masked Autoencoder*, «autoencoder mascherato»: gli si copre un
pezzo di immagine e deve ridisegnarlo). Il secondo studia col **gioco delle
coppie**: mescola foto e didascalie e impara a dire quali vanno insieme e quali
no; è il metodo *contrastivo*, quello di **CLIP** (*Contrastive Language–Image
Pre-training*, addestramento per contrasto di lingua e immagini), che avvicina
ogni immagine alla sua descrizione
e la allontana dalle altre. Il terzo (la via JEPA) studia **prevedendo il
riassunto**: copre un pezzo e, invece di ricopiarlo, ne prevede la
*descrizione*, confrontandola con quella di una copia lenta di sé. Non è una
classifica: il primo metodo ha vinto nel linguaggio, il secondo ha unito
immagini e parole, il terzo scommette sul futuro e sul video. Sono tre
risposte diverse alla stessa domanda: che cosa, di ciò che manca, vale la pena
prevedere?

`````

`````{tab} Superiore

**Generativa**: si ricostruisce l'input nello spazio dell'input. Il masked
language modeling di BERT {cite}`devlin2019bert` predice i token mascherati
con una cross-entropia sul vocabolario; MAE fa lo stesso con i pixel delle
patch mascherate, con loss $L_2$. Funziona magnificamente sul testo (dove i
token sono discreti e la softmax rappresenta senza sforzo l'incertezza) e
resta più goffa su segnali continui ad alta dimensione, dove l'equivalente
della softmax non esiste e ricostruire costringe a modellare l'irrilevante: è
l'argomento centrale di {cite}`lecun2022path`. **Contrastiva**: si impara una
geometria, avvicinando le coppie compatibili e allontanando quelle
incompatibili (CLIP {cite}`radford2021learning` con la loss InfoNCE su coppie
immagine–didascalia). Nel lessico del capitolo precedente: energia abbassata
sulle coppie giuste e *alzata esplicitamente* sui controesempi, con la nota
difficoltà di trovarne mai abbastanza in alta dimensione. **Predittiva nello
spazio latente**: la famiglia JEPA; energia = errore di predizione tra
embedding, nessuna ricostruzione, nessuna coppia negativa, collasso evitato
per asimmetria architetturale (lo stop-gradient, con l'EMA a stabilizzare) o
per regolarizzazione esplicita (varianza/covarianza), che è poi lo stesso
mestiere che la famiglia contrastiva svolge per un'altra via, alzando
l'energia sui controesempi. È la più giovane delle tre, e quella su cui pesa la
scommessa più grossa.

`````

## Una scommessa aperta

Chiudiamo con l'onestà dovuta. Quella raccontata in questa sezione è una
**linea di ricerca in corso**, non un traguardo raggiunto. Le rappresentazioni
JEPA sono eccellenti e costano poco, e V-JEPA 2-AC ha mostrato che un world
model auto-supervisionato può guidare un robot vero; ma dell'architettura a
sei moduli del 2022 la maggior parte resta sulla carta: la JEPA **gerarchica**,
cioè fatta a livelli, dove quello alto pianifica a grandi passi («esco di casa,
vado alla stazione») e quelli sotto ne riempiono i dettagli, ciascuno sulla
propria scala di tempo; il configuratore; il ragionamento a lungo **orizzonte**,
cioè su catene lunghe di conseguenze.

I critici, dal canto loro, fanno notare che la storia recente non è stata
tenera con le previsioni di insufficienza: i modelli generativi, cresciuti
abbastanza in taglia e in dati, continuano a esibire capacità che «non
avrebbero dovuto» avere. L'argomento più forte di quella sponda non è
un'impressione, ed è nella sezione seguente: un modello addestrato soltanto a
indovinare la mossa successiva si costruisce dentro una rappresentazione dello
stato del gioco, e la usa {cite}`li2023emergent`. Sull'idea che la coerenza
fisica dei generatori di video migliori da sé man mano che li si ingrandisce,
invece, conviene essere cauti quanto lo siamo con l'altra sponda: le fonti su
cui poggia sono, per i sistemi più spinti, gli annunci aziendali con
dimostrazioni scelte di cui parla la prossima sezione. È un'affermazione da
verificare, non da concedere. Se per capire
il mondo serva davvero smettere di generarlo, o se generare *sia* un modo di
capire, è esattamente la domanda su cui il campo è spaccato. LeCun, come
ricordato in apertura di capitolo, ci ha scommesso la carriera: ha lasciato
Meta per una startup dedicata ai world model. La prossima sezione attraversa il
fronte opposto del dibattito: i simulatori generativi di video, da Sora a
Genie, e la domanda se un modello che *disegna* futuri plausibili abbia capito
la fisica o abbia solo imparato a imitarla.

## Una mini-JEPA in PyTorch

Tutti i pezzi della sezione (encoder, copia lenta EMA, predictor, loss tra
embedding) stanno comodamente in una pagina di PyTorch. L'esperimento è
volutamente in miniatura: ogni «immagine» è una scena finta fatta di 8
**patch**, cioè di 8 tessere (è il modo in cui i Vision Transformer tagliano
un'immagine; qui le tessere nascono da un contenuto comune più rumore). Il
modello vede 6 tessere di contesto e deve prevedere l'**embedding** (non i
valori!) delle 2 tessere coperte. Il commento chiave è sull'**asimmetria**: il
bersaglio non riceve gradiente, e per questo non può mettersi d'accordo con
l'encoder. Quel «non riceve gradiente» ha un nome, **stop-gradient**, ed è la
traduzione in codice dell'insegnante che non può lamentarsi del voto: è lui a
tenere il sistema lontano dal collasso. L'EMA rende il bersaglio più lento e
più stabile, cosa che nei sistemi veri conta parecchio, ma non è lei a reggere
il muro, e qui sotto lo si misura.

```python
import copy
import torch
from torch import nn

torch.manual_seed(0)

DIM_PATCH, DIM_EMB = 16, 32
N_PATCH, N_CONTESTO = 8, 6          # per scena: 6 patch visibili, 2 mascherate

# Mappa fissa dal "contenuto" della scena all'aspetto delle patch
PROIEZIONE = torch.randn(4, DIM_PATCH)

def genera_batch(n=256):
    """Ogni scena nasce da un contenuto nascosto comune alle sue 8 patch."""
    contenuto = torch.randn(n, 1, 4)               # il "succo" della scena
    patch = contenuto @ PROIEZIONE                 # come il succo appare
    return patch + 0.25 * torch.randn(n, N_PATCH, DIM_PATCH)  # dettagli casuali

# Encoder (l'allievo), predictor, ed encoder target (la copia lenta)
encoder = nn.Sequential(
    nn.Linear(DIM_PATCH, 64), nn.ReLU(), nn.Linear(64, DIM_EMB))
predictor = nn.Sequential(
    nn.Linear(DIM_EMB, 64), nn.ReLU(), nn.Linear(64, DIM_EMB))

encoder_target = copy.deepcopy(encoder)
for p in encoder_target.parameters():
    p.requires_grad_(False)          # stop-gradient: il bersaglio non si allena

@torch.no_grad()
def aggiorna_target(m=0.996):
    """EMA: il target insegue lentamente l'encoder, e non ne riceve mai
    il gradiente. L'anti-collasso è proprio quel 'mai': senza gradiente
    il target non può accordarsi con l'encoder per appiattire tutti gli
    embedding sulla stessa costante. L'EMA aggiunge la lentezza."""
    for p, p_t in zip(encoder.parameters(), encoder_target.parameters()):
        p_t.mul_(m).add_((1.0 - m) * p)

opt = torch.optim.Adam(
    list(encoder.parameters()) + list(predictor.parameters()), lr=1e-3)

for passo in range(1, 601):
    patch = genera_batch()                          # (256, 8, 16)
    # contesto -> embedding riassuntivo (media delle 6 patch visibili)
    s_x = encoder(patch[:, :N_CONTESTO]).mean(dim=1)        # (256, 32)
    # target -> embedding calcolato dalla copia lenta, senza gradiente
    with torch.no_grad():
        s_y = encoder_target(patch[:, N_CONTESTO:]).mean(dim=1)  # (256, 32)
    s_y_pred = predictor(s_x)                       # predizione tra embedding
    loss = nn.functional.mse_loss(s_y_pred, s_y)    # voto fra riassunti

    opt.zero_grad()
    loss.backward()
    opt.step()
    aggiorna_target()                               # un passetto di EMA

    if passo in (1, 100, 200, 400, 600):
        # se gli embedding collassassero, questa varietà scenderebbe verso 0
        varieta = s_y.std(dim=0).mean().item()
        print(f"passo {passo}: loss {loss.item():.4f}  "
              f"varietà degli embedding {varieta:.3f}")
```

Eseguendolo, la loss crolla in un centinaio di passi, da circa 0,23 a meno di
0,01. Nel frattempo la «varietà» degli embedding, cioè quanto i riassunti di
scene diverse restano diversi fra loro, non scende affatto verso zero, che è
quel che farebbe se le rappresentazioni si stessero appiattendo: nei numeri
stampati passa da circa 0,4 a 1,0, e cioè cresce. Il
modello impara a prevedere il *contenuto* delle tessere coperte (che è
condiviso con il contesto) e ignora il rumore (che non è prevedibile), senza
appiattire le rappresentazioni.

Il modo di convincersene, però, non è leggere quei numeri: è spegnere il
meccanismo e guardare che cosa succede. Sostituendo la riga del bersaglio con
`s_y = encoder(patch[:, N_CONTESTO:]).mean(dim=1)`, cioè togliendo in un colpo
solo la copia lenta e il `torch.no_grad()`, i due rami tornano a essere la
stessa rete e possono accordarsi: dopo 600 passi la loss scende a cinque
decimillesimi e la varietà crolla a 0,05, cioè i riassunti di scene diverse
sono diventati quasi lo stesso riassunto. Quello è il collasso, ed è l'energia
zero ovunque di cui parlava la sezione. Se invece si toglie la sola
EMA, calcolando il bersaglio dall'encoder vivo ma sempre dentro
`torch.no_grad()`, non succede niente di male: la varietà arriva a 1,64,
più alta che nel codice qui sopra. Non è un miglioramento da inseguire (in un
giocattolo del genere una varietà più alta non vuol dire rappresentazioni
migliori): è la prova che senza l'EMA il collasso non arriva lo stesso, e che
quindi il muro lo regge l'altro ingrediente.

Ciò che qui manca (il ViT al posto del piccolo MLP, i token posizionali che
dicono al predictor *dove* prevedere, milioni di immagini e di ore di video) è
ingegneria; la logica è tutta in queste righe.

`````{tab} Elementare

```{admonition} Da ricordare
:class: important
- Nel 2022 LeCun mette online un documento di 62 pagine
  {cite}`lecun2022path` che non contiene nessun risultato: è un progetto,
  il disegno di come dovrebbe essere fatta secondo lui una macchina che
  capisce il mondo. Sei pezzi, e al centro un modello del mondo che impara
  guardando, senza che nessuno gli spieghi niente.
- **Prevedere l'immagine esatta è la strada sbagliata**, ed è la storia del
  bicchiere in bilico: il futuro può andare in mille modi e nessuna decisione
  dipende dalla forma della terza scheggia, quindi un modello obbligato a
  disegnare *una* foto finisce per disegnare la media sfocata di tutte. La
  proposta è prevedere il succo, non la foto.
- Il pericolo di prevedere il succo è che allievo e insegnante si accordino
  per rispondere sempre «boh»: si chiama **collasso**. Il rimedio è che
  l'insegnante sia una copia lenta dell'allievo e non riceva mai lamentele sul
  voto: non potendo contrattare, non può accordarsi al ribasso.
- La prova sulle immagini è il gioco della cartolina strappata: si coprono
  quattro rettangoli **grandi** e si chiede di *descriverli*, non di
  ridisegnarli. Funziona, e impara con molto meno calcolo dei metodi che
  ridisegnano; ma non perché ogni passo costi meno, perché ne servono molti
  meno.
- Sui video la copertura diventa un **tubo**, ferma nello stesso punto per
  tutta la clip, così che il modello non possa copiare dal fotogramma accanto.
  E l'ultima versione arriva a guidare un braccio robotico in due laboratori
  mai visti, con un'immagine dell'obiettivo al posto delle istruzioni:
  immagina prima, muove poi. Con i piedi per terra, però: afferrare una tazza
  gli riesce due volte su tre, una scatola una volta su quattro, e ogni gesto
  gli costa sedici secondi di calcolo.
- Tre modi di studiare senza professore, e sono tre studenti diversi:
  **ricopiare con i buchi**, il **gioco delle coppie**, **prevedere il
  riassunto**. Il terzo è quello di cui parla questa sezione, ed è il più
  giovane dei tre. La prossima sezione va a sentire l'altra campana.
```

`````

`````{tab} Superiore

```{admonition} Da ricordare
:class: important
- Nel 2022 LeCun pubblica su OpenReview *A Path Towards Autonomous Machine
  Intelligence* {cite}`lecun2022path`: non un paper di risultati ma un
  progetto di architettura; sei moduli (percezione, world model, costo,
  attore, memoria a breve termine, configuratore) attorno a un world model
  appreso per auto-supervisione.
- **Predire nei pixel è la strada sbagliata**: il futuro è molteplice e
  pieno di dettagli irrilevanti; la minimizzazione dell'errore quadratico
  produce la media sfocata dei futuri. La **JEPA** predice nello spazio
  delle rappresentazioni: è un'architettura a energia non normalizzata, dove
  l'energia è l'errore di predizione tra embedding.
- Il pericolo è il solito **collasso** (embedding costanti, energia bassa
  ovunque); la difesa dei sistemi reali è l'asimmetria fra i due rami, e il
  muro è lo **stop-gradient**: il ramo del target non riceve gradiente e non
  può colludere. L'EMA aggiunge lentezza e stabilità al bersaglio, non è lei
  a impedire il collasso.
- **I-JEPA** {cite}`assran2023self` (CVPR 2023): un ViT predice le
  rappresentazioni di quattro blocchi mascherati dal contesto; niente
  augmentation artigianali; con l'1% delle etichette di ImageNet batte i
  metodi a ricostruzione di pixel (73,3% contro 71,5% di MAE) con oltre
  dieci volte meno calcolo, che però viene dalle iterazioni cinque volte meno
  numerose e non dal costo per iterazione, che è il 7% più alto.
- **V-JEPA** {cite}`bardes2024revisiting` porta lo schema al video (maschere
  a tubo estese su tutta la clip); **V-JEPA 2** {cite}`assran2025vjepa`
  scala a oltre un milione di ore di video e, con meno di 62 ore di video
  DROID privi di annotazione su compito, ricompensa ed esito (ma **con** le
  azioni registrate), ottiene pianificazione robotica **zero-shot** su bracci
  Franka mai visti: successo fra il 25% e il 100% secondo il compito, con 16
  secondi di calcolo per azione.
- I numeri a rete congelata vanno letti insieme al protocollo: la sonda è un
  *attentive probe* (per V-JEPA 2, quattro blocchi transformer), quindi
  misurano quanto l'informazione sia **estraibile**, non quanto il modello
  «capisca».
- Tre famiglie di auto-supervisione: **generativa** (ricostruisci: BERT,
  MAE), **contrastiva** (avvicina/allontana: CLIP), **predittiva nello
  spazio latente** (JEPA). La partita tra generare e predire-nelle-idee è
  aperta: la prossima sezione visita l'altra sponda.
```

`````
