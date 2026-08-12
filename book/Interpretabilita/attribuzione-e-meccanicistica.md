# Dentro le reti profonde: attribuzione e interpretabilità meccanicistica

Torniamo un'ultima volta al rilevatore di neve dell'apertura di capitolo, per
una ragione precisa. Le spiegazioni che smascherarono quel classificatore erano
LIME, cioè il metodo agnostico della sezione precedente: trattavano il modello
come una scatola chiusa, gli davano l'immagine con alcune porzioni spente e
guardavano come cambiava la risposta. Doveva andare così, del resto, perché il
modello truccato non era una rete: era una semplice regressione logistica
appoggiata sopra le rappresentazioni di una rete già addestrata, che nessuno
aveva toccato. La scorciatoia («c'è neve → lupo») l'aveva imparata il pezzo
lineare a valle.

Ma quando il modello **è** la rete, e la rete ce l'abbiamo in mano con tutti i
suoi pesi, c'è una strada che la scatola chiusa non permette: guardarci dentro
invece di bussare da fuori.

Nella prima sezione del capitolo abbiamo visto modelli **interpretabili per
costruzione**: una regressione lineare ci consegna un coefficiente per ogni
variabile, e quel numero *è* la spiegazione (leggibile a occhio nudo, come nel
capitolo sul Machine Learning classico). Una rete profonda no. Ha milioni di
parametri intrecciati, e nessuno di essi, preso da solo, dice qualcosa di
sensato. Dobbiamo cambiare domanda. Non «quanto pesa questa variabile in
generale?», ma «quanto ha contribuito *questo* ingresso a *questa*
decisione?». La quota di merito che si assegna a ciascun pezzo dell'ingresso si
chiama **attribuzione**, ed è la parola che intitola questa sezione. La
risposta, sorprendentemente, viene da uno strumento che il libro ha già usato
per tutt'altro mestiere: il **gradiente**.

## Saliency maps: il gradiente come mappa di importanza

Conviene ricordare in due righe che cos'è, perché tutto quel che segue ci si
appoggia. Il gradiente di una quantità rispetto a una manopola risponde alla
domanda: «di quanto cambia la quantità, se giro la manopola di un nulla?» (la
parola tecnica per quel «di quanto cambia se la giro appena» è **derivata**).
Nell'addestramento le manopole erano i pesi della rete e la quantità era
l'errore, e girare le manopole nel verso che abbassa l'errore *è*
l'addestramento, come si è visto nel capitolo sulle reti neurali. Adesso
puntiamo lo stesso strumento altrove: teniamo ferme le manopole dei pesi e
chiamiamo manopola ogni **pixel di ingresso**, mentre la quantità da guardare
non è più l'errore, ma il punteggio che la rete assegna alla classe che ha
predetto. Il risultato è una *saliency map*,
proposta da Simonyan, Vedaldi e Zisserman nel 2014 {cite}`simonyan2014deep`.

`````{tab} Elementare

Immagina di avere una foto e di volerti chiedere: quali pixel, se li toccassi
appena, cambierebbero di più il verdetto della rete? Se sposti di un nulla il
pixel del muso e la fiducia in «cane» crolla, quel pixel è *importante*. Se
tocchi un pixel dello sfondo e non succede niente, quello non conta. La saliency
map è esattamente questa: una mappa in bianco e nero, delle stesse dimensioni
della foto, che si accende dove un piccolo ritocco farebbe la differenza più
grande.

È come cercare i punti fragili di un castello di carte: dai un colpetto qua e
là e guardi cosa fa tremare tutta la struttura. Il difetto è che questi
colpetti, misurati un pixel alla volta, sono rumorosi: la mappa risulta piena di
puntini sparsi, un brusio granuloso da cui la forma dell'oggetto si intravede
appena.

`````

`````{tab} Superiore

Sia $S_c(\mathbf{X})$ il punteggio (il logit, prima della softmax) che la rete
assegna alla classe $c$ per l'immagine $\mathbf{X}$. La saliency map è il modulo
del gradiente del punteggio rispetto all'ingresso:

$$
\mathbf{M} = \left| \frac{\partial S_c}{\partial \mathbf{X}} \right|,
$$

calcolato con una singola *backpropagation* fino allo strato di input anziché
fermarsi ai pesi. L'idea è una **linearizzazione locale**: nell'intorno di
$\mathbf{X}$,
$S_c(\mathbf{X} + \delta) \approx S_c(\mathbf{X}) +
\big(\partial S_c/\partial \mathbf{X}\big)^\top \delta$,
quindi le componenti del gradiente di modulo maggiore individuano i pixel la cui
piccola variazione altera di più il punteggio. Per un'immagine a colori si
prende in genere il massimo del modulo sui tre canali RGB.

Il limite è duplice. Primo, il gradiente è **locale**: coglie la pendenza solo
nel punto $\mathbf{X}$, e le reti profonde sono tutt'altro che lineari. Secondo, è
**rumoroso**, perché la superficie $S_c$ ha derivate che oscillano rapidamente.
Le mappe risultano granulose, e le tecniche successive nascono quasi tutte per
domare questo rumore.

`````

## Grad-CAM: dove guarda la rete

La saliency lavora sui pixel e paga in rumore. Un'alternativa più stabile
rinuncia alla risoluzione fine e chiede una cosa più grossolana ma più
robusta: in quale *regione* dell'immagine la rete ha trovato le prove della sua
decisione? È l'idea di **Grad-CAM** (*Gradient-weighted Class Activation
Mapping*), di Selvaraju e colleghi nel 2017 {cite}`selvaraju2017grad`.

`````{tab} Elementare

Come abbiamo visto nel capitolo sulla Visione Artificiale, una rete
convoluzionale, salendo di strato in strato, smette di ragionare per pixel e
comincia a ragionare per **motivi**: l'ultimo strato convoluzionale non vede più
puntini, ma zone che «assomigliano a un muso», «assomigliano a una ruota». Ognuna
di queste mappe di attivazione è come un faretto puntato su una parte
dell'immagine.

Grad-CAM chiede al gradiente quali faretti contano per la classe che ci
interessa, e poi li accende in proporzione. Se stiamo spiegando la risposta
«cane», i faretti sul muso e sulle orecchie pesano tanto, quelli sull'erba
pesano zero. Sovrapposti alla foto, danno una macchia calda (una *heatmap*)
che dice, letteralmente, *dove* la rete ha guardato per dire «cane». È
grossolana (la risoluzione è quella dell'ultimo strato, non dei pixel), ma è
pulita e onesta: nel caso dell'husky, la macchia calda finirebbe proprio sulla
neve, smascherando l'inganno.

`````

`````{tab} Superiore

Sia $A^k \in \mathbb{R}^{u \times v}$ la $k$-esima *feature map* dell'ultimo
strato convoluzionale e $y^c$ il punteggio della classe $c$. Grad-CAM procede in
due mosse. Prima calcola un peso per ogni mappa, mediando spazialmente il
gradiente della classe rispetto a quella mappa:

$$
\alpha_k^c = \frac{1}{Z} \sum_{i}\sum_{j}
   \frac{\partial y^c}{\partial A^k_{ij}},
$$

dove $Z = u\,v$ è il numero di posizioni (un *global average pooling* del
gradiente). Poi combina le mappe pesate e tiene solo il contributo positivo:

$$
L^c_{\text{Grad-CAM}} = \mathrm{ReLU}\!\left( \sum_k \alpha_k^c\, A^k \right).
$$

Il peso $\alpha_k^c$ misura quanto la mappa $k$ conta per la classe $c$; la
$\mathrm{ReLU}$ scarta le regioni che *abbassano* il punteggio, tenendo solo
quelle che lo sostengono. La heatmap $L^c$ ha la bassa risoluzione dello strato
convoluzionale ($7\times 7$ in una ResNet su input $224\times 224$) e va
sovracampionata alle dimensioni dell'immagine per la sovrapposizione. A
differenza della saliency, non risale ai pixel: guadagna in robustezza al rumore
ciò che perde in dettaglio spaziale, e localizza in modo affidabile l'oggetto
che ha guidato la decisione.

`````

## Integrated Gradients: gli assiomi e il cammino dalla baseline

Sia la saliency sia Grad-CAM misurano il gradiente in **un solo punto**, e qui
si nasconde un problema.

Le funzioni che una rete usa per decidere non salgono all'infinito: crescono
ripide finché la rete è incerta, e poi si appiattiscono, perché una fiducia non
può superare il suo massimo. La curva a esse che si comporta così, ripida in
mezzo e piatta alle due estremità, è la **sigmoide** vista nel capitolo sulle
reti neurali; e di un neurone arrivato sul tratto piatto si dice che è
**saturo**. Su un tratto piatto, però, il gradiente è quasi zero: toccare
l'ingresso non cambia più niente, perché la rete è già convinta. Il paradosso è
che il gradiente dichiara «questo pixel non conta» proprio quando quel pixel è
la ragione per cui la rete è così sicura.

Sundararajan, Taly e Yan,
nel 2017, hanno affrontato la questione partendo non da un'euristica ma da due
**assiomi**: proprietà che una buona spiegazione *deve* soddisfare
{cite}`sundararajan2017axiomatic`.

Il primo è la **sensibilità**, e va enunciato con la sua condizione, altrimenti
dice il falso: se un ingresso e il riferimento neutro da cui si parte
differiscono per **una sola** variabile, e su quei due il modello risponde in
modo diverso, allora quella variabile deve ricevere una quota di merito diversa
da zero. Il secondo è
l'**invarianza all'implementazione**: due reti che calcolano la stessa
funzione matematica, con architetture diverse, devono ricevere le stesse
attribuzioni (la spiegazione riguarda *cosa* la rete calcola, non *come* lo
scrive in codice). Il gradiente locale, da solo, viola la sensibilità proprio
nei casi di saturazione.

`````{tab} Elementare

Invece di misurare la pendenza solo nel punto di arrivo, immagina di partire
da un'immagine «neutra» (di solito tutta nera, la *baseline*) e di arrivare
piano piano all'immagine vera, mescolandole in tante tappe: 10% vera e 90%
nera, poi 20 e 80, e così via fino al 100%. A ogni tappa ti chiedi di quanto
cambierebbe la fiducia della rete se toccassi appena quel pixel, e alla fine
fai la media di tutte le risposte. Così, anche se all'arrivo la rete è satura e
non reagisce più, hai comunque registrato la sua reazione lungo tutta la
salita, quando reagiva eccome.

Questo metodo ha una proprietà bellissima da controllare: se sommi le
attribuzioni di tutti i pixel, ottieni esattamente *quanto* la rete è passata
dalla fiducia sull'immagine nera a quella sull'immagine vera. Niente si perde e
niente si inventa: il conto torna sempre. È come dividere il conto di una cena
tra i commensali in modo che la somma delle quote faccia, al centesimo, il
totale sullo scontrino.

Sull'analogia della cena, però, c'è una domanda scomoda da fare subito: il
conto torna, ma torna **a partire da dove?** Il punto di partenza, quell'immagine
nera, non è un fatto di natura: è una scelta, e la scelta cambia le risposte. La
più chiara delle conseguenze è questa: di ciò che era **già nero** in partenza
non si può misurare nessun contributo, perché fra partenza e arrivo non è
cambiato. Se la ragione della decisione fosse proprio una zona buia della foto
(un'ombra, il cielo notturno, il nero di una radiografia), quel metodo le
darebbe zero, e la somma tornerebbe lo stesso. Che il conto torni non dice che
si è partiti dal punto giusto.

`````

`````{tab} Superiore

Sia $\mathbf{x}'$ la **baseline**, il punto di riferimento «neutro» rispetto a cui si
misura il contributo (per un'immagine, tipicamente il nero,
$\mathbf{x}' = 0$), e $\mathbf{x}$ l'ingresso da spiegare. In forma precisa,
l'assioma di **sensibilità** chiede che se $\mathbf{x}$ e $\mathbf{x}'$
differiscono in una sola componente e
$f(\mathbf{x}) \neq f(\mathbf{x}')$,
quella componente riceva attribuzione non nulla: è proprio ciò che il gradiente
valutato nel solo punto $\mathbf{x}$ non garantisce, perché in regime di saturazione è
quasi zero anche quando quella componente è la ragione dell'uscita.

Gli *Integrated Gradients* integrano il gradiente lungo
il segmento rettilineo da $\mathbf{x}'$ a $\mathbf{x}$:

$$
\mathrm{IG}_i(\mathbf{x}) = (x_i - x'_i)\,
   \int_0^1 \frac{\partial
     f\big(\mathbf{x}' + \alpha\,(\mathbf{x} - \mathbf{x}')\big)}{\partial x_i}\,
   \mathrm{d}\alpha,
$$

dove $f$ è l'uscita della rete per la classe d'interesse, $\alpha \in [0,1]$
parametrizza il cammino e $\mathrm{IG}_i$ è l'attribuzione della $i$-esima
componente d'ingresso. In pratica l'integrale si approssima con una somma di
Riemann su $m$ passi. Integrando lungo il cammino, il metodo cattura anche i
gradienti *prima* della saturazione, dove il segnale è vivo, risolvendo la
cecità del gradiente locale.

La proprietà che ne fa uno strumento affidabile è la **completezza**: le
attribuzioni si sommano esattamente alla differenza di uscita tra input e
baseline,

$$
\sum_i \mathrm{IG}_i(\mathbf{x}) = f(\mathbf{x}) - f(\mathbf{x}').
$$

La completezza implica la sensibilità e conferisce alle attribuzioni un
significato preciso: ciascuna è la *quota* di quel salto di punteggio imputabile
a quella componente. È l'assioma che verificheremo numericamente più avanti.

Va detto però che cosa la completezza **non** garantisce, perché è il punto in
cui il metodo si presta a essere letto per più di quel che promette: essa vale
per **qualunque** baseline. Gli assiomi vincolano come si ripartisce il salto
$f(\mathbf{x}) - f(\mathbf{x}')$, non da dove il salto parte, e la scelta di
$\mathbf{x}'$ resta il grado di libertà principale del metodo. Due conseguenze
concrete. La prima è che il
fattore $(x_i - x'_i)$ davanti all'integrale **azzera per costruzione**
l'attribuzione di ogni componente che coincide con la baseline: con la baseline
nera ogni pixel nero riceve esattamente zero, per definizione e non per misura,
mentre in una radiografia o in una foto notturna il nero non è affatto assenza
di informazione. La seconda è che cambiando baseline le attribuzioni cambiano di
grandezza e perfino di **segno**, mentre la somma continua a tornare: sulla
funzione giocattolo dell'esempio in fondo alla sezione, spostando la baseline da
$(0,0)$ a $(2,0)$ l'attribuzione della componente con il peso maggiore passa da
$1{,}327$ a esattamente $0$, e con la baseline $(-1,2)$ la seconda componente
prende segno positivo invece che negativo; in tutti e tre i casi la completezza
è verificata al quarto decimale. Gli autori chiedono infatti che la baseline sia
scelta e **verificata**: deve rappresentare un'assenza di segnale, e su di essa
il punteggio della classe dev'essere quasi nullo. Le alternative d'uso comune
(rumore gaussiano, immagine sfocata, media del dataset, media su più baseline)
danno attribuzioni diverse, e nessun assioma le ordina
{cite}`sturmfels2020baselines`.

`````

Questo metodo ha una particolarità che vale la pena rendere esplicita: **non
si vede in un fotogramma**, perché il fotogramma è proprio il punto in cui il
gradiente non dice niente. In {numref}`fig-gradienti-integrati` c'è il cammino,
percorso a passi.

```{figure} ../figures/gradienti-integrati.svg
:name: fig-gradienti-integrati
:alt: "A sinistra la curva dell'uscita della rete lungo il segmento che va dalla baseline all'ingresso: parte ripida e si appiattisce. Un pallino la percorre a passi, e a ogni passo un segmento mostra la pendenza in quel punto, che all'inizio è grande e alla fine quasi nulla. A destra una barra accumula la somma delle pendenze e si ferma esattamente sulla riga che segna la differenza fra l'uscita sull'ingresso e quella sulla baseline."
:width: 92%

Il cammino da $\mathbf{x}'$ a $\mathbf{x}$, percorso a otto passi. A sinistra
la pendenza in ciascun punto: viva all'inizio, quasi nulla alla fine. A destra
la somma che si accumula, e la riga che segna dove deve arrivare.
```

Due cose si leggono in {numref}`fig-gradienti-integrati` e non nella formula.
La prima è **quanto** la saturazione morda: sull'esempio della figura la
pendenza vale $3{,}76$ a un ottavo del cammino e $0{,}009$ all'ultimo ottavo,
quattrocento volte meno. Un metodo che guardi solo il punto d'arrivo lavora su
quel $0{,}009$, ed è la ragione per cui restituisce quasi zero anche per una
componente che è tutta la spiegazione. La seconda è che la barra di destra si
ferma **esattamente** sulla riga, e quella non è una coincidenza né un
aggiustamento: è la completezza, e il fatto che valga anche con soli otto passi
è ciò che rende la somma di Riemann un'approssimazione onesta e non una
speranza.

## Le mappe dicono dove, non che cosa

Tre metodi, tre mappe sempre più pulite. Prima di andare avanti conviene
fermarsi e chiedersi una cosa che le mappe, per come sono fatte, non
suggeriscono da sole: **una mappa di importanza è una spiegazione?**

Ci sono due obiezioni, e sono di natura diversa. La prima riguarda cosa una
mappa può dire in linea di principio; la seconda, più grave, riguarda se stia
davvero parlando del modello.

`````{tab} Elementare

La prima obiezione la mette bene Cynthia Rudin: sapere **dove** la rete
guarda dentro l'immagine non dice **che cosa** stia facendo con quella parte.
Una mappa che si accende sul muso del cane è compatibile con «la rete
riconosce la forma di un muso», ma anche con «la rete ha imparato che in
quella zona c'è di solito una texture di pelo», o con «quel pezzo di immagine
è semplicemente ad alto contrasto». La salienza dice ciò che la rete *vede*,
non ciò che la rete *pensa*.

La seconda obiezione è più radicale ed è arrivata da un esperimento tanto
semplice quanto crudele. Prendi una rete addestrata, produci la sua mappa, poi
**cancella quello che ha imparato**: randomizza i pesi, strato per strato, e
rifai la mappa. Se la mappa fosse una spiegazione del modello, dovrebbe
disintegrarsi, perché il modello non c'è più. Per diversi metodi popolari, la
mappa cambia pochissimo, e resta riconoscibile come una sagoma dell'oggetto. I
due che abbiamo visto qui (i colpetti pixel per pixel e i faretti di Grad-CAM)
non sono fra i bocciati; a fallire il test sono due varianti più elaborate che
non abbiamo incontrato.

La conclusione è spiacevole e va detta: quelle mappe stavano in buona parte
descrivendo l'**immagine**, non la rete. Somigliavano a un rilevatore di
bordi, e siccome un rilevatore di bordi su una foto di cane accende il cane,
sembravano sensate.

`````

`````{tab} Superiore

L'obiezione di Rudin {cite}`rudin2019stop` è che una mappa di salienza è
compatibile con troppe spiegazioni diverse dello stesso comportamento: è
un'informazione sulla *posizione* dell'evidenza, non sul calcolo che la usa.
Nei contesti ad alto rischio, sostiene, questo la rende inadatta a sostituire
un modello intrinsecamente interpretabile.

La seconda obiezione è empirica e ha una forma metodologica importante: sono i
**controlli di sanità** di Adebayo e colleghi {cite}`adebayo2018sanity`. Il
ragionamento è che un metodo di attribuzione, per essere utile, deve almeno
essere **sensibile** alle cose da cui la predizione dipende. Da qui due test.

Nel *model parameter randomization test* si randomizzano progressivamente i
pesi del modello, dall'ultimo strato verso il primo, confrontando ogni volta la
mappa con quella originale. Nel *data randomization test* si riaddestra il
modello su etichette permutate a caso, cosicché abbia necessariamente
memorizzato rumore, e si confronta la mappa con quella del modello addestrato
sulle etichette vere.

Un metodo che superi i test deve cambiare drasticamente in entrambi i casi.
Diversi metodi molto usati non lo fanno, e le loro mappe restano visivamente
simili all'originale: si comportano, per usare l'espressione degli autori, come
un rilevatore di bordi indipendente dal modello. Vale la pena nominarli, perché
è l'informazione operativa: sono **Guided BackProp** e **Guided Grad-CAM**, che
restano riconoscibili anche a pesi randomizzati. Metodi come le saliency
semplici e Grad-CAM se la cavano meglio di altri, ma il punto metodologico
resta, ed è quello che vale la pena portarsi via: **la plausibilità visiva di
una spiegazione non è una prova della sua fedeltà**, e un occhio umano non
distingue le due cose. Un metodo di attribuzione va sottoposto a un test che
possa farlo fallire, esattamente come un modello.

`````

Il che non rende inutili le mappe: le ricolloca. Sono strumenti di
**esplorazione** (dove guardare, quali ipotesi formulare, quale scorciatoia
sospettare in un dataset) e non certificati di funzionamento. La sezione
seguente mostra che lo stesso identico dubbio, con la stessa struttura, si è
posto per l'oggetto che sembrava metterne al riparo: i pesi di attenzione.

## L'attenzione è una spiegazione?

C'è una tentazione naturale, per chi lavora con i Transformer del capitolo
dedicato: i pesi di **attenzione** {cite}`vaswani2017attention` sono già lì,
e sembrano dire su quali parole il modello si è
concentrato. Vale la pena richiamare in una riga di che si tratta: per
decidere che cosa fare di una parola, il modello distribuisce una specie di
sguardo sulle altre parole della frase, dando a ciascuna un peso; quei pesi
sono l'attenzione, e sommano sempre a uno, come le fette di una torta divisa
fra tutte le parole. Perché non usarli come spiegazione, gratis?

La comunità ci ha discusso a lungo. Nel 2019 Jain e Wallace
{cite}`jain2019attention`, con un articolo
dal titolo programmatico *«Attention is not Explanation»*, hanno mostrato che
spesso si possono costruire pesi di attenzione **molto diversi** che
portano alla **stessa** predizione: se più configurazioni dei pesi danno lo
stesso verdetto, nessuna di esse può essere *la* spiegazione. Altri (Wiegreffe
e Pinter, sempre nel 2019, con la replica *«Attention is not not
Explanation»* {cite}`wiegreffe2019attention`) hanno ribattuto che dipende da
cosa si pretende: sotto vincoli
più stretti l'attenzione conserva un valore esplicativo. La morale operativa è
di **cautela**: i pesi di attenzione sono un indizio suggestivo, non una
prova; una heatmap di attenzione va letta come una traccia, non come una
confessione.

Un'avvertenza sulla portata, però, va messa accanto al risultato, perché il
titolo è più largo dell'esperimento: Jain e Wallace studiano **un solo strato**
di attenzione sopra un encoder ricorrente (una BiLSTM), su compiti di
classificazione, domanda-risposta e inferenza testuale, non l'auto-attenzione a
più teste e più strati di un Transformer, che nel loro articolo non compare mai.
Nel lavoro stesso, anzi, il comportamento **cambia con l'architettura**: gli
encoder più semplici, a media pesata, si comportano meglio secondo gli stessi
criteri. Il risultato è quindi un monito metodologico solido, non un teorema sui
Transformer.

C'è però una domanda tecnica che precede quella filosofica, e che di solito
viene saltata: **l'attenzione di quale strato?** Un Transformer ne ha decine,
impilati, e guardarne uno solo è come giudicare una catena di montaggio da una
sola stazione.

`````{tab} Elementare

Il problema è che, a ogni piano della pila, una parte dell'informazione non
passa affatto dall'attenzione: prende una **scorciatoia** e scivola dritta al
piano di sopra (sono le connessioni residuali del capitolo sui Transformer).
Quindi i pesi di un singolo strato raccontano solo un pezzo del viaggio: per
sapere quanto ogni parola d'ingresso ha influenzato il risultato in cima
bisogna seguire l'intero percorso, scorciatoie comprese, piano dopo piano. Gli
strumenti che fanno questo conto si chiamano **attention rollout** e
**attention flow**, e restituiscono una mappa sulle parole di partenza, spesso
più sensata di quella del singolo strato.

C'è infine un attrezzo complementare, il **probing** (sondaggio): per scoprire
se a un certo piano della rete è scritta una data informazione (per esempio,
se una parola è un nome o un verbo), si prova a leggerla da lì con lo
strumento più semplice che c'è, un piccolo classificatore addestrato apposta.
Se ci riesce, l'informazione a quel piano c'è; se fallisce, non c'è, o non è
scritta in modo semplice. Con un'avvertenza: uno strumento di lettura troppo
bravo rischia di indovinare da sé ciò che doveva soltanto leggere.

`````

`````{tab} Superiore

Abnar e Zuidema {cite}`abnar2020quantifying` mostrano che la composizione fra
strati non è affatto banale, per una ragione che il capitolo sui Transformer
ha già messo in evidenza: le **connessioni residuali**. A ogni blocco il
valore di un token non viene sostituito da ciò che l'attenzione gli porta, ma
sommato ad esso; quindi una parte dell'informazione che arriva allo strato
$l+1$ non è passata dall'attenzione di quel livello, ma è scivolata lungo la
scorciatoia.

Il rimedio proposto è di tenerne conto e poi comporre. Si corregge la matrice
di attenzione di ogni strato mescolandola con l'identità, che rappresenta
appunto il passaggio diretto,

$$
A^{(l)} = \tfrac{1}{2} W^{(l)}_{\text{att}} + \tfrac{1}{2} I ,
$$

e si moltiplicano gli strati fra loro per ottenere quanto di ogni token di
ingresso è finito in ogni posizione all'altezza voluta. È l'**attention
rollout**. La variante *attention flow* tratta la stessa struttura come un
grafo orientato aciclico e calcola il flusso massimo dal token di ingresso a
quello di arrivo, che è più costoso e tiene conto dei colli di bottiglia lungo
il cammino. In entrambi i casi il risultato è una mappa sui **token
d'ingresso**, cioè finalmente confrontabile con le attribuzioni delle sezioni
precedenti, e visibilmente diversa (spesso più sensata) della matrice del
singolo strato che si è tentati di visualizzare.

Un approccio complementare, più controllato, è il **probing**. L'idea: se una
rappresentazione interna «sa» qualcosa (poniamo, la parte del discorso di una
parola), allora un classificatore *lineare* addestrato su quella
rappresentazione dovrebbe saperlo prevedere. Si congela la rete, si estraggono
le attivazioni di uno strato e ci si allena sopra una semplice regressione
logistica per una proprietà a scelta. Se il probe riesce, l'informazione è
presente e linearmente accessibile in quello strato; se fallisce, non lo è. È
un modo economico per mappare *dove*, nella pila di strati, emergono le varie
proprietà. Il probe lineare è la proposta di Alain e Bengio
{cite}`alain2017understanding`, che lo motivano proprio con la separabilità
lineare e con la convessità del problema di addestramento; l'avvertenza che ne
delimita l'uso è invece di Hewitt e Liang {cite}`hewitt2019control`, e va
attribuita a loro: un probe troppo potente rischia di *imparare* lui la
proprietà invece di limitarsi a leggerla. La loro proposta per accorgersene
sono i *control task*, cioè rifare lo stesso addestramento su un'etichettatura
casuale, e misurare la **selectivity**, lo scarto fra quanto il probe riesce
sulla proprietà vera e quanto riesce sul casuale. Un probe che va bene su
entrambe non stava leggendo: stava risolvendo.

`````

## Interpretabilità meccanicistica: fare reverse-engineering dei circuiti

Attribuzione e probing dicono *cosa* pesa e *dove* sta l'informazione, ma non
*come* la rete la calcola. La frontiera (giovane, ambiziosa, ancora molto
aperta) punta più in alto: **fare reverse-engineering** dei calcoli interni,
come si smonta un circuito elettronico per capire cosa fa ciascun componente.
È l'**interpretabilità meccanicistica**.

`````{tab} Elementare

Finora abbiamo trattato la rete come una scatola su cui bussare da fuori: le
mostri un ingresso, guardi l'uscita, misuri le reazioni. L'interpretabilità
meccanicistica apre la scatola e prova a leggere il circuito dentro.
L'obiettivo è ricostruire i **circuiti**: piccoli gruppi di neuroni collegati
che, insieme, svolgono un compito riconoscibile (un rilevatore di curve, un
pezzo che tiene il conto delle parentesi aperte in un testo).

C'è però un ostacolo curioso, chiamato **sovrapposizione**: la rete ha meno
neuroni dei concetti che deve rappresentare, e allora fa come chi ha poche
scatole e troppa roba; mette più concetti nella stessa scatola, e un singolo
neurone finisce per accendersi per cose scollegate (un po' per i gatti, un po'
per le automobili, un po' per il colore verde).

Una tecnica recente, gli *sparse autoencoder*, prova a «ri-sistemare gli
scatoloni». Serve prima una parola: quando un'immagine o una frase attraversa
la rete, ogni neurone di uno strato produce un numero, e quei numeri tutti
insieme sono le **attivazioni** di quello strato, cioè la fotografia di ciò che
la rete ha in mente lì dentro in quel momento. Lo sparse autoencoder prende
quella fotografia e la riscrive usando **molte più caselle** di quanti erano i
neuroni, chiedendo però che a ogni esempio se ne accendano pochissime. Con
tanto posto a disposizione e l'obbligo di usarne poco, ogni casella ha
convenienza a specializzarsi su **una cosa sola**, che è esattamente quello che
vorremmo poter leggere.

Il campo è giovane e va raccontato per quello che è. La tecnica ha retto la
prova della scala, e si applica ormai a modelli linguistici veri, non a
giocattoli. Quello che non è dimostrato è il punto d'arrivo: che ogni casella
contenga davvero una cosa sola resta un giudizio dato guardandole a campione, e
si sono trovati casi in cui una casella che sembrava pulita non si accende
proprio dove dovrebbe. Nessuno, comunque, ha ancora letto una rete grande per
intero.

`````

`````{tab} Superiore

Il programma dei **circuiti** è stato articolato da Olah e colleghi su
*Distill* nel 2020 {cite}`olah2020zoom`: studiare una rete come un oggetto
scientifico, individuando *feature* (direzioni nello spazio delle attivazioni
che codificano un concetto) e i *circuiti* che le collegano; sottografi di
neuroni e pesi che implementano un calcolo interpretabile, come i rilevatori
di curve nelle prime reti di visione.

L'ostacolo teorico è la **sovrapposizione** (*superposition*), studiata a
fondo da Elhage e colleghi nei *Toy Models of Superposition* (Anthropic, 2022)
{cite}`elhage2022toy`: una rete con $n$ neuroni può rappresentare molte più di
$n$ feature sfruttando direzioni quasi ortogonali in $\mathbb{R}^n$, purché
ciascuna feature sia rara. La
conseguenza pratica è la **polisemanticità**: un singolo neurone risponde a
stimoli non correlati, e diventa illeggibile. Bricken e colleghi, in *Towards
Monosemanticity* (Anthropic, 2023), affrontano il problema con uno **sparse
autoencoder** {cite}`bricken2023monosemanticity`: le attivazioni di uno strato
vengono ricodificate in un dizionario **sovracompleto** (molte più unità dei
neuroni originali) sotto un vincolo di **sparsità**, che spinge poche unità
attive per esempio. Le feature così estratte risultano in buona parte
**leggibili**, e conviene prendere «in buona parte» e «leggibili» per quel che
sono: il giudizio di valutatori umani (o di un modello usato come valutatore)
su un campione di feature, non una proprietà dimostrata.

Il campo è nascente e va preso con l'onestà che si deve alle frontiere, ma vale
la pena dire **dove** stia oggi la frontiera, perché non è dove si tende a
metterla. La prova della **scala** la tecnica l'ha superata: Templeton e
colleghi {cite}`templeton2024scaling` hanno addestrato autoencoder sparsi fino
a decine di milioni di feature sulle attivazioni interne di un modello
linguistico di produzione, non di un giocattolo di laboratorio. Quella che non
ha superato è la prova dell'**affidabilità**: si sono documentati modi
sistematici in cui un latente apparentemente monosemantico non si accende
proprio dove dovrebbe, perché un latente più specifico ne ha assorbito i casi,
e il fenomeno non si risolve cambiando la dimensione del dizionario o il grado
di sparsità. La sovrapposizione resta una spiegazione teorica solida del
*perché* i neuroni siano illeggibili; che gli sparse autoencoder siano *la*
cura è ancora un programma di ricerca, non un risultato acquisito. E nessuno,
in ogni caso, ha ancora «letto» un modello di grande scala per intero. La posta
in gioco, però, è alta, ne parliamo qui sotto.

`````

```{figure} ../figures/toy-models-superposition.svg
:name: fig-superposizione
:alt: "Due piani a due dimensioni. Nel primo, con feature dense, i due assi interni ospitano due sole feature, ad angolo retto fra loro, una per direzione. Nel secondo, con feature sparse, gli stessi due assi ospitano cinque feature disposte a raggiera: non sono ad angolo retto, si sovrappongono, ma poiché raramente sono attive insieme il modello riesce comunque a distinguerle."
:width: 92%

La sovrapposizione. Con due sole direzioni a disposizione si possono tenere più
di due concetti, purché ciascuno si accenda di rado e quasi mai insieme agli
altri: a destra ce ne stanno cinque, sistemati a raggiera invece che ad angolo
retto. Il prezzo è che nessuno di essi ha più una direzione tutta sua.
```

{numref}`fig-superposizione` mostra perché smontare una rete sia difficile
oltre il previsto. La speranza naturale è che ogni neurone corrisponda a un
concetto; se invece i concetti sono più delle direzioni disponibili e ci
convivono a raggiera, il singolo neurone risponde a un miscuglio di cose senza
rapporto fra loro, ed è esattamente ciò che si osserva guardando dentro i
modelli.

```{figure} ../figures/interpretabilita-scatola-nera.svg
:name: fig-sparse-autoencoder
:alt: "A sinistra uno strato di attivazioni disegnato come un fascio di direzioni aggrovigliate, in cui ogni neurone mescola più concetti. Una freccia le fa attraversare uno sparse autoencoder. A destra le feature che ne escono, disegnate come caselle separate, ciascuna con il nome di un concetto leggibile."
:width: 96%

La mossa che scioglie il groviglio. Le attivazioni aggrovigliate entrano da
sinistra e ne escono riscritte: al posto di neuroni che mescolano più cose,
caselle che ne tengono una sola. Dentro la freccia ci sono due mosse insieme, e
servono tutte e due: le caselle in uscita sono **molte di più** dei neuroni in
entrata, e a ogni esempio se ne accendono **pochissime**.
```

La direzione di {numref}`fig-sparse-autoencoder` sembra paradossale (per capire
meglio si fa più largo il posto in cui la rete tiene le sue rappresentazioni,
invece di stringerlo) e invece è la conseguenza diretta della figura
precedente. Se il problema è che troppe cose stanno in troppo poco spazio, la
cura è dare più spazio, e imporre con la sparsità che ciascuna si prenda la
propria direzione invece di dividerla con altre.

Perché tutto questo conta, e non è solo un esercizio di curiosità? Per la
**sicurezza**. Un modello linguistico di grandi dimensioni può apprendere
comportamenti che non vogliamo (inganni, scorciatoie, bias) senza che nulla,
dall'esterno, li tradisca. Poter leggere i circuiti interni significherebbe
accorgersene *prima* che si manifestino: è il ponte, che riprenderemo nel
capitolo sull'AI responsabile, tra l'interpretabilità come curiosità
scientifica e l'interpretabilità come strumento di controllo.

Il quadro concettuale finisce qui. Le due sezioni che seguono sono di bottega:
rifanno coi numeri, e con poche righe di codice, i due metodi centrali del
capitolo, gli Integrated Gradients e Grad-CAM. Chi non programma può saltarle e
andare al riquadro finale.

## Integrated Gradients coi numeri: un esempio eseguibile

Vale più di mille formule vedere la completezza tornare al centesimo. Prendiamo
una funzione giocattolo di due variabili costruita apposta per **saturare**,
cioè per riprodurre il caso in cui il gradiente locale mente. È la solita somma
pesata di un neurone (due volte la prima variabile, meno una volta la seconda)
passata dentro la **tangente iperbolica** $\tanh$, una funzione che schiaccia:
sale ripida vicino allo zero e si appiattisce man mano che l'uscita si avvicina
a 1, come la sigmoide di poco fa. In formule,
$f(\mathbf{x}) = \tanh(\mathbf{w}^\top \mathbf{x})$ con
$\mathbf{w} = (2, -1)$: la scrittura $\mathbf{w}^\top \mathbf{x}$ è il modo
compatto di dire «moltiplica ogni peso per la variabile che gli corrisponde e
somma tutto», cioè appunto la somma
pesata. Nel punto $\mathbf{x} = (2, 1)$ essa vale $2 \cdot 2 - 1 \cdot 1 = 3$, e
$\tanh(3) \approx 0{,}995$: siamo sulla parte piatta della curva, dove la
pendenza è quasi nulla. Un solo gradiente direbbe «qui non conta niente»; gli
Integrated Gradients, integrando dalla baseline tutta a zero, recuperano
l'intero contributo. Nel codice, il commento «regola della catena» segnala il
modo standard di derivare una funzione dentro un'altra: si deriva prima quella
esterna, la $\tanh$, poi quella interna, la somma pesata, e si moltiplicano i
due risultati.

```python
import numpy as np

# funzione giocattolo che satura: f(x) = tanh(w . x)
w = np.array([2.0, -1.0])

def f(x):
    return np.tanh(w @ x)

def grad_f(x):
    z = w @ x
    return (1.0 - np.tanh(z) ** 2) * w   # regola della catena

x = np.array([2.0, 1.0])     # input da spiegare
baseline = np.zeros(2)        # baseline neutra (lo "zero")

# gradiente grezzo nel solo punto x: saturo, quasi nullo -> saliency cieca
print("gradiente in x :", np.round(grad_f(x), 4))       # [ 0.0197 -0.0099]

# Integrated Gradients: media dei gradienti lungo il cammino baseline -> x
m = 200
alphas = (np.arange(1, m + 1) - 0.5) / m   # punti medi delle m tappe
grad_medio = np.zeros(2)
for a in alphas:
    grad_medio += grad_f(baseline + a * (x - baseline))
grad_medio /= m
ig = (x - baseline) * grad_medio
print("attribuzioni IG:", np.round(ig, 4))              # [ 1.3267 -0.3317]

# assioma di completezza: la somma delle attribuzioni = f(x) - f(baseline)
print("somma IG       :", round(ig.sum(), 4))           # 0.9951
print("f(x) - f(base) :", round(f(x) - f(baseline), 4)) # 0.9951
```

Il gradiente nel punto è $(0{,}0197,\, -0{,}0099)$: minuscolo, come previsto per
un neurone saturo. Ma le attribuzioni integrate valgono $(1{,}327,\, -0{,}332)$
e la loro **somma**, $0{,}9951$, coincide al quarto decimale con
$f(x) - f(\text{baseline}) = 0{,}9951$: la completezza è verificata. Notate anche
il segno: la prima variabile ($w_1 = 2 > 0$) spinge il punteggio in alto, la
seconda ($w_2 = -1 < 0$) lo tira giù, esattamente come ci si aspetta.

## Uno sketch di Grad-CAM in PyTorch

Su una rete vera, Grad-CAM si costruisce agganciando due *hook* allo stadio
convoluzionale finale: uno cattura le attivazioni in avanti, l'altro i
gradienti all'indietro. In una ResNet il punto giusto è l'**uscita dell'ultimo
blocco** di `layer4`, dopo la somma residuale: è lì che agganciano le
implementazioni di riferimento, perché fermarsi a una convoluzione interna al
blocco ignorerebbe il contributo della scorciatoia. Ecco lo scheletro su una
ResNet-18 di `torchvision`, con l'API reale.

```python
import torch
import torch.nn.functional as F
from torchvision import models

model = models.resnet18(weights=models.ResNet18_Weights.IMAGENET1K_V1).eval()
target = model.layer4[-1]                # ultimo blocco: uscita post-residuo

att, grad = {}, {}
target.register_forward_hook(lambda m, i, o: att.__setitem__("v", o.detach()))
target.register_full_backward_hook(
    lambda m, gi, go: grad.__setitem__("v", go[0].detach())
)

x = torch.randn(1, 3, 224, 224)          # immagine gia pre-processata
logit = model(x)                          # (1, 1000)
classe = logit.argmax(dim=1)              # classe predetta
model.zero_grad()
logit[0, classe].backward()               # gradiente della sola classe scelta

A = att["v"]                              # attivazioni  (1, C, h, w)
dY = grad["v"]                            # gradienti    (1, C, h, w)
alpha = dY.mean(dim=(2, 3), keepdim=True)  # peso per canale (global avg pool)
heatmap = F.relu((alpha * A).sum(dim=1))   # (1, h, w), solo contributi positivi
heatmap = heatmap / (heatmap.max() + 1e-8) # normalizzata in [0, 1]
# heatmap va poi sovracampionata a 224x224 e sovrapposta all'immagine

print("attivazioni:", tuple(A.shape))
print("gradienti  :", tuple(dY.shape))
print("heatmap    :", tuple(heatmap.shape))
```

Le tre righe stampate dicono che cosa è uscito:

```text
attivazioni: (1, 512, 7, 7)
gradienti  : (1, 512, 7, 7)
heatmap    : (1, 7, 7)
```

cioè 512 mappe di attivazione da $7 \times 7$ ciascuna (il $7\times 7$
annunciato più sopra per una ResNet su ingresso $224 \times 224$), altrettanti
gradienti, e una sola heatmap $7 \times 7$ che le riassume. Il cuore è tutto
nelle ultime tre righe di calcolo: `alpha` è il peso $\alpha_k^c$ (la
media spaziale del gradiente), la somma pesata delle mappe seguita dalla
`relu` è $L^c_{\text{Grad-CAM}}$, e la divisione la porta in $[0,1]$ per
visualizzarla (il $10^{-8}$ evita una divisione per zero nel caso, raro ma
possibile su una classe non predetta, in cui la `relu` azzeri tutta la mappa).
Su un'immagine di cane la macchia calda cadrebbe sul muso; su
un husky del dataset ingannevole, sulla neve, ed è precisamente questo che
volevamo poter vedere.

`````{tab} Elementare

```{admonition} Da ricordare
:class: important
- Una rete profonda non ha numeri leggibili come i coefficienti di un modello
  lineare. Per capirla si cambia domanda: quanto ha pesato *questo* pezzo
  dell'ingresso su *questa* decisione? Si misura di quanto cambierebbe la
  risposta toccandolo appena.
- Le **saliency maps** (Simonyan e colleghi, 2014) danno questi colpetti pixel
  per pixel: informative ma rumorose, e valide solo attorno a quella foto.
  **Grad-CAM** (Selvaraju e colleghi, 2017) accende invece i «faretti»
  dell'ultimo strato convoluzionale in proporzione a quanto contano per la
  classe: più grossolana, ma pulita e affidabile nel dire *dove* la rete ha
  guardato.
- Gli **Integrated Gradients** (Sundararajan e colleghi, 2017) partono da
  un'immagine neutra e arrivano a quella vera per tappe, registrando la
  reazione lungo tutta la salita: così vedono anche i contributi che
  all'arrivo, ormai satura, la rete non segnala più. La somma delle quote fa
  esattamente il salto di fiducia fra le due immagini, come un conto di cena
  che torna al centesimo. Con un'avvertenza: il conto torna **da qualunque
  punto si parta**, quindi che torni non dimostra che il punto di partenza sia
  quello giusto; e di ciò che era già nero in partenza non si misura nessun
  contributo, nemmeno se era la ragione della decisione.
- Una mappa dice **dove**, non **che cosa** (Cynthia Rudin): sapere quale zona
  la rete guarda non dice che cosa ci trovi. E i **controlli di sanità**
  (Adebayo e colleghi, 2018) mostrano che, cancellando ciò che il modello ha
  imparato, per parecchi metodi la mappa resta quasi identica: descriveva
  l'immagine, non la rete. I due metodi di questa sezione il test lo superano;
  a fallirlo sono due varianti più elaborate che qui non abbiamo incontrato.
  Una spiegazione che sembra sensata non è per questo fedele.
- I **pesi di attenzione** sono un indizio, non una prova: pesi molto diversi
  possono portare alla stessa risposta. E guardare un solo strato non basta,
  perché una parte dell'informazione salta l'attenzione e prende la scorciatoia
  verso il piano di sopra; **attention rollout** e *attention flow* rifanno il
  conto lungo tutta la pila. Il **probing** risponde a un'altra domanda: a
  quale piano è scritta una certa informazione.
- L'**interpretabilità meccanicistica** apre la scatola e prova a ricostruire i
  circuiti con cui la rete calcola, sciogliendo la **sovrapposizione** (troppi
  concetti nella stessa scatola, un neurone che si accende per cose scollegate)
  con gli *sparse autoencoder*. La tecnica si applica ormai a modelli veri e
  grandi; che le caselle che ne escono contengano davvero **una cosa sola**
  resta però un giudizio dato guardandole, non una cosa dimostrata. Campo
  giovane, ma centrale per la sicurezza dei modelli grandi.
```

`````

`````{tab} Superiore

```{admonition} Da ricordare
:class: important
- Le reti profonde non hanno coefficienti leggibili: l'**attribuzione** usa il
  **gradiente dell'uscita rispetto all'ingresso** per stimare quanto ogni parte
  dell'input ha pesato su una singola decisione.
- Le **saliency maps** {cite}`simonyan2014deep` sono il gradiente sui pixel:
  informative ma rumorose e locali. **Grad-CAM** {cite}`selvaraju2017grad` pesa
  le mappe dell'ultimo strato convoluzionale coi gradienti della classe e
  localizza in modo robusto *dove* guarda la CNN.
- Gli **Integrated Gradients** {cite}`sundararajan2017axiomatic` integrano il
  gradiente lungo il cammino dalla baseline all'input: fondati su assiomi
  (sensibilità **a input e baseline che differiscono in una sola componente**,
  invarianza all'implementazione), risolvono la saturazione e
  soddisfano la **completezza**,
  $\sum_i \mathrm{IG}_i = f(\mathbf{x}) - f(\mathbf{x}')$. La completezza vale
  però per **qualunque** baseline: $\mathbf{x}'$ è il grado di libertà che gli
  assiomi non vincolano, e ogni componente uguale alla baseline riceve
  attribuzione nulla per costruzione {cite}`sturmfels2020baselines`.
- Una mappa dice **dove**, non **che cosa** {cite}`rudin2019stop`, e i
  **controlli di sanità** {cite}`adebayo2018sanity` mostrano che per diversi
  metodi popolari la mappa cambia pochissimo randomizzando i pesi del modello:
  descriveva l'immagine, non la rete. Nel paper i bocciati sono **Guided
  BackProp** e **Guided Grad-CAM**, mentre gradiente semplice e Grad-CAM
  passano. La plausibilità visiva di una spiegazione
  non è una prova della sua fedeltà.
- I **pesi di attenzione** non sono di per sé una spiegazione affidabile
  (dibattito *«Attention is not Explanation»*, {cite}`jain2019attention` e
  {cite}`wiegreffe2019attention`, con l'avvertenza che quegli esperimenti sono
  su un singolo strato di attenzione sopra una BiLSTM, non su un Transformer).
  E prima ancora c'è un
  problema tecnico: comporre gli strati richiede di tener conto delle
  connessioni residuali, che è ciò che fanno **attention rollout** e
  *attention flow* {cite}`abnar2020quantifying`. Il **probing** con
  classificatori lineari {cite}`alain2017understanding` mappa invece dove sta
  l'informazione negli strati interni, con i *control task* di Hewitt e Liang
  {cite}`hewitt2019control` a misurare che stia leggendo e non risolvendo.
- L'**interpretabilità meccanicistica** (circuiti {cite}`olah2020zoom` e
  sparse autoencoder {cite}`bricken2023monosemanticity`) punta a fare
  reverse-engineering dei calcoli interni. La scala non è più il limite
  {cite}`templeton2024scaling`; la **monosemanticità** delle feature estratte
  sì, ed è tuttora contesa. Campo giovane, ma centrale per la sicurezza degli
  LLM.
```

`````
