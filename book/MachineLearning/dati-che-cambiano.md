# Quando i dati cambiano

Nel 2008 Google lanciò un servizio dal nome ambizioso: **Google Flu Trends**.
L'idea, presentata anche sulle pagine di *Nature*, era elegante: chi si sente
febbricitante corre a cercare in rete "sintomi influenza" o "febbre alta
rimedi". Contando queste ricerche si poteva stimare la diffusione
dell'influenza negli Stati Uniti in tempo quasi reale, con una o due settimane
di anticipo sui CDC, l'ente federale che raccoglie i dati dai medici. Per
qualche anno funzionò, e il servizio divenne il manifesto di una promessa che
in quegli anni si sentiva dappertutto: adesso che ogni azione lascia una traccia
in rete, perché fare sondaggi lenti e costosi, quando basta leggere i dati che
la rete produce già?

Poi arrivò l'inverno 2012–2013. A febbraio 2013 Google Flu Trends stimava una
quota di visite mediche per sintomi influenzali più che doppia rispetto a
quella registrata dai CDC. Non fu un incidente isolato. Tra l'agosto 2011 e il
settembre 2013 il servizio aveva sovrastimato l'influenza in 100 settimane
su 108. Nel 2014 un gruppo di ricercatori firmò su *Science* l'autopsia del
progetto, con un titolo diventato proverbiale: *The Parable of Google Flu* (la
"parabola", nel senso del racconto che ammonisce {cite}`lazer2014parable`).
Nell'agosto 2015 Google chiuse il servizio.

Che cosa era andato storto? Gli autori dell'articolo su *Science* indicano due
cause, e solo la seconda è un invecchiamento. La prima nasce con il modello,
perché i termini di ricerca erano stati scelti cercando, fra cinquanta milioni
di candidati, quelli che seguivano meglio 1152 rilevazioni settimanali dei CDC,
e molti seguivano l'inverno più che l'influenza. La versione iniziale,
scrivono, era in parte un rilevatore di influenza e in parte un rilevatore
d'inverno, tanto che nel 2009 mancò la pandemia di H1N1, arrivata fuori
stagione, e l'algoritmo fu corretto.

La seconda causa è il tempo. Il modo di cercare in rete era cambiato, e Google
stessa aggiornava il motore. Nel giugno 2011 annunciò di aver modificato i
risultati per proporre altri termini da cercare (a chi chiedeva dell'influenza
suggeriva di cercarne le cure), nel febbraio 2012 di rispondere alle ricerche
sui sintomi con le diagnosi possibili (chi cercava «febbre» o «tosse» si vedeva
proporre l'influenza). Sono due spinte verso l'influenza che arrivavano anche a
chi stava benissimo, e gli autori giudicano plausibile che abbiano gonfiato le
ricerche sul tema, e il modello, che contava quelle ricerche, le avrebbe lette
come malati in più. Nell'inverno 2012–2013 si aggiunsero i giornali, che
parlavano di epidemia, e molti cercavano per curiosità e non per febbre. Il
modello, tarato sul mondo di qualche anno prima, continuava a leggere il
presente con gli occhiali di allora. Che cosa succede quando i dati che un
modello incontra non somigliano più a quelli su cui è stato addestrato?

## L'ipotesi nascosta: che la distribuzione non cambi

L'impianto dell'apprendimento supervisionato (addestramento, validazione, test)
poggia su un'assunzione che di solito non viene dichiarata: che i dati su cui il
modello è addestrato e quelli che incontrerà dopo siano estratti dalla stessa
distribuzione, e che ogni esempio sia estratto senza essere condizionato dagli
altri. È l’**ipotesi i.i.d.** (*indipendenti e identicamente distribuiti*).
Finché vale, l'errore misurato sugli esempi di test stima l'errore che il modello
farà sugli esempi futuri. Nessuna legge assicura che il mondo continui a
rispettarla.

`````{tab} Elementare

Un sondaggio elettorale intervista mille persone e prevede il voto di milioni.
Funziona solo perché le mille persone sono scelte in modo da *somigliare* ai
milioni. Training, validation e test sono, in fondo, un sondaggio: peschiamo
esempi da un'urna e contiamo, fidandoci che l'urna di domani sia la stessa di
oggi.

Se in un'urna ben mescolata il 30% delle palline è rosso, in una manciata di
cento ne troverai più o meno trenta rosse: la manciata "parla" per l'urna
intera. Ma se stanotte qualcuno sostituisce metà delle palline, la manciata
pescata ieri non dice più nulla sull'urna di oggi. È successo a
Google Flu Trends: l'urna (il modo in cui la gente usa un motore di ricerca)
era cambiata, e nessuno aveva avvisato il modello.

La manciata parla per l'urna anche a un'altra condizione, che riguarda il
momento in cui si decide la domanda. Se «Quante rosse?» viene stabilito prima di
pescare, la manciata risponde per l'urna. Stabilito dopo, no: chi pesca cento
palline e poi va a cercare, fra mille domande possibili, quella a cui quelle
cento rispondono nel modo più vistoso ha ritagliato la domanda addosso alla
manciata, e sull'urna intera quella domanda non dice niente. È l'overfitting
visto dal lato dell'urna, e la prima delle due cause di Google Flu Trends, che
aveva scelto i suoi termini di ricerca fra cinquanta milioni guardando come
andavano su quei dati. La difesa è la stessa: tenere corto l'elenco delle
domande che ci si concede prima di pescare.

Questa urna ha anche un nome tecnico. La distribuzione di una grandezza è il
resoconto di quanto spesso capita ciascun valore: le palline rosse al 30% e
le altre al 70% *sono* la distribuzione dei colori in quell'urna. Dire che la
distribuzione degli input è cambiata equivale a dire che l'urna non è più
quella, e un'urna si cambia in più di un modo: mettendoci dentro palline
diverse, cambiando quanto spesso la risposta giusta è una o l'altra, oppure
cambiando il significato del colore di una pallina.

`````

`````{tab} Superiore

Formalmente, assumiamo che le coppie $(\mathbf{x}^{(i)}, y^{(i)})$ del training
e quelle che il modello vedrà in produzione siano estratte in modo indipendente
e identicamente distribuito (i.i.d.) da un'unica distribuzione congiunta
$P(\mathbf{x}, y)$. Sotto questa ipotesi l'errore misurato sul campione
converge, per la legge dei grandi numeri, all'errore atteso di qualunque modello
*fissato in anticipo*. Perché la stessa garanzia valga per il modello scelto
minimizzando sui dati serve di più: che la classe di ipotesi abbia capacità
limitata (in gergo, la convergenza *uniforme*, che la {doc}`teoria
dell'apprendimento </TeoriaApprendimento/overview>` tratta per esteso). È lo
stesso controllo della complessità incontrato con l'overfitting, che di quella
garanzia è appunto il controesempio: quando la capacità non è limitata, l'errore
sul campione può essere azzerato senza che questo dica più nulla sull'errore
atteso. Quando invece valgono entrambe le condizioni, campionamento i.i.d. e
capacità limitata, minimizzare la loss empirica è una buona approssimazione del
minimizzare il rischio vero.

Quando le due distribuzioni divergono, $P_{\text{train}}(\mathbf{x}, y) \neq
P_{\text{prod}}(\mathbf{x}, y)$, con $P_{\text{prod}}$ la distribuzione che il
modello incontra in produzione, si parla di **dataset shift**
{cite}`quinonero2009dataset`, e le garanzie cadono: il minimo della loss sotto
$P_{\text{train}}$ non è più, in generale, un buon punto sotto
$P_{\text{prod}}$. La congiunta si scrive in due modi,

$$
P(\mathbf{x}, y) = P(y \mid \mathbf{x})\, P(\mathbf{x}) = P(\mathbf{x} \mid y)\, P(y),
$$

dove $P(\mathbf{x})$ è la distribuzione degli input, $P(y)$ quella delle
etichette, $P(y \mid \mathbf{x})$ la relazione input-etichetta che il modello
cerca di apprendere e $P(\mathbf{x} \mid y)$ come si presentano gli input di
ciascuna classe. I fattori non cambiano in modo indipendente (con $P(y \mid
\mathbf{x})$ ferma, cambiare $P(\mathbf{x})$ cambia anche $P(y)$), e le famiglie
di shift si distinguono da quale condizionata resta ferma.

`````

## Tre modi in cui il mondo cambia

Non tutti i cambiamenti sono uguali. Conta quale parte della congiunta è
cambiata, perché da questo dipende che cosa si può correggere senza etichette
nuove. Le famiglie più usate sono tre, *covariate shift*, *label shift* (o
*prior probability shift*) e *concept shift*, nella sistemazione di
Moreno-Torres e coautori {cite}`morenotorres2012unifying`, e nella pratica
arrivano mescolate. {numref}`fig-distribution-shift` mostra il caso più
semplice da visualizzare: le densità di una stessa caratteristica (i metri
quadri, l'età, il numero di ricerche) in addestramento e in produzione, con i
valori in orizzontale e in verticale quanto spesso ciascuno capita. Le due
curve sfalsate dicono che i valori frequenti in addestramento non sono quelli
frequenti in produzione, e che il modello viene interrogato dove ha visto pochi
esempi.

```{figure} ../figures/distribution-shift.svg
:name: fig-distribution-shift
:alt: Due curve a campana sfalsate lungo l'asse degli input, una per i dati di addestramento e una spostata a destra per i dati in produzione; una linea tratteggiata indica il punto dove il modello è tarato, lontano dal grosso dei dati nuovi.
:width: 85%

Due distribuzioni a confronto: la curva dei dati di addestramento e quella dei
dati che il modello incontra una volta al lavoro. La linea tratteggiata segna il
valore attorno a cui il modello è tarato, cioè dove i dati di ieri si
addensavano; i dati di oggi cadono in gran parte altrove, dove non
ha quasi mai visto esempi.
```

`````{tab} Elementare

Cambiano le domande, cioè gli input che arrivano al modello (*covariate
shift*). Un'app che riconosce le piante, addestrata su foto scattate d'estate,
viene usata d'inverno: luce bassa, rami spogli, neve sullo sfondo. Le foto che
arrivano sono diverse da quelle viste a lezione, ma un abete resta un abete. La
*regola* che collega foto e risposta non è cambiata; è cambiato il tipo di foto
che arriva. Davanti a un ramo spoglio nella neve dà lo stesso una risposta, e
la dà su un caso che d'estate non poteva capitarle, estendendo la sua
esperienza oltre il punto in cui l'ha acquisita.

Cambiano le proporzioni delle risposte (*label shift*). Un modello aiuta a
diagnosticare una malattia che, quando è stato addestrato, colpiva una persona
su mille. Arriva un'epidemia e i malati diventano uno su cinquanta. I sintomi
della malattia sono identici a prima; cambia solo *quanto spesso* la risposta
giusta è «positivo».

Se i sintomi sono gli stessi, perché il modello dovrebbe sbagliare? Perché la
rarità, di nascosto, è entrata nel suo giudizio. Un modello che ha imparato su
un mondo in cui i malati sono uno su mille ha anche imparato che dire «sano»
paga: ci prende 999 volte su 1000, e per convincerlo del contrario servono
sintomi molto chiari. Davanti a un caso dubbio resterà prudente e dirà «sano»:
ieri era la scommessa giusta, oggi che i malati sono venti volte tanti è quella
sbagliata.

Cambia la regola stessa (*concept shift*). Che cos'è lo spam? Le stesse parole
("offerta", "clicca qui", "solo per oggi") che nel 2005 gridavano truffa, oggi
arrivano da negozi legittimi; e intanto i truffatori hanno imparato a scrivere
come una banca. Qui cambiano le domande e cambia anche la *risposta giusta alla
stessa domanda*. Allo spam sono serviti anni; altrove la regola cambia da un
giorno all'altro, come quando una legge nuova stabilisce che cosa conta come
transazione sospetta. È il caso più insidioso, perché nessuna quantità di dati
vecchi può insegnare una regola nuova.

Nel mondo vero i tre cambiamenti non arrivano uno alla volta e ben separati.
L'inverno porta insieme foto più scure e più abeti che margherite, cioè domande
diverse e proporzioni diverse in una volta sola. Capire quale dei tre pesi di
più, guardando i soli dati in arrivo, è difficile, perché per sapere se l'app
ha sbagliato bisogna che qualcuno guardi la foto e dica che pianta era, e quel
qualcuno arriva tardi, o non arriva affatto.

`````

`````{tab} Superiore

Con le due fattorizzazioni della congiunta, le tre famiglie si distinguono da
quale condizionata resta ferma {cite}`morenotorres2012unifying`:

- **Covariate shift**: cambia $P(\mathbf{x})$, resta invariata $P(y \mid
  \mathbf{x})$. Le foto invernali hanno una distribuzione diversa da quelle
  estive, ma la mappa immagine $\to$ specie è la stessa. È il caso della figura:
  il modello è accurato dove $p_{\text{train}}(\mathbf{x})$ è densa, e viene
  interrogato dove è quasi nulla (di fatto, un’estrapolazione).
- **Label shift** (o *prior probability shift*): cambia $P(y)$, resta invariata
  $P(\mathbf{x} \mid y)$. La malattia si presenta come prima, ma la sua
  prevalenza è diversa. Un classificatore bayesiano tarato sul *prior* vecchio
  produce probabilità a posteriori sistematicamente distorte.
- **Concept shift**: cambia $P(y \mid \mathbf{x})$. La relazione input-etichetta
  stessa si sposta, gradualmente (i gusti musicali) o di colpo (una nuova legge
  cambia cosa è "transazione sospetta"). Nella letteratura sui flussi di dati è
  il *real concept drift*; *concept drift* senza aggettivi indica di solito
  qualunque cambiamento della congiunta nel tempo {cite}`gama2014survey`.

Ciascuna famiglia è un'ipotesi su come le due distribuzioni si legano, e senza
nessuna ipotesi non si impara niente: se le etichette si capovolgessero a parità
di $P(\mathbf{x})$, nessun dato di addestramento potrebbe dirlo. Il covariate
shift è l'ipotesi naturale quando $\mathbf{x}$ causa $y$ (le caratteristiche di
una casa determinano il prezzo), il label shift quando $y$ causa $\mathbf{x}$
(la malattia produce i sintomi).

Nella pratica le tre forme arrivano mescolate, e distinguere quale domini a
partire dai soli dati è un problema difficile: spesso mal posto, se le etichette
nuove tardano ad arrivare. La terna non esaurisce i casi: un cambiamento di
$P(\mathbf{x} \mid y)$ a $P(y)$ fisso, per esempio, non è nessuna delle tre
{cite}`morenotorres2012unifying`.

`````

## Perché la validazione classica non protegge

Validation set, test tenuto da parte e cross-validation, presentati in
{doc}`Overfitting e validazione <overfitting-validazione>`, restano necessari,
ma hanno un punto cieco: estraggono i loro esempi dallo stesso passato da cui
viene il training set, quindi stimano l'errore sotto la distribuzione di
addestramento, non sotto quella che il modello incontrerà dopo.

`````{tab} Elementare

Train, validation e test sono tre ritagli della *stessa fotografia*. Se la
fotografia invecchia, invecchiano tutti e tre insieme, e i loro voti restano
alti mentre il modello, nel mondo reale, comincia a sbagliare. È come guidare
guardando lo specchietto retrovisore, che ti dice benissimo la strada già
percorsa ma non la curva che sta arrivando.

Google Flu Trends andava bene sui dati di validazione, perché era stato
validato sul passato. Quel voto misurava quanto il modello somigliasse al
passato, non quanto avrebbe retto a un'altra stagione o a un altro modo di
cercare, e fra i termini scelti ce n'erano di legati all'inverno più che
all'influenza. La validazione risponde a "quanto sbaglierò su dati *come
questi*?", non a "quanto sbaglierò *domani*?".

C'è però un modo di rendere onesta anche la domanda, e vale ogni volta che i
dati hanno una data sopra, e invece di tagliarli a caso si taglia nel tempo. Il
modello studia su gennaio-ottobre e viene interrogato su novembre-dicembre, che
al momento dell'addestramento erano il futuro. Se già lì peggiora, è un segnale
serio che in mezzo al mondo vero peggiorerà (a meno che la finestra scelta abbia
una stagionalità sua, come le feste di dicembre); se non peggiora non è una
garanzia, ma è un indizio molto migliore di un rimescolamento che gli lascia
sbirciare il domani.

`````

`````{tab} Superiore

La stima di validazione approssima $\mathbb{E}_{(\mathbf{x},y)\sim
P_{\text{train}}}\!\left[\ell\big(f_\theta(\mathbf{x}), y\big)\right]$: un
valore atteso sotto la distribuzione di addestramento. Se la distribuzione
operativa è un'altra, questo numero non vincola l'errore reale senza un'ipotesi
sulla distanza fra le due distribuzioni: può restare ottimo mentre l'errore
sotto $P_{\text{prod}}$ diverge.

Con dati temporali c'è di peggio: la cross-validation rimescolata distrugge
l'ordine cronologico e lascia che il modello "veda il futuro" dei fold di
validazione (una forma di *leakage* temporale), gonfiando le stime. Lo stress
test più onesto è lo **split temporale**: addestrare sul passato e validare sul
futuro relativo (ad esempio, addestrare su gennaio–ottobre e validare su
novembre–dicembre). Ripetuto su più tagli successivi, con l'origine che avanza,
è la validazione *a origine mobile*, quella che `TimeSeriesSplit` di
scikit-learn implementa. Se le prestazioni degradano già lì, è probabile che
degradino anche in produzione (salvo effetti stagionali propri della finestra);
se non degradano, non è comunque una garanzia, solo un indizio migliore.

`````

## Rimedi onesti

Nessun algoritmo rende un modello immune al tempo, e i rimedi di uso più comune
sono operativi prima che matematici. Sono tre:

1. **Monitoraggio** del modello in produzione, cioè acceso su dati veri, con
   utenti veri: va trattato come un impianto, non come un quadro appeso, e si
   confrontano nel tempo la distribuzione degli input, quella delle predizioni
   e, appena le etichette arrivano, l'errore effettivo, misurabile con gli
   strumenti descritti in {doc}`Sorvegliare un modello vivo
   </MLOps/monitoring-e-drift>`.
2. **Riaddestramento** (*retraining*) periodico su dati recenti, perché la
   distribuzione di addestramento segua quella di produzione.
3. **Valutazione su dati successivi** all'addestramento, raccolti dopo che il
   modello è stato fissato, e non su un altro ritaglio dei dati di partenza.

`````{tab} Elementare

Un modello in produzione è come la bilancia del mercato, che per legge va
ritarata periodicamente, perché con l'uso e il tempo si starano tutte, ed è
meglio accorgersene prima del cliente. In pratica si tengono d'occhio le
domande, le uscite e, appena si scopre la risposta giusta, gli errori veri. Se un
filtro antispam riceveva email lunghe in media 80 parole e ora ne arrivano da
200, è un campanello sulle domande. Se ieri segnalava come spam il 20% dei
messaggi e oggi il 45%, qualcosa è cambiato, nel mondo o nel
modello. Se gli utenti ripescano dal cestino molte email segnalate, gli errori
veri stanno salendo, e lo si sa.

Sui conteggi si può fare un aggiustamento che gli istituti di sondaggi fanno da
sempre, la ripesatura. Il campione di mille persone ne contiene 50 sotto i
trent'anni, il 5%, mentre nel paese quella fascia pesa il 20%, quattro volte
tanto. Allora ogni giovane intervistato viene contato quattro volte (20 diviso
5), e chi viene da una fascia sovrarappresentata conta meno di una persona
intera. Nessuno torna a bussare a nessuna porta, cambia solo il peso che ogni
risposta ha nella media finale, e il totale torna a somigliare al paese invece
che al campione. La stessa ripesatura ripara il caso della malattia diventata
comune, dove i malati erano uno su mille e adesso sono uno su cinquanta: ogni
malato del vecchio mucchio di esempi conta venti volte.

Ripesare corregge un solo difetto, quello di aver intervistato le persone
sbagliate. Se nel frattempo la gente ha cambiato idea, i giovani del campione
hanno espresso le opinioni di allora, e moltiplicare per quattro una risposta
vecchia dà una risposta vecchia quattro volte.

Se sotto i trent'anni non è stato intervistato nessuno, non c'è peso che
tenga, perché qualunque numero moltiplicato per zero fa zero. Di quella fascia
il sondaggio non sa niente, e nessun calcolo può inventare le interviste che non
sono state fatte.

Il caso che inganna di più sta in mezzo. Se sotto i trent'anni ne sono
capitati due invece di cinquanta, quei due devono valere per il 20% del paese
e contano cento volte ciascuno; basta che uno dei due cambi idea perché un
partito perda dieci punti. Il calcolo resta corretto e il risultato balla. Le
fasce quasi vuote si moltiplicano appena si ripesa per più cose insieme, età e
regione e titolo di studio e reddito, perché più le caselle sono strette e
meno gente ci finisce dentro.

Un modello può anche passare la mano. Davanti a un caso che non somiglia
a nulla di già visto, meglio dire "non lo so, decida un umano" che sparare una
risposta sicura e sbagliata.

`````

`````{tab} Superiore

Sul versante algoritmico, il rimedio classico per il *covariate shift* puro è
l’**importance weighting**: ripesare la loss di training così che gli esempi
frequenti in produzione ma rari in addestramento contino di più,

$$
\mathcal{L}_w(\theta) = \frac{1}{m} \sum_{i=1}^{m}
w\big(\mathbf{x}^{(i)}\big)\,
\ell\big(f_\theta(\mathbf{x}^{(i)}),\, y^{(i)}\big),
\qquad
w(\mathbf{x}) = \frac{p_{\text{prod}}(\mathbf{x})}{p_{\text{train}}(\mathbf{x})},
$$

dove $w(\mathbf{x})$ è il rapporto tra la densità degli input in produzione e
quella in addestramento e $m$ è il numero di esempi. Il motivo è un cambio di
misura: per ogni $\theta$ fissato,

$$
\mathbb{E}_{P_{\text{prod}}}\big[\ell(f_\theta(\mathbf{x}),y)\big]
= \iint \ell\;p_{\text{prod}}(y\mid\mathbf{x})\,\frac{p_{\text{prod}}(\mathbf{x})}{p_{\text{train}}(\mathbf{x})}\,p_{\text{train}}(\mathbf{x})\,d\mathbf{x}\,dy
= \mathbb{E}_{P_{\text{train}}}\big[w(\mathbf{x})\,\ell(f_\theta(\mathbf{x}),y)\big],
$$

dove l'ultimo passo usa $p_{\text{prod}}(y\mid\mathbf{x}) =
p_{\text{train}}(y\mid\mathbf{x})$, cioè il covariate shift, e la divisione
chiede $p_{\text{train}}(\mathbf{x}) > 0$ dovunque
$p_{\text{prod}}(\mathbf{x}) > 0$: il supporto di produzione deve stare dentro
quello di addestramento, e dove manca nessun peso può inventare esempi mai
raccolti. Quindi $\mathcal{L}_w(\theta)$ è una stima corretta del rischio in
produzione per ogni $\theta$, e con una classe di ipotesi di capacità limitata
il suo minimo converge a quello giusto al crescere di $m$.

Con $m$ finito la varianza di $\mathcal{L}_w$ cresce con $\mathbb{E}[w^2]$, e
stimare il rapporto di densità in alta dimensione è difficile: quando pochi
esempi ricevono pesi enormi, la varianza esplode. Il rimedio di Shimodaira è
appiattire i pesi, $w^\lambda$ con $\lambda \in [0,1]$: $\lambda = 0$ ignora
lo shift, $\lambda = 1$ lo corregge del tutto, e in mezzo si scambia distorsione
con varianza {cite}`shimodaira2000improving`. La taglia effettiva dice a quanti
esempi equivale il campione pesato, $m_{\text{eff}} = \bigl(\sum_i
w_i\bigr)^2 / \sum_i w_i^2$, e crolla appena pochi pesi dominano. Le due
densità, però, non vanno stimate una per una: un classificatore che distingue
addestramento e produzione, addestrato su $m_{\text{tr}}$ e $m_{\text{prod}}$
esempi, dà
$w(\mathbf{x}) =
\frac{m_{\text{tr}}}{m_{\text{prod}}}\,\frac{P(\text{prod}\mid\mathbf{x})}{1 -
P(\text{prod}\mid\mathbf{x})}$. È lo stesso classificatore di dominio che serve
ad accorgersi della deriva. Il ripeso serve soprattutto quando il modello è mal
specificato: se la famiglia contiene la vera $P(y \mid \mathbf{x})$, il minimo
non pesato è già consistente e i pesi aggiungono soltanto varianza
{cite}`shimodaira2000improving`.

Per il *label shift* il calcolo è più semplice, perché il rapporto dipende dalla
sola classe. Con $P(\mathbf{x} \mid y)$ invariata, $p_{\text{prod}}(y \mid
\mathbf{x}) \propto p_{\text{train}}(y \mid
\mathbf{x})\;\pi_{\text{prod}}(y)/\pi_{\text{train}}(y)$, dove $\pi(y)$ è la
prevalenza della classe: un classificatore calibrato si corregge senza
riaddestrarlo, moltiplicando le posteriori per
$\pi_{\text{prod}}(y)/\pi_{\text{train}}(y)$ e rinormalizzando (per la malattia
passata da uno su mille a uno su cinquanta, $20$ sui positivi e $0{,}98/0{,}999
\approx 0{,}98$ sui negativi). Resta da stimare $\pi_{\text{prod}}$, che senza
etichette nuove si ricava con un EM sulle posteriori
{cite}`saerens2002adjusting` o invertendo la matrice di confusione del modello
{cite}`lipton2018detecting`.

Una difesa complementare è l’**out-of-distribution detection**, che riconosce
gli input troppo lontani dalla distribuzione di addestramento e, invece di
predire con finta sicurezza, si astiene o segnala. Il problema è particolarmente
delicato per le reti profonde, che su input fuori distribuzione tendono a essere
*confidenti e sbagliate* insieme. Due strumenti già visti vanno in questa
direzione, la densità di una mistura gaussiana ({doc}`Riduzione e clustering
<riduzione-clustering>`) e la One-Class SVM ({doc}`Il kernel trick
<svm-kernel>`), e segnalano gli input che cadono dove la distribuzione di
addestramento ha densità bassa.

`````

Un cambiamento della distribuzione nel tempo si chiama **deriva** (*drift*), e
per accorgersi di una deriva negli input basta un **classificatore di dominio**
(*classifier two-sample test*) {cite}`lopezpaz2017revisiting`. Si uniscono i
dati di addestramento e quelli raccolti in produzione, si scarta l'etichetta
originale $y$ e se ne assegna una nuova a ogni esempio: 0 se viene
dall'addestramento, 1 se viene dalla produzione. Un modello addestrato a
indovinare la nuova etichetta dalle sole caratteristiche $\mathbf{x}$, una
specie di *detective*, ci riesce solo se le due distribuzioni degli input
differiscono, e il suo punteggio misura quanto.

Il punteggio da guardare è l'AUC della {doc}`sezione sulle metriche
<metriche>`: con $0{,}5$ il detective non distingue le due popolazioni, cioè
tira a indovinare, e con $1$ le separa senza errori.

Un'AUC vicina a $0{,}5$ non equivale a «va tutto bene», per due ragioni. La
prima è che il detective vede solo gli input, non le risposte: per costruzione
non può accorgersi del cambio di regola (in cui gli input restano gli stessi e
cambia la risposta giusta), il caso più insidioso dei tre. La seconda è che
un'AUC vicina a $0{,}5$ può anche indicare che gli esempi nuovi sono ancora
troppo pochi: uno scarto piccolo ma reale resta sotto il rumore di
campionamento dell'AUC, e un classificatore di capacità limitata può non vedere
un'interazione fra molte caratteristiche. Un'AUC alta è quindi un allarme
attendibile, mentre un'AUC vicina a $0{,}5$ non dimostra che la deriva manchi.

```python
import numpy as np
from sklearn.datasets import make_classification
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.model_selection import cross_val_score, train_test_split

X, y = make_classification(n_samples=400, n_features=8, n_informative=5,
                           n_redundant=1, random_state=0)
X_vecchi, X_nuovi, _, _ = train_test_split(X, y, test_size=0.25, random_state=0)

# in produzione arrivano gli stessi input, ma con la prima caratteristica
# scivolata di 1,5: è la deriva che il detective deve scoprire
X_derivati = X_nuovi.copy()
X_derivati[:, 0] += 1.5

def sospetto(X_prima, X_dopo):
    """Quanto bene un modello indovina da quale epoca viene un esempio."""
    X_tutti = np.vstack([X_prima, X_dopo])
    origine = np.hstack([np.zeros(len(X_prima)), np.ones(len(X_dopo))])
    return cross_val_score(GradientBoostingClassifier(random_state=0),
                           X_tutti, origine, cv=5, scoring="roc_auc").mean()

print(f"produzione con la deriva : {sospetto(X_vecchi, X_derivati):.3f}")
print(f"produzione senza deriva  : {sospetto(X_vecchi, X_nuovi):.3f}")
```

```text
produzione con la deriva : 0.736
produzione senza deriva  : 0.508
```

I due numeri sono le due letture possibili dello strumento e vanno letti
insieme. Il secondo, $0{,}508$, è il caso in cui non è cambiato niente: il
detective tira a indovinare, come chi lanciasse una monetina. Il primo,
$0{,}736$, è l'allarme: una sola caratteristica scivolata di un'unità e mezza
basta perché l'epoca di provenienza diventi in buona parte indovinabile.

## Imparare un esempio alla volta

Riaddestrare a intervalli aggiorna il modello a scatti: fra un riaddestramento e
il successivo resta fermo. L'alternativa è aggiornarlo a ogni esempio, appena se
ne conosce la risposta giusta, tenendo soltanto i parametri e non gli esempi già
visti: è l’**apprendimento online** (*online learning*). Cambia la domanda di
partenza: non più quanto il modello sbaglierà su esempi estratti dalla stessa
distribuzione dell'addestramento, perché qui non si suppone che gli esempi
vengano da una distribuzione fissa. Il metro del successo diventa il
**rimpianto** (*regret*): quanto si è perso rispetto alla migliore scelta fissa,
fatta col senno di poi su tutti gli esempi arrivati.

`````{tab} Elementare

Il filtro antispam impara una email alla volta. Arriva un messaggio, il filtro
dà il suo verdetto, e poco dopo tu gli dici se aveva ragione, lasciando l'email
dov'è o ripescandola dal cestino. Allora il filtro corregge un poco le sue
regole, nella direzione che su quell'email avrebbe ridotto l'errore, e passa
alla successiva. È la discesa del gradiente, fatta un esempio alla volta. Non
conserva le email vecchie, perché gli bastano le regole che ha adesso. Tiene
inoltre le regole dentro un recinto (tutti i pesi insieme non possono crescere
oltre una certa misura), così che una parola che compare in mille spam non
finisca per pesare all'infinito.

Tutto dipende da quanto correggere ogni volta. Correzioni grandi inseguono ogni
singola email; correzioni piccole imparano con lentezza. La ricetta che
funziona meglio sta nel mezzo e cambia col tempo, perché la correzione si
accorcia man mano che le email si accumulano, come uno diviso la radice di
quante ne sono arrivate, e alla centesima vale un decimo della prima.

A fine anno, con tutte le email davanti, cerchi il miglior filtro fisso: quello
che, applicato dal primo giorno, avrebbe sbagliato meno di tutti. Il rimpianto è
quanto il filtro che ha imparato strada facendo ha sbagliato in più. Un teorema
garantisce che il rimpianto cresce al più come la radice del numero di email,
cioè più piano delle email stesse. Se dopo cento email fosse dieci, dopo
diecimila sarebbe al più cento, e il rimpianto per email scenderebbe da un
decimo a un centesimo. Per email, quindi, tende a zero.

Più piano della radice, in generale, non si può promettere. Se le email fossero
decise a testa o croce nessun filtro potrebbe indovinarle; eppure a fine anno
uno dei filtri fissi sembrerebbe più bravo degli altri, per pura fortuna, di una
quantità che cresce come la radice (lanciando cento monete ci si
aspettano cinquanta teste, e cinquantacinque sono normali; con diecimila,
cinquemila più o meno cinquanta).

Si fa meglio solo se il problema aiuta. Quando ogni email punisce con decisione
qualunque filtro lontano dal migliore, con un errore che sale come una conca
ripida e non come un fondovalle piatto, la correzione si può accorciare più in
fretta, come uno diviso il numero di email, e il rimpianto cresce appena come
il logaritmo, cioè quasi niente. La garanzia non chiede niente alle
email, e vale anche se chi le scrive prova apposta a ingannare il filtro,
perché il paragone è sempre con il miglior filtro fisso su quelle stesse email.

Se poi le email vengono tutte dalla stessa urna, c'è un regalo in più.
Fai la media dei filtri usati giorno per giorno, regola per regola (il peso
medio che ogni parola ha avuto lungo l'anno), e il filtro che ne esce va bene su
email nuove quasi quanto il miglior filtro fisso, che con un'urna fissa è il
migliore in assoluto; e il mucchio intero non l'hai mai dovuto tenere da parte.
Si prende la media e non l'ultimo filtro, perché il rimpianto è una somma su
tutti i giorni, e la garanzia sulla somma passa alla media dei filtri, non
all'ultimo.

Il punto di rottura sta nel paragone. Il rimpianto si misura contro il miglior
filtro *fisso*, e se a metà anno gli spammer cambiano trucco nessun filtro
fisso va bene per tutto l'anno: un rimpianto piccolo, allora, promette poco.
Peggio, le correzioni che si accorciano rendono il filtro sempre più lento.
Dopo diecimila email ogni correzione vale un centesimo della prima, e il trucco
nuovo trova un filtro testardo. Con correzioni di misura fissa il filtro resta
pronto a seguire, e lo paga oscillando anche quando non cambia niente, perché
ogni email lo strattona della stessa misura, che avesse già ragione o no. Quale
misura scegliere dipende da quanto spesso gli spammer cambiano trucco, che non
si sa in anticipo. C'è anche chi cambia il paragone e confronta il filtro,
invece che con il miglior filtro fisso, con il miglior filtro a cui è permesso
cambiare qualche volta durante l'anno: contro quello il rimpianto torna a
promettere qualcosa.

`````

`````{tab} Superiore

Il quadro è l’*online convex optimization*. A ogni turno $t = 1, \dots, T$
l'algoritmo sceglie $\mathbf{w}_t$ in un insieme convesso $\mathcal{K}$ di
diametro $D$, poi viene rivelata una perdita convessa $\ell_t$ e l'algoritmo
paga $\ell_t(\mathbf{w}_t)$. Sulle $\ell_t$ non si fa nessuna ipotesi
statistica, e può sceglierle un avversario che conosce l'algoritmo. Il
rimpianto misura la perdita in più rispetto al miglior punto fisso col senno di
poi,

$$
R_T = \sum_{t=1}^{T} \ell_t(\mathbf{w}_t)
- \min_{\mathbf{w}\in\mathcal{K}} \sum_{t=1}^{T} \ell_t(\mathbf{w}),
$$

e un algoritmo è *senza rimpianto* (*no-regret*) se $R_T = o(T)$, cioè se il
rimpianto medio $R_T/T$ tende a zero. L'idea di competere con la migliore
strategia fissa col senno di poi viene dai giochi ripetuti
{cite}`hannan1957approximation`.

La **discesa del gradiente online** {cite}`zinkevich2003online` fa un passo di
gradiente sulla perdita appena pagata e proietta su $\mathcal{K}$,

$$
\mathbf{w}_{t+1} = \Pi_{\mathcal{K}}\big(\mathbf{w}_t - \eta_t\,\nabla \ell_t(\mathbf{w}_t)\big),
$$

e con $\|\nabla \ell_t\| \le G$ e $\eta_t = D/(G\sqrt{t})$ garantisce
$R_T \le \tfrac{3}{2}\,GD\sqrt{T}$ {cite}`hazan2016introduction`. L'ordine
$\sqrt{T}$ non si migliora senza altre ipotesi: con perdite lineari scelte a
caso ogni algoritmo subisce un rimpianto atteso $\Omega(GD\sqrt{T})$. Con più
struttura sì: se le $\ell_t$ sono $\alpha$-fortemente convesse, il passo
$\eta_t = 1/(\alpha t)$ porta il rimpianto a
$O\big((G^2/\alpha)\log T\big)$ {cite}`hazan2007logarithmic`.

Quando invece i dati sono i.i.d., il rimpianto diventa una garanzia statistica.
Con una perdita convessa nel primo argomento e a valori in $[0,1]$, il punto
medio $\bar{\mathbf{w}} = \frac{1}{T}\sum_t \mathbf{w}_t$ ha, con probabilità
almeno $1-\delta$, un rischio che non supera la perdita media pagata online più
$\sqrt{2\ln(1/\delta)/T}$ {cite}`cesabianchi2004generalization`, dove il
rischio $L(\mathbf{w}) = \mathbb{E}[\ell(\mathbf{w};\mathbf{x},y)]$ è la
perdita attesa su un esempio nuovo. La perdita media online supera di $R_T/T$
quella del miglior punto fisso sul campione, che a sua volta non supera la
perdita empirica del minimizzatore del rischio, e questa, per la disuguaglianza
di Hoeffding, sta entro $\sqrt{\ln(1/\delta)/(2T)}$ dal suo rischio: messi
insieme, con probabilità almeno $1-2\delta$ il rischio di $\bar{\mathbf{w}}$
resta entro $\min_{\mathbf{w}\in\mathcal{K}} L(\mathbf{w}) + R_T/T +
O\big(\sqrt{\ln(1/\delta)/T}\big)$. È la conversione *online-to-batch*, in cui
il rimpianto medio fa la parte dell'errore di stima.

Il punto di rottura è il confronto con un punto fisso. Se il bersaglio si
muove, il metro giusto è il rimpianto dinamico contro una successione
$\mathbf{u}_1, \dots, \mathbf{u}_T$, di lunghezza di cammino
$P_T = \sum_t \|\mathbf{u}_{t+1} - \mathbf{u}_t\|$; con passo fisso $\eta$ la
discesa online lo tiene sotto
$\frac{7D^2}{4\eta} + \frac{D\,P_T}{\eta} + \frac{\eta\,G^2 T}{2}$
{cite}`zinkevich2003online`. Il terzo termine è il costo del passo fisso, che
resta anche quando niente si muove; il secondo punisce il passo piccolo
quando il bersaglio si sposta, e il passo che bilancia i due,
$\eta \propto \sqrt{(D^2 + D\,P_T)/(G^2 T)}$, dipende da $P_T$, che non si
conosce in anticipo. Nel caso a esperti lo stesso problema ha la risposta di
Herbster e Warmuth, che confrontano con il miglior esperto autorizzato a
cambiare un numero fissato di volte {cite}`herbster1998tracking`.

`````

Il rimpianto torna più avanti in due posti: nell’{doc}`apprendimento per
imitazione </DeepReinforcementLearning/imitazione>`, dove è la proprietà che
l'algoritmo DAgger chiede alla successione dei modelli che addestra, e in
{doc}`Capire è accorciare </AutoSupervisione/capire-e-accorciare>`, nel capitolo
sull'auto-supervisione, dove si misura in bit, come lunghezza di un file
compresso rispetto al miglior compressore possibile.

Il primo esperimento misura il rimpianto su flussi di esempi sempre più lunghi.
Il modello è la regressione logistica, che si corregge dopo ogni esempio con
passo $1/\sqrt{t}$ e riporta i pesi dentro una palla di raggio $5$ se ne
escono, perché la garanzia vuole un insieme limitato. Il passo è del tipo
$c/\sqrt{t}$ con $c = 1$ e non con il $c = D/G$ del teorema di Zinkevich, dove
$D$ è il diametro dell'insieme ammesso per i pesi e $G$ il massimo della norma
del gradiente: con un $c$ qualunque il rimpianto resta di ordine $\sqrt{T}$, e
cambia soltanto la costante davanti. La palla di raggio $5$ è l'insieme convesso
$\mathcal{K} = \{\mathbf{w} : \lVert\mathbf{w}\rVert \le 5\}$ della
formula, i pesi la cui lunghezza non supera $5$. La perdita è la sua
cross-entropy (o *log-loss*), il modo della regressione logistica di contare
quanto ha sbagliato. Il miglior modello fisso col senno di poi si trova
risolvendo la regressione sull'intero flusso (il suo minimo cade dentro la
palla, quindi è anche il migliore in $\mathcal{K}$), e il rimpianto è la
differenza fra le due perdite totali.

```python
import numpy as np

rng = np.random.default_rng(0)
w_vero = np.array([2.0, -1.5, 1.0, 0.5, -0.5])   # la regola che genera le risposte
RAGGIO = 5.0                                      # K: i pesi di norma al più 5

def perdita(w, X, y):
    """La cross-entropy della regressione logistica, esempio per esempio (y vale 0 o 1)."""
    z = X @ w
    return np.logaddexp(0, z) - y * z

def online(X, y):
    """Discesa del gradiente online, passo 1/radice(t): risponde, paga, corregge."""
    w = np.zeros(X.shape[1])
    pagato = 0.0
    for t in range(len(y)):
        z = X[t] @ w
        pagato += np.logaddexp(0, z) - y[t] * z       # prima risponde, poi scopre y
        w -= (1 / np.sqrt(t + 1)) * (1 / (1 + np.exp(-z)) - y[t]) * X[t]
        if np.linalg.norm(w) > RAGGIO:                # la proiezione su K
            w *= RAGGIO / np.linalg.norm(w)
    return pagato

def col_senno_di_poi(X, y, passi=25):
    """La perdita del miglior w fisso sull'intero flusso (metodo di Newton)."""
    w = np.zeros(X.shape[1])
    for _ in range(passi):
        p = 1 / (1 + np.exp(-X @ w))
        w -= np.linalg.solve((X * (p * (1 - p))[:, None]).T @ X, X.T @ (p - y))
    return perdita(w, X, y).sum()

for T in (1_000, 10_000, 100_000):
    X = rng.uniform(-1, 1, size=(T, 5))
    y = (rng.random(T) < 1 / (1 + np.exp(-X @ w_vero))).astype(float)
    R = online(X, y) - col_senno_di_poi(X, y)
    print(f"T = {T:>6}: rimpianto {R:5.1f}   per esempio {R / T:.4f}"
          f"   diviso la radice di T {R / np.sqrt(T):.2f}")
```

```text
T =   1000: rimpianto  20.0   per esempio 0.0200   diviso la radice di T 0.63
T =  10000: rimpianto  27.6   per esempio 0.0028   diviso la radice di T 0.28
T = 100000: rimpianto  63.2   per esempio 0.0006   diviso la radice di T 0.20
```

Il rimpianto cresce, da $20$ a $63$, mentre gli esempi crescono di cento volte,
e diviso per il numero degli esempi scende da due centesimi a sei
decimillesimi. Anche diviso per $\sqrt{T}$ scende. Il $\sqrt{T}$ della garanzia
è il caso peggiore, quello di un avversario, e su esempi estratti da una
distribuzione fissa il rimpianto può crescere più piano, come fa qui; ma nessuna
delle due garanzie sul rimpianto lo promette per questo algoritmo: quella
logaritmica chiede perdite fortemente convesse dappertutto, e la cross-entropy
non lo è. Tre flussi estratti una volta sola, poi, non bastano a dire con quale
legge cresca.

Il secondo esperimento mette alla prova il punto di rottura. Venti flussi di
ventimila esempi, con la regola che a metà cambia; quattro allievi: uno
addestrato sui primi duemila esempi e poi lasciato fermo, e tre che imparano
online con passi diversi. Per ciascuno si misura la perdita in più rispetto a
chi conosce la regola vera, prima del cambio, subito dopo e alla fine.

```python
import numpy as np

rng = np.random.default_rng(0)
FLUSSI, T, CAMBIO = 20, 20_000, 10_000
w_prima = np.array([2.0, -1.5, 1.0, 0.5, -0.5])
w_dopo = np.array([-1.0, -1.5, 2.0, 0.5, 1.5])    # a metà flusso la regola cambia
RAGGIO = 5.0                                       # K: i pesi di norma al più 5

X = rng.uniform(-1, 1, size=(FLUSSI, T, 5))
Z = np.where(np.arange(T) < CAMBIO, X @ w_prima, X @ w_dopo)   # i punteggi veri
y = (rng.random((FLUSSI, T)) < 1 / (1 + np.exp(-Z))).astype(float)
perdita = lambda z, y: np.logaddexp(0, z) - y * z
minima = perdita(Z, y)          # quello che paga chi conosce la regola, istante per istante

def online(passo):
    """Venti flussi insieme, uno per riga: la perdita pagata a ogni esempio."""
    w = np.zeros((FLUSSI, 5))
    pagata = np.empty((FLUSSI, T))
    for t in range(T):
        z = (X[:, t] * w).sum(axis=1)
        pagata[:, t] = perdita(z, y[:, t])            # prima risponde, poi scopre y
        w -= passo(t + 1) * (1 / (1 + np.exp(-z)) - y[:, t])[:, None] * X[:, t]
        norme = np.sqrt((w * w).sum(axis=1, keepdims=True))
        w *= np.minimum(1.0, RAGGIO / norme)          # la proiezione su K
    return pagata

def fermo():
    """Addestrato una volta sui primi 2000 esempi di ogni flusso, e mai più toccato."""
    pagata = np.empty((FLUSSI, T))
    for f in range(FLUSSI):
        A, b, w = X[f, :2000], y[f, :2000], np.zeros(5)
        for _ in range(25):                           # metodo di Newton
            p = 1 / (1 + np.exp(-A @ w))
            w -= np.linalg.solve((A * (p * (1 - p))[:, None]).T @ A, A.T @ (p - b))
        pagata[f] = perdita(X[f] @ w, y[f])
    return pagata

finestre = {"prima del cambio": slice(5_000, CAMBIO),
            "i 1000 dopo": slice(CAMBIO, CAMBIO + 1_000),
            "gli ultimi 2000": slice(T - 2_000, T)}
print(" " * 22 + "".join(f"{nome:>18}" for nome in finestre))
for nome, pagata in [("fermo", fermo()),
                     ("passo 1/radice(t)", online(lambda t: 1 / np.sqrt(t))),
                     ("passo fisso 0,05", online(lambda t: 0.05)),
                     ("passo fisso 0,2", online(lambda t: 0.2))]:
    in_piu = pagata - minima      # la perdita in più rispetto a chi conosce la regola
    print(f"{nome:22}" + "".join(f"{in_piu[:, s].mean():18.4f}" for s in finestre.values()))
```

```text
                        prima del cambio       i 1000 dopo   gli ultimi 2000
fermo                             0.0012            0.3884            0.3894
passo 1/radice(t)                 0.0008            0.2246            0.0010
passo fisso 0,05                  0.0033            0.0678            0.0031
passo fisso 0,2                   0.0138            0.0303            0.0133
```

Il modello fermo è buono finché il mondo resta quello dei suoi duemila esempi, e
dopo il cambio paga $0{,}39$ in più a ogni esempio, per sempre. Fra gli allievi
online l'ordine si rovescia da una colonna all'altra. Nei periodi tranquilli il
passo che si accorcia è il più preciso, con $0{,}0008$ contro i $0{,}0138$ del
passo fisso più lungo; subito dopo il cambio è il più lento, con $0{,}2246$
contro $0{,}0303$, più di sette volte tanto. Il passo fisso più corto sta in
mezzo in tutte le colonne, un compromesso fra prontezza e precisione, che
nessun passo fisso può sciogliere. La figura segue due degli allievi su un solo
flusso, e invece della perdita misura quanto i loro pesi distano da quelli della
regola vera ({numref}`fig-bersaglio-che-si-sposta`).

```{figure} ../figures/bersaglio-che-si-sposta.svg
:name: fig-bersaglio-che-si-sposta
:alt: "Animazione: due curve avanzano da sinistra a destra, la distanza fra i pesi di due modelli e quelli della regola vera, esempio dopo esempio. La curva teal, col passo che si accorcia, scende presto e resta bassa e liscia; la curva terracotta, col passo fisso, scende altrettanto presto ma resta più alta e tremolante. A metà, dove una linea verticale segna il cambio della regola, tutte e due saltano in alto, perché la regola vera si è spostata: la terracotta torna giù in poche centinaia di esempi, la teal ridiscende lentamente per migliaia. Alla fine la teal è di nuovo la più bassa."
:width: 92%

La distanza fra i pesi di due allievi online e quelli della regola vera, su un
flusso come quelli dell'esperimento. Col passo che si accorcia la distanza
scende e resta bassa, ma quando a metà la regola cambia ci mette migliaia di
esempi a riavvicinarsi; col passo fisso da $0{,}2$ si riavvicina in poche
centinaia, e in cambio non smette mai di oscillare.
```

## Quando è il modello a cambiare i dati

C'è un ultimo caso, il più sottile: quello in cui i dati cambiano *a causa* del
modello. Gli autori della "parabola" lo avevano notato già in Google Flu
Trends: aggiornando il motore di ricerca (i termini di ricerca suggeriti, le
diagnosi proposte per i sintomi), era Google a cambiare i dati che il suo stesso
modello leggeva {cite}`lazer2014parable`.

Nei sistemi moderni questo **circuito di retroazione** (*feedback loop*) è
frequente. Un sistema di raccomandazione mostra i contenuti che prevede
piaceranno; l'utente sceglie tra *quelli*, e i clic raccolti confermano al
modello che aveva ragione, qualunque cosa avesse mostrato. Un modello di
credito nega il prestito a chi giudica rischioso. Di quelle persone non sapremo
mai se avrebbero restituito i soldi, e i dati futuri conterranno solo le storie
di chi il prestito l'ha avuto. In entrambi i casi il modello non osserva più il
mondo: osserva le conseguenze delle proprie decisioni. Sui propri numeri può
perfino sembrare sempre più bravo, mentre restringe il mondo alle sue decisioni
passate. Chi guarda un video di cucina ne riceve altri dieci e non scoprirà mai
la musica.

Il riaddestramento periodico qui non basta, perché ogni riaddestramento usa
dati che il modello precedente ha contribuito a produrre, e resta da chiedersi
se questa rincorsa si ferma.

`````{tab} Elementare

Un sistema che consiglia video viene riaddestrato ogni settimana sui clic della
settimana prima. Quei clic, però, li ha prodotti lui, e gli utenti hanno scelto
fra i video che lui aveva deciso di mostrare. Così ogni riaddestramento insegue un
bersaglio che il modello precedente ha spostato.

Possono succedere due cose. Se i consigli spostano i gusti di poco, meno di
quanto basterebbe a cambiare la lezione che il modello ne ricava (l'utente
guarda qualche video di cucina in più, ma resta quello che era), le rincorse si
accorciano settimana dopo settimana, e il modello si ferma in un punto in cui,
riaddestrato sui clic che lui stesso produce, non cambia più, in un equilibrio.
Se invece i consigli spostano molto i gusti, ogni settimana il modello trova un
pubblico diverso da quello su cui si è preparato, e i riaddestramenti possono
girare senza fermarsi mai.

Anche quando ci si ferma, l'equilibrio può non essere il migliore possibile,
perché il modello e i gusti che ha contribuito a creare si danno ragione a
vicenda. Il video di cucina che non porterà mai alla musica è un equilibrio di
questo tipo.

`````

`````{tab} Superiore

Il quadro formale è la *performative prediction*
{cite}`perdomo2020performative`. Il modello di parametri $\theta$ induce la
distribuzione dei dati futuri, $\mathcal{D}(\theta)$, e il rischio che conta è
quello performativo, $\mathrm{PR}(\theta) = \mathbb{E}_{z \sim
\mathcal{D}(\theta)}\,\ell(z;\theta)$, con $z = (\mathbf{x}, y)$ un esempio.
Riaddestrare a intervalli è l'iterazione

$$
\theta_{t+1} = \arg\min_\theta\,
\mathbb{E}_{z\sim\mathcal{D}(\theta_t)}\,\ell(z;\theta).
$$

Se la perdita è $\gamma$-fortemente convessa in $\theta$ e $\beta$-liscia (in
$\theta$ e nei dati insieme), e il modello sposta i dati poco, l'iterazione
converge in modo lineare a un unico punto *performativamente stabile*: un
$\theta$ che resta ottimo sui dati che esso stesso produce. La condizione di
vicinanza è $W_1\big(\mathcal{D}(\theta), \mathcal{D}(\theta')\big) \le
\varepsilon\,\lVert\theta - \theta'\rVert$, con $W_1$ la distanza di
Wasserstein-1 fra distribuzioni e $\varepsilon < \gamma/\beta$. Sono ipotesi
strette: se una cade, il riaddestramento può non convergere affatto. Un punto
stabile non è necessariamente ottimo: il punto fisso del riaddestramento in
generale non minimizza $\mathrm{PR}$.

`````

Il {doc}`capitolo sui sistemi di raccomandazione
</SistemiRaccomandazione/overview>` riprende il circuito di retroazione come
struttura stessa del problema, e non come effetto collaterale.

`````{tab} Elementare

```{admonition} Da ricordare
:class: important
- L'apprendimento supervisionato poggia su un'ipotesi tacita: che i dati di ieri
  e quelli di domani vengano dalla stessa urna, e nessuna legge assicura che il
  mondo la rispetti.
- Tre modi in cui l'urna cambia, e vanno distinti perché chiedono rimedi
  diversi: cambiano le domande (l'app che riconosce le piante, addestrata
  d'estate e usata d'inverno: le foto sono altre, ma un abete resta un abete);
  cambiano le proporzioni delle risposte (la malattia rara che diventa
  un'epidemia: i sintomi sono gli stessi, la loro frequenza no); cambia la
  regola (le parole che nel 2005 gridavano truffa e oggi arrivano da un
  negozio serio). L'ultimo è il peggiore, perché nessuna quantità di dati
  vecchi può insegnare una regola nuova.
- La validazione classica non protegge, perché studio, prove ed esame sono
  tre ritagli della stessa vecchia fotografia: è guidare guardando lo
  specchietto retrovisore. Se i dati hanno una data, la prova onesta è
  addestrare sul passato e verificare sul futuro.
- I rimedi più usati sono abitudini prima che formule: sorvegliare il modello
  mentre lavora (gli ingressi somigliano a quelli di ieri? le risposte sono
  cambiate di colpo?), riaddestrarlo ogni tanto su dati recenti, giudicarlo su
  dati freschi, e lasciargli la possibilità di dire «non lo so».
- Un modello può anche imparare mentre lavora, un esempio alla volta. Lo si
  giudica col rimpianto, quanto sbaglia in più del miglior modello fisso
  scelto col senno di poi, e la garanzia vale qualunque cosa facciano i dati;
  ma se il mondo cambia a metà strada un rimpianto piccolo promette poco,
  correzioni che si accorciano lo rendono lento davanti al cambio, e
  correzioni di misura fissa lo fanno oscillare sempre.
- Attenzione a quando è il modello stesso a fabbricare i dati di domani: se
  mostra solo certi contenuti, vedrà solo clic su quelli; se nega il prestito,
  non saprà mai chi avrebbe restituito. Da lì in avanti guarda le conseguenze
  delle proprie decisioni invece del mondo, e riaddestrarlo spesso può non
  bastare: la rincorsa si ferma, a certe condizioni, quando il modello sposta i
  dati di poco, e anche allora l'equilibrio non è per forza il migliore.
- Google Flu Trends è la parabola da ricordare: un modello che va bene sul
  passato può sbagliare per due ragioni, perché ha imparato l'inverno invece
  dell'influenza e perché il mondo in cui cerca cambia sotto di lui, e gli esami
  fatti sul passato non se ne accorgono.
```

`````

`````{tab} Superiore

```{admonition} Da ricordare
:class: important
- L'apprendimento supervisionato poggia su un'ipotesi tacita: dati di
  addestramento e dati reali vengono dalla stessa distribuzione (i.i.d.).
- Tre famiglie di dataset shift, distinte da quale condizionata resta ferma:
  *covariate shift* (cambia $P(\mathbf{x})$, resta $P(y \mid \mathbf{x})$),
  *label shift* (cambia $P(y)$, resta $P(\mathbf{x} \mid y)$), *concept shift*
  (cambia $P(y \mid \mathbf{x})$, cioè la regola stessa). La terna non esaurisce
  i casi, e *concept drift* in letteratura indica spesso qualunque cambiamento
  nel tempo.
- La validazione classica non protegge: validation e test sono ritagli dello
  stesso passato, e senza un'ipotesi sulla distanza fra le distribuzioni non
  vincolano l'errore in produzione. Con dati temporali, meglio lo split
  temporale a origine mobile.
- Rimedi onesti: monitoraggio in produzione, retraining periodico, validazione
  su dati freschi; l’*importance weighting* corregge il covariate shift (è un
  cambio di misura) ma solo se il supporto di produzione sta dentro quello di
  addestramento e il rapporto di densità si sa stimare, e con pesi appiattiti
  per tenere bassa la varianza.
- Apprendimento online: su perdite convesse, la discesa del gradiente online ha
  rimpianto $O(GD\sqrt{T})$ senza ipotesi statistiche sui dati, e con dati
  i.i.d. la media degli iterati eredita la garanzia (*online-to-batch*). Contro
  un bersaglio che si muove il passo decrescente è lento e il passo fisso paga
  $\eta G^2 T/2$ anche da fermo: il passo giusto dipende da quanto si muove il
  mondo.
- Attenzione ai feedback loop: quando le decisioni del modello generano i dati
  futuri (raccomandazioni, credito), il modello smette di osservare il mondo e
  inizia a osservare se stesso. Nel quadro della *performative prediction* il
  riaddestramento ripetuto converge a un punto stabile se la perdita è
  fortemente convessa e liscia e $\varepsilon < \gamma/\beta$, e stabile non
  vuol dire ottimo.
- Google Flu Trends resta la parabola di riferimento: un modello che va bene sul
  passato può invecchiare senza avvisare, e il suo adattamento iniziale ai dati era
  in parte spurio {cite}`lazer2014parable`.
```

`````

Un modello non avverte quando lo si interroga lontano dai suoi esempi, e il
covariate shift fa accadere proprio questo. I
{doc}`processi gaussiani <processi-gaussiani>` accompagnano invece ogni
previsione con la propria incertezza, larga dove gli esempi mancano e stretta
dove abbondano. Sul cambio di regola non avvertiranno nemmeno loro, perché la
loro incertezza guarda dove stanno gli esempi, non che cosa dicono.
