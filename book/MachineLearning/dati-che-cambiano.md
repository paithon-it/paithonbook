# Quando i dati cambiano

Nel 2008 Google lanciò un servizio dal nome ambizioso: **Google Flu Trends**.
L'idea, presentata anche sulle pagine di *Nature*, era elegante: chi si sente
febbricitante corre a cercare in rete "sintomi influenza" o "febbre alta
rimedi". Contando queste ricerche si poteva stimare la diffusione
dell'influenza negli Stati Uniti in tempo quasi reale, con una o due settimane
di anticipo sui CDC, l'ente federale che raccoglie i dati dai medici. Per
qualche anno funzionò, e il servizio divenne il manifesto di una promessa che
in quegli anni si sentiva dappertutto: adesso che ogni gesto lascia una traccia
in rete, perché fare sondaggi lenti e costosi, quando i dati arrivano da soli?

Poi arrivò l'inverno 2012–2013. A febbraio 2013 Google Flu Trends stimava una
quota di visite mediche per sintomi influenzali **più che doppia** rispetto a
quella registrata dai CDC. E non era un incidente isolato: tra l'agosto 2011 e
il settembre 2013 il servizio aveva sovrastimato l'influenza in **100
settimane su 108**. Nel 2014 un gruppo di ricercatori firmò su *Science*
l'autopsia del progetto, con un titolo diventato proverbiale: *The Parable of
Google Flu* (la "parabola", nel senso del racconto che ammonisce
{cite}`lazer2014parable`). Nell'agosto 2015 Google chiuse il servizio.

Che cosa era andato storto? Il modello non si era rotto: era **invecchiato**.
Il modo di cercare in rete era cambiato, e Google stessa aggiornava il motore:
dal giugno 2011 cominciò a **proporre altri termini da cercare** (a chi
chiedeva dell'influenza suggeriva di cercarne le cure), dal febbraio 2012 a
**rispondere alle ricerche sui sintomi con le diagnosi possibili** (chi cercava
«febbre» o «tosse» si vedeva proporre l'influenza). Erano due spinte verso
l'influenza che arrivavano anche a chi stava benissimo, e la catena si chiude
da sé: più suggerimenti, più ricerche sull'influenza; e siccome il modello
contava proprio quelle ricerche, più malati stimati. Intanto i giornali
parlavano di epidemia e la gente cercava per curiosità, non per febbre. Il
modello, tarato sul mondo del 2008, continuava a leggere il presente con gli
occhiali di allora. Questa sezione
parla esattamente di questo: che cosa succede quando i dati che un modello
incontra non somigliano più a quelli su cui è stato addestrato.

## L'ipotesi nascosta di tutto il libro

C'è un'assunzione che regge, in silenzio, ogni pagina scritta finora: che i
dati di addestramento e i dati che il modello incontrerà dopo siano fatti
*della stessa pasta* (pescati, per così dire, dalla stessa urna). Ha un nome
tecnico, **ipotesi i.i.d.**, che è la sigla di «indipendenti e identicamente
distribuiti»: ogni esempio pescato senza che gli altri lo condizionino
(*indipendenti*) e tutti dalla stessa urna (*identicamente distribuiti*).
Finché vale, tutto l'impianto che abbiamo
costruito funziona. Il problema è che nessuno ha firmato un contratto con il
mondo perché continui a valere.

`````{tab} Elementare

Un sondaggio elettorale intervista mille persone e prevede il voto di milioni.
Funziona per una sola ragione: le mille persone sono scelte in modo da
*somigliare* ai milioni. Tutto quello che abbiamo visto finora (training,
validation, test) è, in fondo, un sondaggio: peschiamo esempi da un'urna e
contiamo, fidandoci che l'urna di domani sia la stessa di oggi.

Se in un'urna ben mescolata il 30% delle palline è rosso, in una manciata di
cento ne troverai più o meno trenta rosse: la manciata "parla" per l'urna
intera. Ma se stanotte qualcuno sostituisce metà delle palline, la manciata
pescata ieri non dice più nulla sull'urna di oggi. È quello che è successo a
Google Flu Trends: l'urna (il modo in cui la gente usa un motore di ricerca)
era cambiata, e nessuno aveva avvisato il modello.

Una parola su questa urna, perché nel resto della sezione compare col suo nome
tecnico. La **distribuzione** di una cosa è, semplicemente, il resoconto di
quanto spesso ciascun valore capita: le palline rosse al 30% e le altre al 70%
*sono* la distribuzione dei colori in quell'urna. Quando più avanti si legge
«la distribuzione degli input è cambiata», vuol dire esattamente «l'urna non è
più quella».

`````

`````{tab} Superiore

Formalmente, assumiamo che le coppie $(\mathbf{x}^{(i)}, y^{(i)})$ del
training e quelle che il modello vedrà in produzione siano estratte in modo
**indipendente e identicamente distribuito** (i.i.d.) da un'unica
distribuzione congiunta $P(X, y)$. Sotto questa ipotesi l'errore misurato sul
campione converge, per la legge dei grandi numeri, all'errore atteso di
qualunque modello *fissato in anticipo*. Perché la stessa garanzia valga per
il modello scelto minimizzando sui dati serve di più: che la classe di ipotesi
abbia capacità limitata (in gergo, la convergenza *uniforme* della teoria
dell'apprendimento statistico). È lo stesso controllo della complessità
incontrato con l'overfitting, che di quella garanzia è appunto il
controesempio: quando la capacità non è limitata, l'errore sul campione può
essere azzerato senza che questo dica più nulla sull'errore atteso. Quando
invece valgono entrambe le condizioni, campionamento i.i.d. e capacità
limitata, minimizzare la loss empirica è una buona approssimazione del
minimizzare il rischio vero.

Quando le due distribuzioni divergono,
$P_{\text{train}}(X, y) \neq P_{\text{test}}(X, y)$, si parla di **dataset
shift** {cite}`quinonero2009dataset`, e le garanzie cadono: il minimo della
loss sotto $P_{\text{train}}$ non è più, in generale, un buon punto sotto
$P_{\text{test}}$. Per classificare i modi in cui la congiunta può cambiare
conviene fattorizzarla:

$$
P(X, y) = P(y \mid X)\, P(X) = P(X \mid y)\, P(y),
$$

dove $P(X)$ è la distribuzione degli input, $P(y)$ quella delle etichette e
$P(y \mid X)$ la relazione input–etichetta che il modello cerca di apprendere.
Ognuno dei tre fattori può cambiare per conto suo, ed è la tassonomia del
prossimo paragrafo.

`````

## Tre modi in cui il mondo cambia

Non tutti i cambiamenti sono uguali. Chi studia il fenomeno ne distingue tre
famiglie
{cite}`quinonero2009dataset`, e vale la pena impararle con esempi quotidiani,
perché la diagnosi giusta suggerisce il rimedio giusto.
{numref}`fig-distribution-shift` mostra il caso più semplice da visualizzare:
i dati nuovi arrivano in una zona che l'addestramento ha quasi ignorato. Il
grafico va letto in un modo nuovo rispetto a quelli visti finora, dove i punti
erano esempi: qui sull'asse orizzontale c'è il valore di una caratteristica (i
metri quadri, l'età, il numero di ricerche) e
sulla verticale **quanto spesso** quel valore capita, così che dove la curva è
alta ci sono tanti esempi e dove è schiacciata quasi nessuno. Due curve
sfalsate vogliono dire che i valori frequenti ieri non sono quelli frequenti
oggi.

```{figure} ../figures/distribution-shift.svg
:name: fig-distribution-shift
:alt: Due curve a campana sfalsate lungo l'asse degli input, una per i dati di addestramento e una spostata a destra per i dati in produzione; una linea tratteggiata indica il punto dove il modello è tarato, lontano dal grosso dei dati nuovi.
:width: 85%

Due urne a confronto: la curva dei dati di addestramento e quella dei dati che
il modello incontra una volta al lavoro. La linea tratteggiata segna il valore
attorno a cui il modello è tarato, cioè dove i dati di ieri si addensavano; i
dati di oggi cadono in gran parte altrove, dove di esempi non ne ha quasi mai
visti.
```

`````{tab} Elementare

**Cambiano le domande** (*covariate shift*). Un'app che riconosce le piante,
addestrata su foto scattate d'estate, viene usata d'inverno: luce bassa, rami
spogli, neve sullo sfondo. Le foto che arrivano sono diverse da quelle viste a
lezione, ma attenzione: un abete resta un abete. La *regola* che collega foto
e risposta non è cambiata; è cambiato il tipo di foto che arriva.

**Cambiano le proporzioni delle risposte** (*label shift*). Un modello aiuta a
diagnosticare una malattia che, quando è stato addestrato, colpiva una persona
su mille. Arriva un'epidemia e diventa una su cinquanta. I sintomi della
malattia sono identici a prima: cambia solo *quanto spesso* la risposta giusta
è «positivo».

Se i sintomi sono gli stessi, perché il modello dovrebbe sbagliare? Perché la
rarità, di nascosto, è entrata nel suo giudizio. Un modello che ha imparato su
un mondo in cui i malati sono uno su mille ha anche imparato che dire «sano»
paga: ci prende 999 volte su 1000, e per convincerlo del contrario servono
sintomi molto chiari. Davanti a un caso dubbio resterà prudente e dirà «sano»,
il che era la scommessa giusta ieri ed è quella sbagliata oggi, che i malati
sono venti volte tanti.

**Cambia la regola stessa** (*concept shift*, o *drift*). Che cos'è lo spam?
Le stesse parole ("offerta", "clicca qui", "solo per oggi") che nel 2005
gridavano truffa, oggi arrivano da negozi legittimi; e intanto i truffatori
hanno imparato a scrivere come una banca. Qui non cambiano solo le domande:
cambia la *risposta giusta alla stessa domanda*. È il caso più insidioso,
perché nessuna quantità di dati vecchi può insegnare una regola nuova.

`````

`````{tab} Superiore

Con la fattorizzazione $P(X,y) = P(y \mid X)\,P(X) = P(X \mid y)\,P(y)$, le
tre famiglie canoniche sono {cite}`quinonero2009dataset`:

- **Covariate shift**: cambia $P(X)$, resta invariata $P(y \mid X)$. Le foto
  invernali hanno una distribuzione diversa da quelle estive, ma la mappa
  immagine $\to$ specie è la stessa. È il caso della figura: il modello è
  accurato dove $p_{\text{train}}(x)$ è densa, e viene interrogato dove è
  quasi nulla (di fatto, un'**estrapolazione**).
- **Label shift** (o *prior probability shift*): cambia $P(y)$, resta
  invariata $P(X \mid y)$. La malattia si presenta come prima, ma la sua
  prevalenza è diversa. Un classificatore bayesiano tarato sul *prior* vecchio
  produce probabilità a posteriori sistematicamente distorte.
- **Concept shift** (o *concept drift*): cambia $P(y \mid X)$. La relazione
  input–etichetta stessa si sposta, gradualmente (i gusti musicali) o di colpo
  (una nuova legge cambia cosa è "transazione sospetta").

Nella pratica le tre forme arrivano mescolate, e distinguere quale domini a
partire dai soli dati è un problema difficile: spesso mal posto, se le
etichette nuove tardano ad arrivare.

`````

## Perché la validazione classica non protegge

Obiezione naturale: "ma noi le pagine sulla validazione le abbiamo studiate!
Validation set, test chiuso nel cassetto, cross-validation…". Tutto vero, e
tutto necessario. Ma c'è un punto cieco: **il validation set viene dallo
stesso passato del training set**.

`````{tab} Elementare

Train, validation e test sono tre ritagli della *stessa fotografia*. Se la
fotografia invecchia, invecchiano tutti e tre insieme, e i loro voti restano
alti proprio mentre il modello, nel mondo reale, comincia a sbagliare. È come
guidare guardando lo specchietto retrovisore: ti dice benissimo la strada già
percorsa, ma non la curva che sta arrivando.

Google Flu Trends, sui propri dati di validazione, era eccellente: era stato
validato sul passato, e sul passato funzionava davvero. Il voto d'esame era
onesto; era la domanda a essere sbagliata. La validazione risponde a "quanto
sbaglierò su dati *come questi*?", non a "quanto sbaglierò *domani*?".

C'è però un modo di rendere onesta anche la domanda, e vale ogni volta che i
dati hanno una data sopra: invece di tagliarli a caso, si taglia **nel tempo**.
Il modello studia su gennaio-ottobre e viene interrogato su novembre-dicembre,
che al momento dell'addestramento erano il futuro. Se già lì peggiora, in
mezzo al mondo vero peggiorerà di sicuro; se non peggiora non è una garanzia,
ma è un indizio molto migliore di un rimescolamento che gli lascia sbirciare
il domani.

`````

`````{tab} Superiore

La stima di validazione approssima
$\mathbb{E}_{(X,y)\sim P_{\text{train}}}\!\left[\mathcal{L}\big(f_\theta(X), y\big)\right]$:
un valore atteso **sotto la distribuzione di addestramento**. Se la
distribuzione operativa è un'altra, questo numero non vincola in alcun modo
l'errore reale: può restare ottimo mentre l'errore sotto $P_{\text{test}}$
diverge.

Con dati temporali c'è di peggio: la cross-validation rimescolata distrugge
l'ordine cronologico e lascia che il modello "veda il futuro" dei fold di
validazione (una forma di *leakage* temporale), gonfiando le stime. Lo stress
test più onesto è lo **split temporale**: addestrare sul passato e validare
sul futuro relativo (ad esempio, addestrare su gennaio–ottobre e validare su
novembre–dicembre). Se le prestazioni degradano già lì, degraderanno anche in
produzione; se non degradano, non è comunque una garanzia, solo un indizio
migliore.

`````

## Rimedi onesti

Non esiste la bacchetta magica: nessun algoritmo rende un modello immune al
tempo. I rimedi più efficaci non sono matematici ma *organizzativi*, e sono
tre:

1. **Sorvegliare il modello mentre lavora.** «In produzione» vuol dire proprio
   questo: non più le prove in laboratorio, ma il modello acceso sul serio, con
   utenti veri e dati che arrivano ogni giorno. Un modello in produzione va
   trattato come un impianto, non come un quadro appeso, e le cose da tenere
   d'occhio sono tre: come sono fatte le domande che arrivano, come sono fatte
   le risposte che dà, e, appena si scopre qual era la risposta giusta, quanto
   ha sbagliato davvero.
2. **Riaddestrare a intervalli regolari** (in gergo *retraining*) su dati
   recenti, così che la "fotografia" non invecchi troppo.
3. **Giudicarlo su dati freschi**: su un campione *nuovo*, raccolto dopo
   l'addestramento, non sull'ennesimo ritaglio del mucchio di esempi di
   partenza.

`````{tab} Elementare

Un modello in produzione è come la bilancia del mercato: per legge va
**ritarata periodicamente**, perché con l'uso e il tempo si starano tutte, ed
è meglio accorgersene prima del cliente. In pratica si tengono d'occhio tre
cose. Primo: gli ingressi somigliano ancora a quelli di ieri? Se un filtro
antispam riceveva email lunghe in media 80 parole e ora ne arrivano da 200, è
un campanello. Secondo: le uscite. Se ieri segnalava come spam il 20% dei
messaggi e oggi il 45%, qualcosa è cambiato: nel mondo o nel modello. Terzo:
appena si scopre la risposta giusta (l'utente ha ripescato l'email dal
cestino?), confrontarla con la predizione. E un buon sistema dovrebbe anche
saper **passare la mano**: davanti a un caso che non somiglia a nulla di già
visto, meglio dire "non lo so, decida un umano" che sparare una risposta
sicura e sbagliata.

`````

`````{tab} Superiore

Sul versante algoritmico, il rimedio classico per il *covariate shift* puro è
l'**importance weighting**: ripesare la loss di training così che gli esempi
frequenti in produzione ma rari in addestramento contino di più,

$$
\mathcal{L}_w(\theta) = \frac{1}{m} \sum_{i=1}^{m}
w\big(\mathbf{x}^{(i)}\big)\,
\mathcal{L}\big(f_\theta(\mathbf{x}^{(i)}),\, y^{(i)}\big),
\qquad
w(\mathbf{x}) = \frac{p_{\text{test}}(\mathbf{x})}{p_{\text{train}}(\mathbf{x})},
$$

dove $w(\mathbf{x})$ è il rapporto tra la densità degli input in produzione e
quella in addestramento e $m$ è il numero di esempi. In teoria, minimizzare
$\mathcal{L}_w$ equivale a minimizzare l'errore atteso sotto
$P_{\text{test}}$. In pratica i limiti sono seri: vale solo se $P(y \mid X)$
non cambia; richiede che i supporti si sovrappongano, dove
$p_{\text{train}}(\mathbf{x}) = 0$ ma $p_{\text{test}}(\mathbf{x}) > 0$ nessun
peso può inventare esempi mai raccolti; e stimare il rapporto di densità in alta
dimensione è difficile, con pesi enormi su pochi esempi che fanno esplodere la
varianza. Correzioni analoghe esistono per il *label shift*, ripesando per
classi. Complementare a tutto questo è l'**out-of-distribution detection**:
riconoscere gli input troppo lontani dalla distribuzione di addestramento e,
invece di predire con finta sicurezza, astenersi o segnalare; un problema
particolarmente delicato per le reti profonde, che su input fuori
distribuzione tendono a essere *confidenti e sbagliate* insieme.

`````

C'è un trucco pratico per accorgersene, e usa solo strumenti che già
conosciamo. Si mescolano i dati di addestramento con quelli raccolti mentre il
modello lavorava, si cancella qualsiasi altra etichetta e si addestra un
secondo modello a rispondere a una domanda sola: **questo esempio viene da ieri
o da oggi?** Se ci riesce, ieri e oggi sono distinguibili, cioè la deriva c'è; e
il suo punteggio dice pure quanto è grossa.

Il punteggio giusto da guardare qui è l'AUC della sezione sulle metriche, che
per un detective come questo si legge benissimo: $0{,}5$ vuol dire che sta
tirando a indovinare, cioè che i due mucchi gli sembrano identici, e $1$ vuol
dire che li separa senza sbagliare un colpo.

Attenzione però a leggere il silenzio. Un'AUC vicina a $0{,}5$ dice che le due
epoche sono indistinguibili **per lui**, che è una
conclusione più debole di «va tutto bene», per due ragioni. La prima è che
questo detective guarda soltanto le domande in arrivo, non le risposte: del
cambio di regola, dove le domande restano le stesse e a cambiare è la risposta
giusta, non può
accorgersi per costruzione, ed è il caso più insidioso dei tre. La seconda è
che un'AUC vicina a $0{,}5$ può anche voler dire che gli esempi nuovi
sono ancora troppo pochi, o che la deriva sta in un intreccio fra più
caratteristiche che quel detective, preso singolarmente, non coglie: non aver
trovato non è aver dimostrato che non c'è niente. È uno strumento che vale la
pena avere, purché letto per quello
che è: un allarme quando suona, non un certificato quando tace.

```python
import numpy as np
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.model_selection import cross_val_score

# X_train: input di addestramento; X_prod: input raccolti in produzione
X_tutti = np.vstack([X_train, X_prod])
origine = np.hstack([np.zeros(len(X_train)), np.ones(len(X_prod))])

# un "detective" prova a indovinare da quale epoca viene ogni esempio
detective = GradientBoostingClassifier()
auc = cross_val_score(detective, X_tutti, origine, cv=5, scoring="roc_auc")

print(auc.mean())  # ~0.5: nessuno shift rilevabile; verso 1: allarme
```

## Quando è il modello a cambiare i dati

C'è un ultimo caso, il più sottile: quello in cui i dati non cambiano
*nonostante* il modello, ma *a causa sua*. Gli autori della "parabola" lo
avevano notato già in Google Flu Trends: era anche Google, aggiornando il
motore di ricerca (il completamento automatico, le ricerche suggerite) a
cambiare i dati che il suo stesso modello leggeva {cite}`lazer2014parable`.

Nei sistemi moderni questo **circuito di retroazione** (*feedback loop*) è
ovunque. Un sistema di raccomandazione mostra i contenuti che prevede
piaceranno; l'utente sceglie tra *quelli*, e i clic raccolti confermano al
modello che aveva ragione, qualunque cosa avesse mostrato. Un modello di
credito nega il prestito a chi giudica rischioso: di quelle persone non
sapremo mai se avrebbero restituito i soldi, e i dati futuri conterranno solo
le storie di chi il prestito l'ha avuto. In entrambi i casi il modello non
osserva più il mondo: osserva le conseguenze delle proprie decisioni. E sui
propri numeri può perfino sembrare sempre più bravo, mentre in realtà sta
restringendo il mondo a ciò che aveva già deciso: chi guarda un video di cucina
ne riceve altri dieci di cucina e non scoprirà mai la musica, e chi si è visto
negare un prestito non avrà mai modo di dimostrare che l'avrebbe restituito.
Ci torneremo nel
capitolo sui sistemi di raccomandazione, dove il circuito di retroazione non è
un effetto collaterale ma la struttura stessa del problema.

`````{tab} Elementare

```{admonition} Da ricordare
:class: important
- Tutto il libro poggia su un'ipotesi tacita: che i dati di ieri e quelli di
  domani vengano **dalla stessa urna**. Il mondo non ha firmato quel contratto.
- Tre modi in cui l'urna cambia, e vanno distinti perché chiedono rimedi
  diversi: **cambiano le domande** (l'app che riconosce le piante, addestrata
  d'estate e usata d'inverno: le foto sono altre, ma un abete resta un abete);
  **cambiano le proporzioni delle risposte** (la malattia rara che diventa
  un'epidemia: i sintomi sono gli stessi, la loro frequenza no); **cambia la
  regola** (le parole che nel 2005 gridavano truffa e oggi arrivano da un
  negozio serio). L'ultimo è il peggiore, perché nessuna quantità di dati
  vecchi può insegnare una regola nuova.
- La **validazione classica non protegge**, perché studio, prove ed esame sono
  tre ritagli della stessa vecchia fotografia: è guidare guardando lo
  specchietto retrovisore. Se i dati hanno una data, la prova onesta è
  addestrare sul passato e verificare sul futuro.
- I rimedi che funzionano non sono formule ma abitudini: **sorvegliare** il
  modello mentre lavora (gli ingressi somigliano a quelli di ieri? le risposte
  sono cambiate di colpo?), **riaddestrarlo** ogni tanto su dati recenti,
  **giudicarlo su dati freschi**. E dargli il permesso di dire «non lo so».
- Attenzione a quando è il modello stesso a **fabbricare i dati di domani**: se
  mostra solo certi contenuti, vedrà solo clic su quelli; se nega il prestito,
  non saprà mai chi avrebbe restituito. Da lì in avanti non guarda più il
  mondo, guarda le conseguenze delle proprie decisioni.
- **Google Flu Trends** è la parabola da ricordare: un modello eccellente sul
  passato può invecchiare in silenzio, restando bravissimo agli esami che si dà
  da solo.
```

`````

`````{tab} Superiore

```{admonition} Da ricordare
:class: important
- Tutto il libro poggia su un'ipotesi tacita: dati di addestramento e dati
  reali vengono dalla **stessa distribuzione** (i.i.d.). Il mondo non ha
  firmato quel contratto.
- Tre famiglie di **dataset shift**: *covariate shift* (cambia $P(X)$: le
  domande), *label shift* (cambia $P(y)$: le proporzioni delle risposte),
  *concept shift* (cambia $P(y \mid X)$, cioè la regola stessa; la barra
  verticale si legge «dato», quindi $P(y \mid X)$ è la probabilità della
  risposta *dato* l'input).
- La **validazione classica non protegge**: validation e test sono ritagli
  dello stesso passato. Con dati temporali, meglio lo split temporale.
- Rimedi onesti: **monitoraggio in produzione**, **retraining periodico**,
  **validazione su dati freschi**; l'*importance weighting* corregge il
  covariate shift ma solo con supporti sovrapposti e densità stimabili.
- Attenzione ai **feedback loop**: quando le decisioni del modello generano i
  dati futuri (raccomandazioni, credito), il modello smette di osservare il
  mondo e inizia a osservare se stesso.
- Google Flu Trends resta la parabola di riferimento: un modello eccellente
  sul passato può invecchiare in silenzio {cite}`lazer2014parable`.
```

`````
