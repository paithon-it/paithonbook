# Quando i dati cambiano

Nel 2008 Google lanciò un servizio dal nome ambizioso: **Google Flu Trends**.
L'idea, presentata anche sulle pagine di *Nature*, era elegante: chi si sente
febbricitante corre a cercare in rete "sintomi influenza" o "febbre alta
rimedi". Contando queste ricerche si poteva stimare la diffusione
dell'influenza negli Stati Uniti in tempo quasi reale, con una o due settimane
di anticipo sui CDC, l'ente federale che raccoglie i dati dai medici. Per
qualche anno funzionò, e il servizio divenne il manifesto della promessa dei
*big data*: perché fare sondaggi lenti e costosi, quando i dati arrivano da
soli?

Poi arrivò l'inverno 2012–2013. A febbraio 2013 Google Flu Trends stimava una
quota di visite mediche per sintomi influenzali **più che doppia** rispetto a
quella registrata dai CDC. E non era un incidente isolato: tra l'agosto 2011 e il settembre 2013
il servizio aveva sovrastimato l'influenza in **100 settimane su 108**. Nel
2014 un gruppo di ricercatori firmò su *Science* l'autopsia del progetto, con
un titolo diventato proverbiale: *The Parable of Google Flu* — la "parabola",
nel senso del racconto che ammonisce {cite}`lazer2014parable`. Nell'agosto
2015 Google chiuse il servizio.

Che cosa era andato storto? Il modello non si era rotto: era **invecchiato**.
Il modo di cercare in rete era cambiato — Google stessa aggiornava di continuo
il motore, e il completamento automatico suggeriva ricerche sull'influenza
anche a chi stava benissimo; i giornali parlavano di epidemia e la gente
cercava per curiosità, non per febbre. Il modello, tarato sul mondo del 2008,
continuava a leggere il presente con gli occhiali di allora. Questa sezione
parla esattamente di questo: che cosa succede quando i dati che un modello
incontra non somigliano più a quelli su cui è stato addestrato.

## L'ipotesi nascosta di tutto il libro

C'è un'assunzione che regge, in silenzio, ogni pagina scritta finora: che i
dati di addestramento e i dati che il modello incontrerà dopo siano fatti
*della stessa pasta* — pescati, per così dire, dalla stessa urna. Ha un nome
tecnico, **ipotesi i.i.d.**, e finché vale tutto l'impianto che abbiamo
costruito funziona. Il problema è che nessuno ha firmato un contratto con il
mondo perché continui a valere.

`````{tab} Elementare

Un sondaggio elettorale intervista mille persone e prevede il voto di milioni.
Funziona per una sola ragione: le mille persone sono scelte in modo da
*somigliare* ai milioni. Tutto quello che abbiamo visto finora — training,
validation, test — è, in fondo, un sondaggio: peschiamo esempi da un'urna e
contiamo, fidandoci che l'urna di domani sia la stessa di oggi.

Se in un'urna ben mescolata il 30% delle palline è rosso, in una manciata di
cento ne troverai più o meno trenta rosse: la manciata "parla" per l'urna
intera. Ma se stanotte qualcuno sostituisce metà delle palline, la manciata
pescata ieri non dice più nulla sull'urna di oggi. È quello che è successo a
Google Flu Trends: l'urna — il modo in cui la gente usa un motore di ricerca —
era cambiata, e nessuno aveva avvisato il modello.

`````

`````{tab} Superiore

Formalmente, assumiamo che le coppie $(X^{(i)}, y^{(i)})$ del training e
quelle che il modello vedrà in produzione siano estratte in modo
**indipendente e identicamente distribuito** (i.i.d.) da un'unica
distribuzione congiunta $P(X, y)$. Sotto questa ipotesi l'errore misurato sul
campione converge, per la legge dei grandi numeri, all'errore atteso:
minimizzare la loss empirica è una buona approssimazione del minimizzare il
rischio vero.

Quando le due distribuzioni divergono — $P_{\text{train}}(X, y) \neq
P_{\text{test}}(X, y)$ — si parla di **dataset shift**
{cite}`quinonero2009dataset`, e le garanzie cadono: il minimo della loss sotto
$P_{\text{train}}$ non è più, in generale, un buon punto sotto
$P_{\text{test}}$. Per classificare i modi in cui la congiunta può cambiare
conviene fattorizzarla:

$$
P(X, y) = P(y \mid X)\, P(X) = P(X \mid y)\, P(y),
$$

dove $P(X)$ è la distribuzione degli input, $P(y)$ quella delle etichette e
$P(y \mid X)$ la relazione input–etichetta che il modello cerca di
apprendere. Ognuno dei tre fattori può cambiare per conto suo — ed è la
tassonomia del prossimo paragrafo.

`````

## Tre modi in cui il mondo cambia

Non tutti i cambiamenti sono uguali. La letteratura ne distingue tre famiglie
{cite}`quinonero2009dataset`, e vale la pena impararle con esempi quotidiani,
perché la diagnosi giusta suggerisce il rimedio giusto.
{numref}`fig-distribution-shift` mostra il caso più semplice da visualizzare:
i dati nuovi arrivano in una zona che l'addestramento ha quasi ignorato.

```{figure} ../figures/distribution-shift.svg
:name: fig-distribution-shift
:alt: Due curve a campana sfalsate lungo l'asse degli input, una per i dati di addestramento e una spostata a destra per i dati in produzione; una linea tratteggiata indica il punto dove il modello è tarato, lontano dal grosso dei dati nuovi.
:width: 85%

La distribuzione dei dati di addestramento e quella dei dati incontrati in
produzione: il modello è accurato dove i dati di ieri abbondavano, ma i dati
di oggi cadono dove non ha quasi mai visto esempi.
```

`````{tab} Elementare

**Cambiano le domande** (*covariate shift*). Un'app che riconosce le piante,
addestrata su foto scattate d'estate, viene usata d'inverno: luce bassa, rami
spogli, neve sullo sfondo. Le foto che arrivano sono diverse da quelle viste a
lezione — ma attenzione: un abete resta un abete. La *regola* che collega foto
e risposta non è cambiata; è cambiato il tipo di foto che arriva.

**Cambiano le proporzioni delle risposte** (*label shift*). Un modello aiuta a
diagnosticare una malattia che, quando è stato addestrato, colpiva una persona
su mille. Arriva un'epidemia e diventa una su cinquanta. I sintomi della
malattia sono identici a prima — cambia solo *quanto spesso* la risposta
giusta è "positivo". Un modello tarato sulla rarità di ieri resterà troppo
prudente, e mancherà casi veri.

**Cambia la regola stessa** (*concept shift*, o *drift*). Che cos'è lo spam?
Le stesse parole — "offerta", "clicca qui", "solo per oggi" — che nel 2005
gridavano truffa, oggi arrivano da negozi legittimi; e intanto i truffatori
hanno imparato a scrivere come una banca. Qui non cambiano solo le domande:
cambia la *risposta giusta alla stessa domanda*. È il caso più insidioso,
perché nessuna quantità di dati vecchi può insegnare una regola nuova.

`````

`````{tab} Superiore

Con la fattorizzazione $P(X,y) = P(y \mid X)\,P(X) = P(X \mid y)\,P(y)$, le
tre famiglie canoniche sono {cite}`quinonero2009dataset`:

- **Covariate shift** — cambia $P(X)$, resta invariata $P(y \mid X)$. Le foto
  invernali hanno una distribuzione diversa da quelle estive, ma la mappa
  immagine $\to$ specie è la stessa. È il caso della figura: il modello è
  accurato dove $p_{\text{train}}(x)$ è densa, e viene interrogato dove è
  quasi nulla — di fatto, un'**estrapolazione**.
- **Label shift** (o *prior probability shift*) — cambia $P(y)$, resta
  invariata $P(X \mid y)$: la malattia si presenta come prima, ma la sua
  prevalenza è diversa. Un classificatore bayesiano tarato sul *prior* vecchio
  produce probabilità a posteriori sistematicamente distorte.
- **Concept shift** (o *concept drift*) — cambia $P(y \mid X)$: la relazione
  input–etichetta stessa si sposta, gradualmente (i gusti musicali) o di colpo
  (una nuova legge cambia cosa è "transazione sospetta").

Nella pratica le tre forme arrivano mescolate, e distinguere quale domini a
partire dai soli dati è un problema difficile — spesso mal posto, se le
etichette nuove tardano ad arrivare.

`````

## Perché la validazione classica non protegge

Obiezione naturale: "ma noi le pagine sulla validazione le abbiamo studiate!
Validation set, test chiuso nel cassetto, cross-validation…". Tutto vero, e
tutto necessario. Ma c'è un punto cieco: **il validation set viene dallo
stesso passato del training set**.

`````{tab} Elementare

Train, validation e test sono tre ritagli della *stessa fotografia*. Se la
fotografia invecchia, invecchiano tutti e tre insieme — e i loro voti restano
alti proprio mentre il modello, nel mondo reale, comincia a sbagliare. È come
guidare guardando lo specchietto retrovisore: ti dice benissimo la strada già
percorsa, ma non la curva che sta arrivando.

Google Flu Trends, sui propri dati di validazione, era eccellente: era stato
validato sul passato, e sul passato funzionava davvero. Il voto d'esame era
onesto; era la domanda a essere sbagliata. La validazione risponde a "quanto
sbaglierò su dati *come questi*?", non a "quanto sbaglierò *domani*?".

`````

`````{tab} Superiore

La stima di validazione approssima $\mathbb{E}_{(X,y)\sim
P_{\text{train}}}\!\left[\mathcal{L}\big(f_\theta(X), y\big)\right]$: un
valore atteso **sotto la distribuzione di addestramento**. Se la distribuzione
operativa è un'altra, questo numero non vincola in alcun modo l'errore reale —
può restare ottimo mentre l'errore sotto $P_{\text{test}}$ diverge.

Con dati temporali c'è di peggio: la cross-validation rimescolata distrugge
l'ordine cronologico e lascia che il modello "veda il futuro" dei fold di
validazione (una forma di *leakage* temporale), gonfiando le stime. Lo stress
test più onesto è lo **split temporale**: addestrare sul passato e validare
sul futuro relativo — ad esempio, addestrare su gennaio–ottobre e validare su
novembre–dicembre. Se le prestazioni degradano già lì, degraderanno anche in
produzione; se non degradano, non è comunque una garanzia, solo un indizio
migliore.

`````

## Rimedi onesti

Non esiste la bacchetta magica: nessun algoritmo rende un modello immune al
tempo. I rimedi più efficaci non sono matematici ma *organizzativi*, e sono
tre:

1. **Monitoraggio in produzione** — trattare il modello come un impianto, non
   come un quadro appeso: sorvegliare la distribuzione degli input, la
   distribuzione delle predizioni e, appena le etichette vere arrivano,
   l'errore effettivo.
2. **Retraining periodico** — riaddestrare a intervalli regolari su dati
   recenti, così che la "fotografia" non invecchi troppo.
3. **Dati di validazione freschi** — valutare il modello su un campione
   *nuovo*, raccolto dopo l'addestramento, non sull'ennesimo ritaglio del
   dataset storico.

`````{tab} Elementare

Un modello in produzione è come la bilancia del mercato: per legge va
**ritarata periodicamente**, perché con l'uso e il tempo si starano tutte, ed
è meglio accorgersene prima del cliente. In pratica si tengono d'occhio tre
cose. Primo: gli ingressi somigliano ancora a quelli di ieri? Se un filtro
antispam riceveva email lunghe in media 80 parole e ora ne arrivano da 200, è
un campanello. Secondo: le uscite. Se ieri segnalava come spam il 20% dei
messaggi e oggi il 45%, qualcosa è cambiato — nel mondo o nel modello. Terzo:
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
w\big(X^{(i)}\big)\, \mathcal{L}\big(f_\theta(X^{(i)}),\, y^{(i)}\big),
\qquad
w(x) = \frac{p_{\text{test}}(x)}{p_{\text{train}}(x)},
$$

dove $w(x)$ è il rapporto tra la densità degli input in produzione e quella in
addestramento e $m$ è il numero di esempi. In teoria, minimizzare
$\mathcal{L}_w$ equivale a minimizzare l'errore atteso sotto
$P_{\text{test}}$. In pratica i limiti sono seri: vale solo se $P(y \mid X)$
non cambia; richiede che i supporti si sovrappongano — dove
$p_{\text{train}}(x) = 0$ ma $p_{\text{test}}(x) > 0$ nessun peso può
inventare esempi mai raccolti; e stimare il rapporto di densità in alta
dimensione è difficile, con pesi enormi su pochi esempi che fanno esplodere la
varianza. Correzioni analoghe esistono per il *label shift*, ripesando per
classi. Complementare a tutto questo è l'**out-of-distribution detection**:
riconoscere gli input troppo lontani dalla distribuzione di addestramento e,
invece di predire con finta sicurezza, astenersi o segnalare — un problema
particolarmente delicato per le reti profonde, che su input fuori
distribuzione tendono a essere *confidenti e sbagliate* insieme.

`````

Un trucco pratico per il monitoraggio, tutto fatto con strumenti che già
conosciamo: addestrare un classificatore a distinguere i dati di
addestramento da quelli raccolti in produzione. Se non ci riesce (AUC vicina
a 0,5), le due distribuzioni sono indistinguibili e possiamo dormire
tranquilli; se ci riesce bene, lo shift è già arrivato.

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

C'è un ultimo caso, il più sottile: quello in cui i dati non cambiano *nonostante*
il modello, ma *a causa sua*. Gli autori della "parabola" lo avevano notato già
in Google Flu Trends: era anche Google, aggiornando il motore di ricerca — il
completamento automatico, le ricerche suggerite — a cambiare i dati che il suo
stesso modello leggeva {cite}`lazer2014parable`.

Nei sistemi moderni questo **circuito di retroazione** (*feedback loop*) è
ovunque. Un sistema di raccomandazione mostra i contenuti che prevede
piaceranno; l'utente sceglie tra *quelli* — e i clic raccolti confermano al
modello che aveva ragione, qualunque cosa avesse mostrato. Un modello di
credito nega il prestito a chi giudica rischioso: di quelle persone non
sapremo mai se avrebbero restituito i soldi, e i dati futuri conterranno solo
le storie di chi il prestito l'ha avuto. In entrambi i casi il modello non
osserva più il mondo: osserva le conseguenze delle proprie decisioni, e sui
propri numeri può perfino sembrare sempre più bravo mentre amplifica i gusti
già espressi e cementa nei dati le proprie distorsioni. Ci torneremo nel
capitolo sui sistemi di raccomandazione, dove il circuito di retroazione non è
un effetto collaterale ma la struttura stessa del problema.

```{admonition} Da ricordare
:class: important
- Tutto il libro poggia su un'ipotesi tacita: dati di addestramento e dati
  reali vengono dalla **stessa distribuzione** (i.i.d.). Il mondo non ha
  firmato quel contratto.
- Tre famiglie di **dataset shift**: *covariate shift* (cambia $P(X)$: le
  domande), *label shift* (cambia $P(y)$: le proporzioni delle risposte),
  *concept shift* (cambia $P(y \mid X)$: la regola stessa).
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
