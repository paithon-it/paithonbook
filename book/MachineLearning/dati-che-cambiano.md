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
quota di visite mediche per sintomi influenzali più che doppia rispetto a
quella registrata dai CDC. E non era un incidente isolato: tra l'agosto 2011 e
il settembre 2013 il servizio aveva sovrastimato l'influenza in 100
settimane su 108. Nel 2014 un gruppo di ricercatori firmò su *Science*
l'autopsia del progetto, con un titolo diventato proverbiale: *The Parable of
Google Flu* (la "parabola", nel senso del racconto che ammonisce
{cite}`lazer2014parable`). Nell'agosto 2015 Google chiuse il servizio.

Che cosa era andato storto? Il modello non si era rotto: era invecchiato.
Il modo di cercare in rete era cambiato, e Google stessa aggiornava il motore:
dal giugno 2011 cominciò a proporre altri termini da cercare (a chi
chiedeva dell'influenza suggeriva di cercarne le cure), dal febbraio 2012 a
rispondere alle ricerche sui sintomi con le diagnosi possibili (chi cercava
«febbre» o «tosse» si vedeva proporre l'influenza). Erano due spinte verso
l'influenza che arrivavano anche a chi stava benissimo, e la catena si chiude
da sé: più suggerimenti, più ricerche sull'influenza; e siccome il modello
contava proprio quelle ricerche, più malati stimati. Intanto i giornali
parlavano di epidemia e la gente cercava per curiosità, non per febbre. Il
modello, tarato sul mondo del 2008, continuava a leggere il presente con gli
occhiali di allora. Ed è esattamente questo il punto: che cosa succede quando i
dati che un modello incontra non somigliano più a quelli su cui è stato
addestrato.

## L'ipotesi nascosta: che l'urna non cambi

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

C'è una condizione in più, e riguarda il momento in cui si decide la domanda.
«Quante rosse?» stabilito prima di pescare, e la manciata risponde per l'urna.
Stabilito dopo, no. Chi pesca cento palline e poi va a cercare, fra mille
domande possibili, quella che su quelle cento torna meglio (le rosse escono a
coppie, la quarta e la nona sono gialle) ha ritagliato la domanda addosso alla
manciata, e sull'urna intera quella domanda non dice niente. È l'overfitting
visto dal lato dell'urna, e ci si difende allo stesso modo, tenendo corto
l'elenco delle domande che ci si concede prima di pescare.

Una parola su questa urna, che ha anche un nome tecnico. La distribuzione
di una cosa è, semplicemente, il resoconto di quanto spesso ciascun valore
capita: le palline rosse al 30% e le altre al 70% *sono* la distribuzione dei
colori in quell'urna. «La distribuzione degli input è cambiata» vuol dire
esattamente «l'urna non è più quella». E un'urna si cambia in più di un modo,
mettendoci dentro palline diverse, oppure cambiando quanto spesso la risposta
giusta è una o l'altra, oppure cambiando quello che il colore di una pallina
significa.

`````

`````{tab} Superiore

Formalmente, assumiamo che le coppie $(\mathbf{x}^{(i)}, y^{(i)})$ del
training e quelle che il modello vedrà in produzione siano estratte in modo
indipendente e identicamente distribuito (i.i.d.) da un'unica
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
Ognuno dei tre fattori può cambiare per conto suo, e a ciascuno corrisponde
una famiglia di shift.

`````

## Tre modi in cui il mondo cambia

Non tutti i cambiamenti sono uguali. Chi studia il fenomeno ne distingue tre
famiglie {cite}`quinonero2009dataset`, e conviene impararle con esempi
quotidiani, perché la diagnosi giusta suggerisce il rimedio giusto.
{numref}`fig-distribution-shift` mostra il caso più semplice da visualizzare:
i dati nuovi arrivano in una zona che l'addestramento ha quasi ignorato. Il
grafico va letto in un modo nuovo rispetto a quelli visti finora, dove i punti
erano esempi: qui sull'asse orizzontale c'è il valore di una caratteristica (i
metri quadri, l'età, il numero di ricerche) e sulla verticale quanto
spesso quel valore capita, così che dove la curva è alta ci sono tanti
esempi e dove è schiacciata quasi nessuno. Due curve sfalsate vogliono dire
che i valori frequenti ieri non sono quelli frequenti oggi.

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

Cambiano le domande (*covariate shift*). Un'app che riconosce le piante,
addestrata su foto scattate d'estate, viene usata d'inverno: luce bassa, rami
spogli, neve sullo sfondo. Le foto che arrivano sono diverse da quelle viste a
lezione, ma attenzione: un abete resta un abete. La *regola* che collega foto
e risposta non è cambiata; è cambiato il tipo di foto che arriva. Davanti a un
ramo spoglio nella neve una risposta la dà lo stesso, e la dà su un caso che
d'estate non poteva capitarle, allungando quello che sa oltre il punto in cui
l'ha imparato.

Cambiano le proporzioni delle risposte (*label shift*). Un modello aiuta a
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

Cambia la regola stessa (*concept shift*, o *concept drift*). Che cos'è lo
spam? Le stesse parole ("offerta", "clicca qui", "solo per oggi") che nel 2005
gridavano truffa, oggi arrivano da negozi legittimi; e intanto i truffatori
hanno imparato a scrivere come una banca. Qui non cambiano solo le domande:
cambia la *risposta giusta alla stessa domanda*. Allo spam sono serviti anni;
altrove la regola cambia da un giorno all'altro, come quando una legge nuova
stabilisce che cosa conta come transazione sospetta. È il caso più insidioso,
perché nessuna quantità di dati vecchi può insegnare una regola nuova.

Nel mondo vero i tre cambiamenti non arrivano in fila e ben separati.
L'inverno porta insieme foto più scure e più abeti che margherite, cioè
domande diverse e proporzioni diverse in una volta sola. E capire quale dei
tre pesi di più, guardando i soli dati in arrivo, è difficile: per sapere se
l'app ha sbagliato bisogna che qualcuno guardi la foto e dica che pianta era,
e quel qualcuno arriva tardi, o non arriva affatto.

`````

`````{tab} Superiore

Con la fattorizzazione $P(X,y) = P(y \mid X)\,P(X) = P(X \mid y)\,P(y)$, le
tre famiglie canoniche sono {cite}`quinonero2009dataset`:

- **Covariate shift**: cambia $P(X)$, resta invariata $P(y \mid X)$. Le foto
  invernali hanno una distribuzione diversa da quelle estive, ma la mappa
  immagine $\to$ specie è la stessa. È il caso della figura: il modello è
  accurato dove $p_{\text{train}}(\mathbf{x})$ è densa, e viene interrogato
  dove è quasi nulla (di fatto, un’estrapolazione).
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
tutto necessario. Ma c'è un punto cieco: il validation set viene dallo
stesso passato del training set.

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
dati hanno una data sopra: invece di tagliarli a caso, si taglia nel tempo.
Il modello studia su gennaio-ottobre e viene interrogato su novembre-dicembre,
che al momento dell'addestramento erano il futuro. Se già lì peggiora, in
mezzo al mondo vero peggiorerà di sicuro; se non peggiora non è una garanzia,
ma è un indizio molto migliore di un rimescolamento che gli lascia sbirciare
il domani.

`````

`````{tab} Superiore

La stima di validazione approssima
$\mathbb{E}_{(X,y)\sim P_{\text{train}}}\!\left[\ell\big(f_\theta(X), y\big)\right]$:
un valore atteso sotto la distribuzione di addestramento. Se la
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

1. Sorvegliare il modello mentre lavora. «In produzione» vuol dire proprio
   questo: non più le prove in laboratorio, ma il modello acceso sul serio, con
   utenti veri e dati che arrivano ogni giorno. Un modello in produzione va
   trattato come un impianto, non come un quadro appeso, e le cose da tenere
   d'occhio sono tre: come sono fatte le domande che arrivano, come sono fatte
   le risposte che dà, e, appena si scopre qual era la risposta giusta, quanto
   ha sbagliato davvero.
2. Riaddestrare a intervalli regolari (in gergo *retraining*) su dati
   recenti, così che la "fotografia" non invecchi troppo.
3. Giudicarlo su dati freschi: su un campione *nuovo*, raccolto dopo
   l'addestramento, non sull'ennesimo ritaglio del mucchio di esempi di
   partenza.

`````{tab} Elementare

Un modello in produzione è come la bilancia del mercato, che per legge va
ritarata periodicamente, perché con l'uso e il tempo si starano tutte, ed
è meglio accorgersene prima del cliente. In pratica si tengono d'occhio tre
cose. Gli ingressi, per cominciare. Se un filtro antispam riceveva email
lunghe in media 80 parole e ora ne arrivano da 200, è un campanello. Poi le
uscite. Se ieri segnalava come spam il 20% dei messaggi e oggi il 45%,
qualcosa è cambiato, nel mondo o nel modello. Infine gli errori veri, appena
si scopre la risposta giusta (l'utente ha ripescato l'email dal cestino?).

Sui conti si può fare un aggiustamento, ed è quello che gli istituti di
sondaggi fanno da sempre. Il campione di mille persone ne contiene 50 sotto i
trent'anni, il 5%, mentre nel paese quella fascia pesa il 20%, quattro volte
tanto. Allora ogni giovane intervistato viene contato quattro volte (20
diviso 5), e chi viene da una fascia sovrarappresentata conta meno di una
persona intera. Nessuno torna
a bussare a nessuna porta, cambia solo il peso che ogni risposta ha nella
media finale, e il totale torna a somigliare al paese invece che al campione.
Lo stesso conto ripara il caso della malattia diventata comune, dove i malati
erano uno su mille e adesso sono uno su cinquanta: ogni malato del vecchio
mucchio di esempi conta venti volte.

La ripesatura ripara un guasto solo, quello di aver intervistato le persone
sbagliate. Se nel frattempo la gente ha cambiato idea, i giovani del campione
hanno detto quello che pensavano allora, e moltiplicare per quattro una
risposta vecchia dà una risposta vecchia quattro volte.

E se sotto i trent'anni non è stato intervistato nessuno, non c'è peso che
tenga, perché qualunque numero moltiplicato per zero fa zero. Di quella fascia
il sondaggio non sa niente, e nessun conto può inventare le interviste che non
sono state fatte.

Il caso che inganna di più sta in mezzo. Se sotto i trent'anni ne sono
capitati due invece di cinquanta, quei due devono valere per il 20% del paese
e contano cento volte ciascuno; basta che uno dei due cambi idea perché un
partito perda dieci punti. Il conto resta giusto, e il risultato balla. E le
fasce quasi vuote si moltiplicano appena si ripesa per più cose insieme, età e
regione e titolo di studio e reddito, perché più le caselle sono strette e
meno gente ci finisce dentro.

Da qui viene anche il permesso di passare la mano. Davanti a un caso che
non somiglia a nulla di già visto, meglio dire "non lo so, decida un umano"
che sparare una risposta sicura e sbagliata.

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
w(\mathbf{x}) = \frac{p_{\text{test}}(\mathbf{x})}{p_{\text{train}}(\mathbf{x})},
$$

dove $w(\mathbf{x})$ è il rapporto tra la densità degli input in produzione e
quella in addestramento e $m$ è il numero di esempi. In teoria, minimizzare
$\mathcal{L}_w$ equivale a minimizzare l'errore atteso sotto $P_{\text{test}}$.
In pratica i limiti sono seri: vale solo se $P(y \mid X)$ non cambia; richiede
che i supporti si sovrappongano, dove $p_{\text{train}}(\mathbf{x}) = 0$ ma
$p_{\text{test}}(\mathbf{x}) > 0$ nessun peso può inventare esempi mai
raccolti; e stimare il rapporto di densità in alta dimensione è difficile, con
pesi enormi su pochi esempi che fanno esplodere la varianza. Quanti esempi
contano davvero lo dice la taglia effettiva, $m_{\text{eff}} = \bigl(\sum_i
w_i\bigr)^2 / \sum_i w_i^2$, che crolla appena pochi pesi dominano. Le due
densità, però, non vanno stimate una per una: un classificatore che distingue
addestramento e produzione, addestrato su $m_{\text{tr}}$ e $m_{\text{prod}}$
esempi, dà $w(\mathbf{x}) =
\frac{m_{\text{tr}}}{m_{\text{prod}}}\,\frac{P(\text{prod}\mid\mathbf{x})}{1 -
P(\text{prod}\mid\mathbf{x})}$, ed è lo stesso detective che serve ad
accorgersi della deriva. Infine il ripeso conta soprattutto quando il modello è
mal specificato: se la famiglia contiene la vera $P(y \mid X)$, il minimo non
pesato è già consistente e i pesi aggiungono soltanto varianza
{cite}`shimodaira2000improving`.

Per il *label shift* il conto è più semplice, perché il rapporto dipende dalla
sola classe. Con $P(X \mid y)$ invariata, $p_{\text{test}}(y \mid \mathbf{x})
\propto p_{\text{train}}(y \mid
\mathbf{x})\;\pi_{\text{test}}(y)/\pi_{\text{train}}(y)$, dove $\pi(y)$ è la
prevalenza della classe: un classificatore calibrato si corregge senza
riaddestrarlo, moltiplicando le posteriori per
$\pi_{\text{test}}(y)/\pi_{\text{train}}(y)$ e rinormalizzando (per la malattia
passata da uno su mille a uno su cinquanta, $20$ sui positivi e $0{,}98/0{,}999
\approx 0{,}98$ sui negativi). Resta da stimare $\pi_{\text{test}}$, che senza
etichette nuove si ricava con un EM sulle posteriori
{cite}`saerens2002adjusting` o invertendo la matrice di confusione del modello
{cite}`lipton2018detecting`. Complementare a tutto
questo è l’**out-of-distribution detection**: riconoscere gli input troppo
lontani dalla distribuzione di addestramento e, invece di predire con finta
sicurezza, astenersi o segnalare; un problema particolarmente delicato per le
reti profonde, che su input fuori distribuzione tendono a essere *confidenti e
sbagliate* insieme.

`````

C'è un trucco pratico per accorgersene, e usa solo strumenti che già
conosciamo. Si mescolano i dati di addestramento con quelli raccolti mentre il
modello lavorava, si cancella qualsiasi altra etichetta e si addestra un
secondo modello a rispondere a una domanda sola: questo esempio viene da ieri
o da oggi? Se ci riesce, ieri e oggi sono distinguibili, cioè la deriva c'è; e
il suo punteggio dice pure quanto è grossa.

Il punteggio giusto da guardare qui è l'AUC della {doc}`sezione sulle metriche
<metriche>`, che
per un detective come questo si legge benissimo: $0{,}5$ vuol dire che sta
tirando a indovinare, cioè che i due mucchi gli sembrano identici, e $1$ vuol
dire che li separa senza sbagliare un colpo.

Attenzione però a leggere il silenzio. Un'AUC vicina a $0{,}5$ dice che le due
epoche sono indistinguibili per lui, che è una conclusione più debole di
«va tutto bene», per due ragioni. La prima è che questo detective guarda
soltanto le domande in arrivo, non le risposte: del cambio di regola, dove le
domande restano le stesse e a cambiare è la risposta giusta, non può
accorgersi per costruzione, ed è il caso più insidioso dei tre. La seconda è
che un'AUC vicina a $0{,}5$ può anche voler dire che gli esempi nuovi sono
ancora troppo pochi, o che la deriva sta in un intreccio fra più
caratteristiche che quel detective, preso singolarmente, non coglie: non aver
trovato non è aver dimostrato che non c'è niente. È uno strumento da avere,
purché letto per quello che è: un allarme quando suona, non un certificato
quando tace.

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
    """Quanto bene un modello indovina da quale delle due epoche viene un esempio."""
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

I due numeri sono le due letture possibili di questo strumento, e conviene
tenerle vicine perché da sole non si interpretano. Il secondo, $0{,}508$, è il
caso in cui non è cambiato niente: il detective tira a indovinare, come chi
lanciasse una monetina. Il primo, $0{,}736$, è l'allarme: una sola
caratteristica scivolata di un'unità e mezza basta perché l'epoca di
provenienza diventi in buona parte indovinabile.

## Imparare un esempio alla volta

Riaddestrare a intervalli rifà la fotografia da capo, e fra una fotografia e
l'altra il modello resta fermo. L'alternativa è non fermarlo mai: il modello si
aggiorna a ogni esempio, appena ne conosce la risposta giusta, senza aspettare
di averne raccolto un mucchio, e può farlo tenendo soltanto i propri
parametri, senza conservare gli esempi già visti (che col tempo non starebbero
più da nessuna parte). È l’**apprendimento online** (*online learning*), e
cambia la domanda di partenza, che non è più quanto il modello sbaglierà su
esempi della stessa urna, perché non suppone che gli esempi vengano da
un'urna fissa. Il metro del successo diventa
il **rimpianto** (*regret*): quanto si è perso rispetto alla migliore scelta
fissa, fatta col senno di poi su tutti gli esempi arrivati.

`````{tab} Elementare

Il filtro antispam impara una email alla volta. Arriva un messaggio, il filtro
dà il suo verdetto, e poco dopo l'utente gli dice se aveva ragione, lasciando
l'email dov'è o ripescandola dal cestino. A quel punto il filtro corregge un
poco le sue regole, nella direzione che su quell'email avrebbe ridotto l'errore,
e passa alla successiva. È la discesa del gradiente, fatta un esempio alla
volta. Le email vecchie non le tiene, perché gli bastano le regole che ha
adesso, e le tiene dentro limiti fissati (tutti i pesi insieme non possono
crescere oltre una certa misura), perché una parola che compare in mille spam
non finisca per pesare all'infinito.

Quanto correggere ogni volta è la scelta che decide tutto. Correzioni grandi
inseguono ogni singola email; correzioni piccole imparano con lentezza. La
ricetta che funziona meglio sta nel mezzo e cambia col tempo: la
correzione si accorcia man mano che le email si accumulano, come uno diviso la
radice di quante ne sono arrivate, e alla centesima vale un decimo della prima.

Il filtro si giudica a fine anno. Con tutte le email dell'anno davanti si cerca
il miglior filtro fisso, quello che, applicato dal primo giorno, avrebbe
sbagliato meno di tutti; il rimpianto è quanto il filtro che ha imparato strada
facendo ha sbagliato in più. La garanzia dice che il rimpianto cresce al più
come la radice del numero di email, cioè più piano delle email stesse, e che
quindi il rimpianto per email tende a zero. Più piano della radice, in generale,
non si può promettere: se le email fossero decise a testa o croce nessun filtro
potrebbe indovinarle, eppure col senno di poi uno dei filtri fissi sembrerebbe
più bravo degli altri filtri fissi, per pura fortuna, di una quantità che
cresce proprio come la radice (lanciando cento monete ci si aspettano
cinquanta teste, e cinquantacinque sono normali; con diecimila, cinquemila più
o meno cinquanta). Si fa meglio solo se il problema aiuta: quando ogni email
punisce con decisione qualunque filtro lontano dal migliore, con un errore che
sale come una conca ripida e non come un fondovalle piatto, la correzione si
può accorciare più in fretta, come uno diviso il numero di email, e il
rimpianto cresce appena come il logaritmo, cioè quasi niente. E la garanzia non
chiede niente alle email, e vale anche se chi le scrive prova apposta a
ingannare il filtro, perché il paragone è sempre con il miglior filtro fisso su
quelle stesse email.

Se poi le email vengono davvero tutte dalla stessa urna, c'è un regalo in più.
Si prende la media dei filtri usati giorno per giorno, regola per regola (il
peso medio che ogni parola ha avuto lungo l'anno), e il filtro che ne esce va
bene su email nuove quasi quanto il miglior filtro fisso, che con un'urna
fissa è il migliore in assoluto, senza aver mai tenuto da parte il mucchio
intero. La media e non l'ultimo filtro, perché il rimpianto è una somma su
tutti i giorni, e la media dei filtri è quella che ne eredita la garanzia.

Il punto di rottura sta nel paragone. Il rimpianto si misura contro il miglior
filtro *fisso*, e se a metà anno gli spammer cambiano trucco nessun filtro fisso
va bene per tutto l'anno: un rimpianto piccolo, allora, promette poco. Peggio,
le correzioni che si accorciano rendono il filtro sempre più lento a seguire:
dopo diecimila email ogni correzione vale un centesimo della prima, e un cambio
di regola lo trova testardo. Con correzioni di misura fissa il filtro resta
pronto a seguire, e lo paga oscillando anche quando non cambia niente, perché
ogni email lo strattona della stessa misura anche quando aveva già ragione. Quale
misura scegliere dipende da quanto spesso cambia il mondo, che non si sa in
anticipo. C'è anche chi cambia il paragone: invece del miglior filtro fisso, il
miglior filtro a cui è permesso cambiare qualche volta durante l'anno, e contro
quello il rimpianto torna a promettere qualcosa.

`````

`````{tab} Superiore

Il quadro è l’*online convex optimization*. A ogni turno $t = 1, \dots, T$
l'algoritmo sceglie $\mathbf{w}_t$ in un insieme convesso $\mathcal{K}$ di
diametro $D$, poi viene rivelata una perdita convessa $\ell_t$ e l'algoritmo
paga $\ell_t(\mathbf{w}_t)$. Sulle $\ell_t$ non si fa nessuna ipotesi
statistica, e può sceglierle un avversario che conosce l'algoritmo. Il
rimpianto confronta con il miglior punto fisso col senno di poi,

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
almeno $1-\delta$, rischio al più la perdita media pagata online più
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
{cite}`zinkevich2003online`. Il terzo termine è il prezzo del passo fisso, che
si paga anche quando niente si muove; il secondo punisce il passo piccolo
quando il bersaglio si sposta, e il passo che bilancia i due,
$\eta \propto \sqrt{(D^2 + D\,P_T)/(G^2 T)}$, dipende da $P_T$, che non si
conosce in anticipo. Nel caso a esperti lo stesso problema ha la risposta di
Herbster e Warmuth, che confrontano con il miglior esperto autorizzato a
cambiare un numero fissato di volte {cite}`herbster1998tracking`.

`````

Il rimpianto torna più avanti in due posti. Nell'apprendimento per imitazione è
la proprietà che la garanzia di DAgger chiede alla successione delle politiche,
nella sezione sull’{doc}`imitazione </DeepReinforcementLearning/imitazione>`; e
misurato in bit, come lunghezza di un file compresso rispetto al miglior
compressore possibile, regge l'argomento di
{doc}`Capire è accorciare </AutoSupervisione/capire-e-accorciare>`, nel capitolo
sull'auto-supervisione.

Il primo esperimento misura il rimpianto su flussi sempre più lunghi. Il modello
è la regressione logistica, che si corregge dopo ogni esempio con passo
$1/\sqrt{t}$ (il passo del teorema di Zinkevich; con $D/(G\sqrt{t})$ si
ottiene la costante $\tfrac32 GD$) e riporta i pesi dentro una palla di raggio
$5$ se ne escono, come chiede la garanzia: sono i «limiti fissati» della scena,
la lunghezza dell'elenco dei pesi che non supera $5$. La perdita è la sua
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
e per esempio scende da due centesimi a sei decimillesimi. Anche diviso per
$\sqrt{T}$ scende. Il $\sqrt{T}$ della garanzia è il caso peggiore, quello di un
avversario, e su esempi estratti da un'urna fissa il rimpianto può crescere più
piano, come fa qui; ma nessuna delle garanzie appena viste lo promette per
questo algoritmo: quella logaritmica chiede perdite fortemente convesse
dappertutto, la conca ripida della scena, e la cross-entropy non lo è. E tre
flussi estratti una volta sola non bastano a dire con quale legge cresca.

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

Il modello fermo è buono finché il mondo resta quello dei suoi duemila esempi,
e dopo il cambio paga $0{,}39$ in più a ogni esempio, per sempre. Fra gli
allievi online l'ordine si rovescia da una colonna all'altra. Nei periodi
tranquilli il passo che si accorcia è il più preciso, con $0{,}0008$ contro i
$0{,}0138$ del passo fisso più lungo; subito dopo il cambio è il più lento, con
$0{,}2246$ contro $0{,}0303$, più di sette volte tanto. Il passo fisso più
corto sta in mezzo in tutte le colonne, ed è il compromesso fra prontezza e
precisione, e nessun passo fisso lo scioglie. La figura segue due degli allievi
su un solo flusso, e invece della perdita misura quanto i loro pesi (le regole
del filtro, il peso di ogni parola) distano da quelli della regola vera
({numref}`fig-bersaglio-che-si-sposta`).

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
{doc}`capitolo sui sistemi di raccomandazione </SistemiRaccomandazione/overview>`, dove il circuito di retroazione non è
un effetto collaterale ma la struttura stessa del problema.

`````{tab} Elementare

```{admonition} Da ricordare
:class: important
- Tutto il libro poggia su un'ipotesi tacita: che i dati di ieri e quelli di
  domani vengano dalla stessa urna. Il mondo non ha firmato quel contratto.
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
- I rimedi che funzionano non sono formule ma abitudini: sorvegliare il
  modello mentre lavora (gli ingressi somigliano a quelli di ieri? le risposte
  sono cambiate di colpo?), riaddestrarlo ogni tanto su dati recenti,
  giudicarlo su dati freschi. E dargli il permesso di dire «non lo so».
- Un modello può anche imparare mentre lavora, un esempio alla volta. Lo si
  giudica col rimpianto, quanto sbaglia in più del miglior modello fisso
  scelto col senno di poi, e la garanzia vale qualunque cosa facciano i dati;
  ma se il mondo cambia a metà strada un rimpianto piccolo promette poco,
  correzioni che si accorciano lo rendono lento davanti al cambio, e
  correzioni di misura fissa lo fanno oscillare sempre.
- Attenzione a quando è il modello stesso a fabbricare i dati di domani: se
  mostra solo certi contenuti, vedrà solo clic su quelli; se nega il prestito,
  non saprà mai chi avrebbe restituito. Da lì in avanti non guarda più il
  mondo, guarda le conseguenze delle proprie decisioni.
- Google Flu Trends è la parabola da ricordare: un modello eccellente sul
  passato può invecchiare in silenzio, restando bravissimo agli esami che si dà
  da solo.
```

`````

`````{tab} Superiore

```{admonition} Da ricordare
:class: important
- Tutto il libro poggia su un'ipotesi tacita: dati di addestramento e dati
  reali vengono dalla stessa distribuzione (i.i.d.). Il mondo non ha
  firmato quel contratto.
- Tre famiglie di dataset shift: *covariate shift* (cambia $P(X)$: le
  domande), *label shift* (cambia $P(y)$: le proporzioni delle risposte),
  *concept shift* (cambia $P(y \mid X)$, cioè la regola stessa; la barra
  verticale si legge «dato», quindi $P(y \mid X)$ è la probabilità della
  risposta *dato* l'input).
- La validazione classica non protegge: validation e test sono ritagli
  dello stesso passato. Con dati temporali, meglio lo split temporale.
- Rimedi onesti: monitoraggio in produzione, retraining periodico,
  validazione su dati freschi; l’*importance weighting* corregge il
  covariate shift ma solo con supporti sovrapposti e densità stimabili.
- Apprendimento online: su perdite convesse, la discesa del gradiente online ha
  rimpianto $O(GD\sqrt{T})$ senza ipotesi statistiche sui dati, e con dati
  i.i.d. la media degli iterati eredita la garanzia (*online-to-batch*). Contro
  un bersaglio che si muove il passo decrescente è lento e il passo fisso paga
  $\eta G^2 T/2$ anche da fermo: il passo giusto dipende da quanto si muove il
  mondo.
- Attenzione ai feedback loop: quando le decisioni del modello generano i
  dati futuri (raccomandazioni, credito), il modello smette di osservare il
  mondo e inizia a osservare se stesso.
- Google Flu Trends resta la parabola di riferimento: un modello eccellente
  sul passato può invecchiare in silenzio {cite}`lazer2014parable`.
```

`````

Un modello che invecchia in silenzio ha un difetto in comune con quasi tutti
quelli incontrati fin qui: risponde con un numero secco, e non dice quanto
fidarsi. Resta da vedere un modello che accompagna ogni previsione con la
propria incertezza, larga dove i dati mancano e stretta dove abbondano: sono i
{doc}`processi gaussiani <processi-gaussiani>`.
