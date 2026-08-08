# Modelli trasparenti e importanza delle feature

A metà degli anni Novanta, addestrando un modello per stimare il rischio di
morte dei pazienti ricoverati per polmonite, un gruppo di ricercatori di
Pittsburgh scoprì che l'algoritmo aveva imparato una regola sorprendente: *chi
soffre d'asma ha un rischio più basso*. Preso alla lettera, un consiglio
pericoloso: gli asmatici sono pazienti fragili. La spiegazione era clinica,
non causale: negli ospedali gli asmatici con polmonite venivano mandati subito
in terapia intensiva, e proprio quelle cure aggressive ne abbassavano la
mortalità. Il modello aveva colto una correlazione vera nei dati e ne aveva
tratto una conclusione che, usata per decidere chi mandare a casa, avrebbe
ucciso. La storia (raccontata anni dopo da Rich Caruana) è diventata il
manifesto di un campo: se non possiamo *guardare dentro* un modello, non
sappiamo su quali scorciatoie si regge, e non possiamo fidarcene quando la
posta è alta.

Ci sono due strade per capire un modello. La prima è sceglierlo **trasparente
per costruzione**, così semplice che la sua logica si legge a occhio nudo. La
seconda è tenere il modello com'è (magari una grande rete) e interrogarlo da
fuori con strumenti che ne rivelano il comportamento. Questa sezione apre il
capitolo percorrendo la prima strada, i modelli intrinsecamente
interpretabili, e imboccando la seconda con il primo attrezzo del kit
*post-hoc*: l'**importanza delle feature**. Per un panorama sistematico
dell'intero campo il riferimento è il manuale di Molnar
{cite}`molnar2022interpretable`.

## Modelli trasparenti per costruzione

```{figure} ../figures/alberi-di-decisione.svg
:name: fig-albero-percorso
:alt: "Un albero di decisione con la radice in alto: a ogni nodo una domanda su una singola feature con soglia, e due rami a seconda della risposta; scendendo si arriva a una foglia che porta la predizione. Un percorso dalla radice a una foglia è evidenziato."
:width: 90%

La spiegazione è il percorso. Per sapere perché un esempio ha ricevuto quella
risposta si leggono le domande incontrate scendendo, e sono poche.
```

{numref}`fig-albero-percorso` mostra una forma di trasparenza diversa da
quella dei modelli lineari, e per certi versi più forte. Un modello lineare
spiega con dei pesi, che valgono per tutti gli esempi insieme; un albero
spiega *questo* esempio con una catena di condizioni verificabili una per una.

Alcuni modelli non hanno bisogno di essere spiegati: *sono* la loro
spiegazione. La regressione lineare e quella logistica, incontrate nel
capitolo sul machine learning, ne sono l'esempio più puro: la predizione è una
somma pesata delle feature, e i pesi *sono* la storia che il modello racconta.

```{figure} ../figures/regressione-lineare.svg
:name: fig-retta-residui
:alt: "Una nube di punti attraversata da una retta di regressione. Da ciascun punto scende o sale un segmento verticale fino alla retta: sono i residui, cioè la parte che il modello non spiega. La retta è quella che rende più piccola la somma dei loro quadrati."
:width: 84%

La retta e ciò che le sfugge. I segmenti verticali sono i residui: il modello
sceglie la retta che li rende complessivamente più corti, e li lascia in bella
vista.
```

C'è una qualità di {numref}`fig-retta-residui` che le reti profonde non hanno,
ed è il motivo di questa sezione: l'errore è *localizzato*. Si vede quale
punto il modello sbaglia e di quanto, e la regola che ha usato è una sola riga
di somma. Trasparente non vuol dire accurato, vuol dire che non c'è niente da
scoprire dopo.

`````{tab} Elementare

Riprendiamo il modello che stima il prezzo di una casa come somma di
contributi: tanti euro per ogni metro quadro, tanti per ogni stanza, un bonus
o un malus per il quartiere. Ogni coefficiente è un'etichetta col prezzo
appesa a una caratteristica: «$+2\,000$ € al metro quadro» si legge senza
sforzo. Non serve nessuno strumento esterno per capire perché il modello ha
detto $210\,000$ €: basta leggere la ricevuta, voce per voce.

Vale lo stesso per la versione che classifica (la regressione logistica): un
coefficiente positivo spinge la probabilità verso il «sì», uno negativo verso
il «no», e più è grande più spinge. Un modello così si può stampare su mezza
pagina e discutere con chi non ha mai visto una formula. È questo che
intendiamo per *trasparente*: la regola di decisione è alla luce del sole.

`````

`````{tab} Superiore

In un modello lineare $\hat{y} = W^\top X + b$ ogni coefficiente $w_j$ è
l'effetto marginale della feature $j$: a parità di tutte le altre, un aumento
unitario di $x_j$ sposta la predizione di esattamente $w_j$. Nella regressione
logistica $\hat{y} = \sigma(W^\top X + b)$ l'interpretazione passa alle
*log-odds*: $w_j$ è la variazione del logaritmo del rapporto di probabilità
$\log\frac{p}{1-p}$ per un incremento unitario di $x_j$, cosicché $e^{w_j}$ è
il fattore moltiplicativo sull'*odds ratio*.

Due avvertenze rendono onesta questa lettura. Primo, i coefficienti sono
confrontabili tra loro solo se le feature sono **standardizzate** (stessa
scala): un $w_j$ grande può riflettere semplicemente un'unità di misura
piccola. Secondo, l'inciso «a parità di tutte le altre» è fragile quando le
feature sono **correlate**: se due colonne si muovono insieme, il modello può
spartire il loro effetto in modo arbitrario, e i singoli coefficienti
diventano instabili (la stessa multicollinearità che rende preziosa la
regolarizzazione Ridge/Lasso vista nel capitolo di machine learning).

`````

La trasparenza non finisce con i modelli lineari. Gli **alberi di decisione**,
studiati nel capitolo sul machine learning, sono l'altro archetipo di «scatola
bianca»: una predizione è un percorso di domande sì/no dalla radice a una
foglia, e quel percorso *è* la spiegazione. Sulla stessa famiglia si collocano
i **modelli additivi generalizzati** (GAM), che estendono la regressione
lineare sostituendo a ogni coefficiente una funzione liscia,
$\hat{y} = \beta_0 + \sum_j f_j(x_j)$: ogni $f_j$ si può disegnare come una
curva («come cambia il rischio al variare dell'età») restando leggibile una
feature alla volta. E ci sono i **sistemi a regole**, elenchi di condizioni
del tipo «SE reddito $<$ 20 000 E contratto a termine ALLORA nega», che
decidono in modo del tutto ispezionabile.

Aleggia però un pregiudizio diffuso: che la trasparenza si paghi in
accuratezza, che per essere bravi si debba per forza essere oscuri. È vero solo
in parte.

`````{tab} Elementare

L'idea comune è: «i modelli semplici sono deboli, quelli forti sono
incomprensibili (scegli»). A volte è così, soprattutto su immagini, testo e
suoni, dove le reti profonde vincono senza rivali. Ma su tanti problemi
concreti (quelli a righe e colonne di un foglio di calcolo, come una
valutazione del credito o del rischio clinico) un modello trasparente ben
costruito arriva vicinissimo, a volte alla pari, con la scatola nera. In quei
casi scegliere l'oscurità non compra accuratezza: regala solo opacità.

Il consiglio pratico che ne segue è di buon senso: parti dal modello
trasparente e misura quanto perdi davvero passando a uno più complicato. Se la
differenza è minima, l'interpretabilità è un guadagno netto: soprattutto dove
una decisione sbagliata ha un costo umano.

`````

`````{tab} Superiore

Il presunto compromesso accuratezza/interpretabilità è stato messo in
discussione, in particolare da Cynthia Rudin (2019), che sostiene come su dati
**strutturati** con feature dotate di senso il divario tra un modello
interpretabile ben ingegnerizzato e una scatola nera sia spesso trascurabile o
nullo. La ragione è che il vantaggio del *deep learning* si manifesta soprattutto
là dove serve **apprendere le rappresentazioni** da dati grezzi ad alta
dimensione (pixel, forme d'onda, token); sui dati tabellari le feature sono già
significative, e modelli come gradient boosting o GAM catturano quasi tutta la
struttura utile restando ispezionabili.

Ne discende una gerarchia metodologica: preferire un modello intrinsecamente
interpretabile quando le prestazioni sono comparabili, e riservare gli
strumenti *post-hoc* (importanza delle feature, PDP, e i metodi locali che
vedremo più avanti nel capitolo) ai casi in cui la scatola nera è davvero
necessaria. Gli strumenti post-hoc, va detto subito, spiegano il modello
*dall'esterno* e sono approssimazioni: non sostituiscono la trasparenza di
progetto.

`````

## L'importanza delle feature: quali colonne contano

```{figure} ../figures/feature-selection.svg
:name: fig-feature-selection
:alt: "A sinistra un grafico a barre con il punteggio di importanza di otto feature, ordinate dalla più alta alla più bassa, e una soglia orizzontale che ne separa tre in alto dalle cinque sotto. A destra il dataset ridotto alle sole tre feature che superano la soglia."
:width: 100%

Dalla classifica al taglio. La parte delicata non è ordinare le feature ma
decidere dove passa la riga, perché quella soglia non la suggerisce nessun
dato.
```

La distinzione che {numref}`fig-feature-selection` rende evidente è fra
*misurare* e *decidere*. Una classifica di importanza è un fatto misurabile;
la soglia è una scelta, e va giustificata con qualcosa d'altro (il costo di
raccogliere una colonna, un vincolo di interpretabilità, una prova che il
modello ridotto non peggiora).

Passiamo agli strumenti che interrogano un modello già addestrato, quale che
sia. La prima domanda, la più naturale, è: **su quali colonne si regge?**
Vogliamo una classifica delle feature per quanto contano nelle predizioni.
Cominciamo dal metodo più generale e robusto (la permutazione), perché non
guarda dentro il modello: lo tratta come una scatola chiusa che riceve input e
sputa predizioni.

### Permutation importance

`````{tab} Elementare

L'idea è quasi impertinente: se una colonna conta davvero, allora
**rovinarla** deve far crollare le prestazioni. Prendiamo un modello che
prevede se un cliente restituirà un prestito, e mettiamolo alla prova su 100
clienti mai visti: indovina 90 volte su 100. Ora prendiamo una colonna sola
(il reddito) e ne **rimescoliamo** i valori tra i 100 clienti: ognuno si
ritrova il reddito di qualcun altro. Tutto il resto è intatto, ma quella
colonna è diventata rumore. Riproviamo il modello: ora indovina solo 72 volte.
Ha perso 18 punti *solo* perché gli abbiamo scombinato il reddito: segno che
ci si appoggiava molto. L'importanza del reddito è quel calo,
$90\% - 72\% = 18$ punti.

Rifacciamo lo stesso gioco con una colonna che non c'entra nulla, il colore
preferito: rimescolandola, il modello continua a indovinare 90 volte. Calo
zero, importanza zero. Poiché il rimescolamento è casuale, lo si ripete
qualche volta e si fa la media, per non farsi ingannare da un mescolamento
fortunato. Il bello è che questo trucco funziona con *qualsiasi* modello (una
foresta, una rete, un GAM), perché serve solo poterlo interrogare.

`````

`````{tab} Superiore

Formalizziamo. Sia $f$ il modello addestrato e
$e_{\text{orig}} = \mathcal{L}(f, D)$ il suo errore (o l'opposto di uno
*score*: MSE in regressione, $1-\text{acc}$ in classificazione) su un insieme
di valutazione $D = (X, y)$. Per la feature $j$ si costruisce $X_{\pi_j}$,
copia di $X$ in cui i valori della **sola colonna $j$** sono permutati
casualmente lungo le righe (rompendo il legame tra $x_j$ e $y$ ma
preservandone la distribuzione marginale) e si misura
$e_{\pi_j} = \mathcal{L}(f, (X_{\pi_j}, y))$. L'importanza è il peggioramento

$$
\mathrm{FI}_j = \frac{1}{K}\sum_{k=1}^{K} e_{\pi_j}^{(k)} - e_{\text{orig}},
$$

media su $K$ permutazioni indipendenti (in `scikit-learn`, `n_repeats`), che
fornisce anche una deviazione standard. Introdotta da Breiman con le foreste
casuali {cite}`breiman2001random` e in seguito formalizzata da Fisher, Rudin e
Dominici (2019) come *model reliance* (nella loro variante il rapporto
$e_{\pi_j}/e_{\text{orig}}$ anziché la differenza) è **model-agnostic**:
richiede solo il forward del modello e un insieme etichettato.

Due accortezze. La misura va calcolata su dati **held-out**: sul *training* essa
racconta quanto il modello si è appoggiato a $x_j$ per memorizzare, non quanto
quella feature aiuti a generalizzare. E le feature **correlate** portano due
guai distinti, che conviene non confondere. Il primo: il modello recupera
l'informazione dalla colonna gemella non permutata, e l'importanza, spartita
fra le due, risulta *sottostimata*. Il secondo: la permutazione crea
combinazioni irrealistiche (un'altezza da adulto con un peso da bambino) su
cui il modello viene interrogato fuori dal supporto dei dati, e l'errore così
gonfiato può *sovrastimare* l'importanza delle feature coinvolte (Hooker,
Mentch e Zhou, 2021): la stessa patologia di estrapolazione che ritroveremo
nel PDP.

`````

### Importanza da impurità (e il suo bias)

Le foreste casuali offrono gratis una seconda misura, la **mean decrease in
impurity** (MDI): quanto ogni feature, sommando su tutti gli alberi, ha
ridotto l'impurità (Gini o entropia) negli split in cui compare. È l'attributo
`feature_importances_` che abbiamo già incontrato nel capitolo sugli alberi e
gli ensemble. È rapidissima (si calcola durante l'addestramento) ma va letta
con prudenza, per una ragione che vale la pena rendere esplicita.

`````{tab} Elementare

L'importanza da impurità premia le feature che l'albero *usa spesso* per
tagliare. Il problema è che una feature con tanti valori diversi (un'età
precisa al giorno, un importo in centesimi) offre all'albero un'enorme
quantità di soglie tra cui scegliere, e con così tante possibilità ne trova
quasi sempre una che, per puro caso, separa un po' i dati. Così accumula
«meriti» anche quando non porta vera informazione. Una feature con pochi
valori (sì/no, tre categorie) parte invece svantaggiata: ha poche soglie da
provare.

Il risultato è che l'importanza da impurità tende a **gonfiare** le feature
continue o con molte categorie e a **sminuire** quelle a pochi valori: un
difetto strutturale, non del singolo dataset. Per una classifica di cui
fidarsi, meglio la permutazione, misurata su dati che il modello non ha mai
visto.

`````

`````{tab} Superiore

Il bias della MDI è verso le feature ad **alta cardinalità** e quelle
**continue**. La causa è combinatoria: il numero di split candidati cresce con
il numero di valori distinti, e massimizzare la riduzione d'impurità su molti
tagli equivale a un test statistico con molte comparazioni (una feature
puramente casuale ma continua ottiene, in aspettazione, un guadagno positivo
per sovradattamento locale). La stessa documentazione di `scikit-learn`
avverte che `feature_importances_` è calcolata sul *training set* e può
risultare fuorviante proprio su queste feature.

Rispetto alla permutation importance, la MDI ha due svantaggi: è legata alla
struttura interna del modello (vale solo per gli alberi) ed è misurata sui dati
di addestramento. La permutazione, calcolata su un *hold-out*, è model-agnostic
e riflette la generalizzazione; è la stima che il capitolo sugli ensemble già
raccomandava di preferire. Entrambe, comunque, restano misure di importanza
**globale**: dicono quanto una feature conta *in media su tutto il dataset*, non
per la singola predizione.

`````

## Come agisce una feature: PDP e ICE

Sapere *quanto* una feature conta non dice *come* agisce: se il prezzo salga o
scenda con la metratura, se l'effetto sia lineare o si spenga oltre una soglia.
Per questo servono i grafici degli **effetti**, che tracciano la forma della
relazione tra una feature e la predizione.

`````{tab} Elementare

Il **Partial Dependence Plot** (PDP) risponde a: «tenendo tutto il resto com'è,
come cambia in media la predizione se muovo *questa* feature?». Immagina di
prendere l'intero elenco di clienti e riscrivere a tutti la stessa età, poniamo
40 anni, lasciando invariato tutto il resto; calcoli le predizioni e ne fai la
media. Poi rifai con 41 anni, 42, e così via. Unendo i punti ottieni una curva:
l'effetto medio dell'età. È come chiedere a tutta la popolazione «e se aveste
tutti 40 anni?», poi «e se ne aveste 41?», misurando come si sposta la media.

C'è un limite: la media può nascondere storie opposte. Se l'età fa salire la
predizione per metà dei clienti e scendere per l'altra metà, la curva media
resta piatta e ti fa credere che l'età non conti. Il rimedio è la curva
**ICE** (*Individual Conditional Expectation*): invece della sola media,
disegni *una curva per ogni cliente*. Un fascio di curve che vanno in direzioni
diverse rivela subito che l'effetto non è uguale per tutti.

`````

`````{tab} Superiore

Per la feature $j$, la **partial dependence** è l'attesa della predizione
marginalizzando sulle altre feature $X_{-j}$, stimata sul dataset come

$$
\mathrm{PD}_j(v) = \frac{1}{m}\sum_{i=1}^{m} f\!\big(v,\, X_{-j}^{(i)}\big),
$$

dove si fissa $x_j = v$ e si mediano le predizioni su tutti gli esempi
(Friedman, 2001). La curva **ICE** è la stessa quantità *prima* di mediare:
$f(v, X_{-j}^{(i)})$ per il singolo esempio $i$ (Goldstein et al., 2015). Il PDP
è dunque la media verticale del fascio di ICE; quando le curve ICE si
sventagliano, un effetto medio piatto maschera **interazioni** o eterogeneità.

Il difetto profondo del PDP è l'**estrapolazione con feature correlate**:
fissare $x_j = v$ mentre si tengono i valori reali di $X_{-j}$ genera punti
$(v, X_{-j}^{(i)})$ implausibili (altezza 2 m con peso 50 kg) su cui il
modello viene interrogato fuori dal supporto dei dati, producendo curve
fuorvianti. L'**Accumulated Local Effects** (ALE) di Apley e Zhu (2020)
corregge il tiro: invece di marginalizzare su tutta la distribuzione, media le
*differenze* di predizione entro piccoli intervalli di $x_j$, usando la
distribuzione **condizionata** e restando così nelle regioni densamente
popolate. È la scelta da preferire quando le feature sono marcatamente
correlate.

`````

## In pratica: permutazione contro impurità

Mettiamo a confronto le due misure globali su un caso reale. Il dataset
`diabetes` di `scikit-learn` raccoglie 442 pazienti diabetici, dieci indicatori
clinici (età, sesso, indice di massa corporea `bmi`, pressione `bp`, sei valori
ematici `s1`–`s6`) e, come target, la progressione della malattia a un anno.
Addestriamo una foresta casuale e chiediamo a entrambe le tecniche quali feature
contano.

```python
import numpy as np
from sklearn.datasets import load_diabetes
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.inspection import permutation_importance

dati = load_diabetes()
X, y, nomi = dati.data, dati.target, dati.feature_names
X_tr, X_te, y_tr, y_te = train_test_split(X, y, test_size=0.3, random_state=0)

rf = RandomForestRegressor(n_estimators=300, random_state=0)
rf.fit(X_tr, y_tr)
print("R^2 sul test:", round(rf.score(X_te, y_te), 3))  # -> 0.308

# Importanza da permutazione, misurata sul TEST (10 mescolamenti per feature)
pi = permutation_importance(rf, X_te, y_te, n_repeats=10, random_state=0)

print("feature   perm-import         (impurita)")
for i in np.argsort(pi.importances_mean)[::-1]:      # dalla piu importante
    print(f"{nomi[i]:>5}   {pi.importances_mean[i]:+.3f} +/- "
          f"{pi.importances_std[i]:.3f}   {rf.feature_importances_[i]:.3f}")
```

L'output ordina le feature per importanza di permutazione e affianca, nell'ultima
colonna, quella da impurità:

```text
feature   perm-import         (impurita)
   s5   +0.197 +/- 0.062   0.302
  bmi   +0.174 +/- 0.018   0.303
   bp   +0.034 +/- 0.012   0.098
  sex   +0.006 +/- 0.003   0.009
  age   +0.005 +/- 0.012   0.049
   s2   +0.003 +/- 0.010   0.049
   s6   -0.001 +/- 0.010   0.056
   s1   -0.004 +/- 0.007   0.044
   s4   -0.004 +/- 0.008   0.027
   s3   -0.012 +/- 0.013   0.064
```

Le due misure concordano sull'essenziale: `bmi` e `s5` (un indice metabolico)
dominano, `bp` le segue, il resto conta poco. Ma emergono anche le differenze
attese. Le feature in fondo alla classifica hanno importanza di permutazione
lievemente **negativa**: rimescolarle *migliora* di un soffio il test, cioè il
modello vi si appoggiava solo per rumore; un'informazione onesta che la misura
da impurità, sempre positiva per costruzione, non può darti. E l'impurità
assegna a `s3` o `s6` valori non trascurabili ($\approx 0{,}06$) benché la
permutazione li dichiari inutili: è il bias verso le feature continue in
azione. Con un $R^2$ di circa $0{,}31$ il modello, per inciso, spiega solo una
parte della variabilità: l'importanza descrive *questo* modello, non la verità
clinica.

## Che una feature conti, non come, né perché

Chiudiamo con l'avvertenza più importante, la stessa della storia degli
asmatici. L'importanza delle feature (di permutazione o da impurità) dice
**che** una colonna pesa sulle predizioni del modello. Non dice **come**
agisce (per quello servono PDP, ICE, ALE), non dice se l'effetto sia lo stesso
per tutti (per quello servono i metodi locali del prossimo tratto del
capitolo), e soprattutto **non dice che sia causale**. Il reddito può
risultare importante perché è una spia del quartiere, e il quartiere del vero
fattore in gioco. Il modello riflette le correlazioni nei suoi dati di
addestramento, non i meccanismi del mondo. Confondere «feature importante per
il modello» con «causa del fenomeno» è l'errore che trasforma uno strumento di
*debug* in una fonte di decisioni sbagliate. L'interpretabilità apre la
scatola: sta a noi non leggerci dentro più di quel che c'è.

`````{tab} Elementare

```{admonition} Da ricordare
:class: important
- I **modelli trasparenti** (lineari e logistici, alberi, **modelli additivi
  generalizzati**, che disegnano una curva leggibile per ogni caratteristica,
  sistemi a regole) sono già la propria spiegazione: nel modello che stima il
  prezzo di una casa ogni coefficiente è un cartellino col prezzo appeso a una
  caratteristica, e la predizione si legge come una ricevuta, voce per voce; in
  un albero la spiegazione è il percorso di domande che porta alla risposta. Il
  presunto scambio fra accuratezza e chiarezza **non vale sempre**, e sui dati
  a righe e colonne spesso non vale affatto.
- La **permutation importance** (Breiman, 2001) rimescola i valori di una sola
  colonna e guarda quanto peggiora il modello: se rimescolando il reddito le
  risposte giuste scendono dal $90\%$ al $72\%$, quella colonna vale 18 punti.
  Funziona con qualunque modello, va misurata su dati che il modello non ha mai
  visto in addestramento e ripetuta più volte, facendo la media.
- L'importanza **da impurità** degli alberi arriva gratis con l'addestramento
  ma è **distorta**: premia le colonne con tanti valori diversi (che offrono
  moltissime soglie fra cui scegliere) e penalizza quelle con due o tre valori,
  ed è calcolata sui dati di addestramento. Meglio fidarsi della permutazione.
- Sapere quanto una colonna conta non dice **come** agisce. Il **PDP** riscrive
  a tutti lo stesso valore («e se aveste tutti quarant'anni?») e fa la media
  delle predizioni; l'**ICE** disegna una curva per ogni esempio e rivela i
  casi in cui l'effetto è opposto da persona a persona e la media lo nasconde.
  Attenzione quando due colonne vanno sempre insieme (l'altezza e il peso, per
  dire): riscrivendone una sola, il PDP finisce per chiedere al modello cosa
  pensa di persone che non esistono, alte due metri e pesanti cinquanta chili,
  e la curva che ne esce inganna.
- L'importanza dice **che** una colonna pesa sulle predizioni, non come agisce
  né che ne sia la **causa**: il reddito può contare solo perché fa da spia del
  quartiere. È l'errore della regola sugli asmatici; il panorama completo è nel
  manuale di Molnar.
```

`````

`````{tab} Superiore

```{admonition} Da ricordare
:class: important
- I **modelli trasparenti** (lineari/logistici, alberi, GAM, regole) sono la
  propria spiegazione: nella regressione lineare ogni coefficiente $w_j$ è
  l'effetto marginale della feature $j$. Il presunto compromesso
  accuratezza/interpretabilità **non vale sempre**, specie sui dati tabellari.
- La **permutation importance** {cite}`breiman2001random` mescola i valori di
  una sola colonna e misura il **calo** di performance ($\mathrm{FI}_j =
  e_{\pi_j} - e_{\text{orig}}$): è **model-agnostic**, va calcolata su dati
  **held-out** e mediata su più permutazioni.
- L'importanza da **impurità** (MDI) negli alberi è gratis ma **distorta**:
  gonfia le feature continue e ad alta cardinalità, ed è misurata sul training.
  Preferire la permutazione.
- **PDP** mostra l'effetto marginale *medio* di una feature, **ICE** una curva
  per istanza (rivela le interazioni che il PDP media via); con feature
  **correlate** il PDP estrapola e inganna: meglio **ALE**.
- L'importanza dice **che** una feature conta, non **come** né se è **causale**.
  Correlazione nel modello non è causazione nel mondo. Panoramica completa in
  Molnar {cite}`molnar2022interpretable`.
```

`````
