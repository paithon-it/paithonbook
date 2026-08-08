# Componenti e modelli classici: da ARIMA a Holt-Winters

Nel 1970 George Box e Gwilym Jenkins misero in appendice al loro libro una
tabella di numeri che sarebbe diventata la «cavia» di generazioni di
statistici: i passeggeri mensili delle linee aeree internazionali dal 1949 al
1960 {cite}`box2015time`. A guardarla, quella serie racconta tre storie
sovrapposte. Una **crescita** costante, decennio del boom dei voli: ogni anno
si vola più dell'anno prima. Un **respiro stagionale**: ogni estate un picco,
ogni inverno un avvallamento, puntuali come le stagioni che li causano. E,
sopra a tutto, un **tremolio** irregolare che nessuna regola spiega. Box e
Jenkins notarono anche un dettaglio decisivo: le oscillazioni estive non erano
sempre alte uguali, ma *crescevano* insieme al livello dei voli. È da qui che
parte ogni modello classico di serie temporale: imparare a leggere quelle tre
storie separatamente prima di provare a prevederle.

## Scomporre una serie: trend, stagionalità, residuo

La prima mossa, più vecchia dei calcolatori, è la **decomposizione**: separare
una serie nei suoi ingredienti. Ce ne sono tre, e la {numref}`fig-serie-decomposizione`
li mostra impilati uno sotto l'altro sullo stesso asse del tempo.

```{figure} ../figures/serie-decomposizione.svg
:name: fig-serie-decomposizione
:alt: Quattro pannelli impilati che condividono l'asse del tempo. Dall'alto la serie osservata con tendenza e oscillazione, poi il trend come linea liscia crescente, poi la stagionalità come onda periodica regolare, infine il residuo come piccole barrette attorno allo zero.
:width: 100%

La serie osservata (in alto) come somma di tre parti: un **trend** di fondo, una
**stagionalità** che si ripete a intervalli regolari e un **residuo** irregolare
che oscilla attorno allo zero.
```

`````{tab} Elementare

Pensa alla bolletta della luce. Dentro quel numero ci sono tre cose diverse.
C'è una **parte fissa**, il canone, che cambia poco e semmai cresce piano di
anno in anno: è il *trend*, la direzione di fondo. C'è una **parte
stagionale**: d'estate il condizionatore, d'inverno le luci accese di più (un
su e giù che torna uguale ogni anno). E poi c'è l'**imprevisto**: il mese che
sei stato in ferie e hai consumato meno, l'amico ospite che ha lasciato tutto
acceso; piccoli scarti che non seguono nessuna regola. È il *residuo*.

Scomporre una serie vuol dire fare esattamente questo: guardare la bolletta e
dire «di questi 90 euro, 60 sono il canone di base, 25 sono la stagione calda,
e 5 sono capitati così». Una volta separati i tre pezzi, ciascuno diventa più
facile da capire e da prevedere: il canone lo estrapoli, la stagione la ripeti,
e sull'imprevisto puoi solo dire quanto è grande di solito.

`````

`````{tab} Superiore

Si assume che la serie osservata $x_t$ sia composta da tre componenti latenti:
un **trend-ciclo** $T_t$, una **stagionalità** $S_t$ di periodo $m$ (12 per dati
mensili, 4 per trimestrali) e un **residuo** $R_t$. Le due forme canoniche sono
il modello **additivo** e quello **moltiplicativo**:

$$
x_t = T_t + S_t + R_t
\qquad\text{oppure}\qquad
x_t = T_t \times S_t \times R_t .
$$

Nell'additivo l'ampiezza della stagionalità è *costante* in valore assoluto:
l'estate aggiunge sempre lo stesso numero di unità, qualunque sia il livello.
Nel moltiplicativo l'ampiezza è *proporzionale* al livello: l'estate aggiunge
una percentuale, quindi cresce con la serie (proprio come i passeggeri delle
linee aeree). Un modello moltiplicativo si linearizza prendendo il logaritmo,
$\log x_t = \log T_t + \log S_t + \log R_t$, che riporta al caso additivo. Le
stime pratiche di $T_t$, $S_t$, $R_t$ si ottengono con medie mobili centrate
(classical decomposition) o con metodi più robusti come STL
{cite}`hyndman2021forecasting`.

`````

Un esempio minuscolo rende concreta la differenza. Prendiamo le vendite di
gelato di una gelateria in quattro trimestri (in migliaia di euro):

$$
x = (\underbrace{20}_{\text{inverno}},\ \underbrace{45}_{\text{primav.}},\ \underbrace{80}_{\text{estate}},\ \underbrace{35}_{\text{autunno}}).
$$

Il **trend** di quest'anno, in prima approssimazione, è la media annuale,
$\bar{x} = (20+45+80+35)/4 = 45$. La **stagionalità additiva** è ciò che ogni
trimestre aggiunge o toglie rispetto alla media:

$$
S^{\text{add}} = (20-45,\ 45-45,\ 80-45,\ 35-45) = (-25,\ 0,\ +35,\ -10),
$$

e questi scarti sommano a zero, com'è giusto per una stagionalità additiva. La
**stagionalità moltiplicativa** è invece il rapporto rispetto al livello:

$$
S^{\text{mol}} = \left(\tfrac{20}{45},\ \tfrac{45}{45},\ \tfrac{80}{45},\ \tfrac{35}{45}\right)
\approx (0{,}44,\ 1{,}00,\ 1{,}78,\ 0{,}78),
$$

fattori che stavolta hanno **media** $1$. La lettura è diversa: l'additivo dice
«d'estate si vendono 35 mila euro *in più* del solito»; il moltiplicativo dice
«d'estate si vende il 78% *in più*». Se l'anno prossimo la gelateria raddoppia
il giro d'affari, il modello additivo continuerebbe a prevedere +35 mila,
sottostimando l'estate; quello moltiplicativo prevede +78%, cioè uno scarto
raddoppiato. Il **residuo** è ciò che resta dopo aver tolto trend e stagionalità:
se l'estate avesse fruttato $82$ invece di $80$, il residuo di quel trimestre
sarebbe $82 - (45 + 35) = 2$.

## Stazionarietà e differenziazione

L'introduzione al capitolo ha già dato la definizione: una serie è
**stazionaria** (in senso debole) quando media, varianza e autocovarianza non
dipendono dal tempo. Qui interessa il *perché* quasi tutti i modelli classici la
pretendono. La ragione è semplice: un modello stima pochi parametri e li assume
validi per tutta la serie, passato e futuro. Se la media scivola verso l'alto o
l'ampiezza delle oscillazioni cambia, quei parametri descrivono un pezzo di
serie e ne sbagliano un altro. Rendere la serie stazionaria significa toglierle
di dosso trend e stagionalità, così che ciò che resta *balli sempre allo stesso
modo*.

Lo strumento principe è la **differenziazione**: sostituire ogni valore con la
sua variazione rispetto al precedente,

$$
\nabla x_t = x_t - x_{t-1}.
$$

Toglie una tendenza lineare in un colpo solo. Sulla serie
$100, 110, 120, 130$, che cresce di $10$ a ogni passo, la differenziata è
$10, 10, 10$: il trend è sparito, resta una costante. Per la stagionalità si
usa la **differenziazione stagionale** $\nabla_m x_t = x_t - x_{t-m}$, che
sottrae il valore dello stesso periodo del ciclo precedente (mese contro
stesso mese dell'anno prima). Prima di modellare, si differenzia quanto basta
a stabilizzare la serie, e non di più, per non introdurre rumore inutile.

Come si *diagnostica* la struttura che resta? Con due grafici che sono il pane
quotidiano dell'analista di serie temporali: l'**ACF** e la **PACF**.

`````{tab} Elementare

Immagina di confrontare la serie con una sua copia fatta scivolare indietro nel
tempo di un passo, di due, di tre. Ogni volta ti chiedi: quanto si somigliano?
La risposta, passo per passo, è la **funzione di autocorrelazione** (ACF): un
grafico a barre che dice quanto oggi assomiglia a ieri, a l'altro ieri, e così
via. Se le barre restano alte a lungo, la serie ha una memoria lunga.

C'è però un inganno. Se oggi somiglia a ieri e ieri somigliava a l'altro ieri,
allora oggi somiglierà a l'altro ieri *di rimbalzo*, anche senza un legame
diretto. La **PACF** (autocorrelazione parziale) toglie questo effetto a catena:
misura quanto oggi dipende dal valore di tre giorni fa *una volta scontato* ciò
che passa attraverso ieri e l'altro ieri. È la differenza fra «il nonno somiglia
al nipote» e «il nonno somiglia al nipote al netto del padre».

`````

`````{tab} Superiore

L'**autocorrelazione** a ritardo $k$ è

$$
\rho_k = \frac{\gamma(k)}{\gamma(0)}
= \frac{\mathrm{Cov}(x_t,\, x_{t-k})}{\mathrm{Var}(x_t)},
$$

dove $\gamma(k)$ è l'autocovarianza; la sequenza $\{\rho_k\}$ è l'ACF. La
**PACF** $\phi_{kk}$ è invece la correlazione fra $x_t$ e $x_{t-k}$ dopo aver
rimosso la dipendenza lineare dai ritardi intermedi $x_{t-1},\dots,x_{t-k+1}$;
coincide con l'ultimo coefficiente della regressione di $x_t$ su quei $k$
ritardi. Le due funzioni sono la bussola dell'identificazione di Box-Jenkins,
perché hanno firme complementari: un processo **AR($p$)** ha PACF che si
**annulla** dopo il ritardo $p$ e ACF che decade gradualmente; un processo
**MA($q$)** ha ACF che si **annulla** dopo il ritardo $q$ e PACF che decade.
Leggere dove le barre «cadono nel rumore» suggerisce gli ordini $p$ e $q$ da
provare {cite}`box2015time`.

`````

## Autoregressione: AR($p$)

Il modello più naturale per una serie con memoria dice: il prossimo valore è una
combinazione dei valori appena passati, più una spinta casuale.

`````{tab} Elementare

Domani la temperatura sarà simile a quella di oggi, con una correzione. Se oggi
fa più caldo della media di stagione, è probabile che anche domani sia sopra la
media, ma un po' meno: il caldo «rientra» piano verso il normale. Un modello
**autoregressivo** cattura proprio questo: prende gli ultimi valori, li pesa, li
somma, e aggiunge un pizzico di imprevedibile per il resto. «Auto-regressivo»
vuol dire che la serie fa da predittore *a sé stessa*: guarda il proprio
passato, non variabili esterne.

Il numero di passati che guarda è l'ordine. Un AR(1) guarda solo ieri; un
AR(2) guarda ieri e l'altro ieri. Più passati includi, più la memoria del
modello si allunga, ma anche più parametri devi stimare da una serie che è pur
sempre lunga un tanto.

`````

`````{tab} Superiore

Un processo **autoregressivo di ordine $p$**, AR($p$), è definito da

$$
x_t = c + \phi_1 x_{t-1} + \phi_2 x_{t-2} + \dots + \phi_p x_{t-p} + \varepsilon_t,
$$

dove $\phi_1,\dots,\phi_p$ sono i coefficienti autoregressivi, $c$ una
costante e $\varepsilon_t$ è **rumore bianco**: una sequenza a media nulla,
varianza $\sigma^2$ costante e incorrelata nel tempo. Il valore atteso
condizionato al passato è
$\mathbb{E}[x_t \mid x_{t-1},\dots] = c + \sum_{i=1}^p \phi_i x_{t-i}$: la
parte prevedibile. La stazionarietà richiede che le radici del polinomio
caratteristico $1 - \phi_1 z - \dots - \phi_p z^p$ stiano fuori dal cerchio
unitario; per l'AR(1) questo si riduce alla condizione $|\phi_1| < 1$. In quel
caso la media di lungo periodo è $\mu = c/(1-\phi_1)$ e la serie vi ritorna
dopo ogni scossa.

`````

Vediamolo con i numeri su un AR(1), $x_t = c + \phi\,x_{t-1} + \varepsilon_t$, con
$c = 4$ e $\phi = 0{,}6$. La media di lungo periodo è $\mu = 4/(1-0{,}6) = 10$.
Partiamo da un valore alto, $x_0 = 20$, e seguiamo la parte prevedibile
(trascurando il rumore, $\varepsilon_t = 0$):

$$
\begin{aligned}
x_1 &= 4 + 0{,}6 \cdot 20 = 16, \\
x_2 &= 4 + 0{,}6 \cdot 16 = 13{,}6, \\
x_3 &= 4 + 0{,}6 \cdot 13{,}6 = 12{,}16.
\end{aligned}
$$

La serie scivola $20 \to 16 \to 13{,}6 \to 12{,}16$, avvicinandosi a $10$ a ogni
passo: è la **mean reversion**, il rientro verso la media che $|\phi| < 1$
garantisce. Il rumore, nella realtà, la scompiglia continuamente, ma la spinta
di fondo resta sempre quella verso $\mu$.

## Da AR ad ARIMA: media mobile, integrazione, stagionalità

L'AR è metà della storia. L'altra metà guarda non ai valori passati, ma agli
**urti** passati: è il modello a **media mobile**, sigla MA. Da non confondere
con la «media mobile» usata per lisciare un grafico: qui è una media *degli
imprevisti*, non dei valori.

`````{tab} Elementare

Pensa a un urto imprevisto: una gita scolastica che svuota la gelateria, uno
sciopero che blocca i voli. L'effetto non si esaurisce il giorno stesso: si fa
sentire ancora domani, un po' meno dopodomani, e poi svanisce. Un modello a
media mobile dice proprio questo: il valore di oggi è il livello normale, più
la sorpresa di oggi, più l'eco (sempre più debole) delle sorprese degli ultimi
giorni.

Le due memorie si possono usare insieme: quella dei valori (l'AR appena visto)
e quella degli urti (il MA). E siccome le serie vere hanno quasi sempre una
tendenza, prima la si raddrizza col trucco già incontrato, sostituire ogni
valore con la variazione rispetto al giorno prima, e poi si modella ciò che
resta. Il tutto insieme si chiama **ARIMA**, il modello di punta di Box e
Jenkins: dietro la sigla ci sono solo tre conteggi, quanti valori passati
guardare, quante volte raddrizzare la serie, per quanti giorni far durare
l'eco degli urti. Se c'è anche una stagionalità, si rifà lo stesso gioco sul
calendario (dicembre si confronta con lo scorso dicembre): è la variante
**SARIMA**. Le sigle piene di lettere e numeri che si incontrano nei manuali
non sono che questi conteggi messi in fila.

`````

`````{tab} Superiore

Il modello a **media mobile** MA($q$) è

$$
x_t = \mu + \varepsilon_t + \theta_1 \varepsilon_{t-1} + \dots + \theta_q \varepsilon_{t-q},
$$

dove $\varepsilon_{t-i}$ sono le scosse casuali dei passi precedenti e
$\theta_1,\dots,\theta_q$ i loro pesi: un urto di oggi non si esaurisce
subito, ma continua a farsi sentire per $q$ passi prima di svanire.

Mettendo insieme le due idee si ottiene l'**ARMA($p,q$)**, che spiega il valore
odierno con $p$ valori passati e $q$ errori passati. Ma l'ARMA vive solo su
serie stazionarie, e le serie vere quasi mai lo sono. La soluzione di Box e
Jenkins è incorporare la differenziazione nel modello stesso: nasce
l'**ARIMA($p,d,q$)** {cite}`box2015time`. Le tre lettere:

- **AR($p$)**, l'ordine autoregressivo, quanti valori passati;
- **I($d$)**, *integrated*, quante volte si differenzia la serie per renderla
  stazionaria ($d=1$ toglie un trend lineare, $d=2$ una curvatura);
- **MA($q$)**: l'ordine a media mobile, quanti errori passati.

In pratica si differenzia la serie $d$ volte, si adatta un ARMA($p,q$) al
risultato, e si «re-integra» sommando all'indietro per tornare alla scala
originale. Quando la serie ha una stagionalità marcata, si aggiunge un secondo
blocco di termini che agiscono al ritardo stagionale $m$: è il
**SARIMA($p,d,q$)($P,D,Q$)$_m$**, dove le lettere maiuscole $P,D,Q$ sono gli
ordini AR, di differenziazione e MA *stagionali*, e $m$ è la lunghezza del ciclo.
Un SARIMA$(1,1,1)(1,1,1)_{12}$ è, ancora oggi, un ottimo punto di partenza per
una serie mensile con trend e stagionalità annuale.

`````

## Lisciamento esponenziale: da SES a Holt-Winters

La famiglia ARIMA modella la memoria della serie in modo esplicito. Una seconda
famiglia, altrettanto classica, la modella in modo implicito e leggerissimo: il
**lisciamento esponenziale** (*exponential smoothing*). L'idea sta in una riga:
la previsione è una media di tutto il passato, in cui i valori recenti pesano
di più e quelli lontani sempre meno.

`````{tab} Elementare

Per indovinare le vendite di domani potresti fare la media di tutti i giorni
passati. Ma il mese scorso conta davvero quanto ieri? No. Il lisciamento
esponenziale fa una media *pesata*, in cui ieri pesa molto, l'altro ieri un po'
meno, la settimana scorsa ancora meno, e così via a scendere. A ogni passo
indietro il peso si riduce di una stessa frazione, come l'eco di un suono che
si spegne.

La versione base tiene conto solo del **livello** (dove sta la serie ora). Ma se
la serie sale con costanza, ti serve anche una stima di *quanto* sale: aggiungi
il **trend**. E se ha un respiro stagionale, aggiungi anche quello. Sono i tre
gradini: livello, poi livello + trend, poi livello + trend + stagionalità. Con
tutti e tre, il metodo si chiama Holt-Winters, dai nomi di chi lo mise a punto
alla fine degli anni Cinquanta.

`````

`````{tab} Superiore

Il **lisciamento esponenziale semplice** (SES) tiene solo il livello $\ell_t$ e
lo aggiorna a ogni passo come media pesata fra l'osservazione nuova e la stima
vecchia:

$$
\ell_t = \alpha\, x_t + (1-\alpha)\,\ell_{t-1},
\qquad
\hat{x}_{t+1} = \ell_t,
$$

con $\alpha \in (0,1)$ il fattore di lisciamento. Srotolando la ricorsione
fino all'inizio della serie,
$\ell_t = \alpha \sum_{j=0}^{t-1} (1-\alpha)^j x_{t-j} + (1-\alpha)^t \ell_0$:
i pesi $\alpha(1-\alpha)^j$ **decadono esponenzialmente** e, per $t$ grande
(quando il peso residuo dell'inizializzazione $\ell_0$ è ormai trascurabile),
la loro somma tende a $1$. Con $\alpha = 0{,}3$ valgono
$0{,}30,\ 0{,}21,\ 0{,}147,\ \dots$

Il metodo di **Holt** aggiunge una componente di trend $b_t$:

$$
\ell_t = \alpha\, x_t + (1-\alpha)(\ell_{t-1} + b_{t-1}),
\qquad
b_t = \beta\,(\ell_t - \ell_{t-1}) + (1-\beta)\,b_{t-1},
$$

con previsione a $h$ passi $\hat{x}_{t+h} = \ell_t + h\,b_t$. **Holt-Winters**
aggiunge infine la stagionalità $s_t$ di periodo $m$ (forma additiva):

$$
\begin{aligned}
\ell_t &= \alpha\,(x_t - s_{t-m}) + (1-\alpha)(\ell_{t-1}+b_{t-1}), \\
b_t &= \beta\,(\ell_t - \ell_{t-1}) + (1-\beta)\,b_{t-1}, \\
s_t &= \gamma\,(x_t - \ell_{t-1} - b_{t-1}) + (1-\gamma)\,s_{t-m},
\end{aligned}
\qquad
\hat{x}_{t+h} = \ell_t + h\,b_t + s_{t+h-m(k+1)},
$$

dove $k = \lfloor (h-1)/m \rfloor$: l'indice stagionale ricicla sempre
l'ultimo ciclo stimato, così anche oltre un periodo intero ($h > m$) la
previsione non riferisce mai stagioni non ancora osservate. I tre fattori
$\alpha,\beta,\gamma \in (0,1)$ regolano quanto in fretta livello,
trend e stagionalità si adeguano ai dati nuovi. Questi metodi hanno una veste
moderna nei modelli **ETS** (*Error, Trend, Seasonal*) in forma spazio-stato,
che aggiungono un'interpretazione probabilistica e intervalli di previsione
{cite}`hyndman2021forecasting`.

`````

## Quando i classici bastano (o battono il deep learning)

Verrebbe da pensare che, con le reti neurali del capitolo successivo, questi
modelli di mezzo secolo fa siano roba da manuale di storia. Non è così, e vale
la pena dire perché con onestà. La prova più citata sono le **competizioni M**
dello statistico greco Spyros Makridakis: gare pubbliche in cui decine di
metodi si sfidano su decine di migliaia di serie reali. Il verdetto, ripetuto
edizione dopo edizione, è scomodo per gli entusiasti: i metodi statistici
semplici (ARIMA, Holt-Winters, e loro medie) restano difficilissimi da
battere, e per molti anni hanno superato reti neurali ben più complesse.

Le ragioni sono tre. La **robustezza**: un modello con pochi parametri non ha
molto spazio per adattarsi al rumore, quindi generalizza bene anche quando la
serie è corta o disturbata. La **frugalità di dati**: gran parte delle serie
reali (le vendite mensili di un prodotto, i pazienti di un reparto) hanno
poche decine o centinaia di osservazioni, troppo poche per addestrare una rete
affamata di dati, più che sufficienti per un ARIMA. E l'**interpretabilità**:
un coefficiente $\phi$, una componente stagionale, un intervallo di confidenza
sono oggetti che un analista legge, discute e difende davanti a chi deve
decidere, mentre i pesi di una rete no {cite}`hyndman2021forecasting`. La
regola pratica che ne discende attraversa tutto il forecasting serio: un
modello classico è la **linea di base onesta**. Prima si batte quella, poi si
tira in ballo il deep learning.

## In pratica: stimare un AR(1) ai minimi quadrati

Stimare un AR(1) non richiede librerie sofisticate: è una regressione lineare di
$x_t$ sul suo ritardo $x_{t-1}$. Generiamo una serie dal modello con un $\phi$
noto e verifichiamo di saperlo recuperare, poi facciamo una previsione a un
passo. Tutto in puro NumPy.

```python
import numpy as np

rng = np.random.default_rng(42)

# --- genera una serie dal modello AR(1): x_t = c + phi * x_{t-1} + rumore ---
phi_vero, c_vero, sigma = 0.6, 4.0, 1.0
n = 500
x = np.zeros(n)
x[0] = c_vero / (1 - phi_vero)                 # parte dalla media di lungo periodo (10)
for t in range(1, n):
    x[t] = c_vero + phi_vero * x[t - 1] + rng.normal(0, sigma)

# --- stima ai minimi quadrati: regredisci x_t su [1, x_{t-1}] ---
y = x[1:]                                       # bersaglio: x_t
Xmat = np.column_stack([np.ones(n - 1), x[:-1]])  # colonne: costante e x_{t-1}
beta, *_ = np.linalg.lstsq(Xmat, y, rcond=None)   # risolve i minimi quadrati
c_hat, phi_hat = beta

print(f"phi vero = {phi_vero:.2f}   phi stimato = {phi_hat:.3f}")
print(f"c vero   = {c_vero:.2f}   c stimato   = {c_hat:.3f}")

# --- previsione one-step dopo l'ultima osservazione ---
x_next = c_hat + phi_hat * x[-1]
print(f"ultima osservazione x_T = {x[-1]:.3f}")
print(f"previsione   x_(T+1)    = {x_next:.3f}")
```

Il $\phi$ stimato cade vicino a $0{,}6$ e la costante vicino a $4$: con
cinquecento osservazioni i minimi quadrati ricostruiscono bene i parametri del
processo che ha generato la serie. La previsione a un passo è semplicemente la
formula del modello applicata all'ultimo valore osservato. Da qui in avanti si
può iterare in avanti per orizzonti più lunghi: ricadendo, però, nel problema
dell'accumulo dell'errore visto nell'introduzione al capitolo.

Con questo abbiamo la cassetta degli attrezzi classica: decomposizione per
capire, ARIMA e Holt-Winters per prevedere, ACF e PACF per diagnosticare. La
sezione successiva affronta una domanda che finora abbiamo aggirato (come si
**valida** un modello di serie temporale senza barare col futuro) e come si
trasformano le serie in feature per i modelli tabulari già incontrati nel
capitolo sul Machine Learning.

`````{tab} Elementare

```{admonition} Da ricordare
:class: important
- **Scomporre** una serie vuol dire leggerla come la bolletta della luce: il
  **canone** di fondo (il trend), la **stagione** che torna ogni anno uguale
  (la stagionalità) e l'**imprevisto** che non segue regole (il residuo). La
  stagione può aggiungere sempre la stessa cifra (caso **additivo**) oppure una
  percentuale, e allora cresce insieme al giro d'affari (caso
  **moltiplicativo**): in gelateria, «d'estate 35 mila euro in più» contro
  «d'estate il $78\%$ in più».
- Quasi tutti i modelli classici pretendono una serie **stazionaria**, che balli
  sempre allo stesso modo: la si ottiene sostituendo ogni valore con la
  **variazione** rispetto al precedente, che toglie la tendenza in un colpo
  solo. Per capire che memoria resta si guardano due grafici a barre: l'**ACF**
  (la funzione di autocorrelazione), quanto oggi assomiglia ai giorni passati, e
  la **PACF** (l'autocorrelazione parziale), quanto ci assomiglia al netto degli
  effetti a catena (il nonno e il nipote, scontato il padre). Finché le barre
  restano alte, quel passato pesa ancora; da dove si schiacciano quasi a zero,
  guardare più indietro non serve.
- Ci sono due memorie. Quella dei **valori** passati (l'autoregressione: domani
  somiglia a oggi, con un rientro verso la media) e quella degli **urti**
  passati (la media mobile: lo sciopero si fa sentire ancora domani, meno
  dopodomani). **ARIMA** le usa insieme su una serie già raddrizzata, e dietro
  la sigla ci sono solo tre conteggi; **SARIMA** rifà lo stesso gioco sul
  calendario, confrontando dicembre con lo scorso dicembre
  {cite}`box2015time`.
- Il **lisciamento esponenziale** è una media del passato in cui ieri pesa
  molto e ogni passo indietro pesa una frazione in meno, come l'eco di un suono
  che si spegne. Tre gradini: solo il livello, poi livello più tendenza, poi
  anche la stagione, e con tutti e tre il metodo si chiama Holt-Winters.
- I classici sono **robusti, si accontentano di poche osservazioni e si
  spiegano a chi deve decidere**: nelle competizioni M restano una **linea di
  base** durissima da battere. Prima si supera quella, poi si tira in ballo il
  deep learning {cite}`hyndman2021forecasting`.
```

`````

`````{tab} Superiore

```{admonition} Da ricordare
:class: important
- La **decomposizione** separa una serie in **trend**, **stagionalità** e
  **residuo**, in forma **additiva** ($x_t = T_t + S_t + R_t$, oscillazioni di
  ampiezza costante) o **moltiplicativa** ($x_t = T_t \times S_t \times R_t$,
  ampiezza proporzionale al livello, linearizzabile col logaritmo).
- Quasi tutti i modelli classici richiedono la **stazionarietà**; la si ottiene
  **differenziando** ($\nabla x_t = x_t - x_{t-1}$ toglie un trend). **ACF** e
  **PACF** diagnosticano gli ordini: la PACF si annulla dopo il ritardo $p$ di un
  AR, l'ACF dopo il ritardo $q$ di un MA.
- **AR($p$)** spiega il valore con i $p$ passati; **MA($q$)** con i $q$ errori
  passati; **ARIMA($p,d,q$)** unisce i due sulla serie differenziata $d$ volte,
  e **SARIMA** aggiunge i termini stagionali al ritardo $m$ {cite}`box2015time`.
- Il **lisciamento esponenziale** pesa il passato con pesi che **decadono
  esponenzialmente**: SES (solo livello), Holt (livello + trend), Holt-Winters
  (livello + trend + stagionalità).
- I modelli classici sono **robusti, frugali di dati e interpretabili**: nelle
  competizioni M restano una **linea di base** durissima da battere. Prima si
  supera quella, poi si passa al deep learning {cite}`hyndman2021forecasting`.
```

`````
