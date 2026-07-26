# Il futuro nei numeri: serie temporali e forecasting

Negli anni Settanta dell'Ottocento, in un laboratorio di Glasgow, William
Thomson (che il mondo avrebbe conosciuto come Lord Kelvin) costruì una
macchina di ottone, corde e pulegge che prevedeva le maree. Non con la magia:
osservando anni di misure del livello del mare, Kelvin lo aveva scomposto
nella somma di tante oscillazioni regolari (quella lunare, quella solare,
quelle più sottili) e la macchina, girando una manovella, sommava
meccanicamente quelle onde per *disegnare la marea di un anno intero prima che
accadesse*. È la stessa idea che regge tutto questo capitolo: il futuro si
stima dal passato, purché il passato conservi delle regolarità.

Prevedere è un mestiere antico. Il contadino che legge il cielo per decidere
quando seminare, il mercante che anticipa il prezzo del grano, il meteorologo
che stende le isobare: tutti fanno lo stesso gesto, guardare la storia di un
fenomeno per indovinarne il seguito. La disciplina che formalizza questo gesto
nasce come metodo sistematico nel 1970, quando George Box e Gwilym Jenkins
pubblicano un libro destinato a diventare un classico dell'econometria e
dell'ingegneria {cite}`box2015time`: la loro metodologia (identificare,
stimare, verificare) dà per la prima volta una ricetta ripetibile per
costruire un modello di serie temporale. Mezzo secolo dopo, lo statistico
greco Spyros Makridakis mette alla prova quei metodi su larga scala con le
**competizioni M**, gare pubbliche di previsione su decine di migliaia di
serie reali {cite}`makridakis2020m4`. La lezione che ne esce è tanto tecnica
quanto morale: si prevede, sì, ma con umiltà. Nessun modello domina sempre, e
dichiarare *quanto* siamo incerti conta quanto la previsione stessa.

## Che cos'è una serie temporale

Il punto di partenza è un tipo di dato che finora, nel libro, abbiamo
sottovalutato: dati che arrivano **in ordine**, uno dopo l'altro nel tempo.

`````{tab} Elementare

Immagina un diario in cui ogni riga porta una data: il peso segnato ogni
mattina, la temperatura misurata ogni ora, il numero di scontrini di un
negozio giorno per giorno. Una **serie temporale** è esattamente questo: una
fila di numeri con accanto un orologio.

La differenza con gli altri dati del libro è sottile ma decisiva. Quando
abbiamo parlato di apprendimento supervisionato (riconoscere gatti, filtrare
lo spam), gli esempi erano come palline in un sacchetto: potevi rimescolarle a
piacere senza perdere nulla, l'ordine non contava. Con una serie temporale non
puoi. La temperatura di oggi somiglia a quella di ieri; se mescoli le date,
distruggi proprio l'informazione che ti serve. L'ordine *è* il dato.

`````

`````{tab} Superiore

Una serie temporale è una sequenza di osservazioni $x_1, x_2, \dots, x_T$
indicizzate da un tempo discreto $t$, tipicamente a passo costante (orario,
giornaliero, mensile). Formalmente è la realizzazione di un **processo
stocastico** $\{X_t\}_{t \in \mathbb{Z}}$, cioè una famiglia di variabili
casuali ordinate nel tempo.

Il punto cruciale è che le $X_t$ **non sono indipendenti e identicamente
distribuite** (non i.i.d.). Tutto l'apprendimento supervisionato che abbiamo
incontrato nel capitolo sul Machine Learning poggia, esplicitamente o meno,
sull'ipotesi che gli esempi $(X^{(i)}, y^{(i)})$ siano campionati in modo
indipendente da un'unica distribuzione; è ciò che rende lecito mescolarli e
separare a caso *train* e *test*. Qui l'ipotesi cade due volte: le osservazioni
sono **dipendenti** ($X_t$ è correlata con $X_{t-1}, X_{t-2}, \dots$) e la loro
distribuzione può **cambiare nel tempo**. Ogni tecnica del capitolo nasce per
convivere con questa doppia rottura.

`````

## I problemi che si pongono

«Serie temporale» è la forma dei dati; i compiti che ci si costruisce sopra sono
diversi, e conviene distinguerli subito perché richiedono strumenti e metriche
differenti.

Il più importante è il **forecasting**: stimare i valori *futuri* della serie a
partire dai passati. È l'unico compito in cui la variabile da prevedere è la
stessa serie proiettata in avanti. Attorno ad esso ruotano gli altri:

- **Classificazione di serie**, assegnare un'etichetta a un'intera sequenza:
  un elettrocardiogramma è normale o aritmico? una vibrazione del motore
  segnala un guasto imminente?
- **Rilevamento di anomalie**, individuare i punti in cui la serie si comporta
  in modo inatteso: una frode su una carta, un picco anomalo di traffico, un
  sensore che impazzisce.
- **Imputazione**: ricostruire i valori mancanti *dentro* la serie, quando un
  sensore si è spento per qualche ora e restano dei buchi da riempire.

Il forecasting stesso si declina lungo due assi che ne cambiano la difficoltà.

`````{tab} Elementare

Il primo asse riguarda **quante cose** guardiamo insieme. Se prevedi solo la
temperatura di domani osservando le temperature passate, la serie è
**univariata**: una sola grandezza che scorre. Ma spesso conviene guardarne
tante insieme (temperatura, umidità, pressione), perché si aiutano a vicenda:
è il caso **multivariato**.

Il secondo asse riguarda **quanto lontano** guardiamo. Prevedere solo il
prossimo valore (domani) è un *passo singolo*; prevedere l'intera settimana
che verrà è *a più passi*. E qui c'è una trappola quotidiana: le previsioni
del tempo a un giorno ci azzeccano quasi sempre, quelle a dieci giorni molto
meno. Più ti spingi lontano, più l'errore si accumula: ogni passo eredita
l'incertezza di quelli prima.

`````

`````{tab} Superiore

Nel caso **univariato** la serie è scalare, $x_t \in \mathbb{R}$; nel
**multivariato** è vettoriale, $X_t \in \mathbb{R}^d$, e si vuole sfruttare la
correlazione tra le $d$ componenti. Sul secondo asse, il forecasting
**one-step** stima

$$
\hat{x}_{T+1} = f(x_1, \dots, x_T),
$$

mentre quello **multi-step**, su un orizzonte $h$, stima l'intero blocco
$\hat{x}_{T+1}, \dots, \hat{x}_{T+h}$. Le due strategie principali sono la
previsione **diretta** (un modello per ciascun orizzonte) e quella **ricorsiva**
(un modello one-step riapplicato, alimentando le proprie previsioni come input),

$$
\hat{x}_{T+k} = f(x_1, \dots, x_T, \hat{x}_{T+1}, \dots, \hat{x}_{T+k-1}),
$$

dove $h$ è l'orizzonte e $k$ il passo corrente. La ricorsiva è economica ma
soffre di **error compounding**: l'errore al passo $k$ entra nell'input del
passo $k+1$ e si propaga, ragione per cui l'incertezza cresce con l'orizzonte
{cite}`hyndman2021forecasting`.

`````

## Perché è un problema diverso (e difficile)

Se le serie temporali meritano un capitolo a sé, e non un paragrafo dentro il
Machine Learning, è perché ognuna delle loro proprietà rompe un'assunzione che
altrove davamo per scontata.

La prima è l'**autocorrelazione**: una serie è correlata con sé stessa spostata
nel tempo. Possiamo misurarla direttamente, e vedere quanto sia lontana dal caso
i.i.d. degli altri dataset del libro.

```python
import numpy as np

rng = np.random.default_rng(0)
n = 200
t = np.arange(n)
# serie sintetica: tendenza + stagionalità (periodo 12) + rumore
serie = 0.05 * t + 2.0 * np.sin(2 * np.pi * t / 12) + rng.normal(0, 0.5, n)

def autocorr(x, lag):
    x = x - x.mean()
    return np.sum(x[lag:] * x[:-lag]) / np.sum(x * x)

print(f"autocorrelazione a lag 1:  {autocorr(serie, 1):.3f}")
print(f"autocorrelazione a lag 12: {autocorr(serie, 12):.3f}")

# rimescolando l'ordine, la dipendenza temporale svanisce
mescolata = rng.permutation(serie)
print(f"lag 1 dopo lo shuffle:     {autocorr(mescolata, 1):.3f}")
```

Sulla serie ordinata l'autocorrelazione a un passo è **vicina a 1** (ogni
valore anticipa quasi perfettamente il successivo) e quella a dodici passi
resta alta, perché la stagionalità riporta il fenomeno allo stesso punto del
ciclo. Ma appena rimescoliamo le date, il coefficiente **crolla verso lo
zero**: la permutazione ha cancellato l'unica cosa che rendeva prevedibile la
serie. È il motivo profondo per cui, nel forecasting, **non si può mescolare
futuro e passato**, né nell'addestramento né, soprattutto, nella validazione.
Torneremo su questo punto nella sezione dedicata alla validazione temporale;
per ora basti la regola: si addestra sul passato, si verifica sul futuro, mai
il contrario.

La seconda proprietà è la **non stazionarietà**, ed è quella che manda in crisi
i modelli classici.

`````{tab} Elementare

Pensa a un fiume. Se la sua portata oscilla sempre attorno allo stesso valore
medio, con piene e magre di ampiezza costante, il fiume è «stabile»: chi lo
studia oggi può usare le stesse regole di chi lo studiava vent'anni fa. Questa
stabilità delle *regole statistiche* (la media attorno a cui la serie balla,
l'ampiezza con cui balla) è ciò che i tecnici chiamano **stazionarietà**.

Molte serie vere non sono così. Il prezzo di una casa cresce di decennio in
decennio (la media sale: c'è una **tendenza**), i consumi di gelato salgono
ogni estate e calano ogni inverno (la **stagionalità**), e ogni tanto succede
qualcosa che cambia le regole di colpo, una crisi, una pandemia, una nuova
tecnologia: un **cambio di regime**. Buona parte del lavoro consiste nel
togliere tendenza e stagionalità per riportare la serie a qualcosa di stabile,
su cui i modelli sappiano ragionare.

`````

`````{tab} Superiore

Un processo $\{X_t\}$ è **stazionario in senso debole** (o in covarianza) se i
suoi primi due momenti non dipendono dal tempo:

$$
\mathbb{E}[X_t] = \mu, \qquad
\mathrm{Var}(X_t) = \sigma^2, \qquad
\mathrm{Cov}(X_t, X_{t+k}) = \gamma(k) \;\; \forall t,
$$

cioè media e varianza costanti e autocovarianza $\gamma(k)$ funzione **solo**
del divario $k$ tra due istanti, non della loro posizione assoluta. È
l'ipotesi su cui poggia l'intera famiglia dei modelli ARMA. Le serie reali la
violano in tre modi ricorrenti: una **tendenza** rende $\mu$ variabile nel
tempo, la **stagionalità** rende $\gamma$ periodica, un **cambio di regime**
(rottura strutturale) altera $\mu$, $\sigma^2$ o entrambi da un certo istante
in poi. La strategia standard è ricondurre la serie alla stazionarietà
(tipicamente con la **differenziazione**, $\nabla x_t = x_t - x_{t-1}$, che
elimina una tendenza lineare) prima di modellarla: è la «I» (*integrated*)
dell'ARIMA {cite}`box2015time`. Verificare la stazionarietà è un test
statistico a sé (ADF, KPSS), non un giudizio a occhio.

`````

Autocorrelazione, non stazionarietà, cambi di regime, ordine che conta: sono
quattro facce dello stesso fatto. In una serie temporale l'**indipendenza**
tra gli esempi (la comoda finzione su cui abbiamo costruito il resto del
machine learning supervisionato) semplicemente non c'è, e ogni metodo del
capitolo è un modo diverso di prenderla sul serio.

## Come è organizzato il capitolo

La storia del forecasting è una lunga convivenza tra due famiglie: i metodi
**statistici** classici, trasparenti e sorprendentemente difficili da battere,
e i metodi **neurali**, più affamati di dati ma capaci di catturare pattern
complessi e di condividere ciò che imparano tra migliaia di serie. La grande
lezione empirica delle competizioni M è che la rivalità è meno netta di quanto
sembri: nella M4, con le sue 100.000 serie e 61 metodi in gara, a vincere non
fu né la statistica pura né il deep learning puro, ma un **ibrido** (una rete
ricorrente innestata su un modello di *exponential smoothing*), mentre le
combinazioni di più metodi surclassavano i singoli concorrenti
{cite}`makridakis2020m4`. Il capitolo segue proprio questa parabola, in tre
sezioni:

1. **Componenti e modelli classici**, come scomporre una serie in tendenza,
   stagionalità e residuo, e i due cavalli di battaglia storici: la famiglia
   **ARIMA** di Box e Jenkins e il livellamento esponenziale nella forma
   **Holt-Winters**. Sono ancora oggi la linea di base onesta contro cui
   misurare qualunque metodo più sofisticato.
2. **Validazione temporale e feature**, perché la *k-fold* mescolata che
   usiamo altrove qui è vietata, e come si valida sul tempo (*rolling* e
   *expanding window*, backtesting); e come si trasformano le serie in feature
   (ritardi, medie mobili, indicatori di calendario) per darle in pasto ai
   modelli tabulari già visti nel capitolo sul Machine Learning.
3. **Forecasting neurale**: dalle reti convoluzionali causali (**TCN**) ai
   modelli autoregressivi probabilistici (**DeepAR**), fino ai **Transformer**
   adattati alle serie e ai recenti **foundation model** (come TimesFM o
   Chronos), pre-addestrati su enormi collezioni di serie e capaci di
   prevedere fenomeni mai visti prima.

Attraversa tutte e tre un filo rosso, ereditato da Kelvin e dalle competizioni M:
la previsione seria non è un numero, è un numero *con la sua incertezza*. Un
modello che dice «domani 24 gradi» vale meno di uno che dice «domani tra 22 e 26
gradi, con l'80% di confidenza», perché il secondo sa quanto poco sa.

```{admonition} Da ricordare
:class: important
- Una **serie temporale** è una sequenza di osservazioni ordinate nel tempo; a
  differenza degli esempi i.i.d. del resto del machine learning, i suoi valori
  sono **dipendenti** e l'ordine *è* l'informazione: rimescolarli la distrugge.
- I compiti principali sono **forecasting** (uni/multivariato, a passo singolo o
  a più passi), classificazione di serie, rilevamento di anomalie e imputazione.
  Nel multi-step l'errore si **accumula** con l'orizzonte.
- Ciò che rende il problema difficile è la rottura dell'indipendenza:
  **autocorrelazione**, **non stazionarietà** (tendenza, stagionalità),
  **cambi di regime**. Una serie è **stazionaria** se media, varianza e
  autocovarianza non dipendono dal tempo.
- Nella validazione **non si mescolano futuro e passato**: si addestra sul
  passato e si verifica sul futuro, sempre.
- Il capitolo procede in tre tappe, modelli classici (ARIMA, Holt-Winters),
  validazione temporale e feature, forecasting neurale (TCN, DeepAR,
  Transformer, foundation model), con un filo comune: prevedere significa
  anche **dichiarare la propria incertezza**.
```
