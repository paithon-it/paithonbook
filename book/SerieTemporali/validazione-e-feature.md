# Validare e rappresentare: backtesting e feature temporali

Un analista quantitativo mostra il grafico di un suo modello: rendimento annuo
del 40%, curva che sale liscia come una pista da sci. Lo mette a lavorare sui
soldi veri e nel giro di un mese è in rosso. Cos'è successo? Ricostruendo il
codice, si scopre che tra le variabili in ingresso ce n'era una calcolata
sulla media dell'intero periodo: futuro compreso. Il modello, in fase di
prova, «sapeva» dove sarebbe andato il prezzo. Sul passato era un veggente;
sul futuro, un ciarlatano.

Questa è la trappola numero uno di chi lavora con le serie temporali, e ha un
nome: **leakage**, la fuga di informazione dal futuro verso il passato. La
sezione precedente ci ha dato il vocabolario delle serie: trend, stagionalità,
autocorrelazione. Questa spiega come *valutare* onestamente una previsione e
come *rappresentare* il tempo perché un normale modello tabellare possa
impararlo.

## Perché mescolare i dati è un errore

Nel capitolo sul Machine Learning abbiamo costruito la validazione come un rito:
si divide in train, validation e test, si mischiano gli esempi per non lasciare
ordini spurii, e la **k-fold cross-validation** rimescola più volte per una stima
più stabile. Con le serie temporali quel rimescolare, che altrove è igiene, qui
è veleno.

`````{tab} Elementare

Immagina di allenare uno studente a prevedere il meteo. Gli dai in mano i dati
di tutto l'anno mescolati a caso: alcuni giorni per esercitarsi, altri per
l'esame. Ma tra i giorni d'esame c'è il 3 marzo, e tra quelli d'esercizio il 4 e
il 5 marzo. Lo studente, «esercitandosi» sul 4 e 5, ha di fatto sbirciato cosa
c'era intorno al 3: la sua previsione del 3 marzo sembrerà miracolosa, ma solo
perché ha spiato i giorni vicini nel futuro.

Nel mondo vero non funziona così: quando prevedi domani, hai solo *ieri e
prima*. Non puoi allenarti sui risultati di domani per indovinare domani.
Mescolare i dati di una serie temporale rompe proprio questa regola (mette
futuro e passato nello stesso mucchio) e ti regala un modello che sul foglio
va benissimo e nella realtà crolla.

`````

`````{tab} Superiore

Il problema è che gli esempi di una serie non sono **indipendenti**: sono
ordinati e fortemente autocorrelati. Uno split casuale, o una k-fold con
shuffle, mette nel training istanti $t+1, t+3, \dots$ e nel validation
l'istante $t$: il modello osserva valori *successivi* a quello che deve
prevedere, e sfrutta l'autocorrelazione per «interpolare» all'indietro. La
stima dell'errore che ne esce è sistematicamente ottimista: un caso di *data
leakage*, la stessa fuga di informazione che nel capitolo sulla validazione ci
imponeva di non toccare mai il test.

La regola è netta: **ogni dato usato per addestrare deve precedere nel tempo ogni
dato usato per validare**. Il confine tra train e validation è un istante $t_0$,
non un'estrazione a sorte.

`````

## Validazione temporale: lo split cronologico

La cura è semplice da enunciare: rispettare la freccia del tempo. Si allena sul
passato, si valida sul futuro, mai il contrario. Ma con una serie sola non
possiamo permetterci un unico taglio: butteremmo via troppe informazioni. La
soluzione standard è il **walk-forward** (o *backtesting*), noto anche come
valutazione «su origine mobile» {cite}`hyndman2021forecasting`.

`````{tab} Elementare

L'idea è rifare più volte lo stesso gioco onesto (allena sul prima, prova sul
dopo) spostando ogni volta il confine in avanti. Ci sono due modi.

Con la **finestra espansa** (*expanding*), a ogni giro tieni tutto il passato
disponibile e lo allunghi: prima usi i primi due mesi per prevedere il terzo, poi
i primi tre per prevedere il quarto, e così via. Come uno storico che, più anni
studia, più contesto ha.

Con la **finestra scorrevole** (*rolling*), tieni invece una finestra di
lunghezza fissa che scivola in avanti: sempre, per esempio, gli ultimi dodici
mesi. Utile quando il passato troppo lontano non è più rappresentativo: le
abitudini d'acquisto di dieci anni fa dicono poco su quelle di oggi.

In entrambi i casi il test è **sempre a destra** del train, cioè nel futuro,
come mostra la {numref}`fig-walk-forward-validazione`.

`````

`````{tab} Superiore

Sia la serie $y_1, \dots, y_n$. Fissato un training minimo e un orizzonte $h$, il
walk-forward produce una sequenza di coppie $(\text{train}, \text{test})$ in cui
il blocco di test cade sempre dopo il blocco di train. Nella variante
**espansa** l'$i$-esima iterazione addestra su $y_1, \dots, y_{t_i}$ e valuta su
$y_{t_i+1}, \dots, y_{t_i+h}$, con $t_i$ crescente; nella variante **scorrevole**
il training è $y_{t_i-w+1}, \dots, y_{t_i}$, con ampiezza $w$ costante. L'errore
finale è la media degli errori sui blocchi di test. Rispetto al singolo
train/test split, questa procedura usa più segmenti futuri come banco di prova e
riduce la varianza della stima, senza mai violare l'ordine temporale
{cite}`hyndman2021forecasting`.

`````

```{figure} ../figures/walk-forward-validazione.svg
:name: fig-walk-forward-validazione
:alt: Cinque righe orizzontali, una per iterazione dall'alto verso il basso. In ogni riga una barra del tempo con un segmento teal di train che cresce verso il basso, seguito a destra da un breve segmento terracotta di test, e dal resto della barra in crema tratteggiata. Un asse del tempo con freccia in basso indica che il test è sempre nel futuro del train.
:width: 100%

Walk-forward con finestra espansa: a ogni iterazione il training (teal) si
allunga e il test (terracotta) avanza, restando sempre nel futuro del training.
Nella variante scorrevole il segmento di train avrebbe lunghezza fissa.
```

## Misurare l'errore: dalle metriche note alle metriche scalate

Con lo schema di validazione in mano, resta la domanda del capitolo sulle
metriche: *con che numero* giudichiamo una previsione? Il MAE e l'RMSE, già
incontrati per la regressione, restano i mattoni di base: il **MAE** è l'errore
assoluto medio, nell'unità della serie; l'**RMSE** eleva al quadrato prima di
mediare, e quindi punisce di più i grandi svarioni. Il guaio è che entrambi
dipendono dalla scala: un MAE di 500 è ottimo per il PIL, disastroso per la
temperatura. Servono metriche che si possano confrontare tra serie diverse.

`````{tab} Elementare

Il primo tentativo è misurare l'errore in **percentuale**: sbagliare di 500 su
50 000 è l'1%, su 500 è il 100%. Questa è la **MAPE**, l'errore percentuale
medio. Comoda da spiegare, ma con tre difetti seri. Se il valore vero è
**zero** (un giorno senza vendite), si divide per zero e la metrica esplode.
Ed è **asimmetrica**: prevedere troppo alto o troppo basso non costa uguale, e
alla lunga premia i modelli timidi che sottostimano.

Una scorciatoia più robusta è confrontarsi con una previsione stupida: «di
quanto sbaglio, rispetto a chi si limita a ripetere l'ultimo valore?». È l'idea
della **MASE**. Se viene 1, sei bravo quanto lo sciocco che copia ieri; se viene
$0{,}5$, sbagli la metà; se supera 1, faresti meglio a non usare il modello.
Un solo numero, senza unità di misura, leggibile a colpo d'occhio.

`````

`````{tab} Superiore

Per un blocco di test di $h$ punti, con valori veri $y_t$ e previsioni
$\hat{y}_t$:

$$
\text{MAE} = \frac{1}{h}\sum_{t=1}^{h}\lvert y_t-\hat{y}_t\rvert,
\qquad
\text{RMSE} = \sqrt{\frac{1}{h}\sum_{t=1}^{h}\bigl(y_t-\hat{y}_t\bigr)^2}.
$$

L'errore percentuale medio e la sua versione «simmetrica» sono

$$
\text{MAPE} = \frac{100}{h}\sum_{t=1}^{h}\frac{\lvert y_t-\hat{y}_t\rvert}{\lvert y_t\rvert},
\qquad
\text{sMAPE} = \frac{100}{h}\sum_{t=1}^{h}
\frac{\lvert y_t-\hat{y}_t\rvert}{(\lvert y_t\rvert+\lvert\hat{y}_t\rvert)/2}.
$$

La MAPE è indefinita per $y_t=0$ ed è asimmetrica: la sottostima ($\hat{y}_t<y_t$)
è limitata al $100\%$, la sovrastima no, così la metrica favorisce chi
sottoprevede. La sMAPE mette la somma dei due valori al denominatore per limitare
il problema, ma malgrado il nome resta anch'essa non del tutto simmetrica.

La **MASE** (*Mean Absolute Scaled Error*), proposta da Rob Hyndman e Anne
Koehler nel 2006, scala l'errore del modello sull'errore *in-sample* del naive
calcolato sul training {cite}`hyndman2021forecasting`:

$$
\text{MASE} =
\frac{\dfrac{1}{h}\sum_{t=1}^{h}\lvert y_t-\hat{y}_t\rvert}
{\dfrac{1}{n-m}\sum_{t=m+1}^{n}\lvert y_t-y_{t-m}\rvert}.
$$

Il numeratore è il MAE del modello sul test; il denominatore è il MAE del
predittore naive a passo $m$ sui dati di training ($m=1$ per il naive semplice,
$m$ pari al periodo per la versione stagionale). Poiché è un rapporto tra errori
nella stessa unità, la MASE è **adimensionale** e confrontabile tra serie: valori
$<1$ battono il naive, $>1$ no. Non ha problemi con gli zeri, purché la serie di
training non sia costante.

Quando la previsione non è un singolo numero ma una **distribuzione** (un
intervallo, o un insieme di quantili), si misura la calibrazione con la
**pinball loss** (o *quantile loss*). Per il quantile di livello
$\tau\in(0,1)$, con previsione $\hat{y}_\tau$ e valore vero $y$:

$$
\mathcal{L}_\tau(y,\hat{y}_\tau) =
\begin{cases}
\tau\,(y-\hat{y}_\tau) & \text{se } y \ge \hat{y}_\tau,\\[4pt]
(1-\tau)\,(\hat{y}_\tau-y) & \text{se } y < \hat{y}_\tau.
\end{cases}
$$

Qui $\tau$ è il livello del quantile (per esempio $0{,}9$ per il novantesimo
percentile): la formula penalizza in modo **asimmetrico** gli sforamenti sopra e
sotto, tanto da spingere $\hat{y}_\tau$ verso il vero quantile $\tau$-esimo della
distribuzione. Mediata su più livelli, approssima un punteggio proprio per
l'intera previsione probabilistica {cite}`hyndman2021forecasting`.

`````

## Le baseline che bisogna sempre battere

Prima di dichiarare vittoria con una rete neurale, un modello va confrontato con
avversari volutamente banali. Se non li batte, non serve. Le tre baseline
classiche {cite}`hyndman2021forecasting` sono:

- **Naive**: la previsione per ogni istante futuro è l'**ultimo valore
  osservato**, $\hat{y}_{t+h}=y_t$. Sorprendentemente forte sulle serie quasi
  casuali (i prezzi finanziari, per dire).
- **Naive stagionale**: si ripete il valore dello **stesso istante del periodo
  precedente**, $\hat{y}_{t+h}=y_{t+h-m}$: le vendite di questo dicembre sono
  quelle dello scorso dicembre. È la baseline da battere ogni volta che c'è
  stagionalità.
- **Drift**: come il naive, ma con una **retta di tendenza** stimata dai due
  estremi della serie, $\hat{y}_{t+h}=y_t+h\cdot\frac{y_t-y_1}{t-1}$: prolunga il
  segmento che unisce il primo e l'ultimo punto.

Non è falsa modestia: è il modo per accorgersi quando un modello complicato sta
solo imitando, peggio, ciò che una riga di codice farebbe gratis.

## Trasformare il tempo in una tabella

Ed eccoci alla seconda metà del titolo: *rappresentare*. Buona parte dei
modelli che conosciamo (la regressione, gli alberi con gradient boosting, le
reti) non sanno nulla di «tempo»: vogliono una tabella di righe indipendenti,
ciascuna con le sue feature e il suo target. Il **feature engineering
temporale** costruisce quella tabella a partire dalla serie, riducendo la
previsione a un normale problema **supervisionato** tabellare.

`````{tab} Elementare

Il trucco è dare in pasto al modello, per ogni giorno, un riassunto del suo
recente passato. I mattoni sono quattro.

I **lag**: i valori di ieri, dell'altroieri, di una settimana fa. Sono la
memoria grezza della serie: spesso «quanto ho venduto ieri» è già un'ottima
indicazione su oggi.

Le **finestre mobili**: media e deviazione degli ultimi 7 o 30 giorni. La media
cattura il livello recente lisciando il rumore; la deviazione dice quanto la
serie è stata mossa di recente.

L'**encoding del tempo**: dal calendario ricaviamo il giorno della settimana, il
mese, se è un giorno festivo. Sono le informazioni che spiegano perché il lunedì
è diverso dalla domenica e agosto da novembre.

I **termini di Fourier**: coppie di seni e coseni che disegnano l'onda della
stagionalità, un modo compatto per dire al modello «siamo a questo punto del
ciclo annuale» senza una colonna per ciascuno dei 365 giorni.

`````

`````{tab} Superiore

Data la serie $y_t$, si costruisce una matrice di progetto $X$ in cui la riga
all'istante $t$ contiene solo informazione **fino a $t$** (mai oltre, per non
reintrodurre leakage):

- **Lag**: $y_{t-1}, y_{t-2}, \dots, y_{t-p}$.
- **Statistiche su finestra** di ampiezza $w$: media
  $\frac{1}{w}\sum_{i=1}^{w} y_{t-i}$, deviazione standard, minimo, massimo.
- **Variabili di calendario**: giorno della settimana, mese, indicatori di
  festività, tipicamente *one-hot*.
- **Termini di Fourier** per una stagionalità di periodo $m$: per $k=1,\dots,K$
  si aggiungono le colonne
  $\sin\!\bigl(\tfrac{2\pi k t}{m}\bigr)$ e $\cos\!\bigl(\tfrac{2\pi k t}{m}\bigr)$.
  Poche armoniche ($K$ piccolo) bastano a rappresentare stagionalità lisce con un
  pugno di regressori, invece delle $m-1$ dummy stagionali
  {cite}`hyndman2021forecasting`.

Il target della riga $t$ è $y_{t+h}$ per l'orizzonte $h$ desiderato. A quel
punto qualunque regressore tabellare (dai modelli lineari al gradient
boosting) diventa un modello di forecasting.

`````

## Prevedere più passi avanti

Finora abbiamo parlato di un orizzonte, ma spesso servono molti passi: le vendite
dei prossimi 30 giorni, non solo di domani. Ci sono tre strategie, con
compromessi diversi.

`````{tab} Elementare

La strategia **ricorsiva** allena un solo modello a un passo e poi lo fa girare a
catena: prevede domani, finge che sia successo davvero, e con quel valore prevede
dopodomani, e così via. Semplice, ma ogni previsione poggia sulle precedenti: se
sbagli il primo passo, l'errore si trascina e si **accumula** lungo la catena.

La strategia **diretta** allena un modello *diverso* per ogni orizzonte: uno per
«tra un giorno», uno per «tra sette giorni». Ogni previsione è indipendente e non
eredita gli errori altrui, ma addestrare tanti modelli costa.

La strategia **multi-output** usa un unico modello che sputa fuori l'intero
vettore dei passi futuri in un colpo solo: è la via naturale per le reti
neurali, che possono avere molte uscite.

`````

`````{tab} Superiore

Volendo prevedere $H$ passi $\hat{y}_{t+1}, \dots, \hat{y}_{t+H}$:

- **Ricorsiva** (o *iterata*): si stima un solo modello a un passo
  $\hat{y}_{t+1}=f(y_t, y_{t-1}, \dots)$ e lo si applica in cascata, reinserendo
  le proprie previsioni come input, $\hat{y}_{t+2}=f(\hat{y}_{t+1}, y_t, \dots)$.
  Gli errori si propagano e si **compongono** lungo l'orizzonte, gonfiando la
  varianza sui passi lontani.
- **Diretta**: si addestra un modello distinto $f_h$ per ciascun orizzonte
  $h=1,\dots,H$, con $\hat{y}_{t+h}=f_h(y_t, y_{t-1}, \dots)$. Nessun errore
  ereditato, ma $H$ modelli da stimare e nessuna coerenza imposta tra i passi.
- **Multi-output** (MIMO): un'unica funzione a valori vettoriali
  $(\hat{y}_{t+1}, \dots, \hat{y}_{t+H}) = f(y_t, y_{t-1}, \dots)$, che
  modella congiuntamente le dipendenze tra gli orizzonti (la forma tipica
  delle reti neurali, con $H$ neuroni in uscita).

`````

Non esiste una scelta sempre migliore: la ricorsiva è economica ma fragile sui
lunghi orizzonti, la diretta è robusta ma pesante, la multi-output sta nel mezzo
ed è comoda con le reti. La regola di prima resta sovrana: qualunque strategia si
scelga, la si valuta col walk-forward, mai mescolando il tempo.

## In pratica: walk-forward e MASE con NumPy

Mettiamo insieme i due pezzi centrali della sezione (lo split walk-forward e
la MASE) in poche righe di NumPy puro, su una serie sintetica con trend
leggero e stagionalità settimanale. Confrontiamo due baseline: il naive
stagionale (ripete l'ultima settimana) e il naive semplice (ripete l'ultimo
valore).

```python
import numpy as np

def walk_forward_split(n, min_train, horizon):
    """Split cronologico a finestra espansa (walk-forward / backtesting):
    restituisce coppie (indici_train, indici_test) col test sempre nel futuro."""
    for t in range(min_train, n - horizon + 1, horizon):
        yield np.arange(t), np.arange(t, t + horizon)

def mase(y_vero, y_pred, y_train, m=1):
    """MASE: MAE del modello sul test, scalato sul MAE del naive a passo m
    calcolato in-sample sul training."""
    errore_modello = np.mean(np.abs(y_vero - y_pred))
    errore_naive_train = np.mean(np.abs(y_train[m:] - y_train[:-m]))
    return errore_modello / errore_naive_train

# --- serie sintetica: trend leggero + stagionalità settimanale + rumore ---
rng = np.random.default_rng(0)
n, m = 140, 7
t = np.arange(n)
serie = 10 + 0.05 * t + 3 * np.sin(2 * np.pi * t / m) + rng.normal(0, 0.4, n)

mase_stagionale, mase_semplice = [], []
for idx_train, idx_test in walk_forward_split(n, min_train=28, horizon=m):
    storia, futuro = serie[idx_train], serie[idx_test]
    pred_stagionale = storia[-m:]            # naive stagionale: ripeti l'ultima settimana
    pred_semplice = np.full(m, storia[-1])   # naive semplice: ripeti l'ultimo valore
    # denominatore MASE sempre col naive a passo 1 sul training
    mase_stagionale.append(mase(futuro, pred_stagionale, storia, m=1))
    mase_semplice.append(mase(futuro, pred_semplice, storia, m=1))

print(f"iterazioni di walk-forward: {len(mase_stagionale)}")
print(f"MASE medio - naive stagionale: {np.mean(mase_stagionale):.3f}")
print(f"MASE medio - naive semplice:   {np.mean(mase_semplice):.3f}")
```

Poiché la serie è chiaramente stagionale, il naive stagionale ottiene un MASE
nettamente **sotto 1** (sbaglia molto meno del predittore che copia ieri),
mentre il naive semplice, cieco alla settimana, resta vicino o sopra 1. È
esattamente la lettura che rende la MASE preziosa: un unico numero, senza
unità, che dice al volo se un modello vale più della pigrizia.

```{admonition} Da ricordare
:class: important
- Con le serie temporali la **cross-validation con shuffle è sbagliata**:
  mescolare mette futuro e passato nello stesso mucchio e produce *leakage*, con
  stime dell'errore troppo ottimiste. Ogni dato di training deve **precedere** nel
  tempo ogni dato di validazione.
- Si valida col **walk-forward** (backtesting): split cronologici ripetuti col
  test sempre nel futuro, a **finestra espansa** (tutto il passato) o
  **scorrevole** (ampiezza fissa) {cite}`hyndman2021forecasting`.
- MAE e RMSE dipendono dalla scala; la **MAPE** ha problemi con gli zeri ed è
  asimmetrica; la **MASE** scala l'errore su quello del naive in-sample ed è
  adimensionale (${<}1$ batte il naive). Per le previsioni **probabilistiche** si
  usa la **pinball loss**.
- Vanno sempre battute le **baseline**: naive, naive stagionale, drift. Se il
  modello non le supera, non serve.
- Il **feature engineering temporale** (lag, statistiche su finestra, calendario,
  termini di Fourier) riduce il forecasting a un problema **supervisionato
  tabellare**, senza mai usare informazione dal futuro.
- Per il **multi-step** si sceglie tra strategia **ricorsiva** (economica, ma
  l'errore si accumula), **diretta** (un modello per orizzonte) e **multi-output**
  (un solo modello, tutti i passi).
```
