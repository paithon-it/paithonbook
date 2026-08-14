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
come *rappresentare* il tempo perché un normale modello tabellare (uno che
vuole una tabella di righe, come la regressione o gli alberi) possa impararlo.

Il modo di valutare che vedremo si chiama **backtesting**, ed è esattamente
quello che il nome dice: provare all'indietro. Si finge di essere in un giorno
del passato, si prevede quello che sarebbe successo dopo, e si confronta con
quello che è successo davvero; poi si sposta in avanti quel giorno, e si rifà.

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
leakage*, la stessa fuga di informazione per cui la sezione sulla validazione,
nel capitolo sul Machine Learning, imponeva di non toccare mai il test.

La regola è netta: **ogni dato usato per addestrare deve precedere nel tempo ogni
dato usato per validare**. Il confine tra train e validation è un istante $t_0$,
non un'estrazione a sorte.

`````

## Validazione temporale: lo split cronologico

La cura è semplice da enunciare: rispettare la freccia del tempo. Si allena sul
passato, si valida sul futuro, mai il contrario. Ma con una serie sola non
possiamo permetterci un unico taglio: butteremmo via troppe informazioni. La
soluzione standard porta tre nomi per la stessa identica cosa, e conviene
conoscerli tutti e tre perché si incontrano tutti e tre: **walk-forward**,
*backtesting*, e valutazione «su **origine mobile**»
{cite}`hyndman2021forecasting`.

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

Sia la serie $y_1, \dots, y_n$. In questa sezione la indichiamo con $y$
anziché con $x$: la serie diventa il *target* di un problema supervisionato,
e le previsioni si scrivono $\hat{y}$, come nel resto del libro. Fissato un
training minimo e un orizzonte $h$, il
walk-forward produce una sequenza di coppie $(\text{train}, \text{test})$ in cui
il blocco di test cade sempre dopo il blocco di train. Nella variante
**espansa** l'$i$-esima iterazione addestra su $y_1, \dots, y_{t_i}$ e valuta su
$y_{t_i+1}, \dots, y_{t_i+h}$, con $t_i$ crescente; nella variante **scorrevole**
il training è $y_{t_i-w+1}, \dots, y_{t_i}$, con ampiezza $w$ costante. L'errore
finale è la media degli errori sui blocchi di test, e la
{numref}`fig-walk-forward-validazione` mostra le due varianti una sopra
l'altra. Rispetto al singolo train/test split, questa procedura usa più
segmenti futuri come banco di prova e riduce la varianza della stima, senza mai
violare l'ordine temporale {cite}`hyndman2021forecasting`.

`````

```{figure} ../figures/origine-mobile.svg
:name: fig-walk-forward-validazione
:alt: Tre barre del tempo sulla stessa serie. In alto la k-fold mescolata: i blocchi di test terracotta sono sparsi ovunque, anche prima dei dati di addestramento teal. Sotto la validazione a origine mobile, a finestra espansa e a finestra scorrevole: il training avanza da sinistra e il blocco di test gli sta sempre subito a destra, cioè nel futuro.
:width: 100%

In alto la k-fold mescolata: i blocchi di prova finiscono sparsi fra quelli di
addestramento, e il modello si allena su ciò che dovrà prevedere. In basso
l'origine mobile: il confine avanza di taglio in taglio e il test resta sempre
dopo il training, sia a finestra espansa sia a finestra scorrevole.
```

## Misurare l'errore: dalle metriche note alle metriche scalate

Con lo schema di validazione in mano, resta la domanda che il capitolo sul
Machine Learning si poneva per i modelli tabellari: *con che numero* giudichiamo
una previsione? Il MAE e l'RMSE, già incontrati per la regressione, restano i
mattoni di base. Il **MAE** è l'errore assoluto medio, nell'unità della serie.
L'**RMSE** eleva al quadrato prima di mediare, e quindi punisce di più i grandi
svarioni; poi ne prende la radice quadrata, che è la R del nome (*root*) e
serve solo a riportare il numero nell'unità di partenza, perché senza un errore
in gradi diventerebbe gradi al quadrato.

Il guaio è che entrambi
dipendono dalla scala: un MAE di 500 è ottimo per il PIL, disastroso per la
temperatura. Servono metriche che si possano confrontare tra serie diverse.

`````{tab} Elementare

Il primo tentativo è misurare l'errore in **percentuale**: sbagliare di 500 su
50 000 è l'1%, su 500 è il 100%. Questa è la **MAPE**, l'errore percentuale
medio. Comoda da spiegare, ma con due difetti seri. Se il valore vero è
**zero** (un giorno senza vendite), si divide per zero e la metrica esplode.
Ed è **asimmetrica**: prevedere troppo alto o troppo basso non costa uguale.
La ragione si vede con un conto: se il valore vero è 100 e prevedi 0, hai
sbagliato del 100%, ed è il massimo che puoi sbagliare per difetto, perché sotto
lo zero non si va. Se prevedi 1000, hai sbagliato del 900%, e non c'è nessun
tetto. Sbagliare per eccesso costa quindi molto di più, e alla lunga la metrica
premia i modelli timidi, quelli che sottostimano sempre.

Una scorciatoia più robusta è confrontarsi con una previsione stupida: «di
quanto sbaglio, rispetto a chi si limita a copiare?». È l'idea della **MASE**.
Se viene 1 sbagli quanto lui, se viene $0{,}5$ sbagli la metà, se viene 2 il
doppio: un solo numero, senza unità di misura, leggibile a colpo d'occhio.

Con un'avvertenza che vale la pena prendersi subito, perché è il modo più comune
di sbagliarla: chi copia può copiare ieri, oppure lo stesso giorno della
settimana scorsa se la serie ha un ritmo settimanale. Sono due pigrizie diverse,
e il numero che ne esce è diverso. Va detto sempre quale delle due si è messa al
paragone.

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

La MAPE è indefinita per $y_t=0$ ed è asimmetrica: per serie e previsioni non
negative la sottostima ($\hat{y}_t<y_t$) è limitata al $100\%$, la sovrastima
no, così la metrica favorisce chi sottoprevede.

La sMAPE mette la somma dei due valori al denominatore, e con il nome promette
di aver corretto l'asimmetria. In realtà la **rovescia**: a parità di errore
assoluto penalizza di più chi sottoprevede. Con $y=100$ e uno scarto di $50$
costa il $66{,}7\%$ per difetto contro il $40\%$ per eccesso, e il divario
cresce con l'errore. In più ha un **tetto** del $200\%$ che la MAPE non ha, e
resta **indefinita** quando $y_t = \hat y_t = 0$, cioè proprio sulle serie
intermittenti per cui la si andava cercando. Hyndman e Koehler, gli stessi della
MASE, ne sconsigliano l'uso {cite}`hyndman2006another`.

La **MASE** (*Mean Absolute Scaled Error*), proposta da Rob Hyndman e Anne
Koehler nel 2006 {cite}`hyndman2006another`, scala l'errore del modello
sull'errore *in-sample* del naive calcolato sul training:

$$
\text{MASE} =
\frac{\dfrac{1}{h}\sum_{j=1}^{h}\lvert e_j\rvert}
{\dfrac{1}{n-m}\sum_{t=m+1}^{n}\lvert y^{\text{tr}}_t-y^{\text{tr}}_{t-m}\rvert}.
$$

I due simboli non sono lo stesso oggetto, e la distinzione è la parte che si
sbaglia più spesso: $e_j$ sono gli $h$ errori del modello sul blocco di **test**,
mentre $y^{\text{tr}}$ è la serie di **training**, lunga $n$. Il denominatore
non si calcola mai sul test. Il passo $m$ è il periodo stagionale: vale $1$ su
una serie senza stagionalità, e va posto **pari al periodo** su una serie che ne
ha una {cite}`hyndman2021forecasting`. Altrimenti al denominatore finisce un
avversario che su quella serie sbaglia molto più del dovuto, e ogni modello ne
esce lusingato.

Poiché è un rapporto tra errori nella stessa unità, la MASE è
**adimensionale** e confrontabile tra serie, e non ha problemi con gli zeri
purché la serie di training non sia costante. La lettura, però, va data per
esteso, perché la versione corta («sotto 1 batte il naive») è la fonte di
un equivoco: un valore sotto $1$ vuol dire che il modello sbaglia meno di quanto
sbaglia, **a un passo di stagione e sui dati di addestramento**, il predittore
che copia il ciclo precedente. Non è un duello, è una scala: il denominatore
serve a togliere l'unità di misura della serie, non a fare da avversario. Un
modello con MASE $0{,}9$ su un orizzonte a dodici passi non ha battuto nessuno,
ha sbagliato il 90% di quanto sbaglia a un passo chi copia; il che su dodici
passi è ottimo e su un passo sarebbe mediocre. Se si vuole davvero il duello, il
naive va fatto correre sullo **stesso** test e sullo **stesso** orizzonte, ed è
quello che fanno le linee di base qui sotto.

Quando la previsione non è un singolo numero ma una **distribuzione** (un
intervallo, o un insieme di quantili), si usa la
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

Attenzione però a cosa misura, perché non è la calibrazione. Essere un punteggio
**proprio** significa che è minimizzata in media dalla distribuzione vera, e
quindi che serve benissimo come funzione di costo in addestramento e come
criterio di confronto complessivo. Ma premia insieme la **calibrazione** (la
banda copre davvero quello che dichiara) e la **finezza** (la banda è stretta),
e non sa dire quale delle due manca. Il conto, su dati gaussiani veri e mediando
su nove livelli di quantile: un modello che azzecca la mediana ma dichiara
**metà** dell'incertezza vera prende $0{,}329$, e la sua banda «all'80%» ne copre
in realtà il 48%; un modello prudente, che dichiara il **doppio**
dell'incertezza, ne copre il 99% e prende $0{,}356$, cioè peggio. Il numero li
ordina, ma non dice che il primo sta mentendo sull'incertezza e il secondo la sta
sprecando (il minimo, $0{,}309$, ce l'ha il modello calibrato: è esattamente
questo che vuol dire «punteggio proprio»).

La formulazione canonica della materia è «massimizzare la finezza **sotto
vincolo** di calibrazione», e la calibrazione si controlla a parte, ed è facile:
si conta quante volte il valore osservato cade dentro la banda all'80% e si
guarda se fa l'80%. La macchina per farlo è il walk-forward di questa stessa
sezione.

`````

## Le linee di base che bisogna sempre battere

Prima di dichiarare vittoria con una rete neurale, un modello va confrontato con
avversari volutamente banali. Se non li batte, non serve. In inglese si chiamano
*baseline*, e sono l'idea che il capitolo chiama **linea di base**.

Le tre classiche {cite}`hyndman2021forecasting` si scrivono con tre simboli
soli: $y_t$ è il valore osservato all'istante $t$, il cappellino di
$\hat{y}$ vuol dire «previsto» invece che «osservato», $h$ è quanti passi avanti
si guarda e $m$ la lunghezza del ciclo stagionale (7 per una settimana, 12 per
un anno di mesi).

- **Naive**: la previsione per ogni istante futuro è l'**ultimo valore
  osservato**, $\hat{y}_{t+h}=y_t$. Sorprendentemente forte sulle serie quasi
  casuali (i prezzi finanziari, per dire).
- **Naive stagionale**: si ripete il valore dello **stesso istante del periodo
  precedente**, e quando l'orizzonte supera un ciclo intero si ricicla sempre
  l'ultimo ciclo *osservato*, invece di andare a pescare stagioni che non sono
  ancora accadute: $\hat{y}_{t+h}=y_{t+h-m(k+1)}$ con
  $k = \lfloor (h-1)/m \rfloor$, la stessa contabilità di Holt-Winters (le due
  parentesi tagliate in basso vogliono dire «arrotonda per difetto», e servono a
  contare quanti cicli interi stanno dentro l'orizzonte). In
  parole: le vendite di questo dicembre sono quelle dello scorso dicembre. È la
  linea di base da battere ogni volta che c'è stagionalità.
- **Drift**: come il naive, ma con una **retta di tendenza** stimata dai due
  estremi della serie, $\hat{y}_{t+h}=y_t+h\cdot\frac{y_t-y_1}{t-1}$: la
  frazione è la salita media per passo (quanto è cresciuta la serie dal primo
  all'ultimo punto, diviso quanti passi ci sono voluti), e moltiplicandola per
  $h$ si prolunga il segmento che unisce il primo e l'ultimo punto.

Non è falsa modestia: è il modo per accorgersi quando un modello complicato sta
solo imitando, peggio, ciò che una riga di codice farebbe gratis.

## Le bande di previsione sono più strette di quello che dichiarano

Qui è il posto giusto per una parentesi, perché qui ci sono gli attrezzi per
chiuderla. Il filo rosso del capitolo dice che una previsione seria è un numero
con la sua incertezza; e quasi nessun metodo dichiara l'incertezza giusta.

Le bande che ARIMA, il lisciamento esponenziale e le reti probabilistiche
restituiscono (le forbici dentro cui dicono che cadrà il valore vero) poggiano
su due comodità.

La prima: i numeri del modello (la frazione con cui ieri pesa su oggi,
l'ampiezza tipica degli scarti) vengono trattati come se li conoscessimo, mentre
li abbiamo stimati dalla stessa storia e potevano venire diversi. È questa a
stringere le bande **sempre**, perché ignora un'incertezza che c'è: un
intervallo dichiarato all'80% ne copre meno dell'80%
{cite}`hyndman2021forecasting`.

La seconda comodità è che si dà per buono che gli scarti si dispongano secondo la
campana della statistica classica (la **gaussiana** del capitolo di matematica),
mentre le serie vere di sorprese grosse ne hanno di più. Questa sbaglia in un
verso che **dipende da quanto in là si guarda**, e conviene saperlo perché è
controintuitivo: le code pesanti spostano massa sia al centro sia agli estremi,
quindi una banda gaussiana all'80% finisce per coprire *più* dell'80%, mentre una
al 99% copre meno del 99%. Chi dichiara forbici strette e frequenti è al riparo;
chi promette di coprire quasi tutto, no.

La buona notizia è che questo si misura, e si chiama **copertura empirica**: si
prende il walk-forward, si contano quante volte il valore osservato cade dentro
la banda, e si confronta col livello dichiarato.

Il conto si fa su un AR(1) come quello della sezione precedente ($c=4$,
$\phi=0{,}6$, rumore sparpagliato con deviazione standard $1$, ripetendo la
prova ventimila volte). Con i parametri
noti la banda all'80% copre l'80% esatto, come promette. Appena si stimano i
parametri dalla storia, invece di conoscerli, la copertura cede: a un passo
scende al **77%** con trenta osservazioni di storia, e risale al 79% con cento.
E cede di più via via che l'orizzonte si allunga, perché all'incertezza
dell'ultimo passo si somma quella di tutti i passi in mezzo: a cinque passi,
sempre con trenta osservazioni, resta il **73%**. È la diagnostica più semplice
della previsione probabilistica, costa dieci righe, e quasi nessuno la fa.

## Trasformare il tempo in una tabella

Ed eccoci alla seconda metà del titolo: *rappresentare*. Buona parte dei
modelli che conosciamo (la regressione, gli alberi decisionali, le reti) non
sanno nulla di «tempo»: vogliono una tabella di righe indipendenti, ciascuna con
le sue colonne di ingresso (le **feature**) e la sua risposta giusta (il
**target**), come nel capitolo sul Machine Learning. Costruire quella tabella a
partire da una serie si chiama **feature engineering temporale**, e riduce la
previsione a un normale problema **supervisionato** tabellare.

`````{tab} Elementare

Il trucco è dare in pasto al modello, per ogni giorno, un riassunto del suo
recente passato. I mattoni sono quattro.

I **lag**: i valori di ieri, dell'altroieri, di una settimana fa. Sono la
memoria grezza della serie: spesso «quanto ho venduto ieri» è già un'ottima
indicazione su oggi.

Le **finestre mobili**: media e deviazione degli ultimi 7 o 30 giorni. La media
cattura il livello recente lisciando il rumore; la deviazione (quella standard,
la misura di quanto i valori si sparpagliano attorno alla loro media, del
capitolo di matematica) dice quanto la serie è stata mossa di recente.

L'**encoding del tempo**, cioè trasformare la data in numeri: dal calendario
ricaviamo il giorno della settimana, il mese, se è un giorno festivo. Sono le
informazioni che spiegano perché il lunedì è diverso dalla domenica e agosto da
novembre.

I **termini di Fourier**. Per dire al modello a che punto del ciclo annuale
siamo si potrebbe mettere una colonna per ciascuno dei 365 giorni, con un $1$
sul giorno giusto e $0$ su tutti gli altri: funziona, ma sono 365 colonne per
un'informazione sola. C'è un modo molto più compatto, ed è lo stesso di quando,
nel capitolo sull'audio, un accordo al pianoforte veniva scomposto nelle poche
note che lo compongono: una curva che si ripete si
descrive con poche onde regolari sovrapposte. Quelle onde, in matematica, si
chiamano seno e coseno (sono le due curve ondulate di base, quelle che salgono e
scendono all'infinito sempre uguali a sé stesse), e bastano due o tre coppie per
disegnare quasi ogni stagionalità liscia.

`````

`````{tab} Superiore

Data la serie $y_t$, si costruisce una matrice di progetto $\mathbf{X}$ in cui
la riga
all'istante $t$ contiene solo informazione **fino a $t$** (mai oltre, per non
reintrodurre leakage):

- **Lag**: $y_{t-1}, y_{t-2}, \dots, y_{t-p}$.
- **Finestre mobili** di ampiezza $w$: media
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

Il divieto vale anche per le **trasformazioni**, ed è la fuga con cui si apre
questa sezione: media, deviazione, minimo e massimo usati per scalare le colonne
vanno stimati **sul solo training** di quel giro e poi applicati al test, mai
calcolati sull'intera serie. Uno `StandardScaler` messo prima dello split è
esattamente l'analista con la curva liscia come una pista da sci.

E attenzione a dove cade il taglio, perché qui la regola «netta» enunciata in
apertura di sezione si viola da sé, al bordo. Se si divide train e test
guardando l'istante $t$ delle **feature**, le ultime $h-1$ righe di training
hanno un bersaglio $y_{t+h}$ che sta già dentro il periodo di test, e le
finestre mobili di ampiezza $w$ allungano la sovrapposizione di altri $w-1$ passi.
Si tagliano via quelle righe, ed è un'operazione che ha un nome, la **purga**.
Quante siano si conta senza formule da ricordare: una riga di training
all'istante $t$ tocca le osservazioni da $y_{t-w}$ a $y_{t-1}$ e in più il suo
bersaglio $y_{t+h}$; la prima riga di test, all'istante $t_0+1$, legge
all'indietro fino a $y_{t_0+1-w}$. Perché le due non si sfiorino serve
$t + h < t_0 + 1 - w$, e le righe da togliere in fondo al training sono
**$h + w$**. La tentazione naturale è togliere solo quelle il cui bersaglio
sfora, cioè $h$, e ci si dimentica delle finestre mobili, che allungano
all'indietro la parte di serie che ogni riga di test si porta dentro. Il costo è
un pugno di esempi; il guadagno è che la regola torna vera anche al bordo.

Quanto costa tenersele, quelle righe, dipende da quanto è lungo il training. Su
una serie fortemente autocorrelata ($\phi = 0{,}9$) con $p=5$ ritardi, $w=10$ e
$h=7$, cioè diciassette righe da purgare, la stima dell'errore esce ottimista
di circa l'$1{,}7\%$ quando il training è di centoventi righe e **di qualche
decimo di punto** quando è di quattrocento (confrontando, a parità di numero di
righe, una finestra di addestramento che arriva al confine e una purgata; la
cifra esatta dipende dalla lunghezza del blocco di test e da quale errore si
guarda, quello quadratico o la sua radice). Il guasto si
vede quando i dati sono pochi, cioè proprio quando si è più tentati di tenersele.

L'**embargo**, che nella letteratura sul machine learning finanziario accompagna
sempre la purga, qui invece **non serve**, e vale la pena dirlo perché i due
viaggiano in coppia e chi li importa entrambi butta via dati per difendersi da
una minaccia che non c'è. L'embargo mette una zona morta anche *dopo* il blocco
di test, e serve quando un blocco di addestramento viene dopo un blocco di prova
nel tempo, come nelle validazioni incrociate combinatorie in cui i fold si
alternano lungo la serie. Nella validazione a origine mobile il training è
sempre un prefisso e il test sempre il blocco immediatamente successivo: nessun
dato di addestramento segue mai un dato di prova, e la zona morta a destra non
avrebbe niente da proteggere.

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

La strategia **multi-output** usa un unico modello che sputa fuori tutti i passi
futuri in un colpo solo, tutti i trenta giorni insieme invece che uno per volta:
è la via naturale per le reti neurali, che possono avere molte uscite.

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
leggero e stagionalità settimanale. Confrontiamo due linee di base: il naive
stagionale (ripete l'ultima settimana) e il naive semplice (ripete l'ultimo
valore).

```python
import numpy as np

def walk_forward_split(n, min_train, horizon):
    """Split cronologico a finestra espansa (walk-forward / backtesting):
    restituisce coppie (indici_train, indici_test) col test sempre nel futuro."""
    for t in range(min_train, n - horizon + 1, horizon):
        yield np.arange(t), np.arange(t, t + horizon)

def mase(y_vero, y_pred, scalatore):
    """MASE: MAE del modello sul test, diviso per lo scalatore, che è il MAE
    del naive a passo m calcolato in-sample sul training."""
    return np.mean(np.abs(y_vero - y_pred)) / scalatore

# --- serie sintetica: trend leggero + stagionalità settimanale + rumore ---
rng = np.random.default_rng(0)
n, m = 140, 7
t = np.arange(n)
serie = 10 + 0.05 * t + 3 * np.sin(2 * np.pi * t / m) + rng.normal(0, 0.4, n)

# Lo scalatore è il naive a passo m (la serie ha un ciclo di 7 giorni: il
# metro giusto è chi copia la settimana scorsa, non chi copia ieri) ed è
# fissato UNA volta sul training iniziale, così i MASE dei vari giri sono
# tutti espressi nella stessa unità e si possono mediare.
scalatore = np.mean(np.abs(serie[m:28] - serie[:28 - m]))

mase_stagionale, mase_semplice = [], []
for idx_train, idx_test in walk_forward_split(n, min_train=28, horizon=m):
    storia, futuro = serie[idx_train], serie[idx_test]
    pred_stagionale = storia[-m:]            # naive stagionale: ripeti l'ultima settimana
    pred_semplice = np.full(m, storia[-1])   # naive semplice: ripeti l'ultimo valore
    mase_stagionale.append(mase(futuro, pred_stagionale, scalatore))
    mase_semplice.append(mase(futuro, pred_semplice, scalatore))

print(f"iterazioni di walk-forward: {len(mase_stagionale)}")
print(f"MASE medio - naive stagionale: {np.mean(mase_stagionale):.3f}")
print(f"MASE medio - naive semplice:   {np.mean(mase_semplice):.3f}")
```

Il naive stagionale esce **attorno a 1**, e non poteva che essere così: su una
serie con un ciclo settimanale il metro è lui, quindi sta pareggiando con sé
stesso. Il naive semplice, cieco alla settimana, sta **sopra 5**: sbaglia cinque
volte tanto. La morale non è che il naive stagionale sia bravo, è che su una
serie stagionale il metro giusto è quello, e chi non lo batte non ha un modello.

Vale la pena notare quanto la scelta del metro cambi il verdetto, perché è una
scorciatoia che si incontra spesso: mettendo al denominatore il naive a un passo
invece che a sette, gli stessi due predittori escono a $0{,}34$ e $1{,}64$, e il
primo sembrerebbe bravissimo. Non ha previsto meglio di prima: è cambiato il
righello. È la lettura che rende la MASE preziosa e insieme la sua unica
insidia: un numero senza unità dice al volo se un modello vale più della
pigrizia, purché si dichiari **quale** pigrizia.

`````{tab} Elementare

```{admonition} Da ricordare
:class: important
- Mescolare i dati di una serie **è sbagliato**, ed è la cosa che altrove si fa
  sempre. Mettere futuro e passato nello stesso mucchio è come far esercitare lo
  studente sul 4 e sul 5 marzo e poi interrogarlo sul 3: la previsione sembrerà
  miracolosa, e non lo è. Ogni dato su cui ci si allena deve venire **prima**,
  nel tempo, di ogni dato su cui si verifica.
- Si valuta **provando all'indietro** (*backtesting*): ci si mette in un giorno
  del passato, si prevede il seguito, si confronta con quello che è successo, e
  poi si sposta quel giorno in avanti e si rifà. Il pezzo su cui ci si allena può
  allungarsi ogni volta (finestra **espansa**) o restare lungo uguale e scivolare
  in avanti (finestra **scorrevole**), come nella
  {numref}`fig-walk-forward-validazione`.
- Le misure d'errore che dipendono dall'unità della serie (500 è ottimo per il
  PIL e disastroso per la temperatura) non si possono confrontare fra serie
  diverse. Quella che si può è la **MASE**: dice di quanto sbagli rispetto a chi
  copia e basta. Se viene 1 sbagli quanto lui, se viene $0{,}5$ sbagli la metà.
  Non è però un duello alla pari, perché chi copia viene fatto correre a un passo
  solo: sbagliare quanto lui prevedendo dodici giorni avanti è tutt'altra impresa
  che sbagliare quanto lui prevedendo domani. E va detto **quale** pigrizia si è
  messa al denominatore: su una serie con un ciclo settimanale il paragone giusto
  è con chi copia la settimana scorsa, non con chi copia ieri, e cambiando
  paragone cambia il verdetto.
- Vanno sempre battute le **linee di base**: chi copia l'ultimo valore, chi copia
  il ciclo precedente, chi prolunga la retta fra il primo e l'ultimo punto. Se il
  modello non le supera, non serve.
- Una serie si trasforma in una **tabella** dando al modello, per ogni giorno, un
  riassunto del suo passato recente: i valori dei giorni prima, le medie degli
  ultimi giorni, il calendario, e poche onde regolari per dire a che punto del
  ciclo siamo. Mai niente che venga dal futuro, nemmeno di striscio.
- Per prevedere molti giorni ci sono tre modi: uno alla volta rimettendo dentro
  la propria previsione (**ricorsivo**: economico, ma l'errore si trascina), un
  modello per ciascun giorno futuro (**diretto**: robusto, ma costa), o un
  modello solo che li sputa fuori tutti insieme (**multi-output**).
- Una previsione che dichiara una forbice («fra 22 e 26 gradi») quasi sempre la
  dichiara **più stretta** di quanto sarebbe onesto. Si controlla contando quante
  volte il valore vero cade davvero dentro.
```

`````

`````{tab} Superiore

```{admonition} Da ricordare
:class: important
- Con le serie temporali la **cross-validation con shuffle è sbagliata**:
  mescolare mette futuro e passato nello stesso mucchio e produce *leakage*, con
  stime dell'errore troppo ottimiste. Ogni dato di training deve **precedere** nel
  tempo ogni dato di validazione, e al confine la regola va difesa con la
  **purga** ($h+w$ righe), non con l'embargo, che qui non ha nulla da
  proteggere.
- Si valida col **walk-forward** (backtesting): split cronologici ripetuti col
  test sempre nel futuro, a **finestra espansa** (tutto il passato) o
  **scorrevole** (ampiezza fissa) {cite}`hyndman2021forecasting`.
- MAE e RMSE dipendono dalla scala; la **MAPE** ha problemi con gli zeri ed è
  asimmetrica, e la **sMAPE** non attenua quell'asimmetria, la **rovescia**; la
  **MASE** {cite}`hyndman2006another` scala l'errore su quello del naive
  in-sample **a passo $m$** ed è adimensionale, ma il denominatore è una scala,
  non un avversario: dichiarare quale $m$ si è usato è parte del numero. Per le
  previsioni **probabilistiche** si usa la **pinball loss**, che è un punteggio
  proprio ma premia insieme calibrazione e finezza: la calibrazione si controlla
  a parte, con la **copertura empirica**.
- Vanno sempre battute le **linee di base**: naive, naive stagionale
  ($\hat y_{t+h} = y_{t+h-m(k+1)}$), drift. Se il modello non le supera, non
  serve.
- Il **feature engineering temporale** (lag, finestre mobili, calendario,
  termini di Fourier) riduce il forecasting a un problema **supervisionato
  tabellare**, senza mai usare informazione dal futuro, comprese le statistiche
  usate per scalare le colonne.
- Per il **multi-step** si sceglie tra strategia **ricorsiva** (economica, ma
  l'errore si accumula), **diretta** (un modello per orizzonte) e **multi-output**
  (un solo modello, tutti i passi).
- Le bande di previsione sono calcolate a **parametri noti** e sotto ipotesi di
  normalità: escono sistematicamente troppo strette. Su un AR(1) con trenta
  osservazioni di storia, un intervallo nominale all'80% ne copre il 77% a un
  passo e il 73% a cinque.
```

`````
