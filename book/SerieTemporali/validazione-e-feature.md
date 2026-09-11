# Validare e rappresentare: backtesting e feature temporali

Un analista quantitativo mostra il grafico di un suo modello: rendimento annuo
del 40%, curva che sale liscia come una pista da sci. Lo mette a lavorare sui
soldi veri e nel giro di un mese è in rosso. Cos'è successo? Ricostruendo il
codice, si scopre che tra le variabili in ingresso ce n'era una calcolata
sulla media dell'intero periodo: futuro compreso. Il modello, in fase di
prova, «sapeva» dove sarebbe andato il prezzo. Sul passato era un veggente;
sul futuro, un ciarlatano.

Questa è la trappola numero uno di chi lavora con le serie temporali, e il nome
ce l'ha già: è il *data leakage* della {doc}`sezione su overfitting e
validazione </MachineLearning/overfitting-validazione>`, la fuga di
informazione dai dati su cui il modello sarà giudicato verso quelli su cui
impara. Qui la fuga ha una direzione sua, dal futuro verso il passato, e per
questo si dice *leakage temporale*.

Trend, stagionalità e autocorrelazione dicono che cos'è una serie; a decidere
se una previsione vale qualcosa sono altre due domande: come le si dà un voto
senza barare col futuro, e come si rappresenta il tempo perché un normale
modello tabellare (uno che vuole una tabella di righe, come la regressione o
gli alberi) possa impararlo. Le colonne di quella tabella si chiamano
feature.

Il modo di valutare che vedremo si chiama **backtesting**, ed è esattamente
quello che il nome dice: provare all'indietro. Si finge di essere in un giorno
del passato, si prevede quello che sarebbe successo dopo, e si confronta con
quello che è successo davvero; poi si sposta in avanti quel giorno, e si rifà.

## Perché mescolare i dati è un errore

Per i modelli tabellari la validazione è un
rito. Gli esempi si dividono in tre mucchi: uno su cui il modello impara, uno
su cui lo si mette a punto, uno su cui lo si esamina alla fine e che non si
tocca mai prima. E prima di dividerli si mescolano, perché se arrivassero già in
un ordine suo (tutte le foto di gatti in fondo, per dire) i tre mucchi
verrebbero diversi fra loro senza che sia colpa di nessuno. La k-fold
cross-validation rifà la divisione più volte, a turno, e fa la media: un voto
più stabile.

Con le serie temporali quel rimescolare, che altrove è igiene, qui è veleno.

`````{tab} Elementare

Alleni uno studente a prevedere il meteo. Gli dai in mano i dati di tutto
l'anno mescolati a caso: alcuni giorni per esercitarsi, altri per
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

Il problema è che gli esempi di una serie sono ordinati e fortemente
autocorrelati, quindi tutt'altro che indipendenti. Uno split casuale, o una
k-fold con shuffle, mette nel training istanti $t+1, t+3, \dots$ e nel
validation l'istante $t$: il modello osserva valori *successivi* a quello che
deve prevedere, e sfrutta l'autocorrelazione per «interpolare» all'indietro. La
stima dell'errore che ne esce è sistematicamente ottimista: un caso di *data
leakage*, la stessa fuga di informazione per cui il test non si tocca mai.

La regola è netta: ogni dato usato per addestrare deve precedere nel tempo ogni
dato usato per validare. Il confine tra train e validation è un istante $t_0$,
non un'estrazione a sorte.

`````

## Validazione temporale: lo split cronologico

La cura è semplice da enunciare: rispettare la freccia del tempo. Ci si allena
sul passato, si verifica sul futuro, mai il contrario. Un taglio solo, però, non
basta, e la ragione è che darebbe un voto solo, misurato su una manciata di
giorni: se in quei giorni è capitato un fatto strano (una nevicata, uno
sciopero), il voto racconta la nevicata e non il modello. Meglio tagliare in
molti punti diversi e fare la media dei voti. È il backtesting di poche righe
fa, e conviene sapere che gli altri due nomi con cui lo si incontra sono
**walk-forward** e valutazione «su **origine mobile**»: tre parole, una cosa
sola {cite}`hyndman2021forecasting`.

`````{tab} Elementare

Si rifà più volte lo stesso gioco onesto (allena sul prima, prova sul dopo),
spostando ogni volta il confine in avanti. Ci sono due modi.

Con la **finestra espansa** (*expanding*), a ogni giro tieni tutto il passato
disponibile e lo allunghi: prima usi i primi due mesi per prevedere il terzo, poi
i primi tre per prevedere il quarto, e così via. Come uno storico che, più anni
studia, più contesto ha.

Con la **finestra scorrevole** (*rolling*), tieni invece una finestra di
lunghezza fissa che scivola in avanti: sempre, per esempio, gli ultimi dodici
mesi. Utile quando il passato troppo lontano non è più rappresentativo: le
abitudini d'acquisto di dieci anni fa dicono poco su quelle di oggi.

In entrambi i casi il test è sempre a destra del train, cioè nel futuro,
come mostra la {numref}`fig-walk-forward-validazione`.

`````

`````{tab} Superiore

Sia la serie $y_1, \dots, y_n$. La lettera è $y$ e non $x$ perché la serie
diventa il *target* di un problema supervisionato,
e le previsioni si scrivono $\hat{y}$, come nel resto del libro; altrove nel
capitolo la serie resta $x_t$. Fissato un training minimo e un orizzonte $h$, il
walk-forward produce una sequenza di coppie $(\text{train}, \text{test})$ in cui
il blocco di test cade sempre dopo il blocco di train. Nella variante
espansa l’$i$-esima iterazione addestra su $y_1, \dots, y_{t_i}$ e valuta su
$y_{t_i+1}, \dots, y_{t_i+h}$, con $t_i$ crescente; nella variante scorrevole
il training è $y_{t_i-v+1}, \dots, y_{t_i}$, con ampiezza $v$ costante. L'errore
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

Con lo schema di validazione in mano, resta la domanda che la
{doc}`sezione sulle metriche </MachineLearning/metriche>` si poneva per i
modelli tabellari: *con che numero* giudichiamo
una previsione? Il MAE e l'RMSE, già incontrati per la regressione, restano i
mattoni di base, e la differenza fra i due sta tutta in come trattano gli
sbagli grossi.

Il MAE è la media degli errori presi senza segno: un giorno in cui hai
previsto tre gradi in più e uno in cui ne hai previsti tre in meno per lui sono
la stessa cosa, tre gradi di errore.

L’RMSE fa tre cose in fila, e il nome le elenca al contrario. Prima eleva
al quadrato ogni errore, poi ne fa la media, e infine prende la radice
quadrata del risultato, che è la R del nome (*root*) e serve solo a riportare
il numero nell'unità di partenza, perché senza di essa un errore in gradi
verrebbe fuori in gradi al quadrato. Il pezzo che conta è il primo: siccome il
quadrato di otto è sessantaquattro mentre il quadrato di due è quattro, un solo
sbaglio grosso pesa più di tanti sbagli piccoli messi insieme.

Due modelli che il MAE giudica identici: uno sbaglia di due gradi tutti e
quattro i giorni, l'altro ne azzecca tre e sbaglia di otto il quarto. MAE due
contro due, pari. Con l'RMSE il primo fa
$\sqrt{(4+4+4+4)/4} = \sqrt{4} = 2$ e il secondo
$\sqrt{(0+0+0+64)/4} = \sqrt{16} = 4$: il doppio, perché quel giorno di
disastro gli altri tre non lo compensano.

Il guaio è che entrambi
dipendono dall'unità di misura della serie: un MAE di 500 è ottimo per il PIL,
disastroso per la temperatura. Servono numeri che si possano confrontare fra
serie diverse.

`````{tab} Elementare

Il primo tentativo è misurare l'errore in percentuale: sbagliare di 500 su
50 000 è l'1%, su 500 è il 100%. Questa è la **MAPE**, l'errore percentuale
medio. Comoda da spiegare, ma con tre difetti seri, e il primo si vede
proprio sulla temperatura di poco fa: una percentuale ha senso solo dove lo
zero della scala è uno zero vero, e in gradi Celsius non lo è, sicché lo
stesso errore di un grado vale il 5% a venti gradi e il 50% a due. Gli altri
due riguardano il conto. Se il valore vero è
zero (un giorno senza vendite), si divide per zero e la metrica esplode.
Ed è asimmetrica: prevedere troppo alto o troppo basso non costa uguale.
Col valore vero a 100, se prevedi 0 hai sbagliato del 100%, ed è il massimo che
puoi sbagliare per difetto, perché sotto lo zero non si va. Se prevedi 1000,
hai sbagliato del 900%, e non c'è nessun tetto. Sbagliare per eccesso costa
quindi di più, e alla lunga la metrica premia i modelli timidi.

La strada che funziona è un'altra: invece di guardare l'errore in sé, si guarda
quante volte è più grande dell'errore di qualcuno che non fa niente di
intelligente. Se il tuo modello sbaglia in media di 4 gradi e chi si limita a
copiare il giorno prima ne sbaglia 8, il tuo numero è $4/8 = 0{,}5$. È la
**MASE**, sigla inglese per «errore assoluto medio scalato», e *scalato* vuol
dire proprio questo: diviso per il metro di qualcun altro. Se viene 1 sbagli
quanto lui, se viene $0{,}5$ sbagli la metà, se viene 2 il doppio: un numero
solo, senza unità di misura. Perché ci sia un metro, però, la serie deve
muoversi: su una che si ripete sempre identica chi copia non sbaglia mai, il
suo errore è zero, e per zero non si divide.

Due modi di sbagliarla. Uno è tacere quale pigrizia si è messa al paragone: chi
copia può copiare ieri, oppure lo stesso giorno della settimana scorsa se la
serie ha un ritmo settimanale, e il numero che ne esce è diverso.

L'altro è credere che sia una gara alla pari. Chi copia corre su un altro
tratto: lo si fa girare sulla strada già percorsa, quella su cui ti sei
allenato, e ogni volta gli si chiede solo il giorno dopo, mentre tu magari ne
stai prevedendo dodici ({numref}`fig-mase-non-duello`). Il suo errore medio si
misura una volta sola, lì, prima che la prova cominci, e da quel momento non si
tocca più. E lo si fa apposta, per la ragione che ne dànno gli autori: un
avversario fatto correre sul blocco di prova si può calcolare solo se le
previsioni da confrontare sono parecchie, mentre un metro preso sulla storia
esiste anche quando avanti si guarda una volta sola. Sbagliare
quanto lui, allora, non vuol dire pareggiare: su dodici giorni avanti è un
ottimo risultato, su un giorno solo sarebbe mediocre. Quel numero sotto la
linea di frazione serve a togliere di mezzo l'unità di misura, non a fare da
avversario.

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
di aver corretto l'asimmetria. In realtà la rovescia: a parità di errore
assoluto penalizza di più chi sottoprevede. Con $y=100$ e uno scarto di $50$
costa il $66{,}7\%$ per difetto contro il $40\%$ per eccesso, e il divario
cresce con l'errore. Nella forma qui scritta, con i valori assoluti al
denominatore, ha in più un tetto del $200\%$ che la MAPE non ha, e resta
indefinita quando $y_t = \hat y_t = 0$, cioè proprio sulle serie
intermittenti per cui la si andava cercando. Hyndman e Koehler la definiscono
invece senza quei valori assoluti, e annotano che metterceli sarebbe più
naturale ma non è l'uso corrente: la loro versione perde il tetto e in cambio
può uscire negativa, cioè smette di essere un errore percentuale. In tutte e
due le forme, gli stessi della MASE ne sconsigliano l'uso
{cite}`hyndman2006another`.

La MASE (*Mean Absolute Scaled Error*), proposta da Rob Hyndman e Anne
Koehler nel 2006 {cite}`hyndman2006another` con il passo fisso a uno e
generalizzata al passo stagionale da Hyndman e Athanasopoulos
{cite}`hyndman2021forecasting`, scala l'errore del modello sull'errore
*in-sample* del naive calcolato sul training:

$$
\text{MASE} =
\frac{\dfrac{1}{h}\sum_{j=1}^{h}\lvert e_j\rvert}
{\dfrac{1}{n-m}\sum_{t=m+1}^{n}\lvert y^{\text{tr}}_t-y^{\text{tr}}_{t-m}\rvert}.
$$

I due simboli non sono lo stesso oggetto, e la distinzione è la parte che si
sbaglia più spesso: $e_j$ sono gli $h$ errori del modello sul blocco di test,
mentre $y^{\text{tr}}$ è la serie di training, lunga $n$. Il denominatore
non si calcola mai sul test. Il passo $m$ è il periodo stagionale: vale $1$ su
una serie senza stagionalità, e va posto pari al periodo su una serie che ne
ha una {cite}`hyndman2021forecasting`. Altrimenti al denominatore finisce un
avversario che su quella serie sbaglia molto più del dovuto, e ogni modello ne
esce lusingato.

Poiché è un rapporto tra errori nella stessa unità, la MASE è
adimensionale e confrontabile tra serie, e non ha problemi con gli zeri
purché la serie di training non si ripeta identica a passo $m$: non basta che
non sia costante, perché su una serie perfettamente periodica di periodo $m$
il denominatore è zero comunque, e la MASE stagionale si usa proprio sulle
serie che un periodo ce l'hanno. La lettura, però, va data per
esteso, perché la versione corta («sotto 1 batte il naive») è la fonte di
un equivoco: un valore sotto $1$ vuol dire che il modello sbaglia meno di quanto
sbaglia, a un passo di stagione e sui dati di addestramento, il predittore
che copia il ciclo precedente. È una scala e non un duello: il denominatore
serve a togliere l'unità di misura della serie, non a fare da avversario. Un
modello con MASE $0{,}9$ su un orizzonte a dodici passi non ha battuto nessuno
alla pari: ha sbagliato il 90% di quanto sbaglia a un passo chi copia, il che
su dodici passi è ottimo e su un passo sarebbe mediocre. Se si vuole davvero
il duello, il naive va fatto correre sullo stesso test e sullo stesso
orizzonte, ed è quello che si fa con le linee di base ingenue.

Quando la previsione non è un singolo numero ma una distribuzione (un
intervallo, o un insieme di quantili), si usa la
**pinball loss** (o *quantile loss*). Per il quantile di livello
$\tau\in(0,1)$, con previsione $\hat{y}_\tau$ e valore vero $y$:

$$
\ell_\tau(y,\hat{y}_\tau) =
\begin{cases}
\tau\,(y-\hat{y}_\tau) & \text{se } y \ge \hat{y}_\tau,\\[4pt]
(1-\tau)\,(\hat{y}_\tau-y) & \text{se } y < \hat{y}_\tau.
\end{cases}
$$

Qui $\tau$ è il livello del quantile (per esempio $0{,}9$ per il novantesimo
percentile): la formula penalizza in modo asimmetrico gli sforamenti sopra e
sotto, tanto da spingere $\hat{y}_\tau$ verso il vero quantile $\tau$-esimo della
distribuzione. Mediata su più livelli, approssima un punteggio proprio per
l'intera previsione probabilistica {cite}`hyndman2021forecasting`.

Il fattore due che in molte scritture moltiplica i due rami qui non c'è, ed è
una convenzione e non un pezzo della definizione: Hyndman e Athanasopoulos lo
tengono, e annotano che spesso si omette. Raddoppiare ogni perdita non sposta
il minimo né l'ordine fra due modelli, ma raddoppia le cifre, e due punteggi si
confrontano solo se vengono dalla stessa forma.

Attenzione però a cosa misura, perché non è la calibrazione. Essere un punteggio
proprio significa che è minimizzata in media dalla distribuzione vera, e
quindi che serve benissimo come funzione di costo in addestramento e come
criterio di confronto complessivo. Ma premia insieme la calibrazione (la
banda copre davvero quello che dichiara) e la **finezza** (la banda è stretta),
e non sa dire quale delle due manca. Il conto si fa contro una gaussiana vera,
con tre previsori che azzeccano tutti la mediana e sbagliano solo la larghezza,
e mediando i nove decili.

```python
import numpy as np
from scipy.integrate import quad
from scipy.stats import norm

# La pinball loss è scritta qui senza il fattore due davanti ai due rami.
def pinball(y, q, tau):
    return tau * (y - q) if y >= q else (1 - tau) * (q - y)

livelli = np.round(np.arange(0.1, 0.91, 0.1), 2)   # i nove decili

def punteggio(larghezza):
    """Pinball media contro una N(0,1) vera, per chi dichiara quella larghezza."""
    perdite = []
    for tau in livelli:
        q = larghezza * norm.ppf(tau)       # il quantile che il modello dichiara
        peso = lambda y: pinball(y, q, tau) * norm.pdf(y)
        perdite.append(quad(peso, -12, q)[0] + quad(peso, q, 12)[0])
    return np.mean(perdite)

def copertura(larghezza):
    """Quanto copre davvero la banda fra il decimo e il novantesimo dichiarati."""
    z = larghezza * norm.ppf(0.9)
    return norm.cdf(z) - norm.cdf(-z)

print("larghezza dichiarata   pinball (nove decili)   banda «all'80%»")
for larghezza, come in [(0.5, "metà di quella vera"),
                        (1.0, "quella vera        "),
                        (2.0, "il doppio          ")]:
    print(f"  {come}          {punteggio(larghezza):.3f}"
          f"                {copertura(larghezza):.0%}")
```

```text
larghezza dichiarata   pinball (nove decili)   banda «all'80%»
  metà di quella vera          0.329                48%
  quella vera                  0.309                80%
  il doppio                    0.356                99%
```

Il punteggio li ordina, ma non dice che chi dichiara metà della larghezza
vera sta mentendo sull'incertezza e chi la dichiara doppia la sta sprecando:
il minimo ce l'ha il modello calibrato, ed è quello che ci si aspetta da un
punteggio proprio.

La formulazione canonica della materia, dovuta a Gneiting, Balabdaoui e
Raftery {cite}`gneiting2007probabilistic`, è «massimizzare la finezza sotto
vincolo di calibrazione», e la calibrazione si controlla a parte:
si conta quante volte il valore osservato cade dentro la banda all'80% e si
guarda se fa l'80%. La macchina per farlo è il walk-forward.

`````

Le due corse, messe sulla stessa linea del tempo, si guardano in un colpo
solo ({numref}`fig-mase-non-duello`): il modello parte dove finisce la storia e
copre tutto l'orizzonte in una volta, chi copia resta di qua dal confine e
avanza di un passo per volta.

```{figure} ../figures/mase-non-e-un-duello.svg
:name: fig-mase-non-duello
:alt: "Schema di che cosa mette a confronto la MASE. Una linea del tempo è divisa da una riga verticale tratteggiata: a sinistra la storia su cui il modello si è addestrato, a destra il blocco di prova. Sopra, il modello: una sola freccia lunga parte dalla riga e attraversa tutto il blocco di prova, dodici passi in un colpo solo, e i suoi errori formano il numeratore. Sotto, chi copia: una fila fitta di frecce cortissime, una per ogni giorno della storia già percorsa, tutte a sinistra della riga: sono 18, contro i dodici passi del modello, e i loro errori formano il denominatore. I due non corrono né sullo stesso tratto né sullo stesso orizzonte, e chi copia ne percorre molti di più."
:width: 92%

Che cosa mette a confronto la MASE. Sopra il numeratore, gli errori del
modello sul blocco di prova e sull'orizzonte intero; sotto il denominatore,
gli errori di chi copia sulla storia già percorsa, un passo alla volta e per
tutta la sua lunghezza, quindi molte più volte. Sono due corse su tratti
diversi, a orizzonti diversi e per un numero diverso di passi, ed è per questo
che il loro rapporto è un righello e non un verdetto.
```

## Le linee di base che bisogna sempre battere

Prima di dichiarare vittoria con una rete neurale, un modello va confrontato con
avversari volutamente banali. Se non li batte, non serve. È l'idea di linea di
base incontrata con gli {doc}`alberi decisionali e metodi ensemble
</MachineLearning/alberi-ensemble>`, il termine di paragone volutamente
semplice che un modello più elaborato deve battere; una serie storica ha i
suoi.

Le classiche sono quattro {cite}`hyndman2021forecasting`, e la prima è già in
mano: rispondere sempre la media di tutto quello che si è osservato, parente
stretta della linea piatta contro cui la sezione precedente ha misurato i
modelli classici (là era la costante che l'ARIMA si stima da sé, qui è la
media dei dati osservati e basta).
Le altre tre si scrivono con quattro simboli, e il naive stagionale ne aggiunge
un quinto che definisce sul posto: $y_t$ è il valore osservato all'istante $t$,
e in tutte e tre $t$ è l'ultimo istante osservato, l'origine da cui si guarda
avanti; il cappellino di $\hat{y}$ vuol dire «previsto» invece che «osservato»;
$h$ è quanti passi avanti si guarda, cioè l'orizzonte della previsione; e $m$ è
la lunghezza del ciclo stagionale (7 per una settimana, 12 per un anno di
mesi).

- **Naive**, cioè ingenuo: la previsione per ogni istante futuro è l’ultimo
  valore osservato, $\hat{y}_{t+h}=y_t$. Sembra una resa, e invece è
  durissimo da battere sulle passeggiate aleatorie, quelle che a ogni passo
  fanno un salto sorteggiato: i prezzi finanziari, per dire. Lì ogni scossa
  sposta il livello e ce lo lascia, perché il passo dopo riparte da dove la
  scossa ha portato, non da dove si era prima. Tutto quello che è successo fin
  lì è dunque già dentro il valore di oggi, e quello che verrà è un sorteggio
  non ancora fatto: il punto in cui la serie sta adesso *è* la migliore
  informazione che si ha su domani. Vale finché non c'è anche una deriva a
  tirare la serie da una parte: se c'è, il metodo giusto è il drift, che chiude
  l'elenco.
- **Naive stagionale**: si ripete il valore dello stesso istante del periodo
  precedente. Le vendite di questo dicembre sono quelle dello scorso
  dicembre. Quando l'orizzonte supera un ciclo intero, però, «lo stesso
  istante del periodo precedente» cade a sua volta nel futuro, e non è ancora
  stato osservato; si ricicla allora sempre l'ultimo ciclo *osservato*:
  $\hat{y}_{t+h}=y_{t+h-m(k+1)}$ con
  $k = \lfloor (h-1)/m \rfloor$, la stessa contabilità del metodo Holt-Winters
  della sezione precedente (le due parentesi tagliate in basso vogliono dire
  «arrotonda per difetto», e servono a contare quanti cicli interi si chiudono
  *prima* dell'istante da prevedere, che non è lo stesso che contarli dentro
  l'orizzonte: con $h=m$, cioè un ciclo tondo avanti, $k$ vale zero e la
  previsione è l'ultimo valore osservato). Con i numeri: siamo a dicembre, i
  mesi fanno $m=12$, e vogliamo prevedere quindici mesi avanti, cioè il marzo
  dell'anno dopo il prossimo. Allora $k = \lfloor 14/12 \rfloor = 1$, e l'indice
  da andare a pescare è $t + 15 - 12\cdot 2 = t - 9$, cioè nove mesi fa: il
  marzo scorso, che è l'ultimo marzo che abbiamo davvero visto. È la linea di
  base da battere ogni volta che c'è stagionalità.
- **Drift**, cioè deriva: come il naive, ma con una retta di tendenza
  tirata fra i due estremi della serie,
  $\hat{y}_{t+h}=y_t+h\cdot\frac{y_t-y_1}{t-1}$: la frazione è la salita media
  per passo (quanto è cresciuta la serie dal primo all'ultimo punto, diviso
  quanti passi ci sono voluti), e moltiplicandola per $h$ si prolunga in avanti
  il segmento che unisce il primo e l'ultimo punto. Con i numeri: la serie è
  partita da 10, adesso sta a 40, e fra il primo e l'ultimo punto sono passati
  30 giorni; sale dunque di $30/30 = 1$ al giorno, e la previsione per fra una
  settimana è $40 + 7 = 47$.

Disegnate sulla stessa serie sono quattro forme, e una forma si ricorda meglio
di una formula ({numref}`fig-linee-di-base`): una riga piatta a mezz'altezza,
una riga piatta all'ultimo valore, l'ultimo ciclo ricopiato in avanti, una
retta che prolunga la salita. La scena è quella del conto di poco sopra, una
serie che finisce a dicembre e una previsione a quindici mesi, e l'arco fa
vedere anche la cosa che il conto serve a evitare: oltre l'anno si ricicla
sempre l'ultimo ciclo osservato, non quello dell'anno appena prima di quello da
prevedere, che non è ancora accaduto.

```{figure} ../figures/linee-di-base.svg
:name: fig-linee-di-base
:alt: "Un grafico a linee. A sinistra la serie osservata, 36 mesi che salgono lentamente con un'onda annuale che ha la punta a dicembre; una riga verticale segna l'ultimo mese osservato, un dicembre. A destra della riga, 15 mesi di previsione, con le quattro linee di base sovrapposte alla stessa scala: grigia e piatta a mezz'altezza la media di tutta la serie, ocra e piatta all'altezza dell'ultimo valore il naive, teal e ondulata il naive stagionale, che ricopia in avanti i mesi dell'ultimo anno osservato, e terracotta in salita il drift, che prolunga la retta fra il primo e l'ultimo punto. Un arco collega il marzo osservato, nove mesi prima dell'ultimo, al marzo previsto quindici mesi dopo: è il valore che il naive stagionale copia, e la ragione per cui oltre un ciclo intero si ricicla sempre l'ultimo anno osservato invece di un anno che non è ancora accaduto."
:width: 100%

Le quattro linee di base sulla stessa serie. A sinistra i tre anni osservati,
a destra i quindici mesi previsti da ciascuna: nessuna delle quattro guarda i
dati più di così, ed è per questo che battere tutte e quattro è il minimo che
si chieda a un modello.
```

Farle correre accanto al proprio modello è il modo più diretto di accorgersi
quando un modello complicato sta imitando, e per giunta peggio, quello che una
riga di codice farebbe da sola.

## Le bande di previsione sono più strette di quello che dichiarano

Una previsione che dichiara una forbice («domani fra 22 e 26 gradi») quasi
sempre la dichiara più stretta di quanto sarebbe onesto. Vale per quasi
tutti i metodi del capitolo, ed è la parentesi che
l’{doc}`apertura del capitolo </SerieTemporali/overview>` ha lasciato aperta,
là dove dice che una previsione seria porta con sé la propria incertezza: qui
ci sono gli attrezzi per chiuderla.

Prima però va detto per bene che cosa promette una forbice, perché è una
promessa precisa e si può controllare. Quando un modello dice «fra 22 e 26,
all'80%» sta dicendo: se ripetessi questa previsione mille volte, il valore vero
mi cadrebbe dentro ottocento volte. È un conto che il modello ha fatto, non una
speranza, e poggia su due ipotesi: che i suoi parametri siano noti invece che
stimati, e che gli scarti si distribuiscano secondo la gaussiana della
{doc}`sezione su probabilità e statistica </Matematica/probabilita-statistica>`.
Nessuna delle due è vera, e non sbagliano nello stesso verso.

`````{tab} Elementare

La prima si racconta in una riga. Per dire «fra 22 e 26» il modello usa due
numeri suoi, quanto ieri pesa su oggi e quanto di solito le giornate si
discostano, e quei due numeri non glieli ha dati nessuno: se li è ricavati
dalla stessa storia che sta guardando, e poteva ricavarli un po' diversi. Il
conto della forbice fa finta di no. È un'incertezza che c'è e che nessuno
conta, e siccome nessuno la conta la forbice esce più stretta del dovuto:
sempre, e tanto più quanto la storia è corta e quanto più in là si guarda.

La seconda è più curiosa, perché non sbaglia sempre nello stesso verso. Prendi
due fenomeni che nel complesso si agitano uguale, ma uno dei due ogni tanto fa
un salto enorme. Quei pochi salti enormi, nel bilancio dell'agitazione, pesano
tantissimo, e per la ragione già vista con l'RMSE: il bilancio somma i
quadrati, e il quadrato di otto è sessantaquattro. Siccome il bilancio totale
deve restare lo stesso, tutti gli altri giorni devono essere più tranquilli. I
valori, cioè, si accalcano attorno al centro, qualcuno finisce lontanissimo, e
a diradarsi sono le vie di mezzo.

Adesso contali, e il conto viene al contrario di come sembra. Una forbice
stretta, quella che promette otto casi su dieci, sta proprio lì dove i valori
si sono accalcati, e ne raccoglie più di otto: la promessa è mantenuta e
avanza. Una forbice larghissima, quella che promette novantanove casi su
cento, arriva fin quasi in fondo alla coda, e i pochi mostri gliela passano
oltre: i casi raccolti sono meno di novantanove.

Messe insieme, allora. Sulle forbici strette, quelle che si usano tutti i
giorni, il secondo errore lavora a tuo favore e il primo resta, quindi la
forbice esce comunque un po' troppo stretta. Chi promette di coprire quasi
tutto se li trova tutti e due contro.

`````

`````{tab} Superiore

La prima ipotesi è che i parametri, stimati, vengano trattati come noti.
L'intervallo si
costruisce con $\hat\phi$ e $\hat\sigma$ al posto di $\phi$ e $\sigma$, e da
lì in poi si ragiona come se fossero i valori veri. L'incertezza sulla stima
non entra nella varianza di previsione, che esce quindi più piccola di quella
giusta, e la copertura effettiva sta sotto il livello dichiarato
{cite}`hyndman2021forecasting`. Il verso è uno solo, e lo sconto cresce al
calare della lunghezza della storia e al crescere dell'orizzonte, perché
l'errore sui parametri si compone a ogni passo.

La seconda è che i residui siano gaussiani. Quelli delle serie vere sono di solito
leptocurtici: a parità di varianza hanno più massa al centro *e* più massa
nelle code, e a diradarsi sono le zone intermedie. La conseguenza sui quantili
non ha un verso unico. Su una $t$ di Student a quattro gradi di libertà,
riscalata alla stessa deviazione standard della gaussiana (è la forma con cui
si modellano di solito i rendimenti finanziari), la banda gaussiana dichiarata
all'80% copre l'85,6% e quella dichiarata al 99% copre il 97,8%:
sovracopertura sulle bande strette, sottocopertura su quelle larghe, e il
pareggio attorno al 95%.

Messe insieme: sulle bande di uso quotidiano le due tirano in versi
opposti, la seconda a favore, e resta netto lo sconto della prima: le forbici
escono comunque più strette di quanto dichiarano. Sulle bande molto larghe si
sommano, ed è lì che la sottostima è peggiore. È anche la ragione per cui
allargare a forfait non basta: ripara il livello su cui lo si è tarato e
sposta l'errore su tutti gli altri.

`````

Le due ipotesi, quindi, non tirano dalla stessa parte, e che le forbici escano
troppo strette resta vero per merito della prima. A dover stare in guardia su
tutte e due è chi promette di coprire quasi tutto.

La buona notizia è che tutto questo si misura, e la misura ha un nome,
**copertura empirica**: si prende il walk-forward di poche righe fa, si conta
quante volte il valore osservato è caduto davvero dentro la banda, e si
confronta con il livello dichiarato.

Il conto si fa sulla stessa serie della sezione precedente, quella in cui il
valore di domani è il 60% di quello di oggi più quattro, più una scossa casuale.
I due numeri del modello (il 60% e il quattro) si ricavano dalla storia con la
retta dei minimi quadrati, esattamente come là, e la prova si ripete ventimila
volte.

```python
import numpy as np

def copertura(n_storia, orizzonte, stima, prove=20_000, seme=0):
    """Quante volte il valore vero cade nella banda all'80% dichiarata."""
    c, phi, sigma, z = 4.0, 0.6, 1.0, 1.2816
    rng = np.random.default_rng(seme)
    # una riga per prova: la stessa storia dell'AR(1), rigenerata da capo
    y = np.empty((prove, n_storia))
    y[:, 0] = c / (1 - phi) + rng.normal(0, sigma / np.sqrt(1 - phi**2), prove)
    for t in range(1, n_storia):
        y[:, t] = c + phi * y[:, t - 1] + rng.normal(0, sigma, prove)
    vero = y[:, -1].copy()
    for _ in range(orizzonte):
        vero = c + phi * vero + rng.normal(0, sigma, prove)

    if stima:      # i due numeri si ricavano dalla storia, come nella realtà
        x, b = y[:, :-1], y[:, 1:]
        mx, mb = x.mean(1, keepdims=True), b.mean(1, keepdims=True)
        p = ((x - mx) * (b - mb)).sum(1) / ((x - mx) ** 2).sum(1)
        cc = mb[:, 0] - p * mx[:, 0]
        res = b - (cc[:, None] + p[:, None] * x)
        s = np.sqrt((res ** 2).sum(1) / (n_storia - 3))
    else:          # regalati già giusti: il caso che non esiste in natura
        p, cc, s = (np.full(prove, v) for v in (phi, c, sigma))

    prev = y[:, -1].copy()
    for _ in range(orizzonte):
        prev = cc + p * prev
    var = s ** 2 * sum(p ** (2 * k) for k in range(orizzonte))
    return np.mean(np.abs(vero - prev) <= z * np.sqrt(var))

print("copertura di una banda dichiarata all'80%, su 20.000 prove")
for n_storia, orizzonte, stima in [(30, 1, False), (30, 5, False),
                                   (30, 1, True), (100, 1, True), (30, 5, True)]:
    come = "stimati " if stima else "regalati"
    avanti = "un passo" if orizzonte == 1 else f"{orizzonte} passi"
    print(f"  parametri {come}, {n_storia:3d} di storia, {avanti:8s}: "
          f"{copertura(n_storia, orizzonte, stima):.1%}")
```

```text
copertura di una banda dichiarata all'80%, su 20.000 prove
  parametri regalati,  30 di storia, un passo: 80.2%
  parametri regalati,  30 di storia, 5 passi : 79.9%
  parametri stimati ,  30 di storia, un passo: 77.7%
  parametri stimati , 100 di storia, un passo: 79.5%
  parametri stimati ,  30 di storia, 5 passi : 73.8%
```

Su ventimila prove il margine di questi numeri è di circa mezzo punto, quindi
solo gli scarti più grandi di un punto contano. Se al modello i due numeri si
regalano già giusti, la banda all'80% copre l'80%, e lo copre anche a cinque
passi: la promessa è mantenuta. Appena invece glieli si fa ricavare dalla
storia, la copertura cede, e cede di più via via che l'orizzonte si allunga,
perché l'errore sui due numeri si compone a ogni passo. È la
diagnostica più semplice della previsione probabilistica, e costa poche righe
più del walk-forward che c'è già.

## Trasformare il tempo in una tabella

Ed eccoci alla seconda metà del titolo: *rappresentare*. Buona parte dei
modelli che conosciamo (la regressione, gli alberi decisionali, le reti) non
sanno nulla di «tempo». Vogliono una tabella, come quelle della {doc}`sezione
sull'apprendimento supervisionato
</MachineLearning/apprendimento-supervisionato>`: una riga per ogni caso,
alcune colonne di domanda (le feature) e una colonna di risposta giusta (il
target), e ogni riga deve poter essere letta da sola, senza sapere che cosa
c'è nelle righe accanto. Imparare da una tabella così è ciò che si chiama
apprendimento supervisionato: si chiama così perché per ogni riga qualcuno
ha già scritto la risposta, e il modello impara confrontandosi con quella.
Costruire una tabella del genere a partire da una serie si chiama **feature
engineering temporale**, e serve a questo: una volta fatta, prevedere il futuro
torna a essere il solito problema tabellare che sappiamo già risolvere.

`````{tab} Elementare

Una riga per ogni giorno, con sopra il riassunto del suo recente passato e una
domanda sola: quanto venderò fra una settimana? I mattoni del riassunto sono
quattro.

I **lag**: i valori di ieri, dell'altroieri, di una settimana fa. Sono la
memoria grezza della serie: spesso «quanto ho venduto ieri» è già un'ottima
indicazione su oggi.

Le **finestre mobili**: media e deviazione degli ultimi 7 o 30 giorni. La media
cattura il livello recente lisciando il rumore; la deviazione (quella standard,
che misura quanto i valori si sparpagliano attorno alla loro media) dice quanto
la serie è stata mossa di recente.

L’**encoding del tempo**, cioè trasformare la data in numeri: dal calendario
ricaviamo il giorno della settimana, il mese, se è un giorno festivo. Sono le
informazioni che spiegano perché il lunedì è diverso dalla domenica e agosto da
novembre.

I **termini di Fourier**. Per dire al modello a che punto del ciclo annuale
siamo si potrebbe mettere una colonna per ciascuno dei 365 giorni, con un $1$
sul giorno giusto e $0$ sugli altri: funziona, ma sono 365 colonne per
un'informazione sola. C'è un modo più compatto, lo stesso con cui la
{doc}`sezione dal suono alle feature </Audio/dal-suono-alle-feature>`
scomponeva un accordo al pianoforte nelle poche note che lo compongono: una
curva che si ripete si descrive con poche onde regolari sovrapposte. Quelle
onde si chiamano seno e coseno, salgono e scendono all'infinito sempre uguali a
sé stesse, e bastano due o tre coppie per disegnare quasi ogni stagionalità
liscia.

Un guaio resta sul confine fra i giorni d'allenamento e quelli di prova. Le
ultime righe d'allenamento chiedono di una settimana che cade già di là; e la
prima riga di prova, per fare le sue medie, guarda indietro a giorni di qua. Si
buttano via le ultime righe d'allenamento, tante quanti i giorni d'anticipo
più il tratto più lungo che una riga guarda indietro, cioè la finestra delle
medie o il valore più vecchio che si porta dietro, quello dei due che arriva
più lontano: con una settimana d'anticipo e medie a sette giorni, quattordici
righe, un pugno di esempi in cambio di un confine pulito.
L'operazione si chiama **purga**, e il nome viene dalla finanza: lo usa
Marcos López de Prado nel capitolo sulla cross-validation di *Advances in
Financial Machine Learning* {cite}`lopezdeprado2018advances`.

`````

`````{tab} Superiore

Data la serie $y_t$, si costruisce una matrice di progetto $\mathbf{X}$ in cui
la riga
all'istante $t$ contiene solo informazione fino a $t$ (mai oltre, per non
reintrodurre leakage):

- Lag: $y_{t-1}, y_{t-2}, \dots, y_{t-p}$.
- Finestre mobili di ampiezza $w$: media
  $\frac{1}{w}\sum_{i=1}^{w} y_{t-i}$, deviazione standard, minimo, massimo.
- Variabili di calendario: giorno della settimana, mese, indicatori di
  festività, tipicamente *one-hot*.
- Termini di Fourier per una stagionalità di periodo $m$: per $k=1,\dots,K$
  si aggiungono le colonne
  $\sin\!\bigl(\tfrac{2\pi k t}{m}\bigr)$ e $\cos\!\bigl(\tfrac{2\pi k t}{m}\bigr)$.
  Poche armoniche ($K$ piccolo) bastano a rappresentare stagionalità lisce con un
  pugno di regressori, invece delle $m-1$ dummy stagionali
  {cite}`hyndman2021forecasting`.

Il target della riga $t$ è $y_{t+h}$ per l'orizzonte $h$ desiderato. A quel
punto qualunque regressore tabellare (dai modelli lineari al gradient
boosting) diventa un modello di forecasting.

Il divieto vale anche per le trasformazioni: media, deviazione, minimo e
massimo usati per scalare le colonne vanno stimati sul solo training di
quel giro e poi applicati al test, mai
calcolati sull'intera serie. Uno `StandardScaler` messo prima dello split è
esattamente l'analista con la curva liscia come una pista da sci.

E attenzione a dove cade il taglio, perché la regola «netta» del confine
temporale si viola da sé, al bordo. Se si divide train e test
guardando l'istante $t$ delle feature, le ultime $h$ righe di training
hanno un bersaglio $y_{t+h}$ che sta già dentro il periodo di test, e le
finestre mobili di ampiezza $w$ allungano la sovrapposizione di altri $w$ passi.
Si tagliano via quelle righe, ed è un'operazione che ha un nome, la purga,
preso dal capitolo settimo di *Advances in Financial Machine Learning* di
Marcos López de Prado {cite}`lopezdeprado2018advances`, intitolato appunto
alla cross-validation in finanza. Quante siano si conta senza formule da
ricordare. Ogni riga guarda all'indietro fino a un certo passo, e quel passo è
il maggiore fra i ritardi $p$ e l'ampiezza $w$ delle finestre mobili:
chiamiamolo $r$. Una riga di training all'istante $t$ tocca allora le
osservazioni da $y_{t-r}$ a $y_{t-1}$ e in più il suo bersaglio $y_{t+h}$; la
prima riga di test, all'istante $t_0+1$, legge all'indietro fino a
$y_{t_0+1-r}$. Perché le due non si sfiorino serve $t + h < t_0 + 1 - r$, e le
righe da togliere in fondo al training sono $h + r$. La tentazione naturale è
togliere solo quelle il cui bersaglio sfora, cioè $h$, e ci si dimentica
delle finestre mobili, che allungano all'indietro la parte di serie che ogni
riga di test si porta dentro. Il costo è un pugno di esempi; il guadagno è che
la regola torna vera anche al bordo.

Quanto costa tenersele, quelle righe, dipende da quanto è lungo il training. Su
una serie fortemente autocorrelata ($\phi = 0{,}9$) con $p=5$ ritardi, $w=10$ e
$h=7$, cioè diciassette righe da purgare, la stima dell'errore esce ottimista di un
paio di punti percentuali quando il training è di centoventi righe, e
l'effetto si riduce a qualche decimo quando è di quattrocento (confrontando, a
parità di numero di righe, una finestra di addestramento che arriva al confine
e una purgata; quanto esattamente dipende dalla lunghezza del blocco di test e
da quale errore si guarda, quello quadratico o la sua radice). Il guasto si
vede quando i dati sono pochi, cioè proprio quando si è più tentati di tenersele.

L’**embargo**, che quel capitolo affianca alla purga, qui invece non serve, e
chi li importa tutti e due butta via dati per difendersi da una minaccia che
non c'è. L'embargo mette una zona morta anche *dopo* il blocco di test, e
serve quando un blocco di addestramento viene dopo un blocco di prova nel
tempo, come nelle validazioni incrociate combinatorie in cui i fold si
alternano lungo la serie. Nella validazione a origine mobile il training è
sempre un prefisso e il test sempre il blocco immediatamente successivo:
nessun dato di addestramento segue mai un dato di prova, e la zona morta a
destra non avrebbe niente da proteggere.

`````

## Prevedere più passi avanti

Finora abbiamo parlato di un orizzonte, ma spesso servono molti passi: le vendite
dei prossimi 30 giorni, non solo di domani. Ci sono tre strategie, con
compromessi diversi.

`````{tab} Elementare

La strategia **ricorsiva** allena un solo modello a un passo e poi lo fa girare a
catena: prevede domani, finge che sia successo davvero, e con quel valore prevede
dopodomani, e così via. Semplice, ma ogni previsione poggia sulle precedenti: se
sbagli il primo passo, l'errore si trascina e si accumula lungo la catena.

La strategia **diretta** allena un modello *diverso* per ogni orizzonte: uno per
«tra un giorno», uno per «tra sette giorni». Nessuna previsione poggia su
un'altra, e quindi nessuna eredita gli errori delle altre; ma addestrare tanti
modelli costa, e nessuno di loro sa che cosa hanno risposto gli altri: le
previsioni, messe in fila, possono raccontare storie che non stanno insieme.

La strategia **multi-output** usa un unico modello che sputa fuori tutti i passi
futuri in un colpo solo, tutti i trenta giorni insieme invece che uno per volta,
e proprio perché escono insieme il modello può legare un giorno all'altro: è la
via naturale per le reti neurali, che possono avere molte uscite.

`````

`````{tab} Superiore

Volendo prevedere $H$ passi $\hat{y}_{t+1}, \dots, \hat{y}_{t+H}$:

- Ricorsiva (o *iterata*): si stima un solo modello a un passo
  $\hat{y}_{t+1}=f(y_t, y_{t-1}, \dots)$ e lo si applica in cascata, reinserendo
  le proprie previsioni come input, $\hat{y}_{t+2}=f(\hat{y}_{t+1}, y_t, \dots)$.
  Gli errori si propagano e si compongono lungo l'orizzonte, gonfiando la
  varianza sui passi lontani.
- Diretta: si addestra un modello distinto $f_h$ per ciascun orizzonte
  $h=1,\dots,H$, con $\hat{y}_{t+h}=f_h(y_t, y_{t-1}, \dots)$. Nessun errore
  ereditato, ma $H$ modelli da stimare e nessuna coerenza imposta tra i passi.
- Multi-output (MIMO): un'unica funzione a valori vettoriali
  $(\hat{y}_{t+1}, \dots, \hat{y}_{t+H}) = f(y_t, y_{t-1}, \dots)$, che
  modella congiuntamente le dipendenze tra gli orizzonti (la forma tipica
  delle reti neurali, con $H$ neuroni in uscita).

`````

Non esiste una scelta sempre migliore, e la regola di prima resta sovrana:
qualunque strategia si scelga, la si valuta col walk-forward, mai mescolando il
tempo.

## In pratica: walk-forward e MASE con NumPy

Mettiamo insieme i due pezzi centrali della sezione, lo split walk-forward e la
MASE, in poche righe di {doc}`NumPy </Python/numpy>`, la libreria di calcolo
numerico, e niente altro. La serie è inventata da noi, con una salita leggera e un
ciclo di sette giorni. Confrontiamo due linee di base: il naive stagionale
(ripete l'ultima settimana) e il naive semplice (ripete l'ultimo valore).

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
scalatore_1 = np.mean(np.abs(serie[1:28] - serie[:27]))   # il metro a un passo

mase_stagionale, mase_semplice = [], []
mase_stag_1, mase_sempl_1 = [], []          # gli stessi due, con l'altro metro
for idx_train, idx_test in walk_forward_split(n, min_train=28, horizon=m):
    storia, futuro = serie[idx_train], serie[idx_test]
    # naive stagionale: ricicla l'ultimo ciclo osservato. np.resize lo ripete
    # quanto serve se l'orizzonte supera m: e' la formula con k, in NumPy
    pred_stagionale = np.resize(storia[-m:], len(futuro))
    pred_semplice = np.full(len(futuro), storia[-1])   # ripeti l'ultimo valore
    mase_stagionale.append(mase(futuro, pred_stagionale, scalatore))
    mase_semplice.append(mase(futuro, pred_semplice, scalatore))
    mase_stag_1.append(mase(futuro, pred_stagionale, scalatore_1))
    mase_sempl_1.append(mase(futuro, pred_semplice, scalatore_1))

print(f"iterazioni di walk-forward: {len(mase_stagionale)}")
print(f"MASE medio - naive stagionale: {np.mean(mase_stagionale):.3f}")
print(f"MASE medio - naive semplice:   {np.mean(mase_semplice):.3f}")
print("con il metro a un passo invece che a sette:")
print(f"MASE medio - naive stagionale: {np.mean(mase_stag_1):.3f}")
print(f"MASE medio - naive semplice:   {np.mean(mase_sempl_1):.3f}")

# lo stesso conto su dieci semi: lo scalatore e' stimato su ventun differenze
# sole, quindi il metro balla, ed e' bene sapere di quanto
def mase_stagionale_con(seme):
    rng = np.random.default_rng(seme)
    s = 10 + 0.05 * t + 3 * np.sin(2 * np.pi * t / m) + rng.normal(0, 0.4, n)
    sc = np.mean(np.abs(s[m:28] - s[:28 - m]))
    return np.mean([mase(s[te], np.resize(s[tr][-m:], len(te)), sc)
                    for tr, te in walk_forward_split(n, 28, m)])

su_dieci = [mase_stagionale_con(seme) for seme in range(10)]
print(f"su dieci semi diversi: da {min(su_dieci):.2f} a {max(su_dieci):.2f}")
```

```text
iterazioni di walk-forward: 16
MASE medio - naive stagionale: 1.056
MASE medio - naive semplice:   5.058
con il metro a un passo invece che a sette:
MASE medio - naive stagionale: 0.343
MASE medio - naive semplice:   1.643
su dieci semi diversi: da 0.81 a 1.36
```

Il naive stagionale esce a $1{,}06$, cioè attorno a uno, ed era prevedibile: su
una serie con un ciclo settimanale il metro è lui, quindi sta pareggiando con
sé stesso. Attorno, non esattamente: il denominatore è stimato su ventun
differenze sole, e su dieci semi diversi il valore oscilla fra $0{,}81$ e
$1{,}36$; allargando il campione si scende sotto $0{,}7$ e si sale sopra
$1{,}6$. Il naive semplice, cieco alla settimana, sta a $5{,}06$: sbaglia
cinque volte tanto. La morale è che su una serie stagionale il metro giusto è
quello, e chi non lo batte non ha un modello.

La scelta del metro cambia il verdetto, ed è una
scorciatoia che si incontra spesso: mettendo sotto la linea di frazione il
naive a un passo invece che a sette, gli stessi due predittori escono a
$0{,}343$ e $1{,}643$, come stampano le ultime due righe, e il primo
sembrerebbe bravissimo. Non ha previsto meglio di prima: è cambiato il
righello. È la lettura che rende la MASE preziosa e
insieme la sua unica insidia: un numero senza unità dice al volo se un modello
vale più della pigrizia, purché si dichiari quale pigrizia.

`````{tab} Elementare

```{admonition} Da ricordare
:class: important
- Mescolare i dati di una serie è sbagliato, ed è la cosa che altrove si fa
  sempre. Mettere futuro e passato nello stesso mucchio è come far esercitare lo
  studente sul 4 e sul 5 marzo e poi interrogarlo sul 3: la previsione sembrerà
  miracolosa, e non lo è. Ogni dato su cui ci si allena deve venire prima,
  nel tempo, di ogni dato su cui si verifica.
- Si valuta provando all'indietro (*backtesting*): ci si mette in un giorno
  del passato, si prevede il seguito, si confronta con quello che è successo, e
  poi si sposta quel giorno in avanti e si rifà. Il pezzo su cui ci si allena può
  allungarsi ogni volta (finestra espansa) o restare lungo uguale e scivolare
  in avanti (finestra scorrevole), come nella
  {numref}`fig-walk-forward-validazione`.
- Le misure d'errore che dipendono dall'unità della serie (500 è ottimo per il
  PIL e disastroso per la temperatura) non si possono confrontare fra serie
  diverse. La percentuale toglie l'unità di misura ma ha guai suoi, a partire
  dalle scale il cui zero non è uno zero vero. Quella che funziona è la MASE:
  dice di quanto sbagli rispetto a chi
  copia e basta, e se viene 1 sbagli quanto lui, se viene $0{,}5$ la metà. Non è
  però un duello alla pari, perché chi copia corre a un passo solo e sulla
  strada già percorsa: sbagliare quanto lui su dodici giorni avanti è tutt'altra
  impresa che su domani. E va detto quale pigrizia si è messa al
  denominatore, perché su una serie con un ciclo settimanale il paragone giusto
  è con chi copia la settimana scorsa, e cambiando paragone cambia il verdetto.
- Vanno sempre battute le linee di base: chi risponde sempre la media, chi
  copia l'ultimo valore, chi copia il ciclo precedente, chi prolunga la retta
  fra il primo e l'ultimo punto. Se il modello non le supera, non serve.
- Una serie si trasforma in una tabella dando al modello, per ogni giorno, un
  riassunto del suo passato recente: i valori dei giorni prima, le medie degli
  ultimi giorni, il calendario, e poche onde regolari per dire a che punto del
  ciclo siamo. Mai niente che venga dal futuro, nemmeno di striscio; e al
  confine fra i giorni d'allenamento e quelli di prova la regola si viola da sé,
  perché le ultime righe d'allenamento chiedono di giorni che cadono già di là.
  Si buttano via, ed è la purga.
- Per prevedere molti giorni ci sono tre modi: uno alla volta rimettendo dentro
  la propria previsione (ricorsivo: economico, ma l'errore si trascina), un
  modello per ciascun giorno futuro (diretto: robusto, ma costa), o un
  modello solo che li sputa fuori tutti insieme (multi-output).
- Una previsione che dichiara una forbice («fra 22 e 26 gradi») quasi sempre la
  dichiara più stretta di quanto sarebbe onesto. Si controlla contando quante
  volte il valore vero cade davvero dentro: e qui, a differenza di ogni altro
  numero della pagina, non si punta al più alto né al più basso. Deve venire
  proprio quello promesso, perché una forbice larga il doppio copre quasi
  sempre e non dice più niente.
```

`````

`````{tab} Superiore

```{admonition} Da ricordare
:class: important
- Con le serie temporali la cross-validation con shuffle è sbagliata:
  mescolare mette futuro e passato nello stesso mucchio e produce *leakage*, con
  stime dell'errore troppo ottimiste. Ogni dato di training deve precedere nel
  tempo ogni dato di validazione, e al confine la regola va difesa con la
  purga ($h$ righe più il passo più lontano che una riga guarda
  all'indietro), non con l'embargo, che qui non ha nulla da
  proteggere.
- Si valida col walk-forward (backtesting): split cronologici ripetuti col
  test sempre nel futuro, a finestra espansa (tutto il passato) o
  scorrevole (ampiezza fissa) {cite}`hyndman2021forecasting`.
- MAE e RMSE dipendono dalla scala; la MAPE ha problemi con gli zeri ed è
  asimmetrica, e la sMAPE non attenua quell'asimmetria, la rovescia; la
  MASE {cite}`hyndman2006another` scala l'errore su quello del naive
  in-sample a passo $m$ ed è adimensionale, ma il denominatore è una scala,
  non un avversario: dichiarare quale $m$ si è usato è parte del numero. Per le
  previsioni probabilistiche si usa la pinball loss, che è un punteggio
  proprio ma premia insieme calibrazione e finezza: la calibrazione si controlla
  a parte, con la copertura empirica, che a differenza di tutte le altre non
  si minimizza né si massimizza, deve coincidere col livello dichiarato.
- Vanno sempre battute le linee di base: media, naive, naive stagionale
  ($\hat y_{t+h} = y_{t+h-m(k+1)}$, con $k = \lfloor (h-1)/m \rfloor$), drift.
  Se il modello non le supera, non serve.
- Il feature engineering temporale (lag, finestre mobili, calendario,
  termini di Fourier) riduce il forecasting a un problema supervisionato
  tabellare, senza mai usare informazione dal futuro, comprese le statistiche
  usate per scalare le colonne.
- Per il multi-step si sceglie tra strategia ricorsiva (economica, ma
  l'errore si accumula), diretta (un modello per orizzonte) e multi-output
  (un solo modello, tutti i passi).
- Le bande di previsione escono sistematicamente troppo strette, e a stringerle
  è una sola delle due comodità su cui poggiano: i parametri sono trattati come
  noti mentre sono stimati. La seconda, la normalità degli scarti, tira
  nell'altro verso e dipende dal livello: con code pesanti una banda nominale
  all'80% ne copre di più e una al 99% di meno. Su un AR(1) con trenta
  osservazioni di storia e parametri stimati ai minimi quadrati, un intervallo
  nominale all'80% ne copre il 77,7% a un passo e il 73,8% a cinque.
```

`````
