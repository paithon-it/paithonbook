# Forecasting neurale: da RNN ai Transformer e ai foundation model

Nella seconda metà degli anni Dieci, in Amazon si presentava un problema di scala
brutale: prevedere la domanda di *milioni* di prodotti, ognuno
con la sua piccola storia di vendite (spesso corta, spesso a scatti, a volte
fatta di zeri interrotti da un picco). La ricetta classica direbbe: un modello
ARIMA per prodotto. Ma stimarne milioni è impraticabile, e la maggior
parte di quelle serie è troppo breve e rumorosa perché un modello su misura ci
capisca qualcosa. Nel 2017 un gruppo di ricercatori dell'azienda propose
un'inversione di prospettiva {cite}`salinas2020deepar` (l'articolo circolò
subito online, ma fu stampato su una rivista scientifica solo tre anni dopo,
nel 2020): e se, invece di un modello per serie, addestrassimo *una sola rete
su tutte le serie insieme*, lasciando che ciascuna impari dai pattern delle
altre? È questa idea (il modello **globale**) a segnare il passaggio dal
forecasting statistico a quello neurale.

Le sezioni precedenti hanno costruito la cassetta degli attrezzi classica
(ARIMA, Holt-Winters) e hanno insistito su un punto scomodo: quei metodi di
mezzo secolo fa restano una **linea di base durissima da battere**. Questa
sezione racconta l'altra famiglia, quella delle reti neurali: quando conviene
davvero, come si è evoluta dalle reti ricorrenti alle convoluzioni causali
fino ai Transformer, e dove sta arrivando oggi con i *foundation model*. Con
la stessa onestà: il deep learning non è un miglioramento automatico.

## Quando conviene il deep learning

La domanda non è se una rete sappia imparare una serie temporale: sa farlo. La
domanda è quando ne conviene, e la risposta dipende meno dall'architettura che
dalla forma dei dati che si hanno davanti.

`````{tab} Elementare

Un unico negozio di quartiere ha una storia di vendite corta e ballerina, e un
modello troppo sofisticato ci si perde: con tanti numeri da regolare e pochi
giorni su cui regolarli, finisce per imparare a memoria anche gli sbalzi
capitati per caso, che domani non si ripeteranno. Un buon vecchio ARIMA fa
altrettanto bene con molta meno fatica. Ora prendi una **catena** con
diecimila negozi. Molti pattern sono condivisi: il picco del sabato, il crollo
di Ferragosto, l'effetto di una promozione. Una rete neurale può impararli
*una volta sola* guardando tutti i negozi insieme, e poi applicarli anche al
negozio aperto il mese scorso, che da solo non avrebbe abbastanza storia.

Il deep learning conviene quando ricorrono quattro condizioni, e la catena le
ha tutte: **tante serie collegate** fra loro; relazioni **non lineari** (una
promozione raddoppia le vendite solo sotto una certa soglia di prezzo);
**fattori esterni** che aiutano a prevedere (meteo, festività, prezzo); e il
bisogno di una **stima dell'incertezza** invece di un numero solo, perché chi
ordina la merce deve sapere di quanto rischia di sbagliare. Se invece hai una
sola serie, pulita e lunga, i classici restano spesso la scelta migliore, e
comunque la prima da provare.

Le famiglie, però, non sono due. I metodi classici e le reti provano a
indovinare, ciascuno a modo suo, la regola con cui un giorno genera il
successivo; una terza sta in mezzo, e su moltissimi problemi aziendali basta.
Prende il calendario e ci disegna sopra una curva, sommando pezzi che si
possono guardare uno per uno. Una tendenza di fondo, che è una linea spezzata
(una retta che ogni tanto cambia pendenza, come una strada che sale e a un
certo punto sale meno: dove cambiarla lo decide guardando i dati). Una o più
stagionalità, disegnate con le poche onde regolari (seno e coseno) che la
sezione sulle feature ha appena introdotto, e possono essercene due
sovrapposte, la settimana lavorativa e il ciclo annuale insieme. E gli effetti
delle **festività**, che sono strappi su date dichiarate a mano.

Il programma più usato di questa famiglia si chiama **Prophet** ed è stato
pubblicato nel 2018 da due ricercatori di Facebook, Sean Taylor e Benjamin
Letham {cite}`taylor2018forecasting`. La sua fortuna si spiega in fretta: si
costruisce sui dati in un attimo, e ogni pezzo si può mostrare a chi non fa
questo mestiere.

C'è però una differenza di fondo dalle altre due famiglie, e spiega insieme il
pregio e il limite: qui il valore di domani non dipende dal valore di oggi.
Dipende solo dalla data. È una curva tirata sui
punti, non una catena di giorni che si tengono per mano. Per questo i buchi nei
dati e i valori assurdi non rompono niente (non c'è nessuna catena da
interrompere), e per questo su una
serie in cui oggi somiglia molto a ieri il metodo butta via proprio
l'informazione più preziosa, e perde contro un ARIMA banale.

`````

`````{tab} Superiore

Il vantaggio strutturale è il **modello globale**: un'unica funzione
parametrica $f_\theta$ addestrata sull'intero insieme di $N$ serie
$\{\mathbf{x}^{(i)}_{1:T_i}\}_{i=1}^N$, in luogo di $N$ modelli locali
indipendenti.
Con $N$ grande, $\theta$ vede molti più esempi e può permettersi capacità che
una singola serie non giustificherebbe, catturando pattern condivisi
(stagionalità tipiche, effetti di calendario) e regolarizzando le serie corte
con quelle lunghe. Le altre tre leve sono la **non linearità** (mappe che un
ARMA lineare non rappresenta), le **covariate esogene** $\mathbf{z}_t$
integrate nell'input,
$\hat{x}_{t+1} = f_\theta(\mathbf{x}_{1:t}, \mathbf{z}_{1:t+1})$, e l'uscita
**probabilistica**
$p_\theta(\mathbf{x}_{t+1:t+h}\mid \mathbf{x}_{1:t}, \mathbf{z})$, non una
stima puntuale.

L'onestà impone il rovescio della medaglia. Con **poche** serie o serie
**corte** il regime dati non regge la varianza di un modello ad alta capacità,
e i metodi statistici (robusti, frugali, interpretabili) tengono il campo,
come mostrano ripetutamente le competizioni M discusse nell'introduzione al
capitolo. La regola resta quella: si batte prima la linea di base classica,
poi si tira in ballo la rete.

Fra le due famiglie ne vive una terza, che non è né un processo stocastico né
una rete, e che su moltissimi problemi aziendali basta: i **modelli additivi
decomponibili**, di cui Prophet {cite}`taylor2018forecasting` è l'esemplare
diffuso. La serie si scrive come

$$
y(t) = g(t) + s(t) + h(t) + \varepsilon_t ,
$$

con $g$ una tendenza lineare **a tratti**, i cui punti di svolta sono stimati
dai dati (è la notazione dell'articolo: questa $g$ non ha niente a che vedere
con il $g_\theta$ di DeepAR); $s$ una somma di **serie di
Fourier troncate** a $K$ armoniche (le stesse colonne $\sin(2\pi kt/m)$ e
$\cos(2\pi kt/m)$ della sezione sulle feature, il che permette di sovrapporre
più periodi e di regolare la flessibilità scegliendo $K$); $h$ gli effetti
puntuali su date dichiarate (è la notazione dell'articolo: $h(t)$ è una
funzione del tempo, non l'orizzonte di previsione che $h$ indica altrove);
e $\varepsilon_t$ il residuo. Il modello è scritto in forma bayesiana, ma di
norma la stima si ferma al **massimo a posteriori** invece di campionare
l'intera distribuzione. Anche gli intervalli non vengono da una formula chiusa:
si ottengono simulando in avanti cambi di pendenza con la stessa frequenza e la
stessa ampiezza di quelli visti nel passato, che è un'ipotesi forte, e che gli
autori dichiarano per quello che è.

Il punto che la distingue dalle altre due, e che vale più della formula, è che
$y$ è una regressione **sul tempo**, non sui valori passati: non c'è nessuno
stato, nessuna dipendenza fra $y(t)$ e $y(t-1)$. Non è un processo stocastico,
è una curva interpolata. Da qui vengono insieme il pregio (i dati mancanti e gli
*outlier* non rompono niente, perché non c'è nessuna ricorsione da interrompere,
e la stima è veloce) e il limite: il modello **non sfrutta l'autocorrelazione a
breve**, che è precisamente ciò che rende prevedibile una serie molto correlata,
ed è la ragione per cui su quelle serie perde contro un ARIMA banale.

`````

## Reti ricorrenti per il forecasting

Il primo strumento neurale per le sequenze lo abbiamo già costruito, nel
{doc}`capitolo sul Natural Language Processing </NaturalLanguageProcessing/overview>`:
una rete che legge un pezzo per volta e si porta dietro un riassunto di quello che ha letto fin lì. Quel
riassunto si chiama **stato nascosto**, ed è tutta la memoria che la rete ha.
Le **reti ricorrenti** (RNN) funzionano così; la **LSTM** di Sepp Hochreiter e
Jürgen Schmidhuber {cite}`hochreiter1997long` è la versione che a ogni passo
decide anche che cosa di quel riassunto conviene tenere e che cosa buttare
(sono i suoi **cancelli**). Lì il problema era il linguaggio, qui è una serie
di numeri, ma il meccanismo è lo stesso: una serie temporale, in fondo, è una
frase di numeri.

`````{tab} Elementare

Con una rete così il forecasting si fa in due modi.

Il primo: le dai in pasto gli ultimi giorni, mettiamo gli ultimi trenta, e le
chiedi un numero solo, quello di domani. Si addestra come qualunque modello che
deve indovinare un numero: in ingresso la finestra di giorni, come risposta
giusta il giorno dopo. Per andare più in là di domani si riapplica la rete a
catena, rimettendole dentro le sue stesse previsioni: è la strategia
**ricorsiva** della sezione precedente, con il suo errore che si trascina.

Il secondo: una prima rete legge tutta la storia e ne fa un riassunto, una
seconda srotola da quel riassunto l'intera settimana futura in un colpo solo. È
lo stesso schema con cui si traduce una frase, ed è proprio da lì che, nel
trattamento del linguaggio, è nata l'idea di lasciare che il modello decida da
sé a quali pezzi del passato dare peso.

Ci sono però due guai, e nel forecasting pesano più che altrove. Il primo è che
la memoria di queste reti **si consuma**: più il passato è lontano, meno ne
resta nel riassunto, e le LSTM migliorano le cose senza risolverle. Il secondo è
che queste reti si addestrano **in fila indiana**, perché il giorno $t$ non si
può calcolare finché non è finito il $t-1$: il lavoro non si può spartire fra le
migliaia di unità di calcolo di una GPU, e l'addestramento è lento.

`````

`````{tab} Superiore

Con una RNN il forecasting prende due forme. Nel **seq-to-one** («da sequenza a
uno») la rete legge la finestra dei giorni passati, cioè gli ultimi $w$ valori
$x_{t-w+1}, \dots, x_t$, dove $w$ è quanto la si vuole lunga, e produce un solo
valore, la previsione del prossimo passo: si allena come una regressione, con la
finestra come input e $x_{t+1}$ come bersaglio. Per prevedere più in là si
riapplica la rete in modo **ricorsivo**, reiniettando le proprie stime, con
l'accumulo dell'errore già visto nell'introduzione. Nel **seq-to-seq**, invece,
una prima rete (l’*encoder*) riassume tutta la storia in una manciata di numeri,
e una seconda (il *decoder*) srotola da quel riassunto l'intero orizzonte futuro
in un colpo, esattamente come una traduzione genera l'intera frase d'uscita. È lo
schema che, nell'NLP, ha fatto nascere il meccanismo di attenzione.

I limiti, però, sono gli stessi del capitolo NLP, e nel forecasting pesano
persino di più. La memoria delle RNN semplici si **dissolve** su orizzonti
lunghi (il gradiente che svanisce) e le LSTM lo mitigano ma non lo cancellano;
l'addestramento è **sequenziale**, mal parallelizzabile sulle GPU perché il
passo $t$ aspetta il $t-1$.

`````

Da qui la ricerca di architetture che guardino lontano nel tempo *senza* pagare
la ricorrenza. La prima risposta arriva, curiosamente, dalle convoluzioni.

## TCN: convoluzioni che guardano solo indietro

Una **convoluzione** è l'operazione del {doc}`capitolo sul Deep Learning </DeepLearning/overview>`: una piccola
finestra che scorre sui dati e a ogni posizione fa sempre lo stesso conto,
prendere i valori che ha sotto, pesarli e sommarli. Là scorreva su un'immagine,
qui scorre su una fila di giorni. Il vantaggio, rispetto a una rete che legge un
giorno per volta, è che tutte le posizioni si possono calcolare **insieme**,
perché nessuna aspetta il risultato di un'altra.

Nel 2018 Shaojie Bai, Zico Kolter e Vladlen Koltun pubblicarono un confronto
sistematico tra reti ricorrenti e reti convoluzionali sulle sequenze, e la
conclusione fece rumore: su un ampio ventaglio di compiti una semplice rete
convoluzionale, opportunamente adattata, eguagliava o superava le LSTM
{cite}`bai2018empirical`. La chiamarono, riprendendo un nome che altri
ricercatori usavano già, **Temporal Convolutional Network** (TCN).

Gli accorgimenti che la rendono adatta al tempo sono due. Il primo è la
**causalità**: la finestra può prendere il giorno corrente e quelli prima, mai
quelli dopo (altrimenti la rete «bara», guardando la risposta). Il secondo è la
**dilatazione**: per abbracciare un passato lungo senza impilare centinaia di
strati, la finestra di ogni strato non prende giorni attaccati fra loro ma
giorni distanziati, e la distanza raddoppia salendo. La
{numref}`fig-tcn-convoluzioni-causali` li mostra tutti e due insieme.

```{figure} ../figures/tcn-convoluzioni-causali.svg
:name: fig-tcn-convoluzioni-causali
:alt: Una rete convoluzionale temporale con convoluzioni causali dilatate. In basso otto nodi di ingresso lungo l'asse del tempo; sopra tre strati di nodi collegati da archi che saltano sempre all'indietro con dilatazione 1, 2 e 4. Nessun arco punta al futuro. Un cono ombreggiato evidenzia il campo recettivo, pari a otto istanti. In alto a destra il nodo di uscita in terracotta.
:width: 100%

Una TCN con convoluzioni **causali** (nessun arco viene dal futuro) e
**dilatate** (i salti raddoppiano a ogni strato: 1, 2, 4). Ogni nodo guarda due
soli nodi dello strato sotto, ma bastano tre strati perché il nodo in cima ne
raccolga $2^3 = 8$: quanto passato arriva a un singolo nodo di uscita si chiama
**campo recettivo**, e cresce in modo esponenziale con la profondità.
```

`````{tab} Elementare

Dieci strati bastano per tenere d'occhio più di mille giorni di diario, e
nessuno di essi sbircia una pagina non ancora scritta. Guardare soltanto
all'indietro è la causalità. Coprire tanti giorni con pochi strati, invece, è
questione di salti che si allargano: il primo strato guarda ieri e oggi, il
secondo salta di due giorni, il terzo di quattro, il quarto di otto. Quattro
strati arrivano così a sedici giorni indietro, dieci a più di mille. È la
**dilatazione**.

Uno strato non riscrive il diario da capo: si tiene accanto la pagina com'era e
ci annota soltanto quello che ha da aggiungere, e anche una pila alta di strati
resta facile da correggere. Il tutto senza ricorrenza: ogni istante si calcola
in parallelo agli altri, e l'addestramento vola sulle GPU invece di procedere
in fila.

`````

`````{tab} Superiore

Una convoluzione 1-D **causale** con kernel di dimensione $k$ calcola l'uscita
al tempo $t$ usando solo $x_t, x_{t-1}, \dots, x_{t-(k-1)}$. Aggiungendo un
fattore di **dilatazione** $d$, i campioni vengono presi a distanza $d$:

$$
y_t = \sum_{i=0}^{k-1} w_i \, x_{t - i\cdot d},
$$

dove $w_0, \dots, w_{k-1}$ sono i pesi del filtro (condivisi lungo tutta la
sequenza) e $d$ è il passo di dilatazione. Impilando $L$ strati con dilatazioni
$d_\ell = 2^{\ell}$ per $\ell = 0, \dots, L-1$, il **campo recettivo** è

$$
r = 1 + (k-1)\sum_{\ell=0}^{L-1} 2^{\ell} = 1 + (k-1)\,(2^{L}-1),
$$

cioè cresce **esponenzialmente** con la profondità $L$: con $k=2$ e $L=3$ si
ha $r = 8$, come in figura; con $k=3$ e $L=6$ si arriva a $r = 127$ istanti.
La formula vale per **una** convoluzione per livello, che è la forma del codice
più avanti; l'implementazione originale di Bai e colleghi ne mette **due** per
blocco residuo, e allora il campo recettivo raddoppia,
$1 + 2(k-1)(2^L-1)$, cioè quel $127$ diventa $253$. Chi applica la formula
sbagliata a una TCN vera ne sottostima il campo recettivo di un fattore due.

In pratica ogni blocco aggiunge una **connessione residua** (nello spirito
delle ResNet) per addestrare pile profonde senza che il gradiente svanisca.
Qui «residua» vuol dire un'altra cosa ancora rispetto ai residui statistici del
capitolo: non «quello che avanza dopo aver tolto trend e stagione», ma una
scorciatoia che porta l'ingresso oltre il blocco, così che al blocco resti da
imparare soltanto la differenza. Rispetto a una RNN, il calcolo è interamente
parallelizzabile lungo il tempo: $O(1)$ passi sequenziali invece di $O(n)$.

`````

## DeepAR: una rete per mille serie, e una distribuzione

Torniamo al problema di Amazon con cui si è aperta la sezione. **DeepAR**
{cite}`salinas2020deepar` è la risposta neurale, e porta due idee da tenere
distinte.

La prima è il modello **globale**, già discusso: una rete sola, addestrata su
tutte le serie insieme, che legge un giorno per volta e si porta dietro il
proprio riassunto del passato (è una LSTM, quella del {doc}`capitolo sul Natural
Language Processing </NaturalLanguageProcessing/overview>`). A ogni passo riceve due cose, il valore del giorno prima e
le informazioni esterne di quel giorno: il calendario, il prezzo, una promozione
già decisa. Sono le variabili **esogene** della sezione sui modelli classici, che
in questa letteratura cambiano nome e si chiamano **covariate**.

La seconda idea è più sottile: DeepAR non predice un *numero*, predice un
ventaglio di valori possibili con le loro probabilità. Un oggetto del genere, in
statistica, si chiama **distribuzione**.

`````{tab} Elementare

Un bollettino serio non dice «domani piove»: dice «70% di probabilità di
pioggia». DeepAR fa lo stesso con le vendite. A ogni passo, invece di sputare
una cifra, descrive un **ventaglio di futuri plausibili**: il valore più
probabile e quanto ci si può discostare. E un bollettino così si giudica in un
modo solo, da quanta probabilità aveva dato a quello che poi è successo
davvero: alzare quel punteggio, su tutte le serie insieme, è tutto
l'addestramento della rete. E ogni serie, prima di entrare, passa dalla
bilancia: la si divide per la propria media, e alla fine la previsione si
rimoltiplica per quella. Senza, il prodotto da trentamila pezzi al giorno
coprirebbe quello da tre, e la rete imparerebbe soltanto dal più grande.

Per prevedere una settimana intera, la rete «tira i dadi» tante volte (genera
migliaia di storie possibili, e in ciascuna il valore appena tirato diventa il
punto di partenza del tiro successivo) e poi legge il ventaglio. Il valore che
sta esattamente in mezzo, con metà delle storie sotto e metà sopra, si chiama
**mediana**, ed è la previsione; e si scarta il 10% delle storie più basse e il
10% delle più alte, così quello che resta in mezzo è la **banda di
incertezza**, dentro cui cadono otto storie su dieci. Questo modo di procedere,
tira un valore e ripartici, ha un nome: **campionamento ancestrale**. Una
previsione seria è un numero *con la sua incertezza*.

`````

`````{tab} Superiore

A ogni passo la rete emette i **parametri** $\boldsymbol{\lambda}_t$ di una
distribuzione di verosimiglianza $p(x_t \mid \boldsymbol{\lambda}_t)$: una
gaussiana, e allora $\boldsymbol{\lambda}_t=(\mu_t,\sigma_t)$, per dati reali;
una **binomiale negativa** per conteggi non negativi (come le vendite). È in
grassetto perché ha più di una componente; la stessa lettera altrove nel libro
è il coefficiente di penalità, qui no. La loss è la log-verosimiglianza
cambiata di segno, sommata su tutte le serie,

$$
\mathcal{L}(\theta) = -\sum_{i=1}^{N} \sum_{t} \log p\big(x^{(i)}_t \mid
\boldsymbol{\lambda}^{(i)}_t\big), \qquad
\boldsymbol{\lambda}^{(i)}_t = g_\theta\big(\mathbf{h}^{(i)}_t\big),
$$

dove $\mathbf{h}^{(i)}_t$ è lo stato nascosto della LSTM per la serie $i$,
$\boldsymbol{\lambda}^{(i)}_t$ i parametri d'emissione che ne discendono (quindi
anch'essi propri di quella serie) e $\theta$ i parametri *condivisi* fra tutte
le serie: minimizzare $\mathcal{L}$ equivale a massimizzare la verosimiglianza
dei dati.

C'è poi un pezzo senza il quale un modello globale non sta in piedi, e che il
racconto di solito salta: prima di entrare nella rete ogni serie viene divisa per
una sua **scala** (tipicamente la media dei suoi valori), e la previsione viene
rimoltiplicata per la stessa. Senza, una serie da trentamila unità e una da tre
non possono stare nella stessa rete, e la seconda sparisce nel gradiente della
prima.

La previsione multi-passo avviene per **campionamento ancestrale**, e siccome
si lavora su una serie alla volta l'indice $i$ resta d'ora in poi sottinteso:
si estrae $\hat{x}_{t+1}\sim p(\cdot\mid \boldsymbol{\lambda}_{t+1})$, lo si
reinietta come input, si ripete fino all'orizzonte; molte traiettorie così
ottenute forniscono, per ogni passo, i **quantili** della previsione. Nessuna
formula chiusa per gli intervalli: è Monte Carlo.

`````

Il campionamento ancestrale è così centrale che conviene vederlo girare, su un
modellino giocattolo di venti righe. La rete
vera, a ogni passo, calcola due numeri: attorno a che valore si aspetta il
giorno dopo, e di quanto quel giorno può discostarsene. Qui quei due numeri non
li impara nessuno, glieli diamo noi con la regola più semplice che abbiamo, la
stessa AR(1) della sezione sui modelli classici (il 60% del valore di ieri, più
quattro), perché quello che conta guardare è il meccanismo: campiona,
reinietta, ripeti.

```python
import numpy as np

rng = np.random.default_rng(0)

# A ogni passo il "modello" predice media e deviazione del prossimo valore.
def prossimo(x_prec):
    mu = 4.0 + 0.6 * x_prec     # parte deterministica (media condizionata)
    sigma = 1.0                 # incertezza a un passo
    return mu, sigma

# Previsione probabilistica a 5 passi per CAMPIONAMENTO ANCESTRALE:
# molte traiettorie, ciascuna reinietta il proprio campione come input.
orizzonte, n_traj = 5, 20000
x_T = 12.0
traj = np.zeros((n_traj, orizzonte))
for j in range(n_traj):
    x = x_T
    for h in range(orizzonte):
        mu, sigma = prossimo(x)
        x = rng.normal(mu, sigma)   # si CAMPIONA, non si prende la media
        traj[j, h] = x

# Dai campioni ricaviamo i quantili: la banda di previsione.
q10, q50, q90 = np.percentile(traj, [10, 50, 90], axis=0)
for h in range(orizzonte):
    print(f"t+{h+1}:  mediana {q50[h]:5.2f}   banda 80% [{q10[h]:5.2f}, {q90[h]:5.2f}]"
          f"   larga {q90[h] - q10[h]:.3f}")
```

```text
t+1:  mediana 11.20   banda 80% [ 9.91, 12.47]   larga 2.556
t+2:  mediana 10.73   banda 80% [ 9.22, 12.22]   larga 3.008
t+3:  mediana 10.42   banda 80% [ 8.87, 12.02]   larga 3.149
t+4:  mediana 10.25   banda 80% [ 8.68, 11.83]   larga 3.159
t+5:  mediana 10.15   banda 80% [ 8.58, 11.74]   larga 3.166
```

Cinque righe, una per giorno previsto: la mediana delle ventimila storie
generate e i due estremi della banda.

Si vede la mediana rientrare verso la media di lungo periodo,
che è la stessa della sezione sui classici, cioè il valore che passando per la
regola resta uguale a sé stesso, $\mu = 4/(1-0{,}6) = 10$. E si vede la banda
allargarsi: la distanza fra i suoi due estremi, che il programma stampa in
fondo a ogni riga, passa da $2{,}56$ a $3{,}01$ a $3{,}15$.

Poi, dal terzo giorno in poi, la salita **quasi si spegne**: $3{,}149$,
$3{,}159$, $3{,}166$, cioè meno di un centesimo per volta. Ed è la ragione per
cui la banda va stampata con tre decimali e non con due: arrotondata al
centesimo, la crescita degli ultimi tre giorni sparirebbe, e si concluderebbe
che si è fermata. Non si è fermata: quel che le resta da crescere è ormai una
manciata di millesimi. Sta arrivando al suo limite, che qui vale $3{,}20$, e al
quinto giorno ne ha già raggiunto il $99{,}7\%$.[^banda-limite]

Ed è la parte più istruttiva del programmino. È il rovescio del rientro verso la
media dei modelli classici: una serie che torna sempre verso il proprio valore
centrale non può diventare indefinitamente imprevedibile, e la sua banda si
assesta su quella di lungo periodo.

A crescere senza fermarsi è invece l'incertezza delle serie che un valore
centrale non ce l'hanno, le **non stazionarie** dell'introduzione al capitolo.
E ci sarebbe da aggiungere l'incertezza sul modello stesso, che qui non c'è per
un motivo un po’ furbesco: la regola con cui la serie viene generata è la stessa
con cui la prevediamo, perché l'abbiamo scritta noi da tutte e due le parti. Su
una serie vera quella regola non la conosce nessuno, va indovinata dai dati, e
può venire sbagliata.

Un'ultima onestà sul giocattolo. Qui la regola è così semplice che quei cinque
intervalli si potrebbero calcolare anche con carta e penna, senza generare
nessuna storia, e infatti i due conti coincidono a un paio di centesimi: la
differenza che resta è il tremolio del sorteggio, non un difetto del metodo.
Generare tante storie a caso e leggere il ventaglio che ne viene fuori ha un
nome, **metodo Monte Carlo**, come il casinò, ed è la sola strada praticabile
nella rete vera, dove il conto passa per una LSTM e per ventagli che una formula
chiusa non ce l'hanno. Ma è questo, in ogni caso, che una previsione
probabilistica dichiara e una puntuale nasconde.

## N-BEATS: solo percettroni, ma interpretabili

Nel 2019 Boris Oreshkin e colleghi mostrarono che per battere i metodi statistici
non servivano né ricorrenza né convoluzioni. Bastavano gli strati più ordinari
che ci siano, quelli in cui ogni neurone guarda tutti quelli dello strato
precedente (i **percettroni** del capitolo sulle Reti neurali), montati con la
giusta architettura {cite}`oreshkin2020nbeats`.[^date-nbeats] **N-BEATS**
(*Neural Basis Expansion Analysis for Time Series*) è fatto solo di quelli, e la
sua eleganza sta in un'idea di contabilità: il **doppio residuo**.

`````{tab} Elementare

Una serie si spiega a strati, come si pela una cipolla. Il primo blocco guarda
la finestra passata e produce due cose: una **ricostruzione** di
ciò che ha capito del passato (il *backcast*) e un pezzo di **previsione** del
futuro (il *forecast*). A questo punto si *sottrae* la ricostruzione dal
passato: ciò che resta è quello che il primo blocco non ha saputo spiegare, e
passa al secondo blocco, che ripete il gioco su quel residuo. Blocco dopo
blocco, ogni strato spiega un pezzo in più; le previsioni parziali si
**sommano** a formare quella finale.

Il bello viene quando a qualche blocco si lega la mano. A uno si concede di
disegnare soltanto curve dolci, e finisce per occuparsi della tendenza di
fondo; a un altro soltanto onde che si ripetono, e finisce sui cicli. Così la
rete non solo prevede, ma **mostra** quanto della previsione è tendenza e
quanto è ciclo (cosa rara per una rete neurale).

`````

`````{tab} Superiore

Ogni blocco $b$ riceve un residuo $\mathbf{x}^{(b)}$ e, tramite una pila di
strati densi seguita da una proiezione su una base, produce due uscite: un
**backcast** $\hat{\mathbf{x}}^{(b)}$ (la parte d'ingresso che sa spiegare) e un
**forecast** $\hat{\mathbf{y}}^{(b)}$ (il suo contributo alla previsione). Il
**doppio residuo** li combina così:

$$
\mathbf{x}^{(b+1)} = \mathbf{x}^{(b)} - \hat{\mathbf{x}}^{(b)}, \qquad
\hat{\mathbf{y}} = \sum_{b} \hat{\mathbf{y}}^{(b)},
$$

dove l'ingresso del blocco successivo è ciò che il blocco corrente **non** ha
saputo spiegare, e la previsione finale è la **somma** dei contributi. Nella
variante *interpretabile* la base è vincolata (polinomi di grado basso per il
trend, termini di Fourier per la stagionalità), così che i due stack
restituiscano componenti leggibili; in quella *generica* la base è appresa
liberamente. Nessuna componente statistica innestata: solo strati densi.

`````

Una nota sulla parola, adesso che la cosa c'è. «Residuo» prende qui il terzo
senso del capitolo, e i tre sono imparentati: era l'imprevisto che avanzava
dalla decomposizione, era l'errore che avanzava da un modello stimato, e adesso
è quello che avanza da un blocco della rete e passa al blocco dopo. Ogni volta
è «ciò che non è stato spiegato», e cambia solo chi ha provato a spiegarlo.

La variante in cui alcuni blocchi si occupano della tendenza e altri della
stagione si chiama **interpretabile**, e riallaccia il forecasting neurale alla
**decomposizione classica** della prima sezione, chiudendo il cerchio con i
modelli statistici. Per arrivarci, mostrò N-BEATS, non serviva innestare pezzi
di statistica dentro la rete, come faceva l'ibrido che aveva vinto la M4 (quello
dell'introduzione al capitolo): contro l'opinione corrente di allora, come
scrivono gli autori, i mattoni del deep learning si bastavano da soli
{cite}`oreshkin2020nbeats`.

## Transformer per le serie, e un lineare che li imbarazza

Nel {doc}`capitolo sui Transformer </Transformers/overview>` il libro ha raccontato una rete che, invece di
leggere una sequenza un pezzo per volta, guarda tutti i pezzi in una volta sola
e decide da sé a quali dare peso: quel «decidere a quali dare peso» è
l’**attenzione**, e nel trattamento del linguaggio ha spazzato via le reti che
leggevano in fila {cite}`vaswani2017attention`. Portarla nelle serie temporali
era una tentazione irresistibile, per una ragione precisa: un Transformer mette
in comunicazione due giorni lontanissimi con **un solo passaggio**, mentre una
rete ricorrente deve trascinarsi l'informazione attraverso tutti i giorni in
mezzo, ed è per questo che se la dimentica.

Fioccarono architetture dedicate, e le due più citate sono due risposte opposte
allo stesso problema. Il problema è il tempo di calcolo: confrontare ogni giorno
con ogni altro giorno vuol dire, su una finestra di mille giorni, un milione di
confronti, e più si allunga la finestra più il conto esplode. **Informer** taglia
i confronti, e ne fa solo una parte, scelta bene, invece di tutti.
**Autoformer** cambia proprio domanda: invece di confrontare coppie di giorni
cerca le somiglianze della serie con sé stessa fatta scivolare indietro, cioè
l’**autocorrelazione** della prima sezione, da cui viene il nome, e per
calcolarle tutte in fretta usa una scorciatoia che il libro ha già incontrato
nel capitolo sull'audio, la trasformata di Fourier (la stessa famiglia di conti
da cui vengono il seno e il coseno della sezione precedente). Fra un blocco e
l'altro, poi, scompone la serie in tendenza e stagionalità.

Poi, nel 2022, una doccia fredda.

`````{tab} Elementare

Un gruppo di ricercatori pose una domanda scomoda già nel titolo: «I
Transformer servono davvero, per prevedere le serie temporali?». La risposta,
sui banchi di prova più usati, fu spiazzante: dei modelli **lineari**
semplicissimi, poco più di una retta tirata sui dati, battevano quei Transformer
sofisticati. Ne provarono più d'uno: il più elaborato della famiglia separa
prima la serie in tendenza e stagionalità e poi tira una retta su ciascuna delle
due, e lo chiamarono **DLinear**; il più semplice è una retta e basta.

E non si fermarono al risultato. Con la retta più semplice fecero due prove che
valgono più della classifica. Nella prima mescolarono l'ingresso: presero i
giorni passati da dare in pasto al modello e li rimisero in ordine sparso. Un
modello che usa davvero l'ordine del tempo, così, dovrebbe crollare. Sui cambi
fra valute i Transformer non se ne accorsero per niente, mentre la retta
peggiorò di un quarto: lì il tempo lo stava usando lei. Su altre serie, invece,
mescolare fa male a tutti quanti. Che è già una lezione: quanto un modello usi
l'ordine del tempo non è una sua proprietà fissa, e si scopre misurandola,
banco di prova per banco di prova.

Nella seconda allungarono il passato da leggere. Chi tira fuori qualcosa da una
storia lunga dovrebbe prevedere meglio quando gliene si dà di più; i
Transformer, con la finestra più lunga, restavano fermi o peggioravano, mentre
la retta migliorava quasi dappertutto.

La morale non è «i Transformer non servono», ma qualcosa di più prezioso: la
complessità non è mai un vantaggio gratuito. Prima di celebrare un modello
elaborato, va confrontato con la linea di base più stupida che ti viene in
mente, e va controllato che stia usando l'informazione che dice di usare. A
volte la retta vince.

`````

`````{tab} Superiore

Il costo quadratico $O(n^2)$ dell'attenzione piena sulle sequenze lunghe aveva
motivato le varianti efficienti (Informer, Autoformer, FEDformer). Zeng e
colleghi {cite}`zeng2023transformers` proposero come confronto una famiglia di
modelli lineari: il più semplice è una singola mappa
$\hat{\mathbf{x}}_{t+1:t+h} = \mathbf{W}\,\mathbf{x}_{t-w+1:t}$, e la
variante **DLinear** decompone prima la serie in trend e stagionalità e applica
una mappa a ciascuna componente. Su nove dataset di forecasting a lungo
orizzonte quella famiglia eguaglia o supera i Transformer dedicati.

Il verdetto da solo sarebbe una classifica, e le classifiche invecchiano. Quello
che non invecchia sono le due prove con cui gli autori lo spiegano, fatte con il
modello lineare semplice, ed è a quello che si riferiscono i numeri.

*La prima è il mescolamento dell'ingresso.* Se un modello usa davvero l'ordine
temporale, rimescolare a caso le posizioni della finestra passata deve
rovinarlo. Sui tassi di cambio le prestazioni dei metodi basati su Transformer
**non si muovono** (lo scostamento medio è dell'ordine di un decimo di punto
percentuale, e per due dei tre è perfino in meglio) mentre lo stesso
trattamento
fa perdere il 27% al modello lineare: lì il tempo lo sta usando la retta.

Su ETTh1 il quadro è un altro, e va detto perché è la metà che si cita di meno:
lì mescolando peggiorano anche i Transformer che hanno un'idea del tempo dentro,
FEDformer del 73% e Autoformer del 57% (il lineare dell'81%). Informer no: si
ferma al 2%. La conclusione onesta, quindi, non è che i Transformer siano
ciechi al tempo per costruzione, ed è più utile: un banco di prova su cui un
modello non peggiora affatto quando gli si distrugge l'ordine non sta misurando
quello che dichiara di misurare.

*La seconda è la lunghezza della finestra passata.* Un modello che estrae
relazioni temporali da una storia lunga deve migliorare quando gliene si dà di
più. I Transformer, allungando la finestra, restano fermi o peggiorano; i
modelli lineari migliorano sulla maggior parte di quei dataset (non su tutti:
sui tassi di cambio, dicono gli autori, la finestra lunga non aiuta nemmeno
loro). Messe insieme, le due
prove dicono che su quei banchi di prova l'attenzione non stava estraendo le
relazioni temporali che dichiarava di estrarre, il che è una critica al
**metodo di valutazione** prima che all'architettura.

Non è la condanna dei Transformer: il **Temporal Fusion Transformer** di Lim e
colleghi {cite}`lim2021temporal` resta forte quando servono **covariate
multiple** (statiche, note nel futuro, osservate nel passato) e
**interpretabilità**, grazie alle reti di selezione delle variabili, alla
scomposizione dell'importanza per orizzonte e a pesi di attenzione
ispezionabili. Su quest'ultimo punto vale la cautela che il capitolo
sull'interpretabilità impone: i pesi di attenzione sono un indizio suggestivo,
non una spiegazione affidabile {cite}`jain2019attention`, e le prime due leve
reggono anche senza di lui. Resta comunque un promemoria di metodo: sempre una
linea di base, sempre onesta.

`````

## Foundation model: la «GPT delle serie»

L'ultima frontiera prende in prestito l'idea che ha reso possibili i grandi
modelli linguistici: pre-addestrare **un** modello enorme su una collezione
sterminata di serie, e poi usarlo per prevedere serie **mai viste**, senza
riaddestrarlo. È il forecasting *zero-shot*, e il nome che gira è, appunto, «la
GPT delle serie temporali».

`````{tab} Elementare

Come un modello linguistico impara la lingua leggendo miliardi di frasi e poi
sa scrivere su argomenti nuovi, un *foundation model* per le serie impara la
«grammatica» dei fenomeni temporali (tendenze, cicli, picchi) leggendo milioni
di serie di ogni tipo. Poi gli dai una serie che non ha mai incontrato, per
esempio le vendite del tuo negozio, e lui prevede il seguito **senza
addestrarsi da capo**. Il modello Chronos, di Amazon, usa un trucco
sorprendente: trasforma i numeri in **parole**. Si sceglie un vocabolario di
qualche migliaio di simboli e si stabilisce che ciascuno copre una fettina di
valori, così «$23{,}7$ gradi» diventa, mettiamo, il simbolo numero $1372$;
a quel punto una serie è una frase, e prevedere il seguito è completare la
frase, che è esattamente il mestiere per cui i modelli linguistici sono già
fatti. E siccome una frase si può finire in molti modi, il modello ne scrive
tante e legge il ventaglio che ne viene: anche qui la previsione arriva con la
sua incertezza.

Promettente, ma è un campo giovane: non batte sempre i metodi su misura, né i
vecchi classici. E quando lo si mette alla prova su una serie, resta il sospetto
che quella serie fosse già fra i milioni che ha letto: chi ha già visto il
compito non sta improvvisando.

`````

`````{tab} Superiore

**Chronos** {cite}`ansari2024chronos` scala e **quantizza** i valori reali in
un vocabolario finito di *token*, poi addestra un modello linguistico della
famiglia T5 con la consueta *cross-entropy* sul token successivo, su un grande
corpus di serie reali e sintetiche. In inferenza campiona traiettorie di token
e le riconverte in valori, ottenendo una previsione **probabilistica** senza
architetture ad hoc. Un'alternativa è **TimesFM** di Google, *decoder-only*
che lavora su *patch* di istanti, pre-addestrato su ordini di grandezza di
miliardi di punti temporali. Entrambi mostrano uno *zero-shot* competitivo con
modelli allenati sul singolo compito: un risultato notevole. La cautela, però,
è d'obbligo: il campo ha pochi anni, la valutazione soffre di possibili
**contaminazioni** tra corpus di pre-addestramento e benchmark, e su serie
corte o molto specifiche i metodi classici restano spesso preferibili.
Promettente, non risolto.

`````

## In pratica: una TCN in PyTorch

Traduciamo la TCN della {numref}`fig-tcn-convoluzioni-causali` in codice. Il
cuore è la convoluzione causale, e ottenerla richiede un piccolo trucco, che è
poi quello dell'implementazione originale.

Il problema è questo. Una finestra che scorre su una fila di numeri, arrivata al
primo giorno, non ha niente alla sua sinistra: le mancano dei valori, e la
libreria li mette a zero. Solo che li mette **da tutte e due le parti**, anche a
destra, dove il tempo non è ancora arrivato, e così il risultato viene più lungo
dell'ingresso. La soluzione è chiedere gli zeri lo stesso e
poi **tagliare via la coda a destra**: quello che resta è come se gli zeri
fossero stati messi soltanto a sinistra, la sequenza è lunga quanto prima e
l'uscita del giorno $t$ non ha mai pescato nel futuro.

Quanti zeri chiedere lo dicono i due numeri che descrivono lo strato: quanti
valori la finestra prende ($k$, la sua ampiezza) e quanto sono distanziati fra
loro ($d$, il salto). Il conto è $(k-1)\,d$: con una finestra da tre valori
distanziati di due, per esempio, ne servono quattro.

```python
import torch
import torch.nn as nn

class Taglia(nn.Module):
    """Rimuove gli ultimi `n` istanti: preserva la causalità."""
    def __init__(self, n):
        super().__init__()
        self.n = n

    def forward(self, x):                 # x: (batch, canali, tempo)
        return x[:, :, :-self.n].contiguous() if self.n > 0 else x

class BloccoTCN(nn.Module):
    def __init__(self, c_in, c_out, kernel=3, dilation=1):
        super().__init__()
        pad = (kernel - 1) * dilation      # padding causale (a sinistra nel tempo)
        self.conv = nn.Conv1d(c_in, c_out, kernel, padding=pad, dilation=dilation)
        self.taglia = Taglia(pad)          # elimina il padding di troppo a destra
        self.relu = nn.ReLU()
        # connessione residua: adatta i canali con una conv 1x1 se necessario
        self.giu = nn.Conv1d(c_in, c_out, 1) if c_in != c_out else None

    def forward(self, x):
        y = self.relu(self.taglia(self.conv(x)))   # uscita causale, stessa lunghezza
        r = x if self.giu is None else self.giu(x)
        return self.relu(y + r)

class TCN(nn.Module):
    # nel disegno le finestre erano da 2 valori e tre strati arrivavano a 8
    # istanti; qui sono da 3, e con tre strati il campo recettivo diventa 15
    # (in generale 1+(k-1)(2^L-1)): passare finestre più lunghe è sprecato
    def __init__(self, c_in=1, canali=32, kernel=3, n_blocchi=3):
        super().__init__()
        strati = []
        for i in range(n_blocchi):
            d = 2 ** i                     # dilatazione 1, 2, 4, ...
            ci = c_in if i == 0 else canali
            strati.append(BloccoTCN(ci, canali, kernel, dilation=d))
        self.rete = nn.Sequential(*strati)
        self.testa = nn.Linear(canali, 1)  # dall'ultimo istante -> previsione

    def forward(self, x):                  # x: (batch, tempo), serie univariata
        h = self.rete(x.unsqueeze(1))      # (batch, canali, tempo)
        return self.testa(h[:, :, -1])     # ultimo istante -> (batch, 1)
```

L'addestramento è il consueto ciclo di discesa del gradiente del capitolo su
PyTorch. Si taglia la serie in coppie, la finestra dei giorni passati e il
valore del giorno dopo, rispettando la separazione temporale imposta dalla
sezione sulla validazione e senza mai mescolare futuro e passato. Poi si passano
le finestre al modello e si spingono le sue uscite verso i valori giusti,
misurando la distanza con l'errore quadratico (`nn.MSELoss`) e lasciando che a
correggere i pesi ci pensi `torch.optim.Adam`.

Basta poco per farne un modello **probabilistico**, nello spirito di DeepAR:
invece di far uscire dalla rete un numero solo se ne fanno uscire due, il valore
centrale e quanto ci si può discostare, e si cambia il bersaglio
dell'addestramento. Non più «avvicina la previsione al valore vero», ma «rendi
il valore vero il più probabile possibile secondo la forbice che stai
dichiarando»: in gergo, si sostituisce la MSE con la log-verosimiglianza
gaussiana cambiata di segno. La rete impara così anche a dire quando non sa.

Dalle reti ricorrenti ai foundation model le architetture cambiano, ma i due
comandamenti del capitolo restano gli stessi: prevedere significa **dichiarare
l'incertezza**, e nessun modello, per quanto profondo, è dispensato dal
confronto con la **linea di base** classica.

`````{tab} Elementare

```{admonition} Da ricordare
:class: important
- Il deep learning conviene quando hai **tante serie collegate** fra loro (la
  catena di diecimila negozi, non il negozio di quartiere), quando i legami non
  sono semplici proporzioni, quando ci sono cause esterne che aiutano, e quando
  ti serve non un numero ma una forbice. Su poche serie corte i **classici**
  spesso vincono ancora.
- Fra le due famiglie ce n'è una terza, e su moltissimi problemi aziendali
  basta: **Prophet** non insegue la regola con cui un giorno genera il
  successivo, prende il calendario e ci disegna sopra una curva fatta di pezzi
  che si guardano uno per uno (una tendenza spezzata, le stagioni, le
  festività). Per questo i buchi nei dati non lo rompono. E per lo stesso
  motivo, su una serie in cui oggi somiglia molto a ieri, butta via
  l'informazione migliore che ha e perde contro un ARIMA banale
  {cite}`taylor2018forecasting`.
- Le **reti ricorrenti** leggono la serie un giorno per volta portandosi dietro
  una memoria, ma su storie lunghe se la dimenticano e sono lente da addestrare.
  Le **TCN** {cite}`bai2018empirical` risolvono entrambe le cose rileggendo il
  diario **a salti che raddoppiano**: quattro strati vedono sedici giorni, dieci
  ne vedono più di mille, e nessuno strato può sbirciare in avanti.
- **DeepAR** {cite}`salinas2020deepar` addestra **una sola rete su tutte le
  serie insieme** e non prevede un numero, prevede un ventaglio di futuri
  possibili: tira i dadi tante volte, ogni volta ripartendo dal valore appena
  tirato, e poi legge il ventaglio. Su una serie che rientra sempre verso la
  propria media, quel ventaglio si allarga per qualche passo e poi **si ferma**.
- **N-BEATS** {cite}`oreshkin2020nbeats` pela la serie a strati come una
  cipolla: ogni blocco spiega quello che può e passa al successivo quello che
  resta. Con un vantaggio raro per una rete: si può fare in modo che mostri
  quanto della previsione è tendenza e quanto è ciclo.
- Sui **Transformer** per le serie conviene la cautela: una retta ben usata li
  ha battuti su molti banchi di prova {cite}`zeng2023transformers`, e su alcuni
  di quei banchi, mescolando l'ordine dei giorni, i Transformer non se ne sono
  nemmeno accorti: il tempo, lì, non lo stavano usando. I **foundation model**
  come **Chronos**
  {cite}`ansari2024chronos` imparano la «grammatica» dei fenomeni temporali su
  milioni di serie e poi prevedono serie mai viste: campo promettente e giovane,
  non risolto.
```

`````

`````{tab} Superiore

```{admonition} Da ricordare
:class: important
- Il deep learning conviene con **molte serie collegate**, relazioni **non
  lineari**, **covariate esterne** e quando serve una **previsione
  probabilistica**; su poche serie corte i **classici** spesso vincono ancora.
  Fra le due famiglie stanno i **modelli additivi decomponibili** (Prophet
  {cite}`taylor2018forecasting`), che regrediscono sul tempo e non sui valori
  passati: robusti ai buchi, ciechi all'autocorrelazione a breve.
- Le **RNN/LSTM** {cite}`hochreiter1997long` fanno forecasting *seq-to-one* o
  *seq-to-seq*, ma soffrono orizzonti lunghi e addestramento sequenziale. Le
  **TCN** {cite}`bai2018empirical` usano convoluzioni **causali dilatate**: campo
  recettivo esponenziale, $1+(k-1)(2^L-1)$ con una convoluzione per livello e il
  doppio meno uno con le due del blocco originale, e calcolo parallelizzabile.
- **DeepAR** {cite}`salinas2020deepar` è una RNN autoregressiva **globale** (una
  rete per molte serie) che emette una **distribuzione**; la previsione
  multi-passo è per campionamento ancestrale. La banda si allarga con
  l'orizzonte **se** il processo non è stazionario; se lo è, converge alla
  varianza di lungo periodo e si ferma.
- **N-BEATS** {cite}`oreshkin2020nbeats` usa blocchi di soli **MLP**
  (percettroni multistrato: pile di strati densi) con **doppio residuo**
  backcast/forecast, ed è interpretabile quando la base è vincolata. Il suo
  contributo è di meccanismo: per battere gli ibridi statistico-neurali non
  serviva innestare statistica dentro la rete.
- Sui Transformer per le serie, cautela: la famiglia **LTSF-Linear** di Zeng e
  colleghi (il lineare semplice e la sua variante con decomposizione,
  **DLinear**) li eguaglia o supera su nove dataset
  {cite}`zeng2023transformers`, e le due prove che lo spiegano, fatte sul
  lineare semplice (su una serie mescolare l'ingresso non li scalfisce, e
  allungare la finestra non li migliora), dicono che su quei banchi l'ordine
  temporale non era quello che stavano sfruttando. Il **TFT**
  {cite}`lim2021temporal` resta utile per covariate multiple e interpretabilità,
  purché i suoi pesi di attenzione si leggano come indizio e non come prova
  {cite}`jain2019attention`. I **foundation model** come **Chronos**
  {cite}`ansari2024chronos` promettono forecasting *zero-shot*: campo promettente
  ma giovane, non risolto.
```

`````

In tutto il capitolo l'unica conoscenza a disposizione è stata la storia del
fenomeno. Nessuno ha spiegato al modello perché la marea sale, e la marea si
prevede lo stesso, finché le regolarità tengono. Il {doc}`capitolo sulle PINN </PINN/overview>` parte dal caso opposto, quello in cui la legge che governa il fenomeno si conosce
benissimo e a scarseggiare sono le misure.

[^banda-limite]: I conti, per chi li vuole. Un AR(1) con $|\phi|<1$ ha
    varianza di lungo periodo $\sigma^2/(1-\phi^2)$, che con $\sigma=1$ e
    $\phi=0{,}6$ dà una deviazione di $1{,}25$; una banda all'80% è larga
    $2\times 1{,}2816$ deviazioni, quindi al limite vale $3{,}20$. A $h$ passi
    la deviazione vale $\sigma\sqrt{(1-\phi^{2h})/(1-\phi^2)}$, e il rapporto
    fra le due è $\sqrt{1-\phi^{2h}}$, che a cinque passi (quindi con
    l'esponente $10$, non $5$) fa $0{,}997$. La banda vera al terzo, quarto e
    quinto passo è dunque $3{,}13$, $3{,}18$, $3{,}19$: cresce ancora, di
    centesimi, ed è quello che il campionamento non riesce più a distinguere.

[^date-nbeats]: L'articolo circolò online nel 2019 e fu presentato in conferenza
    l'anno dopo, come era successo a DeepAR. Questi due lavori sono datati alla
    prima circolazione, che è la data in cui l'idea è entrata nel campo. Quando
    le due date si allontanano e la distanza conta, il capitolo la dichiara; la
    voce in bibliografia resta invece quella della pubblicazione, ed è normale
    che le due non coincidano.
