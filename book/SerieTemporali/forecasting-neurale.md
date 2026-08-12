# Forecasting neurale: da RNN ai Transformer e ai foundation model

Nella seconda metà degli anni Dieci, in Amazon si presentava un problema di scala
brutale: prevedere la domanda di *centinaia di milioni* di prodotti, ognuno
con la sua piccola storia di vendite (spesso corta, spesso a scatti, a volte
fatta di zeri interrotti da un picco). La ricetta classica direbbe: un modello
ARIMA per prodotto. Ma stimarne cento milioni è impraticabile, e la maggior
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

`````{tab} Elementare

Immagina un unico negozio di quartiere. La sua storia di vendite è corta e
ballerina: un modello troppo sofisticato ci si perde, e un buon vecchio ARIMA
fa altrettanto bene con molta meno fatica. Ora immagina una **catena** con
diecimila negozi. Molti pattern sono condivisi: il picco del sabato, il crollo
di Ferragosto, l'effetto di una promozione. Una rete neurale può impararli
*una volta sola* guardando tutti i negozi insieme, e poi applicarli anche al
negozio aperto il mese scorso, che da solo non avrebbe abbastanza storia.

Il deep learning conviene quando ricorrono quattro condizioni: hai **tante
serie collegate** fra loro; le relazioni sono **non lineari** (una promozione
raddoppia le vendite solo sotto una certa soglia di prezzo); ci sono **fattori
esterni** che aiutano a prevedere (meteo, festività, prezzo); e vuoi non un
singolo numero ma una **stima dell'incertezza**. Se invece hai una sola serie,
pulita e lunga, i classici restano spesso la scelta migliore, e comunque la
prima da provare.

C'è poi una terza famiglia, che sta in mezzo e che vale la pena conoscere prima
di aprire il capitolo neurale, perché su moltissimi problemi aziendali basta.
Le prime due famiglie provano a indovinare la **regola** con cui un giorno
genera il successivo. Questa fa una cosa diversa: prende il calendario e ci
disegna sopra una curva, sommando pezzi che si possono guardare uno per uno. Una
tendenza di fondo, che è una linea spezzata (una retta che ogni tanto cambia
pendenza, come una strada che sale e a un certo punto sale meno). Una o più
stagionalità, disegnate con le poche onde regolari che abbiamo appena
incontrato, e possono essercene due sovrapposte, la settimana lavorativa e il
ciclo annuale insieme. E gli effetti delle **festività**, che sono strappi su
date dichiarate a mano.

Il programma più usato di questa famiglia si chiama **Prophet** ed è stato
pubblicato nel 2018 da due ricercatori di Facebook, Sean Taylor e Benjamin
Letham {cite}`taylor2018forecasting`. La sua fortuna si spiega in fretta: si
stima in un attimo, non si rompe se mancano dei giorni o se c'è un valore
assurdo, e ogni pezzo si può mostrare a chi non fa questo mestiere.

C'è però una differenza di fondo dalle altre due famiglie, e conviene tenerla a
mente perché spiega insieme il pregio e il limite: qui **il valore di domani non
dipende dal valore di oggi**. Dipende solo dalla data. È una curva tirata sui
punti, non una catena di giorni che si tengono per mano. Per questo i buchi non
rompono niente (non c'è nessuna catena da interrompere), e per questo su una
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
e i metodi statistici (robusti, frugali, interpretabili), tengono il campo,
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
dai dati; $s$ una somma di **serie di Fourier troncate** a $K$ armoniche (le
stesse colonne $\sin(2\pi kt/m)$ e $\cos(2\pi kt/m)$ della sezione sulle
feature, il che permette di sovrapporre più periodi e di regolare la
flessibilità scegliendo $K$); $h$ gli effetti puntuali su date dichiarate; e
$\varepsilon_t$ il residuo. La stima è bayesiana, e da lì vengono anche gli
intervalli.

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
capitolo sul *Natural Language Processing*: le **reti ricorrenti** (RNN) e la
loro versione con i cancelli, la **LSTM** di Sepp Hochreiter e Jürgen
Schmidhuber {cite}`hochreiter1997long`. Lì il problema era il linguaggio; qui
è una serie di numeri, ma il meccanismo è identico: una cella che scorre nel
tempo mantenendo uno **stato nascosto**, la sua memoria di ciò che ha letto
finora. Una serie temporale, in fondo, è una frase di numeri.

Con una RNN il forecasting prende due forme. Nel **seq-to-one** («da sequenza a
uno») la rete legge la finestra dei giorni passati, cioè gli ultimi $w$ valori
$x_{t-w+1}, \dots, x_t$, dove $w$ è quanto la si vuole lunga, e produce un solo
valore, la previsione del prossimo passo: si allena come una regressione, con la
finestra come input e $x_{t+1}$ come bersaglio. Per prevedere più in là si
riapplica la rete in modo **ricorsivo**, reiniettando le proprie stime, con
l'accumulo dell'errore già visto nell'introduzione. Nel **seq-to-seq**, invece,
una prima rete (l'*encoder*) riassume tutta la storia in una manciata di numeri,
e una seconda (il *decoder*) srotola da quel riassunto l'intero orizzonte futuro
in un colpo, esattamente come una traduzione genera l'intera frase d'uscita. È lo
schema che, nell'NLP, ha fatto nascere il meccanismo di attenzione.

I limiti, però, sono gli stessi del capitolo NLP, e nel forecasting pesano
persino di più. La memoria delle RNN semplici si **dissolve** su orizzonti
lunghi (il gradiente che svanisce) e le LSTM lo mitigano ma non lo cancellano;
l'addestramento è **sequenziale**, mal parallelizzabile sulle GPU perché il
passo $t$ aspetta il $t-1$. Da qui la ricerca di architetture che guardino
lontano nel tempo *senza* pagare la ricorrenza. La prima risposta arriva,
curiosamente, dalle convoluzioni.

## TCN: convoluzioni che guardano solo indietro

Nel 2018 Shaojie Bai, Zico Kolter e Vladlen Koltun pubblicarono un confronto
sistematico tra reti ricorrenti e reti convoluzionali sulle sequenze, e la
conclusione fece rumore: su un ampio ventaglio di compiti una semplice rete
convoluzionale, opportunamente adattata, eguagliava o superava le LSTM
{cite}`bai2018empirical`. La chiamarono, riprendendo un nome che altri
ricercatori usavano già, **Temporal Convolutional Network** (TCN). Due
accorgimenti la rendono adatta al tempo, e la
{numref}`fig-tcn-convoluzioni-causali` li mostra insieme.

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

Il primo accorgimento è la **causalità**: l'uscita all'istante $t$ può
dipendere solo da $t$ e dagli istanti precedenti, mai da quelli futuri
(altrimenti la rete «bara», guardando la risposta). Il secondo è la
**dilatazione**: per abbracciare un passato lungo senza impilare centinaia di
strati, ogni strato salta indietro sempre più lontano.

`````{tab} Elementare

Pensa di ripercorrere un diario, ma con una regola: puoi guardare solo le pagine
*già scritte*, mai quelle future. Questa è la causalità. Per abbracciare mesi di
diario senza rileggere ogni singolo giorno, procedi a salti sempre più larghi:
il primo strato guarda ieri e oggi, il secondo salta di due giorni, il terzo di
quattro, il quarto di otto. In pochi salti che raddoppiano hai coperto una
finestra lunghissima. È la **dilatazione**: raddoppiando l'ampiezza a ogni
strato, quattro strati vedono sedici giorni indietro, dieci ne vedono più di
mille. Il tutto senza ricorrenza: ogni istante si calcola in parallelo agli
altri, così l'addestramento vola sulle GPU invece di procedere in fila.

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
$1 + 2(k-1)(2^L-1)$, cioè quel $127$ diventa $253$. Chi dimensiona una TCN vera
con la formula sbagliata la sottodimensiona di un fattore due.

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
{cite}`salinas2020deepar` è la risposta neurale, e porta due idee che vale la
pena tenere distinte. La prima è il modello **globale**, già discusso: un'unica
LSTM autoregressiva addestrata su tutte le serie insieme, che a ogni passo riceve
il valore precedente e le covariate e aggiorna il proprio stato. La seconda, più
sottile, è che DeepAR non predice un *numero* ma una *distribuzione*.

`````{tab} Elementare

Un bollettino serio non dice «domani piove»: dice «70% di probabilità di
pioggia». DeepAR fa lo stesso con le vendite. A ogni passo, invece di sputare
una cifra, descrive un **ventaglio di futuri plausibili**: il valore più
probabile e quanto ci si può discostare. Per prevedere una settimana intera,
la rete «tira i dadi» tante volte (genera centinaia di storie possibili, e in
ciascuna il valore appena tirato diventa il punto di partenza del tiro
successivo) e poi legge il ventaglio: la mediana è la previsione, l'ampiezza
tra il decimo e il novantesimo percentile è la **banda di incertezza**. Questo
modo di procedere, tira un valore e ripartici, ha un nome che ricorrerà fra
poco: **campionamento ancestrale**. È il filo rosso del capitolo: una previsione
seria è un numero *con la sua incertezza*.

`````

`````{tab} Superiore

A ogni passo la rete emette i **parametri** $\lambda_t$ di una distribuzione di
verosimiglianza $p(x_t \mid \lambda_t)$: una gaussiana $\lambda_t=(\mu_t,\sigma_t)$
per dati reali, una **binomiale negativa** per conteggi non negativi (come le
vendite). La loss è la log-verosimiglianza cambiata di segno, sommata su tutte
le serie,

$$
\mathcal{L}(\theta) = -\sum_{i=1}^{N} \sum_{t} \log p\big(x^{(i)}_t \mid
\lambda^{(i)}_t\big), \qquad
\lambda^{(i)}_t = g_\theta\big(\mathbf{h}^{(i)}_t\big),
$$

dove $\mathbf{h}^{(i)}_t$ è lo stato nascosto della LSTM per la serie $i$,
$\lambda^{(i)}_t$ i parametri d'emissione che ne discendono (quindi anch'essi
propri di quella serie) e $\theta$ i parametri *condivisi* fra tutte le serie:
minimizzarla equivale a massimizzare la verosimiglianza dei dati. La
previsione multi-passo
avviene per **campionamento ancestrale**, e siccome si lavora su una serie alla
volta l'indice $i$ resta d'ora in poi sottinteso: si estrae
$\hat{x}_{t+1}\sim p(\cdot\mid \lambda_{t+1})$, lo si reinietta come input, si
ripete fino all'orizzonte; molte traiettorie così ottenute forniscono, per ogni
passo, i **quantili** della previsione. Nessuna formula chiusa per gli
intervalli: è Monte Carlo.

`````

Il campionamento ancestrale (tira un valore a caso fra quelli plausibili,
rimettilo dentro come se fosse successo, ripeti) è così centrale che conviene
vederlo girare, su un modellino giocattolo in puro NumPy. La rete vera
predirebbe a ogni passo la media $\mu_t$ e la deviazione $\sigma_t$ del prossimo
valore; qui le fissiamo con una semplice regola autoregressiva, la stessa
AR(1) della sezione precedente, e ci concentriamo sul meccanismo: campiona,
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
    print(f"t+{h+1}:  mediana {q50[h]:5.2f}   banda 80% [{q10[h]:5.2f}, {q90[h]:5.2f}]")
```

Eseguendolo si vede la mediana rientrare verso la media di lungo periodo, che è
la stessa di prima: il valore che resta uguale a sé stesso passando per la
regola, $\mu = 4/(1-0{,}6) = 10$. E si vede la banda allargarsi: da $2{,}6$ a
$3{,}0$ a $3{,}2$.

Poi, dal terzo passo, la banda **si ferma**, e vale la pena capire perché,
perché è più istruttivo dell'allargamento. È il rovescio del rientro verso la
media della sezione precedente: una serie che torna sempre verso il proprio
valore centrale non può diventare indefinitamente imprevedibile, e la sua banda
converge a quella della distribuzione di lungo periodo (qui la deviazione
asintotica vale $1{,}25$, e al quinto passo siamo già al 99,6% di quel valore).
L'incertezza che invece cresce senza fermarsi è quella delle serie **non
stazionarie**, e quella che viene dall'errore del modello, che qui non c'è
perché il modello lo abbiamo scritto noi.

Un'ultima onestà sul giocattolo: essendo tutto lineare e gaussiano, i quantili
di queste cinque righe si potrebbero anche calcolare a mano, e infatti tornano a
due decimali. Il Monte Carlo diventa l'unica strada nella rete vera, dove la
ricorsione passa per una LSTM e l'emissione può essere una binomiale negativa:
lì una formula chiusa non c'è. Questo è ciò che una previsione puntuale nasconde
e una probabilistica dichiara.

## N-BEATS: solo percettroni, ma interpretabili

Nel 2019 Boris Oreshkin e colleghi mostrarono che per battere i metodi statistici
non servivano né ricorrenza né convoluzioni: bastavano **percettroni**, impilati
con la giusta architettura (l'articolo circolò online quell'anno e fu presentato
in conferenza nel 2020, come per DeepAR: qui i lavori sono datati alla prima
circolazione) {cite}`oreshkin2020nbeats`. **N-BEATS** (*Neural Basis
Expansion Analysis for Time Series*) è fatto di blocchi di soli strati densi, e
la sua eleganza sta in un'idea di contabilità: il **doppio residuo**. Qui
«residuo» è il terzo senso che la parola prende nel capitolo, ed è imparentato
con gli altri due: dopo l'imprevisto della decomposizione e l'errore del modello,
è quello che un blocco della rete non ha saputo spiegare e che passa al blocco
successivo. Il senso lontano è quello della *connessione residua* di poco fa, che
non è quello che avanza ma una scorciatoia fra strati.

`````{tab} Elementare

Immagina di spiegare una serie a strati, come si pela una cipolla. Il primo
blocco guarda la finestra passata e produce due cose: una **ricostruzione** di
ciò che ha capito del passato (il *backcast*) e un pezzo di **previsione** del
futuro (il *forecast*). A questo punto si *sottrae* la ricostruzione dal
passato: ciò che resta è quello che il primo blocco non ha saputo spiegare, e
passa al secondo blocco, che ripete il gioco su quel residuo. Blocco dopo
blocco, ogni strato spiega un pezzo in più; le previsioni parziali si
**sommano** a formare quella finale. Il bello: si può dedicare uno stack a
catturare il *trend* e uno la *stagionalità*, e allora la rete non solo
prevede, ma **mostra** quanto della previsione è tendenza e quanto è ciclo
(cosa rara per una rete neurale).

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

La variante interpretabile riallaccia il forecasting neurale alla
**decomposizione classica** della prima sezione (trend e stagionalità come
uscite separate) chiudendo il cerchio con i modelli statistici. Ed è stato
N-BEATS a mostrare che per arrivarci non serviva innestare pezzi di statistica
dentro la rete, come faceva l'ibrido vincitore della M4: bastavano strati densi
organizzati bene. Contro l'opinione corrente di allora, come scrivono gli
autori, i mattoni del deep learning si bastavano da soli
{cite}`oreshkin2020nbeats`.

## Transformer per le serie, e un lineare che li imbarazza

Se l'attenzione ha spazzato via la ricorrenza nell'NLP (è la tesi di
*«Attention Is All You Need»* {cite}`vaswani2017attention`, cui il libro
dedica un intero capitolo), la tentazione di portarla nelle serie temporali
era irresistibile. Un Transformer, in teoria, collega due istanti lontani con
*un solo salto* di attenzione, aggirando la memoria che si dissolve delle RNN.
Fioccarono così architetture dedicate: **Informer**, con un'attenzione sparsa
per abbattere il costo sulle sequenze lunghe, e **Autoformer**, che sostituisce
l'attenzione con un meccanismo di **autocorrelazione** basato sulla trasformata
di Fourier (da cui il nome, e il costo $O(n\log n)$) e alterna blocchi di
decomposizione in trend e stagionalità.

Poi, nel 2022, una doccia fredda.

`````{tab} Elementare

Un gruppo di ricercatori pose una domanda scomoda già nel titolo: «I
Transformer servono davvero, per prevedere le serie temporali?». La risposta,
sui banchi di prova più usati, fu spiazzante: un modello **lineare**
semplicissimo (poco più di una retta tirata sui dati, dopo averli separati in
trend e stagionalità), che chiamarono **DLinear**, batteva quei Transformer
sofisticati.

E non si fermarono al risultato. Fecero una prova che vale più della classifica:
presero i giorni passati da dare in pasto al modello e li **mescolarono**, in
ordine sparso. Un modello che usa davvero l'ordine del tempo, così, dovrebbe
crollare. I Transformer non se ne accorsero quasi, mentre DLinear peggiorò
moltissimo. Cioè: su quei dati, il modello che stava usando il tempo era la
retta, e i Transformer stavano facendo qualcos'altro.

La morale non è «i Transformer non servono», ma qualcosa di più prezioso: la
complessità non è mai un vantaggio gratuito. Prima di celebrare un modello
elaborato, va confrontato con la linea di base più stupida che ti viene in
mente, e va controllato che stia usando l'informazione che dice di usare. A
volte la retta vince.

`````

`````{tab} Superiore

Il costo quadratico $O(n^2)$ dell'attenzione piena sulle sequenze lunghe aveva
motivato le varianti efficienti (Informer, Autoformer, FEDformer). Zeng e
colleghi {cite}`zeng2023transformers` proposero come confronto **DLinear**: si
decompone la serie in trend e stagionalità e si applica a ciascuna componente
una singola mappa lineare
$\hat{\mathbf{x}}_{t+1:t+h} = \mathbf{W}\,\mathbf{x}_{t-w+1:t}$. Su nove
dataset di forecasting a lungo orizzonte, DLinear eguaglia o supera i
Transformer dedicati.

Il verdetto da solo sarebbe una classifica, e le classifiche invecchiano. Quello
che non invecchia sono le due prove con cui gli autori lo spiegano, ed è la
parte del lavoro che vale la pena portarsi via.

*La prima è il mescolamento dell'ingresso.* Se un modello usa davvero l'ordine
temporale, rimescolare a caso le posizioni della finestra passata deve
rovinarlo. Sui dataset provati, le prestazioni dei metodi basati su Transformer
**non si muovono** (il calo medio è dell'ordine di un decimo di punto
percentuale, cioè niente), mentre lo stesso trattamento fa perdere a DLinear fra
il 27% e l'81%. Il modello che sta usando il tempo è quello lineare.

*La seconda è la lunghezza della finestra passata.* Un modello che estrae
relazioni temporali da una storia lunga deve migliorare quando gliene si dà di
più. I Transformer, allungando la finestra, restano fermi o peggiorano; i
modelli lineari migliorano sempre. Messe insieme, le due prove dicono che su
quei banchi di prova l'attenzione non stava estraendo le relazioni temporali che
dichiarava di estrarre, il che è una critica al **metodo di valutazione** prima
che all'architettura.

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
sorprendente: trasforma i numeri in **parole**, riducendo ogni valore a un
simbolo di un vocabolario, e poi tratta la serie come una frase da completare
(riciclando di peso la macchina dei modelli linguistici). Promettente, ma è un
campo giovane: non batte sempre i metodi su misura, né i vecchi classici.

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

Traduciamo la TCN della {numref}`fig-tcn-convoluzioni-causali` in codice. Il cuore
è la convoluzione causale, e il modo di ottenerla è un piccolo trucco
dell'implementazione originale: si allunga la sequenza a sinistra con degli zeri
(è il `padding` di `nn.Conv1d`), tanti quanti ne servono perché lo strato non
resti più corto, e poi si **taglia** la coda a destra, così che l'uscita al tempo
$t$ non peschi mai nel futuro. Quanti zeri servono lo dicono i due numeri che
descrivono lo strato: quanti nodi guarda ($k$, l'ampiezza del filtro) e quanto
sono distanziati ($d$, il salto), e il conto viene $(k-1)\,d$.

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
    # con kernel=3 e n_blocchi=3 il campo recettivo è 1+(k-1)(2^L-1) = 15
    # istanti: passare finestre molto più lunghe di così è sprecato
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
PyTorch: si taglia la serie in coppie (finestra passata, valore successivo)
(rispettando la separazione temporale imposta dalla sezione sulla validazione,
mai mescolando futuro e passato) si passano le finestre al modello e si
minimizza l'errore quadratico con `nn.MSELoss` e un ottimizzatore come
`torch.optim.Adam`.

Basta poco per farne un modello **probabilistico**, nello spirito di DeepAR:
invece di far uscire dalla rete un numero solo se ne fanno uscire due, il valore
centrale e quanto ci si può discostare, e si cambia il bersaglio
dell'addestramento. Non più «avvicina la previsione al valore vero», ma «rendi
il valore vero il più probabile possibile secondo la forbice che stai
dichiarando»: in gergo, si sostituisce la MSE con la log-verosimiglianza
gaussiana cambiata di segno. La rete impara così anche a dire quando non sa.

Dalle reti ricorrenti ai foundation model, il filo che attraversa tutta la
sezione è duplice: le architetture cambiano, ma restano validi due
comandamenti del capitolo; prevedere significa **dichiarare l'incertezza**, e
nessun modello, per quanto profondo, è dispensato dal confronto con la **linea
di base** classica.

`````{tab} Elementare

```{admonition} Da ricordare
:class: important
- Il deep learning conviene quando hai **tante serie collegate** fra loro (la
  catena di diecimila negozi, non il negozio di quartiere), quando i legami non
  sono semplici proporzioni, quando ci sono cause esterne che aiutano, e quando
  ti serve non un numero ma una forbice. Su poche serie corte i **classici**
  spesso vincono ancora.
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
  ha battuti su molti banchi di prova {cite}`zeng2023transformers`, e mescolando
  l'ordine dei giorni si è scoperto che quei Transformer, il tempo, non lo
  stavano quasi usando. I **foundation model** come **Chronos**
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
  l'orizzonte finché il processo non è stazionario: su un processo stazionario
  converge alla varianza di lungo periodo e si ferma.
- **N-BEATS** {cite}`oreshkin2020nbeats` usa blocchi di soli **MLP**
  (percettroni multistrato: pile di strati densi) con **doppio residuo**
  backcast/forecast, ed è interpretabile quando la base è vincolata. Il suo
  contributo è di meccanismo: per battere gli ibridi statistico-neurali non
  serviva innestare statistica dentro la rete.
- Sui Transformer per le serie, cautela: **DLinear** li eguaglia o supera su
  nove dataset {cite}`zeng2023transformers`, e le due prove che lo spiegano
  (mescolare l'ingresso non li scalfisce, allungare la finestra non li migliora)
  dicono che su quei banchi non stavano usando l'ordine temporale. Il **TFT**
  {cite}`lim2021temporal` resta utile per covariate multiple e interpretabilità,
  purché i suoi pesi di attenzione si leggano come indizio e non come prova
  {cite}`jain2019attention`. I **foundation model** come **Chronos**
  {cite}`ansari2024chronos` promettono forecasting *zero-shot*: campo promettente
  ma giovane, non risolto.
```

`````
