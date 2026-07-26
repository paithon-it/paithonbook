# Forecasting neurale: da RNN ai Transformer e ai foundation model

Verso la fine degli anni Dieci, in Amazon si presentava un problema di scala
brutale: prevedere la domanda di *centinaia di milioni* di prodotti, ognuno con
la sua piccola storia di vendite — spesso corta, spesso a scatti, a volte fatta
di zeri interrotti da un picco. La ricetta classica direbbe: un modello ARIMA
per prodotto. Ma stimarne cento milioni è impraticabile, e la maggior parte di
quelle serie è troppo breve e rumorosa perché un modello su misura ci capisca
qualcosa. Nel 2020 un gruppo di ricercatori propose un'inversione di
prospettiva {cite}`salinas2020deepar`: e se, invece di un modello per serie,
addestrassimo *una sola rete su tutte le serie insieme*, lasciando che ciascuna
impari dai pattern delle altre? È questa idea — il modello **globale** — a
segnare il passaggio dal forecasting statistico a quello neurale.

Le sezioni precedenti hanno costruito la cassetta degli attrezzi classica —
ARIMA, Holt-Winters — e hanno insistito su un punto scomodo: quei metodi di
mezzo secolo fa restano una **linea di base durissima da battere**. Questa
sezione racconta l'altra famiglia, quella delle reti neurali: quando conviene
davvero, come si è evoluta dalle reti ricorrenti alle convoluzioni causali fino
ai Transformer, e dove sta arrivando oggi con i *foundation model*. Con la
stessa onestà: il deep learning non è un miglioramento automatico.

## Quando conviene il deep learning

`````{tab} Elementare

Immagina un unico negozio di quartiere. La sua storia di vendite è corta e
ballerina: un modello troppo sofisticato ci si perde, e un buon vecchio ARIMA
fa altrettanto bene con molta meno fatica. Ora immagina una **catena** con
diecimila negozi. Molti pattern sono condivisi — il picco del sabato, il crollo
di Ferragosto, l'effetto di una promozione. Una rete neurale può impararli *una
volta sola* guardando tutti i negozi insieme, e poi applicarli anche al negozio
aperto il mese scorso, che da solo non avrebbe abbastanza storia.

Il deep learning conviene quando ricorrono quattro condizioni: hai **tante
serie collegate** fra loro; le relazioni sono **non lineari** (una promozione
raddoppia le vendite solo sotto una certa soglia di prezzo); ci sono **fattori
esterni** che aiutano a prevedere (meteo, festività, prezzo); e vuoi non un
singolo numero ma una **stima dell'incertezza**. Se invece hai una sola serie,
pulita e lunga, i classici restano spesso la scelta migliore — e comunque la
prima da provare.

`````

`````{tab} Superiore

Il vantaggio strutturale è il **modello globale**: un'unica funzione
parametrica $f_\theta$ addestrata sull'intero insieme di $N$ serie
$\{x^{(i)}_{1:T_i}\}_{i=1}^N$, in luogo di $N$ modelli locali indipendenti. Con
$N$ grande, $\theta$ vede molti più esempi e può permettersi capacità che una
singola serie non giustificherebbe, catturando pattern condivisi (stagionalità
tipiche, effetti di calendario) e regolarizzando le serie corte con quelle
lunghe. Le altre tre leve sono la **non linearità** (mappe che un ARMA lineare
non rappresenta), le **covariate esogene** $z_t$ integrate nell'input
— $\hat{x}_{t+1} = f_\theta(x_{1:t}, z_{1:t+1})$ — e l'uscita **probabilistica**
$p_\theta(x_{t+1:t+h}\mid x_{1:t}, z)$, non una stima puntuale.

L'onestà impone il rovescio della medaglia. Con **poche** serie o serie
**corte** il regime dati non regge la varianza di un modello ad alta capacità,
e i metodi statistici — robusti, frugali, interpretabili — tengono il campo,
come mostrano ripetutamente le competizioni M discusse nell'introduzione al
capitolo. La regola resta quella: si batte prima la linea di base classica, poi
si tira in ballo la rete.

`````

## Reti ricorrenti per il forecasting

Il primo strumento neurale per le sequenze lo abbiamo già costruito, nel
capitolo sul *Natural Language Processing*: le **reti ricorrenti** (RNN) e la
loro versione con i cancelli, la **LSTM** di Sepp Hochreiter e Jürgen
Schmidhuber {cite}`hochreiter1997long`. Lì il problema era il linguaggio; qui è
una serie di numeri, ma il meccanismo è identico — una cella che scorre nel
tempo mantenendo uno **stato nascosto**, la sua memoria di ciò che ha letto
finora. Una serie temporale, in fondo, è una frase di numeri.

Con una RNN il forecasting prende due forme. Nel **seq-to-one** la rete legge la
finestra passata $x_{t-w+1}, \dots, x_t$ e produce un solo valore, la previsione
del prossimo passo: si allena come una regressione, con la finestra come input e
$x_{t+1}$ come bersaglio. Per prevedere più in là si riapplica la rete in modo
**ricorsivo**, reiniettando le proprie stime — con l'accumulo dell'errore già
visto nell'introduzione. Nel **seq-to-seq**, invece, un *encoder* ricorrente
comprime tutta la storia in un vettore e un *decoder* srotola l'intero orizzonte
futuro in un colpo, esattamente come una traduzione genera l'intera frase
d'uscita. È lo schema che, nell'NLP, ha fatto nascere il meccanismo di
attenzione.

I limiti, però, sono gli stessi del capitolo NLP, e nel forecasting pesano
persino di più. La memoria delle RNN semplici si **dissolve** su orizzonti
lunghi — il gradiente che svanisce — e le LSTM lo mitigano ma non lo
cancellano; l'addestramento è **sequenziale**, mal parallelizzabile sulle GPU
perché il passo $t$ aspetta il $t-1$. Da qui la ricerca di architetture che
guardino lontano nel tempo *senza* pagare la ricorrenza. La prima risposta
arriva, curiosamente, dalle convoluzioni.

## TCN: convoluzioni che guardano solo indietro

Nel 2018 Shaojie Bai, Zico Kolter e Vladlen Koltun pubblicarono un confronto
sistematico tra reti ricorrenti e reti convoluzionali sulle sequenze, e la
conclusione fece rumore: su un ampio ventaglio di compiti una semplice rete
convoluzionale, opportunamente adattata, eguagliava o superava le LSTM
{cite}`bai2018empirical`. La chiamarono **Temporal Convolutional Network**
(TCN). Due accorgimenti la rendono adatta al tempo, e la
{numref}`fig-tcn-convoluzioni-causali` li mostra insieme.

```{figure} ../figures/tcn-convoluzioni-causali.svg
:name: fig-tcn-convoluzioni-causali
:alt: Una rete convoluzionale temporale con convoluzioni causali dilatate. In basso otto nodi di ingresso lungo l'asse del tempo; sopra tre strati di nodi collegati da archi che saltano sempre all'indietro con dilatazione 1, 2 e 4. Nessun arco punta al futuro. Un cono ombreggiato evidenzia il campo recettivo, pari a otto istanti. In alto a destra il nodo di uscita in terracotta.
:width: 100%

Una TCN con convoluzioni **causali** (nessun arco viene dal futuro) e
**dilatate** (i salti raddoppiano a ogni strato: 1, 2, 4). Con kernel 2 e tre
strati, un singolo nodo di uscita raccoglie informazione da $2^3 = 8$ istanti
passati: il **campo recettivo** cresce in modo esponenziale con la profondità.
```

Il primo accorgimento è la **causalità**: l'uscita all'istante $t$ può dipendere
solo da $t$ e dagli istanti precedenti, mai da quelli futuri — altrimenti la rete
«bara», guardando la risposta. Il secondo è la **dilatazione**: per abbracciare
un passato lungo senza impilare centinaia di strati, ogni strato salta indietro
sempre più lontano.

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

cioè cresce **esponenzialmente** con la profondità $L$: con $k=2$ e $L=3$ si ha
$r = 8$, come in figura; con $k=3$ e $L=6$ si arriva a $r = 127$ istanti. In
pratica ogni blocco aggiunge una **connessione residua** — nello spirito delle
ResNet — per addestrare pile profonde senza che il gradiente svanisca. Rispetto
a una RNN, il calcolo è interamente parallelizzabile lungo il tempo: $O(1)$ passi
sequenziali invece di $O(n)$.

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
pioggia». DeepAR fa lo stesso con le vendite. A ogni passo, invece di sputare una
cifra, descrive un **ventaglio di futuri plausibili**: il valore più probabile e
quanto ci si può discostare. Per prevedere una settimana intera, la rete «tira i
dadi» tante volte — genera centinaia di storie possibili, ognuna coerente con la
precedente — e poi legge il ventaglio: la mediana è la previsione, l'ampiezza tra
il decimo e il novantesimo percentile è la **banda di incertezza**. È il filo
rosso del capitolo: una previsione seria è un numero *con la sua incertezza*.

`````

`````{tab} Superiore

A ogni passo la rete emette i **parametri** $\theta_t$ di una distribuzione di
verosimiglianza $p(x_t \mid \theta_t)$: una gaussiana $\theta_t=(\mu_t,\sigma_t)$
per dati reali, una **binomiale negativa** per conteggi non negativi (come le
vendite). L'addestramento massimizza la log-verosimiglianza su tutte le serie,

$$
\mathcal{L}(\Phi) = \sum_{i=1}^{N} \sum_{t} \log p\big(x^{(i)}_t \mid
\theta_t\big), \qquad \theta_t = g_\Phi\big(h^{(i)}_t\big),
$$

dove $h^{(i)}_t$ è lo stato nascosto della LSTM per la serie $i$ e $\Phi$ i
parametri *condivisi* fra tutte le serie. La previsione multi-passo avviene per
**campionamento ancestrale**: si estrae $\hat{x}_{t+1}\sim p(\cdot\mid
\theta_{t+1})$, lo si reinietta come input, si ripete fino all'orizzonte; molte
traiettorie così ottenute forniscono, per ogni passo, i **quantili** della
previsione. Nessuna formula chiusa per gli intervalli: è Monte Carlo.

`````

Il campionamento ancestrale è così centrale che conviene vederlo girare, su un
modellino giocattolo in puro NumPy. La rete vera predirebbe $\mu_t$ e $\sigma_t$
a ogni passo; qui li fissiamo con una semplice regola autoregressiva, e ci
concentriamo sul meccanismo: campiona, reinietta, ripeti.

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

Eseguendolo si vede la mediana convergere verso la media di lungo periodo
($\mu = 4/(1-0{,}6) = 10$) e, soprattutto, la **banda allargarsi** passo dopo
passo: l'incertezza si accumula con l'orizzonte, proprio come anticipato
nell'introduzione. Questo è ciò che una previsione puntuale nasconde e una
probabilistica dichiara.

## N-BEATS: solo percettroni, ma interpretabili

Nel 2020 Boris Oreshkin e colleghi mostrarono che per battere i metodi statistici
non servivano né ricorrenza né convoluzioni: bastavano **percettroni**, impilati
con la giusta architettura {cite}`oreshkin2020nbeats`. **N-BEATS** (*Neural Basis
Expansion Analysis for Time Series*) è fatto di blocchi di soli strati densi, e
la sua eleganza sta in un'idea di contabilità: il **doppio residuo**.

`````{tab} Elementare

Immagina di spiegare una serie a strati, come si pela una cipolla. Il primo
blocco guarda la finestra passata e produce due cose: una **ricostruzione** di
ciò che ha capito del passato (il *backcast*) e un pezzo di **previsione** del
futuro (il *forecast*). A questo punto si *sottrae* la ricostruzione dal passato:
ciò che resta è quello che il primo blocco non ha saputo spiegare, e passa al
secondo blocco, che ripete il gioco su quel residuo. Blocco dopo blocco, ogni
strato spiega un pezzo in più; le previsioni parziali si **sommano** a formare
quella finale. Il bello: si può dedicare uno stack a catturare il *trend* e uno
la *stagionalità*, e allora la rete non solo prevede, ma **mostra** quanto della
previsione è tendenza e quanto è ciclo — cosa rara per una rete neurale.

`````

`````{tab} Superiore

Ogni blocco $b$ riceve un residuo $x^{(b)}$ e, tramite una pila di strati densi
seguita da una proiezione su una base, produce due uscite: un **backcast**
$\hat{x}^{(b)}$ (la parte d'ingresso che sa spiegare) e un **forecast**
$\hat{y}^{(b)}$ (il suo contributo alla previsione). Il **doppio residuo** li
combina così:

$$
x^{(b+1)} = x^{(b)} - \hat{x}^{(b)}, \qquad
\hat{y} = \sum_{b} \hat{y}^{(b)},
$$

dove l'ingresso del blocco successivo è ciò che il blocco corrente **non** ha
saputo spiegare, e la previsione finale è la **somma** dei contributi. Nella
variante *interpretabile* la base è vincolata — polinomi di grado basso per il
trend, termini di Fourier per la stagionalità — così che i due stack
restituiscano componenti leggibili; in quella *generica* la base è appresa
liberamente. Nessuna componente statistica innestata: solo strati densi.

`````

La variante interpretabile riallaccia il forecasting neurale alla
**decomposizione classica** della prima sezione — trend e stagionalità come
uscite separate — chiudendo il cerchio con i modelli statistici. E i numeri
contano: nelle competizioni M, N-BEATS migliora di circa il **3%** rispetto al
vincitore della M4 {cite}`oreshkin2020nbeats`, la prima volta che un modello di
*puro* deep learning, senza ibridazioni statistiche, supera un ibrido su quel
banco di prova.

## Transformer per le serie, e un lineare che li imbarazza

Se l'attenzione ha spazzato via la ricorrenza nell'NLP — è la tesi di *«Attention
Is All You Need»* {cite}`vaswani2017attention`, cui il libro dedica un intero
capitolo — la tentazione di portarla nelle serie temporali era irresistibile. Un
Transformer, in teoria, collega due istanti lontani con *un solo salto* di
attenzione, aggirando la memoria che si dissolve delle RNN. Fioccarono così
architetture dedicate: **Informer**, con un'attenzione sparsa per abbattere il
costo sulle sequenze lunghe, e **Autoformer**, che innesta la decomposizione in
trend e stagionalità dentro il blocco di attenzione.

Poi, nel 2023, una doccia fredda.

`````{tab} Elementare

Un gruppo di ricercatori pose una domanda scomoda già nel titolo: «I Transformer
servono davvero, per prevedere le serie temporali?». La risposta, sui banchi di
prova più usati, fu spiazzante: un modello **lineare** semplicissimo — poco più
di una retta tirata sui dati, dopo averli separati in trend e stagionalità —
batteva quei Transformer sofisticati. La morale non è «i Transformer non
servono», ma qualcosa di più prezioso: la complessità non è mai un vantaggio
gratuito. Prima di celebrare un modello elaborato, va confrontato con la linea di
base più stupida che ti viene in mente. A volte la retta vince.

`````

`````{tab} Superiore

Il costo quadratico $O(n^2)$ dell'attenzione piena sulle sequenze lunghe aveva
motivato le varianti efficienti (Informer, Autoformer, FEDformer). Zeng e colleghi
{cite}`zeng2023transformers` proposero come confronto **DLinear**: si decompone la
serie in trend e stagionalità e si applica a ciascuna componente una singola
mappa lineare $\hat{x}_{t+1:t+h} = W\,x_{t-w+1:t}$. Su nove dataset di
forecasting a lungo orizzonte, DLinear eguaglia o supera i Transformer dedicati.
Le spiegazioni proposte: l'attenzione è *invariante alla permutazione* e rischia
di disperdere l'ordine temporale, e i benchmark premiano soprattutto la cattura
di trend e stagionalità, che una mappa lineare già coglie. Non è la condanna dei
Transformer — il **Temporal Fusion Transformer** di Lim e colleghi
{cite}`lim2021temporal` resta forte quando servono **covariate multiple** (statiche,
note nel futuro, osservate nel passato) e **interpretabilità**, grazie a reti di
selezione delle variabili e pesi di attenzione ispezionabili — ma un promemoria
di metodo: sempre una linea di base, sempre onesta.

`````

## Foundation model: la «GPT delle serie»

L'ultima frontiera prende in prestito l'idea che ha reso possibili i grandi
modelli linguistici: pre-addestrare **un** modello enorme su una collezione
sterminata di serie, e poi usarlo per prevedere serie **mai viste**, senza
riaddestrarlo. È il forecasting *zero-shot*, e il nome che gira è, appunto, «la
GPT delle serie temporali».

`````{tab} Elementare

Come un modello linguistico impara la lingua leggendo miliardi di frasi e poi sa
scrivere su argomenti nuovi, un *foundation model* per le serie impara la
«grammatica» dei fenomeni temporali — tendenze, cicli, picchi — leggendo milioni
di serie di ogni tipo. Poi gli dai una serie che non ha mai incontrato, per
esempio le vendite del tuo negozio, e lui prevede il seguito **senza addestrarsi
da capo**. Il modello Chronos, di Amazon, usa un trucco sorprendente: trasforma i
numeri in **parole**, riducendo ogni valore a un simbolo di un vocabolario, e poi
tratta la serie come una frase da completare — riciclando di peso la macchina dei
modelli linguistici. Promettente, ma è un campo giovane: non batte sempre i
metodi su misura, né i vecchi classici.

`````

`````{tab} Superiore

**Chronos** {cite}`ansari2024chronos` scala e **quantizza** i valori reali in un
vocabolario finito di *token*, poi addestra un modello linguistico della famiglia
T5 con la consueta *cross-entropy* sul token successivo, su un grande corpus di
serie reali e sintetiche. In inferenza campiona traiettorie di token e le
riconverte in valori, ottenendo una previsione **probabilistica** senza
architetture ad hoc. Un'alternativa è **TimesFM** di Google, *decoder-only* che
lavora su *patch* di istanti, pre-addestrato su ordini di grandezza di miliardi
di punti temporali. Entrambi mostrano uno *zero-shot* competitivo con modelli
allenati sul singolo compito — un risultato notevole. La cautela, però, è
d'obbligo: il campo ha pochi anni, la valutazione soffre di possibili
**contaminazioni** tra corpus di pre-addestramento e benchmark, e su serie corte
o molto specifiche i metodi classici restano spesso preferibili. Promettente, non
risolto.

`````

## In pratica: una TCN in PyTorch

Traduciamo la TCN della {numref}`fig-tcn-convoluzioni-causali` in codice. Il cuore
è la convoluzione causale: si usa `nn.Conv1d` con un `padding` a sinistra pari a
$(k-1)\,d$, e poi si **taglia** la coda a destra, così che l'uscita al tempo $t$
non peschi mai nel futuro. È il *trick* dell'implementazione originale.

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
PyTorch: si taglia la serie in coppie (finestra passata, valore successivo) —
rispettando la separazione temporale imposta dalla sezione sulla validazione,
mai mescolando futuro e passato — si passano le finestre al modello e si
minimizza l'errore quadratico con `nn.MSELoss` e un ottimizzatore come
`torch.optim.Adam`. Sostituire la testa lineare con due uscite $(\mu, \log\sigma)$
e la MSE con la log-verosimiglianza gaussiana basta a trasformare questa TCN in
un forecaster **probabilistico**, nello spirito di DeepAR.

Dalle reti ricorrenti ai foundation model, il filo che attraversa tutta la
sezione è duplice: le architetture cambiano, ma restano validi due comandamenti
del capitolo — prevedere significa **dichiarare l'incertezza**, e nessun modello,
per quanto profondo, è dispensato dal confronto con la **linea di base** classica.

```{admonition} Da ricordare
:class: important
- Il deep learning conviene con **molte serie collegate**, relazioni **non
  lineari**, **covariate esterne** e quando serve una **previsione
  probabilistica**; su poche serie corte i **classici** spesso vincono ancora.
- Le **RNN/LSTM** {cite}`hochreiter1997long` fanno forecasting *seq-to-one* o
  *seq-to-seq*, ma soffrono orizzonti lunghi e addestramento sequenziale. Le
  **TCN** {cite}`bai2018empirical` usano convoluzioni **causali dilatate**: campo
  recettivo esponenziale $1+(k-1)(2^L-1)$, calcolo parallelizzabile.
- **DeepAR** {cite}`salinas2020deepar` è una RNN autoregressiva **globale** (una
  rete per molte serie) che emette una **distribuzione**; la previsione multi-passo
  è per campionamento ancestrale, con la banda che si allarga con l'orizzonte.
- **N-BEATS** {cite}`oreshkin2020nbeats` usa blocchi di soli MLP con **doppio
  residuo** backcast/forecast, è interpretabile e batte il vincitore della M4.
- Sui Transformer per le serie, cautela: un semplice **lineare (DLinear)** li
  supera su molti benchmark {cite}`zeng2023transformers`; il **TFT**
  {cite}`lim2021temporal` resta utile per covariate multiple e interpretabilità.
  I **foundation model** come **Chronos** {cite}`ansari2024chronos` promettono
  forecasting *zero-shot*: campo promettente ma giovane, non risolto.
```
