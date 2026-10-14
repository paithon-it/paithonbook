# Forecasting neurale: da RNN ai Transformer e ai foundation model

Nella seconda metà degli anni Dieci, in Amazon si presentava un problema di scala
brutale: prevedere la domanda di *milioni* di prodotti, ognuno
con la sua piccola storia di vendite (spesso corta, spesso a scatti, a volte
fatta di zeri interrotti da un picco). La ricetta classica direbbe: un modello
ARIMA per prodotto. Ma stimarne milioni è impraticabile, e la maggior
parte di quelle serie è troppo breve e rumorosa perché un modello su misura ci
capisca qualcosa. Nel 2017 un gruppo di ricercatori dell'azienda mise al centro
un'altra prospettiva {cite}`salinas2020deepar`: e se, invece di un modello per
serie, addestrassimo *una sola rete su tutte le serie insieme*, lasciando che
ciascuna impari dai pattern delle altre? L'idea di condividere informazione fra
le serie non nasce lì, e gli autori citano chi li aveva preceduti; ma lì prende
la forma che si sarebbe imposta, il modello **globale**, e da lì passa il
forecasting neurale. (L'articolo circolò subito online e fu
stampato su una rivista scientifica solo tre anni dopo, nel 2020.)

Le sezioni sui {doc}`modelli classici </SerieTemporali/componenti-e-classici>`
e sulla {doc}`validazione </SerieTemporali/validazione-e-feature>` hanno
costruito la cassetta degli attrezzi classica (ARIMA, Holt-Winters) e hanno
insistito su un punto scomodo: quei metodi di mezzo secolo fa restano una linea
di base durissima da battere. Tocca ora alle reti neurali: quando
conviene, come si sono evolute dalle reti ricorrenti alle convoluzioni causali
fino ai Transformer, e dove stanno arrivando oggi con i *foundation model*. Il
criterio resta lo stesso: il deep learning non è un miglioramento automatico.

## Quando conviene il deep learning

Che una rete sappia imparare una serie temporale è fuori discussione; resta da
stabilire in quali casi convenga, e la risposta dipende meno dall'architettura
che dalla forma dei dati. Le famiglie in gioco sono tre: accanto ai modelli
classici e alle reti ci sono i modelli additivi come Prophet, che disegnano la
serie sul calendario.

`````{tab} Elementare

Un unico negozio di quartiere ha una storia di vendite corta e ballerina, e un
modello troppo sofisticato ci si perde: con tanti numeri da regolare e pochi
giorni su cui regolarli, finisce per imparare a memoria anche gli sbalzi
capitati per caso, che domani non si ripeteranno. Un buon vecchio ARIMA fa
altrettanto bene con molta meno fatica. In una catena con diecimila negozi,
invece, molti pattern sono condivisi: il picco del sabato, il crollo di
Ferragosto, l'effetto di una promozione. Una rete neurale può impararli *una
volta sola* guardando tutti i negozi insieme, e poi applicarli anche al negozio
aperto il mese scorso, che da solo non avrebbe abbastanza storia.

Il deep learning conviene quando ricorrono alcune di queste quattro condizioni,
e la catena le ha tutte: **tante serie collegate** fra loro; relazioni **non
lineari** (una promozione raddoppia le vendite solo sotto una certa soglia di
prezzo); **fattori esterni** che aiutano a prevedere (meteo, festività,
prezzo); e il bisogno di una **stima dell'incertezza** invece di un numero
solo, perché chi ordina la merce deve sapere di quanto rischia di sbagliare. Se
invece hai una sola serie, pulita e lunga, i classici restano spesso la scelta
migliore, e comunque la prima da provare.

Poi c'è la terza famiglia. I metodi classici e le reti provano a
indovinare, ciascuno a modo suo, la regola con cui un giorno genera il
successivo; la terza sta in mezzo, e su moltissimi problemi aziendali basta.
Prende il calendario e ci disegna sopra una curva, somma di pezzi che si
guardano uno per uno. Il primo è una tendenza di fondo a linea spezzata (una
retta che ogni tanto cambia pendenza, come una strada che sale e a un certo
punto sale meno; dove cambiarla lo decide guardando i dati). Il secondo è una o
più stagionalità, disegnate con le poche onde regolari (seno e coseno) che la
sezione sulle feature ha introdotto; possono essercene due sovrapposte, la
settimana lavorativa e il ciclo annuale. Il terzo sono gli effetti delle
festività, strappi su date dichiarate a mano.

Il programma più usato di questa famiglia si chiama **Prophet** e lo hanno
pubblicato nel 2018 due ricercatori di Facebook, Sean Taylor e Benjamin Letham
{cite}`taylor2018forecasting`. Ha avuto fortuna perché si costruisce sui dati in
un attimo e ogni pezzo si può mostrare a chi non lavora nel settore.

Rispetto alle altre due famiglie c'è una differenza di fondo, da cui vengono
insieme il pregio e il limite: il valore di domani non dipende da quello di
oggi, ma solo dalla data. Il modello è una curva tirata sui punti, non una
catena di giorni che si tengono per mano. I buchi nei dati non lo rompono,
perché non c'è nessuna catena da interrompere; per la stessa ragione, su una
serie in cui oggi somiglia molto a ieri butta via l'informazione più preziosa,
quella che un ARIMA invece usa. I valori assurdi, al contrario, lo disturbano:
li scambia per cambi di tendenza, e chi lo ha scritto consiglia di toglierli e
lasciare il buco.

`````

`````{tab} Superiore

Il vantaggio strutturale è il modello globale: un'unica funzione
parametrica $f_\theta$ addestrata sull'intero insieme di $N$ serie
$\{\mathbf{x}^{(i)}_{1:T_i}\}_{i=1}^N$, in luogo di $N$ modelli locali
indipendenti.
Con $N$ grande, $\theta$ vede molti più esempi e può permettersi capacità che
una singola serie non giustificherebbe, catturando pattern condivisi
(stagionalità tipiche, effetti di calendario) e regolarizzando le serie corte
con quelle lunghe. Le altre tre leve sono la non linearità (mappe che un
ARMA lineare non rappresenta), le covariate esogene $\mathbf{z}_t$
integrate nell'input,
$\hat{x}_{t+1} = f_\theta(\mathbf{x}_{1:t}, \mathbf{z}_{1:t+1})$, e l'uscita
probabilistica
$p_\theta(\mathbf{x}_{t+1:t+h}\mid \mathbf{x}_{1:t}, \mathbf{z})$, non una
stima puntuale. Un avviso sulle lettere, per chi andrà a leggere l'articolo di
DeepAR: là $z$ è la serie da prevedere e le covariate si chiamano $x$, il
contrario di qui.

Il modello globale ha però un costo strutturale. Con poche serie, o serie
corte, i dati non bastano a controllare la varianza di un modello ad alta
capacità, e i metodi statistici (robusti, frugali, interpretabili) tengono il
campo, come mostrano le competizioni M raccontate in {doc}`Il futuro nei numeri
</SerieTemporali/overview>`. Il modello globale non è una prerogativa delle
reti: lo è altrettanto un albero potenziato su feature ritardate e covariate,
il tipo di modello che ha vinto la M5. Una rete conviene rispetto all'albero
quando servono un'uscita probabilistica per costruzione, sequenze lunghe con
una struttura da imparare, o molti ingressi di tipo diverso; altrimenti
l'albero globale è la linea di base da battere. La regola resta quella: si
batte prima la linea di base, poi si passa alla rete.

Fra le due famiglie ne vive una terza, che non è né un processo stocastico né
una rete, e che su moltissimi problemi aziendali basta: i **modelli additivi
decomponibili**, di cui Prophet {cite}`taylor2018forecasting` è l'esemplare
diffuso. La serie si scrive come

$$
y(t) = g(t) + s(t) + h(t) + \varepsilon_t ,
$$

con $g$ una tendenza lineare a tratti (è quella di default; esiste anche una
crescita che satura verso un tetto dichiarato). La lettera $g$ è quella
dell'articolo, e non ha niente a che vedere con il $g_\theta$ di DeepAR. I punti di
svolta sono scelti dai dati fra molti candidati fissati in anticipo,
venticinque per default e sparsi nel primo 80% della storia, con un prior di
Laplace sulle variazioni di pendenza, cioè una regolarizzazione L1 che ne
spegne la maggior parte; la sua scala (`changepoint_prior_scale`, $0{,}05$ per
default) regola il compromesso fra una tendenza rigida e una che insegue il
rumore. Il resto della formula: $s$ una somma di serie di Fourier troncate a
$K$ armoniche (le stesse colonne $\sin(2\pi kt/m)$ e $\cos(2\pi kt/m)$ della
sezione sulle feature, il che permette di sovrapporre più periodi e di regolare
la flessibilità scegliendo $K$); $h$ gli effetti puntuali su date dichiarate (è
la notazione dell'articolo: $h(t)$ è una funzione del tempo, non l'orizzonte di
previsione che $h$ indica altrove); e $\varepsilon_t$ il residuo. Il modello è
scritto in forma bayesiana, ma di norma la stima si ferma al massimo a
posteriori invece di campionare l'intera distribuzione. Anche gli intervalli
non vengono da una formula chiusa: si ottengono simulando in avanti cambi di
pendenza con la stessa frequenza e la stessa ampiezza di quelli visti nel
passato, che è un'ipotesi forte, dichiarata come tale dagli autori.

A distinguerla dalle altre due, più della formula, è che $y$ è una regressione
sul tempo, non sui valori passati: non c'è nessuno stato, nessuna dipendenza
fra $y(t)$ e $y(t-1)$. È una curva interpolata e non un processo stocastico. Da
qui vengono insieme il pregio (i dati mancanti non rompono niente, perché non
c'è nessuna ricorsione da interrompere, e la stima è veloce) e il limite: il
modello non sfrutta l'autocorrelazione a breve, da cui dipende la prevedibilità
di una serie molto correlata; su quelle serie un modello che la sfrutta, come
un ARIMA, parte avvantaggiato per costruzione. Hyndman e Athanasopoulos, che lo
mettono alla prova su due esempi, annotano che raramente dà previsioni più
accurate delle alternative {cite}`hyndman2021forecasting`. Gli *outlier* hanno
un effetto: la documentazione della libreria spiega che Prophet li assorbe con
cambi di tendenza, che gonfiano gli intervalli futuri, che un gruppo di valori
estremi può deformare per sempre la stima della stagionalità, e che il rimedio
è toglierli e lasciare il buco.

`````

## Reti ricorrenti per il forecasting

Il primo strumento neurale per le sequenze è quello descritto nella
{doc}`sezione sui modelli di sequenza
</NaturalLanguageProcessing/modelli-sequenza>`: una rete che legge un pezzo per
volta e si porta dietro un riassunto di quello che ha letto fin lì. Quel
riassunto, lo stato nascosto, è tutta la memoria della rete. Le reti ricorrenti
(RNN) funzionano così. La LSTM di Sepp Hochreiter e Jürgen Schmidhuber
{cite}`hochreiter1997long`, completata da Gers e colleghi con il cancello di
dimenticanza {cite}`gers2000learning`, decide a ogni passo che cosa di quel
riassunto tenere e che cosa buttare: sono i suoi cancelli. Lì il problema era il
linguaggio, qui è una serie di numeri: la ricorrenza è la stessa, mentre
cambiano l'ingresso (un numero reale a
ogni passo invece di un token) e l'uscita (un valore, oppure i parametri di una
distribuzione).

`````{tab} Elementare

Con una rete così il forecasting si fa in due modi.

Il primo: le dai in pasto gli ultimi giorni, mettiamo gli ultimi trenta, e le
chiedi un numero solo, quello di domani. Si addestra come qualunque modello che
deve indovinare un numero: in ingresso la finestra di giorni, come risposta
giusta il giorno dopo. Per andare più in là di domani si riapplica la rete a
catena, rimettendole dentro le sue stesse previsioni: è la strategia
ricorsiva della {doc}`sezione sulle previsioni a più passi
</SerieTemporali/validazione-e-feature>`, con il suo errore che si trascina.

Il secondo: una prima rete legge tutta la storia e ne fa un riassunto, una
seconda srotola da quel riassunto l'intera settimana futura in un colpo solo. È
lo stesso schema con cui si traduce una frase, e da lì è nata, nel trattamento
del linguaggio, l'idea di lasciare che il modello decida da sé a quali pezzi del
passato dare peso.

Ci sono però due guai, e nel forecasting pesano più che altrove. Il primo è che
la memoria di queste reti si consuma: più il passato è lontano, meno ne
resta nel riassunto, e le LSTM migliorano le cose senza risolverle. Il secondo è
che queste reti si addestrano un passo dopo l'altro, perché il giorno $t$ non si
può calcolare finché non è finito il $t-1$: il lavoro non si può spartire fra le
migliaia di unità di calcolo di una GPU, e l'addestramento è lento.

`````

`````{tab} Superiore

Con una RNN il forecasting prende due forme. Nel **seq-to-one** («da sequenza a
uno») la rete legge la finestra dei giorni passati, cioè gli ultimi $w$ valori
$x_{t-w+1}, \dots, x_t$, dove $w$ è quanto la si vuole lunga, e produce un solo
valore, la previsione del prossimo passo: si allena come una regressione, con
la finestra come input e $x_{t+1}$ come bersaglio. Per prevedere più in là si
riapplica la rete in modo ricorsivo, reiniettando le proprie stime, con
l'accumulo dell'errore descritto per la strategia ricorsiva nella {doc}`sezione
sulle previsioni a più passi </SerieTemporali/validazione-e-feature>`. Nel
**seq-to-seq**, invece, una prima rete (l’*encoder*) riassume tutta la storia
in una manciata di numeri, e una seconda (il *decoder*) srotola da quel
riassunto l'intero orizzonte futuro in un colpo, come una traduzione genera
l'intera frase d'uscita. È lo schema che, nell'NLP, ha fatto nascere il
meccanismo di attenzione.

I limiti, però, sono quelli della {doc}`sezione sui modelli di sequenza
</NaturalLanguageProcessing/modelli-sequenza>`, e nel forecasting pesano
persino di più. La memoria delle RNN semplici si dissolve su orizzonti
lunghi (il gradiente che svanisce) e le LSTM lo mitigano ma non lo cancellano;
l'addestramento è sequenziale, mal parallelizzabile sulle GPU perché il
passo $t$ aspetta il $t-1$.

`````

Da qui la ricerca di architetture che guardino lontano nel tempo *senza*
ricorrenza. La prima risposta viene dalle convoluzioni.

## TCN: convoluzioni che guardano solo indietro

Una convoluzione è l'operazione della {doc}`sezione sulle reti
convoluzionali </DeepLearning/reti-convoluzionali>`: una piccola finestra che
scorre sui dati e a ogni posizione fa sempre lo stesso conto, prendere i valori
che ha sotto, pesarli e sommarli. Là scorreva su un'immagine, qui scorre su una
fila di giorni. Il vantaggio, rispetto a una rete che legge un giorno per
volta, è che tutte le posizioni si possono calcolare insieme, perché
nessuna aspetta il risultato di un'altra.

Nel 2018 Shaojie Bai, Zico Kolter e Vladlen Koltun pubblicarono un confronto
sistematico tra reti ricorrenti e reti convoluzionali sulle sequenze, con una
conclusione che sorprese: su un ampio ventaglio di compiti una semplice rete
convoluzionale, opportunamente adattata, eguagliava o superava le LSTM
{cite}`bai2018empirical`. I compiti erano problemi sintetici di memoria, cifre
scritte a mano lette un pixel alla volta, musica polifonica e modelli di
linguaggio: nessuno era la previsione di una serie reale, e il risultato va
portato qui con questa avvertenza. La chiamarono, riprendendo un nome che altri
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
:alt: Schema di una rete convoluzionale temporale con convoluzioni causali dilatate. In basso otto nodi di ingresso disposti lungo l'asse del tempo, dall'istante t meno sette all'istante t. Sopra, tre strati di nodi collegati da archi che saltano sempre all'indietro, verso il passato, con dilatazione crescente uno, due e quattro. Nessun arco punta verso il futuro. Il singolo nodo di uscita in alto a destra, evidenziato in terracotta, riceve informazione da tutti e otto gli ingressi: il campo recettivo, mostrato come cono ombreggiato, è pari a due elevato alla tre uguale otto istanti.
:width: 100%

Una TCN con convoluzioni causali (nessun arco viene dal futuro) e
dilatate (i salti raddoppiano a ogni strato: 1, 2, 4). Ogni nodo guarda due
soli nodi dello strato sotto, ma bastano tre strati perché il nodo in cima ne
raccolga $2^3 = 8$: quanto passato arriva a un singolo nodo di uscita si chiama
campo recettivo, e cresce in modo esponenziale con la profondità.
```

`````{tab} Elementare

La storia della serie è un diario, una pagina per giorno. Dieci
strati bastano per tenere d'occhio più di mille pagine, e nessuno di essi
sbircia una pagina non ancora scritta. Guardare soltanto
all'indietro è la causalità. Coprire tanti giorni con pochi strati, invece, è
questione di salti che si allargano. Diamo a ogni strato una finestra da due
giorni, che è la scelta più piccola possibile: il primo guarda ieri e oggi, il
secondo salta di due giorni, il terzo di quattro, il quarto di otto. I salti
sommati fanno $1 + 2 + 4 + 8 = 15$ giorni, più oggi fa sedici; con dieci
strati si passano i mille. È la dilatazione, e con finestre più larghe di due
la copertura cresce ancora. Il sedicesimo giorno indietro, però, con quattro
strati non c'è: non sbiadito, assente del tutto, e nessuna finestra di ingresso
più lunga glielo fa comparire. Per guardare più indietro si aggiunge uno
strato, o si allarga la finestra di ciascuno.

Impilare dieci strati, però, crea un problema che il calcolo dei giorni non
mostra, e la rete lo risolve con un accorgimento di costruzione: uno strato non
riscrive il diario da capo, si tiene accanto la pagina com'era e ci annota
soltanto quello che ha da aggiungere. Il vantaggio si vede in addestramento,
quando la rete corregge i propri pesi a partire dall'errore della previsione:
quell'avviso d'errore parte dall'ultimo strato e risale la pila fino al primo
(è la retropropagazione), un viaggio dentro la rete e non dentro il calendario.
Se gli strati sono tanti, per strada si smorza fino a sparire; se ogni strato
però conserva la pagina di partenza, la correzione trova sempre una scorciatoia
per arrivare in fondo, e anche una pila alta resta correggibile. Il tutto senza
ricorrenza: ogni istante si calcola in parallelo agli altri, e l'addestramento
vola sulle GPU invece di procedere un passo alla volta.

`````

`````{tab} Superiore

Una convoluzione 1-D causale con kernel di dimensione $k$ calcola l'uscita
al tempo $t$ usando solo $x_t, x_{t-1}, \dots, x_{t-(k-1)}$. Aggiungendo un
fattore di dilatazione $d$, i campioni vengono presi a distanza $d$:

$$
y_t = \sum_{i=0}^{k-1} w_i \, x_{t - i\cdot d},
$$

dove $y_t$ è l'uscita dello strato al tempo $t$ (non la serie, che qui resta
$x$), $w_0, \dots, w_{k-1}$ sono i pesi del filtro, condivisi lungo tutta la
sequenza, e $d$ è il passo di dilatazione. Dentro questa formula, e solo qui,
$w$ è un peso: altrove nella sezione $w$ è la lunghezza della finestra passata.
Impilando $L$ strati con dilatazioni $d_\ell = 2^{\ell}$ per
$\ell = 0, \dots, L-1$, il campo recettivo è

$$
r = 1 + (k-1)\sum_{\ell=0}^{L-1} 2^{\ell} = 1 + (k-1)\,(2^{L}-1),
$$

cioè cresce esponenzialmente con la profondità $L$: con $k=2$ e $L=3$ si
ha $r = 8$, come in figura; con $k=3$ e $L=6$ si arriva a $r = 127$ istanti.
La formula vale per una convoluzione per livello, che è la forma della TCN in
PyTorch; l'implementazione originale di Bai e colleghi ne mette due per
blocco residuo, e allora il campo recettivo quasi raddoppia,
$1 + 2(k-1)(2^L-1)$, che è il doppio meno uno: quel $127$ diventa $253$. Chi
applica la formula sbagliata a una TCN vera ne sottostima il campo recettivo di
quasi un fattore due.

In pratica ogni blocco aggiunge una connessione residua (nello spirito delle
ResNet) per addestrare pile profonde senza che il gradiente svanisca. Qui
«residua» ha un senso diverso dai residui statistici dei modelli classici: non
«quello che avanza dopo aver tolto trend e stagione», ma una scorciatoia che
porta l'ingresso oltre il blocco, così che al blocco resti da imparare soltanto
la differenza. Rispetto a una RNN, in addestramento il calcolo è
parallelizzabile lungo il tempo: la profondità sequenziale è quella della pila,
$O(L)$, cioè $O(\log r)$ strati per un campo recettivo $r$ con dilatazioni
esponenziali, contro gli $O(n)$ passi di una ricorrente su una sequenza lunga
$n$. In generazione autoregressiva, invece, il passo $t$ resta in attesa del
$t-1$ anche qui.

Il punto di rottura c'è, di specie diversa da quello delle RNN: una ricorrente
sbiadisce la memoria lontana, una TCN la taglia di netto. Oltre $r$ istanti non
arriva niente, mai, per costruzione. Allungare la finestra di
ingresso oltre il campo recettivo non serve a nulla, e per guardare più
indietro si aggiungono strati o si allarga il filtro.

`````

## DeepAR: una rete per mille serie, e una distribuzione

Torniamo al problema di scala affrontato da Amazon. **DeepAR**
{cite}`salinas2020deepar` è la risposta neurale, e porta due idee da tenere
distinte.

La prima è il modello globale, già discusso: una rete sola, addestrata su tutte
le serie insieme, che legge un giorno per volta e si porta dietro il proprio
riassunto del passato (è una LSTM, quella della {doc}`sezione sui modelli di
sequenza </NaturalLanguageProcessing/modelli-sequenza>`). A ogni passo
riceve due cose, il valore del giorno prima e le informazioni esterne di quel
giorno: il calendario, il prezzo, una promozione già decisa. Sono le variabili
esogene della sezione sui modelli classici, che in questa letteratura cambiano
nome e si chiamano **covariate**.

La seconda idea è più sottile: DeepAR non predice un *numero*, predice un
ventaglio di valori possibili con le loro probabilità. Un oggetto del genere, in
statistica, si chiama distribuzione.

`````{tab} Elementare

Un bollettino serio non dice «domani piove»: dice «70% di probabilità di
pioggia». DeepAR fa lo stesso con le vendite. A ogni passo, invece di sputare
una cifra, descrive un ventaglio di futuri plausibili: il valore più
probabile e quanto ci si può discostare. Un bollettino così si giudica da
quanta probabilità aveva assegnato all'esito che poi si è verificato: alzare quel
punteggio, su tutte le serie insieme, è tutto l'addestramento della rete. Ogni
serie, prima di entrare, passa dalla bilancia: la si divide per la propria
media, e alla fine la previsione si rimoltiplica per quella. Senza, il prodotto
da trentamila pezzi al giorno coprirebbe quello da tre, e la rete imparerebbe
soltanto dal più grande.

Per prevedere una settimana intera, la rete «tira i dadi» tante volte (genera
migliaia di storie possibili, e in ciascuna il valore appena tirato diventa il
punto di partenza del tiro successivo) e poi legge il ventaglio. Il valore che
sta in mezzo, con metà delle storie sotto e metà sopra, è la **mediana** e fa da
previsione; scartando poi il 10% delle storie più basse e il 10% delle più alte,
quelle che restano formano la **banda di incertezza**, dentro cui cadono otto
storie su dieci. Questo modo di procedere, tirare un valore e ripartire da lì,
si chiama campionamento ancestrale, perché ogni valore estratto è l'antenato di
quelli che vengono dopo di lui.

`````

`````{tab} Superiore

A ogni passo la rete emette i parametri $\boldsymbol{\lambda}_t$ di una
distribuzione di verosimiglianza $p(x_t \mid \boldsymbol{\lambda}_t)$: una
gaussiana, e allora $\boldsymbol{\lambda}_t=(\mu_t,\sigma_t)$, per dati reali;
una **binomiale negativa** per conteggi non negativi (come le vendite). Il
$\boldsymbol{\lambda}_t$ è in grassetto perché ha più di una componente; la
stessa lettera altrove nel libro è il coefficiente di penalità, qui no. La loss
è la log-verosimiglianza cambiata di segno, sommata su tutte le serie,

$$
\mathcal{L}(\theta) = -\sum_{i=1}^{N} \sum_{t} \log p\big(x^{(i)}_t \mid
\boldsymbol{\lambda}^{(i)}_t\big), \qquad
\boldsymbol{\lambda}^{(i)}_t = g_\theta\big(\mathbf{h}^{(i)}_t\big),
$$

dove $\mathbf{h}^{(i)}_t$ è lo stato nascosto della LSTM per la serie $i$,
$\boldsymbol{\lambda}^{(i)}_t$ i parametri d'emissione che ne discendono
(quindi anch'essi propri di quella serie) e $\theta$ i parametri *condivisi*
fra tutte le serie: minimizzare $\mathcal{L}$ equivale a massimizzare la
verosimiglianza dei dati. Fra addestramento e inferenza c'è un'asimmetria che
la formula non mostra: in addestramento lo stato $\mathbf{h}_t$ si calcola dal
valore osservato del passo prima, in inferenza da quello appena campionato. È
il *teacher forcing* della {doc}`sezione sul seq2seq
</NaturalLanguageProcessing/seq2seq-traduzione>`, e il disallineamento fra le
due fasi non si chiude. Gli autori però dichiarano di non averne osservato
effetti negativi nel forecasting, a differenza di quanto è stato riportato per
il linguaggio, e di aver provato varianti dello *scheduled sampling* senza
guadagni di accuratezza e con una convergenza più lenta.

La loss è la ricetta di
{doc}`Da dove viene la loss </RetiNeurali/da-dove-viene-la-loss>`, applicata a
un passo temporale alla volta.

Un modello globale non sta in piedi senza una gestione della scala: prima di
entrare nella rete ogni serie viene divisa per una scala $\nu_i$ propria (nel
lavoro citato, uno più la media dei suoi valori nella finestra di
condizionamento, che gli autori chiamano un'euristica), e i parametri della
distribuzione che esce vengono riportati alla scala di partenza, ciascuno con
la sua regola: la media si moltiplica per $\nu_i$, la dispersione della
binomiale negativa si divide per $\sqrt{\nu_i}$. Senza, una serie da trentamila
unità e una da tre non possono stare nella stessa rete, e la seconda sparisce
nel gradiente della prima. Alla scala si affianca un secondo accorgimento, che
pesa quanto il primo: la scala pareggia le ampiezze, il sorteggio decide quante
volte una serie viene guardata. Le finestre di addestramento si estraggono con
probabilità proporzionale alla scala della serie, e non in modo uniforme,
perché le serie grandi sono poche e, a sorteggio uniforme, la rete le vedrebbe
troppo di rado.

La previsione multi-passo avviene per campionamento ancestrale, e siccome
si lavora su una serie alla volta l'indice $i$ resta d'ora in poi sottinteso:
si estrae $\hat{x}_{t+1}\sim p(\cdot\mid \boldsymbol{\lambda}_{t+1})$, lo si
reinietta come input per calcolare lo stato successivo,
$\mathbf{h}_{t+2} = \text{LSTM}(\mathbf{h}_{t+1}, \hat{x}_{t+1},
\mathbf{z}_{t+2})$, e da quello i parametri del passo dopo; si ripete fino
all'orizzonte; molte traiettorie così
ottenute forniscono, per ogni passo, i quantili della previsione. Gli
intervalli non hanno una formula chiusa: si ottengono per Monte Carlo.

`````

Il campionamento ancestrale è il motore della previsione, e lo si vede
all'opera su un modellino giocattolo di venti righe. La rete vera, a ogni
passo, calcola due numeri: attorno a che valore si aspetta il giorno dopo, e di
quanto quel giorno può discostarsene. Qui quei due numeri non li impara
nessuno, glieli diamo noi con la regola più semplice che abbiamo, la stessa
AR(1) della sezione sui modelli classici (il 60% del valore di ieri, più
quattro), perché interessa il meccanismo: campionare, reiniettare, ripetere.

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

La mediana rientra verso la media di lungo periodo, la stessa della sezione sui
classici, cioè il valore che la regola lascia invariato, $\mu = 4/(1-0{,}6) =
10$. La banda si allarga: la distanza fra i suoi due estremi, stampata
nell'ultima colonna, passa da $2{,}56$ a $3{,}01$ a $3{,}15$.

Poi, dal terzo giorno in poi, la salita quasi si spegne: $3{,}149$, $3{,}159$,
$3{,}166$. La banda vera cresce ancora di qualche centesimo, ma a quella scala
ventimila storie non bastano più a separare l'allargamento dal tremolio del
sorteggio. (Il programma stampa la larghezza a parte perché, ricavata dagli
estremi arrotondati al centesimo, sembrerebbe non crescere affatto.) Il limite,
qui, vale $3{,}20$, e al quinto giorno la banda vera ne ha già raggiunto il
$99{,}7\%$.[^banda-limite]
Disegnata su dieci giorni ({numref}`fig-ventaglio-che-si-assesta`), quella
salita che si spegne è la forma dell'intero ventaglio.

```{figure} ../figures/ventaglio-che-si-assesta.svg
:name: fig-ventaglio-che-si-assesta
:alt: "Due grafici affiancati. A sinistra, il ventaglio di previsione: in ascissa i giorni previsti in avanti, da oggi a dieci, in ordinata il valore. Dall'ultimo giorno osservato, un punto in alto a sinistra, parte una linea terracotta, la mediana, che scende e si appiattisce sulla riga ocra della media di lungo periodo, che vale dieci; attorno a essa si apre una banda teal. A destra, la larghezza di quella stessa banda contro l'orizzonte: parte da 2,563 al primo giorno, sale a 3,128 al terzo e arriva a 3,2038 al decimo, dove la riga tratteggiata del tetto, 3,2039, le resta appena sopra: a questa scala le due linee non si distinguono. Quasi tutto l'allargamento sta nei primi tre giorni."
:width: 100%

Il ventaglio a sinistra e la sua larghezza a destra, sulla stessa regola AR(1)
del programma, con la banda calcolata invece che sorteggiata: per questo i
millesimi non coincidono con quelli che il programma stampa. La mediana rientra
verso la media di lungo periodo e la banda si allarga, ma non all'infinito:
quasi tutto l'allargamento sta nei primi tre giorni, poi la larghezza si
appoggia da sotto a un tetto che non supera.
```

L'assestamento è l'altra faccia del rientro verso la media dei modelli
classici: una serie che torna sempre verso il proprio valore centrale non può
diventare indefinitamente imprevedibile, e la sua banda si stabilizza su quella
di lungo periodo.

Cresce invece senza fermarsi l'incertezza delle serie prive di un valore
centrale, le non stazionarie. Nel giocattolo manca poi l'incertezza sul modello
stesso, per un motivo semplice: la regola con cui prevediamo è, per ipotesi,
quella che genera la serie, perché l'abbiamo scritta noi da entrambe le parti.
Su una serie vera nessuno la conosce, va stimata dai dati e può risultare
sbagliata.

Nel giocattolo la regola è così semplice che quei cinque intervalli si potrebbero
calcolare anche con carta e penna, senza generare nessuna storia, e infatti i
due calcoli coincidono a meno di tre centesimi (lo scarto più grande, al quinto
giorno, è $3{,}166$ contro $3{,}194$): la differenza che resta è il tremolio del
sorteggio, non un difetto del metodo. Generare tante storie a caso e leggere il
ventaglio che ne viene si chiama metodo Monte Carlo, come il casinò, e nella
rete vera è l'unica strada praticabile, perché il calcolo passa per una LSTM e
per gli intervalli non esiste una formula chiusa. Il ventaglio, in ogni caso, è
l'informazione che una previsione probabilistica dichiara e una puntuale nasconde.

## N-BEATS: solo percettroni, ma interpretabili

Nel 2019 Boris Oreshkin e colleghi mostrarono che per battere i metodi
statistici non servivano né ricorrenza né convoluzioni. Bastavano gli strati
più ordinari che ci siano, quelli in cui ogni neurone guarda tutti quelli dello
strato precedente (i percettroni della {doc}`sezione sul percettrone
</RetiNeurali/percettrone>`), montati con la giusta architettura
{cite}`oreshkin2020nbeats`.[^date-nbeats] **N-BEATS**
(*Neural Basis Expansion Analysis for Time Series*) è fatto solo di quelli, e la
sua eleganza sta in un'idea di contabilità: il **doppio residuo**.

`````{tab} Elementare

Una serie si spiega a strati, come si pela una cipolla. Il primo blocco guarda
la finestra passata e produce due cose: una ricostruzione di
ciò che ha capito del passato (il *backcast*) e un pezzo di previsione del
futuro (il *forecast*). A questo punto si *sottrae* la ricostruzione dal
passato: ciò che resta è quello che il primo blocco non ha saputo spiegare, e
passa al secondo blocco, che ripete il gioco su quel residuo. Blocco dopo
blocco, ogni strato spiega un pezzo in più; le previsioni parziali si
sommano a formare quella finale.

Il bello viene quando a qualche blocco si lega la mano. A uno si concede di
disegnare soltanto curve dolci, e finisce per occuparsi della tendenza di
fondo; a un altro soltanto onde che si ripetono, e finisce sui cicli. Così la
rete non solo prevede, ma mostra quanto della previsione è tendenza e
quanto è ciclo (cosa rara per una rete neurale).

`````

`````{tab} Superiore

Ogni blocco $b$ riceve un residuo $\mathbf{x}^{(b)}$ e, tramite una pila di
strati densi seguita da una proiezione su una base, produce due uscite: un
**backcast** $\hat{\mathbf{x}}^{(b)}$ (la parte d'ingresso che sa spiegare) e un
**forecast** $\hat{\mathbf{y}}^{(b)}$ (il suo contributo alla previsione). Il
doppio residuo li combina così:

$$
\mathbf{x}^{(b+1)} = \mathbf{x}^{(b)} - \hat{\mathbf{x}}^{(b)}, \qquad
\hat{\mathbf{y}} = \sum_{b} \hat{\mathbf{y}}^{(b)},
$$

dove l'ingresso del blocco successivo è ciò che il blocco corrente non ha
saputo spiegare, e la previsione finale è la somma dei contributi. Nella
variante *interpretabile* la base è vincolata (polinomi di grado basso per il
trend, termini di Fourier per la stagionalità), così che i blocchi, raccolti in
due gruppi, restituiscano una componente leggibile ciascuno; in quella
*generica* la base è appresa liberamente, e lì di componenti statistiche
innestate non ce n'è nessuna: solo strati densi. Nel gruppo della tendenza il
forecast di un blocco è $\hat{\mathbf{y}}^{(b)} =
\mathbf{T}\,\mathbf{c}^{(b)}$, con $\mathbf{T} = [\mathbf{1}, \mathbf{t},
\dots, \mathbf{t}^p]$, $\mathbf{t} = (0, 1, \dots, h-1)^\top / h$ e $p$ piccolo
(gli autori usano 2); nel gruppo della stagionalità $\mathbf{T}$ è sostituita
dalla matrice dei $\cos(2\pi i\,\mathbf{t})$ e dei $\sin(2\pi i\,\mathbf{t})$,
per $i$ fino a $\lfloor h/2 \rfloor - 1$. I coefficienti $\mathbf{c}^{(b)}$ li
predice la pila densa del blocco, e lo stesso vale per il backcast, con la base
calcolata sulla finestra passata. Il gruppo della tendenza precede quello della
stagionalità, così che la tendenza è già tolta dalla finestra quando questa
raggiunge la stagionalità.

Il risultato sulla M4 va letto con il suo protocollo. Gli autori ottengono un
miglioramento di circa l’11% sul riferimento statistico e del 3% sul vincitore
della gara, ma da un insieme di 180 reti: tre funzioni di perdita (sMAPE, MASE
e MAPE), sei lunghezze della finestra passata, da due a sette volte
l'orizzonte, e diverse inizializzazioni casuali, con la mediana come regola di
aggregazione. In appendice mostrano che già con 18 reti il risultato resta il
migliore della M4 {cite}`oreshkin2020nbeats`. La valutazione è successiva alla
gara e usa i suoi dati: N-BEATS non vi ha partecipato.

`````

«Residuo» prende qui un terzo senso, imparentato con i primi due: era
l'imprevisto che avanzava dalla decomposizione, era l'errore che avanzava da un
modello stimato, e adesso è l'avanzo di un blocco della rete, che passa al
blocco dopo. Ogni volta è «ciò che non è stato spiegato», e cambia solo chi ha
provato a spiegarlo. Fuori dalla famiglia sta la *connessione residua* delle
TCN, che somma l'ingresso di un blocco alla sua uscita: lì non avanza niente,
l'ingresso viene soltanto portato oltre il blocco.

La variante in cui alcuni blocchi si occupano della tendenza e altri della
stagione si chiama interpretabile, e riallaccia il forecasting neurale alla
decomposizione della sezione sui modelli classici, chiudendo il cerchio con i
modelli statistici. N-BEATS mostrò poi che per battere i metodi statistici non
serviva innestare pezzi di statistica dentro la rete, come faceva l'ibrido
ES-RNN di Smyl che aveva vinto la M4: sui dati della M4, scrivono gli autori,
contro l'opinione corrente di allora, i mattoni del deep learning bastavano da
soli {cite}`oreshkin2020nbeats`.

## Transformer per le serie, e un modello lineare che li batte

Nel trattamento del linguaggio il {doc}`meccanismo di attenzione
</Transformers/attenzione>`, che guarda tutti i pezzi di una sequenza insieme e
pesa quanto ciascuno conta, ha preso il posto delle reti che leggevano in fila
{cite}`vaswani2017attention`. Portarlo nelle serie temporali era naturale: un
Transformer mette in comunicazione due giorni lontanissimi con un solo
passaggio, mentre una rete ricorrente deve trascinare l'informazione attraverso
tutti i giorni in mezzo, e per questo la dimentica.

Ne sono nate molte architetture dedicate, e le due più note partono dallo
stesso inciampo e prendono strade diverse. L'inciampo è il costo: confrontare
ogni giorno con ogni altro giorno comporta, su una finestra di mille giorni, un
milione di confronti, e il numero cresce col quadrato della lunghezza.
**Informer** {cite}`zhou2021informer` ne calcola solo una parte, scelta con
cura, invece di tutti. **Autoformer** {cite}`wu2021autoformer` cambia domanda:
invece di confrontare coppie di giorni cerca le somiglianze della serie con sé
stessa fatta scivolare indietro, cioè l’autocorrelazione (da cui il nome). Per
calcolarle tutte in fretta usa la trasformata di Fourier, quella con cui la
{doc}`sezione dal suono alle feature </Audio/dal-suono-alle-feature>`
scomponeva un accordo nelle sue note, e da cui vengono anche il seno e il
coseno dei termini stagionali. Fra un blocco e l'altro scompone inoltre la
serie in tendenza e stagionalità.

Poi, nel 2022, un confronto con modelli lineari ridimensionò quei risultati.

`````{tab} Elementare

Un gruppo di ricercatori pose una domanda scomoda già nel titolo: «I
Transformer sono efficaci per prevedere le serie temporali?». La risposta,
sui banchi di prova più usati, fu spiazzante: dei modelli lineari
semplicissimi, poco più di una retta tirata sui dati, battevano quei Transformer
sofisticati. Ne provarono più d'uno: uno di essi separa
prima la serie in tendenza e stagionalità e poi tira una retta su ciascuna delle
due, e lo chiamarono **DLinear**; il più semplice è una sola retta.

Fecero poi due prove che valgono più della classifica. Nella prima, con la retta
più semplice, mescolarono l'ingresso: presero i giorni passati da dare in pasto
al modello e li rimisero in ordine sparso. Un modello che sfrutta l'ordine del
tempo, così, dovrebbe crollare. Sui cambi fra valute i Transformer non se ne
accorsero, mentre la retta peggiorò di più di un quarto: lì il tempo lo stava
usando lei. Su un'altra serie, invece, mescolare fa male quasi a tutti. Quanto
un modello usi l'ordine del tempo, dunque, non è una sua proprietà fissa: si
scopre misurandola, banco di prova per banco di prova.

Nella seconda, questa volta con tutta la famiglia, allungarono il passato da
leggere. Chi tira fuori qualcosa da una storia lunga dovrebbe prevedere meglio
quando gliene si dà di più; i Transformer, con la finestra più lunga, restavano
fermi o peggioravano, mentre i modelli lineari miglioravano quasi dappertutto.

La conclusione non è che i Transformer non servano: la complessità non è mai un
vantaggio gratuito. Prima di celebrare un modello elaborato va confrontato con
la linea di base più semplice a cui riesci a pensare, e va controllato che usi
l'informazione che dichiara di usare.

Un Transformer costruito apposta, come il Temporal Fusion Transformer, continua
invece a servire quando le informazioni esterne sono molte e di tipi diversi:
quelle che non cambiano mai (la città del negozio), quelle già scritte in
calendario anche per i giorni da prevedere (le feste), quelle che si conoscono
solo per i giorni già passati (il meteo misurato). Serve anche quando si vuole
sapere su che cosa il modello si è appoggiato: indica a quali informazioni e a
quali giorni ha dato più peso, cosa che una retta non sa fare. La riserva è che
questa indicazione è un indizio su dove il modello ha guardato, non la prova di
che cosa ha usato per decidere.

`````

`````{tab} Superiore

Il costo quadratico $O(n^2)$ dell'attenzione piena sulle sequenze lunghe aveva
motivato una fila di architetture dedicate (Informer {cite}`zhou2021informer`,
Autoformer {cite}`wu2021autoformer`, FEDformer {cite}`zhou2022fedformer`), che
scendono tutte sotto il quadrato pur dichiarando, le ultime due, di inseguire
l'accuratezza prima dell'efficienza. Zeng e colleghi
{cite}`zeng2023transformers` proposero come confronto una famiglia di modelli
lineari, che nei lavori successivi si trova come LTSF-Linear: il più semplice è
una singola mappa $\hat{\mathbf{x}}_{t+1:t+h} =
\mathbf{W}\,\mathbf{x}_{t-w+1:t}$, dove $w$ è la lunghezza della finestra
passata, $h$ l'orizzonte e $\mathbf{W}$ la matrice appresa, e la variante
DLinear decompone prima la
serie in trend e stagionalità e applica una mappa a ciascuna componente. Su
nove dataset di forecasting a lungo orizzonte quella famiglia eguaglia o supera
i Transformer dedicati. Il protocollo va detto, perché sta in appendice e
cambia la lettura del confronto: nella tabella principale i modelli lineari
ricevono una finestra passata di $336$ istanti e i Transformer di $96$. Gli
autori lo giustificano con la prova sulla lunghezza della finestra passata, e i
lavori successivi hanno ricopiato quella scelta dichiarandola.

Il verdetto, preso a sé, sarebbe una classifica, e le classifiche invecchiano.
Non invecchiano invece le due prove con cui gli autori lo spiegano. I numeri
della prima si riferiscono al modello lineare semplice; la seconda è fatta su
tutta la famiglia.

*La prima è il mescolamento dell'ingresso.* Se un modello usa l'ordine
temporale, rimescolare a caso le posizioni della finestra passata deve
rovinarlo. Sui tassi di cambio le prestazioni dei metodi basati su Transformer
non si muovono (lo scostamento medio è dell'ordine di un decimo di punto
percentuale, e per due dei tre è perfino in meglio) mentre lo stesso
trattamento
fa perdere il 27% al modello lineare: lì il tempo lo sta usando la retta.

Su ETTh1, che raccoglie la temperatura dell'olio di un trasformatore elettrico
(la macchina di una cabina, non l'architettura) e sei misure di carico, il
quadro è un altro: mescolando peggiorano anche i Transformer che hanno un'idea
del tempo dentro, FEDformer del 73% e Autoformer del 57% (il lineare dell’81%).
Informer no: si ferma al 2%. Se ne ricava che un banco di prova su cui un
modello non peggiora quando gli si distrugge l'ordine non misura ciò che
dichiara di misurare; i Transformer non sono ciechi al tempo per costruzione.

*La seconda è la lunghezza della finestra passata.* Un modello che estrae
relazioni temporali da una storia lunga deve migliorare quando gliene si dà di
più. I Transformer, allungando la finestra, restano fermi o peggiorano; i
modelli lineari migliorano sulla maggior parte di quei dataset (non su tutti:
sui tassi di cambio, dicono gli autori, la finestra lunga non aiuta nemmeno
loro). Messe insieme, le due
prove dicono che su quei banchi di prova l'attenzione non stava estraendo le
relazioni temporali che dichiarava di estrarre. Gli autori lo attribuiscono
all'invarianza per permutazione dell'attenzione e al sovradattamento al rumore,
e ne concludono che l'efficacia dei Transformer su questo compito va messa in
discussione.

Non è la condanna dei Transformer: il **Temporal Fusion Transformer** di Lim e
colleghi {cite}`lim2021temporal` resta forte quando servono covariate multiple
(statiche, note nel futuro, osservate solo nel passato) e interpretabilità. Le
covariate statiche passano in codificatori che condizionano tutto il resto;
reti di selezione delle variabili, costruite su blocchi residui con un *gating*
che può spegnere i rami inutili, pesano gli ingressi a ogni passo; un
codificatore-decodificatore LSTM tratta il contesto vicino e un'attenzione
multi-testa interpretabile quello lontano; e l'uscita è un insieme di quantili,
addestrati con la pinball loss della sezione sulla validazione sommata sui
livelli $\tau$, quindi senza assumere una famiglia di distribuzioni, a
differenza di DeepAR. L'interpretabilità viene dalla selezione delle variabili,
dalla scomposizione dell'importanza per tipo di ingresso e da pesi di
attenzione ispezionabili. Su quest'ultimo punto vale la cautela della
{doc}`sezione su attribuzione e meccanicistica
</Interpretabilita/attribuzione-e-meccanicistica>`: i pesi di attenzione sono
un indizio suggestivo, non una spiegazione affidabile
{cite}`jain2019attention`, e le prime due leve restano valide anche senza quei
pesi.

La classifica si è già mossa. Nie e colleghi hanno risposto a Zeng sugli
stessi banchi e con la stessa finestra lunga: **PatchTST** taglia la finestra
passata in segmenti di $16$ istanti con passo $8$ e usa ciascun segmento come
un token, alla maniera dei *patch* di un Vision Transformer; tratta poi ogni
variabile come una serie a sé con pesi condivisi (*channel independence*).
Con token che portano già una forma locale, e una sequenza di token circa otto
volte più corta, l'attenzione torna a guadagnare dalla finestra lunga, e il
modello supera DLinear sulla maggior parte di quei dataset
{cite}`nie2023time`. Le due prove di Zeng restano valide come metodo. Secondo
Nie e colleghi colpivano due scelte insieme, un istante per token e tutte le
variabili mescolate nello stesso token, e l'ablazione mostra che contano tutte e
due, i segmenti e il trattamento indipendente delle variabili.

`````

## Foundation model per le serie temporali

I *foundation model* per le serie temporali riprendono l'idea dei grandi
modelli linguistici: pre-addestrare un modello su una collezione molto vasta di
serie, e poi usarlo per prevedere serie mai viste, senza riaddestrarlo. È il
forecasting *zero-shot*. I primi sono comparsi dal 2023, e rispetto a un
modello linguistico sono piccoli: da qualche decina a qualche centinaio di
milioni di parametri.

`````{tab} Elementare

Come un modello linguistico impara la lingua leggendo miliardi di frasi e poi
sa scrivere su argomenti nuovi, un *foundation model* per le serie impara la
«grammatica» dei fenomeni temporali (tendenze, cicli, picchi) leggendo milioni
di serie di ogni tipo. Poi gli dai una serie che non ha mai incontrato, per
esempio le vendite del tuo negozio, e lui prevede il seguito senza
addestrarsi da capo. Il modello Chronos, di Amazon, usa un trucco
sorprendente: trasforma i numeri in parole. Si sceglie un vocabolario di
qualche migliaio di simboli e si stabilisce che ciascuno copre una fettina di
valori, così «$23{,}7$ gradi» diventa, mettiamo, il simbolo numero $1372$;
a quel punto una serie è una frase, e prevedere il seguito è completare la
frase, cioè il compito per cui i modelli linguistici sono già fatti.
Siccome una frase si può finire in molti modi, il modello ne scrive
tante e legge il ventaglio che ne viene: anche qui la previsione arriva con la
sua incertezza.

Promettente, ma è un campo giovane: non batte sempre i metodi su misura, né i
vecchi classici. Quando lo si mette alla prova su una serie, resta inoltre il
sospetto che quella serie fosse già fra i milioni che ha letto: chi ha già
visto il compito non improvvisa.

`````

`````{tab} Superiore

**Chronos** {cite}`ansari2024chronos` (Amazon, 2024) scala e quantizza i valori
reali in un vocabolario finito di *token*, poi addestra un modello linguistico
della famiglia T5, da 20 a 710 milioni di parametri, con la consueta
*cross-entropy* sul token successivo, su un grande corpus di serie reali e
sintetiche. In inferenza campiona traiettorie di token e le riconverte in
valori, ottenendo una previsione probabilistica senza architetture ad hoc.
Un'alternativa è **TimesFM** di Google {cite}`das2024timesfm` (2023), un
modello *decoder-only* da 200 milioni di parametri che lavora su *patch* di
istanti, pre-addestrato su un corpus dell'ordine di cento miliardi di punti
temporali. Entrambi dichiarano prestazioni *zero-shot* competitive con modelli
allenati sul singolo compito, e il protocollo conta: dei 42 dataset su cui
Chronos si misura, 15 fanno parte anche del suo addestramento, e il confronto
lì è in dominio, mentre lo *zero-shot* vale per gli altri 27, tenuti fuori. Di
qui la cautela: il campo ha pochi anni, i confronti dipendono da quali serie il
modello ha già visto e da quale banco di prova si usa, e su serie corte o molto
specifiche i metodi classici restano spesso preferibili.

`````

## In pratica: una TCN in PyTorch

Traduciamo la TCN della {numref}`fig-tcn-convoluzioni-causali` in codice. Il
cuore è la convoluzione causale, e ottenerla richiede un piccolo trucco, che è
poi quello dell'implementazione originale.

Una finestra che scorre su una fila di numeri, arrivata al
primo giorno, non ha niente alla sua sinistra: le mancano dei valori, e la
libreria li mette a zero. Solo che li mette da tutte e due le parti, anche a
destra, dove il tempo non è ancora arrivato, e così il risultato viene più lungo
dell'ingresso. La soluzione è chiedere gli zeri lo stesso e
poi tagliare via la coda a destra: quello che resta è come se gli zeri
fossero stati messi soltanto a sinistra, la sequenza è lunga quanto prima e
l'uscita del giorno $t$ non ha mai pescato nel futuro.

Gli zeri da chiedere sono $(k-1)\,d$, dove $k$ è l'ampiezza della finestra
(quanti valori prende) e $d$ il salto che li distanzia: con una finestra da tre
valori distanziati di due, per esempio, ne servono quattro.

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

Che la rete guardi quindici istanti, e non uno di più, si verifica dando
uno scossone a un solo punto della finestra e guardando se l'uscita cambia.

```python
torch.manual_seed(0)
tcn = TCN().eval()
x = torch.randn(1, 40)                  # una finestra di 40 istanti
with torch.no_grad():
    base = tcn(x)
    for indietro in (0, 14, 15, 20):    # quanti passi prima dell'ultimo
        x2 = x.clone()
        x2[0, -1 - indietro] += 5.0     # uno scossone in quel punto
        print(indietro, round((tcn(x2) - base).abs().item(), 4))
```

```text
0 1.1355
14 0.1235
15 0.0
20 0.0
```

Quattordici passi indietro l'uscita risente ancora dello scossone, a quindici
non lo vede più: è il campo recettivo $1+(k-1)(2^L-1)=15$, e oltre quello la
rete è cieca per costruzione.

L'addestramento è il consueto ciclo di discesa del gradiente del {doc}`training
loop di PyTorch </PyTorch/addestramento>`. Si taglia la serie in coppie, la
finestra dei giorni passati e il valore del giorno dopo, rispettando la
separazione temporale imposta dalla sezione sulla validazione e senza mai
mescolare futuro e passato. Poi si passano le finestre al modello e si spingono
le sue uscite verso i valori giusti, misurando la distanza con l'errore
quadratico (`nn.MSELoss`) e lasciando che a correggere i pesi ci pensi
`torch.optim.Adam`.

Per farne un modello probabilistico, nello spirito di DeepAR, si fanno uscire
dalla rete due numeri invece di uno, il valore centrale e quanto ci si può
discostare (passando per una funzione che la tiene positiva, come l'esponenziale
o la softplus), e
si cambia il bersaglio dell'addestramento. Non più «avvicina la
previsione al valore vero», ma «rendi il valore vero il più probabile possibile
secondo il ventaglio che stai dichiarando»: in gergo, si sostituisce la MSE con
la log-verosimiglianza gaussiana cambiata di segno. La larghezza che la rete
impara così, però, va controllata come ogni altra banda, con la copertura
empirica della sezione sulla validazione, perché niente garantisce che sia
quella giusta.

Dalle reti ricorrenti ai foundation model le architetture cambiano, e restano
due criteri: prevedere significa dichiarare l'incertezza, e nessun modello, per
quanto profondo, è dispensato dal confronto con la linea di base classica.

`````{tab} Elementare

```{admonition} Da ricordare
:class: important
- Il deep learning conviene quando hai tante serie collegate fra loro (la
  catena di diecimila negozi, non il negozio di quartiere), quando i legami non
  sono semplici proporzioni, quando ci sono cause esterne che aiutano, quando ti
  serve non un numero ma un ventaglio: non tutte insieme, ma più ce ne sono,
  meglio è. Su poche serie corte i classici spesso vincono ancora.
- Fra le due famiglie ce n'è una terza, e su moltissimi problemi aziendali
  basta: Prophet non insegue la regola con cui un giorno genera il
  successivo, prende il calendario e ci disegna sopra una curva fatta di pezzi
  che si guardano uno per uno (una tendenza spezzata, le stagioni, le
  festività). Per questo i buchi nei dati non lo rompono
  {cite}`taylor2018forecasting`, mentre i valori assurdi vanno tolti. E per lo
  stesso motivo, su una serie in cui oggi somiglia molto a ieri, butta via
  l'informazione migliore che ha, quella che un ARIMA invece usa.
- Le reti ricorrenti leggono la serie un giorno per volta portandosi dietro
  una memoria, ma su storie lunghe se la dimenticano e sono lente da addestrare.
  Le TCN {cite}`bai2018empirical` rimediano a tutt'e due rileggendo il
  diario a salti che raddoppiano: quattro strati vedono sedici giorni, dieci
  ne vedono più di mille, e nessuno strato può sbirciare in avanti. In
  cambio, oltre quei giorni non vedono niente, mentre una rete ricorrente
  sbiadisce il passato lontano senza cancellarlo.
- DeepAR {cite}`salinas2020deepar` addestra una sola rete su tutte le
  serie insieme e non prevede un numero, prevede un ventaglio di futuri
  possibili: tira i dadi tante volte, ogni volta ripartendo dal valore appena
  tirato, e poi legge il ventaglio. Su una serie che rientra sempre verso la
  propria media, quel ventaglio si allarga in fretta per qualche passo e poi
  quasi si ferma, avvicinandosi a una larghezza massima che non supera mai.
- N-BEATS {cite}`oreshkin2020nbeats` pela la serie a strati come una
  cipolla: ogni blocco spiega quello che può e passa al successivo quello che
  resta. Con un vantaggio raro per una rete: si può fare in modo che mostri
  quanto della previsione è tendenza e quanto è ciclo.
- Sui Transformer per le serie conviene la cautela: una retta ben usata li
  ha battuti su molti banchi di prova {cite}`zeng2023transformers`, e su uno di
  quei banchi, i cambi fra valute, mescolando l'ordine dei giorni i Transformer
  non se ne sono nemmeno accorti: il tempo, lì, non lo stavano usando. I
  foundation model come Chronos
  {cite}`ansari2024chronos` imparano la «grammatica» dei fenomeni temporali su
  milioni di serie e poi prevedono serie mai viste: campo promettente e giovane,
  non risolto.
```

`````

`````{tab} Superiore

```{admonition} Da ricordare
:class: important
- Il deep learning conviene con molte serie collegate, relazioni non
  lineari, covariate esterne e quando serve una previsione
  probabilistica; su poche serie corte i classici spesso vincono ancora.
  Il modello globale non è una prerogativa delle reti: alla M5 hanno vinto
  alberi potenziati globali. Fra le due famiglie stanno i modelli additivi
  decomponibili (Prophet {cite}`taylor2018forecasting`), che regrediscono sul
  tempo e non sui valori passati: robusti ai buchi, non agli outlier, che
  assorbono come cambi di tendenza, e ciechi all'autocorrelazione a breve.
- Le RNN/LSTM {cite}`hochreiter1997long` fanno forecasting *seq-to-one* o
  *seq-to-seq*, ma soffrono orizzonti lunghi e addestramento sequenziale. Le
  TCN {cite}`bai2018empirical` usano convoluzioni causali dilatate: campo
  recettivo esponenziale, $1+(k-1)(2^L-1)$ con una convoluzione per livello e il
  doppio meno uno con le due del blocco originale, e in addestramento una
  profondità sequenziale $O(L)$ invece di $O(n)$.
  Oltre quel raggio però non arriva niente, per costruzione: dove una
  ricorrente sbiadisce, una TCN taglia.
- DeepAR {cite}`salinas2020deepar` è una RNN autoregressiva globale (una rete
  per molte serie) che emette una distribuzione; la previsione multi-passo è
  per campionamento ancestrale. La banda segue la varianza dell'errore a $h$
  passi, $\sigma^2\sum_{j<h}\psi_j^2$: si allarga senza limite se il processo
  non è stazionario; se lo è, tende dal basso alla larghezza di regime, $2
  z_{1-\alpha/2}$ volte la deviazione standard del processo, che l'AR(1) del
  giocattolo non raggiunge in nessun numero finito di passi e un MA($q$)
  raggiunge al passo $q+1$, quando i $\psi_j$ si annullano.
- N-BEATS {cite}`oreshkin2020nbeats` usa blocchi di soli MLP
  (percettroni multistrato: pile di strati densi) con doppio residuo
  backcast/forecast, ed è interpretabile quando la base è vincolata. Il suo
  contributo è di meccanismo: per battere gli ibridi statistico-neurali non
  serviva innestare statistica dentro la rete, anche se il risultato sulla M4
  viene da un insieme di 180 reti (e regge con 18).
- Sui Transformer per le serie, cautela: la famiglia LTSF-Linear di Zeng e
  colleghi (il lineare semplice e la sua variante con decomposizione,
  DLinear) li eguaglia o supera su nove dataset
  {cite}`zeng2023transformers`, e le due prove che lo spiegano, la prima sul
  lineare semplice e la seconda su tutta la famiglia (su una serie mescolare
  l'ingresso non scalfisce i Transformer, e allungare la finestra non li
  migliora), dicono che su quei banchi l'ordine temporale non era sempre quello
  che stavano sfruttando. Il TFT {cite}`lim2021temporal` resta utile per
  covariate multiple e interpretabilità, purché i suoi pesi di attenzione si
  leggano come indizio e non come prova {cite}`jain2019attention`. I foundation
  model come Chronos {cite}`ansari2024chronos` promettono forecasting
  *zero-shot*, che va letto sui soli dataset tenuti fuori dall'addestramento:
  campo promettente ma giovane, non risolto.
```

`````

Tutti questi modelli hanno per materia prima la storia del fenomeno, con poco
altro accanto: il periodo di un ciclo, un calendario, le date delle feste,
qualche informazione esterna. Nessuno ha spiegato al modello perché la marea
sale, e la marea si prevede lo stesso finché le regolarità tengono. (La
macchina di Kelvin, a dire il vero, un pezzo di quella spiegazione lo
conteneva: i periodi delle sue onde li dava l'astronomia, e dalle misure si
ricavavano soltanto l'ampiezza e la fase di ciascuna.) Nel {doc}`capitolo sulle
PINN </PINN/overview>` la spiegazione che qui mancava entra nell'addestramento,
sotto forma di equazione.

[^banda-limite]: I conti, per chi li vuole. Un AR(1) con $|\phi|<1$ ha
    varianza di lungo periodo $\sigma^2/(1-\phi^2)$, che con $\sigma=1$ e
    $\phi=0{,}6$ dà una deviazione di $1{,}25$; una banda all'80% è larga
    $2\times 1{,}2816$ deviazioni, quindi al limite vale $3{,}20$. A $h$ passi
    la deviazione vale $\sigma\sqrt{(1-\phi^{2h})/(1-\phi^2)}$, e il rapporto
    fra le due è $\sqrt{1-\phi^{2h}}$, che a cinque passi (quindi con
    l'esponente $10$, non $5$) fa $0{,}997$. La banda vera al terzo, quarto e
    quinto passo è dunque $3{,}13$, $3{,}18$, $3{,}19$: cresce ancora, di
    centesimi, differenze che il campionamento non riesce più a distinguere.

[^date-nbeats]: L'articolo circolò online nel 2019 e fu presentato in conferenza
    l'anno dopo; a DeepAR era successa una cosa simile, con tre anni e una
    rivista al posto di una conferenza. Questi due lavori sono datati alla
    prima circolazione, che è la data in cui l'idea è entrata nel campo. Quando
    le due date si allontanano e la distanza conta, il capitolo la dichiara; la
    voce in bibliografia resta invece quella della pubblicazione, ed è normale
    che le due non coincidano.
