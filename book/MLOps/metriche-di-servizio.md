# Misurare la generazione: TTFT, TPOT e goodput

Nel 1968, in un articolo intitolato *Response Time in Man-Computer
Conversational Transactions*, Robert B. Miller (ricercatore dell'IBM di
Poughkeepsie) cataloga i tempi di risposta che una persona tollera quando
dialoga con una macchina. Venticinque anni dopo Jakob Nielsen ne distilla, in
*Usability Engineering*, tre soglie diventate proverbiali: entro
**0,1 secondi** la risposta sembra istantanea, entro **1 secondo** il filo del
pensiero non si spezza, oltre **10 secondi** l'attenzione se ne va altrove.
Numeri che hanno retto mezzo secolo, perché non misurano un computer: misurano
una persona.

C'è però un presupposto nascosto, e i modelli generativi lo mandano in pezzi.
Miller dà per scontato che la risposta sia un **evento**, l'istante in cui la
macchina consegna il risultato. Per un classificatore è ancora così, e la
latenza è un numero solo, il tempo fra la domanda e la risposta; la sezione
«Servire un modello» ci ha insegnato a prometterlo per percentili.

Ma un modello che genera testo non consegna niente in un istante: consegna un
pezzo alla volta, per secondi, e mentre consegna il lettore sta già leggendo.
La domanda «quanto ci mette?» ha smesso di avere una risposta sola, e non è una
sottigliezza: tutte le tecniche della sezione precedente (riempire la sala
appena una sedia si libera, non riservare tavoloni, far indovinare il modello
piccolo, comprimere i pesi) si valutano male se le si misura con il metro
sbagliato. Prima di ottimizzare bisogna decidere che cosa vuol dire, per questo
servizio, *andare veloce*.

## Scomporre la latenza

La generazione ha due fasi, molto diverse fra loro, e sono quelle incontrate
nel capitolo sui Transformer parlando di KV cache.

Nella prima il modello **legge la domanda**: tutte le parole insieme, in un
colpo solo, prendendosi gli appunti che gli serviranno dopo (sono proprio gli
appunti della KV cache). Questa fase si chiama **prefill**.

Nella seconda **scrive la risposta**, un token alla volta, e ogni token lo
decide guardando tutti quelli già scritti. È il **decode**, ed è la fase lenta,
quella dove il tempo se ne va nel rileggersi i pesi anziché nel calcolare: è il
punto su cui è costruita l'intera sezione precedente.

Le due misure di base cadono esattamente su questa frattura.

`````{tab} Elementare

Pensa a un menù degustazione. Due cose diverse ti fanno impazientire, e non
vanno confuse: **quanto aspetti la prima portata** (sei seduto davanti a un
tavolo vuoto e non succede niente) e **ogni quanto arrivano le portate dopo**
(se si susseguono a ritmo la serata scorre, se fra una e l'altra passano venti
minuti ti innervosisci, anche a parità di durata totale).

Nella generazione di testo è identico. La prima attesa si chiama **TTFT**
(*time to first token*): il tempo che passa da quando premi invio a quando
compare la prima parola. La seconda si chiama **TPOT** (*time per output
token*): la pausa **media** fra una parola e la successiva.

Su quel «media» conviene fermarsi un istante, perché più avanti questa pagina
dirà che le medie mentono, e non si vuole che sembri una contraddizione. La
media va benissimo per descrivere un ritmo *regolare*, ed è per questo che il
TPOT esiste. Ma se in mezzo a duecento pause da un ventesimo di secondo ne
capita una da due secondi, la media si sposta appena e il testo, sotto gli
occhi, si è piantato in mezzo a una frase. Quindi accanto al TPOT si sorveglia
sempre anche **la pausa più lunga** di quella risposta: è lei che il lettore
ricorda.

Mettiamoci dei numeri, e teniamo a mente che un token è un pezzetto di parola
(più corto di una parola: ci torniamo fra poco con il conto esatto). TTFT di
350 millisecondi, TPOT di 25 millisecondi, risposta lunga 200 token, cioè un
centinaio di parole. Il primo token arriva dopo 0,350 secondi; poi ne
mancano 199, uno ogni 0,025 secondi, cioè 4,975 secondi. In tutto **5,325
secondi**. E il testo scorre sotto gli occhi a uno diviso 0,025 secondi (i 25
millisecondi riscritti in secondi, che è il passaggio che si dimentica), cioè
**40 token al secondo**.

`````

`````{tab} Superiore

Siano $\text{TTFT}$ il tempo dalla ricezione della richiesta all'emissione del
primo token, $\text{TPOT}$ il tempo medio fra due token consecutivi della
stessa risposta e $N_{\text{out}}$ il numero di token generati. La latenza
totale si ricompone come

$$
T = \text{TTFT} + (N_{\text{out}} - 1)\cdot \text{TPOT},
$$

dove $N_{\text{out}} - 1$ conta gli intervalli fra token, uno in meno dei token
stessi. Invertendo si ha la definizione con cui il TPOT si misura davvero,
$\text{TPOT} = (T - \text{TTFT}) / (N_{\text{out}} - 1)$. La grandezza percepita
non è $T$ ma la **velocità di scorrimento** $1/\text{TPOT}$, in token al
secondo.

Con $\text{TTFT} = 0{,}350$ s, $\text{TPOT} = 0{,}025$ s e
$N_{\text{out}} = 200$: $T = 0{,}350 + 199 \times 0{,}025 = 5{,}325$ s, con uno
scorrimento di $1/0{,}025 = 40$ token/s. Il rapporto
$N_{\text{out}}/T = 200/5{,}325 \approx 37{,}6$ token/s è invece il throughput
medio della *richiesta*: ingloba l'attesa iniziale e non descrive nessun istante
dell'esperienza.

Il TPOT è però una media, mentre ciò che l'utente vive è la distribuzione degli
intervalli, la **ITL** (*inter-token latency*): coincidono solo se il flusso è
regolare. Un singolo intervallo da due secondi in mezzo a duecento da 20
millisecondi sposta il TPOT di dieci millisecondi e rovina la risposta, e per
questo va sorvegliata anche l'ITL massima per richiesta.

`````

Che le due misure non siano intercambiabili si vede spendendo la stessa attesa
totale in due modi. Un sistema con TTFT di 0,350 s e TPOT di 25 ms impiega
5,325 secondi per 200 token; un secondo con TTFT di 2,340 s e TPOT di 15 ms
impiega $2{,}340 + 199 \times 0{,}015 = 5{,}325$ secondi, identici al
millesimo. Ma il primo comincia a scrivere quasi subito e scorre a 40 token al
secondo; il secondo lascia lo schermo bianco per oltre due secondi e poi sputa
il testo a $1/0{,}015 \approx 67$ token al secondo. Con un numero solo sarebbero
indistinguibili.

Il secondo, per giunta, spreca la sua velocità, e per capire perché serve
sapere quanto vale un token in parole. Un token non è una parola: è più corto,
perché le parole lunghe il modello le spezza in due o tre pezzi. Su un paragrafo
italiano il conto, misurato con due tokenizzatori diversi, sta fra un token e
mezzo e due e mezzo per parola. Un lettore adulto legge tre o quattro parole al
secondo, quindi sta consumando qualcosa come 5–10 token al secondo. Un sistema
che ne consegna 40 va dunque da quattro a otto volte più veloce di chi legge, e
accelerare ancora non si vede: le parole erano già lì prima che l'occhio le
raggiungesse. Il TTFT invece si sente sempre, perché è tempo in cui sullo
schermo non succede niente. È il primo criterio di progetto: **oltre una certa
soglia il TPOT smette di essere percepibile, il TTFT no**.

## Prefill e decode sono due mestieri diversi

Fin qui la scomposizione sembra contabile. Non lo è: le due metriche misurano
fasi che stressano la GPU in modo opposto, ed è questa la chiave di tutto il
resto.

`````{tab} Elementare

Immagina una fotocopiatrice industriale che, per stampare anche una sola
pagina, deve prima scaldarsi per un minuto. Con duemila pagine da copiare quel
minuto si spalma su duemila fogli e non lo noti; con una pagina sola aspetti un
minuto per un foglio, e la macchina passa quasi tutto il tempo a scaldarsi.

Il prefill è il primo caso: il modello legge tutte le parole del prompt in una
volta, quindi «scaldare la macchina» (portare i miliardi di numeri del modello
dalla memoria ai circuiti di calcolo) è ripagato da un mucchio di lavoro utile.
Il decode è il secondo: per una parola sola bisogna rileggere tutto il modello,
e i circuiti restano quasi fermi ad aspettare. Sono due lavori che non
convivono bene sulla stessa macchina nello stesso momento: se, mentre venti
persone ricevono la risposta parola per parola, arriva qualcuno con un prompt
lunghissimo, la fotocopiatrice si dedica a quello e gli altri vedono il testo
bloccarsi a metà frase. Un singhiozzo.

`````

`````{tab} Superiore

Riprendiamo l'**intensità aritmetica** del modello roofline, vista nel capitolo
sulla GPU: quanti FLOP si eseguono per ogni byte letto dalla memoria. Per un
modello da $N_p$ parametri il costo di una passata in avanti è circa $2 N_p$
FLOP **per token** elaborato, mentre i pesi in 16 bit occupano $2 N_p$ byte e
vanno letti una volta sola per passata. Se una passata elabora $n_{\text{tok}}$
token insieme:

$$
I \approx \frac{2 N_p \, n_{\text{tok}}}{2 N_p} = n_{\text{tok}}
\quad \text{FLOP/byte},
$$

dove $I$ è l'intensità aritmetica e $n_{\text{tok}}$ il numero di token che
viaggiano nella stessa passata (trascurando KV cache e attenzione, che spostano
il conto ma non la conclusione). Si noti che $n_{\text{tok}}$ conta i *token*,
non le richieste: è la stessa quantità in prefill e in decode, ma la si riempie
in due modi diversi. Le due fasi cadono così ai due lati del ginocchio del
roofline, che sulle schede da datacenter sta fra qualche decina e qualche
centinaio di FLOP/byte:

- **prefill**: un prompt di 2.048 token dà $I \approx 2048$ FLOP/byte, ben oltre
  il ginocchio. È **compute-bound**, e il tempo cresce all'incirca linearmente
  con la lunghezza del prompt (finché il termine quadratico dell'attenzione
  resta minoritario rispetto a quello lineare degli strati densi): ecco perché
  il TTFT è dominato da quella.
- **decode**: una sequenza sola dà $I \approx 1$ FLOP/byte, profondamente
  **memory-bound** come stabilito nella sezione precedente. Il batching serve
  proprio a spostare $I$ verso destra: 64 sequenze insieme, un token ciascuna,
  portano l'intensità a circa 64 FLOP/byte.

Quando le due fasi condividono la GPU nella stessa iterazione dello scheduler,
la lunga si mangia la corta. Un prefill da 8.000 token può occupare la scheda
per centinaia di millisecondi, e ogni sequenza in decode aspetta quel tempo
prima del token successivo: **head-of-line blocking** classico, con la coda
dell'ITL che si allunga e la p99 del TPOT che peggiora mentre la media resta
accettabile {cite}`agrawal2024taming`.

`````

Contro questo scontro si sono affermati due rimedi, che risolvono lo stesso
problema con filosofie opposte: uno fa convivere meglio le due fasi, l'altro le
separa.

`````{tab} Elementare

Il primo rimedio è quello della cassa del supermercato. Se arriva un cliente col
carrello pieno, la cassiera non gli passa tutta la spesa in un colpo lasciando
in attesa chi ha in mano solo il pane: gli passa una decina di articoli, poi
serve chi ha il pane, poi altri dieci, e così via. Il carrello finisce un po'
più tardi, ma nessuno resta fermo a lungo. Applicato ai modelli si chiama
**chunked prefill**: il prompt lungo viene spezzato in pezzi, e fra un pezzo e
l'altro si infilano i passi di generazione di tutti gli altri.

Il secondo è più radicale: **due reparti separati**. Un gruppo di macchine legge
solo i prompt, un altro genera solo le risposte, ciascuno organizzato per il
proprio mestiere. È la **disaggregazione**, e il prezzo è che gli appunti presi
leggendo (la KV cache) vanno trasferiti dal primo reparto al secondo, il che
costa tempo e cavi veloci.

`````

`````{tab} Superiore

Il **chunked prefill** {cite}`agrawal2024taming` sostituisce lo scheduling per
richiesta con uno scheduling a **budget di token per iterazione**: un prefill di
$P$ token è spezzato in $\lceil P/c \rceil$ pezzi di dimensione $c$, e a ogni
iterazione lo scheduler compone un batch con un pezzo di prefill più tutte le
sequenze in decode pronte. L'idea nasce in Sarathi, che accosta i *chunked
prefill* a decodifiche «a rimorchio» (*piggybacked*); Sarathi-Serve battezza
*stall-free batching* lo scheduling che ne risulta, quello che non sospende mai
le generazioni in corso. Il guadagno è doppio: il decode non si ferma mai per
più del tempo di un pezzo, e il pezzo di prefill riempie di lavoro
compute-bound un'iterazione che sarebbe stata memory-bound.

Il parametro $c$ è un compromesso esplicito fra le due metriche. Con $c$ grande
il prefill finisce prima (TTFT più basso) ma il singhiozzo si allunga (TPOT
peggiore); con $c$ piccolo vale il contrario, e in più il pezzo $k$-esimo deve
rileggere la KV cache dei pezzi $1, \dots, k-1$, quindi spezzare troppo fa
ricomparire il traffico di memoria che il prefill evitava. Un ordine di
grandezza: 8.000 token in pezzi da 512 danno
$\lceil 8000/512 \rceil = 16$ iterazioni, e uno stallo massimo che dura quanto
un pezzo invece che quanto l'intero prompt, cioè quasi sedici volte meno.

La **disaggregazione** {cite}`zhong2024distserve` prende la strada opposta:
istanze distinte per prefill e decode, ciascuna dimensionata e parallelizzata
per il proprio collo di bottiglia (il prefill vuole calcolo e batch di token, il
decode vuole banda e batch di sequenze). Nessuna interferisce con l'altra, e i
due obiettivi di servizio si regolano in modo indipendente. Il costo è il
**trasferimento della KV cache** fra i due nodi, proporzionale alla lunghezza
del prompt: su interconnessioni veloci (NVLink, InfiniBand) resta una frazione
del tempo di prefill, su reti lente diventa il nuovo collo di bottiglia. E
servono abbastanza richieste per tenere pieni due gruppi di GPU: sotto una certa
scala, due reparti mezzi vuoti costano più di uno pieno.

`````

## Il goodput, ovvero contare solo ciò che è servito bene

Con queste misure in mano si può dire perché il **throughput** da solo inganna:
conta le richieste servite in un secondo, e non chiede *come* siano state
servite. Il termine che ripara il difetto lo prendiamo in prestito dalle reti,
dove **goodput** indica da sempre la parte di traffico che serve davvero a
chi lo aspetta: non tutto ciò che passa nel cavo, ma solo quello che arriva a
destinazione ed è utile.

`````{tab} Elementare

Un ristorante che stipa duecento coperti a sera, ma dove metà dei clienti
aspetta il primo piatto quaranta minuti e se ne va prima del dolce, non sta
servendo duecento coperti: ne serve cento e ne scontenta altrettanti. Se il
proprietario guarda solo il numero dei coperti, il conto gli torna e continuerà
a stipare.

Il **throughput** è il numero dei coperti: quante richieste il sistema ha
sfornato in un secondo. Il **goodput** è il numero dei clienti serviti *bene*:
si contano solo le richieste che hanno rispettato le promesse fatte, per esempio
«il primo token entro mezzo secondo e gli altri a non più di 50 millisecondi
l'uno dall'altro». Il conto quindi è una moltiplicazione: si prendono le
richieste servite in un secondo e si tiene la frazione che ha rispettato tutte
e due le promesse. Se ne servi venti al secondo e solo l'$85\%$ è a posto, il
goodput è $20 \times 0{,}85 = 17$: ne hai servite venti e ne hai contentate
diciassette.

La differenza non è filosofica: allargare il batch (servire
più richieste nello stesso mazzo) fa quasi sempre salire il throughput,
perché la GPU lavora su più cose insieme. E, una volta che il sistema sta già
dietro alle richieste che arrivano (la condizione posta in «Servire un
modello»), allunga l'attesa di ciascuno, finché comincia a sfondare gli
obiettivi. Il throughput sale mentre il goodput crolla: si servono più persone,
e se ne accontentano meno.

`````

`````{tab} Superiore

Siano $\tau_{\text{f}}$ e $\tau_{\text{p}}$ le soglie dichiarate per TTFT e
TPOT, e $R$ le richieste completate in una finestra di durata $T$. Il goodput è

$$
G = \frac{1}{T}\sum_{i=1}^{R}
\mathbb{1}\!\left[\text{TTFT}_i \le \tau_{\text{f}}
\ \wedge\ \text{TPOT}_i \le \tau_{\text{p}}\right],
$$

dove $\mathbb{1}[\cdot]$ vale $1$ se la richiesta $i$ rispetta **entrambe** le
soglie e $0$ altrimenti. Costruito così, però, il goodput eredita il difetto
del TPOT, che è una media: la richiesta con duecento intervalli da 20 ms e uno
da due secondi ha $\text{TPOT} = 29{,}85$ ms, quindi passa una soglia
$\tau_{\text{p}} = 50$ ms e viene contata fra quelle servite bene, benché
l'utente l'abbia vista bloccarsi a metà frase. La stessa costruzione con
$\max_i \text{ITL}$ al posto del TPOT dà un goodput più severo e più aderente
all'esperienza, ed è quello che si sorveglia quando la fluidità conta. Il
throughput è la stessa somma senza l'indicatore,
$R/T$: il goodput è dunque il throughput moltiplicato per la frazione conforme,
e non può mai superarlo. Nella pianificazione della capacità se ne usa la
variante duale, quella con cui il termine si è diffuso nella letteratura sul
serving degli LLM {cite}`zhong2024distserve`: il **massimo tasso di richieste al
secondo per GPU** che mantiene la conformità sopra una quota fissata (per
esempio il $90\%$). Definito così è la metrica su cui si dimensiona il servizio,
perché tiene insieme il costo (le GPU) e la promessa (le soglie).

Due avvertenze. Il goodput dipende dalle soglie, quindi non è confrontabile fra
sistemi che ne dichiarano di diverse: è un numero interno, non un vanto da
comunicato. E il throughput misurato in **token al secondo** inganna più di
quello in richieste al secondo, perché somma i token di prefill a quelli di
decode: un carico di prompt lunghi e risposte corte produce un numero
spettacolare senza che nessun utente veda il testo scorrere più in fretta.

`````

Vale la pena vederlo su due configurazioni dello stesso sistema, con le
promesse fissate a 500 ms sul TTFT e 50 ms sul TPOT.

La prima serve mazzi da 16 richieste: ne smaltisce 20,0 al secondo e ne tiene
il 92,5% dentro entrambe le promesse, quindi il goodput è
$20{,}0 \times 0{,}925 = 18{,}5$. La seconda allarga il mazzo a 64: le
richieste servite salgono a 32,0 al secondo, il $60\%$ in più, ma la quota di
quelle a posto crolla al 49,4% e il goodput scende a
$32{,}0 \times 0{,}494 = 15{,}8$, quasi il $15\%$ in meno di prima. Il numero
che si guarda per abitudine dice che la seconda configurazione è migliore;
quello che conta dice il contrario, e ha ragione lui.

Un'avvertenza sulla provenienza di queste cifre: escono dalla simulazione di
poche righe più avanti, non da una misura su un sistema vero. Servono a
mostrare *che* throughput e goodput possono muoversi in direzioni opposte, non
a dire di quanto succeda su un modello particolare. Nella simulazione, per
giunta, allargare il mazzo peggiora l'attesa di tutti della stessa quantità,
mentre in un sistema vero peggiora molto di più chi ha la sfortuna di accodarsi
in fondo.

C'è poi una grandezza che si misura per richiesta e non è un tempo: quanti
token quella richiesta consuma. Vale la pena guardarla qui, perché è la stessa
di cui parla tutta questa sezione, vista dal lato del conto invece che da
quello dell'orologio.

```{figure} ../figures/costo-per-forma-di-richiesta.svg
:name: fig-costo-per-caso-uso
:alt: "Quattro barre orizzontali in scala logaritmica che confrontano quanti token consuma un'operazione a seconda della forma della richiesta: una quarantina per classificare una frase, circa quattromilatrecento per una domanda su documenti allegati, circa novemila per una conversazione di otto turni, circa sessantatremila per un report tratto da un dossier lungo. Le tacche verticali segnano cento, mille, diecimila e centomila token."
:width: 96%

Stesso modello, consumi incomparabili. La conversazione è la riga da guardare.
Un modello non ha memoria fra un turno e l'altro: per rispondere gli si rimanda
ogni volta tutto quello che ci si è detti fin lì. Se ogni turno aggiunge 250
token in tutto (la domanda più la risposta), all'ottavo gliene sono passati
$250 \times (1 + 2 + \dots + 8)$, cioè novemila. È il conto del caso peggiore,
quello in cui il modello rilegge tutto da capo ogni volta; l'ultima sezione di
questa pagina mostra come si evita. E attenzione a leggere le barre: **una
tacca in più non vuol dire un
po' di più, vuol dire dieci volte tanto** (è la scala logaritmica, l'unico modo
di far stare quaranta e sessantatremila nello stesso disegno).
```

Il divario di {numref}`fig-costo-per-caso-uso` dice una cosa sola, e non
riguarda i listini: quello che fa il costo è **quanti token servono**, cioè
quanto testo entra e quanto ne esce, e quello lo decide la forma della
richiesta. Classificare una frase manda poche parole e ne riceve una;
riassumere un documento lungo ne manda migliaia; una conversazione le rimanda
tutte a ogni turno. E i token che si pagano sono esattamente gli stessi che
occupano lo spazio che il modello ha per leggere, riempiono gli appunti della
KV cache e allungano l'attesa: chi ne fa risparmiare uno risparmia insieme
denaro, memoria e tempo. È la leva su cui si chiude questa pagina.

## Le medie mentono, e qui in tre modi

I **percentili** li abbiamo imparati in «Servire un modello»: la p95 è il tempo
entro cui è servito il 95% delle richieste, la p99 quello entro cui ne è
servito il 99%, e la promessa si scrive su quelli e non sulla media. Manca
solo un nome, che da qui in poi torna in ogni paragrafo: il gruppetto di
richieste sfortunate che resta *oltre* il percentile si chiama la **coda**.
Attenzione, non è la coda nel senso di fila: è la coda della cometa, cioè la
striscia di ritardatarie che si allunga dietro a tutte le altre. Sono poche,
sono molto più lente, e sono quelle che fanno arrabbiare le persone.

Quando il modello genera, quella regola vale doppio, per tre ragioni sue.

**La prima**: i percentili vanno riportati **separati per ciascuna delle due
attese**, non su quella totale. La coda del TTFT e quella del TPOT si allungano
per cause diverse (la prima per i prompt lunghi e per la fila all'ingresso, la
seconda per i mazzi troppo grandi e per le letture di prompt che si infilano fra
un token e l'altro), e un numero solo le mescola e non dice a nessuno dove
mettere le mani.

**La seconda**: quando una risposta è fatta di più pezzi, le code dei pezzi si
combinano fra loro, e *come* si combinano dipende da com'è fatto il sistema. Il
caso che morde è quello in cui una risposta aspetta molte
chiamate lanciate **tutte insieme**: venti pezzi di documento da andare a
recuperare in venti archivi diversi, oppure venti programmi esterni a cui il
modello chiede una cosa ciascuno (che ora, un cambio, un prezzo) prima di
poter rispondere. Lì non si aspetta la media, si
aspetta **la più lenta di tutte**, e basta che una sia finita nella coda perché
l'intera risposta ci finisca. Se ciascuna ha l'$1\%$ di probabilità di essere
lenta, la probabilità che almeno una delle venti lo sia è
$1 - 0{,}99^{20} \approx 18\%$: si calcola la probabilità che vadano bene tutte
e venti ($0{,}99$ moltiplicato per sé stesso venti volte, cioè circa l'$82\%$) e
la si toglie da uno. Una p99 rassicurante sul singolo passo diventa un utente
scontento su cinque sull'intera interazione, ed è l'argomento di Dean e Barroso
{cite}`dean2013tail`.

Nel caso opposto l'effetto si rovescia, e conviene saperlo per non applicare il
conto dove non vale. Un agente che fa venti chiamate **una dopo l'altra** non
aspetta la più lenta, le somma, e sommando la sfortuna si diluisce. Venti passi
da un secondo fanno venti secondi; se uno va male e ne impiega tre, il totale
diventa ventidue, cioè il $10\%$ in più, non il $200\%$ che quel passo ha
subìto per conto suo. Lì il problema non è la coda: è il **totale**, venti volte
più grande, che sfonda la promessa da solo.

**La terza** si vede nel confronto fra le due configurazioni di poco fa, e sono
i numeri della tabella che il codice in fondo alla pagina stampa. Con mazzi da
64 il TTFT **medio** è 457 ms, cioè dentro l'obiettivo di 500 ms: guardando
quello, il sistema mantiene la promessa. Ma la p95 passa da 486 a 729 ms, e lì
la riga dei 500 viene attraversata di netto. E la tabella ha una colonna che
conta proprio le sforate: a mazzi da 64 sfora il mezzo secondo il **33,4%**
delle richieste, cioè **una su tre**, contro il $3{,}5\%$ dei mazzi da 16. Chi
riportasse la media lo farebbe in buona fede, e sarebbe smentito da un terzo
dei suoi utenti. È il modo più comune in cui un cruscotto tutto verde copre un
servizio in rosso.

Vale poi, a maggior ragione, il caso che qui non si vede e che in produzione
capita: **una media che migliora mentre la p95 o la p99 peggiorano è un
peggioramento**, da trattare come un guasto. È il primo dei tre quadranti del
cruscotto di
«Sorvegliare un modello vivo» (quello che dice se il servizio è vivo e risponde
in fretta, prima ancora di chiedersi se risponde *bene*), declinato sulle due
attese che la generazione ha invece di una.

## La leva che resta: riusare il prefisso

Scelto come si servono le richieste e come si alternano lettura e scrittura,
quale leva resta per far comparire prima la prima parola? Una soprattutto, e
non riguarda il modello ma il traffico: nei sistemi reali le richieste **non
sono indipendenti fra loro**, cominciano quasi tutte allo stesso modo.

`````{tab} Elementare

Pensa a uno studio notarile dove ogni atto comincia con le stesse quattro pagine
di premesse, e solo dalla quinta si parla del caso. Un copista che ricopiasse
ogni atto da capo riscriverebbe quelle pagine centinaia di volte: basta tenerne
una copia pronta e ricopiare solo il seguito.

Nei sistemi che servono modelli generativi succede esattamente questo, e tre
casi coprono quasi tutto il traffico. L'istruzione di sistema (le righe che
spiegano al modello come comportarsi) è identica per tutti gli utenti. Un
documento allegato su cui si fanno dieci domande è lo stesso dieci volte. E
soprattutto una conversazione: al decimo turno il prompt è tutta la
conversazione più l'ultima domanda, e i primi nove turni li abbiamo già letti
nove volte.

Gli appunti che il modello prende su un pezzo di testo dipendono solo da quel
pezzo e da ciò che lo precede: se l'inizio è identico, gli appunti sull'inizio
sono identici. I conti della conversazione lo dicono meglio di ogni argomento.
Riprendiamo i 250 token per turno della figura di poco fa (la domanda più la
risposta), così che al primo turno il modello ne legga 250, al secondo 500, al
terzo 750 e via salendo. In dieci turni, rileggere tutto ogni volta costa
$250 \times (1+2+\dots+10) = 13\,750$ token di lettura; riusare gli appunti
significa leggerne 250 per turno, cioè $250 \times 10 = 2\,500$ in tutto,
cinque volte e mezzo di meno. Sono i novemila token della figura di poco fa,
visti dall'altra parte: quel numero era il conto senza riuso, e il riuso lo
taglia di altrettanto. (Il risparmio è di **lavoro**, cioè di tempo e di
memoria; quanto di quel lavoro risparmiato finisca poi sulla fattura dipende da
chi vende il servizio, e non è materia di questa pagina.) E il risparmio cresce
con la lunghezza della
conversazione, perché la somma $1+2+\dots+n$ vale $n(n+1)/2$: il rapporto fra
le due letture è quindi $(n+1)/2$, che non dipende da quanto pesa un turno e
che a dieci turni fa cinque e mezzo, a venti dieci e mezzo. Più lunga è la
conversazione, più conviene.

`````

`````{tab} Superiore

Nell'attenzione causale la coppia $(\mathbf{k}, \mathbf{v})$ della posizione $j$ dipende solo dai
token $1, \dots, j$. Due richieste che condividono un prefisso hanno quindi, per
quelle posizioni, una KV cache **bit a bit identica**, a parità di pesi,
precisione, eventuale adattatore LoRA e codifica posizionale. La chiave del
riuso è la sequenza esatta di **identificativi di token**, non la stringa.

Il meccanismo di riferimento è la **RadixAttention** di SGLang
{cite}`zheng2024sglang`: la cache non è una tabella piatta ma un **albero dei
prefissi** compresso (un radix tree) i cui archi sono sequenze di token e i cui
nodi puntano ai blocchi di KV cache. Una richiesta nuova cammina sull'albero
finché i token coincidono, riusa i blocchi trovati e calcola il prefill solo per
la coda non trovata; il ramo nuovo si innesta e resta disponibile per le
richieste successive. Lo sfratto è a politica LRU con un conteggio dei
riferimenti, così che i blocchi in uso non vengano rimossi, e lo scheduler può
ordinare la coda per affinità di prefisso, in modo da massimizzare i colpi a
cache prima che i rami vengano sfrattati.

Il legame con la sezione precedente è diretto: la condivisione è possibile
*perché* la KV cache è già paginata in blocchi di taglia fissa con una block
table {cite}`kwon2023efficient`. Condividere significa far puntare due block
table allo stesso blocco fisico e incrementare un contatore, la stessa idea di
*copy-on-write* dei sistemi operativi; RadixAttention aggiunge l'indice che
rende la condivisione sistematica **fra richieste diverse nel tempo**, non solo
fra sequenze compresenti nello stesso batch. L'effetto sul TTFT è quasi
proporzionale alla frazione di prompt trovata in cache, e in una conversazione
di $n$ turni da $m$ token ciascuno (il prompt del turno $k$ è lungo $k\,m$) il
prefill totale (dove $n$ è il numero di turni e $m$ i token che ciascuno
aggiunge) scende da $m\,n(n+1)/2$ a $m\,n$: da quadratico a lineare nei turni,
con un risparmio di un fattore $(n+1)/2$.

`````

Una cautela va aggiunta, perché riguarda la sicurezza e non le prestazioni. Se
gli appunti si riusano fra clienti diversi, quel magazzino diventa una **stanza
in comune**, e il tempo di risposta si trasforma in una spia. Chiunque può
provare a scrivere un testo e cronometrare: se la prima parola arriva
stranamente in fretta, vuol dire che gli appunti su quel testo c'erano già,
cioè che *qualcun altro* lo aveva mandato prima. Si scopre così un pezzo di
quello che stanno chiedendo gli altri, senza vedere niente di loro. È un canale
laterale a base di
tempo, della stessa famiglia degli attacchi che il capitolo sull'AI
responsabile affronterà parlando di privacy. Le difese sono di progetto, non di
taratura: si partiziona la cache per cliente, e si condividono solo i prefissi
dichiaratamente pubblici, tipicamente l'istruzione di sistema del prodotto.

Stessa disciplina per la correttezza. Gli appunti tenuti da parte valgono solo
per chi li ha scritti, e dipendono da tre cose: da quale modello li ha
calcolati, da quali eventuali pesi aggiuntivi lo stessero specializzando (la
LoRA vista nel capitolo sui Transformer) e da con quante cifre i suoi numeri
erano scritti. Se il magazzino non tiene conto di tutte e tre, consegna a un
modello gli appunti presi da un altro, e nessuno se ne accorge.

## Misurare in venti righe

Il codice qui sotto prende, per ogni richiesta arrivata in dieci secondi, il suo
TTFT e il suo TPOT, e ne ricava throughput, goodput e percentili. I due tempi
non sono misurati ma estratti a sorte, e la forma con cui si sorteggia non è
scelta a caso: dev'essere quella che i tempi di risposta hanno davvero, cioè
tantissime richieste ammassate attorno a un valore tipico e poche, sempre più
rare, che si allontanano verso i tempi lunghi. È esattamente la coda di cui
parla tutta questa pagina, e ha anche la proprietà che serve, cioè che non
esce mai un tempo negativo. Quella forma ha un nome, **lognormale**, e nel
codice è la riga `rng.lognormal`.

```python
import numpy as np

rng = np.random.default_rng(0)

SLO_TTFT, SLO_TPOT = 0.500, 0.050   # obiettivi dichiarati: 500 ms e 50 ms
FINESTRA = 10.0                     # secondi di traffico osservato

def misura(nome, n, ttft_mediano, tpot_mediano, sigma=0.35):
    """Simula n richieste servite nella finestra e ne riassume le metriche."""
    ttft = rng.lognormal(np.log(ttft_mediano), sigma, n)  # code lunghe a destra
    tpot = rng.lognormal(np.log(tpot_mediano), sigma, n)
    ok = (ttft <= SLO_TTFT) & (tpot <= SLO_TPOT)          # rispetta ENTRAMBE
    sfora = (ttft > SLO_TTFT).mean()                      # sfora SOLO il TTFT
    p50, p95, p99 = np.percentile(ttft, [50, 95, 99]) * 1000
    print(f"{nome:<9}{n / FINESTRA:8.1f}{ok.sum() / FINESTRA:9.1f}{ok.mean():10.1%}"
          f"{sfora:9.1%}{ttft.mean() * 1000:9.0f}{p50:7.0f}{p95:7.0f}{p99:7.0f}")

print(f"{'config':<9}{'ric/s':>8}{'good/s':>9}{'conformi':>10}{'TTFT>SLO':>9}"
      f"{'TTFTmed':>9}{'p50':>7}{'p95':>7}{'p99':>7}")
misura("batch 16", 200, 0.28, 0.028)
misura("batch 64", 320, 0.43, 0.043)
```

L'output è la tabella dell'esempio di poco fa:

```text
config      ric/s   good/s  conformi TTFT>SLO  TTFTmed    p50    p95    p99
batch 16     20.0     18.5     92.5%     3.5%      297    285    486    530
batch 64     32.0     15.8     49.4%    33.4%      457    429    729    893
```

Le due colonne di percentuali contano cose diverse, e vale la pena tenerle
separate. `conformi` è la quota di richieste che rispettano **tutte e due** le
promesse, quella sul primo token e quella sul ritmo; `TTFT>SLO` è la quota che
sfora **solo la prima**. Per questo a mazzi da 64 si legge sia «una su tre
sfora il mezzo secondo» sia «una su due non è a posto»: sono due bocciature
diverse, e la seconda comprende la prima.

Sono le due righe già commentate. Le stesse venti righe, girate su misure vere
invece che su numeri sorteggiati, e ripetute a ogni finestra di dieci secondi,
sono lo scheletro di un cruscotto: throughput, goodput e percentili sono tutto
quello che serve per sapere se un servizio che genera testo sta funzionando.

## Che cosa vuol dire funzionare

Queste metriche non sono contabilità da presentare a fine mese: sono la
**definizione operativa** di cosa vuol dire, per questo servizio, funzionare.
Sceglierle equivale a decidere quali richieste contano e quali no, e ogni
ottimizzazione successiva si muoverà nella direzione che quella scelta indica.
Un sistema tarato sul throughput diventerà bravissimo a servire molte richieste
male; uno tarato sul TTFT medio diventerà bravissimo a nascondere la coda lenta
a chi guarda i grafici. È lo stesso avvertimento già dato sulle metriche di un
classificatore e sui surrogati del giudizio umano: ottimizzare contro una misura
sbagliata non produce un fallimento rumoroso, produce un successo apparente.

`````{tab} Elementare
```{admonition} Da ricordare
:class: important
- Per un modello che genera testo la velocità **non è un numero solo**, come al
  menù degustazione: c'è l'attesa della prima parola (**TTFT**), che dipende da
  quanto è lungo il prompt da leggere, e la pausa fra una parola e la
  successiva (**TPOT**), cioè il ritmo con cui il testo scorre. Due sistemi che
  finiscono nello stesso istante possono essere l'uno piacevole e l'altro
  irritante.
- **Leggere il prompt e generare le parole sono due lavori opposti**: leggere
  tiene la macchina piena di lavoro utile, generare la costringe a scaldarsi
  ogni volta per un foglio solo. Sulla stessa scheda l'uno blocca l'altro, e i
  rimedi sono due: spezzare il prompt lungo in pezzi e infilare fra un pezzo e
  l'altro le parole di tutti gli altri (la cassa che alterna il carrello e chi
  ha in mano solo il pane), oppure separare i reparti, al prezzo di trasferire
  gli appunti presi leggendo.
- Il **goodput** conta solo le richieste servite **entro le promesse
  dichiarate**. Servirne di più in una volta alza il numero dei coperti e
  abbassa quello dei clienti contenti: nell'esempio, $+60\%$ di richieste
  servite e $-15\%$ di richieste servite *bene*.
- **Le medie mentono**: si guardano i percentili alti (p95 e p99, cioè il caso
  peggiore su venti e su cento), riportati separatamente per ciascuna delle due
  attese. E quando una risposta aspetta venti richieste **lanciate insieme**, si
  aspetta la più lenta: se una su cento è lenta, la probabilità che almeno una
  delle venti lo sia arriva al $18\%$. Una media che migliora mentre il caso
  peggiore peggiora è un peggioramento.
- La leva che resta è **riusare l'inizio**: istruzione di sistema, documento
  allegato e cronologia della conversazione si ripetono identici a ogni
  richiesta, come le pagine di premesse dello studio notarile. Tenerne gli
  appunti già pronti e ricopiare solo il seguito accorcia l'attesa della prima
  parola, e più la conversazione è lunga più conviene.
- Quegli appunti però sono **condivisi fra utenti diversi**: una risposta
  anormalmente rapida rivela che qualcun altro aveva già inviato quel testo. Si
  tengono separati per cliente e si condivide solo ciò che è dichiaratamente
  pubblico.
```
`````

`````{tab} Superiore
```{admonition} Da ricordare
:class: important
- Per un modello che genera, la latenza non è un numero solo: si scompone in
  **TTFT** (attesa del primo token, dominata dal **prefill** e quindi dalla
  lunghezza del prompt) e **TPOT** o **ITL** (pausa fra token successivi, la
  fase **decode** memory-bound), e si ricompone come
  $T = \text{TTFT} + (N_{\text{out}} - 1)\,\text{TPOT}$. La velocità percepita è
  $1/\text{TPOT}$.
- **Prefill e decode sono mestieri opposti**: l'intensità aritmetica è pari al
  numero di token elaborati insieme, quindi il prefill è compute-bound e il
  decode memory-bound. Sulla stessa GPU l'uno blocca l'altro; i rimedi sono il
  **chunked prefill** {cite}`agrawal2024taming`, che spezza il prompt e lo
  intercala ai passi di decode, e la **disaggregazione**
  {cite}`zhong2024distserve`, che li manda su GPU diverse al prezzo di
  trasferire la KV cache.
- Il **goodput** conta solo le richieste servite **entro gli obiettivi
  dichiarati**: allargare il batch alza il throughput e può abbassare il goodput
  (nell'esempio $+60\%$ di richieste servite e $-15\%$ di richieste servite
  *bene*). È la misura che rende visibile il compromesso fra throughput e
  latenza.
- **Le medie mentono**: p50, p95 e p99 vanno riportati per ciascuna metrica. Le
  code si compongono nel **fan-out**, dove si aspetta la più lenta di $n$
  chiamate parallele ($1 - 0{,}99^{20} \approx 18\%$ con venti)
  {cite}`dean2013tail`; in una **catena sequenziale** invece si sommano e la
  coda relativa si stringe, ma sfonda lo SLO il budget totale. Una media che
  migliora mentre la p99 peggiora è una regressione.
- Il **riuso del prefisso** è la leva che resta: istruzione di sistema,
  documenti allegati e cronologia di conversazione rendono identica una parte
  della KV cache. Un **albero dei prefissi** la condivide
  fra richieste diverse {cite}`zheng2024sglang` (possibile perché la cache è già paginata in blocchi
  {cite}`kwon2023efficient`) e in una conversazione porta il prefill totale da
  quadratico a lineare nei turni.
- La cache condivisa è però una **superficie fra utenti**: un TTFT anormalmente
  basso rivela che quel prefisso era già stato inviato da qualcuno. Si partiziona
  per cliente e si condividono solo i prefissi pubblici.
```
`````
