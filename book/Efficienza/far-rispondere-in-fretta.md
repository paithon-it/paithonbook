# Starci non è rispondere: la mappa dell’altra metà

Le tre leve del capitolo hanno stretto il modello. Un modello stretto ci sta in
memoria, e questo risolve il primo problema. «Ci sta» e «risponde in fretta»
sono però due domande diverse, e la seconda non si risolve del tutto
rimpicciolendo.

Tre risposte importanti alla seconda domanda non toccano il modello, e stanno
nella {doc}`sezione sui grandi modelli linguistici </Transformers/llm>` e in
quella su {doc}`LLMOps </MLOps/llmops>`, che tratta di come mettere in servizio
quei modelli. Prima vediamo perché le due domande sono diverse.

## Perché rispondere è un problema di traffico

La risposta sta in un calcolo che non misura niente e non dipende da nessuna
macchina: è aritmetica sul lavoro da fare.

`````{tab} Elementare

Hai la cantina piena di casse e la cucina al piano di sopra. Per apparecchiare
devi scendere, caricarti una cassa e risalire. La cassa pesa uguale
comunque, e le scale sono sempre quelle: il viaggio costa lo stesso che tu
serva una persona o venti.

Se apparecchi per venti, scendi una volta e la fatica si spalma su venti
coperti. Se apparecchi per uno, scendi una volta lo stesso, e quella fatica se
la prende un coperto solo. Se ti tocca scendere una volta per ogni coperto,
passi la giornata sulle scale e in cucina non fai niente.

A decidere se il lavoro si blocca sulle scale o in cucina è un paragone fra due
numeri: quanti coperti riesci ad apparecchiare nel tempo di un viaggio in
cantina, e quanti coperti servi con il carico che porti su in un viaggio.

Ad esempio, se nel tempo di un viaggio ne apparecchieresti cento e con il carico
ne servi uno, comandano le scale. Se invece con il carico ne servi duecento,
mentre apparecchi c’è tutto il tempo per il viaggio dopo, e a comandare tornano
le mani. Quel confine dipende dalla casa, e con scale più corte, o mani più
lente, si sposta.

In un computer succede esattamente questo. I pesi del modello stanno «in
cantina», cioè nella memoria, e per fare con essi un calcolo qualunque bisogna
portarli su; i coperti sono gli elementi che si elaborano insieme con gli stessi
pesi. I pesi sono tanti (in un modello vero, gigabyte), e il viaggio costa
uguale che li si usi per un elemento solo o per duecento insieme.

Le casse però non pesano per forza così. Scrivere ogni peso con metà delle
cifre è una cantina in cui ogni cassa pesa la metà. In braccio ne stanno due, i
viaggi si dimezzano, e gli stessi coperti escono da metà scale, quindi il
rapporto fra coperti e viaggi raddoppia. Ecco perché togliere bit ai pesi non
serve soltanto a farli entrare in cantina. Finché a tenerti fermo sono le
scale, fa anche arrivare la cena prima.

Dalla cantina, poi, non salgono solo le casse. Ogni coperto ha la sua roba
(piatto, bicchiere, posate), e quella cresce col numero dei coperti. Per un
coperto o quattro non si sente. Per duecentocinquantasei aggiunge un ottavo al
carico, e da ogni viaggio escono meno coperti di quanti il calcolo pulito ne
prometta. Il calcolo tiene finché i coperti sono pochi rispetto a quanto pesano
le casse.

Quindi la domanda decisiva è quante operazioni si riescono a fare per ogni
viaggio in cantina, più che quante ce ne siano da fare. La tabella dei conti
per byte è quella domanda, messa in numeri.

`````

`````{tab} Superiore

Consideriamo uno strato che moltiplica una matrice di pesi $n \times n$ per un
blocco di ingressi $n \times k$, dove $k$ è il numero di elementi elaborati
insieme. Le operazioni in virgola mobile sono $2 n^2 k$, e il fattore due viene
dal fatto che per ogni casella del risultato si fanno $n$ moltiplicazioni e
altrettante somme. È la convenzione con cui la {doc}`sezione su GEMM e tensor
core </GPU/gemm-e-tensor-core>` conta i FLOP. I byte letti dalla memoria sono
invece $n^2 b / 8$, con $b$ i bit per peso, e non dipendono da $k$, perché i
pesi sono gli stessi per tutti gli elementi.

Il rapporto fra le due quantità,

$$
I = \frac{2 n^2 k}{n^2 b / 8} = \frac{16\,k}{b},
$$

è l’intensità aritmetica, la grandezza in ascissa del modello roofline della
{doc}`sezione sulla gerarchia della memoria </GPU/gerarchia-memoria>`. Il regime
in cui si cade (legati alla banda o legati al calcolo) dipende dal confronto fra
$I$ e il rapporto fra prestazione di picco e banda della macchina. In
particolare, con pesi in sedici bit si semplifica in $I = k$, che è la colonna
di destra della tabella dei conti per byte.

Il calcolo dà due indicazioni. La prima è che $I$ non dipende dalla larghezza
dello strato, ma solo perché si stanno contando i byte dei pesi e non quelli di
ingressi e uscite, che sono $n k b / 8$ ciascuno. L’approssimazione vale quindi
per $k \ll n$, e all’ultima riga della tabella ($k = 256$ contro $n = 4096$)
sbaglia già del dodici e mezzo per cento.

La seconda indicazione, la più utile, è che dimezzare $b$ raddoppia $I$. Di
conseguenza, in regime legato alla banda e finché il traffico è fatto dai soli
pesi, quantizzare non serve solo a far stare il modello in memoria, ma dimezza
al più il tempo di ogni passo. Al più, perché la cache e gli ingressi non si
dimezzano, e riportare i pesi corti alla precisione del calcolo richiede
operazioni in più.

`````

```python
N = 4096                       # la larghezza di uno strato
BIT = 16                       # ogni peso in sedici bit

print(f"{'cose insieme':>13} {'conti':>16} {'byte letti':>12} {'conti per byte':>16}")
for k in (1, 4, 16, 64, 256):
    conti = 2 * N * N * k      # per ogni casella, n moltiplicazioni e n somme
    byte = N * N * BIT / 8     # i pesi si leggono una volta sola
    print(f"{k:>13} {conti/1e6:>11.0f} milioni {byte/1e6:>9.0f} MB "
          f"{conti/byte:>16.1f}")
```

```text
 cose insieme            conti   byte letti   conti per byte
            1          34 milioni        34 MB              1.0
            4         134 milioni        34 MB              4.0
           16         537 milioni        34 MB             16.0
           64        2147 milioni        34 MB             64.0
          256        8590 milioni        34 MB            256.0
```

La colonna dei byte letti non si muove mai: sono sempre gli stessi
trentaquattro megabyte di pesi, da leggere dalla memoria a ogni passo. Quella
dei conti si moltiplica. L’ultima colonna è il rapporto fra le due, cioè quanti
conti si fanno per ogni byte letto, e la {doc}`sezione sulla gerarchia della
memoria </GPU/gerarchia-memoria>` la chiama intensità aritmetica.

Se l’intensità è bassa, il processore resta fermo ad aspettare i dati, e un
processore più veloce non cambia il tempo; se è alta, i dati fanno in tempo ad
arrivare e il processore lavora.

Quando un modello legge una domanda, la legge tutta insieme. Se ad esempio la
domanda è di duecento parole (più precisamente token, i pezzi in cui il modello
divide il testo), le cose insieme sono duecento, vicino all’ultima riga della
tabella. Che quel regime sia legato al calcolo, però, dipende dal ginocchio
della macchina. A sedici bit l’intensità vale il numero di cose insieme, e il
ginocchio sta a 161 conti per byte su una A100 e a 295 su una H100, quindi
servono domande di centinaia di token, non di decine.

Quando invece scrive la risposta, il modello la scrive una parola alla volta,
perché per scegliere la parola dopo deve aver scelto quella prima: le cose
insieme sono una, e il computer sta nella prima riga.

Leggere la domanda e scrivere la risposta sono la stessa moltiplicazione con lo
stesso modello, eppure stanno ai due estremi opposti della tabella. A meno dei
byte degli ingressi che il calcolo trascura, il rapporto fra le due efficienze
è la lunghezza della domanda: con duecento parole, scrivere è quasi duecento
volte meno efficiente che leggere. Non perché l’operazione sia più difficile,
ma perché si paga la stessa lettura di trentaquattro megabyte di pesi per una
parola sola invece che per duecento.

Per questo motivo rimpicciolire il modello aiuta anche il tempo di risposta
(meno pesi da leggere vuol dire meno traffico), ma non basta: finché si scrive
una parola alla volta, si è nella prima riga della tabella qualunque sia la
dimensione del modello.

## Le tre risposte, e dove stanno

Tre idee rispondono alla seconda domanda senza toccare il modello. La seconda e
la terza vengono dalla tabella, perché alzano il numero di cose insieme; la
prima viene da un’altra parte, cioè dal lavoro che altrimenti si rifarebbe.

La prima idea è non rifare due volte lo stesso lavoro. Per scegliere la parola
numero cinquecento, un modello che scrive testo ricalcolerebbe tutte le
quattrocentonovantanove di prima. Di ciascuna di quelle parole gli serve però
soltanto una coppia di vettori per strato (la chiave e il valore), già
calcolata quando la parola è stata scritta. Quindi la si conserva invece di
rifarla, come si tiene un segnalibro per non rileggere il libro da capo a ogni
pagina.

Questo deposito si chiama cache delle chiavi e dei valori. Ne ha già mostrato
il peso la {doc}`sezione su FlashAttention </GPU/flash-attention>`, e la
{doc}`sezione sui grandi modelli linguistici </Transformers/llm>`, nel capitolo
sui Transformer (l’architettura di cui quei modelli sono fatti), spiega come
funziona nella generazione. Non risolve il problema
dell’intensità aritmetica: sposta il traffico dai pesi alla cache, che cresce a
ogni parola scritta.

La seconda idea è riempire la riga, raggruppando le richieste. Se scrivere una
parola per un solo utente sta nella prima riga, scriverla per
duecentocinquantasei utenti insieme sta nell’ultima: i pesi si leggono una
volta e servono a tutti. È il motivo per cui un servizio che risponde a molti
costa, a testa, molto meno di uno che risponde a uno. La {doc}`sezione su LLMOps
</MLOps/llmops>` lo tratta parlando di come si gestiscono le richieste che
arrivano insieme e la memoria che ciascuna si porta dietro.

La terza idea è indovinare avanti e farsi correggere. Un modello piccolo propone
qualche parola di seguito; il modello grande le verifica tutte in una passata
sola, cioè su tutta la bozza insieme invece che su una parola sola, ne accetta
un tratto, e alla prima che rifiuta ne mette una sua.

In questo modo ogni passata produce almeno una parola e, se la bozza è buona,
parecchie al costo di una. Il tempo del modello piccolo che si butta via è
poco, e il testo che ne esce ha la stessa distribuzione del testo che avrebbe
scritto il solo modello grande. È la **decodifica speculativa**, e sta nella
{doc}`sezione su LLMOps </MLOps/llmops>`, con la figura che mostra la bozza
accettata e il punto in cui il modello grande la taglia.

Non sono le sole risposte. Altre toccano il modello o il modo di calcolarlo:
l’attenzione con meno teste per chiavi e valori rimpicciolisce la cache
({doc}`sezione sull’attenzione in pratica
</Transformers/attenzione-in-pratica>`), la mixture of experts fa attraversare
a ogni parola solo una parte dei parametri ({doc}`sezione sulla mixture of
experts </Transformers/mixture-of-experts>`), FlashAttention riduce i byte che
l’attenzione sposta ({doc}`sezione su FlashAttention </GPU/flash-attention>`).
Le tre leve del capitolo tornano inoltre, applicate al servizio, nella sezione
«Comprimere per servire» di {doc}`LLMOps </MLOps/llmops>`.

`````{tab} Elementare

```{admonition} Da ricordare
:class: important
- Starci in memoria e rispondere in fretta sono due problemi diversi. Le
  tre leve del capitolo risolvono il primo, e aiutano il secondo solo di
  riflesso.
- La ragione è che leggere i pesi dalla memoria costa sempre uguale, che
  li si usi per un elemento solo o per duecento insieme. Un modello che legge
  una domanda di duecento parole le elabora tutte insieme; quando scrive la
  risposta va una parola alla volta, e paga la stessa lettura per una parola
  sola.
- Da qui le tre idee che non toccano il modello: tenersi gli appunti invece di
  rifarli (la cache delle chiavi e dei valori, nel {doc}`capitolo sui
  Transformer </Transformers/overview>`), servire molti insieme perché i pesi
  letti una volta valgono per tutti (nel {doc}`capitolo su MLOps
  </MLOps/overview>`), e far scrivere una bozza a un modello piccolo che il
  grande controlla tutta in una volta (la decodifica speculativa, ancora in
  MLOps).
```

`````

`````{tab} Superiore

```{admonition} Da ricordare
:class: important
- L’intensità aritmetica di uno strato $n \times n$ su un blocco $n \times k$
  vale $16k/b$, cioè $k$ a sedici bit: le operazioni crescono con $k$, i byte
  dei pesi no. Non dipende da $n$ solo finché si trascurano i byte di ingressi
  e uscite, il che vale per $k \ll n$.
- Scrivere una parola alla volta (la generazione autoregressiva) impone $k = 1$
  per passo, quindi è strutturalmente legata alla banda: a sedici bit $I = 1$,
  contro un ginocchio che sugli acceleratori sta nell’ordine delle centinaia
  (161 su A100, 295 su H100). L’elaborazione del testo in ingresso ha $k$ pari
  alla lunghezza della sequenza, e risulta legata al calcolo solo quando $k$
  supera il ginocchio. Sono lo stesso modello in due regimi diversi: per questo
  le due fasi si misurano con due grandezze separate.
- Le tre tecniche a modello invariato abbassano il costo di ogni parola
  scritta, ma da lati diversi del rapporto $I$. Il raggruppamento delle
  richieste e la decodifica speculativa alzano $k$ a parità di byte dei pesi
  letti; la seconda lo fa verificando in parallelo una bozza prodotta da un
  modello più economico, e con la regola di accettazione e rifiuto non cambia
  la distribuzione di uscita. La cache delle chiavi e dei valori fa il
  contrario: toglie le operazioni del ricalcolo e aggiunge i byte della cache,
  che cresce con il contesto, quindi abbassa l'intensità aritmetica del passo
  di decodifica invece di alzarla. Resta vantaggiosa perché per ogni posizione
  del contesto toglie proiezioni dell'ordine di $n^2$ operazioni e aggiunge la
  lettura di due vettori dell'ordine di $n$ numeri, con $n$ la larghezza dello
  strato.
- Nessuna delle tre cambia il modello, e per questo stanno altrove: la cache
  nella {doc}`sezione sui grandi modelli linguistici </Transformers/llm>`; il
  raggruppamento delle richieste e la decodifica speculativa nella
  {doc}`sezione su LLMOps </MLOps/llmops>`; il modo di misurare separatamente
  i due regimi nella {doc}`sezione sulle metriche di servizio
  </MLOps/metriche-di-servizio>`, tutte e due in MLOps. Altre risposte toccano
  il modello o il kernel: l’attenzione con meno teste per chiavi e valori, la
  mixture of experts, FlashAttention.
```

`````

Le tre leve si pagano, e il costo di ciascuna conta più dell’elenco delle
tecniche. Arrotondare a otto bit sposta l’uscita di uno strato dell’uno per
cento, e a quattro molto di più; soltanto la prova sul modello dice quanto
questo costi in accuratezza. Potare novanta pesi su cento costa un punto di
accuratezza e, sulla moltiplicazione densa che quasi tutti eseguono, non regala
un millisecondo. Chi imita un maestro, infine, ne prende anche gli errori, e un
maestro sbagliato fa peggio di nessuno. Il costo cambia da un modello
all'altro, e chi adotta una di queste leve senza valutarne l'impatto sul
modello che usa sta scegliendo alla cieca.

Il calcolo dell’intensità aritmetica riguarda uno strato solo, e un modello
vero ne ha decine, uno dopo l’altro. Perché una rete sia fatta di molti
strati sottili invece che di uno solo largo è la domanda del {doc}`capitolo sul
deep learning </DeepLearning/overview>`, che ci risponde fin dalla sua prima
pagina, sotto il titolo «Profondo, non solo largo».
