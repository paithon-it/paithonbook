# Starci non è rispondere: la mappa dell’altra metà

Le tre leve del capitolo hanno stretto il modello. Un modello stretto ci sta in
memoria, e questo era il problema del conto in apertura. Ma «ci sta» e
«risponde in fretta» sono due domande diverse, e la seconda non si risolve del
tutto rimpicciolendo.

Tre risposte importanti alla seconda domanda non toccano il modello, e stanno
nella {doc}`sezione sui grandi modelli linguistici </Transformers/llm>` e in
quella su {doc}`LLMOps </MLOps/llmops>`, il mestiere di mettere in servizio
quei modelli. Prima serve vedere perché le due domande siano diverse.

## Perché rispondere è un problema di traffico

La risposta sta in un conto che non misura niente e non dipende da nessuna
macchina: è aritmetica sul lavoro che c’è da fare.

`````{tab} Elementare

Hai la cantina piena di casse e la cucina al piano di sopra. Per apparecchiare
devi scendere, caricarti una cassa e risalire. La cassa pesa uguale
comunque, e le scale sono sempre quelle: il viaggio costa lo stesso che tu
serva una persona o venti.

Se apparecchi per venti, scendi una volta e la fatica si spalma su venti
coperti. Se apparecchi per uno, scendi una volta lo stesso, e quella fatica se
la prende un coperto solo. E se ti tocca scendere una volta per ogni coperto,
passi la giornata sulle scale e in cucina non fai niente.

Sulle scale o in cucina, dove ti si ferma la giornata lo decide un paragone fra
due numeri: quanti coperti riesci ad apparecchiare nel tempo di un viaggio in
cantina, e quanti coperti servi con quello che porti su in un viaggio. Se nel
tempo di un viaggio ne apparecchieresti cento e con il carico ne servi uno,
comandano le scale. Se con il carico ne servi duecento, mentre apparecchi c’è
tutto il tempo per il viaggio dopo, e a comandare tornano le mani. Quel confine
dipende dalla casa, e con scale più corte, o mani più lente, si sposta.

Un calcolatore fa esattamente questo. I pesi del modello stanno «in cantina»,
cioè nella memoria, e per farci un conto qualunque bisogna portarli su; i
coperti sono le cose che si elaborano insieme con gli stessi pesi. I pesi sono
tanti (in un modello vero, gigabyte) e il viaggio costa uguale che li si usi
per una cosa sola o per duecento insieme.

Le casse però non pesano per forza così. Scrivere ogni peso con metà delle
cifre è una cantina in cui ogni cassa pesa la metà. In braccio ne stanno due, i
viaggi si dimezzano, e gli stessi coperti escono da metà scale, quindi il
rapporto fra coperti e viaggi raddoppia. Ecco perché togliere bit ai pesi non
serve soltanto a farli entrare in cantina. Finché a tenerti fermo sono le
scale, fa anche arrivare la cena prima.

Dalla cantina, poi, non salgono solo le casse. Ogni coperto ha la sua roba
(piatto, bicchiere, posate), e quella cresce col numero dei coperti. Per un
coperto o quattro non si sente. Per duecentocinquantasei aggiunge un ottavo al
carico, e da ogni viaggio escono meno coperti di quanti il conto pulito ne
prometta. Il conto tiene finché i coperti sono pochi rispetto a quanto pesano
le casse.

Quindi la domanda che decide tutto è quanti conti si riescono a fare per ogni
viaggio in cantina, più che quanti conti ci siano da fare. La tabella dei conti
per byte è quella domanda, messa in numeri.

`````

`````{tab} Superiore

Uno strato moltiplica una matrice di pesi $n \times n$ per un blocco di
ingressi $n \times k$, dove $k$ è quante cose si elaborano insieme. Le
operazioni in virgola mobile sono $2 n^2 k$, e il due ha una ragione: per ogni
casella del risultato si fanno $n$ moltiplicazioni e altrettante somme, ed
è la convenzione con cui la {doc}`sezione su GEMM e tensor core
</GPU/gemm-e-tensor-core>` conta i FLOP. I byte letti dalla memoria sono invece
$n^2 b / 8$, con $b$ i bit per peso come nel resto del capitolo, e non
dipendono da $k$, perché i pesi sono gli stessi.

Il rapporto fra le due quantità,

$$
I = \frac{2 n^2 k}{n^2 b / 8} = \frac{16\,k}{b},
$$

è l’intensità aritmetica, la grandezza in ascissa del modello roofline del
capitolo sulla GPU, e il regime in cui si cade (legati alla banda o legati al
calcolo) lo decide il confronto fra $I$ e il rapporto fra prestazione di picco
e banda della macchina. Con pesi in sedici bit si semplifica in $I = k$, che è
la colonna di destra della tabella dei conti per byte.

Il conto dice due cose. La prima: $I$ non dipende dalla larghezza dello strato,
ma solo perché si stanno contando i byte dei pesi e non quelli di ingressi e
uscite, che sono $n k b / 8$ ciascuno; l’approssimazione vale per $k \ll n$ e
all’ultima riga della tabella ($k = 256$ contro $n = 4096$) sbaglia già del
dodici per cento. La seconda, che è la conseguenza più utile: dimezzare $b$
raddoppia $I$, quindi in regime legato alla banda, e finché il traffico è fatto
dai soli pesi, quantizzare non fa solo stare il modello in memoria ma dimezza
al più il tempo di ogni passo. Al più, perché la cache e gli ingressi non si
dimezzano, e riportare i pesi corti alla precisione del calcolo costa conti in
più.

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
conti si fanno per ogni byte letto, e nella {doc}`sezione sulla gerarchia della
memoria </GPU/gerarchia-memoria>` ha un nome, intensità aritmetica. Quando è
bassa il processore resta fermo ad aspettare i dati, e un processore più veloce
non cambia il tempo; quando è alta i dati fanno in tempo ad arrivare e il
processore lavora.

E adesso il punto. Quando un modello legge una domanda, la legge tutta
insieme: se la domanda è di duecento parole (più esattamente token, i pezzi in
cui il modello divide il testo), le cose insieme sono duecento, vicino
all’ultima riga della tabella. Che quel regime sia legato al calcolo dipende dal
ginocchio della macchina: a sedici bit l’intensità vale il numero di cose
insieme, e il ginocchio sta a 161 conti per byte su una A100 e a 295 su una
H100, quindi servono domande di centinaia di token, non di decine. Quando
scrive la risposta, la scrive una parola alla volta, perché per scegliere la
parola dopo deve aver scelto quella prima: le cose insieme sono una, ed è la
prima riga.

Sono la stessa moltiplicazione con lo stesso modello, e stanno ai due estremi
opposti della tabella. Il rapporto fra le due efficienze è, a meno dei byte
degli ingressi che il conto trascura, la lunghezza della domanda: con duecento
parole, scrivere è quasi duecento volte meno efficiente che leggere. E non
perché l’operazione sia più difficile: perché è la stessa lettura di
trentaquattro megabyte di pesi, spesa per una parola sola invece che per
duecento.

Questo è il motivo per cui rimpicciolire il modello aiuta anche il tempo di
risposta (meno pesi da leggere è meno traffico), ma non basta: finché si scrive
una parola alla volta, si è nella prima riga della tabella qualunque sia la
dimensione del modello.

## Le tre risposte, e dove stanno

Tre idee rispondono alla seconda domanda senza toccare il modello. Due vengono
dalla tabella, perché alzano il numero di cose insieme; la prima viene da
un’altra parte, dal lavoro che altrimenti si rifarebbe.

Non rifare due volte lo stesso lavoro. Un modello che scrive testo, per
scegliere la parola numero cinquecento, rimetterebbe in conto tutte le
quattrocentonovantanove di prima. Ma di ciascuna di quelle parole gli serve
soltanto una coppia di vettori per strato, la chiave e il valore, già calcolata
quando la parola è stata scritta, quindi invece di rifarla la si conserva, come
si tiene un segnalibro invece di rileggere il libro da capo a ogni pagina. Quel
deposito si chiama cache delle chiavi e dei valori, e lo costruisce la
{doc}`sezione sui grandi modelli linguistici </Transformers/llm>`, nel capitolo
sui Transformer (l’architettura di cui quei modelli sono fatti), perché è lì
che si capisce che cosa siano chiavi e valori. Non risolve il problema della
tabella dei conti per byte: sposta il traffico dai pesi al deposito, che cresce
a ogni parola scritta.

Riempire la riga. Se scrivere una parola per un solo utente sta nella prima
riga, scriverla per duecentocinquantasei utenti insieme sta nell’ultima: i pesi si
leggono una volta e servono a tutti. È il motivo per cui un servizio che
risponde a molti costa, a testa, molto meno di uno che risponde a uno, e la
{doc}`sezione su LLMOps </MLOps/llmops>` lo tratta parlando di come si
gestiscono le richieste che arrivano insieme e la memoria che ciascuna si
porta dietro.

Indovinare avanti e farsi correggere. Un modello piccolo propone qualche parola
di seguito; il modello grande le verifica tutte in una passata sola, cioè su
tutta la bozza insieme invece che su una parola sola, ne accetta un tratto, e
alla prima che rifiuta ne mette una sua. Ogni passata produce almeno una parola
e, se la bozza è buona, parecchie al prezzo di una; il tempo del modello
piccolo che si butta via è poco, e il testo che ne esce ha la stessa
distribuzione di quello che avrebbe scritto il solo modello grande. È la
**decodifica speculativa**, e sta nella {doc}`sezione su LLMOps
</MLOps/llmops>`, con la figura che mostra la bozza accettata e il punto in cui
il modello grande la taglia.

Non sono le sole risposte. Altre toccano il modello o il modo di calcolarlo:
l’attenzione con meno teste per chiavi e valori rimpicciolisce la cache
({doc}`sezione sull’attenzione in pratica
</Transformers/attenzione-in-pratica>`), la mixture of experts fa attraversare
a ogni parola solo una parte dei parametri ({doc}`sezione sulla mixture of
experts </Transformers/mixture-of-experts>`), FlashAttention riduce i byte che
l’attenzione sposta ({doc}`sezione su FlashAttention </GPU/flash-attention>`).
E le tre leve del capitolo tornano, applicate al servizio, nella sezione
«Comprimere per servire» di {doc}`LLMOps </MLOps/llmops>`.

`````{tab} Elementare

```{admonition} Da ricordare
:class: important
- Starci in memoria e rispondere in fretta sono due problemi diversi. Le
  tre leve del capitolo risolvono il primo, e aiutano il secondo solo di
  riflesso.
- La ragione è che leggere i pesi dalla memoria costa sempre uguale, che
  li si usi per una cosa sola o per duecento insieme. Un modello che legge
  una domanda di duecento parole le elabora tutte insieme; quando scrive la
  risposta va una parola alla volta, e paga la stessa lettura per una parola
  sola.
- Da qui le tre idee che non toccano il modello: tenersi gli appunti invece
  di rifarli (la cache delle chiavi e dei valori, nel capitolo sui
  Transformer), servire molti insieme perché i pesi letti una volta valgono
  per tutti (nel capitolo su MLOps), e far scrivere una bozza a un modello
  piccolo che il grande controlla tutta in una volta (la decodifica
  speculativa, ancora in MLOps).
```

`````

`````{tab} Superiore

```{admonition} Da ricordare
:class: important
- L’intensità aritmetica di uno strato $n \times n$ su un blocco
  $n \times k$ vale $16k/b$, cioè $k$ a sedici bit: i conti crescono con $k$, i
  byte dei pesi no. Non dipende da $n$ solo finché si trascurano i byte di
  ingressi e uscite, il che vale per $k \ll n$.
- Scrivere una parola alla volta (la generazione autoregressiva) impone
  $k = 1$ per passo, quindi è strutturalmente legata alla banda: a sedici bit
  $I = 1$, contro un ginocchio che sugli acceleratori sta nell’ordine delle
  centinaia (161 su A100, 295 su H100). L’elaborazione del testo in ingresso
  ha $k$ pari alla lunghezza della sequenza, ed è legata al calcolo solo quando
  $k$ supera il ginocchio. Sono lo stesso modello in due regimi diversi, ed è la
  ragione per cui le due fasi si misurano con due grandezze separate.
- Le tre mosse a modello invariato abbassano il costo di ogni parola
  scritta, ma da lati diversi del rapporto $I$. Il raggruppamento delle
  richieste e la decodifica speculativa alzano $k$ a parità di byte dei pesi
  letti; la seconda lo fa verificando in parallelo una bozza prodotta da un
  modello più economico, e con la regola di accettazione e rifiuto non cambia
  la distribuzione di uscita. La cache delle chiavi e dei valori fa il
  contrario: toglie i conti del ricalcolo e aggiunge i byte della cache, che
  cresce con il contesto, quindi abbassa l'intensità aritmetica del passo di
  decodifica invece di alzarla. Resta un affare perché per ogni posizione del
  contesto toglie proiezioni dell'ordine di $n^2$ operazioni e aggiunge la
  lettura di due vettori dell'ordine di $n$ numeri, con $n$ la larghezza dello
  strato.
- Nessuna delle tre cambia il modello, e per questo stanno altrove: la cache
  la costruisce la {doc}`sezione sui grandi modelli linguistici
  </Transformers/llm>`; il raggruppamento delle richieste e la decodifica
  speculativa la {doc}`sezione su LLMOps </MLOps/llmops>`; il modo di misurare
  separatamente i due regimi la {doc}`sezione sulle metriche di servizio
  </MLOps/metriche-di-servizio>`, tutte e due in MLOps. Altre risposte toccano
  il modello o il kernel: l’attenzione con meno teste per chiavi e valori, la
  mixture of experts, FlashAttention.
```

`````

Le tre leve si pagano, e il prezzo conta più dell’elenco delle tecniche.
Arrotondare a otto bit sposta l’uscita di uno strato dell’uno per cento e a
quattro molto di più, e quanto costi in accuratezza lo dice soltanto la prova
sul modello; potare novanta pesi su cento costa un punto di accuratezza e,
sulla moltiplicazione densa che quasi tutti eseguono, non regala un
millisecondo; imitare un maestro vuol dire prenderne anche gli errori, e un
maestro sbagliato fa peggio di nessuno. Il prezzo cambia da un modello
all'altro, e chi ne adotta una senza misurarlo sul proprio sta scegliendo alla
cieca.

La tabella dei conti per byte guarda uno strato solo, e un modello vero ne
mette in fila decine, uno dopo l’altro. Perché una rete sia fatta di molti
strati sottili invece che di uno solo largo è la domanda del {doc}`capitolo sul
deep learning </DeepLearning/overview>`, che ci risponde fin dalla sua prima
pagina, sotto il titolo «Profondo, non solo largo».
