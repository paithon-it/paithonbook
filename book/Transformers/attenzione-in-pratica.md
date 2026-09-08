# L'attenzione in pratica

Nel novembre del 2019 esce un articolo di nove pagine, firmato da una persona
sola, che si intitola *Fast Transformer Decoding: One Write-Head is All You
Need* {cite}`shazeer2019fast`. Il gioco di parole è deliberato: l'autore è
Noam Shazeer, secondo firmatario dell'articolo del 2017. Due anni dopo aver
scritto che serviva l'attenzione, torna sulla stessa formula per dire che, al
momento di generare, di teste per le chiavi ne basta una. La formula non è
cambiata di una virgola. È cambiato che cosa costa.

Il meccanismo montato fin qui gira in due modalità profondamente diverse, e
quasi tutto ciò che si legge sui costi dell'attenzione (che sia quadratica, che
la memoria esploda, che le teste condivise facciano risparmiare) diventa
comprensibile solo dopo aver separato le due.

## Addestrare e generare: due regimi della stessa formula

Un modello che scrive lo fa in modo **autoregressivo**: produce un token, se lo
rimette in ingresso, produce il successivo. La
{doc}`sezione sulla struttura del Transformer <architettura>` ha montato la
maschera causale che rende questo possibile, e da lì discende tutto quello che
segue sui costi.

`````{tab} Elementare
Un correttore di bozze e uno scrittore lavorano sullo stesso testo con due
ritmi opposti. Il correttore ha davanti il testo finito: può guardare la
pagina tutta insieme, e su ogni parola fare il suo lavoro nello stesso momento
in cui lo fa su tutte le altre. Lo scrittore no: la parola numero venti non
esiste finché non ha scritto la diciannovesima, quindi procede per forza uno
alla volta.

L'addestramento assomiglia al correttore. Il testo c'è già tutto, il modello
lo elabora in un colpo solo, e il divieto di sbirciare avanti glielo impone la
maschera invece del tempo. La generazione assomiglia allo scrittore, e non
c'è nessun trucco che la renda parallela: le parole che non ha ancora prodotto
non esistono da nessuna parte.

Da qui viene lo spreco che ha fatto nascere tutto il resto. Rifare per intero
il lavoro a ogni parola vorrebbe dire, per una risposta di mille parole,
rileggere da capo il testo mille volte: alla parola numero mille si
ricalcolerebbero per la millesima volta cose calcolate quando la prima parola
era appena uscita. E il punto che rende la cosa riparabile è che quelle cose
non sono cambiate. Poiché nessuna parola può guardare avanti, quello che
il modello aveva ricavato dalla parola numero tre quando la frase finiva lì è
identico a quello che ne ricaverebbe adesso che la frase è lunga mille.

C'è però una cosa che non si conserva, e distinguerla è metà del lavoro. La
domanda che una parola ha posto serviva a quella parola e a quel momento:
serve una volta, dà il suo risultato e si butta. Da tenere è invece ciò con cui
una parola si fa trovare e ciò che consegna, perché a quelli attingeranno tutte
le parole che verranno dopo.

E tutto questo regge finché vale il divieto di guardare avanti. Se ogni parola
potesse tenere d'occhio anche quelle che verranno, aggiungerne una in fondo
cambierebbe l'etichetta di tutte le altre, e quanto si era scritto prima
andrebbe buttato e rifatto da capo a ogni parola nuova: non resterebbe niente
da riusare. Le macchine che leggono un testo già finito lavorano proprio così,
e infatti non conservano niente da una parola all'altra.
`````

`````{tab} Superiore
Sia $T$ la lunghezza del prefisso già elaborato. Le due modalità in cui lo
stesso strato viene chiamato hanno forme diverse.

Nell’**elaborazione a prefisso pieno** (addestramento, oppure il *prefill* di
un prompt) molte posizioni di query vengono trattate insieme: $L = S = T$, la
matrice dei punteggi è $T \times T$, e il calcolo è un prodotto fra matrici
denso, ideale per l'hardware parallelo. La maschera causale garantisce che il
risultato coincida con quello che si otterrebbe elaborando le posizioni una
per una.

Nella **decodifica incrementale** arriva un token per volta: $L = 1$ (o poche
righe, con la decodifica speculativa) e $S = T + 1$. La matrice dei punteggi ha
una riga sola, e il collo di bottiglia si sposta dal calcolo alla banda di
memoria.

La ragione per cui la seconda modalità può riusare il lavoro della prima è un
fatto sulla maschera, non un'euristica. Con attenzione causale, la
rappresentazione della posizione $j$ a ogni strato dipende solo dalle posizioni
$\le j$; quindi $\mathbf{k}_j$ e $\mathbf{v}_j$, che sono proiezioni lineari di
quella rappresentazione, sono determinati appena il token $j$ è stato
elaborato, e non cambiano quando la sequenza si allunga. Formalmente,
$\mathbf{k}_j = \mathbf{W}^{K\top} \mathbf{h}_j^{(\ell)}$ con
$\mathbf{h}_j^{(\ell)} = f^{(\ell)}(\mathbf{x}_{1:j})$, indipendente da
$\mathbf{x}_{>j}$.

La query gode della proprietà opposta, ed è il motivo per cui non si conserva
mai: $\mathbf{q}_t$ serve a calcolare la riga $t$ della matrice dei pesi, viene
consumata in quel prodotto e non compare in nessun conto successivo. Conservare
$\mathbf{Q}$ sarebbe occupare memoria per un oggetto che nessuno rileggerà.

Un avvertimento sulla portata di tutto questo: la garanzia vale per
l'attenzione causale. In un encoder bidirezionale la rappresentazione di ogni
posizione dipende anche da ciò che viene dopo, quindi aggiungere un token
invalida le chiavi già calcolate, e non c'è niente da riusare.
`````

## La cache KV, come struttura dati

Da quell'osservazione nasce un oggetto che ogni sistema di inferenza ha in
pancia, e che si chiama KV cache. Conviene guardarlo per quello che è, cioè
una coppia di tensori che crescono di una riga per token generato, allocata per
ogni strato e per ogni testa di chiave e valore.

`````{tab} Elementare
Un taccuino, uno per ogni piano del palazzo e per ogni lettore. Ogni volta che
il modello elabora una parola nuova ci scrive due righe: l'etichetta con cui
quella parola si farà trovare, e l'informazione che consegnerà a chi la sceglie.
Le righe già scritte non si correggono mai, si aggiungono e basta, e per la
parola nuova bastano loro: la domanda che quella parola pone si confronta con
tutte le etichette del taccuino, e il miscuglio si fa con le informazioni che
ci stanno accanto.

Il risparmio si vede contando che cosa si evita. Senza taccuino, ogni parola
nuova costringe a ricavare etichetta e informazione per tutte le parole già
scritte; con il taccuino, se ne ricavano due sole, quelle della parola nuova.
Il resto era già lì.

Quello che il taccuino non fa sparire è il confronto. La parola nuova deve
comunque guardare tutte le righe scritte finora, quindi la fatica di ogni
parola cresce insieme al testo: la millesima costa mille confronti, la
duemillesima duemila. Chi dice che il taccuino rende la scrittura gratis sta
guardando solo la metà che si risparmia.

E c'è un prezzo di spazio che si vede subito appena si prova a immaginare il
taccuino di una conversazione lunga. Un palazzo di trentadue piani con
trentadue lettori per piano vuol dire mille taccuini, ciascuno con due righe
per ogni parola scritta: qualche centinaio di kilobyte a parola, che per una
conversazione da ottomila parole diventano gigabyte di carta da tenere
aperta. Da lì in poi la domanda non è più come far scrivere il modello, ma
dove mettere i taccuini.
`````

`````{tab} Superiore
Per ogni strato $\ell$ e ogni testa di chiave-valore, la cache tiene due
matrici che crescono per righe:

$$
\mathbf{K}^{(\ell)} \in \mathbb{R}^{T \times d_k}, \qquad
\mathbf{V}^{(\ell)} \in \mathbb{R}^{T \times d_k} .
$$

Al passo $t$ lo strato calcola $\mathbf{q}_t, \mathbf{k}_t, \mathbf{v}_t$ per
il solo token nuovo, appende $\mathbf{k}_t$ e $\mathbf{v}_t$ alle rispettive
matrici e valuta l'attenzione di $\mathbf{q}_t$ (una riga) contro le $t$ righe
accumulate. È il caso $L \neq S$ per eccellenza: $1 \times t$ invece di
$t \times t$.

Il conto degli elementi conservati per token, sommando su tutti gli strati, è

$$
2 \, n_{\text{strati}} \, h_{kv} \, d_k ,
$$

dove il $2$ conta chiavi e valori, $n_{\text{strati}}$ è quanti sono gli strati
(da non confondere con l'indice $\ell$ che ne individua uno), $h_{kv}$ è il
numero di teste di chiave-valore e $d_k$ la dimensione per testa (qui e nel
seguito $d_v = d_k$, come in quasi tutte le implementazioni). In byte si
moltiplica per la dimensione del tipo numerico. Il conto non dipende dal numero
di teste di query, e questa è l'osservazione su cui si regge tutto ciò che
segue.

Il costo in calcolo, per token e per strato, si divide in due termini di natura
diversa: $O(t\, d)$ per l'attenzione contro il prefisso, e $O(d^2)$ per le
proiezioni e la rete feed-forward, con $d = d_{\text{model}}$. Il secondo non
dipende dalla lunghezza; il primo sì, e cresce linearmente in ciò che è già
stato letto. La generazione di $n$ token costa quindi $O(n^2 d + n\, d^2)$ per
strato, contro l’$O(n^3 d + n^2 d^2)$ di un forward rifatto da capo a ogni
passo.

Una precisazione da tenere a mente in vista di quel che segue: la cache
riusa le proiezioni passate, non abolisce l'interazione con il
prefisso. L'affermazione «la cache rende la decodifica a costo costante» è
falsa per l'attenzione densa.
`````

Il modo più rapido di convincersi che le due strade danno la stessa cosa è
eseguirle entrambe. Il blocco seguente costruisce uno strato di attenzione
causale con pesi casuali, lo esegue una volta sulla sequenza intera e una volta
un token per volta con la cache, e confronta le due uscite. La funzione
`attenzione` è una sola, chiamata con forme diverse: è il punto della faccenda.

```python
import torch

torch.manual_seed(0)
d_model, d_k, T = 16, 4, 6
W_Q, W_K, W_V = (torch.randn(d_model, d_k) / d_k**0.5 for _ in range(3))
X = torch.randn(T, d_model)          # i sei token del prefisso

def attenzione(Q, K, V):
    """Attenzione causale con L query e S chiavi, allineate a destra:
    l'ultima query vede tutte le S chiavi."""
    L, S = Q.shape[0], K.shape[0]
    punteggi = Q @ K.T / Q.shape[-1] ** 0.5
    vietato = torch.triu(torch.ones(L, S, dtype=torch.bool), diagonal=S - L + 1)
    return torch.softmax(punteggi.masked_fill(vietato, float("-inf")), -1) @ V

# 1) tutto il prefisso in un colpo: L = S = 6, matrice quadrata
O_pieno = attenzione(X @ W_Q, X @ W_K, X @ W_V)

# 2) un token per volta, con la cache: L = 1, S che cresce
K_cache, V_cache, uscite = torch.empty(0, d_k), torch.empty(0, d_k), []
for t in range(T):
    x_t = X[t:t+1]                                # il solo token nuovo
    K_cache = torch.cat([K_cache, x_t @ W_K])     # la cache cresce di una riga
    V_cache = torch.cat([V_cache, x_t @ W_V])     # la query non si conserva
    uscite.append(attenzione(x_t @ W_Q, K_cache, V_cache))
O_cache = torch.cat(uscite)

print("cache dopo sei token:", tuple(K_cache.shape), tuple(V_cache.shape))
print("le due strade coincidono:", torch.allclose(O_pieno, O_cache, atol=1e-5))
```

```text
cache dopo sei token: (6, 4) (6, 4)
le due strade coincidono: True
```

Tre righe di quel blocco meritano di essere lette due volte. La riga della
maschera usa `diagonal=S - L + 1`, che nel caso quadrato vale 1 e dà il solito
triangolo, e nel caso a una riga sola non vieta niente: è la convenzione di
allineamento a destra, e scriverla in funzione di $L$ e $S$ invece che
fissarla a 1 è ciò che permette alla stessa funzione di servire tutti e due i
regimi. Le due righe che estendono la cache sono le uniche a scrivere; e di
$\mathbf{q}_t$ non resta traccia da nessuna parte.

## Quante teste per le chiavi: da MHA a MLA

Il conto della cache dipende dal numero di teste di chiave-valore e ignora
quelle di query. È una porta lasciata aperta, e dal 2019 in poi ci sono
passate MQA, GQA e MLA, ciascuna con un modo diverso di stringere lo stesso
bullone.

`````{tab} Elementare
In un palazzo con otto lettori per piano, finora ognuno teneva il proprio
taccuino: otto etichette e otto informazioni per ogni parola, per ogni piano.
Guardando i taccuini, però, viene un sospetto: i lettori si distinguono per le
domande che pongono, non per il materiale che consultano. E allora perché non
farli leggere dallo stesso taccuino?

La versione estrema lo fa: un taccuino solo per piano, otto lettori che ci
attingono tutti, ciascuno con le sue domande. Lo spazio si divide per otto, la
scrittura diventa molto più svelta, e si perde qualcosa in qualità, perché
otto punti di vista che consultavano appunti diversi adesso consultano gli
stessi.

La versione di mezzo divide i lettori in gruppi: due gruppi da quattro, e un
taccuino per gruppo. Si sceglie quanti gruppi fare, e con quella manopola si
decide dove stare fra i due estremi. È la soluzione che hanno adottato quasi
tutti, perché con pochi gruppi la perdita si assottiglia fino a non vedersi. E
c'è una seconda ragione, che ha pesato quanto la prima: un palazzo già
costruito, con il suo taccuino per ogni lettore, si converte ai gruppi senza
tirarlo giù. Si fondono gli appunti dei lettori di uno stesso gruppo, si
rimette a punto il tutto per un tempo breve rispetto a quello che era costato
costruirlo, e il palazzo riapre.

L'ultima strada cambia mestiere. Invece di ridurre il numero di taccuini,
riscrive che cosa ci si annota: al posto delle etichette e delle informazioni
per esteso, un riassunto compatto da cui le une e le altre si possono
ricostruire al momento del bisogno. Un taccuino di appunti stenografati, che
occupa una frazione dello spazio e si rilegge quando serve. La stenografia ha
un punto scomodo, e riguarda il modo in cui il modello sa in che ordine stanno
le parole: quel segnale va tenuto fuori dal riassunto, su una riga a parte,
altrimenti il risparmio che rende conveniente la stenografia non c'è
più.

Un equivoco da togliere subito, perché è quello che si sente ripetere. Nessuna
di queste strade tocca il numero di lettori. Le domande restano otto, i punti
di vista restano otto: quello che si condivide o si comprime è il materiale su
cui si legge.
`````

`````{tab} Superiore
MHA, MQA e GQA si distinguono per come trattano $h_{kv}$ rispetto a $h_q$;
MLA cambia invece l'oggetto conservato.

**Multi-head attention (MHA)**, la formulazione originale: $h_{kv} = h_q$, una
proiezione di chiave e una di valore per ogni testa di query. Cache per token
$2\,n_{\text{strati}}\,h_q\,d_k$.

**Multi-query attention (MQA)** {cite}`shazeer2019fast`: $h_{kv} = 1$. Tutte le
teste di query condividono un'unica testa di chiave e un'unica testa di valore.
La cache si riduce di un fattore $h_q$, e con essa la banda di memoria
richiesta a ogni passo di decodifica; il prezzo è una perdita di qualità
rispetto a MHA.

**Grouped-query attention (GQA)** {cite}`ainslie2023gqa`: $1 < h_{kv} < h_q$.
Le teste di query sono partizionate in $h_{kv}$ gruppi, e ogni gruppo
condivide una testa di chiave-valore. Il numero di gruppi è la manopola fra i
due estremi, e gli autori mostrano che con pochi gruppi si arriva vicino alla
velocità di MQA restando vicini alla qualità di MHA. Nello stesso lavoro c'è
la parte che ne ha decretato l'adozione: un modello già addestrato con MHA si
converte a GQA con un *uptraining* che costa una frazione del preaddestramento
originale, mediando i pesi delle teste di ciascun gruppo.

**Multi-head latent attention (MLA)**, introdotta con DeepSeek-V2
{cite}`deepseekv2`: cambia la rappresentazione conservata invece del numero di
teste. Una compressione congiunta a rango basso produce un latente
$\mathbf{c}^{KV}_t = \mathbf{W}^{DKV}\mathbf{h}_t \in \mathbb{R}^{d_c}$ con
$d_c \ll h_q d_k$, e da lì due proiezioni apprese ricostruiscono chiavi e
valori. In decodifica si conserva il solo latente, e le matrici di
ricostruzione si assorbono in $\mathbf{W}^Q$ e $\mathbf{W}^O$, quindi chiavi e
valori per esteso non vengono nemmeno materializzati.

La descrizione precisa richiede un'avvertenza che di solito si omette, e senza
la quale la cache di MLA risulta più piccola di quella vera. RoPE è
incompatibile con la compressione: applicandola alle chiavi ricostruite, la
matrice di ricostruzione resterebbe incastrata fra $\mathbf{W}^Q$ e una matrice
di rotazione che dipende dalla posizione corrente, e l'assorbimento non sarebbe
più possibile. DeepSeek-V2 risolve con una **RoPE disaccoppiata**: una
componente di chiave a parte, condivisa fra le teste, che porta il segnale
posizionale. La cache contiene quindi il latente più quella componente,
$(d_c + d_k^R)\,n_{\text{strati}}$ elementi per token.

| formulazione | elementi in cache per token | teste di chiave-valore |
|---|---|---|
| MHA | $2\,n_{\text{strati}}\,h_q\,d_k$ | quante quelle di query |
| GQA | $2\,n_{\text{strati}}\,h_{kv}\,d_k$ | un numero intermedio |
| MQA | $2\,n_{\text{strati}}\,d_k$ | una |
| MLA | $(d_c + d_k^R)\,n_{\text{strati}}$ | altra formulazione |

Con i valori di DeepSeek-V2 ($d_c = 4 d_k$, $d_k^R = d_k/2$) la cache di MLA
equivale a quella di un GQA con 2,25 gruppi. Il punto da non confondere: MLA
non è riducibile a un GQA con meno teste, perché cambia l'oggetto conservato e
l'algebra delle proiezioni usata in inferenza.

MQA e GQA muovono $h_{kv}$, MLA cambia l'oggetto conservato. Quello che
nessuna delle tre tocca è il numero di teste di query, e con esso la
molteplicità dei sottospazi in cui il modello calcola le compatibilità: quello
che si condivide, o si comprime, è la rappresentazione conservata.
`````

I numeri rendono l'idea meglio della formula. Il blocco che segue prende un
modello di taglia ordinaria (trentadue strati, trentadue teste di query,
$d_k = 128$, quindi $d_{\text{model}} = 4096$) con chiavi e valori a 16 bit, e
conta quanta memoria vuole la cache per ogni token e per una finestra di
ottomila.

```python
strati, teste_q, d_k, byte, contesto = 32, 32, 128, 2, 8192
pesi = 7e9 * byte                    # sette miliardi di parametri, a 16 bit

varianti = [
    (f"MHA   (h_kv = {teste_q})",       2 * strati * teste_q * d_k),
    ("GQA   (h_kv = 8)",                2 * strati * 8 * d_k),
    ("MQA   (h_kv = 1)",                2 * strati * 1 * d_k),
    ("MLA   (d_c = 4 d_k, d_R = d_k/2)", strati * (4 * d_k + d_k // 2)),
]

print(f"{'variante':34}{'KiB/token':>11}{'GiB a 8192 token':>19}")
for nome, elementi in varianti:
    per_token = elementi * byte
    print(f"{nome:34}{per_token/1024:11.0f}{per_token*contesto/1024**3:19.2f}")
print(f"\npesi del modello: {pesi/1024**3:.1f} GiB")
```

```text
variante                            KiB/token   GiB a 8192 token
MHA   (h_kv = 32)                         512               4.00
GQA   (h_kv = 8)                          128               1.00
MQA   (h_kv = 1)                           16               0.12
MLA   (d_c = 4 d_k, d_R = d_k/2)           36               0.28

pesi del modello: 13.0 GiB
```

Quattro gigabyte per una sola conversazione da ottomila token, contro i tredici
che occupano i pesi: bastano quattro richieste servite insieme perché la
cache costi più del modello. La riga GQA dice perché la si trova
quasi ovunque nei modelli recenti, e la riga MLA perché qualcuno abbia cambiato
la formulazione dell'attenzione per un problema di memoria. La riga MHA, per
inciso, è il conto che la {doc}`sezione sui grandi modelli linguistici <llm>`
fa sul suo modello da sette miliardi di parametri: mezzo megabyte a token è
questa tabella nel caso in cui ogni testa di query si tiene le sue chiavi.

## Che cosa è quadratico, e di quale risorsa si parla

Il {doc}`confronto coi modelli precedenti <confronti>` ha messo il conto sul
tavolo separando il tempo dalla memoria. Quella distinzione va portata un passo
più in là, perché sotto la parola «quadratico» stanno grandezze che si
comportano in modo diverso e che una discussione sui costi tratta come se
fossero la stessa.

Il **lavoro aritmetico** dell'attenzione densa su una sequenza di lunghezza $n$
ha un termine $O(n^2 d)$ per strato, ed è un fatto sull'operazione: le coppie
query-chiave sono $n^2$ e vanno calcolate tutte. La **memoria intermedia
materializzata** è un fatto sull'implementazione: un'implementazione ingenua
scrive la matrice $n \times n$ dei punteggi, e allora anche lo spazio è
quadratico; una che non la scrive tutta insieme non lo è. Il **traffico di
memoria**, cioè quanti byte si spostano fra i livelli della gerarchia, è un
terzo fatto ancora, e sulle schede grafiche di oggi è spesso quello che decide
il tempo di esecuzione, a parità di conti da fare.

Tenerle distinte permette di dire due cose insieme senza contraddirsi:
l'attenzione densa fa un numero quadratico di interazioni, e un'implementazione
esatta può evitare di scrivere in memoria l'intera matrice quadratica e girare
molto più in fretta. La seconda è FlashAttention: stessa attenzione, esecuzione
diversa, e la costruisce per intero la {doc}`sezione sulla FlashAttention
</GPU/flash-attention>`. Una famiglia diversa di rimedi cambia invece quali
coppie si calcolano, con finestre locali o schemi sparsi, e allora l'attenzione
non è più densa; un'altra ancora riscrive la softmax per non formare mai le
coppie, ed è il {doc}`capitolo sull'attenzione lineare
</AttenzioneLineare/overview>`.

In decodifica il quadro cambia forma, e dirlo con precisione evita quasi tutti
gli slogan sbagliati. Per ogni token generato, con la cache, il lavoro
dell'attenzione cresce linearmente nella lunghezza già letta, non
quadraticamente; il quadrato ricompare quando si somma su tutti i token
generati. La memoria della cache, invece, cresce linearmente nella lunghezza e
non se ne va mai: è lei, e non il tempo, a fissare quanto contesto un servizio
riesce a tenere aperto.

## Gli errori che si fanno davvero

Chi scrive uno strato di attenzione da zero sbaglia quasi sempre sulle forme,
sugli assi e sulle convenzioni, quasi mai sulla formula. La formula si ricorda;
a mordere è il resto, e il guaio è che parecchi di questi errori non fanno
rumore: il codice gira, i tensori hanno la forma giusta, il modello si addestra
e impara qualcosa. Solo un po’ peggio di quanto dovrebbe.

Il più insidioso di tutti si diagnostica in due righe. Normalizzare lungo
l'asse sbagliato definisce un'operazione diversa, e il sintomo è secco: le
righe smettono di sommare a uno.

```python
import torch

punteggi = torch.tensor([[2., 0., 1.],
                         [0., 3., 1.],
                         [1., 1., 4.]])
print("somma per riga, normalizzando sulle chiavi:",
      torch.softmax(punteggi, dim=-1).sum(dim=-1))
print("somma per riga, normalizzando sulle query: ",
      torch.softmax(punteggi, dim=0).sum(dim=-1).round(decimals=4))
```

```text
somma per riga, normalizzando sulle chiavi: tensor([1.0000, 1.0000, 1.0000])
somma per riga, normalizzando sulle query:  tensor([0.7525, 0.9791, 1.2684])
```

`````{tab} Elementare
Gli errori si raccontano tutti sul tabellone e sui taccuini.

Il conto fatto per colonna invece che per riga. Invece di dare a ogni parola
che chiede una sua unità di colore da spartire, si dà una unità a ogni parola
che risponde, da spartire fra chi la cerca. Le righe smettono di sommare a
uno, e il colore che ogni parola riceve dipende da quante altre la cercano.

La divisione dimenticata prima di colorare. Il modello impara lo stesso e va un
po’ peggio, tanto più quanto più le liste sono lunghe, e non se ne accorge
nessuno perché niente si rompe.

Colorare prima e cancellare dopo. La riga resta con meno di un'unità di colore,
e il miscuglio esce sbiadito.

Il tabellone dato per quadrato. Quando chi chiede e chi risponde sono due liste
di lunghezza diversa, la regola «cancella tutto quello che sta sopra la
diagonale» va riscritta dicendo da che parte le due liste sono allineate.

I lettori contati al posto dei taccuini. Da quando i lettori condividono gli
appunti i due numeri sono diversi, e una stima di quanta carta serve fatta
contando i lettori sbaglia in eccesso, cioè nella direzione che non fa mai
suonare nessun allarme.

Le intensità del colore consegnate al posto del miscuglio. Le intensità servono
a decidere le proporzioni e poi escono di scena; quello che si consegna al
piano dopo è la miscela delle informazioni, che ha tutt'altra forma e tutt'altro
significato.

Due misure di memoria confrontate senza dire che cosa ci si è messo dentro. I
numeri del modello, il tabellone di passaggio, i taccuini e tutto quello che il
calcolatore tiene aperto per lavorare sono cose distinte, e due conti che non
dichiarano quali voci comprendono non si possono paragonare.
`````

`````{tab} Superiore
La softmax lungo la dimensione sbagliata. L'attenzione normalizza sulle
chiavi, per ogni query: `dim=-1` su un tensore `(..., L, S)`. Normalizzare
lungo le query definisce un'altra operazione, e le righe di $\mathbf{A}$ non
sommano più a 1.

La scala dimenticata. Senza il fattore $1/\sqrt{d_k}$ il modello si
addestra lo stesso e degrada al crescere di $d_k$, in un modo che nessun
controllo segnala.

La maschera applicata dopo la softmax. Azzerare i pesi vietati a valle
lascia righe che non sommano a 1, e con un logit vietato abbastanza grande
manda in underflow quelli permessi. Il divieto va sommato ai punteggi prima
della normalizzazione.

La maschera quadrata dove $L \neq S$. Un triangolo costruito
sull'assunzione che righe e colonne siano in corrispondenza uno a uno è
sbagliato appena le due lunghezze differiscono, cioè nell'attenzione incrociata
e nella decodifica con la cache. L'allineamento va dichiarato, e va fatto
combaciare con quello che la libreria si aspetta.

Le teste di query confuse con quelle di chiave-valore. In GQA i due numeri
differiscono, e ogni conto sulla cache va fatto con $h_{kv}$. Usare $h_q$
sbaglia la stima di un fattore $h_q/h_{kv}$, cioè di quante teste di query
si dividono la stessa testa di chiave-valore: con trentadue teste di query e
otto di chiave-valore il fattore è quattro, non otto come i gruppi. E sbaglia
in eccesso, che è la direzione che non fa mai fallire niente.

I pesi di attenzione presi per l'uscita dello strato. $\mathbf{A}$ è un
intermedio $L \times S$; l'uscita è $\mathbf{A}\mathbf{V}$, seguito dalla
concatenazione delle teste, dalla proiezione $\mathbf{W}^O$ e dal resto del
blocco. Restituire i pesi da una funzione che deve restituire l'uscita produce
tensori di forma plausibile e risultati senza senso.

La memoria dell'attenzione confrontata senza dichiarare che cosa si conta.
Pesi, matrice temporanea dei punteggi, cache, attivazioni e spreco
dell'allocatore sono grandezze distinte, e due misure che non dicono quali voci
includono non sono confrontabili.
`````

## Le frasi che si sentono dire

Attorno all'attenzione circolano alcune formule fatte che sopravvivono proprio
perché sono quasi vere, e che si smontano una per una.

«L'attenzione è la memoria del modello.» L'attenzione è un'operazione di
mescolamento, e non conserva niente da una chiamata all'altra. Quello che
persiste durante la generazione è la cache, che appartiene al sistema di
inferenza e viene buttata a fine risposta. Fra una conversazione e l'altra il
modello non ricorda nulla: ciò che sembra memoria è testo che qualcuno gli
rimette davanti, e la {doc}`sezione sul context engineering
</Agenti/context-engineering>` racconta come lo si sceglie.

«Più teste è meglio.» A $d_{\text{model}}$ fisso, aggiungere teste le
assottiglia: con $d_{\text{model}} = 512$, otto teste lavorano in dimensione
64 e trentadue in dimensione 16. Oltre un certo punto ogni testa ha uno spazio
troppo stretto perché la compatibilità che calcola dica qualcosa, e il guadagno
di varietà si mangia quello di risoluzione. Il numero di teste è un compromesso
che si taglia sul modello, non una quantità da massimizzare.

«I pesi di attenzione sono la spiegazione.» È la più diffusa e la più
delicata, perché quei pesi si disegnano bene: sono distribuzioni sulle
posizioni d'ingresso, e la tentazione di leggere un peso alto come «il modello
ha usato questo token perché contava» è forte. Il dibattito ha due voci, e
conoscerle entrambe serve. Jain e Wallace {cite}`jain2019attention` hanno
mostrato casi in cui i pesi appresi si legano debolmente alle misure di
importanza basate sul gradiente, e in cui distribuzioni di attenzione molto
diverse davano previsioni quasi identiche. Wiegreffe e Pinter
{cite}`wiegreffe2019attention` hanno contestato il rifiuto categorico,
sostenendo che la risposta dipende da che cosa si chiama spiegazione e da quali
termini di paragone si adottano. La posizione difendibile sta in mezzo, ed è
più stretta di tutti e due gli slogan: un peso alto dice dove è andata
l'aggregazione, e non dice perché il modello abbia risposto così. Leggerlo come
una spiegazione richiede un metodo e una validazione, che è il mestiere della
{doc}`sezione su attribuzione e interpretabilità meccanicistica
</Interpretabilita/attribuzione-e-meccanicistica>`.

## L'attenzione che si usa davvero

Scrivere l'attenzione a mano serve a capirla; chi la deve far girare chiama una
funzione di libreria, che sceglie da sé l'implementazione adatta all'hardware
che ha davanti. In PyTorch è `scaled_dot_product_attention`, e la sua firma
raccoglie quasi tutti gli oggetti nominati fin qui.

```{code-block} python
:class: pt-non-eseguibile

import torch.nn.functional as F

# query: (batch, teste_q, L, d_k)    chiave e valore: (batch, teste_kv, S, d_k)
uscita = F.scaled_dot_product_attention(
    query, chiave, valore,
    attn_mask=None,     # maschera booleana o additiva; alternativa a is_causal
    dropout_p=0.0,      # da azzerare in valutazione
    is_causal=True,     # la maschera causale, senza materializzarla
    scale=None,         # None significa 1/sqrt(d_k)
    enable_gqa=True,    # teste_kv < teste_q: la condivisione la fa la funzione
)
```

Due trappole di questa interfaccia meritano di essere scritte, perché sono
esattamente del genere descritto poco fa. La prima: nella maschera booleana di
questa funzione `True` significa che la posizione partecipa all'attenzione,
cioè l'opposto della convenzione del `key_padding_mask` di
`nn.MultiheadAttention`, dove `True` marca ciò che va escluso. Due funzioni
della stessa libreria, due convenzioni opposte, nessun errore se le si scambia:
soltanto un modello che guarda esattamente le posizioni sbagliate. La seconda:
`is_causal=True` non fa la maschera che la funzione scritta poco fa costruisce
a mano. Allinea in alto a sinistra, quindi con una riga di query e un prefisso
lungo la sola query vede la posizione iniziale e nient'altro; l'allineamento
che serve a generare è quello a destra, che la libreria espone a parte. In
decodifica con la cache la maschera non serve comunque, perché con una query
sola non c'è niente da vietare. I vincoli di divisibilità di `enable_gqa`,
invece, dipendono dalla versione e vanno letti in quella installata.

`````{tab} Elementare
```{admonition} Da ricordare
:class: important
- La stessa formula gira in due regimi opposti. Con il testo già scritto
  davanti, il modello lavora su tutte le parole insieme; mentre scrive, va per
  forza una parola alla volta, perché quella dopo non esiste ancora.
- Quello che una parola ha calcolato quando la frase finiva lì resta valido
  quando la frase si allunga, perché nessuno può guardare avanti. Da qui il
  taccuino: si conservano l'etichetta e l'informazione di ogni parola, e la
  domanda no, perché serviva una volta sola.
- Il taccuino toglie la fatica di ricalcolare, non quella di confrontare: ogni
  parola nuova guarda comunque tutte quelle di prima, quindi il costo di
  scrivere cresce insieme al testo.
- E il taccuino occupa spazio, tanto: per un modello di taglia ordinaria mezzo
  megabyte per ogni parola, e già quattro conversazioni lunghe servite
  insieme occupano più memoria del modello stesso.
- Di lì le varianti: far leggere più lettori dallo stesso taccuino (tutti, o a
  gruppi), oppure annotare un riassunto stenografato da cui etichette e
  informazioni si ricostruiscono. Nessuna di queste tocca il numero di lettori,
  cioè i punti di vista restano quanti erano.
- Sotto la parola «quadratico» stanno cose diverse: quanti conti si fanno,
  quanta memoria si occupa di passaggio, e quanti byte si spostano. Tenerle
  insieme è la fonte di quasi tutti gli equivoci sui costi.
- Gli errori che si fanno montando questa macchina riguardano quasi sempre le
  righe, le colonne e le convenzioni, e quasi mai la formula. Il più delle
  volte niente si rompe: il modello impara un po’ peggio, e basta.
```
`````

`````{tab} Superiore
```{admonition} Da ricordare
:class: important
- Prefisso pieno e decodifica incrementale sono due regimi della stessa
  operazione: nel primo $L = S = T$ e il calcolo è denso e parallelo, nel
  secondo $L = 1$ e $S = T + 1$, con il collo di bottiglia sulla banda di
  memoria.
- Con attenzione causale $\mathbf{k}_j$ e $\mathbf{v}_j$ dipendono solo dalle
  posizioni $\le j$, quindi non cambiano quando la sequenza si allunga: è la
  garanzia su cui poggia la cache. $\mathbf{q}_t$ viene consumata nel prodotto
  e non si conserva mai. In un encoder bidirezionale la garanzia cade e non c'è
  niente da riusare.
- La cache tiene $2\,n_{\text{strati}}\,h_{kv}\,d_k$ elementi per token: il
  conto non dipende da $h_q$. Per trentadue strati, $h_{kv} = 32$ e
  $d_k = 128$ a 16 bit sono 512 KiB per token, cioè 4 GiB per una finestra di
  8192.
- MQA porta $h_{kv}$ a 1, GQA a un valore intermedio con le teste di query
  raggruppate, MLA conserva un latente compresso più una componente di RoPE
  disaccoppiata. Tutte riducono o comprimono le teste di chiave-valore, nessuna
  tocca $h_q$.
- Per token generato, con la cache, l'attenzione costa $O(t\,d)$, lineare nel
  prefisso; il quadrato riappare sommando sui token generati. Lavoro
  aritmetico, memoria intermedia materializzata e traffico fra i livelli di
  memoria sono grandezze distinte, e «quadratico» da solo non dice quale.
- Gli errori ricorrenti sono di asse e di convenzione: softmax sulla dimensione
  sbagliata, scala omessa, maschera applicata dopo la normalizzazione, triangolo
  quadrato con $L \neq S$, $h_q$ usato al posto di $h_{kv}$, pesi restituiti al
  posto di $\mathbf{A}\mathbf{V}$.
- Un peso di attenzione alto localizza l'aggregazione e non la spiega: trattarlo
  come spiegazione richiede un metodo e una validazione.
```
`````

La formula del 2017 è rimasta quella, e attorno a lei è cambiato tutto il
resto: dove si conservano i suoi risultati intermedi, quante teste li
condividono, in che ordine si toccano i livelli di memoria. È
utile tenerlo presente nelle pagine che seguono, dove i Transformer si mettono
al lavoro su compiti veri: quello che si scarica e si esegue in tre righe è
questa macchina, con addosso dieci anni di accorgimenti che nessuno racconta
più perché sono diventati il modo normale di farla girare.
