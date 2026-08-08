# Come si spezza il testo: BPE, WordPiece e SentencePiece

Nella sezione precedente abbiamo dato per buono che il testo si tagli in
token, e abbiamo detto che i sistemi moderni usano pezzi più piccoli della
parola. Resta però la domanda che conta davvero, ed è una domanda di
ingegneria: **quali** pezzi? Nessuno decide a mano che *tokenizzazione* vada
spezzata in `token` + `izzazione`. Quella decisione è il risultato di un
algoritmo che ha letto un corpus enorme e ha scelto, uno per uno, i mattoncini
da tenere. Questa sezione apre quell'algoritmo.

Il punto di partenza è un vicolo cieco. Se il vocabolario contiene parole
intere, non chiude mai: qualunque soglia scegliate, prima o poi arriverà una
parola che non c'è. Un cognome (*Rossellini*), un refuso (*gattto*), un
termine tecnico (*ortogonalizzazione*), un composto tedesco costruito sul
momento. Il sistema la sostituisce con un simbolo speciale, `<UNK>`, e da quel
momento quella parola per il modello non esiste più: era il nome del paziente,
il numero di serie, il soggetto della frase, e adesso è un buco. Peggio:
`<UNK>` non si può nemmeno *generare*, perché un modello che scrive non ha
modo di riempirlo.

All'estremo opposto c'è la soluzione radicale: un vocabolario di singoli
caratteri. Le parole sconosciute spariscono per costruzione, perché ogni testo
è fatto di lettere che il vocabolario contiene tutte. Ma si paga due volte. La
prima: le sequenze si allungano di brutto, perché una parola italiana media
sta attorno ai cinque o sei caratteri, e siccome il costo dell'attenzione (lo
vedremo nel capitolo sui Transformer) cresce con il **quadrato** della
lunghezza, una sequenza cinque volte più lunga costa venticinque volte il
lavoro. La seconda: il modello deve reimparare da zero che le lettere si
raggruppano in parole, spendendo capacità per riscoprire una struttura che gli
si poteva regalare.

Le **sotto-parole** (*subword*) stanno in mezzo, e sono il compromesso che ha
vinto: le parole frequenti restano intere (un token per *rosso*), quelle rare
si scompongono in pezzi noti, e niente resta fuori. Il resto della sezione
spiega come si costruisce, concretamente, quell'elenco di pezzi.

## Il dilemma del vocabolario

Prima di guardare gli algoritmi vale la pena capire *cosa* stiamo ottimizzando,
perché la scelta della taglia del vocabolario non è un dettaglio: è un baratto
fra due costi che tirano in direzioni opposte.

`````{tab} Elementare

Immaginate di dover preparare una scatola di mattoncini per costruire
qualunque parola italiana. Se mettete nella scatola solo le ventuno lettere
dell'alfabeto, potete costruire tutto, ma ogni parola richiede cinque o sei
pezzi e ci mettete un'eternità. Se invece mettete un pezzo già fatto per ogni
parola del dizionario, costruite in un colpo solo, ma la scatola diventa
enorme e comunque, il giorno che vi serve un cognome o una parola inventata,
non c'è.

La soluzione è una scatola mista: i pezzi grandi per le cose che ricorrono
(*rosso*, *casa*, *-mente*, *-zione*) e le lettere singole come riserva per
tutto il resto. Il problema diventa allora: quali pezzi grandi conviene
tenere, dato che lo spazio nella scatola è limitato? La risposta di tutti gli
algoritmi che vedremo è la stessa in spirito: **teniamo i pezzi che fanno
risparmiare di più**, cioè quelli che ricorrono spesso. E il modo per
scoprirli è guardare un mucchio di testo e contare.

`````

`````{tab} Superiore

Sia $V$ il vocabolario e $|V|$ la sua taglia. Due quantità dipendono da $|V|$
in verso opposto.

La prima è la **lunghezza media della sequenza** dopo la tokenizzazione, la
*fertilità* del tokenizzatore (token prodotti per parola di testo). Cresce al
ridursi di $|V|$: nel limite dei caratteri vale la lunghezza media in lettere,
nel limite delle parole intere vale $1$. La lunghezza $n$ della sequenza è la
grandezza che governa il costo dell'attenzione, $O(n^2 d_{\text{model}})$, e
il consumo della finestra di contesto.

La seconda è il **numero di parametri legati al vocabolario**. La matrice di
embedding ha forma $|V| \times d_{\text{model}}$, e altrettanto la matrice di
proiezione finale se non è condivisa con essa. Con $|V| = 50\,000$ e
$d_{\text{model}} = 4096$ sono $204{,}8$ milioni di parametri per la sola
tabella degli embedding: circa otto volte i parametri di una ResNet-50 intera
(attorno ai 25 milioni), spesi solo per dare un vettore a ciascun token.
Inoltre la softmax finale corre su $|V|$ classi, e il suo costo cresce
linearmente con la taglia.

Le taglie usate in pratica (da qualche decina di migliaia a qualche centinaio
di migliaia di token) stanno dove queste due curve si incontrano, e la scelta
si sposta verso l'alto quando il modello deve coprire molte lingue. Nessun
algoritmo di questa sezione *sceglie* $|V|$: è un iperparametro, e ciascuno si
limita a riempire i posti disponibili nel modo che ritiene migliore.

`````

## Byte Pair Encoding: da compressore a tokenizzatore

Il primo e più usato di questi algoritmi non nasce nella linguistica
computazionale, ma nella compressione dei dati. Nel febbraio 1994 Philip Gage
pubblica sul *C Users Journal* un algoritmo semplicissimo
{cite}`gage1994new`: trova la coppia di byte adiacenti più frequente nel file,
sostituiscila ovunque con un byte inutilizzato, annota la sostituzione in una
tabella, ripeti. Alla fine il file è più corto e la tabella dice come
ricostruirlo. Il nome che Gage gli dà è **byte pair encoding**, BPE.

```{figure} ../figures/tokenizzazione-bpe.svg
:name: fig-tokenizzazione-bpe
:alt: "Catena in tre stadi. La parola «straordinariamente», che nel vocabolario non esiste per intero, viene spezzata in tre sottoparole; ciascuna sottoparola viene poi convertita nel proprio identificativo numerico, e il modello riceve i tre numeri."
:width: 92%

Cosa arriva davvero al modello. Una parola lunga e rara non entra nel
vocabolario intera: si ricompone da pezzi che ci sono, e ogni pezzo diventa un
numero.
```

La catena di {numref}`fig-tokenizzazione-bpe` chiarisce un equivoco diffuso:
il modello non vede mai le parole, né i caratteri. Vede identificativi
numerici, e il confine fra un identificativo e l'altro lo ha deciso un
algoritmo di frequenze, non la grammatica.

Una ventina d'anni dopo, nel 2015, Rico Sennrich, Barry Haddow e Alexandra Birch
{cite}`sennrich2016neural` si accorgono che quell'algoritmo di compressione
risolve un problema completamente diverso: la traduzione automatica delle
parole rare. Cambiano una cosa sola, cioè fondono caratteri (e sequenze di
caratteri) invece che byte, e invece di comprimere si fermano quando il
vocabolario ha raggiunto la taglia voluta. Nel loro articolo il numero di
fusioni è, testualmente, l'unico iperparametro dell'algoritmo. Il lavoro viene
presentato ad ACL nel 2016 ed è, ancora oggi, la base di quasi tutti i
tokenizzatori in circolazione.

`````{tab} Elementare

Il meccanismo si racconta in quattro righe.

1. Spezzate tutte le parole del corpus nelle loro lettere. Il vocabolario di
   partenza sono le lettere, e basta.
2. Guardate tutto il corpus e contate **quali due simboli vicini compaiono
   insieme più spesso**. In italiano sarà qualcosa come `ss`, o `ch`, o `zi`.
3. Incollateli in un simbolo nuovo, che da adesso conta come un pezzo solo, e
   segnatevi la fusione su un elenco.
4. Tornate al punto 2, e ripetete finché il vocabolario è grande quanto
   volete.

Alla fine avete due cose: un elenco di pezzi (il vocabolario) e, soprattutto,
l'**elenco ordinato delle fusioni**. Il secondo è più importante del primo,
perché è la ricetta. Per tokenizzare una parola nuova non serve cercarla da
nessuna parte: la si spezza in lettere e le si riapplicano le stesse fusioni,
nello stesso ordine in cui erano state imparate. Se la parola contiene pezzi
familiari, si ricompongono da soli; se non ne contiene nessuno, resta una fila
di lettere. In nessun caso resta fuori.

`````

`````{tab} Superiore

Sia $C$ il corpus, rappresentato come multiinsieme di parole con le loro
frequenze, e sia $\Sigma$ l'alfabeto dei caratteri che vi compaiono. Ogni
parola è inizialmente una sequenza di simboli in $\Sigma$. A ogni passo si
sceglie

$$
(a^\star, b^\star) \;=\; \arg\max_{(a,b)} \ \mathrm{freq}(ab),
$$

dove $\mathrm{freq}(ab)$ è il numero di occorrenze della coppia adiacente
$(a,b)$ nell'intero corpus, ciascuna pesata per la frequenza della parola che
la contiene. Si sostituisce ogni occorrenza di $(a^\star,b^\star)$ con il
simbolo concatenato $a^\star b^\star$, che entra nel vocabolario, e la coppia
viene accodata alla lista delle fusioni $M = (m_1, \dots, m_k)$. Il processo
si arresta quando $|\Sigma| + k$ raggiunge la taglia $|V|$ desiderata: il
numero di fusioni è quindi $k = |V| - |\Sigma|$, e il modello finale è la
coppia $(\Sigma, M)$.

La codifica di una stringa mai vista è il **replay** della lista: si parte dai
caratteri e si applicano $m_1, \dots, m_k$ in quest'ordine. L'ordine è
sostanziale, non convenzionale: una fusione tardiva può operare solo su
simboli prodotti da quelle precedenti, e invertirne due dà in generale una
segmentazione diversa. La procedura è deterministica e priva di ricerca: BPE
non cerca la segmentazione ottima di una parola, ma quella che le sue fusioni
producono, il che è una proprietà da tenere a mente quando i risultati
sorprendono.

Sul costo: contare le coppie da zero a ogni passo costa $O(N)$ con $N$
lunghezza totale del corpus in simboli, per un totale $O(Nk)$. Le
implementazioni serie non lo fanno: mantengono un indice dalla coppia alle
posizioni in cui compare e aggiornano solo i conteggi toccati dalla fusione,
con una coda di priorità sulle frequenze. L'addestramento resta comunque
un'operazione da fare una volta sola. La codifica di una parola di $L$
caratteri con il replay ingenuo costa $O(kL)$, ma in pratica si tiene una
cache parola $\to$ token e il costo ammortizzato crolla, perché la
distribuzione delle parole è fortemente sbilanciata.

`````

Una precisazione tecnica che eviterà confusione più avanti. Così com'è
descritto, BPE non sa dove finisce una parola: `basso` e `bassotto` gli
arrivano come due sequenze separate, e i pezzi imparati non portano traccia
della loro posizione. Le implementazioni reali aggiungono un marcatore, un
simbolo di fine parola come `</w>` nel lavoro originale, oppure un marcatore
di *inizio* attaccato allo spazio, che è la strada di SentencePiece e che
vedremo fra poco. Nell'esempio che segue lo omettiamo per non appesantire i
conti.

## L'esempio svolto: cinque parole, quattro fusioni

Tutto questo diventa chiaro solo facendo i conti a mano. Prendiamo un corpus
giocattolo di cinque parole italiane, con le loro frequenze, scelte perché si
somigliano abbastanza da condividere pezzi:

| parola | frequenza |
|---|---|
| `basso` | 6 |
| `bassotto` | 2 |
| `bosso` | 3 |
| `rosso` | 9 |
| `rossetto` | 5 |

In tutto sono $6 \cdot 5 + 2 \cdot 8 + 3 \cdot 5 + 9 \cdot 5 + 5 \cdot 8 = 146$
caratteri. Ogni parola parte spezzata nelle sue lettere: `b a s s o`,
`b a s s o t t o`, e così via.

**Passo 1.** Contiamo ogni coppia adiacente, pesandola con la frequenza della
parola. La coppia `s`+`s` compare una volta in ciascuna delle cinque parole,
quindi vale $6+2+3+9+5 = 25$; la coppia `s`+`o` compare in tutte tranne
`rossetto` (dove alla doppia s segue una e), quindi $6+2+3+9 = 20$; la coppia
`r`+`o` solo in `rosso` e `rossetto`, $9+5 = 14$. L'elenco completo:

| coppia | conteggio | dove compare |
|---|---|---|
| `s` `s` | **25** | in tutte e cinque le parole |
| `s` `o` | 20 | tutte tranne `rossetto` |
| `o` `s` | 17 | `bosso`, `rosso`, `rossetto` |
| `r` `o` | 14 | `rosso`, `rossetto` |
| `b` `a` | 8 | `basso`, `bassotto` |
| `a` `s` | 8 | `basso`, `bassotto` |
| `t` `t` | 7 | `bassotto`, `rossetto` |
| `t` `o` | 7 | `bassotto`, `rossetto` |
| `s` `e` | 5 | `rossetto` |
| `e` `t` | 5 | `rossetto` |
| `b` `o` | 3 | `bosso` |
| `o` `t` | 2 | `bassotto` |

Vince `s`+`s` con 25. Prima fusione: **`ss`**. Il corpus diventa
`b a ss o`, `b a ss o t t o`, `b o ss o`, `r o ss o`, `r o ss e t t o`.

**Passo 2.** Si ricontano le coppie sulla nuova segmentazione. Ora `ss` è un
simbolo unico, e le coppie che lo coinvolgono sono `ss`+`o` (in `basso`,
`bassotto`, `bosso`, `rosso`: $6+2+3+9 = 20$) e `o`+`ss` (in `bosso`, `rosso`,
`rossetto`: $3+9+5 = 17$). Le altre non sono cambiate: `r o` resta 14, `b a` e
`a ss` valgono 8, `t t` e `t o` valgono 7. Vince `ss`+`o` con 20. Seconda
fusione: **`sso`**.

**Passo 3.** Quattro parole su cinque contengono ora il simbolo `sso`:
`b a sso`, `b a sso t t o`, `b o sso`, `r o sso`; `rossetto` è rimasta
`r o ss e t t o` perché lì alla doppia s non segue una o. I conteggi:
`r`+`o` vale 14 (in `rosso` e `rossetto`), `o`+`sso` vale $3+9 = 12$,
`b`+`a` e `a`+`sso` valgono 8 a testa. Vince `r`+`o` con 14. Terza fusione:
**`ro`**.

**Passo 4.** Adesso succede la cosa interessante. La coppia più frequente è
`ro`+`sso`, con 9, cioè tutte e sole le occorrenze di `rosso`, e la fusione
produce **`rosso`**: una parola intera diventa un singolo token. Non c'è nulla
di speciale nella regola, è sempre la stessa; è la parola più frequente del
corpus, quindi i suoi pezzi si trovano insieme più spesso di chiunque altro e
si saldano per primi. È il motivo per cui, nei tokenizzatori veri, *casa* o
*the* sono un token solo mentre *ortogonalizzazione* ne prende cinque.

Dopo quattro fusioni il vocabolario contiene le sette lettere del corpus
(`a b e o r s t`) più `ss`, `sso`, `ro`, `rosso`. Al passo successivo si
presenterebbe un pareggio, `b`+`a` e `a`+`sso` a quota 8, e serve una regola
di spareggio: la fissiamo in modo esplicito nel codice qui sotto (a parità di
conteggio, la coppia prima in ordine alfabetico), perché un tokenizzatore deve
essere riproducibile bit per bit.

### La parola mai vista

Il collaudo è tokenizzare qualcosa che nel corpus non c'era. Prendiamo
`bassetto`. Si parte dalle lettere, `b a s s e t t o`, e si riapplicano le
fusioni imparate **nell'ordine in cui sono state imparate**. Con le prime
quattro: `ss` si applica (`b a ss e t t o`), `sso` no (dopo la doppia s c'è
una e), `ro` no, `rosso` no. Con dieci fusioni, come nel codice della prossima
sezione, entrano in gioco anche `to`, `tto` e `etto`, e il risultato è

```
b | a | ss | etto
```

Quattro token, nessun `<UNK>`, e due dei quattro sono pezzi imparati. C'è però
un dettaglio che merita attenzione, perché smonta un equivoco comune: nel
vocabolario, a quel punto, c'è il token `basso`, eppure in `bassetto` la `b` e
la `a` restano due token separati. Il motivo è che l'unica strada per cui
quelle due lettere si saldano passa per `a`+`sso` e poi `b`+`asso`, e in
`bassetto` dopo la doppia s non c'è una o: la catena si spezza al primo anello,
e la coppia `b`+`a`, che pure è frequente, non è mai stata imparata come
fusione a sé. **BPE non cerca la scomposizione migliore: riapplica una
ricetta.** La segmentazione che ne esce somiglia spesso alla morfologia, ma non
è morfologia, e quando le due divergono vince la ricetta.

Un caso più estremo: un cognome come `rossellini`, mai visto, diventa

```
ross | e | l | l | i | n | i
```

sette token per una parola sola. Nessuna informazione persa, ma un costo alto:
è il prezzo che le sotto-parole fanno pagare a ciò che è raro, ed è la radice
di due delle conseguenze pratiche di cui parleremo alla fine.

## Trenta righe di Python

L'algoritmo è abbastanza piccolo da entrare in una pagina, senza librerie.
Conta le coppie, fonde la vincitrice, ripete; poi riapplica le fusioni a una
parola nuova.

```python
from collections import Counter

# corpus giocattolo: parola -> quante volte compare
corpus = {"basso": 6, "bassotto": 2, "bosso": 3, "rosso": 9, "rossetto": 5}


def conta_coppie(pezzi, corpus):
    """Frequenza di ogni coppia adiacente, pesata sulle occorrenze della parola."""
    coppie = Counter()
    for parola, simboli in pezzi.items():
        for coppia in zip(simboli, simboli[1:]):
            coppie[coppia] += corpus[parola]
    return coppie


def fondi(simboli, coppia):
    """Sostituisce ogni occorrenza della coppia con il simbolo unito."""
    uniti, i = [], 0
    while i < len(simboli):
        if i < len(simboli) - 1 and (simboli[i], simboli[i + 1]) == coppia:
            uniti.append(simboli[i] + simboli[i + 1])
            i += 2
        else:
            uniti.append(simboli[i])
            i += 1
    return tuple(uniti)


def addestra(corpus, n_fusioni):
    pezzi = {parola: tuple(parola) for parola in corpus}   # si parte dai caratteri
    fusioni = []
    for _ in range(n_fusioni):
        coppie = conta_coppie(pezzi, corpus)
        if not coppie:
            break
        # la piu' frequente; a parita' di conteggio, la prima in ordine alfabetico
        coppia = min(coppie, key=lambda c: (-coppie[c], c))
        fusioni.append(coppia)
        pezzi = {p: fondi(s, coppia) for p, s in pezzi.items()}
    return fusioni


def tokenizza(parola, fusioni):
    """Riapplica le fusioni imparate, nello stesso ordine."""
    simboli = tuple(parola)
    for coppia in fusioni:
        simboli = fondi(simboli, coppia)
    return simboli


iniziali = {parola: tuple(parola) for parola in corpus}
print("coppie al primo passo:", conta_coppie(iniziali, corpus).most_common(5))

fusioni = addestra(corpus, 10)
for i, (a, b) in enumerate(fusioni, 1):
    print(f"{i:2d}. {a} + {b} -> {a + b}")

print("bassetto   ->", tokenizza("bassetto", fusioni))
print("rossellini ->", tokenizza("rossellini", fusioni))
```

L'output, che è poi il modo per verificare che i conti a mano fossero giusti:

```
coppie al primo passo: [(('s', 's'), 25), (('s', 'o'), 20), (('o', 's'), 17), (('r', 'o'), 14), (('b', 'a'), 8)]
 1. s + s -> ss
 2. ss + o -> sso
 3. r + o -> ro
 4. ro + sso -> rosso
 5. a + sso -> asso
 6. b + asso -> basso
 7. t + o -> to
 8. t + to -> tto
 9. e + tto -> etto
10. ro + ss -> ross
bassetto   -> ('b', 'a', 'ss', 'etto')
rossellini -> ('ross', 'e', 'l', 'l', 'i', 'n', 'i')
```

Le prime quattro fusioni sono esattamente quelle calcolate a mano, con gli
stessi conteggi. Cambiate le frequenze del corpus e cambierà l'ordine: è tutto
qui il "sapere" di un tokenizzatore, ed è tutto ereditato dal testo su cui è
stato addestrato. Tenetelo a mente, perché è la chiave di quasi tutto quello
che segue.

## WordPiece: non il più frequente, il meno casuale

BPE ha un difetto che si vede bene nell'esempio appena svolto. Ha fuso per
prima la coppia `ss` perché la `s` è una lettera comunissima e le doppie
italiane sono ovunque: la coppia è frequente soprattutto perché i suoi pezzi
lo sono. Ma "frequente" e "significativo" non sono la stessa cosa. Nel nostro
corpus la `a` compare **solo** dopo la `b`: `ba` è una coppia rara in assoluto
(8 occorrenze contro 25), ma è una coppia che non capita mai per caso.

Da qui il criterio alternativo di **WordPiece**, introdotto da Mike Schuster e
Kaisuke Nakajima nel 2012 per la ricerca vocale in giapponese e coreano
{cite}`schuster2012japanese` e diventato noto anni dopo come il tokenizzatore
di BERT {cite}`devlin2019bert`. La struttura dell'algoritmo è identica a
quella di BPE, si parte dai caratteri e si fonde una coppia per volta fino a
riempire il vocabolario. Cambia solo *quale* coppia si sceglie.

`````{tab} Elementare

Immaginate di spulciare un giornale contando quali parole compaiono vicine.
Trovate spesso «di» seguito da «un»: capita in continuazione, ma non vuol dire
niente, capita perché entrambe sono parole comunissime. Trovate anche «acqua»
seguito da «minerale»: molto più raro in assoluto, eppure ogni volta che
leggete «minerale» prima c'era «acqua». La seconda coppia dice qualcosa; la
prima è rumore di fondo.

WordPiece fa esattamente questa distinzione. Invece di chiedersi «quante volte
questi due pezzi si trovano attaccati?», si chiede: «si trovano attaccati **più
di quanto ci si aspetterebbe** se si fossero incontrati per caso?». Il conto è
semplice: si prende quante volte la coppia compare e la si divide per quanto
sono comuni i due pezzi presi singolarmente. Un pezzo che è dappertutto viene
penalizzato, e le sue coppie devono essere davvero frequenti per vincere.

Nel nostro corpus di cinque parole, con questo criterio, la prima fusione non è
più `ss` ma `ba`: la `s` è talmente diffusa che le sue coppie non stupiscono
nessuno, mentre la `a`, che si presenta sempre e solo dopo la `b`, è una
compagnia troppo fedele per essere una coincidenza. In una frase: BPE premia
ciò che **ricorre**, WordPiece premia ciò che **sta insieme per una ragione**.

`````

`````{tab} Superiore

Il criterio del lavoro originale è: a ogni passo si aggiunge al vocabolario
l'unità ottenuta combinandone due esistenti che **aumenta di più la
verosimiglianza dei dati** sotto il modello di linguaggio. Nella forma in cui
l'algoritmo si è poi diffuso quel modello è un **unigramma**, cioè un modello
che assegna a una segmentazione $x = (x_1, \dots, x_\ell)$ la probabilità
$P(x) = \prod_i p(x_i)$ con $p$ stimata per frequenza relativa. Sotto questo
modello, il guadagno esatto di log-verosimiglianza di una fusione cresce
(circa) come $\mathrm{freq}(ab)$ volte il logaritmo di quanto la coppia è più
frequente del previsto: pesa cioè anche *quante volte* la fusione si applica.
Il criterio adottato in pratica lascia cadere quel peso e valuta il guadagno
*per occorrenza*; un'euristica ispirata alla verosimiglianza, più che una sua
conseguenza:

$$
(a^\star, b^\star) \;=\;
\arg\max_{(a,b)} \ \frac{\mathrm{freq}(ab)}{\mathrm{freq}(a)\cdot\mathrm{freq}(b)},
$$

dove $\mathrm{freq}(a)$ è il numero di occorrenze del simbolo $a$ nel corpus
segmentato allo stato corrente e $\mathrm{freq}(ab)$ quello della coppia
adiacente. È una trasformazione monotona della **informazione mutua puntuale**
(PMI): passando dalle frequenze assolute a quelle relative, il rapporto diventa
$\frac{p(ab)}{p(a)\,p(b)}$ a meno di un fattore che dipende solo dalla taglia
del corpus, quindi costante entro un singolo passo e ininfluente
sull'$\arg\max$. Quel rapporto è l'esponenziale della PMI fra $a$ e $b$: vale
$1$ quando i due simboli si incontrano esattamente come farebbero per caso, più
di $1$ quando si attirano.

Sul corpus giocattolo dell'esempio svolto, al primo passo (frequenze dei simboli:
`s` 50, `o` 44, `t` 14, `r` 14, `b` 11, `a` 8, `e` 5, su 146 caratteri):

| coppia | $\mathrm{freq}(ab)$ | $\mathrm{freq}(a)$ | $\mathrm{freq}(b)$ | punteggio |
|---|---|---|---|---|
| `b a` | 8 | 11 | 8 | **0,0909** |
| `e t` | 5 | 5 | 14 | 0,0714 |
| `t t` | 7 | 14 | 14 | 0,0357 |
| `r o` | 14 | 14 | 44 | 0,0227 |
| `s s` | 25 | 50 | 50 | 0,0100 |

La coppia più frequente in assoluto, `s s`, scende all'ottavo posto su dodici;
vince `b a`, che ha un terzo delle occorrenze ma due componenti rari. Il
denominatore è esattamente il correttivo: penalizza le coppie che devono la
loro frequenza alla diffusione dei pezzi.

Sul piano pratico, nell'implementazione resa popolare da BERT i pezzi **non
iniziali** portano il prefisso `##`, così che `bassetto` possa uscire come
`bass`, `##etto` e la ricomposizione sia priva di ambiguità (`##etto` non è lo
stesso token di `etto` a inizio parola: la distinzione fra prefissi e suffissi
è codificata nel vocabolario, non ricostruita a valle; il lavoro del 2012
otteneva lo stesso effetto con un marcatore di spazio attaccato alle unità).
BERT usa un vocabolario di circa $30\,000$ token così costruiti. In fase di
codifica, inoltre, quella implementazione non replica una lista di fusioni ma
applica una scansione **greedy del prefisso più lungo** presente nel
vocabolario: differenza sottile che può produrre segmentazioni diverse da
quelle che il replay di BPE darebbe a parità di vocabolario.

`````

## SentencePiece e i byte: togliere gli ultimi presupposti

BPE e WordPiece, così come li abbiamo descritti, danno per scontata una cosa:
che il testo arrivi **già diviso in parole**. Entrambi partono da un
dizionario parola $\to$ frequenza, e quel dizionario qualcuno lo deve
costruire, tagliando il testo sugli spazi e sulla punteggiatura. È un
presupposto innocuo in italiano e in inglese, e falso in giapponese, in cinese
e in thailandese, dove gli spazi fra le parole semplicemente non ci sono. In
quelle lingue serve un segmentatore addestrato a parte, che diventa un pezzo
in più da mantenere e una fonte in più di errori.

**SentencePiece**, presentato da Taku Kudo e John Richardson nel 2018
{cite}`kudo2018sentencepiece`, toglie il presupposto nel modo più diretto
possibile: non tratta il testo come una lista di parole, ma come un **flusso
grezzo di caratteri** in cui lo spazio è un carattere come gli altri.

`````{tab} Elementare

Il trucco è quasi banale, e per questo bello. Prima di tokenizzare, ogni
spazio viene sostituito da un simbolo visibile, `▁` (una barretta bassa, non
un trattino di sottolineatura), e attaccato alla parola che segue. La frase
«il gatto nero» diventa la stringa `▁il▁gatto▁nero`, senza più spazi veri: una
collana ininterrotta di simboli, su cui si può far girare BPE esattamente come
prima. In giapponese, dove gli spazi non ci sono, non cambia nulla: il flusso
era già ininterrotto.

Il guadagno più prezioso arriva in uscita. Per ricostruire il testo da una
lista di token non serve nessuna regola («metti uno spazio prima di *gatto* ma
non prima della virgola, e nemmeno dopo l'apostrofo»): basta incollare i pezzi
e ritrasformare ogni `▁` in uno spazio. Si riottiene il testo di partenza
**identico**, apostrofi e punteggiatura compresi (identico, cioè, a come lo si
era normalizzato all'inizio: se si è deciso di uniformare certi caratteri, quel
passo resta). Sembra un dettaglio da manutentori, e invece è la differenza fra
un sistema che restituisce il testo com'era e uno che lo restituisce quasi
com'era.

Resta un ultimo buco. Anche lavorando sui caratteri, i caratteri sono tanti:
Unicode ne definisce oltre centomila, e nessun corpus li contiene tutti. Un
ideogramma raro, un simbolo matematico, un'emoji nuova, e siamo di nuovo al
punto di partenza con un `<UNK>` in mano. La soluzione è scendere ancora di un
piano: sotto i caratteri ci sono i **byte**, e i byte sono 256. Non 256 nel
corpus, 256 *e basta*: qualunque cosa esista o esisterà, sul computer è una
sequenza di byte. Partendo da lì, il vocabolario di base copre tutto per
costruzione, e la parola «sconosciuto» esce definitivamente dal dizionario.

`````

`````{tab} Superiore

SentencePiece è insieme un formato e una libreria, e va guardato su due piani:
come tratta il testo in ingresso, e con quale algoritmo costruisce il
vocabolario.

Il primo piano è la **normalizzazione e la reversibilità**. L'input è trattato
come una sequenza Unicode, passata per una normalizzazione (NFKC nella
configurazione predefinita), in cui lo spazio è rimpiazzato dal meta-simbolo
`▁` (U+2581, LOWER ONE EIGHTH BLOCK) prefisso al segmento che segue. La
decodifica è la concatenazione dei token seguita dalla sostituzione inversa, e
vale l'identità

$$
\mathrm{decode}\bigl(\mathrm{encode}(\mathrm{norm}(x))\bigr)
= \mathrm{norm}(x),
$$

dove $x$ è il testo grezzo e $\mathrm{norm}$ la normalizzazione scelta. È la
proprietà che gli autori chiamano tokenizzazione *lossless*, e che nelle
pipeline basate su tokenizzatori dipendenti dalla lingua non è garantita,
perché la detokenizzazione è lì una collezione di regole ad hoc.

Il secondo piano è l'algoritmo di costruzione del vocabolario, dove
SentencePiece offre BPE e in alternativa il modello **unigram**
{cite}`kudo2018subword`, che procede al contrario: si parte da un vocabolario
candidato ampio e lo si **pota**. Fissato $V$, il modello è lo stesso
unigramma già incontrato con WordPiece, cioè assegna a una segmentazione
$x = (x_1,\dots,x_\ell)$ di una stringa la probabilità $P(x) = \prod_i p(x_i)$,
ma qui la verosimiglianza di una stringa $s$ è la somma su tutte le
segmentazioni possibili, $P(s) = \sum_{x \in S(s)} P(x)$. Le $p(x_i)$ si
stimano con
l'algoritmo EM; poi, per ogni token, si calcola quanto la verosimiglianza
totale calerebbe rimuovendolo, e si elimina la frazione di token meno utili.
Si itera fino alla taglia voluta. La segmentazione di una stringa nuova è
quella di massima probabilità, trovata con Viterbi in tempo lineare nella
lunghezza. Due proprietà distinguono unigram da BPE: è un modello
**probabilistico**, quindi sa dire *quanto* una segmentazione è buona e
campionarne di alternative (la *subword regularization*, che addestrando su
segmentazioni diverse della stessa frase fa da regolarizzatore), e non dipende
da un ordine di fusioni.

Una terza mossa, che SentencePiece non ha inventato ma che si combina con le
prime due, è il **BPE a livello di byte**, adottato da GPT-2
{cite}`radford2019language`: si applica BPE non ai caratteri Unicode (che sono
oltre centomila, un alfabeto di base già proibitivo) ma alla codifica UTF-8 del
testo. L'alfabeto di base ha allora esattamente $|\Sigma| = 256$ elementi, e il
tasso di `<UNK>` è **zero per costruzione**, su qualunque input: testo, codice
sorgente, dati binari mascherati. Il prezzo è che un carattere fuori ASCII
occupa da 2 a 4 byte, e se le sue sequenze non sono abbastanza frequenti da
meritare una fusione, un singolo ideogramma può costare più token di una parola
inglese intera. La
copertura universale non è gratis: si paga in lunghezza di sequenza, sempre
per le lingue meno rappresentate nel corpus di addestramento del
tokenizzatore.

`````

## Quattro conseguenze che incontrerete davvero

Fin qui la meccanica. Ma la ragione per cui vale la pena conoscerla è che il
tokenizzatore, che sembra un dettaglio di preprocessing, produce quattro
effetti visibili a chiunque usi un modello di linguaggio, e nessuno dei
quattro è una curiosità: sono tutti conseguenze dirette dell'algoritmo appena
descritto.

**Primo: i numeri si spezzano in modo irregolare, e l'aritmetica ne soffre.**
Le fusioni si scelgono per frequenza, e le cifre non fanno eccezione. Le
sequenze numeriche comuni in un corpus web (gli anni recenti, i numeri tondi,
`100`, `000`, le cifre singole) diventano token unici; quelle rare no. Il
risultato è che un numero non viene spezzato secondo il suo *valore
posizionale*, cioè in unità, decine, centinaia, ma secondo la frequenza delle
sue sottostringhe: numeri di lunghezza uguale possono ricevere segmentazioni
di forma completamente diversa, e la stessa cifra può trovarsi ogni volta in un
token diverso. Chiedere a un modello di sommare due numeri lunghi significa
chiedergli di allineare colonne che nella sua rappresentazione non sono
allineate. Non spiega da solo tutti gli errori di calcolo dei modelli, ma è un
contributo strutturale e riconosciuto: tanto che vari tokenizzatori recenti
forzano la segmentazione delle cifre, una per una o a gruppi fissi di tre,
proprio per restituire al modello una griglia regolare.

**Secondo: l'italiano costa più token dell'inglese, a parità di significato.**

```{figure} ../figures/italiano-costa-piu-token.svg
:name: fig-italiano-token
:alt: "La stessa frase scritta in inglese e in italiano, con i confini di token marcati sopra ciascuna. Nella versione inglese le parole restano quasi tutte intere e i token sono pochi; nella versione italiana molte parole risultano spezzate in due o tre pezzi, e il conto totale dei token è sensibilmente più alto."
:width: 96%

Stessa frase, due conti diversi. Le parole italiane si frammentano perché il
vocabolario è stato costruito su un corpus in prevalenza inglese, e i posti se
li sono presi le sottostringhe inglesi.
```

Come mostra {numref}`fig-italiano-token`, il costo non è metaforico: si paga
a token, in denaro e in finestra di contesto. Anche questa è aritmetica di
fusioni. Se il corpus su cui il tokenizzatore è
stato addestrato è in prevalenza inglese, le fusioni che "pagano" sono le
sottostringhe inglesi, e i posti nel vocabolario finiscono lì. Le parole
italiane vengono allora ricostruite con pezzi presi in prestito, e si
frammentano. Si aggiunge un secondo effetto, indipendente e cumulativo: le
lingue morfologicamente ricche moltiplicano le forme. *Gatto*, *gatta*,
*gatti*, *gatte*, *gattino*, *gattini* sono sei parole distinte da imparare, e
ognuna singolarmente più rara del corrispondente inglese *cat*, che sta al
posto di quasi tutte. Più rara vuol dire meno probabile che meriti un token
suo. Le conseguenze sono concrete e tutte nella stessa direzione: lo stesso
testo occupa più posti nella finestra di contesto, costa di più dove si paga a
token, e fa lavorare di più l'attenzione. Quest'ultima in modo non lineare: se
una lingua consuma il 50% di token in più, il costo quadratico dell'attenzione
su quel testo cresce di $1{,}5^2 = 2{,}25$ volte. È una disparità che non
nasce da una scelta contro l'italiano, ma dalla composizione di un corpus, e
che si corregge solo addestrando il tokenizzatore su dati più bilanciati.

**Terzo: uno spazio in più o in meno cambia i token.** Nei tokenizzatori
moderni lo spazio non è un separatore invisibile, è parte del token: `▁gatto`
e `gatto` sono due voci diverse del vocabolario, con due embedding diversi e
due statistiche diverse. Ne segue che un prompt che finisce con uno spazio
mette il modello in una condizione differente da uno che finisce senza, perché
la continuazione naturale del primo è un token *senza* la barretta iniziale,
che è la variante rara. È il motivo per cui uno spazio di troppo in coda a una
richiesta può degradare la risposta in modo apparentemente inspiegabile, e
perché nelle interfacce di completamento conviene non lasciarne. Non è
fragilità del modello: è che gli avete dato in ingresso una sequenza diversa
da quella che credevate.

**Quarto: il vocabolario si fissa prima dell'addestramento e non si cambia
dopo.** Questa è la conseguenza più vincolante, ed è architetturale. La
matrice di embedding ha una riga per token e la proiezione finale una colonna
per token: aggiungerne uno significa aggiungere parametri privi di
addestramento, e cambiare la segmentazione di un token esistente significa che
tutto ciò che il modello ha imparato su quella riga si riferisce a un'altra
cosa. Il tokenizzatore è quindi parte del modello quanto i suoi pesi, si
distribuisce insieme a essi, e se il dominio d'uso non era rappresentato nel
corpus su cui è stato costruito (una lingua minore, la notazione chimica, un
linguaggio di programmazione poco diffuso) quel testo resterà frammentato per
tutta la vita del modello. Si può porre rimedio solo riaddestrando, o almeno
estendendo il vocabolario e riadattando gli embedding nuovi: entrambe
operazioni costose, che è il motivo per cui la scelta del tokenizzatore va
fatta all'inizio e con calma.

## Un'idea che vale oltre il testo

Vale la pena chiudere allargando lo sguardo. Tutto quello che avete letto
serve a una cosa sola: costruire un **alfabeto discreto** su cui un modello
autoregressivo possa lavorare, cioè un insieme finito di simboli in cui
qualunque input si possa scrivere e da cui qualunque output si possa
ricomporre. Il testo ce l'aveva già mezzo pronto (i caratteri) e il lavoro è
stato scegliere i raggruppamenti giusti.

Altri segnali quell'alfabeto non ce l'hanno affatto. L'audio è un'onda
continua, e per darla in pasto a un Transformer bisogna prima inventarsi dei
simboli: è esattamente quello che fanno i codec neurali del capitolo
sull'audio, dove un quantizzatore vettoriale trasforma ogni frammento di suono
in un indice intero. Il problema è lo stesso di questa sezione, la soluzione è
diversa perché diversa è la materia prima. E la domanda che resta aperta, in
entrambi i casi, è se il testo e il suono debbano davvero passare per dei
simboli, o se un giorno i modelli lavoreranno direttamente sui byte grezzi.
Per ora la risposta è economica più che teorica: i simboli accorciano le
sequenze, e la lunghezza delle sequenze è ciò che si paga.

```{admonition} Da ricordare
:class: important
- Un vocabolario di parole intere produce `<UNK>` (informazione persa e non
  generabile), uno di caratteri allunga le sequenze e fa pagare il costo
  quadratico dell'attenzione: le **sotto-parole** sono il compromesso.
- **BPE** {cite}`sennrich2016neural` parte dai caratteri e fonde, una alla
  volta, la **coppia adiacente più frequente**. Il modello è la lista
  **ordinata** delle fusioni, e tokenizzare una parola nuova vuol dire
  riapplicarle nello stesso ordine: nessuna ricerca, nessuna ottimizzazione.
- **WordPiece** {cite}`schuster2012japanese` ha la stessa struttura ma sceglie
  la coppia che massimizza
  $\mathrm{freq}(ab)/(\mathrm{freq}(a)\,\mathrm{freq}(b))$: non ciò che ricorre,
  ma ciò che ricorre più di quanto ci si aspetterebbe dal caso.
- **SentencePiece** {cite}`kudo2018sentencepiece` tratta il testo come flusso
  grezzo con lo spazio marcato da `▁`: nessun bisogno di pre-segmentare in
  parole (il che lo rende usabile per cinese e giapponese, dove gli spazi fra
  le parole non esistono) e detokenizzazione esatta. Il
  **livello dei byte** elimina gli `<UNK>` per costruzione, perché i byte sono
  256.
- Le conseguenze si vedono a valle: **numeri** segmentati in modo irregolare
  (e aritmetica fragile), **lingue non inglesi** che consumano più token e più
  contesto, **spazi** che cambiano la sequenza in ingresso, e un vocabolario
  **congelato** prima dell'addestramento, perché è parte del modello quanto i
  suoi pesi.
```
