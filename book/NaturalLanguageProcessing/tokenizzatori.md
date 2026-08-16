# Come si spezza il testo: BPE, WordPiece e SentencePiece

Nella sezione precedente abbiamo dato per buono che il testo si tagli in
token, e abbiamo detto che i sistemi moderni usano pezzi più piccoli della
parola. Resta però la domanda che conta davvero, ed è una domanda di
ingegneria: **quali** pezzi? Nessuno decide a mano che *tokenizzazione* vada
spezzata in `token` + `izzazione`. Quella decisione è il risultato di un
algoritmo che ha letto un corpus enorme e ha scelto, uno per uno, i mattoncini
da tenere. Questa sezione apre quell'algoritmo.

Il punto di partenza è un vicolo cieco. Immaginate di riempire il vocabolario
di parole intere, e di dover decidere quante tenerne: cinquantamila, centomila,
un milione. Qualunque numero scegliate, quel vocabolario non si chiude mai,
perché prima o poi arriverà una parola che non c'è. Un cognome
(*Rossellini*), un refuso (*gattto*), un termine tecnico
(*ortogonalizzazione*), un composto tedesco costruito sul momento. Il sistema
la sostituisce con un simbolo speciale, `<UNK>` (da *unknown*, sconosciuto), e
da quel momento quella parola per il modello non esiste più: era il nome del
paziente, il numero di serie, il soggetto della frase, e adesso è un buco.
Peggio: questi programmi non si limitano a leggere, scrivono anche, e `<UNK>`
non si può *scrivere*. Un modello che lo tirasse fuori avrebbe prodotto un
buco, e nessuno saprebbe con che cosa riempirlo.

All'estremo opposto c'è la soluzione radicale: un vocabolario di singoli
caratteri. Le parole sconosciute spariscono per costruzione, perché ogni testo
è fatto di lettere che il vocabolario contiene tutte. Ma si paga due volte.

**La prima: le sequenze si allungano di brutto.** Una parola italiana media sta
attorno ai cinque o sei caratteri, quindi contare a lettere invece che a parole
allunga il testo di cinque volte. E la lunghezza si paga cara. Il motivo è un
meccanismo che incontreremo fra qualche sezione, con la traduzione automatica,
e per esteso nel capitolo sui Transformer: l’**attenzione**, cioè il modo in
cui un modello moderno guarda tutte le parole della frase insieme e decide
quali contano davvero. Per farlo confronta ogni posizione con ogni altra, e
quel conto cresce con il **quadrato** della lunghezza: raddoppiare le posizioni
quadruplica i confronti, e una sequenza cinque volte più lunga costa
venticinque volte il lavoro.

**La seconda: si spreca il modello.** Un modello ha una quantità finita di
numeri regolabili dentro di sé, ed è quella la sua capienza: tutto ciò che
impara deve stare lì. Partire dalle lettere vuol dire spenderne una parte per
riscoprire che *g*, *a*, *t*, *t*, *o* stanno spesso in quest'ordine, cioè per
riscoprire le parole, che gliele si poteva regalare in partenza.

Le **sotto-parole** (*subword*) stanno in mezzo, e sono il compromesso che ha
vinto: le parole frequenti restano intere (un token per *rosso*), quelle rare
si scompongono in pezzi noti, e niente resta fuori. Il resto della sezione
spiega come si costruisce, concretamente, quell'elenco di pezzi.

## Il dilemma del vocabolario

Prima di guardare gli algoritmi vale la pena capire che cosa stiamo cercando di
rendere migliore, perché la scelta della taglia del vocabolario non è un
dettaglio: è un baratto fra due costi che tirano in direzioni opposte.

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
computazionale, ma nel mestiere di far stare i file in meno spazio. Serve
allora una parola sola, che poi torna fino in fondo alla sezione: il **byte**.
Un byte è il mattoncino minimo con cui un computer scrive qualunque cosa, otto
caselle che valgono 0 o 1; le combinazioni possibili sono $2^8 = 256$, né una
di più né una di meno, e un file di testo non è che una fila di byte. Di quei
256 valori, però, un testo qualunque ne usa solo una parte, quelli che
corrispondono alle lettere, alle cifre e alla punteggiatura che contiene: tutti
gli altri restano liberi, e sono i «byte inutilizzati» della ricetta che segue.

Nel febbraio 1994 Philip Gage pubblica sul *C Users Journal* un algoritmo
semplicissimo {cite}`gage1994new`: trova la coppia di byte adiacenti più
frequente nel file, sostituiscila ovunque con uno di quei byte liberi, annota
la sostituzione in una tabella, ripeti. Alla fine il file è più corto (dove
c'erano due byte adesso ce n'è uno) e la tabella dice come ricostruirlo. Il
nome che Gage gli dà è **byte pair encoding**, BPE.

```{figure} ../figures/tokenizzazione-bpe.svg
:name: fig-tokenizzazione-bpe
:alt: "Catena in tre stadi. La parola «straordinariamente», che nel vocabolario non esiste per intero, viene spezzata in tre sottoparole; ciascuna sottoparola viene poi convertita nel proprio identificativo numerico, e il modello riceve i tre numeri."
:width: 92%

Cosa arriva davvero al modello. Una parola lunga e rara non entra nel
vocabolario intera: si ricompone da pezzi che ci sono, e ogni pezzo diventa un
numero.
```

La catena di {numref}`fig-tokenizzazione-bpe` chiarisce un equivoco diffuso:
il modello non vede mai le parole, né i caratteri. Vede numeri. Quei numeri non
significano niente, sono i numeri di riga del vocabolario: una volta che
l'elenco dei pezzi è stato deciso, lo si scrive in ordine e il pezzo che sta
alla riga 4821 diventa, per il modello, «4821». È un codice d'inventario, come
il numero di scaffale in un magazzino, e il confine fra un pezzo e l'altro lo
ha deciso un algoritmo di frequenze, non la grammatica.

Una ventina d'anni dopo, nel 2015, Rico Sennrich, Barry Haddow e Alexandra Birch
{cite}`sennrich2016neural` si accorgono che quell'algoritmo di compressione
risolve un problema completamente diverso: la traduzione automatica delle
parole rare. Cambiano una cosa sola, cioè fondono caratteri (e sequenze di
caratteri) invece che byte, e invece di comprimere si fermano quando il
vocabolario ha raggiunto la taglia voluta. Nel loro articolo il numero di
fusioni è, testualmente, l'unico **iperparametro** dell'algoritmo: l'unica
manopola, cioè, che chi lo usa deve girare a mano, perché tutto il resto lo
decidono i conteggi. Il lavoro viene presentato nel 2016 alla conferenza ACL
ed è, ancora oggi, la base di quasi tutti i tokenizzatori in circolazione.

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
l’**elenco ordinato delle fusioni**. Il secondo è più importante del primo,
perché è la ricetta. Per tokenizzare una parola nuova non serve cercarla da
nessuna parte: la si spezza in lettere e le si riapplicano le stesse fusioni,
nello stesso ordine in cui erano state imparate. Se la parola contiene pezzi
familiari, si ricompongono da soli; se non ne contiene nessuno, resta una fila
di lettere. In nessun caso resta fuori, **purché la scatola contenga davvero
tutte le lettere che potranno arrivare**: è un «purché» che pesa più di quanto
sembri, e alla fine della sezione vedremo come lo si toglie di mezzo per
sempre.

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
descritto, BPE lavora su una parola per volta e non lascia sui pezzi nessuna
traccia di *dove* si trovavano. Il guaio si vede in uscita: il pezzo `to`
ritagliato dalla fine di `bassotto` e il pezzo `to` che apre una parola come
`tornare` sono, per il modello, la stessa identica voce del vocabolario, anche
se il primo è una desinenza e il secondo l'inizio di un verbo. Nel rimettere
insieme i pezzi, poi, non c'è modo di sapere dove finisce una parola e comincia
la successiva. Le implementazioni reali aggiungono perciò un marcatore: nel
lavoro originale è un simbolo di fine parola, `</w>`; altrove è un simbolo che
segna l’*inizio*, attaccato allo spazio che precede la parola, ed è la strada
di SentencePiece che vedremo fra poco. Nell'esempio che segue lo omettiamo per
non appesantire i conti.

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
caratteri (il totale tornerà utile fra qualche pagina, quando confronteremo
questo criterio con quello di WordPiece, che divide proprio per delle
frequenze). Ogni parola parte spezzata nelle sue lettere: `b a s s o`,
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

Il risultato dei quattro passi, e l'elenco che li registra, stanno tutti nella
{numref}`fig-bpe-fusioni`: a sinistra come sono ridotte le cinque parole a fine
corsa, a destra le quattro fusioni in ordine, ciascuna con il conteggio che le
ha fatte vincere.

```{figure} ../figures/bpe-fusioni.svg
:name: fig-bpe-fusioni
:alt: A sinistra le cinque parole del corpus giocattolo dopo quattro fusioni, ciascuna spezzata in scatole, una per token, con accanto la propria frequenza: basso è b, a, sso; bassotto è b, a, sso, t, t, o; bosso è b, o, sso; rosso è una scatola sola, evidenziata; rossetto è ro, ss, e, t, t, o. A destra le quattro fusioni in ordine con il loro conteggio: ss 25, sso 20, ro 14, rosso 9. In basso a sinistra il totale, 78 token contro i 146 caratteri di partenza.
:width: 96%

Il corpus dopo quattro fusioni, e l'elenco ordinato che le registra. La parola
più frequente, `rosso`, è finita in una scatola sola, e il testo è passato da
146 pezzi a 78.
```

Dopo quattro fusioni il vocabolario contiene le sette lettere del corpus
(`a b e o r s t`) più `ss`, `sso`, `ro`, `rosso`. Al passo successivo si
presenterebbe un pareggio, `b`+`a` e `a`+`sso` a quota 8, e serve una regola
di spareggio: la fissiamo in modo esplicito nel codice qui sotto (a parità di
conteggio, la coppia prima in ordine alfabetico). Non è pignoleria: un
tokenizzatore, rilanciato domani sullo stesso corpus, deve produrre esattamente
lo stesso vocabolario, altrimenti tutto ciò che il modello ha imparato punta ai
pezzi sbagliati.

### La parola mai vista

Il collaudo è tokenizzare qualcosa che nel corpus non c'era. Prendiamo
`bassetto`. Si parte dalle lettere, `b a s s e t t o`, e si riapplicano le
fusioni imparate **nell'ordine in cui sono state imparate**. Con le prime
quattro: `ss` si applica (`b a ss e t t o`), `sso` no (dopo la doppia s c'è
una e), `ro` no, `rosso` no.

Fermarsi a quattro fusioni sarebbe però un vocabolario ridicolo: lasciamo
correre l'algoritmo fino a dieci, che è quello che fa il programma della
prossima pagina. Le sei che si aggiungono sono, in ordine, `a`+`sso` → `asso`,
`b`+`asso` → `basso`, `t`+`o` → `to`, `t`+`to` → `tto`, `e`+`tto` → `etto`,
`ro`+`ss` → `ross`. Con queste in mano, `bassetto` esce così:

```
b | a | ss | etto
```

Quattro token, nessun `<UNK>`, e due dei quattro sono pezzi imparati. C'è però
un dettaglio che merita attenzione, perché smonta un equivoco comune. Nel
vocabolario, a quel punto, il token `basso` c'è (è la sesta fusione), eppure in
`bassetto` la `b` e la `a` restano due token separati. Il motivo è che l'unica
strada per cui quelle due lettere si saldano passa prima per `a`+`sso` e poi
per `b`+`asso`, e in `bassetto` dopo la doppia s non c'è una o: la prima delle
due fusioni non scatta, la catena si spezza al primo anello, e la coppia
`b`+`a`, che pure nel corpus è frequente, non è mai stata imparata come fusione
a sé. **BPE non cerca la scomposizione migliore: riapplica una ricetta.** La
segmentazione che ne esce somiglia spesso alla morfologia, ma non è morfologia,
e quando le due divergono vince la ricetta.

Un caso più estremo: un cognome come `rossellini`, mai visto, diventa

```
ross | e | l | l | i | n | i
```

sette token per una parola sola. È il prezzo che le sotto-parole fanno pagare a
ciò che è raro, e alla fine della sezione lo ritroveremo due volte: nei numeri,
che si spezzano a casaccio, e nelle lingue diverse dall'inglese, che si
frammentano più dell'inglese.

E c'è dell'altro, che conviene guardare in faccia invece di girarci intorno,
perché è il punto in cui la promessa «niente resta fuori» mostra la sua
condizione.

Guardate le lettere di `rossellini`. Le `l`, le `i` e la `n` nel nostro corpus
giocattolo non compaiono mai: quel corpus è fatto di cinque parole, e le
lettere che ci stanno dentro sono sette in tutto, `a b e o r s t`. Un
tokenizzatore vero, prima di consegnare un pezzo, controlla di averlo nel
vocabolario; e siccome quelle cinque lettere nel vocabolario non ci sono, al
posto loro metterebbe cinque `<UNK>`, uno per ciascuna. Sette token, cinque dei
quali buchi. Il programma della prossima pagina non lo fa soltanto perché
riapplica le fusioni alla cieca, senza mai chiedersi se i simboli rimasti siano
noti: è un programma didattico, non un tokenizzatore di produzione.

Il punto vero è quello, però, e vale la pena metterlo per iscritto.
L'affermazione «con le sotto-parole non resta fuori niente» non è una proprietà
dell'algoritmo: è una **scommessa sull'alfabeto di partenza**. Si vince finché
il corpus di addestramento conteneva ogni carattere che potrà mai arrivare. Con
cinque parole la scommessa è persa in partenza; con un corpus vero è quasi
sempre vinta, e a tradirla bastano un ideogramma raro o un'emoji uscita l'anno
scorso. È il buco che il livello dei byte, fra qualche pagina, chiuderà per
costruzione e per sempre.

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
più `ss` ma `ba`, e il conto si può rifare a mano con due divisioni. Nei 146
caratteri del corpus la `s` compare 50 volte e la coppia `ss` 25: il punteggio è
25 diviso (50 per 50), cioè 0,01. La `b` compare 11 volte, la `a` 8 e la coppia
`ba` 8: il punteggio è 8 diviso (11 per 8), cioè circa 0,09, nove volte tanto.
La `s` è talmente diffusa che le sue coppie non stupiscono nessuno, mentre la
`a`, che si presenta sempre e solo dopo la `b`, è una compagnia troppo fedele
per essere una coincidenza. In una frase: BPE premia ciò che **ricorre**,
WordPiece premia ciò che **sta insieme per una ragione**.

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
sull’$\arg\max$. Quel rapporto è l'esponenziale della PMI fra $a$ e $b$: vale
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
punto di partenza con un `<UNK>` in mano. È lo stesso buco che avevamo
intravisto con `rossellini`: se una lettera non era nel corpus di partenza,
niente la rappresenta.

La soluzione è scendere ancora di un piano: sotto i caratteri ci sono i
**byte**, quelli con cui si è aperta la sezione, e i byte sono 256. Ed è qui
che sta tutta la differenza: non sono 256 *nel corpus*, sono 256 *e basta*, e
non li ha scelti nessuno guardando dei testi. Qualunque cosa esista o
esisterà, sul computer è una sequenza di quei 256 mattoncini: un ideogramma ne
occuperà tre, un'emoji quattro, ma saranno sempre e solo quelli. Partendo da
lì, il vocabolario di base copre tutto per costruzione, la scommessa
sull'alfabeto non c'è più, e la parola «sconosciuto» esce definitivamente dal
dizionario.

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
l'algoritmo EM (lo stesso delle misture gaussiane, nel capitolo sul Machine
Learning: qui la variabile che renderebbe facile la stima e non si osserva non
è l'identità della componente, è la segmentazione);
poi, per ogni token, si calcola quanto la verosimiglianza
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
tokenizzatore, che sembra un dettaglio della preparazione dei dati, produce
quattro effetti visibili a chiunque usi un modello di linguaggio, e nessuno
dei quattro è una curiosità: sono tutti conseguenze dirette dell'algoritmo
appena descritto.

**Primo: i numeri si spezzano in modo irregolare, e l'aritmetica ne soffre.**
Le fusioni si scelgono per frequenza, e le cifre non fanno eccezione. Le
sequenze numeriche comuni sul web (gli anni recenti, i numeri tondi, `100`,
`000`, le cifre singole) si guadagnano un token tutto loro; quelle rare no.

Il guaio si vede con due numeri quasi uguali. Su un corpus in cui `2024` è
frequentissimo e `2025` meno, il primo può uscire come **un token solo** e il
secondo come **due**, `20` e `25`. Due numeri della stessa lunghezza, tagliati
in modo diverso, e la cifra `2` che nel primo caso sta dentro un pezzo unico e
nel secondo apre il pezzo `20`. Adesso pensate a come si fa una somma in
colonna a scuola: si incolonnano le unità sotto le unità, le decine sotto le
decine. Chiedere a un modello di sommare due numeri lunghi significa
chiedergli di incolonnare cifre che nella sua rappresentazione non sono
incolonnate affatto, perché è stato tagliato tutto secondo la frequenza e non
secondo il posto che ogni cifra occupa. Non spiega da solo tutti gli errori di
calcolo dei modelli, ma è un contributo strutturale e riconosciuto: tanto che
vari tokenizzatori recenti forzano la segmentazione delle cifre, una per una o
a gruppi fissi di tre, proprio per restituire al modello una griglia regolare.

**Secondo: l'italiano costa più token dell'inglese, a parità di significato.**

```{figure} ../figures/italiano-costa-piu-token.svg
:name: fig-italiano-token
:alt: "La stessa frase scritta in inglese e in italiano, una sopra l'altra, con i confini fra un token e l'altro marcati sopra ciascuna. In inglese, «The cat is on the table», le sei parole restano sei token interi. In italiano, «Il gatto è sopra il tavolo», due parole si spezzano in due pezzi ciascuna, «gatto» in «g» e «atto» e «tavolo» in «tav» e «olo», e i token diventano otto."
:width: 96%

Stessa frase, due conti diversi. Le parole italiane si frammentano perché il
vocabolario è stato costruito su un corpus in prevalenza inglese, e i posti se
li sono presi le sottostringhe inglesi.
```

Come mostra {numref}`fig-italiano-token`, il costo non è metaforico. Nasce da
due cause indipendenti che tirano nella stessa direzione.

La prima è la composizione del corpus. Se il testo su cui il tokenizzatore è
stato addestrato è in prevalenza inglese, le fusioni che «pagano» sono quelle
inglesi, e i posti nel vocabolario finiscono lì. Alle parole italiane restano i
pezzi avanzati, presi in prestito da altre parole, e si frammentano.

La seconda è la nostra grammatica. Le lingue che declinano e coniugano tutto
moltiplicano le forme: *gatto*, *gatta*, *gatti*, *gatte*, *gattino*,
*gattini* sono sei parole distinte, ciascuna delle quali va imparata per conto
suo, e ciascuna singolarmente più rara dell'inglese *cat*, che sta al posto di
quasi tutte. Più rara vuol dire meno probabile che si meriti un token suo.

E in che valuta si paga? In tre.

- **In denaro.** I servizi che danno accesso a un modello di linguaggio (un
  programma che, dato un testo, scommette su come continua: è il tema di una
  sezione più avanti) fanno pagare un tanto a token. La stessa richiesta
  scritta in italiano costa dunque più della stessa richiesta in inglese, e di
  quanto dipende dal tokenizzatore: nella frase della figura sono otto token
  contro sei, cioè un terzo in più.
- **In posti occupati.** Un modello può tenere davanti agli occhi solo una
  certa quantità di testo per volta, misurata anch'essa in token: è la sua
  **finestra di contesto**. Un documento che in inglese ci sta, in italiano può
  non starci.
- **In lavoro.** E qui il conto non è proporzionale, per via del costo
  quadratico dell'attenzione di cui si diceva all'inizio: se una lingua consuma
  il 50 per cento di token in più, l'attenzione su quel testo costa
  $1{,}5^2 = 2{,}25$ volte tanto.

È una disparità che non nasce da una scelta contro l'italiano, ma dalla
composizione di un corpus, e che si corregge solo addestrando il tokenizzatore
su dati più bilanciati.

**Terzo: uno spazio in più o in meno cambia i token.** Nei tokenizzatori
moderni lo spazio non è un separatore invisibile, è attaccato al token che lo
segue: `▁gatto` e `gatto` sono due voci diverse del vocabolario, con due
posizioni diverse sulla mappa e due storie diverse alle spalle. La prima è
comunissima, perché quasi sempre *gatto* è preceduto da uno spazio; la seconda
è rara, perché ricorre solo dove *gatto* attacca senza spazio davanti, cioè
quasi mai.

Ne segue una cosa che sorprende chiunque non l'abbia mai sentita. Il testo che
si scrive a un modello per farlo lavorare (una domanda, un'istruzione, l'inizio
di un documento da completare) si chiama **prompt**. Se il vostro prompt
finisce con uno spazio, quello spazio l'avete già speso voi, e al modello
tocca continuare con un token *senza* barretta iniziale, cioè con la variante
rara, quella su cui ha molta meno esperienza. Un solo carattere invisibile in
coda alla richiesta, e la risposta può peggiorare senza che si capisca il
perché. Non è fragilità del modello: è che gli avete dato in ingresso una
sequenza diversa da quella che credevate.

**Quarto: il vocabolario si fissa prima dell'addestramento e non si cambia
dopo.** Questa è la conseguenza più vincolante, e riguarda com'è fatto il
modello per dentro. In ingresso c'è una tabella con **una riga per ogni token
del vocabolario**: la riga contiene i numeri con cui quel pezzo di parola viene
rappresentato, ed è quella che nel resto del libro si chiama *matrice di
embedding*. In uscita ce n'è una seconda che fa il lavoro opposto, e infatti è
girata di novanta gradi: ha **una colonna per ogni token**, e serve a dare a
ciascun pezzo un punteggio per decidere quale scrivere. Una per entrare, una
per uscire. Aggiungere un token al vocabolario vuol dire allora aggiungere
una riga e una colonna vuote a due tabelle che l'addestramento ha già
riempito: numeri che nessuno ha mai regolato, in mezzo a numeri regolati per
mesi. E cambiare la segmentazione di un token esistente è peggio, perché
tutto ciò che il modello ha imparato su quella riga si riferisce ormai a
un'altra cosa. Il tokenizzatore è quindi parte del modello quanto i suoi pesi, si
distribuisce insieme a essi, e se il dominio d'uso non era rappresentato nel
corpus su cui è stato costruito (una lingua minore, la notazione chimica, un
linguaggio di programmazione poco diffuso) quel testo resterà frammentato per
tutta la vita del modello. Si può porre rimedio solo riaddestrando, o almeno
estendendo il vocabolario e riadattando gli embedding nuovi: entrambe
operazioni costose, che è il motivo per cui la scelta del tokenizzatore va
fatta all'inizio e con calma.

## Un'idea che vale oltre il testo

Vale la pena chiudere allargando lo sguardo. Tutto quello che avete letto
serve a una cosa sola: costruire un **alfabeto discreto** su cui possa lavorare
un modello che scrive un pezzo per volta, guardando quelli che ha già scritto
(è ciò che si intende con *autoregressivo*). Discreto vuol dire fatto di pezzi
separati e contabili, come le lettere di un alfabeto e non come le sfumature di
un colore: un insieme finito di simboli in cui qualunque testo in ingresso si
possa scrivere e da cui qualunque testo in uscita si possa ricomporre. Il testo
quell'alfabeto ce l'aveva già mezzo pronto (i caratteri) e il lavoro è stato
scegliere i raggruppamenti giusti.

Altri segnali quell'alfabeto non ce l'hanno affatto. L'audio è un'onda
continua, e per darlo in pasto allo stesso tipo di modello bisogna prima
inventarsi dei simboli. Il modo è più semplice di quanto sembri: ci si prepara
un catalogo fisso di frammenti sonori campione, diciamo mille, e poi si scorre
la registrazione un pezzetto alla volta, si cerca nel catalogo il campione che
somiglia di più a quello che si ha davanti, e al posto del suono si scrive il
suo numero di catalogo. L'onda diventa così una fila di numeri fra mille, cioè
un testo in un alfabeto di mille lettere. Questo mestiere ha un nome che
incontrerete nel capitolo sull'audio, ed è il **codec neurale**; il pezzo che
sceglie il campione più vicino si chiama *quantizzatore vettoriale*. Il
problema è lo stesso di questa sezione, la soluzione è diversa perché diversa è
la materia prima.

E la domanda che resta aperta, in entrambi i casi, è se il testo e il suono
debbano davvero passare per dei simboli, o se un giorno i modelli lavoreranno
direttamente sui byte grezzi. Per ora la risposta è economica più che teorica:
i simboli accorciano le sequenze, e la lunghezza delle sequenze è ciò che si
paga.

`````{tab} Elementare
```{admonition} Da ricordare
:class: important
- La **scatola dei mattoncini**: con le sole lettere si costruisce qualunque
  parola, ma ci vuole un'eternità; con un pezzo già fatto per ogni parola del
  dizionario si va veloci, ma la scatola non basta mai. I pezzi
  **sotto-parola** sono il compromesso che ha vinto: pezzi grandi per ciò che
  ricorre, lettere singole di riserva per tutto il resto.
- **BPE** parte dalle lettere e incolla ogni volta la coppia di pezzi vicini
  che compare più spesso, segnandosi la fusione su un elenco. Per spezzare una
  parola mai vista non la cerca da nessuna parte: riapplica l'elenco nello
  stesso ordine. Non cerca la scomposizione migliore, ripete una ricetta.
- **WordPiece** cambia una cosa sola: non incolla la coppia più frequente, ma
  quella che sta insieme più di quanto ci si aspetterebbe per caso («acqua
  minerale» contro «di un»).
- **SentencePiece** tratta il testo come una collana ininterrotta di simboli,
  con lo spazio scritto come `▁`: non c'è bisogno di tagliarlo prima in parole
  (indispensabile per cinese e giapponese, che gli spazi non li usano) e il
  testo si ricompone identico. Sotto i caratteri ci sono i **byte**, che sono
  256 comunque vada: partendo da lì non resta fuori più niente, mai.
- Le conseguenze si toccano con mano: **i numeri** vengono spezzati secondo la
  frequenza e non secondo il posto delle cifre, e i conti ne soffrono;
  **l'italiano costa più token dell'inglese**, cioè più soldi e più posti
  occupati nella finestra di contesto; **uno spazio di troppo** in fondo a una
  richiesta cambia davvero la domanda; e il vocabolario, una volta scelto,
  **non si cambia più**, perché fa parte del modello quanto i numeri che ha
  imparato.
```
`````

`````{tab} Superiore
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
  le parole non esistono) e detokenizzazione esatta. A livello di **carattere**
  l'assenza di `<UNK>` dipende da quali caratteri stavano nel corpus; il
  **livello dei byte** la rende una proprietà per costruzione, perché i byte
  sono 256 e basta.
- Le conseguenze si vedono a valle: **numeri** segmentati in modo irregolare
  (e aritmetica fragile), **lingue non inglesi** che consumano più token e più
  contesto, **spazi** che cambiano la sequenza in ingresso, e un vocabolario
  **congelato** prima dell'addestramento, perché è parte del modello quanto i
  suoi pesi.
```
`````
