# Come si spezza il testo: il Byte Pair Encoding

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
e per esteso nel {doc}`capitolo sui Transformer </Transformers/overview>`: l’**attenzione**, cioè il modo in
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

Prima di guardare gli algoritmi conviene capire che cosa stiamo cercando di
rendere migliore, perché la scelta della taglia del vocabolario è un baratto
fra due costi che tirano in direzioni opposte.

`````{tab} Elementare

Quanti pezzi diversi servono, in una scatola di mattoncini, per costruire
qualunque parola italiana? Con le sole ventuno lettere dell'alfabeto si
costruisce tutto, ma ogni parola richiede cinque o sei pezzi e ci vuole
un’eternità. Con un pezzo già fatto per ogni parola del dizionario si
costruisce in un colpo solo, ma la scatola diventa enorme e comunque, il giorno
che serve un cognome o una parola inventata, non c'è.

Una scatola enorme, poi, si paga in due modi. Ogni pezzo vuole il suo
scomparto, e chi costruisce deve tenere a mente che cosa c'è in ciascuno: mille
scomparti, mille cose da ricordare; centomila scomparti, centomila. E per
scegliere il pezzo da mettere adesso si passa in rassegna la scatola intera,
quindi con dieci volte gli scomparti ci vuole dieci volte il tempo.

La soluzione è una scatola mista: i pezzi grandi per le cose che ricorrono
(*rosso*, *casa*, *-mente*, *-zione*) e le lettere singole come riserva per
tutto il resto. Quanti scomparti ci siano non lo decide l'algoritmo: il numero
si fissa prima di cominciare, qualche decina di migliaia di solito, e chi deve
cavarsela in venti lingue lo alza, perché ogni lingua porta i suoi pezzi. Il
problema diventa allora: quali pezzi grandi conviene tenere, dato che gli
scomparti sono contati? La risposta di tutti gli algoritmi che vedremo è la
stessa in spirito: si tengono i pezzi che fanno risparmiare di più, cioè quelli
che ricorrono spesso. E il modo per scoprirli è guardare un mucchio di testo e
contare.

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
di migliaia di token) stanno nella fascia in cui nessuno dei due costi domina
l'altro, e la scelta si sposta verso l'alto quando il modello deve coprire
molte lingue. Nessuno di questi algoritmi *sceglie* $|V|$: è un iperparametro,
e ciascuno si limita a riempire i posti disponibili nel modo che ritiene
migliore.

`````

## Byte Pair Encoding: da compressore a tokenizzatore

Il primo e più usato di questi algoritmi non nasce nella linguistica
computazionale, ma nel mestiere di far stare i file in meno spazio. Serve
allora una parola sola, che poi torna fino in fondo al capitolo: il **byte**.
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
2. Guardate tutto il corpus e contate quali due simboli vicini compaiono
   insieme più spesso. Su un testo italiano vero vincono coppie banali e
   frequentissime, `re` o `to`.
3. Incollateli in un simbolo nuovo, dappertutto dove compaiono: da adesso
   quello conta come un pezzo solo, e la fusione va segnata su un elenco.
4. Tornate a contare, e ripetete finché il vocabolario è grande quanto volete.
   I giri da fare sono la taglia della scatola meno le lettere con cui si è
   partiti: una scatola da 30.000 pezzi, con 100 lettere iniziali, vuole
   29.900 fusioni.

Contare è la parte lenta: a ogni giro si rilegge tutto il testo da capo, e i
giri sono migliaia. Chi ha fretta si tiene annotato in quali punti ogni coppia
compare, e dopo una fusione aggiorna solo quelli che sono cambiati. In un modo
o nell'altro è un lavoro che si fa una volta sola.

Alla fine avete due cose: un elenco di pezzi (il vocabolario) e, soprattutto,
l’**elenco ordinato delle fusioni**. Il secondo è più importante del primo,
perché è la ricetta. Per tokenizzare una parola nuova non serve cercarla da
nessuna parte: la si spezza in lettere e le si riapplicano le stesse fusioni,
nello stesso ordine in cui erano state imparate. Se la parola contiene pezzi
familiari, si ricompongono da soli; se non ne contiene nessuno, resta una fila
di lettere. Ripassare l'elenco intero per ogni parola sarebbe uno spreco,
perché le stesse poche parole tornano di continuo: il risultato si scrive
accanto alla parola la prima volta, e dalla seconda in poi si legge e basta.

Fuori non resta niente, purché la scatola contenga davvero tutte le lettere che
potranno arrivare: è un «purché» che pesa più di quanto sembri, e più avanti
vedremo come lo si toglie di mezzo per sempre.

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
sostanziale, non convenzionale: una fusione tardiva può agire su simboli che
solo le precedenti sanno produrre, e invertirne due dà in generale una
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
correre l'algoritmo fino a dieci, che è quello che fa il programma qui sotto.
Le sei che si aggiungono sono, in ordine, `a`+`sso` → `asso`,
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
quali buchi. Il programma qui sotto non lo fa, perché
riapplica le fusioni alla cieca, senza mai chiedersi se i simboli rimasti siano
noti: è un programma didattico, non un tokenizzatore di produzione.

Il punto vero è quello, però, e conviene metterlo per iscritto. L'affermazione
«con le sotto-parole non resta fuori niente» non è una proprietà
dell'algoritmo: è una **scommessa sull'alfabeto di partenza**. Si vince finché
il corpus di addestramento conteneva ogni carattere che potrà mai arrivare.
Con cinque parole la scommessa è persa in partenza; con un corpus vero è quasi
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
```
`````

La ricetta si può cambiare in un punto solo, il criterio con cui si sceglie la
coppia da incollare, ed è da lì che nascono [WordPiece, SentencePiece e i
byte](oltre-il-bpe.md), con le conseguenze che arrivano fino alla bolletta.
