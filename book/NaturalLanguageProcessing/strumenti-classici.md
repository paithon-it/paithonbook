# La cassetta degli attrezzi: espressioni regolari, normalizzazione e distanza di edit

Nell'Introduzione abbiamo incontrato ELIZA, il programma con cui Joseph
Weizenbaum dimostrò (suo malgrado) quanto sia facile attribuire
un'intelligenza a una macchina {cite}`weizenbaum1966eliza`. Conviene riaprire
il cofano: dentro non c'era nessuna comprensione del linguaggio, ma un gioco
di **pattern matching**, la ricerca di schemi nel testo. Se l'utente scriveva
«mi sento triste», ELIZA agganciava lo schema «mi sento X» e riassemblava i
pezzi in «Da quanto tempo ti senti X?», seguendo regole scritte a mano da
Weizenbaum stesso. Tutto qui.

Prima delle reti neurali che occuperanno il resto del capitolo, l'NLP era in
larga parte questo: schemi, regole, conteggi. Sarebbe però un errore liquidare
questi attrezzi come pezzi da museo. Sono ovunque, anche oggi: nel modulo di
iscrizione che ti avvisa che «l'indirizzo email non è valido», nei comandi con
cui i programmatori setacciano file da mezzo secolo, e soprattutto in quel
lavoro poco raccontato che è **rimettere in ordine i dati** prima di darli in
pasto a qualunque programma. Che poi vuol dire: togliere le righe doppie,
uniformare le date scritte in quattro modi, accorgersi che «Milano» e «MILANO»
sono la stessa città. Chiedete a chiunque lavori nel settore quanto tempo
porta via: è la parte più grossa di qualunque progetto reale. Prima di
insegnare a una rete neurale a leggere, conviene imparare a usare la cassetta
degli attrezzi.

```{figure} ../figures/nlp-classico-era-llm.svg
:name: fig-nlp-classico-vs-llm
:alt: "Due catene di lavoro sovrapposte, per lo stesso risultato. In alto la strada classica, in quattro stadi: si raccolgono i dati, li si etichetta a mano, si addestra un modello, lo si valuta; il cartellino dice «settimane». In basso la strada a prompt, in tre stadi: si scrive l'istruzione con qualche esempio, risponde un modello generalista che nessuno ha addestrato per quel compito, si controlla l'uscita; il cartellino dice «ore». Sotto, tre righe mettono a confronto i costi delle due strade."
:width: 100%

Due strade per lo stesso risultato, e due conti diversi. In alto quella
classica: dati raccolti, etichettati a mano, un modello addestrato apposta e
messo alla prova. In basso quella di oggi: si scrive l'istruzione in italiano,
e risponde un modello che per quel compito non è stato addestrato. La prima
chiede settimane prima di dare qualcosa e poi costa pochissimo a ogni testo;
la seconda parte in poche ore e paga a ogni chiamata.
```

Il testo che si scrive al modello della seconda strada si chiama **prompt**,
ed è la parola con cui si indica oggi qualunque richiesta fatta a un modello:
la ritroveremo alla fine del capitolo. E si paga «a ogni chiamata» perché
quel modello non sta sul vostro computer: sta sui server di qualcun altro, e
ogni volta che gli mandate un testo scatta il tassametro.

{numref}`fig-nlp-classico-vs-llm` mette a confronto tempi e costi, che è il
modo in cui la scelta si presenta a chi deve consegnare un lavoro. Ma c'è una
seconda ragione, e riguarda proprio gli attrezzi di questa sezione: si
spiegano in una riga, si ispezionano un passaggio alla volta e si correggono a
mano. Un modello vi dice che quell'indirizzo email è valido *quasi sempre*;
un'espressione regolare o combacia o non combacia, e se sbaglia potete aprirla
e vedere dove. Quando serve sapere *perché* è uscita una certa risposta, o
serve una garanzia invece di un «molto probabilmente», sono ancora loro a
stare dentro i sistemi moderni.

## Le espressioni regolari: descrivere uno schema, non una parola

Il primo attrezzo risponde a una domanda concreta: come si cerca in un testo
qualcosa che non è una parola precisa ma una *forma*? Tutte le date, tutti i
CAP, tutti gli importi in euro. La risposta si chiama **espressioni regolari** (*regular expressions*, o *regex*): la descrizione di
una *forma* invece che di una parola, e la si scrive con una riga di simboli.

Il nome è intimidatorio, e la storia è curiosa quanto basta a renderlo
simpatico. Le espressioni regolari nascono negli anni Cinquanta, e non nascono
per cercare nei testi: nascono per descrivere che cosa sa riconoscere una rete
di neuroni artificiali. Il logico Stephen Kleene stava studiando i modellini
matematici di neurone proposti nel 1943 da Warren McCulloch e Walter Pitts,
gli stessi da cui parte il capitolo sulle reti neurali, e per dire quali
sequenze di segnali una rete del genere sa distinguere si inventò questa
notazione. Conviene fermarsi un secondo su questo: gli attrezzi «vecchi» e
quelli «nuovi» di questo capitolo hanno lo stesso atto di nascita.

A portarle dentro i programmi fu Ken Thompson, uno dei padri del sistema
operativo Unix, alla fine degli anni Sessanta. Le mise in due programmi per
scrivere testo, QED e poi `ed`, e da lì viene il nome di `grep`, il comando che
i programmatori usano ancora oggi per cercare dentro un file digitandone il
nome invece che con il mouse. `grep` è la sigla di `g/re/p`, l'istruzione che
in `ed` voleva dire «cerca ovunque (`g`) l'espressione regolare (`re`) e stampa
quello che trovi (`p`)».

`````{tab} Elementare

In una caccia al tesoro l'indizio non dice «trova la parola 95125», dice «trova
cinque cifre di fila». Un'espressione regolare è esattamente questo: la
*descrizione di uno schema*, invece di una parola esatta. «Cinque cifre di
fila» trova tutti i CAP d'Italia; «una o due cifre, una barra, una o due cifre,
una barra, quattro cifre» trova tutte le date scritte come 3/7/2026; «la radice
*gatt-* seguita da una vocale» trova *gatto*, *gatta*, *gatti* e
*gatte* in un colpo solo.

È la funzione Trova del tuo editor di testi, ma con i superpoteri: invece di
controllare lettera per lettera, controlla *tipo* di lettera per tipo di
lettera (qui voglio una cifra, qui una lettera qualsiasi, qui uno spazio).
Quando un sito ti dice al volo che il numero di telefono che hai digitato non
è valido, nove volte su dieci c'è un'espressione regolare che ha confrontato
quello che hai scritto con lo schema atteso e ha trovato che non combacia.

C'è però una cosa che questi superpoteri non sanno fare, e non per distrazione.
Un'espressione regolare **non sa contare**. Scorre il testo da sinistra a
destra e a ogni carattere ricorda solo in che punto dello schema si trova, non
quante volte ci è già passata. Quindi non c'è modo di scriverne una che
verifichi «ogni parentesi aperta ne ha una chiusa» quando le parentesi si
possono annidare quanto si vuole: dovrebbe tenere il conto di quante ne ha
aperte, e non ha dove segnarlo. Vale per le parentesi e vale per le frasi
dentro le frasi, che sono la stessa cosa fatta di parole: «il gatto che dorme
sul divano che ho comprato quando…». Per quelle serviranno gli attrezzi delle
prossime sezioni.

Lo schema è una cosa, l'attrezzo che lo esegue un'altra. La passata unica
appena descritta è quella di `grep`, una lettura sola dall'inizio alla fine, e
il tempo cresce come la lunghezza del testo. Altri programmi lavorano
per tentativi: provano una strada, e se non porta da nessuna parte tornano al
bivio e ne provano un'altra. In cambio offrono qualcosa in più, per esempio
chiedere che un pezzo di testo si ripeta identico più avanti, ed è proprio quel
di più a costringerli ai tentativi. Quasi sempre finiscono comunque in un
lampo. Ma basta uno schema ambiguo, di quelli che chiedono «gruppi di una o più
lettere» dove le lettere sono tutte uguali, e le strade da provare si
moltiplicano. Tre lettere di fila si spezzano in gruppi in quattro modi, e ogni
lettera in più raddoppia il conto. Su una riga di poche decine di caratteri il
programma si pianta, e chi mette un'espressione regolare su un modulo aperto al
pubblico deve saperlo, perché quella riga gliela scrive uno sconosciuto.

`````

`````{tab} Superiore

Un'espressione regolare è una stringa che definisce un insieme di stringhe (un
*linguaggio*). I costrutti essenziali sono pochi:

| Costrutto | Significato | Esempio | Trova |
|---|---|---|---|
| `[oaie]` | una tra le lettere elencate | `gatt[oaie]` | *gatto*, *gatta*, … |
| `\d`, `\w`, `\s` | cifra, carattere di parola, spazio | `\d\d` | *42* |
| `*`, `+`, `?` | zero o più, una o più, opzionale | `carr?o` | *caro*, *carro* |
| `{n}`, `{n,m}` | esattamente $n$, da $n$ a $m$ ripetizioni | `\d{5}` | *95125* |
| `^`, `$`, `\b` | inizio riga, fine riga, confine di parola | `^Il` | *Il* a inizio riga |
| `(...)` | gruppo da catturare | `(\d+)/(\d+)` | giorno e mese, separati |
| `\|` | alternanza (oppure) | `gatto\|micio` | *gatto* o *micio* |

Dietro la sintassi c'è un teorema: i linguaggi descrivibili con espressioni
regolari sono esattamente quelli riconoscibili da un **automa a stati
finiti**, e l'automa scandisce il testo in tempo lineare nella sua lunghezza.
Il rovescio della medaglia è un limite espressivo preciso: un automa a stati
finiti non sa *contare*, quindi nessuna espressione regolare può verificare
strutture annidate a profondità arbitraria (parentesi bilanciate, subordinate
dentro subordinate). Per la sintassi delle lingue naturali servono strumenti
più potenti, o, come vedremo, modelli che la imparano dai dati.

Una cautela pratica prima di scendere al codice: il teorema, e con esso la
garanzia di tempo lineare, riguarda i motori che compilano davvero l'automa,
come `grep` o RE2. Il modulo `re` di Python, che useremo tra poco, procede
invece per *backtracking*, e i suoi costrutti aggiuntivi (le *backreference*)
descrivono anche linguaggi che regolari non sono: fuori dalla portata del
teorema, il costo non è più garantito, e su pattern patologici come `(a+)+$`
applicato a un input ostile il tempo può degradare fino a essere esponenziale.

`````

In Python le espressioni regolari vivono in una cassetta di funzioni già
pronte, che si chiama `re` e che il programma richiama con la prima riga,
`import re`. Prima di leggere il codice conviene avere sotto mano il minimo
dizionario per decifrarlo, perché sono cinque simboli e poi si legge tutto:
`\d` vuol dire «una cifra qualsiasi», `{5}` vuol dire «cinque volte quello che
precede», `{1,2}` vuol dire «una o due volte», le parentesi quadre elencano le
lettere ammesse (`[oaie]` = «una fra queste quattro») e `\b` segna il confine
di una parola, cioè impedisce di pescare un pezzo dentro una parola più lunga.
Con questo, `\b\d{5}\b` si legge «cinque cifre isolate», ed è un CAP. Una
convenzione e basta: la `r` che precede le virgolette, come in `r"\d{5}"`,
serve solo a dire a Python di prendere le barre rovesciate alla lettera invece
di interpretarle a modo suo.

Mettiamo alla prova la nostra frase preferita, arricchita di qualche dettaglio
da estrarre:

```python
import re

testo = ("Il gatto nero salta sul muro di via dei Tigli 42. La gatta lo "
         "guarda dal balcone: CAP 95125, visita dal veterinario il "
         "3/7/2026 alle 18:30.")

# tutte le forme di "gatto": la radice gatt- più una vocale finale
re.findall(r"\bgatt[oaie]\b", testo)
# ['gatto', 'gatta']

# il CAP: esattamente cinque cifre isolate
re.findall(r"\b\d{5}\b", testo)
# ['95125']

# una data giorno/mese/anno
re.findall(r"\b\d{1,2}/\d{1,2}/\d{4}\b", testo)
# ['3/7/2026']

# gruppi: catturare giorno, mese e anno separatamente
m = re.search(r"(\d{1,2})/(\d{1,2})/(\d{4})", testo)
m.group(1), m.group(2), m.group(3)
# ('3', '7', '2026')
```

Una regola d'onestà: le espressioni regolari non *capiscono* niente. Trovano
forme, non significati: proprio come ELIZA, che agganciava «mi sento X» senza
avere idea di cosa fosse un sentimento. Per estrarre un CAP bastano; per
decidere se una recensione è entusiasta o sarcastica no. È il confine esatto
tra ciò che questa sezione può fare e ciò per cui servirà il resto del
capitolo.

## Normalizzare il testo: decidere cosa è «la stessa parola»

Il secondo attrezzo è meno appariscente ma altrettanto indispensabile, e parte
da un fatto da mettere a fuoco adesso, perché regge tutto il capitolo: **per
un calcolatore ogni lettera è un numero**. Esiste una convenzione
internazionale, Unicode, che assegna un numero a ogni carattere di ogni lingua
del mondo, e confrontare due parole vuol dire confrontare due file di numeri.

Da qui la pignoleria. Per una macchina `Muro`, `muro` e `MURO` sono tre parole
diverse: la `M` maiuscola e la `m` minuscola hanno numeri diversi, punto. C'è
di peggio, ed è il caso in cui la macchina ha ragione da vendere e il risultato
è comunque assurdo. La parola `perché` si può scrivere in due modi che sullo
schermo sono identici: o con una `é` sola, che è un numero solo, o con una `e`
seguita da un segno di accento a parte che le si posa sopra, e allora sono due
numeri. Stesso disegno sulla pagina, contenuto diverso in memoria, e la parola
risulta diversa da sé stessa.

Prima di contare le parole di un testo, cosa che faremo eccome nella prossima
sezione, bisogna dunque decidere quali varianti contare *insieme*. Questa
scelta si chiama **normalizzazione**.

`````{tab} Elementare

I fusilli integrali e i fusilli normali sono «pasta» o due cose diverse?
Dipende da cosa vuoi cucinare, e finché non lo hai deciso non puoi nemmeno
contare i barattoli della dispensa. Con le parole si fa la stessa scelta, e
riguarda quali varianti vanno nello stesso barattolo.

Il minuscolo mette insieme *Muro* a inizio frase e *muro* in mezzo. Poi si
uniforma il modo in cui le lettere stanno in memoria. Le due scritture di
*perché*, quella con la *é* intera e quella con l'accento appoggiato sopra,
sono due confezioni identiche sullo scaffale con due codici a barre diversi:
senza questa mossa la cassa le batte come due prodotti. Via anche la
punteggiatura e le **stopword**, le parole-colla come *il*, *di*, *che*, *e*,
che stanno dappertutto e proprio per questo non dicono nulla sull'argomento del
testo. Resta la mossa più delicata, raggruppare le forme della stessa parola.
*Andavamo*, *andiamo* e *andrò* sono tutte facce del verbo *andare*.

Per quest'ultima mossa ci sono due attrezzi. Lo **stemming** lavora di
forbici, e taglia la coda delle parole secondo regole fisse, sempre alla stessa
misura. È rapidissimo, e su *gatto*, *gatta* e *gatti* fa centro, perché da
tutti e tre resta *gatt*. Sui verbi irregolari invece cade proprio dove
serviva. Le forbici più usate per l'italiano riducono *andavamo* ad *andavam*,
*andiamo* ad *andiam* e *andare* ad *andar*, mentre su *andrò* non trovano
nemmeno una coda da tagliare. Quattro etichette diverse, quattro barattoli, e
il verbo che si voleva raccogliere resta sparpagliato come prima.

La **lemmatizzazione** lavora di vocabolario, risale alla forma base (il
**lemma**) e da *andavamo* ricava davvero *andare*. Il vocabolario da solo però
non basta, e lo sa chiunque ne abbia aperto uno. *Porta* può essere quella di
casa, oppure quello che fa chi porta la spesa, e per decidere quale delle due
bisogna leggere le parole intorno. Più precisa, dunque, ma più lenta e molto
più faticosa da costruire.

Raggruppare ha comunque un prezzo, e in dispensa si vede meglio che sulla
pagina. Chi cerca «pasta» adesso trova ogni cosa, ed era lo scopo; chi cercava
proprio i fusilli integrali non li distingue più dagli altri. Nel testo
funziona uguale, si trova di più e si distingue di meno. In italiano, dove un
verbo ha decine di forme, il baratto conviene quasi sempre.

`````

`````{tab} Superiore

Normalizzare significa definire una funzione che manda ogni variante
superficiale in un rappresentante canonico: minuscolizzazione (*case
folding*), normalizzazione Unicode (le forme NFC/NFKC unificano caratteri
composti e precomposti, come la *é* codificata in un modo o in due),
rimozione di punteggiatura e stopword, riduzione morfologica.

Per quest'ultima, lo **stemming** applica regole di troncamento dei suffissi:
il capostipite è l'algoritmo di Porter (1980), per l'inglese, esteso
all'italiano nella famiglia Snowball. È una funzione puramente ortografica,
senza dizionario, e si vede: lo stemmer Snowball italiano manda *gatto*,
*gatta* e *gatti* correttamente in *gatt*, ma spezza il paradigma di *andare*
in tre gambi diversi; *andavamo* → *andavam*, *andiamo* → *andiam*, *andare* →
*andar*. La **lemmatizzazione** richiede invece un'analisi morfologica con
dizionario e contesto (per disambiguare, ad esempio, *porta* sostantivo da
*porta* voce del verbo *portare*) e restituisce il lemma: *andavamo* →
*andare*. Nei sistemi a conteggio la riduzione morfologica aumenta la *recall*
(query e documento si incontrano anche se flessi diversamente) al prezzo di un
po’ di *precision* (forme distinte collassano); per una lingua flessiva come
l'italiano il compromesso è quasi sempre favorevole.

`````

In Python bastano poche righe per una catena di normalizzazione essenziale. Il
programma fa tre cose in fila: uniforma le codifiche (è la prima riga, quella
che risolve il caso del `perché` scritto in due modi: `NFKC` è il nome della
regola che sceglie sempre la versione a un carattere solo), manda tutto in
minuscolo, butta via la punteggiatura e le parole-colla. Nella terza riga,
`[^\w\s]` si legge «tutto ciò che *non* è né una lettera o cifra (`\w`) né
uno spazio (`\s`)»: il `^` dentro le parentesi quadre rovescia l'elenco, e
quindi quel pezzo di schema aggancia esattamente la punteggiatura.

```python
import re
import unicodedata

STOPWORD = {"il", "lo", "la", "i", "gli", "le", "un", "una", "di", "a",
            "da", "in", "su", "sul", "per", "con", "e", "che", "è"}

def normalizza(testo):
    testo = unicodedata.normalize("NFKC", testo)  # codifiche Unicode uniformi
    testo = testo.lower()                         # tutto minuscolo
    testo = re.sub(r"[^\w\s]", " ", testo)        # via la punteggiatura
    return [p for p in testo.split() if p not in STOPWORD]

normalizza("Il gatto NERO salta sul muro!")
# ['gatto', 'nero', 'salta', 'muro']
```

Quando serve tutto questo? Quando si rappresenta il testo *contando le
parole*, ed è proprio quello che faremo nella prossima sezione: lì, e nei
motori di ricerca classici, normalizzare bene fa la differenza tra trovare e
non trovare un documento.

Le reti neurali moderne, invece, hanno progressivamente smesso di buttare via
l'informazione, perché maiuscole, accenti e desinenze *portano significato*.
Mandate tutto in minuscolo e la frase «Rosa è rosa» diventa «rosa è rosa»:
avete perso il nome proprio, e con lui la battuta. Al posto della potatura
usano un taglio diverso, che conserva il testo com'è e lo spezza in unità più
piccole della parola: *straordinariamente* non finisce nel dizionario intero,
ci finiscono `stra`, `ordinaria` e `mente`, che ricorrono in mille altre
parole. Come si scelgano quei pezzi è il tema di una sezione fra due, quella
sui **tokenizzatori**. La normalizzazione aggressiva è dunque un attrezzo da
usare quando si conta, non un obbligo universale.

## La distanza di edit: quante mosse da una parola all'altra

Il terzo attrezzo nasce da un'esperienza quotidiana: digiti «gatot» e il
telefono capisce che intendevi «gatto». Come fa a sapere che «gatot» somiglia
a «gatto» più che a «divano»? Serve un modo per *misurare* la distanza tra due
parole. La misura standard porta il nome del matematico sovietico Vladimir
Levenshtein, che la introdusse nel 1966 {cite}`levenshtein1966binary`, in un
articolo di poche pagine che non parlava affatto di parole: parlava di codici
binari per correggere errori di trasmissione, e le sue «parole» erano sequenze
di 0 e 1. Il nome **distanza di Levenshtein** per la versione sul testo si
affermò solo in seguito; è uno di quei casi in cui un'idea nata in un campo
finisce per fare fortuna in un altro.

`````{tab} Elementare

Sul tavolo cinque tessere formano *carta*, e bisogna arrivare a *casa*. I gesti
permessi sono tre: cambiare una tessera con un'altra (una sostituzione),
toglierne una (una cancellazione), infilarne una nuova (un inserimento). Ogni
gesto vale una mossa, vince chi ne fa meno, e il numero di mosse della strada
più corta è la distanza di edit fra le due parole.

Da *casa* a *cosa* basta girare la prima *a* in *o*, distanza 1. Da *carta* a
*casa* le mosse sono due: via la *r* (*carta* → *cata*), poi *t* → *s* (*cata*
→ *casa*). Con una sola non ce la fate per quanto proviate, perché le tessere
sono in numero diverso: una va tolta per forza, e tolta quella il resto ancora
non combacia.

Più corta è la strada, più le due parole si somigliano. *Gatot* dista 2 da
*gatto* e 5 da *divano*: una *d* e una *i* davanti (*digatot*), *g* → *v*
(*divatot*), *t* → *n* (*divanot*), via la *t* finale (*divano*). Per questo il
correttore del telefono scommette su *gatto*.

Ma quella da cinque è *una* strada, e che sia la più corta nessuno l'ha ancora
promesso: è l'unica difficoltà del gioco. Su quattro lettere si vede a occhio,
su parole lunghe le strade sono troppe. Serve allora un foglio a quadretti,
*muro* lungo il bordo di sinistra e *mare* lungo quello di sopra. Ogni casella
riguarda solo l'inizio delle due parole, quello letto fino a quella riga e fino
a quella colonna, e dice quante mosse servono per passare dall'uno all'altro:

|   | (niente) | m | a | r | e |
|---|---|---|---|---|---|
| **(niente)** | 0 | 1 | 2 | 3 | 4 |
| **m** | 1 | 0 | 1 | 2 | 3 |
| **u** | 2 | 1 | 1 | 2 | 3 |
| **r** | 3 | 2 | 2 | 1 | 2 |
| **o** | 4 | 3 | 3 | 2 | **2** |

Prima riga e prima colonna sono regalate: per andare da niente a *m*, *ma*,
*mar*, *mare* si infilano 1, 2, 3, 4 tessere. Ogni altra casella guarda le tre
vicine, e ciascuna è un gesto: da quella di sopra si arriva togliendo l'ultima
tessera di sinistra, da quella a sinistra infilando l'ultima di sopra, da
quella in diagonale accoppiandole. Si prende la vicina più piccola, si paga 1,
e lo sconto è uno solo: se l'ultima tessera di sinistra e l'ultima di sopra
portano la stessa lettera si ricopia la diagonale senza pagare niente, perché
quelle due tessere già combaciano.

Facciamone una insieme, riga *u* e colonna *a*. Sopra c'è 1, a sinistra 1, in
diagonale 0. La *u* e la *a* sono lettere diverse, niente sconto, quindi si
prende 0 e si aggiunge 1: fa 1, il numero che sta nella casella. Provatene
un'altra, il meccanismo è sempre questo.

In fondo a destra c'è la risposta, 2: da *muro* a *mare* si girano due tessere,
*u* → *a* e *o* → *e*. Ed è davvero il minimo, perché ogni casella ha scelto il
più economico fra i tre gesti, e un quarto modo di arrivarci non esiste:
nessuna scorciatoia può sfuggire. Il foglio si riempie in un lampo anche su
parole lunghe.

`````

`````{tab} Superiore

Date due stringhe $a = a_1 \cdots a_n$ e $b = b_1 \cdots b_m$, la distanza di
Levenshtein è il costo minimo per trasformare $a$ in $b$ con inserzioni,
cancellazioni e sostituzioni di costo unitario. Si calcola con la
**programmazione dinamica**: sia $D_{i,j}$ la distanza tra il prefisso
$a_1 \cdots a_i$ e il prefisso $b_1 \cdots b_j$. Allora

$$
D_{i,0} = i, \qquad D_{0,j} = j,
$$

$$
D_{i,j} = \min
\begin{cases}
D_{i-1,\,j} + 1 & \text{(cancellazione di } a_i\text{)}\\[2pt]
D_{i,\,j-1} + 1 & \text{(inserzione di } b_j\text{)}\\[2pt]
D_{i-1,\,j-1} + \mathbb{1}[a_i \neq b_j] & \text{(sostituzione, o lettere uguali)}
\end{cases}
$$

dove $\mathbb{1}[a_i \neq b_j]$ vale 1 se le lettere differiscono e 0 se
coincidono: l'ultima lettera di ciascun prefisso o si cancella, o si
inserisce, o si mette in corrispondenza con l'altra, e ogni caso riconduce a
un sottoproblema più piccolo, già risolto. Compiliamo la tabella per
*muro* → *mare* (riga per riga, ogni cella applica la ricorrenza; la colonna
e la riga di $\varepsilon$, la stringa vuota, sono i casi base):

|   | $\varepsilon$ | m | a | r | e |
|---|---|---|---|---|---|
| $\varepsilon$ | **0** | 1 | 2 | 3 | 4 |
| **m** | 1 | **0** | 1 | 2 | 3 |
| **u** | 2 | 1 | **1** | 2 | 3 |
| **r** | 3 | 2 | 2 | **1** | 2 |
| **o** | 4 | 3 | 3 | 2 | **2** |

L'angolo in basso a destra dà $D_{4,4} = 2$: bastano due sostituzioni (*u* →
*a*, *o* → *e*), e il percorso ottimo (in grassetto) scende lungo la
diagonale, pagando 1 solo dove le lettere differiscono. La tabella ha
$(n+1)(m+1)$ celle e ogni cella costa un confronto: complessità $O(nm)$ in
tempo, riducibile a $O(\min(n,m))$ in memoria tenendo in vita solo due righe
della tabella, orientata lungo la stringa più corta. La formulazione tabellare
è nota anche come algoritmo di Wagner–Fischer (1974). Una variante dovuta a
Fred Damerau (1964) aggiunge lo **scambio** di due lettere adiacenti come
quarta mossa: per «gatot» → «gatto» la distanza scende da 2 a 1, coerente con
l'osservazione di Damerau che circa quattro refusi su cinque sono a una sola
mossa dalla parola giusta.

`````

Il programma fa esattamente quello che avete fatto voi a mano sulla griglia,
una riga per volta, e di tutta la tabella tiene in memoria solo la riga
precedente, perché è l'unica che serve per calcolare quella dopo:

```python
def levenshtein(a, b):
    prec = list(range(len(b) + 1))          # riga dei casi base D[0][j] = j
    for i, ca in enumerate(a, start=1):
        cur = [i]                           # caso base D[i][0] = i
        for j, cb in enumerate(b, start=1):
            costo = 0 if ca == cb else 1
            cur.append(min(prec[j] + 1,          # cancellazione
                           cur[j - 1] + 1,       # inserzione
                           prec[j - 1] + costo)) # sostituzione o lettera uguale
        prec = cur
    return prec[-1]

levenshtein("muro", "mare")   # 2
levenshtein("carta", "casa")  # 2
levenshtein("gatot", "gatto") # 2
```

## Dal refuso al correttore: l'idea del canale rumoroso

Con la distanza di edit in mano, il correttore ortografico sembra fatto:
suggerisci la parola più vicina e via. Non funziona, e basta un esempio per
capire perché. Digito «cane» quando volevo «case»: le due parole distano una
mossa sola, ma a distanza uno da «cane» ci sono anche «pane», «rane», «cani»,
«can» e altre. Sono tutte ugualmente vicine, e la vicinanza non sa dire quale
volevo: bisogna anche chiedersi *quanto è verosimile che io abbia sbagliato in
quel modo*, e quanto quella parola è frequente.

La cornice giusta viene dalla **teoria dell'informazione**, la disciplina
fondata da Claude Shannon nel 1948 {cite}`shannon1948mathematical`. Shannon
studiava che cosa succede a un messaggio quando viaggia lungo un canale che lo
può sporcare: una linea telefonica disturbata, una radio, un disco graffiato.
Il correttore prende in prestito quell'immagine. Chi scrive aveva in mente la
parola giusta; poi quella parola è passata dentro un **canale rumoroso** (le
dita, la tastiera, la fretta) che ogni tanto la storpia. Correggere vuol dire
risalire il canale e indovinare che cosa c'era all'ingresso.

Si fa in due tempi.

**Primo tempo: una lista corta di sospetti.** Ci si tengono le parole del
vocabolario a distanza di edit 1 o 2 da quello che è arrivato. Il 2 non è una
legge di natura, è una scelta pratica, e poggia su un dato: la stragrande
maggioranza dei refusi sta a una sola mossa dalla parola giusta. Allargare a 3
farebbe entrare migliaia di candidati per pochissimi refusi in più.

**Secondo tempo: il confronto fra i sospetti.** A ogni candidato si danno due
voti e li si **moltiplica**, come si fa per due cose che devono capitare
insieme. Il primo voto è quanto quella parola è frequente nella lingua; il
secondo è quanto è facile che il rumore l'abbia trasformata proprio in ciò che
si legge. Quest'ultimo non lo decide nessuno a mano: si conta, su un archivio
di refusi veri, quante volte una certa svista è capitata davvero. Prendiamo
«gatot». Il candidato *gatto* è frequente (diciamo una parola su ventimila) e
l'errore che servirebbe è lo scambio di due lettere vicine, una svista che
capita eccome quando si scrive in fretta sulla tastiera, diciamo una volta su
venti. Il candidato *gatot* così com'è,
se anche fosse una parola, sarebbe rarissimo. Il primo prodotto è enormemente
più grande del secondo, e il telefono scrive *gatto*.

È un'idea messa in pratica già nel 1990 da Mark Kernighan, Kenneth Church e
William Gale, con un correttore che non conteneva nemmeno una regola di
grammatica: solo conteggi[^kern].

[^kern]: Il nome di battesimo qui non è un vezzo. Mark D. Kernighan, degli AT&T
Bell Laboratories, non va confuso con Brian W. Kernighan, coautore del
linguaggio di programmazione C e uno degli artefici di quell'Unix da cui
vengono `ed` e `grep`, raccontati poche pagine fa: stesso cognome, stessi
laboratori, argomenti confinanti. L'articolo del 1990 è *A Spelling Correction
Program Based on a Noisy Channel Model*.

La griglia della distanza di edit, del resto, non corregge solo refusi. Con
qualche ritocco (per esempio facendo costare più di 1 certe mosse) la stessa
tabella mette in fila due sequenze di DNA in biologia, e ritrova le persone
registrate due volte in un archivio, «Giovanni Rossi» contro «Givanni Rossi».
E non abbiamo finito di incontrarla: tornerà nel {doc}`capitolo sul riconoscimento
vocale </SpeechRecognition/overview>` come metro di giudizio dei programmi che trascrivono il parlato. Lì le
mosse non si contano più sulle lettere ma sulle parole, cioè quante parole un
programma ha sbagliato, saltato o aggiunto rispetto a quello che era stato
detto davvero; il rapporto fra queste e il totale è il **WER**, *word error
rate*, il tasso di errore per parola. Chi volesse approfondire l'intera
cassetta degli attrezzi di questa sezione trova la trattazione di riferimento
in Jurafsky e Martin {cite}`jurafsky2026speech`.

`````{tab} Elementare
```{admonition} Da ricordare
:class: important
- Un’**espressione regolare** descrive una *forma* («cinque cifre di fila»),
  non una parola precisa: è la funzione Trova con i superpoteri. Perfetta per
  estrarre e per controllare che un dato sia scritto bene, incapace di capire
  un significato e incapace, per un limite preciso e non per distrazione, di
  seguire le frasi dentro le frasi: per farlo dovrebbe tenere il conto di
  quante ne ha aperte, e non sa contare.
- ELIZA, `grep`, il modulo che vi dice che l'email non è valida: cercare
  schemi è l'NLP «a regole», ed è ancora ovunque nel lavoro di ripulire i dati.
- **Normalizzare** vuol dire decidere che cosa contare come «la stessa
  parola»: tutto minuscolo, codifiche uniformi, via le parole-colla, e le
  forme di uno stesso verbo raggruppate. Lo **stemming** lavora di forbici
  (taglia la coda, e sui verbi irregolari il paradigma resta sparpagliato), la
  **lemmatizzazione** di dizionario (*andavamo* → *andare*).
- Si normalizza con decisione quando si *conta*, ed è quello che faremo nella
  prossima sezione (motori di ricerca, sacchetto di parole). I modelli neurali
  di oggi preferiscono invece conservare il testo com'è e spezzarlo in pezzi
  più piccoli della parola: come si scelgono quei pezzi è il tema della sezione
  sui tokenizzatori, due più avanti.
- La **distanza di edit** è il numero minimo di mosse (sostituisci, cancella,
  inserisci) per passare da una parola all'altra: *gatot* dista 2 da *gatto* e
  5 da *divano*. Si calcola riempendo una griglia, senza che nessuna
  scorciatoia possa sfuggire.
- Il **correttore ortografico** la usa dentro l'idea del *canale rumoroso*:
  prima si fa la lista corta delle parole vicine a quella digitata, poi si dà a
  ciascuna due voti e li si moltiplica. Primo voto: quanto quella parola è
  comune. Secondo voto: quanto è facile che un dito distratto la trasformi
  proprio in ciò che è arrivato. Vince il prodotto più alto. La stessa
  distanza, contata sulle parole invece che sulle lettere, tornerà nel capitolo
  sul riconoscimento vocale per misurare gli errori di una trascrizione.
```
`````

`````{tab} Superiore
```{admonition} Da ricordare
:class: important
- Le **espressioni regolari** descrivono *schemi* («cinque cifre di fila»),
  non parole esatte: perfette per estrarre e validare, incapaci (per un limite
  matematico preciso) di gestire strutture annidate o significati.
- ELIZA, `grep`, i validatori dei moduli web: il pattern matching è l'NLP
  «a regole», ed è ancora ovunque nella pulizia dei dati.
- La **normalizzazione** (minuscole, Unicode, stopword) decide cosa contare
  come «la stessa parola»; lo **stemming** taglia i suffissi con regole
  fisse, la **lemmatizzazione** risale alla forma di dizionario
  (*andavamo* → *andare*).
- Normalizzare in modo aggressivo serve quando si *conta* (ricerca,
  *bag-of-words*: la prossima sezione); i modelli neurali moderni preferiscono
  conservare il testo e spezzarlo in unità sotto la parola, ed è il tema della
  sezione sui tokenizzatori, due più avanti.
- La **distanza di Levenshtein** {cite}`levenshtein1966binary` è il numero
  minimo di inserzioni, cancellazioni e sostituzioni tra due stringhe; si
  calcola per programmazione dinamica in tempo $O(nm)$.
- Il **correttore ortografico** la usa dentro il modello del *canale
  rumoroso*: tra i candidati vicini vince la parola frequente che il rumore
  trasforma facilmente in ciò che è stato digitato. La stessa distanza,
  contata sulle parole, diventerà il WER del riconoscimento vocale.
```
`````
