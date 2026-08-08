# La cassetta degli attrezzi: espressioni regolari, normalizzazione e distanza di edit

Nell'Introduzione abbiamo incontrato ELIZA, il programma con cui Joseph
Weizenbaum dimostrò (suo malgrado) quanto sia facile attribuire
un'intelligenza a una macchina {cite}`weizenbaum1966eliza`. Vale la pena
riaprire il cofano: dentro non c'era nessuna comprensione del linguaggio, ma
un gioco di **pattern matching**, la ricerca di schemi nel testo. Se l'utente
scriveva «mi sento triste», ELIZA agganciava lo schema «mi sento X» e
riassemblava i pezzi in «Da quanto tempo ti senti X?», seguendo regole scritte
a mano da Weizenbaum stesso. Tutto qui.

Prima dei modelli neurali che occuperanno il resto del capitolo, l'NLP era in
larga parte questo: schemi, regole, conteggi. Sarebbe però un errore liquidare
questi attrezzi come pezzi da museo. Sono ovunque, anche oggi: nel comando
`grep` con cui i programmatori setacciano file da mezzo secolo, nel modulo web
che ti avvisa che «l'indirizzo email non è valido», nella pulizia dei dati che
(chiedete a chiunque lavori nel settore) occupa la parte più grossa di
qualunque progetto reale. Prima di insegnare a una rete neurale a leggere,
conviene imparare a usare la cassetta degli attrezzi.

## Le espressioni regolari: descrivere uno schema, non una parola

Il primo attrezzo risponde a una domanda concreta: come si cerca in un testo
qualcosa che non è una parola precisa ma una *forma*? Tutte le date, tutti i
CAP, tutti gli importi in euro. La risposta ha un nome intimidatorio,
**espressioni regolari** (*regular expressions*, o *regex*), e una storia
curiosa: nascono negli anni Cinquanta dai lavori del logico Stephen Kleene,
che studiava proprio i modelli matematici dei neuroni di McCulloch e Pitts; le
espressioni regolari e le reti neurali, gli attrezzi «vecchi» e quelli «nuovi»
di questo capitolo, condividono l'atto di nascita. A portarle nei programmi fu
Ken Thompson alla fine degli anni Sessanta, prima nell'editor QED e poi in
`ed`, il primo editor di Unix: il nome del comando `grep` viene proprio dal
comando `g/re/p` di `ed`, «cerca ovunque l'espressione regolare e stampa».

`````{tab} Elementare

Pensa a una caccia al tesoro dove l'indizio non dice «trova la parola 95125»
ma «trova cinque cifre di fila». Un'espressione regolare è esattamente questo:
la *descrizione di uno schema*, invece di una parola esatta. «Cinque cifre di
fila» trova tutti i CAP d'Italia; «una o due cifre, una barra, una o due
cifre, una barra, quattro cifre» trova tutte le date scritte come 3/7/2026;
«la radice *gatt-* seguita da una vocale» trova *gatto*, *gatta*, *gatti* e
*gatte* in un colpo solo.

È la funzione Trova del tuo editor di testi, ma con i superpoteri: invece di
controllare lettera per lettera, controlla *tipo* di lettera per tipo di
lettera (qui voglio una cifra, qui una lettera qualsiasi, qui uno spazio).
Quando un sito ti dice al volo che il numero di telefono che hai digitato non
è valido, nove volte su dieci c'è un'espressione regolare che ha confrontato
quello che hai scritto con lo schema atteso e ha trovato che non combacia.

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

In Python le espressioni regolari vivono nel modulo `re`. Mettiamo alla prova
la nostra frase preferita, arricchita di qualche dettaglio da estrarre:

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

Il secondo attrezzo è meno appariscente ma altrettanto indispensabile. Un
calcolatore è un pignolo assoluto: per lui `Muro`, `muro` e `MURO` sono tre
stringhe diverse, e `perché` scritto con due codifiche Unicode diverse è
diverso da sé stesso. Prima di contare le parole di un testo (cosa che faremo,
eccome, nella prossima sezione), bisogna decidere quali varianti contare
*insieme*. Questa scelta si chiama **normalizzazione**.

`````{tab} Elementare

Immagina di riordinare la dispensa: prima di contare i barattoli devi decidere
cosa va nello stesso barattolo. I fusilli integrali e i fusilli normali sono
«pasta» o due cose diverse? Dipende da cosa vuoi cucinare. Con le parole è
uguale, e le mosse tipiche sono tre. Primo: tutto minuscolo, così *Muro* a
inizio frase e *muro* in mezzo finiscono nello stesso barattolo. Secondo:
togliere le **stopword**, le parole-colla come *il*, *di*, *che*, *e*; sono
dappertutto e proprio per questo non dicono nulla sull'argomento del testo.
Terzo, il più delicato: raggruppare le forme della stessa parola. *Andavamo*,
*andiamo* e *andrò* sono tutte facce del verbo *andare*.

Per quest'ultima mossa ci sono due attrezzi. Lo **stemming** lavora di
forbici: taglia la coda delle parole secondo regole fisse, veloce ma
grossolano; a volte da *andavamo* esce un moncone come *andavam*, che non è
nemmeno una parola. La **lemmatizzazione** lavora di dizionario: cerca la
forma base, il **lemma**, e da *andavamo* ricava proprio *andare*, come
farebbe chiunque consulti un vocabolario. Più precisa, ma più lenta e più
difficile da costruire.

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
po' di *precision* (forme distinte collassano); per una lingua flessiva come
l'italiano il compromesso è quasi sempre favorevole.

`````

In Python bastano poche righe, senza librerie esterne, per una pipeline di
normalizzazione essenziale:

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
parole*: nei motori di ricerca classici e nei modelli a sacchetto di parole
che vedremo nella prossima sezione, normalizzare bene fa la differenza tra
trovare e non trovare un documento. I modelli neurali moderni, invece, hanno
progressivamente smesso di buttare via l'informazione: maiuscole, accenti e
desinenze *portano significato* («Rosa è rosa» perde qualcosa, tutto
minuscolo), e i loro tokenizzatori (li incontreremo tra poche pagine)
preferiscono conservare il testo com'è e spezzarlo in unità più piccole della
parola. La normalizzazione aggressiva è un attrezzo da usare quando si conta,
non un obbligo universale.

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

La distanza di edit è un gioco: quante mosse servono, al minimo, per
trasformare una parola in un'altra? Le mosse permesse sono tre: **sostituire**
una lettera, **cancellarne** una, **inserirne** una. Ogni mossa costa 1.

Da *casa* a *cosa*: sostituisci la prima *a* con una *o*, una mossa. Distanza
1. Da *carta* a *casa* servono invece due mosse: cancella la *r* (*carta* →
   *cata*), poi sostituisci la *t* con una *s* (*cata* → *casa*). Distanza 2,
   e puoi provare quanto vuoi, con meno di due mosse non ce la fai: le parole
   hanno lunghezze diverse (quindi almeno una cancellazione è obbligatoria) e
   una cancellazione da sola non basta mai a far combaciare il resto.

Ecco il senso della misura: più piccola è la distanza, più le parole si
somigliano. «Gatot» dista 2 da «gatto» ma 5 da «divano»: per questo il
correttore scommette su «gatto». L'unica difficoltà vera è garantire che le
mosse trovate siano davvero *il minimo*: per due parole corte si fa a occhio,
per due parole lunghe il computer compila una tabella di distanze parziali,
pezzo per pezzo, senza mai perdersi una scorciatoia, e lo fa in un lampo.

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

L'implementazione segue la ricorrenza alla lettera, tenendo in memoria solo la
riga precedente della tabella:

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

```{figure} ../figures/nlp-classico-era-llm.svg
:name: fig-nlp-classico-vs-llm
:alt: "Due approcci affiancati allo stesso problema. A sinistra la pipeline classica: una catena di stadi separati, ciascuno costruito e messo a punto a mano, dalla normalizzazione all'analisi fino alla risposta. A destra l'approccio a prompt: un solo modello che riceve l'istruzione in linguaggio naturale e produce direttamente la risposta."
:width: 100%

Due modi di risolvere lo stesso compito. La colonna di sinistra non è
obsoleta: è ispezionabile stadio per stadio, e quando serve sapere *perché* è
uscita una certa risposta, resta l'unica delle due che lo dice.
```

Vale la pena tenere {numref}`fig-nlp-classico-vs-llm` in mente leggendo questa
sezione, perché il confronto non è fra vecchio e nuovo ma fra trasparente e
opaco. Gli strumenti che seguono si spiegano in una riga e si correggono a
mano; è per questo che sopravvivono dentro sistemi moderni, nei punti in cui
serve una garanzia e non una probabilità.

Con la distanza di edit in mano, il correttore ortografico è a un passo. Ma il
passo è più sottile di «suggerisci la parola più vicina». Se digito «cassa»,
era un refuso per *casa*, per *cassia*, o era proprio *cassa*? La cornice
giusta viene dalla teoria dell'informazione di Shannon
{cite}`shannon1948mathematical`: immagina che chi scrive avesse in mente la
parola giusta, e che questa sia passata attraverso un **canale rumoroso** (le
dita, la tastiera, la fretta) che ogni tanto la sporca. Il correttore deve
risalire il canale: tra le parole *plausibili* (quelle a distanza di edit 1 o
2 da ciò che leggo), scegliere quella che meglio bilancia due fattori; quanto
è *frequente* nella lingua e quanto è *facile* che il rumore l'abbia
trasformata proprio in ciò che è arrivato. È un'idea messa in pratica già nel
1990 da Kernighan, Church e Gale con un correttore puramente statistico, ed è
la stessa logica del vostro telefono: «gatot» viene corretto in «gatto» perché
*gatto* è frequente e lo scambio di due lettere adiacenti è un errore di
battitura tipicissimo.

La distanza di edit, del resto, non corregge solo refusi: varianti pesate
della stessa ricorrenza allineano sequenze di DNA in bioinformatica e
deduplicano anagrafiche («Giovanni Rossi» contro «Givanni Rossi») nei database
di mezzo mondo. E non abbiamo finito di incontrarla: la ritroveremo nel
capitolo sul riconoscimento vocale, dove (contata a livello di parola invece
che di lettera) misura gli errori dei trascrittori automatici con la metrica
WER. Chi volesse approfondire l'intera cassetta degli attrezzi di questa
sezione trova la trattazione di riferimento in Jurafsky e Martin
{cite}`jurafsky2026speech`.

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
  bag-of-words); i modelli neurali moderni preferiscono conservare il testo e
  spezzarlo in unità sotto la parola: è il tema della prossima sezione.
- La **distanza di Levenshtein** {cite}`levenshtein1966binary` è il numero
  minimo di inserzioni, cancellazioni e sostituzioni tra due stringhe; si
  calcola per programmazione dinamica in tempo $O(nm)$.
- Il **correttore ortografico** la usa dentro il modello del *canale
  rumoroso*: tra i candidati vicini vince la parola frequente che il rumore
  trasforma facilmente in ciò che è stato digitato. La stessa distanza,
  contata sulle parole, diventerà il WER del riconoscimento vocale.
```
