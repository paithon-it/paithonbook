# La struttura nascosta della frase: sintassi e parsing

«Ho visto un uomo con il binocolo». La frase che ha aperto questo capitolo
merita di essere ripresa adesso, con gli attrezzi giusti in mano. Chi ha il
binocolo? Se ce l'ho io, «con il binocolo» accompagna il verbo: dice *come* ho
visto. Se ce l'ha lui, accompagna il nome: dice *quale* uomo ho visto. E c'è
un dettaglio che rende la faccenda istruttiva: le etichette della sezione
precedente, da sole, qui non bastano. Ausiliare, verbo, articolo, nome,
preposizione, articolo, nome: il POS tagging produce la stessa identica
sequenza per entrambe le letture. In «La vecchia porta la sbarra» l'ambiguità
viveva al piano delle categorie grammaticali; qui vive un piano più su. Le due
letture non sono due sfumature dello stesso significato: sono due
**strutture** diverse costruite con gli stessi mattoni.

La disciplina che studia queste strutture è la **sintassi**; costruirle
automaticamente si chiama **parsing**, o analisi sintattica. Se la sezione
precedente era l'analisi grammaticale di scuola, questa è l'analisi logica:
chi fa che cosa, a chi, con che cosa. Per disegnare la struttura di una frase
la linguistica ha prodotto due grandi famiglie di mappe (i **costituenti**,
che raggruppano le parole in blocchi, e le **dipendenze**, che le collegano
con frecce) e il NLP le usa entrambe.

## Scatole dentro scatole: i costituenti

L'osservazione di partenza è che certe sequenze di parole si comportano come
un blocco unico. Nell'esempio ricorrente del libro, «Il gatto nero salta sul
muro», il gruppo «il gatto nero» si muove, si sostituisce e risponde alle
domande come un pezzo solo. Questi blocchi si chiamano **sintagmi** (o
costituenti), e l'idea che la frase sia fatta di blocchi annidati è la
**costituenza**.

`````{tab} Elementare

Pensa a un trasloco: non porti in strada le forchette una per una, le chiudi
in una scatola ed è la scatola che viaggia. La frase funziona uguale. Come si
scopre dove finisce una scatola? Con due prove da fare a orecchio. Prova di
**sostituzione**: se un gruppo di parole si può rimpiazzare con una parola
sola, è una scatola («**Il gatto nero** salta sul muro» diventa «**Lui** salta
sul muro»). Prova di **spostamento**: una scatola si sposta tutta intera,
«**Sul muro** salta il gatto nero» suona benissimo, mentre «*Muro il gatto
nero salta sul*» non è italiano: abbiamo strappato il cartone. E dentro la
scatola «sul muro» c'è una scatolina, «il muro»: scatole dentro scatole, fino
alle parole.

Ora il binocolo. Le due letture sono due modi diversi di inscatolare le stesse
parole. Se l'uomo ha il binocolo, c'è una scatola grande: «Ho visto [un uomo
con il binocolo]», e infatti puoi sostituirla tutta con «l'ho visto», dove
«lo» è l'uomo *col* binocolo. Se il binocolo è mio, le scatole sono due: «Ho
visto [un uomo] [con il binocolo]», e infatti la seconda si sposta da sola:
«Con il binocolo, ho visto un uomo» funziona *solo* in questa lettura. Stesse
sette parole, due disegni di scatole: l'ambiguità è tutta lì.

`````

`````{tab} Superiore

Lo strumento formale è la **grammatica context-free** (CFG), introdotta da
Noam Chomsky in un articolo del 1956 {cite}`chomsky1956three` che confronta
tre modelli matematici del linguaggio: i processi a stati finiti (i parenti
stretti degli $n$-gram incontrati in questo capitolo), le grammatiche a
struttura sintagmatica e le grammatiche trasformazionali. La tesi che fece
scuola: gli stati finiti non bastano, perché la sintassi annida dipendenze a
distanza arbitraria; da quella gerarchia di formalismi (oggi detta *gerarchia
di Chomsky*) l'informatica ha attinto anche i linguaggi di programmazione.

Una CFG è una quadrupla $G = (N, \Sigma, R, S)$, dove $N$ è l'insieme dei
simboli **non terminali** (le categorie sintattiche), $\Sigma$ il
vocabolario dei **terminali** (le parole), $R$ un insieme di **regole di
riscrittura** della forma $A \to \alpha$ con $A \in N$ e $\alpha$ sequenza
di simboli, e $S$ il simbolo iniziale. Una grammatica giocattolo per il
nostro frammento d'italiano:

$$
\begin{align*}
F &\to \text{AUX} \ \text{SV}
  &\qquad \text{SN} &\to \text{DET} \ \text{N} \\
\text{SV} &\to \text{V} \ \text{SN}
  &\qquad \text{SN} &\to \text{SN} \ \text{SP} \\
\text{SV} &\to \text{SV} \ \text{SP}
  &\qquad \text{SP} &\to \text{P} \ \text{SN}
\end{align*}
$$

dove $F$ è la frase, SN, SV e SP i sintagmi nominale, verbale e
preposizionale, e le categorie lessicali (DET, N, V, AUX, P) riscrivono le
parole. Una **derivazione** parte da $F$ e riscrive un simbolo alla volta
finché restano solo parole; la sua storia è l'**albero di derivazione**. Si
noti la ricorsione di $\text{SN} \to \text{SN}\ \text{SP}$: sei regole
generano infinite frasi, ed è proprio questa regola, in concorrenza con
$\text{SV} \to \text{SV}\ \text{SP}$, a produrre l'ambiguità del binocolo
(l'*attacco del sintagma preposizionale* al nome oppure al verbo, lo stesso
bivio annunciato nella panoramica del capitolo).

`````

## Frecce tra le parole: le dipendenze

C'è un secondo modo di disegnare la stessa struttura, che risale alla
tradizione europea di Lucien Tesnière (il suo *Éléments de syntaxe
structurale* uscì postumo nel 1959): niente scatole, ma **frecce** che
collegano ogni parola alla parola da cui *dipende*, con un'etichetta che ne
dichiara il ruolo.

`````{tab} Elementare

Immagina l'organigramma di una piccola azienda. Ogni parola ha esattamente un
capo, tranne una: il verbo principale, che è l'amministratore delegato. In «Il
gatto nero salta sul muro» comanda «salta»: per lui lavorano «gatto», con la
qualifica di *soggetto* (chi compie l'azione), e «muro», con la qualifica di
*luogo*. A loro volta «il» e «nero» lavorano per «gatto», e «sul» per «muro».
Sei parole, cinque frecce, ogni freccia con la sua mansione: questo è tutto il
formalismo.

E l'ambiguità del binocolo? Diventa una sola domanda da ufficio del
personale: *per chi lavora «binocolo»?* Se lavora per «visto», è lo
strumento con cui ho guardato; se lavora per «uomo», è un accessorio
dell'uomo. Una freccia che cambia datore di lavoro, e il significato della
frase si capovolge.

`````

`````{tab} Superiore

Un'**analisi a dipendenze** di una frase di $n$ parole è un albero diretto ed
etichettato: ogni parola ha esattamente una testa (un solo arco entrante), una
parola (la radice, tipicamente il verbo principale) dipende da un nodo
fittizio *root*, e ogni arco porta una **relazione grammaticale**. Nello
schema di Universal Dependencies {cite}`nivre2016universal`, già incontrato
per il POS tagging, le relazioni principali sono `nsubj` (soggetto), `obj`
(oggetto diretto), `det` (determinante), `amod` (aggettivo modificatore),
`case` (preposizione), `obl` (complemento obliquo), `nmod` (modificatore
nominale).

Perché due formalismi? Le dipendenze pagano meglio nelle lingue a **ordine
flessibile**. L'italiano ammette «Il binocolo l'ho visto io» o «Sul muro
salta, il gatto nero»: una grammatica a costituenti deve prevedere regole per
ogni permutazione, mentre l'albero a dipendenze resta *lo stesso*; a cambiare
è solo l'ordine in cui le parole compaiono sulla riga, cioè il disegno, non la
struttura. È il motivo per cui il progetto UD ha scelto le dipendenze come
lingua franca per annotare con gli stessi criteri più di cento lingue,
dall'italiano al finlandese al giapponese. Quando gli archi, disegnati sopra
la frase, arrivano a incrociarsi si parla di albero **non proiettivo**: raro
in inglese, assai più comune nelle lingue a ordine libero.

`````

{numref}`fig-alberi-sintassi` mette i due formalismi fianco a fianco sulla
frase del binocolo, con una malizia: ciascuno mostra una lettura diversa. A
sinistra, nelle scatole, «con il binocolo» sta **dentro** la scatola di «un
uomo», ed è la lettura in cui il binocolo è dell'uomo. A destra, nelle frecce,
«binocolo» lavora per «visto» e non per «uomo», ed è la lettura opposta, quella
in cui il binocolo è di chi guarda. Per scambiarle basta uno spostamento per
parte: a sinistra tirare fuori la scatola del binocolo da quella dell'uomo e
agganciarla a quella del verbo, a destra spostare la coda della freccia da
«visto» a «uomo». Nel gergo dei due formalismi lo stesso spostamento si dice
così: il sintagma preposizionale passa dal sintagma nominale a quello verbale,
e la relazione `obl` (complemento del verbo) diventa `nmod` (modificatore del
nome).

```{figure} ../figures/alberi-sintassi.svg
:name: fig-alberi-sintassi
:alt: "La frase Ho visto un uomo con il binocolo analizzata due volte. A sinistra un albero a costituenti in teal, in cui il sintagma preposizionale con il binocolo è contenuto nel sintagma nominale un uomo: la lettura in cui il binocolo è dell'uomo. A destra un grafo a dipendenze in terracotta con archi etichettati aux, obj, det, case, obl sopra le parole, in cui l'arco obl collega visto a binocolo: la lettura in cui il binocolo è di chi guarda."
:width: 100%

La stessa frase ambigua nei due formalismi: a sinistra i costituenti di una
lettura, a destra le dipendenze dell'altra. La differenza tra le due letture
è un solo attacco: al nome oppure al verbo.
```

## L'esplosione degli alberi

Con un solo complemento le letture sono due: pazienza. Ma allunghiamo la
frase. In «Ho visto un uomo con il binocolo **nel parco**» il conto si fa così.
Se il binocolo è dell'uomo, il parco può ospitare la scena, l'uomo o il
binocolo: tre letture. Se invece il binocolo è mio, il parco può ospitare solo
la scena o il binocolo, e non l'uomo, perché le scatole si annidano e non si
possono incrociare: altre due. Tre più due fa **cinque**, tutte
grammaticalmente ineccepibili. Aggiungete «**dalla finestra**» e salgono a 14,
poi 42, 132, 429… Sono i
**numeri di Catalan**, e crescono in modo esponenziale: ogni complemento in
coda moltiplica gli alberi per un fattore che si avvicina a quattro. Il
fenomeno ha un articolo di riferimento dal titolo tutto un programma: *Coping
with syntactic ambiguity or how to put the block in the box on the table*,
come mettere il blocco nella scatola sul tavolo (Church e Patil, 1982).

La morale è doppia. Una frase di giornale può avere *migliaia* di alberi
grammaticalmente leciti, quasi tutti assurdi per un lettore umano ma
impeccabili per la grammatica, che non giudica la plausibilità. E nessun
parser può permettersi di elencarli uno per uno: serve un modo di
**condividere i pezzi** comuni a molte analisi e un modo di **scegliere**
l'analisi giusta (le probabilità, o una rete neurale). Il primo dei due ha un
nome che in questo capitolo è già passato due volte, con la griglia della
distanza di edit e con il navigatore di Viterbi: si chiama **programmazione
dinamica**, e vuol dire calcolare una volta sola ogni pezzo che servirà più
volte, tenendoselo da parte.

## Costruire l'albero senza provarle tutte

Due strategie classiche dominano il campo, una per formalismo. Le
presentiamo in forma di tour: l'idea, il costo, il compromesso.

`````{tab} Elementare

**Dal basso, per pezzi.** Il primo metodo, chiamato CKY dalle iniziali dei tre
informatici che lo scoprirono indipendentemente negli anni Sessanta (Cocke,
Kasami e Younger), lavora come si monta un mosaico: prima capisce tutti i
pezzi di due parole («un uomo», «il binocolo»), poi quelli di tre («con il
binocolo»), poi di quattro, incollando sempre due pezzi già capiti, e ogni
pezzo lo calcola **una volta sola**, anche se servirà a dieci letture diverse.
È lo stesso risparmio del navigatore di Viterbi della sezione precedente: mai
rifare due volte la stessa strada. Il conto da pagare è che il lavoro cresce
in fretta con la lunghezza: frase doppia, lavoro circa otto volte tanto.

**Da sinistra, a mosse.** Il secondo metodo legge la frase una parola alla
volta, tenendo un vassoio: a ogni passo o *prende* la parola successiva e la
posa sul vassoio, o *collega* con una freccia le due parole in cima. A
decidere la mossa è un piccolo classificatore neurale, addestrato su migliaia
di frasi già analizzate a mano. Una sola passata e l'albero è fatto:
velocissimo. Il difetto è il solito delle scelte ingorde, lo stesso visto per
la traduzione: una mossa sbagliata all'inizio non si recupera più, e il
rimedio è lo stesso, tenere aperte alcune alternative con la beam search.

`````

`````{tab} Superiore

**CKY.** Richiede la grammatica in *forma normale di Chomsky* (regole
binarie $A \to B\,C$ o lessicali $A \to w$; ogni CFG vi si converte). Il
numero di alberi binari su $n$ foglie è il numero di Catalan
$C_{n-1} = \frac{1}{n}\binom{2(n-1)}{n-1}$, che cresce all'incirca come
$4^{n}$: l'enumerazione è fuori discussione. CKY riempie una tabella triangolare
indicizzata dagli **intervalli** della frase: $T[i,j]$ è l'insieme delle
categorie che possono coprire le parole dalla posizione $i$ alla $j$
(esclusa), calcolato dal corto verso il lungo con la ricorrenza

$$
T[i,j] = \big\{\, A \;:\; A \to B\,C \in R,\ \exists\,k,\;
B \in T[i,k],\ C \in T[k,j] \,\big\},
$$

dove $k$ scorre sui punti di taglio interni all'intervallo e $R$ sono le
regole. Le celle sono $O(n^2)$, ogni cella prova $O(n)$ tagli per ognuna delle
$|R|$ regole: costo totale $O(n^3\,|R|)$, contro l'esplosione esponenziale
degli alberi espliciti. La stessa ricorrenza, con somme al posto delle unioni,
*conta* gli alberi (è il codice qui sotto); con probabilità sulle regole
(**PCFG**, stimate contando su un treebank) e massimi al posto delle somme
restituisce l'albero più probabile: è Viterbi, trasportato dai prefissi agli
intervalli.

**Parsing a transizioni.** Per le dipendenze lo standard è lo *shift-reduce*:
una configurazione è una terna (pila $\sigma$, buffer $\beta$, archi $A$) e le
mosse della variante *arc-standard* sono tre; `shift` (sposta la prossima
parola sulla pila), `left-arc` e `right-arc` (creano un arco etichettato tra
le due parole in cima e ne rimuovono la dipendente). Una frase di $n$ parole
si analizza in circa $2n$ mosse: costo **lineare**. La mossa la sceglie un
classificatore sullo stato corrente; dalla svolta neurale (Chen e Manning,
2014) i tratti simbolici sono rimpiazzati dagli embedding delle parole su pila
e buffer, dati in pasto a un MLP. La decodifica greedy propaga gli errori;
beam search e modelli globali attenuano il problema. Sull'inglese del Penn
Treebank i parser neurali moderni superano il 95% di archi corretti (UAS,
*unlabeled attachment score*: la quota di parole agganciate alla testa
giusta).

`````

## Contare le letture in trenta righe di Python

La versione «contabile» di CKY sta in una pagina: una grammatica giocattolo
di sei regole più lessico, e una tabella che invece di memorizzare gli
alberi li conta.

```python
# Grammatica giocattolo in forma normale di Chomsky (6 regole + lessico)
lessico = {
    "ho": {"AUX"}, "visto": {"V"}, "un": {"DET"}, "il": {"DET"},
    "uomo": {"N"}, "binocolo": {"N"}, "cappello": {"N"},
    "gatto": {"N"}, "con": {"P"},
}
regole = [                    # A -> B C
    ("SN", "DET", "N"),       # "un uomo", "il binocolo"
    ("SP", "P",   "SN"),      # "con il binocolo"
    ("SN", "SN",  "SP"),      # attacco al nome: l'uomo HA il binocolo
    ("SV", "V",   "SN"),      # "visto un uomo"
    ("SV", "SV",  "SP"),      # attacco al verbo: ho guardato COL binocolo
    ("F",  "AUX", "SV"),      # "ho" + sintagma verbale
]

def conta_alberi(parole):
    n = len(parole)
    # tab[i][j] = {categoria: quanti alberi coprono parole[i:j]}
    tab = [[{} for _ in range(n + 1)] for _ in range(n + 1)]
    for i, w in enumerate(parole):
        for cat in lessico[w]:
            tab[i][i + 1][cat] = 1
    for lung in range(2, n + 1):          # intervalli dal corto al lungo
        for i in range(n - lung + 1):
            j = i + lung
            for k in range(i + 1, j):     # punto di taglio
                for A, B, C in regole:
                    if B in tab[i][k] and C in tab[k][j]:
                        tab[i][j][A] = (tab[i][j].get(A, 0)
                                        + tab[i][k][B] * tab[k][j][C])
    return tab[0][n].get("F", 0)

print(conta_alberi("ho visto un uomo con il binocolo".split()))        # 2
print(conta_alberi(
    "ho visto un uomo con il binocolo con il cappello".split()))       # 5
```

Le due letture del binocolo ci sono; con il secondo complemento le analisi
diventano cinque, il passo successivo dei numeri di Catalan, compresa quella
in cui è il *binocolo* a indossare il cappello: la grammatica la genera senza
batter ciglio, e sta a un modello probabilistico (o a una rete) declassarla.
Per toccare con mano l'esplosione, aggiungi `"con il gatto"` in coda e
riconta: quattordici.

## Da dove vengono gli alberi: i treebank

Chi glieli insegna, ai modelli, gli alberi giusti? Persone. Un **treebank** è
un corpus in cui ogni frase è accompagnata dal suo albero sintattico,
tracciato e ricontrollato da annotatori esperti: un lavoro linguistico lento e
prezioso, che nel NLP ha fatto da spartiacque. Il capostipite è il **Penn
Treebank** {cite}`marcus1993building`, costruito all'Università della
Pennsylvania nei primi anni Novanta: oltre quattro milioni e mezzo di parole
etichettate per categoria grammaticale e un nucleo di circa un milione di
parole di articoli del *Wall Street Journal* annotato con alberi a
costituenti. Da quelle decine di migliaia di frasi, per vent'anni, i parser
hanno imparato le probabilità delle regole ed è nato lo standard con cui
confrontarli: la grammatica ha smesso di essere scritta a mano ed è diventata
qualcosa che si *legge nei dati*. Il progetto Universal Dependencies ha
rifatto l'operazione in scala mondiale e in salsa a dipendenze: per l'italiano
il treebank di riferimento è ISDT (*Italian Stanford Dependency Treebank*),
circa quattordicimila frasi nate dalla convergenza di risorse costruite negli
anni da gruppi di Torino e Pisa. È su questi alberi che si addestrano (ieri
con le PCFG, oggi con le reti) tutti i parser di cui abbiamo parlato.

## La sintassi al tempo dei modelli giganti

Domanda inevitabile: i grandi modelli linguistici fanno parsing? No, non nel
senso di questa sezione. Un LLM addestrato a predire la parola successiva non
produce alberi, e nessuno glieli ha mostrati. Eppure gli studi di *probing*
(piccole sonde addestrate a leggere le rappresentazioni interne di un modello)
raccontano una storia interessante. Mentre un modello come BERT
{cite}`devlin2019bert` legge una frase, dentro di lui si accendono migliaia di
numeri, uno strato dopo l'altro: sono le sue **attivazioni**, cioè lo stato in
cui la frase lo mette. Ebbene, da quei numeri si possono ricostruire con buona
approssimazione le distanze fra le parole nell'albero a dipendenze, quello che
al modello nessuno ha mai mostrato (Hewitt e Manning, 2019). È un
indizio che qualcosa di simile alla struttura sintattica emerga implicitamente
durante l'addestramento: un indizio, non la prova che il modello la usi come
farebbe un linguista.

Il parsing esplicito, intanto, non è andato in pensione. Serve alla
linguistica computazionale, che con cento lingue annotate con gli stessi
criteri può confrontare le grammatiche del mondo su base empirica; alla
correzione grammaticale, dove non basta accorgersi che una frase «suona
male» ma bisogna dire *dove* e *perché*; alle lingue a poche risorse, per
le quali i miliardi di parole di un LLM non esistono ma un treebank da
qualche migliaio di frasi si può costruire. E resta il modo più onesto di
*misurare* quanto una macchina abbia colto la struttura: l'albero o è
giusto o non lo è.

Con le etichette della sezione precedente e gli alberi di questa, una
macchina può dire chi fa che cosa a chi dentro una frase isolata. Ma le
frasi, nella vita, arrivano in botta e risposta: domande, risposte,
malintesi, sottintesi. La prossima sezione porta tutto questo in scena: il
dialogo tra persone e macchine.

`````{tab} Elementare
```{admonition} Da ricordare
:class: important
- L'ambiguità di «Ho visto un uomo con il binocolo» non sta nelle parole né nel
  mestiere che ciascuna fa: le etichette della sezione precedente sono identiche
  nelle due letture. Sta in **come le parole si raggruppano**, e cioè se «con il
  binocolo» si attacca all'uomo o al vedere.
- Le **scatole del trasloco**: certi gruppi di parole viaggiano insieme, e lo si
  scopre con due prove da fare a orecchio, sostituire il gruppo con una parola
  sola e spostarlo tutto intero. Le scatole stanno dentro altre scatole, fino
  alle singole parole.
- L'**organigramma**: la stessa struttura si può disegnare con delle frecce,
  ogni parola con un capo solo e il verbo principale in cima. L'ambiguità del
  binocolo diventa una domanda sola: per chi lavora «binocolo»? Le frecce
  reggono meglio le lingue che spostano le parole con libertà, come l'italiano,
  ed è il formalismo con cui sono annotate più di cento lingue.
- Le letture possibili **esplodono**: due con un complemento, cinque con due,
  poi 14, 42, 132. Nessun programma può elencarle tutte, quindi ne condivide i
  pezzi (la stessa astuzia della griglia e del navigatore delle sezioni
  precedenti) e poi ne sceglie una, con delle probabilità o con una rete.
- Due modi di costruire l'analisi: **il mosaico**, che capisce prima i pezzi
  corti e poi incolla, sicuro ma costoso; e **il vassoio**, che legge da
  sinistra a destra decidendo mossa per mossa, velocissimo ma senza ripensamenti
  (e il rimedio è quello già visto per la traduzione, tenere aperte alcune
  alternative).
- Gli alberi giusti li insegnano delle **persone**: i *treebank* sono corpora in
  cui ogni frase è stata analizzata a mano, e da lì i programmi imparano. Per
  l'italiano ce n'è uno, ISDT, di circa quattordicimila frasi.
- I modelli giganti l'analisi logica **non la fanno**, e nessuno gliel'ha
  insegnata; ma andando a guardare dentro di loro si ritrova qualcosa che le
  somiglia. È un indizio, non una prova.
```
`````

`````{tab} Superiore
```{admonition} Da ricordare
:class: important
- L'ambiguità di «Ho visto un uomo con il binocolo» è **strutturale**: le
  etichette POS sono identiche nelle due letture; a cambiare è l'*attacco*
  del sintagma preposizionale, al nome o al verbo.
- I **costituenti** raggruppano le parole in sintagmi annidati (prove di
  sostituzione e spostamento); il formalismo è la **grammatica
  context-free** di Chomsky (1956), con regole di riscrittura e alberi di
  derivazione.
- Le **dipendenze** collegano ogni parola alla sua testa con una relazione
  etichettata (`nsubj`, `obj`, `obl`…); reggono bene le lingue a ordine
  flessibile come l'italiano e sono lo standard di **Universal
  Dependencies**.
- Il numero di alberi possibili esplode con i **numeri di Catalan**: 2, 5,
  14, 42, 132… Nessun parser può enumerarli.
- **CKY** è programmazione dinamica sugli intervalli, $O(n^3)$, parente di
  Viterbi; il **parsing a transizioni** costruisce l'albero a dipendenze in
  tempo lineare con mosse shift-reduce scelte da un classificatore neurale.
- I parser si addestrano sui **treebank**, il Penn Treebank per i costituenti,
  le UD (per l'italiano: ISDT) per le dipendenze.
- I **LLM** non producono alberi, ma il probing suggerisce che una parte
  della struttura sintattica emerga nelle loro rappresentazioni; il parsing
  esplicito resta utile a linguistica, correzione grammaticale e lingue a
  poche risorse.
```
`````
