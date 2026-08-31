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
letture sono due **strutture** diverse costruite con gli stessi mattoni, e non
due sfumature dello stesso significato.

La disciplina che studia queste strutture è la **sintassi**; costruirle
automaticamente si chiama **parsing**, o analisi sintattica, e un programma che
lo fa si chiama *parser*. Se la sezione precedente era l'analisi grammaticale
di scuola, questa è l'analisi logica: chi fa che cosa, a chi, con che cosa.

Per disegnare la struttura di una frase la linguistica ha prodotto due grandi
famiglie di mappe. La prima raggruppa le parole in **blocchi** annidati uno
dentro l'altro, e si chiamano *costituenti*; la seconda collega le parole a due
a due con delle **frecce**, e si chiamano *dipendenze*. Due modi di disegnare
la stessa cosa, come una città si può descrivere con i quartieri o con le
strade, e il NLP li usa tutti e due.

## Scatole dentro scatole: i costituenti

L'osservazione di partenza è che certe sequenze di parole si comportano come
un blocco unico. Nell'esempio ricorrente del libro, «Il gatto nero salta sul
muro», il gruppo «il gatto nero» si sposta e si sostituisce come un pezzo solo.
Questi blocchi si chiamano **sintagmi**, o costituenti, e per non perdersi
conviene fissare adesso i tre nomi che tornano più spesso: si dice sintagma
*nominale* il blocco che ha un nome per protagonista («il gatto nero»),
sintagma *verbale* quello che ha un verbo («salta sul muro»), e sintagma
*preposizionale* quello che comincia con una preposizione («sul muro»).

`````{tab} Elementare

Le forchette, in un trasloco, non si portano in strada una per una: si chiudono
in una scatola, ed è la scatola che viaggia. La frase funziona uguale, e si
riconosce quali parole viaggiano insieme con due prove da fare a orecchio.

Prova di **sostituzione**: se un gruppo di parole si può rimpiazzare con una
parola sola, è una scatola. «**Il gatto nero** salta sul muro» diventa «**Lui**
salta sul muro», e regge.

Prova di **spostamento**: una scatola si sposta tutta intera. «**Sul muro**
salta il gatto nero» suona benissimo, mentre «*Muro il gatto nero salta sul*»
non è italiano: abbiamo strappato il cartone e le forchette sono per terra.

E dentro la scatola «sul muro» c'è una scatolina, «il muro»: scatole dentro
scatole, fino alle parole. Verso l'alto, invece, non c'è un tetto. Una scatola
può stare dentro una scatola dello stesso tipo, e quella dentro un'altra
ancora: «il gatto del vicino del piano di sopra» ne infila tre una dentro
l'altra, e si potrebbe continuare per un pezzo. Sono sempre gli stessi pochi
modi di inscatolare, riusati; bastano quelli a costruire frasi lunghe quanto si
vuole, comprese quelle che nessuno ha mai pronunciato.

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
stretti degli $n$-gram), le grammatiche a struttura sintagmatica e le
grammatiche trasformazionali. La tesi che fece scuola: gli stati finiti non
bastano, perché la sintassi annida dipendenze a
distanza arbitraria; da quella gerarchia di formalismi (oggi detta *gerarchia
di Chomsky*) l'informatica ha attinto anche i linguaggi di programmazione.

Una CFG è una quadrupla $G = (N, \Sigma, R, S)$, dove $N$ è l'insieme dei
simboli **non terminali** (le categorie sintattiche), $\Sigma$ il
vocabolario dei **terminali** (le parole), $R$ un insieme di **regole di
riscrittura** della forma $A \to \alpha$ con $A \in N$ e $\alpha$ sequenza
di simboli, e $S$ il simbolo iniziale. Una grammatica giocattolo per il
nostro frammento d'italiano:

$$
\begin{aligned}
F &\to \text{AUX} \ \text{SV}
  &\qquad \text{SN} &\to \text{DET} \ \text{N} \\
\text{SV} &\to \text{V} \ \text{SN}
  &\qquad \text{SN} &\to \text{SN} \ \text{SP} \\
\text{SV} &\to \text{SV} \ \text{SP}
  &\qquad \text{SP} &\to \text{P} \ \text{SN}
\end{aligned}
$$

dove $F$ è il simbolo iniziale, cioè la frase, SN, SV e SP i sintagmi
nominale, verbale e preposizionale, e le categorie lessicali (DET, N, V, AUX,
P) riscrivono le parole. Una **derivazione** parte da $F$ e riscrive un simbolo
alla volta finché restano solo parole; la sua storia è l’**albero di
derivazione**. Si noti la ricorsione di $\text{SN} \to \text{SN}\ \text{SP}$:
sei regole generano infinite frasi, ed è proprio questa regola, in concorrenza
con $\text{SV} \to \text{SV}\ \text{SP}$, a produrre l'ambiguità del binocolo
(l’*attacco del sintagma preposizionale*, al nome oppure al verbo).

`````

## Frecce tra le parole: le dipendenze

C'è un secondo modo di disegnare la stessa struttura, che risale alla
tradizione europea di Lucien Tesnière (il suo *Éléments de syntaxe
structurale* uscì postumo nel 1959): niente scatole, ma **frecce** che
collegano ogni parola alla parola da cui *dipende*, con un'etichetta che ne
dichiara il ruolo.

`````{tab} Elementare

In un organigramma aziendale ogni impiegato ha un capo, e uno solo non ce l'ha.
Una frase, disegnata così, ha la stessa forma: ogni parola dipende da un'altra
parola, tranne il verbo principale, che è l'amministratore delegato. In «Il
gatto nero salta sul muro» comanda «salta»: per lui lavorano «gatto», con la
qualifica di *soggetto* (chi compie l'azione), e «muro», con la qualifica di
*luogo*. A loro volta «il» e «nero» lavorano per «gatto», e «sul» per «muro».
Sei parole in fila, cinque frecce fra una parola e l'altra, ognuna con la sua
mansione. Sopra l'amministratore delegato non c'è nessuno: in cima al foglio
resta solo il nome dell'azienda, il punto da cui il disegno parte. Se contate
«su» e «il» separati, come si fa quando si etichetta parola per parola, le
caselle dell'organigramma diventano sette e le frecce sei: la
struttura non cambia, cambia solo quanto finemente si taglia il testo prima di
disegnarla.

E l'ambiguità del binocolo? Diventa una sola domanda da ufficio del
personale: *per chi lavora «binocolo»?* Se lavora per «visto», è lo
strumento con cui ho guardato; se lavora per «uomo», è un accessorio
dell'uomo. Una freccia che cambia datore di lavoro, e il significato della
frase si capovolge.

C'è un motivo pratico per cui questo disegno è quello che si usa quando si
lavora su molte lingue insieme. In italiano si può dire «il gatto nero salta
sul muro», ma anche «sul muro salta il gatto nero», e in altre lingue le parole
girano ancora di più. Con le scatole ogni riordino richiede regole nuove, perché
le scatole stanno in fila e la fila cambia. Con le frecce no: chi comanda chi
resta identico, cambia solo dove le parole sono scritte sulla riga. Ecco perché
il progetto che annota con gli stessi criteri più di centocinquanta lingue,
quell'Universal Dependencies già incontrato per le etichette, ha scelto le
frecce, e infatti si chiama «delle dipendenze».

C'è un effetto collaterale. Quando le parole si mescolano parecchio, due frecce
disegnate sopra la riga possono accavallarsi, perché ciascuna deve raggiungere
il proprio capo che nel frattempo è finito lontano. Dove l'ordine delle parole
è rigido succede di rado; dove è libero è ordinaria amministrazione, e chi
costruisce il disegno deve mettere in conto anche quel caso.

`````

`````{tab} Superiore

Un’**analisi a dipendenze** di una frase di $n$ parole è un albero diretto ed
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
lingua franca per annotare con gli stessi criteri più di centocinquanta
lingue, dall'italiano al finlandese al giapponese. Quando gli archi, disegnati
sopra la frase, arrivano a incrociarsi si parla di albero **non proiettivo**:
raro in inglese, assai più comune nelle lingue a ordine libero.

`````

{numref}`fig-alberi-sintassi` mette i due disegni fianco a fianco sulla frase
del binocolo, con una malizia: ciascuno mostra una lettura diversa. A sinistra,
nelle scatole, «con il binocolo» sta **dentro** la scatola di «un uomo», ed è
la lettura in cui il binocolo è dell'uomo. A destra, nelle frecce, «binocolo»
lavora per «visto» e non per «uomo», ed è la lettura opposta, quella in cui il
binocolo è di chi guarda.

Per scambiarle basta uno spostamento per parte: a sinistra tirare fuori la
scatola del binocolo da quella dell'uomo e agganciarla a quella del verbo, a
destra spostare la coda della freccia da «visto» a «uomo». Chi lavora in questo
campo lo dice in due gerghi diversi, ed è utile riconoscerli quando si
incontrano. Con le scatole si dice che il sintagma preposizionale passa dal
sintagma nominale a quello verbale, cioè che «con il binocolo» smette di stare
dentro «un uomo» e va a stare dentro «ho visto». Con le frecce si dice che la
relazione cambia sigla, da `obl` a `nmod`: `obl` sta per «complemento del
verbo» e `nmod` per «modificatore del nome», e sono esattamente le due
mansioni che il binocolo può avere nell'organigramma, dipendere dal vedere o
dipendere dall'uomo.

```{figure} ../figures/alberi-sintassi.svg
:name: fig-alberi-sintassi
:alt: "La frase Ho visto un uomo con il binocolo analizzata due volte. A sinistra un albero a costituenti in teal, in cui il sintagma preposizionale con il binocolo è contenuto nel sintagma nominale un uomo: la lettura in cui il binocolo è dell'uomo. A destra un grafo a dipendenze in terracotta con archi etichettati aux, obj, det, case, obl sopra le parole, in cui l'arco obl collega visto a binocolo: la lettura in cui il binocolo è di chi guarda."
:width: 100%

La stessa frase ambigua nei due modi di disegnarla: a sinistra i costituenti
di una lettura, a destra le dipendenze dell'altra. La differenza tra le due
letture è un solo aggancio: al nome oppure al verbo.
```

Adesso guardate il disegno di sinistra, e si capisce anche perché uno schema
del genere si chiama **albero**, che è il nome che si userà da qui in avanti.
Le scatole annidate, tirate su in verticale, formano un albero rovesciato: ogni
scatola diventa un **nodo**, cioè un punto in cui il disegno si dirama, e i
rami che ne escono sono le scatole che quella scatola contiene. In cima c'è il
nodo della frase intera; in fondo, dove non c'è più niente da aprire, ci sono
le **foglie**, che sono le singole parole. Anche l'organigramma delle frecce è
un albero, con il verbo principale come nodo in cima e i suoi sottoposti giù
per i rami. «Albero sintattico» e «analisi di una frase» sono quindi la stessa
cosa detta in due modi.

## L'esplosione degli alberi

Con un solo complemento le letture sono due: pazienza. Ma allunghiamo la
frase, e mettiamoci «Ho visto un uomo con il binocolo **nel parco**». Adesso i
complementi da sistemare sono due, e ciascuno si può agganciare a qualcosa che
lo precede: al *vedere*, all’*uomo*, o al *binocolo*.

La regola da tenere è una sola, ed è quella del trasloco: due scatole o sono
una dentro l'altra, o sono separate, mai mezze sovrapposte. Ecco perché non si
può dire che «nel parco» si aggancia all'uomo mentre «con il binocolo» si
aggancia al vedere: le due scatole si incrocerebbero, e il cartone non lo
permette.

Contiamo, allora, tenendo fermo il primo complemento e provando tutti gli
agganci del secondo.

**Caso A: il binocolo è dell'uomo**, cioè «con il binocolo» sta dentro la
scatola di «un uomo». Dove può andare «nel parco»? Al *vedere* (ho visto nel
parco), all’*uomo con il binocolo* (l'uomo col binocolo che stava nel parco),
oppure al *binocolo* (il binocolo del parco, quello lì in dotazione). Tre.
Attenzione: agganciarlo al solo «uomo» *senza* il binocolo ricade nella stessa
scatola e non in una quarta possibilità: dal momento che il binocolo è già dentro
l'uomo, non c'è modo di infilare il parco fra i due senza tagliare il cartone.

**Caso B: il binocolo è mio**, cioè «con il binocolo» è già agganciato al
vedere. Dove può andare «nel parco»? Al *vedere*, oppure al *binocolo*. Non
all’*uomo*: per farlo dovrebbe scavalcare «con il binocolo», che sta più a
sinistra ma è agganciato più in alto, e le scatole si incrocerebbero. Due.

Tre più due fa **cinque**, tutte grammaticalmente ineccepibili.

Aggiungete «**dalla finestra**» e salgono a 14, poi 42, 132, 429… Sono i
**numeri di Catalan**, dal matematico belga Eugène Catalan che li studiò
nell'Ottocento, e sono la risposta a una domanda che torna dappertutto: in
quanti modi si può mettere fra parentesi una fila di cose. Il che è
esattamente il nostro problema, perché inscatolare le parole e metterle fra
parentesi sono la stessa operazione.

Crescono in fretta, e conviene guardare di quanto. Ogni complemento che si
aggiunge moltiplica gli alberi per un fattore che si può leggere dai numeri
stessi: da 2 a 5 il fattore è due e mezzo, da 5 a 14 quasi tre, da 14 a 42 è
tre esatto, da 42 a 132 poco più di tre. Il fattore, cioè, non è fisso: sale a
ogni passo, e continua a salire avvicinandosi a quattro senza mai arrivarci.
Continuando: 429 con sei complementi, poi 1.430, 4.862, e con nove si arriva a
16.796. Nove complementi in una frase sono tanti ma non impossibili, e le
analisi grammaticalmente lecite sono già sedicimila.
Il
fenomeno ha un articolo di riferimento dal titolo tutto un programma, perché il
titolo stesso è ambiguo: *Coping with syntactic ambiguity or how to put the
block in the box on the table* (Church e Patil, 1982). Come mettere il blocco
nella scatola sul tavolo: ma il tavolo sostiene la scatola, o è lì che va messo
il blocco?

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

Due strategie classiche dominano il campo, una per ciascuno dei due disegni: la
prima costruisce le scatole, la seconda le frecce. Le presentiamo con lo stesso
schema: l'idea, il costo, il compromesso.

`````{tab} Elementare

Sul tavolo c'è un mosaico a metà. Due tessere che combaciano formano un'isola,
un'altra coppia ne forma un'altra, e le isole si uniscono fra loro quando sono
pronte. Il primo metodo monta la frase così, e si chiama **CKY** dalle iniziali
di Cocke, Kasami e Younger, che negli anni Sessanta lo scoprirono ciascuno per
conto suo. Prima mette insieme tutti i tratti lunghi due parole («un uomo», «il
binocolo»), poi quelli di tre («con il binocolo»), poi di quattro, e a ogni
giro incolla due isole già montate. Un'isola montata non si smonta più: «il
binocolo» lo capisce una volta sola, anche se poi servirà a dieci letture
diverse. È il risparmio del navigatore di Viterbi, spostato dalle parole ai
tratti di frase.

Il tavolo però si riempie, e si può contare di quanto. Mettete in fila quattro
parole. Le isole da provare, una per ogni scelta di dove cominciare e dove
finire, sono dieci. Con otto parole diventano trentasei, quasi quattro volte
tante. Ogni isola, per giunta, si può tagliare in due nel doppio dei punti di
prima. Quattro per due fa otto, quindi a raddoppiare la
lunghezza della frase il lavoro sul tavolo diventa circa otto volte tanto.

Il secondo metodo gioca a carte. Le parole della frase arrivano una alla volta,
nel loro ordine, e accanto c'è una **pila** di carte scoperte di cui si vedono
solo le prime due. A ogni turno si fa una mossa sola, e le mosse sono tre.

1. **Prendi**: la parola successiva della frase sale in cima alla pila.
2. **Collega verso sinistra**: fra le due carte in cima comanda quella sopra, e
   quella sotto esce dal tavolo.
3. **Collega verso destra**: il contrario, comanda quella sotto ed esce quella
   sopra.

Le ultime due sono la stessa mossa nei due versi, e servono entrambe perché a
volte comanda la parola arrivata prima («salta» comanda «muro»), a volte quella
arrivata dopo («gatto» comanda «il»). La carta che si collega esce
di scena, perché il suo posto nell'organigramma è ormai deciso. Si tira avanti
finché le parole sono finite e sulla pila resta una carta sola, il capo di
tutti. L'albero è fatto.

A giocare è qualcuno che ha visto migliaia di frasi già analizzate a mano.
Guarda le due carte in cima e quante parole restano da prendere, poi butta giù
la mossa senza pensarci. Due mosse per parola, una sola passata sulla frase, e
la partita è chiusa.

Una carta uscita dal tavolo, però, non ci torna più. Un collegamento messo
storto alla terza parola resta storto fino all'ultima, ed è il difetto di ogni
scelta ingorda, quella della traduzione compresa. Il rimedio è tenere aperte
tre o quattro partite invece di una, e scartare alla fine quelle andate peggio.

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
una configurazione è una terna (pila $\sigma$, buffer $\beta$, archi
$\mathcal{A}$) e le mosse della variante *arc-standard* sono tre; `shift`
(sposta la prossima parola sulla pila), `left-arc` e `right-arc` (creano un
arco etichettato tra le due parole in cima e ne rimuovono la dipendente). La
configurazione finale ha il buffer vuoto e sulla pila la sola radice, e una
frase di $n$ parole ci arriva in circa $2n$ mosse: costo **lineare**. La mossa
la sceglie un
classificatore sullo stato corrente; dalla svolta neurale (Chen e Manning,
2014) i tratti simbolici sono rimpiazzati dagli embedding delle parole su pila
e buffer, dati in pasto a un MLP. La decodifica greedy propaga gli errori;
beam search e modelli globali attenuano il problema. Sull'inglese del Penn
Treebank i parser neurali moderni superano il 95% di archi corretti (UAS,
*unlabeled attachment score*: la quota di parole agganciate alla testa
giusta).

`````

## Contare le letture in trenta righe di Python

Serve prima una parola sulle **regole**, perché nel programma qui sotto ce ne
sono sei e finora non ne abbiamo parlato. Una grammatica, in questo mestiere, è
un elenco di regole di montaggio, e ciascuna dice come due pezzi ne fanno uno
più grande. «Un articolo seguito da un nome fa un sintagma nominale» è una
regola; «un sintagma nominale seguito da un sintagma preposizionale fa un
sintagma nominale più grande» è la regola che genera l'ambiguità del binocolo,
perché permette a «con il binocolo» di entrare dentro «un uomo». Sei regole
così bastano per la nostra frase, e prima dei treebank di cui parliamo fra poco
le grammatiche si scrivevano tutte a mano, regola per regola, da linguisti in
carne e ossa.

La versione «contabile» di CKY sta allora in una pagina: quelle sei regole, un
elenco di parole con la loro categoria, e una tabella che invece di memorizzare
gli alberi si limita a contarli. Una nota per chi confronta con il conto fatto
a mano poco fa: lì il secondo complemento era «nel parco», qui è «con il
cappello», e solo perché il vocabolario giocattolo del programma conosce una
preposizione sola, «con». La frase cambia, la forma no, e infatti il numero che
esce è lo stesso.

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
diventano cinque, il passo successivo dei numeri di Catalan, compresa quella in
cui è il *binocolo* a indossare il cappello. La grammatica la genera senza
batter ciglio, perché la grammatica dice solo che cosa si può montare, non che
cosa ha senso: a scartare le assurdità tocca a qualcos'altro, e cioè a delle
probabilità imparate su frasi vere o a una rete neurale. Per toccare con mano
l'esplosione, aggiungi `"con il gatto"` in coda e riconta: quattordici.

## Da dove vengono gli alberi: i treebank

Chi glieli insegna, ai modelli, gli alberi giusti? Persone. Un **treebank** è
un corpus in cui ogni frase è accompagnata dal suo albero sintattico,
tracciato e ricontrollato da annotatori esperti: un lavoro linguistico lento e
prezioso, che nel NLP ha fatto da spartiacque. Il capostipite è il **Penn
Treebank** {cite}`marcus1993building`, costruito all'Università della
Pennsylvania nei primi anni Novanta: oltre quattro milioni e mezzo di parole
etichettate per categoria grammaticale e un nucleo di circa un milione di
parole di articoli del *Wall Street Journal* annotato con alberi a
costituenti.

Da quelle decine di migliaia di frasi, per vent'anni, i parser hanno imparato
quanto ciascuna regola di montaggio è frequente, e con quei numeri hanno
imparato a scegliere fra le mille analisi possibili. È il ribaltamento
importante: la grammatica ha smesso di essere scritta a mano, regola per
regola, ed è diventata qualcosa che si *conta nei dati*.

Il progetto Universal Dependencies ha rifatto l'operazione in scala mondiale e
con le frecce al posto delle scatole: per l'italiano il treebank di riferimento
è ISDT (*Italian Stanford Dependency Treebank*), circa quattordicimila frasi
nate dalla convergenza di risorse costruite negli anni da gruppi di Torino e
Pisa. È su questi alberi che si addestrano tutti i parser di cui abbiamo
parlato, ieri con le probabilità sulle regole, oggi con le reti neurali.

## La sintassi al tempo dei modelli giganti

Domanda inevitabile: i grandi modelli linguistici (in sigla **LLM**, *large
language model*) fanno parsing? No, non nel senso di questa sezione. Un modello
addestrato a indovinare la parola successiva non produce alberi, e nessuno
glieli ha mai mostrati.

Eppure c'è un filone di studi che racconta una storia interessante, e si chiama
*probing*, «sondaggio». Mentre un modello come BERT {cite}`devlin2019bert` legge
una frase, dentro di lui si accendono migliaia di numeri: sono le sue
**attivazioni**, cioè lo stato in cui quella frase particolare lo mette.

Il probing consiste nel prendere quei numeri e attaccarci sopra una **sonda**,
cioè un secondo programmino, minuscolo, che si addestra a ricavarne una certa
informazione. Il punto sta tutto nel «minuscolo». Se la sonda fosse una rete
grande, potrebbe imparare il compito per conto suo, e allora non avremmo
scoperto niente sul modello: avremmo scoperto che una rete grande impara la
grammatica, cosa che già sapevamo. Se invece la sonda è tenuta così piccola da
non poter imparare quasi nulla e ci riesce lo stesso, l'unica spiegazione è che
l'informazione nei numeri di partenza ci fosse già, e alla sonda sia bastato
andarla a leggere.

Hewitt e Manning, nel 2019, provano a ricavarne le **distanze nell'albero**,
cioè quanti passi bisogna fare, di freccia in freccia, per andare da una parola
a un'altra nell'organigramma della frase. In «il gatto nero salta sul muro»,
«nero» dista un passo da «gatto» e due da «salta». Ebbene: quelle distanze si
ricostruiscono con buona approssimazione, pur essendo una cosa che al modello
nessuno ha mai mostrato. È un indizio che qualcosa di simile alla struttura
sintattica si formi da sola durante l'addestramento: un indizio, non la prova
che il modello la usi come farebbe un linguista.

Il parsing esplicito, intanto, non è andato in pensione, e serve almeno a tre
mestieri.

Serve alla **linguistica**, che avendo tutte quelle lingue annotate con gli
stessi criteri può finalmente confrontare le grammatiche del mondo contando,
invece che per impressione. Serve alla **correzione grammaticale**, dove non
basta accorgersi che una frase suona male: bisogna dire dove e perché, e per
dirlo bisogna avere in mano la struttura. E serve alle **lingue di cui esiste
poco testo**, per le quali i miliardi di parole che un modello gigante
richiederebbe non esistono e non esisteranno, mentre un treebank da qualche
migliaio di frasi si può costruire in un paio d'anni di lavoro.

E resta il modo più onesto di *misurare* quanto una macchina abbia colto la
struttura di una frase, che è anche il più semplice: si prende una frase di cui
qualcuno ha già disegnato l'albero giusto, si guarda l'albero prodotto dalla
macchina, e si contano le parole agganciate al capo corretto. Nessuna
interpretazione, nessun giudizio da dare: quella freccia o punta alla parola
giusta o no. È il contrario di quel che succede con un chatbot, dove la
risposta buona non è una sola, e vedremo nella prossima sezione quanto quella
differenza pesi.

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
  alle singole parole; e una scatola può stare dentro una dello stesso tipo,
  così che pochi modi di inscatolare, riusati, bastino a frasi lunghe quanto si
  vuole, comprese quelle che nessuno ha mai pronunciato.
- L’**organigramma**: la stessa struttura si può disegnare con delle frecce,
  ogni parola con un capo solo e il verbo principale in cima. L'ambiguità del
  binocolo diventa una domanda sola: per chi lavora «binocolo»? Le frecce
  reggono meglio le lingue che spostano le parole con libertà, come l'italiano,
  ed è il modo in cui sono annotate più di centocinquanta lingue.
- Scatole o frecce, il disegno che ne esce si chiama **albero**: in cima la
  frase intera, in fondo le singole parole come foglie.
- Gli alberi possibili **esplodono**: due con un complemento, cinque con due,
  poi 14, 42, 132. Nessun programma può elencarli tutti, quindi ne condivide i
  pezzi (la stessa astuzia della griglia e del navigatore delle sezioni
  precedenti) e poi ne sceglie uno, con delle probabilità o con una rete.
- Due modi di costruire l'analisi: **il mosaico**, che capisce prima i pezzi
  corti e poi incolla, sicuro ma costoso; e **la pila**, che legge da sinistra
  a destra decidendo mossa per mossa, velocissimo ma senza ripensamenti (e il
  rimedio è quello già visto per la traduzione, tenere aperte alcune
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
  etichette POS sono identiche nelle due letture; a cambiare è l’*attacco*
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
