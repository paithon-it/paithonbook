# Chi parla con chi: le topologie del coordinamento

Nell'agosto del 1964 un ingegnere della RAND Corporation, Paul Baran, pubblica
il primo di undici memorandum su come costruire una rete di comunicazione che
continui a funzionare dopo che una parte dei suoi nodi è stata distrutta. La
prima figura non contiene formule: contiene tre disegni. Una rete
**centralizzata**, con un nodo al centro e tutti gli altri appesi a lui. Una
rete **decentralizzata**, fatta di piccoli centri collegati fra loro. Una rete
**distribuita**, una maglia in cui ogni nodo ha molte strade per raggiungere
ogni altro. Stessi apparati, stesso mestiere, tre destini diversi. La domanda
che Baran mette davanti al lettore non è quanti nodi servano: è che forma
debbano avere.

È la domanda di questa sezione. La precedente ha contato quanto costa una
squadra di agenti, e in fondo al conto dei token ha lasciato una frase che vale
la pena riprendere: **la topologia decide l'esponente**. La stessa squadra di
quattro agenti costa $O(N^2R^2)$ token se tutti leggono tutto e $O(NR^2)$ se
ciascuno legge solo il proprio. Non cambia il modello, non cambia il numero dei
partecipanti: cambia chi parla con chi.

## Il sistema è un grafo

La rappresentazione giusta è la più spoglia possibile: un **grafo**, i cui nodi
sono gli agenti e i cui archi sono i canali di comunicazione. Un arco non
significa «questi due *possono* scriversi»: dentro un programma tutti possono
scrivere a tutti, basta una riga di codice. Significa «in questo progetto
questi due *si scrivono*, e gli altri no». La topologia è un vincolo che il
progettista impone, e il suo valore sta esattamente in ciò che vieta.

Su un grafo si misurano tre grandezze, le stesse che governano il progetto di
una rete di calcolatori: **quanta strada** fa l'informazione per arrivare dove
serve, **quanto pesa** sul nodo più sollecitato, e **cosa resta in piedi** se
quel nodo smette di funzionare. Ogni topologia è un compromesso fra loro.

`````{tab} Elementare

Immagina venti persone che devono organizzare una festa, e fatti tre domande
sul modo in cui si sono organizzate.

La prima: **per quante mani passa una notizia** prima di arrivare a tutti? Se
c'è un organizzatore che chiama uno per uno, due passaggi bastano sempre. Se
invece ognuno telefona solo al successivo, come in una catena di sant'Antonio,
per andare dal primo all'ultimo servono diciannove telefonate in fila, e chi
sta in fondo la sera prima non sa ancora niente.

La seconda: **quanto lavoro fa il più carico**? Nel primo caso l'organizzatore
di telefonate ne fa diciannove, gli altri una a testa. Nel secondo nessuno ne
fa più di due. Il primo modo è velocissimo, ma tutta la fatica sta su una
persona sola.

La terza, la più sgradevole: **cosa succede se proprio quella persona prende
l'influenza**? Nel primo caso non si sa più niente: diciannove persone che non
si parlano fra loro restano al buio. Nel secondo la catena si spezza in due
tronconi, e metà del gruppo, quella a monte, la notizia ce l'ha comunque.

Nessuna delle due organizzazioni è migliore in assoluto. Ognuna compra una di
quelle tre cose vendendo le altre due, ed è questo, non il numero di invitati,
che decide come andrà la serata.

`````

`````{tab} Superiore

Sia $G = (V, E)$ il grafo di comunicazione, con $|V| = N$ agenti, e sia
$d(u,v)$ la lunghezza del cammino minimo fra due nodi. Le tre grandezze sono:

$$
\mathrm{diam}(G) = \max_{u,v \in V} d(u,v),
\qquad
\Delta(G) = \max_{v \in V} \deg(v),
\qquad
b(v) = \frac{\bigl|\{(s,t) : v \in \mathrm{sp}(s,t)\}\bigr|}{\binom{N}{2}},
$$

dove $\mathrm{diam}(G)$ è il **diametro** (il più lungo dei cammini minimi,
cioè il caso peggiore di quanti passaggi servono), $\Delta(G)$ è il **grado
massimo** (quanti canali fanno capo al nodo più connesso) e $b(v)$ è il
**carico** del nodo, cioè la frazione di coppie $(s,t)$ per cui almeno un
cammino minimo $\mathrm{sp}(s,t)$ attraversa $v$ senza averlo come estremo. È
una versione semplificata della centralità di intermediazione (la *betweenness*)
su due punti: conta le coppie invece di pesarle con la frazione dei loro cammini
minimi che passa davvero per $v$, e le normalizza sul totale $\binom{N}{2}$
delle coppie anziché sulle sole coppie che escludono $v$. La fragilità si legge
da $\kappa(G)$, la **connettività per vertici**: quanti nodi bisogna rimuovere
perché il grafo si spezzi. Negli alberi e nelle
stelle $\kappa(G) = 1$, e ogni nodo interno è un punto di articolazione.

Le tre non sono indipendenti, ed è questo a rendere il progetto interessante.
Abbassare il diametro richiede archi, gli archi si concentrano da qualche parte
e alzano $\Delta$ e $b$; e un nodo con $b$ alto è, per definizione, quello la
cui rimozione taglia più coppie. In una rete di calcolatori $b(v)$ alto
significa un instradatore congestionato; in una squadra di agenti significa una
finestra di contesto che deve contenere tutto ciò che transita, e questo è un
limite più duro, perché la qualità della lettura degrada molto prima del limite
dichiarato (è il *lost in the middle* del context engineering).

`````

Le cinque forme che si incontrano davvero sono in fila nella
{numref}`fig-multiagente-topologie`.

```{figure} ../figures/multiagente-topologie.svg
:name: fig-multiagente-topologie
:alt: "Cinque piccoli grafi affiancati ed etichettati: supervisore e lavoratori (una stella con il supervisore al centro e tre lavoratori che non si collegano fra loro), catena (quattro nodi in fila con frecce a senso unico), gerarchia (un albero con radice, due nodi intermedi e quattro foglie), lavagna condivisa (quattro nodi collegati da frecce bidirezionali a un rettangolo centrale e a nient'altro), mercato (un banditore collegato a tre offerenti, con una linea tratteggiata verso due di loro e una freccia piena verso il terzo, quello a cui il compito viene assegnato)."
:width: 85%

Le cinque topologie di coordinamento: nelle prime tre gli agenti si scrivono
direttamente, nelle ultime due no.
```

## Supervisore e lavoratori

Un nodo riceve il compito, lo scompone, ne distribuisce i pezzi a $N$
lavoratori, raccoglie i risultati e li ricompone. I lavoratori non si parlano
fra loro. È la stella di Baran, ed è la topologia di gran lunga più diffusa,
anche perché nasce da sé quando un programma orchestra delle chiamate a un
modello. I framework conversazionali la costruiscono per esteso: nel sistema di
programmazione multi-agente presentato insieme ad AutoGen, per esempio, un
agente «commander» riceve la domanda dell'utente, fa scrivere il codice a un
agente «writer», lo fa controllare a un agente «safeguard», lo esegue e
ricompone la risposta, mentre writer e safeguard non si scrivono mai
{cite}`wu2024autogen`.

Ottimizza il **controllo**, e la cosa vale più di quanto sembri: esiste un
posto, uno solo, dove sta la verità sullo stato del compito. Chi lo debugga ha
un registro da leggere invece di una conversazione da ricostruire, e chi lo
ferma sa dove mettere il cancello di verifica.

`````{tab} Elementare

È il capo cantiere. Legge il progetto, dice al muratore di fare la parete e
all'elettricista di passare i cavi, e alla fine controlla che le due cose
combacino. I due non si parlano: se serve una modifica, passa da lui.

Il vantaggio è che, quando qualcosa non torna, c'è una persona sola a cui
chiedere cosa sta succedendo. Lo svantaggio è che tutto passa da quella
persona, e la sua giornata ha ventiquattro ore come quella degli altri. Con
tre operai funziona benissimo. Con trenta, il capo cantiere passa la giornata
ad ascoltare rapporti e non gli resta tempo per decidere; e se si ammala, il
cantiere non rallenta, si ferma.

`````

`````{tab} Superiore

Con $N$ lavoratori la stella ha $\mathrm{diam}(G) = 2$ e $\Delta(G) = N$, tutto
concentrato sul centro. Ma il costo che morde non è il numero di archi, è ciò
che vi scorre dentro: il supervisore riceve $N$ rapporti, e se ciascuno è lungo
$\bar{m}$ token la sua finestra cresce come

$$
c_{\text{sup}}(N) \;=\; c_0 + N\,\bar{m},
$$

dove $c_0$ è il contesto iniziale. È **lineare** in $N$, non quadratica come
nella trascrizione condivisa della sezione precedente, ed è esattamente il
guadagno della stella: i lavoratori non si leggono a vicenda. Ma è lineare su
**un solo nodo**. Con $N = 30$ lavoratori che riportano $\bar{m} = 800$ token,
la finestra del supervisore contiene 24.000 token di rapporti prima di
qualunque ragionamento.

La contromisura non è allargare il contesto: è **comprimere all'ingresso**,
imponendo ai lavoratori un formato di risposta stretto (uno schema, non prosa
libera). Portare $\bar{m}$ da 800 a 150 token vale, in questa formula, quanto
ridurre di oltre cinque volte il numero di lavoratori, e non costa nulla in
capacità di lavoro.

`````

Quello che si paga sta tutto sullo stesso nodo. Il supervisore è insieme il
collo di bottiglia (ogni decisione aspetta il suo turno) e il punto singolo di
rottura: se sbaglia lui, nessun altro può accorgersene, perché i lavoratori
vedono solo il proprio pezzo. Sotto una certa taglia è quasi sempre la scelta
giusta; sopra, la stella non si allarga, si spezza.

## La catena

Ognuno riceve dal precedente, fa la sua parte, passa al successivo ed esce di
scena: chi estrae i dati li passa a chi li analizza, che passa a chi scrive il
rapporto. In inglese si chiama *handoff*, ed è la topologia più semplice che
esista, un grafo lineare senza archi all'indietro. Ottimizza la **semplicità**
(non c'è niente da orchestrare: l'ordine è il programma) e il **contesto**,
perché ogni agente vede solo il suo pezzo, con la finestra più pulita di tutte.

Quello che paga è brutale, e lo abbiamo già calcolato: **non c'è ritorno**. Un
errore commesso al secondo passo arriva al quinto travestito da dato di
partenza, e a valle nessuno ha modo di accorgersene, perché a valle nessuno ha
visto l'originale. È la composizione moltiplicativa degli errori nella sua
forma peggiore: con sei passi corretti al 95% ciascuno, la catena intera regge
$0{,}95^6 \approx 0{,}74$, cioè fallisce una volta su quattro pur essendo fatta
di agenti che non sbagliano quasi mai. E siccome il grafo è un cammino, il
diametro cresce come $N$: quindici agenti in fila sono quattordici passaggi fra
il primo e l'ultimo, quattordici occasioni perché una specifica si deformi
strada facendo. Nella tassonomia dei fallimenti reali questa famiglia ha un
nome, **disallineamento fra agenti**, ed è la più insidiosa perché non produce
sintomi: due agenti che hanno in testa due versioni diverse del compito si
scrivono messaggi cortesi e ben formati fino in fondo {cite}`cemri2025why`.

La catena si usa quando i passi sono davvero sequenziali e ciascuno ha un
criterio di verifica proprio, così che il cancello stia **dentro** ogni anello e
non alla fine. Senza cancelli intermedi, ogni agente aggiunto alla fila è un
fattore moltiplicativo minore di uno.

## La gerarchia

Quando la stella non regge più, la mossa naturale è annidarla: supervisori di
supervisori, ciascuno con la sua squadretta. È l'organigramma, ed è la
topologia che compra la **scala**, per una ragione aritmetica: in un albero con
fattore di ramificazione $b$ nessun nodo ha più di $b$ sottoposti, per quanti
agenti ci siano in fondo.

`````{tab} Elementare

Catena e gerarchia sembrano due cose diverse, e invece sono la stessa forma con
un parametro diverso. In tutte e due, fra due qualsiasi partecipanti esiste una
strada sola: niente scorciatoie, niente anelli. Quello che cambia è quante
persone riferiscono a ciascuno. Nella catena una, e allora la fila si allunga
senza fine: quindici agenti sono quattordici passaggi. Nella gerarchia due, o
cinque, o dieci, e allora la fila si accorcia di colpo, perché a ogni livello
il gruppo si moltiplica invece di allungarsi. Quindici agenti in un albero
binario stanno in quattro livelli, e per andare da una foglia all'altra bastano
sei passaggi invece di quattordici.

Il prezzo lo conosce chiunque abbia lavorato in un'azienda grande. A ogni
livello qualcuno **riassume**: il capo squadra non riporta al direttore tutto
quello che gli hanno detto i suoi, riporta l'essenziale. Riassumere è
un'operazione che perde, sempre, e perde tre volte se i livelli sono tre. Il
conto grossolano: se a ogni passaggio sopravvive l'ottanta per cento di ciò che
contava, in cima arriva $0{,}8 \times 0{,}8 \times 0{,}8$, poco più della metà.
E chi sta in cima non sa che cosa manca: il riassunto sembra completo, è per
questo che è un riassunto.

C'è un secondo prezzo, meno visibile. Quando il risultato finale è sbagliato,
capire *dove* lo sia diventa difficile: la responsabilità si è diluita lungo la
scala, e ogni livello può dire in buona fede di aver fatto la sua parte.

`````

`````{tab} Superiore

Catena e gerarchia sono entrambe **alberi**, cioè grafi connessi e aciclici con
$N-1$ archi e un solo cammino fra ogni coppia di nodi; a distinguerle è il
fattore di ramificazione $b$. Con $b = 1$ si ha il cammino,
$\mathrm{diam}(G) = N-1$. Con $b > 1$ e un albero completo di profondità $D$ si
ha $N = (b^{D+1}-1)/(b-1)$, quindi

$$
D = \log_b\!\bigl(N(b-1)+1\bigr) - 1,
\qquad
\mathrm{diam}(G) = 2D,
\qquad
\Delta(G) = b+1,
$$

dove $D$ è il numero di livelli sotto la radice. Il diametro passa da lineare a
**logaritmico** in $N$: è tutto il guadagno della gerarchia, e con $b = 2$ e
$N = 15$ vale $D = 3$, $\mathrm{diam}(G) = 6$ contro i $14$ della catena.

La perdita per riassunto si modella, in modo dichiaratamente grossolano, come
un fattore per livello: se ogni passaggio verso l'alto conserva una frazione
$\lambda \in (0,1)$ dell'informazione rilevante, in cima ne arriva
$\lambda^{D}$. Il punto non è il valore preciso (nessuno sa misurare $\lambda$
su un riassunto in linguaggio naturale) ma la forma: la perdita è
**esponenziale nella profondità**, mentre il guadagno sul diametro è
logaritmico. Preso da solo, questo modello spingerebbe $b$ all'assurdo: con
$b = N-1$ e $D = 1$ si è ricostruita la stella, che la pagina precedente ha
appena dichiarato insostenibile. Il termine che manca è il carico del singolo
supervisore, la finestra $c_0 + b\,\bar{m}$ vista per la stella, che cresce
linearmente in $b$: l'ottimo bilancia la perdita esponenziale in $D$ contro
quel carico lineare, e la regola operativa che ne esce è alzare $b$ **finché
la finestra di ciascun supervisore regge**, e abbassare $D$ di conseguenza.
È il contrario di ciò che suggerisce l'istinto organizzativo.

Il secondo costo è formale quanto il primo: la gerarchia rende difficile
l'**assegnazione del merito** (e della colpa). Attribuire un esito sbagliato a
un nodo richiede di ripercorrere all'indietro tutti i riassunti, e i riassunti
sono proprio ciò che ha cancellato l'informazione che servirebbe. È lo stesso
problema di *credit assignment* che nell'apprendimento per rinforzo
multi-agente si affronta con la scomposizione del valore, e ne parla la sezione
«Imparare insieme».

`````

Le prime tre topologie sono grafi di agenti veri, e allora conviene misurarle
invece di ragionarci a intuito. Il codice che segue costruisce quattro grafi da
quindici agenti e ne calcola diametro, grado massimo, carico del nodo più
sollecitato e, come misura di fragilità, la frazione di coppie di agenti che
restano collegate **dopo** aver tolto proprio quel nodo.

```python
import numpy as np

INF = 10**6


def distanze(A):
    """Cammini minimi fra tutte le coppie (Floyd-Warshall), grafo non pesato."""
    D = np.where(A == 1, 1, INF)
    np.fill_diagonal(D, 0)
    for k in range(len(A)):
        D = np.minimum(D, D[:, k, None] + D[None, k, :])
    return D


def grafo(n, archi):
    A = np.zeros((n, n), dtype=int)
    for i, j in archi:
        A[i, j] = A[j, i] = 1
    return A


def metriche(A, agenti):
    """Diametro fra agenti, grado massimo, carico del nodo piu' sollecitato
    (frazione di cammini minimi che lo attraversano) e frazione di coppie
    ancora collegate dopo averlo tolto."""
    D = distanze(A)
    coppie = [(s, t) for i, s in enumerate(agenti) for t in agenti[i + 1:]]
    carico = [sum(1 for s, t in coppie
                  if v not in (s, t) and D[s, v] + D[v, t] == D[s, t]) / len(coppie)
              for v in range(len(A))]
    v = int(np.argmax(carico))                       # il nodo piu' sollecitato
    Dv = distanze(np.delete(np.delete(A, v, 0), v, 1))
    resta = [a - (a > v) for a in agenti if a != v]   # indici dopo la rimozione
    vive = sum(1 for i, s in enumerate(resta) for t in resta[i + 1:]
               if Dv[s, t] < INF)
    n = len(resta)
    return (max(D[s, t] for s, t in coppie), A.sum(axis=1).max(),
            carico[v], vive / (n * (n - 1) / 2))


N = 15  # quindici agenti in tutte le topologie: stessa squadra, forme diverse
stella = grafo(N, [(0, i) for i in range(1, N)])
catena = grafo(N, [(i, i + 1) for i in range(N - 1)])
albero = grafo(N, [(i, 2*i + 1) for i in range(7)] + [(i, 2*i + 2) for i in range(7)])
lavagna = grafo(N + 1, [(N, i) for i in range(N)])   # il nodo N e' la lavagna

print(f"{'topologia':10} {'diametro':>9} {'grado max':>10} {'carico':>7} {'residua':>8}")
for nome, A in [("stella", stella), ("catena", catena),
                ("albero", albero), ("lavagna", lavagna)]:
    d, g, c, r = metriche(A, list(range(N)))
    print(f"{nome:10} {d:9d} {g:10d} {c:6.0%} {r:7.0%}")
```

```text
topologia   diametro  grado max  carico  residua
stella             2         14    87%      0%
catena            14          2    47%     46%
albero             6          3    54%     37%
lavagna            2         15   100%      0%
```

Tre letture. La stella compra il diametro minimo possibile pagandolo
con l'87% del traffico su un nodo solo e una connettività residua **nulla**:
se quel nodo si ferma non resta niente. L'albero non batte la catena né sul
carico né sulla fragilità (anzi, sulla fragilità fa peggio): la batte sul
diametro, $6$ contro $14$. La gerarchia compra distanza, non robustezza, ed è
bene non aspettarsi dell'altro. La terza lettura è la più istruttiva: nell'albero
il nodo più sollecitato **non è la radice**, è un capo intermedio, che porta il
54% del traffico contro il 47% della radice. Chi conosce le organizzazioni non
si stupirà; chi progetta squadre di agenti dovrebbe, perché il ruolo da
irrobustire per primo non è quello che sembra.

## La lavagna condivisa

Le prime tre topologie hanno una cosa in comune: gli agenti si scrivono
direttamente. Le due che restano vi rinunciano, e in cambio ottengono qualcosa
che con i canali diretti non si ottiene.

Nella prima, nessuno scrive a nessuno. C'è una struttura dati comune, la
**lavagna**, su cui tutti leggono e scrivono: un agente pubblica un'ipotesi, un
altro la vede comparire e reagisce, un terzo la corregge. Nessuno sa chi siano
gli altri, e non gli serve saperlo; un componente di controllo decide, a ogni
ciclo, chi lasciar agire.

L'architettura nasce con **Hearsay-II**, sviluppato alla Carnegie Mellon
University (la prima versione del nucleo è dell'autunno del 1973) dentro il
programma quinquennale sulla comprensione del parlato finanziato dalla DARPA,
e raccontato nel 1980 in un lungo articolo di rassegna su *ACM Computing
Surveys* firmato da Lee Erman, Frederick Hayes-Roth, Victor Lesser e Raj Reddy
{cite}`erman1980hearsay`. Il compito era riconoscere parlato continuo su un
vocabolario di 1011 parole per interrogare una raccolta di abstract di
informatica («*Which abstracts refer to theory of computation?*»), con
interpretazioni corrette nel 90% delle frasi di prova.

Il motivo per cui la lavagna nasce **lì** è il capitolo sullo Speech
Recognition di questo libro. Riconoscere il parlato vuol dire far collaborare
conoscenze di natura incompatibile (l'acustica del segnale, le sillabe, il
lessico, la sintassi, la semantica del dominio) nessuna delle quali è
affidabile da sola: l'acustica propone parole plausibili e sbagliate, la
sintassi scarta sequenze impossibili, la semantica sa che in quella raccolta si
parla di computazione e non di cucina. Soprattutto, **non esiste un ordine
giusto** in cui interpellarle: a volte è un frammento acustico chiaro a far
partire tutto, a volte è una parola riconosciuta a metà frase che permette di
prevedere quelle attorno. Una pipeline avrebbe dovuto fissare quell'ordine in
anticipo. La lavagna non lo fissa.

`````{tab} Elementare

Pensa alla lavagna di una sala operativa, quella con sopra tutto quello che si
sa finora, scritto a mano e cancellabile. Chi entra non chiede a nessuno il
permesso di parlare: guarda la lavagna, vede se sa aggiungere qualcosa, scrive,
esce. Il meteorologo scrive che il vento gira; il tecnico legge quella riga e
corregge una stima che aveva scritto lui mezz'ora prima; nessuno dei due sa
dell'esistenza dell'altro.

Il pregio è che si può aggiungere uno specialista nuovo senza avvisare nessuno:
se sa leggere la lavagna e scriverci sopra, è dentro, e nessuno degli altri va
modificato. È il contrario del capo cantiere, che va aggiornato ogni volta che
arriva un operaio.

I difetti sono i due che chiunque abbia visto una lavagna vera conosce. Il
primo è la ressa: in due che scrivono nello stesso punto nello stesso momento
si danno fastidio, e serve una regola su chi va prima. Il secondo è che,
guardandola a fine giornata, **non si sa più chi ha scritto cosa**: c'è un
risultato, e se è sbagliato non si sa da quale mano sia uscito l'errore.

`````

`````{tab} Superiore

L'architettura ha tre pezzi, e conviene chiamarli con i nomi originali. La
**blackboard** è una memoria globale strutturata in *livelli* di astrazione
crescente (in Hearsay-II: parametro, segmento, sillaba, parola, sequenza di
parole, frase), su cui vivono le ipotesi, ciascuna con il proprio intervallo
temporale e un punteggio di credibilità. Le **knowledge source** sono programmi
indipendenti, ciascuno una coppia condizione-azione: la condizione dichiara a
quali cambiamenti della lavagna il modulo reagisce, l'azione dice cosa scrive
in risposta. Il terzo pezzo è lo **scheduler**: a ogni ciclo calcola una
priorità per ogni azione applicabile ed esegue quella di valore atteso più
alto, il che rende il controllo *opportunistico* invece che prestabilito.

Topologicamente è una stella il cui centro **non è un agente**: nella tabella
di poco fa la lavagna ha diametro 2 come il supervisore, ma un carico del 100%
(ogni interazione fra agenti la attraversa) e un grado massimo di 15 che non
appartiene a nessun agente. La differenza conta: un centro che *decide* ha per
limite una finestra di contesto e una capacità di ragionamento, un centro che
*conserva* ha per limite la contesa in scrittura e la coerenza, cioè problemi
da basi di dati (transazioni, versioni, blocchi) con soluzioni note. In cambio
il disaccoppiamento è massimo: le knowledge source non si nominano fra loro, e
aggiungerne una non richiede di modificare le altre.

Gli autori stessi elencano **due** debolezze, e valgono ancora. La prima nasce
dalla generalità: ogni decisione passa per la lavagna, il che si è rivelato
desiderabile *fra* moduli e inadatto ai passaggi intermedi *dentro* un modulo,
dove le knowledge source di Hearsay-II usavano strutture dati private e
specializzate (reti sequenziali per il riconoscitore di parole, una grande
matrice di bit delle adiacenze per quello di sequenze); costringerle nello stile
della lavagna, scrivono, o falliva del tutto o degradava le prestazioni in modo
intollerabile. La seconda riguarda l'efficienza: deliberare a ogni ciclo su
quale azione convenga costa, e si giustifica solo finché per lo stesso compito
non esiste un algoritmo esplicito; appena l'algoritmo c'è, compilarlo ed
eseguirlo direttamente rende di più.

Un terzo costo lo aggiungiamo noi, ed è quello che ci riguarda più da vicino: la
**provenienza**. Con decine di scritture asincrone su una struttura condivisa,
ricostruire quale agente abbia prodotto quale contributo richiede di registrarlo
esplicitamente, altrimenti l'informazione è perduta e con essa ogni possibilità
di attribuzione.

`````

## Il mercato

Le prime quattro topologie danno per scontato che si sappia in anticipo chi sa
fare cosa. Ma se non lo si sa? Se gli agenti disponibili cambiano, se le loro
competenze non sono dichiarate, se chi è libero adesso non lo era un minuto fa?

La risposta è vecchia quanto il commercio: si mette il compito **a bando**. Chi
ha un lavoro da far fare lo annuncia, chi si ritiene capace presenta
un'offerta, il banditore valuta e assegna. È il **contract net protocol**,
formulato da Reid G. Smith in un articolo del dicembre 1980 sulle *IEEE
Transactions on Computers* {cite}`smith1980contract` e dimostrato lì su una rete
di sensori distribuiti, che doveva organizzarsi da sé in base a quali nodi
c'erano e a dove si trovavano.

Ottimizza l'**allocazione dinamica senza pianificatore centrale**. Nessun nodo
è designato in anticipo come capo: «gestore» ed «esecutore» sono *ruoli*, che
un nodo assume e lascia durante il lavoro, e tipicamente ricopre entrambi nello
stesso momento per contratti diversi. Un esecutore può spezzare il proprio
compito e bandirne i pezzi, diventando gestore a sua volta: la gerarchia esiste,
ma è temporanea, e la disegna il lavoro invece del progettista.

`````{tab} Elementare

Funziona come un bando di gara, con quattro messaggi in croce. Il banditore
dice: «serve questo lavoro; possono rispondere solo quelli che hanno il tal
requisito; nella risposta scrivetemi queste cose; c'è tempo fino a giovedì».
Chi si sente in grado risponde con un'offerta. Il banditore le confronta e
assegna a chi gli sembra più adatto.

La cosa elegante è che la scelta la fanno **in due**. Il banditore sceglie fra
chi si è offerto, ma prima ancora ciascuno ha scelto se offrirsi e per quale
bando: un agente sommerso di lavoro semplicemente non risponde, e il sistema si
riequilibra da sé senza che nessuno tenga il conto di chi è libero.

Il costo è altrettanto facile da vedere: **il bando è lavoro che non produce
niente**. Se dieci agenti leggono l'annuncio e cinque preparano un'offerta,
sono cinque ragionamenti spesi per decidere chi ne farà uno. Su compiti piccoli
la gara costa più del lavoro che assegna.

E poi c'è il problema che il diritto degli appalti conosce da sempre, con tanto
di nome: l'**offerta anomala**, quella troppo bella per essere vera, che la
stazione appaltante deve verificare prima di aggiudicare. Un modello di
linguaggio a cui si chiede «sei in grado di fare questo?» tende a rispondere di
sì, e allora la gara premia chi si stima meglio, non chi lavora meglio.

`````

`````{tab} Superiore

Il protocollo ha quattro messaggi principali. L'**annuncio di compito**, con
quattro campi: l'*astrazione del compito* (di che lavoro si tratta), la
*specifica di eleggibilità* (i requisiti minimi per poter offrire, un filtro
che serve a tagliare traffico inutile), la *specifica dell'offerta* (quali
informazioni il gestore vuole ricevere) e il *tempo di scadenza*.
L'**offerta**, che descrive il nodo secondo quanto richiesto.
L'**assegnazione**, con la *specifica del compito*. E i **rapporti**, interinali e
finale. Smith attribuisce alla negoziazione quattro proprietà: è locale (nessun
controllo centralizzato), lo scambio è bidirezionale, ciascuna parte valuta
secondo il proprio criterio, e l'accordo finale è per **selezione reciproca**.

Il costo si conta in messaggi. Collocare un compito fra $N$ candidati richiede
un annuncio (uno solo se il canale è a diffusione, altrimenti $N$), fino a $N$
offerte e un'assegnazione:

$$
M(N) \;=\; N + 2 \quad\text{(in diffusione)},
\qquad
M(N) \;=\; 2N + 1 \quad\text{(punto a punto)},
$$

dove $M(N)$ è il numero di messaggi per collocare **un solo** compito. Con
agenti classici sono pacchetti di rete e non contano nulla; con gli LLM ogni
offerta è una chiamata al modello che legge l'annuncio e produce una
valutazione. Con $N = 10$ candidati e un annuncio da 2000 token, la gara costa
dell'ordine di $10 \times 2000 = 20.000$ token in ingresso prima che qualcuno
abbia cominciato a lavorare. Smith lo dice a modo suo, ed è la regola operativa
da portarsi via: la dimensione dei compiti messi a contratto dev'essere
**grande**, altrimenti l'accelerazione ottenuta distribuendo se la mangia lo
sforzo di distribuire.

Il secondo costo è la **calibrazione delle offerte**. Il protocollo assume che
un nodo sappia stimare la propria idoneità, ipotesi ragionevole quando
l'offerta è «ho tre sensori acustici a queste coordinate» e insostenibile
quando è un modello di linguaggio che dichiara di saper fare una cosa. Il
vincitore sistematico diventa l'agente più ottimista, e l'ottimismo non correla
con la competenza: è una selezione avversa in piena regola. Le contromisure
sono due, e nessuna sta nel protocollo: rendere l'offerta **verificabile** (non
«so farlo» ma un risultato parziale su un campione, che il gestore può
controllare) e pesare le offerte con la storia degli esiti passati. È la stessa
lezione dei fallimenti reali dei sistemi multi-agente, dove la verifica
inadeguata è una delle tre famiglie ricorrenti: un verdetto che nessuno può
provare non aggiunge informazione, aggiunge una firma {cite}`cemri2025why`.

`````

## Coordinarsi senza parlarsi

Vale la pena isolare ciò che lavagna e mercato hanno in comune, perché è il
punto più profondo della sezione. Sono **i due modi di coordinarsi senza un
canale diretto fra agenti**, e risolvono lo stesso problema in modi speculari.

La lavagna coordina **attraverso l'ambiente**: nessuno parla a nessuno, ognuno
lascia una traccia in uno spazio comune e reagisce alle tracce che ci trova. Il
coordinamento non è nei messaggi, è nello stato del mondo. L'idea ha un nome
preciso, **stigmergia**, coniato nel 1959 dal biologo francese Pierre-Paul
Grassé per le termiti che ricostruiscono il nido senza progetto e senza capo,
ciascuna reagendo a quello che le altre hanno già costruito; ed è il meccanismo
su cui si reggono le formiche artificiali e gli sciami di particelle
dell'ultima sezione del capitolo.

Il mercato coordina **attraverso un protocollo**: gli agenti non si conoscono,
ma condividono una grammatica di messaggi (annuncia, offri, assegna) che
permette a due sconosciuti di accordarsi su chi fa cosa. Il coordinamento non è
nello stato del mondo, è nella forma della conversazione, che è il tema della
prossima sezione, dove i messaggi diventano atti tipizzati e la decisione
collettiva diventa una regola di voto.

Detto altrimenti: quando il canale diretto non c'è, o si condivide un **posto**
o si condivide una **lingua**.

## Scegliere

La topologia si sceglie dal compito, mai dall'eleganza. Le tre grandezze di
Baran non dicono quale forma sia migliore, dicono cosa si compra e cosa si
vende; il compito dice quale delle tre cose non è negoziabile.

Un compito le cui parti sono **indipendenti** chiede un **supervisore**:
scomponi, distribuisci, ricomponi, e il costo del centro è un prezzo onesto per
avere un posto solo dove sta la verità. Un compito i cui passi sono
**strettamente sequenziali** e verificabili uno per uno chiede una **catena**,
con un cancello dentro ogni anello. Un compito **troppo grande per un
supervisore solo** chiede una **gerarchia**, tenuta più larga che alta perché
la perdita per riassunto è esponenziale nella profondità mentre il guadagno sul
diametro è solo logaritmico; ma «più larga» ha un limite, ed è quanto ciascun
capo intermedio riesce a leggere, cioè la finestra $c_0 + b\,\bar{m}$ già vista
per la stella. Si allarga la ramificazione finché quella regge, e non oltre:
oltre si è semplicemente rifatta la stella, con i suoi guai.
Un compito in cui contributi **eterogenei** devono incontrarsi
senza un ordine stabilito in anticipo chiede una **lavagna**. Un compito in cui
**non si sa chi sa fare cosa** chiede un **mercato**, purché i pezzi messi a
bando siano grandi abbastanza da ripagare il bando.

Chi non riesce a rispondere a queste domande sul proprio compito ha comunque
un'informazione utile: non lo ha ancora capito abbastanza per costruirci
attorno una squadra. Un agente solo, con un buon prompt e un cancello di
verifica, è nel frattempo un ottimo posto dove aspettare.

```{admonition} Da ricordare
:class: important
- Un sistema multi-agente è un **grafo** (nodi gli agenti, archi i canali) e si
  progetta su tre grandezze, le stesse delle reti di calcolatori: **diametro**,
  **carico** sul nodo più sollecitato, e **cosa resta** se quel nodo si ferma.
  Ogni topologia compra una delle tre vendendo le altre.
- **Supervisore e lavoratori** (stella): ottimizza controllo e tracciabilità,
  c'è un posto solo dove sta la verità sul compito {cite}`wu2024autogen`; paga
  con il collo di bottiglia e il punto singolo di rottura. Il contesto del
  centro cresce come $c_0 + N\bar{m}$, e la cura è comprimere all'ingresso, non
  allargare la finestra.
- **Catena** (*handoff*): semplicità massima e contesto pulitissimo, ma nessun
  ritorno, quindi un errore a monte non è più recuperabile a valle: con sei
  passi al 95% la catena regge $0{,}95^6 \approx 0{,}74$. Serve un cancello dentro
  ogni anello, non alla fine {cite}`cemri2025why`.
- **Gerarchia**: catena e gerarchia sono lo stesso albero con ramificazione
  diversa, e alzare $b$ porta il diametro da lineare a logaritmico ($6$ contro
  $14$ su quindici agenti). Paga con la perdita per riassunto, **esponenziale**
  nella profondità, e con la responsabilità diluita: conviene quindi allargare
  la ramificazione e abbassare i livelli, ma solo finché la finestra
  $c_0 + b\,\bar{m}$ di ogni supervisore regge, perché spinta all'estremo la
  regola ricostruisce la stella. Il nodo più carico è un capo intermedio, non
  la radice.
- **Lavagna condivisa**: nata con **Hearsay-II** alla Carnegie Mellon per la
  comprensione del parlato, dove acustica, lessico, sintassi e semantica devono
  contribuire senza un ordine prestabilito {cite}`erman1980hearsay`. Ottimizza
  il disaccoppiamento (si aggiunge un agente senza toccare gli altri); paga con
  la contesa sulla struttura condivisa e con la perdita della provenienza.
- **Mercato** (il *contract net* di Smith, 1980 {cite}`smith1980contract`):
  annuncio, offerta, assegnazione, rapporto, con ruoli assunti e lasciati.
  Ottimizza l'allocazione senza pianificatore; paga il costo del bando ($N+2$
  messaggi, e con gli LLM $N$ inferenze per assegnarne una) e l'**offerta
  anomala**, cioè l'agente che vince perché si stima meglio, non perché lavora
  meglio.
- Lavagna e mercato sono i due modi di coordinarsi **senza canale diretto**:
  lasciando tracce in un ambiente comune (**stigmergia**, che torna negli
  sciami) oppure condividendo un protocollo di annuncio. O si condivide un
  posto, o si condivide una lingua.
```
