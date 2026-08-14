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
squadra di agenti, e in fondo a quel conto ha lasciato una frase che vale la
pena riprendere: **la forma dello schema decide il ritmo con cui il costo
cresce**. Se ognuno legge tutto quello che dicono gli altri, il conto sale molto
più in fretta che se ognuno legge soltanto il proprio pezzo; e non di una
quantità fissa, ma di un divario che si allarga man mano che la squadra cresce.
Non cambia il modello, non cambia il numero dei partecipanti: cambia chi parla
con chi.

Un avviso sulla parola, perché in questo capitolo ha due sensi e li ha vicini.
Nell'apertura «topologica» qualificava la regola degli storni, quella che conta
i vicini invece di misurarli in metri. Qui «topologia» vuol dire una cosa più
vicina all'uso comune: la forma dello schema, chi è collegato a chi. Le due
accezioni hanno la stessa radice, perché in entrambi i casi si guarda la
struttura dei collegamenti e non le distanze, ma non vanno sovrapposte: gli
storni non c'entrano con le cinque forme che seguono.

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

Nessuna delle due organizzazioni è migliore in assoluto: ognuna compra una di
quelle tre cose vendendo le altre due, ed è questo, non il numero di invitati,
che decide come andrà la serata.

Attenzione però a dove finisce quella regola, perché più avanti ci servirà. Vale
per queste due organizzazioni e per tutte quelle in cui, fra due persone
qualsiasi, esiste **una strada sola**: un solo giro di telefonate che porta da
me a te. Se si accetta di dare a ciascuno una o due telefonate in più, in modo
che le strade fra due persone diventino parecchie, il baratto smette di essere
obbligato e si possono comprare tutte e tre le cose insieme. Alla fine di questa
sezione lo vedremo con i numeri, ed è la ragione per cui Baran ha disegnato tre
reti e non due.

`````

`````{tab} Superiore

Sia $G = (V, E)$ il grafo di comunicazione, con $|V| = N$ agenti, e sia
$d(u,v)$ la lunghezza del cammino minimo fra due nodi. Le tre grandezze sono:

$$
\mathrm{diam}(G) = \max_{u,v \in V} d(u,v),
\qquad
\Delta(G) = \max_{v \in V} \deg(v),
\qquad
\ell(v) = \frac{\bigl|\{(s,t) : v \in \mathrm{sp}(s,t)\}\bigr|}{\binom{N}{2}},
$$

dove $\mathrm{diam}(G)$ è il **diametro** (il più lungo dei cammini minimi,
cioè il caso peggiore di quanti passaggi servono), $\Delta(G)$ è il **grado
massimo** (quanti canali fanno capo al nodo più connesso) e $\ell(v)$ è il
**carico** del nodo, cioè la frazione di coppie $(s,t)$ per cui almeno un
cammino minimo $\mathrm{sp}(s,t)$ attraversa $v$ senza averlo come estremo. È
una versione semplificata della centralità di intermediazione (la *betweenness*)
su due punti: conta le coppie invece di pesarle con la frazione dei loro cammini
minimi che passa davvero per $v$, e le normalizza sul totale $\binom{N}{2}$
delle coppie anziché sulle sole coppie che escludono $v$. La fragilità si legge
da $\kappa(G)$, la **connettività per vertici**: quanti nodi bisogna rimuovere
perché il grafo si spezzi. Negli alberi e nelle
stelle $\kappa(G) = 1$, e ogni nodo interno è un punto di articolazione.

Sugli **alberi** le tre non sono indipendenti, ed è questo a rendere il progetto
interessante. Con $N-1$ archi in tutto non c'è margine: abbassare il diametro
obbliga a concentrare gli archi da qualche parte, il che alza $\Delta$ e $\ell$; e
siccome $\kappa(G) = 1$, il nodo con $\ell$ alto è per costruzione quello la cui
rimozione taglia più coppie. In una rete di calcolatori $\ell(v)$ alto significa un
instradatore congestionato; in una squadra di agenti significa una finestra di
contesto che deve contenere tutto ciò che transita, e questo è un limite più
duro, perché la qualità della lettura degrada molto prima del limite dichiarato
(è il *lost in the middle* del context engineering).

Va detto subito fin dove arriva questo vincolo, perché è facile prenderlo per
una legge e non lo è: è la conseguenza di aver scelto la famiglia di grafi più
povera di archi che esista. Non discende dalle definizioni di
$\mathrm{diam}$, $\Delta$ e $\ell$, discende da $|E| = N-1$. Concedendo qualche arco
in più il baratto si scioglie, e questa sezione ci torna sopra quando avrà la
tabella davanti.

`````

Le cinque forme che si incontrano davvero sono in fila nella
{numref}`fig-multiagente-topologie`.

```{figure} ../figures/multiagente-topologie.svg
:name: fig-multiagente-topologie
:alt: "Cinque piccoli grafi affiancati ed etichettati: supervisore e lavoratori (una stella con il supervisore in cima e tre lavoratori sotto, tutti collegati a lui e non fra loro), catena (quattro nodi in fila con frecce a senso unico), gerarchia (un albero con radice, due nodi intermedi e quattro foglie), lavagna condivisa (quattro nodi collegati da frecce bidirezionali a un rettangolo centrale e a nient'altro), mercato (un banditore collegato a tre offerenti, con una linea tratteggiata verso due di loro e una freccia piena verso il terzo, quello a cui il compito viene assegnato)."
:width: 85%

Le cinque topologie di coordinamento, in ordine di apparizione. Un
**supervisore** che spezza il compito, lo distribuisce e ricompone; una
**catena** in cui ognuno passa il lavoro al successivo ed esce di scena; una
**gerarchia**, cioè supervisori di supervisori; una **lavagna** su cui tutti
scrivono e da cui tutti leggono, senza rivolgersi a nessuno; un **mercato**, in
cui chi ha un lavoro da far fare lo mette a bando e lo assegna a chi si offre.
Nelle prime tre chi parla con chi lo decide il progettista; nelle ultime due no.
```

## Supervisore e lavoratori

Un agente riceve il compito, lo scompone, ne distribuisce i pezzi ai
**lavoratori**, raccoglie i risultati e li ricompone. I lavoratori non si
parlano fra loro. È la rete centralizzata di Baran, quella con un nodo al centro
e tutti gli altri appesi a lui, e siccome disegnata sulla pagina somiglia a una
stella da qui in avanti la chiameremo così. È la forma di gran lunga più
diffusa, anche perché nasce da sé appena un programma comincia a interrogare un
modello più volte di seguito. È anche la squadra dell'apertura del capitolo: uno
tiene le fila, uno scrive il codice, uno controlla se è sicuro eseguirlo, e i due
in fondo non si scrivono mai {cite}`wu2024autogen`.

Quello che compra è il **controllo**, e vale più di quanto sembri: esiste un
posto, uno solo, in cui sta la verità su come sta andando il compito. Chi va a
cercare un errore ha un registro da leggere invece di una conversazione da
ricostruire, e chi vuole fermare il lavoro prima che faccia danni sa dove mettere
il controllo che sbarra la strada.

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

Con $N$ lavoratori (quindi $N+1$ nodi in tutto, supervisore compreso) la stella
ha $\mathrm{diam}(G) = 2$ e $\Delta(G) = N$, tutto
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
rapporto. In inglese si chiama *handoff*, «passaggio di consegne», ed è la forma
più semplice che esista: una fila, senza scorciatoie e senza modo di tornare
indietro. Compra due cose. La **semplicità**, perché non c'è niente da
coordinare, l'ordine della fila *è* il programma. E la pulizia di quello che
ciascuno si trova davanti: nessuno legge conversazioni altrui, ognuno vede solo
il proprio pezzo, e le finestre restano le più corte di tutte le cinque forme.
Quello che paga, invece, è brutale.

`````{tab} Elementare

È il telefono senza fili della sezione precedente, con una differenza sola: qui
i bambini in fila non stanno giocando, stanno lavorando. Il primo raccoglie i
dati e li passa al secondo, il secondo li analizza e passa al terzo, il terzo
scrive il rapporto. E come nel gioco, **nessuno ha visto l'originale** tranne il
primo.

Da lì viene tutto il resto. Se al secondo passaggio qualcuno capisce male una
cosa, quell'errore arriva al quinto travestito da verità di partenza, e al
quinto non c'è nessuno che possa accorgersene: per accorgersene bisognerebbe
avere in mano il foglietto iniziale, e chi sta in fondo alla fila non ce l'ha
mai avuto.

E poi c'è il conto che si moltiplica, lo stesso di prima. Sei passaggi fatti
bene novantacinque volte su cento non danno una fila affidabile al novantacinque
per cento: danno una fila che arriva in fondo intatta poco più di settantatré
volte su cento, cioè fallisce **più di una volta su quattro**, pur essendo fatta
di partecipanti che non sbagliano quasi mai. E la fila si può allungare molto: quindici agenti in
fila sono quattordici passaggi, cioè quattordici occasioni perché la richiesta
di partenza si deformi per strada.

La cosa più sgradevole è che non se ne vede niente. Due agenti che hanno in
testa due versioni diverse del compito continuano a scriversi messaggi cortesi e
ben scritti fino in fondo {cite}`cemri2025why`, e il rapporto finale esce
pulito, ordinato e sbagliato.

`````

`````{tab} Superiore

La catena è un **cammino**, cioè l'albero con fattore di ramificazione $b = 1$:
$N-1$ archi, nessun arco all'indietro, $\mathrm{diam}(G) = N-1$.

Quello che paga lo abbiamo già calcolato: **non c'è ritorno**. Un errore
commesso al secondo passo arriva al quinto travestito da dato di partenza, e a
valle nessuno ha modo di accorgersene, perché a valle nessuno ha visto
l'originale. È la composizione moltiplicativa degli errori nella sua forma
peggiore: con sei passi corretti al 95% ciascuno, la catena intera regge
$0{,}95^6 \approx 0{,}74$, cioè fallisce una volta su quattro pur essendo fatta
di agenti che non sbagliano quasi mai. E siccome il diametro cresce come $N$,
quindici agenti in fila sono quattordici passaggi fra il primo e l'ultimo,
quattordici occasioni perché una specifica si deformi strada facendo. Nella
tassonomia dei fallimenti reali questa famiglia ha un nome, **disallineamento
fra agenti**, ed è la più insidiosa perché non produce sintomi: due agenti che
hanno in testa due versioni diverse del compito si scrivono messaggi cortesi e
ben formati fino in fondo {cite}`cemri2025why`.

`````

La catena si usa quando i passi sono davvero uno dopo l'altro e ciascuno ha un
modo suo di verificare il proprio risultato, così che il controllo stia
**dentro** ogni passaggio e non solo alla fine. Senza controlli lungo la strada,
ogni agente aggiunto alla fila toglie un'altra fetta all'affidabilità di quello
che arriva in fondo.

## La gerarchia

Quando la stella non regge più, la mossa naturale è annidarla: supervisori di
supervisori, ciascuno con la sua squadretta. È l'organigramma, ed è la forma che
permette di **diventare grandi**, per una ragione aritmetica: si decide in
anticipo quanti sottoposti può avere ciascun capo, e quel numero non cambia per
quanti agenti ci siano in fondo. Nel gergo dei grafi l'organigramma si chiama
**albero**, il capo supremo è la **radice**, quelli in fondo che non hanno
sottoposti sono le **foglie**, e quanti sottoposti ha ciascun capo è il **fattore
di ramificazione**: le quattro parole servono da qui alla fine della sezione.

`````{tab} Elementare

Catena e gerarchia sembrano due cose diverse, e invece sono la stessa forma con
un parametro diverso. In tutte e due, fra due qualsiasi partecipanti esiste una
strada sola: niente scorciatoie, niente giri alternativi. Quello che cambia è
quante persone riferiscono a ciascuno. Nella catena una, e allora la fila si
allunga senza fine: quindici agenti sono quattordici passaggi. Nella gerarchia
due, o cinque, o dieci, e allora la fila si accorcia di colpo, perché a ogni
livello il gruppo si moltiplica invece di allungarsi.

Facciamo il conto con due sottoposti a testa. Il capo supremo è uno; sotto di
lui due; sotto quei due, quattro; sotto i quattro, otto: uno più due più quattro
più otto fa quindici, ed è tutto l'organigramma in quattro file. Adesso prendi
uno dei sottoposti dell'ultima fila e mandalo a parlare con un altro dell'altra
metà: deve salire tre gradini fino al capo supremo e riscenderne altri tre.
Sei passaggi invece di quattordici, con le stesse quindici persone.

Il prezzo lo conosce chiunque abbia lavorato in un'azienda grande. A ogni
livello qualcuno **riassume**: il capo squadra non riporta al direttore tutto
quello che gli hanno detto i suoi, riporta l'essenziale. Riassumere è
un'operazione che perde, sempre, e perde tre volte se i livelli sono tre. Il
conto grossolano: se a ogni passaggio verso l'alto sopravvive l'ottanta per
cento di ciò che contava, dopo tre passaggi in cima arriva l'ottanta per cento
dell'ottanta per cento dell'ottanta per cento, cioè poco più della metà,
cinquantuno su cento. E chi sta in cima non sa che cosa manca: il riassunto
sembra completo, è per questo che è un riassunto.

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
\Delta(G) = b+1 \;\; (D \ge 2),
$$

dove $D$ è il numero di livelli sotto la radice. Il $b+1$ è il grado di un nodo
interno, $b$ figli più il genitore, e vale appena esiste un nodo interno che non
sia la radice: a $D = 1$ l'albero è la stella e il massimo torna a essere $b$. Il diametro passa da lineare a
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

Le tre forme viste finora sono schemi di agenti veri, e allora conviene
misurarle invece di ragionarci a intuito. Il codice che segue ne costruisce
cinque, le tre più la lavagna che arriva fra poco e una che serve solo da
termine di paragone, e per ciascuna calcola quattro numeri: le tre domande sulla
festa più una.

Il **diametro** è la prima domanda: per quante mani passa una notizia, nel caso
peggiore. Il **grado massimo** è la seconda nella versione più concreta: quanti
interlocutori diretti ha chi ne ha di più (nella festa erano le diciannove
telefonate dell'organizzatore). Il **carico** è la stessa domanda guardata da un
altro lato, e misura quanto uno è di passaggio: quale frazione degli scambi fra
gli altri deve attraversarlo. La **connettività residua** è la terza domanda:
quante coppie riescono ancora a parlarsi dopo che quella persona si è ammalata,
e se fa zero non parla più nessuno.

Le prime quattro forme hanno quindici agenti a testa: stessa squadra, schemi
diversi. La quinta è quella che Baran disegnava per terza e che finora abbiamo
lasciato da parte: una maglia in cui ogni nodo ha più strade per raggiungere
ogni altro. Ne prendiamo la versione più regolare che esista, l'**ipercubo**, e
ne bastano tre cose. La prima è che cosa ne esce: una maglia in cui ogni agente
ha esattamente quattro vicini e in cui nessuno sta più al centro degli altri;
quattro collegamenti a testa, quindi, contro i due o tre di un capo
dell'organigramma. La seconda è che di agenti ne vuole sedici e non quindici,
perché è fatto così e non si può ritagliare: uno in più su quindici è un
vantaggio troppo piccolo per spiegare i numeri che vedremo. La terza, per chi è
curioso, è la ricetta: si numerano i sedici agenti da 0 a 15, si riscrivono i
numeri usando solo zeri e uni (è il modo in cui contano i calcolatori) e si
collegano quelli che differiscono per una cifra sola.

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


N = 15  # quindici agenti in tutte le topologie ad albero: stessa squadra, forme diverse
stella = grafo(N, [(0, i) for i in range(1, N)])
catena = grafo(N, [(i, i + 1) for i in range(N - 1)])
albero = grafo(N, [(i, 2*i + 1) for i in range(7)] + [(i, 2*i + 2) for i in range(7)])
lavagna = grafo(N + 1, [(N, i) for i in range(N)])   # il nodo N e' la lavagna

# La terza rete di Baran, la maglia: ipercubo a 4 dimensioni. Sedici agenti,
# arco fra due numeri che in binario differiscono per una sola cifra.
ipercubo = grafo(16, [(i, i ^ (1 << b))
                      for i in range(16) for b in range(4) if i < (i ^ (1 << b))])

print(f"{'topologia':10} {'diametro':>9} {'grado max':>10} {'carico':>7} {'residua':>8}")
for nome, A, agenti in [("stella", stella, list(range(N))),
                        ("catena", catena, list(range(N))),
                        ("albero", albero, list(range(N))),
                        ("lavagna", lavagna, list(range(N))),
                        ("ipercubo", ipercubo, list(range(16)))]:
    d, g, c, r = metriche(A, agenti)
    print(f"{nome:10} {d:9d} {g:10d} {c:6.0%} {r:7.0%}")

# La tabella stampa solo il nodo piu' carico. Nell'albero conviene guardarli
# tutti, perche' il primo della classe non e' quello che ci si aspetta.
D = distanze(albero)
coppie = [(s, t) for s in range(N) for t in range(s + 1, N)]
carico = [sum(1 for s, t in coppie
              if v not in (s, t) and D[s, v] + D[v, t] == D[s, t]) / len(coppie)
          for v in range(N)]
print(f"\nalbero: radice {carico[0]:.0%}, capo intermedio {carico[1]:.0%}, "
      f"foglia {carico[7]:.0%}")
```

```text
topologia   diametro  grado max  carico  residua
stella             2         14    87%      0%
catena            14          2    47%     46%
albero             6          3    54%     37%
lavagna            2         15   100%      0%
ipercubo           4          4    21%    100%

albero: radice 47%, capo intermedio 54%, foglia 0%
```

Quattro letture, e l'ultima rimette in discussione le prime tre.

**La prima.** La stella compra il diametro più basso possibile fra le forme che
spendono pochi fili (due passaggi: da un lavoratore al centro, e dal centro
all'altro) e lo paga con l'87% del traffico su un nodo solo e una connettività
residua **nulla**: se quel nodo si ferma non resta niente. L'87 e non il 100
perché quando è il centro stesso a parlare con qualcuno non lo si conta come
traffico *di passaggio*, e quelle conversazioni sono quattordici delle
centocinque coppie possibili. La lavagna, che ha lo stesso diametro, arriva
invece al 100%: lei nel mezzo ci sta sempre, perché non parla mai per conto suo.

**La seconda.** L'organigramma non batte la fila né sul carico (54% contro 47%)
né sulla fragilità (37% contro 46%): la batte sul diametro, sei contro
quattordici, e basta. La gerarchia compra distanza, non robustezza, ed è bene
non aspettarsi dell'altro.

**La terza**, ed è la più istruttiva: nell'organigramma il più sollecitato **non
è il capo supremo**, è un capo intermedio, che porta il 54% del traffico contro
il 47% del capo supremo. È la riga che il programma stampa in fondo, dopo la
tabella. La ragione, a guardarla, è ovvia: il capo supremo sta in mezzo solo fra
le due metà dell'organigramma, mentre un capo intermedio sta in mezzo sia fra i
sei che ha sotto di sé, sia fra ciascuno di quei sei e tutto il resto.
Chi progetta squadre di agenti farebbe bene a saperlo, perché il ruolo da
irrobustire per primo non è quello che sembra.

**E poi c'è l'ultima riga della tabella.** L'ipercubo batte l'organigramma a due
sottoposti su **tutte e tre** le grandezze insieme: diametro quattro contro sei,
carico 21% contro 54%, connettività residua 100% contro 37%, cioè togliendo il
nodo più sollecitato non si scollega nessuno. Lo paga con un collegamento in più
sul nodo più connesso, quattro invece di tre; ma lo paga su **tutti**, perché
nell'ipercubo quattro ce li hanno tutti, mentre nell'albero tre ce li hanno solo
i sei capi intermedi, il capo supremo ne ha due e le otto foglie uno.

Contiamoli, i fili. Se sommo i collegamenti che ciascuno ha, ogni filo lo conto
due volte, una per ognuno dei due che sta alle sue estremità: il totale va quindi
diviso a metà. Nell'albero la somma è due (il capo supremo) più tre per sei
(i capi intermedi) più uno per otto (le foglie), cioè ventotto, e diviso due fa
**quattordici** fili. Nell'ipercubo è quattro per sedici, cioè sessantaquattro,
e diviso due fa **trentadue**. Più del doppio, ed è tutto lì il segreto.

Non è un caso fortunato: in uno schema abbastanza regolare nessuno sta più al
centro degli altri, quindi il traffico si distribuisce da sé, e con più strade
fra ogni coppia togliere qualcuno non spezza niente. Il baratto a tre di poche
pagine fa era **il prezzo di chi vuole spendere pochi fili**, non una legge:
vale perché fra due partecipanti c'è una strada sola, e chi sta su quella strada
la porta tutta.

Se la maglia vince su tutto, perché il resto della sezione parla d'altro? Perché
i nodi di Baran erano macchine che smistano, mentre i nostri sono agenti che
leggono. Un ipercubo di sedici agenti chiede a ciascuno di tenere aperti quattro
interlocutori e di rileggere ciò che gli arriva da tutti e quattro, il che
riporta dritti al conto della sezione precedente, quello che esplode quando
ognuno legge gli altri. Baran poteva tirare fili perché un cavo in più costa un
cavo in più; qui un collegamento in più costa, a ogni giro, altro testo da
rileggere. **Ciò che esclude la terza figura di Baran non è la forma, è quanto
ciascuno riesce a leggere**: la sua rete si comprava con il rame, la nostra si
comprerebbe con il testo da rileggere, che si paga a peso. Ed è la ragione per
cui le cinque forme di questa sezione sono, tutte e cinque, varianti dei primi
due disegni di Baran e mai del terzo.

## La lavagna condivisa

Le prime tre topologie hanno una cosa in comune: chi parla con chi lo decide il
progettista, e lo decide prima che il sistema parta. Le due che restano vi
rinunciano, e in cambio ottengono qualcosa che con un grafo fissato in anticipo
non si ottiene.

Nella prima, nessuno scrive a nessuno. C'è un foglio comune, la **lavagna**, su
cui tutti leggono e scrivono: un agente ci mette sopra un'ipotesi, un altro la
vede comparire e reagisce, un terzo la corregge. Nessuno sa chi siano gli altri,
e non gli serve saperlo; un pezzo di programma a parte decide, giro dopo giro,
chi lasciar agire.

L'architettura nasce dentro un riconoscitore di parlato, e il motivo per cui
nasce **lì** è il ritratto in miniatura del problema che risolve (il
riconoscimento del parlato avrà poi un capitolo suo, più avanti nel libro).
Riconoscere il parlato vuol dire far collaborare conoscenze di natura
incompatibile, e nessuna affidabile da sola: com'è fatto il suono, come si
dividono le sillabe, quali parole esistono, come si mettono in fila, che cosa ha
senso dire in quel contesto. Il suono propone parole plausibili e sbagliate, la
grammatica scarta le sequenze impossibili, il significato sa che in quel dominio
si parla di computazione e non di cucina. Soprattutto, **non esiste un ordine
giusto** in cui interpellarle: a volte è un pezzo di suono chiaro a far partire
tutto, a volte è una parola riconosciuta a metà frase che permette di indovinare
quelle attorno. Una catena di montaggio avrebbe dovuto fissare quell'ordine in
anticipo. La lavagna non lo fissa.

Il sistema si chiama **Hearsay-II**, è stato costruito alla Carnegie Mellon
University dentro un programma quinquennale sulla comprensione del parlato
finanziato dalla DARPA, e lo racconta un lungo articolo di rassegna del 1980
su *ACM Computing Surveys* firmato da Lee Erman, Frederick Hayes-Roth, Victor
Lesser e Raj Reddy {cite}`erman1980hearsay`. Doveva capire frasi dette a voce, di
seguito e senza pause, per interrogare una raccolta di riassunti di articoli di
informatica («*Which abstracts refer to theory of computation?*»), con un
vocabolario di 1011 parole. Interpretava correttamente il 90% delle frasi di
prova, e vale la pena sapere quante erano: ventitré, mai sentite prima dal
sistema. Il novanta per cento di ventitré vuol dire ventuno frasi capite e due
sbagliate: una percentuale su così pochi casi va letta sapendo questo.

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
crescente (in Hearsay-II sono sette: parametro, segmento, sillaba, parola,
sequenza di parole, frase, e in cima l'interrogazione alla base dati), su cui
vivono le ipotesi, ciascuna con il proprio intervallo
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

Quello che compra è la possibilità di **assegnare il lavoro strada facendo**,
senza che nessuno tenga l'elenco di chi c'è e di chi sa fare cosa. Nessuno è
designato capo in anticipo: «chi bandisce» e «chi esegue» sono ruoli che si
prendono e si lasciano durante il lavoro, e di solito uno li ricopre entrambi
nello stesso momento, per contratti diversi. Chi ha vinto un lavoro può
spezzarlo e bandirne i pezzi, diventando banditore a sua volta: una gerarchia
c'è, ma è temporanea, e la disegna il lavoro invece del progettista.

`````{tab} Elementare

Funziona come un bando di gara, e i messaggi che servono sono quattro. Il
banditore dice: «serve questo lavoro; possono rispondere solo quelli che hanno
il tal requisito; nella risposta scrivetemi queste cose; c'è tempo fino a
giovedì». Chi si sente in grado risponde con un'offerta. Il banditore le
confronta e assegna a chi gli sembra più adatto. E il quarto, quello che ci si
dimentica sempre, è che chi ha preso il lavoro alla fine deve **riferire**: se
non lo fa, il banditore non ha modo di sapere se il compito è stato svolto.

La cosa elegante è che la scelta la fanno **in due**. Il banditore sceglie fra
chi si è offerto, ma prima ancora ciascuno ha scelto se offrirsi e per quale
bando: un agente sommerso di lavoro semplicemente non risponde, e il sistema si
riequilibra da sé senza che nessuno tenga il conto di chi è libero.

Il costo è altrettanto facile da vedere: **il bando è lavoro che non produce
niente**. Se dieci agenti leggono l'annuncio e cinque preparano un'offerta,
sono cinque ragionamenti spesi per decidere chi ne farà uno. Su compiti piccoli
la gara costa più del lavoro che assegna.

E poi c'è il problema che chiunque abbia mai fatto fare un lavoro conosce:
quello che ti promette di finire in due giorni a metà prezzo, e poi non lo fa.
Nelle gare vere ha perfino un nome, l'**offerta anomala**, ed è quella che chi
bandisce deve controllare prima di assegnare. Un modello di linguaggio a cui si
chiede «sei in grado di fare questo?» tende a rispondere di sì, e allora la gara
premia chi si stima meglio, non chi lavora meglio.

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

## Coordinarsi senza sapere con chi

Vale la pena isolare ciò che lavagna e mercato hanno in comune, perché è il
punto più profondo della sezione. Sono **i due modi di coordinarsi senza un
grafo deciso in anticipo**, e risolvono lo stesso problema in modi speculari. Il
malinteso da togliere di mezzo subito è che rinuncino ai messaggi diretti: il
contract net è fatto *soltanto* di messaggi diretti, quattro tipi in croce, e la
{numref}`fig-multiagente-topologie` disegna le frecce che vanno dal banditore a
ciascun offerente. A sparire non è il canale, è il progettista che stabilisce in
anticipo chi scriverà a chi.

La lavagna ci rinuncia togliendo il **destinatario**: nessuno scrive a nessuno,
ognuno lascia una traccia in uno spazio comune e reagisce alle tracce che ci
trova. Il coordinamento non è nei messaggi, è nello stato del mondo. L'idea ha
un nome preciso, **stigmergia**, coniato nel 1959 dal biologo francese
Pierre-Paul Grassé per le termiti che ricostruiscono il nido senza progetto e
senza capo, ciascuna reagendo a quello che le altre hanno già costruito; ed è il
meccanismo su cui si reggono le formiche artificiali e gli sciami di particelle
dell'ultima sezione del capitolo.

Il mercato ci rinuncia in modo opposto: il destinatario ce l'ha eccome, ma non
lo sceglie il progettista, lo trova il **bando**. Gli agenti non si conoscono e
condividono una grammatica di messaggi (annuncia, offri, assegna) che permette a
due sconosciuti di accordarsi su chi fa cosa; chi risponderà al prossimo annuncio
non lo sa nessuno finché non risponde, e il grafo di quella conversazione esiste
solo dopo che è avvenuta. Il coordinamento non è nello stato del mondo, è nella
forma della conversazione, che è il tema della prossima sezione: là ogni
messaggio si porterà scritto in cima che cosa fa, e la decisione collettiva
diventerà una regola di voto.

Detto altrimenti: quando non si sa in anticipo con chi si parlerà, o si condivide
un **posto** o si condivide una **lingua**.

## Scegliere

La topologia si sceglie dal compito, mai dall'eleganza. Le tre domande sulla
festa non dicono quale forma sia migliore, dicono cosa si compra e cosa si
vende; il compito dice quale delle tre cose non è negoziabile. E vale
l'avvertenza dell'ipercubo: quel baratto obbligato è il prezzo di chi vuole
spendere pochi collegamenti, non una legge di natura, e qui lo si accetta perché
ogni collegamento in più è altro testo da rileggere per qualcuno.

Un compito le cui parti sono **indipendenti** chiede un **supervisore**:
scomponi, distribuisci, ricomponi, e il costo del centro è un prezzo onesto per
avere un posto solo dove sta la verità. Un compito i cui passi vengono
**davvero uno dopo l'altro**, e ciascuno si può verificare per conto suo, chiede
una **catena**, con un controllo dentro ogni passaggio.

Un compito **troppo grande per un supervisore solo** chiede una **gerarchia**,
tenuta più larga che alta. La ragione è uno squilibrio fra due conti: ogni
livello in più fa perdere un'altra fetta di ciò che conta, e le fette si
moltiplicano, mentre l'accorciamento delle distanze che quel livello compra è
molto più modesto. Ma «più larga» ha un limite, ed è quanto ciascun capo
intermedio riesce a leggere: si allarga finché la sua giornata regge, e non
oltre, perché oltre si è semplicemente rifatta la stella con i suoi guai.

Un compito in cui contributi **di natura diversa** devono incontrarsi senza un
ordine stabilito in anticipo chiede una **lavagna**. Un compito in cui **non si
sa chi sa fare cosa** chiede un **mercato**, purché i pezzi messi a bando siano
grandi abbastanza da ripagare il bando.

Chi non riesce a rispondere a queste domande sul proprio compito ha comunque
un'informazione utile: non lo ha ancora capito abbastanza per costruirci
attorno una squadra. Un agente solo, con un buon foglio di istruzioni e un
controllo che sbarra la strada, è nel frattempo un ottimo posto dove aspettare.

`````{tab} Elementare

```{admonition} Da ricordare
:class: important
- Un sistema multi-agente è uno **schema di chi parla con chi**, e si giudica su
  tre domande: per quante mani passa una notizia, quanto lavora il più carico, e
  che cosa resta in piedi se proprio quello si ferma.
- Nelle forme in cui fra due partecipanti c'è **una strada sola** (la fila, il
  capo cantiere, l'organigramma) quelle tre cose si comprano una vendendo le
  altre. Non è una legge, ed è il conto dell'ipercubo a dimostrarlo: dando a
  ciascuno qualche collegamento in più, in modo che le strade fra due qualsiasi
  siano parecchie, si possono avere tutte e tre le cose insieme. Qui non si fa
  perché ogni collegamento in più è, a ogni giro, altra roba da leggere per
  qualcuno, ed è quello il conto che esplode.
- **Capo cantiere** (un supervisore, tanti lavoratori che non si parlano fra
  loro): c'è una persona sola a cui chiedere che cosa sta succedendo
  {cite}`wu2024autogen`, e si paga con la sua giornata, che ha ventiquattro ore
  come quella degli altri, e con il fatto che se si ferma lui si ferma tutto. La
  cura non è dargli più tempo per leggere, è obbligare gli altri a scrivergli
  meno.
- **Fila** (*handoff*): semplicissima e con la testa sgombra per tutti, ma non si
  torna indietro, e chi sta in fondo non ha mai visto il foglio di partenza. Sei
  passaggi fatti bene novantacinque volte su cento arrivano in fondo intatti
  poco più di settantatré volte su cento. Serve un controllo dentro ogni passaggio, non
  solo alla fine {cite}`cemri2025why`.
- **Organigramma**: è la stessa fila, ma con più persone che riferiscono a
  ciascuno, e allora si accorcia di colpo (quindici agenti diventano sei passaggi
  invece di quattordici). Si paga con i riassunti, che perdono qualcosa a ogni
  livello e non dicono che cosa hanno perso, e con la responsabilità che si
  diluisce. Conviene quindi tenerla **larga e bassa**, ma solo finché ogni capo
  riesce a leggere quel che gli arriva. E il più carico non è il capo supremo: è
  un capo intermedio.
- **Lavagna della sala operativa**: nessuno scrive a nessuno, si scrive sulla
  lavagna e si reagisce a quello che ci si trova. Nata con **Hearsay-II** per
  capire il parlato, dove suoni, parole, grammatica e significato devono
  contribuire senza un ordine stabilito prima {cite}`erman1980hearsay`. Ottimo
  per aggiungere uno specialista senza avvisare nessuno; si paga con la ressa in
  scrittura e col fatto che a fine giornata non si sa più chi ha scritto cosa.
- **Bando di gara** (il *contract net* di Smith, 1980 {cite}`smith1980contract`):
  chi ha un lavoro lo annuncia, chi si sente in grado si offre, il banditore
  assegna. Nessuno è capo per decreto, e il sistema si riequilibra da sé perché
  chi è sommerso non risponde. Si paga con il bando stesso, che è lavoro che non
  produce niente, e con l'**offerta anomala**: un modello a cui si chiede «sei in
  grado?» tende a dire di sì, e la gara premia chi si stima meglio, non chi
  lavora meglio.
- Lavagna e mercato sono i due modi di coordinarsi **senza sapere in anticipo con
  chi si parlerà**: o si condivide un **posto** (la lavagna, ed è la
  *stigmergia*, che torna con le formiche) o si condivide una **lingua** (la
  grammatica del bando).
```

`````

`````{tab} Superiore

```{admonition} Da ricordare
:class: important
- Un sistema multi-agente è un **grafo** (nodi gli agenti, archi i canali) e si
  progetta su tre grandezze, le stesse delle reti di calcolatori: **diametro**,
  **carico** sul nodo più sollecitato, e **cosa resta** se quel nodo si ferma.
  Negli **alberi** ($|E| = N-1$, $\kappa(G) = 1$) si compra una delle tre
  vendendo le altre; il vincolo è del numero di archi, non delle tre grandezze,
  e un grafo regolare ben connesso le prende tutte insieme (l'ipercubo $Q_4$ fa
  diametro $4$, carico $21\%$ e connettività residua $100\%$ contro $6$, $54\%$
  e $37\%$ dell'albero binario, con più del doppio degli archi). Qui non si usa
  perché ogni arco in più è una finestra di contesto in più, cioè il conto quadratico della
  sezione precedente.
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
- Lavagna e mercato sono i due modi di coordinarsi **senza un grafo deciso in
  anticipo** (non senza messaggi: il contract net è fatto solo di messaggi
  diretti): la lavagna rinuncia al **destinatario** e lascia tracce in un
  ambiente comune (**stigmergia**, che torna negli sciami), il mercato al grafo,
  e il destinatario lo trova il bando. O si condivide un posto, o si condivide
  una lingua.
```

`````
