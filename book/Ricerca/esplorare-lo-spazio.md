# Cercare senza avversari: a tentoni e con l’euristica

Chiedi al telefono la strada per una città a trecento chilometri e la risposta
arriva in un istante. Eppure fra qui e là ci sono milioni di incroci, e le
strade che li collegano sono ancora di più: se il telefono le provasse tutte,
non finirebbe entro sera.

Non le prova tutte. Guarda quasi solo nella direzione giusta, e una parte del
lavoro l’ha fatta prima che tu chiedessi, calcolando una volta per tutte i
costi di certi percorsi. La prima metà del trucco è la ricerca vera e propria,
e si capisce meglio dove nessuno rema contro: l’avversario arriva con i
{doc}`giochi </Ricerca/giocare-contro-qualcuno>`, e cambia le regole.

## Cercare a tentoni: due modi, e i loro difetti

Prima di guardare nella direzione giusta bisogna sapere che cosa costa non
guardarci. I due modi di esplorare un albero senza sapere niente si chiamano
**ricerca in ampiezza** (*breadth-first search*) e **ricerca in profondità**
(*depth-first search*), e la differenza sta tutta nell’ordine in cui si aprono
i nodi. Aprire un nodo, o *espanderlo*, vuol dire guardare quali mosse ci sono
nel suo stato e generare i nodi che ne escono.

`````{tab} Elementare

Sei in un labirinto e cerchi l’uscita. Hai due strategie, e sono opposte.

In ampiezza: allaghi. È come se l’acqua entrasse da dove sei e avanzasse di
un metro alla volta in tutti i corridoi insieme: fai un passo in ogni
corridoio, poi torni indietro e fai il secondo passo in ogni corridoio, poi il
terzo. Il vantaggio è enorme: quando l’acqua tocca l’uscita, sei sicuro che
quella è la strada più corta, perché niente ha potuto arrivarci prima. Lo
svantaggio pure: per allagare devi ricordarti tutti i punti bagnati, e sono
tantissimi.

In profondità: scegli un corridoio e lo segui fino in fondo; se finisce nel
muro torni all’ultimo bivio e prendi l’altro. È come tenere un filo srotolato
dietro di sé: non devi ricordarti tutti i punti dove sei stato, solo il filo
che hai alle spalle, e un filo lungo quanto sei sceso in profondità costa
pochissimo rispetto ad allagare. In cambio la strada che trovi può essere
ridicola, e non c’è nessun limite a quanto: ti fermi alla prima uscita che
incontri, non alla più vicina, e se il primo corridoio gira per mezzo edificio
prima di sbucare, quella è la tua strada. E se un corridoio prosegue e prosegue
senza mai finire né chiudersi, tu lo segui e basta: non hai nessun motivo per
tornare indietro, non ci torni mai, e il filo che ti stavi srotolando dietro
cresce con te.

Lo stesso ti capita se quel corridoio gira in tondo e ti riporta a un bivio
dove eri già stato: il filo non te lo dice, e tu ci giri dentro per sempre.
Contro i giri in tondo basta un gesso: segni ogni bivio dove metti piede, e
appena ritrovi un segno torni indietro. In un labirinto che da qualche parte
finisce, l’uscita, se c’è, adesso la trovi; solo che i segni te li devi
ricordare tutti, e la memoria che il filo ti faceva risparmiare hai
ricominciato a pagarla.

Le due strategie si possono anche sposare. Giri col filo, ma con un tetto: un
bivio soltanto, e se l’uscita non salta fuori torni all’inizio e rifai tutto
con due, poi con tre. Rifare ogni volta i primi corridoi sembra uno spreco, e
non lo è, perché i bivi vicini all’inizio sono pochissimi rispetto a quelli
lontani. Se da ogni bivio ne partono dieci, ogni giro costa dieci volte quello
prima: se l’ultimo costa mille passi, quelli prima ne sono costati cento,
dieci e uno, centoundici in tutto, cioè circa un nono di mille. Rifarli costa
quindi circa l’undici per cento di lavoro in più. In cambio ti tieni la memoria
del filo e la garanzia dell’acqua: l’uscita che trovi è la più vicina, perché
il giro prima, con un bivio in meno, era andato a vuoto.

Nessuna di queste sa niente di dove sia l’uscita. E il difetto vero è quello,
non la memoria: cercano dappertutto con lo stesso impegno, anche nella
direzione opposta a dove bisogna andare.

`````

`````{tab} Superiore

Le due strategie differiscono per la disciplina della **frontiera**, cioè
dell’insieme dei nodi generati e non ancora espansi. In ampiezza è una coda
(primo entrato, primo uscito), in profondità una pila (ultimo entrato, primo
uscito), e da quella riga sola discendono tutte le proprietà.

| | ampiezza | profondità |
|---|---|---|
| trova sempre una soluzione? | sì (se esiste, e $b$ è finito) | no (rami infiniti, cicli) |
| è la più corta? | sì, a costi uniformi | no |
| tempo | $O(b^d)$ | $O(b^m)$ |
| memoria | $O(b^d)$ | $O(bm)$ |

dove $b$ è il fattore di ramificazione, $d$ la profondità della soluzione più
vicina e $m$ la profondità massima dell’albero. La tabella vale per la ricerca
**ad albero**, cioè senza tenere memoria degli stati già visti: tenendola (che
è quello che fa il programma del rompicapo) la ricerca in profondità diventa
completa su spazi finiti, perché i cicli si riconoscono, ma si paga la memoria
che si era risparmiata. La riga che decide, fra tempo e memoria, è quella della
memoria: la ricerca in ampiezza tiene in memoria un intero livello, e un
livello cresce come $b^d$. È il vincolo che morde per primo, molto prima del
tempo.

Le due si sposano nell’**approfondimento iterativo**: si fa una ricerca in
profondità con un tetto di un passo, poi di due, poi di tre, fino a trovare la
soluzione. Sembra uno spreco, perché i livelli alti si rigenerano ogni volta,
e il conto dice che non lo è. I nodi a profondità $j$ si generano una volta
per ogni tetto da $j$ a $d$, cioè $d - j + 1$ volte, quindi in tutto

$$
N_{\text{iter}} = \sum_{j=1}^{d} (d - j + 1)\, b^{j}
\qquad \text{contro} \qquad
N = \sum_{j=1}^{d} b^{j}
$$

di una visita sola fino a $d$. Il rapporto $N_{\text{iter}}/N$ cresce con $d$
verso $b/(b-1)$ e non lo raggiunge mai: con $b = 10$ vuol dire al più l’undici
per cento di lavoro in più. In cambio si tengono la memoria $O(bd)$ della
ricerca in profondità e le garanzie di quella in ampiezza, cioè la
completezza con $b$ finito e l’ottimalità quando i passi costano tutti uguale.
Korf ha mostrato che fra le ricerche ad albero di crescita esponenziale è
asintoticamente ottimo in tempo, in memoria e nel costo del cammino trovato
{cite}`korf1985depth`, e un programma di scacchi, CHESS 4.5, lo usava già nel
1977 per amministrare il tempo dell’orologio {cite}`russell2020artificial`.
Quando lo spazio degli stati non entra in memoria e non si sa quanto sia
lontana la soluzione, è la scelta di riferimento.

`````

Il costo della ricerca a tentoni si misura su un problema abbastanza piccolo da
poterlo percorrere tutto.

## Un rompicapo con cui contare

Il rompicapo delle otto tessere è una cornice di tre caselle per lato con otto
tessere numerate e un buco. Una mossa fa scivolare nel buco una delle tessere
che gli stanno accanto. I modi di disporre nove cose in nove caselle sono
$9! = 362\,880$ (il punto esclamativo si legge «fattoriale» e vuol dire
$9 \times 8 \times 7 \times \ldots \times 1$: nove scelte per la prima casella,
otto per la seconda, e così via). Facendo scorrere le tessere, però, se ne
raggiunge soltanto la metà, 181.440, e chi toglie dalla cornice due tessere e
le rimette scambiate ha in mano un rompicapo che non si risolve più.

Il motivo è un invariante, una proprietà che nessuna mossa cambia. Una
disposizione si rimette in ordine con una serie di scambi fra due cose alla
volta, tessere o buco (le *trasposizioni*), e quanti scambi servono dipende da
come si procede, ma il loro numero resta sempre pari oppure sempre dispari: è
la *parità* della disposizione. Se la cornice si colora come una scacchiera,
ogni mossa fa due cose insieme: scambia il buco con una tessera, e gira la
parità; porta il buco su una casella dell’altro colore, e gira il colore.
Nella disposizione ordinata gli scambi che servono sono zero, un numero pari,
e il buco sta in un angolo, su una casella che chiamiamo bianca; da lì in poi
parità pari e buco sul bianco vanno sempre insieme, come parità dispari e buco
sul nero. Due tessere scambiate a mano girano la parità e lasciano il buco
dov’è: l’accordo è rotto, e nessuna mossa lo rimette. Le disposizioni con
l’accordo rotto sono esattamente la metà, perché a buco fermo quelle pari sono
tante quante le dispari. Che tutte le altre si raggiungano, invece,
l’invariante non lo dice: lo dice una visita completa a partire dalla
disposizione ordinata.

Il programma del rompicapo fa prima questa visita, in ampiezza, e conta le
posizioni che incontra. Poi cerca la soluzione aprendo gli stati uno alla
volta, e sceglie ogni volta quello che gli sembra più promettente, cioè quello
per cui è più piccola la somma fra i passi già fatti e una stima di quelli che
restano. La stima, per adesso, è messa a zero: il programma non sa niente di
dove sia la meta, ed è costretto a guardarsi intorno in tutte le direzioni
allo stesso modo.

```python
import heapq
from collections import deque

META = (1, 2, 3, 4, 5, 6, 7, 8, 0)
PARTENZA = (7, 2, 4, 5, 0, 6, 8, 3, 1)   # 0 e' la casella vuota


def mosse(s):
    """Gli stati raggiungibili spostando una tessera nella casella vuota."""
    v = s.index(0)
    r, c = divmod(v, 3)
    for dr, dc in ((-1, 0), (1, 0), (0, -1), (0, 1)):
        nr, nc = r + dr, c + dc
        if 0 <= nr < 3 and 0 <= nc < 3:
            n = nr * 3 + nc
            t = list(s)
            t[v], t[n] = t[n], t[v]
            yield tuple(t)


visti, da_visitare = {META}, deque([META])      # la visita in ampiezza
while da_visitare:
    for t in mosse(da_visitare.popleft()):
        if t not in visti:
            visti.add(t)
            da_visitare.append(t)
print(f"posizioni raggiungibili dalla disposizione ordinata: {len(visti)}")


def cerca(stima):
    """Apre sempre lo stato con (passi fatti + stima di quelli che restano)
    piu' piccolo. Restituisce la lunghezza della soluzione e quanti stati
    ha dovuto guardare per trovarla."""
    # a somma pari esce per primo chi ha fatto meno passi: lo dice la tupla,
    # ed e' una scelta: cambiarla cambia quanti stati si aprono
    coda = [(stima(PARTENZA), 0, PARTENZA)]
    costo = {PARTENZA: 0}
    guardati = 0
    while coda:
        _, fatti, s = heapq.heappop(coda)
        if s == META:
            return fatti, guardati
        if fatti > costo[s]:          # gia' raggiunto per una strada migliore
            continue
        guardati += 1
        for t in mosse(s):
            if fatti + 1 < costo.get(t, 10**9):
                costo[t] = fatti + 1
                heapq.heappush(coda, (fatti + 1 + stima(t), fatti + 1, t))
    raise AssertionError("nessuna soluzione")


passi, senza_stima = cerca(lambda s: 0)
print(f"senza nessuna stima:  {passi} mosse, {senza_stima} stati guardati")
```

```text
posizioni raggiungibili dalla disposizione ordinata: 181440
senza nessuna stima:  20 mosse, 48389 stati guardati
```

Le posizioni raggiungibili sono proprio la metà di $9!$. Venti mosse di
soluzione, e per trovarle ne sono state esaminate quarantottomila e passa: più
di un quarto delle posizioni che questo rompicapo può raggiungere. Con la stima
a zero l’algoritmo apre sempre lo stato più vicino alla partenza, e quindi si
allarga in tutte le direzioni allo stesso modo, esattamente come la ricerca in
ampiezza. Non sbaglia mai la risposta, e paga carissimo il non sapere dove sta
andando.

## La stima di quanto manca

Quello che cambia di più le proporzioni è dare alla ricerca un modo di
indovinare, guardando uno stato, quanto lavoro resta da lì alla fine: un
fiuto, che non garantisce niente ma dice da che parte guardare per primo.

Quel fiuto si chiama **euristica**, parola greca per «che aiuta a trovare», ed
è il nome che in informatica si dà a una regola pratica, non garantita, che
indirizza la ricerca. La proprietà che le serve ha anch’essa un nome:
un’euristica si dice **ammissibile** quando non esagera mai, cioè quando il
lavoro che stima non supera mai quello che serve davvero. E il programma del
rompicapo, che apre sempre lo stato con la somma più piccola fra i passi già
fatti e la stima di quelli che restano, esegue l’algoritmo **A\*** (si legge «a
stella»), che ha una data e tre autori: 1968, Peter Hart, Nils Nilsson e
Bertram Raphael {cite}`hart1968formal`.

`````{tab} Elementare

In una città che non conosci c’è la casa di un amico, e sai solo che sta vicino
a una torre che dal tuo incrocio si vede. Quali strade ci portino non lo sai,
né se sono a senso unico. Sai la direzione, e i metri in linea d’aria.

Puntare alla torre a ogni incrocio non funziona. Quando va bene arrivi per un
giro più lungo del necessario; quando va male finisci davanti a un muro, con la
torre dall’altra parte, e le altre strade le hai lasciate al primo bivio.

Allora tiri fuori un foglio e tieni aperte più strade insieme. Accanto a ogni
punto raggiunto scrivi due numeri, tutti e due in metri: quelli che hai già
camminato per arrivarci, e quelli che restano in linea d’aria. Poi allunghi di
un isolato la strada con la somma più piccola, e solo quella.

La somma, non uno dei due numeri. Coi soli metri camminati ti allargheresti in
tondo come l’acqua del labirinto; coi soli metri che restano ricadresti nel
muro di prima. Con la somma i quartieri dalla parte opposta restano bianchi sul
foglio, e la strada corta che partiva male non ti sfugge. Se hai fretta puoi
contare due volte i metri che restano: allunghi meno strade, arrivi prima, e la
strada che trovi non è mai più lunga del doppio della più corta.

Una riga del foglio tocca la casa dell’amico, e tu non ti fermi. Aspetti che
venga il suo turno, cioè che sia la sua somma la più piccola. Chi si ferma alla
prima strada che arriva porta a casa quella, e più giù nel foglio ce n’era una
più corta.

Aspettando il turno arrivi per la strada più corta perché la stima, quei metri
in linea d’aria, non è mai più dei metri veri. Le strade girano, la linea d’aria
no, e non capita che dica «due chilometri» dove la strada ne fa uno e mezzo.
Adesso uno al bar ti dice dieci chilometri dove ce ne sono due. La somma di
quella strada diventa pessima, non la allunghi più, e all’amico ci arrivi da
un’altra parte. Nemmeno te ne accorgi: sei arrivato, la strada c’era, ed era più
lunga del necessario. Sbagliare per difetto costa tempo, sbagliare per eccesso
costa la strada giusta.

Certe volte capiti su un incrocio già scritto, con meno metri camminati della
volta prima. Cancelli il numero vecchio, e quell’incrocio torna in gioco. Su un
incrocio da cui sei già ripartito, invece, non capita, e il motivo sta ancora
nella linea d’aria: camminando cento metri si accorcia al massimo di cento,
quindi lungo una strada la somma non scende mai. Se a un incrocio portasse una
via più corta di quella con cui ci sei arrivato, un pezzo di quella via sarebbe
ancora sul foglio, in attesa; e siccome lungo una via la somma non scende, quel
pezzo avrebbe una somma più piccola, e il turno sarebbe toccato prima a lui.
Quando tocca a un incrocio, quindi, ci sei già arrivato per la strada più corta.

La regola dei cento metri, però, può cadere anche con una stima che non esagera
mai. Se la stima salta, per esempio la linea d’aria a un incrocio e zero a
quello accanto, un punto da cui eri già ripartito può tornare in gioco davvero,
e chi lo dà per chiuso porta a casa una strada più lunga.

Il conto lo paga il foglio, dove ogni incrocio aperto resta scritto per sempre.
Per una città basta, per le strade di un paese intero servirebbe un magazzino.
Allora lasci il foglio e riprendi il filo del labirinto, con un tetto sulla
somma, mettiamo tre chilometri: segui una strada, e appena la sua somma supera
il tetto torni indietro, segnandoti a quanto era arrivata. Se la casa non salta
fuori, guardi gli sforamenti segnati, per esempio tre chilometri e due, tre e
mezzo e quattro: il tetto nuovo è il più piccolo, tre e due, e si riparte da
capo. Un tetto di quattro farebbe passare per buona la prima casa che capita
sotto i quattro, che non è detto sia la più vicina; alzato di quel minimo,
invece, la strada che trovi resta la più corta, per la stessa ragione del
labirinto: il giro prima, col tetto più basso, era andato a vuoto. Questa
versione col filo ha un nome che è una sigla, IDA\*.

`````

`````{tab} Superiore

Si introduce una funzione $h(n) \ge 0$, l’euristica, che stima il costo del
cammino ottimo da $n$ alla meta, e si combina con il costo già pagato $g(n)$:

$$
f(n) = g(n) + h(n),
$$

dove $g(n)$ è il costo del cammino trovato finora dalla partenza a $n$ e $f(n)$
è quindi la stima del costo totale del miglior cammino che passa per $n$.
Espandere sempre il nodo con $f$ minimo è A\*. Con $h \equiv 0$ si riduce alla
ricerca a costo uniforme, cioè al caso con la stima a zero, che è poi
l’algoritmo di Dijkstra con in più un test di arrivo. È la ricerca a costo
uniforme, e non quella in ampiezza, a restituire il cammino più economico
quando i passi costano diversamente: è completa e ottima se ogni passo costa
almeno un $\varepsilon > 0$ (con passi a costo nullo può girare per sempre su
un ciclo gratuito), e costa $O(b^{1+\lfloor C^*/\varepsilon \rfloor})$ in
tempo e in memoria, dove $C^*$ è il costo della soluzione ottima: con passi
piccoli rispetto a $C^*$ è molto più di $b^d$ {cite}`russell2020artificial`.

All’altro estremo, con la sola stima, cioè $f(n) = h(n)$, si ha la **ricerca
best-first golosa** (*greedy best-first search*): punta dritta alla meta e,
quando la stima è buona, apre pochissimo, ma non è ottima, e nella versione ad
albero non è nemmeno completa, perché un minimo locale della stima la tiene
in trappola. In mezzo sta A\* pesato {cite}`pohl1970heuristic`, con
$f(n) = g(n) + w\,h(n)$ e $w > 1$: con $h$ ammissibile la soluzione trovata
costa al più $w\,C^*$, in pratica spesso molto meno, e i nodi aperti calano di
molto. È la strada quando un’euristica ammissibile abbastanza informativa non
c’è, o quando una buona soluzione subito vale più di una ottima tardi
{cite}`russell2020artificial`.

Quanto A\* faccia meglio della ricerca a costo uniforme si misura col
**fattore di ramificazione effettivo** $b^*$: la ramificazione che dovrebbe
avere un albero uniforme profondo $d$ per contenere $N + 1$ nodi, dove $N$ è il
numero di nodi che la ricerca ha aperto. È cioè la radice di

$$
N + 1 = 1 + b^* + (b^*)^2 + \cdots + (b^*)^d .
$$

La misura l’ha proposta Nilsson; Russell e Norvig mettono al posto di $N$ i
nodi generati, e ottengono valori un poco più alti
{cite}`russell2020artificial`. Sul rompicapo delle otto tessere, con $d = 20$,
vale circa $1{,}64$ senza stima, $1{,}42$ con le tessere fuori posto e $1{,}22$
con la distanza a isolati: li stampa il conto che confronta le due stime. Con
un’euristica consistente A\* è anche *ottimamente efficiente*
{cite}`dechter1985generalized`: ogni algoritmo che estende cammini dalla
radice, usa la stessa euristica e garantisce una soluzione ottima deve aprire
tutti i nodi con $f(n) < C^*$, perché ciascuno potrebbe stare su un cammino
ottimo, e sono i nodi che A\* apre. Non per questo A\* smette di essere
esponenziale: quei nodi possono essere esponenzialmente tanti.

Il prezzo, che la tabella di ampiezza e profondità non dice, è scomodo: A\*
tiene in memoria tutti i nodi generati, esattamente come la ricerca in
ampiezza. Riduce enormemente quanti ne genera, e questo è tutto il guadagno, ma
la memoria resta il vincolo che morde per primo. Sul rompicapo delle otto
tessere non si vede; su quello delle quindici, che di posizioni ne ha diecimila
miliardi, sì, e la via d’uscita è sposare A\* con l’approfondimento iterativo,
tenendo un tetto sul valore di $f$ invece che sulla profondità. Il tetto non si
alza a piacere: il successivo è il più piccolo $f$ che ha sforato il
precedente, e senza quella regola la visita restituisce la prima soluzione che
sta sotto il tetto, che ottima non è. È l’**IDA\*** di Richard Korf
{cite}`korf1985depth`, ed è stato il primo metodo di uso
corrente a trovare, dentro limiti di tempo e di memoria praticabili, soluzioni
ottime di istanze del quindici generate a caso, con una memoria che cresce come
la profondità e non come il numero di nodi.

La proprietà che serve a $h$ ha un nome: è ammissibile se non sovrastima
mai, cioè se $h(n) \le h^*(n)$ per ogni $n$, dove $h^*(n)$ è il costo vero del
cammino ottimo da $n$ alla meta. Un’euristica ammissibile è, in altre parole,
ottimista. E con un’euristica ammissibile A\* restituisce una soluzione di
costo minimo.

La ragione, in poche righe e per un grafo a costi non negativi, con due
dettagli che sembrano formalità e non lo sono. A\* dichiara di aver finito
quando estrae dalla frontiera uno stato finale, non quando lo genera: se
bastasse generarlo, restituirebbe la prima soluzione che incontra, che non è la
più corta. Detto questo, supponiamo che stia per restituire una soluzione
peggiore di quella ottima, il cui costo è $C^*$. Lungo il cammino ottimo, il
primo nodo $n$ che non è ancora stato espanso con il suo costo ottimo
$g^*(n)$, cioè con il costo del cammino ottimo dalla partenza a $n$, sta sulla
frontiera: il suo predecessore sul cammino lo è stato, e lo ha generato con
$g(n) = g^*(n)$. Per quel nodo vale quindi
$f(n) = g^*(n) + h(n) \le g^*(n) + h^*(n) = C^*$: un valore non superiore a
$C^*$, e quindi inferiore a quello della soluzione peggiore che stiamo per
restituire. Ma allora A\* avrebbe estratto $n$ prima, perché estrae sempre il
minimo. È la contraddizione che dimostra il risultato.

Il «primo non ancora espanso con il suo costo ottimo» porta tutto il peso
dell’argomento: appartenere al cammino ottimo non basta, bisogna esserci
arrivati lungo di esso, e solo per il primo di quei nodi questo è garantito
dal predecessore. Ed è il punto in cui si sbaglia, perché presuppone di poter
tornare su uno stato già aperto se salta fuori una strada più corta per
arrivarci: un nodo espanso la prima volta con un costo troppo alto deve poter
tornare sulla frontiera. Il programma del rompicapo lo fa (è la riga che
riscrive `costo[t]` e rimette lo stato in coda); una versione che marchiasse
gli stati come «fatti» e non ci tornasse più potrebbe, con un’euristica solo
ammissibile, restituire una soluzione peggiore di quella ottima, e lo fa
davvero: in fondo a questa stessa sezione, con una stima costruita apposta,
porta a casa ventidue mosse invece di venti.

Una proprietà leggermente più forte si chiama **consistenza**: $h$ è
consistente se per ogni nodo $n$ e ogni suo successore $n'$ ottenuto con
l’azione $a$ vale

$$
h(n) \le c(n, a, n') + h(n'),
$$

che è una disuguaglianza triangolare: la stima da qui non può superare il costo
di un passo più la stima da lì. Ogni euristica consistente che valga zero
sugli stati finali è anche ammissibile, e non viceversa; la condizione sugli
stati finali serve davvero, perché $h \equiv 5$ soddisfa la disuguaglianza
triangolare su qualunque grafo a costi non negativi e ammissibile non è.

In cambio la consistenza dà una cosa pratica, ed è esattamente quella che manca
sopra: i valori di $f$ non diminuiscono mai lungo un cammino, perché passando
da $n$ a un successore $n'$ si ha
$f(n') = g(n) + c(n, a, n') + h(n') \ge g(n) + h(n) = f(n)$. Ne segue che la
prima estrazione di uno stato è ottima. Se $n$ venisse estratto con
$g(n) > g^*(n)$, sul cammino ottimo verso $n$ ci sarebbe un primo nodo $n'$
non ancora espanso con il suo costo ottimo, sulla frontiera con
$g(n') = g^*(n')$, e per la monotonia di $f$ lungo quel cammino
$f(n') \le g^*(n) + h(n) < g(n) + h(n) = f(n)$: A\* avrebbe estratto $n'$
prima di $n$. Marcare uno stato come fatto alla prima estrazione e non
tornarci più, allora, è lecito. La parola esatta è **estratto**, non
raggiunto. Uno stato si può *generare* per una strada pessima molto prima di
generarlo per quella buona (succede anche con $h \equiv 0$, che è consistente
ed è l’euristica della ricerca a costo uniforme), e chiudere uno stato alla
prima *generazione* può restituire soluzioni peggiori dell’ottimo, e basta che
succeda una volta perché la garanzia non ci sia più. La riga che riscrive
`costo[t]` esiste esattamente per questo. Sul rompicapo non scatta mai, ma non
basta che ogni mossa costi uno: su un grafo qualunque a costi unitari, anche
con un’euristica consistente, uno stato si può generare prima da un
predecessore con $g$ più alto e poi da uno con $g$ più basso, quando i due
differiscono di un passo in $g$ e di due nella stima. Sul rompicapo lo
impedisce la forma del grafo: ogni mossa porta il buco su una casella
dell’altro colore, quindi il grafo degli stati è bipartito e due strade verso
lo stesso stato differiscono di un numero pari di passi; e a parità di $f$ la
tupla fa uscire per primo chi ne ha fatti meno. A far scattare la riga sono i
passi che costano diversamente e i grafi con cicli dispari: è il caso generale
che la riga difende.

Le due euristiche del rompicapo sono tutte e due consistenti, e valgono
zero sulla configurazione finale.

La consistenza decide anche quale fra due euristiche convenga. Se $h_1$ e $h_2$
sono consistenti e $h_2(n) \ge h_1(n)$ per ogni $n$, si dice che $h_2$
**domina** $h_1$, e A\* con $h_2$ non apre mai un nodo che A\* con $h_1$ non
apra, salvo quelli con $f(n) = C^*$, su cui decide il criterio con cui si
rompono i pari. Infatti con un’euristica consistente A\* apre certamente ogni
nodo con $g^*(n) + h(n) < C^*$ (lungo il cammino ottimo verso di lui $f$ resta
sotto $C^*$, e la prima estrazione è ottima) e nessuno con
$g^*(n) + h(n) > C^*$; e l’insieme dei nodi con $h_2(n) < C^* - g^*(n)$ sta
dentro quello dei nodi con $h_1(n) < C^* - g^*(n)$. Con la sola ammissibilità
l’argomento cade: un nodo con $g^*(n) + h(n) < C^*$ può restare chiuso, se la
strada per arrivarci passa da nodi con $f > C^*$.

`````

A\* sta, con l’asterisco o senza, dentro quasi tutto ciò che cerca un percorso:
i robot che attraversano una stanza, i personaggi di videogioco che aggirano un
muro, e i navigatori stradali, che lo affiancano ai costi calcolati in
anticipo, perché da solo, su una rete di milioni di incroci, non basterebbe.

Le due euristiche classiche per il rompicapo delle otto tessere si ricavano
nello stesso modo, che è uno dei principali per inventarne una: si prende il
problema e gli si tolgono delle regole. Nel rompicapo vero una tessera si può
spostare solo in una casella adiacente e solo se quella casella è vuota. Se si
cancella la seconda regola, una tessera può andare in qualunque casella
adiacente, e il costo per rimettere tutto a posto è la somma di quanto
ciascuna tessera dista dal suo posto contando i passi in orizzontale e in
verticale: una tessera che sta due colonne e una riga lontano dal suo posto
conta tre. Si chiama **distanza a isolati**, e nei testi si trova più spesso
col nome inglese di **distanza di Manhattan**, che è la stessa cosa, perché è
il modo in cui si contano i metri in una città a scacchiera, dove non si taglia
in diagonale. Se si cancellano tutt’e due, una tessera vola dove vuole in una
mossa, e il costo è semplicemente quante tessere sono fuori posto. Sono le due
stime che il programma mette a confronto.

La garanzia è netta: il costo esatto di un problema con meno regole non può
mai superare quello del problema vero, perché tutto quello che si poteva fare
prima si può fare ancora, e magari qualcosa in più. Quindi un problema
alleggerito, risolto esattamente, dà sempre un numero che sta sotto (o al più
pari) a quello vero: è cioè un’euristica ammissibile per costruzione, e non c’è
bisogno di verificarlo caso per caso. Lo stesso argomento dà anche una
proprietà più forte, la *consistenza*: a ogni passo la stima non cala più di
quanto il passo costa, come la linea d’aria, che camminando cento metri si
accorcia al massimo di cento. Il costo esatto di un problema la rispetta in
quel problema, perché arrivare in due tappe non può costare meno che andarci
diritti (è la disuguaglianza triangolare); e siccome ogni mossa del problema
vero è anche una mossa di quello alleggerito, e non costa di meno, la
proprietà si trasporta ai passi veri. Il rilassamento serve, però, solo se il
problema alleggerito si risolve senza cercare: qui basta sommare, tessera per
tessera, un conto di passi, e la stima costa poco a ogni stato che la ricerca
guarda.

Togliere regole non è l’unico modo di costruire una stima che non esagera. Il
massimo fra più stime che non esagerano non esagera nemmeno lui, e vale almeno
quanto ciascuna; e una tabella preparata in anticipo, un *pattern database*,
tiene il costo esatto di un pezzo del problema, per esempio di poche tessere
con le altre lasciate indistinte, e lo usa come stima. Sul rompicapo delle
quindici tessere, con tabelle costruite in modo da potersi sommare, i nodi
generati scendono di un fattore diecimila rispetto alla distanza a isolati
{cite}`russell2020artificial`.

```python
def fuori_posto(s):
    """Quante tessere non sono al loro posto (la casella vuota non conta)."""
    return sum(1 for i, v in enumerate(s) if v and v != META[i])


def a_isolati(s):
    """Per ogni tessera, di quanti passi in orizzontale e in verticale
    e' lontana dal suo posto."""
    d = 0
    for i, v in enumerate(s):
        if v:
            g = META.index(v)
            d += abs(i // 3 - g // 3) + abs(i % 3 - g % 3)
    return d


def fattore_effettivo(aperti, d):
    """La b per cui un albero uniforme profondo d ha aperti + 1 nodi, cioè
    1 + b + b**2 + ... + b**d = aperti + 1, trovata per bisezione."""
    basso, alto = 1.0, 10.0
    for _ in range(60):
        b = (basso + alto) / 2
        if sum(b ** i for i in range(d + 1)) < aperti + 1:
            basso = b
        else:
            alto = b
    return basso


risultati = {"nessuna stima": (passi, senza_stima)}
for nome, stima in (("tessere fuori posto", fuori_posto),
                    ("distanza a isolati", a_isolati)):
    risultati[nome] = cerca(stima)
    passi, guardati = risultati[nome]
    print(f"{nome:22} {passi} mosse, {guardati:5d} stati guardati"
          f"   ({senza_stima / guardati:5.1f} volte meno)")
print("fattore di ramificazione effettivo:")
for nome, (passi, guardati) in risultati.items():
    print(f"  {nome:20} b* = {fattore_effettivo(guardati, passi):.2f}")
```

```text
tessere fuori posto    20 mosse,  3666 stati guardati   ( 13.2 volte meno)
distanza a isolati     20 mosse,   282 stati guardati   (171.6 volte meno)
fattore di ramificazione effettivo:
  nessuna stima        b* = 1.64
  tessere fuori posto  b* = 1.42
  distanza a isolati   b* = 1.22
```

La risposta non cambia: venti mosse in tutti e tre i casi, perché tutte e tre
le stime sono ottimiste e quindi nessuna fa sbagliare strada. Cambia solo
quanto si guarda: quarantottomila stati senza stima, tremilaseicento con
quella grossolana, duecentottantadue con quella più fine. Il fattore di
ramificazione effettivo dice la stessa cosa con un numero solo: è quanti figli
dovrebbe avere ogni nodo di un albero regolare profondo venti per contenere
tanti stati quanti ne sono stati guardati, e scende da 1,64 a 1,22.

I conteggi non dicono dove siano finite le posizioni guardate, ed è la parte
che spiega il resto. Si può disegnare: ogni posizione aperta si mette su un
piano, in orizzontale i passi già fatti per arrivarci e in verticale la
distanza a isolati che le resta ({numref}`fig-frontiera`). Il numero in
verticale è lo stesso nei due riquadri, anche in quello della ricerca senza
stima: quello che cambia è soltanto se l’algoritmo lo usa.

```{figure} ../figures/frontiera-che-si-allarga.svg
:name: fig-frontiera
:alt: "Due riquadri affiancati con lo stesso piano: in orizzontale i passi già fatti dalla partenza, in verticale la stima di quanto la posizione disti ancora dalla meta, contata a isolati. In ciascuno una riga obliqua segna le venti mosse della soluzione. Le posizioni che la ricerca apre si accendono a poco a poco. A sinistra, «senza stima», si accendono dappertutto, e la maggior parte finisce sopra la riga obliqua, cioè in posizioni per cui i passi fatti più quelli stimati superano già il costo della soluzione; il contatore sotto arriva a 48.389. A destra, «con la distanza a isolati», si accendono soltanto lungo una fascia stretta che segue la riga obliqua e non la supera mai, e il contatore si ferma a 282."
:width: 100%

La stessa ricerca, sullo stesso rompicapo, senza e con la stima. La riga
obliqua è la soluzione, cioè la fila dei punti in cui passi fatti e passi
stimati sommano a venti, quante sono le mosse della strada giusta: per una
posizione che sta sopra, i passi già fatti più
quelli stimati superano già la lunghezza della soluzione intera, e aprirla è
tempo perso. Senza la stima la ricerca ci finisce di continuo; con la stima non
ci mette piede, ed è proprio la garanzia che A\* dà.
```

La stima non fa guardare *un po’ meno dappertutto*: impedisce alla ricerca di
salire sopra quella riga. Senza la stima l’algoritmo confronta i soli passi
fatti, non c’è nessuna riga da non superare, e ogni posizione vale quanto
un’altra alla stessa distanza dalla partenza.

Il confronto fra le due stime ha una regola sola, che si legge nelle
definizioni: la distanza a isolati vale sempre almeno quanto il numero di
tessere fuori posto, perché una tessera fuori posto dista almeno un passo da
dove dovrebbe stare. Si dice che la prima *domina* la seconda, e quando tutte
e due sono consistenti, come qui, A\* con la stima che domina non apre mai più
stati dell’altra, a meno dei pari. Gli stati che è costretto ad aprire sono
quelli per cui i passi fatti più la stima stanno sotto il costo della
soluzione: finché quella somma sta sotto non c’è modo di escludere che di là
passi una strada più corta, e l’algoritmo deve andarci a guardare, mentre
alzare la stima restringe quell’insieme. Cercare una buona euristica vuol dire
cercare la stima più alta che non superi mai il vero e che si calcoli in
fretta: la stima perfetta, la distanza vera, farebbe aprire soltanto gli stati
che stanno su una strada più corta, ma calcolarla vorrebbe dire aver già
risolto il problema.

Resta da vedere che cosa compra la consistenza, e lo si vede togliendola. La
stima *a salti* vale la distanza a isolati quando le tessere fuori posto sono
in numero pari, e zero quando sono dispari: non esagera mai, perché non supera
la distanza a isolati, ma fra due posizioni vicine può saltare da zero a una
ventina di passi, e consistente non è. Il conto rifà la ricerca con due
interruttori: se uno stato già aperto si può riaprire quando salta fuori una
strada più corta, e chi esce per primo dalla coda a parità di somma.

```python
def cerca_variante(stima, riapre=True, verso_la_meta=False):
    """Come cerca, con due interruttori: se uno stato gia' aperto si puo'
    riaprire, e chi esce per primo a parita' di somma (con verso_la_meta
    chi ha fatto piu' passi, altrimenti chi ne ha fatti meno)."""
    segno = -1 if verso_la_meta else 1
    coda = [(stima(PARTENZA), 0, PARTENZA)]
    costo, chiusi, guardati = {PARTENZA: 0}, set(), 0
    while coda:
        _, chiave, s = heapq.heappop(coda)
        fatti = segno * chiave
        if s == META:
            return fatti, guardati
        if fatti > costo[s] or s in chiusi:
            continue
        guardati += 1
        if not riapre:
            chiusi.add(s)               # aperto una volta, chiuso per sempre
        for t in mosse(s):
            if t not in chiusi and fatti + 1 < costo.get(t, 10**9):
                costo[t] = fatti + 1
                heapq.heappush(coda, (fatti + 1 + stima(t),
                                      segno * (fatti + 1), t))
    raise AssertionError("nessuna soluzione")


def a_salti(s):
    """La distanza a isolati se le tessere fuori posto sono in numero pari,
    zero altrimenti: non esagera mai, ma salta."""
    return a_isolati(s) if fuori_posto(s) % 2 == 0 else 0


for nome, stima in (("distanza a isolati", a_isolati),
                    ("stima a salti", a_salti)):
    for riapre in (True, False):
        passi, guardati = cerca_variante(stima, riapre=riapre)
        print(f"{nome:18} {'riapre' if riapre else 'non riapre':10} "
              f"{passi} mosse, {guardati:5d} stati guardati")
print("a somma pari esce per primo chi ha fatto più passi:")
for nome, stima in (("tessere fuori posto", fuori_posto),
                    ("distanza a isolati", a_isolati)):
    passi, guardati = cerca_variante(stima, verso_la_meta=True)
    print(f"  {nome:20} {passi} mosse, {guardati:5d} stati guardati")
```

```text
distanza a isolati riapre     20 mosse,   282 stati guardati
distanza a isolati non riapre 20 mosse,   282 stati guardati
stima a salti      riapre     20 mosse,  1804 stati guardati
stima a salti      non riapre 22 mosse,  2880 stati guardati
a somma pari esce per primo chi ha fatto più passi:
  tessere fuori posto  20 mosse,  2312 stati guardati
  distanza a isolati   20 mosse,    93 stati guardati
```

Con la distanza a isolati, che è consistente, chiudere gli stati alla prima
estrazione non costa niente. Con la stima a salti, chi riapre trova le venti
mosse, e chi chiude ne porta a casa ventidue: qualche stato è stato chiuso
quando ci si era arrivati per una strada più lunga, e da chiuso non si
corregge più. Le ultime righe dicono un’altra cosa, che i conteggi di prima non
facevano vedere: a parità di somma, decidere chi esce per primo cambia di
molto quanto si guarda. Il programma del rompicapo fa uscire chi ha fatto meno
passi; facendo uscire chi ne ha fatti di più, cioè chi è più vicino alla meta,
gli stati guardati scendono da 3.666 a 2.312 con le tessere fuori posto e da
282 a 93 con la distanza a isolati, sempre con venti mosse.

`````{tab} Elementare

```{admonition} Da ricordare
:class: important
- Cercare a tentoni si può fare in due modi: allagando il labirinto un
  metro alla volta (la ricerca in ampiezza: si trova la strada più corta, e si
  consuma una memoria enorme) oppure seguendo un corridoio fino in fondo con
  un filo dietro (la ricerca in profondità: il filo costa pochissimo, la strada
  trovata può essere assurda, e dove i corridoi girano in tondo non si esce più
  finché non si segnano i bivi già visti, cioè finché non si ricomincia a
  pagare la memoria risparmiata).
- I due si sposano girando col filo, ma con un tetto che si alza a ogni giro.
  Rifare ogni volta i primi metri costa poco, perché i bivi vicini all’inizio
  sono pochissimi rispetto a quelli lontani, e in cambio si tengono la memoria
  del filo e la garanzia dell’acqua.
- Il difetto vero del cercare a tentoni è che si cerca dappertutto con lo
  stesso impegno, anche dalla parte sbagliata; la memoria viene dopo.
- La cosa che cambia di più le proporzioni è una stima di quanto manca,
  un’euristica: la distanza in linea d’aria dalla torre vicino a casa
  dell’amico. Non dice quale strada prendere, dice solo da che parte guardare
  per primo. A\* tiene aperte più strade e allunga sempre quella in cui i metri
  fatti più la stima danno la somma più piccola; quando il foglio non basta,
  IDA\* fa lo stesso col filo e un tetto sulla somma.
- La stima deve stare sotto al vero, mai sopra, e allora si dice ammissibile.
  Se sbaglia per difetto si guarda qualcosa di troppo; se sbaglia per eccesso
  si scarta la strada buona e si arriva più lunghi, senza nemmeno
  accorgersene. E se salta da un incrocio a quello accanto, chi non torna mai
  sui propri passi può arrivare più lungo anche con una stima che non esagera.
- Una stima si inventa togliendo regole al problema: risolto esattamente,
  un problema con meno regole non può mai costare più di quello vero (al
  massimo costa uguale), quindi la sua soluzione è automaticamente una stima
  che non esagera.
- Sul rompicapo delle otto tessere: senza stima 48.389 posizioni guardate, con
  la stima grossolana 3.666, con quella fine 282. E la risposta è la stessa
  tutte e tre le volte, venti mosse.
```

`````

`````{tab} Superiore

```{admonition} Da ricordare
:class: important
- Ampiezza e profondità differiscono per la disciplina della frontiera (coda o
  pila). L’ampiezza è completa e ottima a costi uniformi ma costa $O(b^d)$ di
  memoria, che è il vincolo che morde per primo; la profondità costa
  $O(bm)$ ma non è né completa né ottima. L’approfondimento iterativo le
  unisce pagando solo un fattore $b/(b-1)$ di lavoro in più, con memoria
  $O(bd)$ {cite}`korf1985depth`.
- A\* {cite}`hart1968formal` espande il nodo di $f(n) = g(n) + h(n)$
  minimo. Con $h$ ammissibile ($h \le h^*$, cioè ottimista) restituisce una
  soluzione di costo minimo, purché si possa tornare su uno stato già aperto
  quando salta fuori una strada più corta per arrivarci; con $h$ consistente
  ($h(n) \le c(n,a,n') + h(n')$, disuguaglianza triangolare) i valori di $f$
  non decrescono lungo un cammino e ogni stato viene estratto dalla
  frontiera in modo ottimo la prima volta (estratto, non generato: per una
  strada pessima lo si genera anche molto prima). Con un’euristica ammissibile
  e non consistente, chiudere gli stati alla prima estrazione dà sul rompicapo
  ventidue mosse invece di venti.
- Con $f = h$ si ha la ricerca golosa, che non è ottima e ad albero nemmeno
  completa; con $f = g + w\,h$ e $w > 1$, A\* pesato restituisce soluzioni di
  costo al più $w\,C^*$. IDA\* tiene un tetto su $f$ al posto della frontiera,
  e la memoria cresce come la profondità.
- Le euristiche ammissibili si costruiscono rilassando il problema: il
  costo esatto di un problema con vincoli in meno è un limite inferiore a
  quello del problema vero, quindi è ammissibile per costruzione, e serve se il
  problema rilassato si risolve senza ricerca. Le due del rompicapo (tessere
  fuori posto, distanza a isolati) sono i rilassamenti che cancellano
  rispettivamente due vincoli e uno.
- Fra due euristiche consistenti, quella con valori sempre maggiori
  domina l’altra, e A\* con la dominante non espande mai più nodi
  dell’altra (con l’eccezione dei nodi a $f = C^*$, dove decide il criterio con
  cui si rompono i pari). L’ipotesi è la consistenza e non la sola
  ammissibilità, perché la dimostrazione passa per «ogni nodo con
  $f(n) < C^*$ viene certamente espanso», che vale sotto consistenza. Sul
  rompicapo delle otto tessere: 48.389 nodi con $h \equiv 0$, 3.666 con le
  tessere fuori posto, 282 con la distanza a isolati, a parità di soluzione
  ottima (20 mosse).
```

`````

Tutto questo vale finché il mondo sta fermo mentre ci pensiamo. Il navigatore
può permettersi di calcolare la strada intera fino in fondo perché la strada,
mentre lui calcola, non cambia idea. Quando si {doc}`gioca contro qualcuno
</Ricerca/giocare-contro-qualcuno>`, invece, dall’altra parte del tavolo c’è
qualcuno che sceglie anche lui, e sceglie apposta il ramo che a noi conviene
meno.
