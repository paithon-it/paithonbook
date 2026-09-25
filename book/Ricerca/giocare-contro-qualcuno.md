# Giocare contro qualcuno: minimax, potatura, orizzonte

In un labirinto i corridoi stanno fermi. Se ne provo uno e non porta da nessuna
parte, il labirinto non si riorganizza per dispetto.

Con un avversario davanti cambia tutto, e cambia in un punto solo: metà delle
mosse non le scelgo io. L’albero è lo stesso, i rami sono gli stessi, ma un
livello sì e uno no li sceglie qualcuno che vuole esattamente il contrario di
quello che voglio io. Non posso più chiedermi «qual è la strada migliore»: devo
chiedermi «qual è la mossa che regge anche alla risposta peggiore».

## Ragionare all’indietro dalla fine

Il modo di rispondere si chiama **minimax**, e il nome sono le sue due metà:
c’è un punteggio solo sul tavolo, e uno dei due giocatori cerca di portarlo al
massimo mentre l’altro cerca di portarlo al minimo. Un punto guadagnato da me è
un punto perso da lui, esattamente: non esiste una mossa che convenga a tutti e
due.

`````{tab} Elementare

Facciamo il conto su un albero piccolissimo, di due mosse soltanto: prima muovo
io, poi muove lui, e a quel punto la partita è finita e si legge il punteggio.
Punteggi alti vuol dire bene per me.

Ho tre mosse. Se gioco la prima, lui può rispondere in tre modi, che portano a
3, 12 e 8 punti. Se gioco la seconda, le sue tre risposte portano a 2, 4 e 6.
Se gioco la terza, portano a 14, 5 e 2.

Adesso l’istinto sbagliato: «gioco la terza, che porta a 14». No. Il 14 non lo
sceglierei io, lo sceglierebbe lui, e lui vuole il numero più piccolo: davanti
alla mia terza mossa risponderebbe con il 2. Quindi la mia terza mossa non vale
14, vale 2.

Rifacciamo il conto come va fatto, dal basso. La prima mossa vale il minimo fra
3, 12 e 8, cioè 3. La seconda vale il minimo fra 2, 4 e 6, cioè 2. La terza
vale il minimo fra 14, 5 e 2, cioè 2. E adesso tocca a me, che voglio il
massimo: fra 3, 2 e 2 scelgo il 3, cioè la prima mossa.

Il gesto è tutto qui, e si chiama ragionare all’indietro: il valore di una
posizione è quello che si ottiene giocando fino in fondo, supponendo che da lì
in avanti giochino bene tutti e due. Il numero più alto che si vede sotto non
c’entra niente, perché a sceglierlo non sono io. E il conto si costruisce
partendo dalle foglie e risalendo, un livello alla volta, alternando «prendi il
massimo» e «prendi il minimo».

Il conto dà per scontato che lui giochi sempre la risposta migliore. Contro
qualcuno che si distrae, la terza mossa potrebbe fruttare davvero 14, e quel 14
il conto non lo mette nemmeno sul tavolo: si tiene il 3 sicuro. Chi ragiona
così gioca contro il migliore avversario possibile, anche quando dall’altra
parte c’è un principiante che gli regalerebbe la partita.

E dà per scontato di arrivare in fondo. Nel nostro alberello la partita finiva
dopo due mosse e il punteggio era scritto. In un gioco vero il fondo resta
fuori portata: se ogni mossa ne apre trenta e si guardano dieci mosse per
parte, le partite da srotolare sono un numero di trenta cifre.

`````

`````{tab} Superiore

Per un gioco a due giocatori, deterministico, a somma zero e a
informazione perfetta (cioè in cui ciascuno vede tutta la posizione: è
un’altra cosa dall’«informazione completa» della teoria dei giochi, che
riguarda il conoscere i guadagni dell’avversario), il valore minimax di uno
stato $s$ è definito ricorsivamente:

$$
\mathrm{minimax}(s) =
\begin{cases}
u(s) & \text{se } s \text{ è terminale},\\[2pt]
\max_{a \in \mathcal{A}(s)} \mathrm{minimax}(\mathrm{ris}(s,a)) & \text{se tocca a chi massimizza},\\[2pt]
\min_{a \in \mathcal{A}(s)} \mathrm{minimax}(\mathrm{ris}(s,a)) & \text{se tocca a chi minimizza},
\end{cases}
$$

dove $u(s)$ è l’**utilità** dello stato terminale letta dal punto di vista di
chi massimizza, $\mathcal{A}(s)$ le mosse legali e $\mathrm{ris}(s,a)$ lo stato che ne
segue. Il valore così definito è quello che si ottiene se entrambi giocano in
modo ottimo da lì alla fine, ed è un’ipotesi forte: contro un avversario che
sbaglia, minimax non è la strategia che ne sfrutta di più gli errori, perché
sceglie sempre la mossa che regge alla risposta migliore e non quella che
guadagna di più dalla risposta probabile.

L’algoritmo è una visita in profondità che scende fino alle foglie e risale
combinando. Costa $O(b^m)$ in tempo, con $b$ il numero di mosse legali per
posizione e $m$ la profondità dell’albero, e $O(bm)$ in memoria. Su un gioco vero è impraticabile per lo stesso conto
dell’apertura del capitolo: agli scacchi $35^{80}$.

Minimax non è un’euristica e non approssima niente: dato l’albero completo, il
valore che restituisce è esatto. Evitare di costruire quell’albero si può fare
in due modi, e confonderli costa caro: calcolare lo stesso valore guardando
meno (la potatura, che non perde niente) e calcolare un valore diverso perché
quello vero è fuori portata (la funzione di valutazione, che perde eccome).

`````

Il conto si può fare per intero su un gioco che finisce davvero: il tris, tre
caselle per lato, quello che si gioca sul tovagliolo e che in mezza Italia si
chiama filetto. Le partite possibili
sono poche abbastanza da poterle percorrere tutte, e il risultato è noto a
chiunque ci abbia giocato abbastanza: giocando bene tutti e due, finisce
sempre in parità.

```python
VINCENTI = [(0,1,2), (3,4,5), (6,7,8), (0,3,6),
            (1,4,7), (2,5,8), (0,4,8), (2,4,6)]


def esito(t):
    """1 se ho vinto io, -1 se ha vinto lui, 0 se e' patta,
    None se la partita non e' ancora finita."""
    for a, b, c in VINCENTI:
        if t[a] and t[a] == t[b] == t[c]:
            return t[a]
    return 0 if all(t) else None


guardate = {"minimax": 0}


def minimax(t, tocca_a_me):
    fine = esito(t)
    if fine is not None:
        guardate["minimax"] += 1
        return fine
    segno = 1 if tocca_a_me else -1
    valori = [minimax(t[:i] + (segno,) + t[i+1:], not tocca_a_me)
              for i in range(9) if not t[i]]
    return max(valori) if tocca_a_me else min(valori)


vuota = (0,) * 9
print(f"esito con gioco perfetto: {minimax(vuota, True)}")
print(f"partite portate fino in fondo: {guardate['minimax']}")
```

```text
esito con gioco perfetto: 0
partite portate fino in fondo: 255168
```

Lo zero è la patta, ed è la risposta giusta. Il numero sotto va letto con
attenzione, perché è il conto che serve: sono duecentocinquantacinquemila
partite intere e non *posizioni diverse* (di quelle un gioco da nove
caselle ne ha molte meno), giocate una per una dalla prima mossa all’ultima.
Sono meno delle $9! = 362\,880$ sequenze con cui si possono riempire nove
caselle, perché una partita si ferma appena qualcuno allinea tre simboli, anche
a tabellone mezzo vuoto. È l’albero dell’apertura del capitolo in miniatura:
piccolo abbastanza da srotolarlo tutto, e già abbastanza grande da far vedere
il problema.

## Smettere di guardare: la potatura

C’è un modo di ottenere esattamente lo stesso numero srotolando una
frazione di quelle partite, e senza nessuna approssimazione: basta accorgersi
che certi rami, qualunque cosa contengano, non possono cambiare la risposta.

`````{tab} Elementare

Torniamo all’alberello di prima, e stavolta guardiamo le foglie una per volta,
da sinistra, come farebbe qualcuno che le scopre a mano a mano.

Della mia prima mossa scopro 3, 12, 8: lui sceglierebbe il minimo, quindi
quella mossa vale 3. Adesso so una cosa che non mollo più: qualunque cosa
succeda, non accetterò meno di 3.

Passo alla mia seconda mossa. Scopro la prima risposta di lui: 2. E qui mi
fermo, perché ho già finito di ragionare. Lui, su questa mossa, prenderà il
minimo fra 2 e le altre due che non ho ancora guardato: quindi al massimo
prenderà 2, e forse meno. Comunque vada, questa mossa non vale più di 2, cioè
meno del 3 che ho già in tasca. Le altre due risposte non le guardo
nemmeno: non c’è nessun numero che possano contenere capace di farmi cambiare
idea. Anche se ci fosse un milione, lui non me lo lascerebbe prendere.

Passo alla terza. Scopro 14: non basta a decidere, perché lui prenderà il
minimo e potrebbe esserci di peggio. Scopro 5: idem. Scopro 2: adesso so che
questa mossa vale 2, meno di 3. Anche questa scartata.

Risposta finale: la prima mossa, che vale 3. La stessa di prima. E ho guardato
sette foglie su nove.

Il gesto ha un nome che si spiega da sé: **potatura**, come i rami che si
tagliano a un albero (per esteso, potatura *alfa-beta*, dai nomi dei due
segnalibri con cui il programma ricorda quanto ciascuno dei due giocatori si è
già garantito). E la frase che la produce è una sola, quella che si dice
a se stessi guardando la seconda mossa: «questa strada è già peggio della
migliore che ho trovato, non la guardo nemmeno».

Il punto di rottura, che conta moltissimo in pratica: quanto si pota dipende
dall’ordine in cui si guardano le mosse. Se la mossa buona capita per prima,
tutte le altre si scartano in fretta perché c’è già un metro alto da superare;
se capita per ultima, il metro resta basso a lungo e non si scarta quasi
niente. Lo stesso algoritmo, sullo stesso albero, può guardare pochissimo o
quasi tutto a seconda dell’ordine.

`````

`````{tab} Superiore

La potatura **alfa-beta**, di cui Knuth e Moore hanno dato l'analisi che si
cita ancora {cite}`knuth1975analysis`, porta lungo la ricorsione due valori:
$\alpha$, il migliore che chi massimizza si è già assicurato lungo il cammino
corrente, e $\beta$, il migliore per chi minimizza. La regola è simmetrica: in
un nodo di massimo si interrompe l’esplorazione dei figli non appena il valore
corrente arriva a $\beta$ o lo supera; in un nodo di minimo, non appena scende
ad $\alpha$ o sotto. Sono le due condizioni `v >= beta` e `v <= alfa`, e il
caso di uguaglianza conta: con la disuguaglianza stretta il taglio scatterebbe
meno spesso, e il risparmio si ridurrebbe di parecchio.

La correttezza si vede con un conto di tre righe sull’albero d’esempio, quello
con foglie $3, 12, 8$ sotto la prima mossa, $2, 4, 6$ sotto la seconda e
$14, 5, 2$ sotto la terza. Chiamando $x$ e $y$ le due foglie del secondo ramo
che non vengono esaminate, il valore alla radice è

$$
\max\big(\min(3,12,8),\ \min(2,x,y),\ \min(14,5,2)\big)
= \max\big(3,\ z,\ 2\big), \qquad z = \min(2,x,y) \le 2,
$$

e siccome $z \le 2 < 3$ il massimo vale 3 indipendentemente da $x$ e $y$.
Non è un’approssimazione: alfa-beta restituisce sempre lo stesso valore di
minimax alla radice.

Il guadagno dipende dall’ordinamento delle mosse. Nel caso migliore, cioè
esaminando per prima la mossa migliore in ogni nodo, alfa-beta esamina
$O(b^{m/2})$ nodi invece di $O(b^m)$: il fattore di ramificazione effettivo
diventa $\sqrt{b}$, che agli scacchi vuol dire circa 6 invece di 35, ossia la
possibilità di guardare il doppio più a fondo nello stesso tempo. Con
ordinamento casuale, e per $b$ moderati, si scende a circa $O(b^{3m/4})$
{cite}`russell2020artificial`.

Da qui il fatto che nei programmi di gioco l’ordinamento delle mosse non è una
rifinitura ma una parte dell’algoritmo. Due tecniche classiche: provare per
prime, in un nodo, le mosse che hanno già prodotto un taglio alla stessa
profondità in un altro ramo dell’albero (le **killer move**: se una mossa ha
confutato una linea, spesso ne confuta anche una parallela), e usare
l’approfondimento iterativo della sezione precedente non
solo per gestire il tempo, ma per ordinare: si cerca a profondità uno, si
ordinano le mosse secondo quel risultato, si cerca a profondità due partendo da
quell’ordine, e così via. Il tempo speso nelle passate superficiali si ripaga
con gli interessi in quelle profonde.

`````

Il conto sul tris si rifà identico, cambiando solo la funzione.

```python
guardate["alfabeta"] = 0


def alfabeta(t, tocca_a_me, alfa=-2, beta=2):
    fine = esito(t)
    if fine is not None:
        guardate["alfabeta"] += 1
        return fine
    if tocca_a_me:
        v = -2
        for i in range(9):
            if not t[i]:
                v = max(v, alfabeta(t[:i] + (1,) + t[i+1:], False, alfa, beta))
                alfa = max(alfa, v)
                if v >= beta:          # lui non mi lascerebbe mai arrivare qui
                    break
        return v
    v = 2
    for i in range(9):
        if not t[i]:
            v = min(v, alfabeta(t[:i] + (-1,) + t[i+1:], True, alfa, beta))
            beta = min(beta, v)
            if v <= alfa:              # io non sceglierei mai questo ramo
                break
    return v


print(f"esito con gioco perfetto: {alfabeta(vuota, True)}")
print(f"partite portate fino in fondo: {guardate['alfabeta']}")
print(f"rapporto: {guardate['minimax'] / guardate['alfabeta']:.1f} volte meno")
```

```text
esito con gioco perfetto: 0
partite portate fino in fondo: 7330
rapporto: 34.8 volte meno
```

Stessa risposta, quasi trentacinque volte meno lavoro. E conviene insistere su
«stessa risposta», perché è la cosa che rende la potatura diversa da tutti gli
altri risparmi della ricerca: non si è rinunciato a niente. I rami
non guardati erano rami di cui si era dimostrato, senza guardarli, che non
potevano cambiare la conclusione.

E l'ordine? Sul tris si può misurare: basta guardare le caselle in un ordine
diverso, il che non cambia il gioco di una virgola.

```python
import random


def con_ordine(ordine):
    """Alfa-beta scandendo le caselle nell'ordine dato. Il risultato non
    cambia mai; cambia solo quanto lavoro serve per ottenerlo."""
    guardate = [0]

    def ab(t, tocca_a_me, alfa=-2, beta=2):
        fine = esito(t)
        if fine is not None:
            guardate[0] += 1
            return fine
        libere = [i for i in ordine if not t[i]]
        if tocca_a_me:
            v = -2
            for i in libere:
                v = max(v, ab(t[:i] + (1,) + t[i+1:], False, alfa, beta))
                alfa = max(alfa, v)
                if v >= beta:
                    break
            return v
        v = 2
        for i in libere:
            v = min(v, ab(t[:i] + (-1,) + t[i+1:], True, alfa, beta))
            beta = min(beta, v)
            if v <= alfa:
                break
        return v

    assert ab(vuota, True) == 0        # la risposta e' sempre la patta
    return guardate[0]


a_caso = []
for seme in range(20):
    mescolato = list(range(9))
    random.Random(seme).shuffle(mescolato)
    a_caso.append(con_ordine(mescolato))

ragionato = con_ordine([4,0,2,6,8,1,3,5,7])
print(f"in ordine di casella (quello di prima): {con_ordine(range(9)):6d}")
print(f"centro e angoli per primi:              {ragionato:6d}")
print(f"bordi per primi:                        {con_ordine([1,3,5,7,0,2,6,8,4]):6d}")
print(f"venti ordini a caso: da {min(a_caso)} a {max(a_caso)}, "
      f"e {sum(g < ragionato for g in a_caso)} su 20 batte il ragionato")
```

```text
in ordine di casella (quello di prima):   7330
centro e angoli per primi:                2893
bordi per primi:                         17002
venti ordini a caso: da 2603 a 13358, e 1 su 20 batte il ragionato
```

Sei volte fra il migliore e il peggiore, sullo stesso gioco, con lo stesso
algoritmo e con la stessa risposta in fondo. E l'ordine ragionato non è
lontano dal migliore che si trovi a tentativi: mettere per primi il centro e
gli angoli vuol dire provare per prime le caselle che nel tris contano di più,
e dei venti ordini pescati a caso uno solo fa meglio. I programmi di scacchi
fanno la stessa scommessa in un altro modo: prima della ricerca vera ne fanno
una corta, di poche mosse, e usano quel risultato per decidere in che ordine
guardare. Il conto della potatura, insomma, non si fa una volta per tutte: si
fa sull’ordine che si è scelto.

## Quando il fondo non si raggiunge

Nel tris la partita finisce, e il punteggio in fondo c’è scritto. Agli scacchi
no: dopo dieci mosse per parte si è ancora in mezzo alla partita, e in fondo
all’albero non c’è nessun numero da leggere.

Allora si fa la cosa che un giocatore umano fa da sempre: si guarda avanti
finché si può, ci si ferma, e si giudica a occhio la posizione a cui si è
arrivati. Quel giudizio è una **funzione di valutazione**, e prende il posto
del punteggio vero. È qui che la ricerca smette di essere esatta.

`````{tab} Elementare

Cinque secondi bastano a un giocatore esperto per dire chi sta meglio a metà
partita. Conta i pezzi, e una torre vale più di un alfiere; guarda il re, al
riparo o allo scoperto, i pedoni che si difendono a vicenda, chi tiene le
caselle in mezzo, da cui si arriva ovunque in fretta. Resta un giudizio, e due
maestri sulla stessa posizione dicono cose diverse.

Il programma si siede sulla stessa sedia con un foglietto di conti. Tanti punti
per ogni pezzo secondo quanto vale, qualche punto per il re al sicuro, qualche
punto per ogni casella centrale che tiene. Somma, e il totale è il suo voto.
Guarda avanti quattro mosse, o sei, o dieci, e dove si ferma scrive quel voto
invece di tirare avanti; poi ragiona all’indietro da quei voti, non dai
punteggi veri.

Un foglietto del genere deve compilarsi in un attimo, perché di posizioni ne
passano milioni. A partita finita deve dire quello che dice il risultato, vinta
o persa senza sfumature. E chi esce col voto più alto deve vincere più spesso,
unica ragione per fidarsene.

Una finta però il foglietto la fa. Conta i pezzi su una riga e i pedoni su
un’altra, come se ciascuno se ne stesse per conto suo. Un alfiere chiuso dietro
i propri pedoni non va da nessuna parte e in partita vale poco, ma sul
foglietto vale quanto uno libero. Grossa com’è, la finta si accetta, perché un
voto grossolano che arriva subito serve più di un voto giusto che non arriva
mai.

Fermarsi sempre alla stessa distanza ha un costo con un nome: **l’effetto
orizzonte**. Il mio alfiere è spacciato, comunque giochi fra sei mosse me lo
prendono, e io guardo avanti otto mosse: quella perdita la vedo, e mi pesa. Do
allora tre scacchi inutili al suo re, che sotto scacco deve rispondere e non può
fare altro: ogni scacco gli ruba una mossa, e regalando un pedone per volta la
cattura slitta a sette mosse, a otto, a nove, fuori dal mio orizzonte. Riguardo,
l’alfiere è salvo, e concludo che regalare pedoni sia un’ottima idea. Nessuno ha
sbagliato a programmare: capita a chiunque giudichi il mondo a una scadenza
fissa, ben oltre gli scacchi. Il disastro sta ancora là, appena oltre il punto
in cui smetto di guardare, e i pedoni li ho pagati davvero.

Un rimedio a metà lo conosce ogni giocatore. Se dove arrivo i pezzi si stanno
ancora mangiando a vicenda, lì non mi fermo. Tiro avanti finché le acque non si
calmano, e solo allora compilo il foglietto. L’orizzonte si sposta dove fa meno
danni; sparire non sparisce.

Muovo cavallo e poi alfiere, oppure alfiere e poi cavallo: la scacchiera
davanti è la stessa, e ricompilare il foglietto sarebbe tempo buttato. Allora
tengo da parte ogni posizione già giudicata col suo voto, e me lo riprendo
quando la stessa scacchiera ricapita per un’altra strada. Agli scacchi tanto
basta per scendere due volte più a fondo nello stesso tempo.

`````

`````{tab} Superiore

Si sostituisce l’utilità terminale $u(s)$ con una valutazione $\mathrm{ev}(s)$
e il test di terminazione con un **test di taglio**, ottenendo il minimax
euristico

$$
\mathrm{h\text{-}minimax}(s, k) =
\begin{cases}
\mathrm{ev}(s) & \text{se il taglio scatta in } (s,k),\\[2pt]
\max_a \mathrm{h\text{-}minimax}(\mathrm{ris}(s,a),\, k+1) & \text{se tocca a chi massimizza},\\[2pt]
\min_a \mathrm{h\text{-}minimax}(\mathrm{ris}(s,a),\, k+1) & \text{se tocca a chi minimizza},
\end{cases}
$$

dove $k$ conta i livelli già scesi lungo la ricorsione (vale zero alla radice e
cresce di uno a ogni mossa giocata: non è il $d$ della profondità della
soluzione), e il taglio scatta quando $k$ arriva al limite fissato o la
posizione è comunque terminale.

Perché $\mathrm{ev}$ sia utile deve concordare con $u$ sugli stati terminali,
essere calcolabile in fretta, e correlare con la probabilità di vittoria. In
pratica è quasi sempre una somma pesata di caratteristiche della posizione, il
che assume implicitamente che i loro contributi siano indipendenti: un’ipotesi
falsa (il valore di un alfiere dipende da com’è la struttura pedonale) e utile
lo stesso.

Nei giochi con il caso (il backgammon, dove muove il dado) fra i livelli dei
due giocatori si inseriscono i **nodi di caso**, e lì la ricorsione prende il
valore atteso: $\mathrm{expectiminimax}(s) = \sum_{e} P(e)\,
\mathrm{expectiminimax}(\mathrm{ris}(s,e))$, dove $e$ corre sugli esiti
possibili
del caso e $P(e)$ è la loro probabilità; nei livelli dei giocatori restano il
massimo e il minimo. Il costo sale a $O(b^m n^m)$, con $n$ il numero di esiti
distinti, e cambia una cosa sottile sulla valutazione: senza caso conta solo
l'ordine dei valori di $\mathrm{ev}$, e una trasformazione monotona non cambia
la mossa scelta; con il caso si fanno medie, contano le distanze, e
$\mathrm{ev}$ deve essere una trasformazione affine positiva della probabilità
di vittoria {cite}`russell2020artificial`.

Due complicazioni che i programmi seri devono affrontare, e sono i punti in cui
la teoria pulita si sporca:

- **l’effetto orizzonte**, cioè la tendenza a preferire mosse che rimandano un
  danno inevitabile oltre la profondità di taglio, pagandolo con un danno
  minore ma reale {cite}`russell2020artificial`. I rimedi sono due, e nessuno
  dei due lo elimina: la **ricerca di quiescenza**, che dove la posizione è
  «agitata» (catture in corso, scacchi) continua a scendere oltre il taglio
  finché non si stabilizza, e le **estensioni singolari**, che prolungano la
  ricerca lungo una mossa chiaramente migliore di tutte le altre, così che le
  mosse dilatorie non riescano a spingere il danno fuori vista;
- **le trasposizioni**. L’albero srotolato dall’algoritmo tratta come nuovi
  stati che sono lo stesso stato raggiunto per un ordine diverso di mosse.
  Tenere una tabella dei valori già calcolati, indicizzata sulla posizione,
  elimina il lavoro ripetuto, e agli scacchi permette di raddoppiare la
  profondità raggiungibile a parità di tempo {cite}`russell2020artificial`. È
  il momento in cui l’albero di ricerca torna a essere, come si diceva
  nell’apertura del capitolo, un grafo.

`````

```{figure} ../figures/alfabeta-pota.svg
:name: fig-alfabeta-pota
:alt: "Un albero a due livelli. In cima un pallino, chi muove per primo, che prende il massimo; sotto, tre pallini dell’avversario, che prendono il minimo; sotto ancora nove caselle con i numeri 3, 12, 8, poi 2, 4, 6, poi 14, 5, 2. Le caselle si scoprono da sinistra a destra. Scoperte le prime tre, il nodo sopra di esse segna 3, e in basso compare il 3 come guadagno già assicurato. Nel secondo gruppo si scopre soltanto il 2: le due caselle che restano e i loro rami diventano grigi e barrati, e il loro nodo segna «minore o uguale a 2», perché quel valore nessuno l’ha misurato fino in fondo. Il terzo gruppo si scopre tutto, 14, 5 e 2, e segna 2. Alla fine la radice segna 3, e la riga in basso conta sette foglie guardate su nove."
:width: 100%

La potatura mentre avviene. Le foglie si scoprono da sinistra; il numero in
basso è il migliore che si è già assicurato chi muove per primo. Appena in un
gruppo compare un valore che sta sotto quel numero, il resto del gruppo si
spegne: non serve guardarlo, perché a sceglierlo sarebbe l’avversario e
l’avversario prenderà comunque il minimo.
```

{numref}`fig-alfabeta-pota` fa vedere quando i due rami si spengono, cosa
che su un disegno fermo non si vedrebbe. E messa accanto alla funzione di
valutazione, la potatura è di natura opposta: spegne rami di cui si è
dimostrato che non possono cambiare la risposta, e non costa niente. La
funzione di valutazione, invece, sostituisce una risposta vera con un giudizio:
costa, e il prezzo si chiama effetto orizzonte.

## Quando il giudizio è esatto: il Nim

L’effetto orizzonte viene tutto da un punto: la funzione di valutazione è una
stima, e la ricerca si ferma proprio dove la stima sbaglia. Esistono però giochi
in cui il giudizio sulla posizione si può scrivere esatto, e dove lo si può
scrivere non resta niente da guardare oltre: basta scegliere, fra le proprie
mosse, quella che lascia all’avversario una posizione persa.

Il caso da manuale è il **Nim**. Sul tavolo stanno alcuni mucchi di oggetti; chi
muove sceglie un mucchio e ne toglie quanti vuole, almeno uno e al più tutto il
mucchio; chi prende l’ultimo vince. Nel 1901 il matematico americano Charles
Bouton ne pubblicò la teoria completa, e il nome lo propose lui: in qualche
college e in qualche fiera il gioco circolava come Fan-Tan, che però è il nome
di un altro gioco, cinese {cite}`bouton1901nim`. La teoria sta in una riga: si
scrivono le taglie dei mucchi in binario e se ne fa lo XOR, l’«o esclusivo»
della {doc}`sezione sul percettrone </RetiNeurali/percettrone>`, applicato
colonna per colonna; se il risultato è zero, chi deve muovere perde contro un
avversario che non sbaglia, e altrimenti vince.

`````{tab} Elementare

Sul tavolo ci sono tre mucchi di fiammiferi, da 3, da 5 e da 8. Il trucco di
Bouton sta in un modo di contare: ogni mucchio si spezza in pacchetti da 1, da
2, da 4 e da 8, con al più un pacchetto per misura. Il 3 è 2 + 1, il 5 è 4 + 1,
l’8 è un pacchetto da 8 e basta. Il modo di spezzare è uno solo, ed è la
scrittura in binario: ogni cifra dice se quel pacchetto c’è o no, e scrivendo i
numeri uno sotto l’altro ogni colonna è una misura.

Poi si contano i pacchetti misura per misura. Da 1 ce ne sono due (nel 3 e nel
5), da 2 uno, da 4 uno, da 8 uno. Una posizione è *in ordine* quando ogni misura
compare un numero pari di volte, e questa non lo è: restano spaiati il 2, il 4 e
l’8. Guardare colonna per colonna se il conto è pari o dispari è proprio lo XOR.

Il modo di giocare diventa allora una frase sola: lascia sempre il tavolo in
ordine. Si guarda il pacchetto spaiato più grosso, qui l’8, e si prende un
mucchio che lo contiene, qui l’unico, quello da 8. Quel mucchio si rifà da capo
passando in rassegna le misure spaiate: quelle che ha le perde, quelle che non
ha le riceve. Il mucchio da 8 perde l’8 e riceve il 4 e il 2, cioè diventa un
mucchio da 6: si tolgono due fiammiferi, e sul tavolo restano 3, 5 e 6, che
fanno 2 + 1, 4 + 1 e 4 + 2, ogni misura due volte.

La mossa si può fare sempre, perché il mucchio scelto perde il suo pacchetto
più grosso e riceve soltanto pacchetti più piccoli, e i più piccoli tutti
insieme non ci arrivano: 4 + 2 + 1 fa 7, meno di 8. Il mucchio scende, e
togliere fiammiferi è una mossa legale.

L’avversario, davanti a un tavolo in ordine, non può fare lo stesso. Qualunque
cosa tolga, tocca un mucchio solo, e quel mucchio cambia taglia, quindi cambia
almeno un pacchetto: una misura che c’era sparisce, o una che non c’era compare.
Negli altri mucchi quella misura è rimasta com’era, e il suo conto cambia di
uno: da pari diventa dispari. Allora tocca di nuovo a me, e rimetto in ordine.
Il tavolo vuoto è in ordine (zero pacchetti per ogni misura, e zero è pari), e
siccome in ordine lo lascio sempre io, l’ultimo fiammifero lo prendo io.

Per giocare così non serve immaginare nemmeno una risposta dell’avversario:
basta un’occhiata alla posizione. È un giudizio che non sbaglia mai, e dove il
giudizio non sbaglia non c’è nessun orizzonte oltre cui nascondere un disastro.
Vale però per le regole dette. Se si gioca che chi prende l’ultimo fiammifero
perde, la regola va corretta alla fine: quando restano solo mucchi da un
fiammifero, la posizione buona da lasciare è quella con un numero dispari di
mucchi.

`````

`````{tab} Superiore

Una posizione del Nim è una $r$-upla di interi non negativi
$(x_1, \dots, x_r)$, e una mossa sostituisce un solo $x_i$ con un
$x_i' < x_i$. Il gioco è finito, perché la somma delle taglie scende a ogni
mossa, e si gioca in *convenzione normale*: chi non ha mosse, cioè chi si
trova davanti a tutti zeri, perde. Le posizioni si dividono allora in due
classi, definite per induzione dal fondo: una posizione è P (vince il
giocatore *precedente*, quello che l’ha lasciata) se tutte le sue mosse portano
in posizioni N, ed è N (vince chi muove, il *next*) se almeno una mossa porta
in una posizione P. È minimax con utilità $\pm 1$, scritto per classi invece
che per valori.

Sia $t = x_1 \oplus x_2 \oplus \cdots \oplus x_r$ la *somma di Nim*, lo XOR
bit a bit delle taglie. Il teorema di Bouton {cite}`bouton1901nim` dice che la
posizione è P se e solo se $t = 0$, e la dimostrazione verifica le tre
proprietà che caratterizzano le posizioni P:

1. la posizione terminale $(0, \dots, 0)$ ha $t = 0$;
2. da $t = 0$ ogni mossa porta a $t' \neq 0$: sostituendo $x_i$ con $x_i'$ la
   somma diventa $t' = t \oplus x_i \oplus x_i'$, e $x_i \oplus x_i' \neq 0$
   perché $x_i' \neq x_i$;
3. da $t \neq 0$ esiste una mossa verso $t' = 0$. Sia $2^j$ il bit più alto
   di $t$: almeno un $x_i$ ha acceso il bit $j$, altrimenti nella colonna $j$
   gli uni sarebbero in numero pari. Il valore $x_i' = x_i \oplus t$ spegne il
   bit $j$ di $x_i$ e lascia invariati quelli più alti, quindi $x_i' < x_i$ e
   la mossa è legale; e $t' = t \oplus x_i \oplus x_i' = t \oplus t = 0$.

Per induzione sulla somma delle taglie, le posizioni con $t = 0$ sono
esattamente le P.

Letto con gli occhi della ricerca, il test $t \neq 0$ è una funzione di
valutazione *esatta*: in ogni stato dice quello che direbbe minimax, cioè se
vince chi muove, senza visitare niente. Con una valutazione esatta basta una
ricerca a profondità uno: si provano le $\sum_i x_i$ mosse e si tiene una che
porta a $t' = 0$, e l’effetto orizzonte non ha dove nascere. Il Nim permette
anche di meglio, perché la dimostrazione costruisce la mossa (il bit più alto di
$t$, un mucchio che lo ha acceso, $x_i \oplus t$) in $O(r \log \max_i x_i)$
operazioni sui bit, contro una visita di un albero che ha $\prod_i (x_i + 1)$
stati distinti e molti più nodi.

Il teorema vale nella convenzione normale. Nella variante *misère*, in cui chi
prende l’ultimo perde e che Bouton segnala come la più conosciuta delle due, la
classificazione cambia soltanto sulle posizioni in cui ogni mucchio ha al più
un oggetto: lì è P quella con un numero dispari di mucchi da uno, e non più
quella con un numero pari.

`````

Il conto rifà minimax su tutte le posizioni piccole e lo confronta con la
regola, poi misura su un caso solo che cosa la regola risparmia.

```python
from functools import lru_cache
from itertools import product


def xor(taglie):
    """La somma di Nim: lo XOR delle taglie, colonna binaria per colonna."""
    t = 0
    for x in taglie:
        t ^= x
    return t


@lru_cache(maxsize=None)
def vince_chi_muove(mucchi):
    """Minimax: vince chi ha una mossa che lascia l'altro in una posizione
    perdente."""
    for i, x in enumerate(mucchi):
        for resto in range(x):          # dal mucchio i si lasciano `resto`
            dopo = tuple(sorted(mucchi[:i] + (resto,) + mucchi[i + 1:]))
            if not vince_chi_muove(dopo):
                return True
    return False                        # senza mosse, o con tutte perdenti


posizioni = sorted({tuple(sorted(p)) for p in product(range(8), repeat=3)})
concordano = all(vince_chi_muove(p) == (xor(p) != 0) for p in posizioni)
print(f"{len(posizioni)} posizioni con tre mucchi fino a 7 oggetti: "
      f"lo XOR concorda con minimax in tutte: {concordano}")


def mossa_di_bouton(mucchi):
    """Il mucchio e quanti oggetti togliere perché lo XOR torni a zero."""
    t = xor(mucchi)
    for i, x in enumerate(mucchi):
        if x ^ t < x:                   # ha acceso il bit più alto di t
            return i, x - (x ^ t)
    return None                         # t = 0: nessuna mossa lo rimette


@lru_cache(maxsize=None)
def partite(mucchi):
    """Le partite diverse da qui alla fine, cioè le foglie che minimax
    visiterebbe senza potatura e senza memoria."""
    if not any(mucchi):
        return 1
    return sum(partite(tuple(sorted(mucchi[:i] + (r,) + mucchi[i + 1:])))
               for i, x in enumerate(mucchi) for r in range(x))


p = (3, 5, 8)
i, quanti = mossa_di_bouton(p)
dopo = p[:i] + (p[i] - quanti,) + p[i + 1:]
print(f"{p}: XOR {xor(p)}; si tolgono {quanti} oggetti dal mucchio "
      f"da {p[i]}, resta {dopo} con XOR {xor(dopo)}")
print(f"partite diverse da {p} alla fine: {partite(p)}")
```

```text
120 posizioni con tre mucchi fino a 7 oggetti: lo XOR concorda con minimax in tutte: True
(3, 5, 8): XOR 14; si tolgono 2 oggetti dal mucchio da 8, resta (3, 5, 6) con XOR 0
partite diverse da (3, 5, 8) alla fine: 51823082
```

Sulle centoventi posizioni con tre mucchi fino a sette oggetti la regola e
minimax dicono la stessa cosa. Da (3, 5, 8) la regola trova la mossa in un
colpo, due oggetti via dal mucchio da otto, e lascia (3, 5, 6) con XOR zero.
Senza la regola, l’albero che minimax attraverserebbe senza tagli e senza
memoria conta più di cinquantuno milioni di partite diverse, contro le
duecentocinquantacinquemila del tris, e sul tavolo ci sono sedici oggetti.

## Ogni gioco imparziale è un mucchio di Nim

Il Nim sembra un caso fortunato, un gioco con un trucco tutto suo. Un teorema
dimostrato in modo indipendente da Roland Sprague nel 1935-36 e da Patrick
Grundy nel 1939 dice il contrario: ogni gioco di una certa famiglia si comporta,
posizione per posizione, come un mucchio di Nim di una taglia che si sa
calcolare {cite}`sprague1936mathematische,grundy1939mathematics`. La famiglia è
quella dei giochi **imparziali**: le mosse possibili dipendono dalla posizione e
non da chi deve muovere, ogni partita finisce, e chi resta senza mosse perde.

`````{tab} Elementare

Un altro gioco: un mucchio solo, e a ogni turno se ne tolgono uno, due o tre
fiammiferi, non di più. Ogni taglia del mucchio riceve un’etichetta, e la
regola per scriverla è una: si guardano le etichette delle taglie in cui si può
andare, e si prende il numero più piccolo che fra loro manca.

Il mucchio vuoto non va da nessuna parte, e prende 0. Da 1 si va solo a 0,
quindi il più piccolo che manca è 1. Da 2 si va a 1 e a 0, e prende 2; da 3 si
va a 2, 1 e 0, e prende 3. Da 4 si va a 3, 2 e 1, e fra le destinazioni manca
lo 0: il 4 prende 0. Da lì la fila si ripete, 1, 2, 3, 0, 1, 2, 3, 0.

L’etichetta 0 vuol dire che chi deve muovere perde. Da un’etichetta 0 non si
arriva mai a un’altra etichetta 0 (se ci si arrivasse, lo 0 non mancherebbe), e
da un’etichetta diversa da 0 c’è sempre una mossa verso uno 0. L’etichetta 0 fa
il mestiere del tavolo in ordine: chi ci si trova davanti può solo uscirne, e
l’altro ce lo riporta.

L’etichetta dice anche di più: una taglia con etichetta 2 si comporta come un
mucchio di Nim da 2. Dal mucchio di Nim da 2 si scende a 1 o a 0, e non si può
restare a 2; dalla taglia con etichetta 2 si arriva a un’etichetta 1 e a
un’etichetta 0 (sono nella lista, se no il più piccolo che manca sarebbe più
basso), e mai a un’etichetta 2 (se ci fosse, il 2 non mancherebbe e l’etichetta
sarebbe un’altra). A volte si può anche salire: dal 6, che ha etichetta 2, si va
al 3, che ha etichetta 3. Ma chi sale non guadagna niente, perché una taglia con
etichetta più alta ha fra le sue destinazioni tutte le etichette più basse, 2
compreso: dal 3 l’avversario toglie un fiammifero e va al 2, che ha di nuovo
etichetta 2. Il mucchio non è più quello di prima, ma l’etichetta sì, ed è
l’etichetta che conta.

Allora due giochi affiancati sullo stesso tavolo, in cui a ogni turno si muove
in uno solo, si giudicano come due mucchi di Nim. Il mucchio «togli uno, due o
tre» da 6 fiammiferi ha etichetta 2, il mucchio di Nim da 2 ha etichetta 2, e
spezzate in pacchetti le due etichette fanno coppia: il tavolo è in ordine, e
chi deve muovere perde.

Le etichette si scrivono una volta sola, una per taglia, e da lì in poi
qualunque tavolo fatto di quei mucchi si giudica con i pacchetti, senza
esplorare le combinazioni. Su un mucchio da solo, però, scrivere l’etichetta
costa quanto guardare tutte le sue mosse: il risparmio arriva quando i mucchi
sono tanti.

Il trucco si ferma davanti al tris. Le etichette si scrivono guardando dove si
può andare, e nel tris dove si può andare dipende da chi muove: le caselle
libere sono le stesse, ma uno ci mette una croce e l’altro un cerchio. Un numero
solo non basta a dire due cose. E nel tris non perde chi resta senza mosse: si
vince allineando tre segni, e si può finire pari. Lì resta l’albero.

`````

`````{tab} Superiore

Un gioco è imparziale se l’insieme delle mosse $\mathcal{A}(s)$ dipende solo
dallo stato $s$ e non dal giocatore che muove; lo si suppone inoltre finito
(ogni partita termina) e giocato in convenzione normale. Il **numero di
Grundy** di uno stato è definito per induzione dal fondo:

$$
g(s) = \operatorname{mex}\,\{\, g(\mathrm{ris}(s,a)) : a \in \mathcal{A}(s) \,\},
$$

dove $\operatorname{mex} S$ (*minimum excludant*) è il più piccolo intero non
negativo che non sta in $S$; negli stati terminali $\mathcal{A}(s)$ è vuoto e
$g(s) = 0$. Ne seguono tre fatti.

- $g(s) = 0$ se e solo se $s$ è una posizione P. Da $g = 0$ nessuna mossa
  porta a $g = 0$, da $g \neq 0$ almeno una sì (lo $0$ sta fra i valori
  raggiunti), e la posizione terminale ha $g = 0$: sono le tre proprietà del
  teorema di Bouton.
- Uno stato con $g(s) = n$ è equivalente al mucchio di Nim da $n$, che si
  indica con $*n$: in qualunque somma lo si può sostituire con $*n$ senza
  cambiare chi vince. L’idea della dimostrazione sta in due proprietà. Le
  mosse di $s$ raggiungono ogni valore $0, \dots, n-1$ e nessuna raggiunge
  $n$, come quelle di $*n$; e le mosse verso valori maggiori di $n$ sono
  *reversibili*, perché da uno stato $s'$ con $g(s') > n$ l’avversario ha una
  mossa verso uno stato di valore $n$ (il mex di $s'$ supera $n$, quindi $n$
  sta fra i suoi valori raggiunti).
- Nella *somma disgiuntiva* $G + H$ si muove, a ogni turno, in esattamente una
  delle due componenti, e vale $g(G + H) = g(G) \oplus g(H)$: è il teorema di
  Sprague-Grundy {cite}`sprague1936mathematische,grundy1939mathematics`. Il Nim
  a $r$ mucchi è la somma di $r$ mucchi singoli, e la regola di Bouton ne è il
  caso particolare.

Nel gioco della sottrazione con mosse $\{1, 2, 3\}$ la ricorsione dà
$g(n) = n \bmod 4$, e con mosse $\{1, \dots, q\}$ dà $n \bmod (q + 1)$.

Il guadagno va letto con precisione. Calcolare $g$ su una componente vuol dire
visitarne il grafo degli stati, come farebbe minimax con una tabella delle
trasposizioni: su un gioco che non si spezza in parti indipendenti il teorema
non fa risparmiare niente. Il risparmio sta nella somma. Con $r$ componenti da
$N$ stati ciascuna lo spazio degli stati della somma ha $N^r$ elementi, mentre i
numeri di Grundy da calcolare sono al più $rN$ (e appena $N$ se le componenti
sono copie dello stesso gioco), e lo XOR ricompone il valore.

Il teorema cade per i giochi *partigiani*, in cui $\mathcal{A}(s)$ dipende da
chi muove: il tris, gli scacchi, il go. Quando un gioco partigiano si spezza in
somme, come Hackenbush o Domineering, esiste ancora una teoria, quella di Conway
sistemata in *Winning Ways* {cite}`berlekamp1982winning`, ma i valori non sono
più soltanto mucchi di Nim e non si compongono con lo XOR. Il tris e gli scacchi
non si spezzano, e al tris manca anche la convenzione normale (la partita
finisce con un allineamento o in parità, non quando le mosse finiscono): per
loro resta l’albero.

`````

Il conto costruisce la fila delle etichette con la ricorsione del mex, poi
mette un mucchio «togli 1, 2 o 3» accanto a un mucchio di Nim e confronta lo
XOR dei due numeri con minimax sulla coppia.

```python
def mex(valori):
    """Il più piccolo intero non negativo che non compare fra i valori."""
    n = 0
    while n in valori:
        n += 1
    return n


MOSSE = (1, 2, 3)                       # quanti se ne possono togliere
grundy = []
for n in range(13):
    grundy.append(mex({grundy[n - k] for k in MOSSE if n - k >= 0}))
print("numeri di Grundy di «togli 1, 2 o 3», taglie da 0 a 12:", grundy)


@lru_cache(maxsize=None)
def vince_affiancati(n, m):
    """Minimax su due giochi affiancati: un mucchio «togli 1, 2 o 3» da n e
    un mucchio di Nim da m. A ogni turno si muove in uno solo dei due."""
    dopo = [(n - k, m) for k in MOSSE if n - k >= 0]
    dopo += [(n, r) for r in range(m)]
    return any(not vince_affiancati(*d) for d in dopo)


concordano = all(vince_affiancati(n, m) == (grundy[n] ^ m != 0)
                 for n in range(13) for m in range(13))
print(f"{13 * 13} coppie: lo XOR dei numeri di Grundy concorda con minimax "
      f"in tutte: {concordano}")
print(f"mucchio da 6 accanto a Nim da 2: numeri {grundy[6]} e 2, "
      f"XOR {grundy[6] ^ 2}, vince chi muove: {vince_affiancati(6, 2)}")
print(f"dieci mucchi «togli 1, 2 o 3» fino a 12: {13 ** 10} posizioni, "
      f"{len(grundy)} numeri di Grundy da calcolare")
```

```text
numeri di Grundy di «togli 1, 2 o 3», taglie da 0 a 12: [0, 1, 2, 3, 0, 1, 2, 3, 0, 1, 2, 3, 0]
169 coppie: lo XOR dei numeri di Grundy concorda con minimax in tutte: True
mucchio da 6 accanto a Nim da 2: numeri 2 e 2, XOR 0, vince chi muove: False
dieci mucchi «togli 1, 2 o 3» fino a 12: 137858491849 posizioni, 13 numeri di Grundy da calcolare
```

La fila ripete 0, 1, 2, 3, e sulle centosessantanove coppie lo XOR dei due
numeri dice chi vince esattamente come minimax. Il mucchio da sei e il mucchio
di Nim da due hanno lo stesso numero, lo XOR è zero e chi muove perde.
L’ultima riga è il guadagno della somma: dieci mucchi dello stesso gioco fanno
più di centotrentasette miliardi di posizioni, e per giudicarle tutte bastano
tredici numeri e uno XOR. Il tris e gli scacchi restano fuori da questa
famiglia, e per loro resta tutto quello che si è visto fin qui: l’albero, la
potatura, il giudizio a occhio con il suo orizzonte.

`````{tab} Elementare

```{admonition} Da ricordare
:class: important
- Con un avversario davanti, metà delle mosse le sceglie lui, e le sceglie
  per farci del male. Il valore di una posizione non è il numero più alto che
  ci si vede sotto: è quello che si ottiene supponendo che da lì in poi
  giochino bene tutti e due.
- Il conto si fa all’indietro, dalle foglie alla radice, alternando «prendi
  il massimo» dove tocca a me e «prendi il minimo» dove tocca a lui.
- La potatura è la frase «questa strada è già peggio della migliore che ho
  trovato, non la guardo nemmeno». Non è un’approssimazione: la risposta è la
  stessa, e sul tris costa quasi trentacinque volte meno.
- Quanto si pota dipende dall’ordine in cui si guardano le mosse: con la
  migliore per prima si scarta quasi tutto, con la migliore per ultima quasi
  niente.
- Nelle partite vere il fondo non si raggiunge, quindi ci si ferma a una certa
  profondità e si giudica a occhio la posizione. Questo sì che costa, e il
  prezzo si chiama effetto orizzonte: il disastro che sta un passo oltre
  l’ultimo che si è guardato non si vede, e conviene perfino spingercelo
  pagando qualcosa. Un rimedio è non fermarsi dove i pezzi si stanno ancora
  mangiando: l’orizzonte si sposta dove fa meno danni, e sparire non sparisce.
- Nel Nim il giudizio è esatto. Si spezzano i mucchi in pacchetti da 1, 2, 4,
  8: se ogni misura compare un numero pari di volte chi deve muovere perde, e
  chi lascia sempre il tavolo così vince senza guardare avanti.
- Ogni gioco in cui le mosse non dipendono da chi le fa si comporta come un
  mucchio di Nim, e la taglia di quel mucchio si scrive partendo dalla fine: è
  il numero più piccolo che manca fra quelli delle posizioni raggiungibili. Il
  risparmio arriva quando i mucchi sono tanti; nel tris, dove le mosse
  dipendono da chi muove, il trucco non vale.
```

`````

`````{tab} Superiore

```{admonition} Da ricordare
:class: important
- Minimax definisce il valore di uno stato per ricorsione, alternando
  massimo e minimo, e restituisce il valore esatto dato l’albero completo.
  Costa $O(b^m)$, cioè è impraticabile su un gioco vero.
- Alfa-beta {cite}`knuth1975analysis` porta lungo il cammino i due limiti
  $\alpha$ e $\beta$ e taglia i rami che non possono influire. Restituisce lo
  stesso valore di minimax alla radice: nel caso migliore $O(b^{m/2})$, cioè
  ramificazione effettiva $\sqrt{b}$ (agli scacchi 6 invece di 35, ossia il
  doppio della profondità a parità di tempo); con ordinamento casuale e $b$
  moderati, circa $O(b^{3m/4})$.
- L’ordinamento delle mosse è quindi parte dell’algoritmo: killer move e
  approfondimento iterativo usato come ordinatore.
- Non potendo raggiungere le foglie si sostituisce $u$ con una funzione di
  valutazione e il test di fine con un test di taglio. Qui la ricerca smette
  di essere esatta, e compare l’effetto orizzonte, che la ricerca di
  quiescenza e le estensioni singolari attenuano senza eliminare.
- Le trasposizioni riportano l’albero al grafo che era: una tabella dei
  valori già calcolati raddoppia, agli scacchi, la profondità raggiungibile
  {cite}`russell2020artificial`.
- Nel Nim la posizione è P se e solo se $x_1 \oplus \cdots \oplus x_r = 0$
  {cite}`bouton1901nim`: una funzione di valutazione esatta, con cui basta
  una ricerca a profondità uno.
- Per un gioco imparziale, finito e in convenzione normale,
  $g(s) = \operatorname{mex}\{g(\mathrm{ris}(s,a)) : a \in \mathcal{A}(s)\}$,
  $g = 0$ esattamente sulle posizioni P, e $g(G + H) = g(G) \oplus g(H)$
  (Sprague-Grundy). Il risparmio sta nelle somme, non nella singola
  componente, e il teorema non vale per i giochi partigiani.
```

`````

Fin qui il mondo è stato generoso in tre modi: ci ha lasciato interrogare le
regole quante volte volevamo, ci ha detto quando eravamo arrivati, e ci ha
permesso di scrivere un giudizio sulle posizioni intermedie. La sezione che
chiude il capitolo toglie quelle tre cose una per volta, e guarda che cosa
resta in piedi.
