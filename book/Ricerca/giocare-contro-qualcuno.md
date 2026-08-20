# Giocare contro qualcuno: minimax, potatura, orizzonte

Nel labirinto della sezione precedente i corridoi stanno fermi. Se ne provo uno
e non porta da nessuna parte, il labirinto non si riorganizza per dispetto.

Con un avversario davanti cambia tutto, e cambia in un punto solo: **metà delle
mosse non le scelgo io**. L’albero è lo stesso, i rami sono gli stessi, ma un
livello sì e uno no li sceglie qualcuno che vuole esattamente il contrario di
quello che voglio io. Non posso più chiedermi «qual è la strada migliore»: devo
chiedermi «qual è la mossa che regge anche alla risposta peggiore».

## Ragionare all’indietro dalla fine

Il modo di rispondere si chiama **minimax**, e il nome dice già tutto: c’è un
punteggio solo sul tavolo, e uno dei due giocatori cerca di farlo salire mentre
l’altro cerca di farlo scendere. Un punto guadagnato da me è un punto perso da
lui, esattamente: non esiste una mossa che convenga a tutti e due.

`````{tab} Elementare

Facciamo il conto su un albero piccolissimo, di due mosse soltanto: prima muovo
io, poi muove lui, e a quel punto la partita è finita e si legge il punteggio.
Punteggi alti vuol dire bene per me.

Ho tre mosse. Se gioco la prima, lui può rispondere in tre modi, che portano a
3, 12 e 8 punti. Se gioco la seconda, le sue tre risposte portano a 2, 4 e 6.
Se gioco la terza, portano a 14, 5 e 2.

Adesso l’istinto sbagliato: «gioco la terza, che porta a 14». No. Il 14 non lo
sceglierei io, lo sceglierebbe **lui**, e lui vuole il numero più piccolo:
davanti alla mia terza mossa risponderebbe con il 2. Quindi la mia terza mossa
non vale 14, vale 2.

Rifacciamo il conto come va fatto, dal basso. La prima mossa vale il minimo fra
3, 12 e 8, cioè **3**. La seconda vale il minimo fra 2, 4 e 6, cioè **2**. La
terza vale il minimo fra 14, 5 e 2, cioè **2**. E adesso tocca a me, che voglio
il massimo: fra 3, 2 e 2 scelgo il **3**, cioè la prima mossa.

Il gesto è tutto qui, e si chiama ragionare all’indietro: **il valore di una
posizione non è quello che ci si vede sopra, è quello che ci si vede in fondo,
supponendo che da lì in poi giochino bene tutti e due**. E si costruisce
partendo dalle foglie e risalendo, un livello alla volta, alternando «prendi il
massimo» e «prendi il minimo».

Il punto di rottura, che l’ultima parte di questa sezione andrà a raccogliere:
tutto questo presuppone di **arrivare in fondo**. Nel nostro alberello la
partita finiva dopo due mosse e il punteggio era scritto. In una partita vera il fondo non si raggiunge mai, e
quello che si fa al suo posto è il resto della sezione.

`````

`````{tab} Superiore

Per un gioco a due giocatori, **deterministico**, a somma zero e a
**informazione perfetta** (cioè in cui ciascuno vede tutta la posizione: è
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
segue. Il valore così definito è quello che si ottiene **se entrambi giocano in
modo ottimo** da lì alla fine, ed è un’ipotesi forte che vale la pena tenere
presente: contro un avversario che sbaglia, minimax non è la strategia che
sfrutta di più i suoi errori.

L’algoritmo è una visita in profondità che scende fino alle foglie e risale
combinando. Costa $O(b^m)$ in tempo, con $m$ la profondità dell’albero, e
$O(bm)$ in memoria. Su un gioco vero è impraticabile per lo stesso conto
dell’apertura del capitolo: agli scacchi $35^{80}$.

Va notato che minimax **non è un’euristica e non è un’approssimazione**: dato
l’albero completo, il valore che restituisce è esatto. Tutto ciò che segue in
questa sezione serve a non costruire quell’albero, e si divide in due mestieri
distinti che conviene non confondere: **calcolare lo stesso valore guardando
meno** (la potatura, che non perde niente) e **calcolare un valore diverso
perché quello vero è fuori portata** (la funzione di valutazione, che perde
eccome).

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
attenzione, perché è il conto che serve: non sono duecentocinquantacinquemila
*posizioni diverse* (di quelle un gioco da nove caselle ne ha molte meno), sono
duecentocinquantacinquemila **partite intere**, giocate una per una dalla prima
mossa all’ultima. È l’albero dell’apertura del capitolo in miniatura: piccolo
abbastanza da srotolarlo tutto, e già abbastanza grande da far vedere il
problema.

## Smettere di guardare: la potatura

C’è un modo di ottenere **esattamente lo stesso numero** srotolando una
frazione di quelle partite, e non è un’approssimazione: è accorgersi che certi
rami, qualunque cosa contengano, non possono cambiare la risposta.

`````{tab} Elementare

Torniamo all’alberello di prima, e stavolta guardiamo le foglie una per volta,
da sinistra, come farebbe qualcuno che le scopre a mano a mano.

Della mia **prima** mossa scopro 3, 12, 8: lui sceglierebbe il minimo, quindi
quella mossa vale 3. Adesso so una cosa che non mollo più: **qualunque cosa
succeda, non accetterò meno di 3**.

Passo alla mia **seconda** mossa. Scopro la prima risposta di lui: 2. E qui mi
fermo, perché ho già finito di ragionare. Lui, su questa mossa, prenderà il
minimo fra 2 e le altre due che non ho ancora guardato: quindi al massimo
prenderà 2, e forse meno. Comunque vada, questa mossa non vale più di 2, cioè
**meno del 3 che ho già in tasca**. Le altre due risposte non le guardo
nemmeno: non c’è nessun numero che possano contenere capace di farmi cambiare
idea. Anche se ci fosse un milione, lui non me lo lascerebbe prendere.

Passo alla **terza**. Scopro 14: non basta a decidere, perché lui prenderà il
minimo e potrebbe esserci di peggio. Scopro 5: idem. Scopro 2: adesso so che
questa mossa vale 2, meno di 3. Anche questa scartata.

Risposta finale: la prima mossa, che vale 3. La stessa di prima. E ho guardato
sette foglie su nove.

Il gesto ha un nome che si spiega da sé: **potatura**, come i rami che si
tagliano a un albero. E la frase che la produce è una sola, quella che si dice
a se stessi guardando la seconda mossa: «questa strada è già peggio della
migliore che ho trovato, non la guardo nemmeno».

Il punto di rottura, che conta moltissimo in pratica: quanto si pota dipende
**dall’ordine in cui si guardano le mosse**. Se la mossa buona capita per prima,
tutte le altre si scartano in fretta perché c’è già un metro alto da superare;
se capita per ultima, il metro resta basso a lungo e non si scarta quasi
niente. Lo stesso algoritmo, sullo stesso albero, può guardare pochissimo o
quasi tutto a seconda dell’ordine.

`````

`````{tab} Superiore

La potatura **alfa-beta** {cite}`knuth1975analysis` porta lungo la ricorsione
due valori: $\alpha$, il migliore che chi massimizza si è già assicurato lungo
il cammino corrente, e $\beta$, il migliore per chi minimizza. La regola è
simmetrica: in un nodo di massimo si interrompe l’esplorazione dei figli non
appena il valore corrente arriva a $\beta$ o lo supera; in un nodo di minimo,
non appena scende ad $\alpha$ o sotto. Sono le due righe `if v >= beta` e
`if v <= alfa` del codice più sotto, e la parità conta: con la disuguaglianza
stretta il taglio scatterebbe meno spesso e il conteggio cambierebbe.

La correttezza si vede con un conto di tre righe sull’albero d’esempio, quello
con foglie $3, 12, 8$ sotto la prima mossa, $2, 4, 6$ sotto la seconda e
$14, 5, 2$ sotto la terza. Chiamando $x$ e $y$ le due foglie del secondo ramo
che non vengono esaminate, il valore alla radice è

$$
\max\big(\min(3,12,8),\ \min(2,x,y),\ \min(14,5,2)\big)
= \max\big(3,\ z,\ 2\big), \qquad z = \min(2,x,y) \le 2,
$$

e siccome $z \le 2 < 3$ il massimo vale 3 **indipendentemente da $x$ e $y$**.
Non è un’approssimazione: alfa-beta restituisce sempre lo stesso valore di
minimax alla radice.

Il guadagno dipende dall’ordinamento delle mosse. Nel caso migliore, cioè
esaminando per prima la mossa migliore in ogni nodo, alfa-beta esamina
$O(b^{m/2})$ nodi invece di $O(b^m)$: il fattore di ramificazione **effettivo**
diventa $\sqrt{b}$, che agli scacchi vuol dire circa 6 invece di 35, ossia la
possibilità di guardare **il doppio più a fondo** nello stesso tempo. Con
ordinamento casuale, e per $b$ moderati, si scende a circa $O(b^{3m/4})$
{cite}`russell2020artificial`.

Da qui il fatto che nei programmi di gioco l’ordinamento delle mosse non è una
rifinitura ma una parte dell’algoritmo. Due tecniche classiche: provare per
prime, in un nodo, le mosse che hanno già prodotto un taglio **alla stessa
profondità** in un altro ramo dell’albero (le **killer move**: se una mossa ha
confutato una linea, spesso ne confuta anche una parallela), e usare
l’**approfondimento iterativo** della sezione precedente non
solo per gestire il tempo, ma per ordinare: si cerca a profondità uno, si
ordinano le mosse secondo quel risultato, si cerca a profondità due partendo da
quell’ordine, e così via. Il tempo speso nelle passate superficiali si ripaga
con gli interessi in quelle profonde.

`````

Il conto sul filetto si rifà identico, cambiando solo la funzione.

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
altri risparmi di questo capitolo: **non si è rinunciato a niente**. I rami
non guardati erano rami di cui si era dimostrato, senza guardarli, che non
potevano cambiare la conclusione.

E l'ordine? Sul filetto si può misurare: basta
scandire le caselle in un ordine diverso, il che non cambia il gioco di una
virgola.

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

print(f"in ordine di casella (quello di prima): {con_ordine(range(9)):6d}")
print(f"centro e angoli per primi:              {con_ordine([4,0,2,6,8,1,3,5,7]):6d}")
print(f"bordi per primi:                        {con_ordine([1,3,5,7,0,2,6,8,4]):6d}")
print(f"venti ordini a caso: da {min(a_caso)} a {max(a_caso)}")
```

```text
in ordine di casella (quello di prima):   7330
centro e angoli per primi:                2893
bordi per primi:                         17002
venti ordini a caso: da 2603 a 13358
```

Sei volte fra il migliore e il peggiore, **sullo stesso gioco, con lo stesso
algoritmo e con la stessa risposta in fondo**. E l'ordine ragionato non è lontano dal migliore che si trovi a tentativi:
mettere per primi il centro e gli angoli vuol dire provare per prime le
caselle che nel filetto contano di più, e su venti ordini presi a caso uno
solo fa meglio, cioè fare esattamente quello che un
programma di scacchi fa quando ordina le mosse con una passata superficiale.
Il conto della potatura, insomma, non si fa una volta per tutte: si fa
sull’ordine che si è scelto.

## Quando il fondo non si raggiunge

Nel filetto la partita finisce, e il punteggio in fondo c’è scritto. Agli
scacchi no: dopo dieci mosse per parte si è ancora in mezzo alla partita, e in
fondo all’albero non c’è nessun numero da leggere.

Allora si fa la cosa che un giocatore umano fa da sempre: si guarda avanti
finché si può, ci si ferma, e si **giudica a occhio** la posizione a cui si è
arrivati. Quel giudizio è una **funzione di valutazione**, e prende il posto
del punteggio vero. È qui che la ricerca smette di essere esatta.

`````{tab} Elementare

Un giocatore di scacchi, guardando una posizione a metà partita, sa dire in
pochi secondi chi sta meglio: guarda quanti pezzi ha lui e quanti ne ha
l’avversario (e non tutti valgono uguale: una torre vale più di un alfiere),
guarda se il re è al riparo o esposto, se i pedoni si difendono a vicenda, chi
tiene le caselle in mezzo alla scacchiera, che sono quelle da cui si raggiunge
tutto il resto in fretta. Non è un calcolo, è un giudizio, ed è approssimativo:
due maestri possono valutare la stessa posizione in modo diverso.

Il programma fa la stessa cosa, ma con una formula, e la formula è meno
misteriosa di quel che sembra: è una somma. Si dà un peso a ciascuna di quelle
cose (tanti punti per ogni pezzo, secondo quanto vale; qualche punto per il re
al sicuro; qualche punto per ogni casella centrale controllata), si sommano e
viene fuori un numero. Il ragionamento allora diventa: guardo avanti quattro,
sei, dieci mosse, e alle posizioni a cui arrivo do quel voto invece di
continuare. Poi ragiono all’indietro come prima, ma partendo da quei voti
invece che dai punteggi veri.

Il difetto di questa mossa è preciso, ha un nome ed è **l’effetto orizzonte**.
Immagina una posizione in cui il mio alfiere è spacciato: qualunque cosa io
faccia, fra sei mosse me lo prendono. Se io guardo avanti solo cinque mosse,
quella perdita non la vedo. E allora scopro una cosa terribile: se do qualche
scacco inutile al suo re, sacrificando un pedone per volta, la cattura
dell’alfiere si sposta a sette mosse, a otto, a nove, cioè **oltre il mio
orizzonte**. Il programma guarda, non vede più la perdita dell’alfiere, e
conclude che sacrificare pedoni è una buona idea.

Non è un errore di programmazione: è quello che succede quando si giudica il
mondo a una distanza fissa. Il disastro non è stato evitato, è stato spinto
appena oltre il punto in cui si smette di guardare, e in cambio si è pagato
davvero. E ricompare ben oltre gli scacchi, ogni volta che qualcuno decide guardando a
una scadenza fissa: è il difetto da portarsi
dietro, perché ricompare ogni volta che qualcuno ottimizza guardando a una
scadenza fissa.

`````

`````{tab} Superiore

Si sostituisce l’utilità terminale $u(s)$ con una **valutazione** $\mathrm{ev}(s)$
e il test di terminazione con un **test di taglio**, ottenendo il minimax
euristico

dove $d$ è la profondità a cui si è scesi (attenzione: **non** è il $d$ della
sezione precedente, che era la profondità della soluzione; qui è un contatore
della ricorsione) e il **taglio** scatta quando $d$ raggiunge il limite fissato
o la posizione è comunque terminale.

$$
\mathrm{h\text{-}minimax}(s, d) =
\begin{cases}
\mathrm{ev}(s) & \text{se il taglio scatta in } (s,d),\\[2pt]
\max_a \mathrm{h\text{-}minimax}(\mathrm{ris}(s,a),\, d+1) & \text{se tocca a chi massimizza},\\[2pt]
\min_a \mathrm{h\text{-}minimax}(\mathrm{ris}(s,a),\, d+1) & \text{se tocca a chi minimizza}.
\end{cases}
$$

Perché $\mathrm{ev}$ sia utile deve concordare con $u$ sugli stati terminali,
essere calcolabile in fretta, e correlare con la probabilità di vittoria. In
pratica è quasi sempre una somma pesata di caratteristiche della posizione, il
che assume implicitamente che i loro contributi siano indipendenti: un’ipotesi
falsa (il valore di un alfiere dipende da com’è la struttura pedonale) e utile
lo stesso.

Due complicazioni che i programmi seri devono affrontare, e che vale la pena
nominare perché sono i punti in cui la teoria pulita si sporca:

- **l’effetto orizzonte**, cioè la tendenza a preferire mosse che rimandano un
  danno inevitabile oltre la profondità di taglio, pagandolo con un danno
  minore ma reale {cite}`russell2020artificial`. Il rimedio parziale si chiama
  ricerca di quiescenza: dove la posizione è «agitata» (catture in corso,
  scacchi) si continua a scendere oltre il taglio finché non si stabilizza;
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

{numref}`fig-alfabeta-pota` mostra in movimento il primo dei due risparmi di
questa sezione, e la ragione per guardarlo è che si veda **quando** i due rami
si spengono, cosa che su un disegno fermo non si vedrebbe. Messi uno accanto
all’altro, i due risparmi sono di natura opposta, ed è la cosa da tenere di
tutta la sezione. **La potatura spegne rami di cui si è dimostrato che non
possono cambiare la risposta: non costa niente.** La funzione di valutazione,
invece, sostituisce una risposta vera con un giudizio: costa, e il prezzo si
chiama effetto orizzonte.

`````{tab} Elementare

```{admonition} Da ricordare
:class: important
- Con un avversario davanti, **metà delle mosse le sceglie lui**, e le sceglie
  per farci del male. Il valore di una posizione non è il numero più alto che
  ci si vede sotto: è quello che si ottiene supponendo che da lì in poi
  giochino bene tutti e due.
- Il conto si fa **all’indietro**, dalle foglie alla radice, alternando «prendi
  il massimo» dove tocca a me e «prendi il minimo» dove tocca a lui.
- La **potatura** è la frase «questa strada è già peggio della migliore che ho
  trovato, non la guardo nemmeno». Non è un’approssimazione: la risposta è la
  stessa, e sul filetto costa quasi trentacinque volte meno.
- Quanto si pota dipende **dall’ordine in cui si guardano le mosse**: con la
  migliore per prima si scarta quasi tutto, con la migliore per ultima quasi
  niente.
- Nelle partite vere il fondo non si raggiunge, quindi ci si ferma a una certa
  profondità e si **giudica a occhio** la posizione. Questo sì che costa, e il
  prezzo si chiama **effetto orizzonte**: il disastro che sta un passo oltre
  l’ultimo che si è guardato non si vede, e conviene perfino spingercelo
  pagando qualcosa.
```

`````

`````{tab} Superiore

```{admonition} Da ricordare
:class: important
- **Minimax** definisce il valore di uno stato per ricorsione, alternando
  massimo e minimo, e restituisce il valore esatto **dato l’albero completo**.
  Costa $O(b^m)$, cioè è impraticabile su un gioco vero.
- **Alfa-beta** {cite}`knuth1975analysis` porta lungo il cammino i due limiti
  $\alpha$ e $\beta$ e taglia i rami che non possono influire. Restituisce lo
  stesso valore di minimax alla radice: nel caso migliore $O(b^{m/2})$, cioè
  ramificazione effettiva $\sqrt{b}$ (agli scacchi 6 invece di 35, ossia il
  doppio della profondità a parità di tempo); con ordinamento casuale e $b$
  moderati, circa $O(b^{3m/4})$.
- L’ordinamento delle mosse è quindi parte dell’algoritmo: **killer move** e
  approfondimento iterativo usato come ordinatore.
- Non potendo raggiungere le foglie si sostituisce $u$ con una **funzione di
  valutazione** e il test di fine con un test di taglio. Qui la ricerca smette
  di essere esatta, e compare l’**effetto orizzonte**, che la ricerca di
  quiescenza attenua senza eliminare.
- Le **trasposizioni** riportano l’albero al grafo che era: una tabella dei
  valori già calcolati raddoppia, agli scacchi, la profondità raggiungibile
  {cite}`russell2020artificial`.
```

`````

Fin qui il mondo è stato generoso in tre modi, e nessuno dei tre ci è stato
chiesto: ci ha lasciato interrogare le regole quante volte volevamo, ci ha
detto quando eravamo arrivati, e ci ha permesso di scrivere un giudizio sulle
posizioni intermedie. La sezione che chiude il capitolo toglie quelle tre cose
una per volta, e guarda che cosa resta in piedi.
