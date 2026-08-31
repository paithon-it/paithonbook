# Cercare senza avversari: dal tentoni all’euristica

Chiedi al telefono la strada per una città a trecento chilometri e la risposta
arriva in un istante. Eppure fra qui e là ci sono milioni di incroci, e le
strade che li collegano sono ancora di più: se il telefono le provasse tutte,
non finirebbe entro sera.

Non le prova tutte, e non perché sia veloce: perché **guarda quasi solo nella
direzione giusta**. Questa sezione racconta come si fa, e la racconta partendo
dal caso in cui nessuno rema contro, cioè da un mondo che non ha avversari e
in cui l’unica difficoltà è la dimensione. L’avversario arriva nella sezione
dopo, e cambia le regole.

## Cercare a tentoni: due modi, e i loro difetti

Prima di guardare nella direzione giusta bisogna sapere che cosa costa non
guardarci. I due modi di esplorare un albero senza sapere niente si chiamano
**in ampiezza** e **in profondità**, e la differenza sta tutta nell’ordine in
cui si aprono i rami.

`````{tab} Elementare

Sei in un labirinto e cerchi l’uscita. Hai due strategie, e sono opposte.

**In ampiezza**: fai un passo in ogni corridoio, poi torni indietro e fai il
secondo passo in ogni corridoio, poi il terzo. È come se il labirinto si
allagasse a partire da dove sei, e l’acqua avanzasse di un metro alla volta
dappertutto. Il vantaggio è enorme: quando l’acqua tocca l’uscita, sei sicuro
che quella è la strada **più corta**, perché niente ha potuto arrivarci prima.
Lo svantaggio pure: per allagare devi ricordarti tutti i punti bagnati, e sono
tantissimi.

**In profondità**: scegli un corridoio e lo segui fino in fondo; se finisce nel
muro torni all’ultimo bivio e prendi l’altro. È come tenere un filo srotolato
dietro di sé: non devi ricordarti tutti i punti dove sei stato, solo il filo
che hai alle spalle, e un filo lungo quanto sei sceso in profondità costa
pochissimo rispetto ad allagare. In cambio, due difetti. La
strada che trovi può essere ridicola, e non c’è nessun limite a quanto: ti
fermi alla prima uscita che incontri, non alla più vicina, e se il primo
corridoio gira per mezzo edificio prima di sbucare, quella è la tua strada. E
se un corridoio prosegue e prosegue senza mai finire né chiudersi, tu lo segui
e basta: non hai nessun motivo per tornare indietro, non ci torni mai, e il
filo che ti stavi srotolando dietro cresce con te.

Lo stesso ti capita se quel corridoio gira in tondo e ti riporta a un bivio
dove eri già stato: il filo non te lo dice, e tu ci giri dentro per sempre.
Contro i giri in tondo basta un gesso: segni ogni bivio dove metti piede, e
appena ritrovi un segno torni indietro. L’uscita, se c’è, adesso la trovi; solo
che i segni te li devi ricordare tutti, e la memoria che il filo ti faceva
risparmiare hai ricominciato a pagarla.

Le due strategie si possono anche sposare. Giri col filo, ma con un tetto:
dieci metri, non uno di più, e se l’uscita non salta fuori torni all’inizio e
rifai tutto con venti, poi con trenta. Rifare ogni volta i primi metri sembra
uno spreco, e non lo è, perché i bivi vicini all’inizio sono pochissimi
rispetto a quelli lontani: se da ogni bivio partono dieci corridoi, ogni giro
costa dieci volte quello prima, e tutti i giri già fatti messi insieme valgono
poco più di un decimo dell’ultimo. Rifarli, in tutto, costa circa l’undici per
cento di lavoro in più. In cambio ti tieni la memoria del filo e la garanzia
dell’acqua: l’uscita che trovi è la più vicina, perché a ogni giro ti sei
fermato al tetto.

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
è quello che fa il codice più sotto) la ricerca in profondità diventa completa
su spazi finiti, perché i cicli si riconoscono, ma si paga la memoria che si
era risparmiata. La riga che decide è quella
della memoria: la ricerca in ampiezza tiene in memoria un intero livello, e un
livello cresce come $b^d$. È il vincolo che morde per primo, molto prima del
tempo.

Le due si sposano nell’**approfondimento iterativo**: si fa una ricerca in
profondità con un tetto di un passo, poi di due, poi di tre, fino a trovare la
soluzione. Sembra uno spreco, perché i livelli alti si rigenerano ogni volta;
non lo è, e il conto lo spiega. In un albero che si moltiplica per $b$ a ogni
livello, i nodi dell’ultimo livello sono la maggioranza schiacciante di tutti:
i figli della radice si rigenerano $d$ volte, i loro figli $d-1$, e quelli
dell’ultimo livello una volta sola. In tutto si paga un fattore $b/(b-1)$
rispetto a generarli una volta sola. Con $b = 10$ vuol dire l’undici per cento in più di lavoro, in
cambio della memoria della profondità e della garanzia dell’ampiezza. Quando lo
spazio degli stati non entra in memoria e non si sa quanto sia lontana la
soluzione, è la scelta di riferimento.

`````

Il costo di cercare a tentoni si misura, e conviene misurarlo su un problema
piccolo abbastanza da starci in una pagina.

## Un rompicapo con cui contare

Il rompicapo delle otto tessere è una cornice di tre caselle per lato con otto
tessere numerate e un buco. Una mossa fa scivolare nel buco una delle tessere
che gli stanno accanto. I modi di disporre nove cose in nove caselle sono
$9! = 362\,880$ (il punto esclamativo si legge «fattoriale» e vuol dire
$9 \times 8 \times 7 \times \ldots \times 1$: nove scelte per la prima casella,
otto per la seconda, e così via). Di quelle disposizioni, però, solo la metà si
può raggiungere facendo scorrere le tessere: **181.440**. La ragione è graziosa
e si controlla su un foglio. Ogni mossa scambia il buco con una tessera, cioè
scambia due cose fra loro, e a ogni scambio una proprietà della disposizione
che i matematici chiamano parità si inverte, come un interruttore: pari,
dispari, pari, dispari. Ma ogni mossa sposta anche il buco di una casella, e
su una scacchiera colorata come quella della dama il buco cambierebbe colore a
ogni passo: bianco, nero, bianco, nero. I due interruttori scattano insieme.
Per riportare il buco dove stava (sul suo colore) servono quindi mosse in
numero pari, cioè scambi in numero pari, e le disposizioni col buco al suo
posto che chiederebbero un numero dispari di scambi non si raggiungono mai:
sono esattamente la metà. Poche abbastanza da
poterle guardare tutte, tante abbastanza da far vedere la differenza fra
guardarle tutte e non guardarle.

Il programma che segue esplora quel rompicapo, e per farlo apre gli stati uno
alla volta: «aprire» uno stato vuol dire guardare quali mosse ci sono e
generare le situazioni che ne escono. Sceglie ogni volta lo stato che gli
sembra più promettente, cioè quello per cui è più piccola la somma fra i passi
già fatti e una **stima** di quelli che restano. La stima, per adesso, è messa
a zero: è come dire che non abbiamo nessun fiuto, e il programma è costretto a
guardarsi intorno in tutte le direzioni allo stesso modo. Che cosa succeda
quando un fiuto ce l’ha è il resto della sezione.

```python
import heapq

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


def cerca(stima):
    """Apre sempre lo stato con (passi fatti + stima di quelli che restano)
    piu' piccolo. Restituisce la lunghezza della soluzione e quanti stati
    ha dovuto guardare per trovarla."""
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
senza nessuna stima:  20 mosse, 48389 stati guardati
```

Venti mosse di soluzione, e per trovarle ne sono state esaminate
quarantottomila e passa: più di un quarto di tutte le posizioni che questo
rompicapo ha. Con la stima a zero l’algoritmo apre sempre lo stato più vicino
alla partenza, e quindi si allarga in tutte le direzioni allo stesso modo,
esattamente come l’acqua del labirinto. Non sbaglia mai la risposta, e paga
carissimo il non sapere dove sta andando.

## La stima di quanto manca

L’unica cosa che cambia le proporzioni è darle un fiuto, cioè un modo di
indovinare, guardando uno stato, **quanto lavoro resta** da lì alla fine.

Quel fiuto ha un nome, e da qui in avanti il capitolo lo userà sempre: si
chiama **euristica**, che è una parola greca per «che aiuta a trovare» ed è il
nome che in informatica si dà a una regola pratica, non garantita, che
indirizza la ricerca. E la proprietà che deve avere ha anch’essa un nome, da
avere in tasca prima di incontrarlo: un’euristica si dice **ammissibile**
quando non esagera mai, cioè quando il lavoro che stima non supera mai quello
che serve davvero.

`````{tab} Elementare

In una città che non conosci c’è la casa di un amico, e sai solo che sta vicino
a una torre che dal tuo incrocio si vede. Quali strade ci portino non lo sai,
né se sono a senso unico. Sai la direzione, e i metri in linea d’aria.

Puntare alla torre a ogni incrocio non funziona. Quando va bene arrivi per un
giro più lungo del necessario; quando va male finisci davanti a un muro con la
torre dietro, e le altre strade le hai lasciate al primo bivio.

Allora tiri fuori un foglio e tieni aperte più strade insieme. Accanto a ogni
punto raggiunto scrivi i passi che ti è costato arrivarci e i metri che restano
in linea d’aria. Poi allunghi di un passo la strada con la **somma** più
piccola, e solo quella.

La somma, non uno dei due numeri. Coi soli passi fatti ti allargheresti in
tondo come l’acqua del labirinto; coi soli metri che restano ricadresti nel
muro di prima. Con la somma i quartieri dalla parte opposta restano bianchi sul
foglio, e la strada corta che partiva male non ti sfugge.

Una riga del foglio tocca la casa dell’amico, e tu non ti fermi. Aspetti che
venga il suo turno, cioè che sia la sua somma la più piccola. Chi si ferma alla
prima strada che arriva porta a casa quella, e più giù nel foglio ce n’era una
più corta.

Certe volte capiti su un incrocio già scritto, con meno passi della volta
prima. Cancelli il numero vecchio, e quell’incrocio torna in gioco. Su uno già
allungato non succede mai, perché un passo camminato accorcia la linea d’aria
al massimo di un passo, mai di colpo. Quando tocca a un incrocio, ci sei già
arrivato per la strada più corta.

Il conto lo paga il foglio, dove ogni incrocio aperto finisce e non esce più.
Per una città basta, per le strade di un paese intero servirebbe un magazzino.
Allora lasci il foglio, riprendi il filo del labirinto e ti dai un tetto di
venti chilometri: torni indietro appena la somma lo supera, e se la casa non
salta fuori alzi il tetto e riparti da capo. La strada che trovi resta la più
corta.

Quei metri in linea d’aria sono la **stima**, e non sono mai più dei metri
veri. Le strade girano, la linea d’aria no, e non capita che dica «due
chilometri» dove la strada ne fa uno e mezzo.

Adesso uno al bar ti dice dieci chilometri dove ce ne sono due. La somma di
quella strada diventa pessima, non la allunghi più, e all’amico ci arrivi da
un’altra parte. Nemmeno te ne accorgi: sei arrivato, la strada c’era, ed era
più lunga del necessario. Sbagliare per difetto costa tempo, sbagliare per
eccesso costa la strada giusta.

`````

`````{tab} Superiore

Si introduce una funzione $h(n) \ge 0$, l’**euristica**, che stima il costo del
cammino ottimo da $n$ alla meta, e si combina con il costo già pagato $g(n)$:

$$
f(n) = g(n) + h(n),
$$

dove $g(n)$ è il costo del cammino trovato finora dalla partenza a $n$ e $f(n)$
è quindi la stima del costo totale del miglior cammino che passa per $n$.
Espandere sempre il nodo con $f$ minimo è l’algoritmo **A\***, di Hart, Nilsson
e Raphael {cite}`hart1968formal`. Con $h \equiv 0$ si riduce alla ricerca a
costo uniforme, cioè al caso con la stima a zero.

Il prezzo, che la tabella di prima non dice, è scomodo: A\* **tiene in memoria
tutti i nodi generati**, esattamente come la ricerca in ampiezza. Riduce
enormemente quanti ne genera, e questo è tutto il guadagno, ma la memoria resta
il vincolo che morde per primo. Sul rompicapo delle otto tessere non si vede;
su quello delle quindici, che di posizioni ne ha diecimila miliardi, sì, e la
via d’uscita è sposare A\* con l’approfondimento iterativo, tenendo un tetto
sul valore di $f$ invece che sulla profondità. È l’**IDA\*** di Richard Korf
{cite}`korf1985depth`, ed è stato il primo metodo a trovare, dentro limiti di
tempo e di memoria praticabili, soluzioni ottime di istanze del quindici
generate a caso.

La proprietà che serve a $h$ ha un nome: è **ammissibile** se non sovrastima
mai, cioè se $h(n) \le h^*(n)$ per ogni $n$, dove $h^*(n)$ è il costo vero del
cammino ottimo da $n$ alla meta. Un’euristica ammissibile è, in altre parole,
**ottimista**. E con un’euristica ammissibile A\* restituisce una soluzione di
costo minimo.

La ragione, in poche righe, e con due dettagli che sembrano formalità e non lo
sono. A\* dichiara di aver finito quando **estrae** dalla frontiera uno stato
finale, non quando lo genera: se bastasse generarlo, restituirebbe la prima
soluzione che incontra, che non è la più corta. Detto questo, supponiamo che
stia per restituire una soluzione peggiore di quella ottima. Sulla frontiera ci
sarebbe allora il **primo** nodo $n$ non ancora espanso lungo il cammino
ottimo; il suo predecessore lungo quel cammino è già stato espanso, e lo ha
generato con $g(n) = g^*(n)$, cioè col costo giusto. Per quel nodo vale quindi
$f(n) = g^*(n) + h(n) \le g^*(n) + h^*(n) = C^*$, un valore non superiore al
costo ottimo e quindi inferiore a quello della soluzione peggiore che stiamo per
restituire. Ma allora A\* avrebbe estratto $n$ prima, perché estrae sempre il
minimo. È la contraddizione che dimostra il risultato.

Il «primo non ancora espanso» porta tutto il peso dell’argomento: appartenere
al cammino ottimo non basta, bisogna esserci **arrivati lungo di esso**, e solo
per il primo dei non espansi questo è garantito dal predecessore.

L’argomento ha però una condizione che resta implicita, ed è il punto in cui
si sbaglia: presuppone di poter **tornare su uno stato già
aperto** se salta fuori una strada più corta per arrivarci. Il codice qui sopra
lo fa (è la riga che riscrive `costo[t]` e rimette lo stato in coda); una
versione che marchiasse gli stati come «fatti» e non ci tornasse più potrebbe,
con un’euristica solo ammissibile, restituire una soluzione peggiore di quella
ottima.

Una proprietà leggermente più forte si chiama **consistenza**: $h$ è
consistente se per ogni nodo $n$ e ogni suo successore $n'$ ottenuto con
l’azione $a$ vale

$$
h(n) \le c(n, a, n') + h(n'),
$$

che è una disuguaglianza triangolare: la stima da qui non può superare il costo
di un passo più la stima da lì. Ogni euristica consistente **che valga zero
sugli stati finali** è anche ammissibile, e non viceversa; la condizione sugli
stati finali serve davvero, perché $h \equiv 5$ soddisfa la disuguaglianza
triangolare su qualunque grafo a costi non negativi e ammissibile non è.

In cambio la consistenza dà una cosa pratica, ed è esattamente quella che manca
sopra: i valori di $f$ non diminuiscono mai lungo un cammino, quindi la prima
volta che uno stato viene **estratto dalla frontiera** ci si sta arrivando in
modo ottimo, e allora marcarlo come fatto e non tornarci più è lecito. La
parola esatta è **estratto**, non raggiunto. Uno stato si può *generare*
per una strada pessima molto prima di generarlo per quella buona (succede anche
con $h \equiv 0$, che è consistente ed è l’euristica della ricerca a costo
uniforme), e chiudere uno stato alla prima *generazione* può restituire
soluzioni peggiori dell'ottimo, e basta che succeda una volta perché la
garanzia non ci sia più. La riga del codice qui sopra che riscrive `costo[t]`
esiste esattamente per questo, e se la frase valesse alla prima generazione
quella riga sarebbe codice morto.

Le due euristiche di questo capitolo sono tutte e due consistenti, e valgono
zero sulla configurazione finale.

`````

L’algoritmo che tiene insieme le due cose, il lavoro già fatto e la stima di
quello che manca, si chiama **A\*** (si legge «a stella»), e ha una data e tre
nomi: 1968, Peter Hart, Nils Nilsson e Bertram Raphael {cite}`hart1968formal`.
È quello che il codice qui sopra esegue, ed è quello che, con l’asterisco o
senza, sta dentro quasi tutto ciò che cerca un percorso: navigatori, robot che
attraversano una stanza, personaggi di videogioco che aggirano un muro.

Le due euristiche classiche per il rompicapo delle otto tessere non sono
inventate a caso, e conviene vedere da dove escono, perché è il modo
principale in cui si inventa un’euristica.

Si prende il problema e gli si **tolgono delle regole**. Nel rompicapo vero una
tessera si può spostare solo in una casella adiacente e solo se quella casella
è vuota. Se si cancella la seconda regola, una tessera può andare in qualunque
casella adiacente, e il costo per rimettere tutto a posto è la somma di quanto
ciascuna tessera dista dal suo posto contando i passi in orizzontale e in
verticale: si chiama **distanza a isolati**, e nei testi si trova più spesso
col nome inglese di **distanza di Manhattan**, che è la stessa cosa, perché è
il modo in cui si contano i metri in una città a scacchiera, dove non si taglia
in diagonale. Se si
cancellano tutt’e due, una tessera vola dove vuole in una mossa, e il costo è
semplicemente quante **tessere sono fuori posto**. Sono i due nomi che
compaiono nell’elenco qui sotto.

E qui c’è la garanzia, che è quasi troppo bella: **il costo esatto di un
problema con meno regole non può mai superare quello del problema vero**,
perché tutto quello che si poteva fare prima si può fare ancora, e magari
qualcosa in più. Quindi un problema alleggerito, risolto esattamente, dà
sempre un numero che sta sotto (o al più pari) a quello vero: è cioè
un’euristica ammissibile per costruzione, e non c’è bisogno di verificarlo caso
per caso.

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


for nome, stima in (("tessere fuori posto", fuori_posto),
                    ("distanza a isolati", a_isolati)):
    passi, guardati = cerca(stima)
    print(f"{nome:22} {passi} mosse, {guardati:5d} stati guardati"
          f"   ({senza_stima / guardati:5.1f} volte meno)")
```

```text
tessere fuori posto    20 mosse,  3666 stati guardati   ( 13.2 volte meno)
distanza a isolati     20 mosse,   282 stati guardati   (171.6 volte meno)
```

Tre righe di numeri che dicono tutta la sezione. **La risposta non cambia**:
venti mosse in tutti e tre i casi, perché tutte e tre le stime sono ottimiste e
quindi nessuna fa sbagliare strada. Cambia solo quanto si guarda:
quarantottomila stati senza stima, tremilaseicento con quella grossolana,
duecentottantadue con quella più fine.

Quello che quei numeri non dicono è **dove** siano finite le posizioni
guardate, ed è la parte che spiega il resto. Si può disegnare: ogni posizione
aperta si mette su un piano, in orizzontale i passi già fatti per arrivarci e
in verticale **la stima** di quanti ne mancano ({numref}`fig-frontiera`). In
verticale c’è la stima e non la distanza vera, e la differenza conta: quello
che il disegno mostra è che con il fiuto la ricerca non apre mai una posizione
per cui la somma dei due numeri superi il costo della soluzione, non che stia
lontana dai vicoli ciechi. Il piano è lo stesso nei due casi, e cambia una cosa
sola: se l’algoritmo quel secondo numero lo guarda oppure no.

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

E qui c’è la cosa che i tre numeri da soli non facevano vedere: la stima non fa
guardare *meno in giro*, le impedisce di salire sopra quella riga. Con la stima
a zero la somma dei due numeri è sempre uguale ai soli passi fatti, quindi non
c’è nessuna riga da non superare, e ogni posizione vale quanto un’altra alla
stessa distanza dalla partenza.

E il confronto fra le due stime ha una regola sola, che si legge nella loro
definizione: contare i passi è **sempre almeno quanto** contare le tessere
fuori posto, perché una tessera fuori posto dista almeno un passo. Tutte e due,
poi, sono stime senza salti: da una mossa alla successiva cambiano al massimo
di uno, ed è la proprietà che i matematici chiamano **consistenza**. Fra due
euristiche consistenti, quella che dà sempre il numero più grande
**domina** l'altra: A\* non apre mai più stati con la dominante che con
l'altra, perché apre comunque tutti quelli per cui i passi fatti più la stima
stanno sotto il costo della soluzione, e alzare la stima quell'insieme lo
restringe. Cercare una buona
euristica vuol dire cercare la stima più alta che non superi mai il vero.

`````{tab} Elementare

```{admonition} Da ricordare
:class: important
- Cercare **a tentoni** si può fare in due modi: allagando il labirinto un
  metro alla volta (si trova la strada più corta, e si consuma una memoria
  enorme) oppure seguendo un corridoio fino in fondo con un filo dietro (il
  filo costa pochissimo, la strada trovata può essere assurda, e dove i
  corridoi girano in tondo non si esce più finché non si segnano i bivi già
  visti, cioè finché non si ricomincia a pagare la memoria risparmiata).
- I due si sposano girando col filo, ma con un tetto che si alza a ogni giro.
  Rifare ogni volta i primi metri costa poco, perché i bivi vicini all’inizio
  sono pochissimi rispetto a quelli lontani, e in cambio si tengono la memoria
  del filo e la garanzia dell’acqua.
- Il difetto vero del cercare a tentoni è che si cerca dappertutto con lo
  stesso impegno, anche dalla parte sbagliata, e non la memoria.
- La cosa che cambia le proporzioni è una **stima di quanto manca**: la
  distanza in linea d’aria dalla torre vicino a casa dell’amico. Non dice
  quale strada prendere, dice solo da che parte guardare per primo.
- La stima deve **stare sotto** al vero, mai sopra. Se sbaglia per difetto si
  guarda qualcosa di troppo; se sbaglia per eccesso si scarta la strada buona
  e si arriva più lunghi, senza nemmeno accorgersene.
- Una stima si inventa **togliendo regole al problema**: risolto esattamente,
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
  **memoria**, che è il vincolo che morde per primo; la profondità costa
  $O(bm)$ ma non è né completa né ottima. L’**approfondimento iterativo** le
  unisce pagando solo un fattore $b/(b-1)$ di lavoro in più.
- **A\*** {cite}`hart1968formal` espande il nodo di $f(n) = g(n) + h(n)$
  minimo. Con $h$ **ammissibile** ($h \le h^*$, cioè ottimista) restituisce una
  soluzione di costo minimo; con $h$ **consistente**
  ($h(n) \le c(n,a,n') + h(n')$, disuguaglianza triangolare) i valori di $f$
  non decrescono lungo un cammino e ogni stato viene **estratto** dalla
  frontiera in modo ottimo la prima volta (estratto, non generato: per una
  strada pessima lo si genera anche molto prima).
- Le euristiche ammissibili si costruiscono **rilassando** il problema: il
  costo esatto di un problema con vincoli in meno è un limite inferiore a
  quello del problema vero, quindi è ammissibile per costruzione. Le due del
  rompicapo (tessere fuori posto, distanza a isolati) sono i rilassamenti che
  cancellano rispettivamente due vincoli e uno.
- Fra due euristiche **consistenti**, quella con valori sempre maggiori
  **domina** l’altra, e A\* con la dominante non espande mai più nodi
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
mentre lui calcola, non cambia idea. Nella sezione che segue dall’altra parte
del tavolo c’è qualcuno che sceglie, e sceglie apposta il ramo che a noi
conviene meno: metà dell’albero smette di essere nostra, e il modo di
ragionarci sopra cambia da capo.
