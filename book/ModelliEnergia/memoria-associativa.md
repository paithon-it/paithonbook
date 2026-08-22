# La memoria associativa di Hopfield

La memoria di un computer funziona per **indirizzo**: ogni dato abita in una
casella numerata, e per recuperarlo bisogna conoscere il numero esatto, sbagli
una cifra e ottieni un dato qualsiasi. La memoria umana funziona per
**contenuto**: bastano tre note stonate fischiettate da un passante per farti
riaffiorare l'intera canzone, un profumo per restituirti una cucina di
trent'anni fa, mezza faccia intravista da un autobus per completare nome e
cognome. Non forniamo indirizzi: forniamo *frammenti*, e il ricordo si
completa da solo. I tecnici la chiamano **memoria associativa**.

Nel 1982 John Hopfield mostra come costruirne una con neuroni artificiali
{cite}`hopfield1982neural`. Era un fisico della materia condensata, cioè di
come si comportano solidi e liquidi, passato poi a studiare i sistemi
biologici, e dal 1980 insegnava al California Institute of Technology.

Non parte da un foglio bianco, ed è giusto dirlo. Memorie che si interrogano
per contenuto circolavano già da un decennio, costruite legando fra loro i
pezzi che nei ricordi vanno d'accordo: le propongono, tutti nel 1972 e ognuno
per conto suo, Teuvo Kohonen, Kaoru Nakano, James Anderson e Shun-ichi Amari,
il cui lavoro è quello che il libro cita {cite}`amari1972learning`. E nel 1974
William Little descrive una rete di
neuroni che si accendono solo quando la spinta ricevuta supera una certa
soglia, e mostra che una rete così può restare a lungo nella stessa
configurazione invece di cambiarne in continuazione: quelle configurazioni le
chiama *stati persistenti* {cite}`little1974existence`. Quello che Hopfield
aggiunge, e che fa ripartire
il campo da lui, è l’**energia**: un solo numero associato a ogni
configurazione della rete, più la dimostrazione che il modo in cui la rete si
aggiorna non lo fa mai salire. Da quel momento i ricordi sono minimi, e
ricordare è una discesa.

La sua mossa è quella di un fisico. Prendiamo un gruppo di neuroni che possono
stare solo «accesi» o «spenti» (in gergo si dicono **binari**, come una fila
di interruttori), collegati fra loro da **pesi**: numeri che dicono quanto due
neuroni tendono a stare d'accordo, positivi se preferiscono trovarsi nella
stessa posizione, negativi se preferiscono l'opposta. E i pesi siano
**simmetrici**, cioè il legame fra due neuroni valga lo stesso nei due versi.
Un aggeggio così, osserva Hopfield, è matematicamente identico a un materiale
magnetico, dove ogni atomo si comporta come una freccina che punta in su o in
giù e sente l'influenza delle vicine.

Quella freccina è la quarta parola presa in prestito: i fisici la chiamano
**spin**, ed è la parola che si incontra in tutti i lavori di fisica su queste
reti. Qui non serve sapere che cos'è uno spin davvero: basta l'immagine
della freccina con due sole posizioni, che è la stessa cosa di un neurone
acceso o spento. Da qui in avanti «stato» e «configurazione» vogliono dire la
stessa cosa: l'elenco di come stanno messi tutti i neuroni in un dato momento.

E di sistemi così la fisica sa tutto, a partire dalla domanda giusta: qual è
l'energia di ogni configurazione, e verso dove scende?

```{figure} ../figures/energia-paesaggio.svg
:name: fig-energia-paesaggio
:alt: Paesaggio di energia con tre valli i cui minimi, segnati in teal, sono i ricordi memorizzati; una pallina ocra etichettata «ricordo parziale / rumoroso» parte da un punto alto e una freccia terracotta la accompagna nel fondo della valle più vicina. L'asse verticale è l'energia E, quello orizzontale lo stato della rete.
:width: 92%

Il paesaggio di energia di una rete di Hopfield. Le parole tecniche del
disegno, sciolte: i «pattern memorizzati» sono i ricordi (nel disegno sono
tre, e si chiamano A, B e C), i «minimi» sono i fondovalle in cui stanno,
sull'asse orizzontale («stato della rete») ci sono tutte le configurazioni
possibili messe in fila, lo stato «rumoroso» da cui
parte la pallina è l'indizio rovinato che diamo alla rete, e un
«aggiornamento» è una casella che guarda i suoi vicini e decide se cambiare.
```

La {numref}`fig-energia-paesaggio` contiene, in un solo disegno, tutta l'idea:
i ricordi sono valli, e a fare il lavoro di richiamo al posto nostro è la
regola con cui la rete si aggiorna, che l'energia può soltanto farla scendere.

Una precisazione conviene metterla subito, perché altrimenti si finisce con
due immagini scollegate in testa. La pallina non è un oggetto in più che si
muove sopra il paesaggio: la pallina *è* la rete, cioè l'elenco di quali
neuroni sono accesi in quel momento, disegnato come un puntino su una carta.

`````{tab} Elementare

Ogni ricordo che la rete ha memorizzato scava una valle nel paesaggio della
figura. Lo stato della rete in un dato momento è una pallina appoggiata da
qualche parte su quel profilo. Dare alla rete un indizio (un ricordo parziale,
o rovinato) significa posare la pallina in un punto alto del pendio, vicino a
una valle ma non sul fondo. Poi non c'è altro da fare: la pallina rotola, e
può soltanto scendere, finché si ferma nel punto più basso nei paraggi. Se
l'indizio somigliava al ricordo B più che agli altri, il fondo più vicino è
proprio la valle di B: arrivarci *è* ricordare, con tutti i dettagli che
l'indizio non conteneva. La melodia stonata del passante ti deposita sul
fianco della valle della canzone giusta, e la discesa fa il resto: il ricordo
non lo *cerchi*, ci *cadi dentro*.

Due avvertenze oneste, sulla valle sbagliata e sulla capienza. La pallina
scende nella valle più *vicina*, non necessariamente in quella *giusta*, e
«vicina» qui vuol dire somigliante: due configurazioni sono vicine quando
differiscono in poche caselle. Se l'indizio è troppo rovinato somiglia più al
ricordo sbagliato che a quello giusto, e da lì si finisce nel ricordo
sbagliato con la stessa naturalezza.

E il paesaggio ha una capienza. Scavando troppe valli in poco spazio i fianchi
si fondono, e compaiono conche a metà strada tra due ricordi: «ricordi
fantasma» che nessuno ha mai memorizzato. A fondersi per prime sono le valli
dei ricordi che si somigliano, perché due ricordi somiglianti scavano vicini.
Su una rete di venticinque neuroni si può misurare: con tre ricordi
presi a caso il richiamo riesce l'86% delle volte, con quattro il 69%, con
cinque il 50%, e più se ne aggiungono più peggiora. Il peggioramento, qui, è
dolce.

In una rete grande il passaggio è netto, e la soglia si sa dov'è, almeno per
ricordi presi a caso: sta intorno al 14% del numero di neuroni, cioè un
ricordo ogni sette. Fin sotto quella quota la rete funziona quasi sempre,
appena sopra smette di funzionare quasi del tutto. A rompersi di colpo è la
rete grande, il contrario di quel che verrebbe da pensare: in venticinque
neuroni ogni sorteggio dei ricordi dà un risultato diverso e la soglia si
spalma, in diecimila le fluttuazioni si mediano fra loro e quel che resta è
uno scalino.

`````

`````{tab} Superiore

La rete è un vettore di $N$ neuroni binari $s_i \in \{-1, +1\}$, collegati da
pesi **simmetrici** ($w_{ij} = w_{ji}$) e senza auto-connessioni
($w_{ii} = 0$). A ogni stato $\mathbf{s}$ è associata l'energia

$$
E(\mathbf{s}) = -\frac{1}{2}\, \mathbf{s}^\top \mathbf{W} \mathbf{s} = -\frac{1}{2} \sum_{i \neq j} w_{ij}\, s_i s_j,
$$

dove $\mathbf{W}$ è la matrice dei pesi e la somma percorre le coppie
**ordinate** (ogni coppia di neuroni compare due volte, una per verso: è da lì
che viene il $\tfrac12$ davanti). Una coppia collegata da peso positivo
abbassa l'energia quando i due neuroni
concordano, e la alza quando discordano (per pesi negativi vale l'opposto).
Manca il termine di soglia $+\sum_i \theta_i s_i$ del modello generale: qui le
soglie sono nulle, com'è nel codice. La
dinamica è l’**aggiornamento asincrono**: si sceglie un neurone $i$, si
calcola il suo campo locale $h_i = \sum_j w_{ij} s_j$ e si pone
$s_i \leftarrow \operatorname{sign}(h_i)$ (con la convenzione
$\operatorname{sign}(0) = s_i$, cioè in caso di parità il neurone resta com'è:
è quello che fa il codice, e senza quella convenzione lo stato uscirebbe da
$\{-1,+1\}$), lasciando tutto il resto fermo.

Che l'energia non possa salire si mostra in tre passaggi, e ciascuno usa
un'ipotesi diversa. Primo: si isolano i termini che
contengono $s_i$. Sono due somme, $-\tfrac12\sum_{l \neq i} w_{il}s_i s_l$ e
$-\tfrac12\sum_{k \neq i} w_{ki} s_k s_i$, che **grazie alla simmetria** sono
uguali e si raccolgono in $-s_i h_i$; e $h_i$ non dipende da $s_i$ **grazie a**
$w_{ii} = 0$. Dunque $E = -s_i h_i + \text{cost}$, dove la costante non
coinvolge $s_i$. Secondo: se il neurone si capovolge, $s_i \to -s_i$, l'energia
varia di $\Delta E = 2\, s_i h_i$. Terzo: il capovolgimento avviene solo quando
$\operatorname{sign}(h_i) \neq s_i$, cioè quando $s_i h_i = -|h_i|$, da cui

$$
\Delta E = -2\,|h_i| \le 0 .
$$

**Ogni aggiornamento fa scendere l'energia o la lascia invariata, mai salire**:
invariata quando il neurone resta com'è (anche in caso di parità, $h_i = 0$),
più bassa di $2|h_i|$ quando si capovolge. Tolta la simmetria il conto
non torna: su una rete casuale con $\mathbf{W}$ asimmetrica si misurano
capovolgimenti con $\Delta E$ positivo, e alcune reti asimmetriche si mettono
davvero a girare in tondo senza fermarsi mai.

Per concludere che la discesa **termina** serve un'ipotesi in più, che di
solito si tace: che ogni neurone venga visitato infinitamente spesso. Gli
stati sono in numero finito ($2^N$), quindi $E$ assume un numero finito di
valori e non può scendere per sempre; ma senza una scansione equa i
capovolgimenti potrebbero cessare mentre da qualche parte resta un neurone
scontento, e quello non sarebbe un punto fisso. Il codice lo garantisce
ripescando ogni volta una permutazione di tutti i neuroni.

Le valli si scolpiscono con la **regola di Hebb**, dal neuropsicologo Donald
Hebb che nel 1949 la propose per le sinapsi biologiche (l'idea che sarebbe poi
stata riassunta nello slogan «i neuroni che si attivano insieme si legano
insieme»). Per memorizzare i pattern $\boldsymbol{\xi}^1, \dots, \boldsymbol{\xi}^M$, ciascuno un
vettore di $\pm 1$:

$$
w_{ij} = \frac{1}{N} \sum_{\mu=1}^{M} \xi_i^{\mu}\, \xi_j^{\mu}
\qquad (i \neq j),
$$

dove $\xi_i^{\mu}$ è l’$i$-esimo bit del pattern $\mu$: ogni pattern rafforza
i legami tra i propri bit concordi. Che questo lo renda un minimo locale di
$E$ si vede in un conto solo, ed è il conto da cui discende tutto il resto
della sezione. Mettendo la rete nello stato $\boldsymbol{\xi}^\mu$, il campo
locale sul neurone $i$ vale

$$
h_i^\mu = \underbrace{\frac{N-1}{N}\, \xi_i^{\mu}}_{\text{segnale}}
\;+\; \underbrace{\frac{1}{N} \sum_{\nu \neq \mu} \xi_i^{\nu}
\sum_{j \neq i} \xi_j^{\nu} \xi_j^{\mu}}_{\text{interferenza}} ,
$$

perché $(\xi_j^\mu)^2 = 1$ per ogni $j$. Il primo termine tira il neurone
esattamente dove il pattern lo vuole; il secondo è la somma delle
sovrapposizioni con **tutti gli altri** ricordi, e il bit resta al suo posto
finché quel disturbo, moltiplicato per $\xi_i^\mu$, non scende sotto
$-(N-1)/N$. Da qui vengono, in un colpo solo, tre cose: che i pattern quasi
ortogonali (interferenza piccola) siano stabili; che aggiungerne troppi
faccia crescere il disturbo finché vince; e che a pesare non sia solo il loro
numero, ma anche quanto si somigliano, come si vedrà con la T, la L e la X.
La capienza è dunque
limitata: l'analisi di meccanica statistica di Daniel Amit, Hanoch Gutfreund e
Haim Sompolinsky {cite}`amit1985storing`, con i metodi dei vetri di spin,
mostra che la memoria associativa esiste solo per $M < \alpha_c N$ con
$\alpha_c \approx 0{,}14$, e che oltre quella soglia il recupero non degrada
dolcemente: collassa (la transizione è del primo ordine). Il valore raffinato
che si trova citato dappertutto, $0{,}138$, viene dall'analisi estesa che gli
stessi tre autori pubblicano due anni dopo; l'articolo del 1985 scrive
$0{,}14$.

Le ipotesi contano, perché sono ciò che rende quel numero un
teorema e non un'osservazione: pattern **casuali e non correlati**, limite
termodinamico $N \to \infty$, temperatura nulla, simmetria di replica, e una
tolleranza per una piccola frazione di bit errati nel richiamo. Fuori di lì il
numero va maneggiato con cura, e la rete di venticinque neuroni mostra quanto:
a $N = 25$ non c'è nessun limite termodinamico e la transizione è del tutto
sfumata. Misurando il richiamo con pattern casuali (quarantamila prove per
punto, gli $M$ pattern ridisegnati a ogni prova, sei bit invertiti su
venticinque) si
ottiene $0{,}86$ a $M = 3$, $0{,}69$ a $M = 4$, $0{,}50$ a $M = 5$ e ancora
$0{,}33$ a $M = 6$, cioè a quasi il doppio della soglia ($\alpha_c N \approx 3{,}5$,
e $6/3{,}5 = 1{,}7$): nessun crollo, una discesa
regolare. Il collasso è un fenomeno di reti
grandi, e citare $\alpha_c N$ su una rete da venticinque neuroni è un modo
elegante di sbagliare.

E anche sotto soglia il paesaggio contiene minimi non richiesti: gli opposti
$-\boldsymbol{\xi}^{\mu}$ di ogni pattern e miscele spurie di tre o più ricordi.

`````

## Una memoria che si ripara da sola, in poche righe

Tutto questo si può toccare con mano, e in poche righe. Il codice che segue
costruisce una rete di venticinque neuroni disposti in una griglia di cinque
per cinque caselle, e le fa memorizzare tre lettere stilizzate: una T, una L e
una X. Memorizzare, qui, vuol dire una cosa sola, e la fa una riga sola:
legare fra loro le caselle che nelle tre lettere vanno d'accordo, tanto più
forte quanto più spesso ci vanno. È la regola che scava le valli, e porta il
nome del neuropsicologo Donald
Hebb, che nel 1949 propose per le sinapsi del cervello proprio questo: due
cellule che si accendono insieme rafforzano il legame che le unisce. Poi il
codice rovina una lettera
invertendo sei caselle a caso (sei su venticinque, il 24%) e lascia che la
rete si aggiusti da sé, una casella alla volta, finché nessuna vuole più
cambiare.

Conviene vedere che numero esce da quel «legare», perché è l'unico conto del
capitolo che si fa a mente. Prendiamo la seconda e la terza casella della
prima riga: nella T sono accese tutte e due, nella L sono spente tutte e due,
nella X sono spente tutte e due. Vanno d'accordo tre volte su tre, e il loro
legame vale $3/25 = 0{,}12$, il massimo che si possa avere con tre ricordi.
Prendiamo invece la prima casella e la seconda della stessa riga: vanno
d'accordo solo nella T (accese entrambe) e discordano nella L e nella X. Un
accordo e due disaccordi fanno $1 - 1 - 1 = -1$; si divide per il numero di
caselle, venticinque, perché così i legami restano della stessa taglia anche
se la griglia cresce, e viene $-0{,}04$: un legame debole e di segno
contrario, che quelle due caselle tenderà a tenerle diverse. Tutti i legami
della rete sono numeri così, e sono tutto ciò che la rete «sa».

Il codice costruisce i legami con quella regola, e poi mette la rete alla
prova.

```python
import numpy as np

rng = np.random.default_rng(42)

# Tre lettere stilizzate 5x5: '#' = pixel acceso (+1), '.' = spento (-1)
LETTERE = {
    "T": ["#####",
          "..#..",
          "..#..",
          "..#..",
          "..#.."],
    "L": ["#....",
          "#....",
          "#....",
          "#....",
          "#####"],
    "X": ["#...#",
          ".#.#.",
          "..#..",
          ".#.#.",
          "#...#"],
}

def a_vettore(disegno):
    """Da lista di stringhe a vettore di +1/-1."""
    return np.array([1 if c == "#" else -1
                     for riga in disegno for c in riga])

def a_righe(s):
    """Da vettore di +1/-1 a cinque stringhe stampabili."""
    griglia = np.where(s == 1, "#", ".").reshape(5, 5)
    return ["".join(riga) for riga in griglia]

pattern = np.array([a_vettore(d) for d in LETTERE.values()])   # forma (3, 25)
N = pattern.shape[1]

# Regola di Hebb: somma dei prodotti esterni, diagonale a zero
W = (pattern.T @ pattern) / N
np.fill_diagonal(W, 0.0)

def energia(s):
    """E(s) = -1/2 s^T W s (soglie nulle)."""
    return -0.5 * s @ W @ s

def richiama(s, max_passate=10):
    """Aggiornamento asincrono fino a un punto fisso (minimo locale)."""
    s = s.copy()
    for _ in range(max_passate):
        cambiato = False
        for i in rng.permutation(N):        # un neurone alla volta
            campo = W[i] @ s                # campo locale sul neurone i
            nuovo = np.sign(campo) if campo != 0 else s[i]
            if nuovo != s[i]:
                s[i] = nuovo
                cambiato = True
        if not cambiato:                    # nessun cambiamento: stato stabile
            break
    return s

def corrompi(s, quanti=6):
    """Inverte 'quanti' bit scelti a caso (6 su 25 = 24%)."""
    s = s.copy()
    indici = rng.choice(N, size=quanti, replace=False)
    s[indici] = -s[indici]
    return s

for nome, disegno in LETTERE.items():
    originale = a_vettore(disegno)
    rumoroso = corrompi(originale)          # 24% dei pixel invertiti
    recuperato = richiama(rumoroso)
    esito = ("recuperato" if np.array_equal(recuperato, originale)
             else "NON recuperato")
    print(f"{nome}:  E = {energia(rumoroso):+.2f} "
          f"-> {energia(recuperato):+.2f}  ({esito})")
    print("   corrotto   richiamato")
    for r1, r2 in zip(a_righe(rumoroso), a_righe(recuperato)):
        print(f"   {r1}      {r2}")
    print()
```

Eseguendolo, tutte e tre le lettere riemergono intatte dalle loro versioni
sfigurate. Per la T, ad esempio, l'energia scende da $-2{,}08$ dello stato
corrotto a $-11{,}20$ della lettera richiamata:

```text
T:  E = -2.08 -> -11.20  (recuperato)
   corrotto   richiamato
   #.###      #####
   ..#..      ..#..
   #.#.#      ..#..
   .##..      ..#..
   .###.      ..#..
```

Quei due numeri li ha calcolati la rete, sommando un contributo per ogni
coppia di caselle: chi va d'accordo con il legame che lo unisce abbassa il
totale, chi lo contraddice lo alza. Che vengano negativi non vuol dire niente
di speciale, e conviene toglierselo di mezzo subito, perché l'analogia della
carta geografica qui inganna: l'energia non è un'altitudine sul livello del
mare, non ha uno zero suo. Da sola non dice nulla; dice tutto se confrontata
con un'altra energia dello stesso paesaggio. $-11{,}20$ sta più in basso di
$-2{,}08$, e qui è l'unico fatto che conta.

Quella stampa mostra il prima e il dopo, e nasconde la parte interessante: i
passi in mezzo. In {numref}`fig-hopfield-ricorda` ci sono tutti, e la figura
mette per la prima volta accanto le due immagini che finora sono andate
separate. A sinistra c'è la griglia di venticinque caselle, dove a ogni
*aggiornamento* (una casella guarda i suoi vicini e decide se cambiare) una
sola casella cambia colore. A destra c'è la pallina, cioè l'energia. Sono la
stessa cosa vista da due parti: ogni casella che cambia colore è uno scalino
che la pallina scende. E si vede la proprietà che rende la rete una memoria e
non un pasticcio: l'energia non risale mai. Scende quando una casella cambia,
e resta ferma quando la casella si guarda intorno e decide di stare com'è; nel
disegno si contano solo i sei cambi, perché sono i soli passi in cui succede
qualcosa. Che i cambi siano sei come le caselle rovinate non è una regola, è
il caso migliore: vuol dire che ogni casella sbagliata si è raddrizzata una
volta sola e che nessuna casella giusta si è mossa per sbaglio. Alla fine
nessuna casella vuole più cambiare, e quello è ciò che i tecnici chiamano un
**punto fisso**.

```{figure} ../figures/hopfield-ricorda.svg
:name: fig-hopfield-ricorda
:alt: A sinistra una griglia di cinque per cinque neuroni: parte da una lettera T con sei pixel invertiti, e a ogni passo un solo neurone si capovolge finché la T è ricomposta. A destra l'energia della rete, che a ogni aggiornamento scende a scatti e non risale mai: da −2,08 dello stato corrotto a −11,20 del ricordo richiamato, in sei aggiornamenti. Arrivata in fondo, la rete si ferma da sola.
:width: 92%

Sei aggiornamenti, un neurone alla volta: la T corrotta si ricompone e
l'energia scende a ogni passo, senza mai risalire, finché nessun neurone vuole
più cambiare.
```

Tre dettagli del codice meritano un'occhiata, perché sono la teoria in forma
eseguibile. Primo: nessuna casella è collegata a se stessa. Secondo: le
caselle si aggiornano *una alla volta*, in ordine casuale. Terzo: il ciclo si
ferma quando nessuna casella vuole più cambiare, cioè in fondo a una valle.

I primi due sono le due condizioni che garantiscono che l'energia non risalga
mai, insieme a una terza che sta nel modo in cui i legami sono costruiti, cioè
che il legame fra due caselle valga lo stesso nei due versi. L'intuizione sta
in poche righe.

Una casella cambia solo quando è in disaccordo con la spinta che riceve da
tutte le altre; se nel frattempo le altre non si sono mosse, il totale può
soltanto essere sceso. Se invece due caselle cambiassero insieme, ciascuna
avrebbe deciso credendo l'altra ferma, e la mossa buona per l'una potrebbe
rovinare quella dell'altra, esattamente come due persone che si scansano dallo
stesso lato: ecco perché una alla volta.

Il legame uguale nei due versi fa la sua parte in questo stesso conto: se
valesse cinque da una parte e meno due dall'altra, la casella che decide
vedrebbe un costo e il totale ne conterebbe un altro, e la sua mossa potrebbe
far salire l'energia pur sembrandole conveniente. E la casella scollegata da
se stessa serve a che, mentre decide, la spinta che sente non dipenda dalla
sua stessa posizione.

Poi c'è l'onestà statistica, che qui è più istruttiva della riuscita. Quella
stampa qui sopra viene da un unico sorteggio. Il 42 che compare nel codice è
il numero da cui parte il sorteggiatore: serve a far uscire sempre gli stessi
numeri «a caso», così che chi esegue il codice veda la stessa stampa, e
cambiandolo cambiano le sei caselle rovinate e cambia tutto il resto.
Rovinando le tre lettere in trentamila modi diversi ciascuna, il recupero
perfetto riesce il 92% delle volte: più dell'86% che si ottiene con tre
ricordi *presi a caso*, perché T, L e X sono state scelte apposta, e quella
scelta pesa più di quanto sembri.

Nelle altre la rete si ferma altrove, e non sempre dove ci si aspetterebbe. Su
dieci fallimenti, quasi otto finiscono in una conca a metà strada fra due
lettere, che nessuno ha mai memorizzato; poco più di uno in **un'altra
lettera**; e il resto nell'immagine capovolta di un'altra lettera.

Che le lettere capovolte compaiano è inevitabile, e capire perché aiuta:
scambiando acceso e spento dappertutto, le caselle che andavano d'accordo
continuano ad andarci, quindi ogni ricordo si porta dietro un gemello
capovolto, profondo esattamente uguale. Quello che non compare mai è il
gemello della lettera *da cui si è partiti*: in novantamila prove non capita
una volta sola, ed è troppo lontano perché capiti. Lo stato di partenza
differisce dalla lettera in sei caselle su venticinque, e quindi dalla sua
immagine capovolta in diciannove: la discesa non attraversa mezzo mondo. Il gemello è profondo uguale, ma sta
dall'altra parte, e una discesa che si muove una casella per volta non ci
arriva mai.

E c'è un punto in cui questa rete è più fortunata di quanto la teoria le
concederebbe. Le tre lettere non sono state pescate a caso: sono state scelte
in modo da somigliarsi il meno possibile, e si può misurare quanto. Per ogni
coppia di lettere si contano le caselle su cui concordano, si sottraggono
quelle su cui discordano, e del saldo si tiene solo il numero senza il segno,
perché anche un ricordo che è l'esatto opposto di un altro conta come una
sovrapposizione. Fra T, L e X viene in media 1,7 caselle su venticinque (3 fra
T e L, 1 nelle altre due coppie); fra tre disegni presi a caso ce ne si
aspettano 4,0, ed è un conto che si può rifare tirando a sorte tre griglie e
mediando. Meno della metà, dunque.

E questo conta, perché è proprio la somiglianza fra i ricordi a far fondere i
fianchi delle valli: la capienza di cui si diceva è calcolata su ricordi presi
a caso e su reti grandi, e qui i ricordi a caso non sono e la rete grande non
è. Quel 92%, allora, non è merito del codice: è merito della forma di questo
paesaggio, e la forma l'abbiamo scelta noi scegliendo le lettere.

`````{tab} Elementare
```{admonition} Da ricordare
:class: important
- Una **memoria associativa** si interroga con un frammento, non con un
  indirizzo: tre note fischiettate da un passante e la canzone riaffiora
  intera.
- Nella **rete di Hopfield** ogni ricordo memorizzato è una valle scavata nel
  paesaggio. L'indizio dice dove posare la pallina, la pallina rotola (può
  soltanto scendere) e il fondo in cui si ferma è il ricordo completo: la
  regola che scava le valli si limita a legare fra loro le caselle che nei
  ricordi vanno d'accordo.
- La **capienza** è limitata, e per le reti grandi si sa di quanto: circa il
  14% del numero di neuroni. Superata quella quota il richiamo non peggiora un
  poco alla volta, crolla tutto insieme.
- Su una rete piccola come la nostra quel 14% non si applica, e non c'è
  nessuna soglia netta: misurando si trova un peggioramento dolce (con tre
  ricordi presi a caso ne recupera l'86%, con quattro il 69%, con cinque il
  50%). Nel paesaggio compaiono anche conche a metà strada fra due ricordi,
  che nessuno ha mai memorizzato, e il gemello capovolto di ogni ricordo.
- La pallina finisce nella valle più *vicina*, non necessariamente in quella
  giusta. E la rete ricorda soltanto: non inventa, e può solo scendere. Sono i
  due limiti che la prossima sezione affronta con la temperatura e i neuroni
  nascosti.
```
`````

`````{tab} Superiore
```{admonition} Da ricordare
:class: important
- Una **memoria associativa** si interroga con un frammento, non con un
  indirizzo: il ricordo si completa da solo.
- Nella **rete di Hopfield** {cite}`hopfield1982neural` i ricordi sono minimi
  dell'energia $E(\mathbf{s}) = -\tfrac{1}{2}\, \mathbf{s}^\top \mathbf{W} \mathbf{s}$;
  la regola di Hebb scava le valli e l'aggiornamento asincrono (che non fa mai
  salire $E$) completa i ricordi corrotti scendendo nel minimo più vicino.
- La **capienza** è di circa il 14% del numero di neuroni
  ($\alpha_c \approx 0{,}14$) {cite}`amit1985storing`, e oltre soglia il
  richiamo non degrada: collassa. È però un risultato asintotico, per pattern
  casuali e non correlati: su reti piccole la transizione è sfumata e la
  degradazione dolce. Il paesaggio ospita anche minimi spuri, cioè ricordi che
  nessuno ha memorizzato.
- La rete *ricorda* ma non *inventa*, e può solo scendere: due limiti che la
  prossima sezione affronta con la temperatura e i neuroni nascosti.
```
`````
