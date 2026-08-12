# La memoria associativa di Hopfield

La memoria di un computer funziona per **indirizzo**: ogni dato abita in una
casella numerata, e per recuperarlo bisogna conoscere il numero esatto, sbagli
una cifra e ottieni un dato qualsiasi. La memoria umana funziona per
**contenuto**: bastano tre note stonate fischiettate da un passante per farti
riaffiorare l'intera canzone, un profumo per restituirti una cucina di
trent'anni fa, mezza faccia intravista da un autobus per completare nome e
cognome. Non forniamo indirizzi: forniamo *frammenti*, e il ricordo si
completa da solo. I tecnici la chiamano **memoria associativa**.

Nel 1982 John Hopfield (un fisico della materia condensata prestato alla
biologia, dal 1980 al California Institute of Technology) mostra come
costruirne una con neuroni artificiali {cite}`hopfield1982neural`. Non parte da
un foglio bianco, ed è giusto dirlo: memorie che si interrogano per contenuto
circolavano da un decennio, costruite legando fra loro i pezzi che nei ricordi
vanno d'accordo (Teuvo Kohonen, Kaoru Nakano, James Anderson e Shun-ichi Amari,
tutti nel 1972 {cite}`amari1972learning`), e già nel 1974 William Little aveva
descritto una rete di neuroni a soglia che si assesta in stati persistenti
{cite}`little1974existence`. Quello che Hopfield aggiunge, e che fa ripartire
il campo da lui, è l'**energia**: un solo numero associato a ogni
configurazione della rete, più la dimostrazione che la dinamica lo fa scendere.
Da quel momento i ricordi sono minimi, e ricordare è una discesa.

La sua mossa è quella di un fisico: notare che un gruppo di neuroni binari,
ciascuno «acceso» o «spento», collegati da **pesi** (numeri che dicono quanto
due neuroni tendono a stare d'accordo: positivo se preferiscono lo stesso
stato, negativo se preferiscono l'opposto) e per giunta **simmetrici** (il
legame fra due neuroni vale lo stesso nei due versi), è matematicamente
identico a un materiale magnetico in cui ogni atomo ha una freccina, lo
*spin*, che punta in su o in giù e sente l'influenza dei vicini. E di sistemi
così la fisica sa tutto, a partire dalla domanda giusta: qual è l'energia di
ogni configurazione, e verso dove scende?

```{figure} ../figures/energia-paesaggio.svg
:name: fig-energia-paesaggio
:alt: Paesaggio di energia con tre valli i cui minimi, segnati in teal, sono i ricordi memorizzati; una pallina ocra etichettata «ricordo parziale / rumoroso» parte da un punto alto e una freccia terracotta la accompagna nel fondo della valle più vicina. L'asse verticale è l'energia E, quello orizzontale lo stato della rete.
:width: 92%

Il paesaggio di energia di una rete di Hopfield. Le parole del disegno,
tradotte: i «pattern memorizzati» sono i ricordi, i «minimi» i fondovalle in
cui stanno, e lo stato «rumoroso» da cui parte la pallina è l'indizio rovinato
che diamo alla rete.
```

La {numref}`fig-energia-paesaggio` contiene, in un solo disegno, tutta l'idea:
i ricordi sono valli, lo stato attuale della rete è una pallina, e la fisica
del sistema (l'energia che può solo scendere) fa il lavoro di richiamo al
posto nostro.

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

Due avvertenze oneste. Primo: la pallina scende nella valle più *vicina*, non
necessariamente in quella *giusta*: un indizio troppo rovinato può depositarti
sul pendio sbagliato, e da lì si finisce nel ricordo sbagliato con la stessa
naturalezza. Secondo: il paesaggio ha una capienza. Se provi a scavare troppe
valli in poco spazio, i fianchi si fondono e compaiono conche a metà strada
tra due ricordi: «ricordi fantasma» che nessuno ha mai memorizzato. Una rete
con 25 neuroni, come quella che costruiremo tra poco, regge bene tre ricordi;
con quattro sbaglia già una volta su tre, e più se ne aggiungono più il
richiamo peggiora. In una rete grande il peggioramento non è graduale: fin
sotto una certa soglia funziona quasi sempre, appena sopra smette di
funzionare quasi del tutto.

`````

`````{tab} Superiore

La rete è un vettore di $N$ neuroni binari $s_i \in \{-1, +1\}$, collegati da
pesi **simmetrici** ($w_{ij} = w_{ji}$) e senza auto-connessioni
($w_{ii} = 0$). A ogni stato $\mathbf{s}$ è associata l'energia

$$
E(\mathbf{s}) = -\frac{1}{2}\, \mathbf{s}^\top \mathbf{W} \mathbf{s} = -\frac{1}{2} \sum_{i \neq j} w_{ij}\, s_i s_j,
$$

dove $\mathbf{W}$ è la matrice dei pesi e la somma percorre tutte le coppie di neuroni:
una coppia collegata da peso positivo abbassa l'energia quando i due neuroni
concordano, e la alza quando discordano (per pesi negativi vale l'opposto). La
dinamica è l'**aggiornamento asincrono**: si sceglie un neurone $i$, si
calcola il suo campo locale $h_i = \sum_j w_{ij} s_j$ e si pone
$s_i \leftarrow \operatorname{sign}(h_i)$, lasciando tutto il resto fermo. Se
il neurone cambia segno, l'energia varia di $\Delta E = -2\,|h_i| < 0$: **ogni
aggiornamento la fa scendere o la lascia invariata, mai salire**. Poiché gli
stati sono in numero finito ($2^N$) ed $E$ è limitata dal basso, la discesa
termina in un punto fisso, un minimo locale dell'energia. È qui che serve la
simmetria dei pesi: è lei a garantire che questa $E$ scenda a ogni
aggiornamento. Senza simmetria la garanzia cade, e alcune reti asimmetriche
si mettono davvero a girare in tondo, senza fermarsi mai.

Le valli si scolpiscono con la **regola di Hebb**, dal neuropsicologo Donald
Hebb che nel 1949 la propose per le sinapsi biologiche (l'idea che sarebbe poi
stata riassunta nello slogan «i neuroni che si attivano insieme si legano
insieme»). Per memorizzare i pattern $\boldsymbol{\xi}^1, \dots, \boldsymbol{\xi}^P$, ciascuno un
vettore di $\pm 1$:

$$
w_{ij} = \frac{1}{N} \sum_{\mu=1}^{P} \xi_i^{\mu}\, \xi_j^{\mu}
\qquad (i \neq j),
$$

dove $\xi_i^{\mu}$ è l'$i$-esimo bit del pattern $\mu$: ogni pattern rafforza
i legami tra i propri bit concordi, e così diventa (con alta probabilità, se i
pattern sono quasi ortogonali) un minimo locale di $E$. La capienza però è
limitata: l'analisi di meccanica statistica di Daniel Amit, Hanoch Gutfreund e
Haim Sompolinsky {cite}`amit1985storing`, con i metodi dei vetri di spin,
mostra che la memoria associativa esiste solo per $P < \alpha_c N$ con
$\alpha_c \simeq 0{,}138$, e che oltre quella soglia il recupero non degrada
dolcemente: collassa.

Vale la pena enunciare le ipotesi, perché sono ciò che rende quel numero un
teorema e non un'osservazione: pattern **casuali e non correlati**, limite
termodinamico $N \to \infty$, temperatura nulla, simmetria di replica, e una
tolleranza per una piccola frazione di bit errati nel richiamo. Fuori di lì il
numero va maneggiato con cura, e la rete della prossima sezione mostra quanto:
a $N = 25$ non c'è nessun limite termodinamico e la transizione è del tutto
sfumata. Misurando il richiamo con pattern casuali (2000 prove per punto, sei
bit invertiti su venticinque) si ottiene $0{,}86$ a $P = 3$, $0{,}70$ a
$P = 4$, $0{,}48$ a $P = 5$ e ancora $0{,}29$ a $P = 6$, cioè al doppio della
soglia: nessun crollo, una discesa regolare. Il collasso è un fenomeno di reti
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
nome del neuropsicologo Donald Hebb. Poi il codice rovina una lettera
invertendo sei caselle a caso (sei su venticinque, il 24%) e lascia che la
rete si aggiusti da sé, una casella alla volta, finché nessuna vuole più
cambiare.

Chi non programma può saltare direttamente al risultato stampato più sotto: il
codice fa esattamente quello che si è appena detto, e i commenti in italiano
dicono, riga per riga, a quale pezzo del discorso corrispondono.

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
non un pasticcio: l'energia scende a ogni aggiornamento e non risale mai,
finché nessuna casella vuole più cambiare, che è quello che i tecnici chiamano
un **punto fisso**.

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
caselle si aggiornano *una alla volta*, in ordine casuale, ed è questo a
garantire che l'energia non risalga mai. Terzo: il ciclo si ferma quando
nessuna casella vuole più cambiare, cioè in fondo a una valle, cioè su un
ricordo.

Poi c'è l'onestà statistica, che qui è più istruttiva della riuscita. Con
corruzioni casuali diverse da quella del seme fissato il recupero perfetto
riesce circa nove volte su dieci (misurato su trentamila prove: il 92%). Nelle
altre la rete si ferma altrove, e non sempre dove ci si aspetterebbe: quasi
nove fallimenti su dieci finiscono in un ricordo che nessuno ha memorizzato
(una conca a metà strada, oppure una lettera con tutti i pixel invertiti), ma
**circa uno su dieci finisce in un'altra lettera**, che è la valle sbagliata
di cui parla la scheda Elementare qui sopra.

E c'è un punto in cui questa rete è più fortunata di quanto la teoria le
concederebbe. Le tre lettere non sono state pescate a caso: sono state scelte
in modo da somigliarsi il meno possibile, e la loro somiglianza reciproca è
due volte e mezzo più bassa di quella tipica fra tre disegni casuali. A una
rete di Hopfield è proprio la somiglianza fra i ricordi a dare fastidio, e la
capienza di cui parla la scheda Superiore è calcolata su ricordi presi a caso
e su reti grandi. Il limite non è nel codice, è nella forma del paesaggio: e
la forma di questo paesaggio l'abbiamo scelta noi.

`````{tab} Elementare
```{admonition} Da ricordare
:class: important
- Una **memoria associativa** si interroga con un frammento, non con un
  indirizzo: tre note fischiettate da un passante e la canzone riaffiora
  intera.
- Nella **rete di Hopfield** ogni ricordo memorizzato è una valle scavata nel
  paesaggio. L'indizio dice dove posare la pallina, la pallina rotola (può
  soltanto scendere) e il fondo in cui si ferma è il ricordo completo: la
  regola che scava le valli si limita a legare fra loro i pixel che nei
  ricordi vanno d'accordo.
- La **capienza** è di circa il 14% del numero di neuroni: venticinque
  neuroni reggono tre o quattro ricordi. In una rete grande quella percentuale
  è una soglia netta, e appena la si supera il richiamo non peggiora un poco
  alla volta, crolla tutto insieme; in una piccola come la nostra è soltanto
  una tendenza, e il peggioramento è graduale. Nel paesaggio compaiono anche
  conche a metà strada fra due ricordi, che nessuno ha mai memorizzato.
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
  ($\alpha_c \simeq 0{,}138$) {cite}`amit1985storing`, e oltre soglia il
  richiamo non degrada: collassa. È però un risultato asintotico, per pattern
  casuali e non correlati: su reti piccole la transizione è sfumata e la
  degradazione dolce. Il paesaggio ospita anche minimi spuri, cioè ricordi che
  nessuno ha memorizzato.
- La rete *ricorda* ma non *inventa*, e può solo scendere: due limiti che la
  prossima sezione affronta con la temperatura e i neuroni nascosti.
```
`````
