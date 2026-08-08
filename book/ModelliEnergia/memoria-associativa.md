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
costruirne una con neuroni artificiali {cite}`hopfield1982neural`. La sua
mossa è quella di un fisico: notare che un gruppo di neuroni binari, ciascuno
«acceso» o «spento», collegati da pesi simmetrici, è matematicamente identico
a un materiale magnetico in cui ogni atomo ha uno spin che punta in su o in
giù e sente l'influenza dei vicini. E di sistemi così la fisica sa tutto, a
partire dalla domanda giusta: qual è l'energia di ogni configurazione, e verso
dove scende?

```{figure} ../figures/energia-paesaggio.svg
:name: fig-energia-paesaggio
:alt: Paesaggio di energia con tre valli i cui minimi, segnati in teal, sono i ricordi memorizzati; una pallina ocra etichettata «ricordo parziale / rumoroso» parte da un punto alto e una freccia terracotta la accompagna nel fondo della valle più vicina. L'asse verticale è l'energia E, quello orizzontale lo stato della rete.
:width: 92%

Il paesaggio di energia di una rete di Hopfield: ogni ricordo memorizzato è
una valle, e il richiamo è la discesa dallo stato corrotto al minimo più
vicino.
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
con 25 neuroni, come quella che costruiremo tra poco, regge tre o quattro
ricordi: oltre, va in confusione tutta insieme.

`````

`````{tab} Superiore

La rete è un vettore di $N$ neuroni binari $s_i \in \{-1, +1\}$, collegati da
pesi **simmetrici** ($w_{ij} = w_{ji}$) e senza auto-connessioni
($w_{ii} = 0$). A ogni stato $s$ è associata l'energia

$$
E(s) = -\frac{1}{2}\, s^\top W s = -\frac{1}{2} \sum_{i \neq j} w_{ij}\, s_i s_j,
$$

dove $W$ è la matrice dei pesi e la somma percorre tutte le coppie di neuroni:
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
insieme»). Per memorizzare i pattern $\xi^1, \dots, \xi^P$, ciascuno un
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
$\alpha_c \approx 0{,}14$ (il valore raffinato dagli sviluppi successivi è
$0{,}138$), tollerando una piccola frazione di bit errati nel richiamo. Oltre
quella soglia il recupero non degrada dolcemente: collassa. E anche sotto
soglia il paesaggio contiene minimi non richiesti: gli opposti $-\xi^{\mu}$ di
ogni pattern e miscele spurie di tre o più ricordi.

`````

## Una memoria che si ripara da sola, in poche righe

Tutto questo si può toccare con mano. Il codice che segue costruisce una rete
di Hopfield completa in NumPy: tre lettere stilizzate su una griglia 5×5
(quindi $N = 25$ neuroni), la regola di Hebb per i pesi, la funzione di
energia e il richiamo per aggiornamento asincrono a partire da una versione
corrotta, con il 24% dei pixel invertiti.

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
corrotto a $-11{,}20$ del pattern richiamato:

```text
T:  E = -2.08 -> -11.20  (recuperato)
   corrotto   richiamato
   #.###      #####
   ..#..      ..#..
   #.#.#      ..#..
   .##..      ..#..
   .###.      ..#..
```

Vale la pena notare tre dettagli del codice, perché sono la teoria in forma
eseguibile: la diagonale di $W$ è azzerata (niente auto-connessioni), i
neuroni si aggiornano *uno alla volta* in ordine casuale (la discesa asincrona
che non fa mai salire $E$), e il ciclo si ferma quando nessun neurone vuole
più cambiare: un minimo locale, cioè un ricordo. Onestà statistica: con tre
pattern su 25 neuroni siamo proprio al limite della capienza
$0{,}138 \times 25 \approx 3{,}4$; le tre lettere sono state scelte quasi
ortogonali tra loro, e con corruzioni casuali diverse dal seme fissato il
recupero perfetto riesce circa nove volte su dieci. Nelle altre, la pallina
finisce in un minimo spurio: il limite non è nel codice, è nella matematica.

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
  neuroni reggono tre o quattro ricordi, e oltre quella soglia il richiamo non
  peggiora un poco alla volta, crolla tutto insieme. Nel paesaggio compaiono
  anche conche a metà strada fra due ricordi, che nessuno ha mai memorizzato.
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
  dell'energia $E(\mathbf{s}) = -\tfrac{1}{2}\, \mathbf{s}^\top W \mathbf{s}$;
  la regola di Hebb scava le valli e l'aggiornamento asincrono (che non fa mai
  salire $E$) completa i ricordi corrotti scendendo nel minimo più vicino.
- La **capienza** è di circa il 14% del numero di neuroni
  ($\alpha_c \approx 0{,}138$) {cite}`amit1985storing`, e oltre soglia il
  richiamo non degrada: collassa. Il paesaggio ospita anche
  minimi spuri, cioè ricordi che nessuno ha memorizzato.
- La rete *ricorda* ma non *inventa*, e può solo scendere: due limiti che la
  prossima sezione affronta con la temperatura e i neuroni nascosti.
```
`````
