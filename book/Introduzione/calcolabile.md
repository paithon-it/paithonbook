# Che cosa si può calcolare

Nel settembre del 1930 a Königsberg si tengono più convegni negli stessi
giorni. Il 7, nella discussione finale di un incontro sui fondamenti delle
scienze esatte, un logico dell'Università di Vienna di ventiquattro anni, Kurt
Gödel, annuncia di passaggio un risultato che quasi nessuno dei presenti
raccoglie (lo capisce al volo il matematico John von Neumann). Il giorno dopo,
nella stessa città, David Hilbert, il matematico più influente del suo tempo,
chiude un discorso trasmesso per radio con una frase che oggi è incisa sulla
sua tomba a Gottinga: «Dobbiamo sapere, sapremo». Il risultato di Gödel,
annunciato il giorno prima, diceva che quella promessa non si può mantenere del
tutto: qualunque insieme di regole si scelga per dimostrare le cose dei numeri,
purché non si contraddica, resteranno affermazioni vere che quelle regole non
dimostrano.

Hilbert chiedeva, fra le altre cose, un procedimento meccanico capace di
decidere se un'affermazione scritta nel linguaggio della logica segue dagli
assiomi, le verità di partenza. Un'affermazione come «ogni numero pari maggiore
di due è la somma di due numeri primi», per dire: il procedimento avrebbe
dovuto rispondere sì o no, sempre. Lo chiamava il problema della decisione,
l’*Entscheidungsproblem*, e lo aveva posto con Wilhelm Ackermann nel 1928.
Rispondere di no chiedeva prima di dire che cosa sia un procedimento meccanico,
perché per sostenere che nessun procedimento ci riesce bisogna sapere quali
sono tutti; cioè bisognava dare una definizione precisa dell'algoritmo,
descritto finora come una ricetta sull'esempio di {doc}`Euclide
</Introduzione/overview>`. La diede Alan Turing nel 1936, con una macchina
immaginaria {cite}`turing1936computable`; quasi negli stessi mesi Alonzo Church
ne diede un'altra, equivalente {cite}`church1936unsolvable`. Da allora si sa
che cosa una macchina può calcolare, e soprattutto che cosa non può: ed è il
fondo su cui poggia tutto quello che segue, perché un modello che impara resta
un programma, e i limiti dei programmi sono anche i suoi.

## Una macchina con un nastro e una matita

`````{tab} Elementare

Turing non pensava a un calcolatore, che nel 1936 non esisteva. Pensava a una
persona che fa un conto con carta e matita, e si chiedeva che cosa stia
facendo davvero. Scrive e cancella simboli su un foglio, guarda un pezzetto di
foglio alla volta, e decide che cosa fare dopo in base a quello che vede e a
quello che ha in mente in quel momento. Tutto il resto, osservava, è superfluo.

La sua macchina fa esattamente questo, ridotto all'osso. C'è un nastro lungo
quanto serve, diviso in caselle, e in ogni casella c'è un simbolo o niente.
C'è una testina che guarda una casella sola. Quello che la persona «ha in
mente» diventa lo *stato* della macchina, indicato con una parola. E c'è una
tabellina di regole, fatte tutte allo stesso modo: «se sei nello stato *vai* e
leggi un 1, scrivi 1, spostati di una casella a destra e resta nello stato
*vai*». Quando per la
situazione in cui si trova non c'è nessuna regola, la macchina si ferma.

Con sei regole di questo tipo la macchina somma uno a un numero scritto in
cifre binarie, cioè con i soli 0 e 1, dove ogni cifra vale il doppio di quella
alla sua destra: $1011$ è otto, più zero, più due, più uno, cioè undici
({numref}`fig-nastro-e-testina`). È l'addizione in colonna: si va all'ultima
cifra a destra, e se sommando uno viene due (in binario $10$) si scrive 0 e si
porta uno. La macchina, che vede una casella alla volta, l'ultima cifra deve
andarla a cercare. Le regole, dette a parole, sono queste. La testina parte
sulla prima cifra a sinistra, nello stato *vai*. Nello stato *vai* la macchina
riscrive la cifra che legge (uno 0 o un 1) e va a destra; quando trova la
casella bianca torna indietro di una e passa allo stato *riporto*. Nello stato
*riporto* un 1 diventa 0 e si va a sinistra; uno 0, o una casella bianca,
diventa 1 e la macchina passa allo stato *fine*, per il quale non c'è nessuna
regola, e si ferma. Da $1011$, undici, arriva così a $1100$, dodici, in otto
passi.

Il passo che rende l'idea potente è un altro. Le regole di una macchina si
possono scrivere sul nastro, come simboli fra gli altri, e Turing costruì una
macchina che legge quelle regole e le esegue: una macchina **universale**, che
imita qualunque altra macchina di cui le si dia la descrizione. Le sue regole
sono poche e fisse, e dicono soltanto come leggere le regole scritte sul nastro
e applicarle. È l'idea del
calcolatore che fa girare programmi diversi senza cambiare un filo, più di
dieci anni prima che un calcolatore così venisse costruito.

Resta una domanda: questa macchina così povera calcola davvero tutto quello che
si può calcolare con un procedimento? I linguaggi di programmazione di oggi,
Python compreso, sanno fare esattamente le stesse cose, né una di più né una di
meno, e questo si dimostra: ciascuno sa imitare la macchina, e la macchina sa
imitare ciascuno. Che non esista un procedimento che nessuno dei due sappia
imitare, invece, non si dimostra, perché «procedimento» è una parola
dell'esperienza, non della matematica: nessuno ne ha mai trovato uno, ed è una
constatazione. Di un sistema che sa imitare una
macchina di Turing si dice che è **Turing-completo**, con una clausola che
conta: purché gli si dia tutta la memoria che chiede. Un calcolatore vero ne ha
una quantità finita, e davanti a un calcolo abbastanza lungo si ferma per
esaurimento dove la macchina di Turing, con il suo nastro senza fine, andrebbe
avanti.

`````

`````{tab} Superiore

Una macchina di Turing è una tupla $M=(Q,\Gamma,\sqcup,\delta,q_0,F)$: un
insieme finito di stati $Q$, un alfabeto finito di nastro $\Gamma$ con il
simbolo bianco $\sqcup\in\Gamma$, lo stato iniziale $q_0$, gli stati di arresto
$F\subseteq Q$ e la funzione di transizione (parziale)

$$
\delta : (Q\setminus F)\times\Gamma \to Q\times\Gamma\times\{-1,0,+1\},
$$

che a ogni coppia (stato, simbolo letto) associa il nuovo stato, il simbolo da
scrivere e lo spostamento della testina. Una *configurazione* è la terna
(stato, contenuto del nastro, posizione), e un passo di calcolo la trasforma
secondo $\delta$; $M$ si arresta su $x$ se da $(q_0, x, 0)$ raggiunge una
configurazione in cui $\delta$ non è definita. Una funzione è *calcolabile* se
una macchina, partendo dalla codifica di $x$, si arresta con la codifica di
$f(x)$ sul nastro. Nell'articolo del 1936, ricevuto dalla London Mathematical
Society il 28 maggio e letto il 12 novembre, la prima macchina d'esempio ha
quattro stati (Turing li chiama *m-configurazioni* e li scrive in fraktur:
$\mathfrak{b}$, $\mathfrak{c}$, $\mathfrak{e}$, $\mathfrak{k}$) e stampa la
successione $0\,1\,0\,1\ldots$ senza fermarsi mai.

Il risultato decisivo è la **macchina universale** $U$: codificata ogni
macchina come stringa $\langle M\rangle$, esiste $U$ tale che
$U(\langle M\rangle, x)$ si arresta se e solo se $M(x)$ si arresta, e in quel
caso con la stessa uscita. Il programma diventa un dato, ed è lo schema
concettuale del calcolatore a programma memorizzato.

La **tesi di Church-Turing** afferma che le funzioni calcolabili da un
procedimento effettivo sono esattamente quelle calcolabili da una macchina di
Turing, o equivalentemente $\lambda$-definibili (Church) o ricorsive generali
(Gödel, Herbrand, Kleene). Il nome «tesi di Church» è di Kleene (1952), quello
composto compare nel suo testo del 1967. Non è un teorema, perché uno dei due
lati è informale; a suo favore stanno le equivalenze dimostrate fra tutti i
formalismi proposti. Un modello è **Turing-completo** se simula ogni macchina di
Turing; per un calcolatore fisico, che ha memoria finita, la simulazione vale
fino all'esaurimento della memoria, e per un modello a precisione finita
l'affermazione chiede cautela (è il caso del {doc}`Transformer
</Transformers/tendenzefuture>`, la cui completezza è dimostrata sotto ipotesi
di precisione e lunghezza illimitate).

`````

```{figure} ../figures/nastro-e-testina.svg
:name: fig-nastro-e-testina
:alt: "Animazione: un nastro di caselle con scritto 1011; una testina a triangolo si sposta di casella in casella verso destra fino alla prima casella bianca, poi torna indietro trasformando i due 1 finali in 0 e lo 0 in 1. Sotto il nastro il nome dello stato cambia da vai a riporto a fine. Alla fine sul nastro c'è 1100 e la testina è ferma sulla seconda casella."
:width: 90%

La macchina che somma uno: sei regole, due stati di lavoro e uno di arrivo.
Va fino in fondo al numero, poi torna indietro portandosi dietro il riporto, e
si ferma quando lo lascia su uno 0. Da $1011$ (undici) a $1100$ (dodici) in
otto passi.
```

Le sei regole della figura stanno nella tabellina `incremento`, una per riga:
a sinistra lo stato e il simbolo letto, a destra il simbolo da scrivere, lo
spostamento (`+1` a destra, `-1` a sinistra, `0` fermo) e lo stato nuovo. La
funzione `esegui`, in cima, le applica finché ne trova una, e quello che conta
è che cosa stampa. Dopo la macchina che somma uno gira la prima macchina
dell'articolo di Turing, che non si ferma mai.

```python
def esegui(regole, nastro, stato, passi_max=1000, mostra=False):
    """Una macchina di Turing. `regole` dice, per ogni coppia (stato, simbolo
    letto), che cosa scrivere, dove spostare la testina (-1 a sinistra, +1 a
    destra, 0 ferma) e in quale stato passare. Senza una regola la macchina si
    ferma. Restituisce il nastro, i passi fatti e se si è fermata."""
    caselle = dict(enumerate(nastro))            # le caselle scritte; il resto è bianco
    testina = 0
    leggi = lambda: "".join(caselle.get(i, " ")
                            for i in range(min(caselle, default=0),
                                           max(caselle, default=-1) + 1)).strip()
    for passo in range(passi_max):
        letto = caselle.get(testina, " ")
        if (stato, letto) not in regole:
            return leggi(), passo, True
        caselle[testina], sposta, stato = regole[(stato, letto)]
        testina += sposta
        if mostra:
            print(f"  dopo il passo {passo + 1}: {leggi():<5} stato {stato}")
    return leggi(), passi_max, False

# aggiungere uno a un numero scritto in binario: si va in fondo, poi si torna
# indietro cambiando gli 1 in 0 finché non si trova uno 0 o una casella bianca
incremento = {
    ("vai", "0"): ("0", +1, "vai"),
    ("vai", "1"): ("1", +1, "vai"),
    ("vai", " "): (" ", -1, "riporto"),
    ("riporto", "1"): ("0", -1, "riporto"),
    ("riporto", "0"): ("1", 0, "fine"),
    ("riporto", " "): ("1", 0, "fine"),
}
nastro, passi, fermata = esegui(incremento, "1011", "vai", mostra=True)
print(f"1011 + 1 = {nastro}, in {passi} passi")
print(f"111 + 1 = {esegui(incremento, '111', 'vai')[0]}")

# la prima macchina dell'articolo di Turing (1936): quattro stati, b, c, e, k,
# che stampano 0 e 1 alternati lasciando una casella vuota fra una cifra e l'altra
macchina_I = {
    ("b", " "): ("0", +1, "c"),
    ("c", " "): (" ", +1, "e"),
    ("e", " "): ("1", +1, "k"),
    ("k", " "): (" ", +1, "b"),
}
nastro, passi, fermata = esegui(macchina_I, "", "b", passi_max=16)
print(f"macchina I dopo {passi} passi: {nastro!r}; si è fermata? {fermata}")
```

```text
  dopo il passo 1: 1011  stato vai
  dopo il passo 2: 1011  stato vai
  dopo il passo 3: 1011  stato vai
  dopo il passo 4: 1011  stato vai
  dopo il passo 5: 1011  stato riporto
  dopo il passo 6: 1010  stato riporto
  dopo il passo 7: 1000  stato riporto
  dopo il passo 8: 1100  stato fine
1011 + 1 = 1100, in 8 passi
111 + 1 = 1000
macchina I dopo 16 passi: '0 1 0 1 0 1 0 1'; si è fermata? False
```

La seconda macchina, lasciata andare, scriverebbe per sempre. Per lei lo si vede
leggendo le sue quattro regole: ciascuna scrive una cifra o salta una casella,
va a destra e passa allo stato successivo, e dall'ultimo si torna al primo, in
cerchio, senza mai arrivare in uno stato senza uscita. Per una macchina
qualunque, invece, sapere se si fermerà è una domanda a cui nessun programma sa
rispondere.

## La domanda che nessun programma risolve

L'algoritmo di Euclide si ferma sempre, e lo si dimostra: il secondo numero
scende a ogni passo e non può scendere per sempre. Il **problema della
fermata** chiede se esista un procedimento che, data una macchina e il suo
ingresso (i dati da cui parte), dica sempre se quella macchina si fermerà.

`````{tab} Elementare

Mettiamo che qualcuno sostenga di aver scritto un programma, chiamiamolo il
Profeta, che risponde a questa domanda. Gli si dà un programma qualunque e il
suo ingresso, e lui risponde «si ferma» oppure «gira per sempre», senza
sbagliare mai.

Allora si può scrivere un secondo programma, il Dispetto, che fa una cosa sola.
Un programma, come si è visto con la macchina universale, si può scrivere sul
nastro e dare in pasto a un altro come se fosse un dato. Il Dispetto prende un
programma, chiede al Profeta che cosa farebbe quel programma se gli si desse in
pasto la sua stessa descrizione, e fa il contrario: se il Profeta dice «si
ferma», il Dispetto entra in un giro senza fine; se dice «gira per sempre», il
Dispetto si ferma subito. È costruito apposta per smentire il Profeta, ed è
tutto il trucco: una domanda su un programma che fa l'opposto di qualunque
risposta. Finora, però, il Profeta non ha sbagliato niente: il colpo arriva
adesso.

Adesso si dia al Dispetto la sua stessa descrizione. Se il Profeta dice che il
Dispetto si ferma, il Dispetto gira per sempre; se dice che gira per sempre, il
Dispetto si ferma. In tutti e due i casi il Profeta ha sbagliato. E il trucco
funziona con qualunque Profeta: chiunque ne scriva uno, gli si costruisce il
suo Dispetto. Quindi un Profeta che non sbaglia mai non può esistere. Non è
questione di programmatori poco bravi: nessuno potrà mai scriverlo.

Ed è la risposta alla domanda di Hilbert. Se esistesse il procedimento che
decide se un'affermazione segue dagli assiomi, basterebbe scrivergli
l'affermazione «questa macchina, con questo ingresso, si ferma» e chiederglielo:
sarebbe il Profeta, e il Profeta non esiste.

Non vuol dire che non si sappia mai se un programma si ferma. Metà della
domanda, anzi, è facile: se il programma si ferma, basta lasciarlo girare e
prima o poi lo si vede. Il guaio è l'altra metà, perché «non si è ancora
fermato» non vuol dire «non si fermerà mai». Per Euclide lo si sa in un altro
modo, con una dimostrazione. Il Dispetto dice che non esiste un metodo solo che
funzioni per tutti, e ci sono programmi di cinque righe per cui nessuno lo sa.
Uno lo si scrive a mente: prendi un numero, se è pari dimezzalo, se è dispari
triplicalo e aggiungi uno, e ripeti finché arrivi a 1. Partendo da $27$ ci
vogliono $111$ passi, salendo fino a $9232$ prima di scendere; nessuno ha mai
trovato un numero da cui non si arrivi a 1, e nessuno ha mai dimostrato che non
esista.

La conseguenza tocca anche i programmi che imparano. Qualunque domanda
interessante su che cosa farà un programma, non su come è scritto («dirà mai
una parolaccia?» è una domanda su che cosa farà, «quante righe ha?» su come è
scritto), ha lo stesso destino: non esiste un controllore universale che la
risolva per tutti i programmi. Si possono controllare casi particolari, e lo si
fa, ma la garanzia generale non c'è.

`````

`````{tab} Superiore

Sia $\mathrm{HALT}=\{\langle M,x\rangle : M \text{ si arresta su } x\}$.
**Teorema** (Turing, 1936, nella forma moderna): $\mathrm{HALT}$ non è
decidibile. Se una macchina $H$ decidesse $\mathrm{HALT}$, la macchina $D$ che
su $\langle M\rangle$ esegue $H(\langle M,\langle M\rangle\rangle)$ e poi entra
in un ciclo infinito se la risposta è «si arresta», e si arresta altrimenti,
darebbe su $\langle D\rangle$ la contraddizione $D(\langle D\rangle)\downarrow
\iff D(\langle D\rangle)\uparrow$. È la diagonale di Cantor: le macchine si
enumerano, e $D$ differisce dalla $n$-esima sul suo stesso indice.
$\mathrm{HALT}$ è però *semidecidibile* (ricorsivamente enumerabile): basta
eseguire $M$ su $x$ e rispondere quando si arresta. Il complemento non lo è.

Turing formulò il risultato in altri termini: chiama *circolare* una macchina
che stampa solo un numero finito di cifre (perché si blocca, o perché continua
a muoversi senza stamparne più) e dimostra che nessuna macchina decide, data
una descrizione, se è quella di una macchina non circolare; la forma con
l'arresto è di Kleene (1952), e il nome *halting problem* di Davis (1958).
Dall'insolubilità segue la risposta negativa all’*Entscheidungsproblem*: una
formula del primo ordine può codificare «$M$ si arresta su $x$», e un
procedimento che decidesse la validità delle formule deciderebbe
$\mathrm{HALT}$.

Il **teorema di Rice** {cite}`rice1953classes` generalizza: ogni proprietà non
banale della funzione calcolata da un programma (non del suo testo) è
indecidibile. Non esiste quindi un verificatore generale del comportamento di un
programma. Dall'indecidibilità della fermata segue anche che non è calcolabile
la {doc}`complessità di Kolmogorov </AutoSupervisione/capire-e-accorciare>`, la
lunghezza del programma più corto che produce una stringa. Che l'arresto sia una
domanda difficile anche su programmi minuscoli lo mostra la congettura di
Collatz, ancora aperta: l'iterazione $n\mapsto n/2$ per $n$ pari,
$n\mapsto 3n+1$ per $n$ dispari, raggiunge $1$ da ogni intero positivo? Al
calcolatore è verificata per tutti gli interi fino a $2^{68}$
{cite}`barina2021convergence`, e nessuna dimostrazione.

`````

Ecco quel conto per qualche numero di partenza, e la ricerca, sotto
diecimila, di quello che ci mette di più.

```python
def collatz(n):
    """Pari: dimezza. Dispari: triplica e aggiungi uno. Fino a 1, se ci arriva."""
    passi, massimo = 0, n
    while n != 1:
        n = n // 2 if n % 2 == 0 else 3 * n + 1
        passi, massimo = passi + 1, max(massimo, n)
    return passi, massimo

for n in (6, 7, 27):
    passi, massimo = collatz(n)
    print(f"partendo da {n}: arriva a 1 in {passi} passi, salendo fino a {massimo}")
record = max(range(1, 10_000), key=lambda n: collatz(n)[0])
print(f"sotto 10 000 il cammino più lungo parte da {record}: {collatz(record)[0]} passi")
```

```text
partendo da 6: arriva a 1 in 8 passi, salendo fino a 16
partendo da 7: arriva a 1 in 16 passi, salendo fino a 52
partendo da 27: arriva a 1 in 111 passi, salendo fino a 9232
sotto 10 000 il cammino più lungo parte da 6171: 261 passi
```

Quello che si può fare è controllare, un numero alla volta, e ogni volta si
arriva a 1. Nessuna quantità di controlli di questo tipo diventa una
dimostrazione.

## Una frase che parla di sé

Il risultato annunciato a Königsberg riguarda i sistemi formali: un elenco di
affermazioni di partenza, gli assiomi, e di regole con cui se ne ricavano altre,
i teoremi, in modo meccanico. Il gioco MIU che segue ne è un esempio in
miniatura, con un solo assioma e quattro regole. Il **teorema di incompletezza**
di Gödel {cite}`godel1931formal` dice che in ogni sistema del genere abbastanza
ricco da contenere l'aritmetica, e che non si contraddica, esiste
un'affermazione vera che il sistema non riesce a dimostrare.

`````{tab} Elementare

Douglas Hofstadter, nel libro *Gödel, Escher, Bach*, fa giocare il lettore con
un sistema minuscolo {cite}`hofstadter1979godel`. Ci sono tre lettere, M, I e
U, una stringa di partenza, MI, e quattro regole per produrne di nuove: se una
stringa finisce per I le si può attaccare una U; quello che segue la M si può
raddoppiare (da MIU si passa a MIUIU); tre I di fila si possono sostituire con
una U; due U di fila si possono cancellare. La domanda è: si può arrivare a MU?

Si può provare a giocare per ore, e MU non arriva mai. Ma giocare non basta a
concludere niente, perché la mossa giusta potrebbe essere la prossima. La
risposta arriva uscendo dal gioco e guardando le regole da fuori: si contano le
I. MI ne ha una. La prima e la quarta regola non le toccano, la seconda ne
raddoppia il numero, e il doppio di un numero è multiplo di tre solo se lo era
già; la terza ne toglie tre, e togliere tre non cambia il resto della divisione
per tre. Partendo da una, nessuna di queste mosse porta mai a un numero di I
che sia multiplo di tre, e MU ne ha zero, che è
un multiplo di tre. Quindi MU non si raggiunge. Una quantità che nessuna regola
cambia, come qui il resto del numero di I diviso tre, si chiama **invariante**.

Gödel fa una cosa simile, ma dentro il gioco invece che fuori. Assegna a ogni
simbolo della matematica un numero, e a ogni frase il numero che si ottiene
mettendo insieme quelli dei suoi simboli, in modo che dal numero si possa
sempre risalire alla frase, come dal numero di una scheda si risale al libro
in una biblioteca. A quel punto l'aritmetica, che parla di numeri, può parlare
anche delle proprie frasi e delle proprie dimostrazioni: «la frase numero 12 si
ricava dalla numero 5» diventa una relazione fra i numeri 12 e 5, che
l'aritmetica sa scrivere. E Gödel costruisce una frase che, letta attraverso
questa numerazione, dice: «la frase numero tale non si può dimostrare», dove
quel numero è il suo. Che una frase possa contenere il proprio numero sembra un
trucco, ed è la parte tecnica del lavoro di Gödel: si può fare.

È il paradosso del mentitore («questa frase è falsa») con una parola cambiata, e
la parola cambiata salva tutto dalla contraddizione. Se la frase si potesse
dimostrare, il sistema, che sa controllare le proprie dimostrazioni,
dimostrerebbe anche che la frase si può dimostrare, cioè il contrario di quello
che la frase dice: si contraddirebbe. Quindi, se il sistema non si contraddice,
la frase non si dimostra; e siccome dice proprio questo, è vera. Una verità che
il sistema non raggiunge. Aggiungerla alle regole non aiuta, perché il sistema
nuovo ha la sua frase nuova. Hofstadter chiama questo tornare su sé stessi,
salendo di livello e trovandosi al punto di partenza, uno **strano anello**.

Gödel dimostrò anche una seconda cosa, che servirà fra poco. Un sistema
abbastanza ricco, se non si contraddice, non riesce a dimostrare di non
contraddirsi: la garanzia della propria coerenza deve venire da fuori.

`````

`````{tab} Superiore

Sia $T$ una teoria del primo ordine con assiomi ricorsivamente enumerabili che
interpreti l'aritmetica di Robinson. La *numerazione di Gödel* associa a ogni
simbolo un intero e a ogni sequenza di simboli $s_1\dots s_k$ il numero
$\prod_{i=1}^{k} p_i^{\,\#s_i}$, con $p_i$ l’$i$-esimo primo; per l'unicità
della fattorizzazione la codifica è invertibile, e una dimostrazione, sequenza
di formule, si codifica allo stesso modo. Il predicato «$y$ è il codice di una
dimostrazione della formula di codice $x$» è allora rappresentabile in $T$ da
una formula $\mathrm{Prf}_T(y,x)$, e
$\mathrm{Prov}_T(x)\equiv\exists y\,\mathrm{Prf}_T(y,x)$. Il **lemma di
diagonalizzazione** dà, per ogni formula $\varphi(x)$, un enunciato $G$ con
$T\vdash G\leftrightarrow\varphi(\ulcorner G\urcorner)$; con
$\varphi(x)=\neg\mathrm{Prov}_T(x)$ si ottiene l'enunciato di Gödel.

**Primo teorema di incompletezza.** Se $T$ è coerente, $T\nvdash G$; se $T$ è
$\omega$-coerente, anche $T\nvdash\neg G$. Gödel dimostrò il teorema con
l'ipotesi di $\omega$-coerenza; Rosser (1936) la ridusse alla sola coerenza con
un enunciato modificato. **Secondo teorema.** Se $T$ è coerente, non dimostra
l'enunciato $\mathrm{Con}_T$ che formalizza la propria coerenza, purché
$\mathrm{Prov}_T$ sia costruito nel modo naturale, cioè soddisfi le condizioni
di derivabilità di Hilbert, Bernays e Löb. Gödel stesso, nell'introduzione
dell'articolo, nota l'analogia con le antinomie di Richard e del mentitore; la
differenza è che «vero» non è definibile nell'aritmetica (Tarski), mentre
«dimostrabile» lo è, e per questo l'antinomia diventa un teorema.
L'incompletezza, nella forma «esiste un enunciato vero che $T$ non dimostra»,
segue anche dall'insolubilità della fermata, e per questa via basta la coerenza.
Se ogni enunciato vero della forma «$M$ non si arresta su $x$» fosse
dimostrabile in $T$, si deciderebbe $\mathrm{HALT}$ simulando $M$ su $x$ e
intanto enumerando le dimostrazioni di $T$ in cerca di quella negazione: se $M$
si arresta lo dice la simulazione, se non si arresta la dimostrazione prima o
poi arriva. E la risposta sarebbe giusta, perché quando $M$ si arresta
l'enunciato che lo afferma è $\Sigma_1$ e vero, quindi dimostrabile
nell'aritmetica di Robinson e in $T$, e una $T$ coerente non ne dimostra anche
la negazione.

Il sistema MIU di Hofstadter illustra il gesto metamatematico in piccolo: la
domanda «MU è un teorema?» non si risolve derivando, ma con un invariante sul
sistema. Il numero di I modulo $3$ vale $1$ nell'assioma; la regola che
raddoppia manda $1\mapsto 2$ e $2\mapsto 1$, quella che toglie tre I lo
conserva, le altre non lo toccano; resta quindi diverso da zero in ogni
teorema, e MU, che ne ha zero, non lo è.

`````

Al gioco MIU una macchina gioca nell'unico modo che conosce: produce
tutte le stringhe che le regole raggiungono senza mai superare dodici lettere, e
poi guarda il resto del numero di I diviso tre.

```python
# il gioco MIU di Hofstadter: si parte da MI e si applicano quattro regole
def figli(s):
    if s.endswith("I"):
        yield s + "U"                            # regola I:   xI  -> xIU
    yield s + s[1:]                              # regola II:  Mx  -> Mxx
    for i in range(len(s) - 2):
        if s[i:i + 3] == "III":
            yield s[:i] + "U" + s[i + 3:]        # regola III: III -> U
    for i in range(len(s) - 1):
        if s[i:i + 2] == "UU":
            yield s[:i] + s[i + 2:]              # regola IV:  UU  -> niente

teoremi, frontiera = {"MI"}, ["MI"]
while frontiera:
    nuovi = [t for s in frontiera for t in figli(s) if len(t) <= 12 and t not in teoremi]
    teoremi.update(nuovi)
    frontiera = list(dict.fromkeys(nuovi))
print(f"stringhe ottenute da MI, lunghe al più 12: {len(teoremi)}")
print(f"MU fra queste? {'MU' in teoremi}")
resti = {t.count('I') % 3 for t in teoremi}
print(f"resti del numero di I diviso 3, su tutte: {sorted(resti)}")
```

```text
stringhe ottenute da MI, lunghe al più 12: 216
MU fra queste? False
resti del numero di I diviso 3, su tutte: [1, 2]
```

La ricerca non trova MU fra le 216 stringhe, e da sola non potrebbe dire di
più: una stringa più lunga di dodici lettere, accorciata con le regole III e IV,
potrebbe ancora arrivarci. A chiudere la questione è l'invariante: la riga
dopo mostra che su quelle 216 stringhe il resto non vale mai zero, e il
ragionamento sulle regole dice perché non lo varrà mai, su nessuna stringa. La
ricerca ha visto, l'invariante ha dimostrato.

## Che cosa non se ne ricava

Nel 1961 il filosofo John Lucas sostenne che il teorema di Gödel dimostra che
la mente umana non è una macchina {cite}`lucas1961minds`, e Roger Penrose ha
ripreso l'argomento in due libri. Il ragionamento si presenta rigoroso, e per
questo è fra gli usi sbagliati più diffusi del teorema.

`````{tab} Elementare

L'argomento di Lucas suona così. Se la mente fosse una macchina, sarebbe un
sistema di regole. Quel sistema avrebbe la sua frase di Gödel, che il sistema
non riesce a dimostrare. Ma noi, guardandola, vediamo che è vera. Quindi
facciamo una cosa che la macchina non fa, e non siamo quella macchina.

Il punto debole sta in «vediamo che è vera». La frase di Gödel è vera solo se il
sistema non si contraddice mai; se il sistema si contraddicesse, dimostrerebbe
qualunque cosa (da una contraddizione, nella logica, segue qualunque
affermazione), e il ragionamento cadrebbe. Per «vedere» che la frase è vera,
quindi, bisogna sapere che il sistema è senza contraddizioni. E il secondo
risultato di Gödel dice proprio che un sistema abbastanza ricco non riesce a
dimostrarlo di sé stesso. Se la mente fosse un sistema di regole complicato come
un cervello, nemmeno lei saprebbe dimostrare di non contraddirsi mai, e chi
conosce un po' le persone ha buone ragioni per dubitarne.

Turing aveva già risposto nel 1950, prima di Lucas. Che ogni singola macchina
abbia dei limiti è dimostrato; che la mente umana non ne abbia è stato soltanto
affermato. E chi trova la domanda che mette in difficoltà una macchina non ha
battuto tutte le macchine: ce ne può essere un'altra che quella domanda la
supera, e ne avrà una sua.

`````

`````{tab} Superiore

Lucas: se una mente $\mathcal{M}$ fosse una macchina, sarebbe descritta da un
sistema formale $T_{\mathcal{M}}$; la mente riconosce come vero
$G_{T_{\mathcal{M}}}$, che $T_{\mathcal{M}}$ non dimostra; dunque la mente
eccede ogni macchina. Il passaggio che non regge è il secondo: $G_T$ è vero se
$T$ è coerente, e l'argomento richiede che la mente sappia che
$T_{\mathcal{M}}$ è coerente. Per il secondo teorema, se $T_{\mathcal{M}}$ è
coerente non dimostra $\mathrm{Con}_{T_{\mathcal{M}}}$; l'argomento presuppone
quindi che la mente conosca la coerenza di un sistema di complessità arbitraria
che la descrive, cioè presuppone la conclusione. È l'obiezione di Putnam
{cite}`putnam1960minds`, e Benacerraf {cite}`benacerraf1967god` ne ha dato la
forma più discussa: al più, se la mente è una macchina, non può sapere quale.

La replica di Turing sta nell'articolo del 1950 {cite}`turing1950computing`,
fra le obiezioni al gioco dell'imitazione (la terza, «matematica»): i limiti di
ogni macchina *particolare* sono dimostrati, quelli dell'intelletto umano sono
stati soltanto affermati, e quando una macchina sbaglia la domanda critica
«potrebbero esserci uomini più bravi di qualunque macchina data, ma potrebbero
anche esserci altre macchine più brave ancora». Hofstadter, in *Gödel, Escher,
Bach*, giudica l'argomentazione di Lucas «erronea», ma in un modo che trova
«affascinante».
Quello che i teoremi limitativi dicono è che nessun sistema formale coerente e
abbastanza ricco decide tutta l'aritmetica, e che nessun programma decide il
comportamento di tutti i programmi; nulla, in essi, separa le menti dalle
macchine.

`````

Di questi teoremi sui limiti, che si chiamano teoremi limitativi, resta anche
una lezione di metodo: a volte la risposta arriva solo guardando il sistema da
fuori, come l'invariante del gioco MIU ha chiuso una domanda che nessuna
ricerca poteva chiudere.

`````{tab} Elementare
```{admonition} Da ricordare
:class: important
- Turing ha dato un senso preciso alla parola «algoritmo»: tutto ciò che una
  macchina con un nastro, una testina e una tabellina di regole sa fare. Una
  sola macchina, quella universale, imita tutte le altre leggendone le regole.
- Nessun programma può dire, per ogni programma, se si fermerà: il Profeta e il
  Dispetto si contraddicono. Per singoli programmi lo si può sapere, per tutti
  insieme no, ed è il no alla domanda di Hilbert.
- In ogni sistema di regole abbastanza ricco e senza contraddizioni c'è una
  frase vera che il sistema non dimostra (Gödel). Non ne segue che la mente
  non sia una macchina: per vedere quella verità bisogna sapere di non
  contraddirsi, e il sistema non può dimostrarlo di sé.
```
`````

`````{tab} Superiore
```{admonition} Da ricordare
:class: important
- Macchina di Turing $M=(Q,\Gamma,\sqcup,\delta,q_0,F)$; esiste la macchina
  universale; la tesi di Church-Turing identifica il calcolabile effettivo con
  il Turing-calcolabile (tesi, non teorema).
- $\mathrm{HALT}$ è semidecidibile e non decidibile (diagonale), e ne segue il
  no all’*Entscheidungsproblem*; Rice: ogni
  proprietà semantica non banale è indecidibile; la complessità di Kolmogorov
  non è calcolabile.
- Gödel: $T$ coerente, r.e., che interpreti l'aritmetica, è incompleta
  (Rosser: basta la coerenza) e non dimostra $\mathrm{Con}_T$. L'argomento di
  Lucas-Penrose presuppone la conoscenza della coerenza che il secondo teorema
  esclude.
```
`````

Questi limiti non riguardano la potenza di un calcolatore né la bravura di chi
lo programma: valgono per qualunque procedimento, e quindi anche per i
programmi che imparano dai dati, che restano programmi. Dentro quei confini,
però, c'è un territorio vastissimo, ed è lì che lavorano le macchine che si
muovono nel mondo e decidono dove mettere le ruote, a cui è dedicata la
sezione sulla {doc}`robotica <applicazioni>`.
