# Le basi di Python

Guido van Rossum progettò Python con un vincolo severo: la grammatica del
linguaggio doveva stare in testa. Non centinaia di **parole chiave** (le parole
che il linguaggio si tiene per sé, come `if`, `for` e `def`, e che quindi non
puoi usare per altro), ma poche decine; non una regola speciale per ogni caso,
ma una manciata di idee che si combinano. È per questo che si impara a *leggere* Python in un pomeriggio, e si
passa poi il resto del tempo a imparare *cosa* dire, non *come* dirlo. Questa
sezione è quel pomeriggio: i mattoni con cui è costruito tutto il resto del
libro.

## Tipi fondamentali e variabili

Il primo gesto della programmazione è dare un nome a un valore, per poterlo
richiamare più avanti: quel nome si chiama **variabile**. In Python la crei con
il segno `=`, che qui non significa «è uguale a» come in matematica, ma «prendi
il valore di destra e mettilo dentro il nome di sinistra»: `eta = 34` si legge
«da adesso `eta` vale 34». (Per *chiedere* se due cose sono uguali serve il
doppio uguale, `==`, che incontreremo fra poche pagine.) Ogni valore appartiene
a una famiglia, il suo **tipo**: numeri, testo, vero-o-falso. La comodità è che
il tipo non lo devi dichiarare tu: Python guarda il valore e lo riconosce da
sé.

I valori che maneggeremo hanno tre forme ricorrenti, e la differenza fra loro
conta più di quanto sembri: un numero singolo, una **stringa** (una fila di
caratteri, cioè del testo) e una **lista** (una fila di valori in ordine).

```{figure} ../figures/numeri-stringhe-liste.svg
:name: fig-tre-contenitori
:alt: "Tre rappresentazioni affiancate: il numero come una scatolina sigillata che contiene un solo valore; la stringa come una catena di caselle, una per carattere, che si possono leggere ma non riscrivere; la lista come una fila di caselle numerate a partire da zero, che si possono sia leggere sia sostituire."
:width: 96%

Tre forme di contenitore. La differenza che conta non è cosa contengono ma se
si può cambiarne il contenuto: la lista sì, la stringa no.
```

Nel disegno le caselle della lista sono numerate **a partire da zero**: la
prima è la numero 0, la seconda la numero 1. Sembra una stranezza, ed è invece
la convenzione che fa tornare i conti: il numero di una casella dice *quanti
passi fare dall'inizio* per arrivarci, e sulla prima sei già.

La distinzione mostrata in {numref}`fig-tre-contenitori` tornerà spesso, con
il nome di **mutabilità**, ed è all'origine di parecchi errori dei primi
giorni. Una stringa che sembra modificata è in realtà una stringa nuova, e la
vecchia resta com'era; una lista invece si cambia sul posto, e chi la sta
guardando da un altro punto del programma vede il cambiamento. È una differenza
che ha conseguenze, e le vedremo quando parleremo di funzioni, più avanti in
questa pagina.

```python
eta = 34             # int   -> un numero intero
temperatura = 36.6   # float -> un numero con la virgola
nome = "Ada"         # str   -> una stringa di testo
attiva = True        # bool  -> vero (True) o falso (False)
```

Sono i quattro tipi di partenza: interi (`int`), decimali (`float`), stringhe
(`str`) e booleani (`bool`, cioè i due soli valori `True` e `False`). Per
sapere di che tipo è un valore si **chiama** `type`: chiamare vuol dire
scriverne il nome e mettere fra parentesi ciò su cui deve lavorare, e in
risposta si ottiene un valore (le cose che si chiamano così si chiamano
*funzioni*, e le vediamo per bene fra qualche sezione).

```python
type(temperatura)    # <class 'float'>
type(nome)           # <class 'str'>
```

La risposta nomina una *classe*, che è il nome tecnico di una famiglia di
valori (le classi sono l'ultima sezione di questa pagina): per ora si legge
«è un `float`».

Un'ultima cosa sulle stringhe, che userai in ogni pagina di questo libro: le
**f-string**. Mettendo una `f` prima delle virgolette puoi infilare direttamente
il valore di una variabile fra parentesi graffe, invece di concatenare pezzi.

```python
nome, eta = "Ada", 36            # due nomi a sinistra, due valori a destra:
                                 # nome <- "Ada", eta <- 36, in ordine
f"{nome} ha {eta} anni"          # 'Ada ha 36 anni'
f"la metà di {eta} è {eta / 2}"  # 'la metà di 36 è 18.0'
```

Dentro le graffe può stare qualsiasi espressione, e dopo i due punti si mette
il formato, comodissimo per stampare numeri leggibili. Il formato è fatto di
due pezzi: quante cifre dopo la virgola, e che aspetto deve avere il numero
(`f` per un decimale normale, `%` per una percentuale, che moltiplica per cento
e aggiunge il segno).

```python
loss = 0.0347218                 # "loss" è l'errore di un modello che impara
f"loss: {loss:.3f}"              # 'loss: 0.035'      tre cifre, da decimale
f"accuratezza: {0.8723:.1%}"     # 'accuratezza: 87.2%'  una cifra, in percento
```

`````{tab} Elementare

Pensa a una variabile come a un'**etichetta adesiva** che attacchi a un valore.
`nome = "Ada"` vuol dire "d'ora in poi l'etichetta `nome` sta su questa
stringa". La cosa comoda è che puoi spostare l'etichetta su un valore di tipo
diverso quando vuoi:

```python
x = 5          # x è appiccicata a un intero
x = "cinque"   # ora la stessa etichetta sta su una stringa
```

Nessuna cerimonia, nessuna dichiarazione anticipata: assegni e vai. Questo modo
di fare ha un nome, *tipizzazione dinamica*, e vuol dire soltanto che il tipo lo
decide il valore, non una dichiarazione scritta da te. È una delle ragioni per
cui Python è veloce da scrivere.

`````

`````{tab} Superiore

Python è **dinamicamente tipizzato**: il tipo appartiene all'*oggetto*, non al
nome. Un nome è solo un riferimento; `x = 5` lega il nome `x` all'oggetto
intero `5`. Ogni valore è un oggetto con un tipo a runtime, e lo stesso nome può
essere rilegato a oggetti di tipo diverso in momenti diversi.

Dettagli che contano più avanti: gli `int` hanno **precisione arbitraria**
(non c'è overflow a 64 bit), i `float` sono double IEEE 754 a 64 bit
(attenzione agli errori di arrotondamento), le `str` sono sequenze Unicode
**immutabili**, e `bool` è una sottoclasse di `int`, infatti `True == 1` e
`False == 0`. Ne segue lo stile *duck typing*: conta cosa un oggetto *sa
fare*, non a quale classe appartiene.

Il modello a riferimenti ha due conseguenze immediate. La prima è l'*alias*:
`b = a` non copia niente, dà un secondo nome allo stesso oggetto, e
`b.append(3)` si vede anche da `a` (per copiare servono `a.copy()`, che è
superficiale, o `copy.deepcopy`). La seconda è la distinzione fra i due
confronti: `is` chiede «sono lo stesso oggetto?», `==` chiede «valgono la
stessa cosa?». L'unico uso di `is` da imparare a memoria è `if x is None`;
usarlo sui numeri dà risultati che dipendono da come CPython riusa gli oggetti
(`x = 1000; y = 1000; x is y` risponde `True` se le due righe stanno nello
stesso blocco compilato e `False` se le si digita separatamente all'interprete,
che è esattamente il motivo per cui non va usato così).

`````

## Le strutture dati di ogni giorno

Quattro contenitori bastano per il 90% del lavoro: la **lista**, la **tupla**,
il **dizionario** e l'**insieme**. Nel codice si scrivono con i nomi inglesi
(`list`, `tuple`, `dict`, `set`), e ciascuno si riconosce dalle parentesi che
usa:

```python
numeri = [3, 1, 4, 1, 5]                 # list: ordinata, modificabile
punto  = (45.46, 9.19)                   # tuple: ordinata, immutabile
prezzi = {"pane": 1.2, "latte": 0.9}     # dict: coppie chiave -> valore
unici  = {3, 1, 4}                        # set: insieme senza duplicati
```

Nel dizionario ogni valore si trova cercando la sua **chiave**, cioè la parola
scritta a sinistra della freccia: `prezzi` non si interroga per posizione, ma
per nome, come una rubrica.

```{figure} ../figures/strutture-dati-python.svg
:name: fig-strutture-dati
:alt: Quattro schede a confronto (list ordinata e modificabile, tuple ordinata e immutabile, dict con coppie chiave-valore, set senza duplicati).
:width: 90%

Che aspetto hanno, una accanto all'altra: le quadre della lista, le tonde
della tupla, le coppie del dizionario, e il set in cui i doppioni si fondono
da soli.
```

Fra i quattro si sceglie rispondendo a due domande, non a memoria: gli elementi
hanno un ordine? e si possono modificare dopo la creazione?

```{figure} ../figures/tuple-dizionari-set.svg
:name: fig-scelta-contenitore
:alt: "Tabella decisionale con una riga per contenitore, lista, tupla, dizionario e set, e una colonna per ciascuna proprietà: se mantiene l'ordine, se si può modificare dopo la creazione, se ammette duplicati e come vi si accede, per posizione o per chiave."
:width: 96%

Le quattro righe rispondono alle due domande qui sopra. Scegliere il
contenitore è rispondere a quelle, non ricordare a memoria quale si usa di
solito.
```

La colonna della modificabilità in {numref}`fig-scelta-contenitore` è quella
che decide più spesso. Un contenitore che non si può modificare dà due
garanzie: nessun'altra parte del programma può cambiartelo sotto il naso, e (se
anche i valori che contiene sono fissi) può fare da chiave in un dizionario.
Non sono due limitazioni: sono le due ragioni per cui la tupla esiste.

`````{tab} Elementare

Le analogie aiutano ({numref}`fig-strutture-dati`):

- **list**, la lista della spesa: elementi in fila, che aggiungi e togli.
- **tuple**, le coordinate scritte a penna: una coppia fissa, non si cancella.
- **dict**, la rubrica del telefono: cerchi per *nome* (la chiave) e trovi il
  numero (il valore).
- **set**, un sacchetto in cui i doppioni si fondono: mettere due volte lo
  stesso elemento non cambia nulla.

`````

`````{tab} Superiore

La differenza tecnica è **mutabilità** e **hashabilità**. `list`, `dict` e
`set` sono mutabili; `tuple` è immutabile, ed è *hashable* solo se lo sono
anche i suoi elementi: solo in quel caso può fare da chiave di dizionario.
`dict` e `set` sono tabelle hash: l'accesso e il test di appartenenza sono in
media $O(1)$, contro l'$O(n)$ della ricerca lineare in una lista. Le chiavi di
un `dict` e gli elementi di un `set` devono essere hashable, cioè avere un
hash che non cambia nel tempo. Per i tipi built-in la proprietà coincide in
pratica con l'immutabilità (una tupla che contiene una lista, per esempio,
non è hashable), ed è il motivo per cui una lista non può stare in un set ma
una tupla di numeri sì; fuori dai built-in la coincidenza cade, perché
un'istanza di una classe definita dall'utente è mutabile ed è hashable per
default (l'hash deriva dall'identità dell'oggetto).

`````

Le operazioni comuni sono brevi e ricorrono ovunque nel libro:

```python
numeri[0]            # il primo elemento (si conta da zero) -> 3
numeri[-1]           # l'ultimo, contando dalla fine        -> 5
numeri.append(9)     # aggiunge in coda   -> [3, 1, 4, 1, 5, 9]
prezzi["latte"]      # accesso per chiave -> 0.9
"pane" in prezzi     # test di appartenenza -> True
len(unici)           # quanti elementi    -> 3
```

In queste quattro righe convivono due scritture diverse, e vale la pena
separarle subito perché torneranno per tutto il libro. `len(unici)` è una
funzione generica: si scrive il nome e le si passa fra parentesi la cosa su cui
lavorare. `numeri.append(9)` invece si legge da sinistra a destra come una
frase, «alla lista `numeri`, aggiungi 9»: il punto vuol dire «di», prima del
punto c'è la cosa (un oggetto, o una libreria intera) e dopo il punto qualcosa
che quella cosa **sa fare**, che si chiama *metodo*. Righe come `df.head()`,
`np.array(...)` o `modello.fit(...)`, che riempiono le pagine seguenti, sono
tutte di questa seconda forma.

## Decidere e ripetere: il controllo di flusso

Qui si vede la scelta di stile più visibile di Python. Un **blocco** è un
gruppo di righe che vanno insieme (quelle da eseguire se una condizione è vera,
per esempio), e in Python si segna facendolo **rientrare** verso destra: due
punti, si va a capo, si spinge il testo dentro di qualche spazio. Il rientro
non è cosmesi, è sintassi: sono quegli spazi a dire dove il blocco comincia e
dove finisce, e sbagliarli è un errore come sbagliare una parola.

```{figure} ../figures/controllo-di-flusso.svg
:name: fig-controllo-di-flusso
:alt: "Diagramma di flusso di un ciclo: dall'inizio si arriva a un rombo di condizione; se la condizione è vera si entra nel corpo del ciclo, si esegue l'aggiornamento e si torna alla condizione; se è falsa si esce. Il corpo e l'aggiornamento sono racchiusi in un blocco rientrato."
:width: 62%

Il ciclo come figura. Il rientro nel codice disegna esattamente il riquadro
che qui è tracciato: dice dove il corpo comincia e dove finisce.
```

Il rombo di {numref}`fig-controllo-di-flusso` è il punto in cui si annidano
quasi tutti i cicli infiniti dei primi giorni. Se dentro il corpo non c'è
niente che avvicini la condizione al diventare falsa, la freccia di ritorno
non porta da nessuna parte, e il programma gira per sempre senza dare errore.

```python
voto = 27
if voto >= 28:
    esito = "ottimo"
elif voto >= 18:      # "else if": una condizione alternativa
    esito = "promosso"
else:
    esito = "bocciato"

print(esito)          # -> promosso
```

L'ultima riga fa uscire il risultato: `print` scrive sullo schermo ciò che gli
si mette fra parentesi (il nome viene da «stampare», ma la stampante non
c'entra). Senza un `print` il programma calcola e tace, ed è la prima cosa che
sconcerta chi comincia.

Per ripetere ci sono due cicli. Il `for` scorre gli elementi di un contenitore;
il `while` continua finché una condizione resta vera.

```python
for n in numeri:      # n diventa a turno ogni elemento della lista
    print(n * 2)

i = 0
while i < 3:           # ripete finché la condizione è vera
    print("giro", i)
    i += 1             # scorciatoia di "i = i + 1": senza, ciclo infinito
```

Un programma finito in un ciclo infinito non si schianta: continua, e basta. Lo
si ferma premendo `Ctrl+C` nella finestra in cui gira.

`````{tab} Elementare

Un dettaglio comodo, e molto *pythonico* (si dice così di ciò che è scritto nel
modo in cui Python vuole essere scritto): nelle condizioni, i valori "vuoti"
contano come falsi. Una lista vuota, la stringa vuota, lo zero e `None` (che è
il modo in cui Python dice «niente»: un valore che sta al posto di un valore
mancante) si comportano come `False`; tutto il resto come `True`.

```python
carrello = []
if carrello:                  # una lista vuota è "falsa"
    print("hai articoli")
else:
    print("carrello vuoto")   # <- viene stampato questo
```

Così scrivi `if carrello:` invece di `if len(carrello) > 0:`, e si legge come
una frase.

`````

`````{tab} Superiore

Il `for` di Python è un *for-each* costruito sul **protocollo di iterazione**:
funziona su qualunque oggetto *iterabile* (liste, tuple, stringhe, dizionari,
generatori). `range(n)` produce gli indici in modo pigro, senza materializzare
la lista; `enumerate` dà `(indice, valore)` e `zip` allinea più sequenze:

```python
for i, prezzo in enumerate([1.2, 0.9]):   # 0 1.2 ; 1 0.9
    print(i, prezzo)
```

`break` interrompe il ciclo, `continue` salta all'iterazione successiva. La
"verità" di un oggetto è definita dai metodi speciali `__bool__` o, in
mancanza, `__len__`: da qui il fatto che un contenitore vuoto valga `False`.

`````

## Funzioni: dare un nome a un blocco di lavoro

Una funzione impacchetta un pezzo di logica dietro un nome, così lo riusi senza
riscriverlo. Si definisce con `def`, riceve degli argomenti e restituisce un
risultato con `return`.

```{figure} ../figures/funzioni-args-kwargs-lambda.svg
:name: fig-funzione-scatola
:alt: "Una funzione disegnata come una scatola: a sinistra entrano gli argomenti, dentro sta il corpo che li elabora, e a destra esce il valore di ritorno. Chi chiama la funzione vede soltanto ciò che entra e ciò che esce, non quello che accade dentro."
:width: 78%

La funzione come scatola. Chi la usa deve conoscere solo i due lati, cosa
darle e cosa ne ottiene; il corpo può cambiare senza che nessuno se ne
accorga.
```

La linea che in {numref}`fig-funzione-scatola` separa il dentro dal fuori è
tutto il valore delle funzioni, e vale ben oltre Python. Finché i due lati
restano gli stessi, il corpo si può riscrivere, ottimizzare o correggere senza
toccare una riga del codice che la chiama.

```python
def area_rettangolo(base, altezza):
    return base * altezza

def saluta(nome, saluto="Ciao"):   # 'saluto' ha un valore di default
    return f"{saluto}, {nome}!"

area_rettangolo(3, 4)     # -> 12
saluta("Ada")             # -> "Ciao, Ada!"
saluta("Ada", "Salve")    # -> "Salve, Ada!"
```

`````{tab} Elementare

Immagina una funzione come una **macchinetta**: dentro metti gli ingredienti
(gli argomenti), esce un risultato. `area_rettangolo` prende base e altezza,
restituisce l'area.

Il **valore di default** è un ingrediente preimpostato: `saluto="Ciao"` significa
"se non mi dici come salutare, uso Ciao". Puoi chiamare `saluta("Ada")` e
lasciar decidere alla funzione, oppure passare il tuo saluto.

`````

`````{tab} Superiore

Gli argomenti si passano per **posizione** o per **nome** (keyword):
`saluta(nome="Ada", saluto="Salve")` è equivalente ma esplicito. Fra i
parametri **posizionali**, quelli con un default vengono dopo quelli
obbligatori (`def g(a=1, b)` è un errore di sintassi). Dopo un `*` nudo
cominciano invece i parametri *keyword-only*, che si passano solo per nome e
che possono essere obbligatori anche venendo dopo dei default: `def f(a=1, *,
b)` è legale, e `b` va passato. È il motivo per cui tante firme di scikit-learn
e PyTorch hanno un asterisco solitario in mezzo: costringere a scrivere il nome
degli argomenti rende leggibile il codice di chi chiama e permette di
aggiungere parametri senza rompere il codice altrui.

Una trappola classica: il valore di default è valutato **una sola volta**, alla
definizione. Non usare mai un oggetto mutabile come default (`def f(x, acc=[])`)
o l'`acc` sarà condiviso tra le chiamate. Le funzioni in Python sono
*first-class*: si assegnano a variabili, si passano ad altre funzioni, si
restituiscono. Senza `return` esplicito, una funzione restituisce `None`.

Per le funzioni usa-e-getta c'è una forma compatta, la **lambda**: una
funzione anonima che sta su una riga e restituisce il valore dell'unica
espressione che contiene.

```python
quadrato = lambda x: x ** 2      # equivale a: def quadrato(x): return x ** 2

# l'uso tipico: passarla a qualcosa che si aspetta una funzione
coppie = [("Ada", 36), ("Alan", 41), ("Grace", 45)]
sorted(coppie, key=lambda c: c[1])       # ordina per età
```

La regola pratica: va bene per un'espressione breve passata a `sorted`, `map`,
`filter` o alle API di scikit-learn e PyTorch che accettano callable. Se la
logica cresce oltre la riga, o se la funzione ha bisogno di un nome per essere
capita, usa `def`.

`````

## List comprehension: l'idioma pythonico

Costruire una lista trasformando un'altra è talmente comune che Python ha una
sintassi dedicata. Il nome inglese, *list comprehension*, viene dalla
matematica: un insieme si può definire *per elencazione* (0, 1, 4, 9, …) oppure
*per comprensione* («i quadrati dei numeri da 0 a 9»), ed è esattamente ciò che
fa questa scrittura. Il ciclo esplicito funziona, ma la comprehension dice la
stessa cosa in una riga, e chi programma in Python la legge a colpo d'occhio:

```{figure} ../figures/codice-pythonic-stile.svg
:name: fig-stile-pythonico
:alt: "Due versioni dello stesso programma affiancate. A sinistra la versione goffa: un ciclo che scorre gli indici, crea una lista vuota e vi aggiunge un elemento alla volta, in quattro righe. A destra la versione idiomatica: la stessa cosa in una list comprehension di una riga, che si legge come la frase che descrive il risultato."
:width: 100%

Lo stesso risultato, due modi di dirlo. La versione di destra non è più corta
per vezzo: nomina il risultato invece di descrivere la procedura per
ottenerlo.
```

La differenza di {numref}`fig-stile-pythonico` è ciò che si intende con
«pythonico», parola che altrimenti suona come una questione di gusto. Non lo
è: la comprehension si legge come una frase dichiarativa, e chi la scorre non
deve simulare il ciclo nella testa per capire cosa produrrà.

```python
# la via lunga
quadrati = []
for n in range(10):        # range(10): i dieci numeri da 0 a 9
    quadrati.append(n ** 2)          # ** è l'elevamento a potenza

# la via pythonica: stesso risultato, una riga
quadrati = [n ** 2 for n in range(10)]

# con filtro: solo i numeri pari
pari = [n for n in range(10) if n % 2 == 0]    # -> [0, 2, 4, 6, 8]
```

Tre segni nuovi, tutti nell'ultima riga. `%` è il **resto** della divisione
(`7 % 2` fa 1); `==` chiede «sono uguali?», ed è doppio proprio per non
confondersi con l'uguale singolo, che invece assegna; messi insieme,
`n % 2 == 0` significa «il resto della divisione di `n` per due è zero», cioè
«`n` è pari». La riga intera si legge quasi in italiano: "il quadrato di `n`,
per ogni `n` da 0 a 9, se `n` è pari". La stessa forma esiste per i dizionari
(`{k: v for ...}`) e per i set.

## Un assaggio di oggetti: le classi

Un **oggetto** è una cosa che Python tiene in memoria e che porta con sé due
cose insieme: dei dati, e le azioni che sa fare su quei dati (i *metodi* del
punto, poche sezioni fa). In Python *tutto* è un oggetto: numeri, stringhe,
liste, perfino le funzioni. Quando ti servono oggetti su misura, definisci una
**classe**: uno stampo che descrive quali dati un oggetto contiene e cosa sa
fare.

```{figure} ../figures/oop-classi-attributi-metodi.svg
:name: fig-classe-istanze
:alt: "A sinistra la classe Sensore, disegnata come uno stampo che elenca gli attributi e i metodi senza valori. A destra tre istanze prodotte dallo stampo: hanno la stessa struttura ma valori diversi negli attributi, e condividono gli stessi metodi."
:width: 92%

Uno stampo e tre pezzi. La classe dice quali dati esistono e cosa si può
fare; ogni istanza porta i propri valori, mentre i metodi restano quelli
della classe.
```

La divisione di {numref}`fig-classe-istanze` è ciò che si intende quando si
dice che una classe è «uno stampo»: la struttura si scrive una volta, i valori
tante. Ed è anche il motivo per cui i metodi ricevono `self` come primo
argomento, che è il pezzo su cui stanno lavorando.

```python
class Punto:
    def __init__(self, x, y):    # il metodo che prepara un nuovo oggetto:
        self.x = x               # si chiama sempre così, con i quattro
        self.y = y               # trattini bassi. 'self' è l'oggetto stesso

    def distanza_origine(self):
        return (self.x ** 2 + self.y ** 2) ** 0.5   # ** 0.5 = radice quadrata

p = Punto(3, 4)
p.distanza_origine()     # -> 5.0   (il triangolo 3, 4, 5 di Pitagora)
```

`````{tab} Elementare

Una classe è come uno **stampo per biscotti**: `Punto` è lo stampo, ogni `p`
che crei è un biscotto diverso fatto con lo stesso stampo. I *metodi* (come
`distanza_origine`) sono le cose che l'oggetto sa fare; gli *attributi* (`x`,
`y`) sono ciò che l'oggetto ricorda. La parola `self` è il modo in cui
l'oggetto parla di sé stesso.

`````

`````{tab} Superiore

`__init__` è il **costruttore**, invocato quando scrivi `Punto(3, 4)`; `self` è
il riferimento esplicito all'istanza, primo parametro di ogni metodo. Gli
attributi assegnati con `self.x = x` sono dati d'istanza. Le classi supportano
l'**ereditarietà** (`class Punto3D(Punto): ...`), che permette di specializzare
comportamenti riusando il codice della classe base.

Questo pattern è ovunque nell'ecosistema: uno stimatore di scikit-learn è un
oggetto che addestri chiamandone i metodi (`modello.fit(X, y)`,
`modello.predict(X)`), e in PyTorch ogni rete neurale è una classe che
**eredita** da `nn.Module`, con i suoi attributi (i pesi) e i suoi metodi.
Capire le classi ora rende familiare tutto il codice di machine learning che
verrà.

`````

## I decoratori: la `@` che troverai ovunque

Più avanti nel libro incontrerai righe come `@torch.no_grad()` sopra la
definizione di una funzione. Vale la pena sapere cosa fanno: la sintassi
compare spesso e sembra magica solo finché non si guarda sotto.

```{figure} ../figures/decorator-property.svg
:name: fig-decoratore
:alt: "Schema di una funzione avvolta da un decoratore: gli argomenti entrano nel wrapper esterno, che esegue del codice prima, chiama la funzione originale racchiusa al suo interno, esegue del codice dopo, e infine restituisce il risultato al chiamante."
:width: 88%

Un decoratore è un involucro. La funzione originale resta intatta al centro:
quello che cambia è che chi la chiama passa prima e dopo per il codice
dell'involucro.
```

Guardando {numref}`fig-decoratore` si capisce perché la `@` non sia una parola
chiave speciale ma solo una scorciatoia di scrittura: `@qualcosa` sopra una
funzione significa «prendi questa funzione, passala a `qualcosa`, e tieni ciò
che torna al suo posto».

`````{tab} Elementare

Il punto di partenza è una proprietà di Python che si dà per scontata: **le
funzioni sono oggetti**. Puoi darle un secondo nome, passarle come ingrediente
a un'altra funzione, o farle restituire da un'altra funzione.

```python
def buongiorno():
    return "Buongiorno!"

copia = buongiorno    # nessuna parentesi: non la chiamo, la passo
print(copia())        # Buongiorno!
```

Un **decoratore** è una funzione che prende una funzione e ne restituisce una
versione "avvolta". L'analogia è la cover del telefono: il telefono resta
identico, ma tutto ciò che gli arriva passa prima dalla cover, che può
aggiungere qualcosa senza che il telefono ne sappia niente.

```python
import time

def cronometra(funzione):         # riceve una funzione...
    def involucro():              # ...e ne prepara una versione avvolta
        inizio = time.perf_counter()      # un cronometro molto preciso
        risultato = funzione()            # qui in mezzo gira l'originale
        print(f"ci ha messo {time.perf_counter() - inizio:.2f} secondi")
        return risultato
    return involucro              # restituisce l'involucro, non il risultato

@cronometra                       # equivale a: addestra = cronometra(addestra)
def addestra():
    time.sleep(0.3)               # facciamo finta di lavorare
    return "fatto"

addestra()                        # stampa il tempo, poi restituisce "fatto"
```

Due cose spiegano la forma strana di questo codice. L'involucro è scritto
*dentro* `cronometra` perché deve ricordarsi quale funzione sta avvolgendo, e
lo può fare solo se nasce lì; e `cronometra` restituisce l'involucro senza
chiamarlo (`return involucro`, senza parentesi) perché il suo compito è
consegnare la funzione nuova, non eseguirla.

La riga `@cronometra` non è sintassi magica: è una scorciatoia per
`addestra = cronometra(addestra)`, cioè «prendi questa funzione, dalla a
`cronometra`, e da qui in poi chiama «addestra» quello che ne torna». Tutto
qui.

Quando in PyTorch vedrai `@torch.no_grad()`, saprai leggerlo: quella funzione
viene avvolta in qualcosa che, per la durata della chiamata, spegne il calcolo
dei **gradienti** (le quantità con cui una rete neurale impara: le vedremo nei
capitoli sulle reti, e spegnerle serve quando la rete non deve più imparare ma
solo rispondere).

`````

`````{tab} Superiore

Due dettagli separano un decoratore giocattolo da uno usabile.

**Accettare qualunque firma, e non falsificare l'identità.** L'involucro
sostituisce l'originale, quindi deve poter ricevere gli argomenti che
riceveva quella: `*args` raccoglie in una tupla tutti gli argomenti passati per
posizione, `**kwargs` in un dizionario tutti quelli passati per nome. E poiché
è l'involucro a prendere il posto dell'originale, `__name__`, `__doc__` e la
firma diventano i suoi, con danni a `help()`, ai debugger e a qualunque codice
che faccia introspezione. Il rimedio è una riga, `functools.wraps`:

```python
import functools
import time

def cronometra(funzione):
    @functools.wraps(funzione)             # ricopia nome, docstring, annotazioni
    def involucro(*args, **kwargs):        # accetta qualunque argomento...
        inizio = time.perf_counter()
        risultato = funzione(*args, **kwargs)   # ...e lo gira all'originale
        print(f"{funzione.__name__}: {time.perf_counter() - inizio:.3f}s")
        return risultato
    return involucro

@cronometra
def addestra(epoche, lr=0.1):
    time.sleep(0.05 * epoche)
    return "fatto"

addestra(2, lr=0.01)          # -> "fatto", preceduto dal tempo
addestra.__name__             # 'addestra', non 'involucro'
```

**Le parentesi di `@torch.no_grad()`, però, non sono quelle di un decoratore
parametrico.** `torch.no_grad` non è una funzione: è una **classe**.
`torch.no_grad()` costruisce un oggetto, e quell'oggetto sa fare due mestieri
perché definisce sia `__enter__`/`__exit__` (e allora sta dopo `with`) sia
`__call__` (e allora sta dopo `@`). È per questo che le due forme sono la
stessa cosa, e infatti PyTorch accetta anche `@torch.no_grad` senza parentesi
(la classe base si chiama `_NoParamDecoratorContextManager`, cioè
«decoratore senza parametri»). I decoratori che i parametri li prendono davvero
(`@functools.lru_cache(maxsize=128)`) hanno invece un livello in più: sono
funzioni che *restituiscono* il decoratore, quindi parametri, funzione,
chiamata.

Il parente stretto è **`@property`**, che fa sembrare attributo il risultato di
un metodo. Serve quando un valore è derivato o va validato, ma si vuole
continuare a scrivere `oggetto.valore` invece di `oggetto.get_valore()`:

```{code-block} python
:class: pt-non-eseguibile

class Modello:
    def __init__(self, pesi):
        self._pesi = pesi

    @property
    def n_parametri(self):                 # si legge come attributo...
        return sum(p.size for p in self._pesi)

m = Modello([...])
m.n_parametri                              # ...non m.n_parametri()
```

È il motivo per cui in scikit-learn e PyTorch alcune cose si leggono come dati
e altre si chiamano come metodi: la distinzione non è arbitraria, è una scelta
di interfaccia.

`````

### Il parente stretto: `with`

C'è un secondo modo di dire «per la durata di questo blocco, cambia qualcosa e
poi rimetti a posto», ed è la parola chiave **`with`**. L'esempio classico è
leggere un file dal disco: `open` lo apre e restituisce un oggetto da cui il
testo si tira fuori con `f.read()`, e un file aperto va sempre richiuso,
altrimenti resta occupato.

```{code-block} python
:class: pt-non-eseguibile

with open("dati.csv") as f:      # il file si apre...
    testo = f.read()
                                 # ...e si chiude da solo, anche in caso di errore
```

L'oggetto usato con `with` si chiama *context manager* («gestore di
contesto»): definisce cosa fare all'ingresso e cosa all'uscita del blocco. Il
valore sta tutto nella seconda metà, cioè nel rimettere a posto: chiudere il
file, liberare la memoria, riaccendere ciò che si era spento. E avviene
**comunque**, anche quando il programma, dentro il blocco, incontra un errore e
si interrompe (in Python un errore si chiama *eccezione*, e si dice che viene
*sollevata*: il programma smette di eseguire le righe e stampa il punto in cui
è successo). Chiudere un file a mano è facile da dimenticare; con `with` non
serve ricordarsene.

In PyTorch lo incontrerai soprattutto così:

```{code-block} python
:class: pt-non-eseguibile

with torch.no_grad():            # dentro il blocco niente gradienti
    previsioni = modello(x)
```

ed è la stessa identica cosa del decoratore `@torch.no_grad()` visto sopra: la
differenza è solo l'ambito (il decoratore avvolge un'intera funzione, `with`
avvolge un blocco). Molte API offrono entrambe le forme proprio per questo.

## Un lavoratore alla volta: il GIL

C'è un fatto su Python che non si può non conoscere, perché spiega scelte che
il libro farà più avanti senza spiegarle di nuovo. Riguarda che cosa succede
quando si prova a far fare più cose insieme allo stesso programma.

`````{tab} Elementare

Un computer moderno ha molti nuclei di calcolo, e viene naturale pensare che
per andare più veloci basti dividere il lavoro fra loro. In Python la cosa non
funziona come ci si aspetta, e la ragione ha un nome: il **GIL**, il *global
interpreter lock*, che si può tradurre come «il lucchetto dell'interprete».

L'immagine giusta è una cucina professionale con un solo coltello. Puoi
assumere quattro cuochi, ma il coltello è uno: mentre uno taglia, gli altri
tre aspettano il loro turno. Assumerne altri non fa uscire i piatti più in
fretta, anzi li rallenta un po', perché il coltello va passato di mano.

I cuochi, in un programma, si chiamano **thread**: sono le linee di lavoro che
procedono in parallelo dentro lo stesso programma, e condividono tutto, come
quattro cuochi nella stessa cucina. Il coltello è il permesso di eseguire
istruzioni Python, e ce n'è uno solo: **un thread alla volta**. Da cui la
regola pratica, che è tutto ciò che serve ricordare:

- se il lavoro è **aspettare** (scaricare pagine, leggere file, interrogare un
  database), i thread aiutano eccome: mentre uno aspetta posa il coltello, e
  gli altri lavorano;
- se il lavoro è **calcolare**, i thread non servono a niente. Per usare
  davvero più nuclei bisogna avviare **processi** separati, che sono cucine
  diverse, ciascuna con il suo coltello.

I processi hanno un prezzo: non condividono la memoria, quindi tutto ciò che si
scambiano va impacchettato, spedito e riaperto dall'altra parte. Per questo si
avviano una volta sola e si tengono, invece di crearne uno per ogni pezzetto di
lavoro.

E c'è un'ultima cosa, che è il motivo per cui in pratica il problema si sente
molto meno di quanto questa spiegazione faccia temere: quando il conto vero
avviene dentro NumPy o PyTorch, quelle librerie **posano il coltello** prima di
mettersi a calcolare, perché il calcolo lo fanno in C e non hanno bisogno
dell'interprete. Nel codice che conta per questo libro, insomma, il lucchetto è
aperto quasi sempre.

`````

`````{tab} Superiore

Il **Global Interpreter Lock** è un mutex nell'implementazione di riferimento
di Python (CPython) che protegge lo stato interno dell'interprete, in
particolare il **conteggio dei riferimenti** con cui ogni oggetto tiene traccia
di quanti nomi lo puntano (è il meccanismo primario con cui CPython libera la
memoria; il `gc` vero e proprio gli sta sopra e serve a raccogliere i cicli).
La sua conseguenza è netta: **un solo thread per processo esegue bytecode Python in
un dato istante**. Non è una proprietà del linguaggio, è una scelta di
implementazione (Jython e IronPython non ce l'hanno), ma è la scelta
dell'interprete che tutti usano.

Le tre vie alla concorrenza in Python vanno quindi tenute distinte, perché
risolvono problemi diversi:

- **`threading`**: thread veri del sistema operativo, memoria condivisa,
  serializzati dal GIL sul bytecode. Il GIL viene però rilasciato durante le
  operazioni di I/O bloccanti e dentro le estensioni C che lo dichiarano.
  Utile per lavoro **I/O-bound**, inutile per quello **CPU-bound**.
- **`multiprocessing`**: processi separati, ciascuno con il proprio interprete
  e il proprio GIL, quindi parallelismo reale sui nuclei. Il costo è che la
  memoria non è condivisa: gli argomenti e i risultati attraversano una
  serializzazione (`pickle`), che per tensori grandi può dominare il guadagno.
- **`asyncio`**: un solo thread, multitasking **cooperativo** su un ciclo di
  eventi. Un `await` cede il controllo esplicitamente. Nessun parallelismo di
  calcolo, ma scala a decine di migliaia di attese contemporanee senza il costo
  di altrettanti thread. È il modello dei server, ed è quello dei client verso
  le API dei modelli, dove il tempo se ne va aspettando la rete.

La ragione per cui il GIL, in pratica, morde meno di quanto sembri: **NumPy e
PyTorch lo rilasciano** durante le operazioni pesanti, che girano in codice C o
in kernel BLAS/CUDA già multi-thread al loro interno. Una moltiplicazione fra
matrici usa tutti i nuclei anche da un solo thread Python. Il GIL torna a
mordere sul codice Python puro: i cicli sui campioni, la decodifica delle
immagini, il *preprocessing*. È esattamente lì che il `DataLoader` di PyTorch
avvia processi con `num_workers`, e la stessa ragione per cui `DataParallel`,
che pilota più GPU da un solo processo, è sconsigliato in favore di
`DistributedDataParallel`, che ne usa uno per GPU.

```{admonition} Il GIL non è per sempre
:class: note
Con la **PEP 703** il GIL sta diventando opzionale. CPython 3.13 ha introdotto
una *build* sperimentale senza GIL (*free-threading*); con la **PEP 779**
quella build passa da sperimentale a ufficialmente supportata in CPython 3.14,
pur non essendo ancora quella predefinita, con un costo residuo che l'ultima
misura ufficiale dà fra l'1% (macOS aarch64) e l'8% (Linux x86-64) sul codice a
thread singolo: nel 3.13 era attorno al 40%, quasi tutto dovuto all'interprete
adattivo disattivato in quella build. Farne il default è una terza fase
annunciata ma non ancora datata. È materia in movimento: quel che resta vero,
e che vale la pena portarsi via, è la **distinzione** fra lavoro che aspetta e
lavoro che calcola, e il fatto che condividere memoria e condividere nuclei
sono due problemi diversi.
```

`````

Che i thread non aiutino a calcolare, e aiutino invece ad aspettare, si può
vedere in una ventina di righe, ed è il tipo di misura che val la pena fare una
volta con le proprie mani (l'ambiente per farlo è quello preparato nella prima
sezione del capitolo). Non serve capire ogni riga del programma
che segue: quello che conta sono i numeri che stampa, e ognuno di essi è
stampato due volte, perché i tempi da guardare sono due. Il **tempo di parete**
è quello dell'orologio appeso al muro, cioè quanto si è aspettato; il **tempo
di CPU** è quanto lavoro ha fatto davvero il processore, sommato su tutti i
lavoratori. È la differenza fra «quanto ci ha messo» e «quanta fatica ha
fatto», e senza la seconda misura questo esperimento non dimostra niente.

```python
import multiprocessing as mp
import sys
import time
from concurrent.futures import ThreadPoolExecutor   # un gruppo di thread

def calcola(n):
    """Lavoro puro di CPU: nessun file, nessuna rete, solo somme."""
    return sum(i * i for i in range(n))

def aspetta(secondi):
    """Lavoro puro di attesa: il processore non fa niente."""
    time.sleep(secondi)

def durata(funzione, lavori, thread=1):
    """Esegue i lavori e restituisce (tempo di parete, tempo di CPU)."""
    t0, c0 = time.perf_counter(), time.process_time()
    if thread == 1:
        [funzione(x) for x in lavori]
    else:
        with ThreadPoolExecutor(thread) as ex:
            list(ex.map(funzione, lavori))
    return time.perf_counter() - t0, time.process_time() - c0

CPU = [2_000_000] * 4        # quattro lavori di calcolo identici
ATTESE = [0.25] * 4          # quattro attese da un quarto di secondo

for etichetta, funzione, lavori, thread in [
    ("CPU, in sequenza    ", calcola, CPU, 1),
    ("CPU, con 4 thread   ", calcola, CPU, 4),
    ("attesa, in sequenza ", aspetta, ATTESE, 1),
    ("attesa, con 4 thread", aspetta, ATTESE, 4),
]:
    parete, cpu = durata(funzione, lavori, thread)
    print(f"{etichetta}: parete {parete:.2f} s | CPU {cpu:.2f} s")

# I processi hanno ciascuno il proprio interprete, quindi il proprio GIL.
def lavoratore(n, coda):
    coda.put(calcola(n))

t0 = time.perf_counter()
ctx = mp.get_context("fork")    # fuori da Linux serve "spawn", e la funzione
coda = ctx.Queue()              # dev'essere definita in un modulo importabile
processi = [ctx.Process(target=lavoratore, args=(n, coda)) for n in CPU]
[p.start() for p in processi]
risultati = [coda.get() for _ in CPU]
[p.join() for p in processi]
print(f"CPU, con 4 processi : parete {time.perf_counter() - t0:.2f} s")

if hasattr(sys, "_is_gil_enabled"):     # la domanda esiste da Python 3.13
    print("GIL attivo:", sys._is_gil_enabled())
```

I numeri assoluti dipendono dalla macchina, e anche il tempo di parete dipende
da quanto la macchina è occupata: con quattro thread può uscire un po' più alto
o un po' più basso di quello in sequenza, e non è quello il punto. Il punto è
il **tempo di CPU**, che non cambia. Sul lavoro di calcolo i quattro thread ne
consumano quanto la versione in sequenza, cioè non stanno lavorando in quattro:
si stanno passando il turno, ed è esattamente ciò che significa GIL.
Sull'attesa, invece, i quattro thread scendono a poco più di un quarto del
tempo di parete, e il tempo di CPU resta quasi zero: lì il lucchetto è posato e
nessuno si ostacola. Con i processi il lavoro di calcolo accelera davvero,
quanto lo permettono i nuclei disponibili. Tre confronti, e la regola resta in
mente.

`````{tab} Elementare

```{admonition} Da ricordare
:class: important
- Il **tipo** di un valore lo decide il valore, non tu: si assegna con `=` e si
  va avanti. I mattoni sono i numeri interi (`int`), quelli con la virgola
  (`float`), il testo (`str`) e il vero-o-falso (`bool`).
- Quattro contenitori coprono quasi tutto: **list** (fila ordinata, si
  modifica), **tuple** (fila fissa), **dict** (si cerca per nome, cioè per
  chiave), **set** (senza doppioni).
- I blocchi si delimitano **rientrando** le righe: `if`/`elif`/`else` per
  decidere, `for` e `while` per ripetere. `print` è ciò che fa uscire un valore
  sullo schermo: senza, il programma calcola e tace.
- Una **funzione** (`def` … `return`) è un pezzo di lavoro con un nome, da
  riusare; una **list comprehension** costruisce una lista in una riga; una
  **classe** è uno stampo per fabbricare oggetti su misura, ciascuno con i
  propri dati e le proprie azioni.
- Le **f-string** (`f"{nome} ha {eta} anni"`) infilano valori dentro il testo, e
  dopo i due punti si dice che aspetto devono avere: `f"{loss:.3f}"` sono tre
  cifre dopo la virgola.
- Un **decoratore** (`@qualcosa`) avvolge una funzione senza toccarla:
  `@cronometra` è solo `f = cronometra(f)`. `with` fa la stessa cosa su un
  blocco, e garantisce che alla fine si rimetta tutto a posto anche se qualcosa
  va storto.
- Il **GIL** lascia eseguire istruzioni Python a **un thread alla volta**: i
  thread aiutano quando il lavoro è *aspettare* (rete, disco), non quando è
  *calcolare*. Per usare più nuclei servono **processi**, cucine separate che
  però non condividono niente e devono impacchettare e spedirsi tutto quello
  che si scambiano. NumPy e PyTorch, mentre fanno i conti pesanti, **posano il
  coltello**: è per questo che in pratica il problema morde molto meno di
  quanto sembri.
```

`````

`````{tab} Superiore

```{admonition} Da ricordare
:class: important
- Python è **dinamicamente tipizzato**: assegni un valore e il tipo si deduce
  da solo. I mattoni sono `int`, `float`, `str`, `bool`.
- Quattro strutture dati coprono quasi tutto: **list** (ordinata, modificabile),
  **tuple** (immutabile), **dict** (chiave → valore, accesso immediato per
  chiave), **set** (senza duplicati).
- I blocchi sono definiti dall'**indentazione**; `if/elif/else`, `for` e `while`
  bastano per il controllo di flusso.
- Le **funzioni** (`def` … `return`) e le **list comprehension** rendono il
  codice conciso; le **classi** modellano oggetti su misura: la stessa forma
  di scikit-learn e PyTorch.
- Le **f-string** (`f"{nome} ha {eta} anni"`) inseriscono valori nel testo, con
  il formato dopo i due punti: `f"{loss:.3f}"`.
- Un **decoratore** (`@qualcosa`) avvolge una funzione senza modificarla:
  `@cronometra` è solo `f = cronometra(f)`. `with` fa la stessa cosa su un
  blocco invece che su una funzione, e garantisce la pulizia anche in caso di
  errore.
- Il **GIL** lascia eseguire codice Python a **un thread alla volta**: i thread
  aiutano quando il lavoro è *aspettare* (rete, disco), non quando è
  *calcolare*. Per usare più nuclei servono **processi**, che però non
  condividono la memoria e devono serializzare ciò che si scambiano;
  `asyncio` è la terza via, un solo thread che gestisce molte attese. NumPy e
  PyTorch **rilasciano il GIL** durante i conti pesanti, ed è per questo che in
  pratica morde molto meno di quanto sembri.
```

`````
