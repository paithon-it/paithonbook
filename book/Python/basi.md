# Le basi di Python

Guido van Rossum progettò Python con un vincolo severo: la grammatica del
linguaggio doveva stare in testa. Non centinaia di parole chiave, ma poche
decine; non una regola speciale per ogni caso, ma una manciata di idee che si
combinano. È per questo che si impara a *leggere* Python in un pomeriggio, e si
passa poi il resto del tempo a imparare *cosa* dire, non *come* dirlo. Questa
sezione è quel pomeriggio: i mattoni con cui è costruito tutto il resto del
libro.

## Tipi fondamentali e variabili

Il primo gesto della programmazione è dare un nome a un valore, per poterlo
richiamare più avanti: quel nome si chiama **variabile**. In Python la crei
assegnandole un valore con `=`, e non devi mai dichiarare in anticipo di che
tipo è: il linguaggio deduce da sé se sta maneggiando un numero, del testo o
un valore di verità.

```python
eta = 34             # int   -> un numero intero
temperatura = 36.6   # float -> un numero con la virgola
nome = "Ada"         # str   -> una stringa di testo
attiva = True        # bool  -> vero (True) o falso (False)
```

Sono i quattro tipi di partenza: interi (`int`), decimali (`float`), stringhe
(`str`) e booleani (`bool`). La funzione `type()` te lo conferma:

```python
type(temperatura)    # <class 'float'>
type(nome)           # <class 'str'>
```

Un'ultima cosa sulle stringhe, che userai in ogni pagina di questo libro: le
**f-string**. Mettendo una `f` prima delle virgolette puoi infilare direttamente
il valore di una variabile fra parentesi graffe, invece di concatenare pezzi.

```python
nome, eta = "Ada", 36
f"{nome} ha {eta} anni"          # 'Ada ha 36 anni'
f"la metà di {eta} è {eta / 2}"  # 'la metà di 36 è 18.0'
```

Dentro le graffe può stare qualsiasi espressione, e dopo i due punti si mette il
formato — comodissimo per stampare numeri leggibili:

```python
loss = 0.0347218
f"loss: {loss:.3f}"              # 'loss: 0.035'   tre decimali
f"accuratezza: {0.8723:.1%}"     # 'accuratezza: 87.2%'
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

Nessuna cerimonia, nessuna dichiarazione anticipata: assegni e vai. È questa
tipizzazione *dinamica* che rende Python veloce da scrivere.

`````

`````{tab} Superiore

Python è **dinamicamente tipizzato**: il tipo appartiene all'*oggetto*, non al
nome. Un nome è solo un riferimento; `x = 5` lega il nome `x` all'oggetto
intero `5`. Ogni valore è un oggetto con un tipo a runtime, e lo stesso nome può
essere rilegato a oggetti di tipo diverso in momenti diversi.

Dettagli che contano più avanti: gli `int` hanno **precisione arbitraria** (non
c'è overflow a 64 bit), i `float` sono double IEEE 754 a 64 bit (attenzione agli
errori di arrotondamento), le `str` sono sequenze Unicode **immutabili**, e
`bool` è una sottoclasse di `int` — infatti `True == 1` e `False == 0`. Ne
segue lo stile *duck typing*: conta cosa un oggetto *sa fare*, non a quale
classe appartiene.

`````

## Le strutture dati di ogni giorno

Quattro contenitori bastano per il 90% del lavoro. Si distinguono per due
domande: gli elementi hanno un ordine? e si possono modificare dopo la
creazione?

```python
numeri = [3, 1, 4, 1, 5]                 # list: ordinata, modificabile
punto  = (45.46, 9.19)                   # tuple: ordinata, immutabile
prezzi = {"pane": 1.2, "latte": 0.9}     # dict: coppie chiave -> valore
unici  = {3, 1, 4}                        # set: insieme senza duplicati
```

```{figure} ../figures/strutture-dati-python.svg
:name: fig-strutture-dati
:alt: Quattro schede a confronto — list ordinata e modificabile, tuple ordinata e immutabile, dict con coppie chiave-valore, set senza duplicati.
:width: 90%

Le quattro strutture dati native di Python a confronto: cambiano per ordine,
modificabilità e modo di accedere agli elementi.
```

`````{tab} Elementare

Le analogie aiutano ({numref}`fig-strutture-dati`):

- **list** — la lista della spesa: elementi in fila, che aggiungi e togli.
- **tuple** — le coordinate scritte a penna: una coppia fissa, non si cancella.
- **dict** — la rubrica del telefono: cerchi per *nome* (la chiave) e trovi il
  numero (il valore).
- **set** — un sacchetto in cui i doppioni si fondono: mettere due volte lo
  stesso elemento non cambia nulla.

`````

`````{tab} Superiore

La differenza tecnica è **mutabilità** e **hashabilità**. `list`, `dict` e
`set` sono mutabili; `tuple` è immutabile (e quindi *hashable*, perciò usabile
come chiave di dizionario). `dict` e `set` sono tabelle hash: l'accesso e il
test di appartenenza sono in media $O(1)$, contro l'$O(n)$ della ricerca lineare
in una lista. Le chiavi di un `dict` e gli elementi di un `set` devono essere
hashable, cioè immutabili — motivo per cui una lista non può stare in un set,
ma una tupla sì.

`````

Le operazioni comuni sono brevi e ricorrono ovunque nel libro:

```python
numeri.append(9)     # aggiunge in coda   -> [3, 1, 4, 1, 5, 9]
prezzi["latte"]      # accesso per chiave -> 0.9
"pane" in prezzi     # test di appartenenza -> True
len(unici)           # quanti elementi    -> 3
```

## Decidere e ripetere: il controllo di flusso

Qui si vede la scelta di stile più visibile di Python: i blocchi non hanno
parentesi graffe, ma sono delimitati dall'**indentazione**. Due punti, poi si
va a capo e si rientra. Il rientro non è cosmesi: è sintassi.

```python
voto = 27
if voto >= 28:
    esito = "ottimo"
elif voto >= 18:      # "else if": una condizione alternativa
    esito = "promosso"
else:
    esito = "bocciato"
```

Per ripetere ci sono due cicli. Il `for` scorre gli elementi di un contenitore;
il `while` continua finché una condizione resta vera.

```python
for n in numeri:      # itera sugli ELEMENTI, non sugli indici
    print(n * 2)

i = 0
while i < 3:           # ripete finché la condizione è vera
    print("giro", i)
    i += 1             # senza questo, ciclo infinito
```

`````{tab} Elementare

Un dettaglio pythonico comodo: nelle condizioni, i valori "vuoti" contano come
falsi. Una lista vuota, la stringa vuota, lo zero e `None` si comportano come
`False`; tutto il resto come `True`.

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
`saluta(nome="Ada", saluto="Salve")` è equivalente ma esplicito. Gli argomenti
con default vengono dopo quelli obbligatori.

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
sintassi dedicata. Il ciclo esplicito funziona, ma la *comprehension* dice la
stessa cosa in una riga, e i pythonisti la leggono a colpo d'occhio:

```python
# la via lunga
quadrati = []
for n in range(10):
    quadrati.append(n ** 2)

# la via pythonica: stesso risultato, una riga
quadrati = [n ** 2 for n in range(10)]

# con filtro: solo i numeri pari
pari = [n for n in range(10) if n % 2 == 0]    # -> [0, 2, 4, 6, 8]
```

Si legge quasi in italiano: "il quadrato di `n`, per ogni `n` da 0 a 9, se `n`
è pari". La stessa forma esiste per i dizionari (`{k: v for ...}`) e per i set.

## Un assaggio di oggetti: le classi

In Python *tutto* è un oggetto — numeri, stringhe, liste, perfino le funzioni.
Quando ti servono oggetti su misura, definisci una **classe**: uno stampo che
descrive quali dati un oggetto contiene e cosa sa fare.

```python
class Punto:
    def __init__(self, x, y):    # costruttore: prepara il nuovo oggetto
        self.x = x               # 'self' è l'oggetto stesso
        self.y = y

    def distanza_origine(self):
        return (self.x ** 2 + self.y ** 2) ** 0.5

p = Punto(3, 4)
p.distanza_origine()     # -> 5.0
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

`````{tab} Elementare

Il punto di partenza è una proprietà di Python che si dà per scontata: **le
funzioni sono oggetti**. Puoi assegnarle a una variabile, passarle come
argomento, restituirle da un'altra funzione.

```python
def saluta():
    return "Ciao!"

copia = saluta        # nessuna parentesi: non la chiamo, la passo
print(copia())        # Ciao!
```

Un **decoratore** è una funzione che prende una funzione e ne restituisce una
versione "avvolta". L'analogia è la cover del telefono: il telefono resta
identico, ma tutto ciò che gli arriva passa prima dalla cover, che può
aggiungere qualcosa senza che il telefono ne sappia niente.

```python
import time

def cronometra(funzione):
    def involucro(*args, **kwargs):        # accetta qualunque argomento
        inizio = time.perf_counter()
        risultato = funzione(*args, **kwargs)
        print(f"{funzione.__name__}: {time.perf_counter() - inizio:.3f}s")
        return risultato
    return involucro

@cronometra                                # = addestra = cronometra(addestra)
def addestra(epoche):
    time.sleep(0.1 * epoche)
    return "fatto"

addestra(3)                                # stampa il tempo, poi restituisce "fatto"
```

La riga `@cronometra` non è sintassi magica: è una scorciatoia per
`addestra = cronometra(addestra)`. Tutto qui.

Quando in PyTorch vedrai `@torch.no_grad()`, saprai leggerlo: quella funzione
viene avvolta in qualcosa che spegne il calcolo dei gradienti per la durata
della chiamata e li riaccende all'uscita.

`````

`````{tab} Superiore

Due dettagli separano un decoratore giocattolo da uno usabile.

**Preservare l'identità della funzione.** L'involucro sostituisce l'originale,
quindi `__name__`, `__doc__` e la firma diventano quelli dell'involucro — con
danni a `help()`, ai debugger e a qualunque codice che faccia introspezione. Il
rimedio è una riga:

```python
import functools

def cronometra(funzione):
    @functools.wraps(funzione)             # ricopia nome, docstring, annotazioni
    def involucro(*args, **kwargs):
        ...
    return involucro
```

**I decoratori con argomenti hanno un livello in più.** `@torch.no_grad()` ha
le parentesi perché non è il decoratore: è una *fabbrica* che restituisce il
decoratore. Servono quindi tre livelli — parametri, funzione, chiamata.

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
poi rimetti a posto», ed è la parola chiave **`with`**:

```{code-block} python
:class: pt-non-eseguibile

with open("dati.csv") as f:      # il file si apre...
    testo = f.read()
                                 # ...e si chiude da solo, anche in caso di errore
```

L'oggetto usato con `with` si chiama *context manager*: definisce cosa fare
all'ingresso e cosa all'uscita del blocco. Il valore sta tutto nella seconda
metà — **la pulizia avviene comunque**, anche se dentro il blocco viene sollevata
un'eccezione. Chiudere un file a mano è facile da dimenticare; con `with` non
serve ricordarsene.

In PyTorch lo incontrerai soprattutto così:

```{code-block} python
:class: pt-non-eseguibile

with torch.no_grad():            # dentro il blocco niente gradienti
    previsioni = modello(x)
```

ed è la stessa identica cosa del decoratore `@torch.no_grad()` visto sopra: la
differenza è solo l'ambito — il decoratore avvolge un'intera funzione, `with`
avvolge un blocco. Molte API offrono entrambe le forme proprio per questo.

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
  codice conciso; le **classi** modellano oggetti su misura — la stessa forma di
  scikit-learn e PyTorch.
- Le **f-string** (`f"{nome} ha {eta} anni"`) inseriscono valori nel testo, con
  il formato dopo i due punti: `f"{loss:.3f}"`.
- Un **decoratore** (`@qualcosa`) avvolge una funzione senza modificarla:
  `@cronometra` è solo `f = cronometra(f)`. `with` fa la stessa cosa su un
  blocco invece che su una funzione, e garantisce la pulizia anche in caso di
  errore.
```
