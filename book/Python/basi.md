# Le basi di Python

Python è disegnato attorno a un vincolo severo: la grammatica del
linguaggio deve stare in testa. Non centinaia di **parole chiave** (le parole
che il linguaggio si tiene per sé, come `if`, `for` e `def`, e che quindi non
puoi usare per altro), ma poche decine; non una regola speciale per ogni caso,
ma una manciata di idee che si combinano. Le regole stanno in poche pagine; il resto del tempo si passa a decidere che
cosa dire. Qui ci sono i mattoni: dare un nome a un valore, tenere insieme più
valori, decidere, ripetere, impacchettare un pezzo di lavoro dietro un nome.

## Tipi fondamentali e variabili

Il primo gesto della programmazione è dare un nome a un valore, per poterlo
richiamare più avanti: quel nome si chiama **variabile**. In Python la crei con
il segno `=`, che qui non significa «è uguale a» come in matematica, ma
«attacca il nome di sinistra al valore di destra», come si attacca
un'etichetta: `eta = 34` si legge «da adesso `eta` vale 34». (Per *chiedere* se
due cose sono uguali serve il
doppio uguale, `==`, che incontreremo più sotto.) Ogni valore appartiene
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
giorni. Una stringa non si può cambiare: si può solo fabbricarne una nuova a
partire da quella vecchia, che resta intatta. Una lista invece si cambia sul
posto, e chi la stava guardando da un altro nome vede il cambiamento. Detta
così sembra una sottigliezza; è invece la ragione per cui, quando arriveremo
alle funzioni, ne vedrai una modificare la lista che le passi e non la
stringa.

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
type(temperatura)    # -> <class 'float'>
type(nome)           # -> <class 'str'>
```

La risposta nomina una *classe*, che è il nome tecnico di una famiglia di
valori (un assaggio di classi arriva più sotto): per ora si legge
«è un `float`».

Un'ultima cosa sulle stringhe, che userai in ogni pagina di questo libro: le
**f-string**. Mettendo una `f` prima delle virgolette puoi infilare direttamente
il valore di una variabile fra parentesi graffe, invece di concatenare pezzi.

```python
nome, eta = "Ada", 36            # due nomi a sinistra, due valori a destra:
                                 # nome <- "Ada", eta <- 36, in ordine
f"{nome} ha {eta} anni"          # -> 'Ada ha 36 anni'
f"la metà di {eta} è {eta / 2}"  # -> 'la metà di 36 è 18.0'
```

Perché `18.0` e non `18`? Perché in Python la divisione con la barra `/`
restituisce sempre un numero con la virgola, anche quando il conto tornerebbe
tondo: chiedere una divisione vuol dire dichiarare che accetti un risultato
decimale. (Se ti serve invece il quoziente intero esiste la doppia barra:
`36 // 2` dà `18`.)

Dentro le graffe può stare qualsiasi **espressione**, cioè qualunque cosa che
Python sappia ridurre a un valore: un conto, il nome di una variabile, una
chiamata a una funzione. E dopo i due punti si mette
il formato, comodissimo per stampare numeri leggibili. Il formato è fatto di
due pezzi: quante cifre dopo la virgola, e che aspetto deve avere il numero
(`f` per un decimale normale, `%` per una percentuale, che moltiplica per cento
e aggiunge il segno). È il gesto che si fa ogni volta che si vuole guardare un
numero senza tutte le cifre che il computer si porta dietro:

```python
loss = 0.0347218                 # un numero con troppe cifre da leggere
f"loss: {loss:.3f}"              # -> 'loss: 0.035'      tre cifre, da decimale
f"accuratezza: {0.8723:.1%}"     # -> 'accuratezza: 87.2%'  una cifra, in percento
```

`````{tab} Elementare

Una variabile è un’**etichetta adesiva** attaccata a un valore.
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

E niente impedisce di mettere due etichette sullo stesso valore: `b = a`
attacca il secondo nome alla stessa cosa a cui è attaccato il primo, senza
fabbricare niente di nuovo. Con i numeri la differenza non si sente; con una
lista si sente eccome, perché la lista si cambia sul posto: se `a` è `[1, 2]`
e dopo `b = a` scrivi `b.append(3)`, anche `a` adesso è `[1, 2, 3]`. La lista
è una sola, con due etichette attaccate sopra, e chi la guarda da `a` vede
quello che è successo passando da `b`.

`````

`````{tab} Superiore

Python è **dinamicamente tipizzato**: il tipo appartiene all’*oggetto*, non al
nome. Un nome è solo un riferimento; `x = 5` lega il nome `x` all'oggetto
intero `5`. Ogni valore è un oggetto con un tipo a runtime, e lo stesso nome può
essere rilegato a oggetti di tipo diverso in momenti diversi.

Dettagli che contano più avanti: gli `int` hanno **precisione arbitraria**
(non c'è overflow a 64 bit), i `float` sono double IEEE 754 a 64 bit
(attenzione agli errori di arrotondamento), le `str` sono sequenze Unicode
**immutabili**, e `bool` è una sottoclasse di `int`, infatti `True == 1` e
`False == 0`. Ne segue lo stile *duck typing*: conta cosa un oggetto *sa
fare*, non a quale classe appartiene.

Il modello a riferimenti ha due conseguenze immediate. La prima è l’*alias*:
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

Quattro contenitori coprono quasi tutto quello che si fa ogni giorno: la
**lista**, la **tupla**,
il **dizionario** e l’**insieme**. Nel codice si scrivono con i nomi inglesi
(`list`, `tuple`, `dict`, `set`), e ciascuno si riconosce dalle parentesi che
usa:

```python
numeri = [3, 1, 4, 1, 5]                 # list: ordinata, modificabile
punto  = (45.46, 9.19)                   # tuple: ordinata, immutabile
prezzi = {"pane": 1.2, "latte": 0.9}     # dict: coppie chiave: valore
unici  = {3, 1, 4, 1}                    # set: il secondo 1 sparisce da solo
```

Nel dizionario ogni valore si trova cercando la
sua **chiave**, cioè la parola scritta prima dei due punti: `prezzi` non si
interroga per posizione, ma per nome, come una rubrica. E nel set i valori
scritti sono quattro mentre quelli che restano sono tre: mettere due volte lo
stesso elemento non dà errore, semplicemente non aggiunge niente. È la ragione
per cui il set esiste, e fra poche righe `len(unici)` risponderà `3`.

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
:alt: "Tabella decisionale con una riga per contenitore, lista, tupla, dizionario e set, e quattro colonne: se mantiene l'ordine, se si può modificare dopo la creazione, se ammette duplicati, e infine la domanda guida che porta a sceglierlo. Una nota in calce ricorda che il dizionario conserva l'ordine di inserimento delle chiavi da Python 3.7."
:width: 96%

Le quattro righe rispondono alle due domande qui sopra. Scegliere il
contenitore è rispondere a quelle, non ricordare a memoria quale si usa di
solito.
```

La colonna della modificabilità in {numref}`fig-scelta-contenitore` è quella
che decide più spesso, e spiega a cosa serva una tupla, che a prima vista
sembra soltanto una lista con qualcosa in meno. La prima garanzia è che nessuna
altra parte del programma può cambiartela sotto il naso: se scrivi le
coordinate di Milano in una tupla, quelle restano.

La seconda è che una tupla può fare da **chiave di un dizionario**, e una lista
no. La ragione è che il dizionario ritrova un valore andando dritto al posto
che alla chiave compete, e quel posto lo calcola dal contenuto della chiave: se
il contenuto cambiasse, il valore resterebbe in un posto in cui nessuno lo
cerca più. Ecco perché si può usare come chiave solo qualcosa che non cambia, e
perché una coppia di numeri è comodissima:

```python
distanze = {("Milano", "Roma"): 573, ("Milano", "Napoli"): 770}
distanze[("Milano", "Roma")]     # -> 573
```

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
media $O(1)$, contro l’$O(n)$ della ricerca lineare in una lista. Le chiavi di
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
numeri[1:4]          # una fetta: da 1 incluso a 4 escluso  -> [1, 4, 1]
numeri.append(9)     # aggiunge in coda   -> [3, 1, 4, 1, 5, 9]
numeri.remove(9)     # toglie il primo 9  -> [3, 1, 4, 1, 5]
prezzi["latte"]      # accesso per chiave -> 0.9
"pane" in prezzi     # test di appartenenza -> True
len(unici)           # quanti elementi    -> 3
```

La terza riga introduce la **fetta** (in inglese *slice*), che è il modo di
prendere un pezzo di una lista invece di un elemento solo: si scrivono due
indici separati dai due punti, e il secondo è **escluso**. Sembra una
scortesia, ed è la scelta che fa quadrare i conti: la lunghezza della fetta è
la differenza dei due numeri, e due fette scritte di seguito, `[0:3]` e
`[3:6]`, si incastrano senza sovrapporsi e senza buchi. Una fetta di una lista
è sempre una **copia**: modificarla non tocca l'originale (con NumPy, nella
sezione seguente, non sarà più così, ed è una delle differenze che fa più
danni).

In queste sei righe convivono due scritture diverse, e conviene separarle
subito perché torneranno per tutto il libro. `len(unici)` è una funzione
generica: si scrive il nome e le si passa fra parentesi la cosa su cui
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

Si comincia dal decidere. `if` esegue il blocco rientrato solo se la
condizione che lo precede è vera, e se non lo è si passa alle alternative:

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

Con `voto` che vale 27 la prima condizione è falsa, la seconda è vera, e delle
tre righe rientrate viene eseguita solo la seconda. L'ultima riga fa uscire il
risultato: `print` scrive sullo schermo ciò che gli si mette fra parentesi (il
nome viene da «stampare», ma la stampante non c'entra). In un programma
salvato in un file, senza un `print` il calcolo avviene e non se ne vede
niente: è la prima cosa che sconcerta chi comincia, e la ragione per cui
l'interprete, che invece risponde da sé, è più comodo per provare.

Per ripetere ci sono due cicli. Il `for` scorre gli elementi di un contenitore;
il `while` continua finché una condizione resta vera.

```python
for n in numeri:      # n diventa a turno ogni elemento della lista
    print(n * 2)      # -> 6, 2, 8, 2, 10, uno per riga

i = 0
while i < 3:           # ripete finché la condizione è vera
    print("giro", i)   # -> giro 0, giro 1, giro 2
    i += 1             # scorciatoia di "i = i + 1": senza, ciclo infinito
```

Il `while` è quello a cui guardare con sospetto, ed è il soggetto della
{numref}`fig-controllo-di-flusso`.

```{figure} ../figures/controllo-di-flusso.svg
:name: fig-controllo-di-flusso
:alt: "Diagramma di flusso di un ciclo: dall'inizio si arriva a un rombo di condizione; se la condizione è vera si entra nel corpo del ciclo, si esegue l'aggiornamento e si torna alla condizione; se è falsa si esce. Il corpo e l'aggiornamento sono racchiusi da un riquadro tratteggiato, che è il blocco rientrato."
:width: 62%

Il ciclo come figura. Il riquadro tratteggiato è il blocco rientrato del
codice: il rientro disegna esattamente quel contorno, e dice dove il corpo
comincia e dove finisce.
```

Il rombo è il punto in cui si annidano quasi tutti i cicli infiniti dei primi
giorni. Nel codice qui sopra a far scendere il sipario è `i += 1`: senza quella
riga `i` resterebbe zero, la condizione `i < 3` sarebbe vera per sempre, e la
freccia di ritorno non porterebbe da nessuna parte. Un programma finito così
non si schianta: continua, e basta. Lo si ferma premendo `Ctrl+C` nella
finestra in cui gira.

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
riscriverlo. Si definisce con `def`; i valori che le si danno da lavorare si
chiamano **argomenti** (in italiano la parola fa pensare a «di cosa si parla»,
qui invece sono gli ingredienti che le passi); e il risultato lo consegna
indietro con `return`.

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

**A chi restituisce, `return`?** È la domanda giusta, e la risposta è: a chi ha
scritto la chiamata. Quando `area_rettangolo(3, 4)` finisce, quella scritta
*diventa* il numero 12, lì dove sta, come se avessi scritto 12 con le tue mani.
Da quel momento ci fai quello che vuoi: gli dai un nome, lo sommi, lo passi a
un'altra funzione, lo stampi.

```python
risultato = area_rettangolo(3, 4)   # il 12 va a finire in 'risultato'
print(risultato)                    # -> 12
print(area_rettangolo(3, 4) + 8)    # -> 20  (il 12 e' li', e ci si somma 8)
```

Ed è anche il motivo per cui `return` e `print` non sono la stessa cosa, benché
nell'interprete si somiglino: `print` scrive sullo schermo e non consegna
niente a nessuno, `return` consegna un valore al programma e non scrive niente.
Una funzione che stampasse invece di restituire sarebbe inutilizzabile dentro
un conto più grande.

Gli argomenti si possono anche passare **con il loro nome davanti**, e allora
l'ordine non conta più:

```python
saluta(saluto="Salve", nome="Ada")   # -> "Salve, Ada!"
```

È una scrittura che vedrai molto spesso, perché rende leggibile chi chiama:
`bins=20` dice cosa sono quei venti, mentre un `20` da solo lascia indovinare.

`````{tab} Elementare

Una funzione lavora come una **macchinetta**: dentro metti gli ingredienti
(gli argomenti), esce un risultato. `area_rettangolo` prende base e altezza,
restituisce l'area.

Il **valore di default** è un ingrediente preimpostato: `saluto="Ciao"` significa
"se non mi dici come salutare, uso Ciao". Puoi chiamare `saluta("Ada")` e
lasciar decidere alla funzione, oppure passare il tuo saluto.

C'è però un punto in cui l'immagine della macchinetta inganna, ed è la
mutabilità. Una macchinetta vera non tocca gli ingredienti che le hai dato;
una funzione Python, se l'ingrediente è una lista, può modificartela per
davvero, perché la lista che riceve non è una copia, è la tua.

```python
def aggiungi_zero(fila):
    fila.append(0)          # modifica la lista che ha ricevuto

def maiuscolo(parola):
    parola = parola.upper() # fabbrica una stringa nuova e attacca il nome a lei
    return parola

mia_lista = [1, 2]
aggiungi_zero(mia_lista)
mia_lista                   # -> [1, 2, 0]   la mia lista e' cambiata

mia_parola = "ciao"
maiuscolo(mia_parola)       # -> 'CIAO'
mia_parola                  # -> 'ciao'      la mia stringa e' intatta
```

La differenza non sta nella funzione, sta nel tipo: `append` cambia la lista
sul posto, mentre `.upper()` non può cambiare una stringa (nessuno può) e
quindi ne restituisce una nuova. Ecco perché la lista di fuori si trova
modificata e la stringa no. Quando una funzione ti cambia i dati sotto il naso
senza che tu l'abbia chiesto, la causa è quasi sempre questa.

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
o l’`acc` sarà condiviso tra le chiamate. Le funzioni in Python sono
*first-class*: si assegnano a variabili, si passano ad altre funzioni, si
restituiscono. Senza `return` esplicito, una funzione restituisce `None`.

Per le funzioni usa-e-getta c'è una forma compatta, la **lambda**: una
funzione anonima che sta su una riga e restituisce il valore dell'unica
espressione che contiene.

```python
# lambda x: x ** 2   fa lo stesso lavoro di   def quadrato(x): return x ** 2
# ma non si scrive «quadrato = lambda x: ...»: se le serve un nome, usa def.

# l'uso tipico: passarla a qualcosa che si aspetta una funzione
coppie = [("Grace", 45), ("Ada", 36), ("Alan", 41)]
sorted(coppie, key=lambda c: c[1])       # ordina per età
# -> [('Ada', 36), ('Alan', 41), ('Grace', 45)]
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
:alt: "Due versioni dello stesso programma affiancate. A sinistra la versione goffa, otto righe: due cicli che scorrono gli indici, una somma accumulata a mano e una lista vuota riempita un elemento alla volta. A destra la versione idiomatica, due sole istruzioni, di cui una list comprehension che si legge come la frase che descrive il risultato."
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

## Quando il programma si rompe: il traceback, e il paracadute

Prima o poi una riga va storta, e Python risponde con un **traceback**: la
ricostruzione dell'incidente, stampata un attimo prima di fermarsi. Salvato in
un file `errore.py`, questo programma chiede il voto di una persona che nel
registro non c'è:

```{code-block} python
:class: pt-non-eseguibile

voti = {"Ada": 30, "Bruno": 28}

def voto_di(nome):
    return voti[nome]

print(voto_di("Carla"))
```

Eseguito con `python3 errore.py`, si ferma così:

```text
Traceback (most recent call last):
  File "errore.py", line 6, in <module>
    print(voto_di("Carla"))
  File "errore.py", line 4, in voto_di
    return voti[nome]
KeyError: 'Carla'
```

`````{tab} Elementare

Un traceback si legge **dal fondo**. L'ultima riga dice *che cosa* è successo:
`KeyError: 'Carla'`, cioè «ho cercato la chiave `Carla` e non c'era». Le righe
sopra dicono *dove*: ogni coppia «`File`, `line`» è una tappa del viaggio che
il programma stava facendo, dalla prima chiamata in cima fino alla riga
incriminata in fondo. Qui il viaggio ha due tappe: la riga 6 ha chiamato
`voto_di`, e dentro `voto_di` la riga 4 è quella che è caduta.

Quindi, davanti a un traceback lungo uno schermo: niente panico, ultima riga
per il che cosa, e poi si risale con calma per il dove. È la prima cosa da
imparare a leggere in Python, perché la si incontra più spesso di qualunque
altra.

E si può fare di meglio che schiantarsi: mettere un **paracadute** attorno
alla riga che può cadere. Si scrive `try:` («prova»), e sotto, con `except`
(«se è andata male»), che cosa fare invece, nominando l'incidente che ci si
aspetta: `except KeyError:` si legge «se la chiave non c'era». Il paracadute si
mette attorno alla riga fragile, non attorno al programma intero, e si apre per
quell'incidente e per quello solo: un paracadute che copre tutto nasconde anche
gli errori che avresti voluto vedere.

`````

`````{tab} Superiore

Il paracadute, in codice, è il costrutto `try`/`except`, e la regola d'oro è
catturare **poco**:

- si dichiara sempre *quale* eccezione si sta aspettando (`except KeyError:`),
  perché un `except:` nudo prende qualunque cosa, compreso il `Ctrl+C` con cui
  vorresti fermare il programma, e trasforma ogni guasto in un silenzio;
- con `except KeyError as e:` l'oggetto-eccezione finisce in `e`, e porta il
  messaggio;
- `raise` senza argomenti, dentro un `except`, rilancia l'eccezione appena
  catturata: si usa quando si vuole registrare il guasto ma non nasconderlo;
- i rami facoltativi: `else` gira solo se il `try` è filato liscio, `finally`
  gira **comunque**, ed è il posto delle pulizie (chiudere un file, rilasciare
  una risorsa). Il mestiere di `finally` è esattamente quello che il
  costrutto `with` automatizza.

Lo stile che ne esce ha un nome, **EAFP** (*easier to ask forgiveness than
permission*): si prova e si gestisce il fallimento, invece di controllare
tutto prima (`if nome in voti: ...`, lo stile *look before you leap*). In
Python l'EAFP è idiomatico anche per una ragione pratica: fra il controllo e
l'uso il mondo può cambiare, mentre il `try` è un gesto solo.

`````

Il paracadute, sul programma di prima:

```python
voti = {"Ada": 30, "Bruno": 28}

def voto_di(nome):
    try:
        return voti[nome]
    except KeyError:
        return f"{nome} non e' a verbale"

print(voto_di("Bruno"))
print(voto_di("Carla"))
```

```text
28
Carla non e' a verbale
```

Adesso la mancanza di Carla è un fatto gestito, non un incidente: il programma
decide lui che cosa significa, e lo dice con le sue parole.

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

Ogni pezzo uscito dallo stampo, nel gergo, si chiama **istanza** della classe.
La parola da sola non dice niente, e conviene sapere perché: è ricalcata
sull'inglese *instance*, che vuol dire «caso», «esemplare». In italiano
un'istanza è la domanda che si presenta a un ufficio, e qui non c'entra
niente. Leggila come «un esemplare concreto di quella classe» e il senso
torna. La teniamo perché è la parola che troverai in ogni documentazione, in
ogni corso e in ogni colloquio di lavoro, non perché sia una bella traduzione.

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
l’**ereditarietà** (`class Punto3D(Punto): ...`), che permette di specializzare
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
definizione di una funzione. Conviene sapere cosa fanno: la sintassi compare
spesso e sembra magica solo finché non si guarda sotto.

```{figure} ../figures/decorator-property.svg
:name: fig-decoratore
:alt: "Schema di una funzione avvolta da un decoratore: gli argomenti entrano nel wrapper esterno, che esegue del codice prima, chiama la funzione originale racchiusa al suo interno, esegue del codice dopo, e infine restituisce il risultato al chiamante."
:width: 88%

Un decoratore è un involucro. La funzione originale resta intatta al centro:
quello che cambia è che chi la chiama passa prima e dopo per il codice
dell'involucro.
```

In {numref}`fig-decoratore` la scatola grande è l'involucro e quella piccola
al centro è la
funzione di partenza, che resta intatta: la `@` non è una parola chiave
speciale, è la scorciatoia con cui si dice «prendi questa funzione, passala a
`qualcosa`, e tieni al suo posto ciò che torna».

`````{tab} Elementare

Al banco all'ingresso di un laboratorio c'è un custode con un cronometro. Chi
vuole parlare con la ricercatrice della stanza in fondo passa di lì: il custode
fa partire il cronometro, lo lascia entrare e all'uscita segna quanto è durata
la visita. La ricercatrice lavora come sempre e del cronometro non sa niente.

Il banco si regge su due fatti che in Python valgono per tutte le funzioni. Il
nome di una funzione è la targa sulla porta: le parentesi sono il bussare, e
una targa si stacca e si riappende altrove senza che la stanza cambi.

```python
def buongiorno():
    return "Buongiorno!"

copia = buongiorno    # nessuna parentesi: non la chiamo, le do un secondo nome
copia()               # -> 'Buongiorno!'   chiamare 'copia' chiama 'buongiorno'
```

L'altro fatto: un banco si costruisce su ordinazione, e l'ordinazione se la
ricorda per sempre. Chiedine uno tarato sul 3 e triplica tutto quello che gli
passa davanti, anche quando l'officina ha chiuso da un pezzo: il cinque che
entra esce quindici.

```python
def moltiplicatore(n):        # fabbrica funzioni, non numeri
    def moltiplica(x):
        return x * n          # 'n' e' quello che c'era quando 'moltiplica' e' nata
    return moltiplica         # niente parentesi: restituisce la funzione

triplica = moltiplicatore(3)  # ora 'triplica' e' una funzione
triplica(5)                   # -> 15
```

Il custode col cronometro esce dalla stessa officina, e stavolta l'ordinazione
è una porta invece che un numero.

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

addestra()        # stampa il tempo (circa 0.30 s), poi restituisce 'fatto'
```

Il banco, nel codice, si chiama `involucro`, e sta dentro `cronometra` perché
è lì che gli viene detta la porta: solo nascendo lì se la ricorda. E
`cronometra` finisce con
`return involucro`, senza parentesi: consegna il banco senza farci passare
nessuno.

La riga `@cronometra` è la targa nuova, e vale
`addestra = cronometra(addestra)`: da quel momento chi cerca `addestra` trova
il banco. Dietro c'è ancora la funzione di prima, che dorme i suoi tre decimi
di secondo e risponde `"fatto"`, mentre sul foglio del custode finisce 0.30.

`@torch.no_grad()` mette al banco un custode di un altro mestiere: per la
durata della visita tiene spento il calcolo dei **gradienti**, le quantità con
cui una rete neurale impara, e all'uscita lo riaccende. Serve quando la rete
deve solo rispondere e non più imparare.

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

Nella prima riga la parolina `as` significa «e chiamalo»: `open` consegna
l'oggetto che rappresenta il file aperto, e `as f` gli dà il nome `f`, che
useremo dentro il blocco. Il nome lo scegli tu, `f` è solo l'abitudine.

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

Un computer di oggi ha quattro, otto, sedici nuclei di calcolo, e viene
naturale pensare che per andare più in fretta basti dividere il lavoro fra
loro. Con Python non funziona così, e la ragione ha tre lettere che conviene
conoscere adesso, perché decide come si scrivono i programmi che caricano
dati. Riguarda che cosa succede
quando si prova a far fare più cose insieme allo stesso programma.

`````{tab} Elementare

Le tre lettere sono **GIL**, *global interpreter lock*: «il lucchetto
dell'interprete».

L'immagine giusta è una cucina professionale con un solo coltello. Puoi
assumere quattro cuochi, ma il coltello è uno: mentre uno taglia, gli altri
tre aspettano il loro turno. Assumerne altri non fa uscire i piatti più in
fretta: i piatti escono più o meno nello stesso tempo, perché il coltello va
passato di mano e quel passaggio si mangia il poco che si guadagna.

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
molto meno di quanto la cucina con un coltello solo faccia temere: quando il
conto vero avviene dentro NumPy o PyTorch, quelle librerie **posano il
coltello** prima di mettersi a calcolare, perché il calcolo lo fanno in C e
non hanno bisogno dell'interprete. Nel codice che addestra modelli e macina
numeri, insomma, il lucchetto è aperto quasi sempre.

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
misura ufficiale dà fra l'1% (macOS aarch64) e l'8% (Linux x86-64) sul codice
a thread singolo: nel 3.13 era attorno al 40%, il grosso dovuto
all'interprete adattivo disattivato in quella build. Farne il default è una
terza fase annunciata ma non ancora datata. È materia in movimento: quel che
resta vero, e che conviene portarsi via, è la **distinzione** fra lavoro che
aspetta e lavoro che calcola, e il fatto che condividere memoria e condividere
nuclei sono due problemi diversi.
```

`````

Che i thread non aiutino a calcolare, e aiutino invece ad aspettare, si può
vedere in una ventina di righe, con una misura da fare una volta con le
proprie mani (l'ambiente per farlo è quello preparato
nell'{doc}`apertura del capitolo </Python/overview>`). Il programma che segue
stampa i numeri che contano, e ognuno di essi è
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

# Il trattino basso dentro un numero e' solo un separatore per l'occhio:
# 2_000_000 e' due milioni, e Python lo legge come se non ci fosse.
# La lista per un numero, invece, la ripete: [x] * 4 fa una lista di quattro x.
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
# Fuori da Linux il metodo è "spawn": il figlio non eredita la memoria del
# padre, reimporta il modulo, e quindi la funzione dev'essere definita in un
# modulo importabile e il codice che avvia i processi va protetto da
# `if __name__ == "__main__":`. Senza quella riga il figlio riesegue anche
# l'avvio, e i processi si moltiplicano finché la macchina non cede.
# Cosi' com'e', con "fork", questo blocco gira su Linux e su Colab.
ctx = mp.get_context("fork")
coda = ctx.Queue()
processi = [ctx.Process(target=lavoratore, args=(n, coda)) for n in CPU]
for p in processi:
    p.start()
# il trattino basso da solo e' un nome come un altro, e per convenzione dice
# «questo valore non mi serve»: qui conta solo quante volte girare
risultati = [coda.get() for _ in CPU]   # svuotare la coda PRIMA del join
for p in processi:
    p.join()
print(f"CPU, con 4 processi : parete {time.perf_counter() - t0:.2f} s")

if hasattr(sys, "_is_gil_enabled"):     # la domanda esiste da Python 3.13
    print("GIL attivo:", sys._is_gil_enabled())
```

Su una macchina a quattro nuclei che non stia facendo altro, stampa qualcosa
del genere:

```text
CPU, in sequenza    : parete 0.26 s | CPU 0.26 s
CPU, con 4 thread   : parete 0.31 s | CPU 0.29 s
attesa, in sequenza : parete 1.00 s | CPU 0.00 s
attesa, con 4 thread: parete 0.25 s | CPU 0.00 s
CPU, con 4 processi : parete 0.12 s
```

I numeri assoluti dipendono dalla macchina, ma la grandezza da guardare è
sempre la stessa: il **tempo di parete con quattro thread confrontato con
quello in sequenza**. Sul lavoro di calcolo non scende: resta lì dov'era, e da
un'esecuzione all'altra oscilla di qualche centesimo in su o in giù, perché
passarsi il turno ha un costo dello stesso ordine del rumore della misura: i quattro thread non stanno lavorando
in quattro, si stanno alternando, ed è esattamente ciò che significa GIL.
Sull'attesa scende invece a un quarto, da un secondo tondo a 0,25, perché lì il
lucchetto è posato e nessuno si ostacola. Con i processi accelera anche il
calcolo, quanto lo permettono i nuclei disponibili. Tre confronti, e la regola
resta in mente.

Va fatto su una
macchina che non stia facendo nient'altro, e il modo di accorgersi che non è
così è guardare la prima riga: se il tempo di parete è molto più alto del tempo
di CPU (qui sono uguali, 0,26 e 0,26), vuol dire che il programma ha passato la
maggior parte del tempo in coda dietro a qualcun altro, e le misure che seguono
non parlano più del GIL. Sotto carico pesante può perfino capitare il
risultato opposto, i
quattro thread che finiscono *prima* di quello solo: non lavorano in
parallelo (il tempo di CPU resta identico), ma con quattro candidati pronti in
coda capita più spesso che almeno uno di loro abbia il turno. È un effetto
instabile, e un motivo in più per misurare a macchina scarica.

Il **tempo di CPU** stampato accanto serve a distinguere le due situazioni in
cui la parete non scende, e va letto con una cautela da dire: `process_time()`
somma il lavoro di tutti i thread, quindi resta all'incirca uguale sia quando
i thread si alternano sia quando lavorano davvero insieme, e da solo non
dimostra nulla. Quello che dice è un'altra cosa, utile: se la parete non
scende **e** la CPU è alta, si sta calcolando a turno (il caso del GIL); se la
parete non scende e la CPU è quasi zero, si sta solo aspettando.

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
- Un **traceback** si legge dal fondo: l'ultima riga dice *che cosa* è
  successo, risalendo si scopre *dove*. E il **paracadute** (`try`/`except`)
  si mette attorno alla riga fragile, non attorno a tutto il programma.
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
- I blocchi sono definiti dall’**indentazione**; `if/elif/else`, `for` e `while`
  bastano per il controllo di flusso.
- Le **funzioni** (`def` … `return`) e le **list comprehension** rendono il
  codice conciso; le **classi** modellano oggetti su misura: la stessa forma
  di scikit-learn e PyTorch.
- Le **f-string** (`f"{nome} ha {eta} anni"`) inseriscono valori nel testo, con
  il formato dopo i due punti: `f"{loss:.3f}"`.
- Le eccezioni si gestiscono con `try`/`except` **stretto** (mai `except:`
  nudo), `else`/`finally` per il seguito e le pulizie, `raise` per rilanciare:
  è lo stile EAFP, e il traceback si legge dall'ultima riga risalendo la pila
  delle chiamate.
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
