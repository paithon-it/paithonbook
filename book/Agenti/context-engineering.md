# Context engineering: il contesto è l'interfaccia

Due squadre costruiscono un assistente per l'assistenza clienti. Si rivolgono
allo stesso identico modello, dallo stesso fornitore, con le stesse
impostazioni. Una ottiene
risposte precise e nel tono giusto; l'altra, risposte vaghe che inventano
politiche di rimborso mai esistite. Nessuno ha addestrato niente: la
differenza sta tutta in *cosa* le due squadre scrivono nella finestra del
modello prima di premere invio (quali istruzioni, quali esempi, quali
documenti, in quale ordine). Il modello è lo stesso motore; cambia il
carburante che gli versi nel serbatoio.

È il cuore di questo capitolo. Con un LLM istruito non si programma scrivendo
codice: si programma scrivendo il **contesto**. Nel capitolo sui Transformer
abbiamo chiamato questa scoperta *in-context learning* (si descrive un compito
nel prompt e il modello lo esegue, senza toccare un solo peso) e abbiamo detto
che il prompt è «un'interfaccia di programmazione in linguaggio naturale,
potente e fragile insieme». Il **context engineering** è il mestiere che nasce
da lì: l'arte di riempire bene una finestra limitata. Per un po' lo si è
chiamato *prompt engineering*, come se il problema fosse trovare la formula
magica, la frase che sblocca il modello. Chi costruisce applicazioni LLM ha
imparato che il problema vero è un altro: non la frase perfetta, ma il
**governo di ciò che entra nella finestra** a ogni passo (un problema di
ingegneria, con vincoli, budget e compromessi).

E per un agente il problema è ancora più acuto. Come abbiamo visto nelle sezioni
sul tool use, il ciclo osserva → ragiona → agisci **riempie il contesto da sé**:
ogni pensiero, ogni chiamata a uno strumento, ogni osservazione di ritorno è
testo che si accumula. Dopo dieci passi la finestra trabocca di cronologia, e
decidere cosa tenere e cosa buttare non è un dettaglio: è ciò che distingue un
agente che arriva in fondo da uno che si perde.

Questa sezione ne dà la versione essenziale, quella che serve a un agente: la
meccanica del budget, il *lost in the middle*, le forme di memoria. Il tema ha
però un capitolo dedicato (*Prompt, contesto e loop*) che lo allarga oltre
l'agente, in tre livelli concentrici: il **prompt engineering** (come si
scrive il singolo messaggio), il **context engineering** come disciplina a sé
(le quattro mosse per governare la finestra, e i modi in cui un contesto si
guasta) e il **loop engineering** (il ciclo che ri-riempie la finestra a ogni
passo, e che non lascia passare un risultato finché qualcosa non lo ha
verificato). Qui restiamo sul filo del ragionamento agentico; là si guarda il
quadro intero.

## Il prompt come artefatto, non come incantesimo

Cominciamo col ridimensionare la parola «prompt». Nell'uso comune evoca la frase
d'istruzione che si scrive nella casella della chat. Ma nelle applicazioni serie
il prompt è un **artefatto strutturato**, cioè un oggetto costruito a pezzi e
non una frase scritta di getto: lo monta il programma prima di interpellare il
modello, ed è fatto di parti con ruoli diversi. Le istruzioni di fondo, gli
esempi che mostrano il comportamento voluto, il formato preciso in cui vogliamo
la risposta, e solo alla fine la richiesta dell'utente.

`````{tab} Elementare

Immagina di affidare un compito a un nuovo collega. Puoi buttargli lì un
ordine di corsa («rispondi ai reclami») e sperare bene: farà del suo meglio,
ma a modo suo, e non ti stupire se sbaglia tono o inventa una regola. Oppure
gli lasci un **briefing scritto**: «sei l'assistente dell'assistenza; rispondi
in modo cortese e conciso; non promettere mai rimborsi oltre i 30 giorni; ecco
tre esempi di reclami con la risposta giusta accanto; scrivi sempre nel
formato: saluto, soluzione, firma». Stesso collega, risultati incomparabili.
Il briefing non è una parola magica: è un documento di lavoro, e come ogni
documento si scrive, si rilegge, si corregge quando qualcosa non va.

`````

`````{tab} Superiore

Un prompt strutturato ha almeno tre strati. Il **system prompt** fissa il ruolo
e le regole invarianti («sei un assistente di supporto; non riveli dati
interni»); resta identico a ogni richiesta ed è la spina dorsale del
comportamento. Le **istruzioni** e gli **esempi** (*few-shot*) mostrano il
compito: qualche coppia richiesta → risposta corretta condiziona il modello
senza alcun aggiornamento dei pesi. È lo stesso meccanismo dell'in-context
learning del capitolo sui Transformer: il modello stima

$$
\hat{y} = \arg\max_{y}\; P\!\left(y \mid I,\; (x_1, y_1), \dots, (x_k, y_k),\; x\right),
$$

dove $I$ sono le istruzioni, le coppie $(x_i, y_i)$ sono i $k$ esempi svolti,
$x$ è la richiesta corrente e $\hat{y}$ la risposta generata. Gli esempi non
addestrano nulla: sono *condizionamento*, contesto che sposta la distribuzione
del modello verso lo stile e il formato desiderati. Il terzo strato è il
**formato dell'output**, spesso uno schema (JSON, campi obbligatori), che
rende la risposta interpretabile dal programma a valle, non solo leggibile da
un umano.

`````

La conseguenza pratica è netta, e la riprenderemo nel capitolo su MLOps: **il
prompt è codice**. Quella riga d'istruzione che orienta il modello è fragile
(una parola diversa cambia la risposta) e quindi va trattata come si tratta il
software. Se ne tiene la **storia**, cioè si conserva ogni versione con la data
e il motivo del cambiamento, invece di sovrascriverla; la si prova su una
batteria di casi noti; e prima di sostituirla si fanno girare le due versioni
in parallelo sugli stessi casi, per vedere quale risponde meglio. Non è
pignoleria: è l'unico modo di sapere se la modifica di ieri ha migliorato o
peggiorato le risposte di oggi. Il prompt magico non esiste; esiste il prompt
*provato*.

## La finestra è piccola e preziosa

Se il contesto è l'interfaccia, la finestra di contesto è lo schermo su cui la
disegniamo, ed è uno schermo finito. Ogni modello ha un tetto massimo di token
che può leggere in una volta, e riempirlo non è gratis. Lo sappiamo dal
capitolo sui Transformer: il segnalibro che il modello si tiene per non
rileggere ogni volta da capo (la **KV cache**) cresce con la lunghezza del
contesto; e il prezzo che si paga in memoria, in secondi di attesa e in denaro
per ogni parola che entra ed esce (il **costo per token**) sarà uno dei temi
del capitolo su MLOps. Un prompt gonfio è una bolletta più salata e una
risposta più lenta. Riempire la finestra fino all'orlo «per sicurezza» è quasi
sempre un cattivo affare.

```{figure} ../figures/context-window.svg
:name: fig-context-window
:alt: "Una barra orizzontale rappresenta il budget di token di una finestra di contesto, ripartita in segmenti di ampiezza diversa: il system prompt, le descrizioni degli strumenti, la cronologia della conversazione, i documenti recuperati e lo spazio che resta per la risposta."
:width: 92%

La finestra come budget da ripartire. Ogni segmento toglie spazio agli altri,
e l'ultimo (lo spazio per la risposta) è quello che si dimentica di contare
finché il modello non la tronca a metà.
```

Messa così, come in {numref}`fig-context-window`, la finestra smette di
sembrare un limite tecnico e diventa quello che è davvero: un **budget**. Il
primo segmento della barra, il *system prompt*, è il foglio di istruzioni di
fondo che il programma antepone sempre, uguale a ogni richiesta, e che
l'utente non vede né scrive: è la parte fissa della spesa. E come ogni budget
si può spendere bene o male, perché le voci competono fra loro. Una descrizione
di strumento scritta larga, una cronologia che nessuno pota mai, dieci
documenti recuperati dove ne bastavano tre: nessuna di queste è un errore in
sé, ma insieme mangiano lo spazio della risposta.

C'è di peggio, e va contro l'intuizione: **anche quando lo spazio ci sarebbe,
riempirlo può danneggiare la risposta**. Nel 2023 Nelson Liu e colleghi lo
hanno misurato in un lavoro dal titolo eloquente, *Lost in the Middle*
{cite}`liu2024lost`: i modelli usano bene l'informazione che sta
all'**inizio** e alla **fine** del contesto, e trascurano quella sepolta **in
mezzo**.

`````{tab} Elementare

Pensa alla tua scrivania: ci sta solo un certo numero di fogli davanti a te,
oltre quelli finiscono nel cassetto e li dimentichi. Ma c'è un secondo
effetto, più sottile, che chiunque abbia studiato conosce: di una pila di
fogli, l'occhio cade sul **primo** e sull'**ultimo**. Quelli in mezzo li
sfogli distrattamente. Se metti l'informazione che conta proprio lì (impilata
al centro, tra decine di altre carte), rischi di non «vederla» nemmeno se ce
l'hai sotto il naso. Vale per te alla scrivania e, sorprendentemente, vale
anche per il modello: il posto peggiore dove mettere la cosa importante è il
centro di un contesto lungo.

`````

`````{tab} Superiore

Liu e colleghi variano la **posizione** del documento che contiene la risposta
dentro un contesto di molti documenti, e misurano l'accuratezza al variare di
quella posizione. La curva non è piatta: ha una forma a **U**. Detta $j$ la
posizione del passaggio rilevante su $n$ passaggi totali, l'accuratezza è
massima agli estremi ($j = 1$ e $j = n$) e cala vistosamente verso il centro
($j \approx n/2$): in alcuni casi il modello con l'informazione a metà
contesto fa *peggio* dello stesso modello a cui quell'informazione non viene
data affatto. Il calo si accentua man mano che il contesto si allunga. Due
implicazioni operative dirette. Primo: allungare il contesto non è un pasto
gratis; più passaggi si infilano, più è probabile seppellire quello giusto in
una zona cieca. Secondo: **l'ordine conta**. Se recuperiamo dei passaggi (per
esempio con un sistema RAG) e ne conosciamo la rilevanza stimata, conviene
collocare i più importanti in testa o in coda, non nel ventre molle del
contesto.

`````

## La memoria: oltre la finestra

La finestra, allora, è la **memoria di lavoro** dell'agente: piccola, veloce,
volatile (quello che tiene «in testa» adesso). Ma un agente serio ha bisogno
anche di ricordare oltre il singolo scambio: chi è l'utente, cosa si è detto
ieri, cosa contengono mille pagine di documentazione che non entrerebbero mai
tutte nella finestra. Serve una **memoria a lungo termine**, e per forza deve
stare *fuori* dal contesto.

`````{tab} Elementare

È la differenza tra ciò che tieni a mente e ciò che ti appunti su un taccuino.
A mente tieni giusto le poche cose che ti servono *ora*: è veloce, ma ci sta
poco e svanisce. Il taccuino invece contiene tutto quello che hai annotato nel
tempo: non lo leggi tutto insieme, lo apri alla pagina giusta quando ti serve.
Un agente fa lo stesso. Nel breve termine usa un **foglio di brutta** dentro
la finestra: ci scrive i risultati intermedi, i conti a metà, gli appunti del
compito in corso. Nel lungo termine tiene un archivio esterno (un grande
schedario) e la sua bravura sta nel **pescare dallo schedario solo la pagina
che serve adesso** e portarla nella finestra, invece di tenere tutto aperto
sul tavolo (dove non ci starebbe, e dove si perderebbe nel mezzo).

`````

`````{tab} Superiore

La **memoria a breve termine** è lo *scratchpad*: uno spazio nel contesto in
cui l'agente scrive i propri stati intermedi (la traccia ReAct delle sezioni
precedenti ne è un esempio) e che vive quanto vive la finestra. La **memoria a
lungo termine** è esterna e persistente. Tre forme ricorrono. La prima è il
**database vettoriale**: i ricordi (documenti, scambi passati) vengono
codificati in embedding e recuperati per similarità quando servono; è
esattamente il **RAG** visto nel capitolo sui Transformer, letto qui come un
meccanismo di memoria, non solo di recupero. La seconda è il **riassunto
progressivo**: quando la cronologia della conversazione si allunga, la si
comprime in un sunto che ne conserva l'essenziale a costo di token molto
minore, liberando finestra. La terza sono i **fatti strutturati** (preferenze,
identità, vincoli dell'utente) tenuti a parte e reiniettati quando pertinenti.
Il nodo difficile non è memorizzare: è la **politica di rimozione**
(*eviction*) (cosa promuovere a lungo termine, cosa comprimere, cosa
dimenticare), perché la finestra è il collo di bottiglia e ogni token speso a
ricordare è un token in meno per ragionare.

`````

## Assemblare il contesto, con un budget

Mettiamo insieme le tre lezioni (il prompt come artefatto, la finestra
costosa, il *lost in the middle*) in un pezzo di codice che le rende concrete.
Un **context builder** è la funzione che, a ogni passo, monta il prompt
effettivo rispettando un **budget di token**. Non è un dettaglio
implementativo: è il punto in cui le decisioni di context engineering
diventano operative.

`````{tab} Elementare

È come fare la valigia con un limite di peso. Alcune cose non si discutono,
documenti, biglietti: entrano comunque. Per il resto, con lo spazio che
avanza, metti prima l'indispensabile, poi il molto utile, e ciò che resta
fuori resta fuori. Se un oggetto quasi ci sta, a volte lo porti a metà (la
crema in un flaconcino invece del barattolone). E i regali più importanti li
metti in cima, dove li ritrovi subito. Il context builder fa la valigia del
modello: obbligatori dentro, il resto per priorità fino a esaurire il budget,
il più prezioso bene in vista.

`````

`````{tab} Superiore

È, in piccolo, un problema di **zaino** (*knapsack*): scegliere il
sottoinsieme di passaggi che massimizza la rilevanza totale $\sum_i r_i$ sotto
il vincolo che la somma dei costi in token $\sum_i c_i$ non superi il budget
disponibile, con system prompt e domanda pre-allocati come costi fissi. La
soluzione esatta è combinatoria; in pratica si usa un'euristica **greedy**
(passaggi in ordine di rilevanza decrescente, accettati finché entrano) con
due raffinamenti che vengono diritti dalle sezioni precedenti.

Il primo: l'ultimo passaggio che sfora viene **troncato** per riempire lo
spazio residuo invece di essere buttato del tutto. Va detto che è un
raffinamento discutibile, e conviene saperlo: un troncamento a metà frase
occupa token e restituisce un frammento che non afferma niente, quindi spesso
conviene tagliare a confine di frase, e scartare il passaggio se non ne resta
almeno una intera.

Il secondo: i passaggi scelti vengono **riordinati**, e non semplicemente
messi in ordine crescente di rilevanza. La curva di Liu e colleghi è una **U**,
si legge bene all'inizio *e* alla fine, quindi disporre per rilevanza crescente
ottimizzerebbe un estremo solo e regalerebbe l'altro, quello di apertura, al
pezzo peggiore. La disposizione che segue la curva è a **V**: il più rilevante
in fondo, appena sopra la domanda, il secondo in testa, e i meno rilevanti
sepolti nel mezzo, dove costano meno perderli. Nelle librerie di RAG questo
riordino porta il nome di *long-context reorder*.

Un'ultima nota di rigore, che non cambia il risultato ma cambia la regola.
Avendo ammesso il troncamento, lo zaino è diventato **frazionario**, e per
quel problema l'ottimo greedy si ottiene ordinando per **densità** $r_i / c_i$
(rilevanza per token), non per la sola rilevanza $r_i$. Sui numeri dell'esempio
che segue i due criteri scelgono gli stessi passaggi, ma la regola enunciata
non è quella che il modello dello zaino richiederebbe.

`````

Ecco il context builder in puro Python, nessuna libreria, il conteggio dei
token approssimato contando le parole, così che il meccanismo resti in piena
vista. Una cautela che sembra un dettaglio e non lo è: nel budget entrano anche
i **marcatori** che il montaggio aggiunge (`[fonte 0.95]` e simili). Sono
testo, il modello li legge, e un budget che non conta ciò che il montaggio
aggiunge non è un budget.

```python
# Un "context builder": dato un budget di token, assembla il prompt
# scegliendo i passaggi piu' importanti, troncando o scartando il resto,
# e collocando il pezzo piu' rilevante IN FONDO (contro il "lost in the middle").

def conta_token(testo):
    """Stima i token contando le parole: grezza, ma sufficiente per il budget."""
    return len(testo.split())

# System prompt e domanda sono obbligatori: entrano sempre, non si toccano.
system_prompt = (
    "Sei un assistente che risponde citando solo i passaggi forniti. "
    "Se l'informazione non c'e', dillo."
)
domanda = "In che anno e' stato pubblicato il paper sui Transformer?"

# I passaggi recuperati, ciascuno con una rilevanza (piu' alta = piu' utile).
passaggi = [
    (0.95, "Il paper 'Attention Is All You Need' introduce i Transformer nel 2017."),
    (0.20, "Le reti convoluzionali dominarono la visione artificiale negli anni 2010."),
    (0.60, "L'architettura Transformer abbandona la ricorrenza in favore dell'attenzione."),
    (0.10, "Il primo modello GPT fu addestrato su un corpus di libri."),
    (0.75, "L'attenzione scaled dot-product e' il cuore del Transformer."),
]


def riga_fonte(punteggio, testo, troncato=False):
    """La riga come finira' nel prompt. Il marcatore e' testo anche lui:
    entra nella finestra, quindi si paga e va contato."""
    return f"[fonte {punteggio:.2f}{' (troncata)' if troncato else ''}] {testo}"


COSTO_MARCATORE = conta_token(riga_fonte(0.0, "", troncato=True))


def costruisci_contesto(system_prompt, passaggi, domanda, budget):
    """Assembla un prompt che sta nel budget di token.
    Obbligatori: system prompt e domanda. I passaggi entrano per rilevanza
    decrescente finche' c'e' spazio; l'ultimo che sfora viene troncato; la
    disposizione finale e' a V, il migliore in fondo e il secondo in testa."""
    coda = f"Domanda: {domanda}"
    residuo = budget - conta_token(system_prompt) - conta_token(coda)
    if residuo < 0:
        raise ValueError("budget insufficiente perfino per system prompt e domanda")

    ordinati = sorted(passaggi, key=lambda p: p[0], reverse=True)
    scelti = []  # (punteggio, testo, troncato?), gia' per rilevanza decrescente
    for punteggio, testo in ordinati:
        costo = conta_token(riga_fonte(punteggio, testo))
        if costo <= residuo:                          # ci sta intero
            scelti.append((punteggio, testo, False))
            residuo -= costo
        elif residuo >= COSTO_MARCATORE + 2:          # non ci sta: lo tronco
            quante = residuo - COSTO_MARCATORE - 1    # -1 per il segno di taglio
            troncato = " ".join(testo.split()[:quante]) + " …"
            scelti.append((punteggio, troncato, True))
            residuo -= conta_token(riga_fonte(punteggio, troncato, True))
            break
        # altrimenti lo scarto e provo il prossimo (piu' corto o meno rilevante)

    # "lost in the middle": la curva e' a U, si legge bene all'inizio E alla
    # fine. Disposizione a V: il migliore in coda (a ridosso della domanda),
    # il secondo in testa, i peggiori sepolti nel mezzo.
    testa, fondo = [], []
    for n, scelto in enumerate(scelti):
        (fondo if n % 2 == 0 else testa).append(scelto)
    corpo = "\n".join(riga_fonte(p, txt, t) for p, txt, t in testa + fondo[::-1])

    prompt = f"{system_prompt}\n\n{corpo}\n\n{coda}"
    return prompt, conta_token(prompt)   # il conto vero, marcatori compresi


BUDGET = 58
prompt, usati = costruisci_contesto(system_prompt, passaggi, domanda, BUDGET)
print(prompt)
print(f"\nToken usati: {usati}/{BUDGET}")
```

L'esecuzione mostra le decisioni prese: dei cinque passaggi, i due più
rilevanti entrano interi, il terzo viene troncato per riempire l'ultimo
spazio, i due meno rilevanti restano fuori. E la disposizione finale è a **V**:
il passaggio decisivo, quello che contiene il 2017, finisce in fondo, a ridosso
della domanda; il secondo apre; il frammento troncato, che è la parte meno
utile perché non afferma niente, finisce nel mezzo, dove perderlo costa meno.

```text
Sei un assistente che risponde citando solo i passaggi forniti. Se l'informazione non c'e', dillo.

[fonte 0.75] L'attenzione scaled dot-product e' il cuore del Transformer.
[fonte 0.60 (troncata)] L'architettura Transformer abbandona la …
[fonte 0.95] Il paper 'Attention Is All You Need' introduce i Transformer nel 2017.

Domanda: In che anno e' stato pubblicato il paper sui Transformer?

Token usati: 58/58
```

Il numero in fondo è il conteggio del prompt **davvero montato**, marcatori
compresi, non la somma dei pezzi che abbiamo scelto: sono otto token di
differenza su cinquanta, cioè il sedici per cento, ed è esattamente il tipo di
sforamento che si scopre quando il modello tronca la risposta a metà.

Sono poche decine di righe che non «capiscono» nulla, eppure incarnano tre
scelte di progetto: cosa è obbligatorio, cosa entra per priorità, dove va il
pezzo più importante. In un sistema reale la rilevanza non è un numero scritto
a mano ma esce da un recupero (il RAG), il conteggio dei token usa il vero
tokenizzatore del modello, e le politiche sono più ricche, ma l'ossatura è
questa, ed è questa a fare la differenza tra le due squadre da cui siamo
partiti.

## Pensare costa token: il ragionamento come context engineering

Un'ultima osservazione chiude il cerchio. Nelle sezioni sul tool use abbiamo
fatto «ragionare ad alta voce» l'agente prima di agire: è la
**chain-of-thought** {cite}`wei2022chain`, i passaggi intermedi scritti nel
contesto prima della conclusione. Vista con gli occhi di questa sezione, la
catena di ragionamento è *anch'essa* context engineering: si spende
deliberatamente una parte del budget in token di «pensiero» per comprarne
qualità di risposta. Il ragionamento non è gratis (occupa finestra, costa
latenza) ma spesso rende più di quanto costa.

L'idea si può spingere oltre. Invece di una singola catena lineare, si possono
esplorare **più linee di ragionamento in parallelo**, valutarle e tenere le
migliori: è il **Tree of Thoughts** («albero di pensieri»)
{cite}`yao2023tree`. L'immagine è quella di chi risolve un labirinto: a ogni
bivio si prova una strada, e se dopo qualche passo non promette niente di
buono si torna al bivio e si prende l'altra, invece di andare avanti per
inerzia. Al posto di un unico filo di ragionamento c'è un ventaglio di strade
tentate e abbandonate. Il guadagno in problemi che richiedono
pianificazione è reale; il prezzo pure, ed è sempre lo stesso: più token, più
tempo, più costo. È il compromesso di fondo del context engineering, in una
forma nuova: la finestra è un budget, e ogni cosa che ci metti (istruzioni,
esempi, memoria recuperata, o il pensiero stesso del modello) la paghi, e va
messa dove rende di più.

Sei righe per rileggere la sezione.

`````{tab} Elementare

```{admonition} Da ricordare
:class: important
- Con un modello istruito non si programma scrivendo codice, ma scrivendo il
  **contesto**: quello che gli metti davanti prima di fargli la domanda è
  l'unico comando che hai. Il mestiere di riempire bene quello spazio vale più
  di qualunque «frase magica».
- Il **prompt** non è un incantesimo, è un **documento di lavoro** montato a
  pezzi: le istruzioni di fondo, qualche esempio svolto, il formato in cui si
  vuole la risposta, e solo alla fine la richiesta. E siccome una parola diversa
  cambia il risultato, va trattato come si tratta il software: se ne conserva
  la storia, lo si prova su casi noti, si confrontano due versioni prima di
  sostituirne una.
- La finestra è **piccola e costosa**: ogni parola che ci metti la paghi in
  memoria, in attesa e in denaro. E c'è la trappola dei **fogli in mezzo alla
  pila** (in inglese *lost in the middle* {cite}`liu2024lost`): il modello usa
  bene l'inizio e la fine di quello che legge, e trascura il centro. Quindi
  l'**ordine conta**.
- **Memoria**: a breve termine il **foglio di brutta** dentro la finestra, dove
  l'agente scrive i conti a metà; a lungo termine uno **schedario esterno** da
  cui pescare solo la pagina che serve adesso (i documenti recuperati, i
  riassunti di quello che si è detto, i fatti sull'utente tenuti a parte). Il
  problema difficile non è ricordare, è decidere cosa dimenticare.
- Assemblare il contesto è come fare la **valigia con un limite di peso**:
  prima l'indispensabile, poi il resto per priorità finché entra, e quel che
  quasi ci sta lo porti a metà. La cosa più importante va messa dove la
  ritrovi, cioè in fondo, appena prima della domanda.
- Anche **pensare costa**: far ragionare il modello a voce alta prima di
  rispondere {cite}`wei2022chain`, o fargli provare più strade e tornare
  indietro da quelle che non promettono (**Tree of Thoughts**
  {cite}`yao2023tree`), compra qualità spendendo spazio nella finestra. Come
  ogni spesa, va fatta dove rende.
```

`````

`````{tab} Superiore

```{admonition} Da ricordare
:class: important
- Con un LLM istruito non si programma col codice ma col **contesto**: ciò che
  metti nella finestra prima di chiedere è l'interfaccia. Il **context
  engineering** è il mestiere di riempirla bene: più del «prompt magico».
- Il **prompt** è un artefatto strutturato (system prompt, esempi *few-shot*
  come condizionamento, formato dell'output), non un incantesimo. Ed è **codice**:
  va versionato, testato e confrontato, come vedremo in LLMOps.
- La finestra è **finita e costosa**: ogni token pesa su KV cache e costo per
  token, e nel budget vanno contati anche i marcatori che il montaggio
  aggiunge. E c'è il **lost in the middle** {cite}`liu2024lost`: i modelli usano
  bene l'inizio e la fine del contesto, male il centro. Quindi l'**ordine
  conta**.
- **Memoria**: a breve termine lo *scratchpad* nella finestra; a lungo termine
  una memoria esterna (database vettoriale/RAG, riassunti progressivi, fatti
  strutturati). Il problema difficile è decidere cosa ricordare e cosa dimenticare.
- Assemblare il contesto è un problema di **budget** (uno zaino, e per giunta
  frazionario una volta ammessa la troncatura): obbligatori fissi, passaggi per
  rilevanza finché entrano, e disposizione a **V**, il più rilevante in fondo e
  il secondo in testa, perché la curva del *lost in the middle* è a U.
- Anche il **ragionamento** è context engineering: chain-of-thought
  {cite}`wei2022chain` e la sua estensione ad albero, il **Tree of Thoughts**
  {cite}`yao2023tree`, comprano qualità spendendo token di «pensiero».
```

`````
