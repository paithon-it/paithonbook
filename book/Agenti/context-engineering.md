# Context engineering: il contesto è l'interfaccia

Due squadre costruiscono un assistente che risponde ai clienti di un negozio.
Si rivolgono allo stesso identico modello, dallo stesso fornitore, con le
stesse impostazioni. Una ottiene risposte precise e nel tono giusto; l'altra,
risposte vaghe che inventano politiche di rimborso mai esistite. Nessuno ha
addestrato niente: la differenza sta tutta in *cosa* le due squadre scrivono
davanti al modello prima di premere invio, cioè quali istruzioni, quali
esempi, quali documenti e in quale ordine. Il modello è lo stesso motore;
cambia il carburante che gli versi nel serbatoio.

Quello «davanti al modello» ha un nome preciso ed è il perno di questa
sezione. Un modello legge tutto in un colpo solo, e quanto testo riesca a
tenere davanti agli occhi in una volta è un numero fisso, deciso da chi l'ha
costruito. Quello spazio si chiama **finestra di contesto**, e ciò che ci
scrivi dentro si chiama **contesto**. Larga quanto vuoi, resta finita.

È il cuore di questo capitolo: con un modello di oggi non si programma
scrivendo codice, si programma scrivendo il contesto. Il modello non si tocca
e non si modifica; l'unica cosa su cui hai davvero le mani è il testo che gli
metti davanti. Nel {doc}`capitolo sui Transformer </Transformers/overview>` questa scoperta ha un nome
inglese, l’*in-context learning*, l'imparare dal contesto: gli descrivi il
compito lì dentro, magari con due esempi, e lui lo esegue senza che nessuno
abbia cambiato una virgola dentro di lui. È un modo di comandare un programma
scrivendo in italiano invece che in codice, e come tutti i comandi è potente e
fragile insieme: una parola diversa cambia la risposta.

Il mestiere che nasce da lì si chiama **context engineering**, l'ingegneria
del contesto, ed è l'arte di riempire bene quella finestra. Per un po’ lo si è
chiamato *prompt engineering*, come se il problema fosse trovare la formula
magica, la frase che sblocca il modello. Chi costruisce applicazioni ha
imparato che il problema vero è un altro: non la frase perfetta, ma il governo
di **tutto** ciò che entra nella finestra, a ogni passo, con i vincoli e i
compromessi di qualunque problema di ingegneria.

E per un agente il problema è ancora più acuto. Come abbiamo visto nelle
sezioni precedenti, il ciclo osserva → ragiona → agisci **riempie il contesto
da sé**: ogni pensiero, ogni chiamata a uno strumento, ogni osservazione di
ritorno è testo che si accumula. Dopo dieci passi la finestra trabocca di
cronologia, cioè dell'elenco di tutto quel che si è detto e fatto finora, e
decidere cosa tenere e cosa buttare è ciò che distingue un
agente che arriva in fondo da uno che si perde.

Qui ne diamo la versione essenziale, quella che serve a un agente: come si
spende lo spazio, dove il modello legge bene e dove male, e che forme prende
la memoria. Il tema ha però un capitolo tutto suo più avanti, *Prompt,
contesto e loop*, che lo allarga oltre l'agente. Là si vedrà come si scrive il
singolo messaggio, in quanti modi un contesto si guasta, e come si costruisce
il ciclo che quella finestra la ri-riempie a ogni passo. Qui restiamo sul filo
del ragionamento dell'agente; là si guarda il quadro intero.

## Il prompt come artefatto, non come incantesimo

Cominciamo col ridimensionare la parola «prompt». Nell'uso comune evoca la
frase d'istruzione che si scrive nella casella della chat. Ma in
un'applicazione vera fra l'utente e il modello c'è di mezzo un programma, ed è
quel programma a comporre il testo che il modello riceve: prende l'ultima cosa
scritta dall'utente e le cuce attorno tutto il resto, ogni volta da capo.

Il prompt, allora, è un **oggetto montato a pezzi** e non una frase scritta di
getto, e i pezzi hanno ruoli diversi. Le istruzioni di fondo, gli esempi che
mostrano il comportamento voluto, il formato preciso in cui vogliamo la
risposta, e solo alla fine la richiesta dell'utente.

`````{tab} Elementare

Un collega nuovo arriva il lunedì e gli affidi i reclami dei clienti. Puoi
buttargli lì un ordine di corsa («rispondi ai reclami») e sperare bene: farà
del suo meglio, ma a modo suo, e non ti stupire se sbaglia tono o inventa una
regola. Oppure gli lasci un **briefing scritto**, che resta sulla scrivania e
vale per ogni reclamo che arriva: «sei l'assistente dell'assistenza; rispondi
in modo cortese e conciso; non promettere rimborsi oltre i 30 giorni; ecco tre
reclami già evasi, con la risposta giusta accanto; scrivi sempre nel formato:
saluto, soluzione, firma». L'ultima riga serve a chi maneggia la risposta dopo
di lui, perché quel foglio finisce in archivio, dove la firma si cerca sempre
allo stesso posto, e uno composto a modo proprio lì si inceppa. Stesso collega,
nessun corso di formazione, risultati incomparabili. Il briefing è un documento
di lavoro, e come ogni documento si scrive, si rilegge, si corregge quando
qualcosa non va.

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

La conseguenza pratica è netta, e la riprenderemo nel {doc}`capitolo su MLOps </MLOps/overview>`: **il
prompt è codice**. Quella riga d'istruzione che orienta il modello è fragile
(una parola diversa cambia la risposta) e quindi va trattata come si tratta il
software. Se ne tiene la **storia**, cioè si conserva ogni versione con la data
e il motivo del cambiamento, invece di sovrascriverla; la si prova su una
batteria di casi noti; e prima di sostituirla si fanno girare le due versioni
in parallelo sugli stessi casi, per vedere quale risponde meglio. È l'unico
modo di sapere se la modifica di ieri ha migliorato o
peggiorato le risposte di oggi. Il prompt magico non esiste; esiste il prompt
*provato*.

## La finestra è piccola e preziosa

La finestra di contesto ha una misura, e la misura è un numero preciso. Non si
conta in parole né in pagine, ma in **token**: i pezzetti in cui una frase
viene tagliata prima di entrare nel modello, ciascuno grande all'incirca una
parola, spesso un po’ meno. Ogni modello dichiara quanti token riesce a
leggere in una volta, e oltre quel numero non si va.

Riempirli, poi, non è gratis, e da qui in avanti «quanto costa» vorrà sempre
dire «quanti token».

Il prezzo si paga in tre valute, e le conosciamo già. In **memoria**, perché
il segnalibro che il modello si tiene per non rileggere ogni volta da capo (nel
{doc}`capitolo sui Transformer </Transformers/overview>` lo chiamavamo **KV cache**) cresce con la lunghezza
del contesto. In **secondi di attesa**, perché più testo c'è, più tempo passa
prima che compaia la risposta. E in **denaro**, perché un modello si paga a
consumo, un tanto per ogni token che entra e per ogni token che esce: quel
listino si chiama *costo per token*, e sarà uno dei temi del capitolo su
MLOps. Un prompt gonfio è una bolletta più salata e una risposta più lenta, e
riempire la finestra fino all'orlo «per sicurezza» è quasi sempre un cattivo
affare.

Il disegno che segue mostra come una finestra si riempie in una giornata di
lavoro vera. Due delle voci hanno un nome che non abbiamo ancora dato. La
prima è il *system prompt*: il foglio di istruzioni di fondo che il programma
antepone sempre, uguale a ogni richiesta, e che l'utente non vede né scrive. La
seconda sono le *definizioni tool*, cioè il catalogo degli strumenti della
prima sezione, con nome, descrizione e argomenti di ciascuno: sta lì dentro
perché il modello lo legge come legge tutto il resto, e quindi si paga.

```{figure} ../figures/context-window.svg
:name: fig-context-window
:alt: "Una barra orizzontale rappresenta il budget di una finestra di contesto da centoventottomila token, ripartita in segmenti di ampiezza diversa e via via crescente: il system prompt (circa seimila token), le descrizioni degli strumenti (diecimila), la cronologia della conversazione (trentacinquemila, e cresce a ogni turno), i documenti allegati (cinquantottomila) e, tratteggiato in coda, lo spazio che resta per la risposta: diciannovemila token. In fondo l'avvertenza che i valori sono indicativi."
:width: 92%

La finestra come budget da ripartire. Ogni segmento toglie spazio agli altri,
e l'ultimo (lo spazio per la risposta) è quello che si dimentica di contare
finché il modello non la tronca a metà.
```

Messa così, come in {numref}`fig-context-window`, la finestra smette di
sembrare un limite tecnico e diventa quello che è davvero: un **budget**. I
centoventottomila token in cima al disegno sono grosso modo un romanzo, e
sembrano tantissimi finché non si guarda quanti se ne prende ciascun
commensale. E come ogni budget si può spendere bene o male, perché le voci
competono fra loro: una descrizione di strumento scritta larga, una cronologia
che nessuno accorcia mai, dieci documenti recuperati dove ne bastavano tre.
Nessuna di queste è un errore in sé, ma insieme mangiano lo spazio della
risposta, che è l'ultimo segmento e l'unico che nessuno pensa a contare.

C'è di peggio, e va contro l'intuizione: **anche quando lo spazio ci sarebbe,
riempirlo può danneggiare la risposta**. Nel 2023 Nelson Liu e colleghi lo
hanno misurato in un lavoro dal titolo eloquente, *Lost in the Middle*
{cite}`liu2024lost`: i modelli usano bene l'informazione che sta
all’**inizio** e alla **fine** del contesto, e trascurano quella sepolta **in
mezzo**.

`````{tab} Elementare

Sulla tua scrivania ci sta solo un certo numero di fogli davanti a te: oltre
quelli finiscono nel cassetto e li dimentichi. C'è poi un secondo effetto, più
sottile, che chiunque abbia studiato conosce: di una pila di
fogli, l'occhio cade sul **primo** e sull’**ultimo**. Quelli in mezzo li
sfogli distrattamente, e più la pila cresce più quel centro si allarga. Se
metti l'informazione che conta proprio lì, rischi di non «vederla» nemmeno se
ce l'hai sotto il naso: rispondi come se quel foglio non l'avessi mai avuto, e
certe volte peggio, come se le carte scorse in fretta ti avessero confuso
invece di aiutarti. Vale per te alla scrivania e, sorprendentemente, vale
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

La finestra, allora, è quello che l'agente tiene «in testa» adesso: la sua
**memoria di lavoro**, veloce da consultare, stretta, e che sparisce appena la
conversazione finisce. Ma un agente serio deve ricordare anche oltre il singolo
scambio: chi è l'utente, cosa si è detto ieri, cosa contengono mille pagine di
documentazione che nella finestra non entrerebbero mai tutte insieme. Serve
una **memoria a lungo termine**, e per forza deve stare *fuori* dal contesto.

`````{tab} Elementare

A mente tieni giusto le poche cose che ti servono *ora*: è veloce, ma ci sta
poco e svanisce. Su un taccuino invece finisce tutto quello che hai annotato
nel tempo: non lo leggi tutto insieme, lo apri alla pagina giusta quando ti
serve.

Un agente fa lo stesso. Nel breve termine usa un **foglio di brutta** dentro
la finestra: ci scrive i risultati intermedi, i conti a metà, gli appunti del
compito in corso. Nel lungo termine tiene uno **schedario** fuori, e la sua
bravura sta nel pescarne solo la pagina che serve adesso, invece di tenere
tutto aperto sul tavolo (dove non ci starebbe, e dove si perderebbe nel mezzo).

Nello schedario finiscono tre generi di cose, e conviene distinguerle perché si
recuperano in modi diversi. I **documenti**, che si vanno a cercare come
abbiamo visto parlando di RAG. I **riassunti** di quello che si è già detto:
quando una conversazione si allunga troppo, invece di portarsela dietro parola
per parola se ne tiene un sunto, che costa una frazione dello spazio. E i
**fatti sull'utente**, cioè le poche cose che valgono sempre (come si chiama,
che lingua parla, cosa ha già chiesto tre volte), tenute a parte e rimesse
davanti al modello quando c'entrano.

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
Il nodo difficile è la **politica di rimozione**
(*eviction*) (cosa promuovere a lungo termine, cosa comprimere, cosa
dimenticare), perché la finestra è il collo di bottiglia e ogni token speso a
ricordare è un token in meno per ragionare.

`````

Conviene insistere su dove sia la difficoltà, perché non è dove sembra.
Ricordare è facile: uno schedario si allarga quanto si vuole, e scriverci
dentro non costa quasi niente. La difficoltà è a ogni singolo passo, quando
bisogna decidere che cosa di tutto quel materiale merita di occupare la
finestra *adesso*. Ogni riga che ci metti per ricordare è una riga in meno per
ragionare, e il conto lo si paga subito. Il problema è di scelta, prima che
di memoria: la domanda difficile non è cosa tenere, è cosa lasciare
fuori.

## Assemblare il contesto, con un budget

Mettiamo insieme le tre lezioni (il prompt montato a pezzi, la finestra
costosa, l'informazione che si perde nel mezzo) in un pezzo di codice che le
rende concrete. Il programma di cui parlavamo all'inizio, quello che a ogni
passo cuce insieme il testo da mandare al modello, ha un nome: si chiama
**context builder**, il montatore del contesto. È il punto in cui quelle
decisioni smettono di
essere opinioni e diventano righe che qualcuno esegue.

`````{tab} Elementare

È come fare la valigia con un limite di peso. Alcune cose non si discutono,
documenti, biglietti: entrano comunque. Per il resto non provi tutte le
combinazioni possibili di ciò che entra e ciò che no, sono troppe e il taxi è
sotto; scendi per ordine finché lo spazio dura, prima l'indispensabile, poi il
molto utile, e fra due cose che servono uguale quella che pesa meno. Ciò che
resta fuori resta fuori. Se qualcosa quasi ci sta, a volte lo porti a metà,
sapendo che è un compromesso zoppo: mezzo maglione non tiene caldo, e mezza
frase non dice niente. E se sai che chi la aprirà guarderà per prima cosa
quello che sta sopra e quello che sta sotto, mentre quello sepolto in mezzo
rischia di non vederlo, le cose che contano le metti ai due estremi. Il context
builder fa la valigia del modello: gli obbligatori dentro, il resto per
priorità fino a esaurire il budget, e il più prezioso mai nel mezzo.

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
spazio residuo invece di essere buttato del tutto. Il raffinamento è
discutibile: un troncamento a metà frase occupa token e restituisce un
frammento che non afferma niente, quindi spesso conviene tagliare a confine di
frase, e scartare il passaggio se non ne resta almeno una intera.

Il secondo: i passaggi scelti vengono **riordinati**, e non semplicemente
messi in ordine crescente di rilevanza. La curva di Liu e colleghi è una **U**,
si legge bene all'inizio *e* alla fine, quindi disporre per rilevanza crescente
ottimizzerebbe un estremo solo e regalerebbe l'altro, quello di apertura, al
pezzo peggiore. La disposizione che segue la curva è a **V**: i due passaggi migliori ai due
estremi, e i meno rilevanti sepolti nel mezzo, dove costano meno perderli.
Quale dei due estremi meriti il migliore la curva non lo dice, e qui il più
rilevante va in fondo, a ridosso della domanda, mentre il secondo va in testa. Nelle librerie di RAG questo
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
spazio, i due meno rilevanti restano fuori. E la disposizione finale ha la
forma di una **V**: molto ai due estremi, poco nel mezzo. È la risposta
esatta alla pila di fogli, cioè al fatto che il modello legge bene l'inizio e
la fine e trascura il centro. Il passaggio decisivo, quello che contiene il
2017, finisce in fondo, a ridosso della domanda; il secondo apre; il frammento
troncato, che è la parte meno utile perché tagliato a metà non afferma niente,
finisce nel mezzo, dove perderlo costa meno.

```text
Sei un assistente che risponde citando solo i passaggi forniti. Se l'informazione non c'e', dillo.

[fonte 0.75] L'attenzione scaled dot-product e' il cuore del Transformer.
[fonte 0.60 (troncata)] L'architettura Transformer abbandona la …
[fonte 0.95] Il paper 'Attention Is All You Need' introduce i Transformer nel 2017.

Domanda: In che anno e' stato pubblicato il paper sui Transformer?

Token usati: 58/58
```

Il numero in fondo è il conteggio del prompt **davvero montato**, marcatori
compresi, non la somma dei pezzi che abbiamo scelto. Quelli, contati a parte,
pesano cinquantuno token: quindici il system prompt, undici la domanda (dieci
di testo più la parola «Domanda:»), venticinque i tre passaggi messi in fila.
I sette che mancano all'appello sono i `[fonte 0.95]` e simili: sette token su
cinquantuno, cioè quasi il quattordici per cento in più di quanto sembrava di
aver speso. È esattamente il tipo di sforamento che si scopre tardi, quando il
modello tronca la risposta a metà.

Alla selezione per sola rilevanza manca però un occhio: i passaggi più
rilevanti per la stessa domanda tendono a somigliarsi fra loro, e un budget
speso su due passaggi quasi uguali è mezzo budget. Il correttivo classico si
chiama **maximal marginal relevance** (MMR) {cite}`carbonell1998use`: invece
di prendere i passaggi in ordine di rilevanza, a ogni giro si sceglie quello
che massimizza

$$
\lambda \,\mathrm{sim}(p, q) \;-\; (1-\lambda) \max_{p' \in S} \mathrm{sim}(p, p'),
$$

cioè la somiglianza con la domanda $q$ **meno** la somiglianza con il più
vicino fra i passaggi $S$ già scelti, pesate da un $\lambda$ fra zero e uno.
Il secondo termine compra la novità: un passaggio rilevantissimo ma fotocopia
di uno già dentro perde il posto a favore di uno un po’ meno rilevante che
aggiunge qualcosa. Con $\lambda = 1$ si torna alla pura rilevanza.

Sono poche decine di righe che non «capiscono» nulla, eppure incarnano tre
scelte di progetto: cosa è obbligatorio, cosa entra per priorità, dove va il
pezzo più importante. In un sistema reale la rilevanza non è un numero scritto
a mano ma esce dalla ricerca nell'archivio della sezione precedente; il
conteggio dei token non si fa a parole ma con lo stesso programma che li taglia
davvero per quel modello, il **tokenizzatore**; e le politiche sono più ricche.
Ma l'ossatura è questa, ed è questa a fare la differenza tra le due squadre da
cui siamo partiti.

## Pensare costa token: il ragionamento come context engineering

Un'ultima osservazione chiude il cerchio. Nelle sezioni precedenti abbiamo
fatto «ragionare ad alta voce» l'agente prima di agire, cioè scrivere i
passaggi intermedi nel contesto prima della conclusione: è la catena di
ragionamento, la **chain-of-thought** {cite}`wei2022chain`. Vista con gli occhi
di questa sezione, quella catena è *anch'essa* ingegneria del contesto: si
spende deliberatamente una parte del budget in token di «pensiero» per
comprarne qualità di risposta. Il ragionamento non è gratis, perché occupa
finestra e fa aspettare, ma spesso rende più di quanto costa.

L'idea si può spingere oltre. Invece di seguire un unico filo fino in fondo, si
possono aprire **più strade di ragionamento**, guardare dove portano e tenere
solo le migliori: è il **Tree of Thoughts** («albero di pensieri»)
{cite}`yao2023tree`. L'immagine è quella di chi risolve un labirinto: a ogni
bivio si prova una strada, e se dopo qualche passo non promette niente di
buono si torna al bivio e si prende l'altra, invece di andare avanti per
inerzia. Le strade si possono anche tentare tutte insieme, ma la cosa che
conta è un'altra, ed è quella che il filo unico non permette: poter **tornare
indietro** da una strada che non promette. Il guadagno in problemi che richiedono
pianificazione è reale; il prezzo pure, ed è sempre lo stesso: più token, più
tempo, più costo. È il compromesso di fondo del context engineering, in una
forma nuova: la finestra è un budget, e ogni cosa che ci metti (istruzioni,
esempi, memoria recuperata, o il pensiero stesso del modello) la paghi, e va
messa dove rende di più.



`````{tab} Elementare

```{admonition} Da ricordare
:class: important
- Con un modello istruito non si programma scrivendo codice, ma scrivendo il
  **contesto**: quello che gli metti davanti prima di fargli la domanda è
  l'unico comando che hai. Il mestiere di riempire bene quello spazio vale più
  di qualunque «frase magica».
- Il **prompt** è un **documento di lavoro** montato a
  pezzi, non un incantesimo: le istruzioni di fondo, qualche esempio svolto, il
  formato in cui si
  vuole la risposta, e solo alla fine la richiesta. E siccome una parola diversa
  cambia il risultato, va trattato come si tratta il software: se ne conserva
  la storia, lo si prova su casi noti, si confrontano due versioni prima di
  sostituirne una.
- La finestra è **piccola e costosa**: ogni parola che ci metti la paghi in
  memoria, in attesa e in denaro. E c'è la trappola dei **fogli in mezzo alla
  pila** (in inglese *lost in the middle* {cite}`liu2024lost`): il modello usa
  bene l'inizio e la fine di quello che legge, e trascura il centro. Quindi
  l’**ordine conta**.
- **Memoria**: a breve termine il **foglio di brutta** dentro la finestra, dove
  l'agente scrive i conti a metà; a lungo termine uno **schedario esterno** da
  cui pescare solo la pagina che serve adesso (i documenti recuperati, i
  riassunti di quello che si è detto, i fatti sull'utente tenuti a parte). Il
  problema difficile è decidere cosa **lasciare fuori** dalla
  finestra adesso: ogni riga spesa a ricordare è una riga in meno per ragionare.
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
  bene l'inizio e la fine del contesto, male il centro. Quindi l’**ordine
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
