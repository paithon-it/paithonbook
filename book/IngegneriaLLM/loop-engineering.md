# Loop engineering: progettare il ciclo

Per un paio d'anni la domanda, in ogni team che provava a costruire qualcosa
con un LLM, è stata sempre la stessa: qual è il prompt giusto? Si limava una
frase, si aggiungeva un esempio, si spostava una parola, come chi cerca la
combinazione di una cassaforte. Poi, tra chi con questi strumenti costruisce
davvero, la domanda ha cominciato a spostarsi. Boris Cherny, tra gli autori di
Claude Code, l'ha riassunta in una battuta ripresa ovunque: ormai non fa quasi
più prompt al modello, e costruisce invece loop che quei prompt li fanno per
lui. Non è una provocazione: è uno spostamento di leva. La cosa su cui vale la
pena lavorare non è più il singolo messaggio, ma il **sistema di controllo** che
attorno a quel messaggio decide quando parte, cosa gli si mette davanti, come si
verifica il risultato e cosa succede dopo.

Nelle due sezioni precedenti abbiamo lavorato sul **prompt** (il singolo
messaggio) e sul **contesto**: la finestra come sistema. Questa sezione sale
al terzo e più esterno dei cerchi concentrici da cui è partito il capitolo: il
**loop**. È l'anello in cui il prompt e il contesto smettono di essere una
cosa che scrivi *tu, adesso* e diventano una cosa che un programma monta,
esegue e rimette in moto, magari mentre dormi. Peter Steinberger e Addy
Osmani, che di queste automazioni scrivono da mesi, insistono su un punto che
è metà tecnico e metà etico: costruisci il loop come chi ha intenzione di
**restare l'ingegnere**, non come chi vuole solo premere «vai» e andarsene. La
differenza, come vedremo, è tutta lì.

## Il ciclo come unità di progetto

L'unità di lavoro del loop engineering non è la richiesta, ma il **ciclo**:
una sequenza che si ripete (*pianifica, esegui, verifica, rifletti*) e poi
ricomincia, portandosi dietro ciò che ha imparato. Non è un'idea nuova: è la
stessa spina dorsale del ciclo *osserva → ragiona → agisci* che abbiamo
incontrato nel capitolo sugli Agenti. La novità del loop engineering è
riconoscere che di cicli, in un sistema serio, ce ne sono **due, annidati**, e
che sono cose diverse.

`````{tab} Elementare

Immagina un artigiano al banco. Il suo ciclo di lavoro è stretto: guarda il
pezzo, decide la prossima mossa, la fa, guarda di nuovo; avanti così finché
l'oggetto è finito. Questo è il ciclo *interno*, quello dentro la sua testa e
le sue mani, e dura quanto dura un lavoro.

Ma sopra l'artigiano c'è il **capobottega**. Lui non intaglia: decide *quando*
si comincia (lunedì mattina, o ogni notte alle tre), tiene un **registro** di
cosa è stato fatto, **controlla** il pezzo finito prima di spedirlo, e se non va
lo rimanda indietro con un appunto. Il capobottega è il ciclo *esterno*. Il
loop engineering è il mestiere di progettare il capobottega: non le singole
intagliature, ma la cadenza, il registro, il controllo, la ripartenza. Un
artigiano bravissimo senza capobottega lavora finché lo guardi; con un buon
capobottega lavora anche di notte, e quello che consegna è già stato
controllato.

`````

`````{tab} Superiore

Il **loop interno** è il ciclo dell'agente visto negli Agenti: *osserva →
ragiona → agisci*, con lo stato che vive nella finestra di contesto ed è
effimero (finita la conversazione, svanisce). Lo diamo per acquisito e non lo
riespandiamo qui.

Il **loop esterno** è ciò che il loop engineering progetta, e ha proprietà che
il loop interno non ha:

- è **schedulato**, parte a una cadenza (un cron, un evento, un trigger), non
  solo quando un umano digita;
- ha **stato persistente**, non tiene la memoria nella finestra, ma *fuori*,
  su disco o in un database, così che sopravviva alla singola invocazione (il
  contrario dello *scratchpad* effimero visto nel context engineering);
- ha una **verifica deterministica**, un cancello che decide, con un criterio
  esterno e non con l'autovalutazione del modello, se il ciclo è riuscito;
- spesso a ogni giro **istanzia un agente fresco**, con contesto pulito,
  invece di accumulare cronologia all'infinito: riprendendo lo stato
  dall'esterno.

In termini di controllo, il loop interno è un *controllore reattivo* dentro un
singolo episodio; il loop esterno è l'*orchestratore* che decide quanti episodi
avviare, con quali condizioni iniziali, e come giudicarne l'esito.

`````

Il ciclo esterno, disegnato per esteso, ha quattro stazioni. La
{numref}`fig-loop-ciclo` le mostra chiuse in cerchio, con un dettaglio che è
il cuore di tutta la sezione: un **cancello** (spesso presidiato da un umano)
che può interrompere il giro invece di lasciarlo chiudere in automatico.

```{figure} ../figures/loop-engineering-ciclo.svg
:name: fig-loop-ciclo
:alt: "Diagramma di un ciclo a quattro nodi disposti in cerchio. In alto il nodo \"pianifica\", a destra \"esegui\", in basso \"verifica\", a sinistra \"rifletti\"; frecce curve li collegano in senso orario a chiudere il cerchio da \"rifletti\" di nuovo a \"pianifica\". Sul nodo \"verifica\" si innesta un cancello, etichettato \"human gate\", con due uscite: un ramo \"passa\" che prosegue verso \"rifletti\" e un ramo \"ferma\" che esce dal ciclo e lo interrompe."
:width: 90%

Il ciclo esterno del loop engineering: pianifica → esegui → verifica → rifletti,
chiuso ad anello. Sul passaggio di verifica si innesta un cancello (spesso un
controllo umano) che può lasciar proseguire il giro oppure fermarlo.
```

## I componenti di un loop

Un capobottega non è un'idea astratta: è fatto di attrezzi concreti. Il repo
*loop-engineering* di Cobus Greyling raccoglie il repertorio ormai ricorrente di
chi questi cicli li costruisce sul serio. Vale la pena elencarlo, perché ogni
voce risponde a un problema pratico che il ciclo esterno pone. Gli esempi
vengono quasi tutti dal mondo di chi programma, che di questi loop è il primo
cantiere: dove compare un attrezzo del mestiere, accanto c'è la sua traduzione
in parole comuni.

- **Automazione e scheduling.** Il loop ha una *cadenza*: una sveglia
  programmata (un *cron*) che lo avvia ogni notte, un campanello (un *webhook*)
  che lo sveglia ogni volta che qualcuno consegna una modifica, una coda che
  gli passa compiti. Senza cadenza non c'è ciclo, c'è solo un comando che
  qualcuno lancia a mano.
- **Worktree isolati.** Ogni giro lavora in una copia separata del progetto.
  Così più esecuzioni girano **in parallelo senza pestarsi i piedi**, e un giro
  che fa danni li fa in un recinto, non sulla copia buona. Il programma che
  custodisce il codice e la sua storia si chiama git; nel suo gergo, quella
  copia separata è un *worktree*.
- **Skill e prompt riusabili.** Le istruzioni non si riscrivono ogni volta: si
  impacchettano in *skill* versionate, richiamabili per nome. È il **prompt
  come codice** del context engineering, portato al livello del loop.
- **Split maker / checker.** Il lavoro si separa in due ruoli affidati a
  **sotto-agenti** distinti: un *implementatore* che produce, un *verificatore*
  che giudica. Ci torniamo tra poco: è il pezzo più importante.
- **Memoria e stato esterni.** Lo stato del loop non vive nella conversazione,
  ma in **file** che il ciclo legge e riscrive: un `STATE.md` con dove siamo
  arrivati, un `LOOP.md` con il piano e le decisioni. Sono la memoria a lungo
  termine dell'agente, discussa negli Agenti, qui in forma di file leggibili
  anche da un umano.
- **Integrazione con strumenti esterni.** Il loop non è un monologo: propone
  modifiche da far approvare (le *pull request*), commenta le segnalazioni
  aperte (i *ticket*), chiama servizi esterni tramite un protocollo apposito
  (l'**MCP**, *Model Context Protocol*), registra il lavoro nella storia del
  progetto con `git`. È così che il ciclo tocca il mondo invece di limitarsi a
  produrre testo.

Nessuno di questi attrezzi è «intelligente». Sono impalcatura, e come tutta la
buona impalcatura, è ciò che tiene in piedi la parte intelligente.

### Due agenti, non uno: il maker e il checker

Fra i componenti, la separazione tra chi fa e chi controlla merita una riga in
più, perché è controintuitiva: costa un secondo agente, eppure quasi sempre
ripaga.

`````{tab} Elementare

Pensa a uno scrittore e a un redattore. Lo scrittore butta giù il pezzo; il
redattore lo legge, segna cosa non va e lo rimanda indietro. Potresti chiedere
allo scrittore di rileggersi da solo, ma tutti sappiamo com'è: l'autore è il
peggior giudice del proprio testo, perché legge quello che *voleva* scrivere,
non quello che ha scritto. Tenere due ruoli separati serve proprio a questo:
il controllore arriva senza aver visto la fatica di chi ha prodotto, e giudica
il risultato per quello che è. Nel loop, il *maker* scrive, il *checker*
controlla, e sono due «persone» diverse: due agenti con teste separate.

`````

`````{tab} Superiore

Il pattern è due sotto-agenti con **contesti separati** e **prompt distinti**:
il *maker* riceve il compito e produce la modifica; il *checker* riceve solo
il risultato e i criteri, e restituisce un verdetto (passa / non passa) con le
motivazioni. La separazione dei contesti serve a due scopi. Primo, evita che
il maker «corregga il proprio compito»: un modello che valuta il testo che ha
appena generato è condizionato dalla propria traccia di ragionamento e tende a
ratificarla. Secondo, **decorrela i fallimenti**: se lo stesso agente, con lo
stesso contesto, sbaglia a produrre *e* a giudicare, i due errori sono
perfettamente correlati e il controllo è teatro. Un checker con contesto
pulito, e magari con criteri più severi, rompe questa correlazione. In cambio
si paga un secondo giro di inferenza (token e latenza in più) che va messo a
bilancio come ogni altra spesa del loop.

`````

## Il validation gate: verificare, non sperare

Arriviamo alla stazione che dà senso a tutte le altre: la **verifica**. Nel
ciclo della {numref}`fig-loop-ciclo` è il punto in cui si decide se il giro è
riuscito, e la scelta di progetto è netta: la verifica dev'essere un
**cancello**, non un augurio. Un cancello ha due stati, aperto o chiuso; non
esiste il «quasi passato». I test devono passare, il linter non deve
protestare, i tipi devono tornare: *prima* di considerare fatto il lavoro, non
dopo averlo già spedito.

```{figure} ../figures/codex-2021.svg
:name: fig-codice-verificato
:alt: "Una descrizione in linguaggio naturale entra nel modello, che genera del codice Python. Il codice non viene accettato così com'è: viene eseguito contro una batteria di test, e solo se li supera è considerato corretto; altrimenti si torna indietro a rigenerare."
:width: 96%

Il cancello, applicato al codice. Il modello propone; a decidere se la
proposta vale è l'esecuzione dei test, che è un giudizio esterno e non
opinabile.
```

La ragione per cui la programmazione è il terreno d'elezione di questi loop si
legge in {numref}`fig-codice-verificato`: esiste un oracolo automatico e
gratuito. Nei domini dove quell'oracolo manca (scrivere una relazione,
progettare un'interfaccia) il cancello va costruito a mano, ed è lì che il
loop engineering diventa difficile.

Questa idea ha una radice accademica precisa, in due lavori che il capitolo
sugli Agenti ha già introdotto e che qui rileggiamo dal lato del loop. ReAct
{cite}`yao2023react` ha mostrato che intrecciare **ragionamento e azione**
(pensare a parole *e* usare strumenti) rende più del solo agire. E siccome ogni
pensiero è legato a ciò che gli strumenti hanno davvero riportato, ReAct si
inventa meno cose del ragionamento lasciato a sé stesso, cioè della
chain-of-thought {cite}`wei2022chain`, che pensa a voce alta senza mai andare a
controllare. Questo non vuol dire che ReAct la batta sempre: in certe prove (le
domande che richiedono più ricerche in fila) la chain-of-thought resta avanti,
e il risultato migliore arriva dai due metodi usati insieme. Reflexion
{cite}`shinn2023reflexion` ha aggiunto il tassello mancante:
dopo un fallimento, l'agente **riflette a parole** sul proprio errore, scrive
quella riflessione in memoria e la usa per condizionare il tentativo
successivo. È esattamente la stazione «rifletti» del nostro ciclo.

`````{tab} Elementare

Il cancello è come un tornello alla metropolitana: o il biglietto è valido e
passi, o non lo è e resti fuori. Non c'è un tornello che ti fa passare «a
metà». Quando resti fuori, però, non è finita: leggi *perché* (biglietto
scaduto, importo sbagliato), rimedi e riprovi. Un buon loop fa così. Prova,
sbatte contro il cancello, **legge il motivo del rifiuto** (proprio come uno
studente che rilegge le correzioni in rosso prima di riscrivere il tema) e
riprova con quel motivo in mano. Ripete finché passa o finché ha esaurito i
tentativi che gli hai concesso.

`````

`````{tab} Superiore

Il ciclo pratico è **genera → verifica → raffina**. La verifica è un predicato
**deterministico ed esterno** (la suite di test, il type-checker, il linter)
che ritorna un booleano, non un giudizio del modello su sé stesso. La
riflessione (Reflexion) è invece *interna*: il modello propone una diagnosi in
linguaggio naturale dell'errore e la usa come contesto per il tentativo
seguente. La divisione dei ruoli è la chiave dell'affidabilità: **il modello
propone, il cancello deterministico dispone**. Ci si affida al giudizio del
modello per *migliorare*, mai per *dichiarare fatto*: quel verdetto lo dà un
criterio che il modello non può compiacere. Il loop termina alla prima
verifica positiva o all'esaurirsi di un budget di tentativi: un limite
esplicito, senza il quale un ciclo che non converge gira all'infinito
bruciando token.

`````

Ecco lo scheletro in puro Python, eseguibile. Il *generatore* è un finto LLM
(una lista di tentativi via via migliori), perché qui interessa il
**meccanismo del loop**, non il modello; il *verificatore*, invece, è reale
(controlla che uno slug rispetti tre regole) e il ciclo itera finché il
cancello passa o finiscono i tentativi:

```python
# Un loop generate -> verify -> refine. Il generatore e' un finto LLM;
# il verificatore e' reale: e' il "cancello" (validation gate) del ciclo.

def verifica(slug):
    """Il gate: ritorna (ok, motivo). Nessun 'quasi': o passa o no."""
    if slug != slug.lower():
        return False, "deve essere tutto minuscolo"
    if " " in slug:
        return False, "niente spazi: usa il trattino"
    if len(slug) > 20:
        return False, f"troppo lungo ({len(slug)} > 20 caratteri)"
    return True, "ok"


# Finto LLM: a ogni tentativo restituisce una candidata, migliorando quando
# riceve il feedback del verificatore. In un sistema vero qui c'e' il modello.
def genera(richiesta, feedback):
    tentativi = [
        "Guida Introduttiva a PyTorch",            # maiuscole + spazi
        "guida introduttiva a pytorch",            # ancora spazi
        "guida-introduttiva-a-pytorch-per-tutti",  # troppo lungo
        "guida-pytorch",                           # finalmente ok
    ]
    return tentativi[min(len(feedback), len(tentativi) - 1)]


def loop(richiesta, max_tentativi=5):
    feedback = []  # la memoria del loop: cresce a ogni riflessione
    for i in range(1, max_tentativi + 1):
        candidata = genera(richiesta, feedback)      # execute
        ok, motivo = verifica(candidata)             # verify (il gate)
        print(f"tentativo {i}: {candidata!r} -> {motivo}")
        if ok:
            return candidata
        feedback.append(motivo)                      # reflect: annota l'errore
    raise RuntimeError(f"gate non superato in {max_tentativi} tentativi")


risultato = loop("crea uno slug per una guida a PyTorch")
print("accettato:", risultato)
```

L'esecuzione mostra il ciclo che si autocorregge, raccogliendo a ogni giro il
motivo del rifiuto e ripartendo con quello in memoria:

```text
tentativo 1: 'Guida Introduttiva a PyTorch' -> deve essere tutto minuscolo
tentativo 2: 'guida introduttiva a pytorch' -> niente spazi: usa il trattino
tentativo 3: 'guida-introduttiva-a-pytorch-per-tutti' -> troppo lungo (38 > 20 caratteri)
tentativo 4: 'guida-pytorch' -> ok
accettato: guida-pytorch
```

Poche righe che non «capiscono» nulla, eppure incarnano la spina dorsale del
loop esterno: un cancello che non si compiace, una memoria del fallimento che
cresce, un tetto ai tentativi. In un sistema vero il generatore è il modello e
la `verifica` è la vera suite di test, ma l'ossatura è questa.

## Governance: gli errori si moltiplicano

Fin qui la parte esaltante. Ora quella onesta, perché un loop mal governato è
uno strumento per sbagliare più in fretta. Tre problemi vanno guardati in
faccia.

```{figure} ../figures/swe-bench-agenti-programmano.svg
:name: fig-swe-bench
:alt: "Catena di valutazione: da una issue reale di GitHub e dal codice del repository si parte, l'agente produce una patch, la patch viene applicata e sottoposta ai test del progetto, e il verdetto è binario, la issue è risolta oppure no."
:width: 100%

Un banco di prova che non ammette interpretazioni. Il compito viene da una
issue vera, il giudizio dai test che il progetto già aveva: nessuno dei due
l'ha scritto chi valuta.
```

Il pregio di {numref}`fig-swe-bench` è anche il suo limite, e vale la pena
dirlo qui fra i problemi. Un banco di prova così misura ciò che i test
sanno vedere: una patch che li supera peggiorando la leggibilità o
introducendo un debito passa lo stesso, e nessuna percentuale lo registra.

Il primo è **aritmetico**, e lo abbiamo già incontrato negli Agenti: gli errori
si accumulano lungo il ciclo. Se a ogni passo la probabilità di *non* introdurre
un errore non rilevato è $1 - p$, e se ogni passo sbaglia per conto proprio,
con lo stesso rischio $p$ tutte le volte, la probabilità che il loop attraversi
$n$ passi pulito è

$$
P(\text{pulito}) = (1 - p)^n,
$$

dove $p$ è la probabilità d'errore per passo e $n$ il numero di passi del
ciclo. Il prodotto crolla in fretta: con $p = 0{,}05$ e $n = 20$,
$P(\text{pulito}) \approx 0{,}36$; due giri su tre inciampano da qualche
parte. Il conto però regge solo finché ogni passo sbaglia per conto suo, e nei
loop veri non è così: il *context poisoning* visto nel context engineering fa
sì che uno sbaglio ne tiri dietro altri. Quando gli errori vengono a grappoli
il prodotto smette di essere la risposta esatta (i giri tendono a dividersi fra
del tutto puliti e rovinati in blocco), quindi prendilo per quello che è, un
conto all'ingrosso e non una regola sicura. Resta la ragione per cui il
**cancello di verifica** non è un lusso: alzando l'affidabilità effettiva di
ogni passo, tiene il prodotto lontano dallo zero.

`````{tab} Elementare

Nessuno dà a un nuovo assunto le chiavi dell'azienda il primo giorno. La prima
settimana scrive solo **relazioni** che tu leggi: osserva e riferisce, decidi
tu. Guadagnata un po' di fiducia, può **proporre correzioni** che tu approvi
prima che partano. Solo dopo, e solo per cose di cui ti fidi, lavora **da solo,
anche di notte**. Con i loop è identico: si concede autonomia a scaglioni, non
tutta subito. Chi consegna le chiavi al primo giorno non sta risparmiando
tempo: sta preparando il disastro che dovrà poi ripulire.

`````

`````{tab} Superiore

Il rollout maturo procede per livelli, allargando il **raggio d'azione** solo
quando le metriche lo giustificano:

- **L1, solo report.** Il loop osserva e *propone*: apre una segnalazione,
  scrive una diagnosi. L'umano applica. Raggio d'azione nullo sul sistema.
- **L2, fix assistiti.** Il loop produce la modifica (una pull request, una
  patch) ma non la integra: c'è un **cancello umano** che rivede e fonde. È il
  regime di gran lunga più comune in produzione.
- **L3, non presidiato.** Il loop integra da solo, ma **dentro i confini** di
  una *allow-list* (quali file, quali comandi, quali repository) e sotto
  monitoraggio continuo. Vi si sale solo dopo che L2 ha dato numeri buoni.

A ogni livello si accompagnano le difese di sicurezza già viste per gli agenti:
**allow-list e deny-list** dei comandi e delle risorse, worktree isolati come
recinto, e cancelli umani ai punti irreversibili.

`````

Il secondo problema è più sottile e non si risolve con un test. Addy Osmani lo
chiama **comprehension debt**, il debito di comprensione: un loop produce più
codice, più modifiche, più decisioni di quante un umano ne legga: e il conto,
prima o poi, arriva. I loop **amplificano il giudizio**: quello buono e quello
cattivo con la stessa efficienza. Un'architettura pulita si propaga in fretta;
un errore di impostazione anche. Per questo la regola del «restare
l'ingegnere» non è retorica: chi mantiene il sistema deve **leggere ciò che
parte**, non solo guardare la spia verde dei test. Un loop che nessuno capisce
più è un passivo, per quanto verdi siano i suoi cancelli.

Il terzo problema è **economico**, e ci riporta al capitolo su LLMOps. Ogni
giro del loop consuma token, apre chiamate, occupa macchine: un ciclo
schedulato che gira ogni notte ha una **bolletta** e va messo a budget come
qualsiasi processo. E siccome gira quando non lo guardi, va **monitorato** con
la stessa cura: metriche di riuscita, costo per giro, tasso di intervento
umano, allarmi quando qualcosa degenera. Il loop engineering, in fondo, sposta
la leva dal prompt al sistema; ma un sistema, a differenza di una frase, va
sorvegliato mentre lavora.

```{admonition} Da ricordare
:class: important
- Il loop engineering sposta la leva **dal singolo prompt al sistema di
  controllo**: non si cerca più la frase perfetta, si progetta il ciclo che di
  frasi ne fa tante. È il terzo cerchio, il più esterno, dopo prompt e contesto.
- Ci sono **due cicli annidati**: il *loop interno* dell'agente (osserva →
  ragiona → agisci, già visto negli Agenti) e il *loop esterno* che il loop
  engineering progetta (schedulato, con stato persistente **fuori** dalla
  finestra e verifica esterna).
- Il ciclo esterno ha quattro stazioni (**pianifica → esegui → verifica →
  rifletti**) e un **cancello** (spesso umano) che può fermarlo. I suoi
  componenti: scheduling, worktree isolati, skill riusabili, split
  **maker/checker**, stato su file, integrazione (MCP/git/ticket).
- La **verifica è un cancello, non un augurio**: un predicato deterministico
  (test, lint, tipi) che il modello non può compiacere. Il modello *propone*
  (riflessione alla Reflexion {cite}`shinn2023reflexion`, azione+ragionamento
  alla ReAct {cite}`yao2023react`), il cancello *dispone*.
- Gli errori si **moltiplicano** lungo il ciclo, $(1-p)^n$ decade in fretta, e
  per questo il gate è essenziale. Rollout a fasi: **L1** solo report → **L2**
  fix assistiti con cancello umano → **L3** non presidiato entro allow-list.
- Onestà sui limiti: il **comprehension debt** (i loop amplificano il giudizio
  buono *e* cattivo; chi mantiene deve leggere ciò che parte), la sicurezza
  (allow/deny-list, cancelli umani) e il **costo** per giro, da mettere a budget
  e monitorare come insegna LLMOps.
```
