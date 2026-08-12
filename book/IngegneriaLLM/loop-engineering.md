# Loop engineering: progettare il ciclo

Nei primi anni la domanda, in ogni team che provava a costruire qualcosa
con un LLM, è stata sempre la stessa: qual è il prompt giusto? Si limava una
frase, si aggiungeva un esempio, si spostava una parola, come chi cerca la
combinazione di una cassaforte. Poi, tra chi con questi strumenti costruisce
davvero, la domanda ha cominciato a spostarsi. Nel 2026 Boris Cherny, tra gli
autori di Claude Code (uno degli assistenti di programmazione che si usano da
riga di comando), l'ha riassunta in una battuta: non fa quasi
più prompt al modello, e costruisce invece loop che quei prompt li fanno per
lui {cite}`cherny2026loops`. Non è una provocazione: è uno spostamento di
leva. La cosa su cui vale la
pena lavorare non è più il singolo messaggio, ma il **sistema di controllo** che
attorno a quel messaggio decide quando parte, cosa gli si mette davanti, come si
verifica il risultato e cosa succede dopo.

Conviene dire subito da dove viene il materiale di questa sezione, perché è
diverso da quello delle due precedenti. Il vocabolario del loop engineering è
recente e lo hanno scritto quasi tutto dei praticanti, cioè persone che questi
cicli li costruiscono e ne raccontano il funzionamento, non gruppi di ricerca
che li misurano. Quello che riportiamo qui è il **meccanismo** (la cadenza, lo
stato tenuto fuori dalla finestra, il cancello di verifica, l'isolamento),
perché è la parte che ha buone probabilità di sopravvivere ai nomi con cui
oggi la si chiama; dove un'affermazione è di mestiere e non misurata, lo
diciamo.

Nelle due sezioni precedenti abbiamo lavorato sul **prompt** (il singolo
messaggio) e sul **contesto**: la finestra come sistema. Questa sezione sale
al terzo e più esterno dei cerchi concentrici da cui è partito il capitolo: il
**loop**. È l'anello in cui il prompt e il contesto smettono di essere una
cosa che scrivi *tu, adesso* e diventano una cosa che un programma monta,
esegue e rimette in moto, magari mentre dormi. Peter Steinberger lo dice in
una riga («non dovresti più fare prompt agli agenti che programmano: dovresti
progettare i cicli che fanno i prompt al posto tuo»)
{cite}`steinberger2026loops`, e Addy Osmani ne trae il punto che è metà
tecnico e metà etico {cite}`osmani2026loop`: costruisci il loop come chi ha
intenzione di **restare l'ingegnere**, non come chi vuole solo premere «vai» e
andarsene. La differenza, come vedremo, è tutta lì.

## Il ciclo come unità di progetto

Prima, però, una parola che cambia peso, e conviene dirlo invece di lasciarlo
capire. Aprendo il capitolo il «loop» era la conversazione: tu chiedi, guardi
la risposta, storci il naso, richiedi meglio. Quel loop esiste ancora ed è il
più comune di tutti; solo che lo giri **tu**, a mano, e finisce quando chiudi
la finestra. Qui il loop diventa un'altra cosa: lo stesso giro affidato a un
programma, che lo fa partire da sé, lo ripete e ne conserva l'esito. Non è una
sostituzione, è un annidamento: dentro c'è ancora il ciclo di prima. Quello che
cambia è chi lo mette in moto e chi decide quando è finito, e quel «chi», da
qui in avanti, non è più una persona davanti a una tastiera.

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
il cuore di tutta la sezione: alla stazione di verifica c'è un **cancello**,
cioè un controllo che decide se il giro può chiudersi o va rifatto, e a
deciderlo non è il modello. Più avanti vedremo che nei sistemi che toccano
cose difficili da disfare se ne mette un secondo, presidiato da una persona;
qui basta il primo.

```{figure} ../figures/loop-engineering-ciclo.svg
:name: fig-loop-ciclo
:alt: "Diagramma di un ciclo a quattro stazioni disposte in cerchio, percorse in senso orario: pianifica, esegui, verifica, rifletti, e da qui di nuovo a pianifica. La stazione di verifica è il cancello: da lì parte una freccia verso l'esterno, la consegna, che si imbocca solo se i test passano, e una freccia di ritorno verso la riflessione se invece falliscono."
:width: 90%

Il ciclo esterno del loop engineering: pianifica → esegui → verifica → rifletti,
chiuso ad anello. Alla verifica c'è il cancello: chi lo supera esce e viene
consegnato, chi non lo supera torna indietro con il motivo del rifiuto.
```

## I componenti di un loop

Un capobottega non è un'idea astratta: è fatto di attrezzi concreti. Il repo
*loop-engineering* di Cobus Greyling {cite}`greyling2026loop` raccoglie il
repertorio di
chi questi cicli li costruisce sul serio. Vale la pena elencarlo, perché ogni
voce risponde a un problema pratico che il ciclo esterno pone. Gli esempi
vengono quasi tutti dal mondo di chi programma, che di questi loop è il primo
cantiere: ogni voce nomina prima la **funzione**, che resta, e poi l'attrezzo
di oggi, che è solo il modo in cui quella funzione si realizza adesso; dove
compare un attrezzo del mestiere, accanto c'è la sua traduzione in parole
comuni.

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
- **Istruzioni riusabili.** Le istruzioni non si riscrivono ogni volta: si
  impacchettano in unità richiamabili per nome, tenute sotto controllo di
  versione come il codice, così che una correzione fatta una volta valga per
  tutti i giri successivi (nei ferri del 2026 si chiamano *skill*). È il
  **prompt come codice** del context engineering, portato al livello del loop.
- **Chi fa e chi controlla, separati.** Il lavoro si divide in due ruoli
  affidati a due **agenti distinti**, cioè a due copie del modello con
  istruzioni e contesto propri: un *implementatore* che produce, un
  *verificatore* che giudica. Ci torniamo tra poco: è il pezzo più importante.
- **Memoria e stato esterni.** Lo stato del loop non vive nella conversazione,
  ma in **file** che il ciclo legge e riscrive: uno con il punto a cui si è
  arrivati, uno con il piano e le decisioni prese (nei repo che li usano si
  chiamano `STATE.md` e `LOOP.md`, e sono file di testo semplice). Sono la
  memoria a lungo termine dell'agente, discussa negli Agenti, qui in forma
  leggibile anche da un umano, che è il punto: chi arriva la mattina dopo
  capisce che cosa è successo di notte senza rileggere una conversazione.
- **Integrazione con strumenti esterni.** Il loop non è un monologo: propone
  modifiche da far approvare (le *pull request*), commenta le segnalazioni
  aperte (i *ticket*), chiama servizi esterni tramite un protocollo condiviso
  che descrive quali operazioni un modello può richiedere e come (dal 2024
  quello diffuso è l'**MCP**, *Model Context Protocol*), registra il lavoro
  nella storia del progetto con `git`. È così che il ciclo tocca il mondo
  invece di limitarsi a produrre testo.

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
ratificarla, ed è un effetto misurato, non un timore
{cite}`panickssery2024selfpreference`. Secondo, **attenua la correlazione dei
fallimenti**: se lo stesso agente, con lo
stesso contesto, sbaglia a produrre *e* a giudicare, i due errori sono
perfettamente correlati e il controllo è teatro. Un checker con contesto
pulito, e magari con criteri più severi, quella correlazione la abbassa; non
la annulla. Se maker e checker sono lo stesso modello cambia il
condizionamento, non i punti ciechi, e un errore che nasce da una lacuna del
modello lo vedono tutt'e due allo stesso modo. Quanto ne resta non lo sappiamo:
nessuno ha misurato la correlazione residua fra due istanze dello stesso
modello con contesti separati, ed è una delle ragioni per cui, sopra il
checker, il cancello resta deterministico. In cambio
si paga un secondo giro di inferenza (token e latenza in più) che va messo a
bilancio come ogni altra spesa del loop.

`````

## Il validation gate: verificare, non sperare

Arriviamo alla stazione che dà senso a tutte le altre: la **verifica**. Nel
ciclo della {numref}`fig-loop-ciclo` è il punto in cui si decide se il giro è
riuscito, e la scelta di progetto è netta: la verifica dev'essere un
**cancello**, non un augurio. Un cancello ha due stati, aperto o chiuso; non
esiste il «quasi passato». Nel caso del codice i controlli sono tre, e sono
tutti automatici: i **test** devono passare (piccoli programmi scritti apposta
per verificare che il codice faccia quel che promette), il **linter** non deve
protestare (un programma che rilegge il codice e segnala le sciatterie: una
variabile mai usata, una riga scritta in un modo che confonde), e i **tipi**
devono tornare (ogni valore deve essere della specie che il codice si aspetta:
un numero dove serve un numero, un testo dove serve un testo). Tutto questo
*prima* di considerare fatto il lavoro, non dopo averlo già spedito.

```{figure} ../figures/codex-2021.svg
:name: fig-codice-verificato
:alt: "Una descrizione in linguaggio naturale entra nel modello, che genera del codice Python. Il codice non viene accettato così com'è: viene eseguito contro una batteria di test, e solo se li supera è considerato corretto; altrimenti si torna indietro a rigenerare."
:width: 96%

Il cancello, applicato al codice. Il modello propone; a decidere se la
proposta vale è l'esecuzione dei test, che è un giudizio esterno e non
opinabile.
```

La ragione per cui la programmazione è il terreno d'elezione di questi loop si
legge in {numref}`fig-codice-verificato`: esiste un **oracolo** automatico e
gratuito. «Oracolo» qui non ha niente a che vedere con il futuro: in
informatica è il nome di qualcosa che sa dire, senza discutere, se un
risultato è giusto o sbagliato. Per il codice quell'oracolo sono i test, e li
si esegue in un secondo, quante volte si vuole. Nei domini dove manca
(scrivere una relazione, progettare un'interfaccia) il cancello va costruito a
mano, ed è lì che il loop engineering diventa difficile.

È anche il banco su cui la cosa fu misurata per la prima volta. Nel 2021, per
valutare Codex (un modello addestrato sul codice, antenato degli assistenti di
programmazione di oggi), OpenAI pubblicò **HumanEval**: 164 problemi scritti a
mano, ciascuno con una manciata di test, dove una soluzione conta solo se li
supera **tutti** {cite}`chen2021evaluating`. È la forma pura del cancello:
nessun giudizio, nessuna sfumatura, un programma che gira o non gira.

Questa idea ha una radice accademica precisa, in due lavori che il capitolo
sugli Agenti ha già introdotto e che qui rileggiamo dal lato del loop. ReAct
{cite}`yao2023react` ha mostrato che intrecciare **ragionamento e azione**
(pensare a parole *e* usare strumenti) rende più del solo agire. E siccome ogni
pensiero è legato a ciò che gli strumenti hanno davvero riportato, ReAct si
inventa meno cose del ragionamento lasciato a sé stesso, cioè della
chain-of-thought {cite}`wei2022chain`, che pensa a voce alta senza mai andare a
controllare. Questo non vuol dire che ReAct la batta sempre. I due metodi sono
stati messi a confronto su raccolte di domande costruite apposta, in cui per
rispondere bisogna incrociare più informazioni cercandole una dopo l'altra; e
lì la chain-of-thought resta avanti,
mentre il risultato migliore arriva dai due metodi usati insieme. Reflexion
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

Ecco lo scheletro in puro Python, eseguibile. Il *generatore* è un finto LLM,
che al posto di ragionare guarda l'ultimo motivo di rifiuto e corregge quello:
è una caricatura di ciò che fa un modello vero, ma commette lo stesso gesto,
che è il gesto della sezione (il **contenuto** del fallimento guida il
tentativo dopo). Il *verificatore*, invece, è reale (controlla che uno slug
rispetti tre regole) e il ciclo itera finché il cancello passa o finiscono i
tentativi:

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


# Finto LLM: legge l'ULTIMO motivo di rifiuto e corregge quello, come farebbe
# un modello a cui si passa il feedback. In un sistema vero qui c'e' il modello.
def genera(richiesta, feedback):
    if not feedback:                                  # primo tentativo, a freddo
        return "Guida Introduttiva a PyTorch"
    ultimo = feedback[-1]
    if "minuscolo" in ultimo:
        return "guida introduttiva a pytorch"
    if "spazi" in ultimo:
        return "guida-introduttiva-a-pytorch-per-tutti"
    if "lungo" in ultimo:
        return "guida-pytorch"
    return "guida-pytorch"


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

## Tenere il ciclo in mano: gli errori si moltiplicano

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

Il banco di prova disegnato in {numref}`fig-swe-bench` è **SWE-bench**
{cite}`jimenez2024swebench`, che nel capitolo sugli Agenti abbiamo già usato e
discusso. Il suo pregio è anche il suo limite, e vale la pena ricordarlo qui
fra i problemi. Un banco di prova così misura ciò che i test
sanno vedere: una patch che li supera peggiorando la leggibilità o
introducendo un debito passa lo stesso, e nessuna percentuale lo registra. E
c'è un'obiezione più dura, che negli Agenti abbiamo riportata per esteso:
rileggendo a mano i successi, una quota consistente si rivela già scritta
dentro la segnalazione da risolvere {cite}`aleithan2024swebenchplus`. Un
oracolo automatico è una gran cosa, ma resta un oracolo su ciò che qualcuno ha
deciso di misurare.

Il primo problema è **aritmetico**, e lo abbiamo già incontrato negli Agenti:
gli errori si accumulano lungo il ciclo. Chiamiamo $p$ la probabilità che un
singolo passo introduca un errore che nessuno intercetta; allora $1 - p$ è la
probabilità che quel passo vada liscio, e se ogni passo sbaglia per conto
proprio, con lo stesso rischio $p$ tutte le volte, la probabilità che il loop
attraversi $n$ passi senza guai è

$$
P(\text{pulito}) = (1 - p)^n,
$$

dove $n$ è il numero di passi del ciclo: moltiplicare venti volte un numero
appena sotto l'uno porta molto più in basso di quanto sembri. Con
$p = 0{,}05$ e $n = 20$ si ottiene
$P(\text{pulito}) \approx 0{,}36$; cioè due giri su tre inciampano da qualche
parte. Il conto però regge solo finché ogni passo sbaglia per conto suo, e nei
loop veri non è così: il *context poisoning* visto nel context engineering fa
sì che uno sbaglio ne tiri dietro altri. Quando gli errori vengono a grappoli
quella moltiplicazione smette di essere la risposta esatta (i giri tendono a
dividersi fra del tutto puliti e rovinati in blocco), quindi prendila per
quello che è, un conto all'ingrosso e non una regola sicura. Resta la ragione
per cui il
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
  livello a cui conviene fermarsi finché il costo di una revisione umana resta
  minore del costo atteso di un errore integrato senza guardarlo.
- **L3, non presidiato.** Il loop integra da solo, ma **dentro i confini** di
  una *allow-list* (quali file, quali comandi, quali repository) e sotto
  monitoraggio continuo. Vi si sale solo dopo che L2 ha dato numeri buoni.

A ogni livello si accompagnano le difese di sicurezza già viste per gli agenti:
**allow-list e deny-list** dei comandi e delle risorse, worktree isolati come
recinto, e cancelli umani ai punti irreversibili.

`````

Il secondo problema è più sottile e non si risolve con un test. Addy Osmani lo
chiama **comprehension debt** {cite}`osmani2026comprehension`, il debito di
comprensione (l'espressione, va detto, circolava già prima di lui): un loop produce più
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

`````{tab} Elementare

```{admonition} Da ricordare
:class: important
- Il terzo cerchio sposta l'attenzione **dalla frase al processo**: non si
  cerca più il messaggio perfetto, si progetta il giro che di messaggi ne fa
  tanti. Il giro che fai tu, chiedendo e richiedendo, resta dentro: quello che
  si aggiunge è un capobottega che lo mette in moto, lo controlla e lo rimanda
  indietro quando non va.
- Il capobottega ha quattro stazioni (**pianifica, esegui, verifica,
  rifletti**) e un **cancello** alla verifica. Cancello vuol dire due stati e
  basta, come un tornello: non esiste il «quasi passato».
- Il cancello non lo tiene il modello. **Il modello propone, il cancello
  dispone**, e il cancello è un controllo automatico (per il codice: dei
  programmi di prova che girano da soli). Chiedere al modello se il proprio
  lavoro va bene è teatro: tende a dirsi di sì.
- Quando il cancello respinge, quello che serve è **il motivo**: si riparte da
  lì, non da capo. E si mette sempre un tetto ai tentativi, altrimenti un giro
  che non converge gira per sempre.
- **Gli errori si moltiplicano.** Un giro lungo con un rischio piccolo a ogni
  passo finisce male più spesso di quanto l'intuito dica: venti passi con il
  cinque per cento di rischio ciascuno arrivano puliti in fondo solo una volta
  su tre. È la ragione per cui il cancello non è un lusso.
- **L'autonomia si concede a scaglioni**, come a un nuovo assunto: prima solo
  relazioni da leggere, poi proposte da approvare, e solo alla fine, e solo
  dentro confini scritti, il permesso di fare da sé.
- Tre cose da tenere d'occhio: un ciclo produce più roba di quanta se ne
  riesca a leggere (e il conto arriva), amplifica il giudizio buono **e**
  quello cattivo, e costa: gira mentre non lo guardi, e la bolletta arriva
  lo stesso.
```

`````

`````{tab} Superiore

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
- Onestà sui limiti: il **comprehension debt** {cite}`osmani2026comprehension`
  (i loop amplificano il giudizio
  buono *e* cattivo; chi mantiene deve leggere ciò che parte), la sicurezza
  (allow/deny-list, cancelli umani) e il **costo** per giro, da mettere a budget
  e monitorare come insegna LLMOps. E lo statuto delle fonti: il vocabolario
  del loop engineering viene da chi costruisce, non da chi misura; qui si
  riporta il meccanismo, che dura, non i nomi degli attrezzi, che cambiano.
```

`````
