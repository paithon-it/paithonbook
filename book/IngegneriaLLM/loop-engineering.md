# Loop engineering: progettare il ciclo

Nei primi anni di questa tecnologia la domanda, in ogni squadra che provava a
costruire qualcosa con un LLM, è stata sempre la stessa: qual è il prompt
giusto? Si limava una
frase, si aggiungeva un esempio, si spostava una parola, come chi cerca la
combinazione di una cassaforte. Poi, tra chi con questi strumenti costruisce
davvero, la domanda ha cominciato a spostarsi. Nel 2026 Boris Cherny, tra gli
autori di Claude Code (un assistente di programmazione che non ha finestre né
bottoni: gli si scrive e risponde, come si faceva con i computer prima che
arrivassero i mouse), l'ha riassunta in una battuta: non fa quasi più prompt
al modello, scrive **loop** (giri, cicli) che quei prompt li fanno per lui
{cite}`cherny2026loops`. Non è una provocazione: è uno spostamento di leva. La
cosa su cui vale la pena lavorare non è più il singolo messaggio, ma il
**sistema di controllo** che attorno a quel messaggio decide quando parte,
cosa gli si mette davanti, come si verifica il risultato e cosa succede dopo.

Questa sezione sale così al terzo e più esterno dei cerchi da cui è partito il
capitolo: dopo il **prompt** (il singolo messaggio) e il **contesto** (la
finestra come sistema), il **loop**. È l'anello in cui il prompt e il contesto
smettono di essere una cosa che scrivi *tu, adesso* e diventano una cosa che
un programma monta, esegue e rimette in moto, magari mentre dormi. Peter
Steinberger lo dice quasi con le stesse parole di Cherny: non si fanno più
prompt agli agenti che programmano, si progettano i cicli che quei prompt li
fanno da soli {cite}`steinberger2026loops`. E Addy Osmani ne trae una
conseguenza che è metà tecnica e metà morale {cite}`osmani2026loop`: il ciclo
va costruito da
chi ha intenzione di **restare l'ingegnere**, cioè di continuare a capire e a
rispondere di quello che esce, non da chi vuole premere «vai» e andarsene.

Una nota, perché qui le fonti sono diverse da quelle delle due sezioni
precedenti: il vocabolario del loop engineering lo hanno scritto quasi tutto
dei praticanti, gente che questi cicli li costruisce, non gruppi di ricerca
che li misurano. Di quel racconto teniamo il **meccanismo**, che ha buone
probabilità di durare più dei nomi; e dove un'affermazione è di mestiere e non
misurata, lo diciamo.

## Il ciclo come unità di progetto

Prima, però, va detto che una parola qui cambia significato, e conviene dirlo
invece di lasciarlo capire. Aprendo il capitolo il «loop» era la
conversazione: tu chiedi, guardi
la risposta, storci il naso, richiedi meglio. Quel loop esiste ancora ed è il
più comune di tutti; solo che lo giri **tu**, a mano, e finisce quando chiudi
la finestra. Qui il loop diventa un'altra cosa: lo stesso giro affidato a un
programma, che lo fa partire da sé, lo ripete e ne conserva l'esito. Non è una
sostituzione, è un annidamento: dentro c'è ancora il ciclo di prima. Quello che
cambia è chi lo mette in moto e chi decide quando è finito, e quel «chi», da
qui in avanti, non è più una persona davanti a una tastiera.

L'unità di lavoro del loop engineering non è la richiesta, ma il **ciclo**:
una sequenza che si ripete (*pianifica, esegui, verifica, rifletti*) e poi
ricomincia, portandosi dietro ciò che ha imparato. Non è un'idea nuova, è la
stessa ossatura del giro *osserva, ragiona, agisci* con cui lavora un agente.
La novità del loop engineering è riconoscere che di cicli, in un sistema
serio, ce ne sono **due, annidati**, e che sono cose diverse.

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

Nella bottega, l'artigiano e il capobottega sono tutti e due dei programmi. Tu
sei il proprietario: non stai al banco e non fai i turni, ma decidi quanta
corda dare al capobottega, e la bottega resta tua, compreso quello che ne
esce.

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
deciderlo non è il modello. Più avanti vedremo che dove il lavoro tocca cose
difficili da disfare (mandare una mail a un cliente, cancellare dei dati,
pubblicare qualcosa) di cancelli se ne mette un secondo, e a tenerlo è una
persona; qui basta il primo.

```{figure} ../figures/loop-engineering-ciclo.svg
:name: fig-loop-ciclo
:alt: "Diagramma di un ciclo a quattro stazioni disposte in cerchio, percorse in senso orario: pianifica, esegui, verifica, rifletti, e da qui di nuovo a pianifica. La stazione di verifica è il cancello: da lì parte una freccia verso l'esterno, la consegna, che si imbocca solo se i test passano, e una freccia di ritorno verso la riflessione se invece falliscono."
:width: 90%

Il ciclo esterno del loop engineering: pianifica → esegui → verifica → rifletti,
chiuso ad anello. Alla verifica c'è il cancello: chi lo supera esce e viene
consegnato, chi non lo supera torna indietro con il motivo del rifiuto.
```

## I componenti di un loop

Un capobottega non è un'idea astratta: è fatto di attrezzi concreti. Cobus
Greyling {cite}`greyling2026loop` ne ha raccolto il repertorio, e vale la pena
scorrerlo, perché ogni voce risponde a un problema pratico che il ciclo
esterno pone. Gli esempi vengono quasi tutti dal mondo di chi programma, che
di questi loop è il primo cantiere; e di ogni voce diciamo prima a che cosa
serve, che è la parte che dura, poi come si chiama l'attrezzo che oggi la fa.

- **La sveglia.** Il loop parte da sé, a una cadenza sua: ogni notte a un'ora
  fissata (l'orologio che lo fa partire, nei sistemi che ospitano i programmi,
  si chiama *cron*), oppure ogni volta che succede qualcosa, per esempio
  quando un programmatore aggiunge del codice al progetto (l'avviso che
  arriva in quel momento si chiama *webhook*). Senza una sveglia non c'è
  ciclo, c'è solo un comando che qualcuno lancia a mano.
- **Il recinto.** Ogni giro lavora su una copia separata del progetto, non su
  quella buona. Così più giri possono andare insieme senza pestarsi i piedi, e
  quello che combina danni li combina nella sua copia. Il programma che
  custodisce il codice e la sua storia si chiama git; nel suo gergo una di
  quelle copie separate è un *worktree*.
- **Le istruzioni riusabili.** Le istruzioni non si riscrivono ogni volta: si
  impacchettano una volta sola, si dà loro un nome, e si tengono in archivio
  accanto al codice, con la loro storia delle modifiche. Così una correzione
  fatta oggi vale per tutti i giri di domani (negli attrezzi del 2026 questi
  pacchetti si chiamano *skill*). È l'idea, già incontrata nel capitolo sugli
  Agenti, che un prompt vada trattato come si tratta il codice: archiviato,
  corretto in un posto solo, con la storia delle sue versioni.
- **Chi fa e chi controlla, separati.** Il lavoro si divide in due ruoli
  affidati a due **agenti distinti**, cioè a due copie del modello ciascuna
  con le sue istruzioni e la sua finestra: una produce, l'altra giudica. Ci
  torniamo fra poco: è il pezzo più importante.
- **La memoria fuori dalla finestra.** Quello che il ciclo sa non vive nella
  conversazione, ma in **file** che legge e riscrive a ogni giro: uno con il
  punto a cui si è arrivati, uno con il piano e le decisioni prese. Sono la
  memoria a lungo termine di cui parlava il capitolo sugli Agenti, qui in una
  forma che legge anche una persona, e questo è il punto: chi arriva la
  mattina dopo capisce che cosa è successo di notte senza doversi rileggere
  una conversazione.
- **Le mani sul mondo.** Il loop non parla soltanto: propone modifiche da far
  approvare, risponde nelle segnalazioni aperte (quelle che gli utenti scrivono
  quando qualcosa non funziona), registra il proprio lavoro nella storia del
  progetto. E può rivolgersi a programmi esterni, purché
  qualcuno gli abbia detto quali operazioni esistono e come si chiedono: dal
  2024 c'è un modo aperto di dirglielo, adottato da più fornitori, che si
  chiama **MCP** (*Model Context Protocol*). È così che il ciclo tocca il
  mondo invece di limitarsi a produrre testo.

Nessuno di questi attrezzi è «intelligente». Sono impalcatura, e come ogni
buona impalcatura sono quello che tiene in piedi la parte intelligente.

### Due agenti, non uno: chi fa e chi controlla

Fra i componenti, la separazione fra chi fa e chi controlla merita qualche
riga in più, perché a prima vista sembra uno spreco: vuol dire far lavorare il
modello due volte invece che una, e quindi pagare due volte, in tempo e in
denaro. Eppure quasi sempre ripaga. (I due ruoli, in inglese, si chiamano
*maker* e *checker*, e così si trova scritto lo schema.)

`````{tab} Elementare

Pensa a uno scrittore e a un redattore. Lo scrittore butta giù il pezzo; il
redattore lo legge, segna cosa non va e lo rimanda indietro. Potresti chiedere
allo scrittore di rileggersi da solo, ma tutti sappiamo com'è: l'autore è il
peggior giudice del proprio testo, perché legge quello che *voleva* scrivere,
non quello che ha scritto. Tenere due ruoli separati serve proprio a questo:
il controllore arriva senza aver visto la fatica di chi ha prodotto, e giudica
il risultato per quello che è. Nel loop, il *maker* scrive, il *checker*
controlla, e sono due «persone» diverse: due agenti con teste separate.

Qui è lecito obiettare: se sono due copie dello stesso modello, che senso ha?
Uno pensa come l'altro. La risposta è che la differenza non sta nella testa,
sta in quello che ciascuno ha davanti. Il primo ha davanti il compito e tutta
la strada che ha fatto per svolgerlo; il secondo ha davanti solo il risultato
e i criteri con cui giudicarlo, e non sa nemmeno di chi sia. Non è un
dettaglio: è stato misurato che un modello, messo a giudicare, tende a
preferire il testo che ha scritto lui, e lo preferisce di più quanto meglio lo
riconosce come proprio. Toglierglielo di mezzo cambia il verdetto. Restano
naturalmente i punti ciechi comuni: quello che il modello non sa vedere non lo
vede nemmeno da controllore, ed è per questo che sopra il controllore c'è
sempre un cancello che non è un modello.

`````

`````{tab} Superiore

Il pattern è due sotto-agenti con **contesti separati** e **prompt distinti**:
il *maker* riceve il compito e produce la modifica; il *checker* riceve solo
il risultato e i criteri, e restituisce un verdetto (passa / non passa) con le
motivazioni. La separazione dei contesti serve a due scopi. Primo, evita che
il maker «corregga il proprio compito»: messo a giudicare, un modello preferisce
il testo che ha prodotto lui a uno equivalente prodotto da altri, e lo fa tanto
più quanto meglio riconosce come proprio il testo che sta leggendo. È un
effetto misurato, non un timore {cite}`panickssery2024selfpreference`, e la
conseguenza per il loop è immediata: chi ha scritto è il peggior candidato a
dire se quel che ha scritto va bene. Secondo, **attenua la correlazione dei
fallimenti**: se lo stesso agente, con lo
stesso contesto, sbaglia a produrre *e* a giudicare, i due errori sono
perfettamente correlati e il controllo è teatro. Un checker con contesto
pulito, e magari con criteri più severi, quella correlazione la abbassa; non
la annulla. Se maker e checker sono lo stesso modello cambia il
condizionamento, non i punti ciechi, e un errore che nasce da una lacuna del
modello lo vedono tutt'e due allo stesso modo. Quanto ne resta non lo
sappiamo, e una misura diretta di quella correlazione residua, fra due istanze
dello stesso modello con contesti separati, non ci risulta: è una delle
ragioni per cui, sopra il checker, il cancello resta deterministico. In cambio
si paga un secondo giro di inferenza (token e latenza in più) che va messo a
bilancio come ogni altra spesa del loop.

`````

## Il cancello di verifica: verificare, non sperare

Arriviamo alla stazione che dà senso a tutte le altre: la **verifica**. Nel
ciclo della {numref}`fig-loop-ciclo` è il punto in cui si decide se il giro è
riuscito, e la scelta di progetto è netta: la verifica dev'essere un
**cancello**, non un augurio. Un cancello ha due stati, aperto o chiuso; non
esiste il «quasi passato». (In inglese si chiama *validation gate*, ed è la
stessa cosa: il cancello che convalida.)

Nel caso del codice i controlli sono tre, tutti automatici. I **test** devono
passare: sono piccoli programmi scritti apposta per verificare che il codice
faccia quel che promette. Il **linter** non deve protestare: è un programma
che rilegge il codice e segnala le sciatterie, un valore calcolato e poi mai
usato, una riga scritta in un modo che confonde. E i **tipi** devono tornare:
ogni valore dev'essere della specie che il codice si aspetta, un numero dove
serve un numero, un testo dove serve un testo. Tutto questo *prima* di
considerare fatto il lavoro, non dopo averlo già spedito.

```{figure} ../figures/codex-2021.svg
:name: fig-codice-verificato
:alt: "Una descrizione in linguaggio naturale entra nel modello Codex, che genera del codice Python. Il codice non viene accettato così com'è: passa ai test unitari, e conta come corretto solo se li supera tutti. In basso a sinistra, la scheda del banco di prova HumanEval: 164 problemi, circa 7,7 test ciascuno."
:width: 96%

Il cancello, applicato al codice. Il modello propone; a decidere se la
proposta vale è l'esecuzione dei test, che è un giudizio esterno e non
opinabile.
```

La ragione per cui la programmazione è il campo naturale di questi loop si
legge in {numref}`fig-codice-verificato`: per il codice esiste un **oracolo**
automatico e gratuito. «Oracolo» qui non ha niente a che vedere con il futuro:
in informatica è il nome di qualcosa che sa dire, senza discutere, se un
risultato è giusto o sbagliato. Per il codice quell'oracolo sono i test, e li
si esegue in un secondo, quante volte si vuole.

Fuori dal codice l'oracolo non c'è: nessun programma dice se una relazione è
scritta bene o se un'interfaccia si capisce. Lì il cancello va costruito a
mano, e somiglia più a una lista di controllo con delle domande a cui si
risponde sì o no («ci sono tutti i dati richiesti?», «le cifre tornano con
quelle del bilancio?»), oppure a una persona che guarda prima che si spedisca.
Ed è lì che il loop engineering diventa difficile.

È anche il banco di prova che ha reso questa forma di giudizio la norma. Nel
2021, per valutare Codex (un modello addestrato sul codice, antenato degli
assistenti di programmazione di oggi), OpenAI pubblicò **HumanEval**: 164
problemi scritti a mano, ciascuno con una manciata di test, dove una soluzione
conta solo se li supera **tutti** {cite}`chen2021evaluating`. È la forma pura
del cancello: nessun giudizio, nessuna sfumatura, un programma che gira o non
gira.

Questa idea ha una radice accademica precisa, in due lavori che il capitolo
sugli Agenti ha già introdotto e che qui rileggiamo dal lato del loop. Il
primo si chiama ReAct {cite}`yao2023react`, e mostra che intrecciare
**ragionamento e azione** (pensare a parole *e* usare strumenti) rende più del
solo agire. Siccome ogni pensiero è agganciato a quello che gli strumenti
hanno davvero riportato, ReAct si inventa meno cose del ragionamento lasciato
a sé stesso, cioè della catena di pensiero della sezione sul prompt
{cite}`wei2022chain`, che pensa a voce alta senza mai andare a controllare.

Questo non vuol dire che ReAct vinca sempre, e sono gli stessi autori a
misurarlo. Su una raccolta di domande che per rispondere obbligano a incrociare
più informazioni, cercandole una dopo l'altra, la catena di pensiero resta
avanti: risponde giusto nel 29,4 per cento dei casi contro il 27,4. Su una
raccolta di affermazioni da confrontare con una fonte per dire se reggono, è
ReAct a passare davanti: 60,9 contro 56,3. Sono scarti piccoli, due punti nel
primo caso e quasi cinque nel secondo, e la loro piccolezza è il risultato
interessante: nessuno dei due metodi vince in assoluto, e quale sia il
migliore dipende dal compito. Il risultato più alto, in tutti e due i casi,
arriva infatti dai due usati in coppia, uno che parte e l'altro che subentra
quando il primo si arena, con la catena di pensiero nella sua versione a più
tentativi e voto di maggioranza vista nella sezione sul prompt: 35,1 sulle
domande a più passaggi e 64,6 sulle affermazioni da verificare.

Il secondo lavoro si chiama Reflexion {cite}`shinn2023reflexion`, e aggiunge
il tassello mancante: dopo un fallimento l'agente **riflette a parole** sul
proprio errore, scrive quella riflessione in memoria e se la ritrova davanti
al tentativo dopo. È esattamente la stazione «rifletti» del nostro ciclo.

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

Ecco lo scheletro in puro Python, eseguibile. Chi non legge il Python può
saltare il blocco e guardare le quattro righe di risultato che vengono dopo:
si vede il cancello che respinge tre volte e si apre alla quarta.

Nel programma il generatore è un finto modello: invece di ragionare, guarda
l'ultimo motivo di rifiuto e corregge quello. È una caricatura, ma fa la cosa
che conta in questa sezione, cioè lasciarsi guidare dal **contenuto** del
fallimento. Il verificatore invece è vero: controlla che uno **slug** rispetti
tre regole. Slug è il pezzo di indirizzo web che si ricava da un titolo, tutto
minuscolo e con i trattini al posto degli spazi. Il ciclo va avanti finché il
cancello si apre o finiscono i tentativi:

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

```{figure} ../figures/cancello-che-respinge.svg
:name: fig-cancello-che-respinge
:alt: "Una esecuzione del ciclo genera, verifica e raffina: la stringa candidata si riscrive a ogni tentativo, il cancello resta chiuso tre volte e si apre alla quarta, e ogni rifiuto lascia la sua riga nella colonna della memoria, che non ne perde nessuna."
:width: 100%

Lo stesso ciclo, ma in esecuzione. Il candidato riparte da capo a ogni giro;
i motivi del rifiuto no, si accumulano, e sono quelli che il generatore
rilegge. Alla quarta il cancello si apre, e le tre righe restano tutte lì.
```

Poche righe che non «capiscono» nulla, eppure incarnano la spina dorsale del
loop esterno, ed è quella che {numref}`fig-cancello-che-respinge` mostra in
funzione: un cancello che non fa sconti, una memoria del fallimento che
cresce, un tetto ai tentativi. In un sistema vero il generatore è il modello e
la `verifica` è la batteria di test del progetto, ma l'ossatura è questa.

## Tenere il ciclo in mano: gli errori si moltiplicano

Fin qui la parte esaltante. Ora quella onesta, perché un loop mal governato è
uno strumento per sbagliare più in fretta. Prima però guardiamo il metro con
cui questi cicli vengono misurati, perché è un metro che ha i suoi limiti.

```{figure} ../figures/swe-bench-agenti-programmano.svg
:name: fig-swe-bench
:alt: "Catena di valutazione: da una segnalazione di malfunzionamento vera, aperta su GitHub, e dal codice del progetto si parte; l'agente produce una modifica; la modifica viene applicata e sottoposta ai test che il progetto già aveva; e il verdetto è binario, il problema è risolto oppure no."
:width: 100%

Un banco di prova che non ammette interpretazioni. Il compito viene da una
segnalazione vera (una *issue*, nel gergo di chi programma), il giudizio dai
test che il progetto già aveva: né l'uno né gli altri li ha scritti chi
valuta.
```

Il banco di prova disegnato in {numref}`fig-swe-bench` è **SWE-bench**
{cite}`jimenez2024swebench`, che nel capitolo sugli Agenti abbiamo già usato e
discusso. Il suo pregio è anche il suo limite. Un banco così misura ciò che i
test sanno vedere, e i test non sanno vedere tutto: una modifica che li supera
lasciando dietro di sé del codice più difficile da leggere passa comunque, e
nel punteggio non se ne trova traccia. Il guaio si paga più tardi, quando
qualcuno dovrà rimetterci le mani, ed è per questo che chi programma lo chiama
un **debito**. E c'è un'obiezione più dura, che negli Agenti abbiamo riportata
per esteso: rileggendo a mano le prove superate, in circa una su tre la
soluzione era già scritta dentro la segnalazione da risolvere
{cite}`aleithan2024swebenchplus`, cioè l'agente aveva la risposta sotto gli
occhi insieme alla domanda. Un giudice automatico è una gran cosa, ma giudica
solo ciò che qualcuno ha deciso di misurare.

Detto questo, tre problemi vanno guardati in faccia. Il primo è
**aritmetico**, e lo abbiamo già incontrato negli Agenti: gli errori si
accumulano lungo il ciclo. Chiamiamo $p$ la probabilità che un singolo passo
introduca un errore che nessuno intercetta (una probabilità si scrive come una
frazione di uno: $p = 0{,}05$ vuol dire cinque volte su cento). Allora
$1 - p$ è la probabilità che quel passo vada liscio, e se ogni passo sbaglia
per conto proprio, con lo stesso rischio $p$ tutte le volte, la probabilità
che il loop attraversi $n$ passi senza guai è

$$
P(\text{pulito}) = (1 - p)^n,
$$

dove $n$ è il numero di passi del ciclo: moltiplicare venti volte un numero
appena sotto l'uno porta molto più in basso di quanto sembri. Con
$p = 0{,}05$ e $n = 20$ il conto è $0{,}95^{20} \approx 0{,}36$: su cento giri
ne arrivano puliti in fondo trentasei, e i restanti sessantaquattro inciampano
da qualche parte. Un rischio del cinque per cento a ogni passo, che a leggerlo
sembra poco, diventa la maggioranza dei giri andati storti.

Il conto però regge solo finché ogni passo sbaglia per conto suo, e nei loop
veri non è così: l'avvelenamento del contesto, visto nella sezione precedente,
fa sì che uno sbaglio ne tiri dietro altri. Quando gli errori vengono a
grappoli quella moltiplicazione smette di essere la risposta esatta: i giri
tendono a dividersi fra del tutto puliti e rovinati in blocco, invece di
sparpagliarsi. Va presa per quello che è, insomma, un conto all'ingrosso e non
una regola sicura, e serve a dare l'ordine di grandezza di un rischio che
l'intuito sottovaluta.

Resta il motivo per cui il **cancello di verifica** non è un lusso, e sta
proprio in come è definito quel $p$: non è la probabilità di sbagliare, è la
probabilità di sbagliare **senza che nessuno se ne accorga**. Un cancello
intercetta, e quindi abbassa $p$; e siccome quel numero viene moltiplicato per
sé stesso venti volte, abbassarlo un po' cambia moltissimo il risultato.

Il cancello però lavora dentro il ciclo, e contro l'aritmetica c'è una seconda
difesa che sta invece attorno: non consegnare al loop tutto il potere il primo
giorno.

`````{tab} Elementare

Nessuno dà a un nuovo assunto le chiavi dell'azienda il primo giorno. La prima
settimana scrive solo **relazioni** che tu leggi: osserva e riferisce, decidi
tu. Guadagnata un po' di fiducia, può **proporre correzioni** che tu approvi
prima che partano: ecco il secondo cancello promesso all'inizio della sezione,
quello che non è un controllo automatico ma una persona che guarda e dà il via
libera. Solo dopo, e solo per cose di cui ti fidi, il nuovo assunto lavora **da
solo, anche di notte**. Con i loop è identico: si concede autonomia a
scaglioni, non tutta subito. Chi consegna le chiavi il primo giorno non sta
risparmiando tempo: sta preparando il disastro che dovrà poi ripulire.

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

A ogni livello si accompagnano le difese che il capitolo sull'**AI
responsabile** mette in fila per la sicurezza degli LLM: il **minimo dei
permessi** che servono al compito (quali file, quali comandi, quali archivi),
i worktree isolati come recinto, e la **conferma umana** davanti a ogni azione
irreversibile.

`````

Il secondo problema è più sottile e non si risolve con un test. Un loop produce
più codice, più modifiche, più decisioni di quante una persona ne riesca a
leggere, e quello che nessuno ha letto resta un debito: qualcuno, un giorno,
dovrà capirlo, e lo capirà quando serve, cioè quando qualcosa si è rotto. Addy
Osmani lo chiama **comprehension debt** {cite}`osmani2026comprehension`, debito
di comprensione, e l'espressione, va detto, circolava già prima di lui. Il
punto è che i loop **amplificano il giudizio**, quello buono e quello cattivo
con la stessa efficienza: una scelta di partenza azzeccata si moltiplica in
fretta su tutto il lavoro, e una sbagliata pure. Per questo la regola del
«restare l'ingegnere»
non è retorica: chi mantiene il sistema deve **leggere ciò che parte**, non
solo guardare la spia verde dei test. Un loop che nessuno capisce più è un
peso, per quanto verdi siano i suoi cancelli.

Il terzo problema è **economico**. Ogni giro del loop si porta dietro del
testo da far leggere al modello, cioè dei token, e ogni chiamata al modello si
paga; in più occupa dei computer, che qualcuno affitta. Un ciclo che parte
ogni notte, insomma, ha una **bolletta**, e va messo a bilancio come qualsiasi
altra cosa che consuma. E siccome lavora quando nessuno lo guarda, va anche
**sorvegliato**: quante volte è andato a buon fine, quanto è costato ogni
giro, quante volte è dovuta intervenire una persona, e un allarme che suoni se
qualcosa comincia a degenerare. Di come si tengano in funzione, giorno dopo
giorno, i sistemi costruiti sui modelli si occupa il capitolo su **MLOps**. Il
loop engineering, in
fondo, sposta la leva dal prompt al sistema; ma un sistema, a differenza di
una frase, va sorvegliato mentre lavora.

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
  cinque per cento di rischio ciascuno arrivano puliti in fondo trentasei volte
  su cento, cioè poco più di una su tre. È la ragione per cui il cancello non
  è un lusso.
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
  rifletti**) e alla verifica un **cancello deterministico**, a cui nei sistemi
  che toccano cose irreversibili se ne aggiunge un secondo, umano. I suoi
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
  (permessi minimi, cancelli umani ai punti irreversibili, come nel capitolo
  sull'**AI responsabile**) e il **costo** per giro, da mettere a budget
  e monitorare come insegna LLMOps. E lo statuto delle fonti: il vocabolario
  del loop engineering viene da chi costruisce, non da chi misura; qui si
  riporta il meccanismo, che dura, non i nomi degli attrezzi, che cambiano.
```

`````
