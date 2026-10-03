# Loop engineering: progettare il ciclo

Nei primi anni di questa tecnologia la domanda, in ogni squadra che provava a
costruire qualcosa con un LLM, è stata sempre la stessa: qual è il prompt
giusto? Si limava una frase, si aggiungeva un esempio, si spostava una parola,
come chi cerca la combinazione di una cassaforte. Poi, tra chi con questi
strumenti costruisce davvero, la domanda ha cominciato a spostarsi. Nel giugno
2026 Addy Osmani ha attribuito a Boris Cherny, a capo di Claude Code in
Anthropic (un assistente di programmazione nato per il terminale: gli si
scrive, e lui legge e modifica i file del progetto), una frase che riassume lo
spostamento: Cherny non scrive quasi più prompt al modello, scrive **loop**
(giri, cicli), cioè programmi che quei prompt li mandano al posto suo
{cite}`osmani2026loop`. È uno spostamento di leva e non una provocazione. La
cosa su cui conviene lavorare non è più il singolo messaggio, ma il sistema di
controllo che attorno a quel messaggio decide quando parte, cosa gli si mette
davanti, come si verifica il risultato e cosa succede dopo.

Si sale così al terzo e più esterno dei tre cerchi: dopo il prompt (il
singolo messaggio) e il contesto (la finestra come sistema), il loop. È
l'anello in cui il prompt e il contesto smettono di essere una cosa che scrivi
*tu, adesso* e diventano una cosa che un programma monta, esegue e rimette in
moto, magari mentre dormi. Peter Steinberger, in un post dello stesso giorno,
lo dice quasi con le stesse parole: non si fanno più prompt agli agenti che
programmano, si progettano i cicli che quei prompt li fanno da soli
{cite}`steinberger2026loops`. E Osmani ne trae una conseguenza che è metà
tecnica e metà morale: il ciclo va costruito da chi ha intenzione di restare
l'ingegnere, cioè di continuare a capire e a rispondere di quello che esce,
non da chi vuole premere «vai» e andarsene.

Il vocabolario del loop engineering lo hanno scritto quasi tutto dei
praticanti, persone che questi cicli li costruiscono, non gruppi di ricerca
che li misurano; e lo hanno scritto da pochi mesi, tanto che nessuno di questi
nomi ha ancora avuto il tempo di essere smentito o confermato. Di quel
racconto qui interessa il meccanismo, che ha buone probabilità di durare
più dei nomi: gli attrezzi si leggano come esempi di oggi, non come parti
della materia.

## Il ciclo come unità di progetto

Prima, però: la parola «loop» ha due significati annidati, e conviene tenerli
distinti invece di lasciarli capire. Il primo è il giro di un agente (osserva,
ragiona, agisci), e il giro più comune di tutti è quello che fai tu in una
chat: chiedi, guardi la risposta, storci il naso, richiedi meglio. Quel loop
esiste ancora; solo che lo giri tu, a mano, e finisce quando chiudi la
finestra. Il secondo è lo stesso giro affidato a un programma, che lo fa
partire da sé, lo ripete e ne conserva l'esito. È un annidamento e non una
sostituzione: dentro c'è ancora il ciclo di prima. Quello che cambia è chi lo
mette in moto e chi decide quando è finito, e quel «chi», da qui in avanti,
non è più una persona davanti a una tastiera.

L'unità di lavoro del loop engineering non è la richiesta, ma il ciclo:
una sequenza che si ripete (*pianifica, esegui, verifica, rifletti*) e poi
ricomincia, portandosi dietro ciò che ha imparato. Ha la stessa forma del giro
*osserva, ragiona, agisci* con cui lavora un agente, ma sta un piano sopra: il
loop engineering distingue due cicli annidati, uguali nella forma e diversi
nel mestiere.

`````{tab} Elementare

Un artigiano al banco ha un ciclo di lavoro stretto: guarda il pezzo, decide
la prossima mossa, la fa, guarda di nuovo; avanti così finché l'oggetto è
finito. Questo è il ciclo *interno*, quello dentro la sua testa e
le sue mani, e dura quanto dura un lavoro.

La bottega del capitolo sugli agenti divideva il lavoro fra chi costruisce e
chi collauda; qui serve una figura in più, che nessuno dei due fa. Sopra
l'artigiano c'è il **capobottega**. Lui non intaglia: decide *quando*
si comincia (lunedì mattina, o ogni notte alle tre), tiene un registro di
cosa è stato fatto, controlla il pezzo finito prima di spedirlo, e se non va
lo rimanda indietro con un appunto. Il capobottega è il ciclo *esterno*. Il
loop engineering è il mestiere di progettare il capobottega: non le singole
intagliature, ma la cadenza, il registro, il controllo, la ripartenza. Un
artigiano bravissimo senza capobottega lavora finché lo guardi; con un buon
capobottega lavora anche di notte, e quello che consegna è già stato
controllato.

Finito un pezzo, l'artigiano se ne va, e al banco la mattina dopo ne siede un
altro, riposato e senza un ricordo di ieri. Quello che deve sapere sta sul
registro, e glielo mette davanti il capobottega. La memoria della bottega sta
lì, non nella testa di chi intaglia.

Nella bottega, l'artigiano e il capobottega sono tutti e due dei programmi. Tu
sei il proprietario: non stai al banco e non fai i turni, ma decidi quanta
corda dare al capobottega, e la bottega resta tua, compreso quello che ne
esce.

`````

`````{tab} Superiore

Il **loop interno** è il ciclo dell'agente definito nell’{doc}`anatomia di un
agente </Agenti/overview>`: *osserva → ragiona → agisci*, con lo stato che
vive nella finestra di contesto
ed è effimero (finita la conversazione, svanisce).

Il **loop esterno** è ciò che il loop engineering progetta, e ha proprietà che
il loop interno non ha:

- è schedulato, parte a una cadenza (un cron, un evento, un trigger), non
  solo quando un umano digita;
- ha stato persistente, non tiene la memoria nella finestra, ma *fuori*,
  su disco o in un database, così che sopravviva alla singola invocazione (il
  contrario dello *scratchpad* degli agenti, che vive quanto vive la finestra);
- ha una verifica deterministica, un cancello che decide, con un criterio
  esterno e non con l'autovalutazione del modello, se il ciclo è riuscito;
- spesso a ogni giro istanzia un agente fresco, con contesto pulito,
  invece di accumulare cronologia all'infinito: riprendendo lo stato
  dall'esterno.

Letto come sistema di controllo, il loop interno è un controllore a
retroazione dentro un singolo episodio; il loop esterno è il supervisore che
decide quanti episodi avviare, con quali condizioni iniziali, e come
giudicarne l'esito.

`````

Il ciclo esterno, disegnato per esteso, ha quattro stazioni. La
{numref}`fig-loop-ciclo` le mostra chiuse in cerchio, con il dettaglio che regge
tutto il resto: alla stazione di verifica c'è un **cancello**,
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

Un capobottega è fatto di attrezzi concreti. Cobus
Greyling {cite}`greyling2026loop` ne ha raccolto il repertorio, e conviene
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
- **La copia di lavoro.** Ogni giro lavora su una copia separata dei file del
  progetto, non su quella buona. Così più giri possono andare insieme senza
  pestarsi i piedi, e le modifiche di un giro restano nella sua copia finché
  qualcuno non le accetta. Il programma che custodisce il codice e la sua
  storia si chiama git; nel suo gergo una di quelle copie separate è un
  *worktree*. Non è un recinto, però: separa i file, non quello che i comandi
  lanciati da lì possono toccare (la rete, le credenziali, il resto del
  computer), e per quello servono permessi minimi e un ambiente isolato.
- **Le istruzioni riusabili.** Le istruzioni non si riscrivono ogni volta: si
  impacchettano una volta sola, si dà loro un nome, e si tengono in archivio
  accanto al codice, con la loro storia delle modifiche. Così una correzione
  fatta oggi vale per tutti i giri di domani (negli attrezzi del 2026 questi
  pacchetti si chiamano *skill*). È l'idea, già incontrata nel capitolo sugli
  Agenti, che un prompt vada trattato come si tratta il codice: archiviato,
  corretto in un posto solo, con la storia delle sue versioni.
- **Chi fa e chi controlla, separati.** Il lavoro si divide in due ruoli
  affidati a due agenti distinti, cioè a due copie del modello ciascuna
  con le sue istruzioni e la sua finestra: una produce, l'altra giudica. Ci
  torniamo fra poco.
- **La memoria fuori dalla finestra.** Quello che il ciclo sa non vive nella
  conversazione, ma in file che legge e riscrive a ogni giro: uno con il
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
  chiama MCP (*Model Context Protocol*). È così che il ciclo tocca il
  mondo invece di limitarsi a produrre testo.

Nessuno di questi attrezzi usa il modello per decidere: sono la struttura
attorno al modello (lo *scaffolding*, in inglese) che rende ripetibile il suo
lavoro.

### Due agenti, non uno: chi fa e chi controlla

Fra i componenti, la separazione fra chi fa e chi controlla merita qualche
riga in più, perché a prima vista sembra uno spreco: vuol dire far lavorare il
modello due volte invece che una, e quindi pagare due volte, in tempo e in
denaro. Ripaga quando il controllore sa o può fare qualcosa che chi produce non
fa: ha davanti criteri scritti, esegue i test, consulta la documentazione. Una
seconda copia dello stesso modello con le stesse informazioni, chiamata a
rivedere un ragionamento senza un riscontro esterno, non lo migliora, e a
volte lo peggiora {cite}`huang2024selfcorrect`. (I due ruoli, in inglese, si
chiamano *maker* e *checker*, e così si trova scritto lo schema.)

`````{tab} Elementare

Lo scrittore butta giù il pezzo; il redattore lo legge, segna cosa non va e lo
rimanda indietro. Potresti chiedere allo scrittore di rileggersi da solo, ma
tutti sappiamo com'è: l'autore è il peggior giudice del proprio testo, perché
legge quello che *voleva* scrivere, non quello che ha scritto. Tenere due ruoli
separati serve proprio a questo: il controllore arriva senza aver visto la
fatica di chi ha prodotto, e giudica il risultato per quello che è. Nel loop,
il *maker* scrive, il *checker* controlla, e sono due «persone» diverse: due
agenti separati, ciascuno con le sue istruzioni e il suo foglio davanti.

Qui è lecito obiettare: se sono due copie dello stesso modello, che senso ha?
Uno pensa come l'altro. La risposta è che una parte della differenza sta in
quello che ciascuno ha davanti. Il primo ha davanti il compito e tutta la
strada che ha fatto per svolgerlo, con le sue giustificazioni; il secondo ha
davanti solo il risultato e i criteri con cui giudicarlo, e non si lascia
convincere da una strada che non ha visto.

Un'altra parte, però, la separazione non la tocca. Un modello messo a
giudicare tende a preferire il testo che ha scritto lui, e lo preferisce di
più quanto meglio lo riconosce come proprio. Lo hanno misurato proprio con un
controllore separato, a cui non si diceva quale testo fosse suo: il proprio
stile lo riconosceva lo stesso. E restano i punti ciechi comuni: quello che il
modello non sa vedere non lo vede nemmeno da controllore. Per questo il
controllore serve a migliorare il lavoro, e il verdetto finale lo dà un
cancello che non è un modello.

`````

`````{tab} Superiore

Il pattern è due sotto-agenti con contesti separati e prompt distinti:
il *maker* riceve il compito e produce la modifica; il *checker* riceve solo
il risultato e i criteri, e restituisce un verdetto (passa / non passa) con le
motivazioni. La separazione dei contesti serve a due scopi, e non a un terzo
che le si attribuisce volentieri. Primo, il checker non vede la traiettoria del
maker: giudica il risultato contro i criteri, non le giustificazioni con cui
il maker ci è arrivato, e non ne eredita le premesse sbagliate (il
*poisoning* della {doc}`sezione sul contesto <context-engineering>`). Secondo,
attenua la correlazione dei fallimenti: se lo stesso agente, con lo
stesso contesto, sbaglia a produrre *e* a giudicare, i due errori sono
perfettamente correlati e il controllo è teatro. Un checker con contesto
pulito, e magari con criteri più severi, quella correlazione la abbassa; non
la annulla. Se maker e checker sono lo stesso modello cambia il
condizionamento, non i punti ciechi, e un errore che nasce da una lacuna del
modello lo vedono tutt'e due allo stesso modo. Quanto ne resta non lo
sappiamo, e una misura diretta di quella correlazione residua, fra due istanze
dello stesso modello con contesti separati, non ci risulta.

Il terzo scopo, che la separazione non raggiunge, è togliere la preferenza per
sé. Messo a giudicare, un modello preferisce il testo che ha prodotto lui a uno
equivalente prodotto da altri, e lo fa tanto più quanto meglio lo riconosce come
proprio. Nella prova di Panickssery e colleghi il giudice è già una chiamata
separata dello stesso modello, a cui non si dice quale dei due testi sia suo, e
la preferenza c'è lo stesso {cite}`panickssery2024selfpreference`; il legame con
l'auto-riconoscimento è una correlazione lineare, che i controlli degli autori
rendono plausibile come causa senza dimostrarla. Un giudice di un'altra famiglia
nel testo non si riconosce, ma gli stessi autori avvertono che un modello premia
anche i testi che somigliano ai propri: il problema si sposta più che sparire. È
una delle ragioni per cui, sopra il checker, il cancello resta deterministico;
degli altri pregiudizi di un modello che fa da giudice, la posizione e la
lunghezza, si occupa più avanti la {doc}`sezione su LLMOps </MLOps/llmops>`. In
cambio si paga un secondo giro di inferenza (token e latenza in più) che va
messo a bilancio come ogni altra spesa del loop.

`````

## Il cancello di verifica: verificare, non sperare

Arriviamo alla stazione che dà senso a tutte le altre: la verifica. Nel
ciclo della {numref}`fig-loop-ciclo` è il punto in cui si decide se il giro è
riuscito, e la scelta di progetto è netta: la verifica dev'essere un
cancello, non un augurio. Un cancello ha due stati, aperto o chiuso; non
esiste il «quasi passato». (In inglese si chiama *validation gate*, ed è la
stessa cosa: il cancello che convalida.)

Nel caso del codice i controlli sono tre, tutti automatici. I test devono
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
proposta vale è l'esecuzione dei test, un giudizio che non dipende
dall'opinione di chi valuta, e che vale quanto i test.
```

La ragione per cui la programmazione è il campo naturale di questi loop si
legge in {numref}`fig-codice-verificato`: per il codice esiste un **oracolo**
automatico e gratuito. «Oracolo» qui non ha niente a che vedere con il futuro:
in informatica è il nome di qualcosa che sa dire, senza discutere, se un
risultato è giusto o sbagliato. Per il codice quell'oracolo sono i test, e li
si esegue in un secondo, quante volte si vuole. È però un oracolo parziale:
quando un test fallisce il no è certo, ma il sì vuol dire soltanto che i test
scritti passano.

Fuori dal codice l'oracolo non c'è: nessun programma dice se una relazione è
scritta bene o se un'interfaccia si capisce. Lì il cancello va costruito a
mano, e somiglia più a una lista di controllo con delle domande a cui si
risponde sì o no («ci sono tutti i dati richiesti?», «le cifre tornano con
quelle del bilancio?»), oppure a una persona che guarda prima che si spedisca.
Ed è lì che il loop engineering diventa difficile.

Il banco di prova di {numref}`fig-codice-verificato` è anche quello che ha
reso questa forma di giudizio la norma. Nel 2021, per valutare Codex (un
modello addestrato sul codice, antenato degli assistenti di programmazione di
oggi), OpenAI pubblicò **HumanEval**: 164 problemi scritti a mano, ciascuno con
una manciata di test, dove una soluzione conta solo se li supera tutti
{cite}`chen2021evaluating`. È la forma pura del cancello: nessun giudizio,
nessuna sfumatura, un programma che gira o non gira. E ne mostra anche il
limite, che sta nella manciata. Liu e colleghi hanno moltiplicato per ottanta
i test di HumanEval, e su ventisei modelli la quota di problemi risolti si è
ridotta, nei casi peggiori, del 19,3-28,9 per cento: erano soluzioni sbagliate
che i test originali lasciavano passare {cite}`liu2023evalplus`.

Il ciclo attorno al cancello (provare, farsi respingere, ripensarci,
riprovare) ha invece antecedenti in due lavori del 2022 e del 2023, che il
capitolo sugli Agenti ha già introdotto e che qui rileggiamo dal lato del
loop. Il primo si chiama ReAct {cite}`yao2023react`, e mostra che intrecciare
ragionamento e azione (pensare a parole *e* usare strumenti) rende più del
solo agire. Siccome ogni pensiero è agganciato a quello che gli strumenti
hanno davvero riportato, ReAct si inventa meno cose del ragionamento lasciato
a sé stesso, cioè della catena di pensiero della sezione sul prompt
{cite}`wei2022chain`, che pensa a voce alta senza mai andare a controllare.

Questo non vuol dire che ReAct vinca sempre, e sono gli stessi autori a
misurarlo: sulle domande che obbligano a incrociare più fatti la catena di
pensiero resta di poco avanti, sulle affermazioni da verificare contro una
fonte passa avanti ReAct, e il risultato migliore viene dai due metodi in
coppia, con la regola del cambio di turno che dà la {doc}`sezione sul ciclo
dell'agente </Agenti/agenti-e-tool-use>`. Sono misure di un modello del 2022;
quello che non invecchia è che l'ordine giusto lo detta il compito.

Il secondo lavoro si chiama Reflexion {cite}`shinn2023reflexion`, e aggiunge
il tassello mancante: dopo un fallimento l'agente riflette a parole sul
proprio errore, scrive quella riflessione in memoria e se la ritrova davanti
al tentativo dopo. È esattamente la stazione «rifletti» del nostro ciclo.
Self-Refine fa la stessa cosa senza memoria, con un solo modello che produce,
si critica e si corregge {cite}`madaan2023selfrefine`. Il limite di entrambi
sta nell'origine del segnale. Senza un riscontro esterno, l'auto-correzione
non migliora il ragionamento e a volte lo peggiora, e i guadagni riportati per
Reflexion sul ragionamento vengono da etichette che dicono al modello se ha
sbagliato {cite}`huang2024selfcorrect`; per il codice, nel lavoro originale, i
test li scrive il modello stesso, e su uno dei due banchi, fra le soluzioni
che li superano, circa una su sei è sbagliata, come racconta la
{doc}`sezione sul ciclo dell'agente </Agenti/agenti-e-tool-use>`.

`````{tab} Elementare

Il cancello è come un tornello alla metropolitana: o il biglietto è valido e
passi, o non lo è e resti fuori. Non c'è un tornello che ti fa passare «a
metà», e non c'è modo di convincerlo. Puoi essere sicurissimo del tuo
biglietto, la sbarra resta ferma lo stesso. Quando resti fuori, però, non è
finita: leggi *perché* (biglietto scaduto, importo sbagliato), rimedi e
riprovi. Un buon loop fa così. Prova, sbatte contro il cancello, legge il
motivo del rifiuto (proprio come uno studente che rilegge le correzioni in
rosso prima di riscrivere il tema) e riprova con quel motivo in mano. Ripete
finché passa o finché ha esaurito i tentativi che gli hai concesso, e quel
tetto serve: senza, chi non ne viene fuori resta al tornello fino a domattina
a comprare biglietti nuovi, e i biglietti li paghi tu.

Il tornello, però, controlla solo quello che sa controllare. Se legge soltanto
la data, un biglietto falso con la data giusta passa lo stesso; e se il
biglietto se lo stampa da solo chi deve entrare, il tornello non protegge più
niente. Un cancello vale quanto le sue prove.

`````

`````{tab} Superiore

Il ciclo pratico è genera → verifica → raffina. La verifica è un predicato
deterministico ed esterno (la suite di test, il type-checker, il linter)
che ritorna un booleano, non un giudizio del modello su sé stesso. La
riflessione (Reflexion) è invece *interna*: il modello propone una diagnosi in
linguaggio naturale dell'errore e la usa come contesto per il tentativo
seguente. La divisione dei ruoli è la chiave dell'affidabilità: il modello
propone, il cancello deterministico dispone. Ci si affida al giudizio del
modello per *migliorare*, mai per *dichiarare fatto*: quel verdetto lo dà un
criterio che il modello non può compiacere. Il predicato però è un oracolo
incompleto: con test deboli passano anche correzioni sbagliate, e se i test li
scrive il modello che viene giudicato (o se il ciclo gli permette di
modificarli) il cancello misura la sua coerenza con sé stesso, non la
correttezza. Il loop termina alla prima verifica positiva o all'esaurirsi di
un budget di $K$ tentativi: un limite esplicito, senza il quale un ciclo che
non converge gira all'infinito bruciando token. Se ogni tentativo passasse il
cancello con probabilità $h$, indipendentemente dagli altri, il cancello
resterebbe chiuso dopo $K$ giri con probabilità $(1-h)^K$; i tentativi dello
stesso modello con lo stesso feedback sono correlati, e quella stima va letta
come ottimistica.

`````

Lo scheletro, in puro Python ed eseguibile. Il generatore è un finto modello:
invece di ragionare, guarda l'ultimo motivo di rifiuto e corregge quello. È una
caricatura, ma fa la cosa che conta, cioè lasciarsi guidare dal contenuto
del fallimento. Il verificatore invece è vero: controlla che uno **slug**
rispetti tre regole. Slug è il pezzo di indirizzo web che si ricava da un
titolo, tutto minuscolo e con i trattini al posto degli spazi. Il ciclo va
avanti finché il cancello si apre o finiscono i tentativi, e nel risultato si
vede il cancello che respinge tre volte e si apre alla quarta:

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

Poche righe che non «capiscono» nulla, eppure incarnano le stazioni di
verifica e di riflessione del loop esterno, ed è quello che
{numref}`fig-cancello-che-respinge` mostra in funzione: un cancello che non fa
sconti, una memoria del fallimento che cresce, un tetto ai tentativi. Mancano
la sveglia e lo stato su file, che in un sistema vero fanno ripartire il giro
la notte dopo; e il generatore è il modello, la `verifica` è la batteria di
test del progetto. L'ossatura del cancello, però, è questa.

## Tenere il ciclo in mano: gli errori si moltiplicano

Fin qui la parte esaltante. Ora quella onesta, perché un loop mal governato è
uno strumento per sbagliare più in fretta. Prima però guardiamo il metro con
cui questi cicli vengono misurati, perché è un metro che ha i suoi limiti.

```{figure} ../figures/swe-bench-agenti-programmano.svg
:name: fig-swe-bench
:alt: "Catena di valutazione: da una segnalazione di malfunzionamento vera, aperta su GitHub, e dal codice del progetto si parte; l'agente produce una modifica; la modifica viene applicata e sottoposta ai test che il progetto già aveva; e il verdetto è binario, il problema è risolto oppure no."
:width: 100%

Un banco di prova con un verdetto automatico. Il compito viene da una
segnalazione vera (una *issue*, nel gergo di chi programma), il giudizio dai
test che il progetto già aveva: né l'uno né gli altri li ha scritti chi
valuta.
```

Il banco di prova disegnato in {numref}`fig-swe-bench` è SWE-bench
{cite}`jimenez2024swebench`, che nel capitolo sugli Agenti abbiamo già usato e
discusso. Il suo pregio è anche il suo limite. Un banco così misura ciò che i
test sanno vedere, e i test non sanno vedere tutto: una modifica che li supera
lasciando dietro di sé del codice più difficile da leggere passa comunque, e nel
punteggio non se ne trova traccia. Il guaio si paga più tardi, quando qualcuno
dovrà rimetterci le mani, ed è per questo che chi programma lo chiama un
**debito**. E ci sono due obiezioni più dure, che vengono dallo stesso lavoro.
Rileggendo a mano le prove superate, Aleithan e colleghi trovano che in circa
una su tre la soluzione era già scritta dentro la segnalazione da risolvere,
cioè l'agente aveva la risposta sotto gli occhi insieme alla domanda (negli
Agenti l'abbiamo riportata per esteso); e che il 31% delle correzioni accettate
era sospetto, sbagliato o incompleto, e passava soltanto perché i test erano
troppo deboli per accorgersene. Tolti i casi dubbi, il tasso di successo
dell'agente che misuravano scende dal 12,47% al 3,97%
{cite}`aleithan2024swebenchplus`. E il 23 febbraio 2026 OpenAI ha smesso di
riportare i punteggi di SWE-bench Verified, per test difettosi e contaminazione
{cite}`openai2026sweverified`. Un giudice automatico è una gran cosa, ma giudica
solo ciò che qualcuno ha deciso di misurare, e un cancello vale quanto i suoi
test.

Detto questo, i limiti sono tre. Il primo è aritmetico, e lo abbiamo già
incontrato negli Agenti: gli errori si accumulano lungo il ciclo. Chiamiamo $p$
la probabilità che un singolo passo introduca un errore che nessuno intercetta
(una probabilità si scrive come una frazione di uno: $p = 0{,}05$ vuol dire
cinque volte su cento). Allora $1 - p$ è la probabilità che quel passo vada
liscio, e se ogni passo sbaglia per conto proprio, con lo stesso rischio $p$
tutte le volte, la probabilità che il loop attraversi $n$ passi senza guai è

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
veri non è così: l'avvelenamento del contesto, visto con il {doc}`context
engineering <context-engineering>`, fa sì che uno sbaglio ne tiri dietro
altri. Quando gli errori vengono a grappoli, e $p$ è la frequenza media con cui
i passi sbagliano, $(1-p)^n$ sbaglia in un verso preciso: i giri puliti sono
*di più* di quanto dica, e quelli sporchi sono più sporchi. Gli sbagli, in
media, sono gli stessi; si concentrano in meno giri. Il conto si fa in poche
righe, con il modello più semplice del contagio: dopo un passo sbagliato, il
passo successivo sbaglia una volta su due.

```python
n, p = 20, 0.05   # venti passi, uno sbaglio ogni venti passi in media

# passi indipendenti: ognuno sbaglia con probabilita' p, qualunque cosa sia
# successa prima
pulito_indip = (1 - p) ** n

# errori a grappoli: dopo un passo sbagliato il successivo sbaglia con
# probabilita' r = 0,5; dopo un passo buono con una probabilita' q piu'
# bassa, scelta in modo che la frequenza media degli sbagli resti p
r = 0.5
q = p * (1 - r) / (1 - p)
pulito_grappoli = (1 - p) * (1 - q) ** (n - 1)

# gli sbagli attesi in un giro sono n * p = 1 in tutti e due i casi:
# cambia solo come si distribuiscono fra i giri
casi = (("indipendenti", pulito_indip), ("a grappoli", pulito_grappoli))
for nome, pulito in casi:
    print(f"{nome}: giri puliti {pulito:.2f}, "
          f"sbagli per giro sporco {n * p / (1 - pulito):.1f}")
```

```text
indipendenti: giri puliti 0.36, sbagli per giro sporco 1.6
a grappoli: giri puliti 0.57, sbagli per giro sporco 2.3
```

Con la stessa frequenza media, uno sbaglio ogni venti passi, i giri puliti
salgono da 36 a 57 su cento, e ogni giro sporco porta in media 2,3 sbagli
invece di 1,6. Per errori che si tirano dietro a vicenda la formula dei passi
indipendenti è quindi pessimistica sulla quota di giri puliti; quello che
nasconde è quanto vanno male i giri che vanno male.

Resta il motivo per cui il cancello di verifica non è un lusso, e sta
proprio in come è definito quel $p$: non è la probabilità di sbagliare, è la
probabilità di sbagliare senza che nessuno se ne accorga. Un cancello
intercetta, e quindi abbassa $p$; e siccome il conto moltiplica venti volte
$1 - p$, ogni piccolo guadagno su $p$ si moltiplica anche lui venti volte. Se
il cancello porta gli sbagli che passano inosservati dal cinque al due per
cento, cioè ne intercetta tre su cinque ($0{,}05 \cdot (1 - 0{,}6) = 0{,}02$),
i giri puliti su venti passi salgono da trentasei a sessantasette su cento
($0{,}98^{20} \approx 0{,}67$). Con gli errori a grappoli fa anche di più,
perché fermando il primo sbaglio ferma quelli che si sarebbe tirato dietro.

Il cancello però lavora dentro il ciclo, e contro l'aritmetica c'è una seconda
difesa che sta invece attorno: non consegnare al loop tutto il potere il primo
giorno.

`````{tab} Elementare

Nessuno dà a un nuovo assunto le chiavi dell'azienda il primo giorno. La prima
settimana scrive solo relazioni che tu leggi. Lui osserva e riferisce, a
muovere le cose sei tu. Poi può proporre correzioni, che però passano dalle
tue mani prima di partire, ed ecco il secondo cancello, quello tenuto da una
persona invece che da un programma. Il grado dopo si guadagna sui numeri:
quante delle sue proposte erano buone, quante hai dovuto rifarle. E servono
tanti numeri: cento proposte buone di fila non bastano a dire che sbaglia meno
di una volta su cento, ne servono circa trecento. Il conto da fare è semplice.
Rileggere tutto ti costa un'ora al giorno; uno sbaglio che ti sfugge ti costa
una giornata intera, e ne capita uno ogni tanto. Quando l'ora di rilettura
comincia a pesare più degli sbagli che ti risparmia, allora lavora da solo,
anche di notte: dentro un elenco scritto di quello che può toccare, e con
qualcuno che ogni tanto guarda come sta andando. Per le cose che non si
possono disfare (un pagamento già partito, un archivio cancellato) viene a
chiedere comunque, anche dopo dieci anni di servizio. Con i loop è identico:
l'autonomia si concede un gradino alla volta, e le chiavi consegnate il primo
giorno sono un disastro rimandato.

`````

`````{tab} Superiore

L'autonomia matura si concede per livelli, allargando il raggio d'azione solo
quando le metriche lo giustificano. La scala che segue è quella del
repertorio di Greyling, che raccomanda di salire solo dopo che il verificatore
ha avuto ragione per una settimana {cite}`greyling2026loop`:

- **L1, solo report.** Il loop osserva e *propone*: apre una segnalazione,
  scrive una diagnosi. L'umano applica. Raggio d'azione nullo sul sistema.
- **L2, fix assistiti.** Il loop produce la modifica (una pull request, una
  patch) ma non la integra: c'è un cancello umano che rivede e fonde. È il
  livello a cui conviene fermarsi finché il costo di una revisione resta
  minore del costo atteso di un errore integrato senza guardarlo, cioè finché
  $c_{\text{rev}} < p_{\text{err}}\, c_{\text{err}}$, con $p_{\text{err}}$ la
  frequenza degli errori che la revisione intercetta.
- **L3, non presidiato.** Il loop integra da solo, ma dentro i confini di
  una *allow-list* (quali file, quali comandi, quali repository) e sotto
  monitoraggio continuo.

Una settimana è una regola pratica; il criterio che la rende misurabile è
statistico. Se in $m$ revisioni di L2 non si è trovato nessun errore, l'estremo
superiore dell'intervallo di confidenza al 95% del tasso d'errore è
$1 - 0{,}05^{1/m} \approx 3/m$ (la *regola del tre*): per poter dire che il
loop sbaglia meno di una volta su cento servono circa trecento revisioni
pulite, e la stima di $p_{\text{err}}$ da usare nella disuguaglianza di L2 è
quell'estremo, non lo zero osservato.

A ogni livello si accompagnano le difese che la {doc}`sezione sulla
sicurezza degli LLM </AIResponsabile/sicurezza-llm>` mette in fila, il minimo
dei permessi che servono al compito (quali file, quali comandi, quali
archivi) e la conferma umana davanti a ogni azione irreversibile. Il
*worktree* separato di ogni giro non è una di queste difese: evita che due
giri paralleli si pestino i piedi, ma non limita che cosa i comandi possono
toccare.

`````

Il secondo limite è più sottile e non si risolve con un test. Un loop produce
più codice, più modifiche, più decisioni di quante una persona ne riesca a
leggere, e quello che nessuno ha letto resta un debito: qualcuno, un giorno,
dovrà capirlo, e lo capirà quando serve, cioè quando qualcosa si è rotto. Addy
Osmani lo chiama **comprehension debt** {cite}`osmani2026comprehension`, debito
di comprensione, e l'espressione, va detto, circolava già prima di lui. Il
punto è che i loop amplificano il giudizio, quello buono e quello cattivo
con la stessa efficienza: una scelta di partenza azzeccata si moltiplica in
fretta su tutto il lavoro, e una sbagliata pure. Per questo la regola del
«restare l'ingegnere»
non è retorica: chi mantiene il sistema deve leggere ciò che parte, non
solo guardare la spia verde dei test. Un loop che nessuno capisce più è un
peso, per quanto verdi siano i suoi cancelli.

Il terzo limite è economico. Ogni giro del loop si porta dietro del testo da far
leggere al modello, cioè dei token, e ogni chiamata al modello si paga; in più
occupa dei computer, che qualcuno affitta. Un ciclo che parte ogni notte,
insomma, ha un costo ricorrente, e va messo a bilancio come qualsiasi altra cosa
che consuma. E siccome lavora quando nessuno lo guarda, va anche sorvegliato:
quante volte è andato a buon fine, quanto è costato ogni giro, quante volte è
dovuta intervenire una persona, e un allarme che suoni se qualcosa comincia a
degenerare. Di come si tengano in funzione, giorno dopo giorno, i sistemi
costruiti sui modelli si occupa il {doc}`capitolo su MLOps </MLOps/overview>`.
Il loop engineering, in fondo, sposta la leva dal prompt al sistema; ma un
sistema, a differenza di una frase, va sorvegliato mentre lavora.

`````{tab} Elementare

```{admonition} Da ricordare
:class: important
- Il terzo cerchio sposta l'attenzione dalla frase al processo: non si
  cerca più il messaggio perfetto, si progetta il giro che di messaggi ne fa
  tanti. Il giro che fai tu, chiedendo e richiedendo, resta dentro: quello che
  si aggiunge è un capobottega che lo mette in moto, lo controlla e lo rimanda
  indietro quando non va.
- Il capobottega ha quattro stazioni (pianifica, esegui, verifica,
  rifletti) e un cancello alla verifica. Cancello vuol dire due stati e
  basta, come un tornello: non esiste il «quasi passato».
- Il cancello non lo tiene il modello. Il modello propone, il cancello
  dispone, e il cancello è un controllo automatico (per il codice: dei
  programmi di prova che girano da soli). Chiedere al modello se ha finito
  non basta: tende a dirsi di sì, e preferisce il proprio lavoro anche quando
  non sa che è suo. E il cancello vale quanto le sue prove: se sono poche, o
  se le scrive chi viene controllato, lascia passare anche il lavoro
  sbagliato.
- Quando il cancello respinge, quello che serve è il motivo: si riparte da
  lì, non da capo. E si mette sempre un tetto ai tentativi, altrimenti un giro
  che non converge gira per sempre.
- Gli errori si moltiplicano. Un giro lungo con un rischio piccolo a ogni
  passo finisce male più spesso di quanto l'intuito dica: venti passi con il
  cinque per cento di rischio ciascuno, se ogni passo sbaglia per conto suo,
  arrivano puliti in fondo trentasei volte su cento, poco più di una su tre.
  Nei cicli veri uno sbaglio ne tira dietro altri: i giri puliti sono allora
  un po' di più, ma quelli che si sporcano si sporcano di più. In tutti e due
  i casi il cancello non è un lusso.
- L'autonomia si concede un gradino alla volta, come a un nuovo assunto:
  prima solo relazioni da leggere, poi proposte da approvare, e solo alla
  fine, e solo dentro confini scritti, il permesso di fare da sé. Il gradino
  si guadagna con molti numeri, non con una settimana fortunata. Per le cose
  che non si possono disfare, la conferma di una persona si chiede sempre.
- Tre cose da tenere d'occhio: un ciclo produce più roba di quanta se ne
  riesca a leggere (e il conto arriva), amplifica tanto il giudizio buono
  quanto quello cattivo, e costa: gira mentre non lo guardi, e il conto
  arriva lo stesso.
```

`````

`````{tab} Superiore

```{admonition} Da ricordare
:class: important
- Il loop engineering sposta la leva dal singolo prompt al sistema di
  controllo: non si cerca più la frase perfetta, si progetta il ciclo che di
  frasi ne fa tante. È il terzo cerchio, il più esterno, dopo prompt e contesto.
- Ci sono due cicli annidati: il *loop interno* dell'agente (osserva →
  ragiona → agisci, già visto negli Agenti) e il *loop esterno* che il loop
  engineering progetta (schedulato, con stato persistente fuori dalla
  finestra e verifica esterna).
- Il ciclo esterno ha quattro stazioni (pianifica → esegui → verifica →
  rifletti) e alla verifica un cancello deterministico, a cui nei sistemi
  che toccano cose irreversibili se ne aggiunge un secondo, umano. I suoi
  componenti: scheduling, worktree separati (che isolano i file, non i
  permessi), skill riusabili, split maker/checker, stato su file, integrazione
  (MCP/git/ticket).
- La separazione maker/checker toglie la contaminazione dalla traiettoria del
  maker e attenua la correlazione degli errori, ma non la preferenza per sé
  {cite}`panickssery2024selfpreference`; e una seconda copia dello stesso
  modello senza riscontro esterno non migliora il ragionamento
  {cite}`huang2024selfcorrect`.
- La verifica è un cancello, non un augurio: un predicato deterministico
  (test, lint, tipi) che il modello non può compiacere, e che vale quanto i
  test: con test deboli passano correzioni sbagliate
  {cite}`liu2023evalplus, aleithan2024swebenchplus`. Il modello *propone*
  (riflessione alla Reflexion {cite}`shinn2023reflexion`, azione+ragionamento
  alla ReAct {cite}`yao2023react`), il cancello *dispone*.
- Gli errori si moltiplicano lungo il ciclo: *se* i passi sono
  indipendenti $(1-p)^n$ decade in fretta; se l'avvelenamento del contesto li
  correla, a parità di frequenza media $p$ la formula sottostima i giri
  puliti e nasconde quanto sono gravi quelli sporchi. Autonomia per livelli
  (Greyling): L1 solo report → L2 fix assistiti con cancello umano, finché
  $c_{\text{rev}} < p_{\text{err}}\, c_{\text{err}}$ → L3 non presidiato
  entro allow-list, con $p_{\text{err}}$ stimato per eccesso (regola del tre).
- Onestà sui limiti: il comprehension debt {cite}`osmani2026comprehension`
  (i loop amplificano il giudizio
  buono *e* cattivo; chi mantiene deve leggere ciò che parte), la sicurezza
  (permessi minimi, cancelli umani ai punti irreversibili, come nel capitolo
  sull’AI responsabile) e il costo per giro, da mettere a budget
  e monitorare come insegna LLMOps. E lo statuto delle fonti: il vocabolario
  del loop engineering viene da chi costruisce, non da chi misura; qui si
  riporta il meccanismo, che dura, non i nomi degli attrezzi, che cambiano.
```

`````

Tre cerchi, uno dentro l'altro: la frase, il contesto che le sta intorno, il
ciclo che di frasi ne produce a ripetizione. E in mezzo un'idea che vale ben
oltre i modelli di linguaggio, cioè che chi propone non può essere anche chi
approva. Fin qui, però, a decidere è sempre stato uno solo: il programma che
fa partire il giro, il cancello che dice sì o no. Il capitolo sui
{doc}`sistemi multi-agente </SistemiMultiAgente/overview>` toglie anche questo
presupposto. Quando a decidere sono in molti, ciascuno con la sua finestra,
non basta più contare gli errori di ogni passo: mettersi d'accordo su che cosa
fare costa messaggi, attese e voti, e può sbagliare a modo suo.
