# Dal ciclo al grafo: il disegno di un lavoro che dura

Un harness, disegnato su un foglio, ha quasi sempre la forma di frecce fra
caselle. Quello di Young ha due caselle: il primo turno, e poi la sessione di
lavoro che torna su sé stessa finché l'elenco non è finito. Quello di
Rajasekaran ne ha tre: il pianificatore che scrive la descrizione del prodotto,
il costruttore che lavora una tappa alla volta, il valutatore che lo rimanda
indietro finché la tappa non passa.

Caselle e frecce, in matematica, formano un **grafo** (niente a che vedere con
i grafici delle funzioni): le caselle sono i **nodi**, le frecce gli **archi**.
Il disegno smette di essere un disegno quando le caselle diventano pezzi di
programma, le frecce regole scritte su quale casella viene dopo, e i dati del
lavoro, quelli che passano da una casella all'altra, uno stato salvato a ogni
passo.

## Percorsi scritti e percorsi scelti

Il {doc}`capitolo sugli agenti </Agenti/overview>` ha già distinto, con Erik
Schluntz e Barry Zhang, i *workflow* (i flussi di lavoro), in cui il modello e
gli strumenti seguono percorsi decisi in anticipo nel codice, dagli *agenti*,
in cui è il modello a scegliere da sé i passi e gli strumenti
{cite}`schluntz2024agents`. Qui «agente» ha questo senso stretto, cioè il
modello che sceglie; l'harness intorno resta.

Dei workflow l'articolo descrive cinque
schemi, ciascuno fatto di *chiamate*, cioè di richieste al modello. Nella
concatenazione l'uscita di una chiamata diventa l'ingresso della successiva,
con dei controlli programmati fra un passo e l'altro (i *cancelli*).
L'instradamento classifica la richiesta e la manda al percorso specializzato.
La parallelizzazione divide il compito in parti indipendenti, oppure fa lo
stesso compito più volte per confrontare i risultati. Nell'orchestratore con
esecutori un modello decide al momento, in base alla richiesta, in quali parti
dividere il compito, le affida ad altre chiamate e ne mette insieme i
risultati. Nel valutatore con ottimizzatore una chiamata produce e migliora la
risposta, un'altra la giudica e la rimanda indietro, in un ciclo. Il consiglio
dell'articolo è di cercare la soluzione più semplice possibile, e di aggiungere
complessità soltanto quando serve.

Sulle librerie di programmi che costruiscono questi schemi per conto tuo
l'articolo è prudente: rendono facile cominciare, ma mettono codice in mezzo
che nasconde che cosa si chiede al modello e che cosa risponde, e chi le usa
farebbe bene a capire il codice che c'è sotto.

Un harness per un lavoro lungo mescola le due famiglie. Ad esempio, la routine
del {doc}`passaggio di consegne <passaggio-di-consegne>` è un percorso scritto:
avvia, scegli la voce, lavora, collauda, salva, sempre in quest'ordine. Dentro
la casella «lavora», però, c'è un agente, che decide da sé quali file aprire e
quali comandi lanciare. Il ciclo fra costruttore e valutatore di Rajasekaran è
il valutatore con ottimizzatore, con il contratto di tappa fissato prima.
Quindi, più che scegliere fra percorso scritto e agente, [si tratta di decidere
dove passa il confine fra ciò che decide il codice e ciò che decide il
modello]{.pt-matita}.

`````{tab} Elementare

In bottega certi lavori seguono sempre la stessa strada: si taglia, si fora,
si incolla, si vernicia, e fra un banco e l'altro qualcuno guarda che il pezzo
sia a posto prima di passarlo avanti. Altri lavori no: un restauro comincia
smontando, e solo dopo si sa che cosa c'è da fare, quindi l'artigiano decide
lui, passo per passo. Il primo è un percorso scritto sul tabellone appeso al
muro; il secondo è un percorso scelto da chi lavora. Una strada scritta si può
provare prima, banco per banco, e si sa quanto costa al massimo; una strada
scelta si scopre solo facendola.

Il tabellone ha qualche disegno che torna sempre. Una fila di banchi, con un
controllo fra l'uno e l'altro: più banchi ci sono in fila, più occasioni ha il
pezzo di rovinarsi, e il controllo non impedisce l'errore, ma lo fa vedere
subito, quando si può ancora rifare. Un bivio all'ingresso, che manda le sedie
a un banco e i tavoli a un altro. Quattro artigiani che fanno insieme le
quattro gambe di un tavolo, oppure tre che giudicano la stessa vernice e
tengono il giudizio della maggioranza, che però serve poco se l'hanno guardata
tutti sotto la stessa luce sbagliata, e sbagliano insieme. Un capobottega che,
davanti a un ordine grande, decide lui come dividerlo e a chi darne i pezzi. E
l'artigiano e il collaudatore che si rimandano il pezzo finché non va.

Quasi sempre, poi, le due cose stanno insieme: il tabellone dice la strada, e
dentro ogni banco l'artigiano decide come lavorare. Per uno sgabello il
tabellone non si disegna nemmeno, perché ogni freccia in più è una regola da
tenere e un posto in cui sbagliare; e un tabellone comprato già fatto, con
frecce dappertutto, nasconde proprio quello che succede ai banchi.

`````

`````{tab} Superiore

Un workflow è un grafo $\mathcal{G} = (\mathcal{U}, \mathcal{A})$ di nodi e
archi, con una funzione di transizione $\delta(u, S)$ che sceglie il nodo
successivo fra quelli raggiunti da un arco, $(u, \delta(u, S)) \in
\mathcal{A}$, o la fine. Un agente è il caso limite di un solo nodo che si
ripete, in cui l'azione e la decisione di fermarsi sono campionate dal
modello, $a_{k,t} \sim P_\theta(\cdot \mid C_{k,t})$, come nel ciclo
dell’{doc}`anatomia dell'harness <overview>`. In tutti e due i casi $\delta$ è
codice; a distinguerli è chi scrive i campi che $\delta$ legge (il codice, o il
modello, come nell'instradamento, che è un classificatore sugli ingressi, o
nell'agente, che scrive «fermati») e quanto è aperto lo spazio delle azioni.

La contropartita si misura. In un workflow aciclico, o con cicli che hanno un
tetto, i percorsi si possono elencare, anche se crescono in modo esponenziale
con gli archi condizionali; si possono provare uno per uno, e il numero di
chiamate è limitato, finché i nodi non contengono a loro volta un agente. In un
agente il limite è soltanto il tetto ai passi del ciclo, e i percorsi fino al
tetto non si elencano. In entrambi i casi i nodi che chiamano un modello sono
aleatori, e l'affidabilità si stima campionando.

Che una catena di passi riesca con il prodotto delle probabilità dei passi, se
sono indipendenti, e che un cancello trasformi un errore silenzioso in uno
rilevato, che si può ritentare, è il calcolo del {doc}`loop engineering
</IngegneriaLLM/loop-engineering>`.

Il voto fra $m$ esecuzioni, con due esiti possibili, $m$ dispari e la stessa
probabilità $q > 1/2$ di essere giuste, alza la probabilità che la maggioranza
abbia ragione se gli errori sono indipendenti (il teorema di Condorcet, che il
{doc}`capitolo sui sistemi multi-agente
</SistemiMultiAgente/protocolli-e-consenso>` tratta per esteso); con errori
correlati può persino andare peggio di una esecuzione sola, e le
esecuzioni dello stesso modello sullo stesso ingresso tendono a sbagliare
insieme.

La regola di cominciare dal sistema più semplice si legge così: ogni
nodo e ogni arco in più sono un'ipotesi su che cosa il modello non sa fare da
solo, come ogni componente dell'harness, e un posto in più in cui il sistema
può rompersi.

`````

## Il grafo come struttura dati

Per un lavoro lungo il grafo ha quattro ingredienti. I nodi sono funzioni che
leggono lo stato e restituiscono un aggiornamento, non lo stato intero. Un nodo
tocca soltanto i campi di cui è responsabile, e non può rimettere per sbaglio
un valore vecchio negli altri. Gli archi sono fissi o condizionali. Un arco
condizionale guarda lo stato e sceglie il nodo successivo, ed è così che si
scrive un ciclo, come il ritorno dal collaudo alla costruzione, con un tetto ai
tentativi perché finisca.

Lo stato è un insieme di campi, e ogni campo ha la sua regola di fusione, cioè
il modo in cui l'aggiornamento di un nodo si unisce a ciò che c'era: il
registro si accoda, un contatore si somma, un esito si sostituisce. Senza
regola, infatti, un nodo che restituisce la sua riga di registro cancellerebbe
tutte le righe di prima.

Il quarto ingrediente è il **punto salvato**: dopo ogni passo lo stato, insieme
al nodo da cui ripartire, si scrive in un archivio, sotto l'identificativo del
**filo di lavoro** (in inglese *thread*), cioè di quella singola esecuzione del
grafo; ne possono girare molti insieme, ciascuno con il suo stato. Così un
guasto non costringe a ricominciare da capo, e un filo che deve fermarsi ore ad
aspettare una persona non perde niente nel frattempo.

Un nodo può essere un passo di pochi secondi o una sessione intera; la
granularità dipende da quanto costa rifare un nodo interrotto.

Sono le idee su cui si reggono le librerie per costruire agenti come grafi, che
permettono anche rami paralleli, cioè più nodi attivi nello stesso passo. In
LangGraph, per esempio, un *checkpointer* salva lo stato del grafo a ogni passo
sotto un identificativo di thread, e da lì il grafo riprende dopo
un'interruzione o dopo un guasto; se in un passo alcuni nodi falliscono e altri
no, le scritture di quelli riusciti si conservano, e alla ripresa non si
rifanno {cite}`langgraph2026checkpointers`.

Un motore minimo, in Python puro,
fa vedere tutto questo su un lavoro di quattro nodi (pianifica, costruisci,
collauda, rilascia), con un ciclo fra collaudo e costruzione e il rilascio ai
clienti segnato come irreversibile. Il guasto è simulato. Al primo passaggio
dal collaudo il nodo si interrompe, come farebbe una macchina che si spegne.
Ripreso, il collaudo fallisce, si torna alla costruzione, e il secondo
collaudo passa; a quel punto il filo si ferma davanti al rilascio, che senza
un'approvazione non parte. Il motore lancia quindi il filo tre volte, come
farebbe chi lo rimette in moto dopo il guasto e poi dopo la firma, e stampa che
cosa succede a ogni corsa e il registro alla fine.

```python
# Un grafo minimo per un lavoro lungo. I nodi leggono lo stato e restituiscono
# un aggiornamento; ogni campo dello stato ha la sua regola di fusione.
FUSIONE = {
    "registro": lambda vecchio, nuovo: vecchio + nuovo,    # si accoda
    "tentativi": lambda vecchio, nuovo: vecchio + nuovo,   # si somma
    "esito": lambda vecchio, nuovo: nuovo,                 # si sostituisce
}

def fondi(stato, aggiornamento):
    nuovo = dict(stato)
    for campo, valore in aggiornamento.items():
        nuovo[campo] = FUSIONE[campo](stato[campo], valore)
    return nuovo

def pianifica(stato):
    return {"registro": ["piano: una voce, il totale del carrello"]}

def costruisci(stato):
    tentativo = stato["tentativi"] + 1
    return {"registro": [f"tentativo {tentativo} costruito"], "tentativi": 1}

guasti = {"collauda": 1}             # il primo collaudo si interrompe

def collauda(stato):
    if guasti["collauda"]:
        guasti["collauda"] -= 1
        raise RuntimeError("la macchina si è spenta")     # un guasto simulato
    tentativo = stato["tentativi"]
    esito = "passato" if tentativo >= 2 else "fallito"
    riga = f"collaudo del tentativo {tentativo}: {esito}"
    return {"esito": esito, "registro": [riga]}

def rilascia(stato):
    return {"registro": ["rilasciato ai clienti"]}

NODI = {"pianifica": pianifica, "costruisci": costruisci,
        "collauda": collauda, "rilascia": rilascia}
ARCHI = {"pianifica": "costruisci", "costruisci": "collauda", "rilascia": None}
TETTO = 3                            # tentativi al massimo, poi ci si ferma

def dopo_il_collaudo(stato):         # l'unico arco che dipende dallo stato
    if stato["esito"] == "passato":
        return "rilascia"
    return "costruisci" if stato["tentativi"] < TETTO else None

CONDIZIONALI = {"collauda": dopo_il_collaudo}
IRREVERSIBILI = {"rilascia"}         # prima di questi serve una persona

# I punti salvati, uno per filo di lavoro. Qui è un dizionario; in un harness
# vero è un file o un database, scritto prima di passare al nodo dopo.
archivio = {}

def esegui(filo, approvati=()):
    """Riprende il filo dall'ultimo punto salvato e va avanti finché può.
    `approvati` sono i nodi irreversibili che una persona ha approvato."""
    inizio = {"registro": [], "tentativi": 0, "esito": None}
    nodo, stato = archivio.get(filo, ("pianifica", inizio))
    fatti = []
    while nodo is not None:
        if nodo in IRREVERSIBILI:
            if nodo not in approvati:
                return fatti, f"fermo prima di «{nodo}»: serve un'approvazione"
            stato = fondi(stato, {"registro": [f"«{nodo}» approvato"]})
        try:
            aggiornamento = NODI[nodo](stato)
        except RuntimeError as guasto:
            return fatti, f"guasto in «{nodo}»: {guasto}"
        stato = fondi(stato, aggiornamento)
        esito = aggiornamento.get("esito")
        fatti.append(f"{nodo} ({esito})" if esito else nodo)
        if nodo in CONDIZIONALI:
            nodo = CONDIZIONALI[nodo](stato)
        else:
            nodo = ARCHI[nodo]
        archivio[filo] = (nodo, stato)        # salvato dopo ogni passo
    if stato["esito"] == "fallito":           # uscito dal ciclo per il tetto
        return fatti, f"fermo dopo {TETTO} tentativi: serve una persona"
    return fatti, "fine"

for corsa, approvati in [("prima corsa", ()), ("ripresa", ()),
                        ("dopo l'approvazione", ("rilascia",))]:
    fatti, motivo = esegui("ordine-17", approvati)
    print(f"{corsa}: {' -> '.join(fatti)}")
    print(f"   {motivo}")
    if corsa == "prima corsa":
        nodo, stato = archivio["ordine-17"]
        print(f"   punto salvato: si riparte da «{nodo}», "
              f"tentativi costruiti: {stato['tentativi']}")
print("registro del filo:")
for riga in archivio["ordine-17"][1]["registro"]:
    print(f"   {riga}")
```

```text
prima corsa: pianifica -> costruisci
   guasto in «collauda»: la macchina si è spenta
   punto salvato: si riparte da «collauda», tentativi costruiti: 1
ripresa: collauda (fallito) -> costruisci -> collauda (passato)
   fermo prima di «rilascia»: serve un'approvazione
dopo l'approvazione: rilascia
   fine
registro del filo:
   piano: una voce, il totale del carrello
   tentativo 1 costruito
   collaudo del tentativo 1: fallito
   tentativo 2 costruito
   collaudo del tentativo 2: passato
   «rilascia» approvato
   rilasciato ai clienti
```

Si noti che dopo il guasto il filo non riparte da capo: l'ultimo punto salvato
dice «ripartire dal collaudo, con un tentativo già costruito», e pianificazione
e costruzione non si rifanno. Il collaudo, però, si rifà, perché si era
interrotto proprio lì: un passo interrotto a metà si esegue di nuovo per intero.

Il ciclo fra collaudo e costruzione si chiude al secondo tentativo, e
il filo si ferma da solo davanti al rilascio, finché qualcuno non approva
proprio quel nodo; l'approvazione finisce nel registro. Il guasto, invece, nel
registro non c'è: è successo fra due punti salvati, e lo vede soltanto chi
guarda il giornale degli eventi della {doc}`sezione sull'osservabilità
<quando-nessuno-guarda>`. La {numref}`fig-harness-grafo` mostra lo stesso filo
che attraversa il grafo.

```{figure} ../figures/harness-grafo.svg
:name: fig-harness-grafo
:alt: "Un grafo di quattro nodi in fila, pianifica, costruisci, collauda e rilascia, con un arco che dal collaudo torna alla costruzione, al massimo 3 tentativi, e una firma sull'arco che porta al rilascio, segnato come irreversibile. Un gettone percorre il grafo: pianifica, costruisci, poi al collaudo si spegne (una croce resta sul nodo); riparte dal collaudo, che fallisce, torna alla costruzione, il secondo collaudo passa, si ferma alla firma e infine arriva al rilascio. Sotto, una fila di 8 caselle riporta i passi nell'ordine: 6 passi salvati, il guasto e la firma."
:width: 95%

Il filo di lavoro attraversa il grafo: dopo il guasto nel collaudo riparte
dall'ultimo punto salvato, non da capo, torna una volta alla costruzione, e
dopo il collaudo passato si ferma alla firma finché qualcuno non approva il
rilascio. Sotto, i passi nell'ordine in cui il filo li ha fatti.
```

`````{tab} Elementare

Sul muro della bottega c'è il tabellone: il tabellone è il grafo, ogni banco un
nodo, ogni freccia un arco. Una freccia, quella che esce dal collaudo, ha un
cartello: «se il pezzo non passa, torna alla costruzione; dopo tre volte, ci si
ferma e si chiama il proprietario». Così il pezzo non gira in tondo per sempre.

Ogni ordine porta con sé una scheda di stato, e ogni banco ci scrive soltanto
nelle colonne che lo riguardano, secondo la regola della colonna: nelle note si
aggiunge una riga in fondo, nei tentativi si aggiunge uno al numero che c'è,
nell'esito si cancella e si scrive quello nuovo. Così nessun banco cancella le
righe scritte dagli altri; e l'esito, che si riscrive, lo riscrive un banco
alla volta. Dopo ogni banco si fotografa la scheda, con scritto da quale banco
si riparte. Non è la fotografia della bottega che si scatta la sera, quella
del salvataggio nella storia: è una foto piccola, della sola scheda, dopo ogni
banco.

Una sera salta la corrente mentre l'ordine è al collaudo. [La mattina nessuno
ricomincia dal disegno]{.pt-matita}: si prende l'ultima foto, che dice
«riprendere dal collaudo, un tentativo già costruito», e si riparte da lì. Il
collaudo interrotto si rifà da capo, però, perché nessuno sa a che punto fosse
arrivato quando la luce è andata via. E la scheda può sbagliare in un modo
solo: se la luce salta mentre il camion parte, l'ultima foto è di prima, e la
scheda dice «da spedire». Per questo chi spedisce scrive sulla scheda il numero
della spedizione prima di chiamare il corriere, e la mattina, invece di
rispedire, telefona e chiede se la spedizione con quel numero è già partita; il
corriere, dal canto suo, non accetta due ritiri con lo stesso numero.

Il banco della spedizione ha anche un'altra regola, che non c'entra con i
guasti: lì l'ordine si ferma e aspetta la firma del proprietario, perché un
mobile spedito non torna indietro. Il proprietario non firma tutto, non ne
avrebbe il tempo: firma dove uno sbaglio costerebbe più del tempo che gli
serve per guardare, e quasi sempre è dove lo sbaglio non si rimedia.

Infine, il collaudatore lavora a un banco suo, con davanti il mobile e il
foglio delle prove concordate, ma con la colonna delle note di chi l'ha
costruito coperta: così non eredita i suoi ragionamenti. Le idee che ha in
comune con lui, perché vengono dalla stessa scuola, se le porta dietro lo
stesso.

`````

`````{tab} Superiore

Fissiamo la semantica del passo e quella della ripresa. Lo stato è un
insieme di campi, $S = (S^{(1)}, \dots, S^{(J)})$, e ogni campo ha
un'operazione di fusione $\mathrm{fondi}_j$; un nodo $u$ restituisce un
aggiornamento parziale $\mathrm{agg}_u(S)$, un valore per una parte dei campi,
e il passo è

$$
S_{t+1} = \mathrm{fondi}\big(S_t,\, \mathrm{agg}_{u_t}(S_t)\big), \qquad
u_{t+1} = \delta(u_t, S_{t+1}),
$$

con la fusione applicata campo per campo; dopo il passo si scrive il punto
salvato $(u_{t+1}, S_{t+1})$ sotto l'identificativo del filo.

Con i rami paralleli il passo ha un insieme di nodi attivi $\mathcal{U}_t$, gli
aggiornamenti di tutti si fondono, e $\delta$ restituisce l'insieme dei
successori: è il *super-step* di LangGraph. Perché il risultato non dipenda
dall'ordine di arrivo basta che gli aggiornamenti commutino; succede se
$\mathrm{fondi}_j$ è associativa e commutativa, come la somma. La
concatenazione è associativa ma non commutativa, e i rami si fondono in un
ordine fissato; la sostituzione, con due rami che scrivono lo stesso campo,
non si ammette, e LangGraph la rifiuta.

[La ripresa dopo un guasto riparte dall'ultimo punto salvato]{.pt-matita}, e dà
al nodo in corso una semantica **almeno una volta**: se si è interrotto dopo
aver prodotto un effetto esterno ma prima che il punto fosse scritto, alla
ripresa lo produce di nuovo. Vale a una condizione, che il punto si scriva in
un archivio durevole e in modo sincrono, prima di passare al nodo dopo: un
dizionario in memoria, come quello del motore, sparisce con il processo.

Una semantica **esattamente una volta** per gli effetti esterni chiede qualcosa
in più dal lato di chi li riceve. O nodi idempotenti, la proprietà che si
chiede agli {doc}`strumenti
<gli-strumenti>`, cioè tali che eseguirli due volte equivalga a eseguirli una; o
una chiave d'idempotenza derivata da filo e passo, che il servizio esterno usa
per riconoscere la ripetizione (e che conserva per un tempo limitato, quindi
una ripresa molto tardiva la può superare); o un'intenzione con un
identificativo, scritta nello stato prima dell'effetto, che alla ripresa si
confronta con lo stato del servizio esterno. Osserviamo che l'intenzione da
sola dice soltanto che l'effetto forse c'è stato.

C'è poi l'ambiente: il nodo che lavora sui file cambia il
disco, che il punto salvato non contiene, quindi alla ripresa lo stato è quello
di prima e i file no, ed è un'altra ragione per riportarli al commit dell'ultimo
punto, come nel passaggio di consegne.

La terminazione di un grafo con cicli si garantisce con una variante intera,
limitata dal basso, che decresce a ogni iterazione e che i nodi non possono
scrivere, qui $\text{TETTO} - \text{tentativi}$, come il tetto del ciclo del
loop engineering; non copre le riprese, perché un guasto che si ripete sempre
fa ripartire il filo senza che la variante scenda, e serve un tetto anche ai
tentativi per nodo.

Un nodo irreversibile si comporta come un nodo il cui ingresso arriva da fuori:
il grafo scrive il punto salvato, si ferma, e riprende quando l'approvazione di
quel nodo, in quel filo, c'è. In altri termini, è la persona come nodo, con una
latenza e una capacità limitate.

Nel progetto di Lopopolo, ad esempio, man mano che il codice cresceva,
il collo di bottiglia è diventato proprio la capacità delle persone di
controllare {cite}`lopopolo2026harness`, e le interruzioni vanno dove conviene
revisionare: la condizione del loop engineering, $c_{\text{rev}} <
p_{\text{err}}\, c_{\text{err}}$, con $c_{\text{err}}$ il danno dell'errore
lasciato passare, tutto il danno se l'effetto non si disfa; e con una capacità
finita, nell'ordine di $p_{\text{err}}\, c_{\text{err}} / c_{\text{rev}}$.

Infine, un nodo di verifica con un contesto proprio, montato con le istruzioni
e il catalogo degli strumenti del valutatore, l'artefatto e il contratto,
$C_{\text{val}} = \mathrm{monta}(I_{\text{val}}, \mathcal{T}_{\text{val}},
\text{artefatto}, \text{contratto})$, non dipende dalla cronologia di chi ha
costruito: è l'isolamento del {doc}`context engineering
</IngegneriaLLM/context-engineering>`. Le convinzioni che passano per altre
vie restano: dai pesi, se il modello è lo stesso, dall'artefatto stesso, e dal
contratto, che in Rajasekaran propone chi costruisce.

`````

Il grafo non rende il modello più capace: dà un posto, in un disegno che si può
leggere, provare e riprendere dopo un guasto, a ciò che sta attorno al modello.
Si noti che in un harness lo stato del grafo non sostituisce lo stato
persistente delle sessioni: il registro e gli esiti ne sono campi, il commit è
un effetto esterno, e come tale va trattato alla ripresa.

`````{tab} Elementare

```{admonition} Da ricordare
:class: important
- Certi lavori seguono una strada scritta sul tabellone (il grafo), altri la
  decide l'artigiano passo per passo (l'agente); quasi sempre le due cose
  stanno insieme. Il tabellone più semplice che basta è quello giusto.
- Sul tabellone una freccia può tornare indietro (dal collaudo alla
  costruzione), ma con un limite ai giri, dopo il quale si chiama una persona.
- La scheda di stato ha una regola per colonna (aggiungere, sommare,
  sostituire), ogni banco scrive solo nelle sue colonne, e dopo ogni banco la
  si fotografa: dopo un guasto si riparte dall'ultima foto, il punto salvato,
  non da capo.
- Il banco interrotto si rifà da capo: i lavori che non si possono fare due
  volte devono chiedere a chi li riceve se sono già stati fatti.
- Prima di ciò che non si può disfare, come spedire, serve la firma per quel
  banco; e la firma va dove uno sbaglio costerebbe più del tempo di chi firma.
- Il collaudatore lavora a un banco suo, senza gli appunti di chi ha
  costruito; ma le idee della stessa scuola se le porta dietro.
```

`````

`````{tab} Superiore

```{admonition} Da ricordare
:class: important
- Workflow e agenti {cite}`schluntz2024agents`: la transizione è sempre
  codice, cambia chi scrive i campi che legge. Ogni nodo in più è
  un'ipotesi su che cosa il modello non sa fare da solo.
- Il grafo: nodi come aggiornamenti parziali, archi fissi o condizionali,
  una fusione per campo (con rami paralleli, aggiornamenti che commutano),
  punto salvato dopo ogni passo sotto un identificativo di filo
  {cite}`langgraph2026checkpointers`.
- La ripresa dà al nodo interrotto una semantica almeno una volta, se il punto
  si scrive in modo durevole e sincrono; l'esattamente una volta chiede
  idempotenza, una chiave d'idempotenza o un'intenzione confrontata con il
  servizio esterno.
- La terminazione si garantisce con una variante fuori dalla portata dei
  nodi, più un tetto alle riprese; i nodi irreversibili aspettano
  un'approvazione legata a quel nodo, dove $c_{\text{rev}} < p_{\text{err}}\,
  c_{\text{err}}$.
- Un nodo di verifica con contesto proprio non eredita la cronologia di chi ha
  costruito; le convinzioni che passano dai pesi, dall'artefatto e dal
  contratto restano.
```

`````

Ogni pezzo dell'harness, dalle istruzioni al grafo, l'ha scritto e misurato
finora una persona. Può scriverlo anche un programma, che lo prova e lo cambia:
è {doc}`l'harness che si riscrive <l-harness-che-si-riscrive>`.
