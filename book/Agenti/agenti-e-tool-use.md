# Ragionare e agire: il ciclo dell'agente

Chiedete a un modello di linguaggio che ore sono. Non lo sa. Chiedetegli di
moltiplicare $4831$ per $7092$: sputerà un numero dall'aria plausibile, spesso
sbagliato nelle cifre di mezzo. Chiedetegli cosa è successo ieri: vi parlerà
con sicurezza di un mondo che si è fermato alla fine del suo addestramento. Un
LLM, per quanto grande, è un cervello murato in una stanza senza finestre e
senza orologio: sa moltissimo di ciò che ha letto, nulla del resto, e i conti
lunghi li sbaglia come chiunque li faccia a mente.

E però può fare una cosa preziosa: *decidere* di chiedere aiuto. Invece di
inventare la risposta, può emettere una richiesta — «esegui questa
moltiplicazione», «apri questa pagina», «che ore sono?» — lasciare che
qualcos'altro la esegua, e usare il risultato. È il **tool use**, l'uso degli
strumenti: dare le mani a un cervello. Ed è il primo mattone di ciò che
chiamiamo **agente**: un modello che non si limita a rispondere, ma *osserva*,
*decide* e *agisce*, in un ciclo, finché il compito non è chiuso.

Costruiamo su terreno noto. Nel capitolo sui Transformer abbiamo visto un
modello che, dopo il pre-addestramento, *completa* il testo; che il
post-training — l'instruction tuning in particolare — lo ha reso capace di
*eseguire* una consegna invece di limitarsi a proseguirla; e che il prompt è
diventato un'interfaccia di programmazione in linguaggio naturale, potente e
fragile insieme. Un agente è il passo successivo: quello stesso modello, messo
dentro un anello di controllo, con strumenti a portata di mano e il permesso
di usarli.

## Dare le mani al modello: il tool use

L'idea tecnica ha un nome poco poetico — **function calling**, «chiamata di
funzione» — e un funzionamento sorprendentemente semplice. Diamo al modello,
insieme al prompt, un catalogo di strumenti disponibili: per ognuno un nome,
una descrizione a parole di cosa fa e la lista degli argomenti che accetta.
Quando il modello ritiene che serva uno strumento, non risponde con del testo
per l'utente: emette una **richiesta strutturata** — «chiama `calcola` con
argomento `"4831 * 7092"`». Il sistema che ospita il modello intercetta la
richiesta, esegue davvero la funzione, e restituisce il risultato al modello
come nuovo pezzo di contesto. Solo allora il modello continua.

`````{tab} Elementare

Immagina un dirigente competente ma con una regola personale: non fa mai i
conti a mano e non si fida della propria memoria per i dettagli. Sulla
scrivania ha una calcolatrice, un telefono e uno schedario. Quando gli chiedi
«quanto fa il totale della commessa?» non azzarda una cifra: prende la
calcolatrice. Quando gli chiedi «qual è l'indirizzo del cliente?» apre lo
schedario. La sua bravura non sta nel *sapere* tutto, ma nel **capire quale
attrezzo serve** e nell'usarlo bene.

Il tool use è esattamente questo. Al modello diamo un elenco di attrezzi, ognuno
con un'etichetta che dice a cosa serve: «calcolatrice — fa i conti esatti»,
«motore di ricerca — trova pagine aggiornate», «archivio — cerca un dato».
Quando arriva una domanda, il modello non prova a rispondere di pancia: sceglie
l'attrezzo giusto, scrive cosa infilarci dentro, aspetta il risultato e lo usa.
Un modello che sa dire «questo non lo so a memoria, ma so *chi* lo sa» è più
affidabile di uno che indovina sempre.

`````

`````{tab} Superiore

Uno strumento è descritto da uno **schema**: un nome, una descrizione in
linguaggio naturale e una firma tipata degli argomenti, tipicamente in JSON
Schema. Per una calcolatrice:

```json
{
  "name": "calcola",
  "description": "Valuta un'espressione aritmetica e ne restituisce il valore.",
  "parameters": {
    "espressione": {"type": "string", "description": "es. '4831 * 7092'"}
  }
}
```

La descrizione non è decorazione: è il testo su cui il modello ragiona per
decidere *se* e *quando* invocare lo strumento, ed è quindi parte del prompt a
tutti gli effetti. Il modello, invece di campionare token destinati
all'utente, emette una struttura `{"name": "calcola", "arguments":
{"espressione": "4831 * 7092"}}`; il runtime la valida contro lo schema, esegue
la funzione, e re-inietta il risultato nel contesto come messaggio di ruolo
*tool*. La capacità di scegliere lo strumento e **compilarne gli argomenti**
nel formato giusto non è innata: emerge dall'instruction tuning visto nel
capitolo sui Transformer, cioè dall'addestramento su esempi in cui a una
consegna corrisponde l'azione corretta invece della sua prosecuzione. Il
modello resta un generatore di testo: «chiamare uno strumento» è, sotto il
cofano, generare una particolare sequenza di token che il sistema ha imparato a
interpretare come una chiamata.

`````

C'è una domanda naturale: chi decide *dove*, in un testo, conviene fermarsi e
chiamare uno strumento? Nell'approccio appena descritto glielo insegniamo noi,
con esempi. Ma nel 2023 un gruppo di Meta AI ha mostrato che il modello può
impararlo **da solo**, senza che nessuno annoti a mano le chiamate: è
Toolformer {cite}`schick2023toolformer`.

`````{tab} Elementare

Come impara un bambino a usare la calcolatrice? Provando. Fa un conto a mente,
controlla con la calcolatrice, e nota che nei conti lunghi la calcolatrice ci
azzecca dove lui sbaglia — così, la volta dopo, per i conti lunghi la prende
subito. Toolformer fa qualcosa di simile con se stesso. Prende una montagna di
testo e, qua e là, prova a infilare una chiamata a uno strumento; poi guarda se
quella chiamata lo aiuta a **indovinare meglio le parole che vengono dopo**. Se
sì, tiene la chiamata come buon esempio; se no, la butta. Alla fine ha
fabbricato da solo un quaderno di esercizi — «qui conveniva la calcolatrice»,
«qui conveniva la ricerca» — e ci studia sopra. Nessun insegnante gli ha detto
dove mettere gli attrezzi: l'ha scoperto misurando quanto lo aiutavano.

`````

`````{tab} Superiore

Toolformer usa un'**auto-supervisione** elegante. Partendo da poche
dimostrazioni per ciascuna API (calcolatrice, sistema di domanda-risposta,
motore di ricerca, traduttore, calendario), il modello campiona in molte
posizioni di un corpus delle *candidate* chiamate ad API con i relativi
argomenti. Ogni candidata viene eseguita, e si tiene solo se il suo risultato,
inserito nel contesto, **riduce la loss** (la cross-entropia sul token
successivo) sui token che seguono, rispetto al non chiamare o a un risultato
inutile:

$$
L_i^{\text{con}} < L_i^{\text{senza}} - \tau,
$$

dove $L_i^{\text{con}}$ e $L_i^{\text{senza}}$ sono la perdita futura con e
senza la chiamata inserita in posizione $i$, e $\tau$ è una soglia di utilità.
Le chiamate che superano il filtro diventano un dataset aumentato, e il modello
ci viene messo a punto sopra con il consueto obiettivo auto-supervisionato. Il
risultato è un modello che, a inferenza, decide *da sé* quando emettere una
chiamata, perché ha imparato che in quei punti la chiamata paga in termini di
predizione. Il criterio è puramente interno — «l'attrezzo mi aiuta a
continuare il testo?» — e non richiede alcuna etichetta umana su dove usarlo.

`````

## Ragionare e agire insieme: ReAct

Uno strumento, da solo, non basta a fare un agente. Serve una **procedura**:
quando pensare, quando agire, come usare ciò che l'azione ha restituito. Il
pattern diventato lo standard di fatto si chiama **ReAct** — da *Reasoning +
Acting* — proposto da Shunyu Yao e colleghi {cite}`yao2023react`. L'idea è
intrecciare, in un unico flusso, tre tipi di passi: un **pensiero** (*Thought*,
il ragionamento ad alta voce), un'**azione** (*Action*, la chiamata a uno
strumento) e un'**osservazione** (*Observation*, il risultato che torna
indietro). Il modello genera un pensiero, poi un'azione; il sistema esegue e
restituisce l'osservazione; il modello legge l'osservazione, genera il pensiero
successivo, e così via, in un loop, fino a produrre la risposta finale.

Perché conviene far ragionare il modello *ad alta voce* tra un'azione e
l'altra? Perché è la stessa lezione della chain-of-thought incontrata nel
capitolo sui Transformer {cite}`wei2022chain`: scrivere i passaggi intermedi
prima di concludere riduce gli errori, perché ogni passo può appoggiarsi ai
precedenti invece di indovinare tutto in una volta. ReAct porta l'idea nel
mondo delle azioni: il pensiero decide *quale* strumento usare e *come*
interpretare ciò che è tornato, e l'osservazione àncora il pensiero successivo
a un fatto reale invece che a una fantasia.

`````{tab} Elementare

Pensa a un detective che indaga a voce alta. Non spara subito il colpevole:
alterna ragionamenti e verifiche. «*Penso*: la vittima è stata vista l'ultima
volta al porto, quindi conviene controllare i registri delle navi. *Controllo*
i registri… *Scopro* che quella notte è salpato un solo mercantile. *Penso*:
allora mi interessa chi era a bordo. *Controllo* la lista dell'equipaggio…».
Ogni «penso» decide la prossima mossa; ogni «controllo» è un'azione nel mondo;
ogni «scopro» è ciò che la mossa ha rivelato, e riparte il giro.

La forza del metodo sta nell'alternanza. Un detective che ragionasse soltanto,
senza mai controllare, costruirebbe teorie eleganti e magari sbagliate. Uno che
controllasse a caso, senza ragionare, si perderebbe tra mille indizi inutili.
ReAct fa fare al modello tutti e due i mestieri: pensa per decidere dove
guardare, guarda per correggere ciò che pensa.

`````

`````{tab} Superiore

Il contesto dell'agente cresce come una sequenza strutturata di terne:

```text
Thought: per rispondere mi serve l'anno del paper, non lo so a memoria.
Action: cerca[attention is all you need]
Observation: 2017
Thought: ora calcolo la differenza con il 2026.
Action: calcola[2026 - 2017]
Observation: 9
Thought: ho tutto.
Action: Answer[9 anni, dal 2017]
```

Ogni *Observation* è testo prodotto dall'esterno — non campionato dal modello —
e questo è il punto cruciale: àncora il ragionamento a fatti recuperati,
invece di lasciarlo derivare. Yao e colleghi mostrano che, sui compiti a forte
intensità di conoscenza come la domanda-risposta multi-hop (HotpotQA) e la
verifica di fatti (FEVER), affiancare le azioni di recupero al ragionamento
riduce le allucinazioni rispetto alla sola chain-of-thought, che ragiona bene
ma inventa i fatti su cui ragiona; e che, sui compiti interattivi come ALFWorld
(eseguire istruzioni in un ambiente simulato) e WebShop (navigare un sito per
acquistare), il ragionamento intercalato all'azione batte le politiche che
agiscono senza pensare. Il costo è in token e latenza — ogni pensiero è testo
generato in più — ma la traccia ha un effetto collaterale prezioso: è
**leggibile**, e permette a un umano di capire *perché* l'agente ha fatto una
certa mossa.

`````

## Imparare dai propri errori: la riflessione

Un agente ReAct, però, dentro un singolo tentativo non ha modo di *migliorare*:
se imbocca una strada sbagliata e fallisce, al tentativo dopo rischia di
ripetere lo stesso errore. Nel 2023 Noah Shinn e colleghi propongono un
rimedio semplice e umano — **Reflexion** {cite}`shinn2023reflexion`: dopo un
fallimento, l'agente si ferma e **scrive a parole cosa è andato storto**, poi
riprova tenendo quella critica sotto gli occhi.

`````{tab} Elementare

È ciò che fa un buon studente dopo un compito andato male. Non si limita a
riprovare identico: rilegge l'errore e se lo dice a parole — «ho sbagliato
perché ho applicato la formula prima di convertire le unità; la prossima volta
converto per prima cosa». Quella frase, appuntata a margine, alla prova
successiva vale più di mille esercizi ripetuti a testa bassa, perché indirizza
il tentativo nuovo lontano dallo stesso scoglio.

Reflexion dà all'agente questo quaderno di margine. Quando un tentativo
fallisce, il modello genera una piccola auto-critica in linguaggio naturale — la
sua «memoria verbale» degli errori — e la aggiunge al contesto del tentativo
seguente. Non cambia un solo peso della rete: cambia solo ciò che il modello
*legge* prima di riprovare. Eppure spesso basta, perché l'errore che prima era
invisibile ora è scritto nero su bianco all'inizio della pagina.

`````

`````{tab} Superiore

Shinn e colleghi chiamano il metodo *verbal reinforcement learning*: al posto
di aggiornare i parametri con un gradiente, il segnale di rinforzo è **testo**.
Il ciclo ha tre ruoli: un *attore* (il modello ReAct) che tenta il compito; un
*valutatore* che assegna un esito al tentativo (una ricompensa, il superamento
o meno di test, il raggiungimento dell'obiettivo); e un *modulo di
auto-riflessione* che, letta la traccia fallita e il suo esito, produce una
critica verbale — «l'azione X non ha dato il risultato atteso, conviene
provare Y». Questa critica finisce in una **memoria episodica** che viene
anteposta al contesto del tentativo successivo. Sul benchmark di programmazione
HumanEval, gli autori riportano che Reflexion su GPT-4 raggiunge un $91\%$ di
*pass@1*, contro l'$80\%$ del modello di base senza riflessione: iterare
sull'auto-critica, senza toccare i pesi, recupera una fetta consistente dei
casi inizialmente falliti.

`````

L'onestà impone però un distinguo netto, che questo libro deve al lettore.
L'auto-critica **non è** auto-correzione garantita. La riflessione funziona
bene quando esiste un segnale d'esito *affidabile e esterno* — i test unitari
che passano o falliscono, un risultato numerico verificabile, un obiettivo
raggiunto o no nell'ambiente: lì la critica ha un appiglio solido su cui
costruire. Quando invece l'unico giudice è il modello stesso, senza alcun
riscontro dal mondo, la faccenda si fa scivolosa: un modello convinto di una
risposta sbagliata tende a produrre auto-critiche che *confermano* l'errore, e
può perfino peggiorare una risposta che era corretta, «correggendola» verso il
falso. Riflettere aiuta a patto di avere qualcosa contro cui verificarsi; la
sola introspezione, da sé, non crea competenza che il modello non aveva.

## Un agente giocattolo, in Python

Mettiamo insieme i pezzi nel modo più spoglio possibile: un mini-agente ReAct
che gira davvero, in puro Python, senza rete né librerie esterne. Il trucco per
concentrarci sul *ciclo* è sostituire il modello vero con un **LLM finto
deterministico** — una funzione a regole che, guardando la traccia finora,
decide il prossimo `Thought` e la prossima `Action`. Gli strumenti, invece,
sono **veri**: una calcolatrice che valuta un'espressione aritmetica in modo
sicuro (niente `eval`: un piccolo interprete sugli operatori ammessi) e un
`cerca` che fa lookup in un archivio.

```python
import ast
import operator

# --- due strumenti reali ---

# operatori ammessi: un mini-interprete sicuro, niente eval()
_OP = {
    ast.Add: operator.add, ast.Sub: operator.sub,
    ast.Mult: operator.mul, ast.Div: operator.truediv,
    ast.USub: operator.neg,
}

def _valuta(nodo):
    if isinstance(nodo, ast.Constant):        # un numero
        return nodo.value
    if isinstance(nodo, ast.BinOp):           # a operatore b
        return _OP[type(nodo.op)](_valuta(nodo.left), _valuta(nodo.right))
    if isinstance(nodo, ast.UnaryOp):         # -a
        return _OP[type(nodo.op)](_valuta(nodo.operand))
    raise ValueError("espressione non ammessa")

def calcola(espressione):
    """Valuta un'espressione aritmetica in modo sicuro (senza eval)."""
    return _valuta(ast.parse(espressione, mode="eval").body)

# un piccolo archivio: la memoria esterna che il modello non ha nei pesi
ARCHIVIO = {
    "attention is all you need": "2017",
    "gpt-3": "2020",
    "react": "2023",
}

def cerca(chiave):
    """Cerca un fatto nell'archivio; restituisce sempre una stringa."""
    return ARCHIVIO.get(chiave.lower().strip(), "non trovato")

STRUMENTI = {"calcola": calcola, "cerca": cerca}
```

Il cuore dell'agente sono le altre due funzioni: l'`llm_finto`, che al posto di
un Transformer emette la coppia pensiero-azione a partire dallo stato, e il
loop `esegui_agente`, che alterna decisione ed esecuzione — la stessa struttura
di un agente reale, con l'unica differenza che qui il «modello» è una regola.

```python
# --- l'LLM finto: deterministico, a regole ---

def llm_finto(traccia):
    """Data la traccia finora, emette (pensiero, azione, argomento).
    Un vero LLM genererebbe questo testo; qui lo decide una regola."""
    ultima = traccia[-1]["osservazione"] if traccia else None
    if ultima is None:
        return ("Non conosco a memoria l'anno del paper: lo cerco.",
                "cerca", "attention is all you need")
    if ultima == "2017":
        return ("Il paper è del 2017. Calcolo quanti anni fa, dal 2026.",
                "calcola", "2026 - 2017")
    return (f"Il calcolo dice {ultima}: ho tutto per rispondere.",
            "Answer", "'Attention Is All You Need' è del 2017: 9 anni fa nel 2026.")

# --- il ciclo dell'agente ---

def esegui_agente(domanda, max_passi=5):
    print(f"Domanda: {domanda}\n")
    traccia = []
    for _ in range(max_passi):
        pensiero, azione, argomento = llm_finto(traccia)   # il modello "ragiona"
        print(f"Thought: {pensiero}")
        if azione == "Answer":                             # fine del loop
            print(f"Answer: {argomento}")
            return argomento
        print(f"Action: {azione}[{argomento}]")
        osservazione = str(STRUMENTI[azione](argomento))   # il sistema agisce
        print(f"Observation: {osservazione}\n")
        traccia.append({"azione": azione, "argomento": argomento,
                        "osservazione": osservazione})      # torna nel contesto
    print("(limite di passi raggiunto)")

esegui_agente("In che anno è uscito 'Attention Is All You Need' "
              "e quanti anni fa è, nel 2026?")
```

L'esecuzione stampa la traccia completa: si vedono i tre `Thought`, le due
chiamate agli strumenti con le rispettive `Observation`, e la risposta finale.

```text
Domanda: In che anno è uscito 'Attention Is All You Need' e quanti anni fa è, nel 2026?

Thought: Non conosco a memoria l'anno del paper: lo cerco.
Action: cerca[attention is all you need]
Observation: 2017

Thought: Il paper è del 2017. Calcolo quanti anni fa, dal 2026.
Action: calcola[2026 - 2017]
Observation: 9

Thought: Il calcolo dice 9: ho tutto per rispondere.
Answer: 'Attention Is All You Need' è del 2017: 9 anni fa nel 2026.
```

Il modello finto non sapeva l'anno (l'ha cercato) e non ha fatto la
sottrazione a mente (l'ha delegata): esattamente il comportamento che vogliamo
da un agente. Sostituite `llm_finto` con un vero LLM a cui passate, a ogni giro,
la traccia accumulata e il catalogo degli strumenti, e avete — nella sua ossatura
essenziale — lo stesso ciclo che muove gli assistenti capaci di navigare il web,
eseguire codice e interrogare un database. Tutto il resto, nei sistemi reali, è
robustezza: gestire le chiamate malformate, fermarsi quando l'agente entra in
loop, decidere quali strumenti sono sicuri da esporre.

```{admonition} Da ricordare
:class: important
- Un LLM da solo è murato: non sa l'ora, sbaglia i conti lunghi, ignora ciò
  che è successo dopo l'addestramento. Il **tool use** gli dà le mani: invece
  di rispondere, emette una **chiamata strutturata** a uno strumento, che il
  sistema esegue e il cui risultato rientra nel contesto.
- Ogni strumento è uno **schema** (nome, descrizione, argomenti tipati); la
  capacità di sceglierlo e compilarne gli argomenti emerge dall'instruction
  tuning. **Toolformer** {cite}`schick2023toolformer` impara *da solo*, con
  auto-supervisione, dove conviene chiamare un'API: tiene le chiamate che
  riducono la loss sui token successivi.
- **ReAct** {cite}`yao2023react` intreccia in un loop **Thought → Action →
  Observation**: ragionare ad alta voce (sulla scia della chain-of-thought
  {cite}`wei2022chain`) decide le azioni, e le osservazioni àncorano il
  ragionamento a fatti reali, riducendo le allucinazioni.
- **Reflexion** {cite}`shinn2023reflexion` aggiunge una **memoria verbale**
  degli errori: dopo un fallimento l'agente si auto-critica a parole e riprova
  leggendo la critica — senza toccare i pesi.
- Onestà sui limiti: l'auto-critica **non è** auto-correzione garantita. Aiuta
  quando c'è un esito esterno affidabile (test, risultato verificabile); con il
  solo giudizio del modello può confermare l'errore o peggiorare una risposta
  giusta.
- Il **ciclo dell'agente** — osserva, pensa, agisci, ripeti fino alla risposta
  — è la stessa ossatura del mini-agente in venti righe e degli assistenti che
  navigano il web ed eseguono codice; la differenza è la robustezza attorno.
```
