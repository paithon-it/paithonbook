# Ragionare e agire: il ciclo dell'agente

Chiedete a un modello di linguaggio che ore sono. Non lo sa. Chiedetegli di
moltiplicare $4831$ per $7092$: sputerà un numero dall'aria plausibile, spesso
sbagliato nelle cifre di mezzo. Chiedetegli cosa è successo ieri: vi parlerà
con sicurezza di un mondo che si è fermato alla fine del suo addestramento. Un
LLM, per quanto grande, è un cervello murato in una stanza senza finestre e
senza orologio: sa moltissimo di ciò che ha letto, nulla del resto, e i conti
lunghi li sbaglia come chiunque li faccia a mente.

E però può fare una cosa preziosa: *decidere* di chiedere aiuto. Invece di
inventare la risposta, può emettere una richiesta («esegui questa
moltiplicazione», «apri questa pagina», «che ore sono?») lasciare che
qualcos'altro la esegua, e usare il risultato. È il **tool use**, l'uso degli
strumenti: dare le mani a un cervello. Ed è il primo mattone di ciò che
chiamiamo **agente**: un modello che non si limita a rispondere, ma *osserva*,
*decide* e *agisce*, in un ciclo, finché il compito non è chiuso.

Costruiamo su terreno noto. Nel capitolo sui Transformer abbiamo visto un
modello che, letta una montagna di testo, impara a *completarlo*; che una
seconda fase di addestramento su esempi di consegne svolte lo ha reso capace
di *eseguire* una richiesta invece di limitarsi a proseguirla; e che il
**prompt** (il foglietto di istruzioni che si scrive al modello prima di
lasciarlo rispondere) è diventato il modo in cui gli si dice cosa fare:
potente e fragile insieme, perché una parola diversa cambia il risultato. Un
agente è il passo successivo: quello stesso modello, messo dentro un anello di
controllo, con strumenti a portata di mano e il permesso di usarli.

## Dare le mani al modello: il tool use

L'idea tecnica ha un nome poco poetico (**function calling**, «chiamata di
funzione») e un funzionamento sorprendentemente semplice. Diamo al modello,
insieme al prompt, un catalogo di strumenti disponibili: per ognuno un nome,
una descrizione a parole di cosa fa e la lista di cosa bisogna infilarci
dentro perché funzioni, che in gergo si chiamano gli **argomenti** (per una
calcolatrice, il conto da fare; per una ricerca, le parole da cercare: niente
a che vedere con gli argomenti di cui si discute).
Quando il modello ritiene che serva uno strumento, non risponde con del testo
per l'utente: emette una **richiesta strutturata** («chiama `calcola` con
argomento `"4831 * 7092"`»). Il sistema che ospita il modello intercetta la
richiesta, esegue davvero la funzione, e restituisce il risultato al modello
come nuovo pezzo di contesto. Solo allora il modello continua.

```{figure} ../figures/function-calling-llm-strumenti.svg
:name: fig-function-calling
:alt: "Schema del function calling in tre passi: l'utente chiede «che tempo fa?», il modello decide se e quale strumento usare ed emette una richiesta tool_use con la funzione get_weather e l'argomento Bologna; il codice dell'applicazione esegue la funzione e restituisce un tool_result con «18 gradi, sereno»; il modello produce infine la risposta in linguaggio naturale. Il giro fra richiesta e risultato può ripetersi più volte."
:width: 90%

Il giro del function calling. Il modello non esegue mai niente: chiede, e
l'esecuzione resta nel codice di chi lo ospita. I passi 1 e 2 possono
ripetersi più volte prima che arrivi la risposta finale.
```

Le due etichette al centro di {numref}`fig-function-calling` sono i nomi che
si usano in gergo per i due messaggi: `tool_use` è «chiedo di usare questo
attrezzo, con dentro queste cose», `tool_result` è «ecco cosa ha risposto
l'attrezzo». La divisione dei compiti che si vede nel disegno è la
ragione per cui il tool use è insieme potente e governabile: il modello
propone, il codice dispone. Chi ospita il modello decide quali funzioni
esistono, le valida prima di eseguirle e può rifiutarsi; il modello non ha mai
in mano l'esecuzione, solo la richiesta.

Scrivere a mano il catalogo, sistema per sistema, funziona finché i sistemi
sono due o tre. Da qui nasce l'idea di un **protocollo comune**, cioè di una
lingua unica con cui chiedere a qualunque sistema esterno «che strumenti hai?»
e «esegui questo». Quello che si è diffuso si chiama **MCP** (*Model Context
Protocol*, «protocollo per il contesto del modello»), e la sua architettura è
in {numref}`fig-mcp`.

```{figure} ../figures/mcp-spiegato.svg
:name: fig-mcp
:alt: "Architettura a tre livelli: un host, cioè l'applicazione che contiene il modello, tiene al proprio interno due client; ciascun client parla, tramite un protocollo comune, con un server distinto; ogni server espone verso ciò che gestisce le proprie primitive: strumenti e risorse nel primo caso, che governa dei file, strumenti e prompt nel secondo, che governa un archivio di dati."
:width: 100%

Lo stesso catalogo, ma standardizzato. A parlare il protocollo non è il
modello: è l'applicazione che lo ospita, la quale apre un canale verso ogni
sistema esterno e li interroga tutti allo stesso modo. Al modello arrivano poi
strumenti come gli altri, senza che debba sapere da dove vengono.
```

Il salto di {numref}`fig-mcp` rispetto al catalogo scritto a mano è di scala,
non di meccanismo: sotto resta il giro appena descritto, e il modello continua
a vedere solo un elenco di attrezzi con le loro etichette. Cambia chi scrive
quelle etichette, cioè le descrizioni degli strumenti, che diventano
responsabilità di chi espone il sistema invece che di chi costruisce l'agente.

`````{tab} Elementare

Immagina un dirigente competente ma con una regola personale: non fa mai i
conti a mano e non si fida della propria memoria per i dettagli. Sulla
scrivania ha una calcolatrice, un telefono e uno schedario. Quando gli chiedi
«quanto fa il totale della commessa?» non azzarda una cifra: prende la
calcolatrice. Quando gli chiedi «qual è l'indirizzo del cliente?» apre lo
schedario. La sua bravura non sta nel *sapere* tutto, ma nel **capire quale
attrezzo serve** e nell'usarlo bene.

Il tool use è esattamente questo. Al modello diamo un elenco di attrezzi,
ognuno con un'etichetta che dice a cosa serve: «calcolatrice: fa i conti
esatti», «motore di ricerca: trova pagine aggiornate», «archivio: cerca un
dato». Quando arriva una domanda, il modello non prova a rispondere di
pancia: sceglie l'attrezzo giusto, scrive cosa infilarci dentro, aspetta il
risultato e lo usa. Un modello che sa dire «questo non lo so a memoria, ma so
*chi* lo sa» è più affidabile di uno che indovina sempre.

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
    "type": "object",
    "properties": {
      "espressione": {"type": "string", "description": "es. '4831 * 7092'"}
    },
    "required": ["espressione"]
  }
}
```

Le tre righe che avvolgono il parametro non sono cerimoniale: `type`,
`properties` e `required` sono ciò che rende quel blocco un JSON Schema
valido, e senza di esse ogni interfaccia reale lo rifiuta. Il nome della
chiave che lo contiene invece cambia da un fornitore all'altro
(`parameters`, `input_schema`, `inputSchema`); la forma dello schema no, ed è
quella che vale la pena ricordare.

La descrizione non è decorazione: è il testo su cui il modello ragiona per
decidere *se* e *quando* invocare lo strumento, ed è quindi parte del prompt a
tutti gli effetti. Il modello, invece di campionare token destinati
all'utente, emette una struttura `{"name": "calcola", "arguments":
{"espressione": "4831 * 7092"}}`; il runtime la valida contro lo schema, esegue
la funzione, e re-inietta il risultato nel contesto come messaggio di ruolo
*tool*. Anche qui i nomi esatti cambiano da un fornitore all'altro
(`tool_use`/`tool_result` per gli uni, come nella figura, `function_call` e
ruolo `tool` per gli altri, che per giunta passano gli `arguments` come
stringa JSON invece che come oggetto); il giro è lo stesso. La capacità di
scegliere lo strumento e **compilarne gli argomenti**
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

```{figure} ../figures/toolformer-2023.svg
:name: fig-toolformer
:alt: "Schema di Toolformer: mentre genera la frase «400 su 1400, cioè il…» il modello arriva a un punto di decisione, chiamo un tool? Se la risposta è no continua a scrivere la parola successiva; se è sì emette Calculator(400/1400), lo strumento esterno calcola 0.29 e il risultato rientra nella frase, che riprende come «400 su 1400, cioè il 29%»."
:width: 88%

Toolformer decide *dentro* la frase. Al punto di decisione il modello può
proseguire normalmente oppure inserire una chiamata: il risultato torna nel
testo e la generazione riparte da lì, come se il numero l'avesse scritto lui.
```

Il dettaglio da guardare in {numref}`fig-toolformer` è dove sta la chiamata:
non prima o dopo il testo, ma **dentro**, in mezzo a una parola e l'altra. È
questo che rende sensato il criterio di apprendimento che segue: se
l'inserzione è interna alla frase, si può misurare se aiuta a scrivere il
seguito.

`````{tab} Elementare

Come impara un bambino a usare la calcolatrice? Provando. Fa un conto a mente,
controlla con la calcolatrice, e nota che nei conti lunghi la calcolatrice ci
azzecca dove lui sbaglia: così, la volta dopo, per i conti lunghi la prende
subito. Toolformer fa qualcosa di simile con se stesso. Prende una montagna di
testo e, qua e là, prova a infilare una chiamata a uno strumento; poi guarda
se quella chiamata lo aiuta a **indovinare meglio le parole che vengono
dopo**. Se sì, tiene la chiamata come buon esempio; se no, la butta. Alla fine
ha fabbricato da solo un quaderno di esercizi («qui conveniva la
calcolatrice», «qui conveniva la ricerca») e ci studia sopra. Nessun
insegnante gli ha detto dove mettere gli attrezzi: l'ha scoperto misurando
quanto lo aiutavano.

`````

`````{tab} Superiore

Toolformer usa un'**auto-supervisione** elegante. Partendo da poche
dimostrazioni per ciascuna API (calcolatrice, sistema di domanda-risposta,
motore di ricerca, traduttore, calendario), il modello campiona in molte
posizioni di un corpus delle *candidate* chiamate ad API con i relativi
argomenti. Ogni candidata viene eseguita, e si tiene solo se il suo risultato,
inserito nel contesto, **riduce la cross-entropia pesata** sui token
immediatamente successivi, rispetto al non chiamare o a un risultato inutile:

$$
L_i^{\text{con}} < L_i^{\text{senza}} - \tau,
\qquad
L_i = -\sum_{j \ge i} w_{j-i}\, \log p(x_j \mid \dots),
$$

dove $L_i^{\text{con}}$ e $L_i^{\text{senza}}$ sono la perdita futura con e
senza la chiamata inserita in posizione $i$ ($L_i^{\text{senza}}$ è il
**minimo** fra il non chiamare affatto e il chiamare senza ottenere nulla di
utile), $\tau$ è una soglia di utilità, e i pesi $w_t$ calano linearmente
fino ad **annullarsi dopo cinque token**. Il dettaglio dei pesi non è
pignoleria: dice che ciò che si misura è se la chiamata aiuta a scrivere la
frase in corso, non il resto del documento, ed è la ragione per cui il filtro
non annega nel rumore. Le chiamate che superano il filtro diventano un dataset
aumentato, e il modello ci viene messo a punto sopra con il consueto obiettivo
auto-supervisionato. Il risultato è un modello che, a inferenza, decide *da
sé* quando emettere una chiamata, perché ha imparato che in quei punti la
chiamata paga in termini di predizione. Il criterio è puramente interno
(«l'attrezzo mi aiuta a continuare il testo?») e non richiede alcuna etichetta
umana su dove usarlo.

Vale la pena chiudere con il limite che gli autori dichiarano, perché è
esattamente il confine di questa sezione. Toolformer decide *dove* chiamare,
non *come* comporre: non sa usare gli strumenti in **catena** (l'uscita di uno
come ingresso di un altro) né in modo **interattivo** (raffinare la richiesta
guardando il risultato), ed è ciò che serve a un agente. È il salto che
affronta il pattern della prossima sezione.

`````

## Ragionare e agire insieme: ReAct

Uno strumento, da solo, non basta a fare un agente. Serve una **procedura**:
quando pensare, quando agire, come usare ciò che l'azione ha restituito. Il
pattern diventato lo standard di fatto si chiama **ReAct** (da *Reasoning +
Acting*) proposto da Shunyu Yao e colleghi {cite}`yao2023react`. L'idea è
intrecciare, in un unico flusso, tre tipi di passi: un **pensiero**
(*Thought*, il ragionamento ad alta voce), un'**azione** (*Action*, la
chiamata a uno strumento) e un'**osservazione** (*Observation*, il risultato
che torna indietro). Il modello genera un pensiero, poi un'azione; il sistema
esegue e restituisce l'osservazione; il modello legge l'osservazione, genera
il pensiero successivo, e così via, in un loop, fino a produrre la risposta
finale.

```{figure} ../figures/react-2022.svg
:name: fig-react
:alt: "Il ciclo ReAct come sequenza verticale: un PENSIERO («mi serve il film d'esordio del regista, lo cerco»), un'AZIONE (cerca con il nome del regista), un'OSSERVAZIONE («ha esordito con un film, ma manca l'anno»); poi un secondo giro con un nuovo pensiero, l'azione di cercare il titolo del film e l'osservazione «uscito nel 1994, ora posso rispondere». Una parentesi laterale marca un giro del ciclo."
:width: 62%

Due giri di ReAct su una domanda che nessuna singola ricerca risolve. Ogni
osservazione non chiude il problema: lo restringe, e il pensiero successivo
riparte da lì.
```

L'esempio di {numref}`fig-react` mostra perché il loop serva davvero: la
domanda richiede due fatti, e il secondo si può cercare solo dopo aver
ottenuto il primo. Un sistema che agisse una volta sola non avrebbe modo di
formulare la seconda ricerca, perché non saprebbe ancora cosa cercare.

Perché conviene far ragionare il modello *ad alta voce* tra un'azione e
l'altra? La prima risposta è la chain-of-thought incontrata nel capitolo sui
Transformer {cite}`wei2022chain`, ma va presa per quello che è: i guadagni
misurati di scrivere i passaggi intermedi si concentrano sui compiti
matematici e simbolici, e fuori di lì sono piccoli {cite}`sprague2025cot`,
mentre un loop agentico è fatto in buona parte di altro (scegliere uno
strumento, leggere un risultato, decidere se ripetere). La ragione per cui il
pensiero esplicito serve *qui* è un'altra, e più prosaica: dà al modello un
posto dove scrivere a che punto è del compito prima di scegliere l'azione. È
la stessa idea che ritroveremo, con altro nome, parlando di context
engineering. ReAct la porta nel mondo delle azioni: il pensiero decide *quale*
strumento usare e *come* interpretare ciò che è tornato, e l'osservazione
àncora il pensiero successivo a un fatto reale invece che a una fantasia.

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

Ogni *Observation* è testo prodotto dall'esterno (non campionato dal modello)
e questo è il punto cruciale: àncora il ragionamento a fatti recuperati,
invece di lasciarlo derivare. Sui compiti interattivi come ALFWorld (eseguire
istruzioni in un ambiente simulato) e WebShop (navigare un sito per
acquistare), Yao e colleghi misurano che il ragionamento intercalato all'azione
batte nettamente le politiche che agiscono senza pensare.

Sui compiti a forte intensità di conoscenza, invece, l'ancoraggio va letto per
quello che è: uno **scambio**, non un guadagno secco. Sulla verifica di fatti
(FEVER) ReAct supera la sola chain-of-thought; sulla domanda-risposta
multi-hop (HotpotQA) le resta appena sotto. Le allucinazioni crollano (nei
fallimenti passano da oltre metà a zero) ma il ragionamento si irrigidisce
sulla forma pensiero-azione-osservazione, e gli errori di ragionamento
**triplicano**; per giunta nasce un modo di fallire che prima non esisteva, la
ricerca che torna a mani vuote. Il risultato migliore del lavoro non è ReAct da
solo: è la combinazione dei due, che si alternano quando l'uno si arena.

Il costo è in token e latenza (ogni pensiero è testo generato in più). In
cambio la traccia è **ispezionabile**, ed è un vantaggio operativo vero. Ma
qui va evitata una confusione che costa cara: *leggibile* non vuol dire
*fedele*. La catena di pensieri è testo generato come tutto il resto, e può
razionalizzare a posteriori una scelta compiuta per motivi che non scrive
{cite}`turpin2023unfaithful`; peggio, la fedeltà del ragionamento esplicito
tende a **calare** al crescere della scala del modello {cite}`lanham2023faith`.
La parte della traccia su cui si può contare sono le azioni e le osservazioni,
perché quelle le esegue e le registra il runtime; i pensieri sono un indizio,
non una spiegazione.

`````

## Imparare dai propri errori: la riflessione

Un agente ReAct, però, dentro un singolo tentativo non ha modo di
*migliorare*: se imbocca una strada sbagliata e fallisce, al tentativo dopo
rischia di ripetere lo stesso errore. Nel 2023 Noah Shinn e colleghi
propongono un rimedio semplice e umano, **Reflexion**
{cite}`shinn2023reflexion`: dopo un fallimento, l'agente si ferma e **scrive a
parole cosa è andato storto**, poi riprova tenendo quella critica sotto gli
occhi.

`````{tab} Elementare

È ciò che fa un buon studente dopo un compito andato male. Non si limita a
riprovare identico: rilegge l'errore e se lo dice a parole, «ho sbagliato
perché ho applicato la formula prima di convertire le unità; la prossima volta
converto per prima cosa». Quella frase, appuntata a margine, alla prova
successiva vale più di mille esercizi ripetuti a testa bassa, perché indirizza
il tentativo nuovo lontano dallo stesso scoglio.

Reflexion dà all'agente questo quaderno di margine. Quando un tentativo
fallisce, il modello genera una piccola auto-critica in linguaggio naturale
(la sua «memoria verbale» degli errori) e la aggiunge al contesto del
tentativo seguente. Non cambia un solo peso della rete: cambia solo ciò che il
modello *legge* prima di riprovare. Eppure spesso basta, perché l'errore che
prima era invisibile ora è scritto nero su bianco all'inizio della pagina.

`````

`````{tab} Superiore

Shinn e colleghi chiamano il metodo *verbal reinforcement learning*: al posto
di aggiornare i parametri con un gradiente, il segnale di rinforzo è
**testo**. Il ciclo ha tre ruoli: un *attore* (il modello ReAct) che tenta il
compito; un *valutatore* che assegna un esito al tentativo (una ricompensa, il
superamento o meno di test, il raggiungimento dell'obiettivo); e un *modulo di
auto-riflessione* che, letta la traccia fallita e il suo esito, produce una
critica verbale; «l'azione X non ha dato il risultato atteso, conviene provare
Y». Questa critica finisce in una **memoria episodica** che viene anteposta al
contesto del tentativo successivo. Sui compiti di programmazione gli autori
misurano un guadagno netto di *pass@1*: iterare sull'auto-critica, senza
toccare i pesi, recupera una fetta consistente dei casi inizialmente falliti.

Su quel guadagno conviene però leggere la lettera piccola, perché riguarda il
distinguo che chiude questa sezione. Il *valutatore* che dice «hai sbagliato»
non è, in quegli esperimenti di programmazione, un giudice esterno: è una
batteria di test **generata dal modello stesso**, e gli autori dichiarano che
può promuovere una soluzione sbagliata (tutti i test passano su un programma
errato) o bocciarne una giusta. È un segnale d'esito, ma auto-prodotto: sta
dalla parte scivolosa della riga che stiamo per tracciare, non da quella
solida.

`````

L'onestà impone appunto un distinguo netto, che questo libro deve al lettore.
L'auto-critica **non è** auto-correzione garantita. La riflessione funziona
bene quando esiste un segnale d'esito *affidabile e esterno*: dei **test**
scritti da qualcun altro che passano o falliscono (i test di progetto di
SWE-bench sono l'esempio buono, perché nessuno li ha scritti per far contento
l'agente), un risultato numerico verificabile, un obiettivo raggiunto o no
nell'ambiente. Lì la critica ha un appiglio solido su cui
costruire. Quando invece l'unico giudice è il modello stesso, senza alcun
riscontro dal mondo, la faccenda si fa scivolosa: un modello convinto di una
risposta sbagliata tende a produrre auto-critiche che *confermano* l'errore, e
può perfino peggiorare una risposta che era corretta, «correggendola» verso il
falso. Riflettere aiuta a patto di avere qualcosa contro cui verificarsi; la
sola introspezione, da sé, non crea competenza che il modello non aveva.

## Un agente giocattolo, in Python

Mettiamo insieme i pezzi nel modo più spoglio possibile: un mini-agente ReAct
che gira davvero, in puro Python, senza rete né librerie esterne. Il trucco
per concentrarci sul *ciclo* è sostituire il modello vero con un **LLM finto**
che, a parità di traccia, risponde sempre la stessa cosa (si dice
*deterministico*): non un modello, ma poche righe di regole scritte a mano che,
guardando la traccia finora, decidono il prossimo `Thought` e la prossima
`Action`. Gli strumenti, invece, sono **veri**: una calcolatrice che valuta
un'espressione aritmetica in modo sicuro, e un `cerca` che va a prendere una
voce da un archivio, come si cerca una parola sul vocabolario.

Sulla calcolatrice una parola in più, perché è la parte che vale la pena
riusare. La via facile in Python sarebbe `eval`, la funzione che esegue una
stringa come se fosse codice: comodissima e pericolosa, perché eseguirebbe
*qualunque* cosa il modello scriva, non solo un conto. Al suo posto leggiamo
l'espressione, la spezziamo nei suoi pezzi e la calcoliamo noi, accettando
soltanto gli operatori che abbiamo messo in elenco. Tutto il resto viene
respinto per iscritto, con un messaggio che dice cosa non andava.

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

def _operatore(op):
    """Un operatore fuori elenco esce di qui con un errore leggibile."""
    if type(op) not in _OP:
        raise ValueError(f"operatore non ammesso: {type(op).__name__}")
    return _OP[type(op)]

def _valuta(nodo):
    if isinstance(nodo, ast.Constant):        # un numero, e solo un numero
        if not isinstance(nodo.value, (int, float)):
            raise ValueError("ammessi solo numeri")
        return nodo.value
    if isinstance(nodo, ast.BinOp):           # a operatore b
        return _operatore(nodo.op)(_valuta(nodo.left), _valuta(nodo.right))
    if isinstance(nodo, ast.UnaryOp):         # -a
        return _operatore(nodo.op)(_valuta(nodo.operand))
    raise ValueError("espressione non ammessa")

def calcola(espressione):
    """Valuta un'espressione aritmetica in modo sicuro (senza eval)."""
    return _valuta(ast.parse(espressione, mode="eval").body)

# un piccolo archivio: la memoria esterna che il modello non ha nei pesi
ARCHIVIO = {
    "attention is all you need": "2017",
    "gpt-3": "2020",
    "react": "2022",
}

def cerca(chiave):
    """Cerca un fatto nell'archivio; restituisce sempre una stringa."""
    return ARCHIVIO.get(chiave.lower().strip(), "non trovato")

STRUMENTI = {"calcola": calcola, "cerca": cerca}
```

Il cuore dell'agente sono le altre due funzioni: l'`llm_finto`, che al posto
di un Transformer emette la coppia pensiero-azione a partire dallo stato, e il
loop `esegui_agente`, che alterna decisione ed esecuzione; la stessa struttura
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

Quella traccia, guardata dall'alto, è un cerchio che gira tre volte
({numref}`fig-ciclo-agente`). I tre passi sono sempre gli stessi; quello che
cambia a ogni giro è il contesto, cioè ciò che il modello si ritrova davanti
prima di scegliere la mossa successiva, e che si allunga di un blocco ogni
volta: la chiamata fatta e quello che ha risposto.

```{figure} ../figures/ciclo-agente.svg
:name: fig-ciclo-agente
:alt: "A sinistra il ciclo di un agente: tre caselle collegate in cerchio, pensa (Thought), agisce (Action) e osserva (Observation). L'evidenziazione gira di casella in casella: il modello pensa, chiama uno strumento, il sistema lo esegue e il risultato torna indietro. Al centro un contagiri arriva a giro 3 di 5. Al terzo giro il modello non chiama nessuno strumento: un ramo scende fuori dal cerchio verso la risposta finale, cioè che il paper è del 2017 e sono 9 anni fa nel 2026. A destra la colonna del contesto si allunga di un blocco a ogni giro: prima la sola domanda, poi la ricerca con la sua osservazione 2017, poi il calcolo con la sua osservazione 9, e una barra verticale accanto cresce insieme a loro fino a tre blocchi."
:width: 100%

Lo stesso giro, tre volte. A sinistra i passi che si ripetono; a destra quello
che il modello si rilegge prima di decidere, più lungo di un blocco a ogni
giro. Al terzo giro non serve nessuno strumento e l'agente esce dal ciclo con
la risposta: è uno dei due modi in cui un ciclo finisce, l'altro è il limite di
passi.
```

Il modello finto non sapeva l'anno (l'ha cercato) e non ha fatto la
sottrazione a mente (l'ha delegata): esattamente il comportamento che vogliamo
da un agente. Sostituite `llm_finto` con un vero LLM a cui passate, a ogni
giro, la traccia accumulata e il catalogo degli strumenti, e avete (nella sua
ossatura essenziale) lo stesso ciclo che muove gli assistenti capaci di
navigare il web, eseguire codice e interrogare un database. Tutto il resto,
nei sistemi reali, è robustezza: gestire le chiamate malformate, fermarsi
quando l'agente entra in loop, decidere quali strumenti sono sicuri da
esporre.

Le sei cose da portarsi via da questa sezione, prima di passare al recupero
dei documenti.

`````{tab} Elementare

```{admonition} Da ricordare
:class: important
- Un modello da solo è murato: non sa l'ora, sbaglia i conti lunghi, ignora
  quello che è successo dopo il suo addestramento. Il **tool use** gli dà le
  mani: invece di rispondere di pancia, scrive un **bigliettino d'ordine** per
  uno strumento; il programma che gli sta attorno lo esegue e gli riporta il
  risultato, che il modello ritrova davanti al giro dopo.
- Ogni strumento si presenta con una scheda: come si chiama, a cosa serve, e
  cosa bisogna infilarci dentro perché funzioni. Il modello impara a scegliere
  l'attrezzo giusto e a riempirlo bene. **Toolformer**
  {cite}`schick2023toolformer` lo impara perfino **da solo**, come il bambino
  che scopre quando gli conviene la calcolatrice: prova a infilare una chiamata
  qua e là e tiene quelle che lo aiutano a indovinare meglio le parole
  successive.
- **ReAct** {cite}`yao2023react` è il metodo del detective che ragiona a voce
  alta: **penso → controllo → scopro**, e si ricomincia. Le allucinazioni
  crollano, perché ogni passo si appoggia a un fatto trovato invece che
  immaginato; in cambio il ragionamento si irrigidisce e nasce un modo nuovo di
  sbagliare, la ricerca che non trova niente di utile.
- **Reflexion** {cite}`shinn2023reflexion` è il quaderno di margine: dopo un
  fallimento l'agente si scrive a parole cosa è andato storto e riprova
  leggendo quell'appunto. Non cambia niente dentro la rete: cambia solo quello
  che legge prima di ricominciare.
- Onestà sui limiti: rileggersi **non è** correggersi. Aiuta quando c'è
  qualcuno o qualcosa fuori che dice «giusto» o «sbagliato»; se l'unico giudice
  è il modello stesso, può convincersi di avere ragione avendo torto, e perfino
  rovinare una risposta che era buona.
- Il **giro dell'agente** (guarda, pensa, agisci, ripeti fino alla risposta) è
  lo stesso del mini-agente in venti righe e degli assistenti che navigano il
  web ed eseguono codice. Quello che cambia, nei sistemi veri, è tutto il
  lavoro di rendere il giro robusto quando qualcosa va storto.
```

`````

`````{tab} Superiore

```{admonition} Da ricordare
:class: important
- Un LLM da solo è murato: non sa l'ora, sbaglia i conti lunghi, ignora ciò
  che è successo dopo l'addestramento. Il **tool use** gli dà le mani: invece
  di rispondere, emette una **chiamata strutturata** a uno strumento, che il
  sistema esegue e il cui risultato rientra nel contesto.
- Ogni strumento è uno **schema** (nome, descrizione, argomenti tipati, in
  JSON Schema: `type`, `properties`, `required`); la capacità di sceglierlo e
  compilarne gli argomenti emerge dall'instruction tuning. **Toolformer**
  {cite}`schick2023toolformer` impara *da solo*, con auto-supervisione, dove
  conviene chiamare un'API: tiene le chiamate che riducono la cross-entropia
  **pesata** sui cinque token successivi. Non sa però comporre gli strumenti in
  catena: è il salto che ReAct affronta.
- **ReAct** {cite}`yao2023react` intreccia in un loop **Thought → Action →
  Observation**: le osservazioni àncorano il ragionamento a fatti reali e le
  allucinazioni crollano, ma è uno **scambio**, non un guadagno secco (gli
  errori di ragionamento triplicano e si aggiunge il fallimento della ricerca a
  vuoto). La traccia è ispezionabile, **non** fedele
  {cite}`turpin2023unfaithful, lanham2023faith`: contano le azioni, non i
  pensieri.
- **Reflexion** {cite}`shinn2023reflexion` aggiunge una **memoria verbale**
  degli errori: dopo un fallimento l'agente si auto-critica a parole e riprova
  leggendo la critica, senza toccare i pesi.
- Onestà sui limiti: l'auto-critica **non è** auto-correzione garantita. Aiuta
  quando c'è un esito esterno affidabile (test scritti da altri, risultato
  verificabile); con un giudice auto-prodotto, o con il solo giudizio del
  modello, può confermare l'errore o peggiorare una risposta giusta.
- Il **ciclo dell'agente** (osserva, pensa, agisci, ripeti fino alla risposta)
  è la stessa ossatura del mini-agente in venti righe e degli assistenti che
  navigano il web ed eseguono codice; la differenza è la robustezza attorno.
```

`````
