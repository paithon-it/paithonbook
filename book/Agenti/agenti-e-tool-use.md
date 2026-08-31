# Ragionare e agire: il ciclo dell'agente

Chiedete a un modello di linguaggio che ore sono. Non lo sa. Chiedetegli di
moltiplicare $4831$ per $7092$: sputerà un numero dall'aria plausibile e
spesso sbagliato. Il risultato vero è $34\,261\,452$; un modello che sbaglia
tende a scriverne uno lungo uguale e che comincia allo stesso modo, tipo
$34\,281\,452$, con l'errore nascosto in mezzo, dove nessuno lo cerca.
Chiedetegli cosa è successo ieri, infine, e vi parlerà con sicurezza di un
mondo che si è fermato alla fine del suo addestramento.

Un LLM, per quanto grande, è un cervello murato in una stanza
senza finestre e senza orologio: sa moltissimo di ciò che ha letto, nulla del
resto, e i conti lunghi li sbaglia come chiunque li faccia a mente.

E però può fare una cosa preziosa: *decidere* di chiedere aiuto. Invece di
inventare la risposta, può emettere una richiesta («esegui questa
moltiplicazione», «apri questa pagina», «che ore sono?») lasciare che
qualcos'altro la esegua, e usare il risultato. È il **tool use**, l'uso degli
strumenti: dare le mani a un cervello. Ed è il primo mattone di ciò che
chiamiamo **agente**: un modello che non si limita a rispondere, ma *osserva*,
*decide* e *agisce*, in un ciclo, finché il compito non è chiuso.

Costruiamo su terreno noto. Nel {doc}`capitolo sui Transformer </Transformers/overview>` abbiamo visto tre
cose. La prima è un modello che, letta una montagna di testo, impara a
*completarlo*. La seconda è una fase di addestramento successiva, fatta di
esempi di consegne già svolte, che lo rende capace di *eseguire* una richiesta
invece di limitarsi a proseguirla. La terza è il **prompt**, cioè il foglietto
di istruzioni che si scrive al modello prima di lasciarlo rispondere: è
diventato il modo in cui gli si dice cosa fare, potente e fragile insieme,
perché una parola diversa cambia il risultato.

Un agente è il passo successivo: quello stesso modello, messo dentro il ciclo
osserva-ragiona-agisci, con strumenti a portata di mano e il permesso di
usarli.

## Dare le mani al modello: il tool use

Il meccanismo è più semplice di quanto sembri, e si regge su un catalogo.

Insieme al prompt diamo al modello l'elenco degli attrezzi che ha a
disposizione. Per ognuno tre cose: come si chiama, che cosa fa (scritto a
parole, in italiano) e che cosa bisogna infilarci dentro perché funzioni. Quel
terzo pezzo, in gergo, sono gli **argomenti**: per una calcolatrice il conto da
fare, per una ricerca le parole da cercare, e niente a che vedere con gli
argomenti di cui si discute. Dal lato del programma ogni attrezzo è una
funzione, nel senso informatico che abbiamo dato all'inizio del capitolo, e da
lì il nome inglese di tutto il meccanismo: **function calling**, «chiamata di
funzione».

Quando il modello ritiene che serva uno strumento, non risponde con del testo
per l'utente: emette una **richiesta strutturata**. Non una frase rivolta a una
persona, cioè, ma una riga in un formato fisso, sempre lo stesso, che un
programma sa leggere alla lettera senza doverci capire dentro come si fa con
l'italiano («chiama `calcola` con argomento `"4831 * 7092"`»). Il sistema che
ospita il modello intercetta la richiesta, esegue davvero la funzione, e
restituisce il risultato al modello come nuovo pezzo di **contesto**, cioè del
testo che il modello si ritrova davanti agli occhi al giro dopo. Solo allora il
modello continua.

```{figure} ../figures/function-calling-llm-strumenti.svg
:name: fig-function-calling
:alt: "Schema del function calling in tre passi: l'utente chiede «che tempo fa?», il modello decide se e quale strumento usare ed emette una richiesta tool_use con la funzione get_weather e l'argomento Bologna; il codice dell'applicazione esegue la funzione e restituisce un tool_result con «18 gradi, sereno»; il modello produce infine la risposta in linguaggio naturale. Il giro fra richiesta e risultato può ripetersi più volte."
:width: 90%

Il giro del function calling. Il modello non esegue mai niente: chiede, e
l'esecuzione resta nel codice di chi lo ospita. I passi 1 e 2 possono
ripetersi più volte prima che arrivi la risposta finale.
```

Le due etichette sulle frecce di {numref}`fig-function-calling` sono i nomi che
si usano in gergo per i due messaggi: `tool_use` è «chiedo di usare questo
attrezzo, con dentro queste cose», `tool_result` è «ecco cosa ha risposto
l'attrezzo». La divisione dei compiti che si vede nel disegno è la
ragione per cui il tool use è insieme potente e governabile: il modello
propone, il codice dispone. Chi ospita il modello decide quali funzioni
esistono, le valida prima di eseguirle e può rifiutarsi; il modello non ha mai
in mano l'esecuzione, solo la richiesta.

`````{tab} Elementare

Sulla scrivania di un dirigente competente non c'è nessun attrezzo: c'è un
blocco di moduli. La calcolatrice, il telefono e lo schedario stanno nella
stanza accanto, dove lavora la sua assistente. Alla domanda «quanto fa il
totale della commessa?» lui non azzarda una cifra e non si alza a fare il
conto: riempie un modulo, lo passa di là, e aspetta. Il foglio con il risultato
torna sulla scrivania, fra le carte che rilegge prima di decidere.

Il tool use è questo giro. Al modello diamo un blocco di moduli, uno per
attrezzo, e ognuno dice tre cose: come si chiama («calcolatrice»), a che cosa
serve («fa i conti esatti») e quali caselle riempire perché la richiesta si
possa eseguire («il conto da fare»). La riga che dice a che cosa serve è quella
su cui il modello sceglie: un modulo intestato «pratiche varie» non lo prende
in mano nessuno, perché non si capisce quando servirebbe. E se nessuna casella
è marcata come indispensabile, tocca indovinare quali riempire, e di là arriva
una richiesta che non si può eseguire.

Chi sta di là non esegue a occhi chiusi: legge il modulo, e se manca un dato o
se la cosa chiesta non è nell'elenco torna indietro senza aver fatto niente. Il
modello non tiene mai in mano un attrezzo: tutto quello che fa è scrivere.
Compilare bene quei moduli, del resto, si impara vedendone tanti già compilati
bene.

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

Le tre righe che avvolgono il parametro non sono cerimoniale: `type` dice che
gli argomenti arrivano raccolti in un oggetto, `properties` elenca i campi di
quell'oggetto e `required` dichiara quali non si possono omettere. Nessuna
delle tre è necessaria perché lo schema sia *valido* (uno schema senza
`required` è legittimo e vuol dire «sono tutti facoltativi»), ma è su quelle
righe che il modello decide cosa scrivere, e uno schema che non dice cosa è
obbligatorio glielo lascia indovinare. Il nome della
chiave che lo contiene invece cambia da un fornitore all'altro
(`parameters`, `input_schema`, `inputSchema`); la forma dello schema no, ed è
quella che conta.

La descrizione è il testo su cui il modello ragiona per
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

Quel catalogo, però, va scritto a mano, e va riscritto per ogni sistema
esterno a cui si vuole attaccare l'agente: l'archivio dell'azienda, il
calendario, il gestore dei file. Finché i sistemi sono due o tre va benissimo.
Quando diventano venti conviene mettersi d'accordo su una lingua unica con cui
chiedere a chiunque «che strumenti hai?» e «esegui questo».

Un accordo del genere si chiama **protocollo**, ed è la stessa idea per cui due
computer che non si sono mai visti riescono a scambiarsi una pagina web. Ce
n'è uno pensato apposta per questo, **MCP** («protocollo per il contesto del
modello», dall'inglese *Model Context Protocol*), proposto nel 2024
dall'azienda Anthropic. La sua architettura è in {numref}`fig-mcp`, e conviene
guardarla per la forma più che per il nome: di protocolli così ne nascono e ne
muoiono parecchi, e quale finirà per imporsi è il tipo di cosa che si legge sui
giornali fra un anno; la forma invece è la stessa per tutti.

```{figure} ../figures/mcp-spiegato.svg
:name: fig-mcp
:alt: "A sinistra un riquadro grande, l'applicazione che contiene il modello: dentro ci stanno il modello e due connettori, marcati client 1 e client 2. Ciascun connettore è collegato, con lo stesso protocollo, a un riquadro esterno diverso, il server A e il server B: uno per ogni sistema con cui si vuole parlare. Ogni server dichiara che cosa mette a disposizione, e a destra si vede su cosa comanda: il primo su dei file, il secondo su un archivio di dati. In fondo la scritta: una porta sola, tante periferiche."
:width: 100%

Lo stesso catalogo, ma standardizzato. A parlare il protocollo è
l'applicazione che ospita il modello, la quale apre **un canale per ogni
sistema esterno** (nel disegno sono due, uno che governa dei file e uno che
governa un archivio di dati) e li interroga tutti allo stesso modo. Al modello
arrivano poi strumenti come gli altri, senza che debba sapere da dove vengono.
```

Il salto di {numref}`fig-mcp` non è nel meccanismo, che resta quello di prima
(il modello scrive la richiesta, il programma la esegue), ma nel numero di
sistemi che si riescono a collegare senza scrivere codice nuovo ogni volta.
Cambia anche chi scrive le etichette degli attrezzi: non più chi costruisce
l'agente, ma chi mette a disposizione il sistema dall'altra parte.

Resta però una domanda che finora abbiamo scavalcato: chi ha insegnato al
modello *quando* fermarsi e chiamare un attrezzo? Non basta avere il catalogo:
bisogna anche riconoscere il momento in cui serve. Glielo si insegna
addestrandolo su tanti esempi di chiamate fatte al punto giusto, esempi che
finora ha dovuto scrivere qualcuno, uno per uno, a mano.

Nel 2023, però, un gruppo di Meta AI (il laboratorio di ricerca dell'azienda a
cui appartiene Facebook) ha mostrato che quegli esempi il modello se li può
fabbricare **da solo**: è Toolformer {cite}`schick2023toolformer`. E si
fabbricano in un punto inatteso, non prima o dopo una frase ma dentro, in mezzo
a una parola e l'altra.

```{figure} ../figures/toolformer-2023.svg
:name: fig-toolformer
:alt: "Schema di Toolformer: mentre genera la frase «400 su 1400, cioè il…» il modello arriva a un punto di decisione, chiamo un tool? Se la risposta è no continua a scrivere la parola successiva; se è sì emette Calculator(400/1400), lo strumento esterno calcola 0.29 e il risultato rientra nella frase, che riprende come «400 su 1400, cioè il 29%»."
:width: 88%

Toolformer decide *dentro* la frase. Al punto di decisione il modello può
proseguire normalmente oppure inserire una chiamata: il risultato torna nel
testo e la generazione riparte da lì, come se il numero l'avesse scritto lui.
```

Il dettaglio da guardare in {numref}`fig-toolformer` è appunto quello: la
chiamata sta **dentro** la frase, e il suo risultato ($0{,}29$) rientra nel
testo giusto prima della parola che lo commenta ($29\%$). Ed è questo a rendere
possibile il trucco con cui Toolformer impara. Se la chiamata sta lì in mezzo,
il modello può misurare una cosa che sa misurare benissimo: quanto gli riesce
facile scrivere le parole che vengono subito dopo. Con il numero vero sotto gli
occhi, «29%» diventa quasi obbligato; senza, è un tiro a indovinare. La
differenza fra le due difficoltà è il voto che Toolformer dà alla chiamata.

`````{tab} Elementare

Come impara un bambino a usare la calcolatrice? Provando. Fa un conto a mente,
controlla con la macchina, e nei conti lunghi scopre che la macchina ci azzecca
dove lui sbaglia: la volta dopo, per i conti lunghi, la prende subito.
Toolformer si allena così su se stesso, e il compito su cui si corregge è un
testo già scritto da altri, di cui conosce ogni parola.

Prende quel testo e, qua e là, prova a infilarci dentro la chiamata a uno
strumento (per scriverla gli bastano due o tre esempi già fatti). Poi si copre
il seguito e prova a indovinarlo due volte: una con il risultato dell'attrezzo
davanti agli occhi, una senza. Prendi «quattrocento su millequattrocento, cioè
il 29%»: con «0,29» scritto in mezzo, «29%» viene quasi da sé; senza, è un tiro
a indovinare. Se il salto di facilità è grosso, l'attrezzo lì serviva; se è
piccolo, non conta.

Due cautele tengono onesta la misura. Prova anche a infilare la chiamata
lasciando vuoto il posto del risultato: se le parole dopo diventano facili lo
stesso, ad aiutare non era il numero. E guarda vicino, cioè le parole
che vengono subito dopo, non tutto il resto del foglio.

Le chiamate promosse le tiene, le altre le butta, e sul quaderno di esercizi
che ne esce ci studia sopra. Nessun insegnante gli ha detto dove mettere gli
attrezzi: l'ha scoperto misurando quanto lo aiutavano. Un attrezzo alla volta,
però: cercare un numero da qualche parte e poi usarlo nel conto, quello non lo
impara.

`````

`````{tab} Superiore

Toolformer usa un’**auto-supervisione** elegante. Partendo da poche
dimostrazioni per ciascuna API (calcolatrice, sistema di domanda-risposta,
motore di ricerca, traduttore, calendario), il modello campiona in molte
posizioni di un corpus delle *candidate* chiamate ad API con i relativi
argomenti. Ogni candidata viene eseguita, e si tiene solo se il suo risultato,
inserito nel contesto, **riduce la cross-entropia pesata** sui token
immediatamente successivi, rispetto al non chiamare o a un risultato inutile:

$$
\mathcal{L}_i^{\text{con}} < \mathcal{L}_i^{\text{senza}} - \tau,
\qquad
\mathcal{L}_i = -\sum_{j \ge i} w_{j-i}\, \log p(x_j \mid \dots),
$$

dove $\mathcal{L}_i^{\text{con}}$ e $\mathcal{L}_i^{\text{senza}}$ sono la
perdita futura con e senza la chiamata inserita in posizione $i$
($\mathcal{L}_i^{\text{senza}}$ è il **minimo** fra il non chiamare affatto e
il chiamare senza ottenere nulla di utile), $x_j$ sono i token che nel testo
originale seguono quel punto e $p(x_j \mid \dots)$ la probabilità che il
modello assegna loro, $\tau$ è una soglia di utilità, e i pesi $w_{j-i}$
dipendono solo dalla distanza dal punto della chiamata: calano linearmente
fino ad **annullarsi dopo cinque token**. Quei pesi dicono una cosa precisa:
ciò che si misura è se la chiamata aiuta a scrivere la frase in corso, non il
resto del documento, ed è la ragione per cui il filtro non annega nel rumore.
Le chiamate che superano il filtro diventano un dataset aumentato, e il
modello ci viene messo a punto sopra con il consueto obiettivo
auto-supervisionato. Il risultato è un modello che, a inferenza, decide *da
sé* quando emettere una chiamata, perché ha imparato che in quei punti la
chiamata paga in termini di predizione. Il criterio è puramente interno
(«l'attrezzo mi aiuta a continuare il testo?») e non richiede alcuna etichetta
umana su dove usarlo.

Gli autori dichiarano un limite preciso, ed è il confine fra usare uno
strumento e condurre un compito. Toolformer decide *dove* chiamare,
non *come* comporre: non sa usare gli strumenti in **catena** (l'uscita di uno
come ingresso di un altro) né in modo **interattivo** (raffinare la richiesta
guardando il risultato), ed è ciò che serve a un agente. È il salto che
affronta ReAct, che però non gli è succeduto: ReAct è dell'ottobre 2022,
Toolformer del febbraio successivo. Sono due risposte a due domande diverse, non
due tappe di una scala.

`````

## Ragionare e agire insieme: ReAct

Uno strumento, da solo, non basta a fare un agente. Serve una **procedura**:
quando pensare, quando agire, come usare ciò che l'azione ha restituito. Lo
schema di lavoro che ha dato il nome a questo modo di procedere si chiama
**ReAct**, dall'inglese *reasoning* e *acting*, ragionare e agire, ed è stato
proposto nel 2022 da Shunyu Yao e colleghi {cite}`yao2023react`. L'idea è
intrecciare, in un unico flusso, tre tipi di passi: un **pensiero**
(*Thought*, il ragionamento ad alta voce), un’**azione** (*Action*, la
chiamata a uno strumento) e un’**osservazione** (*Observation*, il risultato
che torna indietro). Il modello genera un pensiero, poi un'azione; il sistema
esegue e restituisce l'osservazione; il modello legge l'osservazione, genera
il pensiero successivo, e così via, fino a produrre la risposta finale.

Una nota per non confondersi con l'ordine. Un cerchio non ha un inizio, e
infatti a volte lo si racconta partendo dall'osservazione (osserva, ragiona,
agisci) e a volte dal pensiero (pensa, agisci, osserva): sono lo stesso giro
guardato da due punti diversi. Nelle tracce scritte si parte dal pensiero,
perché la prima osservazione è la domanda dell'utente, che è già lì.

```{figure} ../figures/react-2022.svg
:name: fig-react
:alt: "Il ciclo ReAct come sequenza verticale: un PENSIERO («mi serve il film d'esordio del regista, lo cerco»), un'AZIONE (cerca con il nome del regista), un'OSSERVAZIONE («ha esordito con un film, ma manca l'anno»); poi un secondo giro con un nuovo pensiero, l'azione di cercare il titolo del film e l'osservazione «uscito nel 1994, ora posso rispondere». Una parentesi laterale marca un giro del ciclo."
:width: 62%

Due giri di ReAct su una domanda che nessuna singola ricerca risolve. Ogni
osservazione non chiude il problema: lo restringe, e il pensiero successivo
riparte da lì.
```

La domanda dell'esempio in {numref}`fig-react` è di quelle che sembrano
banali: «in che anno è uscito il film d'esordio di quel regista?». Per
rispondere servono due fatti, e il secondo si può cercare solo dopo aver
ottenuto il primo: finché non so *quale* sia il film d'esordio, non ho niente
da cercare. Un sistema che agisse una volta sola resterebbe fermo al primo
giro, perché non saprebbe ancora cosa chiedere.

Perché conviene far ragionare il modello *ad alta voce* tra un'azione e
l'altra? La risposta che viene per prima, la catena di ragionamento
{cite}`wei2022chain`, qui vale poco: i suoi guadagni misurati stanno sui conti
e sulla logica {cite}`sprague2025cot`, mentre il ciclo di un agente è fatto in
buona parte di altro, cioè scegliere uno strumento, leggere un risultato e
decidere se ripetere.

La ragione per cui il pensiero esplicito serve *qui* è un'altra, e più
prosaica: dà al modello un posto dove scrivere a che punto è del compito prima
di scegliere l'azione. È la stessa idea che ritroveremo, chiamata **foglio di
brutta**, parlando di come si riempie la finestra di contesto.

Ma il guadagno più grosso sta nell’**osservazione**, più che nel pensiero, e
per apprezzarlo serve un nome. Quando un modello inventa un fatto e lo dice con la
faccia di chi lo sa, si parla di **allucinazione**: il modello genera la
continuazione più plausibile, e nessuno gli ha mai chiesto di controllare.
Un'osservazione che arriva da fuori, invece, non se l'è inventata
lui: è testo che gli è stato messo davanti dal programma. Il pensiero decide
*quale* strumento usare e *come* leggere ciò che è tornato, ma è
l'osservazione a tenerlo attaccato a qualcosa di vero.

`````{tab} Elementare

Un detective indaga a voce alta. Non spara subito il colpevole: alterna
ragionamenti e verifiche. «*Penso*: la vittima è stata vista l'ultima
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

Il guadagno però si paga. Un detective che controlla ogni intuizione prima di
proseguire fa meno voli di fantasia, e finisce anche per pensare di meno: la
mossa dopo gliela detta l'ultimo documento che ha letto, e le catene lunghe di
ragionamento smette di farle. E c'è un modo di fallire che prima non aveva: il
registro può non dirgli niente di utile, e lì resta fermo, mentre chi ragionava
per conto proprio almeno un'ipotesi la produceva.

Un'ultima cosa, su quel parlare a voce alta. Il ragionamento che il detective
recita suona convincente, ma resta un racconto, e certe volte è costruito dopo,
per far quadrare una mossa presa d'istinto. Le cose su cui contare sono i
registri che ha davvero aperto e quello che c'era scritto dentro.

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
sulla forma pensiero-azione-osservazione, e gli errori di ragionamento quasi
**triplicano**, dal 16% al 47% delle traiettorie fallite esaminate; per
giunta nasce un modo di fallire che prima non esisteva, la
ricerca che torna a mani vuote. Il risultato migliore del lavoro viene dalla
combinazione dei due, che si alternano quando l'uno si arena.

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

Un ciclo ReAct finisce in due modi: con la risposta, oppure a mani vuote,
quando i passi concessi si esauriscono o la strada imboccata non porta da
nessuna parte. In quel secondo caso la cosa ovvia da fare è **riprovare da
capo**, e qui salta fuori il problema: l'agente riparte esattamente com'era
partito la prima volta, senza sapere niente di com'è andata, e ha ottime
probabilità di rifare lo stesso errore.

Nel 2023 Noah Shinn e colleghi propongono un rimedio semplice e umano,
**Reflexion** {cite}`shinn2023reflexion`: dopo un fallimento, l'agente si ferma
e **scrive a parole cosa è andato storto**, poi riprova tenendo quella critica
sotto gli occhi.

`````{tab} Elementare

Un buon studente, dopo un compito andato male, non riprova identico: rilegge
l'errore e se lo dice a parole, «ho sbagliato perché ho applicato la formula
prima di convertire le unità; la prossima volta converto per prima cosa». Quella frase, appuntata a margine, alla prova
successiva vale più di mille esercizi ripetuti a testa bassa, perché indirizza
il tentativo nuovo lontano dallo stesso scoglio.

Reflexion dà all'agente questo quaderno di margine. Quando un tentativo
fallisce, il modello genera una piccola auto-critica in linguaggio naturale
(la sua «memoria verbale» degli errori) e la aggiunge al contesto del
tentativo seguente. Non cambia un solo peso della rete: cambia solo ciò che il
modello *legge* prima di riprovare. Eppure spesso basta, perché l'errore che
prima era invisibile ora è scritto nero su bianco all'inizio della pagina.

Il voto, intanto, lo mette il professore, che segna gli errori in rosso e non
ha interesse a essere gentile. Uno studente che si corregge il compito da solo
si dà il visto proprio dove ha sbagliato.

`````

`````{tab} Superiore

Shinn e colleghi chiamano il metodo *verbal reinforcement learning*: al posto
di aggiornare i parametri con un gradiente, il segnale di rinforzo è
**testo**. Il ciclo ha tre ruoli: un *attore* (il modello ReAct) che tenta il
compito; un *valutatore* che assegna un esito al tentativo (una ricompensa, il
superamento o meno di test, il raggiungimento dell'obiettivo); e un *modulo di
auto-riflessione* che, letta la traccia fallita e il suo esito, produce una
critica verbale: «l'azione X non ha dato il risultato atteso, conviene provare
Y». Questa critica finisce in una **memoria episodica** che viene anteposta al
contesto del tentativo successivo. Sui compiti di programmazione gli autori
misurano un guadagno netto di *pass@1*: iterare sull'auto-critica, senza
toccare i pesi, recupera una fetta consistente dei casi inizialmente falliti.

La lettera piccola di quel guadagno riguarda chi fa il giudice. Il
*valutatore* che dice «hai sbagliato» non è, in quegli esperimenti di
programmazione, un giudice esterno: è una
batteria di test **generata dal modello stesso**, e gli autori dichiarano che
può promuovere una soluzione sbagliata (tutti i test passano su un programma
errato) o bocciarne una giusta. È un segnale d'esito, ma auto-prodotto: il
caso in cui l'auto-critica ha meno di solido su cui appoggiarsi.

`````

Detta così, la riflessione sembra magica. Non lo è: l'auto-critica **non è**
auto-correzione garantita, e tutto dipende da chi dice all'agente che ha
sbagliato. La riflessione funziona bene
quando esiste un segnale d'esito *affidabile ed esterno*: dei **test**
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
che gira davvero, in puro Python, senza collegarsi a internet e senza
installare niente. Chiamiamo **traccia** l'elenco di quello che è successo
finora, un blocco per giro (che cosa ho chiesto, che cosa mi è tornato): è la
memoria del nostro agente, e la vedremo allungarsi sotto i nostri occhi.

Il trucco per concentrarci sul *ciclo* è sostituire il modello vero con un
**LLM finto** che, a parità di traccia, risponde sempre la stessa cosa (si
dice *deterministico*): non un modello, ma poche righe di regole scritte a
mano che, guardando la traccia, decidono il prossimo `Thought` e la prossima
`Action`. Gli strumenti, invece, sono **veri**: una calcolatrice che valuta
un'espressione aritmetica in modo sicuro, e un `cerca` che va a prendere una
voce da un archivio, come si cerca una parola sul vocabolario.

Il blocco che segue costruisce quei due strumenti, e la parte più lunga è la
calcolatrice. La via facile in Python sarebbe `eval`, la funzione che esegue
una stringa (cioè un pezzo di testo) come se fosse codice: comodissima e
pericolosa, perché eseguirebbe *qualunque* cosa il modello scriva, non solo un
conto. Al suo posto leggiamo l'espressione, la spezziamo nei suoi pezzi e la
calcoliamo noi, accettando soltanto gli operatori che abbiamo messo in elenco;
tutto il resto viene respinto con un messaggio che dice cosa non andava. Non è una precauzione da manuale: quello che il modello scrive va trattato
come si tratta il testo di uno sconosciuto. Il cuore resta il blocco
dopo.

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

Nell'archivio in fondo al blocco ci sono tre voci, e la prima è quella su cui
verterà la domanda: *Attention Is All You Need* è il titolo dell'articolo
scientifico (in gergo, un **paper**) che nel 2017 ha presentato i Transformer,
l'architettura studiata nel {doc}`capitolo sui Transformer </Transformers/overview>`.

Il cuore dell'agente sono le altre due funzioni. La prima è `llm_finto`: legge
la traccia e restituisce il pensiero e l'azione da fare (l'azione è due cose,
il nome dell'attrezzo e quello che ci va infilato dentro). La seconda è il
ciclo `esegui_agente`, che alterna decisione ed esecuzione. È la stessa
struttura di un agente vero, con l'unica differenza che qui il «modello» è una
regola scritta a mano.

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
navigare il web, eseguire codice e interrogare un archivio di dati.

Tutto il resto, nei sistemi reali, è il lavoro di reggere quando qualcosa va
storto. Sono tre mestieri. Bisogna sapere cosa fare quando il modello scrive
una chiamata malformata, cioè che non rispetta il formato concordato. Bisogna
accorgersi che l'agente si è impantanato e ripete la stessa mossa all'infinito,
e fermarlo. E bisogna decidere quali strumenti è prudente mettergli in mano,
visto che li userà davvero.

(Sul secondo, una nota per non confondersi. In inglese quell'impantanarsi si
dice «entrare in loop», e *loop* è la stessa parola che indica il ciclo che fa
funzionare l'agente. Stessa parola, due significati opposti: uno è il motore,
l'altro è il guasto.)

Le sei cose da portarsi via da questa sezione, prima di passare al recupero
dei documenti.

`````{tab} Elementare

```{admonition} Da ricordare
:class: important
- Un modello da solo è murato: non sa l'ora, sbaglia i conti lunghi, ignora
  quello che è successo dopo il suo addestramento. Il **tool use** gli dà le
  mani: invece di rispondere di pancia, scrive per uno strumento il
  **bigliettino d'ordine** che il cuoco passava al cameriere; il programma che
  gli sta attorno lo esegue e gli riporta il risultato, che il modello ritrova
  davanti al giro dopo.
- Ogni strumento si presenta con un modulo: come si chiama, a cosa serve, e
  cosa bisogna infilarci dentro perché funzioni. Il modello impara a scegliere
  l'attrezzo giusto e a riempire bene il modulo. **Toolformer**
  {cite}`schick2023toolformer` lo impara perfino **da solo**, come il bambino
  che scopre quando gli conviene la calcolatrice: prova a infilare una chiamata
  qua e là e tiene quelle che lo aiutano a indovinare meglio le parole
  successive.
- **ReAct** {cite}`yao2023react` è il metodo del detective che ragiona a voce
  alta: **penso → controllo → scopro**, e si ricomincia (è lo stesso giro di
  prima, raccontato partendo dal pensiero). Le **allucinazioni**, cioè i fatti
  che il modello si inventa dicendoli con sicurezza, crollano, perché ogni
  passo si appoggia a qualcosa che è stato davvero trovato; in cambio il
  ragionamento si irrigidisce e nasce un modo nuovo di sbagliare, la ricerca
  che non trova niente di utile.
- **Reflexion** {cite}`shinn2023reflexion` è il quaderno di margine: dopo un
  fallimento l'agente si scrive a parole cosa è andato storto e riprova
  leggendo quell'appunto. Non cambia niente dentro la rete: cambia solo quello
  che legge prima di ricominciare.
- Onestà sui limiti: rileggersi **non è** correggersi. Aiuta quando c'è
  qualcuno o qualcosa fuori che dice «giusto» o «sbagliato»; se l'unico giudice
  è il modello stesso, può convincersi di avere ragione avendo torto, e perfino
  rovinare una risposta che era buona.
- Il **ciclo dell'agente** (guarda, pensa, agisci, ripeti fino alla risposta) è
  lo stesso del mini-agente di poche decine di righe e degli assistenti che
  navigano il web ed eseguono codice. Quello che cambia, nei sistemi veri, è
  tutto il lavoro di rendere il ciclo robusto quando qualcosa va storto.
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
  conviene chiamare un'API: tiene le chiamate che riducono la cross-entropia **pesata** sui cinque token
  che seguono il punto della chiamata. Non sa però comporre gli strumenti in
  catena: è il salto che ReAct affronta.
- **ReAct** {cite}`yao2023react` intreccia in un loop **Thought → Action →
  Observation**: le osservazioni àncorano il ragionamento a fatti reali e le
  allucinazioni crollano, ma è uno **scambio**, non un guadagno secco (gli
  errori di ragionamento quasi triplicano, dal 16% al 47%, e si aggiunge il
  fallimento della ricerca a vuoto). La traccia è ispezionabile, **non** fedele
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
  è la stessa ossatura del mini-agente di poche decine di righe e degli
  assistenti che navigano il web ed eseguono codice; la differenza è la robustezza attorno.
```

`````
