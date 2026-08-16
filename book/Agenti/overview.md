# Agenti: quando i modelli linguistici agiscono

Fra rispondere bene a una domanda e portare a termine un lavoro c'è un salto, e
qualcuno ha provato a misurarlo con una gara di riparazioni. La gara prende
2.294 segnalazioni di errore vere, di quelle che gli utenti scrivono agli autori
di un programma quando qualcosa non funziona, e le usa come compiti d'esame: al
sistema si dà la segnalazione insieme al codice del progetto, e deve produrre la
correzione. Si chiama **SWE-bench** {cite}`jimenez2024swebench`.

A dire se ha funzionato non c'è una persona, ma i **test** del progetto: pezzi
di programma che gli sviluppatori scrivono apposta per controllare il proprio
lavoro, e che a ogni modifica rispondono «a posto» oppure «rotto». Sono gli
stessi test che avevano approvato la correzione scritta, a suo tempo, da uno
sviluppatore in carne e ossa. È un compito che nessun completamento di testo,
per quanto fluente, chiude in un colpo solo: bisogna trovare i file giusti,
provare, sbagliare, rileggere il messaggio d'errore, correggere. I primi
sistemi ci riuscivano in una **piccola frazione** dei casi, pochi punti
percentuali. Un numero così basso non è una delusione, è la notizia: è la prima
misura pubblica di quanto costi tenere insieme molte mosse di fila.
Sull'affidabilità di quella misura, però, l'ultima sezione avrà qualcosa da
ridire, e non è un dettaglio: si scoprirà che una parte di quei pochi successi
non era stata guadagnata sul campo.

Quel salto, però, non è il salto fra dire e fare. Il capitolo su visione e
linguaggio si è appena chiuso su un modello che *fa*: se si tagliano i comandi
di un braccio robotico in gradini, muovere la mano diventa scrivere sette parole
di fila, e la stessa macchina che compone frasi compone movimenti. Quel modello
nel mondo ci mette le mani sul serio. Quello che non fa è decidere: l'obiettivo
glielo consegna qualcun altro («prendi la tazza»), e la mossa la sceglie
guardando soltanto la fotografia di adesso e l'istruzione ricevuta. Da un
comando al successivo non si porta dietro il ricordo di che cosa ha già provato.

È lì che si apre lo spazio di questo capitolo. La domanda non è se un modello
possa agire, ma chi decide **quando** agire e con quale strumento, chi sceglie
la **sequenza** delle mosse, e chi tiene il conto di **quello che è già
successo** mentre il lavoro va avanti.

Un esempio piccolo lo dice meglio di una definizione. Se chiedi a un modello che
tempo farà domani a Roma non ne ha idea, perché quel dato non esisteva nei mesi
in cui ha studiato (in gergo si dice che è stato **addestrato**: gli si è fatta
leggere una montagna di testo finché non ha imparato a proseguirlo). Può
inventare la risposta, con la stessa sicurezza con cui ne direbbe una vera,
oppure fermarsi a metà, andare a *guardare* il meteo e riprendere da lì. La
seconda strada non è più fluenza: è una decisione presa nel mezzo di una frase.

Fra il 2023 e il 2024 compaiono i sistemi che quella decisione la prendono di
continuo: cercano sul web una notizia di ieri, eseguono un pezzo di codice per
controllare se gira, compilano un modulo, prenotano, propongono una correzione a
un programma vero. È il mondo di SWE-bench, ed è il mondo di questo capitolo.

Prima di andare avanti, mettiamo un paletto che vale per tutto il capitolo. Un
**modello** è la rete che, dato un testo, ne predice la continuazione: quello
che abbiamo studiato nel capitolo sui Transformer. Un **agente** è un
*sistema* costruito attorno a un modello: un programma che guarda l'ambiente,
lascia che il modello decida la mossa successiva, la esegue davvero, osserva
com'è andata e ricomincia. L’**ambiente** è tutto ciò su cui l'agente può
mettere le mani e da cui può ricevere notizie: le pagine del web, i file di un
computer, i servizi a cui si può chiedere qualcosa. Il modello è il motore;
l'agente è l'automobile, con volante, ruote e strada. Questo capitolo è
dedicato all'automobile.

## Dal completare testo all'agire

Il primo passo di un modello fuori da se stesso è stato piccolo e si è visto
nel capitolo sui Transformer: prima di rispondere, va a cercare qualcosa in un
archivio di documenti e se lo rilegge. Quella mossa ha un nome, **RAG**, e il
disegno qui sotto è lo schema con cui è stata presentata al mondo.

```{figure} ../figures/rag-lewis-2020.svg
:name: fig-rag-lewis
:alt: "Schema del RAG originale, da sinistra a destra: la domanda dell'utente entra in un cercatore, che pesca in un archivio ricavato da Wikipedia (una fila di riquadri, ventun milioni di brani) e ne tira fuori i pochi più pertinenti, disegnati come una seconda fila di riquadri; da lì una freccia risale a chi scrive la risposta, il generatore, al quale arriva anche una freccia diretta dal cercatore. Le due sigle in etichetta, DPR e BART, sono i due modelli usati nell'articolo originale. In fondo al disegno: la risposta è condizionata insieme dalla domanda e dai passaggi recuperati."
:width: 96%

Il disegno ha tre pezzi: chi cerca nell'archivio (in inglese il *retriever*,
il cercatore), l'archivio stesso e chi scrive la risposta, il generatore. La
domanda entra da sinistra, passa dal cercatore e arriva a chi scrive insieme
ai pochi brani che il cercatore ha pescato. La conoscenza, così, non sta più
tutta dentro il modello.
```

Le tre lettere stanno per *Retrieval-Augmented Generation*, cioè «generazione
aiutata da un recupero», e sono di Lewis e colleghi {cite}`lewis2020retrieval`.
Il nome dice poco; quel che fa, invece, si dice in una riga: il modello
**sospende la risposta, va a prendere qualcosa fuori di sé, e solo dopo
conclude**. È già qualcosa che il completamento puro non sa fare, ed è il
precedente diretto di tutto questo capitolo.

Il «fuori di sé» va preso alla lettera, e vale la pena fermarsi un istante,
perché è la frattura da cui nasce tutto il resto. Quello che un modello sa lo
tiene in un enorme mucchio di numeri, fissati durante l'addestramento e non più
modificabili: si chiamano i **pesi**, e sono la sua memoria di fabbrica. La RAG
è la prima volta che una parte della conoscenza esce da lì e va a vivere in un
archivio che si può correggere e aggiornare senza riaddestrare niente.

Perché il completamento di testo, da solo, non basta? Perché rispondere è un
atto unico e chiuso, mentre agire nel mondo è un processo: richiede più mosse
in sequenza, ognuna decisa alla luce di come è andata la precedente. Comprare
un biglietto significa cercare i treni, confrontare gli orari, scegliere,
pagare, ricevere conferma, e se a metà strada il treno scelto risulta pieno,
tornare indietro e riprovare. Un oracolo che sputa una risposta e si spegne
non può fare niente di tutto questo.

Quell'andirivieni (una mossa, il suo esito, la mossa seguente decisa alla luce
dell'esito) è la cosa che d'ora in poi chiameremo il **ciclo** dell'agente. È
il pezzo che torna in ogni pagina del capitolo, e conviene averlo in mente
sotto questo nome fin da subito.

`````{tab} Elementare

Pensa alla differenza tra un consulente e un assistente. Il **consulente** ti
dà consigli a parole: «per andare a Milano ti conviene il treno delle 9, poi
prenota un hotel in centro». Ottimo, ma il lavoro resta tutto a te: sei tu che
apri il sito, digiti le date, paghi. L’**assistente**, invece, le cose le
*fa*: telefona, prenota, compila il modulo, ti mette in mano il biglietto. La
stessa testa, ma con le mani.

Un agente è il salto dal consulente all'assistente. Il modello continua a
essere il cervello (sa *cosa* andrebbe fatto) ma attorno gli mettiamo delle
mani (gli strumenti) e un metodo di lavoro: fai una mossa, guarda com'è
andata, decidi la prossima. È la differenza tra chi ti spiega la ricetta e chi
ti cucina la cena.

`````

`````{tab} Superiore

Formalmente, un agente è un ciclo di controllo (**osserva → ragiona → agisci →
osserva**) in cui il modello ricopre il ruolo di *policy*: la funzione che,
dato lo stato corrente, sceglie l'azione. È la stessa nozione di policy vista
nel capitolo sul reinforcement learning, ma qui lo «stato» è una sequenza di
testo (il contesto accumulato: la richiesta, le mosse fatte, i loro risultati)
e l’«azione» è, tipicamente, l'invocazione di uno strumento oppure la risposta
finale. A ogni passo:

1. il sistema fornisce al modello lo stato $s_t$ (il contesto);
2. il modello genera un'azione $a_t$, per esempio «cerca sul web *X*»;
3. il *runtime* esegue $a_t$ e produce un'osservazione $o_t$ (i risultati);
4. si aggiorna lo stato, $s_{t+1} = s_t \oplus a_t \oplus o_t$, e si ricomincia,

dove $\oplus$ denota la concatenazione al contesto e il ciclo termina quando il
modello emette un'azione speciale di «risposta finale» o si raggiunge un limite
di passi. La differenza cruciale con il reinforcement learning classico sta in
chi fa cosa: chi costruisce l'agente, di norma, non ottimizza $\theta$. La
policy è un modello di linguaggio già addestrato (spesso proprio con
ricompense ed episodi, nel post-training che gli ha insegnato a seguire
istruzioni e a usare strumenti) e qui il suo comportamento si governa con
istruzioni in linguaggio naturale: non si aggiornano i pesi, si scrive il
*prompt*.

`````

Un ingrediente aiuta il ciclo: far «ragionare ad alta voce» il modello prima
di agire. Invece di saltare all'azione, il modello scrive il proprio
ragionamento (*«prima di correggere devo capire quale pezzo del programma si è
lamentato»*) e solo dopo sceglie la mossa. Questa catena di ragionamento
scritta ha un nome inglese che incontrerai dappertutto, **chain-of-thought**
{cite}`wei2022chain`, e nel capitolo sui Transformer l'abbiamo vista far
salire il numero di risposte giuste sui problemi che richiedono più passaggi.

Conviene però dire subito dove quel guadagno è stato misurato davvero, perché
è più stretto di come lo si racconta di solito. Una rassegna che rimette
insieme i risultati di oltre cento lavori lo trova concentrato sui compiti
**matematici e simbolici**, quelli in cui si manipolano numeri e regole (un
conto, un'espressione algebrica, un problema di logica), e piccolo altrove
{cite}`sprague2025cot`. In un agente il pensiero scritto serve soprattutto a
un'altra cosa: dare al modello un posto dove annotare a che punto è del
compito, prima di scegliere la mossa. È il collante fra il pensare e il fare,
non una cura generale.

## L'anatomia di un agente

Smontiamo l'automobile. Al di là delle mille varianti, ogni agente ha quattro
ingredienti, e conviene tenerli distinti perché ognuno ha problemi suoi.

- Il **modello** è il cervello: legge il contesto, ragiona, decide la prossima
  azione. È l'unico pezzo che «pensa»; tutto il resto è impalcatura attorno.
- Gli **strumenti** (in inglese *tool*) sono le mani: una ricerca sul web, un
  programma che esegue del codice al posto suo, un'interrogazione a un archivio
  di dati, la richiesta a un servizio esterno. Quest'ultima passa da uno
  sportello: un programma si presenta a un altro con una domanda in un formato
  concordato e ne riceve una risposta, senza sapere niente di come sia fatto
  dentro. Quello sportello si chiama **API** (dall'inglese *application
  programming interface*). Gli strumenti sono ciò che permette all'agente di
  *toccare* il mondo: leggere dati freschi e produrre effetti.
- Il **ciclo di controllo** è il metodo di lavoro: il programma che alterna
  percezione e azione, passa il contesto al modello, esegue l'azione scelta,
  raccoglie il risultato e decide se continuare o fermarsi. In inglese si dice
  *loop*, ed è la parola che si sente più spesso.
- La **memoria** è ciò che l'agente si porta dietro. Nel breve termine è la
  **finestra di contesto**: quanto testo il modello riesce a tenere davanti
  agli occhi in una volta sola, prima di scrivere. È larga ma finita, e nel
  capitolo sui Transformer l'abbiamo studiata da vicino, insieme al segnalibro
  con cui il modello evita di rileggere ogni volta da capo ciò che ha già letto
  (là si chiama **KV cache**). Nel lungo termine è invece una memoria
  *esterna*: un archivio di documenti o di ricordi passati da cui pescare
  quando serve, senza tenere tutto in testa.

```{figure} ../figures/agente-anatomia.svg
:name: fig-agente-anatomia
:alt: "Diagramma dell'anatomia di un agente: al centro il MODELLO (LLM); a sinistra la MEMORIA (contesto e memoria esterna), a destra gli STRUMENTI (web, codice, API), entrambi collegati al modello con frecce bidirezionali; in basso l'AMBIENTE. Due frecce curve chiudono il ciclo: osserva porta dall'ambiente al modello, agisci porta dal modello all'ambiente."
:width: 88%

I quattro ingredienti e il ciclo di controllo: il modello ragiona al centro,
usa gli strumenti per agire sull'ambiente e la memoria per non ripartire ogni
volta da zero. Il ciclo osserva → ragiona → agisci si ripete fino alla
risposta.
```

Come mostra {numref}`fig-agente-anatomia`, il modello non tocca mai il mondo
direttamente: lo fa attraverso gli strumenti, e ogni azione torna indietro
come un'osservazione che rientra nel contesto. Il pezzo più sottile è proprio
questo, l'uso degli strumenti (in inglese **tool use**, ed è il nome che
troverai ovunque). Come fa un modello che sa solo *scrivere testo* a mettere
in moto un pezzo di programma? In informatica un pezzo di programma con un
nome, che fa una cosa quando qualcuno lo invoca, si chiama **funzione**: la
domanda, detta in gergo, è come faccia un modello a *chiamare una funzione*.

`````{tab} Elementare

Il trucco è che il modello non esegue niente: **scrive un bigliettino
d'ordine**. Immagina un cuoco chiuso in cucina che non può uscire in sala:
quando gli serve qualcosa scrive un ordine su un foglietto («portami due
uova») e lo passa a un cameriere. Il cameriere va, prende le uova, torna e le
posa sul bancone. Il cuoco non è mai uscito dalla cucina, ma le uova sono
arrivate.

Con un agente succede lo stesso. Il modello, invece delle uova, scrive
«cerca_sul_web(previsioni Roma domani)». Non è lui a navigare: il programma
che gli sta attorno legge quel foglietto, esegue davvero la ricerca e gli
riporta i risultati, che il modello ritrova nel contesto al giro dopo, come il
cuoco ritrova le uova sul bancone. Le mani sono di qualcun altro; al modello
resta il mestiere di decidere *cosa* ordinare.

`````

`````{tab} Superiore

Meccanicamente, una chiamata a strumento è **testo strutturato** che il
modello impara a produrre. Al modello si descrivono, nel prompt, gli strumenti
disponibili (nome, cosa fanno, quali argomenti accettano) di solito con uno
schema formale (spesso JSON). Quando decide di usarne uno, il modello non
esegue nulla: **emette** una stringa che rappresenta la chiamata, per esempio

```text
cerca_sul_web(query="previsioni meteo Roma domani")
```

Il *runtime* dell'agente intercetta questa stringa, la interpreta, esegue
davvero la funzione corrispondente nel codice ospite, e **appende
l'osservazione** (l'esito della funzione) al contesto. Al passo successivo il
modello vede la propria richiesta *e* la risposta, e prosegue il ragionamento.

Due conseguenze pratiche. Primo: la separazione è netta, il modello *propone*,
il runtime *dispone*; l'esecuzione vera (con i suoi permessi, i suoi limiti, i
suoi controlli di sicurezza) resta fuori dal modello, ed è lì che si mette il
freno alle azioni pericolose. Secondo: ogni chiamata è del testo che entra e
del testo che esce, e quindi consuma finestra di contesto; un loop lungo la
riempie in fretta, come vedremo parlando di context engineering.

`````

## Perché adesso

Tre dei quattro ingredienti appena elencati (qualcosa che decide, degli
strumenti, un ciclo che li mette in moto) non sono un'idea nuova.
L'intelligenza artificiale classica, quella fatta di regole scritte a mano da
un programmatore, costruiva agenti così già fra gli anni Sessanta e Settanta,
e fra poco ne vedremo due. Il quarto ingrediente, la memoria, ce l'avevano in
forma minima, e fra poco vedremo anche quella.

Perché allora gli agenti *basati su LLM* nascono solo ora? Le tre lettere
stanno per *large language model*, «grande modello di linguaggio», e sono la
sigla con cui d'ora in poi chiameremo il modello che completa il testo: quando
leggi «LLM» pensa sempre a quello. La risposta è una capacità che questi
modelli hanno acquisito da poco: capire ed eseguire una consegna scritta come
la scriveresti a una persona, cioè in **linguaggio naturale** (che è il modo in
cui i tecnici chiamano l'italiano, l'inglese e le altre lingue che parliamo,
per distinguerle dai linguaggi di programmazione).

`````{tab} Elementare

Prima, per far usare uno strumento a un programma, dovevi scrivergli tu, riga
per riga, *quando* e *come* usarlo: nessuna sorpresa era ammessa, tutto andava
previsto in anticipo. Era come istruire qualcuno che esegue alla lettera e non
capisce una parola fuori copione.

Gli LLM di oggi hanno imparato, durante l'addestramento, a *seguire
istruzioni* scritte come le scriveresti a una persona: «hai a disposizione una
ricerca web; usala quando ti serve un'informazione che non conosci». Non devi
più programmare ogni caso: descrivi lo strumento e l'obiettivo, e il modello
capisce da sé quando ha senso usarlo. È questa comprensione delle consegne
(non una nuova capacità di calcolo) ad aver reso possibili gli agenti.

`````

`````{tab} Superiore

Due proprietà, entrambe discusse nel capitolo sui Transformer, si combinano.
La prima è l’**instruction tuning**: la fase di post-training in cui il modello
viene addestrato su coppie *istruzione → buona risposta*, imparando a trattare
una consegna in linguaggio naturale come qualcosa da *eseguire*, non solo da
continuare. La seconda è l’**in-context learning**: la capacità, emersa con la
scala, di adattarsi a un compito descritto (magari con qualche esempio) nel
solo prompt, senza toccare i pesi. Messe insieme, rendono *eseguibile* una
consegna come «ecco gli strumenti a tua disposizione, usali per raggiungere
l'obiettivo»: il prompt diventa la specifica del comportamento dell'agente.

`````

Due avvertenze prima di andare avanti, e sono l'onestà su cui insiste il resto
del libro. La prima: gli agenti sono un campo **giovane e in rapido
movimento** {cite}`xi2023rise`. Non c'è una teoria consolidata sotto, ci sono
ricette che qualcuno ha provato e che sembrano funzionare (di una ricetta così,
che non garantisce niente ma spesso va, si dice che è un’**euristica**).

La seconda avvertenza è un problema strutturale: gli errori si **sommano lungo
il ciclo**. Se il modello azzecca una mossa nove volte su dieci, dieci mosse di
fila senza un solo inciampo gli riescono poco più di una volta su tre. Il conto
è quello che sembra: chiamiamo $p$ la probabilità di sbagliare un passo e $n$
il numero di passi, e la probabilità di attraversarli tutti senza errori vale
$(1-p)^n$, che con $p = 0{,}1$ e $n = 10$ fa $0{,}9^{10} \approx 0{,}35$. Il
conto vale finché ogni passo va per conto suo, senza che sbagliare il primo
renda più probabile sbagliare il secondo; nella realtà non è proprio così, e ci
torneremo nell'ultima sezione. Ma la sostanza tiene: non basta essere bravi a
un passo, bisogna esserlo per molti passi di seguito, ed è una delle ragioni
per cui compiti lunghi come quelli di SWE-bench {cite}`jimenez2024swebench`
restano difficili.

## Un antenato: i chatbot a regole

Vale la pena guardarsi indietro, perché l'idea di un sistema che percepisce,
decide e agisce non nasce con gli LLM. Nel capitolo sul Natural Language
Processing abbiamo incontrato i primi programmi capaci di sostenere una
conversazione, quelli che oggi chiamiamo **chatbot**.

Il primo è **ELIZA**, che negli anni Sessanta rispondeva rigirando le parole
dell'interlocutore con delle **espressioni regolari**, schemi scritti a mano
del tipo «se la frase contiene *mia madre*, rispondi *mi parli della sua
famiglia*». Il secondo è **GUS**, del 1977, che faceva l'agente di viaggio: conduceva la
conversazione riempiendo le caselle di un modulo (*dove*, *quando*, *quanti*)
con domande mirate, e a modulo completo prenotava. I sistemi fatti così si
chiamano **a modulo**, o *a frame* dalla parola inglese, e quel modulo mezzo
pieno era la loro piccola memoria: l'unica cosa che si portavano dietro da una
battuta all'altra.

Erano già agenti, a modo loro. Avevano una percezione, cioè quello che arriva
dall'esterno; avevano delle azioni, cioè le risposte da dare e la prenotazione
da fare; e in mezzo avevano una regola che, vista la situazione, sceglieva la
mossa successiva. Quella regola si chiama **politica** (in inglese *policy*),
ed è la stessa parola del capitolo sul reinforcement learning, l'apprendimento
per tentativi e ricompense.

`````{tab} Elementare

Il limite di quei sistemi era la **rigidità**. Un assistente a moduli sa fare
benissimo ciò che è previsto (prenota un volo, imposta una sveglia) ma un
millimetro fuori dai suoi moduli cade nel vuoto: non c'è nessuna casella da
riempire, e lui non sa che pesci prendere. Ogni comportamento era stato
scritto a mano da un programmatore, uno per uno.

L'agente basato su LLM generalizza quella stessa idea (percepire, decidere,
agire) ma sostituisce le regole scritte a mano con un motore linguistico
flessibile, capace di affrontare anche richieste che nessuno aveva previsto.
Il guadagno è la versatilità; il prezzo, come vedremo, è che diventa più
difficile prevedere e controllare esattamente cosa farà.

`````

`````{tab} Superiore

Il salto è da una **policy scritta a mano** a una **policy espressa in
linguaggio**. Nei sistemi a frame la logica di controllo era una macchina a
stati esplicita: slot tipizzati, una domanda per ogni slot vuoto, transizioni
codificate da un ingegnere. Robusta e prevedibile (nessuna risposta inventata,
errori localizzabili, successo misurabile) ma incapace di uscire dallo spazio
di stati previsto.

L'agente LLM tiene l'ossatura (uno stato, una policy, delle azioni) ma la
policy non è più un diagramma di flusso: è un modello di linguaggio orientato
da un prompt. Lo spazio delle azioni si allarga enormemente, e con esso la
copertura dei casi non previsti; in cambio si perde parte del controllo e
della prevedibilità che rendevano affidabili i sistemi a frame. Non è un
rimpiazzo indolore: è uno scambio, e i due mondi convivono ancora; spesso un
agente flessibile viene racchiuso dentro binari rigidi proprio per riottenere
un po’ di quelle garanzie.

`````

## Come è organizzato il capitolo

Le prossime quattro sezioni sviluppano, una alla volta, le parti che qui
abbiamo solo montato insieme.

- **Il ciclo dell'agente**, come un modello chiama davvero gli strumenti e
  compone le azioni in sequenza: lo schema ragiona-agisci-osserva in pratica,
  con il codice che lo fa girare.
- **RAG avanzato**, cioè il recupero dei documenti giusti *prima* di
  rispondere, oltre la forma base già vista nel capitolo sui Transformer:
  interrogazioni multiple, ri-ordinamento dei risultati, recupero guidato
  dall'agente stesso.
- **Il contesto è l'interfaccia**, cioè l'arte di riempire bene la finestra di
  contesto (in gergo, il *context engineering*): cosa metterci e cosa lasciare
  fuori, cosa comprimere, cosa far sopravvivere da un passo all'altro quando lo
  spazio è poco.
- **Architetture e valutazione**, cioè come si compongono più agenti in un
  sistema, e il problema aperto di dare loro un voto. È lì che finisce il
  discorso su SWE-bench {cite}`jimenez2024swebench` cominciato qui sopra. Il
  problema gemello, come si dà un voto a un modello che si limita a rispondere
  quando non esiste una risposta giusta sola, ha invece un posto suo più
  avanti: il capitolo su **MLOps**, che è il mestiere di portare un modello dal
  laboratorio all'uso di tutti i giorni.

Sei punti da portarsi via prima di andare avanti.

`````{tab} Elementare

```{admonition} Da ricordare
:class: important
- Un **modello** indovina come continua un testo; un **agente** è il *sistema*
  che gli mette attorno delle mani e un metodo di lavoro, così che non risponda
  soltanto ma **agisca**: cerchi, esegua, prenoti, corregga il codice. Il
  modello è il motore, l'agente è l'automobile.
- Il cuore è un **ciclo** che si ripete, **osserva → ragiona → agisci →
  osserva**, con il modello nel ruolo di chi sceglie la mossa. Farlo «ragionare
  ad alta voce» prima di agire (la **catena di ragionamento**, in inglese
  *chain-of-thought* {cite}`wei2022chain`) aiuta, ma soprattutto sui conti e
  sui problemi di logica {cite}`sprague2025cot`.
- I quattro ingredienti: il **modello** (il cervello), gli **strumenti** (le
  mani: cercare sul web, far girare del codice, interrogare un servizio
  esterno), il **ciclo di controllo** (guarda, agisci, riguarda) e la
  **memoria** (quello che tiene sott'occhio adesso, più un archivio esterno). Il
  modello scrive il bigliettino d'ordine; il programma che gli sta attorno lo
  esegue.
- Gli agenti nascono **adesso** perché i modelli hanno imparato a capire una
  consegna scritta a parole: basta descrivere lo strumento e l'obiettivo,
  invece di programmare ogni caso. È però un campo giovane {cite}`xi2023rise`,
  e i piccoli errori si sommano lungo il ciclo: nove mosse giuste su dieci
  vogliono dire arrivare in fondo a dieci mosse poco più di una volta su tre.
- I **chatbot a regole** del capitolo sul linguaggio (ELIZA, i sistemi a
  modulo) sono gli antenati rigidi: bravissimi dentro il previsto, muti fuori.
  L'agente guadagna versatilità e perde prevedibilità: è uno scambio, non un
  regalo.
- Nel resto del capitolo: come il modello chiama davvero gli strumenti, come si
  recuperano i documenti giusti prima di rispondere, come si riempie bene la
  finestra di contesto, e come si dà un voto a un agente. Su quest'ultimo punto
  SWE-bench {cite}`jimenez2024swebench` è il banco di prova su compiti veri, e
  vedremo che perfino un banco di prova va messo alla prova.
```

`````

`````{tab} Superiore

```{admonition} Da ricordare
:class: important
- Un **modello** predice la continuazione di un testo; un **agente** è il
  *sistema* che gli mette attorno strumenti e un ciclo di controllo, così che
  non risponda soltanto ma **agisca**: cerchi, esegua, prenoti, corregga il
  codice.
- Il cuore è un ciclo **osserva → ragiona → agisci → osserva**, con l'LLM nel
  ruolo di *policy*: sceglie l'azione dato il contesto. Farlo «ragionare ad alta
  voce» (**chain-of-thought** {cite}`wei2022chain`) aiuta, ma i guadagni
  misurati si concentrano su matematica e ragionamento simbolico
  {cite}`sprague2025cot`.
- I quattro ingredienti: il **modello** (il cervello), gli **strumenti** (le
  mani: web, codice, API), il **loop di controllo** (percezione-azione) e la
  **memoria** (contesto + memoria esterna). Il modello *propone* le azioni; il
  runtime le *esegue*.
- Gli agenti diventano possibili **adesso** perché gli LLM istruiti sanno
  seguire una consegna in linguaggio naturale: **instruction tuning** e
  **in-context learning** rendono eseguibile «usa questo strumento». È però
  un'area giovane {cite}`xi2023rise`, e gli errori si accumulano lungo il
  loop.
- I **chatbot a regole** del capitolo NLP (ELIZA, sistemi a frame) sono gli
  antenati rigidi: l'agente LLM generalizza la stessa idea con un motore
  linguistico flessibile, guadagnando versatilità e perdendo prevedibilità.
- Nel resto del capitolo: **tool use**, **RAG avanzato**, **context
  engineering**, **architetture e valutazione**; di quest'ultima, SWE-bench
  {cite}`jimenez2024swebench` è un banco di prova su compiti reali.
```

`````
