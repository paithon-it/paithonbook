# Agenti: quando i modelli linguistici agiscono

Abbiamo lasciato il modello di linguaggio così: gli scrivi una domanda, lui
completa il testo con la risposta più probabile. Utile, ma passivo: un oracolo
dietro una fessura, che sa *dire* e non può *fare*. Se gli chiedi che tempo
farà domani a Roma, un modello addestrato mesi fa non ne ha idea: quel dato
non esisteva quando ha studiato. Può inventarlo, con la stessa sicurezza con
cui direbbe una cosa vera, ma non può *guardare* il meteo.

Poi, tra il 2023 e il 2024, qualcosa cambia registro. Compaiono sistemi che
non si limitano a rispondere: cercano sul web una notizia di ieri, eseguono un
pezzo di codice per controllare se gira, compilano un modulo, prenotano,
aprono una richiesta di modifica (una *pull request*) su un progetto software
vero. L'esempio più spietato di questo salto è **SWE-bench**
{cite}`jimenez2024swebench` (dall'inglese *software engineering*, ingegneria
del software, più *bench*, banco di prova): 2.294 segnalazioni di errore
(*issue*) prese da progetti reali su GitHub, il sito dove i programmatori
tengono il codice dei loro progetti e si scambiano correzioni. Al sistema si
dà la segnalazione del bug e il codice del progetto; deve produrre una
modifica che lo risolva davvero. A giudicarlo non c'è una persona, ma i
**test** del progetto: dei controlli automatici che il computer fa girare per
vedere se il programma funziona ancora. Sono gli stessi test che, quando il
bug fu risolto per davvero, dissero «a posto» alla correzione scritta da uno
sviluppatore in carne e ossa. È un compito che
nessun completamento di testo, per quanto fluente, chiude in un colpo solo:
bisogna trovare i file giusti, provare, sbagliare, rileggere il messaggio
d'errore, correggere. I primi sistemi ci riuscivano in una **piccola
frazione** dei casi (pochi punti percentuali) e proprio quel numero basso è la
notizia: dà la prima cifra pubblica della distanza fra dire e fare. Che poi
sia una misura *pulita* di quella distanza è un'altra questione, e ci
torneremo nell'ultima sezione: un benchmark misura anche se stesso.

Prima di andare avanti, mettiamo un paletto che vale per tutto il capitolo. Un
**modello** è la rete che, dato un testo, ne predice la continuazione: quello
che abbiamo studiato nel capitolo sui Transformer. Un **agente** è un
*sistema* costruito attorno a un modello: un programma che percepisce
l'ambiente, lascia che il modello decida la mossa successiva, la esegue
davvero, osserva com'è andata e ricomincia. Il modello è il motore; l'agente è
l'automobile, con volante, ruote e strada. Questo capitolo è dedicato
all'automobile.

## Dal completare testo all'agire

```{figure} ../figures/rag-lewis-2020.svg
:name: fig-rag-lewis
:alt: "Schema del RAG originale: la domanda entra in un cercatore (retriever) che consulta un indice costruito su Wikipedia e ne estrae i passaggi più rilevanti; domanda e passaggi entrano insieme nel generatore, che scrive la risposta. Cercatore e generatore sono addestrati insieme, tenendo però fisso l'indice dei documenti: per questo l'archivio si può sostituire senza riaddestrare."
:width: 96%

Il primo passo fuori dai pesi. Il disegno ha tre pezzi: chi cerca nell'archivio
(il *retriever*, il cercatore), l'archivio stesso e chi scrive la risposta (il
generatore). La conoscenza non sta più solo nel modello: una parte vive in un
archivio che si può aggiornare senza riaddestrare niente.
```

{numref}`fig-rag-lewis` è il precedente diretto di tutto questo capitolo, ed è
utile vederlo prima di parlare di agenti: è la **RAG** di Lewis e colleghi
{cite}`lewis2020retrieval`, sigla che sta per *Retrieval-Augmented Generation*,
«generazione aumentata dal recupero», e che nel capitolo sui Transformer
abbiamo già costruito in miniatura. Un modello che consulta un archivio fa già
una cosa che il completamento puro non fa: sospende la risposta, va a prendere
qualcosa fuori di sé, e solo dopo conclude.

Perché il completamento di testo, da solo, non basta? Perché rispondere è un
atto unico e chiuso, mentre agire nel mondo è un processo: richiede più mosse
in sequenza, ognuna decisa alla luce di come è andata la precedente. Comprare
un biglietto significa cercare i treni, confrontare gli orari, scegliere,
pagare, ricevere conferma, e se a metà strada il treno scelto risulta pieno,
tornare indietro e riprovare. Un oracolo che sputa una risposta e si spegne
non può fare niente di tutto questo.

`````{tab} Elementare

Pensa alla differenza tra un consulente e un assistente. Il **consulente** ti
dà consigli a parole: «per andare a Milano ti conviene il treno delle 9, poi
prenota un hotel in centro». Ottimo, ma il lavoro resta tutto a te: sei tu che
apri il sito, digiti le date, paghi. L'**assistente**, invece, le cose le
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
e l'«azione» è, tipicamente, l'invocazione di uno strumento oppure la risposta
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
ragionamento (*«per risolvere il bug devo prima leggere il file dove viene
sollevata l'eccezione»*) e solo dopo sceglie la mossa. È l'idea del
**chain-of-thought** {cite}`wei2022chain`, la catena di ragionamento che nel
capitolo sui Transformer abbiamo visto migliorare i compiti di ragionamento.
Conviene però dire subito dove il guadagno è stato misurato davvero, perché è
più stretto di come lo si racconta di solito: una meta-analisi su oltre cento
lavori lo trova concentrato sui compiti **matematici e simbolici**, e piccolo
altrove {cite}`sprague2025cot`. In un agente il pensiero scritto serve
soprattutto a un'altra cosa, dare al modello un posto dove annotare a che
punto è del compito prima di scegliere la mossa: è il collante fra il pensare
e il fare, non una cura generale.

## L'anatomia di un agente

Smontiamo l'automobile. Al di là delle mille varianti, ogni agente ha quattro
ingredienti, e conviene tenerli distinti perché ognuno ha problemi suoi.

- Il **modello** è il cervello: legge il contesto, ragiona, decide la prossima
  azione. È l'unico pezzo che «pensa»; tutto il resto è impalcatura attorno.
- Gli **strumenti** (*tool*) sono le mani: una ricerca sul web, un programma
  che esegue del codice al posto suo, la chiamata a un servizio esterno
  (un'**API**, dall'inglese *application programming interface*: la presa
  elettrica con cui un programma ne interroga un altro), un'interrogazione a
  un archivio di dati. Sono ciò che permette all'agente di *toccare* il mondo:
  di leggere dati freschi e di produrre effetti.
- Il **loop di controllo** è il metodo di lavoro: il programma che alterna
  percezione e azione, passa il contesto al modello, esegue l'azione scelta,
  raccoglie il risultato e decide se continuare o fermarsi.
- La **memoria** è ciò che l'agente si porta dietro. Nel breve termine è la
  finestra di contesto: la memoria di lavoro, limitata, che abbiamo studiato
  parlando di contesti lunghi e di quel segnalibro con cui il modello evita di
  rileggere da capo ciò che ha già letto (la **KV cache**, dalle iniziali di
  *key* e *value*, i due ingredienti dell'attenzione che vengono messi da
  parte). Nel lungo termine è una memoria
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
il tool use. Come fa un modello che sa solo *scrivere testo* a mettere in moto
un pezzo di programma (in gergo, a *chiamare una funzione*)?

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

Le tre idee (un motore linguistico, degli strumenti, un ciclo) non sono nuove.
L'AI classica costruiva agenti già negli anni Settanta. Perché allora gli
agenti *basati su LLM* (dall'inglese *large language model*, «grande modello
di linguaggio»: è la sigla con cui d'ora in poi chiameremo il modello che
completa il testo) nascono solo ora? La risposta sta in una capacità che i
modelli hanno acquisito da poco: capire ed eseguire una consegna scritta in
linguaggio naturale.

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
La prima è l'**instruction tuning**: la fase di post-training in cui il modello
viene addestrato su coppie *istruzione → buona risposta*, imparando a trattare
una consegna in linguaggio naturale come qualcosa da *eseguire*, non solo da
continuare. La seconda è l'**in-context learning**: la capacità, emersa con la
scala, di adattarsi a un compito descritto (magari con qualche esempio) nel
solo prompt, senza toccare i pesi. Messe insieme, rendono *eseguibile* una
consegna come «ecco gli strumenti a tua disposizione, usali per raggiungere
l'obiettivo»: il prompt diventa la specifica del comportamento dell'agente.

È doverosa però l'onestà su cui insiste il resto del libro. Gli agenti basati
su LLM sono un'area **giovane e in rapido movimento** {cite}`xi2023rise`, senza
un impianto teorico consolidato e con più euristiche che garanzie. E c'è un
problema strutturale: gli errori si **accumulano lungo il loop**. Se a ogni
passo il modello ha una probabilità $p$ di sbagliare la mossa, e i passi sono
indipendenti, la probabilità di attraversare un compito di $n$ passi senza un
solo errore scala come $(1-p)^n$, che precipita al crescere di $n$. È una
delle ragioni per cui
compiti lunghi come quelli di SWE-bench {cite}`jimenez2024swebench` restano
difficili: non basta essere bravi a un passo, bisogna esserlo per molti passi
di fila.

`````

## Un antenato: i chatbot a regole

Vale la pena guardarsi indietro, perché l'idea di un sistema che percepisce,
decide e agisce non nasce con gli LLM. Nel capitolo sul Natural Language
Processing abbiamo incontrato i primi sistemi di dialogo: **ELIZA**, che negli
anni Sessanta rispondeva rigirando le parole dell'interlocutore con delle
**espressioni regolari** (schemi scritti a mano del tipo «se la frase contiene
*mia madre*, rispondi *mi parli della sua famiglia*»), e i **sistemi a frame**
come GUS, che conducevano una conversazione riempiendo le caselle di un modulo
(*dove*, *quando*, *quanti*) con domande mirate. Erano già, a modo loro,
agenti: avevano una percezione (quello che arriva dall'esterno), una
**politica** (in inglese *policy*: la regola che, vista la situazione, sceglie
la mossa successiva, ed è la stessa parola del capitolo sul reinforcement
learning) e delle azioni (le risposte, o la prenotazione a modulo completo).

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
un po' di quelle garanzie.

`````

## Come è organizzato il capitolo

Le prossime quattro sezioni sviluppano, una alla volta, le parti che qui
abbiamo solo montato insieme.

- **Agenti e tool use**, come un modello chiama davvero gli strumenti e
  compone le azioni in sequenza: il pattern ragiona-agisci-osserva in pratica,
  con il codice del loop.
- **RAG avanzato**, cioè il recupero dei documenti giusti *prima* di
  rispondere, oltre la forma base già vista nel capitolo sui Transformer:
  interrogazioni multiple, ri-ordinamento dei risultati, recupero guidato
  dall'agente stesso.
- **Context engineering**, l'arte di riempire la finestra di contesto: cosa
  mettere e cosa lasciare fuori, cosa comprimere, cosa far sopravvivere da un
  passo all'altro quando la memoria di lavoro è stretta.
- **Architetture e valutazione**, come si compongono più agenti in un sistema,
  e il problema aperto di dare loro un voto: valutare un agente che agisce,
  non solo un testo che risponde, è difficile quanto (e più di) dare un voto a
  una risposta libera, quella per cui non esiste una soluzione unica, di cui
  parleremo più avanti nel capitolo su MLOps, alla sezione LLMOps. SWE-bench
  {cite}`jimenez2024swebench` è un esempio di come si prova a farlo su compiti
  reali.

Ecco, in sei righe, quello che vale la pena portarsi via da questa prima
sezione.

`````{tab} Elementare

```{admonition} Da ricordare
:class: important
- Un **modello** indovina come continua un testo; un **agente** è il *sistema*
  che gli mette attorno delle mani e un metodo di lavoro, così che non risponda
  soltanto ma **agisca**: cerchi, esegua, prenoti, corregga il codice. Il
  modello è il motore, l'agente è l'automobile.
- Il cuore è un giro che si ripete, **osserva → ragiona → agisci → osserva**,
  con il modello nel ruolo di chi sceglie la mossa. Farlo «ragionare ad alta
  voce» prima di agire (la **catena di ragionamento**, in inglese
  *chain-of-thought* {cite}`wei2022chain`) aiuta, ma soprattutto sui conti e
  sui problemi di logica {cite}`sprague2025cot`.
- I quattro ingredienti: il **modello** (il cervello), gli **strumenti** (le
  mani: cercare sul web, far girare del codice, interrogare un servizio
  esterno), il **giro di controllo** (guarda, agisci, riguarda) e la **memoria**
  (quello che tiene sott'occhio adesso, più un archivio esterno). Il modello
  scrive il bigliettino d'ordine; il programma che gli sta attorno lo esegue.
- Gli agenti nascono **adesso** perché i modelli hanno imparato a capire una
  consegna scritta a parole: basta descrivere lo strumento e l'obiettivo,
  invece di programmare ogni caso. È però un campo giovane {cite}`xi2023rise`,
  e i piccoli errori si sommano lungo il giro.
- I **chatbot a regole** del capitolo sul linguaggio (ELIZA, i sistemi a
  moduli) sono gli antenati rigidi: bravissimi dentro il previsto, muti fuori.
  L'agente guadagna versatilità e perde prevedibilità: è uno scambio, non un
  regalo.
- Nel resto del capitolo: come il modello chiama davvero gli strumenti, come si
  recuperano i documenti giusti prima di rispondere, come si riempie bene la
  finestra di contesto, e come si dà un voto a un agente. Su quest'ultimo punto
  SWE-bench {cite}`jimenez2024swebench` è il banco di prova su compiti veri.
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
