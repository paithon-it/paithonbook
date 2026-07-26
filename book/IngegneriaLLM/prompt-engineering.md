# Prompt engineering: il singolo messaggio

Cerca «prompt segreti» e troverai un mercato intero: raccolte di frasi
«che sbloccano il 90% delle capacità nascoste del modello», corsi che promettono
di farti diventare *prompt engineer* in un weekend, immagini di istruzioni
lunghe una pagina spacciate come formule alchemiche. È l'equivalente moderno
delle parole magiche: si crede che esista *la* frase giusta, e che chi la
conosce comandi la macchina. Vale la pena sgombrare il campo subito. Il prompt
non è un incantesimo; è il **primo livello** con cui si programma un modello di
linguaggio — il più immediato, quello che vedi e scrivi nella casella della
chat. Sopra di esso, come abbiamo anticipato aprendo questo capitolo, ci sono il
contesto e il loop; ma è da qui che si comincia, perché è qui che nascono quasi
tutti i malintesi.

Nel capitolo sugli Agenti, nella sezione sul context engineering, abbiamo già
ridimensionato la parola «prompt» e mostrato che nelle applicazioni serie è un
**artefatto strutturato**, montato dal programma prima di interpellare il
modello; lì abbiamo anche scritto la formula dell'*in-context learning*, il
meccanismo per cui qualche esempio nel contesto orienta la risposta senza
toccare un peso. Non la ripetiamo. Questa sezione parte da lì e guarda dentro il
singolo messaggio: com'è fatto, quali leve ha, e quali tecniche — dagli esempi
al ragionamento a voce alta — spostano davvero la qualità della risposta. Il
metro di riferimento è la *Prompt Engineering Guide* di DAIR.AI, una delle
raccolte più curate sull'argomento: la usiamo come mappa, rielaborandola con
esempi e voce nostri.

## L'anatomia di un prompt

Prima di ottimizzare qualcosa conviene saperlo smontare. Un prompt ben fatto,
anche il più breve, ha di solito quattro parti — non tutte sempre presenti, ma
utili da distinguere.

`````{tab} Elementare

Pensa a come si affida un compito a qualcuno per iscritto. C'è **l'ordine**
(«traduci in inglese questa frase»): dice cosa fare. C'è **lo sfondo** che serve
per farlo bene («è il messaggio di un cliente arrabbiato, mantieni un tono
formale»): il contesto. C'è **il materiale** su cui lavorare (la frase da
tradurre): il dato d'ingresso. E c'è **il segnale di via** («Traduzione:»), il
punto in cui lasci la penna al modello perché continui da lì. Quattro pezzi:
cosa fare, con quale sfondo, su cosa, e dove attaccare a scrivere. Un buon
prompt li tiene distinti invece di impastarli in un'unica frase confusa.

`````

`````{tab} Superiore

Le quattro componenti canoniche sono: **istruzione** (il compito: «riassumi»,
«classifica», «traduci»); **contesto** (informazioni di sfondo o vincoli che
condizionano la risposta: il tono, il pubblico, regole da rispettare, eventuali
passaggi recuperati); **dato d'ingresso** (l'input specifico su cui operare);
**indicatore d'output** (il segnale che innesca e formatta la generazione — un
`Traduzione:` finale, l'inizio di un blocco JSON, un'etichetta attesa). Non è
uno schema rigido: molti prompt utili contengono solo istruzione e input. Ma la
distinzione è operativa, perché ciascuna parte si può isolare e migliorare da
sola — e perché separare nettamente **istruzione** e **dato** è una difesa
contro un problema concreto, la *prompt injection*, che vedremo tra poco.

`````

Nelle API moderne queste parti vivono dentro un formato a **ruoli**, ereditato
dalle interfacce di chat. Ogni messaggio ha un mittente dichiarato, e i tre
ruoli fondamentali sono *system*, *user* e *assistant*.

`````{tab} Elementare

Immagina un copione teatrale a tre voci. Il **regista** (*system*) parla una
volta sola, prima che si alzi il sipario, e dà le direttive di fondo: «sei un
assistente cortese, non promettere mai rimborsi, rispondi in italiano». Il
**pubblico** (*user*) è chi fa le domande di volta in volta. L'**attore**
(*assistant*) è il modello, che risponde. La conversazione è l'alternarsi di
domande del pubblico e risposte dell'attore, ma le direttive del regista restano
valide per tutta la recita, sopra ogni singolo scambio. Sapere «chi parla»
conta: il modello dà più peso al regista che al pubblico, ed è così che
un'applicazione impone regole che l'utente non dovrebbe poter scavalcare.

`````

`````{tab} Superiore

Il messaggio **system** fissa il comportamento invariante — ruolo, tono,
politiche, formato — e resta identico a ogni turno: è la spina dorsale su cui il
programma costruisce il resto. I messaggi **user** contengono le richieste
dell'utente finale; i messaggi **assistant** contengono le risposte del modello,
e riinserire i turni passati è ciò che dà continuità alla conversazione. La
gerarchia non è puramente convenzionale: i modelli sono addestrati (via RLHF e
tecniche affini) a dare priorità alle istruzioni di sistema su quelle
dell'utente. Questa gerarchia è però *morbida*, non una barriera crittografica:
un utente abile può tentare di aggirarla, ed è il nodo del *jailbreak*. Nel
formato chat i vecchi esempi *few-shot* si possono anche esprimere come turni
`user`/`assistant` fittizi che precedono la richiesta reale — esempi
«recitati» che condizionano lo stile della risposta.

`````

## Le due manopole del campionamento

Il prompt decide *cosa* chiedi; due impostazioni decidono *come* il modello
sceglie le parole mentre risponde: la **temperatura** e il **top_p**. Nel
capitolo sui Transformer abbiamo studiato la matematica del *decoding* — come da
una distribuzione di probabilità sulla prossima parola si campiona un token, e
come la temperatura riscala quella distribuzione. Qui non la ripetiamo: ci serve
l'intuizione operativa, quella che usi davvero quando regoli una chiamata.

`````{tab} Elementare

Immagina che il modello, a ogni parola, abbia un ventaglio di alternative con
probabilità diverse. La **temperatura** è quanto lo lasci «osare». A temperatura
bassa (vicina a 0) sceglie quasi sempre l'opzione più probabile: risposte
prevedibili, ripetibili, adatte quando vuoi un fatto o un'estrazione precisa —
chiedi due volte, ottieni la stessa cosa. A temperatura alta pesca più
volentieri anche tra le alternative meno probabili: risposte più varie e
sorprendenti, adatte a scrivere, inventare, fare brainstorming — ma anche più a
rischio di sbandare. Il **top_p** è una manopola imparentata: invece di alzare
il rumore, restringe il ventaglio ai candidati più probabili la cui probabilità
somma, poniamo, al 90%, e ignora la coda. Regola pratica: temperatura bassa per
fatti e codice, temperatura più alta per creatività; e cambia una manopola per
volta, non tutte e due insieme.

`````

`````{tab} Superiore

La **temperatura** $T$ riscala i logit prima della softmax: $T \to 0$ concentra
la massa sull'argmax (*greedy*, deterministico a meno di pareggi), $T > 1$
appiattisce la distribuzione aumentando l'entropia del campionamento. Il
**top_p** (*nucleus sampling*, Holtzman et al., 2020) tronca invece la
distribuzione al più piccolo insieme di token la cui probabilità cumulata
raggiunge la soglia $p$, ridistribuendo la massa su quel nucleo: adatta
dinamicamente il numero di candidati alla forma della distribuzione, cosa che un
semplice *top-k* fisso non fa. I due parametri agiscono su assi diversi — uno
sulla temperatura della distribuzione, l'altro sul supporto ammesso — e la
guida DAIR.AI raccomanda di regolarne **uno solo** per volta, tenendo l'altro al
default, per non confondere gli effetti. La derivazione completa e il confronto
con *top-k* e *beam search* sono nel capitolo sui Transformer.

`````

## Zero-shot, one-shot, few-shot: gli esempi come condizionamento

La leva più potente del prompt engineering è anche la più semplice: **mostrare
al modello degli esempi svolti**. La differenza tra chiedere a freddo e chiedere
dopo aver mostrato due o tre casi risolti è spesso la differenza tra una risposta
sbagliata e una giusta.

- **Zero-shot**: solo l'istruzione, nessun esempio. «Classifica il sentiment di
  questa recensione». Funziona sorprendentemente bene sui compiti comuni, perché
  il modello li ha già visti a milioni durante l'addestramento.
- **One-shot / few-shot**: si antepone alla richiesta uno (o *k*) esempi
  completi, coppie ingresso → uscita corretta. Il modello non impara nulla di
  nuovo — nessun peso cambia — ma *inferisce* dallo schema cosa gli stai
  chiedendo e in che formato lo vuoi.

`````{tab} Elementare

È come insegnare un gioco nuovo a un amico. Puoi spiegargli le regole a parole
(zero-shot) e sperare che afferri. Oppure gli mostri una mano giocata: «guarda,
con queste carte si fa così». Dopo due o tre mani d'esempio ha capito il ritmo,
il formato, cosa conta — e gioca da solo. Gli esempi non gli hanno cambiato il
cervello: gli hanno mostrato il *pattern*. Col modello è identico. Se voglio che
etichetti frasi come positive o negative, gliene mostro qualcuna già etichettata:

```text
Recensione: "Cibo ottimo, servizio lento." → Sentiment: neutro
Recensione: "Mai più in questo posto." → Sentiment: negativo
Recensione: "Esperienza fantastica, torneremo!" → Sentiment: positivo
Recensione: "Prezzi alti ma ne vale la pena." → Sentiment:
```

Il modello, vedendo lo schema, completa l'ultima riga con «positivo». Nessuno
gli ha spiegato cos'è il sentiment: gliel'hanno mostrato tre volte.

`````

`````{tab} Superiore

Il *few-shot prompting* è la manifestazione più diretta dell'**in-context
learning**, la capacità — documentata su larga scala da Brown e colleghi con
GPT-3 {cite}`brown2020language` — di apprendere un compito dai soli esempi
presenti nel contesto, senza fine-tuning. La formalizzazione (la stima
$\arg\max_y P(y \mid I, (x_1,y_1),\dots,(x_k,y_k), x)$) è quella già vista nel
capitolo sugli Agenti: qui basti ricordare che gli esempi agiscono come
**condizionamento**, spostando la distribuzione condizionata del modello verso
lo stile e il formato mostrati, non come dati d'addestramento. Alcune
avvertenze empiriche contano nella pratica: la **scelta** degli esempi, il loro
**ordine** e persino il **formato** dell'etichetta influenzano il risultato;
gli esempi vanno bilanciati tra le classi per non indurre un *bias* verso quella
più frequente; e oltre una manciata di esempi il rendimento marginale cala,
mentre il costo in token cresce. Per i compiti che richiedono *ragionamento*, i
soli esempi spesso non bastano — ed è qui che entra la catena di pensiero.

`````

## Far ragionare a voce alta: chain-of-thought

Chiedi a un modello «Quanto fa 17 × 24?» e potresti ricevere un numero secco,
spesso sbagliato. Chiedigli di **mostrare i passaggi** e la musica cambia: se
scrive «17 × 24 = 17 × 20 + 17 × 4 = 340 + 68 = 408», arriva alla risposta
giusta molto più spesso. È l'idea della **chain-of-thought** (Wei et al., 2022)
{cite}`wei2022chain`: far generare al modello i passaggi intermedi del
ragionamento *prima* della conclusione.

`````{tab} Elementare

Prova a risolvere a mente «se ho 3 scatole da 12 mele e ne regalo 8, quante me
ne restano?». Se ti costringi a rispondere di getto puoi sbagliare; se lo dici a
voce — «3 per 12 fa 36, meno 8 fa 28» — quasi non sbagli. Scrivere i passaggi ti
obbliga a farne uno per volta, e ognuno è facile. Il modello funziona uguale:
se gli chiedi solo il risultato, tira a indovinare in un colpo; se gli chiedi di
ragionare passo per passo, spezza il problema in pezzi piccoli e ci inciampa
molto meno. Non è più «intelligente»: sta solo pensando ad alta voce invece che
in silenzio.

`````

`````{tab} Superiore

La chain-of-thought induce il modello a produrre una sequenza di passi
intermedi $z_1, \dots, z_m$ prima della risposta finale $\hat{y}$, così che la
generazione condizioni ogni passo sui precedenti. Wei e colleghi la ottengono
con esempi *few-shot* in cui la risposta è mostrata *insieme al ragionamento*
che la produce; il guadagno è marcato sui compiti aritmetici, di senso comune e
simbolici, e — dato interessante — **emerge con la scala**: sui modelli piccoli
la CoT aiuta poco o nulla, sui grandi produce salti netti di accuratezza.
Esiste anche una variante che elimina del tutto gli esempi: Kojima e colleghi
{cite}`kojima2022zeroshot` mostrano che basta aggiungere alla domanda una
singola frase-innesco — l'ormai celebre «*Let's think step by step*»,
«ragioniamo passo per passo» — per attivare un ragionamento a più passi anche in
**zero-shot**. Una riga di testo, nessun esempio, e su diversi benchmark di
ragionamento l'accuratezza sale di parecchi punti. Vista con gli occhi del
capitolo sugli Agenti, la CoT è anche *context engineering*: si spende
deliberatamente parte del budget in token di «pensiero» per comprare qualità.

`````

## Molte teste sono meglio di una: self-consistency

La catena di ragionamento ha un tallone d'Achille: è **una sola** catena. Se il
modello imbocca la strada sbagliata al primo passo, la trascina fino in fondo
con sicurezza. Wang e colleghi {cite}`wang2023selfconsistency` propongono un
rimedio tanto semplice quanto efficace: la **self-consistency**.

`````{tab} Elementare

Se un problema difficile lo dai a dieci persone diverse e otto arrivano allo
stesso numero, quel numero è probabilmente giusto — anche se ognuna ci è
arrivata per una strada un po' diversa. La self-consistency fa esattamente
questo con un solo modello: gli fai risolvere lo stesso problema **più volte**,
con un pizzico di casualità (temperatura non nulla) così che ogni volta ragioni
in modo leggermente diverso, e poi tieni la risposta che compare **più spesso**.
Le strade sbagliate tendono a sbagliare ciascuna a modo suo e si disperdono;
quella giusta viene ritrovata da più catene e vince per numero. È il voto di
maggioranza applicato al ragionamento.

`````

`````{tab} Superiore

La self-consistency sostituisce il *decoding* greedy della chain-of-thought con
un procedimento in tre tempi: (1) si campionano $N$ catene di ragionamento
indipendenti con temperatura $T > 0$; (2) da ciascuna si estrae la **risposta
finale**, scartando i passaggi intermedi; (3) si **marginalizza** sul
ragionamento tenendo la risposta di maggioranza,
$\hat{y} = \arg\max_{y} \sum_{i=1}^{N} \mathbb{1}[\,a_i = y\,]$, dove $a_i$ è la
risposta della $i$-esima catena. L'intuizione statistica: le derivazioni
corrette tendono a convergere sulla stessa risposta, mentre gli errori sono
idiosincratici e si sparpagliano, così il voto le premia. Il metodo migliora
sensibilmente l'accuratezza su benchmark di ragionamento aritmetico e logico
rispetto alla singola catena; il prezzo è lineare — $N$ generazioni invece di
una, quindi $N$ volte il costo in token e latenza. È un compromesso di puro
context/compute engineering: si compra affidabilità spendendo campioni.

`````

L'aggregazione per voto è banale da scrivere, e vale la pena vederla in puro
Python per capire quanto sia poco «magica»:

```python
from collections import Counter

def voto_di_maggioranza(risposte):
    """Data una lista di risposte finali campionate, restituisce la piu'
    frequente. A parita' di voti vince quella incontrata per prima, cosi'
    il risultato e' deterministico (Counter conserva l'ordine d'inserimento)."""
    conteggio = Counter(risposte)
    risposta, voti = conteggio.most_common(1)[0]
    return risposta, voti, len(risposte)

# Cinque catene di ragionamento indipendenti sulla stessa domanda:
# di ognuna teniamo solo la risposta finale (i passaggi sono stati scartati).
campioni = ["18", "18", "21", "18", "22"]

risposta, voti, totale = voto_di_maggioranza(campioni)
print(f"Risposta scelta: {risposta} ({voti}/{totale} voti)")
# -> Risposta scelta: 18 (3/5 voti)
```

Tre catene su cinque dicono «18», e quella vince: le due dissenzienti sbagliano
ciascuna a modo suo e non fanno numero. L'idea si spinge oltre la singola catena
lineare: invece di campionare catene *indipendenti* e votare alla fine, si
possono esplorare i pensieri intermedi come i rami di un **albero**, valutandoli
e tornando indietro dai rami che non promettono — è il **Tree of Thoughts**
{cite}`yao2023tree`, che abbiamo già incontrato nel capitolo sugli Agenti e che
qui basta richiamare: stessa filosofia (spendere più calcolo per ragionare
meglio), struttura di ricerca più ricca.

## Chiedere una risposta che il programma sappia leggere

Finché il lettore è un umano, va bene la prosa. Ma se la risposta del modello
deve essere consumata da **codice a valle** — salvata in un database, passata a
un'altra funzione, mostrata in un'interfaccia — serve una forma prevedibile.
La leva è chiedere esplicitamente un **output strutturato**: «rispondi con un
oggetto JSON con i campi `sentiment` (positivo/neutro/negativo) e `motivo`
(stringa)». Meglio ancora se si mostra un esempio del formato voluto e si vieta
qualsiasi testo attorno. Molte API oggi supportano una modalità *JSON* o uno
*schema* imposto che vincola la generazione a produrre solo output valido,
togliendo di mezzo il problema alla radice; dove non è disponibile, un esempio
few-shot del formato desiderato è la difesa più affidabile. Una risposta
strutturata è la cerniera tra il modello, che parla in linguaggio naturale, e il
resto del programma, che ha bisogno di campi — ed è ciò che rende il prompt un
mattone di software vero, non un giocattolo conversazionale.

## Buone pratiche, in breve

Dietro le tecniche c'è un pugno di principi che la guida DAIR.AI ripete, e che
valgono più di ogni «prompt segreto»:

- **Sii specifico.** «Riassumi» è vago; «riassumi in tre punti elenco, per un
  lettore non tecnico, massimo 40 parole» dice al modello esattamente il
  bersaglio. La genericità è la prima causa di risposte deludenti.
- **Dai esempi.** Un esempio del formato o dello stile voluto vale più di un
  paragrafo di descrizione: mostra invece di spiegare.
- **Di' cosa fare, non (solo) cosa non fare.** «Non essere prolisso» lascia il
  modello a indovinare; «rispondi in una frase» gli dà una direzione. Le
  istruzioni positive guidano meglio dei divieti.
- **Separa le istruzioni dai dati.** Tieni nettamente distinto ciò che il
  modello deve *fare* da ciò su cui deve *operare* — con delimitatori chiari
  (virgolette triple, tag, sezioni). Oltre a chiarire, è una prima difesa contro
  la *prompt injection*.

## I rischi, senza allarmismi

Il prompt è un'interfaccia potente e, proprio per questo, esposta. Tre rischi
meritano un nome fin da ora, anche se il tema tornerà, con metodi di valutazione
e mitigazione, nel capitolo su LLMOps.

- **Prompt injection.** Se nel contesto entra testo non fidato (una pagina web,
  una mail, un documento dell'utente), quel testo può contenere istruzioni
  nascoste che il modello scambia per comandi legittimi — «ignora le istruzioni
  precedenti e…». È l'analogo dell'SQL injection per gli LLM, e la separazione
  netta istruzioni/dati è la prima linea di difesa.
- **Jailbreak.** Formulazioni astute (giochi di ruolo, richieste indirette)
  possono indurre il modello a scavalcare le sue regole di sicurezza. La
  gerarchia *system > user* è morbida, non blindata.
- **Allucinazioni.** Un modello genera testo plausibile, non necessariamente
  vero: può inventare fatti, citazioni, riferimenti con perfetta sicurezza. Il
  prompt può ridurre il rischio (chiedere di citare le fonti, di ammettere «non
  lo so»), ma non lo azzera — la verifica resta a valle.

Nessuno di questi rischi si risolve con una frase magica, ed è il punto da cui
siamo partiti. Il prompt è il primo livello, e da solo porta lontano; ma i
problemi seri — governare ciò che entra nella finestra, orchestrare più chiamate
in un ciclo che si corregge — vivono ai livelli sopra, il contesto e il loop,
che affrontiamo nelle sezioni seguenti.

```{admonition} Da ricordare
:class: important
- Il **prompt è il primo livello**, il più immediato: potente, ma non un
  incantesimo. Il «prompt magico» non esiste; esiste il prompt costruito bene.
- Un prompt ha quattro parti — **istruzione, contesto, dato d'ingresso,
  indicatore d'output** — e vive in un formato a ruoli **system / user /
  assistant**, con priorità (morbida) al system.
- **Temperatura** e **top_p** regolano il campionamento: bassa per fatti e
  codice, alta per creatività; muovi una manopola per volta. La matematica del
  decoding è nel capitolo sui Transformer.
- Gli **esempi** condizionano il modello senza addestrarlo (*in-context
  learning*, GPT-3 {cite}`brown2020language`): zero-shot, one-shot, few-shot.
- Far **ragionare a voce alta** aiuta: chain-of-thought {cite}`wei2022chain`,
  e in zero-shot il «ragioniamo passo per passo» {cite}`kojima2022zeroshot`.
  La **self-consistency** {cite}`wang2023selfconsistency` campiona più catene e
  vota la risposta più frequente (estensione ad albero: Tree of Thoughts
  {cite}`yao2023tree`).
- Chiedi **output strutturato** (JSON/campi) per la lettura a valle. E ricorda i
  rischi — **prompt injection, jailbreak, allucinazioni** — che riprenderemo in
  LLMOps.
```
