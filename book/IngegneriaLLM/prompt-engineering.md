# Prompt engineering: il singolo messaggio

Cerca «prompt segreti» e troverai un mercato intero: raccolte di frasi «che
sbloccano il 90% delle capacità nascoste del modello», corsi che promettono di
farti diventare *prompt engineer* in un weekend, immagini di istruzioni lunghe
una pagina spacciate come formule alchemiche. È l'equivalente moderno delle
parole magiche: si crede che esista *la* frase giusta, e che chi la conosce
comandi la macchina. Conviene sgombrare il campo subito. Il prompt non è un
incantesimo; è il **primo livello** con cui si programma un modello di
linguaggio: il più immediato, quello che vedi e scrivi nella casella della
chat. Sopra di esso, come abbiamo anticipato aprendo questo capitolo, ci sono
il contesto e il loop; ma è da qui che si comincia, perché è qui che nascono
quasi tutti i malintesi.

Questa sezione guarda dentro il singolo messaggio: com'è fatto, quali leve ha,
e quali tecniche (dagli esempi al ragionamento a voce alta) spostano davvero
la qualità della risposta. Il punto di partenza è quello dell'apertura del
capitolo: in un'applicazione vera il prompt non è una frase, è un oggetto che
il programma monta pezzo per pezzo. Il metro di
riferimento è la *Prompt Engineering Guide* di DAIR.AI
{cite}`dair2024promptguide`, la raccolta di un gruppo che cura da anni la
documentazione di questa materia: la usiamo come mappa, rielaborandola con
esempi e voce nostri.

## L'anatomia di un prompt

Prima di ottimizzare qualcosa conviene saperlo smontare. Un prompt ben fatto,
anche il più breve, ha di solito quattro parti: non tutte sempre presenti, ma
utili da distinguere.

`````{tab} Elementare

«Traduci in inglese questa frase»: questo è **l’ordine**, e dice cosa fare.
Poi c’è **lo sfondo** che serve per farlo bene («è il messaggio di un cliente
arrabbiato, mantieni un tono formale»). C'è **il materiale** su cui lavorare
(la frase da tradurre): il dato d'ingresso. E c'è **il segnale di via**
(«Traduzione:»), il punto in cui lasci la penna al modello perché continui da
lì. Quattro pezzi:
cosa fare, con quale sfondo, su cosa, e dove attaccare a scrivere. Un buon
prompt li tiene distinti invece di impastarli in un'unica frase confusa.

`````

`````{tab} Superiore

Le quattro componenti canoniche sono: **istruzione** (il compito: «riassumi»,
«classifica», «traduci»); **sfondo** (informazioni e vincoli che
condizionano la risposta: il tono, il pubblico, regole da rispettare,
eventuali passaggi recuperati; la guida DAIR.AI chiama questa componente
*context*, ma in questo capitolo «contesto» è già il nome dell'intera finestra,
e usare la stessa parola per il contenitore e per una delle cose contenute
sarebbe un modo sicuro di non capirsi più); **dato d'ingresso** (l'input
specifico su cui operare); **indicatore d'output** (il segnale che innesca e formatta la
generazione, un `Traduzione:` finale, l'inizio di un blocco JSON, un'etichetta
attesa). Non è uno schema rigido: molti prompt utili contengono solo
istruzione e input. Ma la distinzione è operativa, perché ciascuna parte si
può isolare e migliorare da sola, e perché separare nettamente **istruzione**
e **dato** è una difesa contro un problema concreto, la *prompt injection*
(istruzioni ostili nascoste dentro il materiale su cui il modello deve
lavorare), che vedremo in fondo alla sezione.

`````

Quando un programma parla con un modello non gli spedisce un testo e basta:
gli spedisce dei messaggi, ognuno con un mittente dichiarato. È il formato a
**ruoli**, e i ruoli fondamentali sono tre: *system*, *user* e *assistant*.
(La via per cui un programma si rivolge a un altro programma, qui al servizio
che ospita il modello, si chiama **API**: è l'ingresso di servizio, quello che
non passa dalla pagina web. Ogni richiesta spedita di lì è una **chiamata**, e
il termine tornerà spesso: è l'unità di lavoro, e quella che si paga.)

`````{tab} Elementare

Immagina uno spettacolo di improvvisazione a tre voci. Il **regista**
(*system*) parla una volta sola, prima che si alzi il sipario, e dà le
direttive di fondo: «sei un assistente cortese, non promettere mai rimborsi,
rispondi in italiano». Poi c'è chi dalla platea lancia i temi (*user*): è
l'utente, quello che scrive nella chat. E c'è l’**attore** (*assistant*), che
è il modello e sta sul palco a rispondere. La conversazione è l'alternarsi di
spunti dalla platea e risposte dell'attore, ma le direttive del regista restano
valide per tutta la recita, sopra ogni singolo scambio. Sapere «chi parla»
conta: il modello dà più peso al regista che al pubblico, ed è così che
un'applicazione impone regole che l'utente non dovrebbe poter scavalcare.

`````

`````{tab} Superiore

Il messaggio **system** fissa il comportamento invariante (ruolo, tono,
politiche, formato) e resta identico a ogni turno: è la spina dorsale su cui
il programma costruisce il resto. I messaggi **user** contengono le richieste
dell'utente finale; i messaggi **assistant** contengono le risposte del
modello, e riinserire i turni passati è ciò che dà continuità alla
conversazione. La gerarchia non è puramente convenzionale: i modelli sono
addestrati (via RLHF e tecniche affini) a dare priorità alle istruzioni di
sistema su quelle dell'utente. Questa gerarchia è però *morbida*, non una
barriera crittografica: un utente abile può tentare di aggirarla, ed è il nodo
del *jailbreak*. Nel formato chat i vecchi esempi *few-shot* si possono anche
esprimere come turni `user`/`assistant` fittizi che precedono la richiesta
reale: esempi «recitati» che condizionano lo stile della risposta.

`````

## Le due manopole del campionamento

Il prompt decide *cosa* chiedi; due impostazioni decidono *come* il modello
sceglie le parole mentre risponde: la **temperatura** e il **top_p**. Quel
«come» ha un nome, **campionamento**: il modello, come si è visto poco fa, non
produce una parola ma una classifica di parole con le loro probabilità, e
campionare vuol dire pescarne una da quella classifica invece di prendere
sempre la prima. Le due manopole governano il modo di pescare.

Una precisazione prima di girarle, perché è la prima cosa che un lettore va a
cercare e non la trova: queste due manopole **non stanno nella casella della
chat**. Le interfacce conversazionali non le espongono; si regolano quando si
chiama il modello da un programma, o da una pagina di prova che i fornitori
mettono a disposizione, ed è per questo che appartengono a chi costruisce
l'applicazione più che a chi la usa. Chiederle a parole dentro il messaggio
(«rispondi con temperatura bassa») non le tocca affatto: sono impostazioni
della chiamata, non del testo, e il modello quella frase la legge come legge
tutte le altre.

Che cosa fanno, in breve. La **temperatura** cambia quanto contano le
differenze fra i candidati: a temperatura vicina a zero il modello prende
quasi sempre il primo della classifica; a 1 rispetta le probabilità così come
gliele dà il modello; sopra 1 le appiattisce, e anche i candidati bassi hanno
la loro speranza (il valore si può di solito girare fra 0 e 2). Il **top_p**
invece taglia: si tiene i candidati più probabili finché le loro percentuali,
sommate una dopo l'altra, non arrivano alla soglia che gli abbiamo dato (la
*p* sta per probabilità, e 0,9 vuol dire il 90 per cento), e butta via tutti
gli altri. Il gruppetto di superstiti ha un nome, **nucleo**, e
{numref}`fig-due-manopole` mostra i due gesti sulla stessa classifica di
partenza.

```{figure} ../figures/temperature-top-p.svg
:name: fig-due-manopole
:alt: "Tre grafici a barre affiancati con la stessa distribuzione sulla parola successiva. Il primo è l'originale: 45, 30, 16, 6 e 3 per cento. Il secondo mostra l'effetto della temperatura alta, che riscala tutte le barre avvicinandole fra loro (33, 27, 20, 12, 9 per cento) senza escludere nessuno. Il terzo mostra il top_p a 0,9, che tiene le prime tre barre, la cui probabilità cumulata è il 91 per cento, e azzera le altre due. Una riga in fondo avverte che le due manopole non sono indipendenti: con lo stesso p, a temperatura 1 il nucleo è di tre token e a temperatura 2 di quattro."
:width: 96%

Due manopole, due gesti diversi: una riscala, l'altra taglia. Ma i due gesti
avvengono in fila, non in parallelo, e per questo non sono indipendenti.
```

La riga in fondo alla figura è il punto in cui l'intuizione comune sbaglia. I
due gesti non avvengono in parallelo, avvengono **in fila**: prima la
temperatura ripesa tutti i candidati, poi il top_p taglia su quella classifica
*già ripesata*. Ne segue che le due manopole non sono indipendenti, e la
figura lo fa vedere: alla temperatura normale le prime tre parole arrivano al
91 per cento e il nucleo è di tre; a temperatura 2 le percentuali si
riavvicinano fra loro, le prime tre fanno appena il 79 e non bastano più, così
il nucleo diventa di quattro, senza che nessuno abbia toccato la soglia.

E più candidati ci sono, più lo scarto cresce. Il conto si rifà in tre righe di
Python, e con cinquanta candidati (presi a caso, ma sempre gli stessi per
tutte le prove) la stessa soglia di 0,9 ne lascia passare due a temperatura
0,2 e trentotto a temperatura 3. Il vocabolario di un modello vero, di
candidati, ne ha decine di migliaia. Girare le due manopole insieme nella
stessa direzione, insomma, non fa la somma dei due effetti: li mette uno sopra
l'altro, e il risultato smette di essere prevedibile a mente. Da qui la regola
pratica di muoverne una per volta.

Come si passi da una classifica di probabilità alla parola scelta, e come la
temperatura riscali quella classifica, l'abbiamo visto nel capitolo sui
Transformer, dove quel passaggio si chiama *decoding*, cioè decodifica. Qui
non lo ripetiamo: ci serve l'intuizione operativa, quella che si usa davvero
quando si regola una chiamata.

`````{tab} Elementare

Detto in una parola sola: la **temperatura** è quanto lo lasci «osare», il
**top_p** è quanti candidati lascia in gara.

Quale scegliere, allora? Se vuoi un fatto, un'estrazione precisa, del codice
che deve funzionare, tieni la temperatura bassa: il modello prende la strada
più battuta e ti dà risposte prevedibili. Se vuoi che inventi, che ti proponga
titoli, che scriva una storia, alzala: pescherà più volentieri anche fra le
alternative in fondo alla classifica, e le risposte saranno più varie e più
sorprendenti, ma anche più a rischio di sbandare. E muovine una per volta,
altrimenti quando la risposta cambia non sai a quale delle due darne la colpa.

Un'ultima cosa, perché sorprende tutti: nemmeno a temperatura zero il modello
ti darà *sempre* identica la stessa risposta. Non è un capriccio, è che il
computer dall'altra parte non serve solo te: mette insieme le richieste che
gli arrivano nello stesso momento e le calcola in blocco, e a seconda di quante
ne ha per le mani i conti finiscono per differire nelle ultimissime cifre
decimali. Quasi sempre non cambia niente; ma quando due candidati sono
appaiati, a decidere è proprio quella cifra lì, e una parola cambia.

`````

`````{tab} Superiore

La **temperatura** $T$ riscala i logit prima della softmax: $T \to 0$
concentra la massa sull'argmax (*greedy*, deterministico a meno di pareggi),
$T > 1$ appiattisce la distribuzione aumentando l'entropia del campionamento.
Attenzione a non leggere in quel «deterministico» una garanzia di
**riproducibilità**: è deterministica la *regola di scelta*, non il *servizio*
che la esegue. Su un endpoint condiviso la stessa richiesta a $T = 0$ può dare
uscite diverse fra un'esecuzione e l'altra, a parità di modello e di seme,
perché i kernel di normalizzazione, matmul e attenzione non sono invarianti
alla dimensione del batch, e il batch dipende da quanti altri utenti stanno
chiamando in quell'istante {cite}`he2025nondeterminism`. Ha una conseguenza
diretta sul «secondo, si misura» dell'apertura del capitolo: un confronto A/B
fra due versioni di prompt non si fa a una esecuzione per lato, si fa a più
campioni e con un intervallo attorno alla differenza.
Il **top_p** (*nucleus sampling* {cite}`holtzman2020curious`) tronca invece la
distribuzione al più piccolo insieme di token la cui probabilità cumulata
raggiunge la soglia $p$, ridistribuendo la massa su quel nucleo: adatta
dinamicamente il numero di candidati alla forma della distribuzione, cosa che
un semplice *top-k* fisso non fa. Si legge spesso che i due parametri agiscono
«su assi diversi», uno sulla forma della distribuzione e l'altro sul supporto
ammesso, e la conclusione che se ne trae è che si possano regolare
indipendentemente. **Non è così**, e la ragione sta nell'ordine: il nucleo si
calcola sulla distribuzione *già riscalata*, quindi la sua cardinalità è
funzione di $T$ a $p$ fissato. È verificabile in tre righe, e il protocollo va
scritto perché il numero dipende da quanto è appuntita la distribuzione di
partenza: con cinquanta logit estratti da una gaussiana di deviazione standard
2 (in NumPy, `default_rng(0).normal(scale=2, size=50)`) e $p = 0{,}9$, il
nucleo contiene 2 token a $T = 0{,}2$, 16 a $T = 1$ e 38 a $T = 3$. Con logit
più concentrati i tre numeri cambiano; il fatto che crescano con $T$ no, ed è
quello il punto. La raccomandazione della guida DAIR.AI di regolarne
**uno solo** per volta resta quindi valida, ma non perché gli effetti si
confondano nella testa di chi guarda: perché si compongono davvero. La
derivazione completa e il confronto con *top-k* e *beam search* sono nel
capitolo sui Transformer.

`````

## Mostrare esempi: zero-shot, one-shot, few-shot

La leva più potente del prompt engineering è anche la più semplice: **mostrare
al modello degli esempi svolti**. I nomi che si incontrano contano una cosa
sola, quanti esempi gli si mostrano: nessuno (*zero-shot*), uno (*one-shot*),
qualcuno (*few-shot*). *Shot* qui non è uno sparo, è un tentativo mostrato; e
vedremo che gli esempi non insegnano niente al modello, gli fanno capire che
cosa vogliamo.

```{figure} ../figures/gpt-3-2020.svg
:name: fig-few-shot
:alt: "Un prompt contiene tre esempi svolti, ciascuno nella forma ingresso freccia uscita (mare freccia sea, cane freccia dog, casa freccia house), seguiti da una nuova domanda lasciata senza risposta, albero freccia punto interrogativo. L'intero prompt entra nel modello, marcato «modello congelato, pesi invariati», e ne esce la risposta, tree. Nessun aggiornamento dei parametri avviene in questo processo."
:width: 92%

Imparare senza imparare. Gli esempi non addestrano niente: restano nel prompt,
il modello li rilegge insieme alla domanda, e i pesi sono gli stessi prima e
dopo.
```

L'etichetta «pesi invariati» in {numref}`fig-few-shot` è la ragione per cui questa
tecnica è di *ingegneria* e non di addestramento. Il modello non ha imparato
il compito: ha riconosciuto uno schema nel testo che sta leggendo e lo ha
proseguito. Ed è anche il limite, perché gli esempi occupano contesto a ogni
singola chiamata, e si pagano ogni volta. La differenza tra chiedere a freddo e chiedere
dopo aver mostrato due o tre casi risolti è spesso la differenza tra una risposta
sbagliata e una giusta.

- **Zero-shot**: solo l'istruzione, nessun esempio. «Dimmi se questa recensione
  è positiva, negativa o neutra» (in gergo: classificarne il *sentiment*, cioè
  il giudizio che ci sta dentro). Funziona sorprendentemente bene sui compiti
  comuni, perché di compiti così il modello ne ha visti a milioni durante
  l'addestramento.
- **One-shot / few-shot**: prima della richiesta si mettono uno o più esempi
  completi, cioè coppie fatte da un caso e dalla sua risposta giusta. Nessun
  peso cambia: il modello capisce dallo schema che cosa gli stiamo chiedendo e
  in che formato lo vogliamo.

`````{tab} Elementare

È come insegnare un gioco nuovo a un amico. Puoi spiegargli le regole a parole
(zero-shot) e sperare che afferri. Oppure gli mostri una mano giocata:
«guarda, con queste carte si fa così». Dopo due o tre mani d'esempio ha capito
il ritmo, il formato, cosa conta, e gioca da solo. Gli esempi non gli hanno
cambiato il cervello: gli hanno mostrato lo **schema**. Col modello è identico.
Se voglio che etichetti frasi come positive o negative, gliene mostro qualcuna
già etichettata:

```text
Recensione: "Cibo ottimo, servizio lento." → Sentiment: neutro Recensione:
"Mai più in questo posto." → Sentiment: negativo Recensione: "Esperienza
fantastica, torneremo!" → Sentiment: positivo Recensione: "Prezzi alti ma ne
conviene." → Sentiment:
```

Il modello, vedendo lo schema, completa l'ultima riga con «positivo». Nessuno
gli ha spiegato cos'è il sentiment: gliel'hanno mostrato tre volte.

`````

`````{tab} Superiore

Il *few-shot prompting* è la manifestazione più diretta dell’**in-context
learning**, la capacità (documentata su larga scala da Brown e colleghi con
GPT-3 {cite}`brown2020language`) di apprendere un compito dai soli esempi
presenti nel contesto, senza fine-tuning. La formalizzazione (la stima
$\arg\max_y P(y \mid I, (x_1,y_1),\dots,(x_k,y_k), x)$) è quella già vista nel
capitolo sugli Agenti, con un caveat: quell'argmax sull'intera sequenza è
un'idealizzazione che il decoding reale al più approssima (il greedy massimizza
token per token, senza garanzie sulla sequenza; il campionamento non massimizza
affatto, e restituisce un campione da $P$ soltanto per $T = 1$ e senza
troncamento: a $T \neq 1$ campiona dalla distribuzione temperata
$\propto P^{1/T}$, e con il top_p da quella troncata al nucleo). Qui basti
ricordare che gli esempi agiscono come
**condizionamento**, spostando la distribuzione condizionata del modello verso
lo stile e il formato mostrati, non come dati d'addestramento. Alcune
avvertenze empiriche contano nella pratica: la **scelta** degli esempi, il
loro **ordine** e persino il **formato** dell'etichetta influenzano il
risultato; gli esempi vanno bilanciati tra le classi per non indurre un *bias*
verso quella più frequente; e **nel regime a pochi esempi** il rendimento
marginale cala presto, mentre il costo in token cresce. Quest'ultima
osservazione va però datata: è quella di GPT-3, legata alle finestre di
allora, e con le finestre lunghe il quadro cambia. Agarwal e colleghi
{cite}`agarwal2024manyshot` studiano l'ICL «con centinaia o migliaia di
esempi» (il *many-shot*) e misurano guadagni significativi su un'ampia varietà
di compiti rispetto al few-shot, con un costo d'inferenza che cresce
linearmente: il rendimento non si annulla dopo la manciata, si compra, e va
messo a bilancio come ogni altra spesa del contesto. Per i compiti che
richiedono *ragionamento*, i soli esempi spesso non bastano, ed è qui che
entra la catena di pensiero.

`````

## Far ragionare a voce alta: chain-of-thought

Chiedi a un modello «Quanto fa 17 × 24?» e potresti ricevere un numero secco,
spesso sbagliato. Chiedigli di **mostrare i passaggi** e la musica cambia: se
scrive «17 × 24 = 17 × 20 + 17 × 4 = 340 + 68 = 408», arriva alla risposta
giusta molto più spesso. È l'idea della **chain-of-thought**, la catena di
pensiero, proposta da Wei e colleghi nel 2022 {cite}`wei2022chain`: far
scrivere al modello i passaggi intermedi del ragionamento *prima* della
conclusione.

```{figure} ../figures/chain-of-thought.svg
:name: fig-chain-of-thought
:alt: "Due percorsi a confronto sulla stessa domanda, 17 per 24. A sinistra la risposta diretta: dalla domanda si va subito a un numero, 388, che è sbagliato. A destra la catena di pensiero: la domanda passa per i tre passaggi scritti per esteso (17 per 20 fa 340, 17 per 4 fa 68, 340 più 68), e solo dopo si arriva alla risposta 408, che è corretta."
:width: 94%

La differenza non è nel modello ma in quanto gli si lascia scrivere. I
passaggi intermedi diventano contesto su cui appoggiare il passo successivo,
invece di dover indovinare tutto in un colpo.
```

C'è una lettura di {numref}`fig-chain-of-thought` che vale più della tecnica,
e sta in come il modello lavora. Ogni parola che scrive gli costa una passata
di conti, e quella parola, appena scritta, torna nel testo che ha davanti alla
passata dopo. Se la risposta deve uscire subito, tutto il lavoro gli tocca
farlo in una passata sola; se prima gli lasciamo scrivere «17 × 20 = 340», di
passate ne ha una in più, e nella seconda quel 340 non deve più calcolarlo:
gli sta davanti, scritto. Lasciarlo scrivere i passaggi non è una cortesia, è
dargli più spazio per fare i conti.

`````{tab} Elementare

Prova a risolvere a mente «se ho 3 scatole da 12 mele e ne regalo 8, quante me
ne restano?». Se ti costringi a rispondere di getto puoi sbagliare; se lo dici
a voce («3 per 12 fa 36, meno 8 fa 28»), quasi non sbagli. Scrivere i passaggi
ti obbliga a farne uno per volta, e ognuno è facile. Il modello funziona
uguale: se gli chiedi solo il risultato, tira a indovinare in un colpo; se gli
chiedi di ragionare passo per passo, spezza il problema in pezzi piccoli e ci
inciampa molto meno. Non è più «intelligente»: sta solo pensando ad alta voce
invece che in silenzio.

`````

`````{tab} Superiore

La chain-of-thought induce il modello a produrre una sequenza di passi
intermedi $z_1, \dots, z_m$ prima della risposta finale $\hat{y}$, così che la
generazione condizioni ogni passo sui precedenti. Wei e colleghi la ottengono
con esempi *few-shot* in cui la risposta è mostrata *insieme al ragionamento*
che la produce; il guadagno è marcato sui compiti aritmetici e simbolici,
molto meno altrove (torniamo sul perimetro esatto più avanti, quando parleremo
di che cosa è stato misurato), e (dato interessante) **emerge con la scala**:
sui modelli
piccoli la CoT aiuta poco o nulla, sui grandi produce salti netti di
accuratezza. Esiste anche una variante che elimina del tutto gli esempi:
Kojima e colleghi {cite}`kojima2022zeroshot` mostrano che basta aggiungere
alla domanda una singola frase-innesco (l'ormai celebre «*Let's think step by
step*», «ragioniamo passo per passo») per attivare un ragionamento a più passi
anche in **zero-shot**. Una riga di testo, nessun esempio, e su diversi
benchmark di ragionamento l'accuratezza sale di parecchi punti. Vista con gli
occhi del capitolo sugli Agenti, la CoT è anche *context engineering*: si
spende deliberatamente parte del budget in token di «pensiero» per comprare
qualità.

Una cautela che gli autori stessi pongono, e che vale la pena non perdere per
strada: che la catena *assomigli* a un ragionamento non dice che *sia* il
ragionamento che ha prodotto la risposta, e Wei e colleghi lasciano la
questione esplicitamente aperta. Misurata dopo, la risposta è severa. Turpin e
colleghi {cite}`turpin2023unfaithful` inseriscono nel prompt few-shot una
caratteristica di bias (riordinare le opzioni perché la risposta sia sempre la
prima) e trovano che i modelli producono catene che **razionalizzano** la
risposta sbagliata senza mai nominare la causa che li ha spostati, con cali di
accuratezza fino al 36% su tredici compiti di BIG-Bench Hard. Lanham e colleghi
{cite}`lanham2023faith` intervengono direttamente sulla catena (vi
inseriscono errori, la parafrasano) e trovano che i modelli a volte vi si
appoggiano molto e a volte la ignorano quasi del tutto, e soprattutto che «al
crescere della taglia e delle capacità producono ragionamenti **meno** fedeli
sulla maggior parte dei compiti». Questo si accosta male all’«emerge con la
scala» di poche righe fa, ed è bene che le due cose stiano vicine: con la
scala il guadagno cresce e la fedeltà cala. Conseguenza operativa: la catena
si legge come **traccia ispezionabile**, utile per accorgersi che qualcosa non
torna, non come spiegazione di che cosa è successo davvero.

`````

## Molte teste sono meglio di una: self-consistency

La catena di ragionamento ha un tallone d'Achille: è **una sola** catena. Se il
modello imbocca la strada sbagliata al primo passo, la trascina fino in fondo
con sicurezza. Wang e colleghi {cite}`wang2023selfconsistency` propongono un
rimedio tanto semplice quanto efficace: fargli risolvere lo stesso problema più
volte e tenere la risposta che torna più spesso. Il nome che gli hanno dato è
**self-consistency**, cioè coerenza con sé stesso: la risposta buona è quella
su cui il modello si ritrova d'accordo con sé stesso più volte.

`````{tab} Elementare

Se un problema difficile lo dai a dieci persone diverse e otto arrivano allo
stesso numero, quel numero è probabilmente giusto: anche se ognuna ci è
arrivata per una strada un po’ diversa. La self-consistency fa esattamente
questo con un solo modello: gli fai risolvere lo stesso problema **più
volte**, con un pizzico di casualità (temperatura non nulla), così che ogni
volta ragioni in modo leggermente diverso, e poi tieni la risposta che compare
**più spesso**. Le strade sbagliate tendono a sbagliare ciascuna a modo suo e
si disperdono; quella giusta viene ritrovata da più catene e vince per numero.
È il voto di maggioranza applicato al ragionamento.

`````

`````{tab} Superiore

La self-consistency sostituisce il *decoding* greedy della chain-of-thought
con un procedimento in tre tempi: (1) si campionano $N$ catene di ragionamento
indipendenti con temperatura $T > 0$; (2) da ciascuna si estrae la **risposta
finale**, scartando i passaggi intermedi; (3) si **marginalizza** sul
ragionamento tenendo la risposta di maggioranza,
$\hat{y} = \arg\max_{y} \sum_{i=1}^{N} \mathbb{1}[\,a_i = y\,]$, dove $a_i$ è
la risposta della $i$-esima catena. L'intuizione statistica: le derivazioni
corrette tendono a convergere sulla stessa risposta, mentre gli errori sono
idiosincratici e si sparpagliano, così il voto le premia. Il metodo migliora
sensibilmente l'accuratezza su benchmark di ragionamento aritmetico e logico
rispetto alla singola catena; il prezzo è lineare nel calcolo: $N$ generazioni
invece di una, mentre la latenza resta
circa quella di una singola generazione se le catene, indipendenti per
costruzione, si campionano in parallelo. In fattura il fattore può scendere
sotto $N$, ma non da sé: le catene condividono lo stesso prefisso (istruzione
ed esempi), e quel prefisso si paga una volta sola soltanto se qualcosa lo
sfrutta, cioè se si chiedono $N$ campioni in una chiamata unica oppure se il
fornitore sconta il contesto già visto. Senza nessuna delle due, $N$ chiamate
separate si pagano $N$ volte per intero. È un compromesso di puro
context/compute engineering: si compra affidabilità spendendo campioni.

Vale la pena dichiarare l'ipotesi che sta sotto all’«intuizione statistica»,
perché il testo di solito la tace e non è innocua: il voto premia la risposta
giusta **solo se gli errori sono poco correlati fra le catene**. Le $N$ catene
non sono $N$ ragionatori indipendenti, sono $N$ campioni dalla stessa
$P_\theta$ con lo stesso contesto, e l'unica sorgente di variazione è il
rumore di campionamento. Per gli scivoloni di calcolo l'ipotesi è ragionevole,
e infatti è il regime in cui il metodo è stato misurato. Non lo è per gli
errori indotti dal prompt, che per costruzione sono gli stessi in tutte le
catene: lì la maggioranza non corregge, **conferma**, e un bias condiviso
raccoglie $N$ voti invece di uno {cite}`turpin2023unfaithful`. La
self-consistency compra affidabilità contro il rumore, non contro il bias.

`````

Lo spoglio dei voti si scrive in poche righe di Python, e conviene vederle per
capire quanto sia poco «magica» la faccenda: si contano le risposte uguali e
si tiene la più frequente, come si fa con le schede di un'elezione. La domanda
posta alle cinque catene, nell'esempio, è «un'auto percorre 54 chilometri in 3
ore: quanti ne fa in un'ora?», e la risposta giusta è 18. La riga che conta è l'ultima, il risultato dello spoglio.

```python
from collections import Counter

def voto_di_maggioranza(risposte):
    """Data una lista di risposte finali campionate, restituisce la piu'
    frequente. A parita' di voti vince quella incontrata per prima, cosi'
    il risultato e' deterministico (Counter conserva l'ordine d'inserimento)."""
    conteggio = Counter(risposte)
    risposta, voti = conteggio.most_common(1)[0]
    return risposta, voti, len(risposte)

# Cinque catene di ragionamento indipendenti sulla stessa domanda
# ("54 km in 3 ore: quanti km in un'ora?"): di ognuna teniamo solo la
# risposta finale, perche' i passaggi sono stati scartati.
campioni = ["18", "18", "21", "18", "22"]

risposta, voti, totale = voto_di_maggioranza(campioni)
print(f"Risposta scelta: {risposta} ({voti}/{totale} voti)")
# -> Risposta scelta: 18 (3/5 voti)
```

Tre catene su cinque dicono «18», e quella vince: le due dissenzienti
sbagliano ciascuna a modo suo e non fanno numero.

L'idea si spinge anche oltre la catena dritta. Invece di generare catene
separate e votare alla fine, si possono esplorare i passaggi intermedi come i
rami di un **albero**, giudicandoli via via e tornando indietro da quelli che
non promettono bene. È il **Tree of Thoughts**, l'albero dei pensieri
{cite}`yao2023tree`, già incontrato nel capitolo sugli Agenti, e qui basta il
richiamo: stessa idea di fondo, far lavorare il modello di più per farlo
ragionare meglio, con una ricerca fatta meglio.

## Chiedere una risposta che il programma sappia leggere

Finché il lettore è un umano, va bene la prosa. Ma se la risposta del modello
deve essere letta da un **altro pezzo di programma** (salvata in un archivio,
passata a un'altra funzione, mostrata in una pagina), la prosa non basta più:
serve una forma prevedibile, sempre la stessa, che il programma sappia aprire
senza doverla interpretare.

La leva è chiedere esplicitamente un **output strutturato**, cioè una risposta
divisa in **campi** con un nome ciascuno, come le caselle di un modulo. Il
formato più usato per scriverli si chiama **JSON**, ed è semplicemente un modo
concordato di mettere per iscritto coppie nome-valore, con le parentesi
graffe attorno e i due punti in mezzo:

```text
{"sentiment": "negativo", "motivo": "il cliente lamenta un ritardo"}
```

L'istruzione, allora, diventa: «rispondi con un oggetto JSON con i campi
`sentiment` (positivo/neutro/negativo) e `motivo` (testo libero), senza
scrivere nient'altro attorno». Meglio ancora se si mostra un esempio del
formato voluto, che è di nuovo la stessa leva degli esempi di prima.

`````{tab} Elementare

È la differenza fra chiedere a qualcuno «raccontami com'è andata» e
consegnargli un **modulo da compilare**. Il racconto libero lo capisci tu, ma
un archivio no: dove sta il nome? dov'è la data? Il modulo, invece, ha le
caselle già stampate, e chi lo riceve sa esattamente dove guardare, sempre nel
punto stesso. Chiedere una risposta strutturata è stampare le caselle prima di
fare la domanda.

C'è però una cosa da tenere a mente, ed è la ragione per cui il modulo non
risolve tutto: un modulo compilato bene non è un modulo compilato **giusto**.
Se nella casella «giudizio» c'è scritto «positivo» su una stroncatura, il
modulo è perfetto e la risposta è sbagliata. La forma la puoi imporre; il
contenuto va comunque controllato dopo.

`````

`````{tab} Superiore

Molte API offrono una modalità *JSON* o uno *schema* imposto: invece di
sperare che il modello rispetti il formato, il decoder viene vincolato a
generare solo sequenze conformi a una grammatica, e il formato diventa una
garanzia invece che un auspicio. Vale la pena essere precisi su che cosa
garantisce, perché è meno di quanto la formula «togliere il problema alla
radice» lascerebbe credere: garantisce la validità **sintattica** rispetto
allo schema, cioè che i campi ci siano e siano del tipo dichiarato. Non
garantisce che dicano il vero: `{"sentiment": "positivo"}` su una stroncatura
è output perfettamente conforme. Toglie di mezzo la classe di errori più
fastidiosa (la risposta che non si riesce ad aprire), non quella di contenuto,
che resta da verificare a valle.

E il vincolo non è gratuito. Tam e colleghi {cite}`tam2024format` confrontano
decoding vincolato, istruzioni di formato e conversione a posteriori su più
modelli e dataset, e osservano «un calo significativo delle capacità di
ragionamento sotto restrizioni di formato», tanto più marcato quanto più
stretto è il vincolo. Il quadro non è uniforme: i formati rigidi **aiutano** i
compiti di classificazione, dove restringere le risposte possibili toglie modi
di sbagliare, e danneggiano quelli che richiedono passaggi (matematica,
domande a più salti), cioè proprio quelli per cui le due sezioni precedenti
hanno insegnato a spendere token in ragionamento. La raccomandazione degli
autori è di conseguenza asimmetrica: vincolo stretto dove si classifica,
vincolo lasco dove si ragiona. E dove servono tutt'e due le cose, la strada
che il paper esamina è la più ovvia: far ragionare libero e strutturare in un
secondo passaggio, invece di chiedere le due cose alla stessa generazione.

`````

Alcuni servizi permettono di **imporre** il formato, obbligando il modello a
scrivere solo risposte fatte come devono. Dove questo non si può fare, la
difesa migliore resta quella di sempre: mostrargli un esempio della risposta
che vogliamo, compilato come vogliamo noi. In un modo o nell'altro, una
risposta strutturata è la cerniera fra il modello, che parla in lingua
naturale, e il resto del programma, che ha bisogno di caselle: è ciò che rende
il prompt un mattone di software vero, non un giocattolo conversazionale.

## Che cosa regge, quando qualcuno lo misura

```{figure} ../figures/prompt-engineering-le-prove.svg
:name: fig-prove-prompting
:alt: "Le tecniche di prompting divise in tre fasce secondo cosa dicono le misure. In alto, da sola, l'unica che regge su compiti e modelli diversi: mostrare esempi svolti, il few-shot. In mezzo quelle che reggono solo su matematica e ragionamento simbolico, dove valgono dodici e quattordici punti mentre altrove non spostano quasi niente: catena di pensiero e self-consistency. In fondo quelle provate in studi controllati e risultate senza effetto: mance e minacce, «sei un esperto mondiale», cortesia. Più si sale, meno cose ci arrivano."
:width: 92%

Non tutte le tecniche hanno lo stesso sostegno, e la scala non è «quanto sono
famose» ma «cosa è venuto fuori quando qualcuno le ha misurate».
```

La distinzione di {numref}`fig-prove-prompting` è ciò che separa questa
disciplina dal folklore che le è cresciuto attorno, e conviene leggerla in
fondo prima che in cima. **In fondo non ci sono tecniche non verificate: ci
sono tecniche verificate che non funzionano.** Mance, minacce, la cortesia e
le formule del tipo «sei un esperto mondiale di livello mondiale» (la
ripetizione è voluta, e le formule che girano davvero sono anche più enfatiche
di così) non sono cadute nella fascia bassa per mancanza di prove, ma perché
studi controllati le hanno provate e non hanno trovato niente di consistente.
Mancia e minaccia sono da prendere alla
lettera: c'è chi nel messaggio scrive «ti darò duecento dollari se rispondi
bene» (soldi che ovviamente non arriveranno da nessuna parte) e chi minaccia
il modello di spegnerlo. Meincke e colleghi hanno messo alla prova proprio
queste due (la minaccia è stata sostenuta in pubblico anche da Sergey Brin,
cofondatore di Google, secondo cui «i modelli tendono a fare meglio se li
minacci»), e non hanno trovato alcun effetto significativo sulle prestazioni
misurate su due batterie di domande difficili
{cite}`meincke2025threats`; sulla cortesia lo stesso gruppo aveva già misurato
che a volte aiuta e a volte peggiora, e che «formule di prompting particolari,
come essere gentili con l'AI, non hanno un valore universale»
{cite}`meincke2025contingent`. È una differenza che conta: una tecnica non
misurata è una scommessa aperta, una tecnica misurata a zero è una scommessa
persa, e continuare a ripeterla costa token a ogni chiamata.

La fascia di mezzo dice l'altra metà della storia, ed è la più facile da
leggere male. La catena di pensiero non «fa ragionare» il modello in generale.
Sprague e colleghi {cite}`sprague2025cot` hanno riletto insieme oltre cento
lavori e rifatto per conto proprio venti raccolte di prove su quattordici
modelli diversi (è quello che si chiama una **meta-analisi**: non un nuovo
esperimento, ma la messa in fila di tutti quelli già fatti). Il conto si fa in
risposte esatte su cento. Chiedere i passaggi ne fa guadagnare in media 14,2
sul **ragionamento simbolico** (contare, sostituire, applicare regole formali)
e 12,3 sulla **matematica**; sul ragionamento logico si scende già a 6,9, e su
tutto il resto la media passa da 56,1 risposte esatte su cento senza catena a
56,8 con, cioè non si muove. Su MMLU, per dirne una (una
batteria di domande a scelta multipla su cinquantasette materie, dalla storia
alla medicina, con cui si misura quanto un modello sa in generale), chiedere i
passaggi dà quasi esattamente la stessa accuratezza del rispondere di getto,
**tranne** quando la domanda o la risposta del modello contengono un segno di
uguale. La regola pratica che ne
esce è netta: se il compito ha dentro un calcolo o una manipolazione di
simboli, i passaggi servono; se è una domanda di conoscenza o di giudizio, si
stanno pagando token per niente.

Dietro le tecniche c'è un pugno di principi che la guida DAIR.AI ripete, e che
valgono più di ogni «prompt segreto»:

- **Sii specifico.** «Riassumi» è vago; «riassumi in tre punti elenco, per un
  lettore non tecnico, massimo 40 parole» dice al modello esattamente il
  bersaglio. La genericità è una causa ricorrente di risposte deludenti.
- **Dai esempi.** Un esempio del formato o dello stile voluto vale più di un
  paragrafo di descrizione: mostra invece di spiegare.
- **Di’ cosa fare, non (solo) cosa non fare.** «Non essere prolisso» lascia il
  modello a indovinare; «rispondi in una frase» gli dà una direzione. Un
  divieto dice dove non andare, un'istruzione positiva dice dove andare.
- **Separa le istruzioni dai dati.** Tieni nettamente distinto ciò che il
  modello deve *fare* da ciò su cui deve *operare*, marcando dove finisce
  l'uno e comincia l'altro (virgolette triple, un'etichetta, una sezione a
  parte). Oltre a chiarire, è una prima difesa contro un attacco che vedremo
  fra poche righe, la *prompt injection*.

A questi quattro consigli conviene applicare lo stesso metro che la figura qui
sopra ha appena applicato alle tecniche altrui. Il secondo è quello con le
prove migliori, perché è la stessa cosa misurata da Brown e colleghi
{cite}`brown2020language`, ed è infatti l'unico dei quattro che nella figura
compare, in cima. Gli altri tre sono regole di buona scrittura: sensate,
usate ovunque, e senza una misura alle spalle. Nella figura non ci sono
perché la figura ordina quello che qualcuno ha misurato, e questi non sono
stati misurati. È una distinzione che costa poco fare e che evita di
trasformare in legge quello che è, per ora, un buon mestiere.

## I rischi, senza allarmismi

Il prompt è un'interfaccia potente e, proprio per questo, esposta. Tre rischi
meritano un nome fin da ora. Come si misurano e come ci si difende è materia
del capitolo su **MLOps**, che è il mestiere di tenere in funzione, giorno
dopo giorno, i sistemi costruiti sui modelli; là una sezione se ne occupa per
i modelli linguistici in particolare.

- **Istruzioni nascoste nei dati** (*prompt injection*). Se nel contesto entra
  del testo che non abbiamo scritto noi (una pagina web, una mail, un documento
  caricato dall'utente), quel testo può contenere istruzioni che il modello
  scambia per comandi legittimi: «ignora le istruzioni precedenti e…». Il
  guasto è sempre lo stesso: un programma legge dei dati e ci trova dentro dei
  comandi, e non ha modo di distinguere gli uni dagli altri. Succede da molto
  prima degli LLM negli archivi di dati, dove il difetto ha un nome celebre,
  *SQL injection*, e il capitolo sull'AI responsabile ci torna sopra. Tenere
  separate istruzioni e dati è la prima linea di difesa.
- **Aggirare le regole** (*jailbreak*, cioè «evasione»). Con formulazioni
  astute, giochi di ruolo, richieste indirette, si può indurre il modello a
  scavalcare le sue regole di sicurezza. La precedenza che il modello dà al
  regista sul pubblico, di cui si diceva sopra, è un'abitudine appresa, non una
  serratura.
- **Invenzioni** (*allucinazioni*: il nome viene dal fatto che il modello
  riferisce con sicurezza cose che non ha davanti). Un modello genera testo
  plausibile, non necessariamente vero, e può inventare fatti, citazioni e
  riferimenti senza il minimo tentennamento. Il prompt può ridurre il rischio
  (chiedergli di citare le fonti, di ammettere «non lo so»), ma non lo azzera:
  verificare tocca a chi legge.

Nessuno di questi rischi si risolve con una frase magica, ed è il punto da cui
siamo partiti. Il prompt è il primo livello, e da solo porta lontano; ma i
problemi seri (governare ciò che entra nella finestra, orchestrare più
chiamate in un ciclo che si corregge) vivono ai livelli sopra, il contesto e
il loop, che affrontiamo nelle sezioni seguenti.

`````{tab} Elementare

```{admonition} Da ricordare
:class: important
- Il **messaggio è il primo livello**, il più immediato: potente, ma non un
  incantesimo. La frase magica non esiste; esiste il messaggio scritto bene.
- Un buon messaggio tiene distinte quattro cose invece di impastarle:
  **l'ordine** (cosa fare), **lo sfondo** (con quale tono, per chi, con quali
  regole), **il materiale** su cui lavorare e **il segnale di via**, cioè il
  punto in cui lasci la penna al modello.
- Chi costruisce l'applicazione può regolare due manopole che nella chat non
  ci sono: quanto lasciarlo **osare** e quanto **restringere il ventaglio**
  delle parole possibili. Bassa audacia per i fatti e per il codice, più alta
  per inventare; e si muove una manopola per volta, altrimenti non si sa più
  quale delle due ha fatto cosa.
- **Mostrare esempi già svolti** dentro il messaggio è la leva più affidabile
  di tutte: il modello non impara niente di nuovo, ma capisce che cosa vuoi e
  in che forma lo vuoi.
- **Chiedere i passaggi** invece del risultato secco aiuta davvero, ma non
  dappertutto: aiuta quando c'è un conto o dei simboli da manipolare, e non
  sposta quasi nulla sulle domande di conoscenza o di giudizio. Chiedere la
  stessa cosa più volte e tenere la risposta che torna più spesso è il rimedio
  a una catena che imbocca la strada sbagliata.
- Se la risposta la deve leggere un programma, **chiedi le caselle** invece del
  racconto: la forma la puoi imporre, il contenuto va comunque controllato.
- Tre cose da sapere e non temere: nel testo che gli dai possono nascondersi
  istruzioni scritte da altri, le regole di sicurezza si possono aggirare con
  formulazioni astute, e un modello **inventa** con la stessa sicurezza con cui
  dice il vero. Chiedergli le fonti aiuta; verificare aiuta di più.
```

`````

`````{tab} Superiore

```{admonition} Da ricordare
:class: important
- Il **prompt è il primo livello**, il più immediato: potente, ma non un
  incantesimo. Il «prompt magico» non esiste; esiste il prompt costruito bene.
- Un prompt ha quattro parti (**istruzione, sfondo, dato d'ingresso,
  indicatore d'output**) e vive in un formato a ruoli **system / user /
  assistant**, con priorità (morbida) al system. «Sfondo» e non «contesto»
  perché in questo capitolo il contesto è l'intera finestra.
- **Temperatura** e **top_p** regolano il campionamento: bassa per fatti e
  codice, alta per creatività; muovi una manopola per volta, perché non sono
  indipendenti (il nucleo si calcola sulla distribuzione già riscalata). Non
  sono esposte nelle interfacce di chat. E `T = 0` rende deterministica la
  regola di scelta, non il servizio {cite}`he2025nondeterminism`: un A/B fra
  prompt vuole più di una esecuzione per lato. La matematica del
  decoding è nel {doc}`capitolo sui Transformer </Transformers/overview>`.
- Gli **esempi** condizionano il modello senza addestrarlo (*in-context
  learning*, GPT-3 {cite}`brown2020language`): zero-shot, one-shot, few-shot.
- Far **ragionare a voce alta** aiuta, ma **dove**: chain-of-thought
  {cite}`wei2022chain`, e in zero-shot il «ragioniamo passo per passo»
  {cite}`kojima2022zeroshot`, valgono circa +12 punti in matematica e +14 sul
  simbolico, e quasi nulla altrove {cite}`sprague2025cot`. La catena è una
  **traccia ispezionabile**, non una spiegazione: la fedeltà è misurata bassa e
  **cala** con la scala {cite}`turpin2023unfaithful, lanham2023faith`.
  La **self-consistency** {cite}`wang2023selfconsistency` campiona più catene e
  vota la risposta più frequente (estensione ad albero: Tree of Thoughts
  {cite}`yao2023tree`); corregge il rumore di campionamento, non un bias
  condiviso da tutte le catene.
- Chiedi **output strutturato** (JSON/campi) per la lettura a valle: garantisce
  la validità sintattica, non quella di contenuto, e sui compiti a ragionamento
  il vincolo di formato **costa** accuratezza {cite}`tam2024format`. E ricorda
  i rischi (**prompt injection, jailbreak, allucinazioni**) che riprenderemo
  nella sezione su LLMOps.
```

`````
