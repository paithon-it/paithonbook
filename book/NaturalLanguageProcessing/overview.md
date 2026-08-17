# Natural Language Processing

Nel 1954 un gruppo di ricercatori di IBM e della Georgetown University
annunciò al mondo una traduzione automatica dal russo all'inglese: sessanta
frasi, tradotte da un calcolatore, davanti alla stampa entusiasta. Il progetto
prometteva di risolvere la traduzione "in tre, forse cinque anni". Ne
servirono molti di più, e la lezione che ne uscì è ancora il cuore di questo
capitolo: il linguaggio umano sembra semplice perché lo maneggiamo senza
sforzo, ma per una macchina è uno dei problemi più ostici che esistano.

Il **Natural Language Processing** (NLP, elaborazione del linguaggio naturale)
è la disciplina che insegna ai calcolatori a leggere, capire e produrre testo.
"Naturale" per distinguerlo dai linguaggi *artificiali* come Python: quelli li
abbiamo progettati noi perché ogni istruzione voglia dire una cosa sola,
l'italiano e l'inglese no.

Arriviamo qui dal deep reinforcement learning, che ci lascia una domanda in
mano: come si scrive una ricompensa che dica davvero quel che vogliamo. La
risposta che il libro darà, molto più avanti, non sarà un numero calcolato da un
programma ma il giudizio di una persona che legge due frasi e dice quale
preferisce. Prima però bisogna sapere che cos'è una frase, per una macchina, ed
è quel che si fa da qui in poi.

## Perché il linguaggio è così difficile

`````{tab} Elementare

Prendi la frase «Ho visto un uomo con il binocolo». Chi ha il binocolo? Tu, che
guardavi, oppure l'uomo che hai visto? Nessuna delle due letture è sbagliata:
la frase è **ambigua**, e solo il contesto scioglie il dubbio.

E il contesto cambia tutto. La parola «campo» vuol dire una cosa in «campo di
grano», un'altra in «campo magnetico», un'altra ancora in «campo da calcio».
Poi ci sono i **sinonimi**: «auto», «macchina» e «vettura» indicano lo stesso
oggetto, e una macchina deve capirlo. Ci sono le parole piccole che da sole non
vogliono dire niente e vanno a pescare il significato in quello che è già stato
detto: in «Marco ha preso il libro e l'ha letto», quel «l’» è il libro, ma per
saperlo bisogna ricordarsi la prima metà della frase. Questo rimando
all'indietro ha un nome, l’**anafora**, e tornerà nell'ultima sezione del
capitolo, quella sul dialogo: lì le parole piccole dovranno pescare il
significato non nella stessa frase, ma in una battuta detta prima da un'altra
persona. E c'è l’**ironia**: se dico «che bella giornata» mentre diluvia,
intendo l'esatto contrario. Nessuna di queste cose è scritta nelle parole: sta
tra le righe, ed è lì che le macchine si perdono.

`````

`````{tab} Superiore

La difficoltà del linguaggio si può decomporre in alcuni fenomeni ricorrenti:

- **Ambiguità** a più livelli. *Lessicale*: un termine ha più sensi (la
  disambiguazione del senso è il compito noto come *word sense
  disambiguation*). *Sintattica*: "Ho visto un uomo con il binocolo" ammette
  due alberi di parsing diversi (attacco del sintagma preposizionale).
- **Dipendenza dal contesto**. Il significato di un token è funzione della
  finestra che lo circonda; i pronomi (*anafora*) vanno risolti rispetto ad
  antecedenti anche lontani.
- **Sinonimia e polisemia**. La stessa forma superficiale copre sensi diversi e
  sensi diversi condividono forme: la relazione tra stringhe e significati è
  molti-a-molti.
- **Pragmatica**. Ironia, sarcasmo e implicature richiedono conoscenza del
  mondo e dell'intenzione del parlante, non ricavabile dalla sola sintassi.

La conseguenza pratica è che non esiste una funzione deterministica
testo $\to$ significato: il NLP moderno la *stima* da dati, modellando
$P(\text{significato} \mid \text{testo}, \text{contesto})$.

`````

## I compiti tipici

Il NLP non è un problema unico ma una famiglia di compiti. Quattro tornano di
continuo, e tre di loro avranno più avanti una sezione tutta per sé.

- **Classificazione** e analisi del sentimento (*sentiment analysis*, che è il
  nome inglese con cui la si trova ovunque): assegnare un'etichetta a un testo.
  È spam o no? Questa recensione è positiva o negativa? Questa email va allo
  sportello "reclami" o "fatturazione"?
- **Traduzione automatica** (*machine translation*): trasformare una frase da
  una lingua all'altra preservandone il senso (proprio la promessa di
  Georgetown del 1954).
- **Riconoscimento di entità nominate** (*Named Entity Recognition*, NER):
  individuare nel testo persone, luoghi, organizzazioni, date. In "Enrico Fermi
  nacque a Roma nel 1901", un sistema NER etichetta *Enrico Fermi* come
  persona, *Roma* come luogo, *1901* come data.
- **Question answering**: rispondere a una domanda posta in linguaggio
  naturale, estraendo la risposta da un testo o generandola. È l'unico dei
  quattro che qui non avrà una sezione sua: per rispondere sul serio bisogna
  prima andare a cercare il testo giusto in un archivio, e quel mestiere ha
  bisogno di attrezzi che arrivano nel capitolo dopo questo.

Sono compiti diversi, e per decenni si sono risolti con programmi diversi: uno
per lo spam, uno per la traduzione, uno per le entità. La storia recente li ha
avvicinati fino quasi a unificarli, e vale la pena vedere come, perché sembra
un gioco di prestigio e non lo è.

Il protagonista è un **modello**: un programma che non è stato scritto
istruzione per istruzione, ma ricavato da una montagna di testo, e il cui
unico mestiere è tirare a indovinare come continua un pezzo di scrittura. Un
mestiere solo, quindi. Il trucco sta nel riscrivere ogni compito in modo che
la risposta sia proprio la continuazione. Non gli si chiede «questa email è
spam?»: gli si dà da finire un testo che dice «Email: *vinci subito un
premio*. Questa email è spam? Risposta:», e la parola con cui continua è il
verdetto. Non gli si chiede di tradurre: gli si dà «Italiano: *il gatto nero
salta sul muro*. Inglese:». Stesso programma, stessi numeri dentro: cambia
soltanto il foglio che gli si mette davanti. Il come, e a quale prezzo, è il
filo che percorre il capitolo fino all'ultima sezione.

## Una parabola storica: dalle regole ai Transformer

Come ci si sia arrivati è una storia in quattro tappe, e la
{numref}`fig-nlp-storia` le mette in fila. A cambiare, di tappa in tappa, è
sempre la stessa cosa: **chi mette la conoscenza della lingua dentro il
programma**. Prima un linguista che scrive regole a mano, alla fine il testo
stesso. L'ultima tappa porta il nome di un'architettura, il *Transformer*, che
è il modo in cui oggi si costruiscono quasi tutte queste macchine; qui basta
sapere che esiste, perché ha un capitolo tutto suo, il prossimo.

```{figure} ../figures/nlp-parabola-storica.svg
:name: fig-nlp-storia
:alt: "Linea del tempo con quattro tappe: regole (anni '50-'80), grammatiche scritte a mano; statistica (anni '90-2000), conteggi su grandi raccolte di testo; reti neurali (dal 2013), word2vec ed embedding densi; Transformer (dal 2017), attention, BERT, GPT. In basso una scritta: meno regole scritte a mano, più conoscenza appresa dai dati."
:width: 90%

Quattro stagioni del NLP. A ogni passaggio la conoscenza linguistica scritta a
mano lascia spazio a programmi che la ricavano dai testi.
```

`````{tab} Elementare

**Prima tappa, le regole.** All'inizio si provò a spiegare la lingua alla
macchina una regola per volta: liste di parole, grammatiche compilate a mano
da linguisti. Funzionava su frasi semplici, ma le eccezioni dell'italiano sono
infinite e le regole diventavano ingestibili.

**Seconda tappa, i conteggi.** Invece di dire alla macchina *come* funziona la
lingua, le si dà da leggere montagne di testo e la si lascia notare le
regolarità: dopo "buon" viene spesso "giorno", raramente "sasso". Con la
diffusione di internet il testo da leggere è diventato praticamente infinito.

**Terza tappa, la mappa delle parole.** Qui i programmi hanno cominciato a
rappresentare ogni parola con **qualche centinaio di numeri**. Sono coordinate,
come la latitudine e la longitudine di un posto sulla carta: ogni parola
diventa un punto su una mappa. Con due soli numeri la mappa si disegnerebbe su
un foglio; con trecento non si disegna più, ma la si può ancora misurare, e due
punti vicini restano due parole affini. Il bello è che la mappa la disegna il
programma da solo, e con una regola sciocca: mette vicine le parole che si
trovano in mezzo alle stesse compagnie. "Re" e "regina" compaiono negli stessi
posti (dopo "il" e "la", vicino a "trono", "corona", "regno"), quindi finiscono
uno accanto all'altra.

**Quarta tappa, il 2017.** Fino a lì un programma leggeva la frase parola per
parola, in fila, e a ogni passo si portava dietro un riassunto di quello che
aveva già letto. L'ultimo salto è stato un modello che guarda l'intera frase
tutta insieme e decide, parola per parola, quali delle altre contano davvero
per capirla: si chiama **Transformer**, e a quel nome è dedicato il capitolo
successivo a questo.

`````

`````{tab} Superiore

La traiettoria del NLP attraversa quattro fasi ({numref}`fig-nlp-storia`):

1. **Sistemi a regole** (anni '50–'80). Grammatiche formali e basi di
   conoscenza scritte a mano; approccio *symbolic AI*. Fragile fuori dal
   dominio previsto, costoso da mantenere.
2. **Metodi statistici** (anni '90–2000). Modelli probabilistici stimati da
   corpora annotati: $n$-gram per il *language modeling*, Hidden Markov Model
   per il *part-of-speech tagging*, traduzione statistica allineata a livello
   di parola. Il paradigma diventa "impara dai dati".
3. **Reti neurali** (dal ~2013). Le parole diventano vettori densi
   (*embedding*): word2vec (Mikolov et al., 2013) colloca ogni parola in
   $\mathbb{R}^d$ così che il prodotto scalare catturi la similarità
   semantica. Reti ricorrenti (RNN, LSTM) modellano le sequenze.
4. **Transformer** (dal 2017). Il paper *Attention Is All You Need*
   {cite}`vaswani2017attention` sostituisce la ricorrenza con il meccanismo di
   *self-attention*, permettendo parallelismo e dipendenze a lungo raggio. Da
   qui BERT, la famiglia GPT e i moderni *large language model*.

Il filo conduttore è una progressiva **riduzione della conoscenza linguistica
inserita a mano** in favore di rappresentazioni apprese direttamente dal testo.

`````

## Come è organizzato il capitolo

Nelle sezioni seguenti seguiremo questa stessa parabola, ma da vicino. Ogni
tappa ha un nome tecnico, e qui lo mettiamo fra parentesi solo perché lo si
riconosca quando arriverà: quello che conta è il lavoro, scritto in italiano.

Si parte dalla **cassetta degli attrezzi classica**: cercare in un testo tutti i
pezzi che hanno una certa forma (le *espressioni regolari*), ripulirlo perché
due scritture della stessa parola non contino come due parole diverse (la
*normalizzazione*), misurare quanto due parole si somigliano (la *distanza di
edit*, quella che sta dietro al correttore del telefono). Poi il testo diventa
numeri: prima lo si taglia in pezzi (la *tokenizzazione*), poi si contano i
pezzi (il *bag-of-words*, il "sacchetto di parole"), infine ogni parola riceve
le sue coordinate sulla mappa di cui si diceva sopra (gli *embedding*).

Con i numeri in mano affrontiamo i compiti, uno alla volta.

- **Dare un'etichetta a un testo intero**: spam o non spam, recensione
  entusiasta o stroncatura (i due metodi si chiamano *Naive Bayes* e
  *regressione logistica*).
- **Scommettere su quale parola verrà**, che è il mestiere della barra dei
  suggerimenti sul telefono (i modelli *n-gram*).
- **Ricordare** ciò che si è letto prima, con le reti che leggono in fila
  tenendo un riassunto aggiornato (le *reti ricorrenti*).
- **Tradurre**: una rete legge la frase in una lingua, una seconda la riscrive
  nell'altra (la coppia si chiama *encoder–decoder*), e fra le due nasce l'idea
  che cambierà tutto, l’*attenzione*.
- **Dire il mestiere di ogni singola parola** (nome, verbo, articolo) e
  riconoscere nomi di persona, di luogo e date: sono il *POS tagging* e il
  *NER*, e li risolve un procedimento del 1967, l’*algoritmo di Viterbi*.
- **Scoprire com'è costruita una frase**, cioè quali parole vanno insieme e
  chi fa che cosa a chi (il *parsing*).
- **Parlare con le macchine**: dialogo e chatbot, che chiude il cerchio aperto
  da ELIZA nell'Introduzione.

Ovunque, esempi in Python su `scikit-learn` e PyTorch, tenendo l'italiano come
lingua di lavoro.

Resta l'ultima tappa, i **Transformer**, che merita un capitolo tutto suo: è
l'architettura che ha ridefinito non solo il NLP ma buona parte dell'AI
contemporanea. Qui basti sapere dove stiamo andando: da un calcolatore che nel
1954 arrancava su sessanta frasi, a modelli che oggi traducono, riassumono e
conversano, senza mai, va detto con onestà, "capire" nel senso in cui capiamo
noi. Il senso preciso di quella riserva è che nessuna di queste macchine ha
mai visto un gatto, aperto una porta o avuto fretta: quello che sa della
parola *gatto* è dove quella parola compare rispetto a tutte le altre, e nulla
di ciò a cui la parola si riferisce. È una conoscenza reale e verificabile,
ed è di un altro tipo dalla nostra.
