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
abbiamo progettati noi per essere privi di ambiguità, l'italiano e l'inglese
no.

## Perché il linguaggio è così difficile

`````{tab} Elementare

Prendi la frase "Ho visto un uomo con il binocolo". Chi ha il binocolo? Tu, che
guardavi, oppure l'uomo che hai visto? Nessuna delle due letture è sbagliata:
la frase è **ambigua**, e solo il contesto scioglie il dubbio.

E il contesto cambia tutto. La parola "campo" vuol dire una cosa in "campo di
grano", un'altra in "campo magnetico", un'altra ancora in "campo da calcio".
Poi ci sono i **sinonimi**: "auto", "macchina" e "vettura" indicano lo stesso
oggetto, e una macchina deve capirlo. Ci sono le parole piccole che da sole non
vogliono dire niente e vanno a pescare il significato in quello che è già
stato detto: in "Marco ha preso il libro e l'ha letto", quel "l" è il libro, ma
per saperlo bisogna ricordarsi la prima metà della frase. Questo rimando
all'indietro ha un nome, l'**anafora**, e tornerà spesso nel capitolo. E c'è
l'**ironia**: se dico "che bella giornata" mentre diluvia, intendo l'esatto
contrario. Nessuna di queste cose è scritta nelle parole: sta tra le righe, ed
è lì che le macchine si perdono.

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
continuo, e attraversano tutto il resto del capitolo.

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
  naturale, estraendo la risposta da un testo o generandola.

Sono compiti diversi, e per decenni si sono risolti con programmi diversi: uno
per lo spam, uno per la traduzione, uno per le entità. La storia recente li ha
avvicinati fino quasi a unificarli, e in un senso molto preciso: oggi si
scrivono tutti e quattro come «continua questo testo», si danno in pasto a un
unico modello e si cambia soltanto la richiesta che gli si fa. Il come, e a
quale prezzo, è il filo che percorre il capitolo fino all'ultima sezione.

## Una parabola storica: dalle regole ai Transformer

```{figure} ../figures/nlp-parabola-storica.svg
:name: fig-nlp-storia
:alt: "Linea del tempo con quattro tappe: sistemi a regole anni '50-'80, metodi statistici anni '90, reti neurali dal 2013, Transformer dal 2017."
:width: 90%

Quattro stagioni del NLP. A ogni passaggio la conoscenza linguistica scritta a
mano lascia spazio a modelli che imparano dai dati.
```

`````{tab} Elementare

All'inizio si provò a spiegare la lingua alla macchina **regola per regola**:
liste di parole, grammatiche compilate a mano da linguisti. Funzionava su frasi
semplici, ma le eccezioni dell'italiano sono infinite e le regole diventavano
ingestibili.

Poi arrivò l'idea di **contare**. Invece di dire alla macchina *come* funziona
la lingua, le si dà da leggere montagne di testo e la si lascia notare le
regolarità: dopo "buon" viene spesso "giorno", raramente "sasso". Con la
diffusione di internet il testo da leggere è diventato praticamente infinito,
e i modelli hanno imparato a **rappresentare ogni parola con una manciata di
numeri**. Sono coordinate, come la latitudine e la longitudine di un posto
sulla carta: ogni parola diventa un punto su una mappa, e la mappa la disegna
il programma da solo, leggendo, mettendo vicine le parole che compaiono negli
stessi contesti. Così "re" e "regina" finiscono uno accanto all'altra. L'ultimo
salto, nel 2017, è stato un modello capace di guardare l'intera frase in una
volta e di decidere, parola per parola, quali delle altre contano davvero per
capirla: si chiama **Transformer**, e a quel nome è dedicato un capitolo
apposta.

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

Con i numeri in mano affrontiamo i compiti, uno alla volta: **dare
un'etichetta a un testo intero** (da Naive Bayes alla regressione logistica),
**scommettere su quale parola verrà** (i modelli *n-gram*), **ricordare** ciò
che si è letto prima (le reti ricorrenti), **tradurre** (l'architettura
encoder–decoder e l'attenzione), **dire il mestiere di ogni singola parola** e
riconoscere nomi di persona, di luogo e date (POS tagging e NER, con il
classico algoritmo di Viterbi), **scoprire com'è costruita una frase** (il
parsing) e, per chiudere il cerchio aperto da ELIZA nell'Introduzione,
**parlare con le macchine**: dialogo e chatbot. Ovunque, esempi in Python su
`scikit-learn` e PyTorch, tenendo l'italiano come lingua di lavoro.

L'ultima tappa, i **Transformer**, merita un capitolo tutto suo: è
l'architettura che ha ridefinito non solo il NLP ma buona parte dell'AI
contemporanea, e la tratteremo in dettaglio nel capitolo dedicato. Qui basti
sapere dove stiamo andando: da un calcolatore che nel 1954 arrancava su
sessanta frasi, a modelli che oggi traducono, riassumono e conversano, senza
mai, va detto con onestà, "capire" nel senso in cui capiamo noi.
