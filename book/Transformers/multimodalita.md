# Grandi modelli di linguaggio e multimodalità

L'architettura del 2017 era una macchina per tradurre. Quello che è successo
dopo somiglia a ciò che accadde col motore a scoppio: inventato per la
carrozza, finì su navi, aerei e generatori. Il Transformer è stato smontato
nelle sue due torri (encoder e decoder) e ciascuna, presa da sola e
ingrandita, è diventata una famiglia di modelli: da un lato quelli che
*capiscono* il testo, dall'altro quelli che lo *generano*. Poi qualcuno ha
provato a dargli in pasto le immagini, e ha funzionato anche lì.

## GPT, BERT, T5: tre modi di studiare la lingua

I tre grandi capostipiti si distinguono per come vengono **pre-addestrati**:
quale esercizio fanno, su miliardi di frasi, prima di essere rifiniti su un
compito specifico.

```{figure} ../figures/bert-vs-gpt.svg
:name: fig-bert-vs-gpt
:alt: "Confronto fra due matrici di attenzione sulla stessa frase. In BERT l'attenzione è bidirezionale: ogni parola può guardare tutte le altre, prima e dopo di sé, e la matrice è piena. In GPT l'attenzione è causale: ogni parola vede solo sé stessa e quelle che la precedono, e la metà superiore della matrice è oscurata."
:width: 96%

La stessa frase, due permessi di lettura. Non cambia l'architettura: cambia
cosa ogni parola ha il diritto di guardare, e da lì discende tutto il resto.
```

{numref}`fig-bert-vs-gpt` spiega perché i due modelli finiscano a fare
mestieri diversi. Chi può guardare anche a destra capisce meglio una frase che
ha già davanti (classificare, estrarre, rispondere su un testo dato); chi vede
solo a sinistra è costretto a indovinare il seguito, ed è esattamente
l'esercizio che serve per generare.

`````{tab} Elementare
Immagina tre studenti con tre metodi diversi. **GPT** studia coprendo con la
mano il resto della pagina: legge "Il gatto nero salta sul..." e prova a
indovinare la parola dopo, milioni di volte (così diventa bravissimo a
*continuare* un testo, cioè a scrivere). **BERT** studia con gli esercizi a
buchi: riceve "Il gatto ___ salta sul muro" e indovina la parola mancante
guardando sia prima che dopo il buco (così diventa bravissimo a *capire* le
frasi, meno a scriverle). **T5** trasforma ogni compito in un tema: "traduci:
...", "riassumi: ...", e la risposta è sempre un testo (un formato unico per
tutti gli esercizi). Quando ChatGPT ti risponde, sotto c'è il metodo di GPT:
indovinare la parola successiva, solo con miliardi di esempi alle spalle.

C'è un quarto modo di studiare, meno noto e istruttivo, che nasce da
un'obiezione all'esercizio a buchi: se si cancella una parola su sette, per sei
parole su sette la rete **non impara niente**, perché non le viene chiesto
nulla. **ELECTRA** cambia il gioco: invece di cancellare, *sostituisce* qualche
parola con un'alternativa plausibile, prodotta da un modellino apposta, e poi
chiede alla rete grande di fare il correttore di bozze, dicendo per **ogni**
parola se è quella originale o un'intrusa. Il compito è più povero (una
risposta sì/no invece di indovinare una parola fra decine di migliaia) ma
riguarda tutto il testo, e a parità di calcolo si impara molto di più.
`````

`````{tab} Superiore
**GPT** (OpenAI, 2018) è un Transformer *decoder-only* con maschera causale,
addestrato come modello di linguaggio autoregressivo: massimizza
$\prod_t p(x_t \mid x_1, \dots, x_{t-1})$. La linea di scala culmina in GPT-3
{cite}`brown2020language` (175 miliardi di parametri), che mostra capacità
*few-shot*: adattarsi a un compito descritto nel prompt, senza aggiornare i
pesi. **BERT** {cite}`devlin2019bert` (Google) è *encoder-only* e
bidirezionale, pre-addestrato con *masked language modeling* (predire il ~15%
di token mascherati) e *next sentence prediction*; eccelle nei compiti di
comprensione (classificazione, estrazione di risposte) previo fine-tuning.
**T5** (Google, 2019) mantiene l'encoder–decoder completo e riformula ogni
task NLP come *text-to-text*, mostrando che un solo formato copre traduzione,
sintesi, classificazione. La lezione comune: pre-addestramento
auto-supervisionato su corpora enormi + adattamento leggero (il *transfer
learning* che avevamo visto per le immagini, arrivato al linguaggio).

**ELECTRA** {cite}`clark2020electra` attacca l'inefficienza del masked language
modeling: mascherando il $15\%$ dei token, il segnale di addestramento arriva
solo da quel $15\%$. La sostituisce con la ***replaced token detection***. Un
**generatore** piccolo (un MLM ordinario) rimpiazza i token mascherati con
campioni plausibili; il **discriminatore**, che è ELECTRA, riceve la sequenza
così corrotta e classifica **ogni posizione** come originale o sostituita. Il
segnale viene da tutta la sequenza, il compito binario è più economico del
softmax sul vocabolario, e a valle si getta il generatore e si rifinisce il
discriminatore. Gli autori riportano che un modello addestrato su una sola GPU
per quattro giorni supera GPT su GLUE pur avendo usato trenta volte meno
calcolo, e che alla scala grande si eguagliano RoBERTa e XLNet con meno di un
quarto del loro.

La somiglianza con una **GAN** è dichiarata dagli autori stessi, e istruttiva
soprattutto per dove si rompe: il generatore **non** è addestrato a ingannare
il discriminatore (è addestrato con la sua verosimiglianza, come un normale
MLM), non c'è vettore di rumore in ingresso, e quando produce per caso il token
giusto quello viene etichettato come *originale* e non come falso. È
un'architettura avversaria nella forma e cooperativa nella sostanza, e chi ha
letto il capitolo sulle GAN riconoscerà quanto di quella instabilità venga
proprio dal pezzo che qui è stato tolto.
`````

## Oltre il testo: Vision Transformer e modelli multimodali

```{figure} ../figures/vit-transformer-immagini.svg
:name: fig-vit
:alt: "Un'immagine viene divisa in una griglia di patch quadrate; le patch vengono messe in fila come una sequenza, proiettate in vettori e date in pasto a un encoder Transformer, la cui uscita produce la classe dell'immagine."
:width: 96%

Il Vision Transformer non inventa un meccanismo nuovo: taglia l'immagine in
tessere e le tratta come parole. Da lì in poi è lo stesso encoder del testo.
```

Il passaggio mostrato in {numref}`fig-vit` è meno innocente di quanto sembri.
Tagliare e mettere in fila butta via ciò che una convoluzione dava per
scontato, cioè che i pixel vicini siano imparentati: il ViT quella parentela
deve impararla dai dati, e infatti ha bisogno di molti più esempi per farlo.

`````{tab} Elementare
E le immagini? Il trucco è di una semplicità disarmante: si taglia la foto in
tessere quadrate, come un mosaico, e si mettono le tessere in fila come se
fossero le parole di una frase. A quel punto il Transformer fa quello che sa
fare: per capire la tessera con l'orecchio del gatto, va a "guardare" anche
quella con la coda, dall'altra parte della foto. I modelli **multimodali**
fanno il passo successivo: imparano testo e immagini insieme, così puoi
mostrare una foto e fare una domanda a parole, e la risposta arriva a parole.
È quello che fa un assistente moderno quando gli carichi l'immagine di un
modulo e gli chiedi di spiegartelo.
`````

`````{tab} Superiore
Il **Vision Transformer** (ViT {cite}`dosovitskiy2021image`) suddivide
l'immagine in patch (tipicamente $16 \times 16$ pixel), le proietta
linearmente in embedding e le tratta come token, con un positional encoding
per la posizione spaziale; con pre-addestramento su dataset molto grandi
raggiunge e supera le CNN del capitolo sulla visione, pur non avendo il loro
*bias induttivo* di località. Sul fronte multimodale, **CLIP**
{cite}`radford2021learning` allinea in uno spazio comune embedding di immagini
e testi tramite addestramento contrastivo su coppie immagine–didascalia; i
modelli generativi di immagini come DALL·E e Stable Diffusion usano componenti
Transformer per condizionare la generazione sul testo; e modelli come GPT-4
(2023) accettano input misti testo+immagine. Il filo conduttore tecnico:
qualunque dato riducibile a una **sequenza di token** (parole, patch,
frammenti audio) è terreno di gioco per l'attenzione.
`````

Qui ci fermiamo al principio, che è il filo di questo capitolo: tutto ciò che
si riduce a una sequenza di token è terreno dell'attenzione. Come si costruisca
davvero un modello che vede e parla è un'altra storia, e ha un capitolo suo più
avanti: allineare due spazi separati senza fonderli, innestare un occhio su un
modello di linguaggio già addestrato, oppure dare a pixel e parole un unico
vocabolario sono tre risposte diverse alla stessa domanda, e ciascuna si paga
in modo diverso.

## Fuori dal linguaggio: AlphaFold 2 e la forma delle proteine

Il caso più clamoroso di attenzione applicata a un dominio che con il testo non
c'entra nulla è arrivato nel novembre 2020, a CASP14: **AlphaFold 2** predice la
struttura tridimensionale delle proteine con un'accuratezza vicina a quella dei
metodi sperimentali, chiudendo di fatto un problema aperto da mezzo secolo.

```{figure} ../figures/alphafold-2.svg
:name: fig-alphafold
:alt: "Catena di elaborazione di AlphaFold 2: dalla sequenza di amminoacidi e dall'allineamento multiplo di sequenze evolutivamente imparentate si passa all'Evoformer, che fa scambiare informazione fra la rappresentazione delle sequenze e quella delle coppie di residui; il modulo struttura converte infine il risultato in coordinate tridimensionali."
:width: 100%

I due ingressi contano quanto l'architettura. Oltre alla sequenza da ripiegare,
AlphaFold legge l'allineamento con le proteine imparentate: l'evoluzione ha già
fatto milioni di esperimenti, e quelli sono i dati.
```

Il blocco centrale di {numref}`fig-alphafold` è dove l'attenzione fa il suo
mestiere. Due residui lontani nella sequenza possono trovarsi vicini nello
spazio, ed è esattamente il tipo di relazione a lungo raggio che le
convoluzioni faticano a vedere e che l'attenzione tratta come un caso normale.

`````{tab} Elementare

Una proteina nasce come una lunga catena di amminoacidi: venti mattoncini
agganciati in un ordine preciso, un'informazione monodimensionale come una
collana di perline. Appena esiste, però, si ripiega su se stessa in una forma
tridimensionale, e da quella forma dipende tutto ciò che sa fare: catalizzare
una reazione, trasportare ossigeno, agganciare un farmaco.

È come una collana di perline magnetiche che, ogni volta che la lasci cadere,
si annoda esattamente allo stesso modo. Prevedere quel nodo dal solo ordine
delle perline era il problema.

L'idea centrale è bellissima e non è di forza bruta: **leggere il segnale che
l'evoluzione ha lasciato nei dati**. Confrontando la stessa proteina in
migliaia di specie diverse si scopre che certe posizioni della catena mutano
sempre *in coppia*: se una cambia, l'altra cambia con lei. Il motivo è che
quelle due posizioni si toccano nello spazio: se una muta e l'altra no, la
proteina non funziona più e l'individuo non lascia discendenti. La
co-evoluzione è quindi una traccia indiretta della vicinanza fisica.

AlphaFold impara a leggere quella traccia e a trasformarla in geometria.

`````

`````{tab} Superiore

Il sistema lavora su due rappresentazioni tenute in dialogo dall'**Evoformer**,
una pila di blocchi che alternano attenzione e aggiornamenti moltiplicativi
sui triangoli:

- l'**allineamento multiplo di sequenze** (MSA), la stessa proteina in molte
  specie, da cui emerge il segnale di co-evoluzione;
- la **rappresentazione di coppia**, una matrice indicizzata su ogni coppia di
  residui $(i,j)$: di fatto un grafo pesato sulla catena.

Le due si aggiornano a vicenda a ogni blocco: quel che si scopre nell'MSA
raffina le coppie, e viceversa. Sulla rappresentazione di coppia agisce la
**triangle attention**, che aggiorna $(i,j)$ guardando i cammini attraverso un
terzo residuo $k$. Serve a rendere esprimibile un vincolo che l'attenzione
ordinaria non vede: le distanze devono rispettare la **disuguaglianza
triangolare**, perché sono distanze in uno spazio reale, non affinità
arbitrarie. L'architettura non impone il vincolo: lo rende rappresentabile,
facendo dipendere ogni arco dagli altri due lati del triangolo. Che venga poi
rispettato è cosa che la rete impara dai dati, non una garanzia strutturale.

Il **modulo di struttura** finale produce le coordinate atomiche trattando ogni
residuo come un sistema di riferimento rigido, e il tutto viene ripassato più
volte (*recycling*): l'uscita rientra come ingresso e la struttura si affina.

Due conseguenze che vale la pena tenere. La prima: il modello stima anche la
**propria confidenza** (pLDDT), e le regioni a bassa confidenza corrispondono
spesso a parti realmente disordinate della proteina; un raro caso in cui
l'incertezza dichiarata ha un significato fisico. La seconda: dipendendo
dall'MSA, il metodo è più debole dove la storia evolutiva è povera (proteine
orfane, anticorpi progettati, molecole di sintesi).

Il database pubblico che ne è seguito contiene oltre **200 milioni** di
strutture predette, praticamente ogni proteina nota.

`````

## Vantaggi e sfide

Il quadro va chiuso con la stessa onestà del capitolo sui confronti. I
vantaggi sono reali: gestione di contesti lunghi, addestramento parallelo,
un'unica architettura per testo, immagini e audio. Ma le sfide non sono
dettagli:

- **Risorse**: addestrare un grande modello richiede cluster di GPU, mesi di
  calcolo e consumi energetici ingenti; anche solo *eseguirlo* può richiedere
  hardware fuori dalla portata di un laboratorio piccolo.
- **Dati**: i corpora da miliardi di parole contengono errori, stereotipi e
  contenuti tossici, e i modelli li assorbono; i **bias** dei dati diventano
  bias del modello.
- **Affidabilità**: un modello autoregressivo produce la continuazione più
  plausibile, non necessariamente quella *vera*: le "allucinazioni" (risposte
  fluenti e sbagliate) sono un limite strutturale, non un incidente.

```{admonition} Da ricordare
:class: important
- **GPT** = decoder-only, indovina la parola successiva, forte nel generare;
  **BERT** = encoder-only bidirezionale, forte nel capire; **T5** = tutto
  come text-to-text.
- La ricetta comune è **pre-addestramento** auto-supervisionato su corpora
  enormi + adattamento (fine-tuning o prompt).
- **ELECTRA** mostra che l'obiettivo conta quanto l'architettura: sostituire
  qualche token e far dire alla rete, **per ogni posizione**, se è originale o
  intrusa, dà segnale su tutta la sequenza invece che sul solo $15\%$
  mascherato, e a parità di calcolo rende molto di più.
- **ViT** tratta l'immagine come una frase di tessere $16\times16$; i modelli
  **multimodali** (CLIP, GPT-4) allineano testo e immagini.
- Costi computazionali, bias nei dati e allucinazioni sono limiti
  strutturali, da mettere in conto quanto i vantaggi.
```
