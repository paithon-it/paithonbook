# Le famiglie di modelli, e oltre il testo

L'architettura del 2017 era una macchina per tradurre. Quello che è successo
dopo somiglia a ciò che accadde col motore a scoppio: inventato per la
carrozza, finì su navi, aerei e generatori. Il Transformer è stato smontato
nelle sue due torri (la torre che legge, l'encoder, e la torre che scrive, il
decoder) e ciascuna, presa da sola e ingrandita, è diventata una famiglia di
modelli: da un lato quelli che *capiscono* il testo, dall'altro quelli che lo
*generano*. Poi qualcuno ha provato a dargli in pasto le immagini, e ha
funzionato anche lì.

## GPT, BERT, T5: tre modi di studiare la lingua

I tre capostipiti, cioè i modelli da cui discendono tutti gli altri, si
distinguono per una cosa sola: l'esercizio che fanno su miliardi di frasi prima
di essere messi al lavoro. Quella fase di studio generale si chiama
**pre-addestramento**, e a seconda dell'esercizio scelto ne esce un modello
bravo a scrivere o uno bravo a capire.

```{figure} ../figures/bert-vs-gpt.svg
:name: fig-bert-vs-gpt
:alt: "Confronto fra due matrici di attenzione sulla stessa frase. In BERT l'attenzione è bidirezionale: ogni parola può guardare tutte le altre, prima e dopo di sé, e la matrice è piena. In GPT l'attenzione è causale: ogni parola vede solo sé stessa e quelle che la precedono, e la metà superiore della matrice è oscurata."
:width: 96%

La stessa frase, due permessi di lettura. Non cambia l'architettura: cambia
cosa ogni parola ha il diritto di guardare, e da lì discende tutto il resto.
```

{numref}`fig-bert-vs-gpt` mostra la sola cosa che li distingue davvero: non
l'architettura, che è la stessa, ma quali parole ciascuno ha il permesso di
guardare. Il primo può guardare anche le parole che vengono dopo, e chi vede
tutta la frase la capisce meglio (dire di che parla, trovarci dentro una
risposta, giudicarla); il secondo vede solo le parole che precedono, ed è
costretto a indovinare come si continua, che è esattamente l'esercizio da fare
per imparare a scrivere. La scheda qui sotto racconta chi sono i due, e conviene
leggerla prima di tornare alla figura.

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

C'è un quarto modo di studiare, meno noto e molto istruttivo, che nasce da
un'obiezione all'esercizio a buchi. Negli esercizi veri di BERT i buchi non
sono uno per frase: si cancella circa il quindici per cento delle parole, cioè
grosso modo una ogni sette, e su quelle sole la rete viene interrogata. Delle
altre sei su sette **non impara niente**, perché non le viene chiesto nulla, e
leggerle è fatica sprecata.

**ELECTRA** cambia il gioco: invece di cancellare, *sostituisce* qualche
parola con un'alternativa plausibile, prodotta da un modellino apposta, e poi
chiede alla rete grande di fare il correttore di bozze, dicendo per **ogni**
parola se è quella originale o un'intrusa. Il compito su ciascuna parola è più
povero (una risposta sì/no invece di indovinarne una fra decine di migliaia) ma
riguarda tutto il testo, e a parità di ore di computer bruciate si impara molto
di più: gli autori misurano che per arrivare dove arrivano i modelli a buchi
del suo tempo, ELECTRA spende meno di un quarto del loro addestramento.
`````

`````{tab} Superiore
**GPT** (OpenAI, 2018) è un Transformer *decoder-only* con maschera causale,
addestrato come modello di linguaggio autoregressivo: massimizza
$\prod_t p(x_t \mid x_1, \dots, x_{t-1})$, dove $x_t$ è il token in posizione
$t$ e il prodotto corre su tutta la sequenza. La linea di scala culmina in GPT-3
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
discriminatore. Il guadagno che gli autori misurano è tutto sull'asse del
**calcolo**: alla scala grande ELECTRA arriva alla resa dei modelli mascherati
confrontabili dell'epoca spendendo meno di un quarto del loro addestramento, e
la versione piccola, quattro giorni su una sola GPU, se la cava meglio di un
modello autoregressivo che di calcolo ne aveva consumato trenta volte tanto.
Non è una vittoria di architettura, è una vittoria di **obiettivo**: la stessa
rete impara di più dalle stesse frasi perché le viene chiesto qualcosa su ogni
posizione invece che su una su sette.

La somiglianza con una **GAN** è dichiarata dagli autori stessi, e istruttiva
soprattutto per dove si rompe: il generatore **non** è addestrato a ingannare
il discriminatore (è addestrato con la sua verosimiglianza, come un normale
MLM), non c'è vettore di rumore in ingresso, e quando produce per caso il token
giusto quello viene etichettato come *originale* e non come falso. È
un'architettura avversaria nella forma e cooperativa nella sostanza, e chi ha
letto il capitolo sulle GAN riconoscerà quanto di quella instabilità venga
proprio dal pezzo che qui è stato tolto.
`````

Fra i tre, il terzo merita un disegno, perché la sua idea è quella che si è
presa il futuro.

```{figure} ../figures/t5-2019.svg
:name: fig-t5-text-to-text
:alt: "Quattro compiti diversi (traduzione, giudizio di accettabilità grammaticale, somiglianza fra due frasi e riassunto) entrano nello stesso modello scritti come testo, ciascuno preceduto da un prefisso che dice di quale compito si tratta; da tutti e quattro esce testo. Nessuna testa specializzata per compito compare nello schema."
:width: 100%

Un solo formato per tutto. Di solito a un modello si attacca in cima un pezzo
diverso per ogni mestiere, uno che sa dare voti, uno che sa scegliere fra due
risposte; T5 non ne attacca nessuno: mette il nome del compito davanti alla
frase, e la risposta esce come testo anche quando è un voto o un'etichetta.
```

L'idea di {numref}`fig-t5-text-to-text` sembra un dettaglio ingegneristico e
invece anticipa il modo in cui oggi si usano i modelli di linguaggio. Se ogni
compito si può scrivere come testo in ingresso e testo in uscita, allora
cambiare compito non richiede di cambiare il modello: basta cambiare quello che
gli si scrive davanti. Quel «quello che gli si scrive davanti» è il **prompt**,
la parola che da qui in avanti tornerà in tutto il capitolo, e che vuol dire
esattamente questo: le istruzioni e il testo che si consegnano al modello prima
che risponda. Dentro il prompt ci si può mettere la sola consegna a parole, o
anche due o tre esercizi già svolti perché il modello capisca che cosa gli si
sta chiedendo; e la scoperta che quei due o tre esempi bastino, senza toccare
un solo numero interno del modello, è una delle cose che hanno stupito di più
chi lavorava su GPT-3.

## Oltre il testo: Vision Transformer e modelli multimodali

Fin qui il Transformer ha sempre avuto in pasto delle parole. Ma se si guarda
bene, l'attenzione non sa niente delle parole: sa solo confrontare liste di
numeri messe in fila. Qualunque cosa si riesca a ridurre a una fila di liste di
numeri, allora, può entrarci dentro, e la prima a provarci è stata la
fotografia.

```{figure} ../figures/vit-transformer-immagini.svg
:name: fig-vit
:alt: "Un'immagine viene divisa in una griglia di patch quadrate; le patch vengono messe in fila come una sequenza, proiettate in vettori e date in pasto a un encoder Transformer, la cui uscita produce la classe dell'immagine."
:width: 96%

Il Vision Transformer non inventa un meccanismo nuovo: taglia l'immagine in
tessere e le tratta come parole. Da lì in poi è lo stesso encoder del testo.
```

Il passaggio mostrato in {numref}`fig-vit` è meno innocente di quanto sembri.
Le reti per le immagini del {doc}`capitolo sul deep learning </DeepLearning/overview>` (le *convoluzioni*, i
filtri che guardano un pezzetto di foto alla volta) hanno una regola scritta
dentro: i puntini vicini fra loro sono imparentati, e vanno guardati insieme.
Tagliare la foto in tessere e metterle in fila butta via quella regola, perché
per l'attenzione due tessere lontanissime e due tessere adiacenti sono
esattamente sullo stesso piano. La parentela fra vicini, allora, il Vision
Transformer deve **impararla**, e imparare qualcosa costa esempi: è la ragione
per cui regge il confronto solo se gli si dà da studiare molta più roba.
Quando i dati sono pochi, la regola scritta a mano vince.

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
per la posizione spaziale. La cosa da portarsi via è la condizione: senza il
*bias induttivo* di località delle CNN del capitolo sulla visione, il ViT
regge il confronto **solo** se pre-addestrato su dataset molto grandi, e sotto
quella soglia resta indietro. La località non è gratis: o la si mette
nell'architettura, o la si compra in dati. Sul fronte multimodale, **CLIP**
{cite}`radford2021learning` allinea in uno spazio comune embedding di immagini
e testi tramite addestramento contrastivo su coppie immagine–didascalia; i
modelli generativi di immagini come DALL·E e Stable Diffusion usano componenti
Transformer per condizionare la generazione sul testo; e modelli come GPT-4
(2023) accettano input misti testo+immagine. Il filo conduttore tecnico:
qualunque dato riducibile a una **sequenza di token** (parole, patch,
frammenti audio) è terreno di gioco per l'attenzione.
`````

Qui ci fermiamo al principio, che è il filo di questo capitolo: tutto ciò che si
riduce a una fila di mattoncini (i *token*: le parole di una frase, le tessere
di una foto, gli spezzoni di un suono) è terreno dell'attenzione. Come si
costruisca davvero un modello che vede e parla è un'altra storia, e ha un
capitolo suo più avanti. Le strade sono tre, e basta averne il nome in mente:
tenere immagini e parole ciascuna nella propria mappa e allenarle a mettere le
cose corrispondenti nello stesso punto; innestare un occhio su un modello di
linguaggio già fatto, lasciando comandare il linguaggio; oppure dare a tessere e
parole un unico vocabolario, come se le tessere fossero le parole di una lingua
in più. Quale convenga, e che cosa costi ciascuna, si vedrà là.

## Fuori dal linguaggio: AlphaFold 2 e la forma delle proteine

L'attenzione, però, non è finita a lavorare solo su testi, foto e suoni. Il caso
più clamoroso è arrivato da una parte che con il linguaggio non c'entra niente:
la biologia.

Le proteine sono le macchine di cui siamo fatti, e ciascuna nasce come una
catena di mattoncini agganciati in fila, che appena esiste si ripiega su sé
stessa in una forma tridimensionale precisa. Da quella forma dipende tutto quel
che la proteina sa fare, e prevederla a partire dalla sola fila di mattoncini
era un problema aperto da mezzo secolo. Nel novembre 2020, alla CASP14, la gara
biennale in cui i programmi che ci provano si sfidano su proteine di cui la
risposta è nota solo agli organizzatori, **AlphaFold 2** ha predetto quelle
forme con un'accuratezza confrontabile con quella delle misure fatte in
laboratorio **nella maggior parte dei casi**, chiudendo di fatto il problema per
le proteine formate da una catena sola.

Le due clausole non sono prudenza di maniera. Restano fuori le proteine fatte di
più catene incastrate, i tratti che una forma stabile non ce l'hanno affatto,
le proteine che ne assumono più d'una a seconda della situazione, e l'effetto
delle mutazioni; e la formula «problema risolto», che allora circolò molto,
stava nel comunicato stampa, non nel giudizio di chi assegnava i punteggi.

```{figure} ../figures/alphafold-2.svg
:name: fig-alphafold
:alt: "Catena di elaborazione di AlphaFold 2: dalla sequenza di amminoacidi e dall'allineamento multiplo di sequenze evolutivamente imparentate si passa all'Evoformer, che fa scambiare informazione fra la rappresentazione delle sequenze e quella delle coppie di residui; il modulo struttura converte infine il risultato in coordinate tridimensionali."
:width: 100%

I due ingressi contano quanto l'architettura. Oltre alla sequenza da ripiegare,
AlphaFold legge l'allineamento con le proteine imparentate: l'evoluzione ha già
fatto milioni di esperimenti, e quelli sono i dati.
```

Il blocco centrale di {numref}`fig-alphafold` è dove l'attenzione fa il suo
mestiere, ed è facile vedere perché sia lei la persona giusta. Gli anelli della
catena, in gergo, si chiamano **residui** (nel libro la parola «residuo» è già
comparsa per le connessioni residue, la scorciatoia attorno a un blocco: è la
stessa parola usata per due cose che non c'entrano niente, e capita). Due
residui lontanissimi lungo la catena possono ritrovarsi appiccicati una volta
che la catena si è ripiegata: è la relazione fra due elementi lontani che i
filtri delle reti per immagini faticano a vedere, ed è esattamente il caso che
l'attenzione tratta come normale, perché per lei ogni coppia è a un passo di
distanza.

`````{tab} Elementare

I mattoncini della catena si chiamano amminoacidi, e ne esistono venti tipi:
una proteina è una fila di quelli, agganciati in un ordine preciso, come una
collana di perline di venti colori. Appena esiste, però, la collana si ripiega
su sé stessa in una forma tridimensionale, e da quella forma dipende tutto ciò
che sa fare: far avvenire una reazione chimica che da sola non avverrebbe,
trasportare ossigeno nel sangue, agganciare un farmaco.

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

Il sistema lavora su due rappresentazioni tenute in dialogo dall’**Evoformer**,
una pila di blocchi che alternano attenzione e aggiornamenti moltiplicativi
sui triangoli:

- l’**allineamento multiplo di sequenze** (MSA), la stessa proteina in molte
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

Il database pubblico che ne è seguito copre la quasi totalità di UniProt, cioè
quasi ogni sequenza proteica catalogata (restano fuori le catene troppo corte o
troppo lunghe, quelle con amminoacidi non standard e le proteine virali): nel
giro di due anni la predizione di struttura è passata da problema di ricerca a
servizio di consultazione.

`````

## Vantaggi e sfide

Il quadro va chiuso con la stessa onestà con cui la sezione sul confronto con
le reti ricorrenti aveva ammesso il costo quadratico. I vantaggi sono reali:
questi modelli reggono testi lunghi senza dimenticare l'inizio, si addestrano
spartendo il lavoro fra migliaia di processori, e una sola architettura basta
per il testo, le immagini e l'audio. Ma le sfide non sono dettagli:

- **Risorse**: addestrare un grande modello richiede centinaia di schede
  grafiche che lavorano insieme per mesi, con i consumi elettrici che ne
  seguono; anche solo *eseguirlo* può richiedere macchine fuori dalla portata di
  un laboratorio piccolo.
- **Dati**: i *corpora* (cioè le grandi raccolte di testi su cui i modelli
  studiano) da miliardi di parole contengono errori, stereotipi e contenuti
  tossici, e i modelli li assorbono. Se in quei testi le infermiere sono sempre
  donne e gli ingegneri sempre uomini, il modello impara quella regola come
  impara la grammatica: sono i **bias**, cioè le distorsioni sistematiche dei
  dati, che diventano distorsioni del modello.
- **Affidabilità**: un modello **autoregressivo** (che scrive una parola alla
  volta, ogni volta scegliendo la continuazione più probabile di quello che ha
  già scritto) produce la continuazione più plausibile, non necessariamente
  quella *vera*: le "allucinazioni" (risposte fluenti e sbagliate) sono un
  limite strutturale, non un incidente.

`````{tab} Elementare
```{admonition} Da ricordare
:class: important
- Tre modi di studiare, tre mestieri. **GPT** copre la pagina con la mano e
  indovina la parola dopo: diventa bravo a scrivere. **BERT** fa gli esercizi a
  buchi guardando prima e dopo: diventa bravo a capire. **T5** riscrive ogni
  compito come un tema, con il nome del compito davanti.
- La ricetta è sempre la stessa: prima si studia tantissimo per conto proprio,
  su montagne di testo e senza nessuno che corregga; poi si aggiusta il tiro sul
  compito che serve, con pochi esempi o solo con le istruzioni scritte davanti
  (il *prompt*).
- **ELECTRA** cambia l'esercizio invece dell'architettura: fa il correttore di
  bozze su **ogni** parola invece di indovinarne una su sette, e a parità di
  fatica impara molto di più.
- Le immagini entrano nello stesso meccanismo tagliandole in tessere e
  mettendole in fila come parole; da lì un modello può guardare una foto e
  rispondere a parole.
- I limiti vanno messi in conto quanto i pregi: costano moltissimo da
  addestrare, si portano dentro i pregiudizi dei testi su cui hanno studiato, e
  scrivono con la stessa sicurezza cose vere e cose inventate.
```
`````

`````{tab} Superiore
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
- **ViT** tratta l'immagine come una frase di tessere $16\times16$, e paga in
  **dati** la località che le CNN hanno gratis nell'architettura; i modelli
  **multimodali** (CLIP, GPT-4) allineano testo e immagini.
- Costi computazionali, bias nei dati e allucinazioni sono limiti
  strutturali, da mettere in conto quanto i vantaggi.
```
`````
