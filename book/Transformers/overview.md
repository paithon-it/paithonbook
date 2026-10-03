# Transformer: quando l'attenzione basta

```{image} ../figures/aperture/transformers.png
:class: pt-apertura only-light
:width: 100%
:alt: Una fila di tessere da cui partono archi che collegano ogni tessera a tutte le altre.
```

```{image} ../figures/aperture/transformers-scura.png
:class: pt-apertura only-dark
:width: 100%
:alt: Una fila di tessere da cui partono archi che collegano ogni tessera a tutte le altre.
```

Nel giugno del 2017 otto ricercatori, tutti passati per Google Brain e Google
Research, pubblicano un articolo dal titolo che suona come una battuta:
*Attention Is All You Need* {cite}`vaswani2017attention` («l'attenzione è tutto
ciò che serve», eco di *All You Need Is Love* dei Beatles). Dentro c'è
un'architettura di rete neurale nuova, il **Transformer**, che scommette su un
meccanismo solo: l'attenzione, usata fino ad allora come accessorio, basta da
sola. Restano fuori i due pezzi su cui si reggevano le reti per le sequenze: la
ricorrenza delle RNN, che leggono una parola alla volta portandosi dietro un
riassunto del già letto, e le convoluzioni del {doc}`capitolo sul deep learning
</DeepLearning/overview>`, filtri che guardano soltanto i vicini. La scommessa
ha retto: oggi il Transformer è l'architettura di quasi tutti i grandi modelli
linguistici, e la «T» di GPT, la sigla che si legge anche nel nome di ChatGPT,
sta proprio per Transformer.

## Il problema: leggere una frase tutta insieme

Il Transformer tiene i pezzi che il {doc}`capitolo sul Natural Language
Processing </NaturalLanguageProcessing/overview>` gli ha consegnato e cambia la
macchina che li porta. Le reti ricorrenti, anche nelle varianti con porte che
decidono che cosa conservare e che cosa cancellare dello stato (LSTM e GRU),
riassumono il passato in uno stato aggiornato un passo alla volta. Il metodo ha
due limiti che stanno nella sua struttura: l'informazione fra due parole
lontane attraversa tutti i passi intermedi, e il passo $t$ non comincia finché
il passo $t-1$ non è finito.

`````{tab} Elementare
Un romanzo letto attraverso una fessura che scopre una parola alla volta, con
tutto il resto da tenere a memoria: dopo dieci pagine, quanto ricordi della
prima? È il problema delle reti ricorrenti: sui testi lunghi il ricordo
dell'inizio sbiadisce. Il taccuino della LSTM, su cui annoti quello che conta,
aiuta parecchio, perché ci scrivi solo l'essenziale e cancelli il resto. Ma la
pagina è una sola, e a furia di aggiungere e cancellare, di quel che c'era
all'inizio resta sempre meno. E c'è un secondo problema: se puoi leggere solo
una parola alla volta, non puoi farti aiutare; cento amici non leggono un
libro più in fretta di te se il libro va comunque letto in fila.

Il Transformer rompe la fessura: guarda tutta la frase insieme, e per ogni
parola decide a quali altre parole prestare attenzione. In "Il gatto nero
salta sul muro", mentre elabora "salta" può guardare direttamente "gatto" (chi
è che salta?) senza passare per un riassunto sbiadito. E siccome ogni parola
viene elaborata insieme alle altre, il lavoro si può dividere: i cento amici
servono, eccome. È soprattutto questo ad aver fatto crescere i modelli fino
alle dimensioni di oggi: leggersi una biblioteca intera diventa una faccenda
di quanti amici riesci a chiamare.

Dividere il lavoro però non lo fa sparire, e il conto sta nelle coppie. Ogni
parola guarda ogni altra, quindi dieci parole fanno cento sguardi e cento
parole ne fanno diecimila: raddoppiare la frase quadruplica il lavoro. Su una
frase nessuno se ne accorge. Su un romanzo intero è la voce di spesa che
comanda tutte le altre, ed è una delle ragioni per cui a questi modelli si
mette un limite su quanto testo possono tenere davanti agli occhi in una
volta.
`````

`````{tab} Superiore
Nelle RNN l'informazione che va dalla prima all'ultima parola di una sequenza
lunga $n$ attraversa $O(n)$ passaggi di stato: il segnale si degrada (gradiente
che svanisce, la cui derivazione per le ricorrenti sta nella {doc}`sezione sui
modelli di sequenza </NaturalLanguageProcessing/modelli-sequenza>`) e le
dipendenze lunghe si perdono, problema che LSTM e GRU mitigano ma non
eliminano. Inoltre
la ricorrenza è intrinsecamente sequenziale: il passo $t$ richiede il passo
$t-1$, e l'hardware parallelo (le GPU) resta sottoutilizzato in addestramento.

Nel Transformer la self-attention collega ogni coppia di posizioni in un
solo passo, lunghezza di cammino $O(1)$, e l'elaborazione di tutte le
posizioni è un prodotto tra matrici, parallelizzabile per costruzione. È
questa seconda proprietà, più ancora della prima, ad aver cambiato la scala
dei modelli: addestrare su corpora enormi è diventato una questione di
hardware, non di architettura. Il prezzo è un costo quadratico nella lunghezza
della sequenza, che il {doc}`confronto coi modelli precedenti <confronti>`
mette sul tavolo.
`````

## Dal meccanismo ai modelli

I Transformer sono importanti, ma non sono magia. Dentro ci sono prodotti fra
matrici, una softmax, normalizzazioni e piccole reti applicate a ogni
posizione, montati in un ordine preciso, e su una frase di tre parole si
rifanno con carta e penna. Chi ha letto la {doc}`matematica di un modello
linguistico </Matematica/matematica-llm>` ne conosce già il meccanismo, con
altri nomi; qui arrivano il vocabolario standard (query, key e value, i tre
ruoli che ogni parola gioca nell'attenzione) e i pezzi che là restavano fuori:
le maschere, che decidono quali parole ciascuna può guardare, e l'architettura
che monta il tutto.

Il capitolo segue l'ordine dell'articolo del 2017. Si comincia dal meccanismo
di attenzione e dalla sua forma con le matrici, poi si monta l’architettura
completa, cioè come i blocchi di attenzione diventano una rete. Segue il
confronto con i modelli precedenti, quelli che leggevano in fila, compresi i
punti dove il Transformer è più debole; poi l'attenzione mentre il modello
genera una parola per volta, che cosa si conserva da un token al successivo e
quanto costa; e due esempi da eseguire.

Su quell'architettura sono cresciuti i modelli che si usano oggi: le famiglie
GPT, BERT e T5 e l'estensione alle immagini; una rete sola addestrata su cento
lingue, che si può rifinire in inglese e usare in italiano; i grandi modelli
linguistici, con le leggi di scala, la scelta della parola da scrivere e il
modo di valutarli; i modelli a esperti, che hanno moltissimi parametri ma per
ogni parola ne usano una piccola parte; il post-training, che trasforma un
completatore di frasi in un assistente che risponde; il retrieval, con cui un
modello consulta dei documenti prima di rispondere. Chiude uno sguardo alle
tendenze e ai limiti.

`````{tab} Elementare

```{admonition} Da ricordare
:class: important
- Il Transformer (Vaswani e colleghi, 2017, *Attention Is All You Need*)
  toglie di mezzo sia la lettura in fila sia i filtri che guardano solo i
  vicini: tutta l'architettura si regge sul meccanismo di attenzione.
- Attacca alla radice i due guai delle reti che leggono una parola alla volta.
  Le dipendenze lunghe: ogni parola guarda direttamente ogni altra, e il
  ricordo dell'inizio non sbiadisce più per strada. E la
  parallelizzazione: le parole si elaborano tutte insieme invece che in
  fila, e i cento amici di prima servono davvero, perché è quello che permette
  di addestrare questi modelli su macchine con migliaia di processori.
- Il conto si paga sulle coppie: ogni parola guarda ogni altra, quindi
  raddoppiare la lunghezza del testo quadruplica il lavoro. È una delle ragioni
  per cui a questi modelli si mette un limite su quanto testo possono tenere
  davanti agli occhi in una volta.
- Proprio perché regge dati e macchine sempre più grandi è diventato la base
  dei grandi modelli linguistici: quella «T» è la stessa di GPT, BERT e
  ChatGPT.
```

`````

`````{tab} Superiore

```{admonition} Da ricordare
:class: important
- Il Transformer (Vaswani et al., 2017, *Attention Is All You Need*)
  sostituisce ricorrenza e convoluzione con la self-attention. La Tabella 1
  dell'articolo mette in fila il guadagno: il cammino massimo fra due posizioni
  qualsiasi scende da $O(n)$ di uno strato ricorrente a $O(1)$, e secondo gli
  autori cammini più corti rendono più facili da imparare le dipendenze
  lunghe. È un'ipotesi e non una garanzia: sull'accordo fra soggetto e verbo
  a distanza le LSTM hanno fatto meglio {cite}`tran2018importance`.
- Cade con la ricorrenza anche il vincolo sequenziale: le operazioni da fare
  una dopo l'altra passano da $O(n)$ a $O(1)$, quindi l'intera sequenza si
  elabora in parallelo e l'addestramento sfrutta l'hardware a molti core.
- Il prezzo sta nell'altra colonna della stessa tabella: il costo per strato è
  $O(n^2 \cdot d)$, con $d$ la dimensione delle rappresentazioni, cioè
  quadratico nella lunghezza $n$ della sequenza; ed è quadratica in $n$ anche
  la memoria per i punteggi, $O(n^2)$, senza il fattore $d$. Quel conto torna
  nel {doc}`confronto coi modelli precedenti <confronti>` e nella sezione
  sull’{doc}`attenzione in pratica <attenzione-in-pratica>`.
- Su questa architettura poggiano i grandi modelli linguistici (GPT, BERT,
  T5): a imporla è stata la capacità di scalare con dati e parametri.
```

`````

