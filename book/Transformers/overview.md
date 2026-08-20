# Transformer: quando l'attenzione basta

Nel giugno del 2017 otto ricercatori, tutti passati per Google Brain e Google
Research, pubblicano un *paper* (un articolo scientifico: il modo in cui chi fa
ricerca racconta agli altri quello che ha trovato) dal titolo che suona come una
battuta: *Attention Is All You Need* {cite}`vaswani2017attention`
(«l'attenzione è tutto ciò che serve», eco di *All You Need Is Love* dei
Beatles). Dentro c'è un'architettura di rete neurale nuova, il **Transformer**,
che fa una scommessa radicale: per capire il linguaggio non servono né la
ricorrenza delle RNN (leggere una parola alla volta, portandosi dietro un
riassunto di quel che è venuto prima) né le convoluzioni (i filtri che scorrono
su un testo o su un'immagine guardando solo i vicini, del {doc}`capitolo sul deep
learning </DeepLearning/overview>`); basta il meccanismo di **attenzione**, usato fino ad allora come
accessorio. La scommessa è vinta oltre ogni previsione: oggi il Transformer è la
base di quasi tutti i grandi modelli linguistici, e quella «T» è la stessa che
trovi nel nome di GPT e di ChatGPT.

## Il problema: leggere una frase tutta insieme

I modelli che abbiamo incontrato nel {doc}`capitolo sul Natural Language Processing </NaturalLanguageProcessing/overview>`
leggono il testo una parola alla volta, portandosi dietro un riassunto di quel
che è venuto prima: sono le **reti ricorrenti** (in sigla RNN) e la loro
versione più raffinata, quella con un taccuino su cui annotare e cancellare
(le **LSTM**). Funziona, ma con due difetti strutturali.

`````{tab} Elementare
Immagina di leggere un romanzo attraverso una fessura che mostra una parola
alla volta, dovendo tenere tutto il resto a memoria. Dopo dieci pagine, quanto
ricordi della prima? È il problema delle reti ricorrenti: sui testi lunghi il
ricordo dell'inizio sbiadisce. E c'è un secondo problema: se puoi leggere solo
una parola alla volta, non puoi farti aiutare; cento amici non leggono un
libro più in fretta di te se il libro va comunque letto in fila.

Il Transformer rompe la fessura: guarda **tutta la frase insieme**, e per ogni
parola decide a quali altre parole prestare attenzione. In "Il gatto nero
salta sul muro", mentre elabora "salta" può guardare direttamente "gatto" (chi
è che salta?) senza passare per un riassunto sbiadito. E siccome ogni parola
viene elaborata insieme alle altre, il lavoro si può dividere: i cento amici
servono, eccome.
`````

`````{tab} Superiore
Nelle RNN l'informazione tra due parole distanti $n$ posizioni attraversa
$O(n)$ passaggi di stato: il segnale si degrada (gradiente che svanisce, come
visto nel capitolo sulle reti neurali) e le dipendenze lunghe si perdono,
problema che LSTM e GRU mitigano ma non eliminano. Inoltre la ricorrenza è
intrinsecamente **sequenziale**: il passo $t$ richiede il passo $t-1$, e
l'hardware parallelo (le GPU) resta sottoutilizzato in addestramento.

Nel Transformer la **self-attention** collega ogni coppia di posizioni in un
solo passo, lunghezza di cammino $O(1)$, e l'elaborazione di tutte le
posizioni è un prodotto tra matrici, parallelizzabile per costruzione. È
questa seconda proprietà, più ancora della prima, ad aver cambiato la scala
dei modelli: addestrare su corpora enormi è diventato una questione di
hardware, non di architettura. Il prezzo è un costo quadratico nella lunghezza
della sequenza, di cui parleremo nella sezione sui confronti.
`````

## Dal meccanismo ai modelli

Una nota di metodo, prima dell'elenco. I Transformer sono importanti, ma non
sono magia: sotto il cofano ci sono tabelle di numeri e operazioni che si fanno
con carta e penna, montate in un ordine particolarmente felice. Chi ha letto i
capitoli sulla matematica e sulle reti neurali ritroverà i pezzi con i loro
nomi (matrici, prodotti scalari, softmax); chi non li ha letti può seguire lo
stesso il filo del livello Elementare, dove ogni pezzo ha un'immagine al posto
della formula.

Il capitolo segue la scia dell'articolo del 2017. Si comincia dal **meccanismo
di attenzione**: cos'è, come si calcola, perché funziona. Poi si monta
l’**architettura** completa, cioè come i blocchi di attenzione diventano una
rete vera. Segue un confronto onesto con i **modelli precedenti**, quelli che
leggevano in fila, inclusi i punti dove il Transformer è più debole, e poi due
**esempi pratici** che si possono eseguire.

Da lì in poi si guarda che cosa è cresciuto su quell'architettura. Le
**famiglie di modelli** (GPT, BERT, T5) e l'estensione alle immagini. Che cosa
succede quando lo stesso modello impara **cento lingue insieme**, compreso il
fatto tutt'altro che ovvio che si possa rifinirlo in inglese e usarlo in
italiano. I **grandi modelli linguistici**: quanto conviene ingrandirli, e come
si sceglie davvero la parola da scrivere. I **modelli a esperti**, che sanno
moltissimo ma per ogni parola accendono solo un pezzetto di sé, e costano quindi
molto meno di quanto siano grandi. Il **post-training**, cioè come un
completatore di frasi diventa un assistente che risponde. E il **retrieval**,
cioè come si insegna a un modello a cercare prima di rispondere. Chiude uno
sguardo alle **tendenze**, con i limiti, che non mancano.

`````{tab} Elementare

```{admonition} Da ricordare
:class: important
- Il **Transformer** (Vaswani e colleghi, 2017, *Attention Is All You Need*)
  toglie di mezzo sia la lettura in fila sia i filtri che guardano solo i
  vicini: tutta l'architettura si regge sul meccanismo di **attenzione**.
- Attacca alla radice i due guai delle reti che leggono una parola alla volta.
  Le **dipendenze lunghe**: ogni parola guarda direttamente ogni altra, e il
  ricordo dell'inizio non sbiadisce più per strada. E la
  **parallelizzazione**: le parole si elaborano tutte insieme invece che in
  fila, e i cento amici di prima servono davvero, perché è quello che permette
  di addestrare questi modelli su macchine con migliaia di processori.
- Proprio perché regge dati e macchine sempre più grandi è diventato la base
  dei grandi modelli linguistici: quella «T» è la stessa di GPT, BERT e
  ChatGPT.
```

`````

`````{tab} Superiore

```{admonition} Da ricordare
:class: important
- Il **Transformer** (Vaswani et al., 2017, *Attention Is All You Need*)
  sostituisce ricorrenza e convoluzione con la **self-attention**. La Tabella 1
  dell'articolo mette in fila il guadagno: il cammino massimo fra due posizioni
  qualsiasi scende da $O(n)$ di uno strato ricorrente a $O(1)$, ed è questa la
  ragione per cui le dipendenze lunghe si imparano meglio.
- Cade con la ricorrenza anche il vincolo sequenziale: le operazioni da fare
  una dopo l'altra passano da $O(n)$ a $O(1)$, quindi l'intera sequenza si
  elabora in parallelo e l'addestramento sfrutta l'hardware a molti core.
- Il prezzo sta nell'altra colonna della stessa tabella: il costo per strato è
  $O(n^2 \cdot d)$, cioè **quadratico** nella lunghezza $n$ della sequenza; ed
  è quadratica in $n$ anche la memoria per i punteggi, $O(n^2)$, senza il
  fattore $d$. È il conto che il capitolo nomina e non salda.
- Su questa architettura poggiano i grandi modelli linguistici (GPT, BERT,
  T5): a imporla è stata la capacità di scalare con dati e parametri.
```

`````

