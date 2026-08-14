# Esempi pratici

Dopo tanta architettura, mettiamo i Transformer al lavoro su due compiti
concreti: tradurre una frase e capire se una recensione è entusiasta o delusa.
Sono gli stessi esempi che un lettore incontra ogni giorno senza pensarci (il
tasto "traduci" sotto un post, il termometro delle recensioni di un prodotto)
e per fortuna non serve addestrare nulla da zero: la libreria `transformers`
di Hugging Face (una *libreria* è una cassetta degli attrezzi già pronta, che
un programma può aprire e usare), costruita su PyTorch, mette a disposizione
migliaia di modelli già addestrati.

Una parola su come leggere questa pagina. Il programma che fa girare i due
esempi sta nelle schede Superiore, ma non serve saperlo leggere per seguire
quel che succede: qui sotto, nel testo comune, c'è scritto in italiano che cosa
entra, che cosa esce, e soprattutto che cosa il modello **sbaglia**, che è la
parte più istruttiva delle due.

## Traduzione automatica

Il compito per cui il Transformer è nato: l'encoder legge la frase di
partenza, il decoder scrive quella d'arrivo, e mentre scrive va a rileggersi
l'originale. Quel «rileggersi l'originale» ha un nome che tornerà spesso, la
**cross-attention**: è la stessa attenzione di sempre, con l'unica differenza
che le domande vengono dalla torre che scrive e le risposte dalla torre che ha
letto.

`````{tab} Elementare
Segui il viaggio di "The cat sits on the mat". Prima la frase viene spezzata
in mattoncini (le parole o pezzi di parola: i *token*) e l'encoder la legge
tutta, costruendo per ogni parola quella rappresentazione ricca di contesto
che conosciamo. Poi il decoder comincia a scrivere in italiano, una parola
alla volta: quando deve produrre "gatto" il suo evidenziatore punta su "cat",
quando produce "siede" punta su "sits". Non è un dizionario che sostituisce
parola per parola: è più simile a un traduttore che legge tutta la frase, la
capisce, e la riscrive. La differenza si vede con una parola ambigua: "bank"
in inglese è sia la banca sia la riva del fiume, e su "The cat sits on the
river bank" il modello scrive "sulla riva del fiume", perché la parola
"river" era lì accanto e l'attenzione l'ha vista. Se il contesto non c'è, il
modello sceglie il significato più comune e può sbagliare: non indovina, usa
quello che gli hai dato.
`````

`````{tab} Superiore
In codice, usando un modello encoder–decoder pre-addestrato della famiglia
OPUS-MT (Università di Helsinki) via Hugging Face:

```{code-block} python
:class: pt-lento

from transformers import AutoTokenizer, AutoModelForSeq2SeqLM

# modello encoder-decoder pre-addestrato inglese -> italiano (su PyTorch)
nome = "Helsinki-NLP/opus-mt-en-it"
tokenizzatore = AutoTokenizer.from_pretrained(nome)
modello = AutoModelForSeq2SeqLM.from_pretrained(nome)

for frase in ["The cat sits on the mat.",
              "The cat sits on the river bank."]:
    ingresso = tokenizzatore(frase, return_tensors="pt")  # testo -> token
    uscita = modello.generate(**ingresso, max_new_tokens=40)  # autoregressiva
    print(tokenizzatore.decode(uscita[0], skip_special_tokens=True))
# -> Il gatto si siede sul tappetino.
# -> Il gatto si siede sulla riva del fiume.
#    (uscite reali al momento della stesura: i pesi remoti possono cambiare)
```

La seconda frase è la disambiguazione lessicale in atto: nessun dizionario
traduce «bank» con «riva», e il modello ci arriva perché la rappresentazione di
quel token è stata costruita pesando anche «river».

Le tre righe di lavoro sono i tre passaggi visti nei capitoli precedenti, qui
scritti in chiaro: **tokenizzazione** (la frase diventa una sequenza di id di
token), **inferenza** con `generate` (encoder e decoder Transformer, con
generazione autoregressiva e maschera causale) e **decodifica** (dagli id di
token al testo). La libreria offre anche una scorciatoia, `pipeline`, che li
incapsula in una riga; qui li teniamo separati perché sono esattamente i pezzi
che il capitolo ha spiegato, e perché i nomi delle scorciatoie cambiano tra
versioni, mentre `AutoTokenizer` e `generate` no. Sotto il cofano il modello è
un `nn.Module` PyTorch come quelli del capitolo su PyTorch: con
`modello.named_parameters()` si ispezionano strati, teste di attenzione e
parametri.
`````

## Analisi del sentiment

Qui il Transformer non deve generare nulla: deve *capire* e dare un voto. Il
modello che useremo lo dà come lo darebbe un cliente su un sito di recensioni,
da una a cinque stelle, e leggeremo quel voto come il termometro
dell'entusiasmo. È un compito perfetto per un modello fatto della sola
torre che legge, senza quella che scrive: in gergo si dice **solo-encoder**, e
il capostipite si chiama BERT (lo presentiamo per bene nella sezione
successiva).

`````{tab} Elementare
"Mi è piaciuto moltissimo questo prodotto!" e "Una delusione totale": per te è
ovvio, e il bello è che ormai lo è anche per la macchina, che legge la frase
intera con l'attenzione invece di contare quante parole positive e negative ci
sono dentro. Aziende e ricercatori lo usano per misurare l'umore di migliaia di
recensioni o commenti in pochi secondi: un lavoro che a mano richiederebbe
settimane.

Le frasi facili però le indovinano tutti, ed è sulle altre che si capisce quanto
un modello abbia davvero capito. Il caso classico in italiano è il complimento
detto negando il contrario, "non è affatto male": nessuna delle tre parole è un
elogio, eppure la frase lo è. Il paragrafo che chiude questa sezione racconta
come se la cava il modello che stiamo usando, e la risposta è: male.
`````

`````{tab} Superiore
```{code-block} python
:class: pt-lento

from transformers import pipeline

# modello multilingue (italiano compreso) che assegna da 1 a 5 stelle.
# top_k=None restituisce TUTTE le classi, non solo la vincente: senza
# questo si vedrebbe solo l'argmax, e l'argmax qui nasconde il fatto.
giudice = pipeline("sentiment-analysis",
                   model="nlptown/bert-base-multilingual-uncased-sentiment",
                   top_k=None)

recensioni = [
    "Mi è piaciuto moltissimo questo prodotto!",
    "Questo prodotto è stato una delusione totale.",
    "Non è affatto male.",
    "Non è male.",
]
for r in recensioni:
    esiti = giudice(r)[0]                       # lista, ordinata per punteggio
    vincente = esiti[0]
    coda = "  ".join(f"{e['label']} {e['score']:.2f}" for e in esiti[:3])
    print(f"{r!r}\n   -> {vincente['label']}   [{coda}]")
```

L'uscita, eseguendo davvero:

```text
'Mi è piaciuto moltissimo questo prodotto!'
   -> 5 stars   [5 stars 0.64  4 stars 0.31  3 stars 0.04]
'Questo prodotto è stato una delusione totale.'
   -> 1 star   [1 star 0.84  2 stars 0.15  3 stars 0.01]
'Non è affatto male.'
   -> 2 stars   [2 stars 0.36  3 stars 0.34  1 star 0.23]
'Non è male.'
   -> 3 stars   [3 stars 0.47  4 stars 0.32  5 stars 0.13]
```

I numeri sono quelli usciti al momento della stesura: come per la traduzione
qui sopra, i pesi stanno su un server di altri e possono cambiare.

Il modello è un BERT multilingue rifinito (*fine-tuned*) su recensioni: la
classificazione usa la rappresentazione del token speciale `[CLS]` passata a
una testa lineare (architettura solo-encoder, senza generazione). Le prime due
righe sono quelle che ci si aspetta; la terza è quella che il paragrafo dopo le
schede analizza, ed è il motivo per cui il codice stampa la graduatoria e non
solo la vincente: su «Non è affatto male» il modello dà $0{,}365$
a due stelle e $0{,}336$ a tre. È in bilico fra le due, e cade dalla parte
sbagliata per meno di tre centesimi; è anche l'unica delle quattro righe in cui
la prima e la seconda classe sono così vicine, e questo lo si vede solo
guardando la coda. È il caso in cui riportare solo
l'`argmax` nasconde tutto quello che c'è da sapere, ed è per questo che
`top_k=None` non è un dettaglio di comodo: senza, la pagina non potrebbe
dimostrare quello che sta per dire.

Si noti quindi la **confidenza**: un classificatore serio si valuta con le
metriche del capitolo sul machine learning (accuratezza, precision/recall), e su
domini diversi da quello di addestramento (ironia, sarcasmo, gergo) le
prestazioni calano sensibilmente.
`````

Quell'errore sulla terza frase merita di stare nel testo per tutti, perché è la
cosa più utile che questa pagina abbia da dare. Il modello dà a «non è affatto
male» **due stelle su cinque**, cioè lo legge come una recensione scontenta,
mentre a «non è male», la stessa frase senza l'avverbio, ne dà tre. Non è un
capriccio, ed è probabile che venga dalla compagnia che «affatto» tiene nei
testi: compare quasi sempre dentro una stroncatura piena («non mi è piaciuto
affatto»), e quella compagnia se la porta dietro. (È una spiegazione
plausibile, non una verifica: i testi su cui questo modello ha studiato non si
possono ispezionare.) Dire una cosa negando il suo contrario (i
retori la chiamano *litote*) chiede di comporre il significato di tre parole in
un verso che nessuna delle tre porta da sola: l'attenzione mette «non»,
«affatto» e «male» in contatto, ma il contatto non garantisce che dalla
composizione esca la cosa giusta. Le due frasi facili, da sole, avrebbero fatto
una bella dimostrazione e insegnato molto meno; una demo di tre righe non è una
validazione, e questa pagina l'ha appena dimostrato su sé stessa.

`````{tab} Elementare
```{admonition} Da ricordare
:class: important
- La **traduzione** usa il Transformer intero: la torre che legge, la torre che
  scrive, e il continuo rileggersi l'originale mentre si traduce.
- Per capire una recensione basta la torre che legge, con in cima un giudice
  che dà il voto (qui, da una a cinque stelle). È la famiglia di modelli che
  qui chiamiamo *solo-encoder*, e il capostipite si chiama **BERT**.
- Non serve costruire niente da zero: esistono cassette degli attrezzi (la
  libreria `transformers`) piene di modelli già addestrati da altri, che si
  usano in poche righe.
- I risultati vanno sempre provati **sui propri testi**: ironia, modi di dire e
  complimenti detti al contrario restano difficili, come mostra il "non è
  affatto male" di questa pagina.
```
`````

`````{tab} Superiore
```{admonition} Da ricordare
:class: important
- La **traduzione** usa il Transformer completo: encoder che legge, decoder
  che genera, cross-attention che li allinea.
- L'**analisi del sentiment** usa un solo-encoder (stile BERT) con una testa
  di classificazione: capire, non generare.
- La libreria `transformers` di Hugging Face (su PyTorch) dà accesso a
  modelli pre-addestrati per entrambi i compiti in poche righe: sotto, sono
  `nn.Module` come quelli del capitolo su PyTorch.
- I risultati vanno **validati sul proprio dominio**: ironia, gergo e litoti
  restano difficili, e la demo di questa pagina ne fornisce il controesempio in
  casa.
```
`````
