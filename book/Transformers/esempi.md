# Esempi pratici

Dopo tanta architettura, mettiamo i Transformer al lavoro su due compiti
concreti: tradurre una frase e capire se una recensione è entusiasta o delusa.
Sono gli stessi esempi che un lettore incontra ogni giorno senza pensarci (il
tasto "traduci" sotto un post, il termometro delle recensioni di un prodotto)
e per fortuna non serve addestrare nulla da zero. Qualcun altro ha già fatto la
parte cara del lavoro: ha preso una di queste macchine ancora vuota, le ha dato
in pasto montagne di testo e ha lasciato che si aggiustasse i numeri da sola
per settimane, spendendo in elettricità quanto una casa in un anno. Il
risultato di quella fatica si scarica e si usa in tre righe, ed è quello che
chiamiamo un modello **pre-addestrato**. A tenerne il catalogo è Hugging Face,
un'azienda che ospita un sito da cui chiunque può scaricarli; il programma con
cui li si adopera si chiama `transformers` {cite}`wolf2020transformers` ed è
una *libreria*, cioè una
cassetta degli attrezzi già pronta che un programma può aprire e usare.
Sotto c'è PyTorch, lo strumento con cui in questo libro si costruiscono le reti.

Dei due esempi conta soprattutto il secondo, e non per quello che indovina:
per quello che **sbaglia**. Il prezzo dell'architettura si è visto in astratto,
contando le coppie; qui si tocca un errore concreto, su una frase italiana di
quattro parole.

## Traduzione automatica

Il compito per cui il Transformer è nato, e quello con cui la sezione
sull'architettura ce l'ha presentato: la torre che legge (l’**encoder**) si
prende la frase di partenza, la torre che scrive (il **decoder**) compone
quella d'arrivo, e mentre la compone torna continuamente a guardare
l'originale.

Quel «tornare a guardare» ha un nome che tornerà spesso, la
**cross-attention**, ed è l'attenzione di sempre applicata fra le due torri
invece che dentro una: le domande (le *query*, cioè «che cosa mi serve
adesso?») le pone la torre che scrive, e le etichette e le informazioni con cui
si risponde (le *key* e i *value*) vengono da quella che ha letto.

`````{tab} Elementare
Segui il viaggio di "The cat sits on the mat". Prima la frase viene spezzata
in mattoncini (le parole o pezzi di parola: i *token*) e l'encoder la legge
tutta, riscrivendo la lista di numeri di ogni parola in modo che si porti dentro
anche il contesto in cui si trova. Poi il decoder comincia a scrivere in
italiano, una parola alla volta: quando deve produrre "gatto" il suo
evidenziatore (l'attenzione della sezione di apertura) punta su "cat",
quando produce "siede" punta su "sits". Somiglia più a un traduttore che legge
tutta la frase, la capisce e la riscrive, che a un dizionario che sostituisce
parola per parola. La differenza si vede con una parola ambigua: "bank"
in inglese è sia la banca sia la riva del fiume, e su "The cat sits on the
river bank" il modello scrive "sulla riva del fiume", perché la parola
"river" era lì accanto e l'attenzione l'ha vista. Se il contesto non c'è, il
modello sceglie il significato più comune e può sbagliare: non indovina, usa
quello che gli hai dato.
`````

`````{tab} Superiore
In codice, usando un modello encoder-decoder pre-addestrato della famiglia
OPUS-MT (Università di Helsinki) via Hugging Face:

```{code-block} python
:class: pt-lento

# pt-lento non per il tempo, ma per i 343 MB di pesi da scaricare la prima
# volta: dopo, il modello resta nella cache locale.
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
```

```text
Il gatto si siede sul tappetino.
Il gatto si siede sulla riva del fiume.
```

La seconda frase è la disambiguazione lessicale in atto: un dizionario elenca
tutti e due i significati di «bank» e lascia la scelta a chi legge, mentre qui
la compie il modello, perché la rappresentazione di quel token è stata
costruita pesando anche «river».

Le tre righe di lavoro sono i tre passaggi visti nei capitoli precedenti, qui
scritti in chiaro: **tokenizzazione** (la frase diventa una sequenza di id di
token), **inferenza** con `generate` (encoder e decoder Transformer, con
generazione autoregressiva e maschera causale) e **decodifica** (dagli id di
token al testo). La libreria offre anche una scorciatoia, `pipeline`, che li
incapsula in una riga; qui li teniamo separati perché sono esattamente i pezzi
che il capitolo ha spiegato, e perché i nomi delle scorciatoie cambiano tra
versioni, mentre `AutoTokenizer` e `generate` no. Sotto il cofano il modello è
un `nn.Module` PyTorch come quelli della {doc}`sezione sui moduli
</PyTorch/moduli>`: con `modello.named_parameters()` si ispezionano strati,
teste di attenzione e
parametri.
`````

## Il termometro delle recensioni

Il secondo compito è capire l'umore di chi scrive: se una recensione è
entusiasta o delusa. Si chiama *sentiment analysis*, «analisi del sentimento»,
ed è quello che sta dietro alla percentuale di soddisfatti che compare sotto un
prodotto in vendita.

Qui il Transformer non deve generare nulla: deve *capire* e dare un voto. Il
modello che useremo lo dà come lo darebbe un cliente su un sito di recensioni,
da una a cinque stelle. Ed è un compito perfetto per un modello fatto della
sola torre che legge, senza quella che scrive: se la risposta è un voto e non
una frase, la torre che scrive non serve, e tenerla costerebbe soltanto. In
gergo un modello così si dice **encoder-only**, "solo encoder", e il
capostipite si chiama BERT.

`````{tab} Elementare
"Mi è piaciuto moltissimo questo prodotto!" e "Una delusione totale": per te è
ovvio, e il bello è che ormai lo è anche per la macchina, che legge la frase
intera con l'attenzione invece di contare quante parole positive e negative ci
sono dentro. Aziende e ricercatori lo usano per misurare l'umore di migliaia di
recensioni o commenti in pochi secondi: un lavoro che a mano richiederebbe
settimane.

Le stelle possibili sono cinque, e la macchina non ne indica una soltanto: ha
cento gettoni di fiducia, li sparpaglia sulle cinque caselle e poi annuncia la
casella dove ne ha messi di più. Ottanta gettoni su una casella e venti sparsi
altrove, oppure due caselle in testa a pochi gettoni una dall'altra: l'annuncio
esce identico, un nome di casella e nient'altro, mentre le due situazioni non
si somigliano. Chiedere la fila completa, casella per casella, distingue la
macchina sicura da quella in bilico.

Le frasi facili però le indovinano tutti, ed è sulle altre che si capisce quanto
un modello abbia davvero capito. Il caso classico in italiano è il complimento
detto negando il contrario, "non è affatto male": nessuna delle tre parole è un
elogio, eppure la frase lo è. Lì il modello sbaglia, e sbaglia per un soffio:
la casella accanto, quella più benevola, resta indietro di pochi gettoni.
`````

`````{tab} Superiore
```{code-block} python
:class: pt-lento

# come sopra, e qui il modello da scaricare e' di 669 MB.
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
    coda = "  ".join(f"{e['label']} {e['score']:.3f}" for e in esiti)
    print(f"{r!r}\n   -> {esiti[0]['label']}\n      {coda}")
```

```text
'Mi è piaciuto moltissimo questo prodotto!'
   -> 5 stars
      5 stars 0.639  4 stars 0.314  3 stars 0.042  1 star 0.003  2 stars 0.003
'Questo prodotto è stato una delusione totale.'
   -> 1 star
      1 star 0.840  2 stars 0.148  3 stars 0.011  4 stars 0.001  5 stars 0.000
'Non è affatto male.'
   -> 2 stars
      2 stars 0.365  3 stars 0.336  1 star 0.227  4 stars 0.057  5 stars 0.015
'Non è male.'
   -> 3 stars
      3 stars 0.471  4 stars 0.318  5 stars 0.125  2 stars 0.062  1 star 0.023
```

I pesi di questo modello stanno sul server di chi lo pubblica: se un giorno lo
riaddestrano, le cifre esatte possono cambiare, mentre la graduatoria e il
fenomeno che segue restano.

Il modello è un BERT multilingue rifinito (*fine-tuned*) su recensioni: la
classificazione usa la rappresentazione del token speciale `[CLS]` passata a
una testa lineare (architettura encoder-only, senza generazione). Le prime due
righe sono quelle che ci si aspetta; la terza no, ed è il motivo per cui il
codice stampa la graduatoria e non solo la vincente. I valori esatti sono
$0{,}365$ a due stelle e $0{,}336$ a tre: uno scarto di ventinove millesimi,
l'unico delle quattro righe in cui le prime due classi si toccano così. Il solo
`argmax` direbbe «2 stars» e si fermerebbe lì, indistinguibile dai verdetti
delle altre tre righe, dove la seconda classe resta indietro di centocinquanta
millesimi o più: `top_k=None` tiene visibile la differenza fra un verdetto
comodo e uno in bilico.

Il punteggio della classe vincente è la **confidenza** del modello, e da solo
non dice niente sulla qualità del classificatore: quella si valuta con le
metriche della {doc}`sezione su come si valuta un modello
</MachineLearning/metriche>` (accuratezza, precision/recall), e su
domini diversi da quello di addestramento (ironia, sarcasmo, gergo) le
prestazioni calano sensibilmente.
`````

Quell'errore sulla terza frase merita di stare nel testo per tutti, perché è la
cosa più utile che questa pagina abbia da dare. Il modello dà a «non è affatto
male» **due stelle su cinque**, cioè lo legge come una recensione scontenta,
mentre a «non è male», la stessa frase senza l'avverbio, ne dà tre.

Il bello è che il verdetto non è netto come sembra. Il modello non sceglie una
risposta sola: dà un voto a tutte e cinque, e qui aveva messo le due stelle a
0,365 e le tre a 0,336, a ventinove millesimi l'una dall'altra. È l'unica delle
quattro frasi in cui i primi due posti stanno così attaccati, e chi guardasse
soltanto il vincitore non se ne accorgerebbe mai. Il quasi-pareggio, però, è fra
due modi di sbagliare e non fra sbagliare e indovinare: le quattro e le cinque
stelle, che sarebbero la lettura giusta di un complimento, si dividono in tutto
sette centesimi.

L'errore viene probabilmente dalla compagnia che
«affatto» tiene nei testi: compare quasi sempre dentro una stroncatura piena
(«non mi è piaciuto affatto»), e quella compagnia se la porta dietro. (È una
spiegazione plausibile, non una verifica: i testi su cui questo modello ha
studiato non si possono ispezionare, perché chi lo ha addestrato non li ha
pubblicati.) Dire una cosa negando il suo contrario (i retori la chiamano
*litote*) chiede di comporre il significato di tre parole in una direzione che
nessuna delle tre porta da sola: l'attenzione mette «non», «affatto» e «male»
in contatto, ma il contatto non garantisce che dalla composizione esca la cosa
giusta. Le due frasi facili, da sole, avrebbero fatto una bella dimostrazione e
insegnato molto meno: quattro frasi provate al volo non sono un collaudo, e
questa pagina l'ha appena dimostrato su sé stessa.

`````{tab} Elementare
```{admonition} Da ricordare
:class: important
- La **traduzione** usa il Transformer intero: la torre che legge, la torre che
  scrive, e il continuo rileggersi l'originale mentre si traduce.
- Per capire una recensione basta la torre che legge, con in cima un giudice
  che dà il voto (qui, da una a cinque stelle). In gergo è la famiglia
  *encoder-only*, e il capostipite si chiama **BERT**.
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
- L’**analisi del sentiment** usa un *encoder-only* (stile BERT) con una testa
  di classificazione: capire, non generare.
- La libreria `transformers` di Hugging Face (su PyTorch) dà accesso a
  modelli pre-addestrati per entrambi i compiti in poche righe: sotto, sono
  `nn.Module` come quelli della sezione sui moduli di PyTorch.
- I risultati vanno **validati sul proprio dominio**: ironia, gergo e litoti
  restano difficili, e la demo di questa pagina ne fornisce il controesempio in
  casa.
```
`````
