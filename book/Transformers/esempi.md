# Esempi pratici

I Transformer lavorano qui su due compiti concreti: tradurre una frase e capire
se una recensione è entusiasta o delusa. Sono gli stessi esempi che si
incontrano ogni giorno senza pensarci (il tasto "traduci" sotto un post, il
termometro delle recensioni di un prodotto). Inoltre non serve addestrare
niente da zero, perché qualcun altro ha già fatto la parte costosa: ha preso
una di queste reti con i parametri ancora a caso, l'ha addestrata su montagne
di testo per giorni interi, su una fila di processori in parallelo, e ne ha
pubblicato i pesi, che si scaricano e si usano in poche righe. Una rete così,
con i parametri già ottimizzati su un grande corpus, si chiama modello
**pre-addestrato**. Hugging Face ospita il catalogo di questi modelli sul suo
sito; la libreria che li carica e li esegue si chiama `transformers`
{cite}`wolf2020transformers`, basata su {doc}`PyTorch </PyTorch/overview>`, lo
strumento con cui si costruiscono le reti.

## Traduzione automatica

È il compito per cui il Transformer è nato, e quello con cui la {doc}`sezione
sulla struttura del Transformer <architettura>` l'ha presentato. L'encoder
legge la frase di partenza, il decoder compone quella d'arrivo, e a ogni parola
prodotta torna a guardare l'originale con la cross-attention della
{doc}`sezione sull'attenzione <attenzione>`. In particolare, il decoder formula
le query («che cosa mi serve adesso?»), mentre dall'encoder vengono le chiavi e
i valori con cui si risponde.

`````{tab} Elementare
Segui il viaggio di "The cat sits on the mat". Prima la frase viene spezzata in
mattoncini (le parole o pezzi di parola: i *token*) e l'encoder la legge tutta,
riscrivendo la lista di numeri di ogni parola in modo che si porti dentro anche
il contesto in cui si trova. Poi il decoder comincia a scrivere in italiano,
una parola alla volta: quando deve produrre "gatto" il suo evidenziatore, cioè
l'attenzione, punta su "cat", e quando produce "siede" punta su "sits".
[Somiglia più a un traduttore che legge tutta la frase, la capisce e la
riscrive]{.pt-matita}, che a un dizionario che sostituisce parola per parola.

La differenza si vede con una parola ambigua: "bank"
in inglese è sia la banca sia la riva del fiume, e su "The cat sits on the
river bank" il modello scrive "sulla riva del fiume", perché la parola
"river" era lì accanto e l'attenzione l'ha vista. Se il contesto non c'è, il
modello sceglie il significato più comune e può sbagliare.
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
              "The cat sits on the bank.",
              "The cat sits on the river bank."]:
    ingresso = tokenizzatore(frase, return_tensors="pt")  # testo -> token
    uscita = modello.generate(**ingresso, max_new_tokens=40)  # autoregressiva
    print(tokenizzatore.decode(uscita[0], skip_special_tokens=True))
```

```text
Il gatto si siede sul tappetino.
Il gatto si siede sulla banca.
Il gatto si siede sulla riva del fiume.
```

La terza frase è la disambiguazione lessicale in atto. Un dizionario elenca
tutti e due i significati di «bank» e lascia la scelta a chi legge, mentre [qui
è il modello a decidere in funzione del contesto]{.pt-matita}: senza «river»,
infatti, la stessa frase diventa «Il gatto si siede sulla banca», il
significato più comune. È il risultato atteso di una rappresentazione di «bank»
costruita dall'attenzione anche sugli altri token della frase.

Il codice esegue tre passaggi: tokenizzazione (la frase diventa una sequenza di
id di token), inferenza con `generate` (encoder e decoder Transformer, con
generazione autoregressiva e maschera causale) e decodifica (dagli id di token
al testo). La libreria offre anche una scorciatoia, `pipeline`, che li
incapsula in una sola riga, come nell'esempio successivo sulle recensioni; qui
restano separati. Dietro le quinte il modello è un `nn.Module` PyTorch come
quelli della {doc}`sezione sui moduli </PyTorch/moduli>`: con
`modello.named_parameters()` si ispezionano strati, teste di attenzione e
parametri.
`````

## Analisi del sentimento: il termometro delle recensioni

Il secondo compito è capire l'umore di chi scrive. Si chiama *sentiment
analysis*, «analisi del sentimento», e sta dietro alla percentuale di
soddisfatti che compare sotto un prodotto in vendita.

Qui il Transformer non genera nulla: *capisce* e assegna un voto, da una a
cinque stelle, come un cliente su un sito di recensioni. Basta quindi il solo
encoder, la parte che legge: se la risposta è un voto e non una frase, il
decoder non serve e aggiungerebbe solo calcolo. In gergo un modello simile si
chiama **encoder-only**, «solo encoder», e il capostipite è BERT.

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
si somigliano.

[Per questo, chiedere la fila completa, casella per casella, distingue la
macchina sicura da quella in bilico]{.pt-matita}. Nemmeno ottanta gettoni su
una casella vogliono dire avere ragione ottanta volte su cento: se la sicurezza
della macchina sia giustificata lo si scopre soltanto provandola su frasi di
cui si sa già la risposta.

Tutti indovinano le frasi facili; quanto un modello abbia capito si vede
sulle altre. Il caso classico in italiano è il
complimento detto negando il contrario, "non è affatto male": nessuna delle tre
parole è un elogio, eppure la frase lo è. Lì il modello sbaglia: mette poco più
di un terzo dei gettoni su due stelle, una recensione scontenta, e appena meno
su tre. È indeciso, ma fra due risposte entrambe sbagliate: le quattro e le
cinque stelle, che di un complimento sarebbero la lettura giusta, insieme ne
prendono sette su cento.
`````

`````{tab} Superiore
```{code-block} python
:class: pt-lento

# come sopra, e qui il modello da scaricare è di 669 MB.
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

I pesi di questo modello risiedono sui server di chi lo pubblica: se in futuro
venissero aggiornati, potrebbe cambiare anche la graduatoria. Per riprodurre
queste cifre si fissa la versione dei pesi con l'argomento
`revision` (l'identificativo di una versione sul sito) e quella della libreria.

Il modello è un BERT multilingue rifinito (*fine-tuned*) su recensioni: la
classificazione usa la rappresentazione del token speciale `[CLS]`, passata
attraverso uno strato denso con tangente iperbolica (il *pooler* di BERT) e poi
inviata a una testa lineare a cinque uscite (architettura encoder-only, senza
generazione).

Le prime due frasi danno il risultato atteso; la terza no, e per questo il
codice stampa la graduatoria e non solo la vincente. Sulla terza i valori sono
$0{,}365$ a due stelle e $0{,}336$ a tre: uno scarto di ventinove millesimi,
l'unico caso fra le quattro frasi in cui le prime due classi risultano quasi
appaiate. Il solo `argmax` direbbe «2 stars» e si fermerebbe lì,
indistinguibile dai verdetti delle altre tre frasi, dove la seconda classe
resta indietro di centocinquanta millesimi o più: [`top_k=None` rende visibile
la differenza fra una classificazione netta e una in bilico]{.pt-matita}.

Il punteggio della classe vincente è la probabilità assegnata dalla softmax, ma
nulla garantisce che sia calibrata, ossia che fra le frasi con punteggio 0,6 ne
risultino corrette sei su dieci. Le reti neurali moderne sono spesso mal
calibrate {cite}`guo2017calibration`; la {doc}`sezione su come si valuta un
modello
</MachineLearning/metriche>` spiega come verificarne la calibrazione. Inoltre,
un punteggio isolato non dice nulla sulla qualità del classificatore, che va
misurata con le metriche di quella sezione (accuratezza, precision e recall)
sui dati del dominio di interesse. Su testi diversi da quelli di addestramento
(ironia, sarcasmo, gergo) le prestazioni di solito calano, e una sola litote
non basta a dire di quanto.
`````

Le quattro frasi sono una lode, una stroncatura e due litoti, «non è affatto
male» e «non è male». L'errore sulla terza è la cosa più utile dei due esempi:
il modello dà a «non è affatto male» due stelle su cinque, cioè la legge come
una recensione scontenta, mentre a «non è male», la stessa frase senza
l'avverbio, ne dà tre. Il modello non sceglie una risposta sola ma
assegna una probabilità a ciascuna delle cinque stelle, e due e tre stelle
restano quasi alla pari. Il quasi-pareggio sta fra due modi di sbagliare e non
fra sbagliare e indovinare: le quattro e le cinque stelle, che sarebbero la
lettura giusta di un complimento, si dividono in tutto sette centesimi.

L'errore dipende probabilmente dai testi in cui compare «affatto», quasi sempre
dentro una stroncatura piena («non mi è piaciuto affatto»): il modello ne
eredita l'associazione. È una spiegazione plausibile e non una verifica, perché
i testi su cui questo modello è stato addestrato non si possono ispezionare:
chi lo ha addestrato non li ha pubblicati.

Più in generale, dire una cosa negando il suo contrario richiede di comporre
tre parole in un significato che nessuna delle tre porta con sé: l'attenzione
mette «non», «affatto» e «male» in contatto, ma il contatto non garantisce che
la composizione dia il risultato giusto. Con le sole due frasi facili la
dimostrazione sarebbe riuscita bene e avrebbe insegnato meno: quattro frasi
provate al volo non sono un collaudo.

`````{tab} Elementare
```{admonition} Da ricordare
:class: important
- La traduzione usa il Transformer intero: la torre che legge, la torre che
  scrive, e il continuo rileggersi l'originale mentre si traduce.
- Per capire una recensione basta la torre che legge, con in cima un giudice
  che dà il voto (qui, da una a cinque stelle). In gergo è la famiglia
  *encoder-only*, e il capostipite si chiama BERT.
- Non serve costruire niente da zero: esistono cassette degli attrezzi (la
  libreria `transformers`) piene di modelli già addestrati da altri, che si
  usano in poche righe.
- I risultati vanno sempre provati sui propri testi: ironia, modi di dire e
  complimenti detti al contrario restano difficili, come mostra il "non è
  affatto male" dell'esempio sulle recensioni.
```
`````

`````{tab} Superiore
```{admonition} Da ricordare
:class: important
- La traduzione usa il Transformer completo: encoder che legge, decoder
  che genera, cross-attention che li allinea.
- L’analisi del sentiment usa un *encoder-only* (come BERT) con una testa
  di classificazione: capire, non generare.
- La libreria `transformers` di Hugging Face (su PyTorch) dà accesso a
  modelli pre-addestrati per entrambi i compiti in poche righe: sotto, sono
  `nn.Module` come quelli della {doc}`sezione sui moduli </PyTorch/moduli>`.
- I risultati vanno validati sul proprio dominio: ironia, gergo e litoti
  restano difficili, e «non è affatto male», letta come scontenta, ne è il
  controesempio. La probabilità della classe vincente non è per forza
  calibrata.
```
`````

BERT, il modello del secondo esempio, è uno dei tre capostipiti da cui nascono
le famiglie di Transformer, e la {doc}`sezione sulle famiglie di modelli
<multimodalita>` lo mette accanto a GPT e a T5.
