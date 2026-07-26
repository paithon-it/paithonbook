# Esempi pratici

Dopo tanta architettura, mettiamo i Transformer al lavoro su due compiti
concreti: tradurre una frase e capire se una recensione è entusiasta o
delusa. Sono gli stessi esempi che un lettore incontra ogni giorno senza
pensarci — il tasto "traduci" sotto un post, il termometro delle recensioni
di un prodotto — e per fortuna non serve addestrare nulla da zero: la
libreria `transformers` di Hugging Face, costruita su PyTorch, mette a
disposizione migliaia di modelli già addestrati.

## Traduzione automatica

Il compito per cui il Transformer è nato: l'encoder legge la frase di
partenza, il decoder scrive quella d'arrivo, la cross-attention le tiene
allineate.

`````{tab} Elementare
Segui il viaggio di "The cat sits on the mat". Prima la frase viene spezzata
in mattoncini (le parole o pezzi di parola: i *token*) e l'encoder la legge
tutta, costruendo per ogni parola quella rappresentazione ricca di contesto
che conosciamo. Poi il decoder comincia a scrivere in italiano, una parola
alla volta: quando deve produrre "gatto" il suo evidenziatore punta su
"cat", quando produce "siede" punta su "sits". Non è un dizionario che
sostituisce parola per parola: è più simile a un traduttore che legge tutta
la frase, la capisce, e la riscrive — infatti se la frase fosse "The cat
sits on the *bank*", saprebbe scegliere tra "panchina" e "banca" guardando
il contesto.
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

frase = "The cat sits on the mat."
ingresso = tokenizzatore(frase, return_tensors="pt")     # testo -> token
uscita = modello.generate(**ingresso, max_new_tokens=40) # decodifica autoregressiva
print(tokenizzatore.decode(uscita[0], skip_special_tokens=True))
# Il gatto si siede sul tappeto.
```

Le tre righe finali sono i tre passaggi visti nei capitoli precedenti, qui
scritti in chiaro: **tokenizzazione** (la frase diventa una sequenza di id di
token), **inferenza** con `generate` (encoder e decoder Transformer, con
generazione autoregressiva e maschera causale) e **decodifica** (dagli id di
token al testo). La libreria offre anche una scorciatoia, `pipeline`, che li
incapsula in una riga; qui li teniamo separati perché sono esattamente i pezzi
che il capitolo ha spiegato — e perché i nomi delle scorciatoie cambiano tra
versioni, mentre `AutoTokenizer` e `generate` no. Sotto il cofano il modello è
un `nn.Module` PyTorch come quelli del capitolo su PyTorch: con
`modello.named_parameters()` si ispezionano strati, teste di attenzione e
parametri.
`````

## Analisi del sentiment

Qui il Transformer non deve generare nulla: deve *capire* e classificare —
positivo, negativo — un compito perfetto per un modello solo-encoder.

`````{tab} Elementare
"Mi è piaciuto moltissimo questo prodotto!" e "Una delusione totale": per te
è ovvio, e il bello è che ormai lo è anche per la macchina — comprese le
sfumature che fregavano i sistemi vecchi, tipo "non è affatto male", dove le
parole "non" e "male" sembrano negative ma la frase è un complimento. Il
modello legge la frase intera con l'attenzione, così "non" e "male" si
guardano a vicenda e il significato combinato emerge. Aziende e ricercatori
lo usano per misurare l'umore di migliaia di recensioni o commenti in pochi
secondi — un lavoro che a mano richiederebbe settimane.
`````

`````{tab} Superiore
```{code-block} python
:class: pt-lento

from transformers import pipeline

# modello multilingue (italiano compreso) che assegna da 1 a 5 stelle
giudice = pipeline("sentiment-analysis",
                   model="nlptown/bert-base-multilingual-uncased-sentiment")

recensioni = [
    "Mi è piaciuto moltissimo questo prodotto!",
    "Questo prodotto è stato una delusione totale.",
    "Non è affatto male.",
]
for r in recensioni:
    esito = giudice(r)[0]
    print(f"{r!r} -> {esito['label']} (confidenza {esito['score']:.2f})")
```

Il modello è un BERT multilingue rifinito (*fine-tuned*) su recensioni: la
classificazione usa la rappresentazione del token speciale `[CLS]` passata a
una testa lineare — architettura solo-encoder, senza generazione. Si noti la
**confidenza**: un classificatore serio si valuta con le metriche del
capitolo sul machine learning (accuratezza, precision/recall), e su domini
diversi da quello di addestramento (ironia, sarcasmo, gergo) le prestazioni
calano sensibilmente. La demo è convincente; la validazione sul *tuo* dominio
resta obbligatoria.
`````

```{admonition} Da ricordare
:class: important
- La **traduzione** usa il Transformer completo: encoder che legge, decoder
  che genera, cross-attention che li allinea.
- L'**analisi del sentiment** usa un solo-encoder (stile BERT) con una testa
  di classificazione: capire, non generare.
- La libreria `transformers` di Hugging Face (su PyTorch) dà accesso a
  modelli pre-addestrati per entrambi i compiti in poche righe: sotto, sono
  `nn.Module` come quelli del capitolo su PyTorch.
- I risultati vanno **validati sul proprio dominio**: ironia, gergo e testi
  lontani dai dati di addestramento restano difficili.
```
