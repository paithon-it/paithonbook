# Contribuire a Paithon Book

Il libro è pubblico perché il lavoro migliora quando qualcuno lo legge
davvero. Le segnalazioni sono benvenute, e più sono precise, più diventano
correzioni.

## Il modo più rapido: seleziona e segnala

Leggendo il libro, **seleziona la frase** che non torna: compare un pulsante
*Segnala* che apre una issue già compilata con la citazione, il link alla
pagina con l'ancora della sezione e il file sorgente. Non devi cercare niente
a mano.

In alternativa: [apri una issue](https://github.com/paithon-it/paithonbook/issues)
scrivendo **dove** (capitolo e sezione), **cosa** c'è scritto e **perché** non
va. Per cose private, `info@paithon.it`.

## Che cosa serve davvero

Utile:

- **errori di fatto**, una data, un autore, un numero, un benchmark sbagliato;
- **conti che non tornano**, un esempio numerico che non si riproduce, una
  formula con un indice fuori posto, un pezzo di codice che non gira;
- **passaggi poco chiari**, se una spiegazione non ha funzionato su di te, è
  un dato: dire *dove* ti sei perso vale più di una riscrittura;
- **link morti**, refusi, figure che non caricano.

Meno utile, o da concordare prima in una issue:

- **capitoli e sezioni nuove** inviate come pull request senza averne parlato:
  l'indice segue un percorso, e ogni pagina deve rispettare regole editoriali
  vincolanti (vedi sotto). Meglio proporre l'idea e poi scriverla;
- **riscritture stilistiche** di testo corretto: la voce del libro è una scelta,
  non una svista;
- contenuti promozionali, link affiliati, tool da recensire.

## Se proponi del testo

Queste regole sono vincolanti. Le tre che contano più delle altre:

1. **Due livelli.** Ogni concetto-chiave si spiega su *Elementare* e
   *Superiore* con le tab. L'Elementare deve superare il "test
   dell'ombrellone": comprensibile a uno studente di liceo, letto di fila.
2. **Ogni livello si regge da solo.** Le tab non sono sincronizzate e il
   lettore ha un interruttore globale: qualcuno attraverserà tutto il libro
   senza vedere mai l'altro livello. Il Superiore non è l'Elementare con più
   formule, e l'Elementare non è il Superiore tagliato.
3. **Ogni affermazione fattuale è verificata su fonti primarie** (paper,
   documentazione ufficiale) e citata in `book/references.bib`. Il codice
   Python va eseguito, non immaginato.

In pratica, scrivendo una pagina:

- le tab si aprono con **cinque backtick**, sempre nell'ordine Elementare →
  Superiore, e avvolgono i *passaggi* dove la profondità cambia le cose (una
  definizione, una derivazione, una formula). Incipit, motivazione e
  transizioni stanno fuori e valgono per tutti i lettori:

  ``````md
  `````{tab} Elementare
  Analogia concreta, zero prerequisiti.
  `````

  `````{tab} Superiore
  Definizione formale, notazione, formule.
  `````
  ``````

- **niente lineette** (`—`): non sono nello stile del libro. Un inciso che si
  potrebbe togliere va fra parentesi, un concetto racchiuso dentro la frase
  fra virgole, una spiegazione dopo i due punti, due proposizioni che si
  oppongono separate dal punto e virgola; per una digressione più lunga, una
  nota a piè di pagina (`[^nota]`). Restano il trattino d'unione e la lineetta
  breve degli intervalli (`2020–2023`);
- **formule** in LaTeX (`$…$` in linea, `$$…$$` in blocco), con i simboli
  spiegati subito dopo. Notazione coerente col resto del libro, su tre
  livelli: matrici in maiuscolo grassetto ($\mathbf{X}$, $\mathbf{W}$),
  vettori in minuscolo grassetto ($\mathbf{x}$, $\mathbf{w}$), scalari e
  indici in minuscolo tondo ($x_i$, $n$); poi $\hat{y}$ per le predizioni,
  $\mathcal{L}$ per la loss, $\theta$ per i parametri. Il grassetto non è un
  vezzo tipografico: dice che l'oggetto ha più di una componente, e distingue
  un vettore da uno scalare;
- **codice**: Python idiomatico ed eseguibile, commenti in italiano e brevi.
  Il framework di deep learning del libro è **PyTorch**: niente
  Keras/TensorFlow, che possono comparire solo come citazione storica o
  confronto. Se una pagina ha un notebook in `notebooks/`, il codice viene
  eseguito dalla CI: un blocco che non gira fa fallire la verifica;
- **figure**: solo SVG geometriche in palette (terracotta `#B5532C`, teal
  `#2D5A5C`, ocra `#C9A961`, warm-black `#1A1A1A`, cream `#F8F5EE`), leggibili
  anche in tema scuro. Niente immagini generate da AI, niente stock, niente
  gradienti. Se una figura non aggiunge comprensione, non serve;
- **ogni nuovo file va aggiunto a `book/_toc.yml`**, altrimenti non viene
  nemmeno costruito (`only_build_toc_files: true`). Il toc raggruppa i
  capitoli in parti (`parts:` con `caption:`, sono i blocchi dell'indice di
  sinistra): un capitolo nuovo va dentro la parte a cui appartiene. Ogni voce
  porta anche un `title:` breve, è l'etichetta nell'indice, dove un titolo
  lungo andrebbe a capo due volte; il titolo esteso resta l'H1 della pagina.
  Un capitolo nuovo vuole anche la sua scheda nella griglia di
  `book/intro.md`: il **numero** non si scrive (lo conta il CSS, lo
  `<span class="pt-card-num">` resta vuoto), e
  `python3 scripts/coerenza.py --solo landing` dice se una scheda manca o è
  fuori ordine.

Una cosa che il libro **non** ospita: la cronaca. Classifiche, benchmark,
prezzi, "questo modello supera quell'altro" invecchiano dentro un testo che si
legge per anni. Qui si spiega come funziona un meccanismo; le notizie stanno su
[paithon.it](https://paithon.it).

## Vedere le proprie modifiche

Il libro è un [Jupyter Book](https://jupyterbook.org), costruito con la
distribuzione [TeachBooks](https://teachbooks.io) (la stessa che usa il
deploy). In locale:

```bash
pip install -r requirements.txt
teachbooks build book      # equivale a `jupyter-book build book`
python -m http.server 8080 --directory book/_build/html
```

Non è obbligatorio: la pubblicazione la fa GitHub Actions a ogni push su
`book/`. Se preferisci segnalare a parole, va benissimo: vedi sopra.

## Come entra una correzione

Il libro si scrive in un repository di lavoro **privato** e qui arriva
pubblicato: è la ragione per cui la storia di questo repository è fatta di
pubblicazioni e non di commit quotidiani. Per chi contribuisce cambia una cosa
sola: la **issue** è la strada diretta, mentre una **pull request** accettata
viene riportata a mano nel ramo di lavoro e torna qui con la pubblicazione
successiva. Il commit che la porta online non è il tuo, ma ti cita come
coautore: il credito segue la correzione.

Se ci vuole qualche giorno, non è disinteresse: è che una correzione entra
quando il capitolo intorno regge ancora.

## Licenza dei contributi

Serve chiarirlo, perché i testi del libro sono sotto **CC BY-NC-ND 4.0**: una
licenza che *vieta le opere derivate*. Un paragrafo corretto è tecnicamente
un'opera derivata, quindi senza una concessione esplicita non potrebbe essere
pubblicato nemmeno se la correzione fosse giusta.

Aprendo una issue o una pull request con del testo, delle figure o del codice,
dichiari che:

1. il contributo è **opera tua**, oppure hai il diritto di conferirlo (in
   particolare: non è copiato da libri, corsi o articoli di terzi);
2. concedi a Francesco Messina / paithon.it una licenza **non esclusiva,
   irrevocabile, gratuita e valida in tutto il mondo** per pubblicarlo,
   modificarlo e distribuirlo come parte del libro, comprese le edizioni
   future, anche a stampa o commerciali;
3. per il **codice**, il contributo si intende sotto
   [Apache 2.0](LICENSE-CODE) come il resto del codice del progetto (è la
   regola di default della licenza stessa, §5), concessione di brevetto
   inclusa.

Resti autore di quello che hai scritto: la concessione è una licenza, non una
cessione. Chi segnala un errore viene citato nel commit che lo corregge: la
storia di git è il registro dei contributi.

