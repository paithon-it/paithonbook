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
2. **Ogni livello si regge da solo, e i due sono la stessa struttura.** Le tab
   non sono sincronizzate e il lettore ha un interruttore globale: qualcuno
   attraverserà tutto il libro senza vedere mai l'altro livello. Il Superiore
   non è l'Elementare con più formule, e l'Elementare non è il Superiore
   tagliato: sono due presentazioni della stessa struttura. Prima di chiudere
   una coppia di tab, elenca le *mosse* del Superiore (la scelta di scala, il
   vincolo, l'ottimizzazione, la garanzia, il caso in cui il metodo si rompe)
   e controlla che ciascuna abbia il suo gesto nella scena dell'Elementare: un
   esempio calzante che illustra il risultato senza rifare le mosse descrive
   un altro algoritmo. **E il controllo va fatto anche al contrario**, perché
   «isomorfismo» vuol dire corrispondenza biunivoca: capoverso per capoverso
   dell'Elementare, a quale mossa del Superiore corrisponde? Un capoverso che
   non corrisponde a niente non arricchisce la scheda, la gonfia, e a forza di
   aggiungerne la scena evapora: quello che resta è il Superiore detto in
   parole piane, che per il suo lettore è inutile. Se dopo aver scritto non
   c'è più niente di concreto (una persona che fa un gesto, un oggetto, un
   luogo) che *porti* la spiegazione, la scheda è da rifare; e una formula
   elementare col suo esempio numerico sta dentro la scena, non è un residuo
   del Superiore, quindi non si toglie per accorciare. Chi passa
   dall'Elementare al Superiore non deve
   disimparare niente; l'Elementare può omettere, non può dire una cosa che il
   Superiore dovrà correggere.
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

- **fra un'Elementare e la sua Superiore non va niente**, nemmeno una riga di
  commento che sembra utile. Separate da un capoverso smettono di essere una
  coppia: diventano due gruppi da una linguetta ciascuno, e una linguetta sola
  non si può chiudere, quindi il livello Superiore resta aperto anche a chi ha
  scelto Elementare. Nel sorgente si legge benissimo, e per questo scappa. Se
  hai qualcosa da dire, va dopo la chiusura della Superiore. Vale anche il caso
  opposto: due coppie che si toccano la build le fonde in una fila di quattro
  linguette, e serve del testo comune in mezzo. Tutti e due li vede
  `python3 scripts/coerenza.py --solo schede`.

- **niente lineette** (`—`): non sono nello stile del libro. Un inciso che si
  potrebbe togliere va fra parentesi, un concetto racchiuso dentro la frase
  fra virgole, una spiegazione dopo i due punti, due proposizioni che si
  oppongono separate dal punto e virgola; per una digressione più lunga, una
  nota a piè di pagina (`[^nota]`). Restano il trattino d'unione e la lineetta
  breve degli intervalli (`2020–2023`). Vale per **tutti e tre** i modi di
  scriverla: il carattere lungo si vede scorrendo la pagina, le sue versioni in
  ASCII no, e sono quelle che scappano più facilmente. Quindi niente nemmeno
  `parola -- parola` né `parola - parola`. Se hai dubbi,
  `python3 scripts/coerenza.py --solo lineette` te lo dice, e non segnala i
  segni meno delle formule né le opzioni da riga di comando;
- **formule** in LaTeX (`$…$` in linea, `$$…$$` in blocco), con i simboli
  spiegati subito dopo. Notazione coerente col resto del libro, su tre
  livelli: matrici in maiuscolo grassetto ($\mathbf{X}$, $\mathbf{W}$),
  vettori in minuscolo grassetto ($\mathbf{x}$, $\mathbf{w}$), scalari e
  indici in minuscolo tondo ($x_i$, $n$); poi $\hat{y}$ per le predizioni,
  $\mathcal{L}$ per la loss, $\theta$ per i parametri. Il grassetto non è un
  vezzo tipografico: dice che l'oggetto ha più di una componente, e distingue
  un vettore da uno scalare. E quando hai scritto la spiegazione dei simboli,
  **rileggila coprendo la formula**: la frase che glossa una formula giusta
  può dire un'altra cosa, e chi legge crede alla prosa, che è quella che
  capisce. «$\gamma = 1/(2\sigma^2)$, quindi $\gamma$ è l'inverso del quadrato
  della larghezza» si smentisce da solo a due parole di distanza, e il rimedio
  è una preposizione: *va come* l'inverso del quadrato, non *è*;
- **il verso di una metrica** si dice per ciascuna. Metriche in fila sotto un
  verbo solo («premiano i gruppi compatti») fanno credere che si leggano tutte
  allo stesso modo; se una si minimizza e le altre si massimizzano, chi cerca
  il valore più alto sceglie il risultato peggiore seguendo la pagina alla
  lettera;
- **codice**: Python idiomatico ed eseguibile, commenti in italiano e brevi.
  Il framework di deep learning del libro è **PyTorch**: niente
  Keras/TensorFlow, che possono comparire solo come citazione storica o
  confronto. Se una pagina ha un notebook in `notebooks/`, il codice viene
  eseguito dalla CI: un blocco che non gira fa fallire la verifica. E se il
  testo **commenta** un numero, quel numero lo deve **stampare il codice**: un
  valore scritto a mano accanto a un blocco che non lo produce si scolla al
  primo ritocco, e chi legge esegue e non lo trova;
- **figure**: solo SVG geometriche in palette (terracotta `#B5532C`, teal
  `#2D5A5C`, ocra `#C9A961`, warm-black `#1A1A1A`, cream `#F8F5EE`), leggibili
  anche in tema scuro. Niente immagini generate da AI, niente stock, niente
  gradienti. Se una figura non aggiunge comprensione, non serve;
- **animazioni**: dove **il tempo è il contenuto** (qualcosa che scorre,
  converge, si accumula, si propaga, si genera passo dopo passo) la figura può
  muoversi, e i generatori stanno in `animazioni/svg/`, uno per figura. Regola
  vincolante: il disegno **fermo è lo stato finale**, con le coordinate vere e
  nessuna trasformazione, perché è quello che vedono la stampa, il PDF e chi ha
  chiesto al sistema di ridurre le animazioni; il movimento parte dall'inverso
  e finisce sull'identità. Solo `@keyframes` CSS, mai `<script>`. Se scrivi un
  capitolo nuovo, la domanda «qui il tempo è il contenuto?» va **posta e
  risposta**: se la risposta è no, si dichiara il capitolo in
  `animazioni/senza-clip.toml` scrivendo perché. Un capitolo a zero clip che
  non compare là fa fallire il controllo, e non per pignoleria: senza quella
  riga la domanda si salta senza che nessuno se ne accorga;
- **la firma visiva non si modifica da qui**: colori, tipografia, sfondo
  animato, regole di stampa e lo stile delle figure animate vivono in
  `book/_static/brand`, che è un submodule del design system condiviso col
  sito paithon.it. Una PR che ci scrive dentro, o che ne aggira un valore
  hard-codando un colore in `_static/custom.css`, non può essere accettata:
  vale solo per il libro e il sito resterebbe indietro. Se pensi che un token
  vada cambiato, aprine una issue e ne parliamo;
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

- **Quello che vale solo online va dichiarato.** Il libro esiste in due
  formati: il sito e un PDF unico, che si scarica dalla home. Una frase come
  «in alto trovi il pulsante *Esegui il codice*» è vera sul sito e falsa su
  carta, quindi si avvolge:

  ````md
  :::{only} html
  In alto trovi il pulsante «Esegui il codice».
  :::

  :::{only} latex
  Il codice di questo capitolo si esegue online, su book.paithon.it.
  :::
  ````

  Attenzione a come si annidano le recinzioni MyST: **l'esterna deve essere
  più lunga dell'interna** (`::::{only}` che contiene `:::{container}`). Al
  contrario il blocco non si chiude dove sembra, e l'errore che ne esce parla
  d'altro.

Una cosa che il libro **non** ospita: la cronaca. Classifiche, benchmark,
prezzi, "questo modello supera quell'altro" invecchiano dentro un testo che si
legge per anni. Qui si spiega come funziona un meccanismo; le notizie stanno su
[paithon.it](https://paithon.it).

## La pagina deve leggersi di fila

La correttezza è la soglia d'ingresso, non il traguardo: una pagina può essere
tutta vera e restare illeggibile. Il traguardo è che uno se la legga di fila,
senza fermarsi. Il metro è un buon articolo divulgativo, non una dispensa
universitaria.

Queste cinque cose sono **difetti**, non questioni di gusto:

- **il periodo lungo con tre subordinate**, che regge la dimostrazione e uccide
  la lettura. Una idea per frase: se una frase ha due «che» e un «il quale»,
  sono due frasi;
- **il termine tecnico prima della cosa che nomina.** Prima si dice cosa fa,
  poi come si chiama: «una rete che passa messaggi ai vicini, il *message
  passing*», non il contrario;
- **l'attacco di sezione più difficile del resto della pagina.** È il difetto
  che rompe più spesso la lettura: chi legge il livello Elementare non ha
  ancora difese, e proprio lì gli si parla come al Superiore. Stessa cosa per le didascalie
  delle figure e i riquadri «Da ricordare», che spesso poggiano su un termine
  definito **solo** nella tab Superiore, cioè in un posto che quel lettore non
  aprirà mai;
- **la parentetica che spezza il respiro** dove il lettore stava prendendo il
  ritmo. Se l'inciso è lungo, è una frase dopo, o una nota;
- **le citazioni infilate a metà periodo**: `{cite}` a fine frase, non fra
  soggetto e verbo.

E la soglia non è la stessa dappertutto: nei capitoli d'apertura
(Introduzione, Python, Matematica, Machine Learning, Reti neurali, PyTorch,
Deep Learning) arriva chi non ha basi, ed è lì che si decide se continuerà a
leggere.

**Se correggi, rileggi.** Una correzione ne introduce di nuove più spesso di
quanto sembri, e chi rilegge quello che ha appena corretto trova guasti che
la correzione stessa ha creato. Due accortezze che costano poco: cerca la tua
correzione in **tutte e due le tab** (il libro dice ogni cosa due volte, e
riparare solo il Superiore lascia l'errore dove fa più danno), e se rendi
*precisa* una frase vaga apri prima la fonte, perché una frase precisa e
sbagliata è peggio di una vaga e innocua.

E cerca la tua correzione anche nel riquadro **«Da ricordare»** in fondo alla
pagina. È il posto dove le correzioni si dimenticano più spesso, perché è un
riassunto e sembra staccato dal punto che hai toccato, mentre in realtà ripete
le affermazioni delle schede quasi parola per parola. Riparata la scheda e non
il riquadro, non resta un errore in un angolo: è la pagina che si contraddice
da sola a dieci righe di distanza.

## Quello che in una pagina non ci deve stare

Un testo tecnico che si legge bene non è più semplice né più povero di
analogie: **non marca tipograficamente il ritmo del proprio ragionamento**. Da
lì questi tetti, che sono vincoli e non preferenze.

- **Niente grassetto su una parola-funzione** (*non*, *solo*, *due*, *mai*,
  *sempre*). Il grassetto presenta un termine al suo primo uso, una volta
  sola, e non avvolge mai una frase intera.
- **Al massimo due «non è X, è Y» per capitolo.** Contano anche le sorelle
  corte: «non è un dettaglio», «non è un caso», «non è teoria». Usato una
  volta è una scossa; ripetuto, diventa il ritmo di fondo della voce e smette
  di segnalare qualcosa.
- **Al massimo una scheda su dieci apre con «Immagina di…» o «Pensa a…».** Si
  entra nella scena invece di ordinare al lettore di immaginarla.
- **Niente formule di riempimento**: «vale la pena…», «conviene notare
  che…», «va detto che…», «Misurato:». Premettono al fatto l'annuncio del
  fatto. Non si sostituiscono con un'altra formula: si toglie il costrutto.
- **Niente triadi di serie** («tre cose, e conviene tenerle distinte»). E una
  lista a cui si torna si richiama per nome, mai per numero.
- **Niente sentenza in grassetto seguita dalla sua spiegazione**: è la cadenza
  di una diapositiva, non di un libro.

**E il libro non parla di sé.** È la riga che conta più di tutte le altre
messe insieme. Nel corpo del testo non devono comparire:

- «questo libro», «questa pagina», «questa sezione», «più avanti in questa
  pagina»; e la variante che sfugge perché non nomina il libro ma indica la
  propria impaginazione, «il conto qui sotto», «la tabella qui sopra», «il
  paragrafo che segue». Un rimescolamento le rende false in silenzio, e nel PDF
  il sopra e il sotto non cadono dove cadono in HTML: si nomina la cosa;
- i nomi dell'impianto editoriale, «la scheda Elementare», «la tab Superiore»,
  «il testo comune». Ogni livello deve reggersi da solo, quindi nessuno dei
  due può rimandare all'altro: chi legge solo l'Elementare non vedrà mai la
  scheda accanto;
- la cronaca della lavorazione: «misurato», «rifatto il conto», i controlli
  automatici, la macchina su cui la prova è stata fatta. Che un numero sia
  stato misurato lo si dimostra stampandolo;
- i permessi al lettore: «si può saltare senza perdere il filo», «chi non
  programma può scorrere». Si autoavverano, e ammettono che quel pezzo non si
  è saputo giustificare;
- le sezioni intitolate alla propria struttura. Annunciare il percorso va
  bene; intitolare una sezione «Come è organizzato il capitolo» no.

Fanno eccezione la Prefazione, la pagina di apertura e le Conclusioni, dove il
libro ha diritto di presentarsi: e anche lì si parla della promessa al
lettore, non del processo di redazione.

**Un rimando si nomina e si linka.** Ogni volta che il testo cita un altro
capitolo o un'altra sezione, quel riferimento va scritto come link MyST
`{doc}` e deve nominare il bersaglio invece di indicarne la posizione: «la
sezione sugli ensemble» non invecchia, «la sezione precedente» sì. Punta alla
**sezione** e non all'apertura del capitolo, quando la cosa promessa sta in
una sezione precisa; e «capitolo» si dice solo di un capitolo, perché chi
legge «sezione» resta sulla pagina e chi legge «capitolo» va nell'indice.

**E se togli un capoverso, guarda la cucitura.** Un capoverso spesso apre
qualcosa che continua più sotto: un ordinale, un nome che scioglie, una lista.
Togliendolo, quel seguito resta senza il suo principio, e il diff non lo
mostra. Prima di cancellare, cerca che cosa quel capoverso *introduceva*, e
apri il file per vedere che cosa resta attaccato sopra e sotto.

L'altro difetto tipico di una riscrittura è la frase che ripete quella dopo,
quando si sostituisce un pezzo con uno che anticipa il seguito. Il markdown
resta valido e nessuno se ne accorge, quindi c'è un controllo:
`python3 scripts/coerenza.py --solo doppioni`.

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

Ogni pubblicazione lascia una voce nella pagina **Aggiornamenti** del libro,
con il numero di versione, la data e il link alle pagine toccate: di lì si
vede quando una segnalazione è diventata una correzione online. Il registro è
`book/_dati/aggiornamenti.yml` e la pagina la scrive
`python3 scripts/genera-aggiornamenti.py`; una pull request non deve
aggiornarli, ci pensa chi pubblica.

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

