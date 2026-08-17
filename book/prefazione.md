# Prefazione

Questo libro è stato scritto una prima volta nel 2019, per uscire su carta.
Aveva un indice di quattordici capitoli. Su carta non è mai uscito, e il
manoscritto è rimasto in un cassetto: la prima forma in cui arriva a qualcuno
è questa, online e gratuita.

Con il senno di poi è stata una fortuna. Quell'indice si fermava al deep
reinforcement learning e prometteva codice in TensorFlow e Keras; la parola
*Transformer* non compariva da nessuna parte, benché l'articolo che la
introduce (*Attention Is All You Need*, di Vaswani e colleghi
{cite}`vaswani2017attention`) avesse già più di due anni, e il capitolo sul
linguaggio naturale finiva alle reti ricorrenti. Se questi nomi non ti dicono
niente, benissimo: sono esattamente le cose che imparerai. Non era una
distrazione, era la faccia che questa materia aveva, vista da chi ci lavorava,
in quel momento. Un libro stampato in quella forma avrebbe cominciato a
invecchiare il giorno stesso in cui arrivava in libreria.

Online il problema si pone in un altro modo. Un errore si corregge il giorno
in cui qualcuno lo segnala, e il testo cresce quando cresce il campo: del
manoscritto è rimasta l'ossatura, e attorno sono nati i capitoli che allora
non si potevano scrivere. Il codice, nel frattempo, è passato a PyTorch. Quello
che non è cambiato è il motivo per cui il libro esiste.

Quello che è cambiato, e molto, è il mondo intorno. Nel 2019 questa era una
materia da addetti ai lavori. Oggi la usano tutti, e attorno si sono mossi
interessi industriali e commerciali di ogni ordine di grandezza. È guardando
quel passaggio che ho capito perché valeva la pena riprendere il manoscritto.

Nel 2021 un articolo molto discusso ha chiamato i modelli linguistici
**pappagalli stocastici** {cite}`bender2021dangers`: macchine che rimettono
insieme pezzi di quello che hanno letto senza capire quello che dicono.
L'espressione ha fatto il giro del mondo. E a un certo punto mi sono accorto
che, a forza di ripeterla, i pappagalli eravamo diventati noi.

Succedeva in due modi, e si somigliavano. Usciva una notizia, e per settimane
la stessa frase rimbalzava identica sui canali di chi commenta questa materia
di mestiere, ripetuta da gente che alla fonte non era mai andata. Nelle aziende
capitava la versione tecnica: si scaricava un modello da Hugging Face, lo si
metteva in produzione perché lo stavano facendo tutti, e spesso nessuno apriva
la licenza per vedere se quell'uso commerciale fosse permesso. Il gesto era lo
stesso: ripetere una cosa senza sapere che cosa si stava dicendo.

Non è un'impressione mia. Emily M. Bender, prima firma di quell'articolo, ha
passato cinque anni a guardare che fine faceva la sua espressione: all'inizio,
racconta, la usava chi il paper l'aveva letto, e poi «la frase ha superato il
paper» {cite}`bender2026unasked`. È il pappagallo che si descrive da solo.

Intanto, di quei modelli, la cosa che si ripeteva più spesso è che si
limitano a indovinare una parola per volta. È vero, ma dice come il testo
esce, non come viene deciso. Il gruppo che uno di questi modelli lo aveva
costruito è andato a guardarci dentro mentre scriveva una poesia in rima
{cite}`lindsey2025biology`, aspettandosi di vederlo procedere parola per
parola e aggiustare la rima all'ultimo momento. Ha trovato il contrario. Prima
ancora di cominciare il verso, il modello aveva già in testa le parole con cui
poteva chiuderlo, e scriveva il verso per arrivarci; tolta quella scelta, ne
scriveva un altro che finiva su una rima diversa. La riga con cui gli autori lo
riassumono è questa: anche se questi modelli sono addestrati a produrre una
parola per volta, per farlo possono pensare su orizzonti molto più lunghi.
Predire un token alla volta non vuol dire ragionare un token alla volta.

Su questo Bender non sarebbe d'accordo, e conviene dirlo. L'obiezione che le
arriva puntuale, racconta, è sempre la stessa: i pappagalli stocastici
andavano bene un tempo, ma adesso non più, perché è appena uscito un modello
che fa una cosa che prima non faceva. Per lei non è nemmeno un'obiezione, dato
che quella non era una previsione da superare ma «una descrizione, o una
metafora» di come sono fatte quelle macchine. E il bersaglio della sua critica,
aggiunge, non sono affatto i modelli: «quello che mi preoccupa sono le azioni
delle persone», il furto dei dati, lo sfruttamento del lavoro, l'indifferenza
per l'impatto ambientale, e «la sorprendente disponibilità di tanti a cedere il
proprio potere e affidarsi a testo sintetico».

Su che cosa faccia una macchina la penso diversamente da lei. Su chi sia il
pappagallo, no. E l'unica cosa che separa chi capisce da chi ripete è sapere
come funziona la cosa di cui si sta parlando: è quello che queste pagine
provano a dare.

Il metodo lo trovi descritto nella {doc}`pagina di apertura </intro>`: ogni
concetto che conta è spiegato due volte, una con un'analogia che non chiede
prerequisiti e una con le formule al posto giusto, e decidi tu da quale parte
stare. Il resto è presto detto: nessuna scorciatoia sulle cose difficili,
nessun entusiasmo che il testo non sia in grado di giustificare, e le fonti
sempre citate, così che tu possa andare ad approfondire.

Lo scopo di questo libro è introdurre i concetti principali del machine
learning, del deep learning e del reinforcement learning (che cosa siano lo
dice il primo capitolo, in una riga per ciascuno, prima di dedicarci il
resto), e fornire gli strumenti utili a costruire un applicativo intelligente,
a valutare un'idea, o semplicemente a riconoscere l'intelligenza artificiale
nelle tecnologie di oggi e di domani.

C'è poi una cosa da dire su **come** è scritto, perché riguarda chi legge.
Buona parte di queste pagine nasce lavorando con l'intelligenza artificiale: è
lei a stendere, a cercare, a proporre. Questo libro è, alla lettera, **l'AI
che spiega se stessa**.

E lo fa **due volte**, in tutti e due i sensi della parola. Il primo lo trovi
in ogni capitolo, ed è quello di cui parlavo qui sopra: ogni concetto che
conta è spiegato prima con un'immagine di tutti i giorni e poi con le formule.

Il secondo senso riguarda la strada che il testo fa per arrivare fin qui.
Quello che una AI ha scritto lo rilegge **un'altra AI**, che alla stesura non
ha partecipato e ha un compito solo: cercare l'errore. Non riscrive per farlo
suonare meglio; rifà i conti, riapre i paper citati, ripercorre le derivazioni
un passaggio alla volta ed esegue il codice invece di guardarlo. Chi ha scritto
una cosa è l'ultimo a poterci trovare uno sbaglio, e questo vale per una
macchina esattamente come per una persona.

Solo dopo il testo arriva a me, e quello è il passaggio che decide che cosa
resta. Non è una formalità, ed è la parte che al posto mio una macchina non sa
fare: un modello punta ad avere ragione, un libro deve farsi capire, e non è
la stessa cosa. La prima si misura sui fatti, la seconda si misura su una
persona che prima non sapeva e adesso sa.

Così una spiegazione esatta ma fredda si riscrive finché non somiglia a come
la racconterei a voce; un esempio ineccepibile che però non fa scattare niente
si butta, anche se funziona; e una pagina che dice tutto il vero senza far
capire niente torna alla domanda da cui era nata, e si ricomincia da lì.

Il segno che vedi in copertina dice esattamente questo, e non per caso. È un
**triangolo di Penrose**: tre lati che si reggono l'un l'altro in cerchio, e
nessuno è il primo. Sono i tre passaggi appena descritti, chi scrive, chi cerca
l'errore e chi decide, con l'ultimo che rimanda al primo.

E poi c'è la seconda cosa che quel triangolo fa, che è la ragione per cui sta
su questo libro e non su un altro: **è una figura impossibile**. Copri con una
mano uno qualsiasi dei tre angoli e quello che resta è corretto, due travi che
si incontrano ad angolo retto, una davanti e una dietro; si costruisce davvero,
con dei pezzi di legno. Vale per tutti e tre, uno alla volta. L'impossibilità
non sta in nessun angolo: sta nel fatto che le tre soluzioni locali non si
possono avere tutte insieme, e per accorgersene bisogna smettere di guardare
l'angolo e seguire una trave per l'intero giro.

È il modo esatto in cui una macchina sbaglia su una materia tecnica. Ogni frase
regge da sola, ogni numero è verificabile, e il montaggio è falso.

Tutto questo è severo per una ragione precisa: un modello sbaglia con la
stessa sicurezza con cui dice il vero, e su una materia tecnica sbaglia proprio
dove chi legge non ha modo di accorgersene. Non sbaglia il tono. Perde il
fattore due davanti a una formula che per il resto è giusta; attribuisce al
paper sbagliato un meccanismo, magari a quello più famoso invece che a quello
che lo ha proposto; salta in una derivazione il passaggio da cui dipende tutto
e la fa comunque tornare; racconta benissimo un aneddoto sbagliando l'unico
dettaglio che ne faceva una lezione. Sono guasti che non si vedono rileggendo:
si vedono solo aprendo la fonte, rifacendo il conto a mano, mandando in
esecuzione il codice.

Su una cosa questo libro scommette apertamente. Il testo che leggi oggi è il
meglio che questo metodo sappia produrre adesso, non il meglio possibile:
qualche errore è rimasto, e più di una pagina si potrà spiegare meglio di
così. Ma un libro online non ha una tiratura da rincorrere con un foglietto di
errata corrige: si riscrive dove sbaglia, il giorno in cui qualcuno se ne
accorge. E gli strumenti con cui è scritto migliorano a loro volta, anno dopo
anno. Se mantengono quello che promettono, ogni versione dovrebbe arrivare un
po’ più completa e un po’ meno sbagliata della precedente, finché le
correzioni diventeranno rare e poi rarissime. È la scommessa che questo libro
fa su se stesso, e la si può verificare: {doc}`Aggiornamenti </aggiornamenti>`
tiene il conto di ogni correzione, una per una.

La responsabilità di quello che leggi è mia.

Perché è il non conoscere ciò che genera paura e alimenta false speranze.

% La firma e’ in HTML perche’ al sito serve l'allineamento a destra. In stampa
% il blocco raw sparisce, e la prefazione restava senza firma: si ripete per
% il solo LaTeX, com'e’ gia’ successo con l'attribuzione della citazione nella
% pagina di apertura.

```{raw} html
<p class="text-right mt-3">Napoli, 15 agosto 2026<br><em>Francesco Messina</em></p>
```

:::{only} latex
```{raw} latex
\vspace{4mm}\hfill Napoli, 15 agosto 2026\par
\vspace{1mm}\hfill\textit{Francesco Messina}\par
```
:::
