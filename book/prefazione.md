# Prefazione

Questo libro è stato scritto una prima volta nel 2019, per uscire su carta.
Aveva un indice di quattordici capitoli. Su carta non è mai uscito, e il
manoscritto è rimasto in un cassetto: la prima forma in cui arriva a qualcuno
è questa, online e gratuita.

Con il senno di poi è stata una fortuna. Quell'indice si fermava al deep
reinforcement learning e prometteva codice in TensorFlow e Keras
{cite}`abadi2016tensorflow,chollet2015keras`; la parola
*Transformer* non compariva da nessuna parte, benché l'articolo che la
introduce (*Attention Is All You Need*, di Vaswani e colleghi
{cite}`vaswani2017attention`) avesse ormai due anni, e il capitolo sul
linguaggio naturale finiva alle reti ricorrenti. Se questi nomi non ti dicono
niente, benissimo: sono esattamente le cose che imparerai. Era la faccia che
questa materia aveva, in quel momento, vista da chi ci lavorava. Un libro stampato in quella forma avrebbe cominciato a
invecchiare il giorno stesso in cui arrivava in libreria.

Online il problema si pone in un altro modo. Un errore si corregge il giorno
in cui qualcuno lo segnala, e il testo cresce quando cresce il campo: del
manoscritto è rimasta l'ossatura, e attorno sono nati i capitoli che allora
non si potevano scrivere. Il codice, nel frattempo, è passato a PyTorch
{cite}`paszke2019pytorch`. Quello
che non è cambiato è il motivo per cui il libro esiste.

Quello che è cambiato, e molto, è il mondo intorno. Nel 2019 questa era una
materia da addetti ai lavori. Oggi ci parla chiunque abbia un telefono, e attorno si sono mossi interessi
industriali e commerciali enormi. È guardando
quel passaggio che ho capito perché valeva la pena riprendere il manoscritto.

Nel 2021 un articolo molto discusso ha chiamato i modelli linguistici (i
programmi che stanno dietro a ChatGPT e alle sue varianti)
**pappagalli stocastici** {cite}`bender2021dangers`: macchine che rimettono
insieme pezzi di quello che hanno letto senza capire quello che dicono.
L'espressione ha fatto il giro del mondo. E a un certo punto mi sono accorto
che, a forza di ripeterla, i pappagalli eravamo diventati noi.

Succedeva in due modi, e si somigliavano. Usciva una notizia, e per settimane
la stessa frase rimbalzava identica sui canali di chi commenta questa materia
di mestiere, ripetuta da gente che alla fonte non era mai andata. Nelle aziende
capitava la versione tecnica: si scaricava un modello già pronto da Hugging
Face, il repository pubblico dove questi modelli si prendono e si condividono
{cite}`wolf2020transformers`, lo si
metteva in produzione perché lo stavano facendo tutti, e spesso nessuno apriva
la licenza per vedere se quell'uso commerciale fosse permesso. Il gesto era lo
stesso: ripetere una cosa senza sapere che cosa si stava dicendo.

La conferma è arrivata da chi l'espressione l'aveva coniata. Emily M. Bender,
prima firma di quell'articolo, ha
passato cinque anni a guardare che fine faceva la sua espressione: all'inizio,
racconta, la usava chi l'articolo l'aveva letto, e poi «la frase ha superato
l'articolo» {cite}`bender2026unasked`. È il pappagallo che si descrive da solo.

Intanto, di quei modelli, la cosa che si ripeteva più spesso è che si
limitano a indovinare una parola per volta. È vero, ma questo riguarda come il
testo viene generato, non come viene deciso. Anthropic, l'azienda che
costruisce una delle migliori famiglie di questi modelli, è andata a guardare
dentro uno dei suoi mentre scriveva una poesia in rima: la parola con cui
avrebbe chiuso il verso ce l'aveva in mente prima di cominciarlo, e il verso lo
costruiva per arrivarci {cite}`lindsey2025biology,anthropic2025tracing`.
Predire una parola alla volta non vuol dire ragionare una parola alla volta.

Su che cosa faccia una macchina la penso diversamente da Bender, e la
discussione è aperta. Su chi sia il pappagallo, no: il bersaglio della sua
critica, dice, non sono affatto i modelli, «quello che mi preoccupa sono le
azioni delle persone». E l'unica cosa che separa chi capisce da chi ripete è
sapere come funziona la cosa di cui si sta parlando: è quello che queste
pagine provano a dare.

Il metodo lo trovi descritto nella {doc}`pagina di apertura </intro>`: ogni
concetto che conta è spiegato due volte, una con un'analogia che non chiede
prerequisiti e una con le formule al posto giusto, e decidi tu da quale parte
stare. Il resto è presto detto: nessuna scorciatoia sulle cose difficili,
nessun entusiasmo che il testo non sia in grado di giustificare, e le fonti
sempre citate, così che tu possa andare ad approfondire.

Lo scopo di questo libro è introdurre i concetti principali dell'intelligenza
artificiale, dal machine learning al deep learning al reinforcement learning
(che cosa siano lo dice l’{doc}`introduzione </Introduzione/overview>`, una
definizione per ciascuno, prima di dedicarci il resto) fino ai capitoli che
questi sistemi li portano in produzione e a quelli che ne discutono le
conseguenze, e fornire gli strumenti utili a costruire un applicativo
intelligente,
a valutare un'idea, o semplicemente a riconoscere l'intelligenza artificiale
nelle tecnologie di oggi e di domani.

C'è poi una cosa da dire sul modo in cui è scritto, perché riguarda chi legge.
Buona parte di queste pagine nasce lavorando con l'intelligenza artificiale: è
lei a stendere, a cercare, a proporre. Questo libro è, alla lettera, l'AI
che spiega se stessa.

E lo fa due volte, in due sensi diversi. Il primo è quello dei due livelli di
lettura di cui parlavo qui sopra. Il secondo riguarda la strada che il testo fa
per arrivare fin qui:
quello che una AI ha scritto lo rilegge **un'altra AI**, che alla stesura non
ha partecipato e ha un compito solo, cercare l'errore. Invece di riscrivere per
farlo
suonare meglio, rifà i conti, riapre gli articoli citati, ripercorre le
derivazioni
un passaggio alla volta ed esegue il codice invece di guardarlo. Chi ha scritto
una cosa è l'ultimo a poterci trovare uno sbaglio, e questo vale per una
macchina esattamente come per una persona.

Solo dopo il testo arriva a me, e quello è il passaggio che decide che cosa
resta. È la parte che al posto mio una macchina non sa
fare: un modello punta ad avere ragione, un libro deve farsi capire. Avere
ragione si controlla riaprendo le fonti; farsi capire, soltanto leggendo la
pagina come la leggerebbe un altro lettore.

Così una spiegazione esatta ma fredda si riscrive finché non somiglia a come
la racconterei a voce; un esempio ineccepibile che però non fa scattare niente
si butta, anche se funziona; e una pagina che dice tutto il vero senza far
capire niente torna alla domanda da cui era nata, e si ricomincia da lì.

Il segno che il libro porta nel nome dice esattamente questo, e non per caso:
la «a» di *paithon* è un **triangolo di Penrose**, tre lati che si reggono l'un l'altro in cerchio, e
nessuno è il primo. Sono i tre passaggi appena descritti, chi scrive, chi cerca
l'errore e chi decide, con l'ultimo che rimanda al primo.

E poi c'è la seconda cosa che quel triangolo fa, che è la ragione per cui sta
su questo libro e non su un altro: è una figura impossibile.

```{figure} figures/triangolo-di-penrose.svg
:name: fig-triangolo-penrose
:alt: Il triangolo di Penrose, il triangolo impossibile: tre travi a sezione quadrata, una terracotta, una teal e una ocra, disposte in anello triangolare. Ogni trave passa davanti a quella che incontra e dietro a quella da cui arriva, e il giro si chiude soltanto in apparenza.
:width: 42%

Il segno del libro, ingrandito: gli stessi tre tracciati del logo.
```

Guarda uno qualsiasi dei tre angoli ({numref}`fig-triangolo-penrose`),
coprendo con una mano tutto il resto:
quello che resta scoperto è corretto, due travi che si incontrano ad angolo
retto, una davanti e una dietro; si costruisce davvero,
con dei pezzi di legno. Vale per tutti e tre, uno alla volta. L'impossibilità
non sta in nessuno dei tre, sta nel fatto che non si possono avere tutti
insieme, e per accorgersene bisogna smettere di guardare l'angolo e seguire
una trave per l'intero giro. Non è un inganno dell'occhio ma un teorema, che
Roger Penrose ha scritto in matematica trentatré anni dopo aver disegnato la
figura con suo padre {cite}`penrose1958impossible,penrose1991cohomology`.

È il modo esatto in cui una macchina sbaglia su una materia tecnica. Ogni frase
regge da sola, ogni numero è verificabile, e il montaggio è falso. Il
triangolo sta sul libro per tutte e due le ragioni insieme: è il giro dei tre
mestieri che si reggono a vicenda, ed è il guasto contro cui quel giro esiste.

Il metodo è severo per una ragione precisa: un modello sbaglia con la
stessa sicurezza con cui dice il vero, e su una materia tecnica sbaglia proprio
dove chi legge non ha modo di accorgersene. Il tono, quello, non lo sbaglia
mai. Dimentica un «per due» in una
formula che per il resto è giusta; attribuisce
un meccanismo all'articolo sbagliato, magari al più famoso invece che a quello
che lo ha proposto; salta in una derivazione il passaggio da cui dipende tutto
e la fa comunque tornare; racconta benissimo un aneddoto sbagliando l'unico
dettaglio che ne faceva una lezione. Sono guasti che non si vedono rileggendo:
si vedono solo aprendo la fonte, rifacendo il conto a mano, mandando in
esecuzione il codice.

% La chiusa di questo capoverso esiste in due versioni perche' la pagina degli
% aggiornamenti online c'e' e in stampa no: `FUORI_STAMPA` in `pt_stampa.py` la
% toglie dal PDF, dove la versione e' una sola ed e' quella del colophon. Senza
% lo sdoppiamento il {doc} resta un rimando cieco, e il PDF prometteva una
% pagina che il lettore non poteva trovare (il log lo diceva, in una riga sola:
% «Hyper reference `aggiornamenti::doc' on page 5 undefined»).

:::{only} html
Su una cosa questo libro scommette apertamente. Il testo che leggi oggi è il
meglio che questo metodo sappia produrre adesso, non il meglio possibile:
qualche errore è rimasto, e più di una pagina si potrà spiegare meglio di
così. Ma un libro online non ha migliaia di copie già stampate da rincorrere
con l’*errata corrige*: si riscrive dove sbaglia, il giorno in cui
qualcuno se ne
accorge. E gli strumenti con cui è scritto migliorano a loro volta, anno dopo
anno. Se mantengono quello che promettono, ogni versione dovrebbe arrivare un
po’ più completa e un po’ meno sbagliata della precedente, finché le
correzioni diventeranno rare e poi rarissime. È la scommessa che questo libro
fa su se stesso, e la si può verificare: {doc}`Aggiornamenti </aggiornamenti>`
tiene il conto delle correzioni, versione per versione.
:::

:::{only} latex
Su una cosa questo libro scommette apertamente. Il testo che leggi oggi è il
meglio che questo metodo sappia produrre adesso, non il meglio possibile:
qualche errore è rimasto, e più di una pagina si potrà spiegare meglio di
così. Ma un libro online non ha migliaia di copie già stampate da rincorrere
con l’*errata corrige*: si riscrive dove sbaglia, il giorno in cui
qualcuno se ne
accorge. E gli strumenti con cui è scritto migliorano a loro volta, anno dopo
anno. Se mantengono quello che promettono, ogni versione dovrebbe arrivare un
po’ più completa e un po’ meno sbagliata della precedente, finché le
correzioni diventeranno rare e poi rarissime. È la scommessa che questo libro
fa su se stesso, e la si può verificare: online, il registro degli
aggiornamenti tiene il conto delle correzioni, versione per versione.
:::

La responsabilità di quello che leggi è mia. L'impegno a renderlo ogni volta
più chiaro nasce da un fatto semplice.

È il non conoscere che genera paura e alimenta false speranze, e per
dissiparle non c'è altra strada che guardare dentro il cuore dell'intelligenza
algoritmica.

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
