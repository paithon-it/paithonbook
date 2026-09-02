# Prefazione

Questo libro è stato scritto una prima volta nel 2019, per uscire su carta.
Aveva un indice di quattordici capitoli. Su carta non è mai uscito, e il
manoscritto è rimasto in un cassetto: la prima forma in cui arriva a qualcuno
è questa, online e gratuita.

Con il senno di poi è stata una fortuna. Quell'indice prometteva codice in
TensorFlow e Keras {cite}`abadi2016tensorflow,chollet2015keras` e il capitolo
sul linguaggio naturale finiva alle reti ricorrenti; la parola *Transformer*
non compariva da nessuna parte, benché l'articolo che la introduce (*Attention
Is All You Need*, di Vaswani e colleghi {cite}`vaswani2017attention`) avesse
due anni. Se questi nomi non ti dicono niente, benissimo: sono esattamente le
cose che imparerai. Un libro stampato così avrebbe cominciato a invecchiare il
giorno in cui arrivava in libreria; online un errore si corregge appena
qualcuno lo segnala. Del manoscritto è rimasta l'ossatura, il codice è passato
a PyTorch {cite}`paszke2019pytorch`, e attorno sono nati i capitoli che allora
non si potevano scrivere.

Nel 2021 un articolo molto discusso ha chiamato i modelli linguistici (i
programmi dietro a ChatGPT e alle sue varianti) **pappagalli stocastici**
{cite}`bender2021dangers`: macchine che rimettono insieme pezzi di quello che
hanno letto senza capire quello che dicono. A un certo punto mi sono accorto
che, a forza di ripetere quella formula, i pappagalli eravamo diventati noi.
Per settimane, dopo ogni notizia, la stessa frase rimbalzava identica sui
canali di chi commenta questa materia di mestiere, ripetuta da chi alla fonte
non era andato. Intanto cambiava il lavoro: prima di ChatGPT chi curava i
dati, chi addestrava i modelli e chi li portava in produzione facevano
mestieri distinti, poi il mercato ne ha chiesto uno solo, capace di prendere
il modello aperto del momento da Hugging Face {cite}`wolf2020transformers`,
istruirlo con un prompt perfezionato sul caso d'uso e rivenderlo come
soluzione propria. Il gesto era lo stesso di chi ripeteva la frase: usare una
cosa senza sapere che cosa fosse.

Emily M. Bender, prima firma di quell'articolo, racconta che all'inizio la sua
espressione la usava chi l'articolo l'aveva letto, e che poi «la frase ha
superato l'articolo» {cite}`bender2026unasked`. È il pappagallo che si
descrive da solo. Su che cosa faccia una macchina la penso diversamente da
lei; su chi sia il pappagallo, no: il bersaglio della sua critica, dice, non
sono affatto i modelli, «quello che mi preoccupa sono le azioni delle
persone». E l'unica cosa che separa chi capisce da chi ripete è sapere come
funziona ciò di cui si parla.

Questo libro sta dall'altra parte: meno caccia all'istruzione magica, più
architetture, iperparametri e funzioni di costo. Dentro ci sono i concetti
principali dell'intelligenza artificiale, dal machine learning al deep
learning al reinforcement learning (li definisce l’{doc}`introduzione
</Introduzione/overview>`), e i capitoli che li portano in produzione e ne
discutono le conseguenze. Ogni concetto che conta è spiegato due volte, una
con un'analogia che non chiede prerequisiti e una con le formule al posto
giusto, e decidi tu da quale parte stare: come funziona lo dice la
{doc}`pagina di apertura </intro>`. Nessuna scorciatoia sulle cose difficili,
nessun entusiasmo che il testo non sia in grado di giustificare, e le fonti
sempre citate.

C'è poi da dire come è scritto, perché interessa a chi legge. Buona parte di
queste pagine nasce da una stretta collaborazione con l'intelligenza
artificiale: sono io a delineare i temi, a dare la direzione e a scegliere le
fonti, e l'AI elabora la prima bozza. Questo libro è, alla lettera, l'AI che
spiega se stessa.

Quello che l'AI ha scritto lo rilegge **un'altra AI**, che alla stesura non ha
partecipato e ha un compito solo, cercare l'errore: rifà i conti, riapre gli
articoli citati, ripercorre le derivazioni ed esegue il codice invece di
guardarlo. Chi ha scritto una cosa è l'ultimo a poterci trovare uno sbaglio, e
vale per una macchina come per una persona.

Solo dopo il testo arriva a me, ed è il passaggio che decide che cosa resta:
un modello punta ad avere ragione, un libro deve farsi capire. Avere ragione
si controlla riaprendo le fonti; farsi capire, soltanto leggendo la pagina
come la leggerebbe chiunque altro.

Il segno che il libro porta nel nome dice esattamente questo: la «a» di
*paithon* è un **triangolo di Penrose**, tre lati che si reggono l'un l'altro
e nessuno è il primo, come i tre passaggi appena descritti. Ed è anche una
figura impossibile, e a leggerne una come un anello che torna su sé stesso ha
insegnato Douglas Hofstadter in *Gödel, Escher, Bach*
{cite}`hofstadter1979godel`.

```{figure} figures/triangolo-di-penrose.svg
:name: fig-triangolo-penrose
:class: dark-light
:alt: Il triangolo di Penrose, il triangolo impossibile: tre travi a sezione quadrata, una terracotta, una teal e una ocra, disposte in anello triangolare. Ogni trave passa davanti a quella che incontra e dietro a quella da cui arriva, e il giro si chiude soltanto in apparenza.
:width: 30%

Il logo del progetto paithon.
```

Guarda uno qualsiasi dei tre angoli ({numref}`fig-triangolo-penrose`),
coprendo con una mano il resto: quello che resta scoperto sono due travi che
si incontrano ad angolo retto, e con dei pezzi di legno si costruisce davvero.
L'impossibilità sta nel non poterli avere tutti insieme, e per accorgersene
bisogna seguire una trave per l'intero giro
{cite}`penrose1958impossible,penrose1991cohomology`. È il modo esatto in cui
una macchina sbaglia su una materia tecnica: ogni frase regge da sola, ogni
numero è verificabile, e il montaggio è falso. Il triangolo sta sul libro per
tutte e due le ragioni: è il giro dei tre ruoli che si reggono a vicenda, ed è
il guasto contro cui quel lavoro esiste.

% La chiusa di questo capoverso esiste in due versioni perche' la pagina degli
% aggiornamenti online c'e' e in stampa no: `FUORI_STAMPA` in `pt_stampa.py` la
% toglie dal PDF, dove la versione e' una sola ed e' quella del colophon. Senza
% lo sdoppiamento il {doc} resta un rimando cieco, e il PDF prometteva una
% pagina che il lettore non poteva trovare (il log lo diceva, in una riga sola:
% «Hyper reference `aggiornamenti::doc' on page 5 undefined»).

:::{only} html
La responsabilità di quello che leggi è mia. Qualche errore sarà rimasto, ma
un libro online non ha migliaia di copie stampate da rincorrere con l’*errata
corrige*: si riscrive dove sbaglia il giorno in cui qualcuno se ne accorge, e
{doc}`Aggiornamenti </aggiornamenti>` tiene il conto delle correzioni. È una
scommessa che il libro fa su se stesso: che questi sistemi continuino a
migliorare, e che ogni versione arrivi un po’ meno sbagliata della precedente.
L'impegno a renderlo ogni volta più chiaro nasce da un fatto semplice.
:::

:::{only} latex
La responsabilità di quello che leggi è mia. Qualche errore sarà rimasto, ma
un libro online non ha migliaia di copie stampate da rincorrere con l’*errata
corrige*: si riscrive dove sbaglia il giorno in cui qualcuno se ne accorge, e
online il registro degli aggiornamenti tiene il conto delle correzioni. È una
scommessa che il libro fa su se stesso: che questi sistemi continuino a
migliorare, e che ogni versione arrivi un po’ meno sbagliata della precedente.
L'impegno a renderlo ogni volta più chiaro nasce da un fatto semplice.
:::

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
