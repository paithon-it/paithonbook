---
titolo_ricerca: "Prefazione: perché questo libro è online e gratuito"
descrizione: "Perché un manoscritto del 2019, mai uscito su carta, è diventato
  un libro online e gratuito sull'intelligenza artificiale, e che cosa
  troverai."
---
# Prefazione

Questo libro è stato scritto una prima volta nel 2019, per uscire su carta.
Aveva un indice di quattordici capitoli. Su carta non è mai uscito, e il
manoscritto è rimasto in un cassetto: la prima forma in cui arriva a qualcuno è
questa, *online* e *gratuita*.

Con il senno di poi è stata una fortuna. Quell'indice prometteva codice in
TensorFlow e Keras {cite}`abadi2016tensorflow,chollet2015keras` e il capitolo
sul linguaggio naturale finiva alle reti ricorrenti; la parola *Transformer*
non compariva da nessuna parte, benché l'articolo che la introduce (*Attention
Is All You Need*, di Vaswani e colleghi {cite}`vaswani2017attention`) avesse
due anni. Se questi nomi non ti dicono niente, benissimo: sono le cose che
imparerai. Un libro stampato così avrebbe cominciato a invecchiare il giorno in
cui arrivava in libreria; online il testo si aggiorna, e un errore si corregge
appena qualcuno lo segnala. Del manoscritto è rimasta l'ossatura, il codice è
passato a PyTorch {cite}`paszke2019pytorch`, e attorno sono nati i capitoli che
allora non si potevano scrivere.

Nel 2021 un articolo molto discusso ha chiamato i modelli linguistici (la
famiglia di programmi a cui appartiene ChatGPT) **pappagalli stocastici**
{cite}`bender2021dangers`: macchine che rimettono insieme pezzi dei testi che
hanno letto senza capire che cosa dicono. A un certo punto mi sono accorto che,
a forza di ripetere quella formula, i pappagalli eravamo diventati noi. Per
settimane, dopo ogni notizia, la stessa frase rimbalzava identica sui canali di
chi commenta questa materia di mestiere, ripetuta da chi non era andato alla
fonte, mosso dalla frenesia di attrarre pubblico grazie all’ultima notizia in
*hype*, che dopo tre ore sarebbe stata su ogni testata giornalistica e pagina
social. Intanto la componente teorica e tecnica del machine learning classico e
il lavoro del data scientist sono stati i primi a diventare obsoleti: prima di
ChatGPT chi curava i dati, chi addestrava i modelli e chi li portava in
produzione facevano mestieri distinti. Poi il mercato ne ha chiesto uno solo,
capace di prendere il modello aperto del momento da Hugging Face
{cite}`wolf2020transformers`, istruirlo con un *prompt* perfezionato sul caso
d'uso e rivenderlo come soluzione innovativa proprietaria (quante start-up ha
creato l’*in-context learning*?). Il gesto era lo stesso di chi ripeteva la
frase: usare una cosa senza sapere che cosa fosse e speculare su API a basso
costo di cui un giorno pagheremo il reale debito (tecnico più che economico).

Questo libro cerca di stare dall'altra parte: meno speculazione, più
architetture, iperparametri e funzioni di costo, con la speranza di rendere
accessibile la più grande rivoluzione democratica della storia dell’uomo. Dentro
ci sono i concetti principali dell'intelligenza artificiale, dal *machine
learning* al *deep learning* al *reinforcement learning* (li definisce
l’{doc}`introduzione </Introduzione/overview>`), e i capitoli che li portano in
produzione e ne discutono le conseguenze. Ogni concetto che conta è spiegato
*due volte*, una con un'analogia che non chiede prerequisiti e una con le
formule al posto giusto, e decidi tu da quale parte stare: la {doc}`pagina di
apertura </intro>` spiega come. Nessuna scorciatoia sulle cose difficili, nessun
entusiasmo che il testo non sia in grado di giustificare, e le fonti sempre
citate.

Buona parte di queste pagine nasce da una stretta collaborazione con
l'intelligenza artificiale stessa: sono io a delineare i temi, a dare la
direzione e a scegliere le fonti, e l'AI elabora la prima bozza. Questo libro
è, alla lettera, *l'AI che spiega se stessa*.

Poi un'altra AI rilegge la bozza. Non avendo partecipato alla stesura, svolge
un ruolo fondamentale: *cercare l'errore*. Rifà i conti, riapre gli articoli
citati, ripercorre le derivazioni ed esegue il codice invece di analizzarlo
solo staticamente. Chi ha scritto una cosa è l'ultimo a poterci trovare uno
sbaglio, e vale per una macchina come per una persona.

Solo dopo il testo arriva a me, e lì si decide che cosa resta: un modello punta
a raggiungere uno scopo tramite euristiche che non sempre aiutano la stesura,
un libro invece deve farsi leggere e comprendere.

Il segno che il libro porta nel nome dice qualcosa di simile: la «a» di
*paithon* è un **triangolo di Penrose**, tre lati che si reggono l'un l'altro e
nessuno è il primo, come i tre passaggi di scrittura del libro. È anche una
figura impossibile, e Douglas Hofstadter, in *Gödel, Escher, Bach*, ha
insegnato a vedere in figure come questa un anello che torna su sé stesso
{cite}`hofstadter1979godel`.

```{figure} figures/triangolo-di-penrose.svg
:name: fig-triangolo-penrose
:class: dark-light
:alt: Il triangolo di Penrose, il triangolo impossibile: tre travi a sezione quadrata, una terracotta, una teal e una ocra, disposte in anello triangolare. Ogni trave passa davanti a quella che incontra e dietro a quella da cui arriva, e il giro si chiude soltanto in apparenza.
:width: 30%

Il logo del progetto paithon.
```

Guarda uno qualsiasi dei tre angoli ({numref}`fig-triangolo-penrose`), coprendo
con una mano il resto: restano scoperte due travi che si incontrano ad angolo
retto, un incrocio che con dei pezzi di legno si costruisce davvero.
L'impossibilità sta nel non poterli avere tutti insieme, e per accorgersene
bisogna seguire una trave per l'intero giro
{cite}`penrose1958impossible,penrose1991cohomology`. È il modo in cui una
macchina sbaglia su una materia tecnica: ogni frase regge da sola, ogni numero
è verificabile, e la composizione è *errata* (almeno per un’intelligenza
naturale). Il triangolo sta sul libro per tutte e due le ragioni: è il giro dei
tre ruoli che si reggono a vicenda, e insieme l'errore che, revisione dopo
revisione, si cerca di minimizzare per rendere tutto più corretto.

Le illustrazioni che aprono i capitoli sono generate con il modello Recraft
V4.1, a partire da una descrizione della scena con cui ciascun capitolo
comincia; ogni pixel viene poi riportato sulle cinque tinte del libro.

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
migliorare, e che ogni versione arrivi con un po’ meno errori della
precedente. L'impegno che investirò in questo lavoro nasce da un fatto
semplice.
:::

:::{only} latex
La responsabilità di quello che leggi è mia. Qualche errore sarà rimasto, ma
un libro online non ha migliaia di copie stampate da rincorrere con l’*errata
corrige*: si riscrive dove sbaglia il giorno in cui qualcuno se ne accorge, e
online il registro degli aggiornamenti tiene il conto delle correzioni. È una
scommessa che il libro fa su se stesso: che questi sistemi continuino a
migliorare, e che ogni versione arrivi con un po’ meno errori della
precedente. L'impegno che investirò in questo lavoro nasce da un fatto
semplice.
:::

È il non conoscere che genera paura e alimenta false speranze, e per dissiparle
non c'è altra strada che guardare dentro il cuore dell'intelligenza
algoritmica.

% La firma e’ in HTML perche’ al sito serve l'allineamento a destra. In stampa
% il blocco raw sparisce, e la prefazione restava senza firma: si ripete per
% il solo LaTeX, com'e’ gia’ successo con l'attribuzione della citazione nella
% pagina di apertura. In stampa data e firma stanno su una riga
% sola: su due righe il nome scivolava da solo su una pagina nuova, e la
% prefazione passava da due pagine a quattro (la terza con il solo nome, la
% quarta bianca perche' il capitolo dopo apre a destra); concedendo alla pagina
% le due righe con `\enlargethispage` la firma finiva schiacciata sul piede.
% Con una riga sola basta allungare la pagina di una.

```{raw} html
<p class="text-right mt-3">Napoli, 15 agosto 2026<br><em>Francesco Messina</em></p>
```

:::{only} latex
```{raw} latex
\enlargethispage{\baselineskip}%
\vspace{3mm}\noindent Napoli, 15 agosto 2026\hfill\textit{Francesco Messina}\par
```
:::
