# Modelli a energia

L'8 ottobre 2024 l'Accademia reale svedese delle scienze annuncia il premio
Nobel per la fisica: va a John Hopfield e Geoffrey Hinton «per scoperte e
invenzioni fondamentali che rendono possibile l'apprendimento automatico con
reti neurali artificiali». La notizia lascia interdetti parecchi addetti ai
lavori (lo stesso Hinton, raggiunto al telefono in un albergo della
California, si dice sbalordito) e per giorni rimbalza la stessa domanda: che
cosa c'entra la *fisica*? Hopfield e Hinton non hanno scoperto particelle né
misurato onde gravitazionali: hanno costruito reti neurali.

La risposta della giuria è seria, ed è la porta d'ingresso di questo capitolo.
Le reti premiate non *assomigliano* a sistemi fisici: si comportano
esattamente come tali. Possiedono una grandezza chiamata **energia**, definita
con la stessa matematica dei materiali magnetici, e la loro dinamica la fa
scendere, come una pallina che rotola verso il fondo di una valle. In questo
quadro *ricordare* significa scivolare in un minimo di energia, e *imparare*
significa scolpire il paesaggio: scavare valli nei punti dove vogliamo che la
rete vada a finire.

Quarant'anni dopo quelle reti, il linguaggio dell'energia non è un pezzo da
museo. È la lingua in cui è scritta la proposta di Yann LeCun per l'AI che
verrà; è, sotto mentite spoglie, ciò che addestra i modelli di diffusione del
capitolo precedente; ed è il modo più economico che conosciamo per rispondere
a una domanda senza essere costretti a rispondere, insieme, a tutte le altre.

## Un numero al posto di una probabilità

```{figure} ../figures/oltre-il-gradiente.svg
:name: fig-paesaggio-energia
:alt: "Un paesaggio di energia con più valli di profondità diversa. Una pallina che segue soltanto la discesa resta intrappolata nella prima valle che incontra, poco profonda. Una traiettoria tratteggiata mostra invece un percorso che accetta di risalire ogni tanto e riesce così a raggiungere la valle più profonda."
:width: 92%

Il paesaggio che dà il nome al capitolo. Se la plausibilità è un'altitudine,
cercare la risposta migliore diventa un problema di esplorazione, e chi scende
soltanto si ferma presto.
```

Il percorso tratteggiato di {numref}`fig-paesaggio-energia` anticipa una
differenza di mentalità che attraversa tutto il capitolo. Dove i modelli visti
finora *ottimizzano*, questi **campionano**: accettano di peggiorare per un
tratto, perché è l'unico modo di uscire da una valle e vederne un'altra.

Un modello probabilistico, per dire quanto è verosimile una risposta, deve
tenere il conto di tutte le risposte possibili: le probabilità sommano a uno,
e quell'uno è un vincolo globale. Un modello a energia rinuncia al vincolo.
Assegna a ogni configurazione un numero (l'energia) e si limita a pretendere
che le configurazioni *plausibili* stiano in basso e le altre in alto. Nessuna
somma da chiudere, nessun totale da rispettare: solo un paesaggio.

`````{tab} Elementare

Immagina una carta geografica in rilievo, con valli e montagne. Ogni punto
della carta è una risposta possibile alla tua domanda: una faccia, una frase,
il fotogramma che verrà. L'altezza dice quanto quella risposta è insensata:
le risposte buone stanno nelle valli, quelle assurde in cima ai monti.
Rispondere significa lasciar rotolare una pallina e guardare dove si ferma;
imparare significa scavare il paesaggio finché le valli non stanno nei punti
giusti.

Il vantaggio si vede confrontandolo con l'altro modo di fare, quello delle
probabilità. Se ti chiedessi «quante probabilità ci sono che dietro l'angolo
ci sia un gatto?» e volessi una percentuale onesta, dovrei aver messo in
conto tutto quello che *non* è un gatto: cani, biciclette, cassonetti,
qualunque cosa esista. È il prezzo del cento per cento: per dire «70%» su una
cosa devi aver pesato tutte le altre. Se invece ti chiedo soltanto «gatto o
cassonetto, quale delle due torna di più?», ti basta confrontare due altezze
sulla carta. Il paesaggio non ti obbliga mai a fare il giro del mondo per
rispondere a una domanda locale.

`````

`````{tab} Superiore

Un modello a energia (*energy-based model*, EBM) è una funzione scalare
$E_\theta(x)$ (o $E_\theta(x, y)$ quando le variabili osservate $x$ e quelle
da predire $y$ vanno distinte) con parametri $\theta$: bassa dove i dati sono
plausibili, alta altrove. L'inferenza è un'ottimizzazione,

$$
\hat{y} = \arg\min_{y \in \mathcal{Y}} E_\theta(x, y),
$$

dove $\hat{y}$ è la risposta predetta e $\mathcal{Y}$ l'insieme delle
risposte ammissibili: nessuna somma su $\mathcal{Y}$, solo una ricerca del
minimo.

Il legame con la probabilità esiste, ed è la distribuzione di Gibbs–Boltzmann:

$$
p_\theta(x) = \frac{e^{-E_\theta(x)}}{Z(\theta)},
\qquad
Z(\theta) = \int e^{-E_\theta(x')}\, dx',
$$

dove $Z(\theta)$ è la **funzione di partizione**, l'integrale (o la somma, nel
caso discreto) su *tutto* lo spazio delle configurazioni. Ogni energia
definisce una densità, e ogni densità strettamente positiva si riscrive come
energia, $E_\theta(x) = -\log p_\theta(x) + \text{cost.}$: le due descrizioni
sono equivalenti *sulla carta*. Non lo sono nei conti. $Z(\theta)$ è il
termine che nessuno sa calcolare quando $x$ è un'immagine, e metà di questo
capitolo è dedicata a ciò che si può fare senza di lui, e all'osservazione,
tutt'altro che ovvia, che moltissimi compiti non ne hanno mai avuto bisogno.

`````

## Perché un capitolo a sé

Perché lo stesso oggetto continua a riaffiorare sotto nomi diversi, e finché
lo si incontra un pezzo per volta non lo si riconosce.

Lo *score* dei modelli di diffusione (il gradiente della log-densità
$\nabla_x \log p(x)$, che il capitolo precedente stima con una rete) è
esattamente $-\nabla_x E(x)$: generare per denoising progressivo *è* scendere
lungo un paesaggio di energia partendo dal rumore. Le architetture JEPA del
prossimo capitolo sono energie non normalizzate che giudicano la compatibilità
fra un pezzo di mondo osservato e uno da predire. Le reti di Hopfield
«moderne» hanno una regola di aggiornamento che coincide, formula alla mano,
con l'attenzione dei Transformer {cite}`ramsauer2021hopfield`. E il programma
che LeCun ripete da anni in fondo alle sue conferenze contiene, come seconda
delle quattro rinunce, «abbandonare il modello probabilistico in favore dei
modelli a energia» {cite}`lecun2022path`.

Quattro cose che sembravano quattro. Sono una sola, e questo capitolo la
guarda in faccia.

## Come è organizzato il capitolo

Cinque tappe, in salita dolce. Si comincia da dove l'idea è nata: la **memoria
associativa** di Hopfield {cite}`hopfield1982neural`, venticinque neuroni che
ricostruiscono un ricordo rovinato rotolando in fondo a una valle, con il
codice per vederlo accadere. Poi la **macchina di Boltzmann**
{cite}`ackley1985learning`, che aggiunge temperatura e neuroni nascosti,
trasforma il paesaggio in una distribuzione di probabilità e incontra per la
prima volta il muro: la funzione di partizione.

La terza tappa è quel muro. **Oltre la partizione** mette in fila le tre
strade che l'hanno aggirato (campionare, con MCMC e dinamica di Langevin;
stimare senza normalizzare, con lo *score matching*
{cite}`hyvarinen2005estimation`, la sua forma denoising
{cite}`vincent2011connection`, e la stima contrastiva col rumore
{cite}`gutmann2010noise`), e mostra come la seconda sia finita, un decennio
dopo, dentro i modelli di diffusione.

La quarta tappa è il gesto di LeCun: il tutorial del 2006
{cite}`lecun2006tutorial` che rilegge quasi ogni modello di apprendimento come
una funzione di **compatibilità** $E(x, y)$, con la sua promessa (niente $Z$)
e il suo pericolo (il collasso). L'ultima guarda al presente: gli EBM
addestrati con Langevin sulle immagini {cite}`du2019implicit`, il
classificatore che era un modello a energia senza saperlo
{cite}`grathwohl2020your`, e le quattro rinunce con cui LeCun chiude le sue
conferenze, discusse per quello che sono: un programma di ricerca, non un
verdetto.

```{admonition} Da ricordare
:class: important
- Un **modello a energia** assegna un numero a ogni configurazione (basso se
  plausibile, alto se no) e risponde cercando il minimo:
  $\hat{y} = \arg\min_y E_\theta(x, y)$. Niente probabilità da far sommare a
  uno.
- Energia e probabilità sono legate dalla distribuzione di Gibbs–Boltzmann,
  $p_\theta(x) = e^{-E_\theta(x)}/Z(\theta)$. Il ponte si paga con la
  **funzione di partizione** $Z(\theta)$, intrattabile in alta dimensione: è
  il personaggio contro cui si scontra metà del capitolo.
- Il premio **Nobel per la fisica 2024** a Hopfield e Hinton ha riportato
  alla luce un filone che non se n'era mai andato: lo *score* della
  diffusione è $-\nabla_x E$, la JEPA è un'energia non normalizzata, le
  Hopfield moderne sono l'attenzione dei Transformer.
- Nel resto del capitolo: memoria associativa, macchine di Boltzmann e
  contrastive divergence, i modi di aggirare $Z$, la cornice
  dell'*energy-based learning* e i modelli a energia di oggi.
```
