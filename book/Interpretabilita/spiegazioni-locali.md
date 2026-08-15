# Spiegazioni locali: LIME, SHAP e controfattuali

Nel 1953 **Lloyd Shapley**, un giovane matematico che a Princeton stava
finendo il dottorato, si pose una domanda che con l'intelligenza artificiale
non c'entrava nulla: se un gruppo di persone collabora a un'impresa e ne ricava
un guadagno, come si divide il merito «in modo equo» tra chi ha partecipato?
Alcuni contano di più, alcuni solo in combinazione con altri; non basta guardare
cosa fa ciascuno da solo. Shapley rispose fissando quattro requisiti così
ragionevoli che nessuno li discuterebbe, e dimostrando che un solo modo di
dividere li rispetta tutti e quattro. Nel 2012 avrebbe vinto il premio Nobel per
l'economia, per un altro filone dei suoi studi. Non poteva immaginare che questa
sua formula sarebbe diventata, più di sessant'anni dopo, uno degli strumenti
più usati per spiegare la singola decisione di una rete neurale.

Questa è la seconda tappa del capitolo, e cambia scala. La sezione precedente
rispondeva a una domanda **globale**: «su quali colonne dei dati (le
**feature**) si regge il modello, *in media*?». Ma chi si vede negare un
prestito non chiede una media: chiede «perché *la mia* domanda?». È una domanda
**locale**, perché riguarda una risposta sola e non l'intero comportamento del
modello. Qui vediamo i modi di risponderle: i tre attrezzi che si incontrano
più spesso, **LIME**, **SHAP** e le spiegazioni **controfattuali**, e poi altre
due forme di risposta, la regola e ciò che manca, che rispondono alla stessa
domanda in un modo abbastanza diverso da meritarsi una sezione.

## Perché proprio *questa* risposta

`````{tab} Elementare

Torniamo al modello che decide i prestiti, e immaginiamo che a Maria l'abbia
rifiutato. Sapere che «in generale il reddito conta molto» non le serve: è una
statistica sul modello, non una risposta sul suo caso. Maria vuole sapere
cosa, *nella sua pratica*, ha spinto verso il no (il reddito, i debiti in
corso, un ritardo di pagamento di due anni fa?) e magari quanto ciascuna cosa
ha pesato.

È la differenza tra chiedere «come giudica in media questo professore» e
«perché ha dato a *me* questo voto». La prima è una spiegazione **globale**,
sul modello intero; la seconda è **locale**, su una singola risposta. Un
modello può essere troppo intricato per capirlo tutto in un colpo, eppure
restare semplice *attorno a un punto*: come una strada di montagna piena di
curve che, guardata da vicino su pochi metri, sembra dritta. È su questa idea
(«complicato ovunque, semplice qui accanto») che si reggono i metodi di questa
sezione.

`````

`````{tab} Superiore

Formalmente, una spiegazione locale riguarda il valore del modello $f$ in un
intorno del punto $\mathbf{x}_0$, non la funzione $f$ sull'intero dominio. La
motivazione è geometrica: la superficie decisionale di un modello complesso
può essere globalmente inintelligibile ma **localmente regolare**, cioè bene
approssimabile da un modello semplice in un intorno abbastanza piccolo di
$\mathbf{x}_0$ (l'analogo di linearizzare una funzione differenziabile con il
suo piano tangente). Ne discende il criterio di qualità già introdotto in
apertura di capitolo, la **fedeltà locale**: una spiegazione è buona se il
surrogato interpretabile concorda con $f$ sui punti campionati vicino a
$\mathbf{x}_0$, senza alcuna pretesa di valere lontano da lì. Cambia anche
l'oggetto restituito: non più una classifica di feature valida ovunque, ma un
vettore di **attribuzioni** $\phi_j$ specifiche di $\mathbf{x}_0$, una per
feature, che dicono quanto ciascuna ha spinto *questa* predizione sopra o sotto
un riferimento.

`````

## LIME: un modello semplice, ritagliato attorno al punto

Il primo attrezzo è **LIME** (*Local Interpretable Model-agnostic
Explanations*) proposto nel 2016 da Marco Túlio Ribeiro, Sameer Singh e Carlos
Guestrin {cite}`ribeiro2016why`, gli stessi dell'esperimento con cui si è aperto
il capitolo, quello del riconoscitore di lupi che in realtà guardava la neve.
L'idea è disarmante nella sua semplicità: se il modello è troppo complicato da
capire tutto, costruiamone una copia *facile* che gli somigli **solo qui
vicino**, attorno al caso da spiegare. Del modello vero non apriremo niente: gli
faremo solo delle domande, come si fa con una scatola chiusa.

`````{tab} Elementare

Immagina di voler capire come il modello decide sul caso di Maria. LIME fa
così: fabbrica tante pratiche finte, **casi-fantasma** con dentro numeri
plausibili (redditi, debiti, età presi da altri clienti o tirati a sorte), e per
ciascuna chiede al modello cosa risponderebbe. Ottiene una nuvola di esempi
inventati, ognuno con la risposta che il modello gli darebbe.

A questo punto, su quella nuvola, costruisce un modello **semplice e
leggibile**, del tipo che abbiamo visto nella sezione precedente: quello che
risponde facendo una somma, tanti punti per il reddito, tanti per i debiti,
tanti per l'anzianità di lavoro. Cerca cioè i numeri che fanno somigliare la
somma alle risposte della nuvola.

C'è però una regola in più, ed è quella che fa tutto il lavoro: nel conto, i
fantasmi più simili alla pratica vera di Maria **contano di più**, quelli
lontani quasi niente. Un conto in cui ogni voce entra con un'importanza sua si
dice **pesato**, e nel caso di LIME il peso è la somiglianza con Maria. Qui sta il «solo qui vicino», e vale la pena sottolineare **dove** sta: nel
conteggio, non nella fabbrica. I fantasmi nascono sparsi un po' dappertutto, ed
è soltanto quando si tirano le somme che quelli lontani vengono messi a tacere.

I numeri di quella somma *sono* la spiegazione: «il reddito basso ha spinto
verso il no di tanto, i debiti di tanto, l'anzianità di lavoro ha spinto un po'
verso il sì». Un ultimo modo di vederla: se il modello vero è una strada di
montagna, LIME appoggia un righello sull'asfalto nel punto in cui ti trovi. Il
righello non racconta la strada, ma della salita sotto i tuoi piedi ti dice
tutto.

Un avvertimento, però, prima di fidarsene troppo. Di scelte, in tutto questo, ce
ne sono parecchie, e nessuna la suggeriscono i dati: quanti fantasmi fabbricare,
quanto in fretta il loro peso deve calare allontanandosi da Maria (cioè quanto
largo prendere il «qui vicino»), e, se al posto della pratica di Maria c'è una
fotografia, in quali pezzi spezzettarla prima di spegnerli a turno. Ognuna di
quelle scelte sposta i numeri della spiegazione. In più i fantasmi sono
fabbricati a caso, quindi rilanciando LIME sullo stesso identico caso si
ottengono numeri un po' diversi: si dice che il metodo è **instabile**.

`````

`````{tab} Superiore

LIME cerca un surrogato $g$ in una classe interpretabile $G$ (tipicamente
modelli lineari sparsi) che minimizzi

$$
\xi(\mathbf{x}_0) = \operatorname*{arg\,min}_{g \in G}\;
   \mathcal{L}\big(f, g, \pi_{\mathbf{x}_0}\big) + \Omega(g),
$$

dove $\mathcal{L}$ è l'infedeltà di $g$ rispetto a $f$ **pesata dalla
prossimità** $\pi_{\mathbf{x}_0}$, e $\Omega(g)$ penalizza la complessità di
$g$ (per esempio il numero di feature non nulle, per una spiegazione corta).

Un passaggio del paper va reso esplicito, perché decide tutto il resto: il
surrogato $g$ **non vive nello spazio dell'input**. Vive in una
**rappresentazione interpretabile** binaria $\{0,1\}^{d'}$, di presenza o
assenza di componenti: segmenti contigui di pixel (*superpixel*) per le
immagini, parole per il testo, intervalli di valore per le colonne numeriche. Le
perturbazioni si ottengono **spegnendo a caso** alcune di quelle componenti,
non muovendo i valori originali, ed è ciò che rende LIME applicabile a
un'immagine: nell'esperimento degli husky la macchia di neve che tutti hanno
visto è un gruppo di superpixel acceso, non un insieme di pixel scelti uno per
uno.

In pratica si campionano dei punti $\mathbf{z}$, si valuta $f(\mathbf{z})$ (la
sola cosa che serve del modello: LIME è **model-agnostic**), si pesa ciascun
campione con un kernel esponenziale
$\pi_{\mathbf{x}_0}(\mathbf{z}) =
\exp\!\big(-d(\mathbf{x}_0, \mathbf{z})^2 / \sigma^2\big)$
che decade con la distanza $d$, e si adatta una regressione lineare pesata. I
coefficienti appresi sono le attribuzioni. Conviene guardare da vicino **dove
entra la località**, perché non è dove ci si aspetta: nell'implementazione
tabellare degli autori i campioni non si generano attorno a $\mathbf{x}_0$, ma
dalla distribuzione marginale del training set (l'opzione
`sample_around_instance` è falsa per default, e le colonne continue vengono per
giunta discretizzate in quartili prima di essere perturbate). La vicinanza a
$\mathbf{x}_0$ la introduce **solo** il peso $\pi_{\mathbf{x}_0}$, a valle. La
conseguenza è concreta: la fedeltà locale dipende da quanti dei punti generati
globalmente cadono davvero vicino a $\mathbf{x}_0$, e in alta dimensione sono
pochi.

Tre limiti vanno dichiarati. Primo, l'**instabilità**: campionamento casuale e
scelta del kernel rendono la spiegazione sensibile ai dettagli; rieseguire
LIME sullo stesso punto può dare coefficienti diversi, ed è in buona parte una
conseguenza del punto appena visto. Secondo, la **larghezza
del vicinato** $\sigma$ è un iperparametro senza una regola universale: un
intorno troppo ampio linearizza una zona in cui $f$ non è affatto lineare
(bassa fedeltà), uno troppo stretto lascia troppo pochi campioni informativi.
Terzo, e più insidioso perché non si presenta nemmeno come un parametro da
tarare: la spiegazione dipende da **come si è deciso di segmentare l'input**,
cioè da quali sono le componenti che si accendono e si spengono. Cambia la
segmentazione, cambia la spiegazione, e la segmentazione la sceglie chi usa lo
strumento. La spiegazione dipende quindi da scelte che l'utente raramente
controlla: un motivo per affiancarle un metodo dai fondamenti più solidi.

`````

## I valori di Shapley: dividere il merito in modo equo

LIME risponde alla domanda, ma il modo in cui lo fa poggia su una lunga catena
di scelte fatte da chi lo usa: quanti casi-fantasma fabbricare, quanto contare
quelli lontani, come spezzettare l'ingresso. Rilanciandolo si ottengono numeri
un po' diversi. Viene voglia di un metodo che, data la domanda, abbia una
risposta sola e dimostrabile. Quel metodo esiste, ed è la formula di Shapley del
1953. Il salto sta nel
guardare una risposta del modello come il bottino di una squadra. Le colonne
sono i giocatori; il bottino da spartire non è il punteggio che il modello ha
dato a Maria, ma **di quanto quel punteggio si scosta** dalla risposta che il
modello darebbe senza sapere niente di lei. Quella risposta a vuoto la
chiameremo, qui e per tutto il resto del capitolo, la **risposta base**. La
domanda «quanto ha contribuito il reddito a questo rifiuto?» diventa così la
vecchia domanda di Shapley: «quanto merito spetta a questo giocatore?».

`````{tab} Elementare

Il punto delicato è che i giocatori non agiscono da soli: contano anche le
**combinazioni**. Prendiamo due colonne sole, per tenere il conto piccolo: il
*reddito* e i *pagamenti passati*, cioè se in passato il cliente ha pagato
puntuale. Chiediamoci quanto vale ciascuna. L'idea di Shapley è: proviamo tutti
gli ordini in cui le colonne possono «entrare in campo» e, per ognuna,
misuriamo quanto aggiunge nel momento in cui entra. Poi facciamo la media.

Prima però va sciolto un dubbio ragionevole: che cosa vuol dire far girare il
modello «senza» una colonna? Il modello vuole tutte le sue caselle piene, non
gli si può lasciare una casella vuota. Quello che si fa è riempirla con
qualcosa di neutro: il valore medio di quella colonna su tutti i clienti, o un
valore preso a caso da un altro cliente, oppure uno zero, messo lì per
convenzione a significare «niente». È una **scelta**, non un fatto, e sposta i numeri:
cambiando ciò che si mette al posto del valore vero cambia la risposta del
modello quando non sa niente, e siccome tutti i meriti sono la ripartizione
della distanza *da quella risposta lì*, cambiano anche loro. Per ora basta
tenere presente che «il modello non sa il reddito» significa «al posto del
reddito c'è qualcosa che non dice niente».

Diciamo allora che il modello dà a ogni cliente un punteggio da 0 a 100, e che
sopra i 35 il prestito è approvato. Senza sapere niente del nostro cliente
parte dalla **risposta base**, che vale **10**. Se conosce solo il reddito,
sale a **30**; se conosce solo i pagamenti passati, sale a **20**; se conosce
entrambe le cose, arriva a **50**, cioè al sì.

Prima di fare il conto, una cosa da notare, perché è il motivo per cui
l'esempio è costruito così. Guardiamo quanto aggiunge ciascuna colonna da sola:
il reddito porta da 10 a 30, cioè aggiunge **20**; i pagamenti portano da 10 a
20, cioè aggiungono **10**. Sommati fanno 30, e partendo da 10 dovremmo arrivare
a 40. Invece si arriva a 50. Ci sono dieci punti in più che non appartengono a
nessuna delle due colonne: nascono dall'averle tutte e due, ed è ragionevole che
sia così, perché guadagnare tanto conta poco se non hai mai dimostrato di saper
pagare, e aver sempre pagato conta poco se guadagni una miseria. Le due cose si
rinforzano a vicenda, e un guadagno che nasce così si chiama **interazione**. È
il caso in cui la domanda «quanto vale ciascuna?» smette di avere una risposta
ovvia, e tornerà in fondo alla sezione.

Con due colonne gli ordini possibili sono due, prima il reddito, oppure prima i
pagamenti:

- *Prima il reddito*: entra e porta il punteggio da 10 a 30, quindi aggiunge
  **20**. Poi entrano i pagamenti e lo portano da 30 a 50, aggiungono **20**.
- *Prima i pagamenti*: entrano e portano da 10 a 20, aggiungono **10**. Poi
  entra il reddito e porta da 20 a 50, aggiunge **30**.

Il merito del reddito è la media di quanto aggiunge nei due ordini:
$(20 + 30)/2 = 25$. Quello dei pagamenti: $(20 + 10)/2 = 15$. E il conto torna:
$25 + 15 = 40$, esattamente quanto separa il punteggio finale (50) dalla
risposta base (10). Tutto il bottino è stato ripartito, senza avanzi: nessun
merito inventato, nessuno perso per strada.

E i dieci punti dell'interazione? Non sono spariti, sono stati divisi a metà, e
il modo in cui è successo si legge nei due ordini qui sopra. Chi entra per
**secondo** se li prende tutti e dieci: nel primo ordine sono i pagamenti (che
da soli valevano 10 e lì aggiungono 20), nel secondo è il reddito (che da solo
valeva 20 e lì aggiunge 30). Siccome ciascuno dei due arriva secondo in metà
degli ordini, ciascuno intasca quei dieci una volta su due: cinque a testa.
Infatti il reddito prende 25, che è i suoi 20 più cinque, e i pagamenti 15, che
sono i loro 10 più cinque.

Perché fare la **media** su tutti gli ordini, e non prendere il primo che
capita? Perché nessun ordine è quello vero. Le colonne non entrano davvero una
dopo l'altra, ci sono tutte insieme; gli ordini sono un espediente per misurare
i contributi, e siccome non c'è ragione di preferirne uno, si tengono tutti e si
fa la media. È esattamente l'idea di equità di Shapley.

Le quattro proprietà con cui Shapley aveva fissato il suo modo di dividere sono
cose ovvie come quella che abbiamo appena visto tornare, e conviene chiamarle
per nome perché il resto del capitolo le usa.

- Che il conto torni senza avanzi, come qui $25 + 15 = 50 - 10$, si chiama
  **efficienza**.
- Che due colonne che fanno esattamente lo stesso mestiere ricevano lo stesso
  merito si chiama **simmetria**. È il principio dietro la divisione in due dei
  dieci punti: rispetto a quel pezzo di guadagno, e solo rispetto a quello, le
  due colonne facevano lo stesso lavoro. Sull'intero conto no, tant'è che
  prendono 25 e 15.
- Che una colonna che non aggiunge mai niente, in nessun ordine, prenda zero si
  chiama **giocatore nullo**. Sembra una banalità e invece torna comoda: quando
  in un conto c'è una colonna che non c'entra, la si può togliere di mezzo e
  fare i conti sulle altre.
- E poi c'è l'**additività**, che è la meno intuitiva delle quattro e la più
  utile, quindi vale un esempio. Immagina che il punteggio di un cliente sia la
  somma di due pagelle separate, una sulla sua situazione economica e una sulla
  sua storia di pagamenti, ciascuna con le sue regole. L'additività dice questo:
  il merito che una colonna prende sul punteggio totale è la somma dei meriti
  che prende su ciascuna delle due pagelle, calcolati separatamente. In altre
  parole, un conto complicato lo si può spezzare in pezzi, fare i conti sui
  pezzi e sommare. Alla fine della sezione lo faremo davvero, e sarà quello a
  rendere leggibile un risultato altrimenti misterioso.

Sono quattro richieste che nessuno discuterebbe, e la cosa notevole (il motivo
per cui questa formula del 1953 è ancora qui) è che c'è un solo modo di dividere
il merito che le soddisfa tutte e quattro insieme.

`````

`````{tab} Superiore

Sia $N = \{1, \dots, n\}$ l'insieme delle feature e $v(S)$ il valore della
**coalizione** $S \subseteq N$: la predizione attesa quando si conoscono solo le
feature in $S$ e si marginalizza sulle altre, con $v(\varnothing) = \mathbb{E}[f]$
il valore base e $v(N) = f(\mathbf{x}_0)$. Il valore di Shapley della feature $i$ è la
media dei suoi **contributi marginali** su tutti gli ordini di ingresso:

$$
\phi_i = \sum_{S \subseteq N \setminus \{i\}}
   \frac{|S|!\,\big(n - |S| - 1\big)!}{n!}\,
   \big[\, v(S \cup \{i\}) - v(S) \,\big],
$$

dove $v(S \cup \{i\}) - v(S)$ è quanto aggiunge $i$ unendosi alla coalizione
$S$, e il coefficiente combinatorio conta la frazione di permutazioni in cui $i$
entra proprio dopo l'insieme $S$. Questa è l'**unica** attribuzione che
soddisfa quattro assiomi:

- **Efficienza**: $\sum_{i} \phi_i = v(N) - v(\varnothing) = f(\mathbf{x}_0) - \mathbb{E}[f]$.
  I contributi sommano esattamente allo scarto della predizione dal valore base:
  niente si crea, niente si perde.
- **Simmetria**: se due feature danno lo stesso contributo a ogni coalizione
  ($v(S \cup \{i\}) = v(S \cup \{j\})$ per ogni $S$), allora $\phi_i = \phi_j$.
- **Giocatore nullo** (*dummy*): una feature che non cambia mai il valore
  ($v(S \cup \{i\}) = v(S)$ per ogni $S$) riceve $\phi_i = 0$.
- **Additività**: i valori di Shapley di una somma di modelli sono la somma
  dei valori; è la proprietà che rende trattabili gli ensemble.

Un'avvertenza che la letteratura ha imparato a proprie spese: la funzione $v$
non è data, va **scelta**, e la scelta conta. Le feature fuori da $S$ si
possono marginalizzare sulla loro distribuzione (variante *interventista*),
condizionare a quelle presenti (variante *condizionale*) o fissare a un
riferimento $r$ (variante *baseline*, dove $v(\varnothing) = f(r)$): le tre
scelte producono attribuzioni diverse per lo stesso modello e lo stesso punto
{cite}`sundararajan2020many`. Il teorema di unicità riguarda il **modo di
ripartire**, dato il gioco; cambiare $v$ vuol dire cambiare gioco, e nessun
assioma dice quale dei tre sia quello da giocare. Non è un difetto della
formula, è di nuovo la forcella dell'apertura
di capitolo: la variante interventista risponde a «su che cosa si appoggia
*questo modello*», quella condizionale a «che cosa dice *il dato*», e sono due
domande diverse a cui è giusto rispondere con due numeri diversi.

Sull'esempio dei due giocatori, con $v(\varnothing)=10$, $v(\{1\})=30$,
$v(\{2\})=20$, $v(\{1,2\})=50$, la formula dà $\phi_1 = 25$ e $\phi_2 = 15$, in
accordo con la media sugli ordini, e $\phi_1 + \phi_2 = 40 = v(N) - v(\varnothing)$
verifica l'efficienza. Il costo è la maledizione combinatoria: la somma per una
singola feature è su $2^{\,n-1}$ coalizioni (i sottoinsiemi che non contengono
$i$), e le coalizioni distinte in tutto sono $2^n$: è quest'ultimo il miliardo
abbondante che si cita per $n = 30$. In entrambi i conteggi si è impraticabili
oltre poche decine di feature. È il problema che SHAP risolve.

`````

## SHAP: i valori di Shapley, resi praticabili

Calcolare i valori di Shapley esatti richiede di provare tutte le
**coalizioni**, cioè tutti i gruppi di feature che si possono formare. Ogni
colonna può esserci o non esserci, quindi i gruppi si contano moltiplicando due
per sé stesso una volta per colonna: con trenta colonne fa oltre un miliardo di
gruppi. Nel 2017 Scott Lundberg e Su-In Lee
{cite}`lundberg2017unified` hanno mostrato come stimarli in modo efficiente,
unificando sotto un'unica teoria (**SHAP**, *SHapley Additive exPlanations*)
metodi fino ad allora scollegati. Anche LIME rientra in quella teoria, come caso
particolare: ha la stessa forma, e a distinguerlo è **come conta la
somiglianza** fra i casi-fantasma, che LIME fissa a occhio e SHAP invece deriva
dalla formula di Shapley. Da qui SHAP è diventato uno degli strumenti più
usati per spiegare una decisione dall'esterno, a modello già addestrato.

`````{tab} Elementare

SHAP non cambia la definizione: restituisce ancora i contributi «equi» di
Shapley. Cambia il *come* li ottiene, con due scorciatoie a seconda del modello.

Se il modello è una scatola chiusa qualunque, si usa **KernelSHAP**, che prova
solo alcuni dei gruppi invece di tutti, scelti a caso, e dai pochi provati
ricostruisce i meriti di tutti. Il ricostruire è di nuovo un conto pesato, come
quello di LIME; la differenza è che qui i pesi non sono scelti a occhio, li dà
la formula di Shapley, ed è questo che garantisce di stare puntando ai numeri
giusti.

Se invece il modello è fatto di alberi di decisione, quelli a catena di domande
sì/no della sezione precedente, si usa **TreeSHAP**, che sfrutta la forma
dell'albero per calcolare i meriti **esatti** senza provare niente a caso.
Esatti, s'intende, rispetto al modo che si è scelto per «spegnere» una colonna:
quella resta una scelta anche qui.

Il risultato più leggibile è il grafico **a cascata** della
{numref}`fig-shap-contributi`: si parte dalla risposta base e si impilano i
contributi, uno per riga. Quelli che **alzano** la risposta sono barre che vanno
verso destra, in terracotta (il rosso mattone); quelli che la **abbassano**
tornano verso sinistra, in teal (il verde-azzurro scuro). Si arriva così alla
risposta per *questo* cliente, e una singola immagine racconta, voce per voce,
perché il modello ha deciso così.

`````

`````{tab} Superiore

KernelSHAP riformula il calcolo come una regressione lineare pesata sulle
coalizioni: campionando sottoinsiemi $S$ e pesandoli con il **kernel di
Shapley** $\pi(S) = \frac{n-1}{\binom{n}{|S|}\,|S|\,(n - |S|)}$, la soluzione
ai minimi quadrati converge ai valori di Shapley; è la scelta di pesi che
distingue SHAP da LIME, i cui pesi euristici non hanno questa garanzia.

A pesi giusti, però, resta da calcolare la funzione valore, e lì la garanzia si
assottiglia in un modo che vale la pena dichiarare. KernelSHAP approssima
$v(S) = \mathbb{E}\big[f(\mathbf{x}) \mid \mathbf{x}_S\big]$ con l'attesa
**marginale**, sostituendo le feature assenti con valori pescati da un insieme
di riferimento: è l'ipotesi di **indipendenza fra le feature**, dichiarata da
Lundberg e Lee nel passaggio in cui derivano l'approssimazione
{cite}`lundberg2017unified`. Quando le feature sono correlate le due quantità
divergono, e la divergenza non è piccola: su due colonne quasi identiche di cui
il modello ne usa una sola, la versione marginale dà tutto il merito alla
colonna usata e zero all'altra, quella condizionale lo divide quasi a metà
{cite}`aas2021explaining`. È ancora la forcella dell'apertura, ed è il punto in
cui morde: KernelSHAP restituisce sistematicamente la prima risposta, mentre chi
la legge crede spesso di star leggendo la seconda.

TreeSHAP (introdotto in un lavoro successivo degli stessi autori) elimina
invece il campionamento per i modelli ad albero, con costo $O(T L D^2)$ ($T$
alberi, $L$ foglie, $D$ profondità), propagando lungo l'albero le popolazioni
delle coalizioni. I valori sono **esatti** rispetto alla $v$ che
l'implementazione adotta: la variante *path-dependent* stima $v$ dalle
popolazioni dei nodi dell'albero e non riproduce la marginalizzazione pura;
la variante *interventional* la calcola davvero, rispetto a un insieme di
riferimento esplicito.

Il vantaggio teorico appartiene ai valori di Shapley, e quindi a ogni loro
calcolo esatto: è la **consistenza**. Se si modifica il modello così che una
feature contribuisca di più in ogni coalizione, il suo valore SHAP non può
diminuire; è una monotonia che l'importanza da impurità della prima sezione
*non* garantisce. Oltre al waterfall (una predizione), i grafici tipici sono
il *force plot*, che dispone gli stessi contributi come forze contrapposte
lungo una retta, e soprattutto il **beeswarm**: impilando i valori SHAP di
migliaia di istanze, una riga per feature, si ricostruisce una vista
**globale** (quali feature contano e in che direzione) a partire da tante
spiegazioni locali. È il ponte tra il locale e il globale che rende SHAP così
usato.

`````

```{figure} ../figures/shap-contributi.svg
:name: fig-shap-contributi
:alt: "Grafico a cascata di una singola risposta del modello. A sinistra una linea verticale tratteggiata al valore base E[f(x)] uguale a 0,20. Quattro barre orizzontali si impilano dal basso verso l'alto: 'storia = buona' aggiunge +0,18 verso destra in terracotta, 'reddito = alto' aggiunge +0,10, 'eta = 40' aggiunge +0,04, mentre in cima 'debito = alto' sottrae 0,08 tornando verso sinistra in teal. Si arriva all'output f(x) uguale a 0,44, segnato da una seconda linea verticale tratteggiata più a destra della prima."
:width: 85%

Un grafico a cascata (*waterfall*) SHAP scompone una singola risposta. Qui il
modello risponde con una probabilità, un numero fra $0$ e $1$, e non con un
punteggio da 0 a 100 come nell'esempio di poco fa: la scala cambia, il
ragionamento no. Si parte dalla riga tratteggiata di sinistra, la **risposta
base**, cioè quanto risponde il modello quando di questo cliente non sa niente:
$0{,}20$. Nel disegno la si trova scritta $\mathbb{E}[f(x)]$, che è il modo dei
matematici di dire «la media delle risposte del modello su tutti i clienti».
È una delle scelte possibili, ed è quella che si fa di solito: al posto di
«niente» si mette la media di tutti, invece di uno zero o dei valori di un
cliente preso a caso. Cambiando quella scelta si sposta la riga tratteggiata di
sinistra, e con essa tutte le barre. Poi si impilano i
contributi, uno per riga, partendo dal basso: le barre che vanno verso destra,
in terracotta, alzano la risposta; quella che torna verso sinistra, in teal, la
abbassa. La seconda riga tratteggiata è la risposta per questo cliente,
$0{,}44$, e la somma dei contributi ci arriva esattamente
($0{,}20 + 0{,}18 + 0{,}10 + 0{,}04 - 0{,}08 = 0{,}44$): è la proprietà che
abbiamo chiamato efficienza.
```

## Controfattuali: cosa sarebbe dovuto cambiare

LIME e SHAP rispondono a «perché questa decisione?». Ma a chi si è visto negare
un prestito interessa spesso un'altra domanda, più pratica: «cosa devo cambiare
perché la prossima volta sia un sì?». È la spiegazione **controfattuale**,
formalizzata nel 2017 da Sandra Wachter, Brent Mittelstadt e Chris Russell
{cite}`wachter2017counterfactual` proprio pensando al «diritto alla spiegazione»
del GDPR europeo, discusso nell'introduzione del capitolo.

`````{tab} Elementare

Un controfattuale è un'affermazione del tipo: «se il tuo reddito fosse stato
30 000 € invece di 24 000, il prestito sarebbe stato approvato». Non ti dice
come funziona il modello dentro; ti dice **la modifica più piccola** che avrebbe
ribaltato la decisione. È **azionabile**: indica una via d'uscita concreta,
non una diagnosi astratta.

La qualità di un buon controfattuale sta in due cose. Primo, dev'essere
**vicino** alla tua situazione reale: «guadagna 200 000 € in più» tecnicamente
ribalta la decisione ma non serve a nessuno; «riduci di una rata i tuoi debiti»
è molto più utile. Secondo, dev'essere **plausibile e realizzabile**: suggerire
di cambiare l'età non ha senso, perché non è una leva su cui puoi agire. Il buon
controfattuale è il consiglio minimo, concreto e onesto che ti mette dalla parte
giusta della decisione.

`````

`````{tab} Superiore

Wachter e colleghi cercano un punto $\mathbf{x}_{\mathrm{cf}}$ che ottenga
l'esito desiderato $y'$ restando il più vicino possibile all'istanza originale
$\mathbf{x}_0$, minimizzando

$$
\operatorname*{arg\,min}_{\mathbf{x}_{\mathrm{cf}}}\;
   \lambda\,\big(f(\mathbf{x}_{\mathrm{cf}}) - y'\big)^2
   + d(\mathbf{x}_{\mathrm{cf}}, \mathbf{x}_0),
$$

dove il primo termine spinge la predizione verso il valore-bersaglio $y'$ (la
soglia di approvazione) e $d$ misura quanto $\mathbf{x}_{\mathrm{cf}}$ si
discosta da $\mathbf{x}_0$: nel
paper è una distanza di Manhattan ($L_1$) pesata, feature per feature, con la
deviazione assoluta mediana, che favorisce modifiche **sparse** e rende
confrontabili scale diverse. Il moltiplicatore $\lambda$ non è un compromesso
da regolare a mano: lo si fa **crescere** finché la predizione non rientra in
una tolleranza fissata attorno a $y'$, così che il primo termine agisca da
vincolo e, sotto quel vincolo, si minimizzi la distanza. Estensioni successive
aggiungono vincoli di **plausibilità** (restare sul supporto dei dati) e di
**azionabilità** (non modificare feature immutabili come l'età o l'etnia).

C'è un parallelo tecnico che vale la pena rendere esplicito. Cercare la
perturbazione minima di $\mathbf{x}_0$ che cambia l'uscita del modello è, formalmente,
lo stesso problema degli **esempi avversari**: le impercettibili modifiche
d'input che ingannano una rete, studiate da Goodfellow, Shlens e Szegedy
{cite}`goodfellow2015explaining` e riprese nel capitolo sull'AI responsabile.
La matematica è la medesima, l'intento opposto: un esempio avversario nasconde
la perturbazione per **ingannare** il modello; un controfattuale la esibisce
per **spiegarlo** e offrire una via d'azione. Lo stesso strumento può violare
o servire l'interesse di chi subisce una decisione, a seconda di come lo si
usa.

`````

## Regole invece di pesi: gli anchor

LIME e SHAP consegnano un elenco di colonne con dei numeri accanto, e per
leggerlo bisogna saper interpretare dei numeri. C'è una forma di spiegazione locale
che non chiede questo sforzo, ed è la più antica che esista: una **regola**.

`````{tab} Elementare

La regola di cui parliamo suona così:

> «Finché il reddito supera i 30 000 € **e** negli ultimi due anni non ci sono
> state rate non pagate, questo modello dice sì, qualunque cosa facciano le
> altre colonne.»

Una regola così si chiama **anchor**, àncora, e la differenza con LIME non è
di stile, è di sostanza. LIME dice quanto ogni feature ha pesato *in questo
caso*; un anchor dice **fin dove** la risposta resta la stessa. La prima è una
descrizione, la seconda è una promessa
verificabile: si può prendere la regola, cercare altri casi che la
soddisfano, e controllare se il modello risponde davvero sempre allo stesso
modo.

Da qui le due misure che accompagnano ogni anchor, e conviene guardarle con dei
numeri in mano. La **precisione** dice quanto spesso la regola azzecca la
risposta del modello: se su cento clienti che soddisfano la regola il modello
dice sì a novantasette, la precisione è del 97%. La **copertura** dice su
quanti clienti la regola si applica: se su diecimila clienti duemila hanno
reddito sopra 30 000 e nessuna rata non pagata, la copertura è del 20%.

Le due tirano in direzioni opposte, ed è ovvio perché: più condizioni si
aggiungono, più la regola diventa infallibile e meno gente ci ricade sotto. Una
regola con dieci condizioni sarà quasi sempre esatta e varrà quasi per nessuno;
una con una condizione sola varrà per molti e sbaglierà spesso. Un buon anchor è
la regola più corta che tiene la precisione richiesta.

E quanto debba essere alta quella precisione non lo dicono i dati: lo decide chi
usa lo strumento, e di solito la si fissa molto in alto, per esempio al 95%.
Attenzione a che cosa promette quel 95%: promette che la regola descrive bene
**il modello**, non che il modello abbia ragione. Un anchor precisissimo su un
modello sbagliato descrive perfettamente uno sbaglio.

`````

`````{tab} Superiore

Gli **anchor** {cite}`ribeiro2018anchors` sono degli stessi autori di LIME, e
nascono per rispondere a un difetto dichiarato di quel metodo: un modello
lineare locale non dice **dove finisce** la sua validità, e il lettore non ha
modo di sapere se l'approssimazione regge un pixel più in là o mezzo dataset.

Un anchor è un predicato $A$ sull'istanza (una congiunzione di condizioni
sulle feature) tale che, campionando perturbazioni $\mathbf{z}$ da una
distribuzione $\mathcal{P}$ condizionata al fatto che $A$ resti soddisfatto, il
modello mantenga la stessa predizione con alta probabilità:

$$
\operatorname{prec}(A) = \mathbb{E}_{\mathbf{z} \sim \mathcal{P}(\cdot \mid A)}
\big[\, \mathbb{1}[\,f(\mathbf{z}) = f(\mathbf{x}_0)\,] \,\big] \;\ge\; \tau ,
$$

tipicamente con $\tau = 0{,}95$, soglia scelta da chi analizza e non dedotta
dai dati. Fra tutti i predicati che soddisfano il
vincolo si cerca quello di **copertura** massima,
$\operatorname{cov}(A) =
\mathbb{E}_{\mathbf{z}\sim\mathcal{P}}[\mathbb{1}[A(\mathbf{z})]]$. La
ricerca procede aggiungendo una condizione alla volta e stimando la precisione
per campionamento; poiché ogni valutazione costa, il problema di quale
candidato affinare è formulato come *best-arm identification*, cioè quello che
i bandit del capitolo sul reinforcement learning risolvono.

Il guadagno rispetto a LIME è la **fedeltà dichiarata**: un anchor non
approssima, delimita, e la sua precisione è un numero misurato invece che una
speranza. Il prezzo è che su feature continue e ad alta dimensione le regole
diventano lunghe o la copertura crolla, e su dati non tabellari (immagini,
testo) bisogna prima definire che cosa sia una «condizione», il che riporta
tutti i problemi di rappresentazione di LIME.

`````

## Quel che manca: i negativi pertinenti

I controfattuali chiedono che cosa cambiare. C'è una domanda gemella e
asimmetrica che vale la pena distinguere, perché risponde a un dubbio diverso:
non «che cosa devo cambiare», ma «che cosa, di ciò che **non** c'è, sta
determinando la risposta».

`````{tab} Elementare

Cambiamo per un attimo mestiere al modello, perché su questo l'esempio si vede
meglio. Immagina un modello che guarda una cifra scritta a mano, di quelle sulle
buste da lettera, e deve dire quale cifra è. Su una certa immagine risponde
$3$. Due domande diverse.

La prima: quali tratti dell'immagine **bastano** perché resti un $3$? Se si
cancella tutto il resto e restano solo quelli, la risposta non cambia. Sono i
**positivi pertinenti**: il minimo indispensabile presente.

La seconda: quale tratto, se ci **fosse**, la farebbe diventare un $8$? Un
piccolo arco a sinistra, che chiuda le due pance. Quel tratto è un **negativo
pertinente**: non c'è nell'immagine, e la sua assenza è parte del motivo per
cui la risposta è $3$ e non $8$.

È la differenza fra dire «è un tre per via di questi tratti» e «è un tre e non
un otto perché manca questo». La seconda è il modo in cui le persone spiegano
davvero le cose, e in medicina è la forma standard del ragionamento: un medico
che esclude l'influenza non lo fa solo per quello che il paziente ha, lo fa
anche per quello che non ha (febbre alta sì, ma nessun dolore muscolare e
nessuna tosse). Una diagnosi si regge tanto sui sintomi presenti quanto su
quelli **attesi e assenti**.

`````

`````{tab} Superiore

Il **Contrastive Explanation Method** {cite}`dhurandhar2018explanations`
formalizza le due nozioni cercando, attorno all'istanza $\mathbf{x}_0$, due
perturbazioni minime di segno opposto.

Il **positivo pertinente** è la porzione minima di $\mathbf{x}_0$ che, da sola,
conserva la classificazione: si cerca $\boldsymbol{\delta}$ sparso tale che
$f(\boldsymbol{\delta})$ dia la stessa classe di $f(\mathbf{x}_0)$, con
$\boldsymbol{\delta}$ contenuto in $\mathbf{x}_0$.
Il **negativo pertinente** è la perturbazione minima **additiva** che cambia
la classe: si cerca $\boldsymbol{\delta}$ tale che
$f(\mathbf{x}_0 + \boldsymbol{\delta})$ dia una classe diversa, con
$\boldsymbol{\delta}$ di norma minima.

La formulazione usa una regolarizzazione elastica ($L_1$ più $L_2$) per
ottenere perturbazioni sparse e interpretabili, e opzionalmente un
autoencoder addestrato sui dati come termine di penalità, che spinge la
soluzione a restare sulla varietà dei dati plausibili invece di finire in una
zona dello spazio che nessun esempio reale abita. È lo stesso vincolo di
plausibilità già incontrato per i controfattuali, imposto qui in modo
esplicito.

La parentela con i due metodi appena visti è stretta e conviene esplicitarla:
il negativo pertinente **è** un controfattuale, cercato però solo fra le
perturbazioni additive e presentato come «ciò che manca» invece che come «ciò
che cambierebbe». Il positivo pertinente, invece, è parente dell'anchor, ma
risponde alla domanda «che cosa basta» in un altro modo: un anchor fissa
alcune condizioni e lascia che tutte le altre feature varino liberamente; il
positivo pertinente spegne tutto il resto sulla baseline e cerca la porzione
minima dell'input che conserva la classe da sola.

`````

## In pratica: i valori di Shapley calcolati da zero

Esistono librerie che calcolano tutto questo in due righe. Qui sotto c'è il
programma che lo fa, ma chi non programma può saltarlo a piè pari: subito dopo
il conto lo rifacciamo per intero a mano, con carta e penna, ed è quella la
parte che conta.

Per capire cosa c'è sotto conviene infatti rifare il conto **provando tutti gli
ordini a uno a uno**,
esattamente come si è fatto con 10, 30, 20 e 50 di qualche pagina fa. Stavolta
però con tre colonne invece che due, il che cambia una cosa sola e vale la pena
dirla subito: con due colonne gli ordini erano due, con tre diventano **sei**
(la prima entrata si può scegliere in tre modi, la seconda nei due rimasti, la
terza è obbligata: $3 \times 2 \times 1$).

Il modellino su cui lo faremo è una formula inventata, con tre colonne che
chiameremo $x_0$, $x_1$ e $x_2$: si numera da zero perché così fa Python, quindi
la «colonna 0» è la prima. La formula è

$$
f(x_0, x_1, x_2) = x_0 + 2\,x_1 + x_0\,x_2 .
$$

Tre addendi. Il primo prende la prima colonna così com'è; il secondo prende la
seconda e la raddoppia; il terzo moltiplica la prima per la terza, e quindi vale
qualcosa solo se **tutte e due** sono diverse da zero. Quel terzo addendo è
un'**interazione**, la stessa cosa che nell'esempio dei prestiti faceva
arrivare a 50 invece che a 40, ed è il motivo per cui questa formula è stata
scelta: è il caso in cui i meriti non sono ovvi.

Per «spegnere» una colonna useremo la più semplice delle tre scelte elencate
sopra: al posto del suo valore vero ci mettiamo uno zero. Il caso da spiegare è
quello in cui tutte e tre le colonne valgono $1$; la risposta base, cioè quella
a tutto spento, è $f(0,0,0) = 0$. La proprietà da verificare alla fine è
l'efficienza: la somma dei tre meriti deve fare esattamente la risposta vera
meno la risposta base, cioè $f(1,1,1) - f(0,0,0) = 4 - 0 = 4$.

```python
import itertools
from math import factorial
import numpy as np

# Un modello giocattolo con un'interazione: la feature 2 "conta" solo con la 0
def f(x):
    return x[0] + 2.0 * x[1] + x[0] * x[2]

# istanza da spiegare e riferimento (baseline) su cui "spegnere" le feature assenti
x = np.array([1.0, 1.0, 1.0])
r = np.array([0.0, 0.0, 0.0])
n = len(x)

# valore della coalizione S: le feature in S prendono il valore di x, le altre di r
def v(S):
    z = r.copy()
    for i in S:
        z[i] = x[i]
    return f(z)

# valori di Shapley per forza bruta: media dei contributi marginali su TUTTI gli ordini
phi = np.zeros(n)
for perm in itertools.permutations(range(n)):
    S = []
    for i in perm:
        prima = v(S)            # coalizione prima di aggiungere i
        S = S + [i]
        dopo = v(S)             # coalizione dopo aver aggiunto i
        phi[i] += dopo - prima  # contributo marginale di i in questo ordine
phi /= factorial(n)             # media sugli n! ordini

print("valori di Shapley:", np.round(phi, 3))
print("somma dei phi:     ", round(float(phi.sum()), 3))
print("f(x) - f(base):    ", round(float(f(x) - f(r)), 3))  # assioma di efficienza
```

L'output:

```text
valori di Shapley: [1.5 2.  0.5]
somma dei phi:      4.0
f(x) - f(base):     4.0
```

Tre numeri: $1{,}5$, $2{,}0$ e $0{,}5$. Da dove escono? Verrebbe da aspettarsi
$1$ e $2$, che sono i due numeri scritti nella formula, e niente per la terza
colonna, che un numero suo non ce l'ha. Il modo più pulito di capire perché non
è così è rifare il conto con carta e penna, usando tre delle quattro proprietà,
una dopo l'altra.

La prima è l'**additività**, quella dell'esempio delle due pagelle: un conto che
è una somma si può spezzare, fare i conti sui pezzi e sommare i risultati. Qui i
pezzi sono i tre addendi, e su ciascuno il merito si vede a occhio.

- $x_0$ da solo. Accendere la prima colonna porta questo addendo da $0$ a $1$,
  in qualunque ordine, e le altre due colonne non lo toccano mai. Meriti:
  $(1,\;0,\;0)$.
- $2\,x_1$ da solo. Stessa cosa, ma il salto è di $2$, e tocca alla seconda
  colonna. Meriti: $(0,\;2,\;0)$.
- $x_0 x_2$ da solo. Qui serve il conto vero. Questo addendo vale $1$ se le
  colonne $0$ e $2$ sono accese entrambe, e $0$ in tutti gli altri casi; la
  colonna $1$ non lo tocca mai, quindi è un **giocatore nullo** e la si può
  togliere di mezzo, restando con un conto a due giocatori come quello dei
  prestiti. Se entra prima la $0$: non aggiunge niente (l'altra è ancora
  spenta), poi entra la $2$ e aggiunge $1$. Se entra prima la $2$: non aggiunge
  niente, poi entra la $0$ e aggiunge $1$. Media per la colonna $0$:
  $(0+1)/2 = 0{,}5$. Media per la $2$: uguale. Meriti:
  $(0{,}5,\;0,\;0{,}5)$.

Adesso si sommano colonna per colonna: la prima prende $1 + 0 + 0{,}5 = 1{,}5$,
la seconda $0 + 2 + 0 = 2$, la terza $0 + 0 + 0{,}5 = 0{,}5$. Sono i tre numeri
stampati dal programma, ottenuti senza programma. E la terza colonna, che nella
formula non aveva un numero suo, prende comunque mezzo punto: se l'è guadagnato
tutto nell'interazione.

La terza proprietà usata è la **simmetria**, e vale la pena vedere dove è
entrata: nel terzo pezzo, e solo lì. Dentro $x_0x_2$ le due colonne fanno
esattamente lo stesso mestiere (nessuna delle due vale niente senza l'altra), e
chi contribuisce allo stesso modo riceve lo stesso: da qui il mezzo punto a
testa. Applicarla al conto **intero** sarebbe invece un errore, perché lì le due
colonne non fanno affatto lo stesso mestiere, e infatti prendono $1{,}5$ e
$0{,}5$. È l'additività a permettere di spezzare, ed è solo dopo aver spezzato
che la simmetria si può usare, nel pezzo in cui vale.

Resta la quarta, l'**efficienza**, che è quella che il programma verifica nelle
ultime due righe: $1{,}5 + 2{,}0 + 0{,}5 = 4{,}0$, che è
esattamente la risposta vera meno la risposta base. Il conto torna.

Un'ultima nota pratica sul costo, e conviene chiarire un punto che altrimenti
confonde: la stessa fatica si può contare in due modi, gli ordini di ingresso
oppure i gruppi di colonne da provare, e crescono a valanga tutti e due. Qui gli
ordini sono sei e i gruppi otto. Con dieci colonne gli ordini superano i tre
milioni; con venti sono più di due miliardi di miliardi; e con trenta i gruppi
passano il miliardo, che è il numero citato all'inizio della sezione. Nella pratica quindi non si provano tutti: se ne prova un campione
a caso, oppure si usa TreeSHAP quando il modello è fatto di alberi. La
definizione, però, è questa.

`````{tab} Elementare

```{admonition} Da ricordare
:class: important
- Una spiegazione **locale** riguarda *una* risposta sola, non il modello
  intero: «perché hanno detto no a *me*», non «che cosa conta in media». Un
  modello può essere intricato dappertutto e semplice qui accanto, come la
  strada di montagna che da vicino sembra dritta.
- **LIME** fabbrica tanti casi-fantasma, chiede al modello che cosa
  risponderebbe per ciascuno, e su quella nuvola costruisce un modellino a
  somma, contando di più i fantasmi più simili al caso da spiegare. Il «solo
  qui vicino» sta tutto in quel conteggio, non nel modo in cui i fantasmi sono
  nati. I numeri di quella somma sono la spiegazione. Funziona con qualunque
  modello, ma è **instabile**: rilanciato sullo stesso caso dà numeri diversi, e
  cambia anche a seconda di quanto largo si prende il vicinato e di come si è
  deciso di spezzettare il caso in parti.
- I **valori di Shapley** (una formula del 1953, nata per dividere fra i soci
  il guadagno di un'impresa) ripartiscono fra le colonne lo scarto fra la
  risposta su questo caso e la **risposta base**, cioè quella che il modello dà
  quando non sa niente. La quota di ogni colonna è la media di quanto aggiunge,
  su tutti gli ordini in cui le colonne possono entrare in campo: è il conto con
  10, 30, 20 e 50.
- Sono l'unico modo di dividere che rispetta quattro richieste ragionevoli: il
  conto torna senza avanzi; chi fa lo stesso lavoro prende uguale; chi non
  aggiunge mai niente prende zero; e un conto che è una somma si può spezzare in
  pezzi, fare i conti sui pezzi e sommare. Con un però: prima bisogna stabilire
  che cosa significa «non far sapere» una colonna al modello, e deciderlo in un
  modo o nell'altro sposta la risposta base, e con essa tutti i meriti.
- **SHAP** è il modo di calcolarli in fretta, perché provare tutte le
  combinazioni è impossibile: ne prova solo alcune, se il modello è una scatola
  chiusa qualsiasi, oppure sfrutta la forma degli alberi per farlo in modo
  esatto. Il risultato si legge nel grafico a cascata, una barra per colonna.
- I **controfattuali** dicono la modifica più piccola che avrebbe ribaltato la
  risposta («se il tuo reddito fosse stato 30 000 invece di 24 000»): una via
  d'uscita concreta, purché resti vicina alla situazione reale e riguardi
  qualcosa su cui si può davvero agire.
- Gli **anchor** sostituiscono i numeri con una **regola** («finché il reddito
  supera 30 000, è sì») e ne dichiarano i limiti: la **precisione**, quanto
  spesso azzecca la risposta del modello, e la **copertura**, su quanti casi si
  applica. Dicono **fin dove** la risposta non cambia, che è la cosa che LIME
  non dice. Attenzione: azzeccare il modello non vuol dire aver ragione.
- I **positivi** e i **negativi pertinenti** sono le due domande del tre e
  dell'otto: quali tratti bastano perché resti un tre, e quale tratto, se ci
  fosse, lo farebbe diventare un otto. La seconda è il modo in cui le persone
  spiegano davvero le cose, ed è la forma del ragionamento medico.
```

`````

`````{tab} Superiore

```{admonition} Da ricordare
:class: important
- Una spiegazione **locale** riguarda *una* predizione $f(\mathbf{x}_0)$, non l'intero
  modello: risponde a «perché *questo* caso?» dove l'importanza globale della
  sezione precedente rispondeva a «cosa conta in media?».
- **LIME** {cite}`ribeiro2016why` approssima il modello con un **surrogato
  lineare** definito su una **rappresentazione interpretabile binaria**
  (superpixel, parole, intervalli) e adattato a punti perturbati pesati per
  prossimità: la località la porta il peso $\pi_{\mathbf{x}_0}$, non il campionamento. È
  model-agnostic ma **instabile**, sensibile alla larghezza del vicinato e alla
  segmentazione scelta.
- I **valori di Shapley** (teoria dei giochi, 1953) ripartiscono fra le feature
  lo scarto fra la predizione $f(\mathbf{x}_0)$ e un **valore base**, cioè quanto
  risponde il modello quando delle feature non sa nulla; la quota di ciascuna è
  la media dei suoi **contributi marginali** su tutti gli ordini di ingresso.
  Sono l'unica attribuzione che soddisfa **efficienza, simmetria, giocatore
  nullo e additività**, una volta stabilito che cosa significa «non far sapere»
  una feature al modello: deciderlo in un modo o nell'altro sposta il valore
  base, e con esso le attribuzioni.
- **SHAP** {cite}`lundberg2017unified` li stima in modo efficiente,
  **KernelSHAP** (agnostico) e **TreeSHAP** (esatto per gli alberi), con la
  proprietà di **consistenza**; si legge col grafico a cascata (*waterfall*),
  il *force plot* e il *beeswarm*. KernelSHAP approssima $v(S)$ con l'attesa
  **marginale**, cioè assumendo le feature **indipendenti**: con feature
  correlate le attribuzioni divergono da quelle condizionali
  {cite}`aas2021explaining`.
- I **controfattuali** {cite}`wachter2017counterfactual` indicano la modifica
  minima e azionabile che ribalterebbe la decisione; sono lo stesso problema
  matematico degli **esempi avversari** {cite}`goodfellow2015explaining`, con
  intento opposto: spiegare invece di ingannare.
- Gli **anchor** {cite}`ribeiro2018anchors` sostituiscono i pesi con una
  **regola** e ne dichiarano i limiti: *precisione* (quanto spesso la regola
  azzecca il modello) e *copertura* (su quanti casi si applica). Dicono **fin
  dove** la risposta non cambia, cosa che LIME non fa.
- Il **CEM** {cite}`dhurandhar2018explanations` distingue i **positivi
  pertinenti** (che cosa basta perché la risposta sia questa) dai **negativi
  pertinenti** (che cosa, assente, la tiene ferma). Il negativo pertinente è un
  controfattuale additivo; il positivo pertinente è la porzione minima che
  basta da sola, col resto cancellato, mentre un anchor fissa alcune condizioni
  e lascia variare tutto il resto.
```

`````
