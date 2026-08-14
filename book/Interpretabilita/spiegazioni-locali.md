# Spiegazioni locali: LIME, SHAP e controfattuali

Nel 1953 **Lloyd Shapley**, un giovane matematico di Princeton che di lì a poco
sarebbe entrato alla RAND Corporation, si pose una domanda che con
l'intelligenza artificiale non c'entrava nulla: se un gruppo di persone
collabora a un'impresa e ne ricava un guadagno, come si divide il merito «in
modo equo» tra chi ha partecipato? Alcuni contano di più, alcuni solo in
combinazione con altri; non basta guardare cosa fa ciascuno da solo. Shapley
rispose fissando quattro requisiti così ragionevoli che nessuno li
discuterebbe, e dimostrando che un solo modo di dividere li rispetta tutti e
quattro. Quasi sessant'anni dopo avrebbe vinto il premio Nobel per l'economia:
non per questa formula, per la verità, ma per un altro filone dei suoi studi di
teoria dei giochi, quello sugli abbinamenti stabili. Non poteva immaginare che
la sua idea sarebbe diventata, nel 2017, lo strumento più usato al mondo per
spiegare la singola decisione di una rete neurale.

Questa è la seconda tappa del capitolo, e cambia scala. La sezione precedente,
sull'importanza delle feature, rispondeva a una domanda **globale**: «su quali
colonne si regge il modello, *in media*?». Ma chi si vede negare un prestito
non chiede una media: chiede «perché *la mia* domanda?». È una domanda
**locale**, perché riguarda una predizione sola e non l'intero comportamento
del modello. In formule quella predizione si scrive $f(\mathbf{x}_0)$: il
modello $f$ applicato a quel caso lì, $\mathbf{x}_0$. Qui vediamo tre modi di
risponderle, tre attrezzi diventati lo standard di fatto: **LIME**, **SHAP** e
le spiegazioni **controfattuali**.

## Perché *questa* predizione

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
Guestrin {cite}`ribeiro2016why`, gli stessi dell'esperimento sugli husky e i
lupi visto in apertura di capitolo. L'idea è disarmante nella sua semplicità:
se la scatola nera è troppo complicata da capire tutta, costruiamone una copia
*facile* che le somigli **solo qui vicino**, attorno al punto da spiegare.

`````{tab} Elementare

Immagina di voler capire come il modello decide sul caso di Maria. LIME fa
così: fabbrica tante pratiche finte, **casi-fantasma** che somigliano più o
meno a quella di Maria (un po' più di reddito, un debito in meno, un'età
diversa), e per ciascuna chiede alla scatola nera cosa risponderebbe. Ottiene
una nuvola di esempi inventati, ognuno con la risposta che il modello gli
darebbe.

A questo punto cerca fra quei punti un modello **semplice e leggibile**: la
retta che ci passa più vicino possibile, come la regressione lineare del
capitolo di machine learning. Con una regola in più, ed è quella che fa tutto
il lavoro: i fantasmi più simili alla pratica vera di Maria contano di più,
quelli lontani quasi niente. È tutto qui il senso della parola «pesata», ed è
qui, nel conteggio, che entra il «solo qui vicino»: non nel modo in cui i
fantasmi sono stati fabbricati.

I coefficienti di quella retta *sono* la spiegazione: «il reddito basso ha
spinto verso il no di tanto, i debiti di tanto, l'anzianità di lavoro ha spinto
un po' verso il sì». È come tracciare la tangente a una curva: non descrive
tutta la strada, ma dice benissimo la pendenza nel punto in cui ti trovi.

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

Quel metodo esiste, ed è la formula di Shapley del 1953. Il salto concettuale è
vedere una predizione come un **gioco cooperativo**: le feature sono i
giocatori, e il «guadagno» da spartire è di quanto la predizione si scosta dal
valore medio del modello. La domanda «quanto ha contribuito il reddito a questo
rifiuto?» diventa la vecchia domanda di Shapley: «quanto merito spetta a questo
giocatore?».

`````{tab} Elementare

Il punto delicato è che i giocatori non agiscono da soli: contano anche le
**combinazioni**. Prendiamo due feature, il *reddito* e la *storia creditizia*,
e chiediamoci quanto vale ciascuna. L'idea di Shapley è: proviamo tutti gli
ordini in cui le feature possono «entrare in campo» e, per ognuna, misuriamo
quanto aggiunge nel momento in cui entra. Poi facciamo la media.

Prima però va sciolto un dubbio ragionevole: che cosa vuol dire far girare il
modello «senza» una colonna? Il modello vuole tutte le sue caselle piene, non
gli si può lasciare una casella vuota. Quello che si fa è riempirla con
qualcosa di neutro: il valore medio di quella colonna su tutti i clienti, o un
valore preso a caso da un altro cliente, o uno zero convenzionale. È una
**scelta**, non un fatto, e più avanti si vedrà che sceglierla in un modo o
nell'altro sposta i numeri; per ora basta tenere presente che «il modello non
sa il reddito» significa «al posto del reddito c'è qualcosa che non dice
niente».

Diciamo allora che il modello, senza sapere niente del cliente, parte da un punteggio
base di **10**. Se conosce solo il reddito, sale a **30**; se conosce solo la
storia, sale a **20**; se le conosce entrambe, arriva a **50**. Con due
feature gli ordini possibili sono due, prima il reddito, oppure prima la
storia:

- *Prima il reddito*: entra e porta il punteggio da 10 a 30, quindi aggiunge
  **20**. Poi entra la storia e lo porta da 30 a 50, aggiunge **20**.
- *Prima la storia*: entra e porta da 10 a 20, aggiunge **10**. Poi entra il
  reddito e porta da 20 a 50, aggiunge **30**.

Il merito del reddito è la media di quanto aggiunge nei due ordini:
$(20 + 30)/2 = 25$. Quello della storia: $(20 + 10)/2 = 15$. E il conto torna:
$25 + 15 = 40$, esattamente quanto separa il punteggio finale (50) da quello
base (10). Tutto il «guadagno» è stato ripartito, senza avanzi: nessun merito
inventato, nessuno perso per strada.

Le quattro proprietà con cui Shapley aveva fissato il suo modo di dividere sono
cose ovvie come quella che abbiamo appena visto tornare, e conviene chiamarle
per nome perché il resto del capitolo le usa. Che il conto torni senza avanzi si
chiama **efficienza**. Che due colonne che fanno esattamente lo stesso mestiere
ricevano lo stesso merito si chiama **simmetria**. Che una colonna che non
aggiunge mai niente, in nessun ordine, prenda zero si chiama **giocatore
nullo**. E che, mettendo insieme due modelli, i meriti si sommino, si chiama
**additività**. Sono quattro richieste che nessuno discuterebbe, e la cosa
notevole (il motivo per cui questa formula del 1953 è ancora qui) è che c'è un
solo modo di dividere il merito che le soddisfa tutte e quattro insieme.

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
metodi fino ad allora scollegati (LIME incluso, come caso particolare). Da qui
il metodo è diventato lo standard pratico dell'interpretabilità post-hoc.

`````{tab} Elementare

SHAP non cambia la definizione: restituisce ancora i contributi «equi» di
Shapley. Cambia il *come* li ottiene, con due scorciatoie a seconda del
modello. Per una scatola nera qualsiasi usa **KernelSHAP**: campiona con
astuzia solo alcune coalizioni invece di tutte, e ne ricostruisce i valori con
una regressione pesata; cugino di LIME, ma con i pesi «giusti» che
garantiscono di puntare ai valori di Shapley. Per i modelli ad alberi (foreste
casuali, gradient boosting, i protagonisti della sezione sugli alberi e gli
ensemble del capitolo sul machine learning), usa **TreeSHAP**, che sfrutta la
struttura ad albero per calcolare i valori **esatti** in tempo ragionevole,
senza campionare nulla. Esatti, s'intende, rispetto al modo che si è scelto per
«spegnere» una colonna: quella resta una scelta anche qui.

Il risultato più leggibile è il grafico **a cascata** (*waterfall*) della
{numref}`fig-shap-contributi`: si parte dal valore base (la predizione media
del modello) e si impilano i contributi, uno per riga. Quelli che **alzano** la
predizione sono barre che vanno verso destra, in terracotta (il rosso mattone);
quelli che la **abbassano** tornano verso sinistra, in teal (il verde-azzurro
scuro). Si arriva così alla predizione di *questo* cliente, e una singola
immagine racconta, voce per voce, perché il modello ha deciso così.

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
:alt: "Grafico a cascata di una predizione. A sinistra una linea verticale tratteggiata al valore base E[f(x)] uguale a 0,20. Quattro barre orizzontali si impilano dal basso verso l'alto: 'storia = buona' aggiunge +0,18 verso destra in terracotta, 'reddito = alto' aggiunge +0,10, 'eta = 40' aggiunge +0,04, mentre in cima 'debito = alto' sottrae 0,08 tornando verso sinistra in teal. Si arriva all'output f(x) uguale a 0,44, segnato da una seconda linea verticale tratteggiata più a destra della prima."
:width: 85%

Un grafico a cascata (*waterfall*) SHAP scompone una singola predizione. Si
parte dalla riga tratteggiata di sinistra, il **valore base**: la risposta
media del modello quando non sa niente di questo cliente, che nel disegno è
scritta $\mathbb{E}[f(x)]$ e qui vale $0{,}20$. Poi si impilano i contributi,
uno per riga, partendo dal basso: le barre che vanno verso destra, in
terracotta, alzano la predizione; quella che torna verso sinistra, in teal, la
abbassa. La seconda riga tratteggiata è la risposta per questo cliente,
$0{,}44$, e la somma dei contributi ci arriva esattamente: è la proprietà che
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

LIME e SHAP consegnano un elenco di feature con dei numeri accanto, e per
leggerlo bisogna saper leggere dei pesi. C'è una forma di spiegazione locale
che non chiede questo sforzo, ed è la più antica che esista: una **regola**.

`````{tab} Elementare

La regola di cui parliamo suona così:

> «Finché il reddito supera i 30 000 € **e** non ci sono insolvenze negli
> ultimi due anni, questo modello dice sì, qualunque cosa facciano le altre
> feature.»

Una regola così si chiama **anchor**, àncora, e la differenza con LIME non è
di stile, è di sostanza. LIME dice quanto ogni feature ha pesato *in questo
caso*; un anchor dice **fin dove** la risposta resta la stessa. La prima è una
descrizione, la seconda è una promessa
verificabile: si può prendere la regola, cercare altri casi che la
soddisfano, e controllare se il modello risponde davvero sempre allo stesso
modo.

Da qui le due misure che accompagnano ogni anchor. La **precisione** dice
quanto spesso la regola azzecca la risposta del modello; la **copertura** dice
su quale porzione dei casi la regola si applica. Le due tirano in direzioni
opposte: una regola con dieci condizioni sarà quasi sempre esatta e varrà
quasi per nessuno; una con una condizione sola varrà per molti e sbaglierà
spesso. Un buon anchor è la regola più corta che mantiene la precisione
richiesta. Quanto debba essere alta quella precisione non lo dicono i dati: lo
decide chi usa lo strumento, e di solito la si fissa molto in alto, per esempio
al 95%, cioè la regola deve azzeccare la risposta del modello in almeno
novantacinque casi su cento fra quelli che ricadono sotto di essa.

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

Prendi una cifra scritta a mano che il modello classifica come un $3$. Due
domande diverse.

La prima: quali tratti dell'immagine **bastano** perché resti un $3$? Se si
cancella tutto il resto e restano solo quelli, la risposta non cambia. Sono i
**positivi pertinenti**: il minimo indispensabile presente.

La seconda: quale tratto, se ci **fosse**, la farebbe diventare un $8$? Un
piccolo arco a sinistra, che chiuda le due pance. Quel tratto è un **negativo
pertinente**: non c'è nell'immagine, e la sua assenza è parte del motivo per
cui la risposta è $3$ e non $8$.

È la differenza fra dire «è un tre per via di questi tratti» e «è un tre e non
un otto perché manca questo». La seconda è il modo in cui le persone
spiegano davvero le cose, e in medicina è la forma standard del ragionamento:
una diagnosi si regge tanto sui sintomi presenti quanto su quelli **attesi e
assenti**.

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

## In pratica: i valori di Shapley con NumPy

La libreria `shap` calcola tutto questo in poche righe, ma per capire davvero
cosa c'è sotto conviene ricostruire i valori di Shapley **a mano**, con la
definizione per forza bruta: la media dei contributi marginali su tutti gli
ordini di ingresso. È lo stesso conto con 10, 30, 20 e 50 di qualche pagina fa,
fatto su tre colonne invece che su due.

Il codice qui sotto usa solo NumPy. Per «spegnere» una feature adotta la più
semplice fra le scelte elencate sopra, la variante *baseline*: al posto del
valore vero mette quello di un riferimento fisso $\mathbf{r}$, qui tutti zeri.
Il valore base è quindi $f(\mathbf{r})$, cioè la risposta del modello quando è
spento tutto, e la proprietà da verificare, l'**efficienza**, diventa
$\sum_i \phi_i = f(\mathbf{x}) - f(\mathbf{r})$: la somma dei tre meriti deve
fare esattamente la differenza fra la risposta vera e quella base.

Il modello giocattolo è una formula inventata di tre variabili, scelta perché
contiene un'**interazione**: il terzo termine, $x_0 x_2$, vale qualcosa solo
quando le due variabili sono accese insieme, e nessuna delle due lo produce da
sola. È il caso su cui vale la pena vedere Shapley al lavoro. Qui $x_0$, $x_1$
e $x_2$ sono le tre **componenti** del caso $\mathbf{x}$, numerate da zero come
si usa in Python: la «feature 0» è la prima, la «feature 1» la seconda, la
«feature 2» la terza.

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

Il conto racconta esattamente cosa fa Shapley, e per leggerlo servono **due**
delle quattro proprietà, in fila.

La prima è l'**additività**: i valori di Shapley di
$f = x_0 + 2x_1 + x_0x_2$ sono la somma dei valori dei tre addendi presi come
giochi separati, e i tre pezzi valgono $(1,\,0,\,0)$, $(0,\,2,\,0)$ e
$(0{,}5,\,0,\,0{,}5)$. Da qui si legge subito la feature 1: compare solo nel
secondo addendo, con coefficiente $2$ e nessuna interazione, e prende $2{,}0$,
tutto suo.

La seconda è la **simmetria**, ma va applicata dove vale. Nel terzo gioco, e
solo lì, le feature 0 e 2 sono intercambiabili: dentro il termine $x_0x_2$
fanno esattamente lo stesso mestiere, e chi contribuisce allo stesso modo
riceve lo stesso, quindi quel punto si divide a metà. Ecco perché la feature 0
arriva a $1{,}5$ (il suo $1$ del primo addendo più $0{,}5$ dell'interazione) e
la feature 2 si ferma a $0{,}5$, non avendo altre entrate. Attenzione a non
applicare la simmetria al gioco **intero**: lì le due feature non sono affatto
intercambiabili, e infatti $\phi_0 = 1{,}5 \neq 0{,}5 = \phi_2$. È l'additività
a permettere di spezzare il conto in tre, ed è solo dopo averlo spezzato che la
simmetria chiude il pezzo dove serve.

E la somma $1{,}5 + 2{,}0 + 0{,}5 = 4{,}0$ coincide con l'ultima riga stampata,
$f(\mathbf{x}) - f(\mathbf{r})$:
l'**efficienza** è verificata numericamente. Nella pratica non si enumerano
tutti gli ordini (sono $n!$) ma si campionano, o si usa TreeSHAP se il modello
è un albero; la definizione, però, è questa.

`````{tab} Elementare

```{admonition} Da ricordare
:class: important
- Una spiegazione **locale** riguarda *una* risposta sola, non il modello
  intero: «perché hanno detto no a *me*», non «che cosa conta in media». Un
  modello può essere intricato dappertutto e semplice qui accanto, come la
  strada di montagna che da vicino sembra dritta.
- **LIME** fabbrica tanti casi-fantasma, chiede al modello che cosa
  risponderebbe per ciascuno, e a quella nuvola adatta una retta contando di
  più i fantasmi più simili al caso da spiegare. Il «solo qui vicino» sta tutto
  in quel conteggio, non nel modo in cui i fantasmi sono nati. I pesi della
  retta sono la spiegazione. Funziona con qualunque modello, ma è
  **instabile**: rilanciato sullo stesso caso dà numeri diversi, e cambia anche
  a seconda di quanto largo si prende il vicinato e di come si è deciso di
  spezzettare l'input in parti.
- I **valori di Shapley** (una formula del 1953, nata per dividere fra i soci
  il guadagno di un'impresa) ripartiscono fra le colonne lo scarto fra la
  risposta su questo caso e la **risposta base**, cioè quella che il modello dà
  quando non sa niente. La quota di ogni colonna è la media di quanto aggiunge,
  su tutti gli ordini in cui le colonne possono entrare in campo: è il conto
  con 10, 30, 20 e 50. Sono l'unico modo di dividere che rispetta quattro
  richieste ragionevoli (il conto torna senza avanzi; chi fa lo stesso prende
  uguale; chi non aggiunge mai niente prende zero; due modelli messi insieme
  sommano i meriti), una volta però stabilito che cosa significa «non far
  sapere» una colonna al modello: deciderlo in un modo o nell'altro sposta la
  risposta base, e con essa tutti i meriti.
- **SHAP** è il modo di calcolarli in fretta, perché provare tutte le
  combinazioni è impossibile: ne prova solo alcune, se il modello è una scatola
  chiusa qualsiasi, oppure sfrutta la forma degli alberi per farlo in modo
  esatto. Il risultato si legge nel grafico a cascata, una barra per colonna.
- I **controfattuali** dicono la modifica più piccola che avrebbe ribaltato la
  risposta («se il tuo reddito fosse stato 30 000 invece di 24 000»): una via
  d'uscita concreta, purché resti vicina alla situazione reale e riguardi
  qualcosa su cui si può davvero agire.
- Gli **anchor** sostituiscono i numeri con una **regola** («finché il reddito
  supera 30 000, è sì») e ne dichiarano i limiti: quanto spesso azzecca la
  risposta del modello, e su quanti casi si applica. Dicono **fin dove** la
  risposta non cambia, che è la cosa che LIME non dice.
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
