# I grandi modelli linguistici

Nel 1951 Claude Shannon pubblicò uno degli esperimenti più casalinghi della
storia dell'informatica {cite}`shannon1951prediction`: copriva una riga di un
testo e chiedeva a una persona di indovinare, lettera dopo lettera, come
continuava. Il risultato, contando i tentativi necessari, è più sorprendente di
quanto sembri. Senza sapere niente del contesto, indovinare una lettera
dell'alfabeto inglese è come scegliere fra ventisei possibilità: un'incertezza
che, nell'unità con cui la si misura (il **bit**, cioè il numero di domande sì
o no che servirebbero a risolverla), vale poco meno di cinque. Sapendo quel che
c'è scritto prima, quell'incertezza scende a **circa uno**: una domanda sola,
cioè quanto un testa-o-croce. Il contesto, insomma, toglie da solo i quattro
quinti dell'incertezza. L'abbiamo raccontato nel capitolo sui richiami di
matematica, quando si parlava di entropia. Settant'anni dopo, GPT-3
{cite}`brown2020language` gioca *esattamente lo stesso gioco*: indovinare come
continua un testo. Niente di più. Non è cambiata la scommessa; è cambiata la
scala. Un adolescente di tredici anni, secondo le stime usate dalla ricerca
sull'acquisizione del linguaggio, ha sentito e letto meno di 100 milioni di
parole; GPT-3 in addestramento ne ha viste circa 300 miliardi contate in
*token*, cioè nei pezzi in cui il testo viene spezzato, che per l'inglese
corrispondono a poco più di 200 miliardi di parole: più di duemila volte tanto.
Questa sezione racconta cosa succede quando la scommessa di Shannon viene
giocata su quella scala, e con quali regole, trucchi e delusioni.

Il terreno è già preparato. Nella sezione su GPT, BERT e T5 abbiamo visto il
metodo di studio della famiglia GPT, quello dello studente che copre la pagina
con la mano e indovina la parola dopo (in gergo: un Transformer
*decoder-only*, addestrato a predire il token successivo). E abbiamo visto che
GPT-3, arrivato a quella scala, sapeva eseguire un compito nuovo solo perché
glielo si descriveva a parole, magari con due o tre esempi svolti, senza
toccare un solo numero interno: si chiama capacità *few-shot*. Qui apriamo
il cofano: da dove vengono i dati, perché "più grande" funziona in modo così
prevedibile da meritarsi delle *leggi*, come si sceglie concretamente la
parola da scrivere, e quale accorgimento di ingegneria rende la generazione
sostenibile. Tra il gioco di Shannon e GPT-3 c'è un gradino intermedio che
vale la pena nominare: GPT-2 {cite}`radford2019language`, 1,5 miliardi di
parametri addestrati nel 2019 su pagine web segnalate dagli utenti di Reddit,
il cui titolo era già un manifesto: *i modelli di linguaggio sono studenti
multitask senza supervisione*.

## Una biblioteca sterminata: il pretraining su scala web

La lunga fase di studio generale in cui un modello legge tutto quel testo si
chiama **pretraining**, «pre-addestramento», e il «pre» dice già che dopo verrà
dell'altro: prima si impara la lingua e il mondo, poi si impara un mestiere.
Qui parliamo del primo tempo.

Trecento miliardi di token non stanno in nessuna enciclopedia: l'unico posto
dove trovarli è il web. Ma il web non è una biblioteca ordinata: è una
soffitta piena di tutto, dove i libri buoni stanno accanto allo spam, alle
pagine duplicate e ai commenti scritti di fretta. Metà del lavoro di chi
costruisce un grande modello non è addestrarlo: è preparare la biblioteca.

`````{tab} Elementare
Immagina di imparare una lingua straniera facendo *un solo tipo di esercizio*:
frasi da completare. Nessuna grammatica, nessun insegnante, nessuna correzione
a penna rossa: solo miliardi di esercizi di completamento, ricavati coprendo
l'ultima parola di frasi vere. «Il gatto nero salta sul ___»: provi, sbagli,
aggiusti, passi alla frase dopo. Con abbastanza esercizi, per completare bene
*devi* assorbire ortografia, grammatica, modi di dire, e perfino nozioni sul
mondo: non puoi completare «la capitale della Francia è ___» senza sapere di
Parigi. Il bello è che gli esercizi si fabbricano da soli: qualunque testo
esistente è già un esercizio con la soluzione inclusa. Serve però una
biblioteca sterminata e *pulita*: se la soffitta è piena di doppioni,
l'allievo impara a memoria invece di imparare la lingua; se è piena di
spazzatura, impara la spazzatura. Per questo, prima di studiare, si butta via
moltissimo: pagine duplicate, testo generato da macchine, contenuti di bassa
qualità.
`````

`````{tab} Superiore
La materia prima tipica è **Common Crawl**, un'istantanea periodica e
liberamente scaricabile del web, che va però raffinata: filtri di qualità (per
GPT-3, un classificatore addestrato a distinguere le pagine simili a corpora
di riferimento dal resto del crawl), **deduplicazione** fuzzy (i duplicati
gonfiano la memorizzazione e falsano la valutazione) e rimozione di contenuti
indesiderati. Il dataset di GPT-3 {cite}`brown2020language` è una miscela
pesata: il Common Crawl filtrato copre il 60% dei token visti in
addestramento, il resto viene da corpora più piccoli ma sovracampionati perché
ritenuti di qualità superiore; WebText2 (22%), due corpora di libri (8% + 8%)
e Wikipedia in inglese (3%). Il testo è segmentato in sub-word con BPE, come
visto nel capitolo sull'NLP {cite}`sennrich2016neural`.

L'unica supervisione è il testo stesso: si minimizza la cross-entropia sul
token successivo,

$$
\mathcal{L}(\theta) = -\sum_{t=1}^{n} \log p_\theta(x_t \mid x_1, \dots, x_{t-1}),
$$

dove $x_t$ è il token in posizione $t$, $p_\theta$ è la distribuzione prodotta
dal Transformer con parametri $\theta$ (softmax sull'intero vocabolario) e la
somma corre sugli $n$ token del corpus. Quando servirà la loss **per token**,
cioè la stessa quantità divisa per $n$, la scriveremo $\bar{\mathcal{L}} =
\mathcal{L}/n$: la distinzione sembra pedanteria e non lo è, perché più avanti
la perplessità si calcola mettendo all'esponente proprio quella, e chi confonde
le due sbaglia di un fattore $n$. È la stessa `nn.CrossEntropyLoss` dei
capitoli precedenti, applicata a un problema di classificazione con decine di
migliaia di classi (le parole possibili) ripetuto miliardi di volte. Nessuna
etichetta umana: per questo si parla di apprendimento **auto-supervisionato**.
Sui rischi di corpora così raccolti (bias, contenuti tossici, opacità) il
dibattito è aperto e acceso {cite}`bender2021dangers`.
`````

## La ricetta a tre ingredienti: le leggi di scala

Perché proprio *grandi* modelli? Non è una moda, è una regolarità che qualcuno
ha misurato.

Prima però serve sapere che cosa si misura, perché in questa sezione «migliora»
e «sbaglia meno» ricorrono di continuo. Un modello di linguaggio ha un solo
compito, indovinare la parola dopo, e su quel compito si può dargli un voto
preciso: gli si fa leggere del testo che non ha mai visto e si guarda quanta
probabilità aveva assegnato alle parole che poi sono comparse davvero. Se ne
dava tanta, ha indovinato bene; se ne dava poca, male. Quel numero, che va
verso il basso quando il modello impara, è **l'errore** di cui si parla qui
sotto (in gergo la *loss*), e più avanti in questa pagina lo ritroveremo sotto
un altro nome, la perplessità, che è lo stesso numero raccontato come un dado.

Tra il 2020 e il 2022 due lavori hanno misurato, con la pazienza di centinaia
di addestramenti, come cambia quell'errore al crescere delle risorse, e hanno
trovato curve così regolari da chiamarle **leggi di scala**.

`````{tab} Elementare
La ricetta di un modello di linguaggio ha tre ingredienti: la **taglia del
modello** (quante manopole interne ha da regolare: sono i numeri che
l'addestramento sposta un'inezia alla volta, e li si trova scritti ovunque con
tre nomi diversi che vogliono dire la stessa identica cosa, **parametri**,
**pesi** o appunto manopole; «un modello da sette miliardi» vuol dire sette
miliardi di manopole), la **quantità di testo** su cui studia, e il **calcolo**
(quante ore di computer può bruciare). La scoperta del 2020
{cite}`kaplan2020scaling` è che aumentando gli ingredienti **tutti insieme**
l'errore cala in modo *prevedibile*: niente salti misteriosi, una curva liscia,
come una ricetta che riesce sempre un po' meglio se si raddoppia ogni
ingrediente. Ma i miglioramenti sono lenti: ogni raddoppio del calcolo lima
l'errore solo di qualche punto percentuale, il tre e mezzo per cento circa. La
seconda scoperta, del 2022
{cite}`hoffmann2022training`, è che gli ingredienti vanno **bilanciati**: è
inutile fare una torta con dieci uova e un cucchiaio di farina. La regola
pratica emersa è circa **20 pezzi di testo per ogni manopola del modello**,
contati in quei pezzi in cui il testo viene spezzato, i token, non in parole:
per un modello da sette miliardi di manopole vuol dire centoquaranta miliardi
di token da leggere. Molti modelli dell'epoca erano enormi ma avevano studiato
troppo poco, e la dimostrazione ha un nome, perché è un modello costruito
apposta: si chiama **Chinchilla**, ha quattro volte meno manopole del suo
rivale diretto (**Gopher**), ha letto quasi cinque volte più testo a parità di
ore di calcolo, e lo batte. Da allora "più grande" non basta più: conta il
rapporto fra modello e dati.
`````

`````{tab} Superiore
Kaplan e colleghi {cite}`kaplan2020scaling` osservano che la loss di test **per
token**, cioè la $\bar{\mathcal{L}}$ di poco fa, segue leggi di
potenza in ciascuna delle tre risorse, quando le altre due non fanno da collo
di bottiglia:

$$
\bar{\mathcal{L}}(N) \approx \left(\frac{N_c}{N}\right)^{\alpha_N}, \qquad
\bar{\mathcal{L}}(D) \approx \left(\frac{D_c}{D}\right)^{\alpha_D}, \qquad
\bar{\mathcal{L}}(C_{\min}) \approx
\left(\frac{C_c}{C_{\min}}\right)^{\alpha_C^{\min}},
$$

dove $N$ è il numero di parametri (esclusi gli embedding), $D$ il numero di
token di addestramento, $C_{\min}$ il calcolo **speso in modo ottimo** fra
modello e dati (che è la grandezza per cui gli autori raccomandano di fare
previsioni: la curva a batch size fissato ha un esponente suo, $\approx
0{,}057$), $N_c$, $D_c$ e $C_c$ costanti di normalizzazione, e gli esponenti
misurati
valgono $\alpha_N \approx 0{,}076$, $\alpha_D \approx 0{,}095$,
$\alpha_C^{\min} \approx 0{,}050$. Esponenti piccoli: raddoppiare il calcolo,
speso bene, riduce
la loss di circa il 3,4% ($2^{-0{,}050} \approx 0{,}966$), poco, ma con
sorprendente affidabilità su molti ordini di grandezza, il che permette di
*estrapolare*: si può stimare la loss di un modello da miliardi di parametri
addestrando modelli da milioni. Un'avvertenza sulla forma: scritte così le tre
leggi mandano la loss a **zero** ingrandendo abbastanza, il che è falso. Valgono
dentro il regime misurato, e la forma completa che si usa per estrapolare
davvero somma un termine costante irriducibile, l'entropia del linguaggio
stesso, che nessuna quantità di parametri toglie di mezzo: è quella che
Hoffmann e colleghi usano nel paragrafo qui sotto.

Hoffmann e colleghi {cite}`hoffmann2022training` correggono la conclusione
operativa di Kaplan (che suggeriva di privilegiare $N$): rifacendo le misure
trovano che, a budget di calcolo $C$ fissato, il minimo della loss si ottiene
scalando all'incirca $N_{\mathrm{opt}} \propto C^{0{,}5}$ e
$D_{\mathrm{opt}} \propto C^{0{,}5}$ (modello e dati crescono *insieme*, in
proporzione) con un rapporto quasi costante $D/N \approx 20$ token per
parametro. La verifica empirica è **Chinchilla**: 70 miliardi di parametri
addestrati su 1.400 miliardi di token che, a parità di calcolo, superano
Gopher (280 miliardi di parametri, circa 300 miliardi di token). Col senno del
2022, GPT-3 era fortemente sotto-addestrato: per 175 miliardi di parametri la
regola prescriverebbe circa 3.500 miliardi di token, contro i 300 effettivi.
`````

```{figure} ../figures/chinchilla-2022.svg
:name: fig-chinchilla
:alt: "Due barre che ripartiscono lo stesso budget di calcolo in modo diverso. Gopher spende in 280 miliardi di parametri e 300 miliardi di token: molta capacità, poca esperienza. Chinchilla spende in 70 miliardi di parametri e 1.400 miliardi di token: meno capacità, molta più esperienza."
:width: 90%

Lo stesso budget, due modi di spenderlo. Chinchilla è quattro volte più
piccolo di Gopher e ha letto quasi cinque volte di più: a parità di calcolo,
vince il secondo.
```

Il confronto di {numref}`fig-chinchilla` è il motivo per cui «quanti
parametri ha?» ha smesso di essere una domanda sensata da sola. Un numero di
parametri dice quanta capacità c'è, non quanta ne è stata riempita, e due
modelli con lo stesso cartellino possono aver letto quantità di testo
incomparabili.

Una parola di prudenza, per intanto: quello che le leggi di scala garantiscono
è che il modello sbaglierà un po' meno a indovinare la parola dopo, non che a
una certa taglia gli spunterà una certa abilità. Che le abilità spuntino
davvero all'improvviso è una faccenda controversa, e ha una sezione tutta sua
in fondo a questa pagina.

## Generare: l'arte di scegliere la parola dopo

Un modello addestrato non sceglie una parola: distribuisce probabilità su tutte
quelle possibili. Per *scrivere*, però, bisogna sceglierne una davvero, poi
un'altra, poi un'altra ancora, e il modo in cui la si sceglie cambia moltissimo
il testo che ne esce.

I due modi classici li abbiamo già visti nel capitolo sull'NLP. Il primo è
prendere ogni volta la parola più probabile e tirare dritto (si chiama
*greedy*, cioè ingorda). Il secondo è meno miope: invece di impegnarsi subito,
si portano avanti in parallelo le $k$ continuazioni più promettenti, si vede
come proseguono, e solo alla fine si tiene la migliore delle $k$ (è la *beam
search*, «ricerca a fascio»). Piccola nota che vale per tutta la sezione:
lettere come $k$, $n$, $T$, $p$ stanno per numeri che sceglie chi usa il
modello, non per costanti di natura; sono manopole, e il libro dice ogni volta
a che cosa servono.

Per la traduzione questi due modi funzionano; per la generazione libera
(scrivere un racconto, rispondere a una domanda aperta) falliscono in un modo
curioso, documentato da Holtzman e colleghi {cite}`holtzman2020curious`. Il
testo che *massimizza* la probabilità è noioso, ripetitivo, e finisce spesso
incastrato a girare in tondo («Il gatto nero salta sul muro. Il gatto nero
salta sul muro. Il gatto nero...»). Gli stessi autori sono andati a
controllare come è fatto il testo scritto da noi: hanno preso pagine di prosa
umana e hanno chiesto al modello, parola per parola, quanto le riteneva
probabili. Il risultato è che noi *non* scriviamo la sequenza più probabile,
mai: le nostre frasi sono punteggiate di scelte a media e bassa probabilità, ed
è quello che le rende vive. Per scrivere come noi, allora, il modello deve
**rischiare**: non scegliere sempre il massimo, ma *campionare*, cioè tirare un
dado truccato secondo le sue probabilità. E il trucco del dado si può regolare.

```{figure} ../figures/generazione-autoregressiva.gif
:name: fig-generazione-autoregressiva
:alt: "Animazione: la frase «Il gatto» viene evidenziata come contesto, sotto compaiono quattro candidati con le rispettive probabilità in barre orizzontali, il più probabile sale a formare la parola successiva, e il ciclo si ripete fino a «Il gatto nero salta sul muro»."
:width: 90%

Il ciclo della generazione: leggi tutto quello che c'è scritto finora, ottieni
un voto di probabilità per ogni parola possibile, scegline una, riattaccala in
fondo e ricomincia da capo.
```

Nella {numref}`fig-generazione-autoregressiva` la scelta cade ogni volta sul
candidato più probabile: è la decodifica *greedy*, quella che produce i loop
di cui sopra. Le manopole che seguono servono esattamente a **non** far
vincere sempre la barra più lunga.

`````{tab} Elementare
Tre manopole, tutte con la stessa filosofia: quanta sorpresa vogliamo?

**La temperatura** regola quanto è truccato il dado. Riprendiamo il gioco: «Il
gatto nero salta sul...» con quattro esiti; muro (probabile), tetto, divano,
pigiama (assurdo). A temperatura *bassa* il dado è truccatissimo: esce «muro»
quasi sempre, il testo è prudente e un po' monotono. A temperatura 1 il dado
rispetta le probabilità del modello. A temperatura *alta* il dado si
"stempera" verso l'equità: ogni tanto esce «pigiama», e il testo si fa
creativo fino allo sproposito. Bassa = affidabile e prevedibile; alta = vivace
e rischiosa. Il nome viene dalla fisica, non dal caldo: nelle formule che
descrivono un gas compare esattamente la stessa manopola, e alzarla vuol dire
far muovere le particelle più a caso. Qui non si scalda niente; è la formula a
essere la stessa.

**Top-k e top-p** tolgono dal mazzo le carte peggiori prima di pescare. Con il
**top-k** tieni solo le $k$ carte migliori: con $k=2$, nel nostro gioco, pesca
solo tra «muro» e «tetto»; «pigiama» non può proprio uscire. Il difetto: $k$ è
fisso, ma a volte le carte buone sono due, a volte venti. Il **top-p** è più
furbo: tieni le carte migliori finché, sommando le loro probabilità, copri
(diciamo) il 90% del totale; a volte bastano due carte, a volte ne servono
dieci, il mazzo si adatta da solo alla situazione. È il metodo proposto
proprio nell'articolo del "caso curioso", con il nome di *nucleus sampling*:
si pesca solo dal nucleo buono del mazzo, e la coda di parole strampalate (che
una per una vale poco, ma sommata pesa) sparisce.
`````

`````{tab} Superiore
Il modello produce per ogni token del vocabolario un punteggio grezzo (il
**logit** $z_i$); la softmax con **temperatura** $T$ lo converte in
probabilità:

$$
p_i = \frac{\exp(z_i / T)}{\sum_{j=1}^{|\mathcal{V}|} \exp(z_j / T)},
$$

dove $|\mathcal{V}|$ è la dimensione del vocabolario (il calligrafico distingue
l'insieme delle parole possibili dalla matrice $\mathbf{V}$ dei *value*), e
$T > 0$ scala i logit prima
della normalizzazione: per $T \to 0$ la distribuzione collassa sul massimo
(si torna alla scelta greedy), per $T \to \infty$ tende all'uniforme.
Esempio numerico completo con quattro parole e logit
$\mathbf{z} = (2{,}0;\; 1{,}0;\; 0{,}0;\; -2{,}0)$:

| parola   | $z_i$  | $T=0{,}5$ | $T=1$   | $T=2$   |
|----------|--------|-----------|---------|---------|
| muro     | $2{,}0$  | $0{,}867$ | $0{,}657$ | $0{,}474$ |
| tetto    | $1{,}0$  | $0{,}117$ | $0{,}242$ | $0{,}287$ |
| divano   | $0{,}0$  | $0{,}016$ | $0{,}089$ | $0{,}174$ |
| pigiama  | $-2{,}0$ | $0{,}000$ | $0{,}012$ | $0{,}064$ |

A $T=0{,}5$ «muro» passa da 0,657 a 0,867 e «pigiama» praticamente scompare; a
$T=2$ la distribuzione si appiattisce e «pigiama» sale a 0,064: un errore ogni
sedici parole, in media.

Il **top-k** limita il campionamento ai $k$ token con probabilità maggiore,
rinormalizzando: con $k=2$ restano muro e tetto con
$0{,}657/0{,}899 \approx 0{,}731$ e $0{,}242/0{,}899 \approx 0{,}269$. Il
**top-p** (*nucleus sampling* {cite}`holtzman2020curious`) sceglie invece il
più piccolo insieme di token (il *nucleo*) la cui probabilità cumulata
raggiunge la soglia $p$:

$$
\mathcal{V}_p = \text{il più piccolo } \mathcal{V}' \subseteq \mathcal{V} \text{ tale che }
\sum_{i \in \mathcal{V}'} p_i \ge p,
$$

ordinando per probabilità decrescente. Con $p=0{,}9$ e $T=1$: la cumulata fa
$0{,}657 \to 0{,}899 \to 0{,}988$; siccome $0{,}899 < 0{,}9$, serve anche
«divano», e il nucleo è {muro, tetto, divano}, rinormalizzato a
$(0{,}665;\; 0{,}245;\; 0{,}090)$. A differenza di $k$, la taglia del nucleo
si adatta alla forma della distribuzione: pochi candidati quando il modello è
sicuro, molti quando è incerto. Holtzman e colleghi mostrano che è la
strategia che meglio riproduce le statistiche del testo umano nella
generazione di testi lunghi. Le tre manopole si compongono: prima la
temperatura, poi i tagli top-k e top-p, infine il campionamento.
`````

```{figure} ../figures/decoding-sampling.svg
:name: fig-decoding-sampling
:alt: "Tre istogrammi della stessa distribuzione sulla parola successiva dopo «Il gatto nero salta sul», a temperatura 0,5, 1 e 2: a temperatura bassa quasi tutta la probabilità va su «muro», a temperatura alta la distribuzione si appiattisce; sul pannello centrale un riquadro tratteggiato racchiude il nucleo del top-p pari a 0,9, che esclude «pigiama»."
:width: 100%

Gli stessi voti di probabilità a tre temperature: più la temperatura è bassa,
più il dado è truccato verso «muro»; il riquadro tratteggiato racchiude le
parole che restano nel mazzo con il taglio top-p.
```

In {numref}`fig-decoding-sampling` si vede il compromesso a colpo d'occhio: la
temperatura decide quanto la distribuzione è appuntita, il top-p dove tagliare
la coda. In pratica si usano insieme (temperature attorno a 0,7–0,8 e $p$
attorno a 0,9 sono punti di partenza comuni) e la scelta dipende dal compito:
per una risposta fattuale conviene un dado truccato, per una poesia un dado
più libero.

## Il segnalibro: la KV cache

C'è un dettaglio pratico che a prima vista sembra un disastro. La generazione
è autoregressiva: come visto nella sezione sull'architettura, il token
prodotto rientra come input e si ricomincia. Ma allora, per ogni nuovo token,
il Transformer dovrebbe rileggere *tutta* la sequenza, e i conti
dell'attenzione sul prefisso sarebbero sempre gli stessi, rifatti da capo a
ogni passo. Nessun sistema reale lavora così: tutti usano la **KV cache**. Il
nome dice già tutto, una volta sciolto: K e V sono la *key* e il *value* della
sezione sull'attenzione, cioè l'etichetta con cui ogni parola si fa trovare e
l'informazione che consegna; *cache* è la dispensa dove si tiene a portata di
mano quello che si è già preparato.

`````{tab} Elementare
Quando leggi un romanzo, non ricominci da pagina 1 ogni volta che ne giri una:
usi un segnalibro, e in testa ti restano gli appunti su quello che è successo.
La KV cache è il segnalibro del modello: gli "appunti" che l'attenzione ha già
calcolato sulle parole lette restano in memoria, e per ogni parola nuova il
modello calcola solo gli appunti *di quella parola*, consultando i vecchi
senza rifarli. Il risparmio è enorme: è la differenza tra girare pagina e
rileggere il libro da capo a ogni pagina. Il prezzo, però, è lo spazio: gli
appunti si accumulano, e più lunga è la conversazione, più scaffali servono
per tenerli. È uno dei motivi per cui i contesti lunghi costano: non solo più
calcolo, ma memoria che cresce parola dopo parola, e che per conversazioni
molto lunghe arriva a pesare quanto il modello stesso.
`````

`````{tab} Superiore
Nella self-attention causale, il token in posizione $t$ calcola la sua query
$\mathbf{q}_t$ e attende alle coppie $(\mathbf{k}_j, \mathbf{v}_j)$ con
$j \le t$. Le key e le value
delle posizioni passate **non cambiano** quando la sequenza si allunga: si
possono calcolare una volta e conservare. La cache memorizza, per ciascuno
degli $L$ strati (e per ogni testa), le matrici $\mathbf{K}$ e $\mathbf{V}$ del
prefisso; al passo $t$ si calcolano solo
$\mathbf{q}_t, \mathbf{k}_t, \mathbf{v}_t$ del token nuovo, si appendono
$\mathbf{k}_t, \mathbf{v}_t$ alla cache e si valuta l'attenzione di
$\mathbf{q}_t$ contro le $t$ chiavi
accumulate: costo $O(t\,d + d^2)$ per passo **e per strato**, invece
dell'$O(t^2 d + t\,d^2)$ di
un forward rifatto da capo sul prefisso. Sull'intera generazione di $n$ token il
totale per strato scende da $O(n^3 d + n^2 d^2)$ a $O(n^2 d + n\,d^2)$; per il
modello intero si moltiplica per gli $L$ strati, ed è la forma con cui vanno
lette le percentuali del capoverso seguente.

I due termini vanno tenuti distinti, perché è facile portarsi via la morale
sbagliata. Il termine quadratico è quello dell'attenzione ed è ineliminabile,
come nel confronto con le RNN; il termine $d^2$ è quello delle matrici dense
(proiezioni e feed-forward), ed è quello che **domina** a tutte le lunghezze di
contesto correnti. Per un modello da 7 miliardi di parametri con $L = 32$,
$d = 4096$ e feed-forward SwiGLU, l'attenzione vale il 4% del calcolo per token
a 1.024 token di contesto e il 14% a 4.096; i due termini si pareggiano
attorno ai 25.000. È il motivo per cui, come dirà la sezione sui modelli a
esperti, generare testo è un lavoro **limitato dalla memoria** più che
dall'aritmetica: il collo di bottiglia è leggere i pesi a ogni parola, non
confrontare la parola con quelle prima.

Il conto della memoria: per ogni token servono
$2 \cdot L \cdot d_{\text{model}}$ numeri ($\mathbf{K}$ e $\mathbf{V}$, per
strato). Per un
modello da 7 miliardi di parametri con $L = 32$ e $d_{\text{model}} = 4096$,
in precisione a 16 bit, sono $2 \times 32 \times 4096 \times 2$ byte
$\approx 0{,}5$ MB per token: una finestra di 4.096 token occupa circa 2 GB
*per ogni sequenza nel batch*, da sommare ai ~14 GB dei pesi. È il motivo per
cui varianti come la *multi-query* e la *grouped-query attention* (molte teste
per le query, poche per key e value) sono diventate standard nei modelli
recenti: riducono proprio la cache. E spiega un'asimmetria che si nota usando
i servizi commerciali: elaborare il prompt (il *prefill*, parallelo) e
generare i token (la *decodifica*, sequenziale e affamata di memoria) hanno
costi molto diversi.
`````

## Programmare con le parole: prompt e in-context learning

Abbiamo visto, nella sezione su GPT, BERT e T5, la scoperta sorprendente di
GPT-3. Il **prompt** è tutto quello che si scrive al modello prima che
risponda: la richiesta, il testo su cui deve lavorare, le istruzioni su come
farlo. Ebbene, descrivere un compito lì dentro (magari con due o tre esempi già
svolti) basta spesso a farglielo eseguire, senza toccargli un solo numero
interno {cite}`brown2020language`.

```{figure} ../figures/gpt-2-2019.svg
:name: fig-gpt2-multitask
:alt: "Al centro un unico modello linguistico, addestrato soltanto a prevedere la parola successiva. Da esso si diramano più compiti diversi (rispondere a domande, riassumere, tradurre) che il modello esegue senza essere stato addestrato specificamente su nessuno di essi."
:width: 96%

Un solo obiettivo, molti compiti. Nessuno ha insegnato a questo modello a
riassumere: il riassunto era già dentro l'esercizio di prevedere la parola
dopo, su un web che contiene testi e i loro riassunti.
```

L'osservazione di {numref}`fig-gpt2-multitask` precede GPT-3 e ne spiega la
premessa. Se il corpus è abbastanza vasto, contiene già esempi impliciti di
quasi ogni compito linguistico, e un modello che lo prevede bene ha dovuto,
per forza, imparare a farli. Vale la pena soffermarsi su quanto è strano. Per
tutto il libro, "adattare un modello" ha significato addestrarlo, cioè
mostrargli esempi, misurare quanto sbaglia e spostargli i numeri interni
un'inezia alla volta, per giorni. Qui no: il compito viene *descritto in
italiano* (o in inglese), e il modello, completando il testo nel modo più
probabile, di fatto lo esegue. Il prompt è diventato un'interfaccia di
programmazione in linguaggio naturale: si "programma" il modello scrivendo, e
l'*in-context learning* (imparare dal contesto della singola richiesta) non
era un obiettivo di progetto, ma un comportamento comparso con la scala.

L'onestà impone però di dire che questa "programmazione" è fragile.
Riformulare la stessa domanda con parole diverse può cambiare la risposta;
l'ordine degli esempi nel prompt influenza il risultato; una frase
d'istruzione che funziona con un modello può fallire con un altro. Non c'è un
manuale del linguaggio di programmazione, perché non è un linguaggio di
programmazione. Sotto non c'è nessuno che esegue un ordine: c'è una macchina
che, dato tutto quello che ha davanti, calcola quanto è probabile ogni
possibile continuazione, e ne sceglie una. Se cambi quello che ha davanti,
cambiano le probabilità; l'istruzione non è un comando, è un pezzo di contesto
come tutti gli altri, e il confine tra "istruire" e "suggestionare" è sottile.
Il *prompt engineering*
(l'artigianato di formulare richieste che funzionano) è utile, ma va preso per
quello che è: una collezione di euristiche su un sistema che nessuno, finora,
sa programmare con garanzie.

## Misurare un gigante

Come si valuta un modello del genere? La misura più naturale è la stessa cosa
che il modello sta imparando a fare: quanto resta indeciso sulla parola
successiva. Si chiama **perplessità**, e si legge come il numero di facce del
dado che il modello si ritrova in mano a ogni passo. Perplessità 1 vuol dire
che sa sempre esattamente che cosa viene dopo; perplessità 20, che è indeciso
come chi tira un dado a venti facce. Meno facce, modello migliore. È un altro
modo di raccontare l'errore di cui parlavano le leggi di scala, cioè
esattamente la cosa che il pretraining fa scendere, e resta il termometro più
affidabile della qualità *come modello di linguaggio*.

`````{tab} Elementare
Il dado è tutto quello che serve. Se un modello ha perplessità 20 su un certo
testo, vuol dire che, in media, a ogni parola si trova nella condizione di uno
che deve indovinare fra venti possibilità equiprobabili. Un modello migliore
scende a dieci, uno molto migliore a cinque, e nessuno arriverà mai a uno,
perché il linguaggio ha una sua imprevedibilità di fondo che nessun modello può
togliere: quella che Shannon misurava all'inizio di questa pagina.

L'unica avvertenza è che il numero non si confronta fra testi diversi. La
perplessità su una raccolta di leggi e quella su un romanzo non si possono
mettere sulla stessa riga, perché le leggi sono scritte in modo molto più
prevedibile: confrontare due modelli ha senso solo sullo stesso testo.
`````

`````{tab} Superiore
In formula, con la stessa definizione del capitolo di matematica e di quello
sull'NLP, la perplessità è $2^H$, dove $H$ è la cross-entropia media per token
**espressa in bit**. La parola «bit» non è un vezzo, ed è il punto in cui si
sbaglia: la loss di questa sezione usa il logaritmo naturale, quindi la loss
per token $\bar{\mathcal{L}}$ è in *nat* e non in bit.
Per passare dagli uni agli altri si moltiplica per $\log_2 e = 1{,}4427$, cioè
$H = \bar{\mathcal{L}}/\ln 2$, e la perplessità si scrive allora più comodamente
$e^{\bar{\mathcal{L}}}$. Chi invece mette $\bar{\mathcal{L}}$ tale e quale
all'esponente di 2 sta usando un esponente più piccolo del dovuto di quel
fattore, e ottiene la perplessità vera elevata a $\ln 2 = 0{,}693$: su una
perplessità di 20 ne stampa 8.
`````

Ma nemmeno la perplessità dice quasi nulla di ciò che interessa a chi il modello
lo usa: sa rispondere a domande di diritto? Sa tradurre? Per questo si
affiancano batterie di test standardizzati, i **benchmark**: il più citato è
stato a lungo MMLU {cite}`hendrycks2021measuring`, cinquantasette materie di
domande a scelta multipla, dal diritto alla fisica.

I benchmark vanno però letti con un sospetto specifico: la **contaminazione**
dei dati di test.

```{figure} ../figures/benchmark-llm-come-si-bara.svg
:name: fig-contaminazione
:alt: "Un grande insieme, i dati di addestramento raccolti dal web, e un piccolo insieme, le domande del benchmark. I due si sovrappongono in una zona evidenziata: le domande di test che compaiono anche nel corpus di addestramento. Su quella zona il punteggio misura ciò che il modello ricorda, non ciò che sa fare."
:width: 88%

La zona di sovrapposizione è il problema. Non serve malafede perché si formi:
basta che il benchmark sia pubblico e il corpus sia il web.
```

Come si vede in {numref}`fig-contaminazione`, la contaminazione non è un
imbroglio ma una conseguenza quasi inevitabile del modo in cui si raccolgono i
dati. Ed è per questo che è difficile da escludere: per dimostrare che una
domanda *non* è nel corpus bisognerebbe poterlo ispezionare tutto, e chi
pubblica un punteggio quasi mai pubblica anche i dati. Se il modello ha
studiato l'intero web, è probabile che abbia già *visto* le domande del test,
che quindi misura la memoria, non la competenza. Non è un rischio teorico: gli
stessi autori di GPT-3 dedicano al
problema un'analisi accurata, e ammettono che, per un bug nella procedura di
pulizia, parte delle sovrapposizioni tra corpus e benchmark non era stata
rimossa {cite}`brown2020language`. Da allora il problema è solo cresciuto:
ogni benchmark pubblicato sul web è, per il modello successivo, potenziale
materiale di studio. Quando leggi «il modello X supera il modello Y di due
punti», la domanda giusta è: su dati che nessuno dei due aveva mai visto? Per
una trattazione sistematica di valutazione e decoding rimandiamo a Jurafsky e
Martin {cite}`jurafsky2026speech`, il riferimento moderno del settore.

### Le abilità emergenti, e il dubbio che siano un miraggio

C'è un'osservazione che ha fatto molto discutere. Su certi compiti (aritmetica
a più cifre, ragionamento a più passi) i modelli piccoli vanno **a zero**, e
poi, superata una certa scala, la prestazione **salta** all'improvviso. Non
migliora gradualmente: appare. Da qui il nome *abilità emergenti*, e l'idea
inquietante che ingrandendo un modello si ottengano capacità non previste.

```{figure} ../figures/emergent-abilities.svg
:name: fig-capacita-emergenti
:alt: "Due grafici affiancati che misurano lo stesso modello al crescere della scala. A sinistra, con una metrica discontinua come la risposta esatta sì o no, la curva resta piatta e poi salta di colpo: sembra un'abilità comparsa all'improvviso. A destra, con una metrica continua come la distanza di edit, la stessa crescita appare come un miglioramento graduale e regolare."
:width: 100%

Lo stesso modello, due righelli. Il gradino di sinistra non è nei dati: è
nel modo di dare i voti, che assegna zero a una risposta quasi giusta finché
non diventa esatta.
```

I due grafici di {numref}`fig-capacita-emergenti` sono lo stesso modello,
misurato in due modi diversi, e la differenza fra loro è tutto quello che
questa sezione ha da dire.

`````{tab} Elementare

L'obiezione arrivata dopo è più interessante della scoperta, ed è un'ottima
lezione su come si misura.

Molti di quei compiti sono valutati **tutto-o-niente**: la risposta a
$134 \times 27$ è giusta solo se tutte le cifre sono giuste. Con una metrica
del genere, un modello che passa dallo sbagliare tre cifre allo sbagliarne una
prende zero in entrambi i casi: poi azzecca l'ultima e prende uno. Il salto è
nella *pagella*, non nel modello.

Se si misura la stessa identica prestazione con un metro graduale (quante
cifre sono corrette, o la probabilità assegnata alla risposta giusta), la
curva diventa liscia e prevedibile. Il miglioramento c'era ed era continuo: la
metrica lo nascondeva.

La morale vale ben oltre gli LLM: **una metrica discontinua trasforma un
progresso graduale in un miracolo apparente.**

`````

`````{tab} Superiore

Le abilità emergenti sono state documentate da Wei e colleghi
{cite}`wei2022emergent`; la critica del *miraggio* è di Schaeffer, Miranda e
Koyejo {cite}`schaeffer2023emergent`.

L'argomento è preciso. Se l'errore per token cala regolarmente con la scala
(come le leggi di potenza della sezione precedente predicono), l'accuratezza
su una risposta di $n$ token valutata in modo esatto vale circa $p^{\,n}$, con
$p$ la probabilità per token. Una funzione del genere resta schiacciata vicino
a zero e poi si impenna: la discontinuità è **prodotta dalla non linearità
della metrica**, non dal modello. Sostituendo l'accuratezza esatta con la
distanza di edit, o con la log-verosimiglianza della risposta corretta, in
molti casi l'emergenza svanisce.

Il dibattito non è chiuso, e conviene tenere distinte due affermazioni. Che gran
parte delle curve «a salto» siano artefatti di misura è ormai ben argomentato.
Che *nessun* cambiamento qualitativo avvenga con la scala è un'affermazione più
forte e non dimostrata: fenomeni come l'*in-context learning* restano difficili
da ridurre a un miglioramento puramente continuo.

La ricaduta pratica è però univoca, e riguarda chiunque valuti un modello:
**usare metriche continue quando si studia un andamento**, e diffidare di
qualunque grafico in cui una capacità «appare». La prima domanda da farsi è come
è stata misurata.

`````

## In pratica: campionare con PyTorch

Le tre manopole della scelta stanno in una funzione di venti righe. Non c'è
nessun modello scaricato: i punteggi grezzi (i *logit*, cioè i voti che il
modello dà a ogni parola possibile prima di trasformarli in probabilità) sono
scritti a mano, così resta in vista solo il meccanismo. Chi legge al livello
Elementare può saltare i due blocchi che seguono senza perdere il filo: dicono
in Python le stesse tre manopole già raccontate con il dado truccato, le carte
tolte dal mazzo e il nucleo che si adatta.

```python
import torch

def sample_next(logits, temperature=1.0, top_k=None, top_p=None):
    """Sceglie il prossimo token dai logits (tensore di forma [V])."""
    if temperature == 0:                       # caso limite: scelta greedy
        return int(torch.argmax(logits))

    logits = logits / temperature              # 1) temperatura

    if top_k is not None:                      # 2) top-k: solo i k migliori
        soglia = torch.topk(logits, top_k).values[-1]
        logits = logits.masked_fill(logits < soglia, float("-inf"))

    if top_p is not None:                      # 3) top-p: il nucleo
        ordinati, indici = torch.sort(logits, descending=True)
        probs_ord = torch.softmax(ordinati, dim=-1)
        cumulate = torch.cumsum(probs_ord, dim=-1)
        # fuori dal nucleo i token oltre la soglia (il migliore resta sempre)
        fuori = (cumulate - probs_ord) > top_p
        ordinati[fuori] = float("-inf")
        logits = torch.full_like(logits, float("-inf")).scatter(0, indici, ordinati)

    probs = torch.softmax(logits, dim=-1)      # 4) di nuovo una distribuzione
    return int(torch.multinomial(probs, num_samples=1))

# --- l'esempio numerico del testo: muro, tetto, divano, pigiama ---
logits = torch.tensor([2.0, 1.0, 0.0, -2.0])
for T in (0.5, 1.0, 2.0):
    print(f"T={T}:", torch.softmax(logits / T, dim=-1).round(decimals=3))
# T=0.5: [0.867, 0.117, 0.016, 0.000]   il dado si trucca verso "muro"
# T=2.0: [0.474, 0.287, 0.174, 0.064]   il dado si appiattisce
```

E un mini-ciclo di generazione, con un "modello" giocattolo al posto di un
vero Transformer, la struttura del loop è identica a quella reale:

```python
torch.manual_seed(0)

vocab = ["il", "gatto", "nero", "salta", "sul", "muro",
         "tetto", "divano", "e", "poi", "dorme", "."]

def modello_giocattolo(sequenza):
    # un vero LLM restituirebbe qui i logits dell'ultima posizione;
    # noi generiamo logits riproducibili a partire dall'ultimo token
    g = torch.Generator().manual_seed(sequenza[-1])
    return torch.randn(len(vocab), generator=g)

sequenza = [vocab.index("il")]
for _ in range(8):
    logits = modello_giocattolo(sequenza)   # in un LLM vero: forward + KV cache
    prossimo = sample_next(logits, temperature=0.8, top_p=0.9)
    sequenza.append(prossimo)

print(" ".join(vocab[i] for i in sequenza))
# testo sgrammaticato, ovviamente: il "modello" è un generatore casuale.
# Ma il ciclo (forward, campiona, appendi, ripeti) è quello vero.
```

Sostituisci `modello_giocattolo` con un Transformer addestrato e hai, per
davvero, il cuore della generazione di ChatGPT e simili.

C'è però un ultimo, decisivo tassello mancante. Il modello che esce dal
pretraining è un *completatore*, non un assistente: alla domanda «Qual è la
capitale della Francia?» può rispondere «Qual è la capitale della Spagna? Qual
è la capitale dell'Italia?»: perché nel web le liste di domande abbondano, e
completare la lista è probabilissimo. Trasformare il completatore in un
interlocutore che risponde, segue istruzioni e rifiuta le richieste dannose
richiede una seconda fase di addestramento, con ricette proprie: è il
**post-training**, e ha una sezione tutta sua poco più avanti. Prima però
conviene fermarsi su un'idea architetturale che le leggi di scala rendono quasi
obbligata. Crescere conviene, questo lo abbiamo visto; ma per scrivere una sola
parola il modello deve moltiplicarla, piano dopo piano, per **tutte** le sue
manopole, e più le manopole sono tante più quel giro costa: la bolletta di ogni
singola parola sale insieme alla taglia del modello. A meno di non fare in modo
che ogni parola passi solo per un pezzetto delle manopole, che è esattamente
l'idea della sezione seguente.

`````{tab} Elementare
```{admonition} Da ricordare
:class: important
- Un grande modello linguistico gioca il **gioco di Shannon** su scala
  industriale: coprire l'ultima parola di una frase vera e provare a
  indovinarla, miliardi di volte, su una biblioteca raccolta dal web e
  ripulita. Nessuno gli corregge i compiti: la soluzione era già nel testo.
- Più **manopole** interne, più testo da leggere e più ore di calcolo danno un
  modello migliore, e in modo prevedibile. Ma gli ingredienti vanno
  **bilanciati**, e la regola pratica è una ventina di pezzi di testo per ogni
  manopola: «quanto è grande?», da sola, ha smesso di essere una domanda
  sensata.
- Per scrivere, il modello **non** prende sempre la parola più probabile:
  verrebbe un testo noioso, che si incarta a ripetere sé stesso. Tira un dado,
  e tre manopole decidono quanto quel dado è truccato (la **temperatura**) e
  quante carte restano nel mazzo da cui pescare (il **top-k** e il **top-p**).
- Il **segnalibro** evita di rileggere tutto da capo a ogni parola: gli appunti
  già presi restano in memoria. Si risparmia tempo e si paga in spazio, ed è
  uno dei motivi per cui le conversazioni lunghe costano.
- Il **prompt** è un modo di programmare scrivendo: potente e fragile insieme.
  E i punteggi dei test vanno letti sapendo che un modello che ha studiato
  tutto il web potrebbe aver già visto le domande.
- Quello che esce da tutto questo è un **completatore** di testo, non un
  assistente: per quello serve una seconda fase, ed è la sezione sul
  post-training.
```
`````

`````{tab} Superiore
```{admonition} Da ricordare
:class: important
- Un LLM gioca il gioco di Shannon su scala industriale: **cross-entropia sul
  token successivo** come unica supervisione, su corpora web filtrati e
  deduplicati (GPT-3: ~300 miliardi di token, 60% da Common Crawl).
- **Leggi di scala**: la loss cala come una legge di potenza in parametri,
  dati e calcolo {cite}`kaplan2020scaling`; il bilanciamento ottimale è circa
  **20 token per parametro** {cite}`hoffmann2022training`. Sulle "capacità
  emergenti" il dibattito è aperto: prudenza.
- Massimizzare la probabilità **degenera** in ripetizioni
  {cite}`holtzman2020curious`: si campiona con **temperatura** (quanto è
  truccato il dado), **top-k** (solo le $k$ carte migliori) e **top-p** (il
  nucleo che copre probabilità cumulata $p$).
- La **KV cache** conserva key e value già calcolati: niente ricalcoli, ma
  memoria che cresce col contesto (~0,5 MB per token in un modello da 7
  miliardi di parametri); ecco perché i contesti lunghi costano.
- Il **prompt** è programmazione in linguaggio naturale: potente e fragile
  insieme. I **benchmark** vanno letti col sospetto della **contaminazione**
  dei dati di test.
- Il pretraining produce un completatore, non un assistente: per quello serve
  il post-training.
```
`````
