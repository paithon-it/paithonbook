# I grandi modelli linguistici

Nel 1951 Claude Shannon pubblicò uno degli esperimenti più casalinghi della
storia dell'informatica {cite}`shannon1951prediction`: copriva una riga di un
testo e chiedeva a una persona di indovinare, lettera dopo lettera, come
continuava. Il risultato, contando i tentativi necessari, è più sorprendente di
quanto sembri. Senza sapere niente del contesto, indovinare una
lettera dell'alfabeto inglese è come scegliere fra ventisei possibilità:
un'incertezza di poco meno di cinque bit (il numero di domande sì o no che
servirebbero a risolverla). Sapendo quel che c'è scritto prima,
quell'incertezza scende a circa uno: una domanda sola, cioè quanto un
testa-o-croce. Il contesto toglie, insomma, i quattro quinti
dell'incertezza, che la {doc}`sezione sulla teoria dell'informazione
</Matematica/teoria-informazione>` chiama entropia. Settant'anni dopo, GPT-3
{cite}`brown2020language` gioca lo stesso gioco, indovinare come continua un
testo, su una scala molto diversa. A confronto, un adolescente di tredici anni,
secondo le stime usate dalla ricerca sull'acquisizione del linguaggio, è stato
esposto a meno di 100 milioni di parole, quasi tutte ascoltate
{cite}`warstadt2023call`; GPT-3 in addestramento ha visto circa 300 miliardi di
*token* (i pezzi in cui viene spezzato il testo), che per l'inglese
corrispondono a poco più di 200 miliardi di parole: più di duemila volte tanto.

Il terreno è già preparato. Nella {doc}`sezione sulle famiglie di modelli
<multimodalita>` la famiglia GPT è quella dei Transformer *decoder-only*,
addestrati a predire il token successivo; e GPT-3, arrivato a quella scala,
sapeva eseguire un compito nuovo solo perché glielo si descriveva nel prompt,
magari con due o tre esempi svolti: è la capacità *few-shot*. Fra il gioco di
Shannon e GPT-3 c'è un gradino intermedio, GPT-2 {cite}`radford2019language`,
1,5 miliardi di parametri addestrati nel 2019 su pagine web segnalate dagli
utenti di Reddit, con un titolo programmatico: *Language Models are
Unsupervised Multitask Learners*, cioè i modelli di linguaggio imparano da soli
a svolgere molti compiti. Restano da spiegare da dove vengono i dati, perché
«più grande» funziona in modo così prevedibile da meritarsi delle *leggi*, come
si sceglie in pratica il token da scrivere e quale accorgimento di ingegneria
rende la generazione sostenibile.

## Il pretraining su scala web: i dati

Il **pre-addestramento** (*pretraining*) è la prima di due fasi: il modello
impara la lingua e una parte delle conoscenze sul mondo, e solo dopo, con il
post-training, impara a svolgere compiti. Trecento miliardi di token non stanno
in nessuna enciclopedia, e il grosso può venire solo dal web, che però mescola
testo di qualità, spam, pagine duplicate e testo generato da macchine, per cui
va filtrato, deduplicato e ripesato prima dell'addestramento, e questa
preparazione è una parte rilevante del lavoro di chi costruisce un grande
modello.

`````{tab} Elementare
Una lingua straniera si può imparare con *un solo tipo di esercizio*: frasi da
completare. Nessuna grammatica, nessun insegnante, nessuna correzione a penna
rossa: solo miliardi di esercizi di completamento, ricavati coprendo a turno
ogni parola di frasi vere. «Il gatto nero salta sul ___»: provi, sbagli,
aggiusti, scopri la parola e passi a coprire quella dopo. Il voto che conta è
quanto si sbaglia in media su una parola: il totale degli sbagli, con miliardi
di parole, direbbe soltanto che le parole erano tante.

Con abbastanza esercizi, per completare bene *devi* assorbire
ortografia, grammatica, modi di dire, e perfino nozioni sul mondo: non puoi
completare «la capitale della Francia è ___» senza sapere di Parigi. Il bello è
che gli esercizi si fabbricano senza fatica: qualunque testo esistente è già un
esercizio con la soluzione inclusa.

Serve però una biblioteca sterminata e *pulita*: se la soffitta è piena di
doppioni, l'allievo impara a memoria invece di imparare la lingua; se è piena
di spazzatura, impara la spazzatura. Per questo, prima di studiare, si butta
via moltissimo: pagine duplicate, testo generato da macchine, contenuti di
bassa qualità. A decidere che cosa buttare non c'è nessuno che legge: c'è un
giudice automatico, a cui sono stati mostrati due o tre scaffali scelti bene, e
che tiene le pagine somiglianti a quelli.

Dopo la pulizia i testi non contano
tutti allo stesso modo: del mucchio raccolto dal web si legge una parte, mentre
gli scaffali migliori si ripassano più volte, così contano nello studio più di
quanto la loro mole direbbe.
`````

`````{tab} Superiore
La materia prima tipica è **Common Crawl**, un'istantanea periodica e
liberamente scaricabile del web, che va però raffinata: filtri di qualità (per
GPT-3, un classificatore addestrato a distinguere le pagine simili a corpora di
riferimento dal resto del crawl), deduplicazione fuzzy (i duplicati gonfiano la
memorizzazione e falsano la valutazione) e rimozione di contenuti indesiderati.
Il dataset di GPT-3 {cite}`brown2020language` è una miscela pesata a mano, e il
peso non segue la dimensione: il Common Crawl filtrato dà il 60% dei token
visti in addestramento; le fonti ritenute migliori, e più piccole, vengono
ripassate più volte (WebText2 quasi tre, Wikipedia più di tre) e pesano il 22%
e il 3%. Completano la miscela due corpora di libri, all'8% ciascuno (la somma
delle quote fa 101 per arrotondamento). Il testo è segmentato in sub-word con
l'algoritmo BPE descritto nella {doc}`sezione sui tokenizzatori
</NaturalLanguageProcessing/tokenizzatori>` {cite}`sennrich2016neural`.

L'unica supervisione è il testo stesso: si minimizza la cross-entropia sul
token successivo,

$$
\mathcal{L}(\theta) = -\sum_{t=1}^{n} \log p_\theta(x_t \mid x_1, \dots, x_{t-1}),
$$

dove $x_t$ è il token in posizione $t$, $p_\theta$ è la distribuzione prodotta
dal Transformer con parametri $\theta$ (softmax sull'intero vocabolario) e la
somma corre sugli $n$ token del corpus, che però è spezzato in sequenze
indipendenti: il condizionamento si ferma alla finestra di contesto, e dentro
ciascuna sequenza riparte da capo.

Quando servirà la loss per token, cioè la stessa quantità divisa per $n$, la
scriveremo $\bar{\mathcal{L}} = \mathcal{L}/n$: la distinzione conta, perché il
calcolo della perplessità richiede la loss media per token, e scambiarla con
quella totale porterebbe a elevare il risultato alla potenza $n$. È la stessa
`nn.CrossEntropyLoss` dei capitoli precedenti, applicata a un problema di
classificazione con decine di migliaia di classi (le parole possibili) ripetuto
miliardi di volte; la sua riduzione predefinita è la media, quindi restituisce
$\bar{\mathcal{L}}$.

Poiché non serve nessuna etichetta umana, si parla di apprendimento
auto-supervisionato. Sui rischi di corpora così raccolti (bias, contenuti
tossici, opacità) il dibattito è aperto e acceso {cite}`bender2021dangers`.
`````

Quell'idea, che qualunque testo esistente sia già un esercizio con la soluzione
inclusa, è l’**apprendimento auto-supervisionato**, e non riguarda soltanto il
linguaggio. È lo stesso meccanismo con cui il {doc}`capitolo sulla visione
</VisioneArtificiale/overview>` ha fatto imparare a rappresentare le immagini
senza etichette, con cui si impara a riconoscere il parlato senza
trascrizioni ({doc}`rappresentazioni auto-supervisionate dell'audio
</Audio/rappresentazioni-auto-supervisionate>`) e con cui si allineano le
immagini alle loro didascalie ({doc}`allineare due spazi
</VisioneLinguaggio/allineare-due-spazi>`). Il
{doc}`capitolo sull'auto-supervisione </AutoSupervisione/overview>` lo tratta
nel dettaglio e discute perché funzioni su tanti problemi: a
ogni token il modello riceve una correzione, un segnale molto più fitto di
un'etichetta per fotografia o di un «hai vinto» a fine partita.

## La ricetta a tre ingredienti: le leggi di scala

Perché proprio *grandi* modelli? È una regolarità che qualcuno ha misurato.

Un modello di linguaggio ha un solo compito: indovinare la parola dopo. Su
questo compito gli si può dare un voto preciso: gli si fa leggere del testo che
non ha mai visto e si guarda quanta probabilità aveva assegnato alle parole che
poi sono comparse. Se ne dava tanta, ha indovinato bene; se ne dava poca, male.
Quel valore, che diminuisce quando il modello impara, è l'errore (in gergo la
*loss*), da cui deriva anche la misura usata per valutare il modello finito:
la perplessità.

Tra il 2020 e il 2022 due lavori hanno misurato, con la pazienza di centinaia
di addestramenti, come cambia quell'errore al crescere delle risorse, e hanno
trovato curve così regolari da chiamarle **leggi di scala**.

`````{tab} Elementare
La ricetta di un modello di linguaggio ha tre ingredienti: la taglia del
modello, la quantità di testo su cui studia, e il calcolo (quante ore di
computer può bruciare). La taglia si misura in parametri, cioè i numeri interni
che l'addestramento sposta un'inezia alla volta: «un modello da sette miliardi»
ha sette miliardi di parametri. La scoperta del 2020 {cite}`kaplan2020scaling`
è che aumentando gli ingredienti tutti insieme l'errore cala in modo
*prevedibile*: niente salti misteriosi, una curva liscia, come una ricetta che
riesce sempre un po’ meglio se si raddoppia ogni ingrediente.

Prevedibile è la parola che conta: si prova la ricetta in piccolo, si guarda di
quanto migliora a ogni raddoppio, e si sa già come verrà quella grande prima di
infornarla. Il guadagno però è lento: ogni raddoppio del calcolo lima l'errore
di poco più del tre per cento. La ricetta perfetta non arriva mai: sotto un
certo punto non si scende comunque, perché una quota dell'incertezza appartiene
alla lingua stessa, e nessuna quantità di parametri la toglie di mezzo.

La seconda scoperta, del 2022 {cite}`hoffmann2022training`, è che gli
ingredienti vanno bilanciati: è inutile fare una torta con dieci uova e un
cucchiaio di farina. La regola pratica emersa è circa 20 pezzi di testo per
ogni parametro del modello, contati in quei pezzi in cui il testo viene
spezzato, i token, non in parole: ad esempio a un modello da sette miliardi di
parametri ne servono centoquaranta miliardi da leggere.

Molti modelli dell'epoca erano enormi ma avevano studiato troppo poco,
e la dimostrazione ha un nome, perché è un modello costruito apposta: si chiama
**Chinchilla**, ha un quarto dei parametri del suo rivale diretto (**Gopher**),
ha letto quasi cinque volte più testo a parità di ore di calcolo, e lo batte.
Da allora "più grande" non basta più: conta il rapporto fra modello e dati.

Venti pezzi per parametro, però, è la proporzione che rende
minimo il costo dell'addestramento: chi deve poi usare il modello moltissimo ne
fa uno più piccolo e gli fa leggere molto di più, perché ogni risposta di un
modello piccolo costa meno.
`````

`````{tab} Superiore
Kaplan e colleghi {cite}`kaplan2020scaling` osservano che la loss di test per
token, cioè la $\bar{\mathcal{L}}$ definita per il pre-addestramento, segue
leggi di potenza in ciascuna delle tre risorse, quando le altre due non fanno
da collo di bottiglia:

$$
\bar{\mathcal{L}}(N) \approx \left(\frac{N_c}{N}\right)^{\alpha_N}, \qquad
\bar{\mathcal{L}}(D) \approx \left(\frac{D_c}{D}\right)^{\alpha_D}, \qquad
\bar{\mathcal{L}}(C_{\min}) \approx
\left(\frac{C_c}{C_{\min}}\right)^{\alpha_C^{\min}},
$$

dove $N$ è il numero di parametri (esclusi gli embedding), $D$ il numero di
token di addestramento, $C_{\min}$ il calcolo ripartito in modo ottimo fra
modello e dati, $N_c$, $D_c$ e $C_c$ costanti di normalizzazione, mentre gli
esponenti misurati valgono $\alpha_N \approx 0{,}076$, $\alpha_D \approx
0{,}095$, $\alpha_C^{\min} \approx 0{,}050$. Gli autori raccomandano di fondare
le previsioni su $C_{\min}$: la curva a dimensione del batch fissata segue un
esponente distinto, $\approx 0{,}057$.

Gli esponenti sono piccoli: raddoppiare il calcolo riduce la loss
di circa il 3,4% ($2^{-0{,}050} \approx 0{,}966$), un guadagno modesto, ma
regolare su molti ordini di grandezza, che permette di *estrapolare*: si può
stimare la loss di un modello da miliardi di parametri addestrando modelli da
milioni.

Scritte in questa forma, le tre leggi
porterebbero la loss a zero su scale arbitrariamente grandi. Valgono in realtà
solo dentro il regime misurato, e la forma completa impiegata per estrapolare
somma un termine costante irriducibile: l'entropia del linguaggio stesso, che
nessuna quantità di parametri può annullare. È la forma adottata da Hoffmann e
colleghi.

Hoffmann e colleghi {cite}`hoffmann2022training` correggono la conclusione
operativa di Kaplan (che suggeriva di privilegiare $N$): rifacendo le misure
trovano che, a budget di calcolo $C$ fissato, il minimo della loss si ottiene
scalando all'incirca $N_{\mathrm{opt}} \propto C^{0{,}5}$ e
$D_{\mathrm{opt}} \propto C^{0{,}5}$ (modello e dati crescono *insieme*, in
proporzione) con un rapporto quasi costante $D/N \approx 20$ token per
parametro.

La verifica empirica è **Chinchilla**: 70 miliardi di parametri
addestrati su 1.400 miliardi di token che, a parità di calcolo, superano
Gopher (280 miliardi di parametri, circa 300 miliardi di token). Col senno del
2022, GPT-3 era fortemente sotto-addestrato.

Il calcolo parte dal costo di un
addestramento, $C \approx 6ND$ operazioni in virgola mobile: due per parametro
e per token nel passaggio in avanti, quattro nella retropropagazione. Ponendo
$D = 20N$ si ricava $C \approx 120N^2$, quindi $N_{\text{opt}} \approx
\sqrt{C/120}$. In codice bastano poche righe:

```python
# il conto di Hoffmann e colleghi: C = 6 N D, e all'ottimo D = 20 N
N, D = 175e9, 300e9                    # GPT-3: parametri e token
C = 6 * N * D
N_opt = (C / 120) ** 0.5               # da C = 6 N (20 N) = 120 N^2
print(f"calcolo di GPT-3: {C:.2e} operazioni")
print(f"all'ottimo: {N_opt/1e9:.0f} miliardi di parametri, "
      f"{20 * N_opt/1e9:.0f} miliardi di token")
print(f"token per 175 miliardi di parametri: {20 * N/1e9:.0f} miliardi")
print(f"Llama 3 8B su 15 000 miliardi di token: {15e12/8e9:.0f} per parametro")
```

```text
calcolo di GPT-3: 3.15e+23 operazioni
all'ottimo: 51 miliardi di parametri, 1025 miliardi di token
token per 175 miliardi di parametri: 3500 miliardi
Llama 3 8B su 15 000 miliardi di token: 1875 per parametro
```

Il budget di calcolo di GPT-3, pari a $3{,}15\cdot10^{23}$ operazioni, sarebbe
stato impiegato in modo ottimale su circa 51 miliardi di parametri e poco più
di mille miliardi di token; tenuti fissi i 175 miliardi, la regola ne
richiederebbe 3.500.

Kaplan prescriveva invece $N_{\text{opt}} \propto
C^{0{,}73}$. Hoffmann e colleghi attribuivano la differenza al programma del
learning rate, che Kaplan teneva uguale per tutti i modelli; Porian e colleghi
{cite}`porian2024resolving` hanno poi riprodotto la legge di Kaplan e ne hanno
trovato altre tre cause: il calcolo dell'ultimo strato, che Kaplan non contava,
la durata del riscaldamento e una regolazione dell'ottimizzatore che dipende
dalla scala. Corretti questi tre punti i due risultati coincidono, e un
decadimento accurato del learning rate non risulta essenziale.

Hoffmann adatta la forma $\bar{\mathcal{L}}(N,D) = E + A/N^{a} + B/D^{b}$, con
$E \approx 1{,}69$ il termine irriducibile, $A$ e $B$ due costanti di
adattamento e $a \approx 0{,}34$, $b \approx 0{,}28$. Sono i valori arrotondati
dell'articolo, e Besiroglu e colleghi {cite}`besiroglu2024chinchilla`,
ricostruendo i punti dai grafici, trovano che quei valori non descrivono bene
l'andamento: il loro adattamento dà $E \approx 1{,}82$, $a \approx 0{,}35$, $b
\approx 0{,}37$, quindi $N_{\text{opt}} \propto C^{0{,}51}$ e un rapporto
ottimo vicino a venti token per parametro. L'esponente attorno a un mezzo resta
valido, i valori arrotondati no.

Il rapporto 20 minimizza il costo dell'addestramento, non quello dell'uso: chi
prevede di servire molte richieste addestra di proposito un modello più piccolo
su molti più token. La famiglia Llama 3 adotta questa scelta anche per i
modelli più piccoli {cite}`grattafiori2024llama3`: quello da 8 miliardi di
parametri ha letto circa 15.000 miliardi di token, quasi 1.900 per parametro.
`````

```{figure} ../figures/chinchilla-2022.svg
:name: fig-chinchilla
:alt: "Due barre che ripartiscono lo stesso budget di calcolo in modo diverso. Gopher spende in 280 miliardi di parametri e 300 miliardi di token: molta capacità, poca esperienza. Chinchilla spende in 70 miliardi di parametri e 1.400 miliardi di token: meno capacità, molta più esperienza."
:width: 90%

Lo stesso budget, due modi di spenderlo. Chinchilla è quattro volte più piccolo
di Gopher e ha letto quasi cinque volte di più: a parità di calcolo, vince
Chinchilla.
```

Il confronto di {numref}`fig-chinchilla` è il motivo per cui «quanti parametri
ha?» non basta più come domanda. Infatti un numero di parametri dice quanta
capacità c'è, non quanta ne è stata riempita, e due modelli con lo stesso
cartellino possono aver letto quantità di testo incomparabili.

Le leggi di scala descrivono come cala la loss di test dentro il regime in cui
sono state misurate, cioè quanto il modello sbaglierà meno a indovinare la
parola dopo; non garantiscono che a una certa scala si manifesti una nuova
abilità. Che certe capacità emergano all'improvviso resta un punto controverso.

## Generare: l'arte di scegliere la parola dopo

Un modello addestrato non sceglie una parola: distribuisce probabilità su tutte
quelle possibili. Per *scrivere*, però, bisogna sceglierne una, poi
un'altra, poi un'altra ancora, e il modo in cui la si sceglie cambia moltissimo
il testo che ne esce.

I due modi classici sono quelli della {doc}`sezione sulla traduzione con le
reti </NaturalLanguageProcessing/seq2seq-traduzione>`. Il primo è prendere ogni
volta la parola più probabile e tirare dritto (si chiama *greedy*, cioè
ingorda). Il secondo è meno miope: invece di impegnarsi subito, si portano
avanti in parallelo le $k$ continuazioni più promettenti, si vede come
proseguono, e solo alla fine si tiene la migliore delle $k$ (è la *beam
search*, «ricerca a fascio»). Le quantità come $k$, $T$ e $p$ sono
iperparametri della decodifica: li sceglie chi usa il modello, e non si
imparano.

Per la traduzione questi due modi funzionano; per la generazione libera
(scrivere un racconto, rispondere a una domanda aperta) falliscono in un modo
curioso, documentato da Holtzman e colleghi {cite}`holtzman2020curious`. Il
testo che *massimizza* la probabilità è noioso, ripetitivo, e finisce spesso
incastrato a girare in tondo («Il gatto nero salta sul muro. Il gatto nero
salta sul muro. Il gatto nero...»). Gli stessi autori sono andati a controllare
come è fatto il testo scritto da noi: hanno preso pagine di prosa umana e hanno
chiesto al modello, parola per parola, quale probabilità assegnasse a ciascuna.
Risulta che noi non scegliamo mai la sequenza più probabile: le nostre frasi
sono punteggiate di scelte a media e bassa probabilità, e sono queste a
renderle vive. Per scrivere come noi, quindi, il modello deve rischiare, cioè
**campionare**: tirare un dado truccato secondo le sue probabilità invece di
scegliere sempre il massimo. Il trucco del dado si può regolare.

```{figure} ../figures/generazione-autoregressiva.gif
:name: fig-generazione-autoregressiva
:alt: "Animazione: la frase «Il gatto» viene evidenziata come contesto, sotto compaiono quattro candidati con le rispettive probabilità in barre orizzontali, il più probabile sale a formare la parola successiva, e il ciclo si ripete fino a «Il gatto nero salta sul muro»."
:width: 90%

Il ciclo della generazione: leggi tutto il testo scritto finora, ottieni
una probabilità per ogni parola possibile, scegline una, riattaccala in fondo e
ricomincia da capo.
```

Nella {numref}`fig-generazione-autoregressiva` la scelta cade ogni volta sul
candidato più probabile: è la decodifica *greedy*, quella che produce i loop in
cui il testo si incarta. Temperatura, top-k e top-p servono a non far vincere
sempre la barra più lunga.

`````{tab} Elementare
Tre manopole, tutte con la stessa filosofia: quanta sorpresa vogliamo?

La temperatura regola quanto è truccato il dado. Riprendiamo il gioco, «Il
gatto nero salta sul...», con quattro possibili esiti: muro (probabile), tetto,
divano, pigiama (assurdo). A temperatura *bassa* il dado è truccatissimo: esce
«muro» quasi sempre, il testo è prudente e un po’ monotono. A temperatura 1 il
dado rispetta le probabilità del modello. A temperatura *alta* il dado si
"stempera" verso l'equità: ogni tanto esce «pigiama», e il testo si fa creativo
fino allo sproposito. Bassa = affidabile e prevedibile; alta = vivace e
rischiosa.

Il nome viene dalla fisica, non dal caldo: nelle formule che
descrivono un gas compare la stessa manopola, e alzarla fa muovere le
particelle più a caso. Qui non si scalda niente; la formula è la stessa.

Top-k e top-p tolgono dal mazzo le carte peggiori prima di pescare. Con il
**top-k** tieni solo le $k$ carte migliori: con $k=2$, nel nostro
gioco, peschi solo tra «muro» e «tetto»; «pigiama» non può uscire. Il difetto è
che $k$ è fisso, mentre le carte buone sono a volte due, a volte venti. Il
**top-p** è più furbo: tieni le carte migliori finché, sommando le loro
probabilità, copri (diciamo) il 90% del totale; a volte bastano due carte, a
volte ne servono dieci, il mazzo si adatta alla situazione. È il metodo
proposto nell'articolo del "caso curioso", con il nome di *nucleus sampling*:
si pesca solo dal nucleo buono del mazzo, e la coda di parole strampalate (che
una per una vale poco, ma sommata pesa) sparisce.

Le tre manopole si usano insieme, e in quest'ordine: prima si decide quanto
truccare il dado, poi si tolgono le carte dal mazzo, e solo alla fine si pesca.
`````

`````{tab} Superiore
Il modello produce per ogni token del vocabolario un punteggio grezzo (il
logit $z_i$); la softmax con temperatura $T$ lo converte in
probabilità:

$$
p_i = \frac{\exp(z_i / T)}{\sum_{j=1}^{|\mathcal{V}|} \exp(z_j / T)},
$$

dove $|\mathcal{V}|$ è la dimensione del vocabolario (il calligrafico distingue
l'insieme delle parole possibili dalla matrice $\mathbf{V}$ dei *value*), e
$T > 0$ scala i logit prima
della normalizzazione: per $T \to 0$ la distribuzione collassa sul massimo
(si torna alla scelta greedy), per $T \to \infty$ tende all'uniforme.
Consideriamo, come esempio numerico completo, quattro parole con logit
$\mathbf{z} = (2{,}0;\; 1{,}0;\; 0{,}0;\; -2{,}0)$:

| parola   | $z_i$  | $T=0{,}5$ | $T=1$   | $T=2$   |
|----------|--------|-----------|---------|---------|
| muro     | $2{,}0$  | $0{,}867$ | $0{,}657$ | $0{,}474$ |
| tetto    | $1{,}0$  | $0{,}117$ | $0{,}242$ | $0{,}287$ |
| divano   | $0{,}0$  | $0{,}016$ | $0{,}089$ | $0{,}174$ |
| pigiama  | $-2{,}0$ | $0{,}000$ | $0{,}012$ | $0{,}064$ |

A $T=0{,}5$ «muro» passa da 0,657 a 0,867 e «pigiama» praticamente
scompare; a $T=2$ la distribuzione si appiattisce e «pigiama» sale a 0,064: un
errore ogni sedici parole, in media.

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

ordinando i token per probabilità decrescente. Con $p=0{,}9$ e $T=1$, la
cumulata fa $0{,}657 \to 0{,}899 \to 0{,}988$; siccome $0{,}899 < 0{,}9$, serve
anche «divano», e il nucleo è {muro, tetto, divano}, rinormalizzato a
$(0{,}665;\; 0{,}245;\; 0{,}090)$. A differenza di $k$, la taglia del nucleo si
adatta alla forma della distribuzione: pochi candidati quando il modello è
sicuro, molti quando è incerto. Holtzman e colleghi mostrano che è la strategia
che meglio riproduce le statistiche del testo umano nella generazione di testi
lunghi. Le tre operazioni si compongono: prima la temperatura, poi i tagli
top-k e top-p, infine il campionamento.
`````

```{figure} ../figures/decoding-sampling.svg
:name: fig-decoding-sampling
:alt: "Tre istogrammi della stessa distribuzione sulla parola successiva dopo «Il gatto nero salta sul», a temperatura 0,5, 1 e 2: a temperatura bassa quasi tutta la probabilità va su «muro», a temperatura alta la distribuzione si appiattisce; sul pannello centrale un riquadro tratteggiato racchiude il nucleo del top-p pari a 0,9, che esclude «pigiama»."
:width: 100%

Le stesse probabilità a tre temperature: più la temperatura è bassa, più il
dado è truccato verso «muro»; il riquadro tratteggiato racchiude le parole che
restano nel mazzo con il taglio top-p.
```

In {numref}`fig-decoding-sampling` si vede il compromesso a colpo d'occhio: la
temperatura decide quanto la distribuzione è appuntita, il top-p dove tagliare
la coda. In pratica si usano insieme (temperature tra 0,7 e 0,8 e $p$
attorno a 0,9 sono punti di partenza comuni) e la scelta dipende dal compito:
per una risposta fattuale conviene un dado truccato, per una poesia un dado
più libero.

## La KV cache

C'è un dettaglio pratico che a prima vista sembra un disastro. La generazione
è autoregressiva, come nel decoder della {doc}`sezione sulla struttura del
Transformer <architettura>`: il token prodotto rientra come input e si
ricomincia. Per ogni nuovo token il Transformer dovrebbe allora rileggere
*tutta* la sequenza, e le operazioni dell'attenzione sul prefisso sarebbero
sempre le stesse, rifatte da capo a ogni passo. Nessun sistema reale lavora
così:
tutti usano la KV cache, che la {doc}`sezione sull'attenzione in pratica
<attenzione-in-pratica>` ha descritto come struttura dati. Il nome dice già
tutto, una volta sciolto: K e V sono la *key* e il *value* dell'attenzione, i
vettori che ogni posizione offre come chiave e come valore; *cache* è la
memoria in cui si conserva il calcolo già fatto.

`````{tab} Elementare
È il taccuino della sezione sull'attenzione in pratica, uno per ogni piano e
per ogni lettore: per ogni parola letta ci restano scritte l'etichetta con cui
si fa trovare e l'informazione che consegna, e per una parola nuova il modello
scrive soltanto le sue due righe, consultando le vecchie senza rifarle. Il
risultato è identico a quello ottenuto rileggendo tutto, perché
nessuna parola può guardare avanti: quanto accade a pagina trecento non cambia
gli appunti di pagina dodici. Il risparmio è enorme, la differenza fra girare
pagina e rileggere il libro da capo a ogni pagina.

Lo svantaggio è lo spazio: il
taccuino si allunga a ogni parola, mezzo megabyte a
parola per un modello da sette miliardi, e per conversazioni molto lunghe
arriva a pesare quanto il modello stesso.

Anche col taccuino, però, resta una fatica che non si può togliere: per ogni
parola scritta il modello ripassa tutti i suoi parametri, miliardi di numeri,
e li deve rileggere dalla memoria ogni volta. Consultare il taccuino, al
confronto, costa poco, e resta la parte minore finché il testo davanti non
diventa lunghissimo: oltre qualche decina di migliaia di parole è il taccuino
a pesare di più.
`````

`````{tab} Superiore
La KV cache evita di ricalcolare chiavi e valori del prefisso: il costo di un
passo scende, per strato, da $O(t^2 d + t\,d^2)$ a $O(t\,d + d^2)$, e quello
dell'intera generazione di $n$ token da $O(n^3 d + n^2 d^2)$ a
$O(n^2 d + n\,d^2)$, come ricava la sezione sull'attenzione in pratica; per il
modello intero si moltiplica per gli $n_{\text{strati}}$ strati.

I due termini vanno tenuti distinti per non trarne la conclusione sbagliata. Il
termine in $t\,d$ è quello dell'attenzione ed è ineliminabile, come nel
confronto con le RNN; il termine in $d^2$ è quello delle matrici dense
(proiezioni e feed-forward), e domina finché il contesto è più corto di qualche
decina di migliaia di token. Ad esempio, per un modello da 7 miliardi di
parametri, con le forme di Llama 2 7B ($n_{\text{strati}} = 32$, $d =
d_{\text{model}} = 4096$, feed-forward SwiGLU con $d_{\text{ff}} = 11\,008$,
vocabolario di $32\,000$ token), si contano due operazioni per parametro delle
matrici dense e $4\,t\,d$ per strato per l'attenzione su $t$ token di contesto:

```python
# le forme di un modello da sette miliardi (quelle di Llama 2 7B)
n_strati, d, d_ff, vocab = 32, 4096, 11008, 32000

per_strato = 4 * d * d + 3 * d * d_ff         # proiezioni e FFN SwiGLU
parametri = n_strati * per_strato + 2 * vocab * d   # embedding e uscita
densi = n_strati * per_strato + vocab * d     # l'embedding non fa conti
print(f"parametri: {parametri/1e9:.2f} miliardi")

# per token: 2 operazioni per parametro denso, e 4 t d per strato
# di attenzione (Q K^T e A V) su t token di contesto
flop_densi = 2 * densi
for t in (1024, 4096, 32768, 131072):
    flop_att = 4 * t * d * n_strati
    quota = flop_att / (flop_att + flop_densi)
    print(f"contesto {t:>6}: attenzione {quota:6.1%}")
print(f"pareggio a {flop_densi/(4 * d * n_strati):.0f} token")

# a una richiesta per volta si rileggono tutti i pesi, 2 byte l'uno
print(f"operazioni per byte di pesi letti: {flop_densi/(2 * parametri):.2f}")
```

```text
parametri: 6.74 miliardi
contesto   1024: attenzione   3.9%
contesto   4096: attenzione  14.0%
contesto  32768: attenzione  56.5%
contesto 131072: attenzione  83.9%
pareggio a 25204 token
operazioni per byte di pesi letti: 0.98
```

A 1.024 token di contesto l'attenzione vale il 4% del calcolo per token, a
4.096 il 14%; i due termini si pareggiano attorno ai 25.000 token, e oltre
quella soglia il calcolo per token è fatto soprattutto di attenzione, l'84% a
131.072.

A una richiesta per volta, però, il tempo dipende da un altro fattore: a
ogni token vanno riletti dalla memoria tutti i pesi, due byte per parametro a
16 bit, per circa due operazioni per parametro, cioè un'operazione per byte
letto, molto sotto il ginocchio del {doc}`modello roofline
</GPU/gerarchia-memoria>`, che per una scheda grafica moderna in mezza
precisione supera le centocinquanta operazioni per byte. Generare testo, a una
richiesta per volta, è quindi un lavoro limitato dalla memoria più che
dall'aritmetica, come dirà anche la {doc}`sezione sui modelli a esperti
<mixture-of-experts>`: il collo di bottiglia è leggere i pesi a ogni parola,
non confrontare la parola con quelle prima.

Il calcolo della memoria riprende quello riportato nella sezione
sull'attenzione in pratica: con una testa di chiave e valore per ogni testa di
query servono $2\,n_{\text{strati}}\,d_{\text{model}}$ numeri per token, mezzo
megabyte a 16 bit per il modello da sette miliardi, e una finestra di 4.096
token occupa circa 2 GB *per ogni sequenza nel batch*, da sommare ai ~14 GB dei
pesi. Con meno teste di chiave e valore questo fabbisogno si riduce (con GQA a
otto gruppi, a un quarto); da qui l'adozione di MQA, GQA e MLA come standard
nei modelli recenti.

Lo stesso ragionamento spiega un'asimmetria che si nota usando
i servizi commerciali: elaborare il prompt (il *prefill*, parallelo) e generare
i token (la *decodifica*, sequenziale e affamata di memoria) hanno costi molto
diversi.
`````

## Programmare con le parole: prompt e in-context learning

Abbiamo visto, fra le {doc}`famiglie di modelli <multimodalita>`, la scoperta
di GPT-3: descrivere un compito nel prompt, con due o tre esempi già svolti,
basta spesso a farlo eseguire {cite}`brown2020language`.

```{figure} ../figures/gpt-2-2019.svg
:name: fig-gpt2-multitask
:alt: "Al centro un unico modello linguistico, addestrato soltanto a prevedere la parola successiva. Da esso si diramano più compiti diversi (rispondere a domande, riassumere, tradurre) che il modello esegue senza essere stato addestrato specificamente su nessuno di essi."
:width: 96%

Un solo obiettivo, molti compiti. Nessuno ha insegnato a questo modello a
riassumere o a tradurre; GPT-2 ci riesce soltanto in modo grezzo (nel riassunto
di articoli di giornale, di poco meglio di tre frasi prese a caso), ma ci
riesce senza aver visto un solo esempio del compito.
```

L'osservazione di {numref}`fig-gpt2-multitask` precede GPT-3 e ne spiega la
premessa. Se il corpus è abbastanza vasto, contiene esempi impliciti di molti
compiti linguistici, e un modello che prevede bene il testo può imparare a
svolgerli: è la congettura degli autori di GPT-2, che i loro risultati
sostengono solo in parte (55 di F1 sulle domande di CoQA, 5 di BLEU nella
traduzione dall'inglese al francese, un riassunto che batte di poco tre frasi
scelte a caso). GPT-3 la riprende con un modello oltre cento volte più grande e
con gli esempi nel prompt.

Il fatto resta strano. Fin qui, "adattare un modello" ha significato
addestrarlo, cioè mostrargli esempi, misurare quanto sbaglia e spostargli i
numeri interni un'inezia alla volta, per giorni. Qui no: il compito viene
*descritto in italiano* (o in inglese), e il modello, completando il testo nel
modo più probabile, di fatto lo esegue. In altri termini, il prompt è diventato
un'interfaccia di programmazione in linguaggio naturale: si "programma" il
modello scrivendo.

L’**in-context learning** (imparare dal
contesto della singola richiesta) non era un obiettivo di progetto: nessuna
loss lo richiede, e migliora con la scala, anche se i suoi meccanismi di base,
come le *teste di induzione* che ricopiano il seguito di uno schema già
comparso nel contesto, si formano già in modelli piccolissimi
{cite}`olsson2022induction`.

Questa "programmazione" è però fragile. L'ordine degli esempi nel
prompt può far oscillare l'accuratezza da valori vicini allo stato dell'arte
fino al livello del caso {cite}`lu2022fantastically`; il solo formato della
richiesta (separatori, maiuscole, spazi) la sposta fino a 76 punti su un
modello da 13 miliardi di parametri {cite}`sclar2024quantifying`; e le
etichette degli esempi contano meno di quanto ci si aspetterebbe, tanto che
sostituirle a caso cambia poco {cite}`min2022rethinking`. Una frase
d'istruzione che funziona con un modello può fallire con un altro. Non c'è un
manuale del linguaggio di programmazione, perché non è un linguaggio di
programmazione.

Sotto non c'è nessuno che esegue un ordine: c'è una macchina
che, dato tutto il contesto, calcola quanto è probabile ogni
possibile continuazione, e ne sceglie una. Se cambi il contesto,
cambiano le probabilità; l'istruzione è un pezzo di contesto come tutti gli
altri e non un comando, e il confine tra "istruire" e "suggestionare" è sottile.
Il **prompt engineering**
(l'artigianato di formulare richieste che funzionano) è utile, ma resta
una collezione di euristiche su un sistema che nessuno, finora,
sa programmare con garanzie.

## Misurare un gigante

Come si valuta un modello del genere? La misura più naturale è quella che il
modello sta imparando a minimizzare: quanto resta indeciso sulla parola
successiva. Si chiama perplessità ed è l'esponenziale della loss media per
token, cioè il numero di scelte equiprobabili fra cui il modello si
troverebbe, in media, a ogni passo. Con perplessità 1 il modello sa sempre quale
parola viene dopo: più il valore è basso, minore è l'incertezza.
È l'errore delle leggi di scala scritto in un'altra forma, che il
pretraining fa scendere, e resta la misura più affidabile della qualità
*come modello di linguaggio*.

`````{tab} Elementare
Basta un dado per capirla. Se un modello ha perplessità 20 su un certo testo, a
ogni parola si trova, in media, nella condizione di chi deve indovinare fra
venti possibilità equiprobabili. Un modello migliore
scende a dieci, uno molto migliore a cinque, e nessuno arriverà mai a uno,
perché il linguaggio ha una sua imprevedibilità di fondo che nessun modello può
togliere: quella che Shannon misurava coprendo una riga di testo e chiedendo a
una persona di indovinare come andava avanti.

L'avvertenza è che il numero non si confronta fra testi diversi. Ad esempio, la
perplessità su una raccolta di leggi e quella su un romanzo non si possono
mettere sulla stessa riga, perché le leggi sono scritte in modo molto più
prevedibile: confrontare due modelli ha senso solo sullo stesso testo e
spezzato allo stesso modo. L'unica alternativa per sfuggire alla scelta dei
pezzi è contare l'indecisione lettera per lettera, che non dipende da come il
testo viene diviso.
`````

`````{tab} Superiore
In formula, con la stessa definizione della
{doc}`sezione sulla teoria dell'informazione </Matematica/teoria-informazione>`
e di quella {doc}`sui modelli n-gram
</NaturalLanguageProcessing/modelli-ngram>`, la perplessità è $2^H$, dove $H$
è la cross-entropia media per token
espressa in bit. La parola «bit» è il punto in cui si
sbaglia: la cross-entropia del pretraining si scrive col logaritmo naturale,
quindi la loss per token $\bar{\mathcal{L}}$ è in *nat* e non in bit.

Per passare dagli uni agli altri si moltiplica per $\log_2 e = 1{,}4427$, cioè
$H = \bar{\mathcal{L}}/\ln 2$, e la perplessità si scrive allora più comodamente
$e^{\bar{\mathcal{L}}}$. Chi invece mette $\bar{\mathcal{L}}$ direttamente
all'esponente di 2 sta usando un esponente più piccolo del dovuto di quel
fattore, e ottiene la perplessità vera elevata a $\ln 2 = 0{,}693$: su una
perplessità di 20 ne stampa 8.

Il valore dipende poi dal corpus e da come il
testo è stato spezzato in token, quindi due modelli si confrontano così solo
sullo stesso testo e con la stessa segmentazione. Con tokenizzatori diversi si
normalizza per i byte invece che per i token, e si usano i *bit per byte*,

$$
\mathrm{BPB} = \frac{\bar{\mathcal{L}}\; n_{\text{token}}}{n_{\text{byte}}\,\ln 2},
$$

dove $n_{\text{token}}$ e $n_{\text{byte}}$ sono i token e i byte dello stesso
testo: il numeratore è la cross-entropia dell'intero testo, in nat, e dividerla
per i byte invece che per i token dà una misura che si confronta anche fra
modelli che spezzano il testo in modo diverso.
`````

Nemmeno la perplessità dice però quasi nulla di ciò che interessa a chi usa il
modello: sa rispondere a domande di diritto? Sa tradurre? Per questo si
affiancano batterie di test standardizzati, i **benchmark**: il più citato è
stato a lungo MMLU (*Massive Multitask Language Understanding*)
{cite}`hendrycks2021measuring`, cinquantasette materie di domande a scelta
multipla, dal diritto alla fisica.

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
dati. Per questo è difficile da escludere: per dimostrare che una
domanda *non* è nel corpus bisognerebbe poterlo ispezionare tutto, e chi
pubblica un punteggio quasi mai pubblica anche i dati. Se il modello ha
studiato l'intero web, è probabile che abbia già *visto* le domande del test,
che quindi misura la memoria, non la competenza.

Il rischio è concreto: gli
stessi autori di GPT-3 dedicano al
problema un'analisi accurata, e ammettono che, per un bug nella procedura di
pulizia, parte delle sovrapposizioni tra corpus e benchmark non era stata
rimossa {cite}`brown2020language`. Da allora il problema è solo cresciuto:
ogni benchmark pubblicato sul web è, per il modello successivo, potenziale
materiale di studio.

Quando leggi «il modello X supera il modello Y di due
punti», la domanda giusta è: su dati che nessuno dei due aveva mai visto?

Una
trattazione sistematica della perplessità, della decodifica e della
valutazione dei grandi modelli linguistici sta nel manuale di Jurafsky e Martin
{cite}`jurafsky2026speech`, nel capitolo sui modelli a n-grammi e nelle parti
dedicate ai grandi modelli.

### Le abilità emergenti, e il dubbio che siano un miraggio

Su certi compiti (aritmetica a più cifre, ragionamento a più passi) i modelli
piccoli ottengono zero e, superata una certa scala, la prestazione salta
all'improvviso, senza una salita graduale. Da qui il nome di **abilità
emergenti**, e l'ipotesi che ingrandendo un modello si ottengano capacità non
previste.

```{figure} ../figures/emergent-abilities.svg
:name: fig-capacita-emergenti
:alt: "Due grafici affiancati che misurano lo stesso modello al crescere della scala. A sinistra, con una metrica discontinua come la risposta esatta sì o no, la curva resta piatta e poi salta di colpo: sembra un'abilità comparsa all'improvviso. A destra, con una metrica continua come la distanza di edit, la stessa crescita appare come un miglioramento graduale e regolare."
:width: 100%

Lo stesso modello, due righelli. Il gradino di sinistra sta nel modo di dare i
voti, non nei dati: quel modo assegna zero a una risposta quasi giusta finché
non diventa esatta.
```

I due grafici di {numref}`fig-capacita-emergenti` mostrano lo stesso modello,
misurato in due modi diversi: la differenza fra le due curve è il cuore della
questione.

`````{tab} Elementare

L'obiezione arrivata dopo mostra che l'errore stava nel modo di misurare.

Molti di quei compiti sono valutati **tutto-o-niente**: la risposta a
$134 \times 27$ è giusta solo se tutte le cifre sono giuste. Con una metrica
del genere, un modello che passa dallo sbagliare tre cifre allo sbagliarne una
prende zero in entrambi i casi: poi azzecca l'ultima e prende uno. Il salto è
nella *pagella*, non nel modello.

Se si misura la stessa identica prestazione con un metro graduale (quante
cifre sono corrette, o la probabilità assegnata alla risposta giusta), la
curva diventa liscia e prevedibile. Il miglioramento c'era ed era continuo: la
metrica lo nascondeva.

Tuttavia questo vale per molti casi, non per tutti. Che gran parte di quelle
curve a scalino sia un effetto del righello è ormai ben argomentato; che
ingrandire un modello non gli cambi *mai* niente di qualitativo è
un'affermazione più forte, e nessuno l'ha dimostrata. Il dibattito è aperto, e
resta tale anche quando la spiegazione furba fa comodo.

La morale vale ben oltre gli LLM: una metrica discontinua trasforma un
progresso graduale in un miracolo apparente.

`````

`````{tab} Superiore

Le abilità emergenti sono state documentate da Wei e colleghi
{cite}`wei2022emergent`; la critica del *miraggio* è di Schaeffer, Miranda e
Koyejo {cite}`schaeffer2023emergent`.

L'argomento è preciso. Se l'errore per token cala regolarmente con la scala
(come predicono le leggi di scala), l'accuratezza su una risposta di $n$ token
valutata in modo esatto vale circa $p_{\text{tok}}^{\,n}$, dove
$p_{\text{tok}}$ è la probabilità di azzeccare un singolo token e il calcolo
suppone che gli errori sui token siano indipendenti l'uno dall'altro. Una
funzione del genere resta schiacciata vicino a zero e poi si impenna. Ne segue
che la discontinuità è prodotta dalla non linearità della metrica, non dal
modello. Sostituendo l'accuratezza esatta con la distanza di edit, o con la
log-verosimiglianza della risposta corretta, in molti casi l'emergenza
svanisce.

Il dibattito non è chiuso, e le due affermazioni vanno tenute distinte. Che gran
parte delle curve «a salto» siano artefatti di misura è ormai ben argomentato.
Che *nessun* cambiamento qualitativo avvenga con la scala è un'affermazione più
forte e non dimostrata: fenomeni come l’*in-context learning* restano difficili
da ridurre a un miglioramento puramente continuo.

La ricaduta pratica è però univoca, e riguarda chiunque valuti un modello:
usare metriche continue quando si studia un andamento, e diffidare di
qualunque grafico in cui una capacità «appare». La prima domanda da porsi è come
sia stata misurata.

`````

## In pratica: campionare con PyTorch

Temperatura, top-k e top-p stanno in una funzione di venti righe. Non c'è
nessun modello scaricato: i punteggi grezzi (i *logit*, quelli che il
modello assegna a ogni token del vocabolario prima di trasformarli in
probabilità) sono scritti a mano, così resta in vista solo il meccanismo. Le
operazioni sono quelle della decodifica, nell'ordine: la temperatura divide i
logit, top-k e top-p tolgono i token fuori dal taglio, e infine si campiona.

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
        fuori = (cumulate - probs_ord) >= top_p
        ordinati[fuori] = float("-inf")
        logits = torch.full_like(logits, float("-inf")).scatter(0, indici, ordinati)

    probs = torch.softmax(logits, dim=-1)      # 4) di nuovo una distribuzione
    return int(torch.multinomial(probs, num_samples=1))

# --- l'esempio numerico del testo: muro, tetto, divano, pigiama ---
logits = torch.tensor([2.0, 1.0, 0.0, -2.0])
for T in (0.5, 1.0, 2.0):
    print(f"T={T}:", torch.softmax(logits / T, dim=-1).round(decimals=3))
```

```text
T=0.5: tensor([0.8670, 0.1170, 0.0160, 0.0000])
T=1.0: tensor([0.6570, 0.2420, 0.0890, 0.0120])
T=2.0: tensor([0.4740, 0.2870, 0.1740, 0.0640])
```

Sono gli stessi numeri dell'esempio su muro, tetto, divano e pigiama: a $T =
0{,}5$ la distribuzione si concentra su «muro», a $T = 2$ si appiattisce. Per
il ciclo di generazione basta un modello giocattolo al posto di un vero
Transformer: la struttura del ciclo è identica a quella reale.

```python
import torch

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

Sostituendo `modello_giocattolo` con un Transformer addestrato si ottiene il
ciclo di generazione di un modello linguistico, a parte la KV cache,
l'elaborazione di più richieste insieme e i criteri per fermarsi.

Il modello che esce dal pretraining è un
**completatore**, non un assistente: alla domanda «Qual è la capitale della
Francia?» può rispondere «Qual è la capitale della Spagna? Qual è la capitale
dell'Italia?», perché nel web le liste di domande abbondano, e completare la
lista è probabile. Trasformare il completatore in un
interlocutore che risponde, segue istruzioni e rifiuta le richieste dannose
richiede una seconda fase di addestramento, con procedure proprie: è il
post-training, e ha una {doc}`sezione tutta sua <post-training>`.

`````{tab} Elementare
```{admonition} Da ricordare
:class: important
- Un grande modello linguistico gioca il gioco di Shannon su scala
  industriale: coprire a turno ogni parola di una frase vera e provare a
  indovinarla, miliardi di volte, su una biblioteca raccolta dal web e
  ripulita. Nessuno gli corregge i compiti: la soluzione era già nel testo.
- Più parametri, più testo da leggere e più ore di calcolo danno un modello
  migliore, e in modo prevedibile. Non all'infinito, però: sotto un certo
  punto non si scende, perché una quota dell'incertezza appartiene alla lingua
  stessa. Gli ingredienti, poi, vanno bilanciati, e la regola pratica è una
  ventina di pezzi di testo per ogni parametro, se si vuole spendere il meno
  possibile per addestrarlo; chi poi deve usare il modello moltissimo ne fa uno
  più piccolo e gli fa leggere di più. «Quanto è grande?» non basta più come
  domanda.
- Per scrivere, il modello non prende sempre la parola più probabile:
  verrebbe un testo noioso, che si incarta a ripetere sé stesso. Tira un dado,
  e tre manopole decidono quanto quel dado è truccato (la temperatura) e
  quante carte restano nel mazzo da cui pescare (il top-k e il top-p).
- Il taccuino della KV cache evita di rileggere tutto da capo a ogni parola:
  gli appunti già presi restano in memoria. Si risparmia tempo e si paga in
  spazio: le conversazioni lunghe costano di più.
- Il prompt è un modo di programmare scrivendo: potente e fragile insieme.
  I punteggi dei test vanno letti sapendo che un modello che ha studiato
  tutto il web potrebbe aver già visto le domande.
- Il modello che esce dal pretraining è un completatore di testo, non un
  assistente: per questo serve una seconda fase, il post-training.
```
`````

`````{tab} Superiore
```{admonition} Da ricordare
:class: important
- Un LLM gioca il gioco di Shannon su scala industriale: cross-entropia sul
  token successivo come unica supervisione, su corpora web filtrati e
  deduplicati (GPT-3: ~300 miliardi di token, 60% da Common Crawl).
- Leggi di scala: la loss cala come una legge di potenza in parametri,
  dati e calcolo {cite}`kaplan2020scaling`, dentro il regime misurato e sopra
  un termine irriducibile; il bilanciamento che minimizza il costo di
  addestramento è circa 20 token per parametro {cite}`hoffmann2022training`, e
  chi prevede di servire molte richieste addestra di proposito un modello più
  piccolo su più token. Sulle "capacità emergenti" il dibattito è aperto:
  prudenza.
- Massimizzare la probabilità degenera in ripetizioni
  {cite}`holtzman2020curious`: si campiona con temperatura (che scala i
  logit), top-k (i $k$ token più probabili) e top-p (il nucleo che copre
  probabilità cumulata $p$).
- La KV cache conserva chiavi e valori già calcolati: niente ricalcoli, ma
  memoria che cresce col contesto (~0,5 MB per token in un modello da 7
  miliardi di parametri con una testa di chiave e valore per ogni testa di
  query, un quarto con GQA a otto gruppi); ecco perché i contesti lunghi
  costano. Oltre qualche decina di migliaia di token di contesto (circa
  $6\,d_{\text{model}}$), poi, il calcolo per token è fatto soprattutto di
  attenzione.
- Il prompt è programmazione in linguaggio naturale: potente e fragile
  insieme. I benchmark vanno letti col sospetto della contaminazione
  dei dati di test.
- Il pretraining produce un completatore, non un assistente: per quello serve
  il post-training.
```
`````

Prima del post-training c'è però un'idea architetturale che la scala rende
interessante: dare a un modello molti parametri senza che ogni parola li
attraversi tutti. È l'idea della {doc}`sezione sui modelli a esperti
<mixture-of-experts>`.
