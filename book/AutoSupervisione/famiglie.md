# Quattro modi di fabbricare il segnale

I nomi in circolazione sono decine e sembrano tutti diversi. Non lo sono. Se si
guarda che cosa fanno invece di come si chiamano, i metodi auto-supervisionati
rispondono tutti alla stessa domanda, ed è una domanda che nasce da una
difficoltà sola.

Il pretesto, dice la sezione precedente, è un compito la cui risposta sta già
nei dati. Ma un compito costruito così ha un difetto di fabbrica: quasi sempre
esiste un modo di vincerlo senza aver capito niente. Prendiamo due **viste**
della stessa foto, cioè due versioni diverse della stessa immagine (un ritaglio
e un altro, oppure la stessa scena con i colori spostati), e chiediamo al
modello di dire che si somigliano. La risposta che vince sempre è dire che
**tutte** le foto si somigliano, descrivendole tutte allo stesso identico modo.
Punteggio pieno, niente da correggere, e un modello che non ha guardato niente.
Quella risposta vuota si chiama **collasso**.

Quindi la domanda vera non è «quale pretesto». È: **che cosa impedisce la
risposta vuota**. Le quattro famiglie sono quattro risposte a questa domanda, e
messe in fila si tengono a mente molto meglio che come quattro elenchi di sigle.

## La prima: respingere

Al modello si dà una delle due viste. L'altra, il suo **gemello**, viene
nascosta in mezzo a una folla di viste prese da foto tutte diverse, che
chiameremo i **rivali**, e il compito è ritrovarla. Non si chiede soltanto di
avvicinarsi al gemello, quindi, ma anche di allontanarsi dai rivali, e la
risposta vuota diventa impossibile per costruzione: se descrivo tutto allo
stesso modo, non distinguo nessun rivale dal gemello e il punteggio crolla.

È la famiglia **contrastiva**, e il libro l'ha già percorsa per intero nel
capitolo sulla visione artificiale, alla sezione «Imparare a vedere senza
etichette»: la ricetta di base, il costo dei rivali, e la coda di rivali già
elaborati che permette di averne molti senza doverli calcolare tutti insieme.
Qui interessa solo il posto che occupa nello schema: **il collasso lo impedisce
una forza che allontana**.

Il prezzo di questa famiglia va ricordato, perché è quello che ha spinto le
due che seguono a cercare un'altra strada: i rivali costano, e servono a
migliaia. La quarta famiglia, invece, non nasce da questo problema e non le deve
niente: sul testo esisteva già prima.

## La seconda: rendere le due reti diverse

Si tolgono i rivali e si mettono due reti che guardano la stessa scena da due
punti diversi, chiedendo a una di indovinare quello che dice l'altra. Non c'è
niente che allontani; a impedire il collasso è che le due reti **non sono
intercambiabili**, una impara e l'altra insegue in ritardo, e una sola delle due
ha un passaggio in più prima del confronto.

Anche questa il libro l'ha già fatta, nello stesso capitolo. Qui basta il posto
nello schema: **il collasso lo impedisce un'asimmetria**, cioè una differenza
costruttiva fra i due rami.

È anche la famiglia di cui si capisce meno *perché* funzioni, e conviene dirlo
perché è proprio da lì che nasce la terza: **funziona, e la spiegazione è
arrivata dopo**, un pezzo alla volta e su modelli semplificati
{cite}`tian2021understanding`. Chi non si accontenta di una proprietà che
spunta fuori da sé mentre il modello si addestra ha una sola strada, ed è
scrivere l'anti-collasso dentro la formula, dove si può leggere.

## La terza: vincolare le statistiche

Qui il libro entra in materia nuova, e conviene dire subito la mossa. Il
riassunto che il modello produce di ogni foto è una fila di numeri, e ognuno di
quei numeri sta in una **casella** sua (nel gergo le caselle si chiamano
*coordinate*). Invece di allontanare gli esempi gli uni dagli altri, o di
sperare che un'asimmetria faccia il suo lavoro, si guarda che cosa il modello
scrive in ciascuna casella e gli si impone una condizione che la risposta vuota
non può soddisfare.

L'idea non nasce nell'informatica. In un saggio raccolto nel volume *Sensory
Communication* del 1961 {cite}`barlow1961possible`, il neurofisiologo Horace
Barlow propose che il compito dei primi stadi del sistema sensoriale fosse
**ridurre la ridondanza**: ricodificare il segnale in modo che le sue componenti
dicessero ciascuna una cosa propria, invece di ripetersi a vicenda. Sessant'anni
dopo un metodo di apprendimento prende il nome da lui proprio per questo, e sono
gli autori stessi a dichiararlo {cite}`zbontar2021barlow`.

`````{tab} Elementare

Ogni fotografia va descritta riempiendo una scheda con otto
caselle. Le caselle non hanno un significato deciso da noi: è il modello a
scoprire che cosa metterci.

Perché la scheda sia buona devono valere due regole, e chiedono cose diverse.

La somiglianza: se compilo la scheda guardando due ritagli diversi della stessa
foto, le due schede devono venire uguali. Prima di confrontarle, però, si guarda
che cosa ogni casella ha scritto sulle altre fotografie, e una casella che
scrive sempre lo stesso numero viene messa da parte: non ha detto niente. Due
schede identiche perché tutte le caselle sono bloccate non si somigliano
affatto, non hanno detto niente su cui somigliarsi.

La varietà: le otto caselle devono dire otto cose diverse. Se la casella 3 dice
sempre la stessa cosa della casella 5, ho una scheda da otto caselle che ne vale
sette, e sto sprecando spazio. Quanto pesi la varietà rispetto alla somiglianza
lo decidiamo noi, con una manopola.

Ecco perché questa famiglia non ha bisogno né di rivali né di trucchi
costruttivi: le due regole vietano guasti diversi, e insieme li coprono. La
risposta vuota, cioè descrivere tutte le foto allo stesso modo, la ferma la
somiglianza: caselle bloccate non si somigliano, non c'è niente da premiare, e
quella regola resta insoddisfatta per intero. La varietà ferma un guasto più
educato, la scheda che cambia da foto a foto ma dice otto volte la stessa cosa:
lì i due ritagli si somigliano quanto devono, e a pagare è lo spreco. In tutti e
due i casi la penalità è scritta nel punteggio, e non arriva per vie traverse.

C'è chi, invece di mettere da parte la casella bloccata, scrive la pretesa nero
su bianco: ogni casella deve variare almeno tanto da una foto all'altra, e se
varia meno si paga. La richiesta sta su ciascuna scheda invece che nel confronto
fra le due, e allora a compilarle possono essere due persone che lavorano in
modo diverso, o perfino su materiali diversi.

`````

`````{tab} Superiore

Siano $\mathbf{Z}^A, \mathbf{Z}^B \in \mathbb{R}^{N \times D}$ le
rappresentazioni di un batch di $N$ esempi nelle due viste, con $D$ coordinate,
ciascuna standardizzata sul batch (media nulla, varianza unitaria). Si costruisce
la **matrice di cross-correlazione**

$$
\mathbf{C} = \frac{1}{N} \, (\mathbf{Z}^A)^\top \mathbf{Z}^B
\in \mathbb{R}^{D \times D},
\qquad
C_{ij} = \frac{1}{N} \sum_{n=1}^{N} Z^A_{ni} \, Z^B_{nj},
$$

dove $C_{ij}$ è la correlazione fra la coordinata $i$ della prima vista e la
coordinata $j$ della seconda. **Barlow Twins** {cite}`zbontar2021barlow` chiede
che $\mathbf{C}$ sia il più vicino possibile alla matrice identità:

$$
\mathcal{L} = \underbrace{\sum_{i=1}^{D} (1 - C_{ii})^2}_{\text{invarianza}}
\; + \; \lambda \underbrace{\sum_{i=1}^{D} \sum_{j \neq i} C_{ij}^2}_{\text{riduzione di ridondanza}},
$$

con $\lambda > 0$ a pesare i due termini. La lettura è diretta: la **diagonale**
a uno impone che ogni coordinata sia invariante alla vista; la **fuori
diagonale** a zero impone che coordinate diverse siano scorrelate, cioè che non
si ripetano.

L'anti-collasso non è una proprietà emergente, ed è utile vedere quale dei due
termini ferma quale collasso, perché non è lo stesso. Se l'uscita è
**costante**, la standardizzazione divide per una deviazione standard nulla e
manda a zero tutte le celle di $\mathbf{C}$: a pagare è allora il termine di
**invarianza**, che vale $D$, mentre quello di ridondanza vale zero e non serve
a niente. Se invece l'uscita varia ma tutte le coordinate portano lo **stesso**
segnale, che è la forma interessante del collasso, dopo la standardizzazione le
colonne di $\mathbf{Z}^A$ sono identiche fra loro, e così quelle di
$\mathbf{Z}^B$: ogni cella di $\mathbf{C}$ vale allora lo stesso numero $c$,
quello che sta sulla diagonale, e il termine di ridondanza paga
$\lambda \, D(D-1) \, c^2$, cioè $D(D-1)$ celle piene quanto la diagonale,
mentre una rappresentazione con la stessa diagonale e coordinate scorrelate
pagherebbe zero. Non c'è niente da dimostrare sulla dinamica
dell'ottimizzazione, perché la penalità è scritta nell'obiettivo.

**VICReg** {cite}`bardes2022vicreg` arriva alla stessa meta con tre termini
espliciti, varianza, invarianza e covarianza, e la differenza pratica sta nella
varianza: un termine che tiene la **deviazione standard di ogni coordinata sopra
una soglia**, con una cerniera, calcolato su ciascun ramo per conto suo. Gli
autori lo scrivono come una critica alla seconda famiglia: il collasso, dicono,
«è spesso evitato attraverso bias impliciti nell'architettura di apprendimento,
che spesso mancano di una giustificazione o di un'interpretazione chiara», e
VICReg «evita esplicitamente il problema del collasso» con un termine di
regolarizzazione sulla varianza. Due conseguenze concrete: le due reti non hanno
bisogno di condividere i pesi né di essere una la copia lenta dell'altra, e i
due rami possono avere architetture diverse o perfino ingressi di natura
diversa.

**SwAV** {cite}`caron2020swav` sta a cavallo fra questa famiglia e la prima. Non
confronta le rappresentazioni a coppie: assegna ogni vista a un insieme di
prototipi e **predice l'assegnazione di una vista dalla rappresentazione
dell'altra**, con un vincolo di equipartizione fra i prototipi che è il pezzo
anti-collasso. Mathilde Caron firma come prima autrice anche il metodo di
distillazione della famiglia precedente {cite}`caron2021emerging`, e
l'equipartizione fa qui il mestiere che là fanno centratura e affilatura:
impedire che una casella se le prenda tutte.

`````

Che chiedere «otto caselle, otto cose diverse» sia un'operazione e non una
metafora si vede in una trentina di righe, senza dataset e senza addestrare
niente di grosso. Partiamo apposta dal caso interessante, cioè da un modello
**ridondante**: otto coordinate che all'inizio dicono quasi tutte la stessa
cosa.

```python
import torch

torch.manual_seed(0)
N, D_IN, D = 512, 32, 8      # esempi, dimensioni in ingresso, coordinate finali

# Due VISTE dello stesso esempio: stesso contenuto, disturbi indipendenti.
contenuto = torch.randn(N, D_IN)
RUMORE_VISTA = 0.3
vista_a = contenuto + RUMORE_VISTA * torch.randn(N, D_IN)
vista_b = contenuto + RUMORE_VISTA * torch.randn(N, D_IN)

# Partenza RIDONDANTE, ed e' il caso interessante: le otto coordinate nascono
# quasi uguali fra loro, cioe' il modello dice otto volte la stessa cosa.
proiettore = torch.nn.Linear(D_IN, D, bias=False)
with torch.no_grad():
    proiettore.weight.copy_(proiettore.weight[0] + 0.05 * torch.randn(D, D_IN))

def correlazione(za, zb):
    """Cross-correlazione fra le due viste, ogni coordinata standardizzata."""
    za = (za - za.mean(0)) / (za.std(0) + 1e-9)
    zb = (zb - zb.mean(0)) / (zb.std(0) + 1e-9)
    return (za.T @ zb) / za.shape[0]

def barlow(c, lam=0.05):
    """Diagonale verso 1 (invarianza), fuori diagonale verso 0 (ridondanza)."""
    diag = torch.diagonal(c)
    fuori = c - torch.diag_embed(diag)
    return ((diag - 1) ** 2).sum() + lam * (fuori ** 2).sum()

def referto(eti):
    with torch.no_grad():
        c = correlazione(proiettore(vista_a), proiettore(vista_b))
        d, f = torch.diagonal(c), c - torch.diag_embed(torch.diagonal(c))
        print(f"{eti:14s} diagonale {d.mean():5.2f}   "
              f"fuori diagonale {f.abs().sum() / (D * D - D):5.2f}")

RUMORE = RUMORE_VISTA
# La diagonale non potra' arrivare a 1: le due viste hanno rumore indipendente,
# quindi la loro correlazione ha un tetto, ed e' questo.
print(f"tetto della diagonale, imposto dal rumore: {1 / (1 + RUMORE ** 2):.2f}\n")

referto("all'inizio")
ott = torch.optim.SGD(proiettore.parameters(), lr=0.05)
for passo in range(1, 601):
    ott.zero_grad()
    barlow(correlazione(proiettore(vista_a), proiettore(vista_b))).backward()
    ott.step()
    if passo in (100, 600):
        referto(f"dopo {passo}")
```

```text
tetto della diagonale, imposto dal rumore: 0.92

all'inizio     diagonale  0.92   fuori diagonale  0.70
dopo 100       diagonale  0.92   fuori diagonale  0.01
dopo 600       diagonale  0.93   fuori diagonale  0.01
```

Le due colonne raccontano due storie diverse, ed è esattamente il punto. La
colonna della **fuori diagonale**, che misura quanto le coordinate si ripetono
l'una con l'altra, crolla da $0{,}70$ a $0{,}01$: le otto coordinate smettono di
ripetersi e cominciano a dire otto cose distinte. La colonna della
**diagonale**, che misura quanto le due viste ricevono lo stesso riassunto,
invece non si muove, perché era già al massimo consentito: le due viste hanno
rumore indipendente, quindi la loro correlazione ha un tetto, che il programma
calcola e stampa in cima, ed è $0{,}92$. Il valore misurato dopo seicento passi
è $0{,}93$, cioè lo stesso numero a meno del campione finito.

Vale la pena fermarsi su questo, perché è la cosa che si sbaglia leggendo la
formula: la diagonale non deve andare a uno per forza. Deve andare **il più in
alto che il rumore consente**, e in una situazione reale quel tetto è imposto
dalle trasformazioni che abbiamo scelto noi. Quello che l'ottimizzazione può
davvero guadagnare, in questo esempio, è tutto nell'altra colonna.

## La quarta: ricostruire

L'ultima famiglia non chiede al modello di riconoscere né di confrontare: gli
copre un pezzo di dato e gli chiede di rifarlo. La risposta vuota qui non è
nemmeno una tentazione: descrivere tutte le immagini allo stesso modo rende
impossibile ricostruirne una in particolare, e il punteggio se ne accorge subito.

È la famiglia **generativa mascherata**, e il libro l'ha percorsa due volte: sul
testo, nel {doc}`capitolo sui Transformer </Transformers/overview>`, e sulle immagini, nel capitolo sulla
visione. Il posto nello schema: **il collasso lo impedisce il compito stesso**,
perché ricostruire un dato specifico richiede di averlo descritto in modo
specifico.

Il prezzo, che le altre tre non pagano, è che il conto si fa sul dato grezzo:
ricostruire i pixel vuol dire spendere capacità anche sul granello di polvere e
sul riflesso, cioè su dettagli che nessuno potrebbe indovinare e che a nessuno
interessano. È l'obiezione che porterà alla JEPA, nel {doc}`capitolo sui world model </WorldModels/overview>`.

## Una rinuncia annunciata, e chi l'ha firmata

C'è un filo che questo capitolo può finalmente chiudere, e che il libro aveva
lasciato aperto.

Nel {doc}`capitolo sui modelli a energia </ModelliEnergia/overview>` compare l'elenco delle rinunce che Yann LeCun
ripete nelle sue conferenze, e la terza dice: abbandonare i metodi contrastivi
in favore di quelli **regolarizzati**. Cioè, nel lessico di questa pagina:
smettere di mostrare al modello dei controesempi da respingere, e costruirlo
invece in modo che non possa dire di sì a tutto.

I metodi regolarizzati sono la terza famiglia di questa sezione, e conviene
guardare chi firma i due lavori: Barlow Twins e VICReg hanno **LeCun stesso
fra gli autori**. La rinuncia e la sua attuazione sono la stessa persona, il
che non la rende né più né meno vera, ma spiega perché quella riga della
diapositiva non fosse una previsione generica.

Se la scommessa sia giusta resta una questione aperta, e il libro non ha motivo
di chiuderla al posto della ricerca. L'argomento di chi ci scommette è che al
crescere della complessità del dato, e soprattutto sul video, le risposte
possibili diventano così tante che nessuna quantità di controesempi basterebbe a
puntellare il modello; l'argomento di chi non ci scommette è che i metodi
contrastivi, nel frattempo, hanno prodotto sistemi che funzionano molto bene.

## Le quattro famiglie in una tabella

| famiglia | il pretesto | che cosa impedisce la risposta vuota | dove sta la difficoltà |
|---|---|---|---|
| contrastiva | ritrovare il gemello fra molti rivali | una forza che **allontana** | nelle trasformazioni scelte a mano |
| distillazione | indovinare che cosa dice l'altra rete | un’**asimmetria** fra i due rami | nel come le due reti sono fatte diverse |
| riduzione di ridondanza | due viste, stessa scheda | un **vincolo scritto nella formula** sulle coordinate | in quali statistiche si decide di vincolare |
| generativa mascherata | rifare il pezzo coperto | il **compito stesso** | in quanta informazione si toglie |

L'ultima colonna è quella che si porta via chi legge. Fabbricare un pretesto
significa decidere **dove mettere la difficoltà**, e ognuna delle quattro
famiglie la mette in un posto diverso: nelle nostre scelte a monte, nella forma
dell'architettura, in una condizione algebrica, o nella dose di informazione
nascosta. Non c'è una risposta migliore in assoluto, c'è una risposta che si
adatta meglio al tipo di dato e a quanto siamo disposti a mettere di nostro
dentro il compito.

## Un avvertimento sulle tassonomie

Le famiglie si possono contare in più di un modo, e conviene sapere quale si
sta usando qui. Questa pagina taglia secondo **che cosa impedisce la risposta
vuota**, ed è la colonna di mezzo della tabella; ne escono quattro famiglie.
La colonna di destra, «dove sta la difficoltà», è invece l'asse che usa
{doc}`Imparare a vedere senza etichette </VisioneArtificiale/senza-etichette>`,
e la coincidenza ha una ragione. Nel
{doc}`capitolo sui world model </WorldModels/overview>` si taglia invece
secondo **dove avviene la previsione**, cioè se il modello prova a rifare il dato
(i pixel, i token) oppure il suo riassunto: da lì escono tre famiglie, e la
terza, quella che predice nello spazio delle rappresentazioni, in questa pagina
non compare affatto perché non è un modo diverso di evitare il collasso.

Non è una contraddizione ed è utile che sia così: i due assi sono
indipendenti, e un metodo ha una posizione su ciascuno dei due. Una JEPA, per
dire, sull'asse del *dove* predice nello spazio delle rappresentazioni, e
sull'asse del *come non collassa* si affida a un'asimmetria, oppure a un vincolo
sulle statistiche. Quando si leggono due elenchi con due numeri diversi, quasi
sempre non è che uno dei due sbagli a contare: stanno guardando la stessa cosa
da due lati.

`````{tab} Elementare

```{admonition} Da ricordare
:class: important
- Ogni esercizio inventato ha un modo di essere vinto senza aver capito niente:
  descrivere **tutto allo stesso modo**. È il **collasso**, e le famiglie di
  metodi si distinguono per come lo impediscono, non per come si chiamano.
- **Respingere**: si mettono in campo dei rivali, e descrivere tutto uguale fa
  perdere. Funziona, ma i rivali servono a migliaia e costano.
- **Rendere le due reti diverse**: niente rivali, ma allievo e insegnante non
  sono intercambiabili. Funziona, e la spiegazione del perché è arrivata dopo il
  risultato.
- **Vincolare le statistiche**: si compila una scheda con otto caselle e si
  chiedono due cose insieme, la somiglianza e la varietà. La somiglianza vuole
  che due ritagli della stessa foto diano la stessa scheda; la varietà vuole che
  **le otto caselle dicano otto cose diverse**. Le due richieste fermano due
  guasti diversi: la risposta vuota, dove ogni casella scrive sempre lo stesso
  numero, la ferma la somiglianza, perché caselle bloccate non si somigliano; la
  scheda che dice otto volte la stessa cosa la ferma la varietà. Nessuna delle
  due arriva per vie traverse: è scritto nel punteggio. L'idea viene dalla
  neurofisiologia degli anni Sessanta.
- **Ricostruire**: si copre un pezzo e si chiede di rifarlo. Qui la risposta
  vuota non serve nemmeno a niente, perché per rifare *quella* foto bisogna
  averla descritta in modo suo. Si paga altrove: si spreca fatica su dettagli
  che nessuno può indovinare.
- Trenta righe di codice fanno vedere il vincolo all'opera: le caselle smettono
  di ripetersi (il numero della ridondanza crolla da $0{,}70$ a $0{,}01$) mentre
  la somiglianza fra le due schede resta dov'era, perché era già al massimo che
  il disturbo consentiva.
```

`````

`````{tab} Superiore

```{admonition} Da ricordare
:class: important
- Le famiglie auto-supervisionate si classificano meglio per **come evitano il
  collasso** che per il pretesto: repulsione (contrastivi), asimmetria
  architetturale (distillazione), vincolo esplicito sulle statistiche
  dell'embedding (regolarizzati), specificità del bersaglio (generativi
  mascherati).
- **Barlow Twins** {cite}`zbontar2021barlow`: si standardizzano le
  rappresentazioni sul batch, si costruisce la cross-correlazione
  $\mathbf{C} = \frac{1}{N}(\mathbf{Z}^A)^\top \mathbf{Z}^B$ e la si porta verso
  l'identità. Diagonale a uno: invarianza. Fuori diagonale a zero: riduzione di
  ridondanza. I due termini fermano due collassi diversi: l'uscita **costante**
  la ferma l'invarianza (la standardizzazione manda $\mathbf{C}$ a zero e quel
  termine vale $D$), mentre le coordinate **tutte uguali** le ferma la
  ridondanza, perché allora ogni cella di $\mathbf{C}$ vale lo stesso numero $c$
  della diagonale e il termine paga $\lambda D(D-1)c^2$, cioè $D(D-1)$ celle
  piene quanto la diagonale, e coordinate scorrelate pagherebbero zero.
- **VICReg** {cite}`bardes2022vicreg`: varianza, invarianza, covarianza. Il
  termine di **varianza** con cerniera tiene la deviazione standard di ogni
  coordinata sopra una soglia, quindi l'anti-collasso è **esplicito** e non un
  bias implicito dell'architettura. I due rami non devono condividere i pesi né
  essere l'uno la media mobile dell'altro, e possono avere architetture o
  ingressi diversi.
- **SwAV** {cite}`caron2020swav`: si predice l'assegnazione a prototipi di una
  vista dalla rappresentazione dell'altra, con equipartizione fra i prototipi
  come vincolo anti-collasso. Sta a cavallo fra i contrastivi e i metodi che
  vincolano le statistiche.
- Nell'esperimento con otto coordinate ridondanti la fuori diagonale scende da
  $0{,}70$ a $0{,}01$ mentre la diagonale non si muove, $0{,}92$ all'inizio e
  $0{,}93$ dopo seicento passi: è il **tetto imposto dal rumore delle viste**,
  $1/(1+\sigma^2)$ con $\sigma = 0{,}3$, non un limite dell'ottimizzazione. La
  diagonale non deve tendere a uno in assoluto, ma al massimo che le
  trasformazioni consentono.
- I metodi regolarizzati sono la **terza rinuncia** dell'elenco di LeCun
  discusso nel {doc}`capitolo sui modelli a energia </ModelliEnergia/overview>`, e Barlow Twins e VICReg hanno
  LeCun fra gli autori. La scommessa è che sul video nessuna quantità di
  negativi basti; la questione è aperta.
```

`````
