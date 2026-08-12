# State Space Model

C'è un'idea che l'ingegneria usa da oltre mezzo secolo per descrivere
qualunque sistema che evolve nel tempo: un termostato, la traiettoria di un
razzo, un circuito elettrico. Si chiama **modello a spazio degli stati**
(*state space model*, in sigla **SSM**, che è l'abbreviazione che useremo da
qui in avanti): un pugno di equazioni che riassumono tutto il passato di un
**segnale** (una grandezza che cambia nel tempo: il livello dell'acqua in una
vasca, un suono, una sequenza di parole) in uno **stato** interno, e da quello
prevedono il futuro. È la matematica dei filtri di Kalman che portarono
l'Apollo sulla Luna. Che cosa ci fa in un libro sull'intelligenza artificiale?

Ci fa che risolve il problema lasciato aperto dal capitolo sui Transformer.
Là, per capire una frase, ogni parola guarda tutte le altre: raddoppiare la
lunghezza del testo **quadruplica** il lavoro, e su un testo molto lungo il
conto diventa proibitivo. Serve una macchina che legga lungo restando veloce,
cioè che quando il testo raddoppia raddoppi il lavoro e basta. È questo che si
intende, in tutto il capitolo, con **costo lineare** (o «a tempo lineare»): il
lavoro cresce di pari passo con la lunghezza, non più in fretta di lei.

La risposta arriva nel 2021, quando Albert Gu, Karan Goel e Christopher Ré
prendono quelle equazioni vecchie di sessant'anni, le impacchettano in uno
strato di rete neurale e le mettono alla prova sul *Long Range Arena*, il
banco di prova delle **dipendenze a lunghissimo raggio**: i legami fra parti
lontane di una sequenza (in un giallo, per capire l'ultima pagina bisogna
ricordare il nome che compariva alla prima). Il loro modello, **S4**
{cite}`gu2022s4`, riesce là dove Transformer e reti ricorrenti si arrendevano:
riconosce strutture che si estendono per **sedicimila** passi, dove un «passo»
è un elemento della sequenza (una parola, un campione audio, un pixel). È
l'atto di nascita di una seconda strada verso il modello di sequenze a costo
lineare: non quella dell'attenzione resa economica del capitolo precedente, ma
quella, apparentemente lontana, dei sistemi dinamici. Alla fine, scopriremo,
le due strade portano allo stesso posto.

## Un sistema che riassume il passato

`````{tab} Elementare

Immagina di dover seguire il livello dell'acqua in una vasca mentre entra ed esce
di continuo. Non ti serve ricordare ogni singola goccia: ti basta un numero, il
livello attuale, che riassume tutta la storia. A ogni istante il livello di prima,
più quello che è entrato, meno quello che è uscito, ti dà il livello nuovo. Quel
numero che si porta dietro il passato è lo **stato**; la regola che lo aggiorna è
il modello.

Uno *state space model* fa esattamente questo con una sequenza: mantiene uno
stato di dimensione fissa che riassume tutto ciò che ha letto finora, e lo
aggiorna a ogni passo. È lo stesso spirito della rete ricorrente vista nel
capitolo sul linguaggio, ma qui la regola di aggiornamento nasce da una teoria
matematica precisa, quella dei sistemi che evolvono nel tempo, e questo, come
vedremo, fa una grande differenza sulla memoria a lungo termine.

C'è poi una proprietà che tornerà in ogni pagina del capitolo, e conviene
prenderla subito. Finché la regola di aggiornamento **resta la stessa a ogni
passo**, lo stesso calcolo si può fare in due modi: «passo dopo passo», una
parola alla volta, oppure «tutto insieme», facendo scorrere un unico filtro
lungo su tutta la sequenza. Sono la stessa identica cosa vista da due lati, e
questa è la **doppia natura** di cui parleremo: si addestra il modello nel
secondo modo, che è veloce perché fa tutti i conti in una volta, e lo si usa
nel primo, che è economico perché non tiene niente in sospeso.

`````

`````{tab} Superiore

Il mattone è un sistema lineare a tempo continuo che mappa un ingresso $u(t)$ in
un'uscita $y(t)$ attraverso uno stato latente $\mathbf{h}(t)$:

$$
\mathbf{h}'(t) = \mathbf{A}\, \mathbf{h}(t) + \mathbf{B}\, u(t), \qquad y(t) = \mathbf{C}\, \mathbf{h}(t).
$$

La matrice $\mathbf{A}$ governa la dinamica interna (come lo stato evolve da solo), $\mathbf{B}$
come l'ingresso vi entra, $\mathbf{C}$ come se ne legge l'uscita. Per usarlo su una
sequenza discreta lo si **discretizza** con un passo $\Delta$, ottenendo una
ricorrenza $\mathbf{h}_t = \bar{\mathbf{A}}\, \mathbf{h}_{t-1} + \bar{\mathbf{B}}\, x_t$. E qui sta la ricchezza:
finché i parametri sono costanti nel tempo, questa ricorrenza ha una **doppia
natura**; si può calcolare passo per passo come una RNN (inferenza a costo
costante) oppure tutta in una volta come una **convoluzione** (addestramento
parallelo). È la stessa dualità parallelo/ricorrente che muove il capitolo
sull'attenzione lineare, raggiunta però dalla teoria dei segnali.

`````

## Due strade, una meta

L'attenzione lineare e gli *state space model* nascono da mondi diversi (l'una
dal meccanismo di attenzione, gli altri dai sistemi dinamici) ma convergono
sullo stesso oggetto: una **ricorrenza lineare a stato fisso**, addestrabile
in parallelo e capace di generare a memoria costante. Detta in italiano, è la
macchina descritta qui sopra: tiene un riassunto di taglia sempre uguale (lo
**stato**) e a ogni parola lo aggiorna con una regola semplice, di quelle in
cui il nuovo riassunto è il vecchio più ciò che entra (è ciò che significa
«lineare»); si addestra lavorando su tutta la sequenza in una volta sola, e
poi genera una parola alla volta senza che la memoria cresca mai. Questa
frase, con parole diverse, tornerà a ogni sezione: è il punto d'arrivo comune
dei due capitoli.

```{figure} ../figures/mamba-2023.svg
:name: fig-attenzione-vs-ssm
:alt: "Due schemi affiancati sulla stessa sequenza di token. A sinistra, sotto il titolo Attention, l'attenzione piena: ogni token è collegato a tutti gli altri, e il numero di connessioni cresce col quadrato della lunghezza (costo proporzionale a n²). A destra, sotto il titolo State space selettivo, i token sono collegati in catena a uno stato che si aggiorna passando da uno al successivo; su ogni passaggio un piccolo rombo ocra rappresenta il filtro che decide, token per token, quanto di ciò che arriva entra nello stato (costo proporzionale a n)."
:width: 100%

Due modi di portarsi dietro il passato. L'attenzione lo tiene tutto e lo
riguarda; la ricorrenza lo riassume in uno stato di taglia fissa e ci scrive
sopra, decidendo di volta in volta che cosa vale la pena scrivere.
```

Il confronto di {numref}`fig-attenzione-vs-ssm` mostra anche dove sta il
prezzo. Un riassunto di taglia fissa deve, prima o poi, dimenticare qualcosa.
Il tema del capitolo è la **selettività** (il rombo sul lato destro della
figura): decidere *cosa* scrivere nello stato, e cosa lasciar cadere, in
funzione di ciò che sta arrivando. Vale la pena essere precisi su che cosa
promette, perché è facile chiedergli troppo: la selettività cambia *come* si
usa lo spazio del riassunto, non lo allarga. Il tetto di un riassunto di
taglia fissa resta, ed è l'argomento dell'ultima sezione del capitolo.

Il secondo filo è una parentela. Alla fine, con **Mamba-2**
{cite}`dao2024mamba2`, vedremo che non è una somiglianza vaga: un *state space
model* di forma opportuna *è* un'**attenzione mascherata**, cioè
un'attenzione che guarda solo all'indietro, in cui il confronto fra due parole
è pesato da quanto della prima è sopravvissuto nel frattempo. Le due famiglie
che raccontiamo in due capitoli sono, in fondo, due viste dello stesso
disegno.

Ma prima c'è una tensione da sciogliere. La doppia natura «passo dopo passo» /
«tutto insieme» vale solo se il sistema è **invariante nel tempo**: le stesse
regole a ogni passo. Ed è proprio questa rigidità che **Mamba**
{cite}`gu2023mamba` romperà, rendendo il sistema *selettivo*, per dargli
qualcosa che a S4 mancava: la capacità di scegliere, in base al contenuto,
cosa ricordare e cosa dimenticare.

## Come è organizzato il capitolo

Quattro tappe, dall'idea di base alla frontiera.

**Dai sistemi dinamici a S4**: che cos'è una macchina che riassume il passato
in un pugno di numeri, come si adatta a una sequenza fatta di passi separati, e
come si fa a darle una memoria lunga (sono i due modelli che danno il titolo
alla sezione, HiPPO e S4).

**Mamba**: come si insegna alla macchina a scegliere, invece di trattare tutte
le parole allo stesso modo; che cosa costa quella scelta (si perde il modo
«tutto insieme»), e con quale trucco si recupera la velocità perduta,
tenendo conto di com'è fatta davvero una scheda grafica.

**La dualità**: la scoperta che questa macchina, scritta in un altro modo, *è*
l'attenzione dei Transformer, e che riscriverla così la fa girare molto più in
fretta. Poi le tre raffinature più recenti, con Mamba-3.

**Panorama e limiti**: una mappa che tiene insieme questo capitolo e il
precedente, che cosa un riassunto di taglia fissa non potrà mai fare, e le
architetture **ibride** che oggi mettono insieme il meglio delle due strade.

`````{tab} Elementare

```{admonition} Da ricordare
:class: important
- Un **modello a spazio degli stati** (in sigla **SSM**) riassume tutto quello
  che ha letto in una specie di taccuino di dimensione **sempre uguale**, e a
  ogni parola lo aggiorna. Sono le stesse equazioni con cui l'ingegneria
  descrive un termostato o la traiettoria di un razzo; **S4** {cite}`gu2022s4`
  le porta dentro una rete neurale, ed è il primo a riconoscere legami fra
  parti di una sequenza distanti **sedicimila** passi.
- Il problema che vengono a risolvere: far guardare ogni parola a tutte le
  altre costa **al quadrato** (testo doppio, lavoro quadruplo). Qui il costo
  cresce di pari passo con la lunghezza, ed è ciò che nel libro si chiama
  **costo lineare**.
- Finché le regole non cambiano da un passo all'altro, lo stesso calcolo si può
  fare in due modi: **passo dopo passo** (economico per generare) oppure
  **tutto insieme** (parallelo, veloce per addestrare). È la **doppia natura**,
  la stessa già vista con l'attenzione lineare.
- **Mamba** {cite}`gu2023mamba` rompe quella regola fissa: lascia decidere alla
  parola in arrivo quanto scrivere e quanto dimenticare (è la **selettività**).
  **Mamba-2** {cite}`dao2024mamba2` mostra poi che, nella sua versione più
  semplice, questa macchina **è** un'attenzione che guarda solo all'indietro:
  le due famiglie si incontrano su quel gradino.
- Il percorso: dai sistemi dinamici a S4 → Mamba (scegliere, e restare veloci)
  → la dualità (Mamba-2 e Mamba-3) → panorama, limiti e ibridi.
```

`````

`````{tab} Superiore

```{admonition} Da ricordare
:class: important
- Uno **state space model** riassume il passato in uno **stato di dimensione
  fissa**, con equazioni che l'ingegneria usa da decenni per i sistemi dinamici;
  **S4** {cite}`gu2022s4` le porta nel deep learning e conquista le dipendenze a
  lunghissimo raggio (fino a $16\,384$ passi sul *Long Range Arena*).
- Discretizzato, un SSM invariante nel tempo ha una **doppia natura**:
  ricorrente (inferenza a costo costante) e convoluzionale (addestramento
  parallelo) (la stessa dualità dell'attenzione lineare, da un'altra strada).
- **Mamba** {cite}`gu2023mamba` rompe l'invarianza temporale con la
  **selettività**; **Mamba-2** {cite}`dao2024mamba2` mostra che un SSM di forma
  opportuna *è* un'attenzione mascherata: le due famiglie si incontrano su quel
  gradino.
- Il percorso: dai sistemi dinamici a S4 → Mamba (selezione e scan) → la dualità
  (Mamba-2 e Mamba-3) → panorama, limiti e ibridi.
```

`````
