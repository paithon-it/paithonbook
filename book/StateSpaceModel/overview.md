# State Space Model

C'è un'idea che l'ingegneria usa da oltre mezzo secolo per descrivere
qualunque sistema che evolve nel tempo: un termostato, la traiettoria di un
razzo, un circuito elettrico. Si chiama **modello a spazio degli stati**
(*state space model*): un pugno di equazioni che riassumono tutto il passato
di un segnale in uno **stato** interno, e da quello prevedono il futuro. È la
matematica dei filtri di Kalman che portarono l'Apollo sulla Luna. Che cosa ci
fa in un libro sull'intelligenza artificiale?

La risposta arriva nel 2021, quando Albert Gu, Karan Goel e Christopher Ré
prendono quelle equazioni vecchie di sessant'anni, le impacchettano in uno
strato di rete neurale e le mettono alla prova sul *Long Range Arena*, il
banco di prova delle dipendenze a lunghissimo raggio. Il loro modello, **S4**
{cite}`gu2022s4`, riesce là dove Transformer e reti ricorrenti si arrendevano:
riconosce strutture che si estendono per **sedicimila** passi. È l'atto di
nascita di una seconda strada verso il modello di sequenze a tempo lineare:
non quella dell'attenzione resa economica del capitolo precedente, ma quella,
apparentemente lontana, dei sistemi dinamici. Alla fine, scopriremo, le due
strade portano allo stesso posto.

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

`````

`````{tab} Superiore

Il mattone è un sistema lineare a tempo continuo che mappa un ingresso $u(t)$ in
un'uscita $y(t)$ attraverso uno stato latente $h(t)$:

$$
h'(t) = A\, h(t) + B\, u(t), \qquad y(t) = C\, h(t).
$$

La matrice $A$ governa la dinamica interna (come lo stato evolve da solo), $B$
come l'ingresso vi entra, $C$ come se ne legge l'uscita. Per usarlo su una
sequenza discreta lo si **discretizza** con un passo $\Delta$, ottenendo una
ricorrenza $h_t = \bar A\, h_{t-1} + \bar B\, x_t$. E qui sta la ricchezza:
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
in parallelo e capace di generare a memoria costante.

```{figure} ../figures/mamba-2023.svg
:name: fig-attenzione-vs-ssm
:alt: "Due schemi affiancati sulla stessa sequenza di token. A sinistra l'attenzione: ogni token è collegato a tutti gli altri, e il numero di connessioni cresce col quadrato della lunghezza. A destra lo state space model selettivo: i token sono collegati in catena a uno stato che si aggiorna passando da uno al successivo, e la dimensione di quello stato non cambia con la lunghezza."
:width: 100%

Due modi di portarsi dietro il passato. L'attenzione lo tiene tutto e lo
riguarda; la ricorrenza lo riassume in uno stato di taglia fissa e ci scrive
sopra.
```

Il confronto di {numref}`fig-attenzione-vs-ssm` mostra anche dove sta il
prezzo. Un riassunto di taglia fissa deve, prima o poi, dimenticare qualcosa;
la parola **selettivo** che compare nel nome è la risposta a questa obiezione,
ed è il tema del capitolo: decidere *cosa* scrivere nello stato in funzione di
ciò che sta arrivando. Non è una coincidenza
superficiale. Alla fine del capitolo, con **Mamba-2** {cite}`dao2024mamba2`,
vedremo che la parentela è esatta: un *state space model* di forma opportuna
*è* un'attenzione mascherata. Le due famiglie che raccontiamo in due capitoli
sono, in fondo, due viste dello stesso disegno.

Ma prima c'è una tensione da sciogliere. La doppia natura
convoluzione/ricorrenza vale solo se il sistema è **invariante nel tempo**:
stessi parametri a ogni passo. Ed è proprio questa rigidità che **Mamba**
{cite}`gu2023mamba` romperà, rendendo il sistema *selettivo*, per dargli
qualcosa che a S4 mancava: la capacità di scegliere, in base al contenuto,
cosa ricordare e cosa dimenticare.

## Come è organizzato il capitolo

Quattro tappe, dall'idea di base alla frontiera.

Si parte **dai sistemi dinamici a S4**: il sistema a spazio degli stati
continuo, la sua discretizzazione, la doppia natura ricorrenza/convoluzione, e
come HiPPO e S4 risolvono la memoria a lungo raggio. Seconda tappa, **Mamba**:
la selettività che rende i parametri dipendenti dall'input, il prezzo (via la
convoluzione, entra lo *scan*) e lo *scan* hardware-aware che lo rende veloce,
dentro il blocco Mamba. Terza tappa, **la dualità**: Mamba-2 che riscopre
l'attenzione dentro l'SSM e recupera i *tensor core*, e Mamba-3 con le sue
raffinatezze (discretizzazione di ordine superiore, stato complesso,
formulazione MIMO). Chiude un **panorama e i limiti**: una mappa unificata di
entrambi i capitoli, l'onesto collo di bottiglia dello stato fisso, e gli
**ibridi** che oggi combinano il meglio dei due mondi.

```{admonition} Da ricordare
:class: important
- Uno **state space model** riassume il passato in uno **stato di dimensione
  fissa**, con equazioni che l'ingegneria usa da decenni per i sistemi dinamici;
  **S4** {cite}`gu2022s4` le porta nel deep learning e conquista le dipendenze a
  lunghissimo raggio (fino a $16\,000$ passi sul *Long Range Arena*).
- Discretizzato, un SSM invariante nel tempo ha una **doppia natura**:
  ricorrente (inferenza a costo costante) e convoluzionale (addestramento
  parallelo) (la stessa dualità dell'attenzione lineare, da un'altra strada).
- **Mamba** {cite}`gu2023mamba` rompe l'invarianza temporale con la
  **selettività**; **Mamba-2** {cite}`dao2024mamba2` mostra che un SSM di forma
  opportuna *è* un'attenzione mascherata: le due famiglie coincidono.
- Il percorso: dai sistemi dinamici a S4 → Mamba (selezione e scan) → la dualità
  (Mamba-2 e Mamba-3) → panorama, limiti e ibridi.
```
