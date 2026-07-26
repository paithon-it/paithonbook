# Physics-Informed Neural Networks

La notte del 23 settembre 1846 l'astronomo Johann Galle punta il telescopio
dell'Osservatorio di Berlino verso un punto preciso del cielo, indicato in una
lettera arrivata quel giorno da Parigi. Il mittente, Urbain Le Verrier, quel
punto non l'ha mai osservato: l'ha *calcolato*, applicando per mesi le leggi
di Newton alle irregolarità dell'orbita di Urano, fino a concludere che a
perturbarla doveva essere un pianeta sconosciuto. Nettuno compare a meno di un
grado dalla posizione prevista. È di François Arago la formula rimasta
celebre: Le Verrier ha scoperto un pianeta «sulla punta della penna».

Per quasi due secoli la scienza ha funzionato così: leggi di natura scritte
come **equazioni differenziali** — regole compatte su come cambiano le cose —
e risolte, a mano finché si è potuto, poi al calcolatore. Previsioni del
tempo, gallerie del vento, reattori: sotto c'è sempre un'equazione.

Il machine learning di questo libro ha fatto finora l'esatto contrario:
niente leggi, solo dati, e una rete che scova le regolarità da sola. Ottimo
quando i dati abbondano e le leggi non le conosce nessuno — quale equazione
governa lo spam? Nelle scienze fisiche però è tutto rovesciato: le leggi le
conosciamo con precisione ammirevole, i dati sono pochi, costosi e rumorosi.
La domanda di questo capitolo, allora: e se si potessero usare *entrambi*?

## Una regola che dice come cambiano le cose

Prima di rispondere, facciamo pace con l'oggetto matematico al centro di
tutto. Un'equazione differenziale ha una particolarità che spiazza: non dice
mai *dov'è* la cosa che ti interessa, ma solo *come cambia*. Eppure basta.

`````{tab} Elementare

Posa una tazza di caffè bollente sulla scrivania. Nessuno sa scrivere d'un
fiato la sua temperatura tra dieci minuti, ma tutti conosciamo una regola più
semplice: **il caffè si raffredda tanto più in fretta quanto più è caldo** —
di corsa quando scotta, piano da tiepido, fermandosi a temperatura ambiente.

La regola parla solo del *cambiamento*, eppure da lì si ricostruisce tutta la
storia. Caffè a 80 °C, stanza a 20 °C; diciamo che ogni minuto il caffè perde
un decimo della differenza con la stanza: al primo minuto la differenza è 60,
quindi perde 6 gradi e scende a 74 °C; poi la differenza è 54, perde 5,4 e va
a 68,6 °C; poi 63,7 °C, e così via fino alla curva completa — ripida
all'inizio, sempre più piatta. Un'equazione differenziale è questo: una regola
sul cambiamento che, partendo da una condizione iniziale (80 °C al minuto
zero), inchioda tutto il futuro. Le leggi di Le Verrier erano regole dello
stesso tipo, con la gravità al posto del caffè.

`````

`````{tab} Superiore

La regola del caffè è la legge del raffreddamento di Newton, un'**equazione
differenziale ordinaria** (ODE):

$$
\frac{du}{dt} = -k\,\big(u(t) - T_a\big), \qquad u(0) = u_0,
$$

dove $u(t)$ è la temperatura al tempo $t$, $T_a$ quella dell'ambiente, $k>0$
una costante che dipende da tazza e materiale, e la **condizione iniziale**
$u(0)=u_0$ seleziona, tra le infinite soluzioni, quella del nostro caffè.
"Ordinaria" perché l'incognita dipende da una sola variabile; con più
variabili indipendenti si parla di **equazione alle derivate parziali** (PDE).
Il capostipite è l'equazione del calore per una sbarra,

$$
\frac{\partial u}{\partial t} = \alpha \frac{\partial^2 u}{\partial x^2},
$$

dove $u(x,t)$ è la temperatura nel punto $x$ al tempo $t$ e $\alpha$ la
diffusività termica; oltre alla condizione iniziale (il profilo a $t=0$)
servono **condizioni al contorno**: cosa accade agli estremi della sbarra.
Pochissime equazioni ammettono soluzioni in forma chiusa; per le altre si
passa al calcolatore, nello spirito dei richiami di analisi numerica: là
abbiamo accettato numeri a precisione finita, qui si accetta un continuo
fatto a punti — una **griglia** su $x$ e $t$, con le derivate rimpiazzate da
**differenze finite**, rapporti incrementali a passo piccolo ma non nullo.
Più fitta la griglia, migliore l'approssimazione e più salato il conto —
proibitivo con molte dimensioni o geometrie irregolari.

`````

## Una rete come candidata soluzione

Nel 2019 Maziar Raissi, Paris Perdikaris e George Karniadakis propongono di
saldare i due mondi {cite}`raissi2019physics`. L'idea delle **Physics-Informed
Neural Networks** (PINN) sta in una frase: una rete neurale come *candidata
soluzione* dell'equazione, penalizzata quando **viola la fisica**; i dati —
pochi, sporchi — e la legge — esatta — collaborano nella stessa loss.

`````{tab} Elementare

Immagina di dover disegnare la curva di raffreddamento del caffè avendo solo
tre misure di termometro, pure un po' ballerine. Una rete addestrata alla
vecchia maniera passerebbe vicino ai tre punti e, nel resto del grafico,
inventerebbe: tra una misura e l'altra potrebbe fare gobbe assurde, magari un
caffè che si riscalda da solo. La PINN aggiunge un secondo esaminatore. Il
primo, classico, controlla col righello che la curva passi vicino alle misure.
Il secondo apre il grafico in punti scelti *a caso* — anche dove nessuno ha
misurato niente — e verifica la regola: qui il caffè è a 60 °C, quindi deve
scendere con la giusta pendenza; se sale, o scende troppo piano, scatta una
penalità. Le penalità si sommano in un punteggio unico, e la rete aggiusta i
pesi per farlo calare. Dove ci sono dati comanda il righello, dove non ce ne
sono comanda la fisica — e la curva non può più inventare.

`````

`````{tab} Superiore

Sia $u_\theta(x,t)$ una rete neurale con parametri $\theta$ che riceve in
ingresso le coordinate $(x,t)$ e restituisce il valore della soluzione
candidata in quel punto. Per l'equazione del calore si definisce il **residuo
fisico**

$$
r_\theta(x,t) = \frac{\partial u_\theta}{\partial t}
- \alpha \frac{\partial^2 u_\theta}{\partial x^2},
$$

che vale zero esattamente dove la rete rispetta l'equazione. La loss somma due
richiami all'ordine:

$$
\mathcal{L}(\theta) =
\underbrace{\frac{1}{N_d} \sum_{i=1}^{N_d}
\big( u_\theta(x_i, t_i) - u_i \big)^2}_{\text{dati}}
\;+\;
\underbrace{\frac{\lambda}{N_c} \sum_{j=1}^{N_c}
r_\theta(x_j, t_j)^2}_{\text{fisica}},
$$

dove $(x_i, t_i, u_i)$ sono le $N_d$ misure disponibili (incluse condizioni
iniziali e al contorno), i $(x_j, t_j)$ sono $N_c$ **punti di collocazione**
estratti a caso nel dominio — nessuna griglia — e $\lambda$ bilancia i due
termini. Il tocco elegante è il calcolo delle derivate di $u_\theta$ rispetto
agli *ingressi*: le fornisce la **differenziazione automatica**, la stessa
regola della catena del backpropagation {cite}`rumelhart1986learning`
applicata a $x$ e $t$ anziché ai pesi. In PyTorch è una chiamata a
`torch.autograd.grad` {cite}`paszke2019pytorch`, e le derivate sono esatte a
meno della precisione di macchina: niente differenze finite, niente passo di
discretizzazione da scegliere.

`````

## Perché ci interessa

Tre proprietà rendono la ricetta interessante, tutte figlie della stessa
radice: la soluzione non vive più su una griglia, ma dentro una funzione
continua interrogabile ovunque. Primo: **niente griglia**. I punti di
collocazione si spargono a pioggia anche in domini dalla forma impossibile —
il condotto di un'aorta, il profilo di un'ala — dove costruire una buona
griglia è un mestiere a sé. Secondo: **dati scarsi**. Dove il laboratorio
arriva con tre sensori, la legge riempie i vuoti con l'unico filo coerente
con l'equazione. Terzo, il più sorprendente: i **problemi inversi**.

`````{tab} Elementare

Il problema *diretto* è quello del caffè: conosco la regola, ricostruisco la
curva. Il problema *inverso* lo ribalta: ho osservato la curva — o qualche suo
punto — e voglio scoprire un pezzo di regola che mi manca. Di notte la casa si
raffredda: dalle temperature segnate ora per ora, quanto isolano i muri? È la
domanda del medico legale (a che ora il decesso, data la temperatura del
corpo?) ed era la domanda di Le Verrier: dai disturbi nell'orbita di Urano,
dov'è il pianeta che non vedo? Per i solutori classici gli inversi sono
notoriamente scomodi: provare un valore, risolvere tutto, confrontare,
riprovare. Per una PINN il coefficiente ignoto è una manopola in più da
addestrare: la aggiusti finché fisica e misure vanno d'accordo.

`````

`````{tab} Superiore

Nel problema inverso un parametro dell'equazione — per esempio la diffusività
$\alpha$ — è incognito. Basta promuoverlo a variabile addestrabile e
minimizzare la stessa loss su entrambi,
$\hat{\theta},\hat{\alpha}=\arg\min_{\theta,\alpha}\mathcal{L}(\theta,\alpha)$:
il residuo dipende ora anche da $\alpha$, il cui gradiente arriva dalla stessa
passata di backpropagation. Nessun ciclo di tentativi: soluzione e parametro
fisico si stimano *insieme*, anche con misure rumorose e incomplete. È
soprattutto questa naturalezza sui problemi inversi ad aver fatto la fortuna
delle PINN {cite}`raissi2019physics`. Il filone che ne è nato — reti vincolate
dalla fisica, operatori neurali, scoperta di equazioni dai dati — va oggi
sotto il nome di **scientific machine learning**
{cite}`karniadakis2021physics`.

`````

## Un'onestà dovuta

Chiariamolo prima di innamorarcene: le PINN **non mandano in pensione i
solutori classici**. Su un problema standard — equazione nota, geometria
regolare, nessun dato da integrare — differenze finite ed elementi finiti
restano più veloci, più accurati e con garanzie di convergenza che
un'ottimizzazione non convessa non può offrire; una PINN può richiedere minuti
di addestramento dove un solutore maturo impiega millisecondi, e a volte
fallisce senza preavviso {cite}`karniadakis2021physics`. Il loro territorio è
un altro: dove dati e leggi vanno fusi nella stessa stima, dove la geometria
mette in crisi le griglie, dove il problema è inverso. Lì i solutori classici
arrancano, e la penna di Le Verrier torna a scrivere.

## Come è organizzato il capitolo

Dalla cornice al banco di lavoro: nella prossima sezione costruiremo il
metodo per intero, con una PINN in PyTorch — rete, residuo fisico, loss
composita, fino al problema inverso con un coefficiente che fingeremo di non
conoscere. Chiuderemo con le applicazioni reali — fluidodinamica, clima,
biomedicina — e una mappa onesta dei limiti: quando convengono, quando no.

```{admonition} Da ricordare
:class: important
- Un'**equazione differenziale** non dice dov'è una grandezza ma **come
  cambia**; con condizioni iniziali e al contorno questo basta a determinarla
  (ODE: una variabile indipendente; PDE: più di una).
- I solutori classici **discretizzano**: differenze finite su una griglia —
  accurati e veloci, ma in difficoltà su geometrie irregolari e molte
  dimensioni.
- Una **PINN** usa una rete $u_\theta(x,t)$ come candidata soluzione, con una
  loss doppia: aderenza ai (pochi) dati più penalità sul **residuo fisico**
  nei punti di collocazione {cite}`raissi2019physics`; derivate dalla
  **differenziazione automatica**, esatte e senza griglia.
- Punti di forza: dati scarsi ma leggi note, domini irregolari, **problemi
  inversi** (il parametro ignoto diventa una variabile addestrabile).
- Onestà: sui problemi standard i solutori classici restano superiori; le
  PINN sono un complemento, non un rimpiazzo {cite}`karniadakis2021physics`.
```
