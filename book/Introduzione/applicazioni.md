# Robotica e AI

Un rover su Marte sceglie da solo dove mettere le ruote fra le rocce, un drone
si rimette dritto in mezzo a una raffica, in un magazzino decine di carrelli
si incrociano senza toccarsi. In tutti e tre i casi c'è un corpo che si muove
e qualcosa che decide come muoverlo, e sono due mestieri diversi: la robotica
costruisce il corpo, l'intelligenza artificiale prova a costruire la mente che
lo comanda. Come nel corpo umano, poi, i due non si staccano mai: un robot
moderno vede, riconosce e decide con tecniche che sono AI a tutti gli effetti.

Un robot, in generale, è una macchina che sente l'ambiente fisico e agisce su
di esso attraverso degli **attuatori**, cioè le parti che si muovono: motori,
ruote, bracci, pinze. Il mestiere della robotica è tutto lì: far muovere quelle
parti nel modo voluto e dosare la forza con cui toccano le cose. E non pensiamo
solo agli «umanoidi», i robot con sembianze umane: sono robot anche i bracci
meccanici delle catene di montaggio, i *rover* che esplorano Marte (cioè i
veicoli a ruote che si muovono da soli su un altro pianeta), gli aspirapolvere
che girano per casa. Per essere un robot non serve nemmeno spostarsi: il
braccio fisso di una catena di montaggio non si sposta di un centimetro, e
nessuno gli nega il titolo. Basta sentire e agire.

Fin dove si può tirare questa definizione? Esiste un vocabolario
internazionale, la norma ISO 8373, in cui i tecnici di tutto il mondo si
mettono d'accordo su che cosa chiamare robot. È più stretto della nostra
definizione: chiede che la macchina faccia almeno una di tre cose, spostarsi,
afferrare, oppure portare qualcosa in un punto preciso. Ne resta fuori la
lavatrice, che pure sente (il carico, la temperatura) e agisce (apre la
valvola, ferma il cestello). A noi qui interessa lo schema, sentire e agire,
perché è quello con cui l'intelligenza artificiale ha a che fare; ma Una
lavatrice resta fuori anche dall'altra definizione, quella data in {doc}`apertura di capitolo </Introduzione/overview>`: la regola
che decide quando fermare il risciacquo l'ha scritta un
tecnico, riga per riga.

Esiste però un ponte fra i due mondi, ed è fatto di programmi che imparano per
tentativi ed errori, incassando un «premio» ogni volta che fanno bene, un po’
come si addestra un cucciolo premiandolo quando obbedisce. Quel ramo
dell'intelligenza artificiale si chiama **reinforcement learning**
(apprendimento per rinforzo). È così che oggi si insegna a un robot a
camminare, ad afferrare oggetti o a mantenere l'equilibrio, e quando a fare il
lavoro sono le reti neurali si parla di *deep reinforcement learning*, cioè lo
stesso con le reti a molti strati. Ne parleremo per esteso nei due capitoli
dedicati all'uno e all'altro.

`````{tab} Elementare
Prendi un robot che deve imparare a camminare. Nessun ingegnere gli spiega come
piegare le ginocchia: si stabilisce solo la regola del gioco, per esempio *un
punto per ogni secondo in cui resti in piedi*. Ai primi tentativi crolla quasi
subito: $2$ punti, poi $3$, poi di nuovo $2$.

Ogni tentativo, però, lascia una traccia: il robot si segna che cosa ha fatto e
quanti punti ne ha ricavato. E le mosse non le sceglie a colpo sicuro, le
sorteggia, come tirando dei dadi: all'inizio i dadi sono onesti e ogni mossa ha
la stessa probabilità di uscire. Dopo ogni prova il programma li ritocca di
pochissimo, e questo è il punto in cui l'imparare succede: rende un po’ più
facile l'uscita delle mosse comparse nelle prove andate bene, un po’ più
difficile quella delle mosse comparse nelle prove andate male. Ripeti migliaia
di volte e i dadi si sbilanciano sempre di più verso il camminare; il punteggio
sale a $10$, poi a $50$, poi a $500$.

Assomiglia al gioco «acqua-fuochino», con due differenze che sono poi tutta la
difficoltà del problema. La prima: nel gioco c'è una persona che sa già dov'è
l'oggetto nascosto, qui non c'è nessuno che sappia come si cammina. La seconda:
il «fuochino» non arriva mentre ti muovi, arriva alla fine del giro, ed è un
numero solo, che dice quanto è andata bene in tutto e non quale passo fosse
quello buono.
`````

`````{tab} Superiore
Nel formalismo che svilupperemo nei due capitoli sul reinforcement learning:
un **agente** osserva
lo stato $s_t$ dell'ambiente, sceglie un'azione $a_t$ secondo una
**politica** $\pi(a \mid s)$ e riceve una ricompensa $r_{t+1}$; l'obiettivo è
trovare la politica che massimizza il ritorno atteso
$\mathbb{E}_{\pi}\!\left[\sum_{t=0}^{\infty} \gamma^{\,t} r_{t+1}\right]$,
dove la media è presa sulle traiettorie che la politica $\pi$ genera e
$\gamma \in [0, 1)$ sconta le ricompense future, il che è anche quello che
rende finita una somma di infiniti termini. Per la robotica, con azioni continue (coppie ai motori), si usano i
metodi a gradiente di policy e actor-critic; l'addestramento avviene in
simulazione, con il passaggio al robot fisico (*sim-to-real*) come problema
aperto. Tutti questi termini avranno il loro capitolo: qui basta la sagoma
del meccanismo.
`````

Di questa storia conviene fissare subito i nomi, perché torneranno per intero
nei due {doc}`capitoli sul reinforcement learning </ReinforcementLearning/overview>`. Si chiama **agente** chi decide,
cioè il robot dell'esempio; **ambiente** tutto il resto con cui l'agente ha a
che fare (il pavimento, la gravità, il cronometro che conta i secondi in
piedi); **stato** la fotografia della situazione in cui l'agente si trova nel
momento in cui deve decidere (com'è messo il corpo, a che velocità sta
cadendo); **ricompensa** il punteggio che l'ambiente gli restituisce dopo ogni
mossa; e **policy** la regola con cui l'agente sceglie la mossa, che è poi la
cosa che deve imparare (in italiano si traduce «politica», ma è una parola che
porta fuori strada, e ovunque troverai scritto *policy*). Il disegno qui sotto
mette in fila queste parole e nient'altro; le letterine in basso segnano
soltanto il momento, $a_t$ è «l'azione alla mossa $t$» e $s_{t+1}$ «lo stato
alla mossa dopo».

```{figure} ../figures/reinforcement-learning-agenti-stati-azioni.svg
:name: fig-agente-ambiente
:alt: "Anello fra due blocchi: l'agente invia un'azione all'ambiente; l'ambiente restituisce all'agente il nuovo stato e una ricompensa numerica, e il giro ricomincia. Nessun altro canale collega i due: tutto ciò che l'agente sa del mondo passa da stato e ricompensa."
:width: 88%

Il giro che regge i due capitoli sul reinforcement learning: l'agente manda la
sua mossa, l'ambiente risponde con la nuova situazione e con la ricompensa, e
si ricomincia. Non passa altro: nessuno spiega mai all'agente *perché* quella
ricompensa sia arrivata.
```

Tra i due, come mostra {numref}`fig-agente-ambiente`, passa pochissimo, ed è
proprio questo a rendere il problema difficile e interessante. Torna indietro
la nuova situazione, e torna indietro un numero: quel numero è l'unico
giudizio che l'agente riceverà mai sul proprio operato, e per giunta arriva
spesso in ritardo di molte mosse rispetto alla scelta che l'ha causato: il
robot cade adesso per un passo storto di tre secondi fa. Mai una spiegazione,
mai la mossa giusta scritta da qualche parte. Capire a quale mossa, fra le
tante, vada assegnato il merito di un punto arrivato dopo è il problema
centrale di questo campo, e i due capitoli dedicati non fanno altro che
girargli attorno.

E quando la «mente» artificiale entra in un corpo meccanico, i risultati si
vedono: rover marziani che scelgono da soli il percorso evitando le rocce,
magazzini in cui flotte di carrelli autonomi si coordinano senza scontrarsi,
droni che si stabilizzano da soli in mezzo alle raffiche.

Le applicazioni crescono giorno dopo giorno, e sempre più spesso
l'intelligenza artificiale sconfina in campi dove la ricerca sembrava arrivata
a un punto morto: raffreddare un capannone pieno di computer accesi, leggere
un elettrocardiogramma, e un problema di biologia rimasto aperto per mezzo
secolo. Sono i tre esempi da cui
{doc}`si riparte </Introduzione/conclusione>`.
