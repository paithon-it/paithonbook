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
perché è quello con cui l'intelligenza artificiale ha a che fare; ma una
lavatrice resta fuori anche dall'altra definizione, quella dei {doc}`compiti
per cui nessuno sa scrivere una ricetta </Introduzione/overview>`: la ricetta
per fermare il risciacquo esiste, ed è corta.

Esiste però un ponte fra i due mondi, ed è fatto di programmi che imparano per
tentativi ed errori, incassando un «premio» ogni volta che fanno bene, un po’
come si addestra un cucciolo premiandolo quando obbedisce. Quel ramo
dell'intelligenza artificiale si chiama **reinforcement learning**
(apprendimento per rinforzo). È così che oggi si insegna a un robot a
camminare, ad afferrare oggetti o a mantenere l'equilibrio, e quando a fare il
lavoro sono le reti neurali si parla di *deep reinforcement learning*, cioè lo
stesso con le reti a molti strati.

`````{tab} Elementare
Un robot deve imparare a camminare, e nessun ingegnere gli spiega come piegare
le ginocchia: si stabilisce solo la regola del gioco, per esempio *un punto per
ogni secondo in cui resti in piedi*. Ai primi tentativi crolla quasi subito:
$2$ punti, poi $3$, poi di nuovo $2$.

Ogni tentativo, però, lascia una traccia: il robot si segna che cosa ha fatto e
quanti punti ne ha ricavato. Nel farsi i conti dà più peso ai punti vicini nel
tempo; uno che arriverà fra molto conta un po’ meno, e i lontanissimi quasi
niente. Così anche una camminata che non finisse mai varrebbe, nei suoi conti,
un totale preciso, e due modi di camminare si possono sempre confrontare.

Le mosse non le sceglie a colpo sicuro, le sorteggia, come tirando dei dadi. Di
dadi ne tiene uno per ogni situazione in cui il corpo può trovarsi (sbilanciato
in avanti, piegato su un ginocchio, in equilibrio), e a ogni istante tira
quello che corrisponde a com'è messo in quel momento. Dadi strani, però: le
facce non sono sei e non sono nemmeno un numero fisso, perché quello che il
robot sorteggia è la spinta da dare a un motore, e quella può valere qualunque
cosa fra il minimo e il massimo. Più che scegliere fra alcune mosse, il dado
decide con quanta forza fare quella che sta facendo. All'inizio tutti i dadi
sono onesti e ogni mossa ha la stessa probabilità di uscire. Dopo ogni prova il
programma li ritocca di pochissimo, e questo è il punto in cui l'imparare
succede: sui dadi tirati nelle prove andate bene rende un po’ più facile
l'uscita delle mosse fatte, sui dadi tirati nelle prove andate male un po’ più
difficile. Ripeti migliaia di volte e i dadi si sbilanciano sempre di più verso
il camminare; il punteggio sale a $10$, poi a $50$, poi a $500$.

Assomiglia al gioco «acqua-fuochino», con due differenze che sono poi tutta la
difficoltà del problema. La prima: nel gioco c'è una persona che sa già dov'è
l'oggetto nascosto, qui non c'è nessuno che sappia come si cammina. La seconda:
il «fuochino» non giudica i passi uno per uno. Il totale del giro racconta
quanto è andata bene la prova in tutto, ma fra le centinaia di tiri di dadi non
segna quale abbia tenuto il robot in piedi e quale l'abbia fatto cadere.

Restano le migliaia di cadute, che il robot vero non potrebbe permettersi: si
sfascerebbe alla decima. Al posto suo cade una copia dentro una simulazione al
computer, una specie di videogioco fedele del suo corpo. È lì che il gioco si
può rompere: il pavimento vero è un filo più scivoloso di quello simulato, i
motori veri rispondono con un soffio di ritardo, e la camminata che nella copia
era perfetta inciampa appena tocca il mondo. Portare quel che ha imparato dal
videogioco al metallo, senza perderlo per strada, resta un problema aperto.
`````

`````{tab} Superiore
Nel formalismo che svilupperemo nei due capitoli sul reinforcement learning:
un **agente** osserva
lo stato $s_t$ dell'ambiente, sceglie un'azione $a_t$ secondo una
**policy** $\pi(a \mid s)$ e riceve una ricompensa $r_{t+1}$; l'obiettivo è
trovare la policy che massimizza il ritorno atteso
$\mathbb{E}_{\pi}\!\left[\sum_{t=0}^{\infty} \gamma^{\,t} r_{t+1}\right]$,
dove la media è presa sulle traiettorie che la policy $\pi$ genera e
$\gamma \in [0, 1)$ sconta le ricompense future; con ricompense limitate è lo
sconto a rendere finita una somma di infiniti termini, e senza quel limite la
serie diverge anche per $\gamma < 1$. Per la robotica, con azioni continue
(coppie ai motori), si usano i metodi a gradiente di policy; l'addestramento
avviene in simulazione, con il passaggio al robot fisico (*sim-to-real*) come
problema aperto.
`````

Questi nomi vanno fissati adesso, perché torneranno per intero nei due
{doc}`capitoli sul reinforcement learning </ReinforcementLearning/overview>`.
Si chiama **agente** chi decide, cioè il robot dell'esempio; **ambiente** tutto il resto con cui l'agente ha a
che fare (il pavimento, la gravità, il cronometro che conta i secondi in
piedi); **stato** la fotografia della situazione in cui l'agente si trova nel
momento in cui deve decidere (com'è messo il corpo, a che velocità sta
cadendo); **ricompensa** il punteggio che l'ambiente gli restituisce dopo ogni
mossa; e **policy** la regola con cui l'agente sceglie la mossa, che è poi la
cosa che deve imparare (in italiano si traduce «politica», ma è una parola che
porta fuori strada, e ovunque troverai scritto *policy*). La
{numref}`fig-agente-ambiente` mette in fila queste parole e nient'altro; le
letterine in basso segnano soltanto il momento, $a_t$ è «l'azione alla mossa
$t$», $r_{t+1}$ «la ricompensa che arriva subito dopo» e $s_{t+1}$ «lo stato
alla mossa dopo».

```{figure} ../figures/reinforcement-learning-agenti-stati-azioni.svg
:name: fig-agente-ambiente
:alt: "Anello fra due blocchi: l'agente invia un'azione all'ambiente; l'ambiente restituisce all'agente il nuovo stato e una ricompensa numerica, e il giro ricomincia. Nessun altro canale collega i due: tutto ciò che l'agente sa del mondo passa da stato e ricompensa."
:width: 88%

L'anello fra l'agente e l'ambiente: l'agente manda la sua mossa, l'ambiente
risponde con la nuova situazione e con la ricompensa, e si ricomincia. Non
passa nient'altro.
```

Tra i due passa pochissimo, ed è proprio questo a rendere il problema
difficile e interessante. Torna indietro la nuova situazione, e torna indietro
un numero: quel numero è l'unico
giudizio che l'agente riceverà mai sul proprio operato, e per giunta arriva
spesso in ritardo di molte mosse rispetto alla scelta che l'ha causato: il
robot cade adesso per un passo storto di tre secondi fa. Mai una spiegazione,
mai la mossa giusta scritta da qualche parte. Capire a quale mossa, fra le
tante, vada assegnato il merito di un punto arrivato dopo è il problema
centrale di questo campo, e i due capitoli dedicati non fanno altro che
girargli attorno.

Fuori dai corpi meccanici, poi, l'intelligenza artificiale sconfina sempre più
spesso in campi dove la ricerca sembrava arrivata a un punto morto:
raffreddare un capannone pieno di computer accesi, leggere un
elettrocardiogramma, e un problema di biologia rimasto aperto per mezzo
secolo. Sono i tre esempi da cui {doc}`si riparte </Introduzione/conclusione>`.
