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
meccanici delle catene di montaggio, i *rover* che esplorano Marte, gli
aspirapolvere che girano per casa. Per essere un robot non serve nemmeno
spostarsi: il braccio fisso di una catena di montaggio non si sposta di un
centimetro, e
nessuno gli nega il titolo. Basta sentire e agire.

Fin dove si può tirare questa definizione? Esiste un vocabolario
internazionale, la norma ISO 8373, rifatta nel 2021, in cui i tecnici di tutto
il mondo si mettono d'accordo su che cosa chiamare robot. È più stretta della
nostra: chiede che la macchina si sposti, maneggi oggetti o porti qualcosa in
un punto preciso, e che lo faccia con un certo grado di autonomia, cioè
decidendo da sé in base a quello che sente. Ne resta fuori la
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

Ogni tentativo lascia una traccia: il robot si segna che cosa ha fatto e quanti
punti ne ha ricavato. Un punto incassato subito conta più di uno che arriverà
fra dieci secondi, e i lontanissimi quasi niente: così due prove si confrontano
sempre, anche se una non finisse mai, perché ogni secondo vale un punto e non
di più.

Le mosse non le sceglie a colpo sicuro, le sorteggia, come tirando un dado; e
quale dado tira dipende da com'è messo il corpo in quel momento, sbilanciato in
avanti, piegato su un ginocchio, in equilibrio. Un dado strano, però, perché
quello che sorteggia è la spinta da dare a un motore, e quella può valere
qualunque cosa fra il minimo e il massimo: più che scegliere fra alcune mosse,
decide con quanta forza fare quella che sta facendo.

All'inizio i dadi sono
onesti e ogni spinta ha la stessa probabilità di uscire. Dopo ogni tornata di
prove il programma li ritocca di pochissimo, ed è qui che l'imparare succede:
sui dadi tirati nelle prove andate bene rende un po’ più facile l'uscita delle
mosse fatte, su quelli delle prove andate male un po’ più difficile. Ripeti
migliaia di volte e i dadi si sbilanciano sempre di più verso il camminare; il
punteggio sale a $10$, poi a $50$, poi a $500$.

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
un agente osserva lo stato $s_t$ dell'ambiente, sceglie un'azione $a_t$ secondo
una policy $\pi(a \mid s)$ e riceve una ricompensa $r_{t+1}$; l'obiettivo è
trovare la policy che massimizza il ritorno atteso
$\mathbb{E}_{\pi}\!\left[\sum_{t=0}^{\infty} \gamma^{\,t} r_{t+1}\right]$,
dove la media è presa sulle traiettorie che la policy $\pi$ genera e
$\gamma \in [0, 1)$ sconta le ricompense future; con ricompense limitate è lo
sconto a rendere finita una somma di infiniti termini, e senza quel limite la
convergenza non è più garantita. Per la robotica, con azioni continue
(coppie ai motori), si usano i metodi a gradiente di policy; l'addestramento
avviene in simulazione, con il passaggio al robot fisico (*sim-to-real*) come
problema aperto.
`````

Questi nomi vanno fissati adesso, perché torneranno per intero nei due
{doc}`capitoli sul reinforcement learning </ReinforcementLearning/overview>`.
Si chiama **agente** chi decide, cioè il robot dell'esempio, e **ambiente**
tutto il resto con cui ha a che fare: il pavimento, la gravità, il cronometro
che conta i secondi in piedi. Lo **stato** è la fotografia della situazione nel
momento in cui l'agente deve decidere: com'è messo il corpo, a che velocità sta
cadendo. La **ricompensa** è il punteggio che l'ambiente gli restituisce dopo
ogni mossa. La **policy** è la regola con cui sceglie la mossa, ed è poi la
cosa che deve imparare; in italiano si traduce «politica», ma è una parola che
porta fuori strada, e ovunque troverai scritto *policy*.

La {numref}`fig-agente-ambiente` mette in fila queste parole e nient'altro. Le
letterine in basso segnano soltanto il momento: $a_t$ è «l'azione alla mossa
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
