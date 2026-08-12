# Robotica e AI

La robotica e l’intelligenza artificiale vengono spesso confuse, ma non sono affatto la stessa cosa: sono due campi distinti, con oggetti diversi, anche se oggi profondamente intrecciati. Il modo più semplice per capire la differenza è pensare al corpo umano, in cui convivono il fisico e la mente; o a uno smartphone, composto dall’elettronica (touchscreen, antenne, processore) e dal sistema operativo che la governa (iOS o Android). La robotica costruisce il corpo; l’intelligenza artificiale prova a costruirne la mente. E come nel corpo umano, i due lavorano di continuo insieme: un robot moderno percepisce, riconosce e pianifica con tecniche che sono AI a tutti gli effetti.

In generale, un robot è una macchina che percepisce l’ambiente fisico e
agisce su di esso attraverso degli **attuatori** (motori, ruote, bracci,
pinze), eseguendo azioni quanto più possibile simili a quelle che erano le
intenzioni del suo “creatore”. Difatti, la robotica si concentra maggiormente
sulla meccanica del movimento e sul controllo della forza da applicare agli
oggetti al fine di manipolarli. E non pensiamo solamente agli “umanoidi”,
cioè ai robot con sembianze umane: sono robot anche i bracci meccanici delle
catene di montaggio, i rover in esplorazione su Marte, gli aspirapolvere che
girano per casa da soli. Per essere un robot non serve spostarsi: il braccio
fisso di una catena di montaggio non si sposta di un centimetro, e nessuno gli
nega il titolo. Basta sentire e agire.

Fin dove si può tirare questa definizione? La definizione tecnica del campo
(la norma ISO 8373) è più stretta della nostra e chiede che la macchina compia
almeno una fra locomozione, manipolazione e posizionamento: lascia quindi
fuori la lavatrice, che pure sente (il carico, la temperatura) e agisce (apre
la valvola, ferma il cestello). A noi qui interessa lo schema, sentire e
agire, perché è quello con cui l'intelligenza artificiale ha a che fare; ma
vale la pena ricordare che una lavatrice resta fuori anche dall'altra
definizione, quella data nella pagina precedente: la regola che decide quando
fermare il risciacquo l'ha scritta un tecnico, riga per riga.

Esiste però un ponte tra i due mondi, ed è una branca dell’intelligenza
artificiale che va sotto il nome di **Reinforcement Learning** (in italiano,
apprendimento per rinforzo): programmi che imparano per tentativi ed errori,
collezionando un “premio” ogni volta che fanno bene; un po’ come si addestra
un cucciolo, premiandolo quando obbedisce. È con queste tecniche (e con le
loro versioni potenziate dalle reti neurali, il Deep Reinforcement Learning)
che oggi si insegna a un robot a camminare, ad afferrare oggetti o a mantenere
l'equilibrio; ne parleremo per esteso nei due capitoli dedicati al
reinforcement learning e alla sua versione profonda.

`````{tab} Elementare
Prendi un robot che deve imparare a camminare. Nessun ingegnere gli spiega
come piegare le ginocchia: si stabilisce solo la regola del gioco, per esempio
*un punto per ogni secondo in cui resti in piedi*. Ai primi tentativi crolla
quasi subito: $2$ punti, poi $3$, poi di nuovo $2$. Ma tra mille prove
qualcuna va meglio, e il robot tende a ripetere ciò che ha preceduto i
punteggi alti: dopo migliaia di cadute arriva a $10$, $50$, $500$ punti, cioè
cammina. È il gioco “acqua-fuochino” portato all’estremo: nessuna istruzione,
solo un segnale che dice *così va meglio, così va peggio*.
`````

`````{tab} Superiore
Nel formalismo che svilupperemo nei due capitoli sul reinforcement learning:
un **agente** osserva
lo stato $s_t$ dell’ambiente, sceglie un’azione $a_t$ secondo una
**politica** $\pi(a \mid s)$ e riceve una ricompensa $r_{t+1}$; l’obiettivo è
trovare la politica che massimizza il ritorno atteso $\mathbb{E}\!\left[\sum_t
\gamma^{\,t} r_{t+1}\right]$, dove $\gamma \in [0, 1)$ sconta le ricompense
future. Per la robotica, con azioni continue (coppie ai motori), si usano i
metodi a gradiente di policy e actor-critic; l’addestramento avviene in
simulazione, con il passaggio al robot fisico (*sim-to-real*) come problema
aperto. Tutti questi termini avranno il loro capitolo: qui basta la sagoma
del meccanismo.
`````

Di questa storia conviene fissare subito i nomi, perché torneranno per intero
nei due capitoli sul reinforcement learning. Si chiama **agente** chi decide,
cioè il robot dell'esempio; **ambiente** tutto il resto con cui l'agente ha a
che fare (il pavimento, la gravità, il cronometro che conta i secondi in
piedi); **stato** la fotografia della situazione in cui l'agente si trova nel
momento in cui deve decidere (com'è messo il corpo, a che velocità sta
cadendo); **ricompensa** il punteggio che l'ambiente gli restituisce dopo ogni
mossa; e **policy** (in italiano *politica*) la regola con cui l'agente
sceglie la mossa, che è poi la cosa che deve imparare. Il disegno qui sotto
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
spesso in ritardo di molte mosse rispetto alla scelta che l'ha causato (il
robot cade adesso per un passo storto di tre secondi fa). Mai una spiegazione,
mai la mossa giusta scritta da qualche parte.

E quando la «mente» artificiale entra in un corpo meccanico, i risultati si
vedono: rover marziani che scelgono da soli il percorso evitando le rocce,
magazzini in cui flotte di carrelli autonomi si coordinano senza scontrarsi,
droni che si stabilizzano da soli in mezzo alle raffiche.

Le applicazioni crescono giorno dopo giorno, e sempre più spesso
l'intelligenza artificiale sconfina in campi dove la ricerca sembrava arrivata
a un punto morto: prevedere la forma tridimensionale delle proteine, per dirne
uno, era un problema aperto da cinquant'anni prima che una rete neurale
cominciasse a farlo bene, e ne riparleremo nella pagina di chiusura del
capitolo.
