# Robotica e AI

La robotica e l’intelligenza artificiale vengono spesso confuse, ma non sono affatto la stessa cosa: sono due campi distinti, con oggetti diversi, anche se oggi profondamente intrecciati. Il modo più semplice per capire la differenza è pensare al corpo umano, in cui convivono il fisico e la mente; o a uno smartphone, composto dall’elettronica (touchscreen, antenne, processore) e dal sistema operativo che la governa (iOS o Android). La robotica costruisce il corpo; l’intelligenza artificiale prova a costruirne la mente. E come nel corpo umano, i due lavorano di continuo insieme: un robot moderno percepisce, riconosce e pianifica con tecniche che sono AI a tutti gli effetti.

In generale, un robot è una macchina che percepisce l’ambiente fisico e
agisce su di esso attraverso degli **attuatori** (motori, ruote, bracci,
pinze), eseguendo azioni quanto più possibile simili a quelle che erano le
intenzioni del suo “creatore”. Difatti, la robotica si concentra maggiormente
sulla meccanica del movimento e sul controllo della forza da applicare agli
oggetti al fine di manipolarli. E non pensiamo solamente agli “umanoidi”,
cioè ai robot con sembianze umane: sono robot anche i bracci meccanici delle
catene di montaggio, i rover in esplorazione su Marte, perfino le semplici
lavatrici che ognuno di noi utilizza in casa, o una casa stessa, se possiede
un minimo di domotica, cioè impianti capaci di regolarsi da soli. Per essere
un robot non serve spostarsi; basta sentire e agire.

Esiste però un ponte tra i due mondi, ed è una branca dell’intelligenza
artificiale che va sotto il nome di **Reinforcement Learning** (in italiano,
apprendimento per rinforzo): programmi che imparano per tentativi ed errori,
collezionando un “premio” ogni volta che fanno bene; un po’ come si addestra
un cucciolo, premiandolo quando obbedisce. È con queste tecniche (e con le
loro versioni potenziate dalle reti neurali, il Deep Reinforcement Learning)
che oggi si insegna a un robot a camminare, ad afferrare oggetti o a mantenere
l’equilibrio; ne parleremo negli ultimi capitoli del libro.

`````{tab} Elementare
Prendi un robot che deve imparare a camminare. Nessun ingegnere gli spiega
come piegare le ginocchia: si stabilisce solo la regola del gioco, per esempio
*un punto per ogni secondo in cui resti in piedi*. Ai primi tentativi crolla
quasi subito: $2$ punti, poi $3$, poi di nuovo $2$. Ma tra mille prove
qualcuna va meglio, e il robot tende a ripetere ciò che ha preceduto i
punteggi alti: dopo migliaia di cadute arriva a $10$, $50$, $500$ punti, cioè
cammina. È il gioco “acqua–fuochino” portato all’estremo: nessuna istruzione,
solo un segnale che dice *così va meglio, così va peggio*.
`````

`````{tab} Superiore
Nel formalismo che svilupperemo negli ultimi capitoli: un **agente** osserva
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

Di questa storia conviene fissare subito i tre nomi, perché torneranno in
tutti gli ultimi capitoli. Si chiama **agente** chi decide, cioè il robot
dell’esempio; **ambiente** tutto il resto con cui l’agente ha a che fare (il
pavimento, la gravità, il cronometro che conta i secondi in piedi); e
**ricompensa** il punteggio che l’ambiente gli restituisce dopo ogni mossa.
Il disegno qui sotto mette in fila queste tre parole e nient’altro.

```{figure} ../figures/reinforcement-learning-agenti-stati-azioni.svg
:name: fig-agente-ambiente
:alt: "Anello fra due blocchi: l'agente invia un'azione all'ambiente; l'ambiente restituisce all'agente il nuovo stato e una ricompensa numerica, e il giro ricomincia. Nessun altro canale collega i due: tutto ciò che l'agente sa del mondo passa da stato e ricompensa."
:width: 88%

Il giro che regge tutti gli ultimi capitoli: l’agente manda la sua mossa,
l’ambiente risponde con la nuova situazione e con la ricompensa, e si
ricomincia. Non passa altro: nessuno spiega mai all’agente *perché* quella
ricompensa sia arrivata.
```

Tra i due, come mostra {numref}`fig-agente-ambiente`, passa pochissimo, ed è
proprio questo a rendere il problema difficile e interessante: un numero solo,
per giunta spesso in ritardo di molte mosse rispetto alla scelta che l’ha
causato, è tutto quello che l’agente riceve per capire come comportarsi. Mai
una spiegazione, mai la mossa giusta scritta da qualche parte.

E quando la “mente” artificiale entra in un corpo meccanico, i risultati si vedono: rover marziani che scelgono da soli il percorso evitando le rocce, robot chirurgici che aiutano il medico a eseguire incisioni più precise e meno invasive di quelle di una mano umana, magazzini in cui flotte di carrelli autonomi si coordinano senza scontrarsi.

Le applicazioni crescono giorno dopo giorno, e sempre più spesso l’intelligenza artificiale sconfina, contaminando positivamente campi in cui la ricerca sembrava arrivata a un punto morto.
