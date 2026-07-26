# Robotica e AI

La robotica e l’intelligenza artificiale vengono spesso confuse, ma non sono affatto la stessa cosa: in realtà, i due campi sono quasi completamente separati e hanno scopi molto diversi. Il modo più semplice per capire la differenza è pensare al corpo umano, in cui convivono il fisico e la mente; o a uno smartphone, composto dall’elettronica (touchscreen, antenne, processore) e dal sistema operativo che la governa (iOS o Android). La robotica costruisce il corpo; l’intelligenza artificiale prova a costruirne la mente.

In generale, un robot è un dispositivo meccanico che si muove nello spazio
autonomamente, eseguendo azioni quanto più possibile simili a quelle che erano
le intenzioni del suo “creatore”. Difatti, la robotica si concentra
maggiormente sulla meccanica del movimento e sul controllo della forza da
applicare agli oggetti al fine di manipolarli. E non pensiamo solamente agli
“umanoidi”, cioè ai robot con sembianze umane: sono robot anche i bracci
meccanici delle catene di montaggio, i rover in esplorazione su Marte, perfino
le semplici lavatrici che ognuno di noi utilizza in casa, o una casa stessa,
se possiede un minimo di domotica, cioè impianti capaci di regolarsi da soli.

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

E quando la “mente” artificiale entra in un corpo meccanico, i risultati si vedono: rover marziani che scelgono da soli il percorso evitando le rocce, robot chirurgici che aiutano il medico a eseguire incisioni più precise e meno invasive di quelle di una mano umana, magazzini in cui flotte di carrelli autonomi si coordinano senza scontrarsi.

Le applicazioni crescono giorno dopo giorno, e sempre più spesso l’intelligenza artificiale sconfina, contaminando positivamente campi in cui la ricerca sembrava arrivata a un punto morto.
