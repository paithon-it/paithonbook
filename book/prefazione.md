# Prefazione

Questo libro è stato scritto una prima volta nel 2019, per uscire su carta.
Aveva un indice di quattordici capitoli. Su carta non è mai uscito, e il
manoscritto è rimasto in un cassetto: la prima forma in cui arriva a qualcuno
è questa, online e gratuita.

Con il senno di poi è stata una fortuna. Quell'indice si fermava al deep
reinforcement learning e prometteva codice in TensorFlow e Keras; la parola
*Transformer* non compariva da nessuna parte, benché l'articolo che la
introduce (*Attention Is All You Need*, di Vaswani e colleghi
{cite}`vaswani2017attention`) avesse già più di due anni, e il capitolo sul
linguaggio naturale finiva alle reti ricorrenti. Se questi nomi non ti dicono
niente, benissimo: sono esattamente le cose che imparerai. Non era una
distrazione, era la faccia che questa materia aveva, vista da chi ci lavorava,
in quel momento. Un libro stampato in quella forma avrebbe cominciato a
invecchiare il giorno stesso in cui arrivava in libreria.

Online il problema si pone in un altro modo. Un errore si corregge il giorno
in cui qualcuno lo segnala, e il testo cresce quando cresce il campo: del
manoscritto è rimasta l'ossatura, e attorno sono nati i capitoli che allora
non si potevano scrivere. Il codice, nel frattempo, è passato a PyTorch. Quello
che non è cambiato è il motivo per cui il libro esiste.

Il metodo lo trovi descritto nella {doc}`pagina di apertura </intro>`: ogni
concetto che conta è spiegato due volte, una con un'analogia che non chiede
prerequisiti e una con le formule al posto giusto, e decidi tu da quale parte
stare. Il resto è presto detto: nessuna scorciatoia sulle cose difficili,
nessun entusiasmo che il testo non sia in grado di giustificare, e le fonti
sempre citate, così che tu possa andare ad approfondire.

Lo scopo di questo libro è introdurre i concetti principali del machine
learning, del deep learning e del reinforcement learning (che cosa siano lo
dice il primo capitolo, in una riga per ciascuno, prima di dedicarci il
resto), e fornire gli strumenti utili a costruire un applicativo intelligente,
a valutare un'idea, o semplicemente a riconoscere l'intelligenza artificiale
nelle tecnologie di oggi e di domani.

C'è poi una cosa da dire su **come** è scritto, perché riguarda chi legge.
Buona parte di queste pagine nasce lavorando con l'intelligenza artificiale: è
lei a stendere, a cercare, a proporre. Questo libro è, alla lettera, **l'AI
che spiega se stessa**.

E lo fa **due volte**, in tutti e due i sensi della parola. Il primo lo trovi
in ogni capitolo, ed è quello di cui parlavo qui sopra: ogni concetto che
conta è spiegato una volta con un'analogia che non chiede prerequisiti e una
volta con le formule al posto giusto.

Il secondo senso riguarda la strada che il testo fa per arrivare fin qui.
Quello che una AI ha scritto lo rilegge **un'altra AI**, che alla stesura non
ha partecipato e ha un compito solo: cercare l'errore. Non riscrive per farlo
suonare meglio, controlla che le date siano quelle, che i numeri tornino, che
le derivazioni si reggano e che il codice giri davvero invece di sembrare
giusto. Chi ha scritto una cosa è l'ultimo a poterci trovare uno sbaglio, e
questo vale per una macchina esattamente come per una persona.

Solo dopo il testo arriva a me, e quello è il passaggio che decide che cosa
resta. Non è una formalità, ed è la parte che al posto mio una macchina non sa
fare: un modello punta ad avere ragione, un libro deve farsi capire, e non è
la stessa cosa. La prima si misura sui fatti, la seconda si misura su una
persona che prima non sapeva e adesso sa.

Così una spiegazione esatta ma fredda si riscrive finché non somiglia a come
la racconterei a voce; un esempio ineccepibile che però non fa scattare niente
si butta, anche se funziona; e una pagina che dice tutto il vero senza far
capire niente torna alla domanda da cui era nata, e si ricomincia da lì.

Il metodo è severo per una ragione precisa. Un modello sbaglia con la stessa
sicurezza con cui dice il vero, e su una materia tecnica sbaglia proprio dove
chi legge non ha modo di accorgersene: una data spostata di un anno, un numero
plausibile, una derivazione che sembra tornare. Perciò la regola qui è che
ogni affermazione fattuale si controlla sulle fonti primarie, ogni esempio
numerico si rifà a mano e ogni blocco di codice si esegue prima di finire in
queste pagine.

Su una cosa questo libro scommette apertamente. Il testo che leggi oggi è il
meglio che questo metodo sappia produrre adesso, non il meglio possibile:
qualche errore è rimasto, e più di una pagina si potrà spiegare meglio di
così. Ma un libro online non ha una tiratura da rincorrere con un foglietto di
errata corrige: si riscrive dove sbaglia, il giorno in cui qualcuno se ne
accorge. E gli strumenti con cui è scritto migliorano a loro volta, anno dopo
anno. Se mantengono quello che promettono, ogni versione dovrebbe arrivare un
po' più completa e un po' meno sbagliata della precedente, finché le
correzioni diventeranno rare e poi rarissime. È la scommessa che questo libro
fa su se stesso, e la si può verificare: {doc}`Aggiornamenti </aggiornamenti>`
tiene il conto di ogni correzione, una per una.

La responsabilità di quello che leggi è mia.

Perché è il non conoscere ciò che genera paura e alimenta false speranze.

% La firma e' in HTML perche' al sito serve l'allineamento a destra. In stampa
% il blocco raw sparisce, e la prefazione restava senza firma: si ripete per
% il solo LaTeX, com'e' gia' successo con l'attribuzione della citazione nella
% pagina di apertura.

```{raw} html
<p class="text-right mt-2"><em>Francesco Messina</em></p>
```

:::{only} latex
```{raw} latex
\vspace{4mm}\hfill\textit{Francesco Messina}\par
```
:::
