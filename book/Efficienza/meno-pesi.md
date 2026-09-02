# Meno pesi: la promessa che si riscuote male

Novanta pesi su cento si possono buttare via e la rete continua a rispondere
quasi come prima. Non è un modo di dire: più sotto è misurato, su una rete
vera, con i numeri stampati dal programma.

È la promessa più grande delle tre, e quello che succede quando la si va a
riscuotere è meno allegro. Perché la rete alleggerita del novanta per cento,
sul calcolatore, **non è più veloce in proporzione a quanto si è
alleggerita**: con lo stesso conto di prima va uguale, e cambiare il conto
conviene solo a certe condizioni.

## Quali pesi si tolgono

La domanda è quale peso togliere, e la risposta che si usa quasi sempre è la
più sbrigativa che ci sia.

`````{tab} Elementare

Un club vince il campionato e a giugno il presidente deve mandare via nove
giocatori su dieci. Chiunque mandi via, la squadra ci rimette, e la domanda è
solo chi costa meno perdere. Quanto valga davvero ciascuno non lo sa nessuno, e
allora si guardano i minuti giocati e si manda via chi ne ha di meno.

In una rete quella cifra c'è già, e sono i pesi. Il peso di un collegamento
dice quanto quel collegamento conta, e uno vicino a zero non sposta quasi
niente: tagliarlo sembra gratis. Si ordinano tutti per grandezza, si tiene una
percentuale dei più grandi, il resto va a zero. Che lavoro faccia ciascun
collegamento non lo guarda nessuno.

Il presidente scommette quattro volte, e perde tutte e quattro. Scommette che
la squadra fosse messa nel modo migliore possibile, mentre a giugno sta dove il
campionato l'ha lasciata. Che due giocatori con gli stessi minuti lascino lo
stesso buco: il portiere gioca quanto un difensore e non si rimpiazza con la
stessa facilità. Che i buchi si sommino:
due che si intendevano bene, tolti insieme, si sentono più della somma dei due
presi uno per uno. E che il conto regga anche a tagliare in blocco: vale per un
giocatore alla volta, e di pesi se ne azzerano nove su dieci in un pomeriggio.
La cosa curiosa è che il criterio funzioni lo stesso, e nessuno sa dire bene
perché.

La prima domenica dice come è andata. Tolti nove pesi su dieci, l'accuratezza
passa da novantotto a trentanove per cento, cioè da «sbaglia una volta su
cinquanta» a «sbaglia tre volte su cinque».

Quello che salva la squadra è il ritiro. Dopo aver tagliato **si riaddestra**,
tenendo però i tagli dove sono: i giocatori rimasti si allenano nei ruoli
lasciati vuoti, e dopo qualche settimana si gioca di nuovo bene. Chi è stato
mandato via non rientra: dopo ogni seduta i pesi tagliati vengono rimessi a
zero, così nessuno di loro può riprendersi il posto.

Il ritiro riesce meglio a scaglioni. Mandarne via nove su dieci in un
pomeriggio e poi allenare quel che resta funziona peggio che tagliarne pochi,
allenare, tagliarne altri pochi: ogni volta si chiede alla squadra un
aggiustamento piccolo, e lo regge.

E a un certo punto nemmeno il ritiro basta. Con la metà dei pesi tolti la rete
non perde niente; con nove su dieci resta indietro di meno di un punto; con
diciannove su venti di quasi cinque. Il ritiro insegna a coprire i buchi, non
a essere in due dove ne servono undici, e sotto un certo numero di giocatori
non c'è allenamento che tenga.

`````

`````{tab} Superiore

La **potatura per grandezza** azzera i pesi il cui valore assoluto sta sotto
una soglia, tipicamente scelta per percentile all’interno di ciascuna matrice.
È il criterio di {cite}`han2015learning`, e *Optimal Brain Damage*
{cite}`lecun1990optimal` non ne è la giustificazione: è il lavoro scritto per
scavalcarlo, che si propone di «andare oltre l'approssimazione che grandezza
uguale importanza» e misura che ordinare per grandezza costa più che ordinare
per l'importanza stimata. Ricostruire il criterio dentro quel quadro serve
proprio a vedere quante approssimazioni nasconde. Spostando i pesi di
$\delta\boldsymbol{\theta}$,

$$
\delta \mathcal{L} = \mathbf{g}^{\top}\delta\boldsymbol{\theta}
+ \tfrac{1}{2}\,\delta\boldsymbol{\theta}^{\top}\mathbf{H}\,\delta\boldsymbol{\theta}
+ O(\|\delta\boldsymbol{\theta}\|^3),
$$

con $\mathbf{g}$ il gradiente e $\mathbf{H}$ l’Hessiana. Di qui in poi OBD
butta via tre pezzi, e li nomina: si ferma al secondo ordine
(**quadratica**), pota ad addestramento finito, dove il gradiente è nullo e il
primo termine sparisce (**estremale**), e trascura i termini fuori diagonale,
così il costo si spezza in un addendo per peso (**diagonale**),
$\tfrac{1}{2}h_{ii}\,\delta
\theta_i^2$; azzerare il peso $i$-esimo vuol dire $\delta\theta_i = -w_i$, cioè
un costo $\tfrac{1}{2}h_{ii}w_i^2$. L’ordinamento che ne esce non è ancora
quello per $|w_i|$: lo diventa con una quarta ipotesi, che la diagonale sia
**uniforme**, e quella OBD non la fa.

Sono quattro approssimazioni, non una, e si sanno false tutte e quattro: il
costo non è quadratico, una rete fermata da Adam non sta in un minimo,
l’Hessiana non è diagonale, e la sua diagonale non è uniforme. Gli autori di
OBD misurano dove si rompono le loro tre: l’accordo con la previsione regge
fino a circa il **trenta per cento** dei pesi tolti, e potare al novanta è tre
volte oltre. La cosa notevole è che il criterio regga lo stesso.

Il taglio da solo non basta perché la rete rimasta è fuori dal minimo in cui
era stata portata: i pesi superstiti sono ottimi rispetto a una funzione che
comprendeva anche quelli azzerati. Il **riaddestramento con maschera fissa**
(la maschera dei pesi sopravvissuti si applica dopo ogni passo
dell’ottimizzatore, così i pesi tagliati non tornano mai) riporta i superstiti
in un minimo della funzione ristretta.

Nella pratica il ciclo si itera: si pota una frazione, si riaddestra, si pota
ancora {cite}`han2015learning`. La potatura **iterativa** raggiunge, a parità
di sparsità finale, accuratezze nettamente migliori di quella in un colpo
solo, e la ragione è quella che rende fragile lo sviluppo: ogni passo chiede
alla rete un adattamento piccolo invece che uno enorme.

`````

Il conto si fa su una rete piccola e su dati piccoli, così sta in una pagina e
gira in mezzo minuto. Il compito è riconoscere cifre scritte a mano: si dà alla
rete un quadratino di otto pixel per otto e lei deve dire quale delle dieci
cifre sia.

```python
import torch
from torch import nn
from sklearn.datasets import load_digits
from sklearn.model_selection import train_test_split

torch.manual_seed(0)
# un thread solo: due esecuzioni di fila danno lo stesso numero. Su un'altra
# macchina le ultime cifre ballano, perche' cambia l'ordine delle somme
torch.set_num_threads(1)

dati = load_digits()
Xtr, Xte, ytr, yte = train_test_split(dati.data / 16.0, dati.target,
                                      test_size=0.3, random_state=0)
Xtr = torch.tensor(Xtr, dtype=torch.float32)
Xte = torch.tensor(Xte, dtype=torch.float32)
ytr, yte = torch.tensor(ytr), torch.tensor(yte)


def costruisci():
    torch.manual_seed(0)
    return nn.Sequential(nn.Linear(64, 256), nn.ReLU(),
                         nn.Linear(256, 256), nn.ReLU(), nn.Linear(256, 10))


def addestra(rete, passi, maschere=None):
    """Se ci sono le maschere, i pesi tagliati vengono rimessi a zero
    dopo ogni passo: l'ottimizzatore non puo' farli risorgere."""
    opt = torch.optim.Adam(rete.parameters(), lr=1e-3)
    matrici = [p for p in rete.parameters() if p.dim() == 2]
    for _ in range(passi):
        nn.functional.cross_entropy(rete(Xtr), ytr).backward()
        opt.step()
        opt.zero_grad()
        if maschere:
            with torch.no_grad():
                for p, m in zip(matrici, maschere):
                    p *= m
    return rete


def accuratezza(rete):
    with torch.no_grad():
        return (rete(Xte).argmax(1) == yte).float().mean().item() * 100


rete = addestra(costruisci(), 600)
pieni = [p.detach().clone() for p in rete.parameters()]
print(f"rete intera: {accuratezza(rete):.1f}%")
print()
print(f"{'tolti':>7} {'subito dopo':>13} {'dopo il riaddestramento':>25}")
for frazione in (.5, .8, .9, .95):
    with torch.no_grad():
        for p, originale in zip(rete.parameters(), pieni):
            p.copy_(originale)                 # si riparte sempre dalla rete intera
        maschere = []
        for p in [q for q in rete.parameters() if q.dim() == 2]:
            soglia = p.abs().flatten().kthvalue(int(frazione * p.numel())).values
            m = (p.abs() >= soglia).float()
            p *= m
            maschere.append(m)
    subito = accuratezza(rete)
    dopo = accuratezza(addestra(rete, 300, maschere))
    print(f"{frazione*100:>6.0f}% {subito:>12.1f}% {dopo:>24.1f}%")
```

```text
rete intera: 97.8%

  tolti   subito dopo   dopo il riaddestramento
    50%         95.9%                     98.0%
    80%         66.5%                     97.6%
    90%         39.3%                     96.9%
    95%         30.4%                     93.0%
```

La colonna di mezzo e quella di destra dicono due cose diverse, e la seconda è
quella che conta. Tolti nove pesi su dieci la rete non sa più leggere una
cifra, e dopo trecento passi di riaddestramento è tornata a 96,9 contro il 97,8
di partenza: ha perso meno di un punto **avendo dentro un decimo dei
collegamenti**. A metà strada, con la metà dei pesi, è perfino salita di due
decimi, e qui bisogna resistere alla tentazione di dedurne qualcosa. Il
riaddestramento dà alla rete potata trecento passi in più **e un ottimizzatore
nuovo**, che la rete intera non riceve. Fatto il controllo (stesso seme, stessi
trecento passi, stesso Adam nuovo, ma **senza** potare nulla) l’accuratezza è
98,0%: identica. Quei due decimi non li ha regalati la potatura, li ha regalati
il riavvio dell’ottimizzatore, e il conto per accorgersene sono quattro
righe.

Il conto qui sopra pota in un colpo solo, perché sta in venti righe. Chi pota
sul serio lo fa **a giri**: toglie una fetta, riaddestra, toglie un’altra
fetta, e così via. {numref}`fig-potatura` fa proprio questo, tredici giri di
fila più lo stato di partenza, ed è una figura che **si muove**: se la si
guarda online i pesi si spengono giro dopo giro e la curva si allunga da
sinistra a destra. La rete lì dentro ha un solo strato nascosto invece di due,
quindi i numeri non combaciano con quelli della tabella e non devono: la cosa
da guardare è la forma della curva, non il valore.

```{figure} ../figures/potatura-che-assottiglia.svg
:name: fig-potatura
:alt: "Due riquadri affiancati. A sinistra una griglia di sedici per sedici quadratini, un campione dei pesi del primo strato di una rete: all'inizio sono tutti pieni, e giro dopo giro se ne svuotano sempre di più, fino a restare vuota o quasi. A destra la curva dell'accuratezza contro la frazione di pesi tolti, tracciata un punto per giro: resta piatta poco sotto il cento per cento mentre si tolgono i primi nove pesi su dieci, e poi precipita negli ultimi giri. Sotto, a ogni giro, quanti pesi sono stati tolti e l'accuratezza corrispondente."
:width: 100%

Tredici giri di potatura iterativa su una rete piccola, più lo stato di
partenza. A sinistra i pesi che restano, a destra quello che costa toglierli.
La curva non scende piano: resta piatta finché si tolgono i primi nove pesi su
dieci, e poi cade. Il punto in cui cade non si sa prima, e per questo si pota
misurando a ogni giro invece che scegliendo una percentuale all’inizio.
```

## Perché il conto non si accorge degli zeri

E adesso il conto che rovina la festa. Una griglia con il novantacinque per
cento di zeri (una griglia **rada**, si dice, per distinguerla da una piena)
viene moltiplicata esattamente come una piena.

```python
torch.manual_seed(0)
# nomi tutti nuovi: nel notebook compagno le pagine si susseguono nello stesso
# spazio dei nomi, e riusare `W` costringerebbe a rieseguirle sempre in ordine
piena = torch.randn(1024, 1024)
soglia = piena.abs().flatten().kthvalue(int(0.95 * piena.numel())).values
rada = piena * (piena.abs() >= soglia)
ingressi = torch.randn(1024, 256)

zeri = (rada == 0).float().mean().item()
print(f"zeri nella matrice rada: {zeri * 100:.0f}%")
print(f"moltiplicazioni che servirebbero: una su {1 / (1 - zeri):.0f}")
print(f"moltiplicazioni che il calcolatore fa, in tutti e due i casi: "
      f"{piena.numel() * ingressi.shape[1] / 1e9:.2f} miliardi")
```

```text
zeri nella matrice rada: 95%
moltiplicazioni che servirebbero: una su 20
moltiplicazioni che il calcolatore fa, in tutti e due i casi: 0.27 miliardi
```

Venti volte meno lavoro utile, e zero lavoro risparmiato: il calcolatore fa le
stesse duecentosessantotto milioni di moltiplicazioni in tutti e due i casi, e
duecentocinquantacinque milioni di quelle (il novantacinque per cento) sono
moltiplicazioni per zero.

Cronometrandolo si vede lo stesso: la matrice rada non va venti volte più
veloce, va **uguale**. Un cronometro dipende dalla macchina e da quanto è
occupata; il conto qui sopra no, e dice la stessa cosa.

`````{tab} Elementare

La ragione è semplice e un po’ deludente: un calcolatore che moltiplica due
matrici non guarda i numeri, li macina. Fa la stessa identica sequenza di
moltiplicazioni comunque siano fatti, e moltiplicare per zero costa quanto
moltiplicare per qualunque altra cosa.

Per guadagnarci bisognerebbe **saltare** gli zeri, e per saltarli bisogna
sapere dove sono, cioè tenere in memoria un elenco delle loro posizioni. Quel
libretto costa a sua volta memoria da leggere, e i salti costano tempo perché
mandano all’aria l’ordine con cui i numeri arrivano dalla memoria.

A volte quel patto conviene e a volte no, e dipende da due cose: da quanto è
vuota la matrice, e da che macchina la moltiplica. Su un processore normale,
con novantacinque zeri su cento, l’elenco conviene e si guadagna davvero; con
la metà degli zeri no, perché tenere il conto delle posizioni costa più di
quanto fa risparmiare. Su una scheda grafica non conviene quasi mai, perché lì
il conto ordinato va così veloce che saltare gli zeri costa più di farli. Ed è
per questo che, in pratica, si sente dire che la potatura non fa guadagnare
tempo: è vero dove i modelli grandi girano davvero.

Quello che invece si guadagna sempre è **lo spazio**: una matrice con novanta
zeri su cento si salva su disco molto più piccola, e per chi deve distribuire
un modello questo conta. Ma spazio su disco e velocità di risposta sono due
cose diverse.

C’è un modo di riscuotere anche la seconda, ed è togliere i pesi **a blocchi**
invece che uno per uno: non il singolo collegamento più debole, ma un neurone
intero con tutti i collegamenti che vi arrivano, cioè una riga intera della
griglia. Una rete a cui si toglie un neurone intero è letteralmente una rete
più piccola: la griglia ha una riga in meno, e moltiplicare una griglia più
piccola costa meno, senza trucchi. Si paga in accuratezza, perché scegliendo a
blocchi si è costretti a buttare via anche i pesi utili che stavano nella riga
sbagliata.

Una via di mezzo esiste, e taglia a gruppi minuscoli invece che a righe intere.
Non la sceglie chi addestra: la decide chi disegna le macchine. Certe schede
grafiche sanno saltare gli zeri, a una condizione: che stiano al loro posto. Di
ogni quattro pesi in fila due devono essere zero e due no, sempre, in tutta la
griglia. Il conto resta ordinato
abbastanza da correre veloce, e chi taglia resta libero abbastanza da non dover
buttare via righe intere. Il vincolo però non si tratta: dove di pesi utili ce
ne sono tre di fila, uno dei tre va a zero lo stesso.

`````

`````{tab} Superiore

Un prodotto matriciale denso è eseguito da un kernel GEMM che opera su
piastrelle regolari, con accessi alla memoria contigui e prevedibili: è il
regime per cui l’hardware è costruito, e la {doc}`sezione su GEMM e tensor core
</GPU/gemm-e-tensor-core>` spiega perché uscirne costi caro. La sparsità **non
strutturata** distrugge esattamente le due proprietà che rendono quel kernel
veloce, la regolarità dell’accesso e la possibilità di riempire le unità
vettoriali. Passare a un **formato rado** (CSR e simili, cioè la matrice
scritta come l'elenco delle sole posizioni non nulle, riga per riga) vuol dire
quindi **cambiare kernel**, non aggiustare quello di prima, e se convenga è una
domanda empirica, non di principio. Misurato sulla matrice rada del conto qui
sopra, su CPU e a tempo di processore: il CSR pareggia il denso intorno al
venti per cento di densità, e al cinque per cento (cioè con i novantacinque
zeri su cento di quell’esperimento) va dalle cinque alle sette volte più
veloce, a seconda della macchina. Su GPU la soglia si sposta molto più in
basso, perché il kernel denso lavora vicino al picco e gli accessi irregolari
costano di più: è la ragione per cui in pratica la sparsità non strutturata si
usa poco, e sta nell’hardware, non nell’aritmetica.

Da qui la distinzione operativa:

- la **sparsità non strutturata** riduce i parametri e non tocca il tempo **del
  kernel denso**, che è quello che quasi tutti eseguono. È utile per la
  dimensione del file, e come strumento di indagine (è quella che serve nel
  paragrafo sul biglietto della lotteria);
- la **sparsità strutturata** rimuove unità intere (righe e colonne di una
  matrice, canali di una convoluzione, teste di attenzione, strati). Il
  risultato è un tensore più piccolo e denso, quindi lo stesso kernel di prima
  su una forma minore: il guadagno è reale e proporzionale. Il costo è che il
  vincolo strutturale esclude molte configurazioni buone, e a parità di
  parametri rimossi l’accuratezza è peggiore;
- una via di mezzo esiste ed è imposta dall’hardware: alcuni acceleratori
  supportano schemi **a densità fissa locale** (per esempio due valori non
  nulli ogni quattro consecutivi), che sono abbastanza regolari da essere
  eseguiti in fretta e abbastanza liberi da non essere una potatura a blocchi.
  È il compromesso che decide chi progetta il silicio, non chi addestra.

`````

## Il biglietto della lotteria, e perché non ci salva

C’è una domanda che a questo punto viene naturale, e il libro l’ha già
incontrata parlando di quanto male il numero di parametri misuri la complessità
di un modello: se alla fine mi resta una rete con un decimo dei pesi che
funziona, perché ho dovuto addestrare quella grande? Perché non parto da quella
piccola?

La risposta sta nella {doc}`sezione su overfitting e validazione
</MachineLearning/overfitting-validazione>`: quella sottorete funziona **solo
se la si riaddestra con i numeri di partenza che aveva**, e reinizializzandola
a caso non impara altrettanto bene. Non era il collegamento a essere buono, era
il collegamento con quella partenza lì, e da qui il nome che l’idea porta: fra
i milioni di collegamenti di una rete grande, inizializzati a caso, qualcuno è
già disposto bene per il compito, e addestrare la rete grande è comprare tutti
i biglietti insieme per ritrovarsi in mano quello vincente.

Qui interessa una conseguenza sola, ed è quella che riguarda l’efficienza: **il
risparmio che si vorrebbe non è disponibile**. Per sapere quali sono i
biglietti vincenti bisogna prima fare l’estrazione, cioè addestrare la rete
grande, e per giunta più volte se si pota a giri. Tutto quello che questa
sezione ha misurato (novanta pesi su cento tolti, un punto di accuratezza perso)
si paga **dopo** un addestramento intero, non al posto suo. La potatura
comprime un modello che esiste già; non insegna a farne uno piccolo.

## Le due leve insieme

L’apertura del capitolo prometteva che le tre leve si compongono e che le
perdite **non si sommano in modo prevedibile**. Adesso ci sono i pezzi per
provarlo: si prende la rete, la si pota al novanta per cento riaddestrandola, e
poi si arrotondano a quattro bit i pesi rimasti con la funzione della sezione
precedente.

```python
def stato_pieno():
    """Rimette la rete com'era dopo il primo addestramento."""
    with torch.no_grad():
        for p, originale in zip(rete.parameters(), pieni):
            p.copy_(originale)


def arrotonda(bit=4, gruppo=64):
    with torch.no_grad():
        for p in rete.parameters():
            if p.dim() == 2:
                p.copy_(quantizza(p, bit, gruppo))     # dalla sezione di prima


stato_pieno()
print(f"rete intera:                {accuratezza(rete):.1f}%")
stato_pieno()
arrotonda()
print(f"solo quattro bit:           {accuratezza(rete):.1f}%")
stato_pieno()
maschere = []
for p in [q for q in rete.parameters() if q.dim() == 2]:
    soglia = p.abs().flatten().kthvalue(int(0.9 * p.numel())).values
    m = (p.abs() >= soglia).float()
    p.data *= m
    maschere.append(m)
addestra(rete, 300, maschere)
print(f"solo potata al 90%:         {accuratezza(rete):.1f}%")
arrotonda()
print(f"potata e poi a quattro bit: {accuratezza(rete):.1f}%")
```

```text
rete intera:                97.8%
solo quattro bit:           98.0%
solo potata al 90%:         96.9%
potata e poi a quattro bit: 96.7%
```

Arrotondare da solo non costa niente (anzi, due decimi in più, che è rumore).
Potare da solo costa 0,9 punti. Fare tutte e due costa **1,1**, cioè più della
somma dei due costi presi separatamente. Su un campione di prova di
cinquecentoquaranta cifre quel decimo di scarto in più sono due cifre, e
ripartendo da un'altra inizializzazione cambia anche di verso: su cinque, in
due casi comporre costa **meno** della somma. Che sia più o meno non si sa
prima, ed è esattamente la cosa che l’apertura prometteva: il budget di errore
non si spartisce a tavolino. Resta comunque un ottimo affare, un decimo dei
pesi e un ottavo dei bit per poco più di un punto di accuratezza.

Componendo le due leve salta fuori un guasto che nessuna delle due mostrava da
sola, e che non dà nessun errore. Dopo la potatura **duecentouno gruppi di
sessantaquattro pesi sono interamente zeri**: la scala di quei gruppi vale
zero, dividere per zero riempie la rete di valori non numerici, e da lì
`argmax` sceglie sempre la stessa cifra. L'accuratezza si ferma all'8,3%, che
non è il caso (il caso sarebbe il dieci per cento) ma la frequenza dello zero
fra gli esempi di prova: la rete risponde «zero» a tutto. Senza un avviso e
senza un errore. È il motivo della riga di protezione nella funzione
`quantizza` della sezione precedente, ed è il genere di guasto che si trova
solo componendo le cose e provandole.

Di tutte e tre le leve, la potatura è quella che si studia di più e si usa di
meno. Adesso si vede perché: la promessa è enorme, il costo in accuratezza è
basso, e in mezzo c’è un calcolatore che quella promessa non la sa incassare.

`````{tab} Elementare

```{admonition} Da ricordare
:class: important
- **Potare** vuol dire mettere a zero i pesi più piccoli. Da solo distrugge la
  rete; quello che la salva è **riaddestrare tenendo i tagli**. Misurato:
  togliendo nove pesi su dieci si passa da 97,8% a 39,3%, e dopo trecento passi
  di riaddestramento si è a 96,9%.
- La promessa però si riscuote male: con il novantacinque per cento di zeri le
  moltiplicazioni **utili** sono una su venti, e il calcolatore **le fa tutte e
  venti** lo stesso: 268 milioni in tutti e due i casi, di cui 255 milioni per
  zero. Non guarda i numeri, li macina.
- Quello che si guadagna sempre è **lo spazio su disco**. Per guadagnare anche
  tempo bisogna togliere i pesi **a blocchi** (un neurone intero, cioè una riga
  intera della griglia): allora la rete è davvero più piccola, ma si buttano
  via anche pesi utili che stavano nella riga sbagliata.
- Il **biglietto della lotteria**, che la {doc}`sezione su overfitting e
  validazione </MachineLearning/overfitting-validazione>` ha già raccontato,
  dice qui una cosa sola: per sapere quali collegamenti tenere
  bisogna prima addestrare la rete grande. La potatura comprime un modello che
  esiste già, non insegna a farne uno piccolo.
```

`````

`````{tab} Superiore

```{admonition} Da ricordare
:class: important
- La **potatura per grandezza** ordina i pesi per $|w_i|$ e azzera sotto una
  soglia percentile. Poggia su **quattro** approssimazioni, tutte e quattro
  note come false: sviluppo fermo al secondo ordine, rete supposta a un minimo,
  Hessiana supposta diagonale (le tre di *Optimal Brain Damage*, che le nomina
  per scavalcare proprio l’ordinamento per grandezza) e diagonale supposta
  uniforme, che è la quarta e serve solo a riportarsi a $|w_i|$.
- Il **riaddestramento con maschera fissa** è la parte non opzionale: la rete
  potata è fuori dal minimo in cui stava, e i superstiti vanno riportati in un
  minimo della funzione ristretta. Misurato a sparsità 0,9: 39,3% subito,
  96,9% dopo.
- **Sparsità non strutturata**: riduce i parametri, non il tempo, perché un
  kernel GEMM denso esegue lo stesso numero di prodotti indipendentemente da
  quanti operandi siano nulli: a sparsità 0,95 il lavoro utile è un ventesimo e
  quello eseguito è identico. Passare a un formato rado è cambiare kernel, e
  conviene o no a seconda della densità e dell’hardware (misurato su CPU: il
  pareggio è intorno al venti per cento di densità). **Strutturata**: rimuove
  unità intere e dà un guadagno reale su qualunque macchina, a un costo
  maggiore in accuratezza. Gli schemi a densità fissa locale sono il
  compromesso imposto dall’hardware.
- L’**ipotesi del biglietto della lotteria** {cite}`frankle2019lottery` sta nel
  capitolo sul machine learning, con i suoi due limiti. Quello che conta qui è
  il primo: la maschera si ottiene addestrando la rete densa, quindi il costo è
  pagato prima e non al posto.
```

`````

Le prime due leve hanno in comune una cosa che finora è passata sotto silenzio:
lavorano tutte e due su un modello **già addestrato**, e non gli chiedono di
imparare niente di nuovo. La terza rovescia il tavolo. Non stringe il modello
grande: ne costruisce un altro, piccolo, e glielo mette accanto come maestro. E
la cosa da capire è che cosa passi fra i due, perché non è la risposta giusta:
quella ce l’avevano già i dati.
