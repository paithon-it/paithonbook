# Il latente che si usa

Con un archivio ben fatto si può fare una cosa che ha l’aria di un gioco di
prestigio: si prende la scheda di un quadro, si cambia **un numero solo**, si
richiama il copista, e il quadro che ridipinge è quello di prima con una cosa
sola diversa. Più luce. La stessa faccia girata di lato. Lo stesso volto con gli
occhiali.

Quando funziona è una meraviglia, perché vuol dire che l’archivista, senza che
nessuno glielo abbia chiesto, ha scoperto da solo di che cosa sono fatti i
quadri. E qui c’è la domanda, ed è in due tempi:
si può *chiedergli* di farlo? E se si chiede, che cosa si paga?

## Una manopola sul costo della scheda

La sezione precedente ha lasciato in mano una perdita fatta di due voci: quanto
male si ricostruisce e quanto costa scrivere la scheda. Chi ha un conto con due
voci prima o poi prova a cambiare il peso di una delle due, ed è esattamente
quello che fecero Irina Higgins e colleghi nel 2017 {cite}`higgins2017beta`:
moltiplicare la seconda voce per un numero, chiamarlo $\beta$, e girare la
manopola. La macchina che ne esce si chiama **$\beta$-VAE**, e il nome dice
già tutto: un VAE con una manopola in più.

`````{tab} Elementare

Girare la manopola vuol dire chiedere all’archivista di essere ancora più
sintetico. Con la manopola a uno siamo al patto della sezione precedente; a
due gli si dice che ogni riga scritta costa il doppio; a quattro, il quadruplo.
Un tetto e un prezzo, per lui, sono lo stesso ordine: invece di scrivergli sul
contratto «non più di tre righe», si alza il prezzo della riga finché di righe
ne scrive tre, e la manopola è quel prezzo.

L’idea è che un archivista sotto pressione debba mettersi in ordine. Se le
righe costano care, gli conviene spenderle bene: usare una riga sola per la
luce, una sola per l’inclinazione, invece di spargere ogni cosa un po’
dappertutto.

Il modo in cui obbedisce ha una parte che sorprende: non accorcia soltanto le
righe, ne **spegne** qualcuna del tutto, lasciando cadere per intero quelle che
gli rendono meno e concentrando su quelle che restano. Fin qui è proprio il
mestiere che gli abbiamo chiesto.

Ma quel mestiere ha un punto in cui si rovescia, ed è lo stesso di cui la
sezione precedente aveva già avvertito. A un archivista a cui la scrittura
costa troppo non conviene più essere sintetico: conviene **smettere di
scrivere**. Prima cadono le righe che servivano meno, poi quelle che servivano
un po’, e alla fine consegna schede vuote. A quel punto il copista dipinge sempre lo
stesso quadro, che è la media di tutti quelli che ha visto, e cambiare i numeri
della scheda non cambia più niente perché non c’è più niente da cambiare.

In certi casi, poi, la manopola c’era già, senza che nessuno l’avesse chiamata
così, il che le toglie l’aria della trovata. Quando il metro con cui si giudica
la copia porta dentro di sé quanto si è disposti a sbagliare un pixel,
scegliere quel metro è già scegliere quanto pesi l’altra voce: chi lo sceglie
gira la manopola senza saperlo. Vale finché quella tolleranza la fissiamo noi;
se a deciderla è il copista, la manopola gratis non c’è più. E il metro usato
qui giudica ogni pixel come una scommessa fra bianco e nero, e una misura così
quella manopola dentro non ce l’ha: bisogna metterla a mano. È quello che
facciamo adesso, girandola su quattro tacche.

`````

`````{tab} Superiore

Il $\beta$-VAE {cite}`higgins2017beta` sostituisce l’ELBO con

$$
\mathcal{E}^{\beta}_{\theta,\phi}(\mathbf{x}) =
\mathbb{E}_{q_\phi(\mathbf{z} \mid \mathbf{x})}
\big[\log p_\theta(\mathbf{x} \mid \mathbf{z})\big]
\;-\; \beta \, D_{\mathrm{KL}}\!\big(q_\phi(\mathbf{z} \mid \mathbf{x})
\,\|\, p(\mathbf{z})\big),
$$

dove $\beta > 0$ pesa il costo di descrizione. Con $\beta = 1$ si torna
all’ELBO. Gli autori lo ricavano come lagrangiana di un problema vincolato,
«massimizza la ricostruzione con $D_{\mathrm{KL}} \le \varepsilon$», dove
$\varepsilon$ è il tetto che ci si dà e $\beta$ è il moltiplicatore: sotto
quella luce la manopola diventa il prezzo ombra di un vincolo di capacità sul
canale latente.

Due osservazioni che tolgono al parametro l’aria di magia. La prima: $\beta$ era
**già lì**, nascosto nella scelta della verosimiglianza. Con un decoder
gaussiano il cui rumore ha varianza $\sigma^2$ **fissata** (è il $\sigma^2$
dell’apertura del capitolo, non la larghezza della zona proposta
dall’encoder), il termine di ricostruzione porta
davanti a sé un fattore $1/(2\sigma^2)$; moltiplicando l’obiettivo per
$2\sigma^2$, che è positivo e quindi non sposta l’ottimo, si ottiene
esattamente l’obiettivo del $\beta$-VAE con $\beta = 2\sigma^2$. Due riserve,
però, e la seconda morde qui: se $\sigma^2$ viene **appreso** invece che
fissato, il termine additivo $-\tfrac{D}{2}\log(2\pi\sigma^2)$ non è più una
costante e la manopola libera sparisce; e il decoder di questo capitolo non è
gaussiano ma di Bernoulli, cioè un $\sigma^2$ da girare non ce l’ha affatto.
Nell’esperimento delle quattro tacche, quindi, $\beta$ è un parametro vero e
non è assorbito da niente.

La seconda: l’effetto atteso è che le componenti latenti si specializzino, e il
meccanismo per cui dovrebbe succedere è la pressione a **spegnerne** alcune.
Ogni componente con $D_{\mathrm{KL}}$ vicino a zero è una componente che
l’encoder ha rinunciato a usare, e in cui $q_\phi(z_j \mid \mathbf{x}) \approx
p(z_j)$: è il collasso della posterior della sezione precedente, che qui compare
non come guasto ma come strumento di selezione. Fra strumento e guasto passa la
posizione della manopola, e l’esperimento sulle quattro tacche misura dove.

`````

```python
import torch
from torch import nn
from torch.nn import functional as F
from sklearn.datasets import load_digits

torch.set_num_threads(1)      # numeri riproducibili su qualunque macchina
X = torch.tensor(load_digits().data / 16.0, dtype=torch.float32)
LATENTE = 8


class VAE(nn.Module):
    def __init__(self):
        super().__init__()
        self.tronco = nn.Sequential(nn.Linear(64, 48), nn.ReLU())
        self.testa = nn.Linear(48, 2 * LATENTE)
        self.decoder = nn.Sequential(nn.Linear(LATENTE, 48), nn.ReLU(),
                                     nn.Linear(48, 64))

    def codifica(self, x):
        return self.testa(self.tronco(x)).chunk(2, dim=1)


def addestra(beta):
    """Lo stesso VAE della sezione scorsa, col costo di descrizione pesato."""
    torch.manual_seed(0)
    vae = VAE()
    opt = torch.optim.Adam(vae.parameters(), lr=3e-3)
    for passo in range(4000):
        media, log_var = vae.codifica(X)
        z = media + torch.exp(0.5 * log_var) * torch.randn_like(media)
        ricostruzione = F.binary_cross_entropy_with_logits(
            vae.decoder(z), X, reduction="sum") / len(X)
        # il costo tenuto separato riga per riga, per poterlo poi leggere
        costo = (-0.5 * (1 + log_var - media ** 2 - log_var.exp())).mean(0)
        perdita = ricostruzione + beta * costo.sum()
        opt.zero_grad()
        perdita.backward()
        opt.step()
    return vae, ricostruzione.item(), costo.detach()


print(f"{'beta':>5} {'ricostruzione':>14} {'costo':>7} {'righe usate':>12}   nat per riga")
reti = {}
for beta in (0.5, 1, 2, 4):
    reti[beta], ricostruzione, costo = addestra(beta)
    print(f"{beta:>5} {ricostruzione:>14.1f} {costo.sum():>7.2f} "
          f"{(costo > 0.05).sum().item():>9}/{LATENTE}   "
          + " ".join(f"{v:.2f}" for v in costo.sort(descending=True).values))
```

```text
 beta  ricostruzione   costo  righe usate   nat per riga
  0.5           18.6    6.03         6/8   1.41 1.40 1.19 0.94 0.70 0.39 0.00 0.00
    1           20.2    3.59         4/8   1.05 1.03 0.89 0.63 0.00 0.00 0.00 0.00
    2           23.6    1.41         3/8   0.51 0.47 0.43 0.00 0.00 0.00 0.00 0.00
    4           27.1    0.00         0/8   0.00 0.00 0.00 0.00 0.00 0.00 0.00 0.00
```

La tabella dice tre cose.

**Il baratto va sempre nella stessa direzione**, almeno sulle quattro tacche
provate. Più la manopola sale, meno si spende in scheda e peggio si
ricostruisce: da 18,6 nat a 27,1. Su nessuna delle quattro si guadagna da tutte
e due le parti, il che suggerisce una cosa importante: non esiste un valore
«giusto» da trovare, c’è solo una scala su cui scegliere dove stare, e la
scelta dipende da che cosa serve.

**Le righe si spengono, e non a una per volta.** Già al valore standard,
quello di tutta la sezione precedente, quattro delle otto righe della scheda
portano zero nat: la rete ha scelto da sé di usarne quattro. Alzando la
manopola ne resta una in meno, e alla tacca dopo cadono le ultime tre insieme.
Chi si aspettava che il latente usasse tutto lo spazio disponibile ha
un’informazione in più: **la dimensione del latente è
quella che la rete decide di pagare**, non quella che si dichiara.

**A quattro, l’archivista ha smesso di scrivere.** Costo zero su tutte le
righe: è il collasso della posterior, arrivato non per sfortuna ma perché lo
abbiamo comprato girando una manopola. E c’è una conferma indipendente, che
viene da due sezioni fa: la ricostruzione a $\beta = 4$ vale **27,1 nat**, che
è lo stesso costo di chi non guarda la cifra e dichiara per ogni pixel il
grigio medio di tutte. Non è una coincidenza. A scheda vuota il copista non
può fare altro che dipingere sempre la stessa cosa, e la cosa che gli conviene
dipingere è proprio quella media: i due numeri **devono** coincidere. La prova
del collasso, però, non è il 27,1, è la colonna del costo, che a quella tacca
vale zero su tutte e otto le righe; il 27,1 è la conferma che arriva da fuori.
E dice una cosa da portarsi via: il collasso non è un modello
brutto, è **nessun modello**.

Il blocco che segue lo fa vedere nel modo più diretto, cioè provando a usare la
scheda.

```python
LIVELLI = " .:-=+*#%"


def affianca(*immagini):
    griglie = [(im.reshape(8, 8) * 8).round().long().clamp(0, 8) for im in immagini]
    return "\n".join("   ".join("".join(LIVELLI[i] for i in g[r]) for g in griglie)
                     for r in range(8))


for beta in (1, 4):
    with torch.no_grad():
        vae = reti[beta]
        media, log_var = vae.codifica(X)
        costo = (-0.5 * (1 + log_var - media ** 2 - log_var.exp())).mean(0)
        riga = int(costo.argmax())
        # a beta = 4 il costo e' zero su tutte, quindi argmax sceglie fra pareggi
        quale = "la piu' carica" if costo[riga] > 0.05 else "una qualunque"
        varianti = media[:1].repeat(5, 1)          # la scheda della prima cifra
        varianti[:, riga] = torch.linspace(-2.5, 2.5, 5)
        print(f"\nbeta = {beta}: la riga {riga}, {quale}, "
              f"portata da -2,5 a +2,5")
        print(affianca(*torch.sigmoid(vae.decoder(varianti))))
```

```text
beta = 1: la riga 5, la piu' carica, portata da -2,5 a +2,5
  -#+.       -**-       -**+:      =##*-      +##+.
  ##**      .#*+*.     .#*=*-     .++=#=     .+=**.
 :%:.#:     :#:.+:     :#::+:      -.-#.      ..+*.
 -*  ==     :*. =-     .*+#+.      .=*+.       -#*:
 -*  -=     -+. +-      :-++.      .*#=.      .+#=.
 -%  +-     :* .*:      ..=+.      .+*-       .+-.
 .#+*#.      *=+*       -=*=       :+=.       :*-
  :##:       -#*:       -#+.       =#:        *+.

beta = 4: la riga 2, una qualunque, portata da -2,5 a +2,5
  -**-.      -**-.      -**-.      -**-.      -**-.
 .+*+=.     .+*+=.     .+*+=.     .+*+=.     .+*+=.
 .+-==.     .+-==.     .+-==.     .+===.     .+===.
 .+=+=.     .+=+=.     .+=+=.     .+=+=.     .+=+=.
 .=++=.     .=++=.     .=++=.     .=++=.     .=++=.
 .-===:     .-===:     .-===:     .-===:     .====:
  =++=:      =++=:      =++=:      =++=:      =++=:
  -**-.      -**-.      -**-.      -**-.      -**-.
```

Con la manopola a quattro le cinque immagini sono la stessa immagine: fra la
prima e l’ultima si contano due caratteri di differenza, uno nella terza
riga e uno nella sesta, e a occhio non si vedono. La scheda non governa più
niente.

Con la manopola a uno, invece, succede qualcosa, ed è il punto della
sezione. A sinistra c’è uno zero, con il buco aperto
in mezzo; spostandosi verso destra il buco si chiude, la figura si stringe e
si sposta di lato, e l’ultima immagine non è più uno zero né si riesce a dire
che cifra sia. Sono cambiate insieme la forma del tratto, la posizione e
l’identità della cifra, e non **una** cosa sola. Quella riga della scheda è una
direzione lungo la quale parecchie cose si muovono insieme, e non «lo
spessore» né «l’inclinazione».

Ed è la regola, non l’eccezione. Chiamiamo **fattori** gli ingredienti di cui un
dato è fatto e che si vorrebbero tenere separati: per un volto, la luce, quanto
la testa è girata, l’espressione. Nel 2019 Francesco Locatello e colleghi hanno
addestrato più di dodicimila modelli di questa famiglia, con tutte le varianti
proposte fino ad allora, per rispondere a una domanda sola: la manopola separa
davvero i fattori? La risposta ha due parti, ed è una delle poche dimostrazioni
di impossibilità che il libro incontra {cite}`locatello2019challenging`.

La prima parte è teorica: **senza ipotesi in più sul modello e sui dati,
separare i fattori senza supervisione è impossibile**, e non per difficoltà
pratica. La ragione si vede con un esempio. Metti che l’archivista ci sia
riuscito: la prima riga della scheda dice quanto la testa è girata, la seconda
quanta luce c’è. Adesso prendi quelle due righe e **falle ruotare insieme**,
come si gira di sbieco una coppia di assi disegnata su un foglio: al posto di
«inclinazione» e «luce» restano due righe che ne portano un po’ per una.

Guarda che cosa non cambia. Il vocabolario comune non se ne accorge, perché non
ha un verso suo: girarlo lo lascia identico a prima. Al copista basta leggere
le righe girate all’indietro dello stesso angolo, e ridipinge esattamente i
quadri di sempre. E il conto del costo torna identico. Niente, in
quello che abbiamo chiesto alla macchina, dice che la coppia di partenza sia
più giusta di quella girata: sono due descrizioni ugualmente buone, e la
macchina non ha modo di preferire quella che a noi sembra sensata.

La seconda parte è sperimentale, ed è la più scomoda. Fra i dodicimila modelli,
a contare non era **quale** metodo si fosse scelto. Contavano il sorteggio con
cui la rete era stata inizializzata, cioè i suoi numeri interni prima di
imparare, e quanto forte fosse la manopola. E senza etichette non c’è modo di
scegliere né l’uno né l’altra: si può solo provare e sperare.

Il che non rende la manopola inutile. La rende quello che è: un modo di
comprare **spazio sulla scheda**, non un modo di comprare significato.

## Quando il latente è fatto di simboli

C’è un’altra cosa che si può chiedere alla scheda, ed è la più conseguente di
tutte per il resto del libro: che invece di numeri porti **simboli**, presi da
un elenco finito di cui decidiamo in anticipo soltanto quanto sia lungo. La
macchina che lo fa si chiama **VQ-VAE**, e le due lettere in più dicono proprio
questo, che si sceglie da un catalogo.

`````{tab} Elementare

Fin qui l’archivista scriveva numeri, cioè poteva mettere sulla scheda
qualunque sfumatura. Adesso gli si dà un prontuario da riempire:
milleventiquattro caselle, che è il numero che il libro usa davvero per il
suono. Le caselle gliele contiamo noi; a riempirle, con le descrizioni-tipo che
tornano più spesso nei quadri, pensa lui. E da lì in poi non descrive più
niente: guarda il quadro, cerca la casella che gli somiglia di più, e scrive
quel numero. La scheda smette di essere una fila di misure e diventa una fila
di **numeri di catalogo**. E la manopola non fa più presa: una casella costa
quanto le altre, quindi la scheda costa uguale comunque la si scriva.

Il guadagno è enorme, e il libro lo ha già incassato due volte. Una fila di
numeri di catalogo è, alla lettera, un testo: simboli in fila presi da un
alfabeto finito, come le parole di una frase sono prese da un vocabolario. E su
una cosa fatta così si può mettere al lavoro tutta la macchina che il libro ha
costruito per il linguaggio, quella che indovina il simbolo dopo. È il modo in
cui una macchina genera musica, e il modo in cui genera parlato.

C’è però un ostacolo, ed è esattamente quello che la sezione precedente aveva
annunciato. Il trucco per far tornare indietro le correzioni funzionava perché
lo scarto si poteva decidere prima e poi appoggiare sulla zona proposta. Fra la
descrizione numero tre e la numero quattro non c’è niente in mezzo, quindi non
c’è nessuno scarto da decidere, e il trucco non si applica. Ci vuole un’altra
idea, e il libro l’ha già raccontata parlando di come si comprime il suono: la
si trova nel {doc}`capitolo sull’audio </Audio/overview>`, nella sezione sui
codec neurali, e torna nel
{doc}`capitolo sulle GAN </GAN/overview>`, le reti che si sfidano, dove la
stessa idea serve per le immagini.

`````

`````{tab} Superiore

Il **VQ-VAE** {cite}`oord2017neural` sostituisce il latente continuo con uno
discreto: l’uscita dell’encoder viene sostituita dalla voce più vicina di un
dizionario appreso di $K$ vettori, e la scheda diventa una sequenza di indici in
$\{1, \dots, K\}$. Il libro lo spiega per esteso nel capitolo sull’audio, dove
serve a fabbricare un alfabeto per il suono, e lo riprende nel capitolo sulle
GAN, dove diventa la base di VQ-GAN; qui interessa solo la sua posizione in
questa famiglia.

La posizione è questa. Con un latente categorico la riparametrizzazione **non è
disponibile**: non esiste una scrittura $\mathbf{z} = g(\boldsymbol{\epsilon},
\phi, \mathbf{x})$ derivabile in $\phi$, perché la mappa da $\phi$ a un indice è
costante a tratti e ha derivata nulla quasi ovunque. Restano tre strade, e il
libro le incontra tutte e tre in punti diversi: lo **stimatore a punteggio**
della sezione precedente, che si applica ma paga in varianza; un
**rilassamento continuo** come la Gumbel-softmax, che il capitolo sull’audio usa
per wav2vec 2.0; e lo **straight-through estimator**, cioè copiare all’indietro
il gradiente saltando la quantizzazione, che è la scelta di VQ-VAE.

Dell’ELBO, poi, resta poco. Con prior uniforme sugli indici e
posterior deterministica il termine di divergenza vale $\log K$, cioè è una
costante: c’è, ma non ha gradiente e non partecipa all’ottimizzazione. Quello
che si minimizza davvero è la ricostruzione più due termini che nell’ELBO non
compaiono affatto, e che servono a tenere insieme dizionario ed encoder. Il
prior sugli indici viene semmai appreso dopo, con un modello autoregressivo
sulla sequenza di simboli, ed è quel modello, non il VQ-VAE, a generare.

`````

## Quattro macchine, adesso che si sa come sono fatte

Questo capitolo è nato per pagare un debito, e adesso si può fare il conto.
Quattro punti del libro montano un modello a variabile latente. Due li abbiamo
già attraversati, e là c’era una promessa al posto della derivazione; due
arrivano dopo, e adesso possono darla per fatta.

**Nei {doc}`codec neurali </Audio/codec-neurali>`**, per fabbricare un alfabeto
del suono. Là la scheda è fatta di simboli e non di numeri, cioè è il caso in
cui il trucco delle correzioni della sezione precedente non si applica.

**Nell’{doc}`offline reinforcement learning
</DeepReinforcementLearning/offline-rl>`**, dove si impara a decidere da
partite già giocate senza poterne giocare di nuove, per l’uso più insolito dei
quattro: là questa macchina non serve né a generare né a comprimere, serve a
**recintare**. Le si danno in pasto le mosse che nei dati compaiono davvero, e
lei dice quali mosse siano plausibili in una certa situazione; il programma poi
sceglie la migliore soltanto **fra quelle**, invece che fra tutte, così non va
a fantasticare su mosse che nessuno ha mai visto. È un modello che fabbrica,
usato come guardiano.

Gli altri due arrivano dopo, e da qui in avanti li si legge sapendo
che cosa c’è dentro.

**In {doc}`Stable Diffusion </ModelliDiffusione/stable-diffusion>`**, per far
stare un generatore di immagini in un computer di casa. Là all’archivista non
si chiede affatto di inventare: gli si chiede solo di rimpicciolire le immagini
di quarantotto volte, così che il generatore vero e proprio possa lavorare su
qualcosa di piccolo. L’archivista impara prima, da solo, e poi **smette di
imparare**; e la seconda voce di spesa, quella che tiene le schede raccolte, è
tenuta apposta piccolissima. È il caso in cui il difetto misurato in questo
capitolo, lo scarto fra il vocabolario comune e l’insieme vero delle schede,
non si risolve: si aggira, perché a decidere che cosa esce dalla scheda pensa
un altro modello.

**Nei {doc}`mondi in miniatura </WorldModels/mondi-in-miniatura>`**, in cui un
programma si allena immaginando invece che giocando, per spremere un fotogramma
di videogioco in trentadue numeri. Là il punto è proprio la proprietà che
questo capitolo ha misurato: se la mappa delle schede avesse buchi, la macchina
che immagina il fotogramma successivo produrrebbe presto una scheda a cui non
corrisponde nessuna immagine, e il sogno si spezzerebbe dopo pochi passi.

`````{tab} Elementare

```{admonition} Da ricordare
:class: important
- Si può mettere una **manopola** sul costo della scheda e chiedere
  all’archivista di essere ancora più sintetico. Sulle quattro tacche provate il
  baratto va sempre nella stessa direzione: la scheda costa meno e la copia
  peggiora, e su nessuna si guadagna da tutte e due le parti.
- Girando la manopola le righe della scheda **si spengono**, a una a una o a
  gruppi: già al valore normale, quattro righe su otto portano zero. La
  dimensione del latente la decide la rete, pagandola, e non la
  dichiarazione.
- Girata troppo, l’archivista smette di scrivere: la scheda non governa più
  niente e il copista dipinge sempre lo stesso quadro.
- La manopola compra **spazio sulla scheda, non significato**: muovendo una
  riga cambiano più cose insieme. Che senza aiuti dall’esterno separare gli
  ingredienti di un dato **non si possa**, e non per difficoltà pratica, è una
  dimostrazione. E che in pratica conti più il sorteggio iniziale del metodo
  scelto lo hanno mostrato, nel 2019, dodicimila modelli.
- La scheda può essere fatta di **simboli** invece che di numeri, e allora
  diventa un testo su cui si può mettere al lavoro la macchina del linguaggio.
  Costa un’altra idea, perché con i simboli il trucco delle correzioni non
  funziona più.
- Questa macchina il libro la monta in **quattro** posti, due già letti e due
  che verranno: fabbrica l’alfabeto dei codec audio, fa da recinto attorno alle
  mosse ammissibili quando si impara da partite già giocate, comprime per
  Stable Diffusion, riassume i fotogrammi dei mondi in miniatura.
```

`````

`````{tab} Superiore

```{admonition} Da ricordare
:class: important
- **$\beta$-VAE** {cite}`higgins2017beta`: il termine
  $D_{\mathrm{KL}}(q_\phi \,\|\, p)$ pesato da $\beta$, ricavabile come
  lagrangiana di «massimizza la ricostruzione con $D_{\mathrm{KL}} \le
  \varepsilon$». Con decoder gaussiano di varianza $\sigma^2$ **fissata** non è
  nemmeno un parametro nuovo, perché $\beta = 2\sigma^2$; con $\sigma^2$
  appreso l’equivalenza cade, e con un decoder di Bernoulli come quello di
  questo capitolo un $\sigma^2$ da girare non c’è affatto: qui $\beta$ è un
  parametro vero.
- Su cifre 8x8 con $L = 8$, passando da $\beta = 0{,}5$ a $\beta = 4$,
  la ricostruzione va da 18,6 a 27,1 nat e le componenti con
  $D_{\mathrm{KL}} > 0{,}05$ passano da 6 a 0. La **dimensione effettiva** del
  latente la sceglie l’ottimizzatore, non chi scrive `LATENTE = 8`.
- La separazione dei fattori non è comprabile con la manopola:
  {cite}`locatello2019challenging` dimostra che senza ipotesi induttive sul
  modello **e** sui dati è **impossibile** in modo non supervisionato, e misura
  su oltre 12 000 modelli che a contare sono il seme e gli iperparametri più
  della scelta del metodo, e che nello studio non si è trovato nessun modo, in
  assenza di etichette, di fissare né gli uni né gli altri: per gli autori la
  selezione del modello senza supervisione resta un problema aperto.
- Latente **discreto** (VQ-VAE {cite}`oord2017neural`): la riparametrizzazione
  non si applica (mappa costante a tratti). Le tre alternative che il libro
  incontra sono lo stimatore a punteggio, la Gumbel-softmax e lo
  *straight-through*. Nell’obiettivo del VQ-VAE il termine di divergenza vale
  $\log K$ ed è quindi costante, cioè resta lì senza avere gradiente, e
  **accanto** compaiono due termini estranei all’ELBO che allineano dizionario
  ed encoder. Ma lo scostamento che conta è un altro: lo
  *straight-through* dà un gradiente **distorto**, quindi non si sta più
  ottimizzando un limite in senso stretto.
- Quattro usi nel libro, due già letti e due che verranno: tokenizzazione del
  suono (codec neurali) e vincolo di supporto sulle azioni (offline RL); poi
  compressione percettiva (Stable Diffusion) e riassunto dello stato (world
  model).
```

`````

Alla fine di questo capitolo abbiamo in mano una macchina che sa fare due cose
insieme: comprimere un dato in poche righe, e restituire un dato nuovo partendo
da righe che nessuno ha mai scritto. Le fa bene tutte e due, e nessuna delle due
benissimo: le immagini che produce sono morbide, e la ragione è nella pagella,
che la punisce molto se dimentica qualcosa di vero e poco se inventa qualcosa
che non esiste. In dubbio, quindi, copre.

Il capitolo che segue butta via la pagella. Niente probabilità, niente limite
inferiore, niente costo di descrizione: al posto di tutto questo, un giudice che
guarda il risultato e dice se ci crede. Si perde la capacità di dire quanto un
dato è probabile, e si guadagna il taglio netto che ai VAE manca.
