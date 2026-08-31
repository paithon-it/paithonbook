# Meno passi: risolvere l'equazione invece di simularla

Il modo che tutti conoscono per seguire numericamente una traiettoria lo scrive
Eulero nelle *Institutiones calculi integralis* del 1768: si guarda in che
direzione si sta andando, si fa un passetto in quella direzione, si guarda di
nuovo. Funziona sempre ed è il modo più lento di arrivare, perché butta via
tutto quello che dell'equazione si sa in anticipo e la tratta come una scatola
nera da interrogare passo per passo.

L'equazione della diffusione, invece, è per metà nota. Una metà descrive il
riscalamento della scala del segnale, e quella metà si risolve a penna,
esattamente, senza approssimare niente; l'altra metà contiene la rete, e solo
quella va integrata numericamente. Accorgersene ha portato il costo di
un'immagine da qualche centinaio di passaggi nella rete a una ventina, con la
stessa rete e gli stessi pesi: nessun riaddestramento, soltanto un modo più
accorto di percorrere la stessa strada.

## Metà dell'equazione si risolve a mano

`````{tab} Elementare

Una barca scende un fiume. Due cose la muovono: la corrente, che la trascina in
un modo perfettamente noto perché il fiume è quello e non cambia, e i colpi di
remo di chi ci sta sopra, che dipendono da dove si trova e da cosa vede.

Chi volesse prevedere dove finirà la barca ha due strade. La prima è misurare
ogni secondo lo spostamento complessivo e sommarlo: onesto, e sprecato, perché
un secondo su due si sta ricalcolando la corrente, che era già nota. La seconda
è tenere i due contributi separati: la corrente si integra una volta per tutte
con carta e penna, e i passi numerici si spendono soltanto per i colpi di remo,
che sono la parte davvero imprevedibile.

L'equazione della diffusione è fatta esattamente così. Un pezzo dice «restringi
lo stato di questo fattore», e il fattore è quello che abbiamo scelto noi
quando abbiamo deciso il programma di rumore: si conosce in forma esatta a
qualunque istante. L'altro pezzo dice «e adesso correggi secondo quello che la
rete indica», e quello è il solo pezzo ignoto.

Un metodo che tratta i due pezzi allo stesso modo, come fa quello di Eulero,
sta pagando passi per ricalcolare qualcosa di già noto. Un metodo che li separa
usa gli stessi passi tutti sul pezzo che conta. Il guadagno è grande perché il
pezzo noto, in queste equazioni, cambia molto in fretta: all'inizio del ritorno
il fattore di scala si muove di ordini di grandezza, e chi lo insegue a
passetti deve farne moltissimi.

`````

`````{tab} Superiore

La PF-ODE, scritta con la parametrizzazione sul rumore, ha struttura
**semilineare**:

$$
\frac{\mathrm{d}\mathbf{x}}{\mathrm{d}t}
= \underbrace{f(t)\,\mathbf{x}}_{\text{lineare, noto}}
+ \underbrace{\frac{g^2(t)}{2\sigma_t}\,
\boldsymbol{\epsilon}_\theta(\mathbf{x},t)}_{\text{non lineare, la rete}} ,
$$

dove si è usata $\nabla\log p_t = -\boldsymbol{\epsilon}_\theta/\sigma_t$. La
parte lineare ammette soluzione esatta per fattore integrante: posto
$\alpha_t = \exp\!\big(\int_0^t f\big)$, la variazione delle costanti dà, per
un passo da $s$ a $t$,

$$
\mathbf{x}_t = \frac{\alpha_t}{\alpha_s}\,\mathbf{x}_s
+ \alpha_t\int_s^t \frac{g^2(u)}{2\,\alpha_u\,\sigma_u}\,
\boldsymbol{\epsilon}_\theta(\mathbf{x}_u,u)\,\mathrm{d}u .
$$

Il primo addendo è **esatto**, per qualunque ampiezza del passo: nessuna
approssimazione, nessun errore accumulato. Tutto l'errore di discretizzazione
sta nell'integrale, cioè nella sola parte che dipende dalla rete.

Questa è la definizione di **integratore esponenziale**, una famiglia di metodi
sviluppata per i problemi *stiff*, quelli in cui la parte lineare ha costanti di
tempo molto più corte del resto. La PF-ODE è stiff proprio in questo senso: il
coefficiente lineare $f(t)=-\tfrac12\beta(t)$ passa da $-0{,}05$ a $-10$ lungo
il percorso, e a passi uniformi nel tempo un metodo generico deve tenerli
piccoli per restare stabile dove $|f|$ è grande. È la ragione per cui il metodo
di Eulero applicato direttamente a questa equazione si comporta peggio del suo
ordine nominale, come il conto misura.

`````

## Il tempo giusto non è il tempo

`````{tab} Elementare

C'è una seconda idea, indipendente dalla prima e altrettanto redditizia: la
scelta di come spaziare i passi.

Distribuirli uniformemente nel tempo sembra la cosa naturale e non lo è, perché
lungo il percorso non succede sempre la stessa quantità di cose. Nella prima
parte del ritorno l'immagine è indistinguibile dal rumore e i cambiamenti sono
grossolani; nell'ultimo tratto si decidono i dettagli, e lì i passi vanno
fitti. Passi uniformi nel tempo ne sprecano parecchi dove non serve.

La grandezza rispetto a cui conviene spaziarli è **quanto segnale c'è rispetto
al rumore**, misurata in scala logaritmica. Distribuire i passi uniformemente
lungo quella scala significa fare in modo che a ogni passo il rapporto fra
segnale e rumore cambi sempre della stessa proporzione, ed è quello che rende
i passi ugualmente informativi.

È un cambio di variabile, cioè un'operazione che non tocca l'equazione ma solo
il modo di percorrerla, e da sola vale una parte importante del risparmio. Chi
usa una libreria trova la spaziatura come impostazione a sé, staccata dal
metodo di calcolo: sono due manopole, e girare la prima cambia l'immagine anche
lasciando ferma la seconda.

`````

`````{tab} Superiore

Si introduce il **log-rapporto segnale-rumore**

$$
\lambda_t := \log\frac{\alpha_t}{\sigma_t},
$$

dove il rapporto è quello fra le ampiezze, come nella sezione sul limite
continuo: gli articoli che definiscono il rapporto sulle potenze scrivono
quindi $2\lambda_t$ dove qui c'è $\lambda_t$.

funzione strettamente decrescente di $t$, quindi invertibile:
$t = t_\lambda(\lambda)$. Cambiando variabile nell'integrale, e usando le
identità $\mathrm{d}\lambda = -\tfrac{g^2}{2\sigma_t^2}\mathrm{d}t$ valide per
i programmi affini, la soluzione esatta assume la forma

$$
\mathbf{x}_t = \frac{\alpha_t}{\alpha_s}\,\mathbf{x}_s
- \alpha_t\int_{\lambda_s}^{\lambda_t} e^{-\lambda}\,
\hat{\boldsymbol{\epsilon}}_\theta(\lambda)\,\mathrm{d}\lambda ,
$$

con $\hat{\boldsymbol{\epsilon}}_\theta(\lambda) :=
\boldsymbol{\epsilon}_\theta(\mathbf{x}_{t_\lambda},t_\lambda)$. È la
riscrittura su cui si fonda DPM-Solver {cite}`lu2022dpm`, e ha due
conseguenze.

La prima è che l'integrale è **pesato esponenzialmente**, quindi si approssima
bene con una espansione di Taylor di $\hat{\boldsymbol{\epsilon}}$ in
$\lambda$: i coefficienti che servono sono gli integrali
$\int e^{-\lambda}\lambda^k\,\mathrm{d}\lambda$, che hanno primitiva
elementare. Nessuna quadratura numerica.

La seconda riguarda la griglia. Spaziare uniformemente in $\lambda$ invece che
in $t$ concentra i passi dove $\hat{\boldsymbol{\epsilon}}$ varia di più, e su
programmi come quello lineare di DDPM la differenza fra le due griglie è
sostanziale a parità di valutazioni. È il motivo per cui i campionatori delle
librerie espongono la griglia come parametro separato dal metodo: sono due
scelte ortogonali, e si sbaglia a considerarle una sola.

`````

## DDIM è il primo gradino della scala, e si vede

`````{tab} Elementare

Supponendo che la risposta della rete resti la stessa per tutta la durata del
passo, si ottiene una formula che era già nota: è **DDIM**, il campionatore
veloce comparso pochi mesi dopo DDPM. All'epoca ci si era arrivati da
tutt'altra parte, ed è la strada che racconta la {doc}`sezione su come funziona
la diffusione </ModelliDiffusione/come-funziona>`; visto da qui è semplicemente
il primo gradino di una scala.

Riconoscerlo per quello che è ha un vantaggio pratico immediato: se DDIM è il
primo gradino, ci sono gradini successivi, e si sa esattamente come
costruirli.

`````

`````{tab} Superiore

Approssimando $\hat{\boldsymbol{\epsilon}}_\theta(\lambda)\approx
\hat{\boldsymbol{\epsilon}}_\theta(\lambda_s)$ costante sul passo, l'integrale
si risolve e resta

$$
\mathbf{x}_t = \frac{\alpha_t}{\alpha_s}\,\mathbf{x}_s
- \sigma_t\big(e^{h}-1\big)\,
\boldsymbol{\epsilon}_\theta(\mathbf{x}_s,s),
\qquad h := \lambda_t - \lambda_s ,
$$

che riscritta è esattamente l'aggiornamento **DDIM**
{cite}`song2021denoising`:

$$
\mathbf{x}_t = \alpha_t\underbrace{\frac{\mathbf{x}_s
- \sigma_s\boldsymbol{\epsilon}_\theta}{\alpha_s}}_{\hat{\mathbf{x}}_0}
+ \sigma_t\,\boldsymbol{\epsilon}_\theta .
$$

Le due scritture sono la stessa cosa, e la seconda è quella con cui DDIM viene
di solito presentato: si stima il dato pulito e lo si rimescola al livello di
rumore successivo. L'errore locale è $O(h^2)$ e quello globale $O(h)$: DDIM è
un metodo del **primo ordine**, ed è un integratore esponenziale, non un
Eulero. La differenza si misura, ed è grande.

`````

Le tre affermazioni (il pezzo noto risolto a penna conviene, spaziare i passi
sul rapporto segnale-rumore conviene, l'ordine alto conviene) si misurano sullo
stesso banco di prova che il capitolo usa da qualche sezione: due sole
modalità, punteggio esatto scritto a mano, nessuna rete addestrata. Isolare i
metodi dalla qualità del modello è precisamente ciò che serve per confrontarli.

```python
import numpy as np

MODI = np.array([-1.5, 1.5])
T_MIN = 1e-3
B_MIN, B_MAX = 0.1, 20.0
beta = lambda t: B_MIN + t * (B_MAX - B_MIN)
alpha = lambda t: np.exp(-0.5 * (B_MIN * t + 0.5 * (B_MAX - B_MIN) * t**2))
sigma = lambda t: np.sqrt(1 - alpha(t)**2)
lam = lambda t: np.log(alpha(t) / sigma(t))          # log-SNR

def eps(x, t):
    """La rete perfetta: il rumore atteso, in forma chiusa."""
    a, s = alpha(t), sigma(t)
    d = x[:, None] - a * MODI[None, :]
    w = np.exp(-0.5 * (d / s)**2)
    w /= w.sum(axis=1, keepdims=True)
    return (w * d).sum(axis=1) / s

# t in funzione di lambda, invertito una volta sola su una griglia fitta
_t = np.linspace(T_MIN, 1.0, 200_001)
_l = lam(_t)
t_di_lam = lambda L: float(np.interp(L, _l[::-1], _t[::-1]))

rng = np.random.default_rng(0)
z = rng.normal(size=2000)

def eulero(M):
    """Eulero in t: tratta l'equazione come una scatola nera."""
    ts = np.linspace(1.0, T_MIN, M + 1)
    x = z.copy()
    for k in range(M):
        t = ts[k]
        campo = -0.5 * beta(t) * x + 0.5 * beta(t) / sigma(t) * eps(x, t)
        x = x + (ts[k + 1] - t) * campo
    return x

def ddim(M):
    """Integratore esponenziale del primo ordine, a passi uguali in lambda."""
    Ls = np.linspace(lam(1.0), lam(T_MIN), M + 1)
    x = z.copy()
    for k in range(M):
        s, t = t_di_lam(Ls[k]), t_di_lam(Ls[k + 1])
        h = Ls[k + 1] - Ls[k]
        x = alpha(t) / alpha(s) * x - sigma(t) * (np.exp(h) - 1) * eps(x, s)
    return x

def dpm2(M):
    """DPM-Solver del secondo ordine: due valutazioni per passo."""
    Ls = np.linspace(lam(1.0), lam(T_MIN), M + 1)
    x = z.copy()
    for k in range(M):
        s, t = t_di_lam(Ls[k]), t_di_lam(Ls[k + 1])
        h = Ls[k + 1] - Ls[k]
        m = t_di_lam(Ls[k] + h / 2)
        u = alpha(m) / alpha(s) * x - sigma(m) * (np.exp(h / 2) - 1) * eps(x, s)
        x = alpha(t) / alpha(s) * x - sigma(t) * (np.exp(h) - 1) * eps(u, m)
    return x

riferimento = dpm2(1000)                     # la soluzione, praticamente esatta
scarto = lambda x: float(np.abs(x - riferimento).mean())

print("valutazioni   Eulero      DDIM        DPM-2")
misure = {}
for N in (8, 16, 32, 64, 128):
    misure[N] = (scarto(eulero(N)), scarto(ddim(N)), scarto(dpm2(N // 2)))
    print(f"{N:11d}   {misure[N][0]:.3e}   {misure[N][1]:.3e}   "
          f"{misure[N][2]:.3e}")
# -> valutazioni   Eulero      DDIM        DPM-2
# ->           8   1.164e-01   1.265e-02   5.940e-02
# ->          16   3.015e-02   4.737e-03   5.012e-03
# ->          32   1.059e-02   2.063e-03   6.528e-04
# ->          64   5.673e-03   9.624e-04   1.411e-04
# ->         128   3.593e-03   4.650e-04   3.655e-05

print("ordine misurato:", [round(float(np.log(misure[32][i] / misure[128][i])
                                       / np.log(4)), 2) for i in range(3)])
# -> ordine misurato: [0.78, 1.07, 2.08]
```

Gli ordini misurati (l'esponente per cui, raddoppiando i passi, l'errore si
divide per due elevato a quell'esponente) dicono che i metodi si comportano
esattamente come la teoria prevede: DDIM è del primo ordine ($1{,}07$),
DPM-Solver del secondo ($2{,}08$). Eulero invece **non raggiunge** il primo
ordine ($0{,}78$), e non per un difetto del metodo ma perché la parte che lui
approssima invece di risolvere è proprio quella che cambia in fretta.

Il confronto a parità di valutazioni della rete, che è la valuta con cui si
paga davvero, dice il resto. Per arrivare allo stesso scarto che DDIM ottiene
con trentadue valutazioni, a Eulero ne servono più di centoventotto; a
DPM-Solver del secondo ordine ne bastano ventidue, e con le stesse trentadue è
già tre volte più preciso.

## Quando l'ordine alto smette di aiutare

La tabella contiene anche il suo controesempio, e va guardata perché è la cosa
che in pratica sorprende.

`````{tab} Elementare

Con otto sole valutazioni il metodo del secondo ordine fa **peggio** del primo
ordine: cinque centesimi contro un centesimo. Non c'è niente di sbagliato nel
conto: i metodi di ordine alto sono più precisi quando i passi sono piccoli, e
più fragili quando sono grandi. Un metodo del secondo ordine guarda avanti
usando l'informazione presa a metà passo; se il passo è enorme, quella
informazione è presa in un posto molto diverso da dove si finirà, e la
correzione fa danni invece che bene.

Ci sono altre due cose che in pratica limitano l'ordine alto, e le conosce chi
usa questi strumenti tutti i giorni.

La prima riguarda il condizionamento sul testo. Quando si chiede al modello di
seguire con forza una descrizione, l'uscita della rete diventa molto più grande
in modulo, perché è la differenza amplificata fra due risposte; a quel punto un
metodo che estrapola quella grandezza esce di strada. La cura è cambiare che
cosa il metodo estrapola: invece del disturbo si estrapola la stima
dell'immagine pulita, che di grandezza normale resta molto più a lungo, e che
quando esagera si può riportare dentro i limiti dell'immagine. È esattamente la
modifica che porta da un campionatore alla sua versione «più», ed è pensata per
il caso guidato, che poi è quello di tutti.

La seconda è che sotto una certa soglia il collo di bottiglia smette di essere
il metodo. Se la rete stima il disturbo con un certo errore suo, nessun
integratore può fare meglio di quell'errore: si arriva a un pavimento, e da lì
in giù aggiungere ordine o passi non serve. Chi vuole scendere sotto quel
pavimento deve cambiare il modello, e i modi per farlo sono l'argomento della
{doc}`sezione sui generatori a un passo </ModelliDiffusione/pochi-passi>`.

`````

`````{tab} Superiore

Tre limiti, in ordine di quanto mordono.

**Stabilità a passo grande.** L'errore locale di un metodo di ordine $p$ è
$O(h^{p+1})$, con costante proporzionale alla derivata $p$-esima di
$\hat{\boldsymbol{\epsilon}}$ in $\lambda$. Per $h$ grande la costante domina
l'esponente, e la gerarchia degli ordini si rovescia. Sul banco di prova il
punto di pareggio fra primo e secondo ordine cade attorno alle sedici
valutazioni; nei modelli veri sta fra le dieci e le venti, ed è il motivo per
cui le librerie predefiniscono il secondo ordine e non il terzo.

**Guida forte.** Con classifier-free guidance la quantità integrata è
$\tilde{\boldsymbol{\epsilon}} = \boldsymbol{\epsilon}_\varnothing
+ w\,(\boldsymbol{\epsilon}_c-\boldsymbol{\epsilon}_\varnothing)$, dove
$\boldsymbol{\epsilon}_\varnothing$ e $\boldsymbol{\epsilon}_c$ sono la
predizione senza e con il testo e $w$ è la forza della guida (un numero, non il
peso $w(t)$ della loss). La norma di $\tilde{\boldsymbol{\epsilon}}$ cresce
linearmente in $w$ e la sua derivata in $\lambda$ cresce con essa.
DPM-Solver++ {cite}`lu2022dpmpp` risolve il problema riscrivendo l'integrale
nella parametrizzazione $\hat{\mathbf{x}}_0$, che a guida alta esce
dall'intervallo dei dati molto meno del rumore predetto (e quel che ne esce si
riporta dentro con una soglia dinamica), e adottando uno schema
**multipasso** (che riusa le valutazioni precedenti, come i metodi di Adams)
invece che a passo singolo: a parità di ordine dimezza le valutazioni. DEIS
arriva alla stessa famiglia per la via dell'estrapolazione polinomiale.

**Il pavimento dell'errore del modello.** Detto $\delta$ l'errore quadratico
medio della rete rispetto al punteggio vero, l'errore del campione generato
porta un termine che dipende da $\delta$ e che nessun integratore riduce: i
limiti noti sono superiori, cioè dicono che quel termine non sparisce
rimpicciolendo il passo, non che esista un pavimento sotto cui non si possa
scendere. Nei modelli veri quel pavimento si incontra fra le venti e le
cinquanta valutazioni, ed è la ragione strutturale per cui la corsa ai solutori
si è fermata lì: sotto quella soglia il guadagno non può più venire dal modo di
percorrere la traiettoria, ma solo da un modello che ne percorra una più corta.

`````

## In pratica: quale campionatore

```python
# la stessa qualita' con quattro budget di valutazioni della rete
for N in (8, 16, 32, 64):
    print(N, f"{misure[N][1]:.1e}", f"{misure[N][2]:.1e}",
          "primo ordine" if misure[N][1] < misure[N][2] else "secondo ordine")
# -> 8 1.3e-02 5.9e-02 primo ordine
# -> 16 4.7e-03 5.0e-03 primo ordine
# -> 32 2.1e-03 6.5e-04 secondo ordine
# -> 64 9.6e-04 1.4e-04 secondo ordine
```

La regola che se ne ricava vale anche sui modelli veri, con le soglie spostate:
fino alle sedici valutazioni conviene il primo ordine, e sopra conviene il
secondo. Nelle librerie i nomi sono `DDIM` o `Euler` per il primo,
`DPMSolverMultistep` (nella variante «più») o `Heun` per il secondo. La scelta
della griglia è un'impostazione a parte, e attenzione al nome: il parametro che
si chiama *timestep spacing* sceglie fra tre spaziature tutte uniformi nel
tempo, mentre la griglia che concentra i passi dove servono si accende con un
altro interruttore, quello che ridistribuisce i livelli di rumore.

`````{tab} Elementare
```{admonition} Da ricordare
:class: important
- L'equazione da percorrere è **per metà nota**: un pezzo riscala lo stato
  secondo il programma di rumore che abbiamo scelto noi e si risolve a penna,
  l'altro contiene la rete. I metodi buoni spendono i passi solo sul secondo,
  quelli generici li spendono su tutti e due.
- I passi vanno spaziati non nel tempo ma rispetto a **quanto segnale c'è
  rispetto al rumore**, in scala logaritmica: così ogni passo cambia le cose
  della stessa proporzione. È un parametro a parte rispetto al metodo, e da
  solo vale una fetta del risparmio.
- **DDIM è il primo gradino** di questa scala, ricavato per un'altra strada nel
  2020. Riconoscerlo come tale dice subito come costruire i gradini successivi.
- Misurato sullo stesso banco: per arrivare dove DDIM arriva con trentadue
  valutazioni, a Eulero ne servono più di centoventotto e a un metodo del
  secondo ordine ventidue; con trentadue, quello del secondo ordine è già tre
  volte più preciso.
- L'ordine alto **non conviene sempre**. Con pochissimi passi è più fragile e
  fa peggio; con la guida forte va riscritto in modo da estrapolare la stima
  dell'immagine pulita invece del disturbo; e sotto una certa soglia il limite
  diventa l'errore della rete, che nessun integratore può togliere.
```
`````

`````{tab} Superiore
```{admonition} Da ricordare
:class: important
- La PF-ODE è **semilineare**: la parte $f(t)\mathbf{x}$ ha soluzione esatta
  per fattore integrante, e l'unico errore di discretizzazione sta
  nell'integrale che contiene
  $\boldsymbol{\epsilon}_\theta$. È la definizione di **integratore
  esponenziale**, famiglia nata per i problemi stiff.
- Nel tempo $\lambda_t=\log(\alpha_t/\sigma_t)$ la soluzione è
  $\mathbf{x}_t = \frac{\alpha_t}{\alpha_s}\mathbf{x}_s -
  \alpha_t\int_{\lambda_s}^{\lambda_t}e^{-\lambda}
  \hat{\boldsymbol{\epsilon}}_\theta\,\mathrm{d}\lambda$: l'integrale è pesato
  esponenzialmente, quindi Taylor in $\lambda$ con coefficienti
  $\int e^{-\lambda}\lambda^k$ in forma chiusa.
- Troncando all'ordine zero si riottiene **DDIM**,
  $\mathbf{x}_t=\frac{\alpha_t}{\alpha_s}\mathbf{x}_s-\sigma_t(e^h-1)
  \boldsymbol{\epsilon}_\theta$, che è quindi un integratore esponenziale del
  primo ordine e non un Eulero. Ordini misurati sul banco di prova: Eulero
  $0{,}78$, DDIM $1{,}07$, DPM-Solver-2 $2{,}08$.
- I tre limiti dell'ordine alto: instabilità a passo grande (il pareggio col
  primo ordine cade attorno alle sedici valutazioni), norma di
  $\tilde{\boldsymbol{\epsilon}}$ che cresce con la guida (donde
  **DPM-Solver++**, che estrapola $\hat{\mathbf{x}}_0$ ed è multipasso), e il
  pavimento dovuto all'errore $\delta$ della rete.
- La griglia è una scelta **ortogonale** al metodo, e va uniforme in
  $\lambda$.
```
`````

Il pavimento dell'errore del modello è il punto in cui questa strada finisce.
Percorrere meglio una traiettoria ha un limite, e per scendere sotto le dieci
valutazioni bisogna cambiare il problema: invece di seguire la traiettoria un
tratto alla volta, insegnare a qualcuno a saltare da un capo all'altro.
