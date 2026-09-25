# Una complessità misurata sui dati: Rademacher e il margine

La dimensione VC si calcola una volta per tutte, guardando la famiglia di
regole e nient'altro: è la sua forza, perché la garanzia vale per qualunque
distribuzione, ed è il suo limite, perché non vede i dati che si hanno
davanti. C'è un altro modo di misurare quanto una famiglia sia ricca, e lo si
calcola proprio su quei dati: chiederle di inseguire il rumore. Da lì viene la
**complessità di Rademacher**, e da lì la ragione per cui una strada larga fra
le classi, quella che la {doc}`SVM </MachineLearning/svm>` va a cercare, vale
più del numero di direzioni in cui la si traccia.

## La prova delle monete

`````{tab} Elementare

Per capire se una famiglia di regole è troppo ricca la si mette alla prova su
un compito impossibile. Si prendono i propri esempi, si butta via la risposta
giusta di ciascuno e al suo posto si lancia una moneta: testa vuol dire «sì»,
croce vuol dire «no». Poi si cerca, nella famiglia, la regola che va più
d'accordo con le monete. È il gioco del righello, con l'avversario che colora
tirando una moneta invece che con malizia. Le monete non hanno niente da
insegnare, quindi se una
regola le indovina lo fa per caso, o perché la famiglia è abbastanza ricca da
inseguire qualunque cosa.

Si ripete tante volte, con monete nuove, e ogni volta si conta di quanto le
monete indovinate dalla regola migliore superano quelle sbagliate, in
proporzione al totale: chi le indovina tutte fa uno, chi ne azzecca la metà fa
zero, e una famiglia che frantuma i punti fa uno a ogni lancio. Una famiglia
che su dieci esempi insegue le monete quasi alla perfezione è come un indovino
che ha una spiegazione per tutto: quando poi indovina anche le risposte vere,
non sta dicendo niente. Una famiglia che resta vicina allo zero, quando
indovina le risposte vere, sta dicendo qualcosa.

Quella media si chiama complessità di Rademacher, dal matematico Hans
Rademacher, che studiò successioni di più uno e meno uno che si comportano come
monete lanciate a caso. Ha un pregio che la dimensione VC non ha: si misura sui
propri esempi, non sul caso peggiore immaginabile. Lo stesso righello, su punti
tutti in fila lungo una linea, può soltanto tagliare la fila in due, una parte
per colore, e insegue le monete peggio che su punti sparpagliati: la misura se
ne accorge, la dimensione VC no.

La garanzia che ne esce ha la stessa forma di quella della dimensione VC.
L'errore vero sta sotto l'errore sugli esempi, più quel punteggio sulle monete,
più un pezzetto che si restringe con il numero degli esempi. E anche il
punteggio si restringe: una
famiglia che insegue dieci monete senza fatica ne insegue mille molto peggio.
Resta un inconveniente pratico: trovare davvero la regola migliore
sulle monete può essere un conto lunghissimo, e per molte famiglie ci si
accontenta di stimarla.

`````

`````{tab} Superiore

Per una famiglia $\mathcal{G}$ di funzioni a valori reali e un campione
$S=(z_1,\dots,z_m)$, la **complessità di Rademacher empirica** è

$$
\hat{\mathfrak{R}}_S(\mathcal{G}) =
\mathbb{E}_{\boldsymbol{\sigma}}\Big[\sup_{g\in\mathcal{G}}
\frac{1}{m}\sum_{i=1}^{m}\sigma_i\, g(z_i)\Big],
$$

con $\sigma_1,\dots,\sigma_m$ indipendenti e uniformi su $\{-1,+1\}$ (le
variabili di Rademacher), e la complessità di Rademacher è
$\mathfrak{R}_m(\mathcal{G})=\mathbb{E}_{S\sim\mathcal{D}^m}\big[\hat{\mathfrak{R}}_S(\mathcal{G})\big]$.
Il sup misura la correlazione migliore che la famiglia raggiunge con un rumore
puro. Per la classificazione binaria,
$\mathcal{H}\subseteq\{-1,+1\}^{\mathcal{X}}$ con perdita 0-1, vale con
probabilità almeno $1-\delta$, per ogni $h\in\mathcal{H}$ (Mohri e colleghi
{cite}`mohri2018foundations`, teorema 3.5),

$$
R(h) \le \hat{R}_S(h) + \mathfrak{R}_m(\mathcal{H}) + \sqrt{\frac{\log\frac{1}{\delta}}{2m}},
\qquad
R(h) \le \hat{R}_S(h) + \hat{\mathfrak{R}}_S(\mathcal{H}) + 3\sqrt{\frac{\log\frac{2}{\delta}}{2m}} .
$$

La costante davanti alla complessità è $1$ e non il $2$ del teorema generale
per funzioni a valori in $[0,1]$, perché la famiglia delle perdite 0-1 ha
complessità pari a metà di quella di $\mathcal{H}$. La dimostrazione combina la
disuguaglianza di McDiarmid, che concentra il sup delle deviazioni attorno
alla sua media, con la stessa simmetrizzazione del bound VC, che trasforma
quella media in una correlazione con segni casuali
{cite}`koltchinskii2001rademacher,bartlett2002rademacher`.

La seconda forma è **dipendente dai dati**: $\hat{\mathfrak{R}}_S$ si calcola
sul campione, e premia i campioni benevoli (punti allineati, poche dicotomie
realizzabili). Il legame con la teoria di prima è il lemma di Massart
(corollario 3.8):
$\mathfrak{R}_m(\mathcal{H})\le\sqrt{2\log\Pi_{\mathcal{H}}(m)/m}$, che con
Sauer dà di nuovo $O\big(\sqrt{d\log m/m}\big)$, cioè Rademacher non è mai
peggio di VC a meno delle costanti, e sui dati buoni è meglio. Il prezzo è
computazionale: calcolare $\hat{\mathfrak{R}}_S$ vuol dire minimizzare l'errore
empirico su etichette casuali, un problema NP-difficile per alcune famiglie, e
nella pratica lo si maggiora invece di calcolarlo.

`````

## La strada larga

Il caso in cui il conto si fa in una riga, e dice qualcosa che la dimensione
VC non può dire, è quello delle funzioni lineari, i righelli in qualunque numero
di direzioni, quando ai loro pesi si impone un tetto, la norma limitata. È la
famiglia su cui lavora una SVM, a meno del termine noto, che si assorbe
aggiungendo ai punti una coordinata costante.

`````{tab} Elementare

Un righello qualsiasi e un righello che deve lasciare libera, fra i due
colori, una strada larga: il secondo ha meno libertà, perché per molte
colorazioni una strada così non passa. È la strada più larga della
{doc}`sezione sulle SVM </MachineLearning/svm>`, ed è anche un modo di tenere
bassa la ricchezza della famiglia.

La larghezza però va misurata rispetto a quanto sono sparsi i punti. Una
strada larga un metro fra punti sparsi su un chilometro è un sentiero; fra
punti raccolti in una stanza è un'autostrada. Quello che conta è il rapporto
fra la larghezza (il margine, la parola con cui la SVM la chiama) e l'ingombro
dei punti, e da quel rapporto dipende quanto il righello riesce a inseguire le
monete.

Non dipende invece dal numero di direzioni in cui i punti possono stare. Su
un foglio un righello qualsiasi frantuma tre punti; in uno spazio di mille
direzioni ne frantuma mille e uno, e con mille esempi il conto della
dimensione VC non garantisce più niente. Al righello con la strada larga, a
parità di rapporto fra larghezza e ingombro, il tetto a quanto riesce a
inseguire le monete è lo stesso in mille direzioni e sul foglio, e la garanzia
regge: la distanza fra due punti si misura in metri, non in direzioni, e se due
punti distano un metro e la strada deve essere larga due, nessun righello li
separa lasciando la strada, in due direzioni come in mille.

C'è un prezzo, ed è dentro la regola stessa. Un punto che cade dentro la
strada, anche dalla parte giusta, si conta come sbagliato: pretendere la
strada larga vuol dire accettare qualche errore in più sugli esempi in cambio
di una garanzia migliore sui nuovi. Ed è questo che permette a una SVM, che
cerca proprio la strada più larga, di funzionare anche quando la trasporta, con
il trucco del {doc}`kernel </MachineLearning/svm-kernel>`, in spazi con
infinite direzioni.

`````

`````{tab} Superiore

Sia $\mathcal{H}=\{\mathbf{x}\mapsto\mathbf{w}^\top\mathbf{x} :
\lVert\mathbf{w}\rVert\le\Lambda\}$ e i punti in una palla di raggio $r$.
Allora (Mohri e colleghi, teorema 5.10; per i nuclei, teorema 6.12)

$$
\hat{\mathfrak{R}}_S(\mathcal{H}) \le \sqrt{\frac{r^2\Lambda^2}{m}},
$$

e la dimostrazione è di due righe: il sup su $\mathbf{w}$ si calcola con
Cauchy-Schwarz,
$\sup_{\lVert\mathbf{w}\rVert\le\Lambda}\frac{1}{m}\mathbf{w}^\top\sum_i\sigma_i\mathbf{x}_i
=\frac{\Lambda}{m}\big\lVert\sum_i\sigma_i\mathbf{x}_i\big\rVert$, e per Jensen
$\mathbb{E}\lVert\sum_i\sigma_i\mathbf{x}_i\rVert\le\big(\sum_i\lVert\mathbf{x}_i\rVert^2\big)^{1/2}\le
r\sqrt{m}$, perché i termini incrociati hanno media nulla. La dimensione dello
spazio non compare da nessuna parte.

Con la perdita di margine, che conta come errore ogni esempio con
$y\,h(\mathbf{x})<\rho$ anche se classificato correttamente, e il suo rischio
empirico $\hat{R}_{S,\rho}(h)$, il bound di margine per le funzioni lineari
(corollario 5.11) è

$$
R(h) \le \hat{R}_{S,\rho}(h) + 2\sqrt{\frac{r^2\Lambda^2/\rho^2}{m}}
+ \sqrt{\frac{\log\frac{1}{\delta}}{2m}} ,
$$

che discende dal bound generale
$R(h)\le\hat{R}_{S,\rho}(h)+\frac{2}{\rho}\mathfrak{R}_m(\mathcal{H})+\sqrt{\log(1/\delta)/(2m)}$
(teorema 5.8). Conta il rapporto $r\Lambda/\rho$, l'inverso del margine
normalizzato, e non $d$: per gli iperpiani senza vincolo sui pesi, con il
termine noto che la SVM ha, $\mathrm{VCdim}=d+1$ (senza, $d$), e con $m\le d+1$
il bound VC è vuoto. Il massimo margine della SVM, $\min\lVert\mathbf{w}\rVert$
a margine funzionale unitario, è la minimizzazione diretta di quel rapporto, e
il bound resta invariato nello spazio di Hilbert a nucleo riproducente del
{doc}`kernel trick </MachineLearning/svm-kernel>`, con $r^2=\sup_{\mathbf{x}}
K(\mathbf{x},\mathbf{x})$: è una giustificazione teorica del fatto che una SVM
a kernel gaussiano, la cui famiglia ha dimensione VC infinita, generalizzi.

`````

Il blocco calcola la complessità di Rademacher empirica delle funzioni lineari
con $\lVert\mathbf{w}\rVert\le 1$, su punti presi a caso sulla sfera unitaria
($r=1$), in due dimensioni e in mille, e la mette accanto al tetto
$r\Lambda/\sqrt{m}$ e ai due pezzi che la complessità aggiunge all'errore
sugli esempi: quello della garanzia di margine, per un margine richiesto
$\rho=0{,}5$, e quello della garanzia VC per gli iperpiani.

```python
import numpy as np
from math import e, log, sqrt

rng = np.random.default_rng(2)

def rademacher_lineare(X, norma_w=1.0, estrazioni=2000):
    """Complessità di Rademacher empirica di {x -> w.x : ||w|| <= norma_w}.
    Il massimo su w si fa a mano: per Cauchy-Schwarz vale
    norma_w / m * ||somma_i sigma_i x_i||, e resta da mediare sulle monete."""
    m = len(X)
    sigma = rng.choice([-1.0, 1.0], size=(estrazioni, m))
    return norma_w / m * np.linalg.norm(sigma @ X, axis=1).mean()

def sulla_sfera(m, d):
    X = rng.standard_normal((m, d))
    return X / np.linalg.norm(X, axis=1, keepdims=True)   # tutti con ||x|| = 1

RHO = 0.5                                    # il margine richiesto
for d in (2, 1000):
    for m in (10, 100, 1000):
        rad = rademacher_lineare(sulla_sfera(m, d))
        # i due termini: margine (2/rho) e VC (iperpiani con termine noto: d + 1)
        margine = 2 / RHO * rad
        vc = sqrt(2 * (d + 1) * log(e * m / (d + 1)) / m) if m > d + 1 else float("inf")
        print(f"d={d:>4} m={m:>4}: Rademacher {rad:.4f} (tetto {1 / sqrt(m):.4f});"
              f" termine di margine {margine:.3f}, termine VC {vc:.3f}")
```

```text
d=   2 m=  10: Rademacher 0.2812 (tetto 0.3162); termine di margine 1.125, termine VC 1.150
d=   2 m= 100: Rademacher 0.0896 (tetto 0.1000); termine di margine 0.359, termine VC 0.520
d=   2 m=1000: Rademacher 0.0283 (tetto 0.0316); termine di margine 0.113, termine VC 0.202
d=1000 m=  10: Rademacher 0.3162 (tetto 0.3162); termine di margine 1.265, termine VC inf
d=1000 m= 100: Rademacher 0.0999 (tetto 0.1000); termine di margine 0.400, termine VC inf
d=1000 m=1000: Rademacher 0.0316 (tetto 0.0316); termine di margine 0.126, termine VC inf
```

Il numero stampato è la correlazione con le monete, che vale $1$ quando la
famiglia le insegue tutte e $0$ quando ne indovina metà: con dieci punti sul
cerchio di raggio uno vale $0{,}28$. In due dimensioni i due pezzi sono dello
stesso ordine. In mille dimensioni la garanzia VC non esiste più, perché mille
esempi non bastano nemmeno a superare la dimensione VC degli iperpiani, mentre
il pezzo di margine ha lo stesso tetto che nel piano, e gli sta a un passo: il
numero di direzioni non entra. Il tetto $1/\sqrt{m}$, anzi, in mille dimensioni
viene toccato quasi esattamente. Punti presi a caso in mille direzioni sono,
come vettori, quasi perpendicolari fra loro, e la lunghezza della loro somma
con segni a caso è quasi la stessa a ogni sorteggio. Nel piano, dove le
direzioni sono due, a volte i vettori si sommano e a volte si cancellano, la
lunghezza oscilla da un sorteggio all'altro, e la media di una lunghezza che
oscilla resta un po' sotto il valore che il tetto calcola.

`````{tab} Elementare
```{admonition} Da ricordare
:class: important
- La complessità di Rademacher chiede alla famiglia di inseguire monete
  lanciate a caso sui propri esempi: più ci riesce, meno vale un suo successo
  sulle risposte vere.
- Si misura sui dati che si hanno, e su dati benevoli dà garanzie migliori
  della dimensione VC.
- Per un righello che deve lasciare una strada larga conta il rapporto fra
  la larghezza della strada e l'ingombro dei punti, non il numero di
  direzioni: è la ragione per cui la SVM funziona anche con infinite
  direzioni. I punti dentro la strada contano come errori.
```
`````

`````{tab} Superiore
```{admonition} Da ricordare
:class: important
- $\hat{\mathfrak{R}}_S(\mathcal{G})=\mathbb{E}_{\boldsymbol\sigma}\big[\sup_{g}\frac1m\sum_i\sigma_i g(z_i)\big]$;
  per la classificazione binaria
  $R(h)\le\hat{R}_S(h)+\mathfrak{R}_m(\mathcal{H})+\sqrt{\log(1/\delta)/(2m)}$,
  e la versione empirica con $3\sqrt{\log(2/\delta)/(2m)}$.
- Massart: $\mathfrak{R}_m\le\sqrt{2\log\Pi_{\mathcal{H}}(m)/m}$, quindi mai
  peggio di VC a meno delle costanti; calcolarla è in generale NP-difficile.
- Lineari con $\lVert\mathbf{w}\rVert\le\Lambda$, $\lVert\mathbf{x}\rVert\le r$:
  $\hat{\mathfrak{R}}_S\le r\Lambda/\sqrt{m}$, e il bound di margine
  $R(h)\le\hat{R}_{S,\rho}(h)+2r\Lambda/(\rho\sqrt{m})+\sqrt{\log(1/\delta)/(2m)}$
  non dipende da $d$.
```
`````

Tutte e tre le misure viste fin qui guardano la famiglia di regole, o la
famiglia insieme ai dati, e mai il modo in cui l'algoritmo sceglie dentro la
famiglia. Per le SVM basta, perché la famiglia è stretta dal vincolo sul
margine. Per le reti neurali, come si vede appena si prova, no.
