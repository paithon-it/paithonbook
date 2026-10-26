<!-- GENERATO da scripts/genera-aggiornamenti.py: non modificare a mano.
     La fonte e' book/_dati/aggiornamenti.yml, e la prossima rigenerazione
     cancella qualunque cosa venga scritta qui dentro. -->

(aggiornamenti)=

# Registro degli aggiornamenti

Paithon Book si aggiorna per versioni successive: si aggiungono capitoli e
sezioni, si correggono gli errori, si riscrivono le spiegazioni che non
funzionano. Questa pagina elenca le modifiche di ogni versione, dalla più
recente alla più antica, con il collegamento alla pagina interessata.

La cronologia completa delle modifiche al sorgente è pubblica [su
GitHub](https://github.com/paithon-it/paithonbook/commits/main); qui sono
riportate soltanto quelle che riguardano il testo.

Gli errori si segnalano [su GitHub](https://github.com/paithon-it/paithonbook/issues).
Le correzioni proposte dai lettori entrano nel registro come le altre, e il
commit che le pubblica cita chi le ha segnalate.

## Il numero di versione

Il numero di versione ha tre cifre, `major.minor.patch`, come nel
versionamento semantico del software:

- **major**, la prima, cambia con l'impianto del libro: una nuova parte
  nell'indice, un riordino del percorso di lettura, la licenza;
- **minor**, la seconda, con l'aggiunta di un capitolo o di una sezione;
- **patch**, la terza, con le correzioni e le revisioni del testo esistente,
  e con le migliorie al sito e al PDF.

Quando una cifra aumenta, quelle alla sua destra tornano a zero. Una versione
che contiene sia sezioni nuove sia correzioni prende il numero della modifica
maggiore, e ogni versione corrisponde a una pubblicazione: sotto ciascuna, il
livello dice quale cifra è salita.

(v1-17-9)=

## 1.17.9 · 26 ottobre 2026

### Patch

- **L'indice segue il dito.** Sul telefono il pannello dell'indice si trascina: resta sotto il dito e, al rilascio, finisce di aprirsi o torna indietro. Il gesto non parte più a ridosso del bordo sinistro, che resta al browser per tornare alla pagina precedente.

(v1-17-8)=

## 1.17.8 · 26 ottobre 2026

### Patch

- **L'indice a portata di dito.** Sul telefono un trascinamento verso destra apre l'indice, e uno verso sinistra lo richiude. Il gesto parte dall'interno della pagina: quello dal bordo dello schermo resta al browser, che torna alla pagina precedente.

(v1-17-7)=

## 1.17.7 · 25 ottobre 2026

### Patch

- **Statistiche di visita, senza cookie.** Il libro conta le visite e i download dei PDF con il Plausible di paithon.it: niente cookie, niente script di terzi, e nessun conteggio se il browser chiede di non essere tracciato. Lo dice una riga in fondo a ogni pagina, con l'informativa.
- **Ogni pagina si presenta da sé.** Nei risultati di ricerca, nella scheda del browser e nell'anteprima di un link condiviso ogni pagina ha un titolo e una descrizione propri. Il titolo nella pagina, l'indice e il PDF non cambiano.

(v1-17-6)=

## 1.17.6 · 24 ottobre 2026

### Patch

- **Sul telefono, anche il capoverso.** Tenendo premuto il capoverso col dito la postilla si scrive al suo posto, e un tocco la richiude.

(v1-17-5)=

## 1.17.5 · 24 ottobre 2026

### Patch

- **Iterazione, non giro.** Fuori dalle schede Elementari la ripetizione di un procedimento si chiama iterazione, come la definisce il capitolo su Python: nel testo, nel codice, nelle figure e nelle postille.
- **Le due schede fanno le stesse mosse.** Nella valutazione dei raggruppamenti e negli autoencoder la scheda Elementare racconta ciò che la Superiore dimostra, compreso dove ciascun risultato smette di valere.
- {doc}`Deep Q-Network (DQN) </DeepReinforcementLearning/dqn>` (Deep Reinforcement Learning). **La figura del DQN parla come il capitolo.** Memoria di replay e rete-target, al posto di buffer e copia congelata.
- **Una frase sottolineata per sezione.** In ogni sezione che ne ha una, la matita segna la frase su cui la sezione si regge: sul sito e nel PDF.
- **Le postille si scrivono sul capoverso.** Premuto il segno, il capoverso sfuma e la postilla si scrive al suo posto; tenendolo premuto la si sbircia, un tocco la lascia scritta.

(v1-17-4)=

## 1.17.4 · 24 ottobre 2026

### Patch

- **La sottolineatura a matita.** Come negli articoli del sito, un tratto teal staccato dalle lettere può segnare la frase che regge una sezione, una per sezione, sul sito e nel PDF.

(v1-17-3)=

## 1.17.3 · 24 ottobre 2026

### Patch

- **Sito e PDF con un motore aggiornato.** L'indice a destra evidenzia la sezione che stai leggendo, il selettore del tema propone anche le impostazioni del sistema, e il PDF ha lo stesso contenuto in meno pagine.

(v1-17-2)=

## 1.17.2 · 22 ottobre 2026

### Patch

- **Altre correzioni di teoria.** Fra le altre: la disuguaglianza di Markov applicata a una funzione crescente dello scarto, il teorema del minimax che vale per i giochi finiti, i pesi dell'attenzione nella traduzione calcolati da una piccola rete, le matrici ortogonali della SVD che comprendono anche le riflessioni.
- **Titoli col nome delle cose.** Più sezioni nominano l'oggetto accanto all'immagine che lo racconta: la smoothing spline, il modality gap, la complessità di Rademacher, l'allucinazione visiva, che è anche la nuova voce dell'indice.
- **Un nome per ogni cosa.** Nel deep reinforcement learning la memoria di replay e la rete-target si chiamano allo stesso modo in tutte le sezioni, e nei modelli di diffusione il peso della guida non diventa più «forza».

(v1-17-1)=

## 1.17.1 · 22 ottobre 2026

### Patch

- {doc}`Imparare a imparare </DeepLearning/meta-apprendimento>` (Deep Learning). **La media delle onde.** La rete allenata su tutte le onde insieme ne impara la media, che è anch'essa un'onda della famiglia: è vicina a poche e lontana da tutte le altre, non diversa da tutte.

(v1-17-0)=

## 1.17.0 · 22 ottobre 2026

### Minor

- **Le postille.** Accanto a molti capoversi una riga dice di che cosa parla il capoverso: sul sito la apre il segno a margine, nel PDF sta nel margine esterno, in corsivo.
- **Prosa rivista in tutto il libro.** Frasi più dirette e capoversi più corti, i passaggi fra un'idea e l'altra detti in chiaro, e nei titoli e nei rimandi il nome che il libro dà alle cose al posto dell'immagine.
- **Correzioni di teoria in più capitoli.** Fra le altre: il rumore che DBSCAN lascia fuori dai gruppi, la disuguaglianza dell'unione nei limiti PAC, la quota che il label smoothing divide fra le classi, i tensor core e l'underflow in bfloat16.

(v1-16-1)=

## 1.16.1 · 20 ottobre 2026

### Patch

- {doc}`Gli strumenti </Harness/gli-strumenti>` (Harness engineering). **Gli strumenti con il loro nome.** La sezione sugli strumenti di un agente si intitolava «Gli attrezzi», la parola dell'analogia della bottega: titolo e rimandi ora usano il termine del resto del libro, e la pagina ha un indirizzo nuovo.

(v1-16-0)=

## 1.16.0 · 20 ottobre 2026

### Minor

- {doc}`Gli strumenti </Harness/gli-strumenti>` (Harness engineering). **Gli strumenti di un agente.** Come si progettano le operazioni che il modello può chiedere: che cosa devono restituire, come si modifica un file senza sbagliare versione, perché la riga di comando va chiusa in un recinto.
- {doc}`Il conto </Harness/il-conto>` (Harness engineering). **Il conto di un compito.** Quanto costa e quanto dura il lavoro di un agente: la cache del prefisso e che cosa la rompe, quale modello chiamare e quando smettere, il cammino critico e le azioni speculative; con una figura animata.
- {doc}`L'harness che si riscrive </Harness/l-harness-che-si-riscrive>` (Harness engineering). **L'harness che si riscrive.** Quando a modificare istruzioni, strumenti e memoria è un altro programma: l'archivio delle varianti, il voto tenuto da parte, e che cosa succede se chi viene misurato può toccare il metro.
- {doc}`Servire un agente </MLOps/servire-un-agente>` (MLOps). **Servire un agente.** Le richieste di un agente si fermano ad aspettare uno strumento: che cosa fare della memoria intanto, a chi dare la precedenza, su quale copia del modello mandarle; con una figura animata.
- **Caratteri, colori e figure scure dal design system di paithon.it.** La pagina usa i font e l'inchiostro del sito, senza più caricarli da fuori, e figure e animazioni in tema scuro sono ridisegnate con la sua mappa di colori.

(v1-15-0)=

## 1.15.0 · 18 ottobre 2026

### Minor

- {doc}`Harness engineering </Harness/overview>`. **Nuovo capitolo: Harness engineering.** Tutto ciò che sta attorno al modello in un agente che lavora per ore: le istruzioni nel repository, il primo turno che prepara, il passaggio di consegne fra sessioni, il collaudo da utente, l'osservabilità e l'entropia, il lavoro disegnato come grafo; con tre figure animate.
- **Prosa rivista nei capitoli su NLP e Transformer.** Frasi più dirette, termini italiani al posto dei calchi, rimandi che nominano la pagina di destinazione.
- **Una pagina più comoda da leggere.** Corpo del testo e titoli ridisegnati, titoli di sezione in oro, indice di pagina più leggero; in apertura, quattro ingressi per scegliere da dove cominciare.

(v1-14-3)=

## 1.14.3 · 14 ottobre 2026

### Patch

- **Approfondimenti nel livello Superiore sulle serie temporali, la diffusione e la raccomandazione.** Aggiunti la regressione del test ADF, l'attenzione congiunta di MM-DiT, il teorema della distillazione di consistenza, le basi di N-BEATS e lo stimatore IPS con le sue ipotesi.
- **Prosa più asciutta in tutti i capitoli.** Ridotti annunci e frasi accessorie; ogni cosa è indicata col suo nome e non con la posizione nella pagina; ogni rimando a un'altra sezione è un collegamento.
- **Attribuzioni e numeri riportati alle fonti.** Corretti, fra gli altri, il cancello di dimenticanza dell'LSTM, i FID di DiT rispetto alla U-Net e il confronto fra reti su grafo nella raccomandazione.

(v1-14-2)=

## 1.14.2 · 4 ottobre 2026

### Patch

- {doc}`Detection e segmentazione </VisioneArtificiale/detection-segmentazione>` (Visione artificiale). **Piramidi di feature e segmentazione panottica.** Il rilevamento di oggetti di scala diversa nella stessa immagine e la segmentazione panottica, che unisce quella semantica e quella d'istanza.
- {doc}`Geometria e profondità </VisioneArtificiale/geometria-e-profondita>` (Visione artificiale). **Omografia, matrice fondamentale e controllo sinistra-destra nella stereo.** Aggiunti nella sezione sulla geometria; nell'apertura del capitolo, il funzionamento del rilevatore di Canny.
- **Revisione della prosa in tutto il libro.** Italiano più diretto, termini italiani al posto dei calchi dall'inglese, rimandi che nominano la pagina di destinazione.
- **Affermazioni corrette in più capitoli.** Fra le altre, il costo della ricerca a griglia, le radici complesse nella diagonalizzazione, la densità di un flusso e il teorema alla base della disputa su COMPAS.

(v1-14-1)=

## 1.14.1 · 3 ottobre 2026

### Patch

- **Condizioni e casi limite nel livello Superiore.** Aggiunti condizioni, dimostrazioni e punti di rottura dei metodi: k-anonimato, rumore di Laplace, dualità di Fokker-Planck e Feynman-Kac, continuous batching, SMOTE e curva precision-recall.
- **Revisione del testo comune ai due livelli.** Ogni concetto compare con il proprio nome e la propria notazione al primo uso, e i capitoli sono raccordati dall'apertura al congedo.
- **Correzioni di fatto in tutto il libro.** Attribuzioni, date e affermazioni riportate alle fonti, con l'AI Act e il GDPR riletti sul testo; schede Elementari ricondotte a quanto sostiene il livello Superiore.
- **Numeri stampati dal codice.** Nuovi blocchi per i numeri commentati nel testo, dalla regolarizzazione nei sistemi di raccomandazione ai minimi quadrati a confronto con la PINN; il lettore è chiamato «tu» in tutto il libro.

(v1-14-0)=

## 1.14.0 · 27 settembre 2026

### Minor

- {doc}`Capacità e costo </MLOps/capacita-e-costo>` (MLOps). **Dimensionamento delle repliche e costo per token.** Capacità di una replica, ritardo dell'autoscaler e rimedi, costo di un token e quota di scheda effettivamente utilizzata.
- {doc}`Davanti ai modelli </MLOps/gateway-e-affidabilita>` (MLOps). **Gateway e affidabilità nell'erogazione dei modelli.** Limitazione delle richieste, ritentativi senza congestione, instradamento verso il modello adatto, tracciamento della richiesta e prove di guasto preventive.
- **Approfondimenti sull'inferenza.** Aggiunti i CUDA Graphs, Medusa ed EAGLE, la KV cache a precisione ridotta, la gestione della memoria esaurita, l'FP8 e la sparsità due su quattro.

(v1-13-0)=

## 1.13.0 · 25 settembre 2026

### Minor

- {doc}`Teoria dell'apprendimento </TeoriaApprendimento/overview>`. **Teoria dell'apprendimento: perché gli esempi bastano.** Nuovo capitolo su modello PAC, dimensione VC, complessità di Rademacher e margini, e su ciò che tali garanzie affermano, e non affermano, per le reti profonde.
- {doc}`Che cosa si può calcolare </Introduzione/calcolabile>` (Introduzione). **Calcolabilità e problema della fermata.** Nuova sezione sulla macchina di Turing, animata, e sulle conseguenze del problema della fermata per l'intelligenza artificiale.
- {doc}`Un'equazione come media </PINN/equazioni-come-medie>` (PINN: reti e fisica). **Equazioni alle derivate parziali come valori medi.** Nuova sezione sulla soluzione di un'equazione mediante cammini aleatori, con i cammini animati.
- **Ampliamento del capitolo sul machine learning.** Aggiunti, fra gli altri, lo spazio delle ipotesi, la logistica multinomiale e la massima entropia, la stima di densità, la LDA come regressione, l'ICA e la curva di validazione.
- **Ampliamento del capitolo di matematica.** Aggiunti il calcolo combinatorio e il coefficiente binomiale, la trasposta, le ricorrenze lineari di Leonardo Pisano, la stima bayesiana e la derivazione del fattore di momento.
- **Nuove sezioni in più parti del libro.** Fra le altre: perdita quadratica e media condizionata, ControlNet, autoencoder contrattivo, reti prototipiche, RuleFit, effetto Rashomon, SAGAN e BigGAN, ELMo, giochi imparziali.
- **Correzioni di teoria in tutto il libro.** Ipotesi mancanti aggiunte alle formule, affermazioni riportate alle fonti, numeri stampati dal codice; corretti il peso sui dati iniziali nelle PINN e la garanzia di MMD-critic.

(v1-12-17)=

## 1.12.17 · 23 settembre 2026

### Patch

- **Illustrazioni d'apertura ridisegnate.** Ognuna raffigura ora un solo oggetto, al centro della pagina, al posto di una scena composita.

(v1-12-16)=

## 1.12.16 · 23 settembre 2026

### Patch

- **Illustrazione d'apertura in ogni capitolo.** Ciascuna richiama l'aneddoto con cui il capitolo comincia, nelle cinque tinte del libro e in una versione per il tema scuro; la prefazione ne descrive le caratteristiche.

(v1-12-15)=

## 1.12.15 · 22 settembre 2026

### Patch

- **Sette figure nuove sui meccanismi.** Griglie di passi dei campionatori veloci, purga al confine fra training e test, regola di accettazione dello speculative decoding, pesi causali di una PINN, JEPA, over-squashing e randomized smoothing.

(v1-12-14)=

## 1.12.14 · 22 settembre 2026

### Patch

- **Sei formule spezzate a capo nel punto sbagliato.** Un segno meno o più a inizio riga interrompeva un elenco o inglobava la prosa nell'equazione; corrette le pagine su bootstrap, testo, rinforzo gerarchico, limite continuo, diffusione e verosimiglianza esatta.

(v1-12-13)=

## 1.12.13 · 22 settembre 2026

### Patch

- **Formule e ipotesi nelle schede Superiori.** Dove un metodo era nominato e non scritto, compare ora la formula con le sue ipotesi: fra gli altri contrastive divergence, VICReg, DreamerV3, GATv2, privacy differenziale, randomized smoothing.
- **Correzioni di sostanza nelle schede.** La purga toglie $h$ righe e non $h+r$; nella JEPA il collasso è evitato da stop-gradient e predictor insieme; una finestra di un token rende il watermark più robusto.
- {doc}`La legge dentro la loss </PINN/come-funziona>` (PINN: reti e fisica). **Numeri del problema inverso stampati dal codice.** Con il programma completo la rigidezza stimata è 3,95 (3,76 con il seme 7) e lo scarto massimo dalla curva vera 0,064, in luogo di 3,92 e 0,07.
- **Raccordo fra i capitoli.** I concetti già spiegati, fra cui il collasso, l'inferenza attiva, Dyna e la scatola nera, sono richiamati anziché ripresentati, e ogni rimando a un altro capitolo è un collegamento.

(v1-12-12)=

## 1.12.12 · 13 settembre 2026

### Patch

- {doc}`I flussi normalizzanti </VerosimiglianzaEsatta/flussi>` (Verosimiglianza esatta). **Campana gaussiana nell'animazione delle due lune.** La figura mostra la gaussiana vista dall'alto, con i punti che vi cadono: la campana corrisponde alla densità lungo la dimensione che il disegno non ha.
- **Scorrevolezza e correzioni in più capitoli.** Eliminate ripetizioni e termini usati prima della definizione; corretti, fra l'altro, la dimensione del batch in `nn.RNN`, `nn.LSTM` e `nn.GRU` e il tetto di accuratezza di 0,866.

(v1-12-11)=

## 1.12.11 · 12 settembre 2026

### Patch

- **Figure a schermo intero nel tema scuro.** Nel visore la figura mantiene la resa del tema scelto: in tema scuro non compare più su fondo chiaro.

(v1-12-10)=

## 1.12.10 · 11 settembre 2026

### Patch

- **Capoversi spezzati in elenchi.** Corretti tre capoversi in cui un a capo prima di «1.», «-» o «+» era letto come voce di elenco: ricerca casuale degli iperparametri, PINN, varianza campionaria di un MA(q).

(v1-12-9)=

## 1.12.9 · 11 settembre 2026

### Patch

- **Sette figure nuove su memoria, parallelismo e geometria ad alta dimensione.** Fra esse la memoria a stato fisso, il coseno fra direzioni casuali in tre, cento e mille dimensioni e il torneo delle configurazioni.
- **Definizioni anticipate: «latente», «iperparametri», «message passing».** Altri termini sono spiegati dove compaiono o rinviati alla sezione che li definisce; corretti anche DBSCAN, le regole sulla potenza statistica e l'esercizio su T5.
- **Riscontro con le fonti su BERT, SoundStream e Whisper.** Il pre-addestramento di BERT è durato quattro giorni su sedici chip, il riavvio dei codebook è di SoundStream, Whisper copre novantasei lingue oltre all'inglese.

(v1-12-8)=

## 1.12.8 · 11 settembre 2026

### Patch

- **Figure animate sui meccanismi dinamici.** Continuous batching, Q-learning, nastro di autograd e RAG originale; fra le figure ferme, il peso del trapezio in Mamba-3 e il ventaglio di previsione.
- **Definizioni anticipate: «rollout», «autograd», «ELBO».** Altri termini sono spiegati dove compaiono o rinviati alla sezione che li definisce, fra cui «negativi», «flussi normalizzanti» e «context engineering»; aggiunte voci bibliografiche.
- **Fonti primarie su quantizzazione, purga e pinball loss.** Il punto di svolta dei 4 bit con Dettmers e Zettlemoyer, le ipotesi di Baldi e Hornik, purga ed embargo di López de Prado, la convenzione della pinball loss.
- **Affermazioni corrette sulle fonti.** Il divieto sulle immagini intime nel regolamento europeo ha tre rami e non due; la macchina di Boltzmann ristretta non nacque per la gara del Netflix Prize.

(v1-12-7)=

## 1.12.7 · 10 settembre 2026

### Patch

- **Figure animate su linee di base, campionamento, modelli latenti e Transformer.** Quattro linee di base sulla stessa serie, le due manopole del campionamento, gli assi che ruotano sulla nuvola di punti e la taratura nei due montaggi.
- **Definizioni anticipate: «validazione», «linea di base», «kernel».** Altri termini sono spiegati dove compaiono o rinviati alla sezione che li definisce; corretta la leave-one-out, meno stabile e non più solida.
- **Riscontro con le fonti su dimenticanza catastrofica e doppia discesa.** Corretti il rimedio alla dimenticanza catastrofica, la doppia discesa con arresto anticipato, il blocco più economico del diffusion transformer e l'accordo fra giudice automatico e umani.
- **Un solo neretto per termine in tutto il libro.** Un termine è marcato soltanto alla prima ricorrenza; gli omonimi, la stessa parola per mestieri diversi, sono dichiarati sul posto.

(v1-12-6)=

## 1.12.6 · 9 settembre 2026

### Patch

- **Correzioni di fatto in Matematica, diffusione, rinforzo profondo, GAN e PINN.** Un «92 %» dove la pagina misura 93, un «vent'anni» fra il 1994 e il 2019, affermazioni più forti delle fonti, promesse senza seguito.
- **Neretto ridotto in tutto il libro.** Segna un termine alla prima occorrenza, salvo dove una definizione serve a un capitolo successivo; le parole non cambiano.

(v1-12-5)=

## 1.12.5 · 8 settembre 2026

### Patch

- {doc}`L'attenzione in pratica </Transformers/attenzione-in-pratica>` (Transformer). **L'attenzione in pratica.** Nuova sezione sui Transformer: costo della stessa formula in addestramento e in generazione, KV cache e varianti che la riducono (MQA, GQA, MLA), con i conti di memoria.
- {doc}`Da dove viene la loss </RetiNeurali/da-dove-viene-la-loss>` (Reti neurali). **Origine della funzione di perdita.** Nuova sezione sulle reti neurali: la perdita si ricava dalla distribuzione dichiarata sull'uscita, con minimi quadrati e cross-entropy come due casi della stessa costruzione.
- **Errori di fatto, termini anticipati e rimandi inesistenti.** Corretti un cancello attribuito al ramo sbagliato, un conto di memoria, una frase smentita dalla scheda gemella e i rimandi a sezioni che non esistono.
- **Neretto limitato alla prima occorrenza.** Il neretto segna un termine la prima volta che compare e non ritorna; le parole non cambiano.

(v1-12-4)=

## 1.12.4 · 7 settembre 2026

### Patch

- **Neretto ridotto in GPU, Audio, Serie temporali e reti su grafo.** Segna ora un termine alla prima occorrenza e non più clausole intere o parole comuni; le parole non cambiano.
- **Versione scura delle figure.** Ogni figura, animazioni comprese, ha una versione per il tema scuro con gli stessi colori sul fondo scuro, in luogo del rettangolo bianco; il PDF resta chiaro.

(v1-12-3)=

## 1.12.3 · 2 settembre 2026

### Patch

- **Colonna di lettura uniforme.** Anche nelle pagine senza sommario laterale (prefazione, bibliografia, indice dei termini, ricerca) larghezza e margine sinistro coincidono con quelli dei capitoli.
- {doc}`Prefazione </prefazione>`. **Triangolo di Penrose in tema scuro.** Il segno del progetto conserva i propri colori anche nel tema scuro; il difetto era presente dalla 1.12.2.

(v1-12-2)=

## 1.12.2 · 1 settembre 2026

### Patch

- {doc}`Dalla softmax alla ricorrenza </AttenzioneLineare/dalla-softmax-alla-ricorrenza>` (Attenzione lineare). **Errore di richiamo calcolato dalla pagina.** Un blocco con seme proprio produce i due valori dell'errore di una memoria di taglia fissa sovraccarica, con l'andamento di confronto.
- {doc}`Modelli trasparenti </Interpretabilita/modelli-trasparenti-e-importanza>` (Interpretabilità). **Accuratezze del confronto fra modello leggibile e opaco.** Le due accuratezze e la differenza di un punto e tre sono ora stampate dalla pagina.
- **Correzioni a fatti, formule, citazioni e dati hardware.** Fatti storici riportati alle fonti, condizioni delle formule precisate, citazioni e numeri delle schede tecniche ricontrollati, scene riscritte perché ogni gesto corrisponda a un passaggio del conto.
- {doc}`Introduzione </Introduzione/overview>`. **Fatti storici nell'Introduzione.** L'epigrafe è la prima riga dell'articolo del 1966, la frase di Pascal è riportata per intero, ELIZA è un capostipite dei chatbot e non il primo.
- {doc}`Robotica e AI </Introduzione/applicazioni>` (Introduzione). **Definizione di robot secondo la norma ISO 8373.** Aggiunta la clausola sul grado di autonomia (edizione 2021); precisato che la limitatezza delle ricompense basta alla convergenza della somma scontata, ma non è necessaria.
- {doc}`Conclusione </Introduzione/conclusione>` (Introduzione). **Protocollo del confronto fra rete e cardiologi.** Nello studio i cardiologi messi a confronto non coincidono con quelli che hanno annotato il riferimento, a differenza di quanto il racconto lasciava intendere.
- {doc}`Il primo programma </Introduzione/massimo_comune_divisore>` (Introduzione). **Uscita salvata nel quaderno del massimo comune divisore.** La cella riporta ora il risultato commentato nella pagina.
- {doc}`Il filtraggio collaborativo </SistemiRaccomandazione/filtraggio-collaborativo>` (Sistemi di raccomandazione). **Fuga di informazione nell'esempio di filtraggio collaborativo.** L'errore sulle celle già viste e quello sulle nuove sono contati separatamente; per un utente appena iscritto la scheda tende a zero con minimi quadrati alternati o decadimento dei pesi.
- {doc}`La raccomandazione neurale </SistemiRaccomandazione/raccomandazione-neurale>` (Sistemi di raccomandazione). **Citazione e ricerca del vicino nella raccomandazione neurale.** Il lavoro citato mostra che l'addestramento sul futuro rende imprevedibile l'ordine fra i metodi; massimo prodotto scalare e vicino più prossimo coincidono solo a norma uguale.
- {doc}`La raccomandazione neurale </SistemiRaccomandazione/raccomandazione-neurale>` (Sistemi di raccomandazione). **Calcolo della NDCG e figura della vetrina.** Completati gli sconti per posizione, così che la classifica ideale sia ricostruibile; la figura mostra ora il sorpasso promesso dalla didascalia.
- {doc}`Dentro la GPU </GPU/architettura-gpu>` (GPU e calcolo parallelo). **Operazioni per ciclo nell'architettura GPU.** Una moltiplicazione-e-somma per ciclo vale due operazioni, e il conto torna ai mille miliardi delle schede tecniche; aggiornata la figura degli ordini di grandezza.
- {doc}`La memoria </GPU/gerarchia-memoria>` (GPU e calcolo parallelo). **Nomi dei livelli di cache.** La cache di secondo livello aveva due nomi, uno dei quali designava il primo livello; i dati che traboccano dai registri non sottraggono memoria condivisa alla squadra.
- {doc}`FlashAttention </GPU/flash-attention>` (GPU e calcolo parallelo). **Teste e gruppi in FlashAttention.** Le teste di lettura sono trentadue e otto i gruppi che condividono chiavi e valori; i gigabyte del risultato sono tredici, in base dieci, e non dodici.
- {doc}`GEMM e tensor core </GPU/gemm-e-tensor-core>` (GPU e calcolo parallelo). **Confronti omogenei fra GEMM e tensor core.** Il rapporto fra operazioni e byte vale un quarto o un mezzo secondo il formato; il processore dedicato citato è quello di prima generazione, a interi.
- {doc}`Imparare a imparare </DeepLearning/meta-apprendimento>` (Deep Learning). **Media della famiglia nel meta-apprendimento.** La sinusoide media appartiene alla famiglia: i suoi due parametri cadono negli intervalli dichiarati, contrariamente a quanto affermavano quattro punti della pagina.
- {doc}`Far funzionare le reti profonde </DeepLearning/ottimizzazione-regolarizzazione>` (Deep Learning). **Decadimento dei pesi e passo di apprendimento.** Anche nella versione separata il decadimento dipende dal passo, e ne cade il fattore dieci; la riserva sul label smoothing compare in entrambi i livelli.
- {doc}`Architetture storiche </DeepLearning/architetture-storiche>` (Deep Learning). **Reti residuali e parametri di Inception.** Rimosso un confronto di leggerezza di addestramento a pari accuratezza che nessuna fonte sostiene; corretti i parametri di Inception e la descrizione testuale della figura.
- {doc}`Reti convoluzionali (CNN) </DeepLearning/reti-convoluzionali>` (Deep Learning). **Equivarianza e pooling nelle reti convoluzionali.** L'equivarianza deriva dalla sola condivisione dei pesi; il pooling comprende anche la media, non soltanto il massimo.
- {doc}`Un modello che imita </Efficienza/un-modello-piccolo-che-imita>` (Efficienza). **Distillazione dal lavoro originale.** L'equivalenza con la forma semplice richiede temperatura alta rispetto ai punteggi e punteggi a media nulla per ciascun esempio; il blocco mostra quanto ci si scosti da quel regime.
- {doc}`Starci non è rispondere </Efficienza/far-rispondere-in-fretta>` (Efficienza). **Rapporto fra calcolo e byte nell'inferenza veloce.** L'approssimazione regge se ingressi e uscite sono molto meno dei pesi di una riga; all'ultima riga della tabella sbaglia del dodici per cento.
- {doc}`Le tre cose date per scontate </Ricerca/quando-il-mondo-non-si-conosce>` (Ricerca e pianificazione). **Scena della ricerca ad albero e punteggio del 1997.** La scena scava ora l'albero, e ogni partita riparte dal fondo del ramo già esplorato; il punteggio della sfida del 1997 è riportato per intero, con le patte.
- {doc}`Ricerca e pianificazione </Ricerca/overview>`. **Profondità media nella ricerca.** Il simbolo usato per la lunghezza tipica di una partita indicava altrove la profondità massima; corretta anche la descrizione testuale di una figura.
- {doc}`Reinforcement Learning </ReinforcementLearning/overview>`. **Fattore di sconto nel rinforzo.** Scena e trattazione formale dicono la stessa cosa: a sconto minimo l'agente è miope; mantenerlo sotto uno è la scelta consueta, non un obbligo nei compiti senza fine.
- {doc}`I bandit a più braccia </ReinforcementLearning/banditi>` (Reinforcement Learning). **Confronto fra strategie nei banditi.** L'affermazione «nessun'altra farà meglio» è limitata alle strategie confrontate (un'altra raggiunge il 91,8 per cento contro l'86,6); il confronto finale usa tre sorteggi invece di due.
- {doc}`I metodi Monte Carlo </ReinforcementLearning/monte-carlo>` (Reinforcement Learning). **Stimatori Monte Carlo con pesi di importanza.** Il primo dei due pesi coglie il valore giusto in media per qualunque numero di partite: illimitata è la varianza, non la distorsione; la contrapposizione vale contando la prima visita.
- {doc}`MDP e funzioni valore </ReinforcementLearning/mdp-valore>` (Reinforcement Learning). **Osservazioni e stato nei processi di decisione.** Guardare gli ultimi fotogrammi serve quando manca il movimento, non un'informazione assente da ogni fotogramma; le caselle più lontane dall'obiettivo sono due.
- {doc}`Q-learning e differenze temporali </ReinforcementLearning/q-learning>` (Reinforcement Learning). **Esempio del precipizio nel Q-learning.** La griglia dichiara le proprie dimensioni e il caso sceglie fra quattro direzioni, così che «una partita su quattro» sia un numero che si può rifare.
- {doc}`Dalla softmax alla ricorrenza </AttenzioneLineare/dalla-softmax-alla-ricorrenza>` (Attenzione lineare). **Contributi e capienza della memoria a taglia fissa.** I tre contributi non compaiono più due volte con valori identici dopo la moltiplicazione; il lato trentadue è presentato prima come capienza, poi come ciò che la erode.
- {doc}`Scrivere nella memoria </AttenzioneLineare/scrivere-nella-memoria>` (Attenzione lineare). **Attribuzione dei fast weight programmer.** Furono progettati da Schmidhuber negli anni Novanta; Schlag e colleghi nel 2021 vi riconobbero l'attenzione lineare. La forma WY è il prodotto compatto delle correzioni in un blocco.
- {doc}`Scrivere nella memoria </AttenzioneLineare/scrivere-nella-memoria>` (Attenzione lineare). **Memorie correttive e taglia fissa.** A parità di taglia correggere batte accumulare e sbiadire, ma il vantaggio non cresce con il modello: su memorie grandi conta più la capienza del modo di aggiornarle.
- {doc}`Architetture lineari </AttenzioneLineare/architetture-lineari>` (Attenzione lineare). **Conteggi dei lavori su RWKV-7 e xLSTM.** I modelli RWKV-7 con pesi aperti sono sette, non quattro; i limiti della LSTM indicati dagli autori di xLSTM sono tre; xLSTM-7B va alla pari con i modelli confrontabili.
- {doc}`Conclusioni </Conclusioni/overview>`. **Mappa dei modelli aperti e chiusi.** Il grafico a due assi non colloca più in alto i soli pesi scaricabili, in accordo con la didascalia: aperto non vuol dire debole, chiuso non vuol dire potente.
- {doc}`Conclusioni </Conclusioni/overview>`. **Ottimizzazione e discesa del gradiente nelle Conclusioni.** «Tutti i problemi di ottimizzazione» diventa «quasi tutti», perché DBSCAN, clustering gerarchico e alberi non hanno obiettivo globale; la discesa si ferma dove il gradiente si annulla.
- {doc}`Conclusioni </Conclusioni/overview>`. **Citazione degli esponenti delle leggi di scala.** I tre esponenti rinviano ora ai lavori da cui provengono, accanto alla figura e nel corpo del testo.
- {doc}`Conclusioni </Conclusioni/overview>`. **Rimandi corretti nelle Conclusioni.** Il richiamo all'introduzione descrive il compito giusto, nascondere la parola successiva, senza estenderlo alle immagini, e indica una sezione e non un capitolo.
- {doc}`Conclusioni </Conclusioni/overview>`. **Manuali consigliati e librerie.** La prima metà del manuale di Géron usa scikit-learn, la libreria di queste pagine; l'avvertenza indica ora dove passa il confine con Keras.
- {doc}`Allineare due spazi </VisioneLinguaggio/allineare-due-spazi>` (Visione e linguaggio). **Perdita contrastiva e spostamenti in blocco.** La perdita non ignora qualunque spostamento in blocco di una nuvola, ma solo una costante aggiunta dentro una riga; la conclusione sul minimo poggia ora su un argomento corretto.
- {doc}`Allineare due spazi </VisioneLinguaggio/allineare-due-spazi>` (Visione e linguaggio). **Riquadro e figura sull'allineamento di due spazi.** Il confronto della coppia giusta vale dentro una riga, non come soglia globale; la temperatura $0{,}01$ è il suo valore minimo; nella figura le lettere dei simboli sono distinte.
- {doc}`Innestare gli occhi </VisioneLinguaggio/innestare-gli-occhi>` (Visione e linguaggio). **Compressione dell'immagine e costo del calcolo.** Da seicento a cinquantadue pezzi i confronti calano di centotrenta volte, ma il lavoro totale di undici volte e mezzo; il quadrato prevale solo a decine di migliaia di pezzi.
- {doc}`Innestare gli occhi </VisioneLinguaggio/innestare-gli-occhi>` (Visione e linguaggio). **Testo alternativo della figura dei connettori.** Riportava quattro milioni di parametri, cifra della prima versione del modello; ora riporta lo stesso valore di disegno e didascalia.
- {doc}`Il costo del dettaglio </VisioneLinguaggio/risoluzione-e-dettaglio>` (Visione e linguaggio). **Riquadri di ricapitolazione sulla risoluzione.** Non affermano più che riordinare le tessere non perda nulla: la misura del posto in cui farle stare decide che cosa passa.
- {doc}`Agenti </Agenti/overview>`. **Istruzioni nel prompt e datazione degli agenti.** Seguire istruzioni è una capacità emersa con la scala, che non basta da sola a rendere possibili gli agenti; ReAct è dell'ottobre 2022, e il 2023-2024 indica la diffusione.
- {doc}`Il ciclo dell'agente </Agenti/agenti-e-tool-use>` (Agenti). **Calcolatore protetto e modello finto negli agenti.** Il calcolatore rifiuta ora `True` come numero; il modello finto dichiara di non sapere quando la ricerca va a vuoto, invece di inventare la risposta.
- {doc}`Il ciclo dell'agente </Agenti/agenti-e-tool-use>` (Agenti). **Rilettura degli errori da parte dell'agente.** La tecnica non migliora sempre la quota di problemi risolti al primo tentativo: su un banco di programmazione la quota scende, e la pagina riporta i due numeri.
- {doc}`RAG avanzato </Agenti/rag-avanzato>` (Agenti). **Soglia del filtro nel RAG avanzato.** Il criterio ha un minimo assoluto oltre alla metà del punteggio migliore, così che una ricerca senza passaggi pertinenti, con punteggi tutti nulli, non lasci passare tutto.
- {doc}`Architetture e valutazione </Agenti/architetture-e-valutazione>` (Agenti). **Banco di prova per la programmazione e versione ripulita.** La versione ripulita non è più presentata come rimedio: il riesame da cui provengono i numeri sui difetti dice che si ritrovano anche lì.
- {doc}`Il contesto è l'interfaccia </Agenti/context-engineering>` (Agenti). **Notazione della probabilità di errore e descrizione della figura del contesto.** La lettera $p$ non indica più tre grandezze; la descrizione riporta che l'ultimo segmento, lo spazio per la risposta, è più stretto del precedente.
- {doc}`Memoria associativa </ModelliEnergia/memoria-associativa>` (Modelli a energia). **Confronto con lo zero nella memoria associativa.** In virgola mobile i legami non tornavano a zero esatto e la regola di parità non si applicava; riparato, le sei misure sono rifatte su tre semi.
- {doc}`Memoria associativa </ModelliEnergia/memoria-associativa>` (Modelli a energia). **Capienza della rete di Hopfield e rete di Little.** Il valore $0{,}138$ sta nell'articolo del 1985, e $0{,}14$ ne è l'arrotondamento nel sommario; i neuroni di Little si accendono con probabilità crescente, non oltre una soglia netta.
- {doc}`Memoria associativa </ModelliEnergia/memoria-associativa>` (Modelli a energia). **Conteggio dei cambiamenti nella figura della memoria associativa.** L'ultimo pannello include l'ultimo capovolgimento, così che i sei cambiamenti annunciati dal testo coincidano con quelli disegnati.
- {doc}`Modelli a energia </ModelliEnergia/overview>`. **Documento di posizione del 2022 e seconda rinuncia.** La frase riportata fra virgolette non compare nel documento: il testo enuncia ora la tesi e attribuisce al documento l'argomento disteso, mentre la formulazione secca viene dalle conferenze.
- {doc}`Oltre la partizione </ModelliEnergia/oltre-la-partizione>` (Modelli a energia). **Passaggio fra le due immagini dell'energia.** Il passaggio dalla pioggia totale all'altezza aggiunta in ogni punto è ora scritto: scendere di un gradino non aggiunge pioggia, la moltiplica.
- {doc}`Paesaggi di oggi </ModelliEnergia/paesaggi-di-oggi>` (Modelli a energia). **Esponente dell'energia di Hopfield e formula di partenza.** L'idea del 2016 riporta nel deep learning un filone di fisica statistica degli anni Ottanta; la formula in cui ogni ricordo abbassa l'energia è ora scritta dove nasce.
- {doc}`Componenti e modelli classici </SerieTemporali/componenti-e-classici>` (Serie temporali). **Riduzione dell'errore nelle componenti classiche.** Da $2{,}49$ a $1{,}49$ la riduzione è del quaranta per cento, due quinti, e non un terzo.
- {doc}`Componenti e modelli classici </SerieTemporali/componenti-e-classici>` (Serie temporali). **Scala dei pannelli nella scomposizione di una serie.** La didascalia dichiara che ogni pannello ha la propria scala verticale: nella serie in alto la stagione vale circa tre volte e mezzo il residuo.
- {doc}`Componenti e modelli classici </SerieTemporali/componenti-e-classici>` (Serie temporali). **Box e Jenkins e fattori di lisciamento.** Box e Jenkins ebbero per primi la procedura di scelta del modello, non la prima procedura applicabile (il lisciamento esponenziale è del 1957); il terzo fattore dipende dal primo.
- {doc}`Serie temporali </SerieTemporali/overview>`. **Incertezza della previsione a più passi.** Si sommano i contributi delle sorprese, pesati da coefficienti che si smorzano, e non le sorprese stesse; per questo la banda di una serie stazionaria si assesta.
- {doc}`Validazione e feature </SerieTemporali/validazione-e-feature>` (Serie temporali). **Copertura delle bande di previsione.** Il degrado a orizzonti lontani deriva dall'errore sui parametri stimati e non dall'accumulo di incertezza; la copertura empirica non va minimizzata né massimizzata, ma coincidere con il livello dichiarato.
- {doc}`Forecasting neurale </SerieTemporali/forecasting-neurale>` (Serie temporali). **Ampiezza della banda nel forecasting neurale.** Al quinto giorno la banda stampata (3,166) raggiunge il 98,8 per cento del limite (3,204); il 99,7 si riferisce alla banda teorica, e la frase lo dichiara.
- {doc}`Collasso e misura </AutoSupervisione/collasso-e-misura>` (Auto-supervisione). **Tetto di informazione nei metodi contrastivi.** Con lotti da quattromila immagini i candidati sono 8191, quindi il tetto è tredici bit e non dodici, in accordo con il capitolo sulla visione artificiale.
- {doc}`Le quattro famiglie </AutoSupervisione/famiglie>` (Auto-supervisione). **Diagonale della matrice nell'esperimento contro il collasso.** La diagonale supera il valore massimo imposto dal rumore perché il proiettore si adatta agli esempi visti; il codice rimisura ora su esempi mai visti, dove il valore rientra.
- {doc}`Capire è accorciare </AutoSupervisione/capire-e-accorciare>` (Auto-supervisione). **Teorema di codifica di sorgente e coefficienti.** Il teorema riguarda la lunghezza attesa, e la codifica aritmetica raggiunge la disuguaglianza su una stringa data; i coefficienti sono $-0{,}935$, $-0{,}937$ e $-0{,}953$, e l'intervallo è arrotondato verso l'esterno.
- {doc}`Capire è accorciare </AutoSupervisione/capire-e-accorciare>` (Auto-supervisione). **Compressione congiunta e compressore ideale.** Comprimere due cose insieme non costa più che separatamente solo per il compressore ideale: è un'ipotesi e non un risultato, e la condizione compare ora anche nel riassunto.
- {doc}`Auto-supervisione </AutoSupervisione/overview>`. **Riquadro finale dell'auto-supervisione.** I bit citati riguardano una finestra di parole e non mezza fotografia, grandezze che differiscono di un fattore quattro; il riquadro è riallineato al testo.
- {doc}`Il dibattito sul rinforzo </AutoSupervisione/dibattito-rl>` (Auto-supervisione). **Passi sbagliati nella figura sul credito.** Il testo alternativo ne contava quattro, la figura ne mostra cinque; nella stessa sezione i passi di una prova riuscita ricevono lo stesso coefficiente, non la stessa spinta.
- {doc}`Spiegazioni locali </Interpretabilita/spiegazioni-locali>` (Interpretabilità). **Controfattuali, Shapley e studio sulla fiducia.** La distanza controfattuale divide ogni caratteristica per la sua dispersione tipica; Shapley nel 1953 fissò tre requisiti e non quattro; gli studenti dell'esperimento sulla fiducia erano già laureati.
- {doc}`Attribuzione e meccanicistica </Interpretabilita/attribuzione-e-meccanicistica>` (Interpretabilità). **Pesi di attenzione e controlli di sanità.** Chi li ha misurati scrive che si comportano meglio, non che siano assolti; il metodo detto fallire del tutto il controllo non lo fallisce nella versione corrente del lavoro.
- {doc}`Interpretabilità </Interpretabilita/overview>`. **Mappa di salienza e livelli di valutazione.** Le macchie sulla fotografia del lupo non sono una mappa di salienza, che la sezione dedicata definisce altrimenti; i tre livelli di rigore sono elencati in ordine crescente.
- {doc}`Modelli trasparenti </Interpretabilita/modelli-trasparenti-e-importanza>` (Interpretabilità). **Importanza per permutazione e valori negativi.** Quando rimescolare una colonna migliora le risposte, la causa è la scarsità dei dati e non il caso: il margine dichiarato misura la dispersione fra ripetizioni, non l'incertezza della media.
- {doc}`Modelli trasparenti </Interpretabilita/modelli-trasparenti-e-importanza>` (Interpretabilità). **Ripartizione del merito fra due colonne.** Un'unica estrazione non regge una ripartizione in parti: su otto semi i valori si spostano troppo; restano i due meccanismi che nel numero si sommano.
- {doc}`Loop engineering </IngegneriaLLM/loop-engineering>` (Prompt, contesto e loop). **Glossa della formula sull'affidabilità a catena.** A essere elevata a potenza è la probabilità di sbagliare un passo e non di riuscirvi: con errore del cinque per cento, venti passi riescono trentasei volte su cento.
- {doc}`Prompt engineering </IngegneriaLLM/prompt-engineering>` (Prompt, contesto e loop). **Etichetta del materiale e istruzioni nel prompt.** L'etichetta indica da dove viene il testo e non lo mette al sicuro: un modello non distingue i dati dai comandi, ed è per questo che l'attacco funziona.
- {doc}`Prompt engineering </IngegneriaLLM/prompt-engineering>` (Prompt, contesto e loop). **Mappa delle tecniche di prompt.** Il guadagno di una meta-analisi non vale per la tecnica che essa esclude dal conteggio, e il ruolo di esperto mondiale non è misurato da nessuna delle due fonti.
- {doc}`Prompt engineering </IngegneriaLLM/prompt-engineering>` (Prompt, contesto e loop). **Somma delle prime tre voci nel prompt engineering.** Le prime tre voci scendono a settantanove e mezzo, che arrotondato dà ottanta: la frase è ora vera sia sul conto sia sulle barre.
- {doc}`Context engineering </IngegneriaLLM/context-engineering>` (Prompt, contesto e loop). **Foglio di appunti nel context engineering.** Lo stesso nome indicava tre cose in due pagine (spazio nella finestra, stato esterno, il suo contrario): ogni uso ha ora il proprio nome.
- {doc}`Prompt, contesto e loop </IngegneriaLLM/overview>`. **Condizioni nei riquadri di sintesi dell'ingegneria dei modelli linguistici.** Ripristinate le riserve sull'indipendenza fra i passi, sulla fedeltà della catena di ragionamento e sulla validità di una misura per un solo modello; parlare bene al modello non è gratis.
- {doc}`I modelli di riconoscimento </SpeechRecognition/modelli-asr>` (Speech Recognition). **Allineamenti del trasduttore.** Sono $\binom{T+U-1}{U}$ e non $\binom{T+U}{U}$, perché l'ultimo passo chiude l'ultimo istante; cinquanta per undici fa cinquecentocinquanta, non «poche centinaia».
- {doc}`I modelli di riconoscimento </SpeechRecognition/modelli-asr>` (Speech Recognition). **Decodifica semplice e ripetizioni.** La decodifica che a ogni passo prende il simbolo più probabile non è la «decodifica del percorso migliore»: il lavoro citato la nomina per la tendenza a incepparsi ripetendo.
- {doc}`I modelli di riconoscimento </SpeechRecognition/modelli-asr>` (Speech Recognition). **Didascalia del simbolo vuoto e distanza di edit.** La rete può non assegnare alcun simbolo a un istante, ed è il motivo del simbolo vuoto; la distanza di edit, già spiegata altrove, è richiamata e non reintrodotta.
- {doc}`Sintesi vocale </SpeechRecognition/sintesi-vocale>` (Speech Recognition). **Frequenza di campionamento del generatore d'onda.** Il modello che produce un campione alla volta ne genera sedicimila al secondo, come nel suo articolo e nel capitolo sull'audio; i ventiquattromila appartengono a un altro modello.
- {doc}`Sintesi vocale </SpeechRecognition/sintesi-vocale>` (Speech Recognition). **Confronto fra punteggi e dilatazione delle convoluzioni.** I ventotto centesimi di scarto confrontano raccolte, voci e frequenze diverse e non sono una differenza valida; la dilatazione raddoppia fino a un limite, poi ricomincia.
- {doc}`Speech Recognition </SpeechRecognition/overview>`. **Punteggio acustico e linguistico.** Le due valutazioni si moltiplicano e non si compongono in un compromesso: basta che una scarti una frase perché la frase esca di gara.
- {doc}`Graph Neural Network </GraphNeuralNetwork/overview>`. **Antibiotico del 2020 e ponti di Königsberg.** Il primo firmatario del lavoro sull'antibiotico è Jonathan Stokes, nel gruppo di James Collins e Regina Barzilay; la memoria di Euler fu presentata nel 1735 e stampata sei anni dopo.
- {doc}`GraphSAGE, GAT e applicazioni </GraphNeuralNetwork/architetture-applicazioni>` (Graph Neural Network). **Grafo delle strade nella stima dei tempi.** Due segmenti sono collegati anche, e per primo, quando si susseguono sulla stessa strada, non soltanto agli incroci.
- {doc}`GraphSAGE, GAT e applicazioni </GraphNeuralNetwork/architetture-applicazioni>` (Graph Neural Network). **Garanzia di iniettività della rete di potere distintivo massimo.** Lo scalare appreso della variante più diffusa è fissato a zero, valore che il corollario non copre: è una scelta sperimentale e non una conseguenza del teorema.
- {doc}`Message passing </GraphNeuralNetwork/message-passing>` (Graph Neural Network). **Costo del filtro polinomiale sul grafo.** Costa un prodotto fra matrice sparsa e vettore per ogni grado, quindi cresce col grado; la forma precedente vale solo per polinomi di primo grado.
- {doc}`Message passing </GraphNeuralNetwork/message-passing>` (Graph Neural Network). **Glosse sulla normalizzazione per i gradi.** La configurazione più liscia assegna a ciascun nodo un valore crescente con i suoi vicini, e su una stella il centro vale più del doppio di una punta.
- {doc}`Message passing </GraphNeuralNetwork/message-passing>` (Graph Neural Network). **Massimo come terzo modo di aggregare i messaggi.** Somma, media e massimo sono ora tutti trattati nella scena, in accordo con il riquadro riepilogativo.
- {doc}`I knowledge graph </GraphNeuralNetwork/knowledge-graph>` (Graph Neural Network). **Conto dei fatti nei grafi di conoscenza.** Eliminato dal testo il numero «tre», in contrasto con i tre fatti diversi elencati nella scheda avanzata; gli esempi restano.

(v1-12-1)=

## 1.12.1 · 31 agosto 2026

### Patch

- **Revisione di rimandi, glosse e numeri.** Rimandi portati al bersaglio giusto, glosse riallineate alle formule che commentano e numeri verificati sui blocchi che li stampano; diradate le contrapposizioni in Matematica e nei sistemi multi-agente.
- {doc}`L'addestramento avversario </GAN/come-funziona>` (GAN). **Discriminatore giocattolo nelle GAN.** Il gradiente è letto per ciò che indica; la conversione fra perdita e frequenza dà un minimo e non un massimo; raddrizzato un verso di metrica.
- {doc}`Comprimere e ricostruire </ModelliLatenti/comprimere-e-ricostruire>` (Modelli latenti). **Affermazioni su autoencoder e compressione.** La non linearità serve nel decoder e non da entrambe le parti; il rapporto di compressione è reso coerente (cinquanta e quarantotto); la PCA probabilistica ha la media.
- {doc}`A che serve saperlo </VerosimiglianzaEsatta/a-che-serve>` (Verosimiglianza esatta). **Citazione nella pagina sulla verosimiglianza esatta.** Le virgolette omettevano il soggetto, la densità appresa dai modelli a flusso; la citazione è ripresa dal testo originale.
- {doc}`Mondi in miniatura </WorldModels/mondi-in-miniatura>` (World Model). **Temperatura e punteggio nei world model.** Alla temperatura più alta l'agente impara meno bene, non smette di imparare: il punteggio resta sopra la soglia, ben oltre una politica casuale, e a cambiare è la deviazione standard.

(v1-12-0)=

## 1.12.0 · 30 agosto 2026

### Minor

- {doc}`Imparare a imparare </DeepLearning/meta-apprendimento>` (Deep Learning). **Meta-apprendimento.** Nuova sezione: scegliere una posizione di partenza da cui pochi passi bastano su un compito mai visto, con il conto che lo dimostra.
- {doc}`Gerarchia e opzioni </DeepReinforcementLearning/gerarchia>` (Deep Reinforcement Learning). **Apprendimento per rinforzo gerarchico.** Nuova sezione sulle opzioni, sulla scala di tempo delle decisioni e sulla ragione per cui scegliere meno spesso può far imparare di più.
- {doc}`Far funzionare le reti profonde </DeepLearning/ottimizzazione-regolarizzazione>` (Deep Learning). **Spiegazione di concetti prima solo accennati.** Label smoothing, Thompson sampling, teacher forcing ed exposure bias, test d'ipotesi e p-value con le correzioni per confronti multipli, nelle sezioni in cui servono.
- {doc}`Componenti e modelli classici </SerieTemporali/componenti-e-classici>` (Serie temporali). **Concetti aggiunti nelle serie temporali e altrove.** L'orizzonte di un MA(q), la passeggiata aleatoria, ROUGE accanto a BLEU, il curriculum e il self-play negli agenti che imparano insieme, l'ordinamento per perdita fra le metriche.
- {doc}`Algebra lineare </Matematica/algebra-lineare>` (Matematica). **Presupposti di matematica spiegati.** Decomposizione ai valori singolari e rango basso, calibrazione con temperature scaling, filtro di Kalman, modelli lineari generalizzati: regressione lineare, logistica e di Poisson sono la stessa macchina.
- **Rimandi, formule e termini corretti in più capitoli.** Rimandi a destinazioni inesistenti, termini usati prima di essere spiegati, due formule errate e un'affermazione vera in aritmetica esatta ma falsa eseguendo la pagina.

(v1-11-0)=

## 1.11.0 · 30 agosto 2026

### Minor

- {doc}`Sistemi lineari </Matematica/sistemi-lineari>` (Matematica). **Cinque sezioni nuove nel capitolo di matematica.** Sistemi lineari, proiezioni e minimi quadrati, determinante come volume, disuguaglianze indipendenti dalla distribuzione e catene di Markov; il capitolo arriva a undici sezioni.
- {doc}`Il limite continuo </ModelliDiffusione/sde-e-ode>` (Modelli di diffusione). **Diffusione: dal limite continuo alla generazione in un passo.** Sei sezioni nuove su equazioni differenziali, flow matching, campionatori veloci, generatori a un passo, condizionamento e diffusione su stati discreti.
- {doc}`Guida e allineamento </ModelliDiffusione/guida>` (Modelli di diffusione). **Forza della guida nella diffusione.** Alzarla non equivale a campionare da una distribuzione inclinata: sul banco di prova esatto i campioni se ne discostano, e il divario si concentra a metà percorso.
- **Diradamento della costruzione «non è X, è Y».** In tutto il libro la contrapposizione resta dove mette in guardia da un errore reale e si riduce altrove.

(v1-10-5)=

## 1.10.5 · 25 agosto 2026

### Patch

- {doc}`Analisi e ottimizzazione </Matematica/analisi-ottimizzazione>` (Matematica). **Tre correzioni nel capitolo di matematica.** La sigmoide non restituisce probabilità con somma uno, la softmax sì; un punto di sella richiede almeno due dimensioni; il gradiente taglia ad angolo retto le curve di livello.
- {doc}`Teoria dell'informazione </Matematica/teoria-informazione>` (Matematica). **Bit di un dado e numero di domande.** Le domande sono intere: su un tiro la strategia migliore ne usa in media due e due terzi, e solo su molti tiri si raggiungono i 2,585 bit.
- {doc}`Probabilità e statistica </Matematica/probabilita-statistica>` (Matematica). **Teorema del limite centrale e intervalli di confidenza.** Esplicitata l'ipotesi, nessun contributo enormemente più grande degli altri messi insieme; nella figura il terzo modello dista dagli altri due più dell'incertezza della misura.
- {doc}`La matematica di un LLM </Matematica/matematica-llm>` (Matematica). **Specializzazione delle teste di attenzione.** Che copie ridondanti costino accuratezza è un'ipotesi e non un risultato: da un modello addestrato se ne può togliere una quota grande senza perdite misurabili, come mostrano i lavori citati.
- **Simboli e figure nel capitolo di matematica.** I simboli presenti solo nella scheda avanzata sono sciolti nelle didascalie; in due figure una scritta non esce più dal margine né ne copre un'altra.
- **Riquadri e blocchi di codice in stampa.** Nessuna cornice vuota in fondo alla pagina; i blocchi di codice spezzati lo indicano; i tre fermi immagine delle animazioni sono dichiarati tre istanti dello stesso movimento.

(v1-10-4)=

## 1.10.4 · 25 agosto 2026

### Patch

- **Codice copiabile nel PDF.** Le coppie di simboli non si uniscono più in un solo segno: `!=`, `>=` e `->` restano tali in pagina e nel testo selezionato.
- **Distinzione dei blocchi da eseguire e da leggere.** Gli esempi con nomi di comodo, file inesistenti o modelli costruiti altrove lo dichiarano, online e in stampa.
- {doc}`Le basi di Python </Python/basi>` (Python). **Presupposti di programmazione in Python.** Scrivere in una tabella, ricerca in un dizionario e in una lista, copia di una lista e lettura di un messaggio d'errore sono ora in entrambi i livelli.
- {doc}`Python </Python/overview>`. **Velocità di NumPy.** Il guadagno non viene dal contenitore ma dal costo che Python paga a ogni giro di un ciclo, pagato una volta sola sul blocco intero.
- {doc}`NumPy </Python/numpy>` (Python). **Riquadri di ripasso in NumPy.** La regola del broadcasting è enunciata in accordo con l'esempio; PyTorch non è più in cima alla torre delle librerie; aggiunti mutabilità, viste di NumPy e Matplotlib.
- **Numeri e fonti nel capitolo di Python.** La media di gruppo ignora i valori mancanti nel divisore; riferimento a Rubin per i dati mancanti; costo del Python senza lucchetto dalla documentazione ufficiale; PyTorch del 2016.
- {doc}`L'addestramento avversario </GAN/come-funziona>` (GAN). **Derivazione del valore di una GAN in stampa.** La formula che riscrive il secondo integrale nello spazio dei dati va a capo sull'uguale e non esce più dal margine.

(v1-10-3)=

## 1.10.3 · 24 agosto 2026

### Patch

- {doc}`Introduzione </Introduzione/overview>`. **Definizione di intelligenza artificiale.** Riguarda i compiti per cui nessuno sa scrivere una ricetta valida nel mondo reale: esclude la lavatrice, include i sistemi esperti, il cui fallimento diventa la premessa del capitolo.
- {doc}`Introduzione </Introduzione/overview>`. **Riordino dell'Introduzione.** Il blocco filosofico e le tre discipline sulla decisione precedono Turing; la funzione obiettivo ha una sezione propria; ARGO è motivata dall'assenza di apprendimento al suo interno.
- **Rinvii con destinazione esplicita.** Nell'Introduzione il rimando sui Transformer è un collegamento, in World Model il collasso rimanda alla sezione sulle JEPA, in Ricerca e pianificazione la promessa è stata tolta.
- **Licenza nei PDF per capitolo.** Ogni file per capitolo riporta a piè di pagina le condizioni d'uso, che il libro intero scrive nel colophon.
- **Comando di esecuzione nelle pagine notebook.** Il comando che esegue il codice è fra i pulsanti in cima alla pagina; da telefono sta accanto al marchio.

(v1-10-2)=

## 1.10.2 · 23 agosto 2026

### Patch

- {doc}`Prefazione </prefazione>`. **Posizione del libro nella prefazione.** La prefazione dichiara la scelta del libro: meno fiducia nelle istruzioni scritte nel prompt, più architetture, iperparametri e funzioni di costo.
- {doc}`Prefazione </prefazione>`. **Prefazione più concisa e riconoscimento di un debito.** Il segno del libro nomina «Gödel, Escher, Bach», da cui vengono la lettura della figura come anello che torna su sé stesso e l'idea di spiegare ogni cosa due volte.
- **Fonti degli strumenti nominati.** PyTorch, TensorFlow, Keras e Hugging Face compaiono in bibliografia con l'articolo degli autori e un indirizzo di approfondimento.

(v1-10-1)=

## 1.10.1 · 22 agosto 2026

### Patch

- **Corrispondenza fra i due livelli.** Nelle scene del livello Elementare sono riportate le mosse presenti solo nel Superiore: condizione di validità di una garanzia, caso di rottura del metodo, conto commentato dal testo.
- **Errori di merito corretti.** L'indice di Davies-Bouldin si minimizza; la perplessità scende con frasi più prevedibili; la proiezione non è il prodotto scalare; `top_p` sceglie quanta classifica resta.
- **Ipotesi esplicitate accanto alle formule.** Una mistura di due gaussiane è bimodale solo se i centri distano più di due deviazioni standard; la simmetria di Shapley chiede meno; l'equivarianza nei grafi richiede l'indipendenza dall'ordine.
- {doc}`AI responsabile </AIResponsabile/overview>`. **Paradosso dell'equità al primo livello.** Nel caso COMPAS le due parti hanno ragione entrambe: il conflitto deriva dalle frequenze diverse nei due gruppi, e la scena del tribunale arriva fin qui.
- {doc}`Generare suono e musica </Audio/generazione-audio>` (Audio oltre la voce). **Capoverso e titolo ripristinati in Audio e Ricerca.** In generazione audio il paragrafo su Jukebox e AudioLM torna prosa dopo essere stato impaginato come codice; in ricerca ad albero un titolo di sezione ha il livello corretto.
- **Notazione: una lettera, un significato.** Rinominati i simboli con doppio uso nella stessa pagina: sequenza di token e vettore continuo, latenza e durata, modello bozza e intero quantizzato, precisione e probabilità.

(v1-10-0)=

## 1.10.0 · 20 agosto 2026

### Minor

- {doc}`Il kernel trick </MachineLearning/svm-kernel>` (Machine Learning). **Kernel trick in una pagina propria.** La sezione sulla Support Vector Machine è divisa in due: il margine massimo da una parte; kernel trick, regressione, rilevamento di anomalie e uso con scikit-learn dall'altra.
- {doc}`MCTS e AlphaGo </DeepReinforcementLearning/mcts-alphago>` (Deep Reinforcement Learning). **Ricerca ad albero Monte Carlo, AlphaGo e allineamento dei modelli linguistici.** Trattati in una pagina propria, separata dal gradiente di policy, che ora si conclude con PPO e con i riquadri finali.
- {doc}`Oltre il BPE </NaturalLanguageProcessing/oltre-il-bpe>` (Natural Language Processing). **Oltre il BPE.** Nuova pagina su WordPiece, SentencePiece e tokenizzazione a byte, con quattro conseguenze pratiche: numeri spezzati, costo in token dell'italiano, spazio superfluo, vocabolario congelato. Il BPE resta nella pagina d'origine.
- {doc}`Le basi di Python </Python/basi>` (Python). **Lettura dei traceback e gestione delle eccezioni.** Le basi di Python spiegano come leggere un errore e come proteggere la riga fragile con un `try`/`except` mirato, secondo lo stile del chiedere scusa anziché del chiedere permesso.
- **Cinque figure nuove.** Apprendimento contrastivo, spettrogramma dell'AST, parte mascherata di wav2vec, generazione audio a token; AlexNet ridisegnata con i due rami su GPU che comunicano soltanto in alcuni punti.

(v1-9-4)=

## 1.9.4 · 20 agosto 2026

### Patch

- {doc}`Deep Q-Network (DQN) </DeepReinforcementLearning/dqn>` (Deep Reinforcement Learning). **Replay con priorità e rete a due rami in DQN.** Aggiunta la sezione sulle due migliorie, con formule e fonti, la correzione dei pesi di campionamento e il richiamo a Rainbow, che le combina.
- {doc}`Prefazione </prefazione>`. **Triangolo impossibile nella Prefazione.** Accanto al passaggio sul segno del libro è inserita l'illustrazione grande dei suoi tre tracciati, da osservare un angolo alla volta.
- **Rifiniture di lettura.** Nelle Conclusioni ogni capitolo citato è collegato con un link alla prima occorrenza soltanto; in due aperture di «Visione e linguaggio» si entra direttamente nella scena.

(v1-9-3)=

## 1.9.3 · 20 agosto 2026

### Patch

- **Formule aggiunte.** Equazione di Bellman per Q, RoPE e RMSNorm, word2vec e GloVe, gradiente di policy con GAE, riparametrizzazione di SAC, MMR e PSI sono ora scritti per esteso.
- {doc}`Detection e segmentazione </VisioneArtificiale/detection-segmentazione>` (Visione artificiale). **Ancore, IoU e famiglia YOLO.** Precisato che il confronto con il riquadro vero esiste solo in addestramento e che in inferenza la IoU serve a eliminare i doppioni; aggiunta la storia di YOLO, con codice.
- {doc}`Prefazione </prefazione>`. **Correzioni in Prefazione, Conclusioni, Python, Ricerca e Visione e linguaggio.** Nella Prefazione la citazione di Anthropic rimanda al resoconto corretto e il teorema di Penrose ha due voci bibliografiche; i conti dei tre rapporti usano numeri verificati.
- {doc}`Conclusioni </Conclusioni/overview>`. **Ripasso finale delle Conclusioni.** Allineato al libro attuale: cinque famiglie generative, capitoli recenti citati con link, rimando ad AlphaGo, coerenza fra i livelli sugli errori di un agente, riquadri prima del congedo.
- {doc}`Esplorare lo spazio </Ricerca/esplorare-lo-spazio>` (Ricerca e pianificazione). **Parità dell'otto-tessere e attribuzioni nella Ricerca.** Corretto l'argomento di parità, che dimostrava il contrario di quanto affermava; indicati gli autori di IDA* e Monte Carlo Go, il punteggio di Deep Blue, il nome della potatura.
- {doc}`Python </Python/overview>`. **Ambiente virtuale su Ubuntu e storia dei framework.** Indicato il componente mancante per il primo comando, l'uscita dall'ambiente e gli altri gestori; ripristinate le date (Theano precede AlexNet); la tabella delle vendite è generata dalla pagina.
- {doc}`Allineare due spazi </VisioneLinguaggio/allineare-due-spazi>` (Visione e linguaggio). **Precisione dei risultati e rimandi in «Allineare due spazi».** Il divario fra i due gruppi è riportato alla precisione che il protocollo sostiene; corretti i rimandi e il conto di SigLIP, sedici domande al posto di otto interrogazioni.
- **Numeri commentati stampati dal codice.** Escono ora da blocchi della pagina il conto del weight decay, il sorpasso dei banditi, i decimali del giudice di recensioni e il confronto della potatura con venti ordini casuali.

(v1-9-2)=

## 1.9.2 · 20 agosto 2026

### Patch

- **Aperture delle schede Elementari.** Meno inviti a immaginare: le schede entrano più spesso direttamente nella scena, e le analogie restano.
- **Immagine della catena di montaggio.** Sostituita dove era soltanto un modo di dire «una cosa dopo l'altra» da immagini scelte per il passaggio; resta dove regge, come nel capitolo sulla GPU e nella retropropagazione.

(v1-9-1)=

## 1.9.1 · 20 agosto 2026

### Patch

- **Diciassette voci bibliografiche.** Aggiunte le fonti di lavori già nominati nel testo, tra cui la policy iteration di Howard, TRPO, DAgger, DistMult, ComplEx e RotatE.
- **Correzioni di formule.** Risoluzione della trasformata a finestra, matrice condivisa della GAT, perdita del VQ-VAE, costo del calcolo a blocchi di Mamba-2, reticolo del trasduttore, penalità di gradiente delle GAN.
- **Numeri ricalcolati.** Ripartizione dei fallimenti della rete di Hopfield, punteggi di AST e V-JEPA 2, MOS di Tacotron 2, accuratezza di Hearsay-II, banda di previsione in «Forecasting neurale», ora stampata dal codice.
- **Coerenza fra schede Elementari e Superiori.** Riallineate le schede in «Verosimiglianza esatta», «Modelli latenti» e «Costo del coordinamento»; l'apertura delle PINN non attribuisce più l'idea a un lavoro che il capitolo fa risalire al 1994.
- **Rimandi fra capitoli come link.** I richiami a capitoli e sezioni, per esempio alla backpropagation e agli ensemble, sono ora collegamenti diretti.
- **Revisione della prosa.** Ridotti i grassetti e le contrapposizioni del tipo «non è X, è Y»; le schede introducono direttamente la scena invece di annunciarla.

(v1-9-0)=

## 1.9.0 · 19 agosto 2026

### Minor

- {doc}`Capire è accorciare </AutoSupervisione/capire-e-accorciare>` (Auto-supervisione). **Compressione e comprensione nell'auto-supervisione.** Nuova sezione sul perché prevedere bene richieda accorciare: complessità di Kolmogorov, rasoio di Occam in bit, argomento di Sutskever, misure di verifica, costo del vocabolario; con esperimento e figura animata.
- {doc}`Capire è accorciare </AutoSupervisione/capire-e-accorciare>` (Auto-supervisione). **Correzioni nella sezione sulla compressione.** Il 99% di iGPT su CIFAR-10 riguarda la rete rifinita, non il sondaggio a rete congelata (96,3%); corretti anche verosimiglianza, argomento di Sutskever e una frase autocontraddittoria.
- **Etichette delle figure.** In tre figure le etichette nere sono sostituite dal colore della curva corrispondente; in quella sulla compressione erano l'unica legenda.

(v1-8-5)=

## 1.8.5 · 19 agosto 2026

### Patch

- **Ricerca con testo già presente nel campo.** Corretto il caso in cui la finestra, aperta da un vecchio collegamento o al ritorno con il browser, non mostrava risultati: cerca ora subito il testo già scritto.

(v1-8-4)=

## 1.8.4 · 19 agosto 2026

### Patch

- **Evidenziazione dei risultati di ricerca.** L'avviso «Nascondi i risultati della ricerca» compare solo se qualcosa è stato evidenziato, con il numero di corrispondenze; corretta l'evidenziazione, che sulle versioni recenti del tema non partiva.
- **Ricerca incrementale.** I risultati compaiono a ogni tasto, raggruppati per pagina e capitolo, e portano alla singola sezione; apertura con Ctrl+K o ⌘K, navigazione da tastiera, termini evidenziati, suggerimento sui refusi.

(v1-8-3)=

## 1.8.3 · 19 agosto 2026

### Patch

- {doc}`Analisi numerica </Matematica/analisi-numerica>` (Matematica). **Riproducibilità numerica fra calcolatori.** Corretta l'affermazione che due variabili d'ambiente rendano identiche le cifre: non vale per i prodotti fra matrici di PyTorch, quindi esecuzioni su macchine diverse si confrontano con una tolleranza.

(v1-8-2)=

## 1.8.2 · 19 agosto 2026

### Patch

- {doc}`Python </Python/overview>`. **Definizione di SIMD.** Sciolta la sigla nel paragrafo sulla velocità di NumPy: una sola istruzione del processore che opera su più numeri insieme, quanti ne entrano nei registri.
- {doc}`Analisi numerica </Matematica/analisi-numerica>` (Matematica). **Prerequisiti nella sezione sui due calcolatori.** Definiti SSE2, AVX2 e AVX-512 (registri da 128, 256 e 512 bit); tolto il riferimento anticipato a PyTorch; uniformata la precisione dei conti e alleggerita la chiusura.

(v1-8-1)=

## 1.8.1 · 19 agosto 2026

### Patch

- {doc}`Analisi numerica </Matematica/analisi-numerica>` (Matematica). **Risultati diversi dallo stesso codice su due calcolatori.** Nuova sezione: l'addizione in virgola mobile non è associativa e il kernel scelto dalla libreria fissa l'ordine delle somme; con esempio, definizione di kernel e variabili per fissare la scelta.
- {doc}`Meno pesi </Efficienza/meno-pesi>` (Efficienza). **Potatura e quantizzazione rimisurate a kernel fissato.** Comporre le due leve costa 1,1 punti di accuratezza (prima 1,7); la rete potata del 90% ne perde meno di uno. La conclusione non cambia.
- **Verifica dei numeri stampati nel repository pubblico.** Il confronto fra i numeri delle pagine e quelli prodotti dal codice gira ora anche sulla copia pubblicata del libro.

(v1-8-0)=

## 1.8.0 · 18 agosto 2026

### Minor

- {doc}`Spline e modelli additivi </MachineLearning/curve-al-posto-di-rette>` (Machine Learning). **Regressione spline e modelli additivi.** Nuova sezione: limiti del polinomio unico, spline con flessibilità localizzata e parametro di regolarizzazione, modelli additivi con una curva per colonna.
- {doc}`Il bootstrap </MachineLearning/il-bootstrap>` (Machine Learning). **Il bootstrap.** Nuova sezione sul metodo di Efron (1979) per stimare l'incertezza di una misura da un solo campione, e sui casi in cui fallisce: massimo, dati non indipendenti, campione raccolto male.
- {doc}`Modelli generativi </MachineLearning/modelli-generativi>` (Machine Learning). **Modelli generativi per la classificazione.** Nuova sezione su analisi discriminante lineare e quadratica e naive Bayes, con il confronto fra approccio generativo e discriminativo sui punti lontani dai dati osservati.
- {doc}`Valutare un raggruppamento </MachineLearning/valutare-un-raggruppamento>` (Machine Learning). **Valutazione di un raggruppamento.** Misure di qualità del clustering: l'indice più usato può preferire la risposta sbagliata, l'indice di Rand grezzo dà 0,9 al caso; un teorema del 2002 impone di dichiarare la scelta.
- {doc}`Alberi e metodi ensemble </MachineLearning/alberi-ensemble>` (Machine Learning). **Uscite dei blocchi nel capitolo di Machine Learning.** Corretti due blocchi che misuravano i dati di un'altra pagina e un rilevatore di deriva che non segnalava scostamenti reali.
- **Correzioni alle sezioni nuove.** Corretti l'argomento della dimostrazione di un teorema, la provenienza non omogenea del più noto insieme di dati della statistica, la conclusione di un esperimento sulla copertura e un'animazione.

(v1-7-0)=

## 1.7.0 · 18 agosto 2026

### Minor

- {doc}`Efficienza </Efficienza/overview>`. **Efficienza dei modelli.** Nuovo capitolo sulle tre leve per ridurre un modello (meno bit, meno pesi, distillazione) e sul loro costo misurato, che non sempre coincide con l'attesa.
- {doc}`Ricerca e pianificazione </Ricerca/overview>`. **Ricerca prima dell'apprendimento.** Nuovo capitolo su albero dei futuri, stima della distanza dall'obiettivo e potatura dei rami, con le tre ipotesi che nel mondo reale spesso mancano e da cui muove il rinforzo.
- {doc}`Modelli latenti </ModelliLatenti/overview>`. **Modelli a variabili latenti.** Nuovo capitolo: compressione con ricostruzione, rimedio ai buchi dello spazio latente (distribuzioni al posto di punti), applicazioni a codec audio, rinforzo, diffusione e world model.
- **Rimandi e spiegazioni uniche dopo l'inserimento di tre capitoli.** Aggiornati numeri e rimandi, ora per nome del capitolo; quantizzazione, potatura e biglietto della lotteria hanno una sola spiegazione, richiamata altrove.
- **Correzioni nei capitoli nuovi.** Corretti la spiegazione elementare di A*, un conto di efficienza difforme da quello sulla GPU, un miglioramento dovuto al riavvio dell'ottimizzatore e non alla potatura, una frase imprecisa nella fonte.

(v1-6-3)=

## 1.6.3 · 17 agosto 2026

### Patch

- **Spiegazioni elementari corrette in auto-supervisione e modelli a energia.** Il collasso è fermato dalla coerenza fra due ritagli della stessa foto, non dal richiedere di non ripetersi; la descrizione locale del paesaggio energetico lascia indeterminata un'altezza per valle.
- **Allineamento fra livello Elementare e Superiore.** Corretti, fra gli altri, il passo di tempo dei modelli a spazio di stato, l'attenzione lineare, l'allineamento diretto e la verosimiglianza «esatta» di un flusso.
- {doc}`Prefazione </prefazione>`. **Chiusa della Prefazione.** Ripristinato il nesso fra le ultime due frasi e accorciato il paragrafo sul triangolo impossibile, che conserva il teorema e i tre isomorfismi che non si compongono.

(v1-6-2)=

## 1.6.2 · 17 agosto 2026

### Patch

- {doc}`Prefazione </prefazione>`. **Il triangolo impossibile come segno del libro.** La Prefazione ne spiega la ragione: l'impossibilità è un teorema, i tre pezzi sono corretti e il difetto sta nella loro composizione, non in un luogo preciso.
- **Ultime formule della prima metà e cinque numeri.** Corretti la precisione media (la definizione elementare premiava un rilevatore quasi cieco), la ripartizione «due terzi e un terzo» in una rete e tre numeri di un confronto fra inizializzazioni.

(v1-6-1)=

## 1.6.1 · 17 agosto 2026

### Patch

- {doc}`Generare suono e musica </Audio/generazione-audio>` (Audio oltre la voce). **Figura animata sulla rappresentazione dello spartito.** In «Generazione audio» mostra la griglia che cresce a ogni sedicesimo anche senza eventi, mentre la rappresentazione a eventi resta ferma e poi salta.
- **Congedi di capitolo.** I capitoli non terminano più sull'elenco «Da ricordare»: chiude un breve raccordo che dice che cosa portare con sé e quale capitolo segue.
- **Simboli e glosse nelle formule.** Presentati i simboli usati senza definizione, distinte le lettere con due significati nello stesso capitolo, corretta una F1 di esempio calcolata con numeri di tabelle diverse.
- **Coerenza fra i livelli di lettura.** Riscritti i passaggi in cui la scheda Elementare affermava ciò che la Superiore smentiva, come la convergenza della discesa del gradiente da qualsiasi punto e l'assestamento graduale del percettrone.
- **Apertura di «Visione e linguaggio».** Il capitolo non ripete più l'aneddoto del 1966 di «Visione artificiale» e si apre con la pipa di Magritte (1929): il disegno non è la cosa.

(v1-6-0)=

## 1.6.0 · 16 agosto 2026

### Minor

- {doc}`Auto-supervisione </AutoSupervisione/overview>`. **Apprendimento auto-supervisionato.** Nuovo capitolo: i quattro modi di costruire il segnale senza etichette, il guasto comune (rispondere sempre la stessa cosa), i rimedi e il dibattito sul peso del rinforzo.
- {doc}`Verosimiglianza esatta </VerosimiglianzaEsatta/overview>`. **Modelli a verosimiglianza esatta.** Nuovo capitolo sulle famiglie, autoregressive e a flussi, che assegnano un numero alla probabilità di un dato: costo e caso in cui probabilità alta e tipicità divergono.
- {doc}`L'inferenza attiva </WorldModels/inferenza-attiva>` (World Model). **Inferenza attiva.** Sezione sull'agente che agisce per ridurre la propria sorpresa, alternativa di principio al rinforzo; il termine che nell'allineamento tiene un modello vicino a quello di partenza risulta formalmente lo stesso.
- {doc}`Evoluzioni e applicazioni </GAN/applicazioni-evoluzioni>` (GAN). **StyleGAN e quantizzazione delle immagini.** Aggiunti la separazione fra stile e contenuto, l'artefatto a goccia rimasto un anno senza spiegazione e la quantizzazione in un vocabolario di simboli, ponte verso i modelli di linguaggio.
- {doc}`Generare suono e musica </Audio/generazione-audio>` (Audio oltre la voce). **Generazione dello spartito.** Confronto fra la rappresentazione a note e durate (venti kilobyte al minuto) e la registrazione (trenta megabyte): che cosa si guadagna, che cosa si perde, effetto della codifica sul tempo.
- **Correzioni di numeri e formule.** Corretti, fra gli altri, un rapporto errato di cento volte, una memoria sovrastimata di otto volte e un fattore invertito in una formula, giusta nella scheda Superiore e rovesciata nell'Elementare.
- **Apostrofi nella composizione del PDF.** Il compositore rendeva l'apostrofo a fine parola come una virgoletta doppia («un po” meno»); corretti in tutto il libro.
- **Aperture e congedi dei capitoli.** Citazione di Galileo sostituita con il passo del Saggiatore; matematica apre con von Neumann e Shannon; corretta l'affermazione sugli agenti; i congedi sostituiscono gli elenchi finali.

(v1-5-8)=

## 1.5.8 · 15 agosto 2026

### Patch

- {doc}`Conclusioni </Conclusioni/overview>`. **Il vantaggio delle macchine nel mondo digitale.** Nuova sezione delle Conclusioni: la superiorità deriva dal terreno e non dall'intelligenza; restano all'uomo la responsabilità della scelta e la decisione su quale partita giocare.
- {doc}`Introduzione </Introduzione/overview>`. **I dati come scarto.** L'Introduzione presenta i dati come sottoprodotto e non come giacimento, con il parallelo dell'ossigeno, rifiuto di certi batteri che prima di diventare respiro fu un veleno.
- **Descrizione uniforme del libro.** Copertina del PDF, anteprima di condivisione e titolo delle pagine riportano la stessa frase, prima in tre versioni; dalla home il libro si apre anche nel browser.

(v1-5-7)=

## 1.5.7 · 15 agosto 2026

### Patch

- **Lettura nel browser e DOI.** Accanto al download del PDF un secondo pulsante apre il libro pagina per pagina; il DOI, identificativo permanente, rimanda all'ultima versione depositata e fornisce la scheda per le citazioni.
- **Riga descrittiva in copertina.** Oltre alla postilla compare la riga che dice che cos'è il libro, la stessa della home e dell'anteprima dei link.

(v1-5-6)=

## 1.5.6 · 15 agosto 2026

### Patch

- **Grafia del nome del libro.** Il titolo si scrive Paithon Book, con le iniziali maiuscole; il logo resta minuscolo perché è un segno disegnato.
- **Logo con il triangolo al posto della «a».** Allineato il logo del libro a quello del sito; rigenerato il file in copertina al PDF, una copia fatta a mano e mai aggiornata.
- **Titolarità del copyright e licenza.** Nel colophon del PDF il copyright è intestato a Francesco Messina e non al dominio paithon.it; la licenza del codice è indicata per esteso, Apache License 2.0.

(v1-5-5)=

## 1.5.5 · 15 agosto 2026

### Patch

- **Citazioni nelle didascalie del PDF.** In tre didascalie la citazione spezzava il rimando alla bibliografia nella versione a stampa; ora sta nel testo, e in un caso indica autori e anno degli insiemi di dati.

(v1-5-4)=

## 1.5.4 · 15 agosto 2026

### Patch

- {doc}`Prefazione </prefazione>`. **Perché il libro esiste oggi.** Nella Prefazione la ragione di oggi: la ripetizione irriflessiva di frasi e l'adozione di modelli senza leggerne la licenza. Chi coniò «pappagalli stocastici» nota: «la frase ha superato il paper».
- **Leggibilità dei capitoli più tecnici.** Riscritti i passaggi più densi degli ultimi capitoli; un conto presentato come eseguibile «in due righe» è ora svolto per intero, e un'imprecisione matematica è corretta.
- **Correzioni di fatto.** L'inganno del cavallo che sapeva contare è del 1904 (il 1907 è l'anno del libro che lo racconta); corretti il fischietto delle telefonate gratuite e un'ipotesi del teorema sull'equità.
- **Percentuali nelle metriche di servizio.** Le due percentuali che reggevano un paragrafo sono ora prodotte dal codice con una colonna aggiunta; una delle due dipendeva da un sorteggio fortunato.
- **Costo dell'attenzione nelle Conclusioni.** Corretta la contraddizione con il capitolo sui Transformer: il costo dell'attenzione supera quello del resto della rete sei volte più in là di quanto indicato.

(v1-5-3)=

## 1.5.3 · 14 agosto 2026

### Patch

- **Leggibilità in altri capitoli.** Riformulati i passaggi più oscuri per chi parte senza basi, fra cui quelli su GAN, reinforcement learning e attenzione lineare.
- **Autonomia del livello Elementare.** Le pagine del percorso elementare non rimandano più a nozioni spiegate soltanto nella scheda Superiore; in un capitolo simboli e formule escono dal testo comune.
- **Fatti e numeri corretti.** Rimossa una citazione attribuita agli autori di HyDE; corretti due numeri, le operatrici del Voder (ventiquattro, non venti) e una frase del 1939 senza riscontro; completato un protocollo sperimentale.
- **Figure in contraddizione con il testo.** In due disegni sulle GAN la freccia indicava il ritorno del verdetto, contro quanto spiega il capitolo; un grafico in scala logaritmica ha riavuto l'asse.

(v1-5-2)=

## 1.5.2 · 14 agosto 2026

### Patch

- **Tre figure nuove.** Le quattro mosse che amministrano la finestra di contesto, le tre forme di una rete multi-compito e il righello che traduce l'entropia nel numero di facce di un dado equivalente.
- **Correzioni di sostanza in tutti i capitoli.** Fra le altre: l'asimmetria del clipping di PPO, la derivazione del gradiente della funzione di partizione, la composizione in TransE, il minimo di sei frame per «PALLA».
- **Leggibilità dei capitoli d'apertura.** Riformulati i passaggi difficili da seguire senza basi, come un conto con il seno e gli alberi usati prima di essere definiti.
- **Figure allineate alle didascalie.** Corretti i disegni difformi dal testo: una griglia detta quasi vuota ma piena al 72%, una scala senza tacche, una freccia delle GAN contraria alla pagina.
- {doc}`Allineamento e governance </AIResponsabile/allineamento-e-governance>` (AI responsabile). **AI Act: pratiche vietate.** Aggiornato l'elenco: dal 2 dicembre 2026 il regolamento europeo del 24 luglio 2026 vieta anche i sistemi che generano materiale intimo non consensuale e materiale di abuso sessuale su minori.
- **Disegno in copertina del PDF.** Sotto il titolo compare una conca con curve di livello e la traiettoria calcolata della discesa, che richiama il gesto del gradiente.

(v1-5-1)=

## 1.5.1 · 13 agosto 2026

### Patch

- {doc}`Introduzione </Introduzione/overview>`. **Due ingressi dell'intelligenza artificiale nella vita quotidiana.** L'Introduzione distingue gli anni Dieci, in cui è entrata inosservata (traduttori, suggerimenti, volti), dal novembre 2022, quando con ChatGPT è diventata interlocutore; il salto era cominciato cinque anni prima.
- {doc}`Introduzione </Introduzione/overview>`. **Terminologia: inverni.** Le due pause dell'AI sono chiamate «inverni», come nel capitolo sulle reti neurali, e non più «gelate».

(v1-5-0)=

## 1.5.0 · 13 agosto 2026

### Minor

- **Versione PDF.** Il libro è scaricabile in un unico PDF impaginato come volume, con indice, numeri di pagina e aperture di capitolo; il link nella pagina di apertura punta all'ultima edizione.
- **Fermi immagine delle animazioni nel PDF.** Per ogni figura animata il PDF riporta tre fotogrammi (inizio, metà, fine) e l'indirizzo della pagina in cui la figura si muove.
- {doc}`Prefazione </prefazione>`. **Modalità di scrittura nella Prefazione.** La Prefazione dichiara come il libro è composto, spiega il sottotesto «due volte» e indica la scommessa che le correzioni diventino rare, versione dopo versione.
- {doc}`Algebra lineare </Matematica/algebra-lineare>` (Matematica). **Caratteri delle figure.** Le etichette delle figure usano ora i caratteri del progetto e non quelli generici del sistema del lettore, e risultano uguali per tutti.
- {doc}`Modelli n-gram </NaturalLanguageProcessing/modelli-ngram>` (Natural Language Processing). **Formule in stampa in «Modelli n-gram».** Corrette quattro formule che il browser tollerava e la stampa no: tre sistemi di equazioni annidati male e una formula spezzata da un a capo.
- {doc}`Prefazione </prefazione>`. **Prefazione come parte autonoma.** La Prefazione esce da «Fondamenta» e apre il libro in una nuova parte dell'indice, «Prima di cominciare».

(v1-4-2)=

## 1.4.2 · 13 agosto 2026

### Patch

- {doc}`Support Vector Machine </MachineLearning/svm>` (Machine Learning). **Derivazione della Support Vector Machine.** Nuova sezione in cinque passi, sui due livelli: larghezza del margine e forma duale in cui i dati compaiono a coppie, base del kernel trick; segue la lezione di Winston.
- {doc}`Support Vector Machine </MachineLearning/svm>` (Machine Learning). **Note storiche nella derivazione della SVM.** Due note riprendono i salti del conto segnalati da Winston; la sezione conclude che Vapnik aveva il kernel già nella tesi degli anni Sessanta e ne capì l'importanza trent'anni dopo.

(v1-4-1)=

## 1.4.1 · 13 agosto 2026

### Patch

- **Sei figure animate.** Ciclo di addestramento con loss in discesa, due modi in cui il DQN rompe la correlazione, scan a raddoppio, agente con contesto crescente, spinta avversaria, residuo di una PINN.
- **Verifica numerica delle figure animate.** Ognuna delle sei calcola i numeri che mostra e non viene generata se non coincidono con quelli del capitolo; la figura della PINN addestra davvero per trentamila epoche.
- {doc}`Mamba </StateSpaceModel/mamba>` (State Space Model). **Scan parallelo in Mamba.** Il testo attribuiva a Blelloch le operazioni della versione in fila, ma il codice è a raddoppio e ne fa di più: si guadagna sui turni, si paga in conti.

(v1-4-0)=

## 1.4.0 · 12 agosto 2026

### Minor

- **Livello Elementare fuori dalle schede.** Didascalie, riquadri «Da ricordare» e raccordi, che usavano la lingua del livello Superiore, sono riscritti; i riquadri sono ora su entrambi i livelli.
- **Tredici figure animate.** Ciascuna calcola i propri numeri e non viene generata se non coincidono con il testo; fra esse fusioni del BPE, k-means, online softmax di Flash Attention, rete di Hopfield.
- **Notazione uniforme.** Il grassetto indica gli oggetti con più componenti (vettori, matrici), il tondo un numero solo: la distinzione valeva solo nei primi due capitoli.
- **Correzioni in tutti i capitoli.** Il codice del libro viene ora eseguito e funziona.
- **Esempi di codice corretti.** Smentiti dal codice e corretti: la frase sulla negazione, le lettere dell'esempio BPE e l'analogia «re meno uomo più donna», che dà «re» senza escludere il punto di partenza.
- {doc}`Il filtraggio collaborativo </SistemiRaccomandazione/filtraggio-collaborativo>` (Sistemi di raccomandazione). **Valutazione nel filtraggio collaborativo.** L'esperimento misurava l'errore sui dati di addestramento; tenendo da parte un quinto dei voti l'errore passa da 0,019 a 0,418.
- {doc}`La legge dentro la loss </PINN/come-funziona>` (PINN: reti e fisica). **Esempio delle PINN.** L'esempio non convergeva come dichiarato; ora mostra il caso che fallisce, una rete con il residuo più basso e la soluzione più sbagliata, con tre controlli nel codice.
- **Errori di fatto e attribuzioni.** Corretti, fra gli altri, il margine massimo (1963), il forget gate della LSTM (2000), le immagini di ImageNet nel 2009 (3,2 milioni) e il segno di BIC e AIC.
- **Notebook scaricabili.** Il pulsante «Esegui il codice» apre ora notebook funzionanti: prima molte celle davano errore di sintassi.
- **Figure incoerenti con la pagina.** Corrette: 17 × 28 invece di 17 × 24, AlphaGo addestrato col solo self-play (è il successore), un blocco di memoria assente, una scelta fuori ordine in un albero.
- **Simboli con più significati.** Le lettere usate per due o tre cose nella stessa pagina sono rinominate; dove il nome è quello standard, il testo avverte che altrove il simbolo si chiama diversamente.
- **Misure su più semi.** I numeri che derivavano da una sola esecuzione, e su altre non reggevano la tesi, sono ora misurati su più semi, con la variabilità dei risultati.

(v1-3-0)=

## 1.3.0 · 11 agosto 2026

### Minor

- {doc}`GraphSAGE, GAT e applicazioni </GraphNeuralNetwork/architetture-applicazioni>` (Graph Neural Network). **Graph Transformer.** Ogni nodo dialoga con tutti gli altri e non solo con i vicini; la codifica di posizione sul grafo generalizza quella dei Transformer, e su una fila sono onde imparentate.
- {doc}`Confronto coi modelli precedenti </Transformers/confronti>` (Transformer). **Attenzione a costo ridotto.** Completate le tre strade (schema prefissato, scelta guidata dai dati con il Reformer, rinuncia alla softmax); aggiunti gli strati reversibili, scambio fra memoria e calcolo.
- {doc}`Multimodalità </Transformers/multimodalita>` (Transformer). **ELECTRA.** Quarto modo di studiare accanto a GPT, BERT e T5, nato dall'obiezione che mascherando una parola su sette per le altre sei non si impara; limite dell'analogia con le GAN.
- {doc}`Tendenze e limiti </Transformers/tendenzefuture>` (Transformer). **Calcolo adattivo nei Transformer.** Perché lo stesso calcolo è speso su domande facili e difficili, e due modi di rimuovere il vincolo: pensare più a lungo in silenzio (meno costoso) o scrivendo (traccia leggibile).
- {doc}`Il meccanismo di attenzione </Transformers/attenzione>` (Transformer). **Antenato dell'attenzione.** Un riquadro mostra che la struttura domanda-archivio nacque a metà degli anni Dieci per far ragionare una rete su un elenco di fatti, ed è quella dei sistemi di recupero.
- {doc}`Dati e pipeline </MLOps/dati-e-pipeline>` (MLOps). **Formati dei dati nelle pipeline.** Il CSV è quasi sempre inadatto all'addestramento, che legge poche colonne su molte; la memorizzazione per colonna di Parquet e Arrow risolve il problema.
- {doc}`Overfitting e validazione </MachineLearning/overfitting-validazione>` (Machine Learning). **Curve di apprendimento e test set.** Come scegliere fra più dati e un altro modello dalla forma di due curve; aggiunti Elastic Net, campionamento stratificato e il rischio di sporcare il test set anche solo guardandolo.
- {doc}`Valutare un modello </MachineLearning/metriche>` (Machine Learning). **Metriche per target ordinali.** Quando la variabile ha un ordine (fasce d'età, stelle, gravità) l'accuratezza non distingue errori piccoli e grandi e sceglie il modello sbagliato; sono indicate misure alternative.
- {doc}`Alberi e metodi ensemble </MachineLearning/alberi-ensemble>` (Machine Learning). **Combinazione di modelli eterogenei.** Per voto o con un modello che li pesa; nell'esperimento della pagina il voto peggiora rispetto al miglior modello singolo e l'alternativa lo supera.
- {doc}`Trovare gli iperparametri </MachineLearning/iperparametri>` (Machine Learning). **Torneo a eliminazione asincrono.** Su un cluster l'attesa della fine di ogni turno lascia ferme quasi tutte le macchine; la versione asincrona evita il problema.
- {doc}`Backpropagation </RetiNeurali/backpropagation>` (Reti neurali). **Cross-entropia ed errore quadratico.** Nella classificazione la cross-entropia non è una convenzione: con l'errore quadratico il modello che sbaglia di più impara più lentamente, l'opposto di ciò che serve.
- {doc}`Far funzionare le reti profonde </DeepLearning/ottimizzazione-regolarizzazione>` (Deep Learning). **Riscaldamento del learning rate.** Perché i primi passi dell'addestramento sono i più rischiosi e conviene farli piccoli; la tecnica compare in ogni ricetta moderna.
- {doc}`Prestazioni e scala </PyTorch/prestazioni>` (PyTorch). **Misura del tempo su GPU.** La GPU esegue quando le è comodo e non quando le si chiede, per cui un cronometro ingenuo misura il nulla; spiegato come misurare correttamente.
- {doc}`Replicare un paper </PyTorch/replicare-un-paper>` (PyTorch). **Diario degli esperimenti.** Le tre regole che lo rendono utile e ciò che gli strumenti di tracciamento non registrano: il motivo per cui un esperimento è stato provato.
- {doc}`Dati su misura </PyTorch/dati-su-misura>` (PyTorch). **Impacchettamento dei dataset grandi.** Conviene perché il costo dominante è aprire i file e non decodificarli; nello stesso passaggio si calcolano le statistiche per la normalizzazione.
- {doc}`Modelli di sequenza </NaturalLanguageProcessing/modelli-sequenza>` (Natural Language Processing). **Addestramento di reti ricorrenti su sequenze lunghe.** Spezzare la sequenza in blocchi impedisce di apprendere legami più lunghi del blocco: parte della «memoria corta» è una scelta ingegneristica e non un limite matematico.
- {doc}`MDP e funzioni valore </ReinforcementLearning/mdp-valore>` (Reinforcement Learning). **Osservabilità parziale.** Quando l'agente vede soltanto una parte rumorosa dello stato, caso comune, alcuni agenti impilano gli ultimi fotogrammi e altri hanno una memoria.
- {doc}`Gradiente di policy </DeepReinforcementLearning/policy-gradient>` (Deep Reinforcement Learning). **Origine del vincolo di PPO.** Il vincolo di PPO è la versione economica di un vincolo preciso, formulato anni prima e troppo costoso da calcolare: meno rigorosa, ma più semplice e sufficiente.
- {doc}`Forecasting neurale </SerieTemporali/forecasting-neurale>` (Serie temporali). **Scomposizione in tendenza e stagionalità.** Terza famiglia fra modelli classici e reti: tendenza, stagionalità e festività con curva interpolata; basta a molti problemi aziendali, e il motivo è anche quello del suo limite.
- {doc}`Parallelismo distribuito </GPU/parallelismo-distribuito>` (GPU e calcolo parallelo). **Scambio dei gradienti ad anello.** Indicato lo schema che l'anello ha sostituito e perché non reggeva l'aumento del numero di macchine.

(v1-2-0)=

## 1.2.0 · 10 agosto 2026

### Minor

- {doc}`Geometria e profondità </VisioneArtificiale/geometria-e-profondita>` (Visione artificiale). **Geometria e profondità in visione artificiale.** Nuova sezione: modello di fotocamera, vincolo epipolare, profondità dalla disparità, flusso ottico e profondità da una sola immagine; il codice verifica i risultati invece di illustrarli.
- {doc}`NeRF e splatting </VisioneArtificiale/rendering-neurale>` (Visione artificiale). **Rendering neurale.** NeRF e splatting gaussiano: una scena come funzione addestrata, rendering volumetrico differenziabile, ruolo della codifica posizionale (come nelle PINN) e delle pose delle fotocamere, ricavate nella sezione precedente.
- {doc}`Una rete, molti compiti </DeepLearning/multi-compito>` (Deep Learning). **Apprendimento multi-compito.** Nuova sezione: un compito in più può aiutare o danneggiare quello di interesse; nelle misure un compito affine riduce l'errore del 65% e uno estraneo lo aumenta del 25%.
- {doc}`Imparare guardando </DeepReinforcementLearning/imitazione>` (Deep Reinforcement Learning). **Apprendimento per imitazione.** Colma il passaggio fra rinforzo e addestramento degli assistenti conversazionali; gli errori si compongono e più il maestro è bravo meno insegna a rimediare.
- {doc}`Riduzione e clustering </MachineLearning/riduzione-clustering>` (Machine Learning). **Misture gaussiane e algoritmo EM.** Probabilità al posto di etichette, forma dei gruppi e criterio per scegliere quanti gruppi, nozioni che i capitoli su tokenizzatori e riconoscimento vocale davano per note.
- {doc}`MCTS e AlphaGo </DeepReinforcementLearning/mcts-alphago>` (Deep Reinforcement Learning). **Ricerca ad albero Monte Carlo.** Le quattro mosse, la scelta della mossa più visitata e non di quella con la media migliore, l'identità con la formula dei bandit; già invocata per AlphaGo e MuZero.
- {doc}`Le basi di Python </Python/basi>` (Python). **Il GIL in Python.** Perché i thread aiutano quando si attende e non quando si calcola; il capitolo su PyTorch usava il termine due volte senza introdurlo.
- {doc}`Rappresentare il testo </NaturalLanguageProcessing/rappresentare-testo>` (Natural Language Processing). **Embedding di frase.** Come ottenere un vettore per un'intera frase: un BERT usato così com'è dà vettori mediocri perché addestrato ad altro; due capitoli sul recupero poggiavano su embedding mai spiegati.
- {doc}`Architetture storiche </DeepLearning/architetture-storiche>` (Deep Learning). **Convoluzione separabile in profondità.** Il mattone di EfficientNet: stessa forma in uscita con quasi nove volte meno pesi, ragione per cui la visione artificiale gira su un telefono.
- {doc}`Componenti e modelli classici </SerieTemporali/componenti-e-classici>` (Serie temporali). **Scelta e diagnostica di un modello ARIMA.** Come si scelgono i parametri e si verifica il modello sui residui; aggiunti i due modi di includere informazioni esterne: variabili esogene e serie che si influenzano a vicenda.
- {doc}`Alberi e metodi ensemble </MachineLearning/alberi-ensemble>` (Machine Learning). **Ensemble di modelli diversi.** Combinazione per voto o con un modello addestrato a pesarli; nella misura riportata il voto peggiora rispetto al miglior modello singolo e l'alternativa lo supera.

(v1-1-7)=

## 1.1.7 · 9 agosto 2026

### Patch

- **Stampa sempre in versione chiara.** Con il tema scuro la stampa dava pagine nere e, a sfondi disattivati, testo chiarissimo su carta bianca; ora è sempre chiara e riporta entrambi i livelli.
- **Contrasto della riga di attribuzione.** La riga con l'autore di una citazione usa il grigio del libro in entrambi i temi, perché il precedente era sotto il contrasto minimo raccomandato.

(v1-1-6)=

## 1.1.6 · 9 agosto 2026

### Patch

- **Numerazione delle figure per capitolo.** Le figure portano il numero del capitolo («Fig. 3.2») e non più la numerazione continua, arrivata a «Fig. 302»; i rimandi seguono e il numero coincide con quello dell'indice.
- **Apertura delle figure ingrandite.** La figura ingrandita si apre alla dimensione della finestra, senza zoom, così da mostrare prima l'intero disegno.

(v1-1-5)=

## 1.1.5 · 9 agosto 2026

### Patch

- **Figure a schermo intero su telefono.** Il testo delle figure scendeva sotto i sei pixel; la figura toccata si apre ora a schermo intero alla dimensione di progetto, con zoom a due dita e trascinamento.

(v1-1-4)=

## 1.1.4 · 9 agosto 2026

### Patch

- **Composizione del testo sullo schermo.** Il corpo cresce con la larghezza dello schermo senza scalini, la riga resta fra trentacinque e settanta caratteri e l'interlinea ne segue la lunghezza; il testo non cambia.
- **Colonna di lettura unica.** Testo, codice, figure e tabelle condividono lo stesso margine sinistro; sugli schermi da portatile l'indice di destra non è più tagliato.

(v1-1-3)=

## 1.1.3 · 8 agosto 2026

### Patch

- **Epigrafe di apertura.** Sostituita la frase attribuita a Marcus du Sautoy, senza fonte, con la domanda di Turing in «Computing Machinery and Intelligence» (1950): non simulare una mente adulta, ma una che impara.

(v1-1-2)=

## 1.1.2 · 8 agosto 2026

### Patch

- {doc}`Introduzione </Introduzione/overview>`. **Massima di Weizenbaum.** «To explain away» è reso con «dissolvere» invece della traduzione letterale, priva di senso in italiano; una nota riporta l'originale e il richiamo nelle Conclusioni è allineato.

(v1-1-1)=

## 1.1.1 · 8 agosto 2026

### Patch

- {doc}`Prefazione </prefazione>`. **Ultima riga della Prefazione.** La formula del 2019 «genera paura e alimenta false notizie» diventa «false speranze»: l'ignoranza sul funzionamento di una macchina produce anche l'attesa che sappia fare ciò che non sa fare.

(v1-1-0)=

## 1.1.0 · 8 agosto 2026

*Il libro ha una prefazione*

È saltata fuori la bozza del 2019, quella mandata a un editore: indice di quindici capitoli e una sola parte scritta, la prefazione. Quasi tutto quello che diceva era già nel libro, sparso fra l'introduzione e la sua conclusione. Mancava la prefazione in quanto tale, cioè l'unica pagina in cui l'autore parla in prima persona.

### Minor

- {doc}`Prefazione </prefazione>`. **Prefazione.** Nuova pagina su perché il libro esiste e perché il mancato progetto a stampa del 2019 sia stato un vantaggio: l'indice non nominava i Transformer. Non è un capitolo.
- {doc}`Conclusione </Introduzione/conclusione>` (Introduzione). **Genealogia dell'«elettricità» di Ng.** Prima della citazione di Andrew Ng sull'AI come «nuova elettricità» è inserita la genealogia a cui allude: le tecnologie che cambiano il modo di risolvere tutti gli altri problemi.
- **Numerazione dei capitoli nell'indice.** Il numero è assegnato ai soli capitoli (la pagina degli aggiornamenti compariva come trentaquattresima) e coincide con quello delle schede in prima pagina.
- {doc}`Conclusioni </Conclusioni/overview>`. **Citazione di Weizenbaum nelle Conclusioni.** Il richiamo cita la frase con cui si apre l'Introduzione, «si dice che spiegare significhi dissolvere», al posto di una priva di riscontro nelle fonti primarie.
- **Comando di apertura delle sezioni nell'indice.** Il controllo che apre le sezioni di un capitolo senza entrarvi, prima un'icona di dodici pixel, è ora un comando alto quanto la riga, in teal, evidenziato al passaggio.

(v1-0-3)=

## 1.0.3 · 8 agosto 2026

### Patch

- **Storia del libro e manoscritto del 2019.** La storia presupponeva un'edizione a stampa mai uscita: il termine di riferimento per i capitoli aggiunti è il manoscritto del 2019, e questa è la prima forma pubblica del testo.

(v1-0-2)=

## 1.0.2 · 8 agosto 2026

### Patch

- **Data di nascita del libro.** Il libro nasce nel 2019, la versione online nel giugno 2024; la nota di copyright parte dal 2019.

(v1-0-1)=

## 1.0.1 · 8 agosto 2026

### Patch

- **Titolo e descrizione per i motori di ricerca.** Bibliografia e pagina degli aggiornamenti usavano titolo e descrizione della copertina; ora ognuna ha i propri. Il difetto riguardava le voci di primo livello dell'indice, prive di capitolo sovraordinato.

(v1-0-0)=

## 1.0.0 · 8 agosto 2026

*Il libro prende un numero*

La 1.0.0 non è il primo giorno del libro, che nasce nel 2019 (la storia breve è in fondo a questa pagina): è il primo giorno in cui il libro ha un numero, e il punto da cui si contano i cambiamenti. L'elenco qui sotto è il lavoro di agosto 2026, l'ultimo mese prima che il registro cominciasse.

### Major

- {doc}`La matematica di un LLM </Matematica/matematica-llm>` (Matematica). **Matematica dei modelli linguistici.** Gli strumenti dei capitoli di matematica applicati a un token che entra in un modello e ne esce, da un'osservazione di Joseph Breeden: l'ostacolo è il vocabolario preso a prestito.
- {doc}`I bandit a più braccia </ReinforcementLearning/banditi>` (Reinforcement Learning). **Banditi a più braccia.** Il dilemma fra esplorare e sfruttare nella forma più pura, dalla domanda posta da Thompson nel 1933 sulle sperimentazioni cliniche; precede gli MDP.
- {doc}`I metodi Monte Carlo </ReinforcementLearning/monte-carlo>` (Reinforcement Learning). **Metodi Monte Carlo nel rinforzo.** Apprendere giocando fino in fondo, senza modello dell'ambiente: il passo fra la value iteration, che lo richiede, e il Q-learning.
- {doc}`I knowledge graph </GraphNeuralNetwork/knowledge-graph>` (Graph Neural Network). **Knowledge graph.** Grafi i cui archi sono fatti con verso e nome, dal principio «things, not strings» di Google alle rappresentazioni di entità e relazioni.
- {doc}`Una rete, cento lingue </Transformers/multilingua>` (Transformer). **Modelli multilingua.** Un solo modello per cento lingue: vocabolari condivisi, trasferimento fra lingue lontane e costo dello spazio comune; riprende il filo della traduzione neurale, fermo al 2016.
- {doc}`Il conto in energia </MLOps/energia-e-impronta>` (MLOps). **Energia e impronta dei modelli.** Come si misura l'energia consumata, quasi mai dichiarata accanto a parametri, FLOP, accuratezza e latenza, e perché il valore cambia di un ordine di grandezza secondo il punto di misura.
- {doc}`Message passing </GraphNeuralNetwork/message-passing>` (Graph Neural Network). **Matrice di adiacenza normalizzata e oversmoothing.** Origine della matrice normalizzata e perché l'oversmoothing è conseguenza del message passing e non un difetto da correggere.
- {doc}`Confronto coi modelli precedenti </Transformers/confronti>` (Transformer). **Self-attention come message passing.** Terzo filo del confronto: la self-attention è message passing su un grafo completo, e ciò collega questo capitolo a quello sulle graph neural network.
- {doc}`L'addestramento avversario </GAN/come-funziona>` (GAN). **Valutazione delle GAN.** Come si misura una GAN, domanda lasciata aperta dall'addestramento avversario: Inception Score, FID e ciò che nessuno dei due coglie.
- {doc}`GEMM e tensor core </GPU/gemm-e-tensor-core>` (GPU e calcolo parallelo). **Array sistolico.** Il circuito contenuto in un tensor core e la ragione per cui la moltiplicazione di matrici in hardware procede come in una catena di montaggio.
- {doc}`I tre errori più comuni </PyTorch/errori-comuni>` (PyTorch). **Lettura delle curve di addestramento.** Quali forme indicano overfitting, quali un learning rate errato, quali un errore nei dati.
- {doc}`Pandas e Matplotlib </Python/pandas-matplotlib>` (Python). **Join fra tabelle.** I quattro tipi di join, usati nel capitolo senza essere spiegati.
- **Riquadri «Da ricordare».** Ogni capitolo si chiude con un riquadro su entrambi i livelli: cinque righe per rileggerlo in un minuto o valutare se aprirlo.
- **Correzioni generali.** Corrette date, attribuzioni, esempi numerici e notazione nei capitoli in cui servivano.
- **Codice e notebook allineati.** Il codice dei blocchi Python è stato eseguito e i notebook del pulsante «Esegui il codice» rigenerati dalle pagine corrette, così testo e notebook coincidono.
- **Palette del tema.** Rimosso il viola residuo delle impostazioni di fabbrica nei fondini delle schede: la palette del libro vale ovunque.
- **Pagina degli aggiornamenti e numero di versione.** Introdotta questa pagina, con il numero di versione in cima all'indice e nella prima pagina del libro.

## Prima della 1.0

Il libro nasce nel 2019 come manoscritto destinato alla stampa, che non è mai
stato pubblicato: la versione online è la sua prima edizione. Le date che
seguono riguardano questa versione, non la stesura del testo.

L'impianto Jupyter Book risale al 13 giugno 2024, con la licenza CC BY-NC-ND
e le prime pagine: l'introduzione, il capitolo su Python e, quattro giorni
dopo, i due livelli di lettura, poi estesi a tutto il libro. Nell'ottobre del
2025 si amplia il capitolo sui Transformer.

Il lavoro riprende nel luglio del 2026: prima i capitoli del manoscritto
(matematica, machine learning, reti neurali, deep learning, visione
artificiale, NLP, GAN, reinforcement learning, speech recognition), poi quelli
che nel 2019 non potevano esserci, da PyTorch e le GPU fino agli agenti, ai
modelli di diffusione, agli state space model e ai sistemi multi-agente.

Queste tappe non hanno un numero di versione, perché non corrispondono a
pubblicazioni: il registro comincia dalla 1.0.0.
