/**
 * PAITHON BOOK — LA RICERCA
 *
 * Che cosa cambia: la finestra della lente (il pulsante «Cerca», o Ctrl/⌘+K)
 * mostra i risultati MENTRE si scrive, raggruppati per pagina e agganciati
 * alla singola sezione, invece di aspettare l'invio e portare su una pagina
 * di risultati a parte.
 *
 * Da dove vengono i risultati: dall'indice che Sphinx costruisce già a ogni
 * build (`searchindex.js`), che contiene il titolo di OGNI sezione del libro
 * con la sua ancora. Non c'è un secondo indice da generare e da tenere
 * allineato: quello in cui si cerca è per costruzione quello che il libro
 * contiene, e una sezione nuova ci entra da sé.
 *
 * Perché l'indice si carica solo alla prima apertura della finestra: pesa
 * 1,3 MB, cioè più di tutto il resto di una pagina, e un lettore che non
 * cerca niente non deve pagarlo. Le versioni recenti di pydata-sphinx-theme
 * lo mettono nel <head> di ogni pagina (quella pubblicata oggi no): se lo
 * trova già caricato, questo file non lo ricarica.
 *
 * A cercare nel corpo del testo è `Search._performSearch` di Sphinx, che non
 * si riscrive: sa già lo stemming italiano (`language: it` in `_config.yml`),
 * i pesi fra titolo e corpo, le corrispondenze parziali. Quello che si
 * riscrive è la RESA, perché quella del tema fa due cose che con la ricerca a
 * ogni tasto non stanno in piedi: stampa una lista piatta senza dire in che
 * capitolo si è finiti, e per ogni risultato SCARICA la pagina intera per
 * ricavarne un estratto. Venti risultati per parola scritta sono decine di
 * megabyte. Qui, dopo il caricamento dell'indice, non si scarica più niente.
 *
 * E ai titoli si aggiunge una passata nostra, che a Sphinx manca proprio dove
 * questa funzione vive: la sua ricerca nei titoli pretende che la stringa
 * cercata sia lunga almeno metà del titolo (`queryLower.length >=
 * title.length/2` in `searchtools.js`), quindi mentre si scrive non trova
 * niente; e la ricerca nel corpo, per le parole che l'indice non ha esatte,
 * fa una corrispondenza per SOTTOSTRINGA, quindi «gan» pesca «elegante» e
 * «organizzare». La passata qui sotto lavora per PREFISSO DI PAROLA sui
 * titoli delle sezioni, che è il modo in cui si cerca digitando.
 */

(function () {
  'use strict';

  // Sphinx emette due volte i file di `html_js_files` (vedi la stessa guardia
  // in custom.js): senza questa, due pannelli di risultati e due gestori di
  // tastiera.
  if (window.__ptRicercaCaricata) return;
  window.__ptRicercaCaricata = true;

  const MAX_ESITI = 40;     // in tutto
  const MAX_PER_PAGINA = 4; // sezioni mostrate sotto una stessa pagina

  // ===== L'EVIDENZIAZIONE SULLA PAGINA D'ARRIVO =====
  /**
   * Sphinx evidenzia le parole cercate sulla pagina su cui si atterra e ci
   * mette sopra il richiamo «Nascondi i risultati della ricerca». Il difetto
   * è che quel richiamo lo scrive SEMPRE, purché in memoria ci siano dei
   * termini, anche su una pagina dove quei termini non compaiono: si torna
   * alla copertina e in cima all'articolo c'è un avviso che invita a
   * nascondere zero evidenziazioni. E i termini in memoria ce li mette la
   * ricerca stessa a ogni tasto battuto (`Search._parseQuery` scrive in
   * `localStorage`), quindi bastava aprire la finestra, scrivere e chiuderla
   * senza aprire niente perché l'avviso comparisse sulla pagina dopo.
   *
   * Qui la funzione si rifà: si evidenzia, si CONTA, e il richiamo esce solo
   * se c'è davvero qualcosa da nascondere. I termini li scrive in memoria
   * soltanto la navigazione verso un risultato (vedi `apri()`), non la
   * digitazione.
   *
   * E c'è un secondo difetto, che si è visto solo provandolo: Sphinx chiama
   * quella funzione solo `if (typeof Search === "undefined")`, una condizione
   * che vuol dire «non siamo sulla pagina dei risultati». Ma
   * pydata-sphinx-theme, dalla 0.16, carica `searchtools.js` su OGNI pagina,
   * quindi `Search` è sempre definito e l'evidenziazione non parte mai: chi
   * arriva da una ricerca non vede evidenziato niente. Il sito pubblicato
   * oggi monta una versione più vecchia del tema, dove invece parte (ed è da
   * lì che nasce l'avviso fantasma segnalato): cioè la stessa riga di codice
   * si comporta in due modi opposti a seconda del tema. Quindi la chiamata
   * non la si lascia a quella condizione: la fa `avvia()`, una volta sola,
   * e la guardia qui sotto regge il caso in cui la faccia anche Sphinx.
   */
  let giaEvidenziato = false;

  function evidenziaArrivo() {
    if (giaEvidenziato || typeof _highlightText === 'undefined') return;
    giaEvidenziato = true;

    const url = new URL(window.location);
    const grezzi = localStorage.getItem('sphinx_highlight_terms') ||
                   url.searchParams.get('highlight') || '';
    localStorage.removeItem('sphinx_highlight_terms');
    if (url.searchParams.has('highlight')) {
      url.searchParams.delete('highlight');
      window.history.replaceState({}, '', url);
    }

    const termini = grezzi.toLowerCase().split(/\s+/).filter(Boolean);
    if (!termini.length) return;

    const corpo = document.querySelector('article.bd-article') ||
                  document.querySelector('div.body') || document.body;
    termini.forEach((t) => _highlightText(corpo, t, 'highlighted'));

    const quante = document.querySelectorAll('span.highlighted').length;
    const scatola = document.getElementById('searchbox');
    if (!quante || !scatola) return;

    const p = document.createElement('p');
    p.className = 'highlight-link pt-evidenziate';
    const a = document.createElement('a');
    a.href = '#';
    a.textContent = quante === 1
      ? 'Nascondi 1 corrispondenza evidenziata'
      : 'Nascondi le ' + quante + ' corrispondenze evidenziate';
    a.addEventListener('click', (e) => {
      e.preventDefault();
      document.querySelectorAll('span.highlighted')
        .forEach((s) => s.classList.remove('highlighted'));
      p.remove();
    });
    p.appendChild(a);
    scatola.appendChild(p);
  }

  if (typeof SphinxHighlight !== 'undefined') {
    SphinxHighlight.highlightSearchWords = evidenziaArrivo;
  }

  // ===== UTILITÀ =====
  const normalizza = (s) => s.toLowerCase()
    .normalize('NFD').replace(/[\u0300-\u036f]/g, '');

  const proteggi = (s) => s.replace(/[&<>"]/g, (c) =>
    ({ '&': '&amp;', '<': '&lt;', '>': '&gt;', '"': '&quot;' }[c]));

  const perRegex = (s) => s.replace(/[.*+\-?^${}()|[\]\\]/g, '\\$&');

  /** Il testo con i termini cercati avvolti in <mark>. Restituisce HTML. */
  function segna(testo, termini) {
    const html = proteggi(testo);
    if (!termini.length) return html;
    const re = new RegExp('(' + termini.map(perRegex).join('|') + ')', 'gi');
    return html.replace(re, '<mark>$1</mark>');
  }

  // ===== L'INDICE, CARICATO QUANDO SERVE =====
  let promessaIndice = null;

  function caricaScript(url) {
    return new Promise((risolvi, rifiuta) => {
      const s = document.createElement('script');
      s.src = url;
      s.onload = risolvi;
      s.onerror = () => rifiuta(new Error(url));
      document.head.appendChild(s);
    });
  }

  function caricaIndice() {
    if (promessaIndice) return promessaIndice;
    // Il tema, dalla 0.16, li carica già lui su ogni pagina; e la pagina
    // `search.html` ce li ha per conto suo. Ricaricare `searchtools.js` non
    // sarebbe innocuo: dichiara `const Search`, e una seconda dichiarazione è
    // un errore di sintassi.
    if (typeof Search !== 'undefined' && Search.hasIndex && Search.hasIndex()) {
      promessaIndice = Promise.resolve();
      return promessaIndice;
    }
    const radice = document.documentElement.dataset.content_root || './';
    promessaIndice = (typeof Search === 'undefined'
      ? caricaScript(radice + '_static/searchtools.js')
          .then(() => caricaScript(radice + '_static/language_data.js'))
      : Promise.resolve()
    ).then(() => caricaScript(radice + 'searchindex.js'));
    return promessaIndice;
  }

  // Le mappe che l'indice non dà fatte: da nome di file a posizione, e da
  // cartella al titolo del capitolo (che è il titolo del suo `overview`, la
  // convenzione del `_toc.yml`).
  let mappe = null;
  function preparaMappe() {
    if (mappe) return mappe;
    const idx = Search._index;
    const posizione = {}, capitoli = {};
    idx.docnames.forEach((nome, i) => {
      posizione[nome] = i;
      const taglio = nome.indexOf('/');
      if (taglio > 0 && nome.slice(taglio + 1) === 'overview') {
        capitoli[nome.slice(0, taglio)] = idx.titles[i];
      }
    });
    mappe = { posizione, capitoli };
    return mappe;
  }

  // ===== LA RICERCA =====
  /**
   * Punteggio di un titolo di sezione: ogni parola scritta deve comparire
   * come PREFISSO di una parola del titolo. Chi comincia col testo cercato
   * vale più di chi lo contiene in mezzo, e a parità vince il titolo corto,
   * che è quello più specifico.
   */
  function punteggio(titolo, parole) {
    let punti = 0;
    for (const parola of parole) {
      // Tutte le occorrenze, non solo la prima: «reti» dentro «segreti» non
      // vale, ma non deve far scartare il titolo se piu' avanti c'e' «reti
      // neurali». Fermandosi alla prima si perdevano risultati buoni.
      let dove = -1;
      for (let i = titolo.indexOf(parola); i >= 0;
           i = titolo.indexOf(parola, i + 1)) {
        if (i === 0 || /[\s(«"'\-,.:;/]/.test(titolo[i - 1])) { dove = i; break; }
      }
      if (dove < 0) return 0;
      punti += dove === 0 ? 30 : 20;
    }
    return punti + Math.max(0, 20 - Math.floor(titolo.length / 6));
  }

  /**
   * Scorre i titoli di tutte le sezioni e raccoglie quelli per cui `voto`
   * restituisce un punteggio. Le due passate (quella esatta e quella che
   * perdona un refuso) differiscono solo per quella funzione: tenerle in due
   * copie voleva dire vederle divergere alla prima correzione.
   */
  function scorriTitoli(voto) {
    const idx = Search._index;
    const esiti = [];
    for (const titolo in idx.alltitles) {
      const punti = voto(titolo);
      if (!punti) continue;
      for (const coppia of idx.alltitles[titolo]) {
        const file = coppia[0], ancora = coppia[1];
        const eLaPagina = idx.titles[file] === titolo;
        esiti.push({
          file: idx.docnames[file],
          ancora: ancora ? '#' + ancora : '',
          sezione: eLaPagina ? '' : titolo,
          punti: punti + (eLaPagina ? 10 : 0),
        });
      }
    }
    return esiti;
  }

  function cercaNeiTitoli(query) {
    const parole = normalizza(query).split(/\s+/).filter(Boolean);
    if (!parole.length) return [];
    return scorriTitoli((titolo) => punteggio(normalizza(titolo), parole));
  }

  /**
   * Vero se due parole differiscono per una sola modifica: una lettera in
   * più, una in meno, una cambiata. È la distanza di edit fermata a uno, che
   * costa una passata sola e non una matrice.
   */
  function entroUno(a, b) {
    if (Math.abs(a.length - b.length) > 1) return false;
    let i = 0, j = 0, differenze = 0;
    while (i < a.length && j < b.length) {
      if (a[i] === b[j]) { i++; j++; continue; }
      if (++differenze > 1) return false;
      if (a.length > b.length) i++;
      else if (a.length < b.length) j++;
      else { i++; j++; }
    }
    return differenze + (a.length - i) + (b.length - j) <= 1;
  }

  /**
   * La rete per i refusi, e gira SOLO quando la ricerca esatta non ha trovato
   * niente: fuori da quel caso costerebbe senza servire. «Trasformer» per
   * «transformer» è l'errore che un lettore italiano fa più spesso, e senza
   * questa passata la finestra risponde che il libro non ne parla.
   * Sotto le cinque lettere non si tenta: a quella lunghezza una modifica
   * cambia la parola invece di correggerla.
   */
  function cercaApprossimato(query) {
    const parole = normalizza(query).split(/\s+/).filter((p) => p.length >= 5);
    if (!parole.length) return [];
    return scorriTitoli((titolo) => {
      const pezzi = normalizza(titolo).split(/[^a-z0-9]+/).filter(Boolean);
      return parole.every((p) => pezzi.some((t) => entroUno(p, t))) ? 1 : 0;
    });
  }

  function cerca(query) {
    // `_parseQuery` vuole `Stemmer` e `stopwords`, che stanno in
    // `language_data.js`: se quel file non e' arrivato muore qui, e senza
    // questa rete la finestra resterebbe muta invece di dare almeno i titoli.
    let analisi;
    try {
      analisi = Search._parseQuery(query);
    } catch (e) {
      analisi = [query, new Set(), new Set(),
                 new Set(query.toLowerCase().split(/\s+/).filter(Boolean)),
                 new Set()];
    }
    // `_parseQuery` scrive i termini in `localStorage` a ogni tasto battuto:
    // è la memoria da cui nasce l'avviso fantasma riparato in cima a questo
    // file. Ce li rimette `apri()`, e solo se si va davvero da qualche parte.
    localStorage.removeItem('sphinx_highlight_terms');
    const evidenzia = Array.from(analisi[3]).filter(Boolean);

    const esiti = cercaNeiTitoli(query);
    const visti = new Set(esiti.map((e) => e.file + e.ancora));

    let daSphinx = [];
    try {
      daSphinx = Search._performSearch(analisi[0], analisi[1], analisi[2],
                                       analisi[3], analisi[4]);
    } catch (e) {
      daSphinx = [];   // un indice a metà, o un'API cambiata: i titoli bastano
    }
    daSphinx.forEach((riga) => {
      const file = riga[0], titolo = riga[1], ancora = riga[2] || '';
      if (visti.has(file + ancora)) return;
      visti.add(file + ancora);
      const taglio = titolo.indexOf(' > ');
      esiti.push({
        file: file,
        ancora: ancora,
        sezione: taglio > 0 ? titolo.slice(taglio + 3) : '',
        punti: Math.min(riga[4], 25),   // sotto ai titoli, sempre
      });
    });

    esiti.sort((a, b) => b.punti - a.punti);

    // Niente in mano: si riprova perdonando un refuso.
    const approssimato = esiti.length === 0;
    if (approssimato) esiti.push(...cercaApprossimato(query));

    // Raggruppati per pagina, nell'ordine in cui la pagina è comparsa.
    const gruppi = new Map();
    for (const e of esiti.slice(0, MAX_ESITI)) {
      if (!gruppi.has(e.file)) gruppi.set(e.file, []);
      gruppi.get(e.file).push(e);
    }
    return { gruppi, evidenzia, approssimato, quanti: esiti.length };
  }

  // ===== LA FINESTRA =====
  function avvia() {
    // Prima di tutto: le parole cercate, evidenziate sulla pagina d'arrivo.
    // Va chiamata da qui e non lasciata alla condizione di Sphinx, che con
    // questo tema non scatta mai (vedi il commento su `evidenziaArrivo`).
    evidenziaArrivo();

    const finestra = document.getElementById('pst-search-dialog');
    if (!finestra) return;
    const campo = finestra.querySelector('input[name=q]');
    if (!campo) return;

    const pannello = document.createElement('div');
    pannello.className = 'pt-ricerca';
    pannello.innerHTML =
      '<div class="pt-ricerca-esiti" role="listbox" aria-label="Risultati"></div>' +
      '<div class="pt-ricerca-piede">' +
        '<span><kbd>↑</kbd><kbd>↓</kbd> scorri · <kbd>↵</kbd> apri · ' +
        '<kbd>esc</kbd> chiudi</span>' +
        '<span class="pt-ricerca-conto"></span>' +
      '</div>';
    finestra.appendChild(pannello);

    const esiti = pannello.querySelector('.pt-ricerca-esiti');
    const conto = pannello.querySelector('.pt-ricerca-conto');
    let voci = [], scelta = -1, ultima = null, attesa = null, evidenzia = [];

    const radice = document.documentElement.dataset.content_root || './';

    function stato(testo) {
      esiti.innerHTML = '<p class="pt-ricerca-stato">' + proteggi(testo) + '</p>';
      voci = []; scelta = -1; conto.textContent = '';
    }

    function disegna(query) {
      const esito = cerca(query);
      evidenzia = esito.evidenzia;
      if (!esito.quanti) {
        stato('Nessun risultato per «' + query + '».');
        return;
      }
      const { posizione, capitoli } = preparaMappe();
      const idx = Search._index;
      // Se i risultati arrivano dalla rete per i refusi si dice: sono
      // risposte a una domanda diversa da quella scritta, e chi legge deve
      // saperlo invece di credere di aver cercato bene.
      let html = esito.approssimato
        ? '<p class="pt-ricerca-forse">Nessun risultato per «' +
          proteggi(query) + '». Forse cercavi:</p>'
        : '';
      esito.gruppi.forEach((righe, file) => {
        const titoloPagina = idx.titles[posizione[file]] || file;
        const capitolo = capitoli[file.split('/')[0]];
        const strada = capitolo && capitolo !== titoloPagina
          ? '<span class="pt-ricerca-strada">' + proteggi(capitolo) + '</span>'
          : '';
        html += '<div class="pt-ricerca-gruppo">' +
          '<a class="pt-ricerca-voce pt-ricerca-pagina" role="option" ' +
          'href="' + radice + file + '.html">' + strada +
          '<span class="pt-ricerca-titolo">' +
          segna(titoloPagina, evidenzia) + '</span></a>';
        righe.filter((r) => r.sezione).slice(0, MAX_PER_PAGINA).forEach((r) => {
          html += '<a class="pt-ricerca-voce pt-ricerca-sezione" role="option" ' +
            'href="' + radice + file + '.html' + r.ancora + '">' +
            segna(r.sezione, evidenzia) + '</a>';
        });
        html += '</div>';
      });
      esiti.innerHTML = html;
      voci = Array.prototype.slice.call(
        esiti.querySelectorAll('.pt-ricerca-voce'));
      scelta = -1;
      muovi(1);
      // Il conto dice quello che si VEDE: le sezioni oltre MAX_PER_PAGINA
      // restano fuori, e un numero piu' grande di quello che c'e' in pagina
      // fa cercare al lettore risultati che non esistono.
      conto.textContent = voci.length === 1
        ? '1 risultato' : voci.length + ' risultati';
    }

    function muovi(passo) {
      if (!voci.length) return;
      if (scelta >= 0) {
        voci[scelta].classList.remove('scelto');
        voci[scelta].removeAttribute('aria-selected');
      }
      scelta = (scelta + passo + voci.length) % voci.length;
      const voce = voci[scelta];
      voce.classList.add('scelto');
      voce.setAttribute('aria-selected', 'true');
      voce.scrollIntoView({ block: 'nearest' });
    }

    /** Si va da qualche parte: SOLO qui i termini finiscono in memoria, così
     *  la pagina d'arrivo li evidenzia e nessun'altra se li porta dietro. */
    function apri(href) {
      if (evidenzia.length) {
        localStorage.setItem('sphinx_highlight_terms', evidenzia.join(' '));
      }
      window.location.href = href;
    }

    function aggiorna() {
      const query = campo.value.trim();
      if (query === ultima) return;
      ultima = query;
      if (query.length < 2) {
        stato(query ? 'Continua a scrivere…'
                    : 'Scrivi per cercare in tutto il libro.');
        return;
      }
      caricaIndice().then(function () {
        if (campo.value.trim() !== query) return;   // ha continuato a scrivere
        disegna(query);
      }).catch(function () {
        stato('Non riesco a caricare l’indice della ricerca.');
      });
    }

    campo.addEventListener('focus', caricaIndice, { once: true });
    campo.addEventListener('input', () => {
      window.clearTimeout(attesa);
      attesa = window.setTimeout(aggiorna, 120);
    });

    campo.addEventListener('keydown', (e) => {
      if (e.key === 'ArrowDown' || e.key === 'ArrowUp') {
        e.preventDefault();
        muovi(e.key === 'ArrowDown' ? 1 : -1);
      } else if (e.key === 'Enter') {
        e.preventDefault();
        if (scelta >= 0) apri(voci[scelta].href);
        else if (campo.value.trim()) {
          apri(radice + 'search.html?q=' +
               encodeURIComponent(campo.value.trim()));
        }
      }
    });

    esiti.addEventListener('click', (e) => {
      const voce = e.target.closest('.pt-ricerca-voce');
      if (!voce) return;
      e.preventDefault();
      apri(voce.href);
    });

    // Alla fine si cerca quello che nel campo c'è GIÀ, invece di scrivere il
    // messaggio di riposo e aspettare un tasto. Sembra un dettaglio e non lo
    // è: il campo può essere pieno prima che questo file abbia agganciato i
    // suoi ascoltatori, e in quel caso l'evento che lo ha riempito è passato
    // e non torna, quindi la finestra resterebbe muta per sempre. Succede
    // quando il testo arriva tutto insieme invece che a lettere (un
    // incollaggio), su `search.html`, dove Sphinx riempie il campo da sé, e
    // quando il browser ripristina il modulo tornando indietro.
    //
    // Il difetto si vedeva SOLO sul sito pubblicato, e la ragione va saputa
    // prima di credere a una prova locale: qui il tema è più recente e
    // precarica l'indice della ricerca su ogni pagina, quindi `caricaIndice`
    // risolve subito e la finestra di corsa non si apre mai. Là l'indice
    // arriva davvero dopo. Adesso il collaudo prova tutt'e due le topologie.
    aggiorna();

    // E lo stesso quando la finestra si riapre: dentro c'è ancora la ricerca
    // di prima, e vederla senza i suoi risultati è peggio che non vederla.
    document.querySelectorAll('.search-button__button').forEach((pulsante) => {
      pulsante.addEventListener('click', () => {
        ultima = null;       // la stessa domanda va rifatta, non saltata
        window.setTimeout(aggiorna, 0);
      });
    });
  }

  if (document.readyState === 'loading') {
    document.addEventListener('DOMContentLoaded', avvia);
  } else {
    avvia();
  }
})();
