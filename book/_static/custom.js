/**
 * PAITHON BOOK - CUSTOM JAVASCRIPT
 * Ottimizzazioni UI/UX e interattività
 */

(function() {
  'use strict';

  // Sphinx emette due volte i file di `html_js_files`, quindi questo file
  // viene eseguito due volte per pagina: senza guardia si ottengono due
  // pulsanti "torna su", due barre di avanzamento e due copie di ogni
  // listener di scroll.
  if (window.__ptCustomCaricato) return;
  window.__ptCustomCaricato = true;

  // ===== UTILITY FUNCTIONS =====
  const debounce = (func, wait) => {
    let timeout;
    return function executedFunction(...args) {
      const later = () => {
        clearTimeout(timeout);
        func(...args);
      };
      clearTimeout(timeout);
      timeout = setTimeout(later, wait);
    };
  };

  // Il pulsante «torna su» lo mette gia' il tema (`#pst-back-to-top`, la
  // pillola in basso): quello nostro, un cerchio terracotta in fisso a
  // destra, faceva la stessa cosa a due dita di distanza. Su schermo piccolo
  // se ne vedevano due, uno sopra l'altro. Tolto il nostro.

  // ===== READING PROGRESS BAR =====
  function addReadingProgressBar() {
    const progressBar = document.createElement('div');
    progressBar.className = 'reading-progress';
    progressBar.style.cssText = `
      position: fixed;
      top: 0;
      left: 0;
      width: 0%;
      height: 4px;
      background: var(--primary-color, #B5532C);
      z-index: 9999;
      transition: width 0.1s ease;
    `;

    document.body.appendChild(progressBar);

    const updateProgressBar = debounce(() => {
      const winScroll = document.body.scrollTop || document.documentElement.scrollTop;
      const height = document.documentElement.scrollHeight - document.documentElement.clientHeight;
      const scrolled = (winScroll / height) * 100;
      progressBar.style.width = scrolled + '%';
    }, 10);

    window.addEventListener('scroll', updateProgressBar);
  }

  // ===== CODE COPY BUTTONS =====
  // Rimosso: Jupyter Book include gia' sphinx-copybutton (button.copybtn).
  // Un secondo bottone iniettato qui si sovrapponeva a quello nativo,
  // creando icone doppie e disallineate sui blocchi di codice.
  // Lo stile del bottone nativo e' in custom.css (button.copybtn).

  // ===== EXTERNAL LINKS INDICATOR =====
  function markExternalLinks() {
    const links = document.querySelectorAll('a[href^="http"]');

    links.forEach(link => {
      // Skip if it's an internal link
      if (link.hostname === window.location.hostname) return;

      link.setAttribute('target', '_blank');
      link.setAttribute('rel', 'noopener noreferrer');

      // Add icon if not already present
      if (!link.querySelector('.external-icon')) {
        const icon = document.createElement('span');
        icon.className = 'external-icon';
        icon.innerHTML = ' ↗';
        icon.style.cssText = `
          font-size: 0.8em;
          opacity: 0.6;
          margin-left: 2px;
        `;
        link.appendChild(icon);
      }
    });
  }

  // ===== TABLE RESPONSIVE WRAPPER =====
  function makeTablesResponsive() {
    const tables = document.querySelectorAll('table');

    tables.forEach(table => {
      if (table.parentElement.classList.contains('table-wrapper')) return;

      const wrapper = document.createElement('div');
      wrapper.className = 'table-wrapper';
      wrapper.style.cssText = `
        overflow-x: auto;
        margin: 2rem 0;
        border-radius: 8px;
        box-shadow: 0 1px 3px rgba(0,0,0,0.06);
      `;

      table.parentNode.insertBefore(wrapper, table);
      wrapper.appendChild(table);
    });
  }

  // ===== LAZY LOADING IMAGES =====
  function setupLazyLoading() {
    const images = document.querySelectorAll('img[data-src]');

    if ('IntersectionObserver' in window) {
      const imageObserver = new IntersectionObserver((entries, observer) => {
        entries.forEach(entry => {
          if (entry.isIntersecting) {
            const img = entry.target;
            img.src = img.dataset.src;
            img.removeAttribute('data-src');
            imageObserver.unobserve(img);
          }
        });
      });

      images.forEach(img => imageObserver.observe(img));
    } else {
      // Fallback for browsers without IntersectionObserver
      images.forEach(img => {
        img.src = img.dataset.src;
        img.removeAttribute('data-src');
      });
    }
  }

  // ===== KEYBOARD SHORTCUTS =====
  function setupKeyboardShortcuts() {
    document.addEventListener('keydown', (e) => {
      // Ctrl/Cmd + K: Focus search
      if ((e.ctrlKey || e.metaKey) && e.key === 'k') {
        e.preventDefault();
        const searchInput = document.querySelector('input[type="search"]');
        if (searchInput) searchInput.focus();
      }

      // Escape: Close modals/sidebars
      if (e.key === 'Escape') {
        // Add your modal/sidebar close logic here
      }
    });
  }

  // ===== IMPROVED SIDEBAR NAVIGATION =====
  // ===== NUMERI DEI CAPITOLI NELL'INDICE DI SINISTRA =====
  /**
   * Scrive in `data-pt-numero` il numero progressivo di ogni CAPITOLO
   * dell'indice di sinistra; a stamparlo poi e' il `::before` in custom.css.
   *
   * Capitolo, qui, vuol dire quello che vuol dire nel `_toc.yml`: una
   * cartella con dentro il suo `overview.md`. Prefazione, bibliografia e
   * aggiornamenti stanno in radice e non sono capitoli, quindi non prendono
   * un numero ne' lo consumano. E' lo stesso criterio di `conta_capitoli()`
   * e dell'asse `landing` di `coerenza.py`: i numeri dell'indice e quelli
   * delle schede della landing devono venire dallo stesso conto, altrimenti
   * la stessa pagina si chiama 3 di qua e 4 di la'.
   *
   * Perche' non lo fa il CSS, che sarebbe il posto naturale e dove stava
   * fino a ieri: il contatore CSS si incrementa su OGNI voce di primo
   * livello, e le pagine che capitoli non sono andavano tolte a mano, una per
   * una, per nome di file. Una lista di eccezioni si dimentica (la pagina
   * degli aggiornamenti, arrivata dopo, si era numerata da se' e nell'indice
   * compariva come il capitolo 34) e, soprattutto, non si puo' scrivere
   * giusta: Sphinx marca la voce della pagina corrente con `href="#"`, quindi
   * qualunque selettore sul nome del file sbaglia proprio il capitolo che si
   * sta leggendo. Qui invece `voce.href` e' l'URL gia' risolto dal browser,
   * che per `href="#"` e' quello della pagina corrente: il criterio regge su
   * tutte le pagine.
   *
   * Da sapere se un giorno un capitolo non si chiamera' `overview.md`: quel
   * capitolo resterebbe senza numero, in silenzio. La convenzione e' scritta
   * in CLAUDE.md, ma qui non c'e' niente che la faccia rispettare.
   */
  function numeraCapitoli() {
    const indice = document.querySelector('nav.bd-links');
    if (!indice) return;

    let numero = 0;
    indice.querySelectorAll('li.toctree-l1 > a.reference').forEach(voce => {
      let percorso;
      try {
        percorso = new URL(voce.href, window.location.href).pathname;
      } catch (e) {
        return;
      }
      // Una cartella, poi `overview.html`: `/main/Matematica/overview.html`
      // si', `/main/prefazione.html` no.
      if (!/\/[^/]+\/overview\.html$/.test(percorso)) return;
      numero += 1;
      voce.dataset.ptNumero = String(numero).padStart(2, '0');
    });
  }

  function improveSidebarNav() {
    const sidebar = document.querySelector('.bd-sidebar');
    if (!sidebar) return;

    // Highlight current section on scroll
    const sections = document.querySelectorAll('h2[id], h3[id]');
    const navLinks = document.querySelectorAll('.bd-sidebar a');

    const highlightNav = debounce(() => {
      let currentSection = '';

      sections.forEach(section => {
        const sectionTop = section.offsetTop;
        if (window.pageYOffset >= sectionTop - 100) {
          currentSection = section.getAttribute('id');
        }
      });

      navLinks.forEach(link => {
        link.classList.remove('current-section');
        if (link.getAttribute('href') === `#${currentSection}`) {
          link.classList.add('current-section');
          link.style.fontWeight = '700';
          link.style.borderLeft = '3px solid var(--primary-color)';
        } else {
          link.style.fontWeight = '';
          link.style.borderLeft = '';
        }
      });
    }, 100);

    window.addEventListener('scroll', highlightNav);
  }

  // ===== MOBILE TOUCH IMPROVEMENTS =====
  function improveMobileTouch() {
    if ('ontouchstart' in window) {
      // Add touch-friendly padding to clickable elements
      const clickables = document.querySelectorAll('a, button, .clickable');
      clickables.forEach(el => {
        el.style.minHeight = '44px'; // iOS recommended tap target size
      });
    }
  }

  // ===== SIDEBAR COLLASSATA → LETTURA PIÙ AMPIA =====
  // Quando il toggle nasconde la barra di sinistra, segnaliamo lo stato con
  // la classe `pt-sidebar-collapsed` su <html>: il CSS allarga la colonna e
  // ingrandisce il testo. Rileviamo lo stato MISURANDO se la sidebar occupa
  // ancora spazio nel layout, così è indipendente dal meccanismo del tema
  // (checkbox, transform off-canvas, media query…).
  function setupSidebarWidening() {
    // sidebar primaria (pydata: .bd-sidebar-primary; fallback: .bd-sidebar)
    const getSidebar = () =>
      document.querySelector('.bd-sidebar-primary') ||
      document.querySelector('.bd-sidebar:not(.bd-sidebar-secondary)');

    // La sidebar "occupa spazio" solo se è visibile, ha larghezza, non è un
    // overlay (fixed/absolute non spinge il contenuto) e non è fuori campo.
    function takesLayoutSpace(el) {
      if (!el) return true; // niente sidebar → non collassiamo nulla
      const cs = getComputedStyle(el);
      if (cs.display === 'none' || cs.visibility === 'hidden' ||
          parseFloat(cs.opacity) === 0) return false;
      if (cs.position === 'fixed' || cs.position === 'absolute') return false;
      const r = el.getBoundingClientRect();
      if (r.width < 2) return false;
      if (r.right <= 2 || r.left >= window.innerWidth - 2) return false;
      return true;
    }

    function evaluate() {
      const collapsed = !takesLayoutSpace(getSidebar());
      document.documentElement.classList.toggle('pt-sidebar-collapsed', collapsed);
    }

    // Ricalcolo più volte durante l'animazione del toggle (la sidebar può
    // scorrere via con un transform, che non emette resize).
    function evaluateSoon() {
      [0, 60, 180, 360].forEach(t => setTimeout(evaluate, t));
    }

    // Click su qualunque toggle della sidebar (capture: prima che il tema agisca).
    document.addEventListener('click', (e) => {
      if (e.target.closest(
        '.primary-toggle, .sidebar-toggle, label[for="__primary"], ' +
        'label[for="pst-primary-sidebar-checkbox"]'
      )) evaluateSoon();
    }, true);

    // Fine transizione della sidebar (scorrimento off-canvas) e resize.
    const sb = getSidebar();
    if (sb) sb.addEventListener('transitionend', evaluate);
    window.addEventListener('resize', debounce(evaluate, 120));

    evaluate(); // stato iniziale
  }

  // ===== LA LENTE: LA FIGURA A SCHERMO INTERO, INGRANDIBILE =====
  //
  // Le figure del libro sono diagrammi con del testo dentro, disegnato intorno
  // ai 13px su un viewBox largo: nella colonna di un telefono quel testo
  // scende sotto i sei pixel e la figura si vede ma non si legge.
  //
  // Sphinx avvolge gia' ogni immagine di figura in un link all'originale: qui
  // si intercetta quel link. Senza JavaScript resta il link di prima e
  // funziona come prima, che e' la ragione per cui non si costruisce un
  // bersaglio nuovo.
  const LENTE_MIN = 1;        // "adattata alla finestra"
  const LENTE_MAX = 8;
  const LENTE_DOPPIO_TAP = 2.5;

  function setupLente() {
    const figure = document.querySelectorAll(
      '.bd-article figure a.image-reference, .bd-article figure a.reference.internal.image-reference');
    if (!figure.length) return;

    let visore = null, scena = null, img = null, didascalia = null, etichetta = null;
    let scala = 1, tx = 0, ty = 0;
    const puntatori = new Map();
    let pizzicoAvvio = null;     // {dist, scala, cx, cy}
    let trascina = null;         // {x, y, tx, ty}
    let ultimoTap = 0;
    let haTrascinato = false;

    function costruisci() {
      visore = document.createElement('dialog');
      visore.className = 'pt-lente';
      visore.innerHTML = `
        <div class="pt-lente__scena" data-pt-scena>
          <img class="pt-lente__figura" alt="" data-pt-figura>
        </div>
        <div class="pt-lente__barra">
          <button type="button" class="pt-lente__comando" data-pt-zoom="-1"
                  aria-label="Rimpicciolisci">&minus;</button>
          <span class="pt-lente__scala" data-pt-scala aria-live="polite">100%</span>
          <button type="button" class="pt-lente__comando" data-pt-zoom="1"
                  aria-label="Ingrandisci">+</button>
          <button type="button" class="pt-lente__comando" data-pt-zoom="0"
                  aria-label="Torna alla dimensione della finestra">Adatta</button>
          <span class="pt-lente__spazio"></span>
          <button type="button" class="pt-lente__comando" data-pt-chiudi
                  aria-label="Chiudi">&#10005;</button>
        </div>
        <div class="pt-lente__didascalia" data-pt-didascalia></div>`;
      document.body.appendChild(visore);

      scena = visore.querySelector('[data-pt-scena]');
      img = visore.querySelector('[data-pt-figura]');
      didascalia = visore.querySelector('[data-pt-didascalia]');
      etichetta = visore.querySelector('[data-pt-scala]');

      visore.addEventListener('click', e => {
        const zoom = e.target.closest('[data-pt-zoom]');
        if (zoom) {
          const v = Number(zoom.dataset.ptZoom);
          if (v === 0) reimposta();
          else applica(scala * (v > 0 ? 1.4 : 1 / 1.4));
          return;
        }
        if (e.target.closest('[data-pt-chiudi]')) { visore.close(); return; }
        // Il velo. Tre cose devono NON chiudere, e tutte e tre sono successe
        // davvero alla prima prova:
        //   - la fine di un trascinamento. `setPointerCapture` dirotta alla
        //     scena anche il click composito, quindi spostare la figura
        //     arrivava qui e la chiudeva in faccia a chi la stava guardando;
        //   - un tocco mentre la figura e' ingrandita: li si sta esplorando,
        //     e il dito che si appoggia non e' una richiesta di uscire;
        //   - il primo dei due tocchi di un doppio tap, che altrimenti chiude
        //     prima che il secondo arrivi.
        // Resta quello che si vuole dire davvero con un click nel vuoto
        // intorno al disegno: ho finito.
        if (e.target === scena || e.target === img) {
          if (haTrascinato || scala > LENTE_MIN + 0.01) return;
          if (dentroIlDisegno(e.clientX, e.clientY)) return;
          visore.close();
        }
      });

      // `close` scatta anche quando chiude Esc, che e' del <dialog> e non
      // passa da qui: la pulizia va agganciata all'evento, non al bottone.
      visore.addEventListener('close', () => {
        document.documentElement.classList.remove('pt-lente-aperta');
        img.removeAttribute('src');
      });

      scena.addEventListener('wheel', e => {
        e.preventDefault();
        const p = puntoRelativo(e.clientX, e.clientY);
        applica(scala * (e.deltaY < 0 ? 1.12 : 1 / 1.12), p);
      }, { passive: false });

      scena.addEventListener('pointerdown', e => {
        scena.setPointerCapture(e.pointerId);
        puntatori.set(e.pointerId, { x: e.clientX, y: e.clientY });
        if (puntatori.size === 2) {
          const [a, b] = [...puntatori.values()];
          pizzicoAvvio = {
            dist: Math.hypot(a.x - b.x, a.y - b.y) || 1,
            scala,
            centro: puntoRelativo((a.x + b.x) / 2, (a.y + b.y) / 2),
          };
          trascina = null;
        } else if (puntatori.size === 1) {
          trascina = { x: e.clientX, y: e.clientY, tx, ty };
          haTrascinato = false;
        }
      });

      scena.addEventListener('pointermove', e => {
        if (!puntatori.has(e.pointerId)) return;
        puntatori.set(e.pointerId, { x: e.clientX, y: e.clientY });

        if (puntatori.size === 2 && pizzicoAvvio) {
          const [a, b] = [...puntatori.values()];
          const d = Math.hypot(a.x - b.x, a.y - b.y) || 1;
          applica(pizzicoAvvio.scala * (d / pizzicoAvvio.dist), pizzicoAvvio.centro);
        } else if (trascina && puntatori.size === 1) {
          if (Math.hypot(e.clientX - trascina.x, e.clientY - trascina.y) > 8) {
            haTrascinato = true;
            scena.dataset.ptTrascina = 'si';
          }
          tx = trascina.tx + (e.clientX - trascina.x);
          ty = trascina.ty + (e.clientY - trascina.y);
          disegna();
        }
      });

      const sufine = e => {
        puntatori.delete(e.pointerId);
        if (puntatori.size < 2) pizzicoAvvio = null;
        if (puntatori.size === 0) {
          delete scena.dataset.ptTrascina;
          // doppio tocco: alterna fra adattata e ingrandita, sul punto toccato
          const ora = Date.now();
          const fermo = trascina &&
                Math.hypot(e.clientX - trascina.x, e.clientY - trascina.y) < 8;
          if (fermo && ora - ultimoTap < 320) {
            if (scala > LENTE_MIN + 0.01) reimposta();
            else applica(LENTE_DOPPIO_TAP, puntoRelativo(e.clientX, e.clientY));
            ultimoTap = 0;
          } else if (fermo) {
            ultimoTap = ora;
          }
          trascina = null;
        }
      };
      scena.addEventListener('pointerup', sufine);
      scena.addEventListener('pointercancel', sufine);

      visore.addEventListener('keydown', e => {
        if (e.key === '+' || e.key === '=') { applica(scala * 1.4); e.preventDefault(); }
        else if (e.key === '-') { applica(scala / 1.4); e.preventDefault(); }
        else if (e.key === '0') { reimposta(); e.preventDefault(); }
      });

      window.addEventListener('resize', () => { if (visore.open) reimposta(); });
    }

    // Coordinate del puntatore rispetto al CENTRO della scena: e' l'origine
    // della trasformazione, e ragionare in quel sistema evita di rincorrere
    // gli offset a ogni calcolo.
    function puntoRelativo(cx, cy) {
      const r = scena.getBoundingClientRect();
      return { x: cx - (r.left + r.width / 2), y: cy - (r.top + r.height / 2) };
    }

    // Zoom "focale": il punto sotto le dita resta sotto le dita. Un punto p
    // dello schermo sta sull'immagine in u = (p - t)/s; imporre che u non
    // cambi passando da s a s' da' t' = p - (p - t)·s'/s.
    function applica(nuova, fuoco) {
      const s = Math.min(LENTE_MAX, Math.max(LENTE_MIN, nuova));
      const p = fuoco || { x: 0, y: 0 };
      tx = p.x - (p.x - tx) * (s / scala);
      ty = p.y - (p.y - ty) * (s / scala);
      scala = s;
      disegna();
    }

    // "Adatta" vuol dire adattata alla finestra, ed e' il minimo consentito.
    function reimposta() {
      scala = LENTE_MIN; tx = 0; ty = 0;
      disegna();
    }

    // Il visore si apre SEMPRE adattato alla finestra: prima si vede tutta la
    // figura, poi si decide dove guardare da vicino. Aprire gia' ingranditi
    // fa risparmiare un gesto e costa l'orientamento, che su un diagramma e'
    // il primo pezzo di informazione.

    // La figura non si porta fuori dalla finestra: oltre il bordo non c'e'
    // niente da guardare, e ritrovarla costa piu' che averla spostata.
    //
    // Il riquadro dell'<img> e' grande quanto la scena, ma il disegno dentro
    // e' quello che `object-fit: contain` ci fa stare: limitare sul riquadro
    // lascerebbe trascinare la figura dentro due bande vuote. Le proporzioni
    // arrivano da naturalWidth/naturalHeight, che per una SVG il browser
    // ricava dal viewBox; se non le sa (0), il riquadro e' il ripiego.
    // Il rettangolo davvero occupato dal disegno, in coordinate di finestra.
    // L'<img> e' grande quanto la scena, il disegno no: intorno restano due
    // bande vuote, ed e' li che un click vuol dire "chiudi".
    function riquadroDisegno() {
      const s = scena.getBoundingClientRect();
      let w = s.width, h = s.height;
      const nw = img.naturalWidth, nh = img.naturalHeight;
      if (nw > 0 && nh > 0) {
        const proporzioni = nw / nh;
        w = Math.min(s.width, s.height * proporzioni);
        h = w / proporzioni;
      }
      const cx = s.left + s.width / 2 + tx;
      const cy = s.top + s.height / 2 + ty;
      return { s, w, h, cx, cy, dw: w * scala, dh: h * scala };
    }

    function dentroIlDisegno(x, y) {
      const r = riquadroDisegno();
      return Math.abs(x - r.cx) <= r.dw / 2 && Math.abs(y - r.cy) <= r.dh / 2;
    }

    function limita() {
      const { s, w, h } = riquadroDisegno();
      const maxX = Math.max(0, (w * scala - s.width) / 2);
      const maxY = Math.max(0, (h * scala - s.height) / 2);
      tx = Math.min(maxX, Math.max(-maxX, tx));
      ty = Math.min(maxY, Math.max(-maxY, ty));
    }

    function disegna() {
      limita();
      img.style.transform = `translate(${tx}px, ${ty}px) scale(${scala})`;
      etichetta.textContent = Math.round(scala * 100) + '%';
      scena.style.cursor = scala > LENTE_MIN + 0.01 ? 'grab' : 'default';
    }

    // La didascalia senza il suo ancoraggio: Sphinx infila in coda a ogni
    // `figcaption` un link "#" per copiarne l'indirizzo, e `textContent` se lo
    // porta dietro. Nel visore diventava un cancelletto appeso al punto
    // finale.
    function testoDidascalia(figura) {
      const cap = figura?.querySelector('figcaption');
      if (!cap) return '';
      const copia = cap.cloneNode(true);
      copia.querySelectorAll('.headerlink').forEach(a => a.remove());
      return copia.textContent.replace(/\s+/g, ' ').trim();
    }

    function apri(link) {
      if (!visore) costruisci();
      const dentro = link.querySelector('img');
      img.alt = dentro ? dentro.getAttribute('alt') || '' : '';
      didascalia.textContent = testoDidascalia(link.closest('figure'));
      document.documentElement.classList.add('pt-lente-aperta');
      visore.showModal();

      // Il riquadro del disegno lo si conosce solo a immagine caricata, e da
      // quello dipendono i limiti dello spostamento: si ridisegna al `load`,
      // non solo adesso. Le figure sono gia' nella cache della pagina, quindi
      // non c'e' nessuna attesa da vedere.
      reimposta();
      img.onload = reimposta;
      img.src = link.getAttribute('href');
    }

    figure.forEach(link => {
      if (link.dataset.ptLente) return;
      link.dataset.ptLente = 'si';

      if (!link.querySelector('.pt-figura-segno')) {
        const segno = document.createElement('span');
        segno.className = 'pt-figura-segno';
        segno.setAttribute('aria-hidden', 'true');
        segno.innerHTML =
          '<svg viewBox="0 0 16 16" width="15" height="15" focusable="false">' +
          '<circle cx="7" cy="7" r="4.6" fill="none" stroke="currentColor" stroke-width="1.5"/>' +
          '<path d="M10.4 10.4 L14 14" stroke="currentColor" stroke-width="1.6" ' +
          'stroke-linecap="round"/>' +
          '<path d="M5 7h4M7 5v4" stroke="currentColor" stroke-width="1.3" ' +
          'stroke-linecap="round"/></svg>';
        link.appendChild(segno);
      }

      link.setAttribute('aria-label',
        'Apri la figura a schermo intero' + (link.title ? ': ' + link.title : ''));

      link.addEventListener('click', e => {
        // Un click con un modificatore vuole aprire in una scheda: e' un'altra
        // intenzione, e portargliela via sarebbe una prepotenza.
        if (e.metaKey || e.ctrlKey || e.shiftKey || e.altKey || e.button !== 0) return;
        e.preventDefault();
        apri(link);
      });
    });
  }

  // Lo scambio di tema per la stampa stava qui. Ora e' in `brand/print.js`,
  // caricato da `_config.yml`: e' la stessa ricetta che serve al sito, che
  // nasce scuro come il libro, e una ricetta sola non puo' divergere.

  // ===== PERFORMANCE: REDUCE MOTION FOR ACCESSIBILITY =====
  function respectReducedMotion() {
    const prefersReducedMotion = window.matchMedia('(prefers-reduced-motion: reduce)');

    if (prefersReducedMotion.matches) {
      document.documentElement.style.setProperty('--transition-fast', '0s');
      document.documentElement.style.setProperty('--transition-normal', '0s');
      document.body.style.scrollBehavior = 'auto';
    }
  }

  // Qui c'erano due funzioni che riscrivevano gli href a mano: `fixAllLinks`
  // metteva sul logo un `intro.html` **relativo**, e `interceptLinkClicks`
  // trasformava i link a `paithon.it/book` nel percorso che seguiva `/book/`,
  // anch'esso relativo. Servivano quando il libro stava sotto `paithon.it/book`.
  //
  // Ora il libro sta su `book.paithon.it/main/`, e quel codice era diventato la
  // causa di un 404: da una pagina in sottocartella — cioe' da quasi tutte —
  // `intro.html` relativo risolve in `Introduzione/intro.html`, che non esiste,
  // e un percorso estratto come `main/Introduzione/intro.html` risolve in
  // `/main/main/Introduzione/intro.html`. Il doppio `main/` veniva da qui.
  //
  // Il tema calcola l'href giusto da se' (`../intro.html` da una sottocartella,
  // `#` sulla landing) e `_templates/pt-logo-compatto.html` usa `pathto()`.
  // Non c'e' niente da aggiustare: bastava non aggiustarlo.

  // ===== INITIALIZE ALL FEATURES =====
  function init() {
    // Wait for DOM to be fully loaded
    if (document.readyState === 'loading') {
      document.addEventListener('DOMContentLoaded', init);
      return;
    }

    // Initialize features
    addReadingProgressBar();
    markExternalLinks();
    makeTablesResponsive();
    setupLazyLoading();
    setupKeyboardShortcuts();
    numeraCapitoli();
    improveSidebarNav();
    setupSidebarWidening();
    improveMobileTouch();
    setupLente();
    respectReducedMotion();

    console.log('✨ Paithon Book UI enhancements loaded');
  }

  // Run initialization
  init();

  // Re-run some features on dynamic content changes (for SPAs)
  //
  // Questo file è caricato nell'`<head>`, quindi qui `document.body` può
  // essere ancora nullo: `observe(null)` lancia, e l'eccezione zittiva
  // l'osservatore per tutta la vita della pagina — le tabelle aggiunte da
  // Thebe restavano senza il loro contenitore scorrevole, e nessuno lo
  // sapeva perché l'errore usciva in console e basta. `init()` la sua guardia
  // ce l'aveva già; questo pezzo era rimasto fuori.
  function osservaContenutoDinamico() {
    const observer = new MutationObserver(debounce(() => {
      markExternalLinks();
      makeTablesResponsive();
    }, 500));

    observer.observe(document.body, { childList: true, subtree: true });
  }

  if (document.readyState === 'loading') {
    document.addEventListener('DOMContentLoaded', osservaContenutoDinamico);
  } else {
    osservaContenutoDinamico();
  }

})();
