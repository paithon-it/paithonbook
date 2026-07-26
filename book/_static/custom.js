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
    improveSidebarNav();
    setupSidebarWidening();
    improveMobileTouch();
    respectReducedMotion();

    console.log('✨ Paithon Book UI enhancements loaded');
  }

  // Run initialization
  init();

  // Re-run some features on dynamic content changes (for SPAs)
  const observer = new MutationObserver(debounce(() => {
    markExternalLinks();
    makeTablesResponsive();
  }, 500));

  observer.observe(document.body, {
    childList: true,
    subtree: true
  });

})();
