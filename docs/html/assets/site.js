// Dev AI Lab docs — site.js
// Theme toggle + TOC scroll-spy + sidebar active state

(function () {
  'use strict';

  // ---------- Theme ----------
  const THEME_KEY = 'devai-docs-theme';
  const root = document.documentElement;
  const saved = localStorage.getItem(THEME_KEY);
  if (saved === 'light' || saved === 'dark') {
    root.setAttribute('data-theme', saved);
  }

  function toggleTheme() {
    const current = root.getAttribute('data-theme') || 'dark';
    const next = current === 'dark' ? 'light' : 'dark';
    root.setAttribute('data-theme', next);
    localStorage.setItem(THEME_KEY, next);
  }
  window.toggleTheme = toggleTheme;

  // ---------- Sidebar active link ----------
  const path = location.pathname.split('/').pop() || 'index.html';
  document.querySelectorAll('.nav-section a').forEach(a => {
    const href = a.getAttribute('href');
    if (href === path || (path === '' && href === 'index.html')) {
      a.classList.add('active');
    }
  });

  // ---------- Auto-generate TOC + click-to-open toggle ----------
  const tocPanel = document.querySelector('.toc-floating');
  const tocList = tocPanel ? tocPanel.querySelector('ul') : null;
  if (tocPanel && tocList) {
    const headings = document.querySelectorAll('.content h2, .content h3');
    headings.forEach(h => {
      if (!h.id) {
        h.id = h.textContent.trim().toLowerCase()
          .replace(/[^\w\s-]/g, '')
          .replace(/\s+/g, '-');
      }
      const li = document.createElement('li');
      const a = document.createElement('a');
      a.href = '#' + h.id;
      a.textContent = h.textContent.replace(/^\d+(\.\d+)*\.?\s*/, '');
      a.className = h.tagName.toLowerCase();
      li.appendChild(a);
      tocList.appendChild(li);
    });

    // Build the toggle button (only when there's something to show)
    if (headings.length > 0) {
      const btn = document.createElement('button');
      btn.type = 'button';
      btn.className = 'toc-toggle';
      btn.setAttribute('aria-expanded', 'false');
      btn.setAttribute('aria-controls', 'toc-panel');
      btn.innerHTML = `
        <svg width="14" height="14" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round" aria-hidden="true">
          <line x1="9" y1="6" x2="21" y2="6"></line>
          <line x1="9" y1="12" x2="21" y2="12"></line>
          <line x1="9" y1="18" x2="21" y2="18"></line>
          <circle cx="4" cy="6" r="1.5"></circle>
          <circle cx="4" cy="12" r="1.5"></circle>
          <circle cx="4" cy="18" r="1.5"></circle>
        </svg>
        <span>On this page</span>`;
      document.body.appendChild(btn);
      tocPanel.id = 'toc-panel';

      const closePanel = () => {
        tocPanel.classList.remove('open');
        btn.setAttribute('aria-expanded', 'false');
      };
      const openPanel = () => {
        tocPanel.classList.add('open');
        btn.setAttribute('aria-expanded', 'true');
      };
      btn.addEventListener('click', e => {
        e.stopPropagation();
        if (tocPanel.classList.contains('open')) closePanel(); else openPanel();
      });
      tocPanel.addEventListener('click', e => {
        if (e.target.tagName === 'A') closePanel();
        else e.stopPropagation();
      });
      document.addEventListener('click', e => {
        if (!tocPanel.contains(e.target) && !btn.contains(e.target)) closePanel();
      });
      document.addEventListener('keydown', e => {
        if (e.key === 'Escape') closePanel();
      });
    }

    // Scroll-spy
    const links = tocList.querySelectorAll('a');
    const observer = new IntersectionObserver(
      entries => {
        entries.forEach(entry => {
          if (entry.isIntersecting) {
            const id = entry.target.id;
            links.forEach(a => {
              a.classList.toggle('active', a.getAttribute('href') === '#' + id);
            });
          }
        });
      },
      { rootMargin: '-30% 0px -60% 0px', threshold: 0 }
    );
    headings.forEach(h => observer.observe(h));
  }

  // ---------- Copy buttons on pre ----------
  document.querySelectorAll('pre').forEach(pre => {
    const btn = document.createElement('button');
    btn.textContent = 'copy';
    btn.className = 'pre-copy';
    btn.style.cssText = `
      position: absolute; top: 6px; right: 10px;
      background: transparent; border: 0; color: var(--text-dim);
      font-size: 0.75rem; cursor: pointer; font-family: inherit;
      opacity: 0; transition: opacity 180ms;
    `;
    pre.addEventListener('mouseenter', () => btn.style.opacity = '1');
    pre.addEventListener('mouseleave', () => btn.style.opacity = '0');
    btn.addEventListener('click', () => {
      const code = pre.querySelector('code');
      navigator.clipboard.writeText(code ? code.textContent : pre.textContent);
      btn.textContent = 'copied!';
      setTimeout(() => (btn.textContent = 'copy'), 1500);
    });
    pre.appendChild(btn);
  });
})();
