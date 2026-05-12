// Shared sidebar/nav, injected so we don't repeat HTML across 15 pages.
(function () {
  const NAV_HTML = `
<aside class="sidebar">
  <a href="index.html" class="brand">
    <span class="brand-mark">D</span>
    <span class="brand-text">
      <strong>Dev AI Lab</strong>
      <span>Documentation</span>
    </span>
  </a>
  <button class="theme-toggle" onclick="toggleTheme()" aria-label="Toggle theme" title="Toggle dark/light">
    <svg width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round">
      <circle cx="12" cy="12" r="4"></circle>
      <path d="M12 2v2M12 20v2M4.93 4.93l1.41 1.41M17.66 17.66l1.41 1.41M2 12h2M20 12h2M4.93 19.07l1.41-1.41M17.66 6.34l1.41-1.41"></path>
    </svg>
  </button>

  <nav>
    <div class="nav-section">
      <h4>Project</h4>
      <ul>
        <li><a href="index.html">Overview</a></li>
        <li><a href="router.html">Router</a></li>
        <li><a href="backends.html">Backends</a></li>
        <li><a href="ollama_models.html">Ollama models</a></li>
        <li><a href="HOST_VFIO_SETUP.html">Host VFIO setup</a></li>
        <li><a href="bench-results.html">Bench results</a></li>
      </ul>
    </div>
    <div class="nav-section">
      <h4>LLM internals</h4>
      <ul>
        <li><a href="attention-and-the-transformer.html">Transformer &amp; attention</a></li>
        <li><a href="nvfp4-number-formats.html">Number formats</a></li>
        <li><a href="nvfp4-coldstart.html">NVFP4 cold-start</a></li>
        <li><a href="llm-tokens-and-speed.html">Tokens &amp; speed</a></li>
        <li><a href="paged-attention-and-vllm-internals.html">Paged attention</a></li>
        <li><a href="mixture-of-experts.html">Mixture of experts</a></li>
        <li><a href="sampling-strategies.html">Sampling strategies</a></li>
      </ul>
    </div>
    <div class="nav-section">
      <h4>Interfaces</h4>
      <ul>
        <li><a href="reasoning-tool-calling-chat-templates.html">Reasoning &amp; tool calling</a></li>
        <li><a href="openai-api-and-streaming.html">OpenAI API &amp; streaming</a></li>
      </ul>
    </div>
  </nav>
</aside>`;

  // Insert at start of body
  document.addEventListener('DOMContentLoaded', () => {
    const wrapper = document.querySelector('.site');
    if (wrapper && !wrapper.querySelector('.sidebar')) {
      wrapper.insertAdjacentHTML('afterbegin', NAV_HTML);
      // Re-run sidebar active link logic
      const path = location.pathname.split('/').pop() || 'index.html';
      document.querySelectorAll('.nav-section a').forEach(a => {
        if (a.getAttribute('href') === path) a.classList.add('active');
      });
    }
  });
})();
