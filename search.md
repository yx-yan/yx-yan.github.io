---
layout: default
title: Search
permalink: /search/
---

<div class="search-container">
  <div class="search-input-wrapper">
    <i class="fas fa-search search-icon"></i>
    <input type="text" id="search-input" placeholder="Type to search (e.g. AI, Data, Python)...">
  </div>
  
  <ul id="results-container" class="post-list"></ul>
</div>

<script src="https://unpkg.com/simple-jekyll-search@latest/dest/simple-jekyll-search.min.js"></script>

<script>
  window.simpleJekyllSearch = new SimpleJekyllSearch({
    searchInput: document.getElementById('search-input'),
    resultsContainer: document.getElementById('results-container'),
    json: '{{ "/search.json" | relative_url }}',
    searchResultTemplate: `
      <li>
        <article class="post-card search-result-card">
          <div class="post-meta">{date}</div>
          <h2 class="post-card-title">
            <a href="{url}">{title}</a>
          </h2>
          <div class="post-meta" style="margin-top: 5px; font-size: 0.8em; color: var(--link-color);">
            <i class="fas fa-tag"></i> {tags}
          </div>
        </article>
      </li>
    `,
    noResultsText: '<li style="text-align:center; color:var(--text-muted);">No posts found. Try another keyword.</li>',
    limit: 10,
    fuzzy: false
  })
</script>