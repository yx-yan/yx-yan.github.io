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

  <div class="tag-cloud">
    <span class="tag-cloud-title">Explore by Tags</span>
    <div class="tags-wrapper">
      {% for tag in site.tags %}
        <button class="search-tag" onclick="filterByTag('{{ tag[0] }}')">
          {{ tag[0] }} <span class="tag-count">{{ tag[1].size }}</span>
        </button>
      {% endfor %}
    </div>
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
    noResultsText: '<li style="text-align:center; color:var(--text-muted); margin-top:30px;">No posts found. Try selecting a tag above.</li>',
    limit: 10,
    fuzzy: false
  })

  // 新增：点击标签触发搜索的函数
  function filterByTag(tagName) {
    const input = document.getElementById('search-input');
    // 1. 将标签名填入搜索框
    input.value = tagName;
    // 2. 模拟用户输入事件，触发 SimpleJekyllSearch 插件进行搜索
    input.dispatchEvent(new Event('input', { bubbles: true }));
    // 3. 页面滚动到结果处 (可选)
    // document.getElementById('results-container').scrollIntoView({ behavior: 'smooth' });
  }
</script>