---
layout: default
title: Bolg
---

<section class="intro-section">
  <h1 class="intro-title">👋 Welcome to YXY.</h1>
  <p class="intro-text">
    Hi, I am YuxuanYan. This is my digital garden where I document my learning notes on <b>AI, Data Science, and Prompt Engineering</b>.
  </p>
  <p class="intro-text" style="font-size: 0.95em;">
    Based on the style of this blog, you can tell I am a big fan of Lil'Log 😉.
  </p>
</section>

<section class="posts-section">
  <ul class="post-list">
    {% for post in site.posts %}
      <li>
        <article class="post-card">
          <div class="post-meta">
            {{ post.date | date: "%B %d, %Y" }}
          </div>
          <h2 class="post-card-title">
            <a href="{{ post.url | relative_url }}">{{ post.title }}</a>
          </h2>
          <p class="post-card-excerpt">
            {{ post.content | strip_html | truncatewords: 35 }}
          </p>
        </article>
      </li>
    {% endfor %}
  </ul>
</section>