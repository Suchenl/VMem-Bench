(function () {
  'use strict';
  const api = window.MemStrataApi;
  const params = new URLSearchParams(location.search);
  const dataset = params.get('dataset') || '';
  const movieId = params.get('movie_id') || '';
  const state = {
    timer: null,
    playing: new Set(),
    playTimes: new Map(),
    cardSig: ''
  };

  const el = (id) => document.getElementById(id);
  function escapeHtml(value) {
    return String(value ?? '')
      .replaceAll('&', '&amp;')
      .replaceAll('<', '&lt;')
      .replaceAll('>', '&gt;')
      .replaceAll('"', '&quot;');
  }
  function toast(msg, kind) {
    const node = el('toast');
    node.hidden = false;
    node.className = 'toast ' + (kind || '');
    node.textContent = msg;
    setTimeout(() => { node.hidden = true; }, 3500);
  }

  function cardsSignature(cards) {
    return (cards || []).map((c) => [
      c.segment_id,
      c.n_rounds,
      c.elapsed_seconds,
      c.queue_seconds,
      c.clip_seconds,
      c.vlm_request_seconds,
      c.vlm_inference_seconds,
      c.accepted ? 1 : 0,
      (c.revised_action || '').length,
      (c.original_action || '').length,
      (c.revised_present || []).join(','),
      (c.original_present || []).join(','),
      (c.risk_reasons || []).join('|')
    ].join(':')).join(';');
  }

  function formatElapsed(seconds) {
    if (seconds == null || seconds === '' || Number.isNaN(Number(seconds))) {
      return '—';
    }
    const s = Number(seconds);
    if (s < 60) return s.toFixed(1) + 's';
    const m = Math.floor(s / 60);
    const rem = s - m * 60;
    return m + 'm' + rem.toFixed(0).padStart(2, '0') + 's';
  }

  function totalTimingHtml(card) {
    const values = [
      card.queue_seconds,
      card.clip_seconds,
      card.vlm_request_seconds,
      card.vlm_inference_seconds
    ];
    const hasBreakdown = values.some((value) => value != null && value !== '');
    const detail = hasBreakdown
      ? [
        '<span>排队：' + escapeHtml(formatElapsed(card.queue_seconds)) + '</span>',
        '<span>裁剪：' + escapeHtml(formatElapsed(card.clip_seconds)) + '</span>',
        '<span>VLM 请求：' + escapeHtml(formatElapsed(card.vlm_request_seconds)) + '</span>',
        '<span>VLM 推理：' + escapeHtml(
          card.vlm_inference_seconds == null || card.vlm_inference_seconds === ''
            ? '—（服务端未上报）'
            : formatElapsed(card.vlm_inference_seconds)
        ) + '</span>'
      ].join('')
      : '<span>该历史记录未采集分项耗时</span>';
    return [
      '<span class="badge timingTotal" tabindex="0">',
      '总耗时 ' + escapeHtml(formatElapsed(card.elapsed_seconds)),
      '<span class="timingTooltip" role="tooltip">',
      '<strong>耗时明细</strong>',
      detail,
      '</span>',
      '</span>'
    ].join('');
  }

  function kindPrefix(kind) {
    const k = String(kind || '').toLowerCase();
    if (k.startsWith('char')) return 'c';
    if (k.startsWith('prop')) return 'p';
    if (k.startsWith('loc')) return 'l';
    return k ? k[0] : '?';
  }

  function entityChipHtml(e) {
    const status = e.status || (e.in_revised === false ? 'removed' : 'kept');
    const label = kindPrefix(e.kind) + ':' + (e.name || e.entity_id);
    const tip = [
      e.entity_id || '',
      status === 'kept' ? '存在且通过' : status === 'removed' ? '存在但未通过/被删' : 'S3 补充',
      e.description || ''
    ].filter(Boolean).join(' · ');
    return (
      '<span class="entityChip ' + escapeHtml(status) + '" title="' + escapeHtml(tip) + '">' +
      escapeHtml(label) +
      '</span>'
    );
  }

  function segmentSortKey(segmentId) {
    const text = String(segmentId || '');
    const match = text.match(/(\d+)\s*$/);
    if (match) return [0, Number(match[1]), text];
    return [1, Number.POSITIVE_INFINITY, text];
  }

  function sortCards(cards) {
    return (cards || []).slice().sort((a, b) => {
      const ka = segmentSortKey(a.segment_id);
      const kb = segmentSortKey(b.segment_id);
      if (ka[0] !== kb[0]) return ka[0] - kb[0];
      if (ka[1] !== kb[1]) return ka[1] - kb[1];
      return String(ka[2]).localeCompare(String(kb[2]));
    });
  }

  function actionCompareHtml(card) {
    const orig = card.original_action || '';
    const rev = card.revised_action || '';
    const changed = Boolean(card.action_changed) || (orig.trim() !== rev.trim() && Boolean(rev.trim()));
    const blocks = [
      '<div class="actionCompare">',
      '<div class="actionBlock original">',
      '<div class="actionLabel">S1/S2 原始 action</div>',
      '<p class="actionText">' + escapeHtml(orig || '（无）') + '</p>',
      '</div>'
    ];
    if (changed) {
      blocks.push(
        '<div class="actionBlock revised">',
        '<div class="actionLabel">S3 修订 action</div>',
        '<p class="actionText">' + escapeHtml(rev || '（无）') + '</p>',
        '</div>'
      );
    }
    blocks.push('</div>');
    return blocks.join('');
  }

  function mediaBlock(card) {
    if (!card.clip_url) {
      return '<div class="muted">无时间戳，暂无 clip</div>';
    }
    if (state.playing.has(card.segment_id)) {
      return (
        '<div class="clipStage">' +
        '<video class="clipVideo" data-segment-id="' + escapeHtml(card.segment_id) +
        '" controls playsinline preload="auto" src="' + escapeHtml(card.clip_url) + '"></video>' +
        '</div>'
      );
    }
    const poster = card.poster_url
      ? ('<img class="clipPoster" src="' + escapeHtml(card.poster_url) + '" alt="' + escapeHtml(card.segment_id) + ' cover" loading="lazy">')
      : '<div class="clipPosterFallback">加载封面中…</div>';
    return (
      '<button type="button" class="clipPlayBtn" data-play="' + escapeHtml(card.segment_id) + '" title="点击播放">' +
      poster +
      '<span class="clipPlayIcon" aria-hidden="true">▶</span>' +
      '</button>'
    );
  }

  function updateProgress(data) {
    const progress = data.progress || {};
    const done = Number(progress.done || data.n_reviews || 0);
    const total = Number(progress.total || 0);
    const pct = total > 0 ? Math.min(100, Math.round(100 * done / total)) : 0;
    el('progressBar').style.width = pct + '%';
    el('progressText').textContent = total
      ? (done + ' / ' + total + ' · ' + pct + '% · phase=' + (progress.phase || '-') + ' · status=' + (progress.status || '-') + (progress.n_shards ? (' · shards=' + progress.n_shards) : ''))
      : ('已写出 ' + done + ' 条 audit（等待 progress.json）');
    el('meta').textContent = data.available
      ? ('实时产物 · 已审核 ' + (data.n_reviews || 0) + ' 段 · 更新于 ' +
         (window.formatBeijingTime ? window.formatBeijingTime(progress.updated_at) : (progress.updated_at || '—')) +
         ' · 封面点击可同时播放多个 clip（播放中暂停刷卡片）')
      : '尚无 S3 实时产物。启动 qwen S3 后这里会自动出现。';
  }

  function bindVideo(video, segmentId) {
    const savedTime = state.playTimes.get(segmentId);
    if (savedTime > 0) {
      video.addEventListener('loadedmetadata', () => {
        try { video.currentTime = savedTime; } catch (_) {}
      }, { once: true });
    }
    video.addEventListener('timeupdate', () => {
      state.playTimes.set(segmentId, video.currentTime || 0);
    });
    video.addEventListener('pause', () => {
      state.playTimes.set(segmentId, video.currentTime || 0);
    });
    video.addEventListener('ended', () => {
      state.playing.delete(segmentId);
      state.playTimes.delete(segmentId);
      if (!state.playing.size) state.cardSig = '';
    });
    video.addEventListener('error', () => {
      state.playing.delete(segmentId);
      state.playTimes.delete(segmentId);
      if (!state.playing.size) state.cardSig = '';
      toast('视频加载失败：' + segmentId, 'fail');
    });
  }

  function activateVideo(card, clipCol) {
    state.playing.add(card.segment_id);
    state.playTimes.set(card.segment_id, 0);
    clipCol.innerHTML = mediaBlock(card);
    const video = clipCol.querySelector('video.clipVideo');
    if (!video) return;
    bindVideo(video, card.segment_id);
    // This is called directly from the user's click handler, so browsers allow
    // each selected clip to start without stopping the clips already playing.
    video.play().catch(() => {});
  }

  function renderCards(cards) {
    const root = el('cards');
    root.querySelectorAll('video.clipVideo').forEach((video) => {
      const segmentId = video.dataset.segmentId;
      if (segmentId) state.playTimes.set(segmentId, video.currentTime || 0);
    });
    root.innerHTML = '';
    const ordered = sortCards(cards);
    if (!ordered.length) {
      root.innerHTML = '<div class="empty">还没有写完的 segment。稍等刷新。</div>';
      return;
    }
    for (const card of ordered) {
      const div = document.createElement('article');
      div.className = 'reviewCard';
      const verdict = card.verdict || (card.accepted ? 'PASS' : 'WARN');
      const entityChips = (card.entities || []).map(entityChipHtml).join('')
        || '<span class="muted">无 present 实体</span>';
      div.innerHTML = [
        '<div class="clipCol">' + mediaBlock(card) + '</div>',
        '<div class="reviewCardBody">',
        '<div class="reviewCardHead"><strong>' + escapeHtml(card.segment_id) + '</strong>',
        '<span class="badge ' + (verdict === 'PASS' ? 'ok' : 'warn') + '">' + escapeHtml(verdict) + '</span>',
        '<span class="badge">' + escapeHtml(card.confidence || '') + '</span>',
        '<span class="badge">rounds=' + escapeHtml(String(card.n_rounds || 1)) + '</span>',
        totalTimingHtml(card),
        '</div>',
        '<div class="muted">' + escapeHtml(String(card.start_seconds ?? '')) + 's – ' + escapeHtml(String(card.end_seconds ?? '')) + 's</div>',
        '<div class="entityChips">' + entityChips + '</div>',
        actionCompareHtml(card),
        (card.risk_reasons && card.risk_reasons.length)
          ? ('<div class="muted">risk: ' + escapeHtml(card.risk_reasons.join(' | ')) + '</div>')
          : '',
        card.recommended_action && card.recommended_action !== 'none'
          ? ('<div class="muted">recommended: ' + escapeHtml(card.recommended_action) + '</div>')
          : '',
        '</div>'
      ].join('');
      const playBtn = div.querySelector('[data-play]');
      if (playBtn) {
        playBtn.addEventListener('click', () => {
          activateVideo(card, div.querySelector('.clipCol'));
        });
      }
      const video = div.querySelector('video.clipVideo');
      if (video) bindVideo(video, card.segment_id);
      root.appendChild(div);
    }
  }

  async function load() {
    el('title').textContent = 'S3 Live · ' + dataset + '/' + movieId;
    let data;
    try {
      data = await api.reviewS3(dataset, movieId);
    } catch (err) {
      el('meta').textContent = '加载失败: ' + err.message;
      toast(err.message, 'fail');
      return;
    }
    updateProgress(data);
    const cards = data.cards || [];
    const sig = cardsSignature(cards);
    // While any clip is open, only refresh progress — rebuilding <video>
    // elements would interrupt one or more simultaneous playbacks.
    if (state.playing.size) {
      return;
    }
    if (sig === state.cardSig && el('cards').children.length) {
      return;
    }
    state.cardSig = sig;
    renderCards(cards);
  }

  function schedule() {
    clearInterval(state.timer);
    if (el('autoRefresh').checked) {
      state.timer = setInterval(() => { load().catch(() => {}); }, 3000);
    }
  }

  if (!api || typeof api.reviewS3 !== 'function') {
    el('meta').textContent = '前端脚本未加载完整（MemStrataApi 缺失）。请硬刷新，或检查 /static/js/api.js 是否被代理拦下。';
    return;
  }
  el('refreshBtn').addEventListener('click', () => {
    state.cardSig = '';
    load().catch((e) => toast(e.message, 'fail'));
  });
  el('autoRefresh').addEventListener('change', schedule);

  async function rerunThisSample() {
    if (!dataset || !movieId) {
      toast('缺少 dataset / movie_id', 'fail');
      return;
    }
    const ok = window.confirm(
      '确认重跑？\n\n将清空 ' + dataset + '/' + movieId + ' 的 tmp/pipeline，并从 S2 从头开始。此操作不可撤销。'
    );
    if (!ok) return;
    try {
      const job = await api.createJob({
        samples: [{ dataset: dataset, movie_id: movieId }],
        force_restart: true,
        resume: false,
        reviewer: 'qwen',
        grounder: 'qwen',
        s4_mode: 'blocking',
        crop_route: 'propose_and_pick',
        proposer: 'fusion',
        task_mode: 'coverage',
        skip_human: false,
        max_review_rounds: 2
      });
      toast('已提交重跑 ' + (job.job_id || '') + '，可回控制台看 Jobs', 'ok');
    } catch (err) {
      toast('重跑失败: ' + err.message, 'fail');
    }
  }
  const rerunBtn = el('rerunBtn');
  if (rerunBtn) rerunBtn.addEventListener('click', () => {
    rerunThisSample().catch((e) => toast(e.message, 'fail'));
  });

  load().then(schedule).catch((e) => toast(e.message, 'fail'));
})();
