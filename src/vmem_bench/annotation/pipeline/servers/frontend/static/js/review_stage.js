(function () {
  'use strict';
  const api = window.MemStrataApi;
  const params = new URLSearchParams(location.search);
  const dataset = params.get('dataset') || '';
  const movieId = params.get('movie_id') || '';
  let stage = params.get('stage') || 's1';
  let refreshTimer = null;
  let loading = false;

  const el = (id) => document.getElementById(id);
  function escapeHtml(value) {
    return String(value == null ? '' : value)
      .replace(/&/g, '&amp;')
      .replace(/</g, '&lt;')
      .replace(/>/g, '&gt;')
      .replace(/"/g, '&quot;');
  }
  function toast(msg, kind) {
    const node = el('toast');
    if (!node) return;
    node.hidden = false;
    node.className = 'toast ' + (kind || '');
    node.textContent = msg;
    setTimeout(() => { node.hidden = true; }, 3500);
  }

  function stageHref(name) {
    return '/review/stage.html?dataset=' + encodeURIComponent(dataset) +
      '&movie_id=' + encodeURIComponent(movieId) +
      '&stage=' + encodeURIComponent(name);
  }

  function renderTabs(rows) {
    el('stageTabs').innerHTML = (rows || []).map((row) => {
      const active = row.stage === stage || row.short.toLowerCase() === String(stage).toLowerCase();
      const cls = 'buttonLink reviewLink' + (active ? ' active' : '');
      const style = row.available ? '' : 'opacity:0.45;pointer-events:none';
      return '<a class="' + cls + '" style="' + style + '" href="' + stageHref(row.stage) + '">' +
        escapeHtml(row.label || row.short) +
        (row.available ? '' : ' · 无产物') + '</a>';
    }).join(' ');
  }

  function isS5(stageName) {
    return stageName === 's5_entities_visual_crop_acquisition' || stageName === 's5';
  }

  function syncRerunVisibility() {
    const s5 = isS5(stage);
    const rerunS5 = el('rerunS5Btn');
    const rerunAll = el('rerunBtn');
    if (rerunS5) rerunS5.hidden = !s5;
    if (rerunAll) rerunAll.hidden = !s5;
  }

  function renderProgress(summary) {
    const panel = el('progressPanel');
    const wrap = el('autoRefreshWrap');
    if (!panel) return false;
    const progress = (summary || {}).progress || {};
    const done = Number(progress.done || 0);
    const total = Number(progress.total || summary.n_tasks || 0);
    const accepted = Number(progress.accepted || 0);
    const status = String(progress.status || '');
    const phase = String(progress.phase || '');
    const running = status === 'running' || (total > 0 && done < total && status !== 'completed' && status !== 'failed');
    const hasProgress = total > 0 || status || phase || done > 0;
    panel.hidden = !hasProgress;
    if (wrap) wrap.hidden = !isS5(stage);
    if (!hasProgress) return false;

    const pct = total > 0 ? Math.min(100, Math.round(100 * done / total)) : Number(progress.pct || 0);
    el('progressBar').style.width = pct + '%';
    const current = progress.current || {};
    const currentLabel = current.name
      ? (String(current.name) + (current.entity_id ? ' (' + current.entity_id + ')' : ''))
      : (current.entity_id || '—');
    const byKind = progress.by_kind || {};
    const kindBits = Object.keys(byKind).map((k) => k + '=' + byKind[k]).join(' · ');
    const updated = progress.updated_at
      ? (window.formatBeijingTime ? window.formatBeijingTime(progress.updated_at) : progress.updated_at)
      : '—';
    el('progressText').textContent = [
      total ? (done + ' / ' + total + ' · ' + pct + '%') : ('已完成 ' + done),
      'accepted=' + accepted,
      phase ? ('phase=' + phase) : '',
      status ? ('status=' + status) : '',
      'current=' + currentLabel,
      kindBits ? ('kinds: ' + kindBits) : '',
      '更新 ' + updated
    ].filter(Boolean).join(' · ');
    return running;
  }

  function renderEntityGroups(groups) {
    const root = el('highlights');
    root.className = 'entityGroups';
    root.innerHTML = '';
    if (!groups.length) {
      root.innerHTML = '<div class="empty">尚无 crop；模型加载或首个 task 完成后会出现分组预览。</div>';
      return;
    }
    for (const group of groups) {
      const section = document.createElement('section');
      section.className = 'entityGroup panel';
      const crops = group.crops || [];
      const grid = crops.length
        ? '<div class="cropGrid">' + crops.map((crop) => {
            const img = crop.image_url
              ? '<img loading="lazy" src="' + escapeHtml(crop.image_url) + '" alt="">'
              : '<div class="emptySmall">无图</div>';
            const badge = crop.accepted ? 'ok' : 'warn';
            return (
              '<article class="cropTile">' +
              img +
              '<div class="cropTileBody">' +
              '<strong>' + escapeHtml(crop.title || '') + '</strong>' +
              '<span class="badge ' + badge + '">' + (crop.accepted ? 'accepted' : 'rejected') + '</span>' +
              '<div class="muted">' + escapeHtml(crop.meta || '') + '</div>' +
              '</div></article>'
            );
          }).join('') + '</div>'
        : '<div class="hint">该实体尚无 crop 图</div>';
      section.innerHTML =
        '<div class="entityGroupHead">' +
        '<div><h3>' + escapeHtml(group.name || group.entity_id || '') + '</h3>' +
        '<div class="muted">' + escapeHtml(group.kind || '') + ' · ' + escapeHtml(group.entity_id || '') + '</div></div>' +
        '<div class="entityGroupStats">' +
        '<span class="badge">' + Number(group.n_accepted || 0) + ' accepted</span> ' +
        '<span class="badge">' + Number(group.n_done || 0) + ' tasks</span> ' +
        '<span class="badge">' + Number(group.n_with_crop || crops.length) + ' crops</span>' +
        '</div></div>' + grid;
      root.appendChild(section);
    }
  }

  function renderFlatHighlights(highlights) {
    const root = el('highlights');
    root.className = 'reviewCards';
    root.innerHTML = '';
    if (!highlights.length) {
      root.innerHTML = '<div class="empty">没有可展示条目；可查看上方摘要与产物文件。</div>';
      return;
    }
    for (const item of highlights) {
      const card = document.createElement('article');
      card.className = 'reviewCard';
      const img = item.image_url
        ? '<img loading="lazy" src="' + escapeHtml(item.image_url) + '" alt="">'
        : '<div class="emptySmall">无图</div>';
      card.innerHTML =
        '<div>' + img + '</div>' +
        '<div class="reviewCardBody">' +
        '<h3>' + escapeHtml(item.title || '') + '</h3>' +
        '<div class="muted">' + escapeHtml(item.meta || '') + '</div>' +
        '<div class="muted">' + escapeHtml(item.text || '') + '</div>' +
        '</div>';
      root.appendChild(card);
    }
  }

  function scheduleRefresh(running) {
    clearInterval(refreshTimer);
    refreshTimer = null;
    if (!isS5(stage)) return;
    const auto = el('autoRefresh');
    if (auto && !auto.checked) return;
    if (!running) return;
    refreshTimer = setInterval(() => { load().catch(() => {}); }, 3000);
  }

  async function load() {
    if (loading) return;
    loading = true;
    try {
      el('title').textContent = stage.toUpperCase() + ' · ' + dataset + '/' + movieId;
      const data = await api.stageInspect(dataset, movieId, stage);
      if (data.stage) stage = data.stage;
      el('title').textContent = (data.short || stage) + ' · ' + dataset + '/' + movieId;
      el('meta').textContent =
        (data.available ? '可检视' : '尚无产物') +
        ' · ' + (data.label || data.stage) +
        ' · ' + (data.stage_dir || '');
      syncRerunVisibility();
      renderTabs(data.stage_reviews || []);
      const summary = data.summary || {};
      // Keep summary readable: drop bulky entity_groups dump if present.
      const summaryView = Object.assign({}, summary);
      delete summaryView.entity_groups;
      el('summary').textContent = JSON.stringify(summaryView, null, 2);
      const files = data.files || [];
      el('files').innerHTML = files.length
        ? '<ul>' + files.map((f) => '<li><code>' + escapeHtml(f.name) + '</code> · ' + escapeHtml(f.bytes) + ' B</li>').join('') + '</ul>'
        : '<span class="hint">无文件</span>';

      const deep = el('deepLink');
      if (data.deep_link) {
        deep.hidden = false;
        deep.href = data.deep_link;
        deep.textContent = data.kind === 's3' ? '打开 S3 实时审核'
          : data.kind === 's4' ? '打开 S4 审核'
            : data.kind === 's6' ? '打开 S6 审核'
              : '打开专用审核页';
      } else {
        deep.hidden = true;
      }

      const running = renderProgress(summary);
      const groups = data.entity_groups || summary.entity_groups || [];
      // S5 live groups + S7 gold groups both use entity grid; other stages stay flat.
      if (groups.length) {
        renderEntityGroups(groups);
      } else if (isS5(stage)) {
        renderEntityGroups([]);
      } else {
        renderFlatHighlights(data.highlights || []);
      }
      scheduleRefresh(running);
    } finally {
      loading = false;
    }
  }

  const refreshBtn = el('refreshBtn');
  if (refreshBtn) refreshBtn.addEventListener('click', () => load().catch((err) => toast(err.message, 'fail')));
  const auto = el('autoRefresh');
  if (auto) {
    auto.addEventListener('change', () => {
      if (auto.checked) load().catch(() => {});
      else clearInterval(refreshTimer);
    });
  }

  async function rerunS5() {
    if (!dataset || !movieId) {
      toast('缺少 dataset / movie_id', 'fail');
      return;
    }
    const ok = window.confirm(
      '确认重跑 S5？\n\n将清空 ' + dataset + '/' + movieId +
      ' 的 S5/S6 产物，保留 S2–S4，并从 crop 采集重新开始。'
    );
    if (!ok) return;
    try {
      const job = await api.continueReview({
        dataset: dataset,
        movie_id: movieId,
        continue_from: 'after_s4',
        rerun_s5: true,
        grounder: 'qwen',
        s4_mode: 'blocking',
        crop_route: 'propose_and_pick',
        proposer: 'fusion',
        task_mode: 'coverage',
        skip_human: false
      });
      toast('已提交重跑 S5 ' + (job.job_id || '') + '，可回控制台看 Jobs', 'ok');
    } catch (err) {
      toast('重跑 S5 失败: ' + err.message, 'fail');
    }
  }

  async function rerunThisSample() {
    if (!dataset || !movieId) {
      toast('缺少 dataset / movie_id', 'fail');
      return;
    }
    const ok = window.confirm(
      '确认重跑整样本？\n\n将清空 ' + dataset + '/' + movieId +
      ' 的 tmp/pipeline，并从 S2 从头开始。此操作不可撤销。'
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
  const rerunS5Btn = el('rerunS5Btn');
  if (rerunS5Btn) {
    rerunS5Btn.addEventListener('click', () => {
      rerunS5().catch((err) => toast(err.message, 'fail'));
    });
  }
  const rerunBtn = el('rerunBtn');
  if (rerunBtn) {
    syncRerunVisibility();
    rerunBtn.addEventListener('click', () => {
      rerunThisSample().catch((err) => toast(err.message, 'fail'));
    });
  }

  load().catch((err) => toast(err.message, 'fail'));
})();
