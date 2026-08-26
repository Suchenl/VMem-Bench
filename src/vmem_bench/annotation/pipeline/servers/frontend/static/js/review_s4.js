(function () {
  'use strict';
  const api = window.MemStrataApi;
  const params = new URLSearchParams(location.search);
  const dataset = params.get('dataset') || '';
  const movieId = params.get('movie_id') || '';
  const PAGE = 12;
  const state = {
    cards: [],
    decisions: {},
    entities: [],
    page: 0,
    nTotalSegments: 0,
    playing: ''
  };

  const el = (id) => document.getElementById(id);
  function toast(msg, kind) {
    const node = el('toast');
    node.hidden = false;
    node.className = 'toast ' + (kind || '');
    node.textContent = msg;
    setTimeout(() => { node.hidden = true; }, 4000);
  }
  function escapeHtml(value) {
    return String(value ?? '').replaceAll('&', '&amp;').replaceAll('<', '&lt;').replaceAll('>', '&gt;').replaceAll('"', '&quot;');
  }

  async function load() {
    el('title').textContent = 'S4 · ' + dataset + '/' + movieId;
    const data = await api.reviewS4(dataset, movieId);
    state.cards = data.cards || [];
    state.entities = data.entities || [];
    state.nTotalSegments = Number(data.n_total_segments || 0);
    const draft = data.draft || {};
    state.decisions = {};
    for (const card of state.cards) {
      const prior = (draft.decisions || {})[card.segment_id] || {};
      state.decisions[card.segment_id] = {
        action: prior.action || 'accept',
        reason: prior.reason || '',
        present_entity_ids: prior.present_entity_ids || (card.revised_present || []).slice(),
        revised_action: prior.revised_action || card.revised_action || ''
      };
    }
    if (draft.film_verdict) el('filmVerdict').value = draft.film_verdict;
    if (draft.reason) el('filmReason').value = draft.reason;
    const sampleText = data.available
      ? ('抽检 ' + state.cards.length + (state.nTotalSegments ? ' / 全片 ' + state.nTotalSegments : '') + ' 个 segment')
      : '无 S4 队列';
    const modeText = data.audit && data.audit.s4_mode_effective
      ? (' · S4→S5=' + data.audit.s4_mode_effective)
      : '';
    el('meta').textContent = sampleText + modeText + (data.audit && data.audit.human_reviewed ? ' · 已人工审核' : ' · 待审核');
    state.page = 0;
    state.playing = '';
    render();
  }

  function pageSlice() {
    const start = state.page * PAGE;
    return state.cards.slice(start, start + PAGE);
  }

  function sameIdSet(a, b) {
    const left = new Set((a || []).map(String));
    const right = new Set((b || []).map(String));
    if (left.size !== right.size) return false;
    for (const id of left) {
      if (!right.has(id)) return false;
    }
    return true;
  }

  /** Auto-pick accept / edit_* from whether action/present differ from the card baseline.
   *  Leaves request_retry / reject_film alone (manual overrides). */
  function syncAutoDecision(segmentId) {
    const card = state.cards.find((c) => c.segment_id === segmentId);
    const item = state.decisions[segmentId];
    if (!card || !item) return item;
    const manual = item.action === 'request_retry' || item.action === 'reject_film';
    if (manual) return item;
    const actionChanged = String(item.revised_action || '') !== String(card.revised_action || '');
    const presentChanged = !sameIdSet(item.present_entity_ids, card.revised_present || []);
    if (actionChanged && presentChanged) item.action = 'edit_both';
    else if (actionChanged) item.action = 'edit_action';
    else if (presentChanged) item.action = 'edit_present';
    else item.action = 'accept';
    return item;
  }

  function mediaBlock(card) {
    if (!card.clip_url) {
      return '<div class="muted">无时间戳，暂无 clip</div>';
    }
    if (state.playing === card.segment_id) {
      return (
        '<div class="clipStage">' +
        '<video class="clipVideo" controls autoplay playsinline preload="auto" src="' + escapeHtml(card.clip_url) + '"></video>' +
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

  function render() {
    const root = el('cards');
    root.innerHTML = '';
    if (!state.cards.length) {
      root.innerHTML = '<div class="empty">没有 S4 review queue。请先跑到 S4（勿勾选跳过人工）。</div>';
      el('pageInfo').textContent = '—';
      return;
    }
    const pages = Math.max(1, Math.ceil(state.cards.length / PAGE));
    if (state.page >= pages) state.page = pages - 1;
    el('pageInfo').textContent = '第 ' + (state.page + 1) + ' / ' + pages + ' 页 · 封面默认显示，点击播放';
    for (const card of pageSlice()) {
      const decision = state.decisions[card.segment_id];
      const div = document.createElement('article');
      div.className = 'reviewCard';
      div.innerHTML = [
        '<div class="clipCol">' + mediaBlock(card) + '</div>',
        '<div class="reviewCardBody">',
        '<h3>' + escapeHtml(card.segment_id) + '</h3>',
        '<div class="muted">' + escapeHtml(String(card.start_seconds ?? '')) + 's – ' + escapeHtml(String(card.end_seconds ?? '')) + 's · verdict=' + escapeHtml(card.verdict || '') + ' · confidence=' + escapeHtml(card.confidence || '') + ' · recommended=' + escapeHtml(card.recommended_action || '') + '</div>',
        '<div class="muted"><strong>Action:</strong> ' + escapeHtml(card.revised_action || '') + '</div>',
        '<div class="muted"><strong>Risk:</strong> ' + escapeHtml((card.risk_reasons || []).join(', ') || '无') + '</div>',
        '<textarea data-revised-action="' + escapeHtml(card.segment_id) + '" rows="3" placeholder="修订 action">' + escapeHtml(decision.revised_action) + '</textarea>',
        '<div class="entityChips" data-seg="' + escapeHtml(card.segment_id) + '"></div>',
        '<div class="decisionRow">',
        '<select data-action="' + escapeHtml(card.segment_id) + '">',
        '<option value="accept">接受当前结果</option>',
        '<option value="edit_action">修改 action</option>',
        '<option value="edit_present">修改 present</option>',
        '<option value="edit_both">同时修改 action/present</option>',
        '<option value="request_retry">请求自动重试</option>',
        '<option value="reject_film">标记为整片问题</option>',
        '</select>',
        '<input data-reason="' + escapeHtml(card.segment_id) + '" placeholder="备注" value="' + escapeHtml(decision.reason) + '">',
        '</div></div>'
      ].join('');
      const chips = div.querySelector('.entityChips');
      const allIds = Array.from(new Set([].concat(card.revised_present || [], decision.present_entity_ids || [], state.entities.map((e) => e.entity_id))));
      for (const eid of allIds.slice(0, 40)) {
        const ent = state.entities.find((e) => e.entity_id === eid) || { entity_id: eid, name: eid, kind: '' };
        const on = decision.present_entity_ids.includes(eid);
        const chip = document.createElement('button');
        chip.type = 'button';
        chip.className = 'entityChip ' + (on ? 'on' : 'off');
        chip.textContent = (ent.kind ? ent.kind[0] + ':' : '') + (ent.name || eid);
        chip.title = eid;
        chip.addEventListener('click', () => {
          const list = state.decisions[card.segment_id].present_entity_ids;
          const idx = list.indexOf(eid);
          if (idx >= 0) list.splice(idx, 1);
          else list.push(eid);
          syncAutoDecision(card.segment_id);
          render();
        });
        chips.appendChild(chip);
      }
      const playBtn = div.querySelector('[data-play]');
      if (playBtn) {
        playBtn.addEventListener('click', () => {
          state.playing = card.segment_id;
          render();
        });
      }
      const video = div.querySelector('video.clipVideo');
      if (video) {
        video.addEventListener('error', () => {
          toast('视频加载失败：' + card.segment_id + '（请确认后端能找到 ffmpeg）', 'fail');
        });
      }
      const select = div.querySelector('[data-action]');
      select.value = decision.action;
      select.addEventListener('change', () => { state.decisions[card.segment_id].action = select.value; });
      const reason = div.querySelector('[data-reason]');
      reason.addEventListener('input', () => { state.decisions[card.segment_id].reason = reason.value; });
      const revisedAction = div.querySelector('[data-revised-action]');
      revisedAction.addEventListener('input', () => {
        const item = state.decisions[card.segment_id];
        item.revised_action = revisedAction.value;
        syncAutoDecision(card.segment_id);
        // Update the select in-place so the caret stays in the textarea.
        select.value = item.action;
      });
      root.appendChild(div);
    }
  }

  function payload() {
    return {
      dataset: dataset,
      movie_id: movieId,
      decisions: state.decisions,
      film_verdict: el('filmVerdict').value,
      reason: el('filmReason').value.trim()
    };
  }

  async function saveDraft() {
    try {
      await api.draftS4(payload());
      toast('草稿已保存', 'ok');
    } catch (err) {
      toast('草稿失败: ' + err.message, 'fail');
    }
  }

  async function save() {
    try {
      const result = await api.applyS4(payload());
      toast('S4 已应用：overrides=' + result.n_overrides, 'ok');
    } catch (err) {
      toast('保存失败: ' + err.message, 'fail');
    }
  }

  async function cont() {
    try {
      const job = await api.continueReview({
        dataset: dataset,
        movie_id: movieId,
        continue_from: 'after_s4',
        grounder: 'qwen',
        s4_mode: 'blocking',
        crop_route: 'propose_and_pick',
        proposer: 'fusion',
        task_mode: 'coverage',
        skip_human: false
      });
      toast('已启动继续任务 ' + job.job_id + '，完成后可进 S6 审核', 'ok');
    } catch (err) {
      toast('继续失败: ' + err.message, 'fail');
    }
  }

  el('draftBtn').addEventListener('click', saveDraft);
  el('saveBtn').addEventListener('click', save);
  el('continueBtn').addEventListener('click', cont);
  el('prevPage').addEventListener('click', () => {
    if (state.page > 0) { state.page -= 1; state.playing = ''; render(); }
  });
  el('nextPage').addEventListener('click', () => {
    const pages = Math.max(1, Math.ceil(state.cards.length / PAGE));
    if (state.page + 1 < pages) { state.page += 1; state.playing = ''; render(); }
  });
  load().catch((err) => toast(err.message, 'fail'));
})();
