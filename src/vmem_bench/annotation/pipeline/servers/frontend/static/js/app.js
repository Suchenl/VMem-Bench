(function () {
  'use strict';
  const api = window.MemStrataApi;
  const PIPELINE_STAGES = [
    's1_vlm_annotation',
    's2_annotation_postprocess',
    's3_segment_auto_review_revise',
    's4_segment_sampling_human_review',
    's5_entities_visual_crop_acquisition',
    's6_entities_visual_crop_human_review',
    's7_freeze_publish'
  ];
  const STAGE_SHORT = {
    s1_vlm_annotation: 'S1',
    s2_annotation_postprocess: 'S2',
    s3_segment_auto_review_revise: 'S3',
    s4_segment_sampling_human_review: 'S4',
    s5_entities_visual_crop_acquisition: 'S5',
    s6_entities_visual_crop_human_review: 'S6',
    s7_freeze_publish: 'S7'
  };

  const state = {
    samples: [],
    selected: new Set(),
    jobs: [],
    sampleActivity: {},
    sampleCounts: null,
    jobsFilter: 'active',
    activeJobId: null,
    detail: null
  };

  const el = (id) => document.getElementById(id);
  const formatBeijingTime = window.formatBeijingTime || function (v) {
    return v == null || v === '' ? '—' : String(v);
  };
  function on(id, event, handler) {
    const node = el(id);
    if (!node) return;
    node.addEventListener(event, handler);
  }
  function toast(msg, kind) {
    const node = el('toast');
    if (!node) return;
    node.hidden = false;
    node.className = 'toast ' + (kind || '');
    node.textContent = msg;
    const ms = kind === 'fail' ? 8000 : 3500;
    setTimeout(() => { node.hidden = true; }, ms);
  }

  function escapeHtml(value) {
    return String(value == null ? '' : value)
      .replace(/&/g, '&amp;')
      .replace(/</g, '&lt;')
      .replace(/>/g, '&gt;')
      .replace(/"/g, '&quot;');
  }

  function numberOrNull(value) {
    const text = String(value || '').trim();
    if (!text) return null;
    const n = Number(text);
    return Number.isFinite(n) ? n : null;
  }

  function selectionKey(sample) {
    return sample.dataset + '::' + sample.movie_id;
  }

  function setText(id, text) {
    const node = el(id);
    if (node) node.textContent = text;
  }

  function updateSelectionHint() {
    setText('selectionHint', '已选 ' + state.selected.size + ' 个');
  }

  function showSampleError(err) {
    const node = el('sampleList');
    if (!node) return;
    node.className = 'datasetSections';
    node.innerHTML = '<div class="errorText">样本列表加载失败: ' +
      escapeHtml((err && err.message) || String(err)) +
      '<br><span class="hint">可点「刷新」重试；若刚打开 S6/检视页，图片请求可能暂时挤占 API。</span></div>';
  }

  function statusBadgeClass(status) {
    const s = String(status || '');
    if (s === 'gold_ready' || s === 'ok' || s === 'done' || s === 'completed' || s === 'succeeded') return 'ok';
    if (s === 'in_progress' || s === 'running' || s === 'awaiting_human' || s === 'needs_human' || s === 'stopping') return 'warn';
    if (s === 'queued') return 'ok';
    if (s === 'failed' || s === 'error' || s === 'missing_source' || s === 's1_missing' || s === 's1_incomplete') return 'danger';
    return '';
  }

  function sampleRunInfo(sample) {
    const activity = state.sampleActivity[selectionKey(sample)];
    if (!activity) return null;
    const jobStatus = String(activity.job_status || '');
    // Per-movie state inside a multi-movie job: trust backend sample_state
    // (progress.json / live stage). Do NOT mirror job_status onto every row —
    // one running batch still has many queued movies waiting their turn.
    let sampleState = String(activity.sample_state || '');
    if (!sampleState) {
      if (jobStatus === 'queued') sampleState = 'queued';
      else if (jobStatus === 'stopping') sampleState = 'stopping';
      else sampleState = 'running';
    }
    return {
      job_id: activity.job_id,
      job_status: jobStatus,
      sample_state: sampleState,
      n_samples: Number(activity.n_samples || 0)
    };
  }

  function sampleStateLabel(sampleState) {
    if (sampleState === 'queued') return '在排队';
    if (sampleState === 'stopping') return '停止中';
    if (sampleState === 'running') return '在跑';
    return sampleState || '—';
  }

  function jobStatusLabel(status) {
    if (status === 'queued') return '在排队';
    if (status === 'running') return '在跑';
    if (status === 'stopping') return '停止中';
    if (status === 'succeeded') return '成功';
    if (status === 'failed') return '失败';
    if (status === 'stopped') return '已停止';
    return status || '—';
  }

  function stageBadgeClass(entry) {
    const status = String((entry && entry.status) || '');
    if (!status) return '';
    if (status === 'ok' || status === 'done' || status === 'completed' || status === 'skipped' ||
        status === 'human_reviewed' || status === 'human_reviewed_complete') return 'ok';
    if (status === 'running' || status === 'in_progress' || status === 'awaiting_human') return 'warn';
    if (status === 'failed' || status === 'error') return 'danger';
    return 'warn';
  }

  function shortStage(name) {
    return STAGE_SHORT[name] || String(name || '').split('_')[0].toUpperCase() || '—';
  }

  function stageReviewHref(sample, stageName) {
    const review = sample.review || {};
    const rows = review.stage_reviews || [];
    const row = rows.find((r) => r.stage === stageName);
    if (row && !row.available) return '';
    const q = 'dataset=' + encodeURIComponent(sample.dataset) + '&movie_id=' + encodeURIComponent(sample.movie_id);
    if (stageName === 's3_segment_auto_review_revise' && (review.s3_available || review.s3_live_available)) {
      return '/review/s3.html?' + q;
    }
    if (stageName === 's4_segment_sampling_human_review' && review.s4_available) {
      return '/review/s4.html?' + q;
    }
    if (stageName === 's6_entities_visual_crop_human_review' && review.s6_available) {
      return '/review/s6.html?' + q;
    }
    if (row && row.available) {
      return '/review/stage.html?' + q + '&stage=' + encodeURIComponent(stageName);
    }
    return '';
  }

  function renderStageCell(sample) {
    const stages = sample.stages || {};
    const current = sample.current_stage;
    const currentStatus = sample.current_status || 'not_started';
    const done = Number(sample.completed_stage_count || 0);
    const total = Number(sample.stage_count || PIPELINE_STAGES.length);
    const title = current
      ? shortStage(current) + ' · ' + currentStatus
      : (sample.has_vlm_output ? '就绪 · 未开跑' : '未开始');
    const badges = PIPELINE_STAGES.map((name) => {
      const entry = stages[name];
      const cls = stageBadgeClass(entry);
      const tip = name + (entry && entry.status ? ': ' + entry.status : ': —');
      const href = stageReviewHref(sample, name);
      const label = escapeHtml(shortStage(name));
      if (href) {
        return '<a class="badge ' + cls + '" href="' + href + '" title="' + escapeHtml(tip + ' · 打开审核/检视') + '">' + label + '</a>';
      }
      return '<span class="badge ' + cls + '" title="' + escapeHtml(tip) + '">' + label + '</span>';
    }).join('');
    return (
      '<div class="stage">' +
      '<strong>' + escapeHtml(title) + '</strong>' +
      '<span>' + done + ' / ' + total + (current ? ' · ' + escapeHtml(current) : '') + '</span>' +
      '<div class="rowActions" style="margin-top:4px">' + badges + '</div>' +
      '</div>'
    );
  }

  function reviewLinksHtml(sample) {
    const review = sample.review || {};
    const q = 'dataset=' + encodeURIComponent(sample.dataset) + '&movie_id=' + encodeURIComponent(sample.movie_id);
    const rows = review.stage_reviews || [
      { stage: 's1_vlm_annotation', short: 'S1', available: !!sample.has_vlm_output, kind: 'inspect', label: 'S1' },
      { stage: 's2_annotation_postprocess', short: 'S2', available: !!review.s2_available, kind: 'inspect', label: 'S2' },
      { stage: 's3_segment_auto_review_revise', short: 'S3', available: !!(review.s3_available || review.s3_live_available), kind: 's3', label: 'S3' },
      { stage: 's4_segment_sampling_human_review', short: 'S4', available: !!review.s4_available, kind: 's4', label: 'S4' },
      { stage: 's5_entities_visual_crop_acquisition', short: 'S5', available: !!review.s5_available, kind: 'inspect', label: 'S5' },
      { stage: 's6_entities_visual_crop_human_review', short: 'S6', available: !!review.s6_available, kind: 's6', label: 'S6' },
      { stage: 's7_freeze_publish', short: 'S7', available: !!(review.s7_available || sample.has_gold), kind: 'inspect', label: 'S7' }
    ];
    const links = rows.map((row) => {
      if (!row.available) {
        return '<span class="badge" title="尚无产物">' + escapeHtml(row.short) + '</span>';
      }
      let href = '/review/stage.html?' + q + '&stage=' + encodeURIComponent(row.stage);
      if (row.kind === 's3') href = '/review/s3.html?' + q;
      if (row.kind === 's4') href = '/review/s4.html?' + q;
      if (row.kind === 's6') href = '/review/s6.html?' + q;
      let label = row.short;
      if (row.short === 'S3') {
        const sp = review.s3_progress || {};
        const done = Number(sp.done || 0);
        const total = Number(sp.total || 0);
        if (total > 0) label = 'S3 ' + done + '/' + total;
        else if (sp.status === 'running' || ((sample.stages || {}).s3_segment_auto_review_revise || {}).status === 'running') {
          label = 'S3…';
        }
      }
      if (row.short === 'S5') {
        const sp = review.s5_progress || {};
        const done = Number(sp.done || 0);
        const total = Number(sp.total || 0);
        if (total > 0) label = 'S5 ' + done + '/' + total;
        else if (sp.status === 'running' || ((sample.stages || {}).s5_entities_visual_crop_acquisition || {}).status === 'running') {
          label = 'S5…';
        }
      }
      return '<a class="buttonLink reviewLink" href="' + href + '" title="' + escapeHtml(row.label || row.stage) + '">' +
        escapeHtml(label) + '</a>';
    });
    if (review.awaiting_human) links.push('<span class="badge warn">待人工</span>');
    return links.join(' ');
  }

  function resumeAction(sample) {
    const review = sample.review || {};
    const stages = sample.stages || {};
    const s5 = stages.s5_entities_visual_crop_acquisition || {};
    const s6 = stages.s6_entities_visual_crop_human_review || {};
    if (s5.status === 'running') {
      const sp = review.s5_progress || {};
      const done = Number(sp.done || 0);
      const total = Number(sp.total || 0);
      const label = total > 0 ? ('S5 运行中 ' + done + '/' + total) : 'S5 运行中';
      return { label: label, disabled: true };
    }
    if (s6.status === 'running') return { label: 'S6 运行中', disabled: true };
    // Older interrupted S5 attempts may have written planning files, making
    // s5_available true without ever reaching S6.  S6 availability, not a
    // partial S5 directory, is the completion boundary for this resume action.
    if (review.s4_human_reviewed && !review.s6_available && s5.status !== 'ok') {
      return {
        label: s5.status === 'failed' ? '重试 S5' : '续跑 S5',
        continueFrom: 'after_s4'
      };
    }
    if (review.s6_human_reviewed && !review.s7_available) {
      return { label: '续跑 S7', continueFrom: 'after_s6' };
    }
    return null;
  }

  function jobFormBody(samples, forceRestart, resume) {
    const modelName = el('reviewerModel').value.trim() || 'qwen3-vl-32b';
    const body = {
      samples: samples,
      execution_target: el('executionTarget').value || 'remote',
      reviewer: el('reviewer').value,
      reviewer_base_url: el('reviewerBaseUrl').value.trim(),
      reviewer_model: modelName,
      grounder_model: modelName,
      max_review_rounds: numberOrNull(el('maxReviewRounds').value) || 2,
      grounder: el('grounder').value,
      s4_mode: el('s4Mode').value || 'auto',
      crop_route: 'propose_and_pick',
      proposer: el('proposer').value || 'fusion',
      task_mode: 'coverage',
      skip_human: el('skipHuman').checked,
      force_restart: !!forceRestart,
      resume: !!resume && !forceRestart,
      // The "自动接受 S4" checkbox is authoritative for both 续跑 (resume) and
      // 重跑 (force_restart): whatever S4 queue this run produces is accepted by
      // the S3 revision. Previously this was gated on !!resume && !forceRestart,
      // so 重跑 silently ignored the checkbox and always stalled at S4.
      auto_accept_s4: el('autoAcceptS4').checked
    };
    const maxTasks = numberOrNull(el('maxTasks').value);
    if (maxTasks != null) body.max_tasks = maxTasks;
    const maxParallel = numberOrNull(el('maxParallelMovies') && el('maxParallelMovies').value);
    if (maxParallel != null) body.max_parallel_movies = maxParallel;
    return body;
  }

  function validateJobForm(body) {
    // qwen with empty URL is OK — backend resolves from runtime/services/vlm_fleet.
    if (body.reviewer === 'qwen' && body.reviewer_base_url) {
      return '';
    }
    return '';
  }

  function renderFleet(data) {
    const list = el('fleetList');
    const hint = el('fleetHint');
    if (!list) return;
    const rows = (data && data.instances) || [];
    const online = Number((data && data.online_count) || 0);
    const busy = Number((data && data.busy_count) || 0);
    const idle = Number((data && data.idle_count) || 0);
    const paused = Number((data && data.break_count) || 0);
    const broke = Number((data && data.broke_count) || 0);
    const total = Number((data && data.total_count) || 0);
    const starting = rows.filter((row) => String(row.console_status || '') === 'starting').length;
    const activeRows = rows.filter((row) => String(row.console_status || '') !== 'broke');
    const archivedRows = rows.filter((row) => !activeRows.includes(row));
    if (hint) {
      hint.textContent = online
        ? ('在线 ' + online + ' · 忙碌 ' + busy + ' · 空闲 ' + idle +
           (starting ? (' · 启动中 ' + starting) : '') +
           (paused ? (' · break ' + paused) : '') +
           (broke ? (' · broke ' + broke) : '') +
           (archivedRows.length ? (' · 已折叠 ' + archivedRows.length + ' 条异常/历史登记') : '') +
           ' · 北京时间')
        : ('在线 0' + (starting ? (' · 启动中 ' + starting) : '') +
           (paused ? (' · break ' + paused) : '') +
           (broke ? (' · broke ' + broke) : '') +
           (archivedRows.length ? (' · 已折叠 ' + archivedRows.length + ' 条异常/历史登记') : '') +
           ' · 先用 start_reviewer_pool.sh 起 VLM');
    }
    if (!rows.length) {
      list.className = 'fleetList hint';
      list.textContent = '尚无 fleet 登记（runtime/services/vlm_fleet 为空）';
      return;
    }
    list.className = 'fleetList';
    const renderRows = (groupRows) => groupRows.map((row) => {
      const label = row.console_status || row.status || 'broke';
      const cls = label === 'idle' ? 'ok' :
        (label === 'busy' ? 'busy' : (label === 'broke' ? 'danger' : 'warn'));
      const wl = row.workload || {};
      const workBits = [
        wl.movie_id || '',
        wl.segment_id || '',
        wl.stage || '',
        wl.leased_at ? ('@' + formatBeijingTime(wl.leased_at)) : ''
      ].filter(Boolean).join(' · ');
      const detail = [
        row.display_name || '',
        (row.host && row.port) ? (row.host + ':' + row.port) : '',
        workBits ? ('工作中: ' + workBits) : '',
        row.break && row.break.reason ? ('暂停: ' + row.break.reason) : '',
        row.heartbeat_at ? ('hb ' + formatBeijingTime(row.heartbeat_at)) : ''
      ].filter(Boolean).join(' · ');
      return (
        '<div class="fleetRow">' +
        '<span class="badge ' + cls + '">' + escapeHtml(label) + '</span>' +
        '<span title="' + escapeHtml(row.base_url || '') + '">' + escapeHtml(detail) + '</span></div>'
      );
    }).join('');
    const byCluster = {};
    activeRows.forEach((row) => {
      const name = String(row.cluster || '其它已启用端点');
      (byCluster[name] || (byCluster[name] = [])).push(row);
    });
    const activeMarkup = Object.keys(byCluster).sort((left, right) => {
      const preferredOrder = ['gpu-h800', 'gpu-a800', 'bdy-a800'];
      const leftPreferred = preferredOrder.indexOf(left);
      const rightPreferred = preferredOrder.indexOf(right);
      if (leftPreferred !== -1 || rightPreferred !== -1) {
        return (leftPreferred === -1 ? preferredOrder.length : leftPreferred) -
          (rightPreferred === -1 ? preferredOrder.length : rightPreferred);
      }
      const leftRow = byCluster[left][0] || {};
      const rightRow = byCluster[right][0] || {};
      return Number(leftRow.cluster_order == null ? 10000 : leftRow.cluster_order) -
        Number(rightRow.cluster_order == null ? 10000 : rightRow.cluster_order) ||
        left.localeCompare(right);
    }).map((cluster) => {
      const byNode = {};
      byCluster[cluster].forEach((row) => {
        const key = String(row.node_id || '?');
        (byNode[key] || (byNode[key] = [])).push(row);
      });
      const nodeMarkup = Object.keys(byNode).sort((left, right) => {
        const leftRow = byNode[left][0] || {};
        const rightRow = byNode[right][0] || {};
        return Number(leftRow.node_order == null ? 10000 : leftRow.node_order) -
          Number(rightRow.node_order == null ? 10000 : rightRow.node_order) ||
          left.localeCompare(right, undefined, { numeric: true });
      }).map((node) => {
        const groupRows = byNode[node].slice().sort((left, right) => {
          return Number(left.gpu_rank == null ? 10000 : left.gpu_rank) -
            Number(right.gpu_rank == null ? 10000 : right.gpu_rank) ||
            String(left.service_name || '').localeCompare(String(right.service_name || ''));
        });
        const nodeMeta = groupRows[0].node_meta || {};
        const nodeLabel = 'node' + node +
          (nodeMeta.role ? (' · ' + nodeMeta.role) : '') +
          (nodeMeta.host ? (' · ' + nodeMeta.host) : '');
        return '<div class="fleetNode"><strong>' + escapeHtml(nodeLabel) +
          '</strong>' + renderRows(groupRows) + '</div>';
      }).join('');
      const onlineInGroup = byCluster[cluster].filter((row) => row.online).length;
      return '<section class="fleetGroup"><strong>' + escapeHtml(cluster) +
        ' · 在线 ' + onlineInGroup + '/' + byCluster[cluster].length +
        '</strong>' + nodeMarkup + '</section>';
    }).join('');
    const archivedMarkup = archivedRows.length
      ? '<details class="fleetArchive"><summary>离线 / 历史登记（' + archivedRows.length + '）</summary>' +
        renderRows(archivedRows) + '</details>'
      : '';
    list.innerHTML = activeMarkup || '<div class="hint">当前没有在线或启动中的端点。</div>';
    list.innerHTML += archivedMarkup;
    const modelInput = el('reviewerModel');
    if (modelInput && data.default_model && !modelInput.dataset.userEdited) {
      modelInput.value = data.default_model;
    }
  }

  async function refreshFleet() {
    if (!api || !api.fleet) return;
    try {
      const data = await api.fleet(false);
      renderFleet(data);
    } catch (err) {
      const list = el('fleetList');
      if (list) {
        list.className = 'fleetList hint';
        list.textContent = 'fleet 加载失败: ' + ((err && err.message) || err);
      }
    }
  }

  async function refreshHealth() {
    const node = el('health');
    try {
      if (!api) throw new Error('MemStrataApi 未加载（检查 /static/js/api.js）');
      const data = await api.health();
      if (!node) return;
      const fleetBit = (data.fleet_online != null)
        ? (' · fleet ' + data.fleet_online + '/' + (data.fleet_total || 0) +
           (data.fleet_busy != null ? (' · busy ' + data.fleet_busy) : ''))
        : '';
      node.className = 'health ok';
      node.textContent = 'ok · backend up' + fleetBit + ' · 北京时间';
      if (data.default_reviewer_model && el('reviewerModel') && !el('reviewerModel').dataset.userEdited) {
        el('reviewerModel').value = data.default_reviewer_model;
      }
    } catch (err) {
      if (!node) return;
      node.className = 'health fail';
      const host = location.hostname || '';
      const viaProxyGateway = /\.example\.com$/i.test(host) || /remote-gpu-host/i.test(host);
      if (err && err.code === 'SSO_REQUIRED') {
        node.textContent =
          'SSO 失效 · 请刷新页面重新登录 the reverse proxy（不是 backend 挂了）';
        return;
      }
      if (err && err.code === 'GATEWAY_BUSY') {
        node.className = 'health';
        node.textContent = '后端繁忙/网关超时 · 正在自动重试（不是登录失效，也不是 backend 挂）';
        return;
      }
      const hint = viaProxyGateway
        ? ' · 经 remote HTTPS 网关访问时：先刷新重登 SSO；本机保活 bash servers/ensure_console.sh --watch'
        : (' · 若持续失败：bash servers/ensure_console.sh --watch');
      node.textContent = 'backend down · ' + ((err && err.message) || err) + hint;
    }
  }

  async function waitJobTerminal(jobId, attempts) {
    const maxAttempts = attempts == null ? 40 : attempts;
    let last = null;
    for (let i = 0; i < maxAttempts; i++) {
      const jobs = await api.listJobs();
      state.jobs = jobs.jobs || jobs || [];
      renderJobs();
      last = state.jobs.find((j) => j.job_id === jobId) || null;
      if (!last) break;
      const status = String(last.status || '');
      if (status !== 'running' && status !== 'stopping' && status !== 'queued') break;
      await new Promise((resolve) => setTimeout(resolve, 800));
    }
    return last;
  }

  async function startSamples(samples, forceRestart, resume) {
    if (!samples.length) {
      toast('请先勾选样本', 'fail');
      return;
    }
    const idle = [];
    const busy = [];
    for (const sample of samples) {
      const info = sampleRunInfo(sample);
      if (info && (info.sample_state === 'running' || info.sample_state === 'queued' || info.sample_state === 'stopping')) {
        busy.push(sample);
      } else {
        idle.push(sample);
      }
    }
    if (!idle.length) {
      toast('选中样本都已在跑或排队，已全部跳过', 'fail');
      return;
    }
    if (forceRestart) {
      const label = idle.length === 1
        ? (idle[0].dataset + '/' + idle[0].movie_id)
        : (idle.length + ' 个选中样本');
      const ok = window.confirm(
        '确认重跑？\n\n将清空 ' + label + ' 的 tmp/pipeline，并从 S2 从头开始。此操作不可撤销。' +
        (busy.length ? ('\n\n另有 ' + busy.length + ' 个已在跑/排队的样本会自动跳过。') : '')
      );
      if (!ok) return;
    }
    const body = jobFormBody(idle, forceRestart, resume);
    const invalid = validateJobForm(body);
    if (invalid) {
      toast(invalid, 'fail');
      return;
    }
    try {
      const job = await api.createJob(body);
      const jobId = job.job_id || '';
      state.activeJobId = jobId || null;
      const nEp = (job.dispatch && job.dispatch.n_reviewer_endpoints) || 0;
      const dispatchBit = nEp ? (' · 派发 ' + nEp + ' 端点') : '';
      const skippedBusy = Array.isArray(job.skipped_busy) ? job.skipped_busy.length : busy.length;
      const skippedUnready = Array.isArray(job.skipped_unready) ? job.skipped_unready.length : 0;
      const startedN = Array.isArray(job.samples) ? job.samples.length : idle.length;
      const skipBits = [];
      if (skippedBusy) skipBits.push('跳过已在跑 ' + skippedBusy);
      if (skippedUnready) skipBits.push('跳过不可跑 ' + skippedUnready);
      const skipBit = skipBits.length ? (' · ' + skipBits.join(' · ')) : '';
      const verb = forceRestart ? '已重跑 ' : (resume ? '已续跑 ' : '已启动 ');
      toast(verb + (jobId || 'job') + ' · ' + startedN + ' 片' + skipBit + dispatchBit, 'ok');
      await loadJobs(true);
      if (jobId) {
        const finalJob = await waitJobTerminal(jobId);
        await loadSamples(true);
        if (!finalJob) {
          toast('任务已提交，但未读到状态', 'fail');
          return;
        }
        if (finalJob.status === 'succeeded') {
          toast('完成 ' + jobId, 'ok');
        } else if (finalJob.status === 'failed') {
          toast(
            '失败 ' + jobId + ': ' + (finalJob.error_summary || '见 Jobs / Log'),
            'fail'
          );
          await loadJobLog(jobId, true);
        } else {
          toast(jobId + ' · ' + (finalJob.status || 'unknown'), 'fail');
        }
      } else {
        await loadSamples(true);
      }
    } catch (err) {
      toast((forceRestart ? '重跑' : (resume ? '续跑' : '启动')) + '失败: ' + err.message, 'fail');
    }
  }

  async function continueSample(sample, continueFrom) {
    const body = {
      ...jobFormBody([{ dataset: sample.dataset, movie_id: sample.movie_id }], false),
      dataset: sample.dataset,
      movie_id: sample.movie_id,
      continue_from: continueFrom,
      grounder: continueFrom === 'after_s4' ? 'qwen' : el('grounder').value,
      s4_mode: 'blocking',
      skip_human: false
    };
    try {
      const job = await api.continueReview(body);
      state.activeJobId = job.job_id || null;
      toast('已提交 ' + (continueFrom === 'after_s4' ? 'S5' : 'S7') + ' 继续任务 ' + (job.job_id || ''), 'ok');
      await Promise.all([loadJobs(true), loadSamples(true)]);
    } catch (err) {
      toast('继续失败: ' + err.message, 'fail');
    }
  }

  async function loadSamples(quiet) {
    if (!api) {
      showSampleError(new Error('MemStrataApi 未加载'));
      throw new Error('MemStrataApi 未加载');
    }
    const params = {
      dataset: (el('datasetFilter') && el('datasetFilter').value) || '',
      status: (el('statusFilter') && el('statusFilter').value) || '',
      q: ((el('searchQ') && el('searchQ').value) || '').trim()
    };
    if (!quiet) {
      const list = el('sampleList');
      if (list) {
        list.className = 'datasetSections loading';
        list.textContent = '加载中…';
      }
    }
    try {
      const data = await api.samples(params);
      state.samples = data.samples || data || [];
      renderSamples();
    } catch (err) {
      showSampleError(err);
      throw err;
    }
  }

  function renderSamples() {
    setText(
      'sampleCountHint',
      '共 ' + state.samples.length + ' 条 · 滚动选择样本，Stage 列显示 S1–S7 进展'
    );
    const list = el('sampleList');
    if (!list) return;
    if (!state.samples.length) {
      list.className = 'datasetSections empty';
      list.textContent = '无匹配样本';
      return;
    }
    const byDataset = {};
    for (const sample of state.samples) {
      (byDataset[sample.dataset] || (byDataset[sample.dataset] = [])).push(sample);
    }
    const parts = [];
    for (const [dataset, rows] of Object.entries(byDataset)) {
      parts.push(
        '<div class="datasetBlock">' +
        '<div class="datasetHead">' +
        '<h3>' + escapeHtml(dataset) + ' <span class="badge">' + rows.length + '</span></h3>' +
        '<button type="button" class="secondary datasetSelectAll" data-select-dataset="' +
          escapeHtml(dataset) + '" title="全选本数据集">全选</button>' +
        '</div>' +
        '<div class="tableWrap"><table><thead><tr>' +
        '<th style="width:36px"></th>' +
        '<th>Movie</th>' +
        '<th>Stage / Progress</th>' +
        '<th>Status</th>' +
        '<th>任务</th>' +
        '<th>Source</th>' +
        '<th>VLM</th>' +
        '<th>Review</th>' +
        '<th>操作</th>' +
        '</tr></thead><tbody>'
      );
      for (const sample of rows) {
        const key = selectionKey(sample);
        const checked = state.selected.has(key) ? ' checked' : '';
        const statusCls = statusBadgeClass(sample.status);
        const resume = resumeAction(sample);
        const resumeDisabled = !!(resume && resume.disabled);
        const resumeTitle = resumeDisabled
          ? escapeHtml(resume.label)
          : (resume && resume.continueFrom
            ? escapeHtml(resume.label)
            : '从当前阶段继续（保留进度）');
        const runInfo = sampleRunInfo(sample);
        let jobCell = '<span class="hint">—</span>';
        if (runInfo) {
          if (runInfo.sample_state === 'stopping') {
            jobCell =
              '<div class="sampleJobCell">' +
              '<span class="badge ' + statusBadgeClass('stopping') + '">停止中</span>' +
              '</div>';
          } else {
            const pauseTip = '暂停该样本所属任务' +
              (runInfo.n_samples > 1 ? '（同任务内其它样本也会停）' : '');
            jobCell =
              '<div class="sampleJobCell" title="' + escapeHtml(runInfo.job_id || '') + ' · 悬停可暂停">' +
              '<span class="sampleJobState badge ' + statusBadgeClass(runInfo.sample_state) + '">' +
              escapeHtml(sampleStateLabel(runInfo.sample_state)) + '</span>' +
              '<button type="button" class="sampleJobPause" data-stop-sample="' + escapeHtml(key) +
              '" title="' + escapeHtml(pauseTip) + '">暂停</button>' +
              '</div>';
          }
        }
        parts.push(
          '<tr>' +
          '<td><input type="checkbox" data-key="' + escapeHtml(key) + '"' + checked + '></td>' +
          '<td><div class="movie"><strong title="' + escapeHtml(sample.movie_dir || '') + '">' +
          escapeHtml(sample.movie_id) + '</strong></div></td>' +
          '<td>' + renderStageCell(sample) + '</td>' +
          '<td><span class="badge ' + statusCls + '">' + escapeHtml(sample.status || '') + '</span></td>' +
          '<td>' + jobCell + '</td>' +
          '<td>' + (sample.source_video_exists ? '<span class="badge ok">yes</span>' : '<span class="badge danger">no</span>') + '</td>' +
          '<td>' + (sample.has_vlm_output ? '<span class="badge ok">yes</span>' : '<span class="badge danger">no</span>') + '</td>' +
          '<td class="rowActions">' + reviewLinksHtml(sample) + '</td>' +
          '<td class="rowActions">' +
          '<button class="primary" data-resume="' + escapeHtml(key) + '"' +
            (resumeDisabled || runInfo ? ' disabled' : '') +
            ' title="' + (runInfo ? '任务进行中，请先暂停' : resumeTitle) + '">续跑</button>' +
          '<button class="warnBtn" data-rerun="' + escapeHtml(key) + '"' +
            (runInfo ? ' disabled title="任务进行中，请先暂停"' : ' title="清空 tmp/pipeline 并从 S2 重跑"') +
            '>重跑</button>' +
          '<button class="secondary" data-detail="' + escapeHtml(key) + '">详情</button>' +
          '</td>' +
          '</tr>'
        );
      }
      parts.push('</tbody></table></div></div>');
    }
    list.className = 'datasetSections';
    list.innerHTML = parts.join('');
    updateSelectionHint();
  }

  function renderStageList(stages) {
    const map = stages || {};
    return (
      '<div class="stageList">' +
      PIPELINE_STAGES.map((name) => {
        const entry = map[name] || {};
        const status = entry.status || '—';
        return (
          '<div class="stageRow">' +
          '<span>' + escapeHtml(shortStage(name) + ' ' + name) + '</span>' +
          '<strong>' + escapeHtml(status) + '</strong>' +
          '</div>'
        );
      }).join('') +
      '</div>'
    );
  }

  async function openDetail(key) {
    const sample = state.samples.find((s) => selectionKey(s) === key);
    if (!sample) return;
    el('drawer').hidden = false;
    el('drawerTitle').textContent = sample.dataset + ' / ' + sample.movie_id;
    el('drawerBody').innerHTML = '<div class="loading">加载中…</div>';
    try {
      const data = await api.sampleDetail(sample.dataset, sample.movie_id);
      state.detail = data;
      const review = data.review || sample.review || {};
      const paths = data.paths || {};
      const gold = data.gold || data.gold_summary || {};
      const stages = data.stages || sample.stages || {};
      const actions = [];
      const q = 'dataset=' + encodeURIComponent(sample.dataset) + '&movie_id=' + encodeURIComponent(sample.movie_id);
      if (review.s3_live_available) actions.push('<a class="buttonLink reviewLink" href="/review/s3.html?' + q + '">打开 S3 实时审核</a>');
      if (review.s4_available) actions.push('<a class="buttonLink reviewLink" href="/review/s4.html?' + q + '">打开 S4 Segment 审核</a>');
      if (review.s6_available) actions.push('<a class="buttonLink reviewLink" href="/review/s6.html?' + q + '">打开 S6 Crop 审核</a>');

      el('drawerBody').innerHTML =
        '<div class="detailActions">' + (actions.join(' ') || '<span class="hint">暂无人工审核入口</span>') + '</div>' +
        '<div class="detailTitle">' + escapeHtml(sample.movie_id) + '</div>' +
        '<div class="detailGrid"><span>status</span><strong>' + escapeHtml(data.status || sample.status || '') + '</strong></div>' +
        '<div class="detailGrid"><span>stage</span><strong>' + escapeHtml((sample.current_stage || '—') + ' · ' + (sample.current_status || '')) + '</strong></div>' +
        '<div class="detailGrid"><span>progress</span><strong>' + escapeHtml((sample.completed_stage_count || 0) + ' / ' + (sample.stage_count || 7)) + '</strong></div>' +
        '<div class="detailSection"><h3>Stages</h3>' + renderStageList(stages) + '</div>' +
        '<div class="detailSection"><h3>路径</h3>' +
        '<div class="detailPath">' + escapeHtml(paths.movie_dir || sample.movie_dir || '') + '</div>' +
        '<div class="detailPath">' + escapeHtml(paths.source_video || sample.source_video || '') + '</div></div>' +
        '<div class="detailSection"><h3>Gold</h3>' +
        '<div>exists=' + escapeHtml(gold.exists) +
        ' · entities/nxt=' + escapeHtml(gold.n_nxt) +
        ' · crops=' + escapeHtml(gold.n_crops) +
        ' · human_reviewed=' + escapeHtml(gold.human_reviewed) + '</div></div>';
    } catch (err) {
      el('drawerBody').innerHTML = '<div class="errorText">' + escapeHtml(err.message) + '</div>';
    }
  }

  function closeDrawer() {
    el('drawer').hidden = true;
  }

  async function loadJobs(quiet) {
    const data = await api.listJobs();
    state.jobs = data.jobs || data || [];
    state.sampleActivity = data.sample_activity || {};
    state.sampleCounts = data.sample_counts || null;
    renderJobs();
    renderSamples();
    if (state.activeJobId) {
      await loadJobLog(state.activeJobId, quiet);
    }
  }

  function filteredJobs() {
    const filter = state.jobsFilter || 'active';
    const jobs = state.jobs.slice();
    if (filter === 'all') return jobs;
    if (filter === 'running') {
      return jobs.filter((job) => String(job.status || '') === 'running');
    }
    if (filter === 'queued') {
      return jobs.filter((job) => String(job.status || '') === 'queued');
    }
    return jobs.filter((job) => {
      const status = String(job.status || '');
      return status === 'queued' || status === 'running' || status === 'stopping';
    });
  }

  function updateJobsToolbar() {
    const active = state.jobs.filter((job) => {
      const status = String(job.status || '');
      return status === 'queued' || status === 'running' || status === 'stopping';
    });
    const jobRunning = active.filter((job) => String(job.status || '') === 'running').length;
    const jobQueued = active.filter((job) => String(job.status || '') === 'queued').length;
    const stopping = active.filter((job) => String(job.status || '') === 'stopping').length;
    const counts = state.sampleCounts || {};
    let sampleRunning = Number(counts.running);
    let sampleQueued = Number(counts.queued);
    if (!Number.isFinite(sampleRunning) || !Number.isFinite(sampleQueued)) {
      const activity = Object.values(state.sampleActivity || {});
      sampleRunning = activity.filter((row) => row && row.sample_state === 'running').length;
      sampleQueued = activity.filter((row) => row && row.sample_state === 'queued').length;
    }
    const hint = active.length
      ? ('活跃任务 ' + active.length +
         '（在跑 ' + jobRunning + ' / 排队 ' + jobQueued +
         (stopping ? (' / 停止中 ' + stopping) : '') +
         '）· 样本在跑 ' + sampleRunning + ' · 样本在排队 ' + sampleQueued)
      : '暂无活跃任务';
    setText('jobsCountHint', hint);
    setText(
      'jobsSummaryHint',
      active.length
        ? ('当前 ' + hint + '；样本状态按片内进度，不是整任务照搬')
        : '查看在跑 / 在排队的任务，可单个停止或全部停止'
    );
    const stopAll = el('stopAllJobsBtn');
    if (stopAll) stopAll.disabled = active.length === 0;
    document.querySelectorAll('.jobsFilter').forEach((btn) => {
      const value = btn.getAttribute('data-jobs-filter');
      btn.classList.toggle('active', value === state.jobsFilter);
    });
  }

  function renderJobs() {
    updateJobsToolbar();
    const jobs = filteredJobs();
    if (!jobs.length) {
      el('jobsList').className = 'jobsList empty';
      el('jobsList').textContent = state.jobsFilter === 'all' ? '暂无任务' : '当前过滤下没有任务';
      return;
    }
    const ordered = jobs.slice().sort((left, right) => {
      const rank = { running: 0, queued: 1, stopping: 2 };
      const leftRank = rank[String(left.status || '')];
      const rightRank = rank[String(right.status || '')];
      if (leftRank != null || rightRank != null) {
        return (leftRank == null ? 9 : leftRank) - (rightRank == null ? 9 : rightRank);
      }
      return String(right.created_at || '').localeCompare(String(left.created_at || ''));
    });
    el('jobsList').className = 'jobsList';
    el('jobsList').innerHTML = ordered.map((job) => {
      const status = String(job.status || '');
      const live = status === 'queued' || status === 'running' || status === 'stopping';
      const active = job.job_id === state.activeJobId ? ' active' : '';
      const liveCls = live ? ' live' : '';
      const samples = (job.samples || []).map((s) => s.movie_id || s).join(', ');
      const err = job.error_summary
        ? '<div class="jobMeta" style="color:var(--danger,#f87171)">' + escapeHtml(job.error_summary) + '</div>'
        : '';
      const when = [
        job.created_at ? ('创建 ' + formatBeijingTime(job.created_at)) : '',
        job.ended_at ? ('结束 ' + formatBeijingTime(job.ended_at)) : '',
        job.stopped_at ? ('停止 ' + formatBeijingTime(job.stopped_at)) : ''
      ].filter(Boolean).join(' · ');
      const canStop = status === 'queued' || status === 'running';
      return (
        '<div class="jobCard' + active + liveCls + '" data-job="' + escapeHtml(job.job_id) + '">' +
        '<div class="jobHead"><strong>' + escapeHtml(job.job_id) + '</strong>' +
        '<span class="badge ' + statusBadgeClass(status) + '">' + escapeHtml(jobStatusLabel(status)) + '</span></div>' +
        '<div class="jobMeta">' + escapeHtml(samples || '—') + '</div>' +
        (when ? '<div class="jobMeta">' + escapeHtml(when) + '</div>' : '') +
        err +
        '<div class="jobActions">' +
        '<button class="secondary" data-view="' + escapeHtml(job.job_id) + '">日志</button>' +
        (canStop
          ? '<button class="dangerBtn" data-stop="' + escapeHtml(job.job_id) + '">停止</button>'
          : '') +
        '</div></div>'
      );
    }).join('');
  }

  async function loadJobLog(jobId, quiet) {
    state.activeJobId = jobId;
    renderJobs();
    if (!quiet) el('jobLog').textContent = '加载日志…';
    try {
      const payload = await api.jobLog(jobId);
      const text = typeof payload === 'string'
        ? payload
        : (payload && payload.text != null ? payload.text : JSON.stringify(payload, null, 2));
      el('jobLog').textContent = text || '(empty log)';
      el('jobLog').scrollTop = el('jobLog').scrollHeight;
    } catch (err) {
      el('jobLog').textContent = '日志读取失败: ' + err.message;
    }
  }

  async function resumeSelectedJob() {
    const samples = state.samples
      .filter((s) => state.selected.has(selectionKey(s)))
      .map((s) => ({ dataset: s.dataset, movie_id: s.movie_id }));
    if (!samples.length) {
      toast('请先勾选样本', 'fail');
      return;
    }
    await startSamples(samples, false, true);
  }

  async function rerunSelectedJob() {
    const samples = state.samples
      .filter((s) => state.selected.has(selectionKey(s)))
      .map((s) => ({ dataset: s.dataset, movie_id: s.movie_id }));
    await startSamples(samples, true, false);
  }

  on('sampleList', 'change', (ev) => {
    const target = ev.target;
    if (!(target instanceof HTMLInputElement) || target.type !== 'checkbox') return;
    const key = target.getAttribute('data-key');
    if (!key) return;
    if (target.checked) state.selected.add(key);
    else state.selected.delete(key);
    updateSelectionHint();
  });

  on('sampleList', 'click', (ev) => {
    const selectDataset = ev.target.closest('[data-select-dataset]');
    if (selectDataset) {
      const dataset = selectDataset.getAttribute('data-select-dataset') || '';
      let n = 0;
      for (const sample of state.samples) {
        if (sample.dataset !== dataset) continue;
        state.selected.add(selectionKey(sample));
        n += 1;
      }
      renderSamples();
      toast('已全选 ' + dataset + ' · ' + n + ' 条', 'ok');
      return;
    }
    const stopSample = ev.target.closest('[data-stop-sample]');
    if (stopSample) {
      const sample = state.samples.find((s) => selectionKey(s) === stopSample.getAttribute('data-stop-sample'));
      if (!sample) return;
      const info = sampleRunInfo(sample);
      const multi = info && info.n_samples > 1;
      const ok = window.confirm(
        multi
          ? ('暂停 ' + sample.dataset + '/' + sample.movie_id + '？\n\n该样本与同任务内其它样本共用一个 job，全部会一起停止。')
          : ('暂停 ' + sample.dataset + '/' + sample.movie_id + '？')
      );
      if (!ok) return;
      api.stopSampleJob(sample.dataset, sample.movie_id).then(() => {
        toast('已请求暂停 ' + sample.movie_id, 'ok');
        return loadJobs();
      }).catch((err) => toast(err.message, 'fail'));
      return;
    }
    const detail = ev.target.closest('[data-detail]');
    if (detail) {
      openDetail(detail.getAttribute('data-detail'));
      return;
    }
    const resumeBtn = ev.target.closest('[data-resume]');
    if (resumeBtn) {
      if (resumeBtn.disabled) return;
      const sample = state.samples.find((s) => selectionKey(s) === resumeBtn.getAttribute('data-resume'));
      if (!sample) return;
      const action = resumeAction(sample);
      if (action && action.continueFrom && !action.disabled) {
        continueSample(sample, action.continueFrom);
      } else {
        startSamples([{ dataset: sample.dataset, movie_id: sample.movie_id }], false, true);
      }
      return;
    }
    const rerun = ev.target.closest('[data-rerun]');
    if (rerun) {
      if (rerun.disabled) return;
      const sample = state.samples.find((s) => selectionKey(s) === rerun.getAttribute('data-rerun'));
      if (sample) startSamples([{ dataset: sample.dataset, movie_id: sample.movie_id }], true, false);
    }
  });

  on('jobsList', 'click', async (ev) => {
    const stop = ev.target.closest('[data-stop]');
    if (stop) {
      try {
        await api.stopJob(stop.getAttribute('data-stop'));
        toast('已请求停止', 'ok');
        await loadJobs();
      } catch (err) {
        toast(err.message, 'fail');
      }
      return;
    }
    const view = ev.target.closest('[data-view]');
    const card = ev.target.closest('[data-job]');
    const jobId = (view && view.getAttribute('data-view')) || (card && card.getAttribute('data-job'));
    if (jobId) await loadJobLog(jobId);
  });

  on('stopAllJobsBtn', 'click', async () => {
    const active = state.jobs.filter((job) => {
      const status = String(job.status || '');
      return status === 'queued' || status === 'running' || status === 'stopping';
    });
    if (!active.length) {
      toast('当前没有可停止的任务', 'fail');
      return;
    }
    const ok = window.confirm('确认停止全部 ' + active.length + ' 个活跃任务？');
    if (!ok) return;
    try {
      const result = await api.stopAllJobs();
      toast('已请求停止 ' + (result.n_stopped || active.length) + ' 个任务', 'ok');
      await loadJobs();
    } catch (err) {
      toast(err.message, 'fail');
    }
  });

  document.querySelectorAll('.jobsFilter').forEach((btn) => {
    btn.addEventListener('click', () => {
      state.jobsFilter = btn.getAttribute('data-jobs-filter') || 'active';
      renderJobs();
    });
  });

  (function initJobsPanelToggle() {
    const panel = document.querySelector('details.jobs');
    if (!panel) return;
    const key = 'memstrata.console.jobsOpen';
    try {
      const saved = localStorage.getItem(key);
      if (saved === '0') panel.open = false;
      else if (saved === '1') panel.open = true;
    } catch (_) { /* ignore */ }
    panel.addEventListener('toggle', () => {
      try {
        localStorage.setItem(key, panel.open ? '1' : '0');
      } catch (_) { /* ignore */ }
    });
  })();

  on('closeDrawerBtn', 'click', closeDrawer);
  on('drawerBackdrop', 'click', closeDrawer);

  on('refreshBtn', 'click', () => {
    Promise.all([refreshHealth(), refreshFleet(), loadSamples(), loadJobs()]).catch(explainFailure);
  });
  on('selectAllBtn', 'click', () => {
    for (const sample of state.samples) state.selected.add(selectionKey(sample));
    renderSamples();
  });
  on('selectReadyBtn', 'click', () => {
    for (const sample of state.samples) {
      if (sample.source_video_exists && sample.has_vlm_output) state.selected.add(selectionKey(sample));
    }
    renderSamples();
  });
  on('clearSelectionBtn', 'click', () => {
    state.selected.clear();
    renderSamples();
  });
  on('resumeJobBtn', 'click', resumeSelectedJob);
  on('rerunJobBtn', 'click', rerunSelectedJob);
  ['datasetFilter', 'statusFilter'].forEach((id) => {
    on(id, 'change', () => loadSamples().catch((e) => toast(e.message, 'fail')));
  });
  let searchTimer = null;
  on('searchQ', 'input', () => {
    clearTimeout(searchTimer);
    searchTimer = setTimeout(() => {
      loadSamples(true).catch((e) => toast(e.message, 'fail'));
    }, 180);
  });
  on('searchQ', 'keydown', (ev) => {
    if (ev.key !== 'Enter') return;
    clearTimeout(searchTimer);
    loadSamples(true).catch((e) => toast(e.message, 'fail'));
  });

  if (window.MemStrataConfig && window.MemStrataConfig.defaultVlmBaseUrl && el('reviewerBaseUrl')) {
    el('reviewerBaseUrl').value = window.MemStrataConfig.defaultVlmBaseUrl;
  }
  if (el('reviewerModel')) {
    el('reviewerModel').addEventListener('input', () => {
      el('reviewerModel').dataset.userEdited = '1';
    });
  }

  function recoverSso(reason) {
    const key = 'memstrata_sso_nav_at';
    const now = Date.now();
    let last = 0;
    try {
      last = Number(sessionStorage.getItem(key) || 0);
    } catch (_) { /* ignore */ }
    // Avoid a tight redirect loop if SSO keeps bouncing.
    if (now - last < 20000) {
      toast(
        'SSO 仍未通过。请新开标签打开当前网址完成登录：' + location.origin + '/',
        'fail'
      );
      return;
    }
    try {
      sessionStorage.setItem(key, String(now));
    } catch (_) { /* ignore */ }
    const raw = (reason && reason.loginUrl) || '';
    let target = '';
    try {
      target = raw ? new URL(raw, location.href).href : location.href;
    } catch (_) {
      target = location.href;
    }
    toast('正在跳转快手 SSO 登录…', 'fail');
    window.location.assign(target);
  }

  function explainFailure(reason) {
    if (reason && reason.code === 'SSO_REQUIRED') {
      recoverSso(reason);
      return;
    }
    toast(reason && reason.message ? reason.message : String(reason), 'fail');
  }

  Promise.allSettled([refreshHealth(), refreshFleet(), loadSamples(), loadJobs()]).then((results) => {
    const failed = results.find((r) => r.status === 'rejected');
    if (failed) explainFailure(failed.reason);
  });

  setInterval(() => {
    if (document.hidden) return;
    refreshHealth().catch(() => {});
    refreshFleet().catch(() => {});
    loadSamples(true).catch(() => {});
    loadJobs(true).catch(() => {});
  }, 8000);
})();
