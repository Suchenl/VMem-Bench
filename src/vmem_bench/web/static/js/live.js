/** Live monitor: SSE, event log, scrubber. */
(function (global) {
  'use strict';
  const STAGE_ZH = global.MemStrataApi.STAGE_ZH;

  global.MemStrataLive = {
    computed: {
      filteredLogs() {
        return this.eventsLog.filter(row => {
          let fCls = 'other';
          if (row.kind === 'chunk_start' || row.kind === 'chunk_done') fCls = 'chunk';
          else if (row.kind === 'role_start' || row.kind === 'role_end' || row.kind === 'tracking_start' || row.kind === 'naming_start') fCls = 'role';
          else if (row.kind === 'registry' || row.kind === 'registry_final' || row.kind === 'cast_roster'
                   || row.kind === 'roster_start' || row.kind === 'roster_progress'
                   || row.kind === 'roster_semantic_dedup' || row.kind === 'roster_resumed') fCls = 'asset';
          if (this.currentFilter === 'all') return true;
          if (this.currentFilter === 'chunk' && fCls === 'chunk') return true;
          if (this.currentFilter === 'role' && fCls === 'role') return true;
          if (this.currentFilter === 'asset' && fCls === 'asset') return true;
          if (this.currentFilter === 'warn' && (row.kind.includes('error') || row.kind === 'name_error' || row.flagged)) return true;
          return false;
        });
      }
    },
    watch: {
      filteredLogs() {
        if (this.autoScrollLog && !this.loadingHistory) this.forceScrollToBottom();
      },
      currentFilter() {
        if (this.autoScrollLog) this.forceScrollToBottom();
      },
      activeMode(mode) {
        if (mode === 'live' && this.autoScrollLog) this.forceScrollToBottom();
      }
    },
    methods: {
      getChunkGridClass(cid) {
        const info = this.liveState.chunks[cid];
        if (!info) return 'bg-[#080a10]/40 border-border/80 text-dimText';
        if (info.active) return 'bg-accentLight/15 border-accentLight text-accentLight shadow-[0_0_10px_#3b82f6] pulse-active';
        if (info.done) {
          if (info.flagged) return 'bg-warning/15 border-warning text-warning shadow-[0_0_6px_#f59e0b]';
          return 'bg-success/15 border-success text-success';
        }
        return 'bg-[#080a10]/40 border-border/80 text-dimText';
      },
      getChunkGridTitle(cid) {
        const info = this.liveState.chunks[cid];
        if (!info) return '';
        if (info.active) return '进行中...';
        if (info.done) return info.flagged ? 'Flagged (已标记不合规项)' : '完成 (一次性通过)';
        return '待处理';
      },
      selectLiveChunk(cid) { this.selectedChunkId = cid; },
      initSSE() {
        this.es = new EventSource('/events');
        this.es.onopen = () => { this.connStatus = 'connected'; };
        this.es.onerror = () => { this.connStatus = 'reconnecting'; };
        this.es.addEventListener('reset', () => { this.resetLiveState(); });
        this.es.onmessage = (m) => {
          let e;
          try { e = JSON.parse(m.data); } catch (err) { return; }
          this.applyEvent(e);
          this.addLogEvent(e);
        };
      },
      resetLiveState() {
        this.eventsLog = [];
        this.liveAssets = [];
        this.liveState = {
          annotatorActive: false, annotatorStage: '', verifierActive: false, verifierStage: '',
          currentShot: 0, totalShots: 0, elapsedSeconds: 0, etaSeconds: null,
          nEntitiesTracked: 0, namingDone: 0, namingTotal: 0, currentNamingName: '',
          rosterDone: 0, rosterTotal: 0, rosterKnown: 0,
          completedChunks: 0, chunks: {}
        };
        this.runStatus = { movie_id: '—', stage: 'chunking', done: false, n_chunks: 0, n_entities: 0 };
        this.loadingHistory = true;
        this.historyBuffer = [];
        this.autoScrollLog = true;
        this.logScrollProgress = 100;
      },
      applyEvent(e) {
        const id = e.chunk_id;
        if (e.stage) this.runStatus.stage = e.stage;
        switch (e.kind) {
          case 'run_start':
            this.runStatus.movie_id = e.movie_id;
            this.runStatus.stage = 'chunking';
            break;
          case 'layout':
            this.runStatus.n_chunks = e.n_chunks;
            this.runStatus.n_entities = e.n_entities || 0;
            this.runStatus.stage = 'roster';
            (e.chunks || []).forEach(ch => {
              this.liveState.chunks[ch.chunk_id] = Object.assign(this.liveState.chunks[ch.chunk_id] || {}, ch);
            });
            break;
          case 'roster_start':
            this.runStatus.stage = 'roster';
            this.liveState.rosterDone = 0;
            this.liveState.rosterTotal = 0;
            this.liveState.rosterKnown = 0;
            break;
          case 'roster_progress':
            this.runStatus.stage = 'roster';
            this.liveState.rosterDone = e.done || 0;
            this.liveState.rosterTotal = e.total || 0;
            this.liveState.rosterKnown = e.n_known || 0;
            break;
          case 'tracking_start':
            this.runStatus.stage = 'tracking';
            this.liveState.annotatorActive = true;
            this.liveState.annotatorStage = `逐镜头跟踪 0/${e.n_shots}`;
            this.liveState.totalShots = e.n_shots;
            this.liveState.currentShot = 0;
            break;
          case 'track_progress':
            this.runStatus.stage = 'tracking';
            this.liveState.annotatorActive = true;
            this.liveState.currentShot = e.shot;
            this.liveState.totalShots = e.n_shots;
            this.liveState.nEntitiesTracked = e.n_entities;
            this.liveState.etaSeconds = e.eta_seconds != null ? Math.round(e.eta_seconds) : null;
            this.liveState.annotatorStage = `逐镜头跟踪 ${e.shot}/${e.n_shots} · ${e.n_entities} 实体`;
            break;
          case 'identity':
            this.runStatus.stage = 'identity';
            this.liveState.annotatorActive = false;
            this.liveState.annotatorStage = '跨镜身份联合完成';
            break;
          case 'naming_start':
            this.runStatus.stage = 'naming';
            this.liveState.annotatorActive = true;
            this.liveState.namingDone = 0;
            this.liveState.namingTotal = e.n_entities;
            this.liveState.annotatorStage = `实体命名 0/${e.n_entities}`;
            break;
          case 'naming_progress':
            this.runStatus.stage = 'naming';
            this.liveState.annotatorActive = true;
            this.liveState.namingDone = e.done;
            this.liveState.namingTotal = e.n_entities;
            this.liveState.currentNamingName = e.name || '';
            this.liveState.annotatorStage = `实体命名 ${e.done}/${e.n_entities}`;
            break;
          case 'naming_done':
            this.runStatus.stage = 'drafting';
            this.liveState.annotatorActive = false;
            break;
          case 'chunk_start': {
            this.runStatus.stage = 'drafting';
            const cell = this.liveState.chunks[id] || {};
            cell.active = true;
            cell.frame_span = e.frame_span;
            this.liveState.chunks[id] = cell;
            break;
          }
          case 'role_start':
            this.runStatus.stage = 'drafting';
            if (e.role === 'annotator') {
              this.liveState.annotatorActive = true;
              this.liveState.annotatorStage = `${STAGE_ZH[e.stage] || e.stage} · c${String(id).padStart(3, '0')}`;
            } else {
              this.liveState.verifierActive = true;
              this.liveState.verifierStage = `${STAGE_ZH[e.stage] || e.stage} · c${String(id).padStart(3, '0')}`;
            }
            break;
          case 'role_end': {
            this.runStatus.stage = 'drafting';
            const info = this.liveState.chunks[id] || {};
            if (e.role === 'annotator') this.liveState.annotatorActive = false;
            else this.liveState.verifierActive = false;
            if (e.entities) info.entities = e.entities;
            if (e.prompt) info.prompt = e.prompt;
            if (e.checks) info.checks = e.checks;
            if (e.crops) info.crops = e.crops;
            this.liveState.chunks[id] = info;
            break;
          }
          case 'chunk_done': {
            this.runStatus.stage = 'drafting';
            const info = this.liveState.chunks[id] || {};
            info.active = false;
            info.done = true;
            info.flagged = e.flagged;
            info.rounds = e.rounds;
            info.seconds = e.seconds;
            this.liveState.chunks[id] = info;
            this.recalculateCompletedChunks();
            break;
          }
          case 'cast_roster':
            this.liveAssets = e.entities || [];
            this.runStatus.n_entities = e.n_entities;
            break;
          case 'registry':
            (e.entities || []).forEach(en => {
              const idx = this.liveAssets.findIndex(x => x.entity_id === en.entity_id);
              if (idx !== -1) this.liveAssets[idx] = Object.assign({}, this.liveAssets[idx], en);
              else this.liveAssets.push(en);
            });
            break;
          case 'registry_final':
            this.liveAssets = e.entities || [];
            break;
          case 'run_done':
            this.runStatus.done = true;
            this.runStatus.stage = 'done';
            this.liveState.annotatorActive = false;
            this.liveState.verifierActive = false;
            break;
        }
      },
      recalculateCompletedChunks() {
        let count = 0;
        Object.values(this.liveState.chunks).forEach(c => { if (c.done) count++; });
        this.liveState.completedChunks = count;
      },
      addLogEvent(e) {
        if (this.loadingHistory) {
          this.historyBuffer.push(e);
          clearTimeout(this.historyTimeout);
          this.historyTimeout = setTimeout(() => {
            this.loadingHistory = false;
            this.eventsLog.push(...this.historyBuffer);
            if (this.eventsLog.length > 1500) this.eventsLog = this.eventsLog.slice(-1500);
            this.historyBuffer = [];
            this.autoScrollLog = true;
            this.forceScrollToBottom();
            setTimeout(() => this.forceScrollToBottom(), 80);
            setTimeout(() => this.forceScrollToBottom(), 250);
          }, 120);
        } else {
          this.eventsLog.push(e);
          if (this.eventsLog.length > 1500) this.eventsLog.shift();
          if (this.autoScrollLog) this.scrollToBottomThrottled();
        }
      },
      scrollToBottomThrottled() {
        if (this.scrollTimeout) return;
        this.scrollTimeout = setTimeout(() => {
          this.scrollTimeout = null;
          this.forceScrollToBottom();
        }, 50);
      },
      forceScrollToBottom() {
        const { nextTick } = Vue;
        nextTick(() => {
          const log = this.$refs.logContainer;
          if (!log) return;
          this.isAutoScrolling = true;
          const scrollToBottom = () => {
            log.scrollTop = Math.max(0, log.scrollHeight - log.clientHeight);
            this.logScrollProgress = 100;
          };
          scrollToBottom();
          requestAnimationFrame(() => {
            scrollToBottom();
            requestAnimationFrame(() => {
              scrollToBottom();
              this.isAutoScrolling = false;
            });
          });
        });
      },
      toggleAutoScroll(val) {
        this.autoScrollLog = val;
        if (val) this.forceScrollToBottom();
      },
      applyLogScrubProgress(clientY, trackEl) {
        const log = this.$refs.logContainer;
        if (!log || !trackEl) return;
        const bounds = trackEl.getBoundingClientRect();
        const progress = Math.max(0, Math.min(100, ((clientY - bounds.top) / Math.max(1, bounds.height)) * 100));
        const maxScroll = Math.max(0, log.scrollHeight - log.clientHeight);
        this.isAutoScrolling = true;
        log.scrollTop = maxScroll * progress / 100;
        this.logScrollProgress = Math.round(progress);
        this.autoScrollLog = progress >= 99.5;
        requestAnimationFrame(() => { this.isAutoScrolling = false; });
      },
      onLogScrubberPointerDown(e) {
        const trackEl = e.currentTarget;
        const clientY = e.touches ? e.touches[0].clientY : e.clientY;
        this.isScrubbingLog = true;
        document.body.style.cursor = 'grabbing';
        this.applyLogScrubProgress(clientY, trackEl);
        this._scrubMoveHandler = (ev) => {
          const y = ev.touches ? ev.touches[0].clientY : ev.clientY;
          this.applyLogScrubProgress(y, trackEl);
        };
        this._scrubUpHandler = () => {
          this.isScrubbingLog = false;
          document.body.style.cursor = '';
          window.removeEventListener('mousemove', this._scrubMoveHandler);
          window.removeEventListener('mouseup', this._scrubUpHandler);
          window.removeEventListener('touchmove', this._scrubMoveHandler);
          window.removeEventListener('touchend', this._scrubUpHandler);
          this._scrubMoveHandler = null;
          this._scrubUpHandler = null;
        };
        window.addEventListener('mousemove', this._scrubMoveHandler);
        window.addEventListener('mouseup', this._scrubUpHandler);
        window.addEventListener('touchmove', this._scrubMoveHandler, { passive: false });
        window.addEventListener('touchend', this._scrubUpHandler);
      },
      onLogScroll(e) {
        if (this.loadingHistory || this.isAutoScrolling || this.isScrubbingLog) return;
        const log = e.target;
        const maxScroll = log.scrollHeight - log.clientHeight;
        if (maxScroll <= 0) this.logScrollProgress = 100;
        else this.logScrollProgress = Math.round((log.scrollTop / maxScroll) * 100);
        this.autoScrollLog = log.scrollTop + log.clientHeight >= log.scrollHeight - 35;
      },
      pollStatus() {
        fetch('/status').then(res => res.json()).then(data => {
          if (data.done) {
            this.runStatus.done = true;
            this.runStatus.stage = 'done';
            if (!this.gold && this.activeMode === 'review') this.fetchGold();
          } else {
            this.runStatus.done = false;
            if (data.stage) this.runStatus.stage = data.stage;
            if (data.movie_id) this.runStatus.movie_id = data.movie_id;
            if (data.n_chunks) this.runStatus.n_chunks = data.n_chunks;
            if (data.n_entities) this.runStatus.n_entities = data.n_entities;
          }
        }).catch(err => console.error('Poll status error', err));
      },
      formatLogTs(ts) {
        return new Date(ts * 1000).toLocaleTimeString('zh-CN', { hour12: false });
      },
      getLogItemClass(ev) {
        let cls = '';
        if (ev.kind.includes('error') || ev.kind === 'name_error') cls += ' text-danger bg-danger/[0.02]';
        else if (ev.flagged) cls += ' text-warning bg-warning/[0.02]';
        else if (ev.kind === 'run_start' || ev.kind === 'run_done' || ev.kind === 'layout') cls += ' text-success';
        return cls;
      },
      getLogKindLabel(ev) {
        switch (ev.kind) {
          case 'run_start': return 'RUN';
          case 'layout': return 'LAYOUT';
          case 'credits_excluded': return '剔除片头尾';
          case 'roster_start': return '演员表开始';
          case 'roster_progress': return '演员表进度';
          case 'roster_resumed': return '演员表续跑';
          case 'roster_semantic_dedup': return '演员去重';
          case 'cast_roster': return '全片演员';
          case 'tracking_start': return '跟踪开始';
          case 'track_progress': return '跟踪进度';
          case 'identity': return '联合Re-ID';
          case 'crop_qa': return '裁剪质检';
          case 'naming_start': return '命名开始';
          case 'naming_progress': return '命名进度';
          case 'naming_done': return '命名结束';
          case 'name_error': return '命名异常';
          case 'chunk_start': return 'CHUNK';
          case 'role_start': return ev.role === 'annotator' ? '标注工起' : '校验工起';
          case 'role_end': return ev.role === 'annotator' ? '标注工迄' : '校验工迄';
          case 'chunk_done': return ev.flagged ? 'QA-FLAG' : 'PASS';
          case 'registry': return '资产流';
          case 'registry_final': return '资产定稿';
          case 'run_done': return 'DONE';
          default: return ev.kind;
        }
      },
      getLogBody(ev) {
        const c = ev.chunk_id !== undefined ? `c${String(ev.chunk_id).padStart(3, '0')}` : '';
        switch (ev.kind) {
          case 'run_start': return `开始执行标注电影 ${ev.movie_id}（当前正在切分视频中…）`;
          case 'layout': return `视频切分完成：${ev.n_chunks} 个 chunk / ${ev.n_shots} 镜头 · 帧率 FPS=${ev.fps}` + ((ev.excluded_segments || []).length ? ` · 已剔除 ${ev.excluded_segments.length} 段片头/片尾` : '');
          case 'credits_excluded': return `剔除非叙事片段 ${ev.n_excluded_frames} 帧：` + (ev.segments || []).map(s => `${s.reason === 'opening_credits' ? '片头' : '片尾'} 帧${s.frame_span[0]}-${s.frame_span[1]}（${s.seconds_span[0]}s-${s.seconds_span[1]}s）`).join('、');
          case 'roster_start': return `开始全片演员表发现（选中 ${ev.n_keyframes_selected != null ? ev.n_keyframes_selected : ev.n_keyframes_budget} 关键帧，下限 ${ev.n_keyframes_budget}${ev.n_keyframes_max ? `/上限 ${ev.n_keyframes_max}` : ''}，每批 ${ev.vlm_batch}）…`;
          case 'roster_progress': return `演员表批次 ${ev.done}/${ev.total} 完成 · 累积候选 ${ev.n_known}`;
          case 'roster_resumed': return `复用已 checkpoint 的演员表：${ev.n_entities} 个实体`;
          case 'roster_semantic_dedup': return `语义去重：${ev.before} ➔ ${ev.after} 实体数（合并 ${ev.merged} 个）`;
          case 'cast_roster': {
            const names = (ev.entities || []).map(e => `${e.name}(${e.kind === 'character' ? '角' : e.kind === 'location' ? '景' : '物'})`).join('、');
            return `候选演员表定稿：发现 ${ev.n_entities} 个实体（涉及 ${ev.n_keyframes} 帧证据截图）` + (names ? ` ➔ ${names}` : '');
          }
          case 'tracking_start': return `开始对 ${ev.n_shots} 镜头执行 detect+track+re-ID 密集检测...`;
          case 'track_progress': {
            const yieldTxt = ev.n_tracklets != null ? ` ➔ 本镜头 ${ev.n_tracklets} 条轨迹` : (ev.n_entities != null ? ` ➔ 检测出 ${ev.n_entities} 实体` : '');
            return `镜头 ${ev.shot}/${ev.n_shots} 跟踪完成${yieldTxt}` + (ev.eta_seconds != null ? ` · 预计剩余 ${Math.round(ev.eta_seconds)}s` : '');
          }
          case 'identity': return `跨镜头 Re-ID 连接完成：联合 ${ev.n_entities} 实体 / ${ev.n_tracklet_spans} 段轨迹段`;
          case 'crop_qa': return `错配裁剪剔除完成 (${ev.method}) ➔ 已识别并自动删除 ${ev.n_flagged} 个瑕疵`;
          case 'naming_start': return `开始对 ${ev.n_entities} 个实体执行大模型 VLM 多模态命名...`;
          case 'naming_progress': return `命名进度: ${ev.done}/${ev.n_entities}` + (ev.name ? ` ➔ ${ev.name}` : '');
          case 'naming_done': return `大模型自动命名结束：共 ${ev.n_entities} 个实体名完成一致性映射`;
          case 'name_error': return `实体 ${ev.entity_id} 命名失败 (非致命): ${ev.error}`;
          case 'chunk_start': return `开始起草 ${c} · frames ${ev.frame_span ? ev.frame_span.join('-') : '?'}`;
          case 'role_start': return `${ev.role} 开始处理 ${c} · ${STAGE_ZH[ev.stage] || ev.stage}`;
          case 'role_end': return `${ev.role} 完成 ${c}` + (ev.prompt ? ` · prompt=${String(ev.prompt).slice(0, 48)}…` : '');
          case 'chunk_done': return `${c} 完成 · rounds=${ev.rounds} · ${ev.seconds}s` + (ev.flagged ? ' · FLAGGED' : '');
          case 'registry': return `资产增量更新 +${(ev.entities || []).length}`;
          case 'registry_final': return `资产定稿 ${(ev.entities || []).length} 个实体`;
          case 'run_done': return '流水线全部完成，可进入人工审核';
          default: return JSON.stringify(ev);
        }
      },
      getLogDetail(ev) {
        if (ev.kind === 'role_end' && ev.prompt) return ev.prompt;
        if (ev.kind === 'chunk_done' && ev.checks) return JSON.stringify(ev.checks, null, 2);
        if (ev.error) return ev.error;
        return '';
      }
    }
  };
})(window);
