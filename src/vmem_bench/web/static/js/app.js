/** MemStrata dashboard app entry (CDN Vue, no build). */
(function (global) {
  'use strict';
  const { createApp } = Vue;
  const Live = global.MemStrataLive;
  const Review = global.MemStrataReview;

  function mergeOptions(parts) {
    const out = { data() { return {}; }, computed: {}, watch: {}, methods: {} };
    const dataFns = [];
    for (const part of parts) {
      if (!part) continue;
      if (part.data) dataFns.push(part.data);
      Object.assign(out.computed, part.computed || {});
      Object.assign(out.watch, part.watch || {});
      Object.assign(out.methods, part.methods || {});
      if (part.mounted) out._mounted = (out._mounted || []).concat([part.mounted]);
      if (part.unmounted) out._unmounted = (out._unmounted || []).concat([part.unmounted]);
    }
    out.data = function () {
      const base = {};
      for (const fn of dataFns) Object.assign(base, fn.call(this));
      return base;
    };
    const mountedHooks = out._mounted || [];
    const unmountedHooks = out._unmounted || [];
    delete out._mounted;
    delete out._unmounted;
    if (mountedHooks.length) {
      out.mounted = function () {
        for (const h of mountedHooks) h.call(this);
      };
    }
    if (unmountedHooks.length) {
      out.unmounted = function () {
        for (const h of unmountedHooks) h.call(this);
      };
    }
    return out;
  }

  const core = {
    data() {
      return {
        connStatus: 'disconnected',
        activeMode: 'live',
        reviewSubTab: 'inbox',
        eventsLog: [],
        historyBuffer: [],
        currentFilter: 'all',
        autoScrollLog: true,
        logScrollProgress: 100,
        loadingHistory: true,
        historyTimeout: null,
        scrollTimeout: null,
        isAutoScrolling: false,
        isScrubbingLog: false,
        _scrubMoveHandler: null,
        _scrubUpHandler: null,
        runStatus: { movie_id: '—', stage: 'chunking', done: false, n_chunks: 0, n_entities: 0 },
        steps: {
          chunking: '镜头切分 (Chunking)',
          roster: '候选去重 (Roster)',
          tracking: '逐镜头跟踪 (Tracking)',
          identity: '跨镜身份 (Re-ID)',
          naming: '实体命名 (Naming)',
          drafting: '快编草稿 (Drafting)',
          done: '完成 (Done)'
        },
        liveState: {
          annotatorActive: false, annotatorStage: '', verifierActive: false, verifierStage: '',
          currentShot: 0, totalShots: 0, elapsedSeconds: 0, etaSeconds: null,
          nEntitiesTracked: 0, namingDone: 0, namingTotal: 0, currentNamingName: '',
          rosterDone: 0, rosterTotal: 0, rosterKnown: 0,
          completedChunks: 0, chunks: {}
        },
        liveAssets: [],
        gold: null,
        workingGold: null,
        goldLoadError: null,
        grayMerges: [],
        reviewQueue: { version: 1, items: [], summary: {} },
        identityPicks: {},
        reviewDispositions: {},
        reviewNotes: {},
        stateEventReviews: {},
        promptReviews: {},
        reviewActionError: '',
        focusedChunkId: null,
        movies: [], selectedMovie: '',
        entityFilter: 'all',
        entitySearch: '',
        chunkFilter: 'all',
        chunkSort: 'id',
        selectedChunkId: null,
        lightboxSrc: null,
        freezeSuccess: false,
        freezeError: null,
        previewResult: null,
        previewing: false,
        savingDraft: false,
        applyingPatch: false,
        freezing: false,
        es: null,
        statusTimer: null
      };
    },
    computed: {
      progressPercent() {
        if (this.activeMode === 'live') {
          const stage = this.runStatus.stage;
          if (stage === 'chunking') return 5;
          if (stage === 'roster') {
            if (!this.liveState.rosterTotal) return 15;
            return Math.round(15 + (this.liveState.rosterDone / this.liveState.rosterTotal) * 5);
          }
          if (stage === 'tracking') {
            if (!this.liveState.totalShots) return 20;
            return Math.round(20 + (this.liveState.currentShot / this.liveState.totalShots) * 40);
          }
          if (stage === 'identity') return 65;
          if (stage === 'naming') {
            if (!this.liveState.namingTotal) return 70;
            return Math.round(70 + (this.liveState.namingDone / this.liveState.namingTotal) * 15);
          }
          if (stage === 'drafting') {
            if (!this.runStatus.n_chunks) return 88;
            return Math.round(85 + (this.liveState.completedChunks / this.runStatus.n_chunks) * 14);
          }
          if (stage === 'done' || this.runStatus.done) return 100;
          return 0;
        }
        return 100;
      },
      progressText() {
        if (this.activeMode === 'live') {
          const stage = this.runStatus.stage;
          if (stage === 'chunking') return { title: '正在切分镜头', detail: 'Decomposing video chunks...' };
          if (stage === 'roster') {
            const done = this.liveState.rosterDone;
            const total = this.liveState.rosterTotal;
            if (total) return { title: '正在构建全片演员表', detail: `VLM roster ${done}/${total} 批 · 已发现 ${this.liveState.rosterKnown} 候选` };
            return { title: '正在去重候选', detail: 'Selecting keyframes & discovering roster...' };
          }
          if (stage === 'tracking') {
            const shot = this.liveState.currentShot;
            const total = this.liveState.totalShots;
            const eta = this.liveState.etaSeconds !== null ? ` (预计剩余 ${this.liveState.etaSeconds}s)` : '';
            return { title: '逐镜头目标追踪中', detail: `Shot ${shot}/${total} · 累积发现 ${this.liveState.nEntitiesTracked} 实体${eta}` };
          }
          if (stage === 'identity') return { title: '跨镜轨迹身份联合', detail: 'Binding tracklets identity...' };
          if (stage === 'naming') {
            const done = this.liveState.namingDone;
            const total = this.liveState.namingTotal;
            const name = this.liveState.currentNamingName ? ` [正在识别: ${this.liveState.currentNamingName}]` : '';
            return { title: '正在逐实体智能命名', detail: `VLM naming ${done}/${total}${name}` };
          }
          if (stage === 'drafting') {
            const done = this.liveState.completedChunks;
            const total = this.runStatus.n_chunks || 0;
            return { title: '正在快编起草 Prompts', detail: `Drafted ${done}/${total} chunks` };
          }
          if (stage === 'done' || this.runStatus.done) return { title: '标注运行圆满跑通', detail: 'Waiting for reviewer...' };
          return { title: '初始化看板', detail: 'Waiting for stream...' };
        }
        return { title: '正在审核定稿中', detail: 'Reviewing gold data' };
      },
      connText() {
        if (this.connStatus === 'connected') return '已连入 (SSE)';
        if (this.connStatus === 'reconnecting') return '重连中...';
        return '断开';
      }
    },
    methods: {
      getStepNum(step) {
        return Object.keys(this.steps).indexOf(step) + 1;
      },
      isStepDone(step) {
        const keys = Object.keys(this.steps);
        return keys.indexOf(step) < keys.indexOf(this.runStatus.stage);
      }
    },
    mounted() {
      fetch('/movies').then(r => r.json()).then(d => { this.movies = d.movies || []; this.selectedMovie = d.selected || ''; }).catch(() => {});
      this.initSSE();
      this.pollStatus();
      this.statusTimer = setInterval(() => this.pollStatus(), 2500);
      window.addEventListener('keydown', (e) => {
        if (e.key === 'Escape' && this.lightboxSrc) this.lightboxSrc = null;
      });
      this.$nextTick(() => {
        this.forceScrollToBottom();
        setTimeout(() => this.forceScrollToBottom(), 200);
      });
    },
    unmounted() {
      if (this.es) this.es.close();
      if (this.statusTimer) clearInterval(this.statusTimer);
      if (this._scrubUpHandler) this._scrubUpHandler();
    }
  };

  const options = mergeOptions([core, Live, Review]);
  createApp(options).mount('#app');
})(window);
