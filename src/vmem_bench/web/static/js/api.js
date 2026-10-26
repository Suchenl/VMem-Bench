/** MemStrata dashboard API helpers (no build). */
(function (global) {
  'use strict';

  const STAGE_ZH = {
    chunking: '镜头切分',
    roster: '候选去重',
    tracking: '逐镜头跟踪',
    identity: '跨镜身份ID',
    naming: '实体大模型命名',
    drafting: '快编 prompt',
    done: '全部完成'
  };

  function emptyPatch() {
    return {
      schema_version: '2.0.0',
      merges: [],
      drops: [],
      renames: {},
      field_edits: [],
      dispositions: {},
      state_event_reviews: {},
      prompt_reviews: {}
    };
  }

  /** Project schema-shaped /gold into ID-keyed maps the review UI expects. */
  function indexGoldPayload(data) {
    if (!data || typeof data !== 'object') return data;
    const registrySrc = data.registry;
    const chunksSrc = data.chunks;
    const registry = {};
    const entities = Array.isArray(registrySrc)
      ? registrySrc
      : (registrySrc && Array.isArray(registrySrc.entities) ? registrySrc.entities : null);
    if (entities) {
      for (const e of entities) {
        if (!e || !e.entity_id) continue;
        registry[e.entity_id] = Object.assign({}, e, {
          isDropped: false,
          mergedInto: '',
          representations: Array.isArray(e.representations) ? e.representations : []
        });
      }
    } else if (registrySrc && typeof registrySrc === 'object') {
      for (const eid of Object.keys(registrySrc)) {
        const e = registrySrc[eid];
        if (!e || typeof e !== 'object' || !e.entity_id) continue;
        registry[eid] = Object.assign({}, e, {
          isDropped: !!e.isDropped,
          mergedInto: e.mergedInto || ''
        });
      }
    }
    const chunks = {};
    const chunkList = Array.isArray(chunksSrc)
      ? chunksSrc
      : (chunksSrc && Array.isArray(chunksSrc.chunks) ? chunksSrc.chunks : null);
    if (chunkList) {
      for (const c of chunkList) {
        if (!c || c.chunk_id === undefined || c.chunk_id === null) continue;
        chunks[String(c.chunk_id)] = Object.assign({}, c);
      }
    } else if (chunksSrc && typeof chunksSrc === 'object') {
      for (const cid of Object.keys(chunksSrc)) {
        const c = chunksSrc[cid];
        if (!c || typeof c !== 'object' || c.chunk_id === undefined) continue;
        chunks[String(cid)] = Object.assign({}, c);
      }
    }
    return Object.assign({}, data, {
      registry: registry,
      chunks: chunks,
      qa: Array.isArray(data.qa) ? data.qa : [],
      layout: data.layout || {}
    });
  }

  function cloneIndexedGold(data) {
    return JSON.parse(JSON.stringify(indexGoldPayload(data)));
  }

  function jsonFetch(url, options) {
    return fetch(url, options).then(function (res) {
      return res.json().then(function (body) {
        return { res: res, body: body };
      }).catch(function () {
        return { res: res, body: null };
      });
    });
  }

  function clearReviewDraft() {
    return fetch('/review/patch', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify(emptyPatch())
    }).then(function (res) { return res.json(); });
  }

  global.MemStrataApi = {
    STAGE_ZH: STAGE_ZH,
    emptyPatch: emptyPatch,
    indexGoldPayload: indexGoldPayload,
    cloneIndexedGold: cloneIndexedGold,
    jsonFetch: jsonFetch,
    clearReviewDraft: clearReviewDraft
  };
})(window);
