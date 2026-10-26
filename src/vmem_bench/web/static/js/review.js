/** Review mode: gold normalize, draft, patch, apply, freeze. */
(function (global) {
  'use strict';
  const Api = global.MemStrataApi;

  global.MemStrataReview = {
    computed: {
      registryCount() {
        return this.gold && this.gold.registry ? Object.keys(this.gold.registry).length : 0;
      },
      chunkCount() {
        return this.gold && this.gold.chunks ? Object.keys(this.gold.chunks).length : 0;
      },
      currentPatch() {
        if (!this.gold || !this.workingGold) return Api.emptyPatch();
        const merges = [];
        const drops = [];
        const renames = {};
        const field_edits = [];
        for (const eid in this.workingGold.registry) {
          const workingEntity = this.workingGold.registry[eid];
          const originalEntity = this.gold.registry[eid];
          if (!originalEntity) continue;
          if (workingEntity.name !== originalEntity.name) renames[eid] = workingEntity.name;
          if (workingEntity.isDropped) {
            drops.push(eid);
          } else if (workingEntity.mergedInto && workingEntity.mergedInto !== '') {
            merges.push([workingEntity.mergedInto, eid]);
          }
          if (workingEntity.description !== originalEntity.description) {
            field_edits.push({ path: `entities[${eid}].description`, value: workingEntity.description });
          }
        }
        for (const cid in this.workingGold.chunks) {
          const workingChunk = this.workingGold.chunks[cid];
          const originalChunk = this.gold.chunks[cid];
          if (!originalChunk) continue;
          if (workingChunk.prompt !== originalChunk.prompt) {
            field_edits.push({ path: `chunks[${cid}].prompt`, value: workingChunk.prompt });
          }
        }
        // Align with review.apply_patch: when dispositions are present, every
        // merge/drop must carry a matching disposition action + non-empty reason.
        const dispositions = {};
        for (const eid in this.reviewDispositions) {
          const value = this.reviewDispositions[eid];
          if (value && value.action && value.reason) dispositions[eid] = value;
        }
        for (const eid of drops) {
          if (!dispositions[eid]) {
            dispositions[eid] = {
              action: 'dropped',
              reason: (this.reviewDispositions[eid] && this.reviewDispositions[eid].reason) ||
                'dropped in entity editor'
            };
          }
        }
        for (const [target, source] of merges) {
          if (!dispositions[source]) {
            dispositions[source] = {
              action: 'merged',
              reason: (this.reviewDispositions[source] && this.reviewDispositions[source].reason) ||
                ('merged into ' + target + ' in entity editor')
            };
          }
        }
        const state_event_reviews = {};
        for (const eventId in this.stateEventReviews) {
          const value = this.stateEventReviews[eventId];
          if (value && value.action && value.reason) state_event_reviews[eventId] = value;
        }
        const prompt_reviews = {};
        for (const itemId in this.promptReviews) {
          const value = this.promptReviews[itemId];
          if (value && value.action && value.reason) prompt_reviews[itemId] = value;
        }
        return {
          schema_version: '2.0.0', merges, drops, renames, field_edits, dispositions,
          state_event_reviews, prompt_reviews
        };
      },
      patchCount() {
        const p = this.currentPatch;
        const renameCount = Object.keys(p.renames).length;
        const dropCount = p.drops.length;
        const mergeCount = p.merges.length;
        const promptCount = p.field_edits.filter(fe => fe.path.includes('chunks[')).length;
        const descCount = p.field_edits.filter(fe => fe.path.includes('entities[')).length;
        const dispositionCount = Object.keys(p.dispositions || {}).length
          + Object.keys(p.state_event_reviews || {}).length
          + Object.keys(p.prompt_reviews || {}).length;
        return {
          total: renameCount + dropCount + mergeCount + promptCount + descCount + dispositionCount,
          renames: renameCount, drops: dropCount, merges: mergeCount,
          prompts: promptCount, descriptions: descCount, dispositions: dispositionCount
        };
      },
      maxFrame() {
        if (!this.workingGold) return 1;
        let maxF = 1;
        for (const cid in this.workingGold.chunks) {
          const c = this.workingGold.chunks[cid];
          if (c.frame_span && c.frame_span[1] > maxF) maxF = c.frame_span[1];
        }
        return maxF;
      },
      filteredEntities() {
        if (!this.workingGold) return [];
        let list = Object.values(this.workingGold.registry);
        if (this.entitySearch) {
          const s = this.entitySearch.toLowerCase();
          list = list.filter(e =>
            (e.name && e.name.toLowerCase().includes(s)) ||
            (e.entity_id && e.entity_id.toLowerCase().includes(s)) ||
            (e.description && e.description.toLowerCase().includes(s)));
        }
        if (this.entityFilter === 'flagged') {
          list = list.filter(e => {
            const reps = e.representations || [];
            return reps.some(r => (r.qa && r.qa.flagged) || r.bbox_source === 'vlm_fallback' ||
              (r.bbox_source === 'grounding_dino' && r.qa && r.qa.grounding_score < 0.5));
          });
        } else if (this.entityFilter !== 'all') {
          list = list.filter(e => e.kind && e.kind.toLowerCase() === this.entityFilter);
        }
        return list;
      },
      filteredChunks() {
        if (!this.workingGold) return [];
        let list = Object.values(this.workingGold.chunks);
        if (this.chunkFilter === 'flagged') {
          list = list.filter(c => this.workingGold.qa.some(q => q.chunk_id === c.chunk_id && q.flagged));
        } else if (this.chunkFilter === 'high_risk') {
          list = list.filter(c => this.getChunkRiskScore(c) >= 3.0);
        }
        if (this.chunkSort === 'risk') list.sort((a, b) => this.getChunkRiskScore(b) - this.getChunkRiskScore(a));
        else list.sort((a, b) => a.chunk_id - b.chunk_id);
        return list;
      }
    },
    methods: {
      switchToReview() {
        if (this.runStatus.done) {
          this.activeMode = 'review';
          if (!this.gold) this.fetchGold();
        }
      },
      selectMovie() {
        fetch('/review/select', {method: 'POST', headers: {'Content-Type': 'application/json'}, body: JSON.stringify({movie_id: this.selectedMovie})})
          .then(r => r.json()).then(d => { if (d.ok) window.location.reload(); else this.reviewActionError = d.error; });
      },
      _primeWorkingFlags(working) {
        for (const eid in working.registry) {
          const ent = working.registry[eid];
          ent.isDropped = false;
          ent.mergedInto = '';
        }
      },
      fetchGold() {
        this.goldLoadError = null;
        fetch('/gold')
          .then(res => {
            if (res.status === 409) throw new Error('Gold 尚未就绪（流水线仍在运行）');
            if (!res.ok) throw new Error('HTTP ' + res.status);
            return res.json();
          })
          .then(data => {
            const indexed = Api.indexGoldPayload(data);
            this.gold = indexed;
            this.workingGold = JSON.parse(JSON.stringify(indexed));
            this._primeWorkingFlags(this.workingGold);
            this.goldLoadError = null;
            this.fetchDraft();
            this.fetchGrayMerges();
            this.fetchReviewQueue();
          })
          .catch(err => {
            console.error(err);
            this.goldLoadError = err.message || String(err);
          });
      },
      fetchGrayMerges() {
        fetch('/auto_review')
          .then(res => res.json())
          .then(data => {
            const merges = (data && data.gray_merges) || [];
            // Only keep proposals whose both ids still exist in working gold.
            this.grayMerges = merges.filter(m =>
              m && m.keep && m.merge
              && this.workingGold && this.workingGold.registry[m.keep]
              && this.workingGold.registry[m.merge]);
          })
          .catch(() => { this.grayMerges = []; });
      },
      fetchReviewQueue() {
        fetch('/review_queue')
          .then(res => res.json())
          .then(data => {
            this.reviewQueue = data && Array.isArray(data.items)
              ? data : { version: 1, items: [], summary: {} };
          })
          .catch(() => { this.reviewQueue = { version: 1, items: [], summary: {} }; });
      },
      identityQueueItems() {
        return ((this.reviewQueue && this.reviewQueue.items) || [])
          .filter(item => item.kind === 'identity');
      },
      identityEvidence(item) {
        return item.evidence && (item.evidence.candidate || ((item.evidence.candidates || [])[0]));
      },
      identityCandidates(item) {
        if (!item || !item.evidence) return [];
        if (Array.isArray(item.evidence.candidates) && item.evidence.candidates.length) {
          return item.evidence.candidates;
        }
        return item.evidence.candidate ? [item.evidence.candidate] : [];
      },
      queueItems(kind) {
        const items = (this.reviewQueue && this.reviewQueue.items) || [];
        return kind ? items.filter(item => item.kind === kind) : items;
      },
      queueReason(item) {
        return String(this.reviewNotes[item.id] || '').trim();
      },
      stateEventIds(item) {
        const events = (item.evidence && item.evidence.events) || [];
        if (events.length) return events.map(e => String(e.event_id)).filter(Boolean);
        // Legacy per-event queue item: the id suffix IS the event id.
        return [String(item.id || '').replace(/^state:/, '')];
      },
      isQueueItemDecided(item) {
        if (!item) return false;
        if (item.kind === 'state') {
          return this.stateEventIds(item).every(id => {
            const decision = this.stateEventReviews[id];
            return !!(decision && decision.action && decision.reason);
          });
        }
        if (item.kind === 'prompt') {
          const decision = this.promptReviews[item.id];
          return !!(decision && decision.action && decision.reason);
        }
        if (item.kind === 'lint') {
          // Lint blockers are fixed by editing gold / applying patch, not by inbox disposition.
          return false;
        }
        if (item.kind === 'identity') {
          const ids = (item.entity_ids || []).filter(Boolean);
          if (!ids.length) return false;
          if (ids.every(eid => this.reviewDispositions[eid])) return true;
          // Merge pattern: all but one are merged into the surviving keep id.
          const merged = ids.filter(eid =>
            this.reviewDispositions[eid] && this.reviewDispositions[eid].action === 'merged');
          if (merged.length !== ids.length - 1) return false;
          const keep = ids.find(eid =>
            !this.reviewDispositions[eid] || this.reviewDispositions[eid].action !== 'merged');
          if (!keep || !this.workingGold) return false;
          return merged.every(eid => {
            const ent = this.workingGold.registry[eid];
            return ent && ent.mergedInto === keep;
          });
        }
        return false;
      },
      setDisposition(entityId, action, reason) {
        this.reviewDispositions[entityId] = { action: action, reason: reason };
      },
      // ---- identity card: self-contained entity grid + multi-select merge -------------------
      entityCover(eid) {
        const ent = this.workingGold && this.workingGold.registry[eid];
        if (!ent) return '';
        if (ent.cover_crop) return ent.cover_crop;
        const rep = (ent.representations || []).find(r => r.crop_path);
        return rep ? rep.crop_path : '';
      },
      entityBrief(eid) {
        const ent = this.workingGold && this.workingGold.registry[eid];
        if (!ent) return eid;
        const n = (ent.representations || []).length;
        return (ent.name || eid) + ' · ' + n + ' 证据';
      },
      isPicked(item, eid) {
        return !!(this.identityPicks[item.id] && this.identityPicks[item.id][eid]);
      },
      togglePick(item, eid) {
        const picks = Object.assign({}, this.identityPicks[item.id] || {});
        picks[eid] = !picks[eid];
        this.identityPicks = Object.assign({}, this.identityPicks, { [item.id]: picks });
      },
      pickedIds(item) {
        const picks = this.identityPicks[item.id] || {};
        return (item.entity_ids || []).filter(eid => picks[eid]);
      },
      applySuggestionGroup(item, group) {
        // One click selects a machine-suggested same-individual subgroup on the grid.
        const picks = {};
        (group || []).forEach(eid => { picks[eid] = true; });
        this.identityPicks = Object.assign({}, this.identityPicks, { [item.id]: picks });
      },
      suggestionGroups(item) {
        const ms = item.evidence && item.evidence.machine_suggestion;
        return (ms && ms.same_individual_groups) || [];
      },
      _defaultReason(item, action) {
        const machine = this.suggestionGroups(item).length ? '（含机器同体建议）' : '';
        const map = { merge: '人工确认为同一个体' + machine, distinct: '人工确认为不同个体',
                      confirmed: '人工确认不可逆状态变化', rejected: '人工判定为可逆/无效事件',
                      adequate: '人工确认 prompt 覆盖充分', missing_entity: '人工确认 prompt 漏实体',
                      narrative_conflict: '人工确认叙述冲突' };
        return this.queueReason(item) || map[action] || '人工审核决定';
      },
      mergeSelected(item) {
        this.reviewActionError = '';
        const picked = this.pickedIds(item);
        if (picked.length < 2) {
          this.reviewActionError = '请先在网格里点选 ≥2 个属于同一个体的实体，再合并。';
          return;
        }
        if (!this.workingGold) { this.reviewActionError = 'Gold 尚未加载。'; return; }
        const reason = this._defaultReason(item, 'merge');
        // Survivor = most evidence among the picked.
        const keep = picked.slice().sort((a, b) =>
          ((this.workingGold.registry[b] || {}).representations || []).length -
          ((this.workingGold.registry[a] || {}).representations || []).length)[0];
        for (const mergeId of picked) {
          if (mergeId === keep) continue;
          const source = this.workingGold.registry[mergeId];
          if (!source) { this.reviewActionError = '候选实体已不在草稿中；请刷新 Gold。'; return; }
          source.mergedInto = keep;
          source.isDropped = false;
          this.setDisposition(mergeId, 'merged', reason);
        }
        this.setDisposition(keep, 'kept_distinct', reason);
        this.identityPicks = Object.assign({}, this.identityPicks, { [item.id]: {} });
      },
      decideIdentity(item, decision) {
        this.reviewActionError = '';
        const ids = (item.entity_ids || []).filter(Boolean);
        const reason = this._defaultReason(item, decision);
        if (!ids.length) {
          this.reviewActionError = '队列项缺少 entity_ids，无法记录处置。';
          return;
        }
        if (decision === 'merge') {
          // Grid multi-select is the primary path; "merge all" remains for 2-entity cards.
          if (this.pickedIds(item).length >= 2) return this.mergeSelected(item);
          if (ids.length < 2) {
            this.reviewActionError = '单实体项无法合并；请用「确认身份」标记为保持独立。';
            return;
          }
          if (!this.workingGold) {
            this.reviewActionError = 'Gold 尚未加载。';
            return;
          }
          const keep = ids[0];
          if (!this.workingGold.registry[keep]) {
            this.reviewActionError = '保留实体已不在当前草稿中；请刷新 Gold。';
            return;
          }
          for (const mergeId of ids.slice(1)) {
            const source = this.workingGold.registry[mergeId];
            if (!source) {
              this.reviewActionError = '候选实体已不在当前草稿中；请刷新 Gold。';
              return;
            }
            source.mergedInto = keep;
            source.isDropped = false;
            this.setDisposition(mergeId, 'merged', reason);
          }
        } else {
          // kept_distinct: pair, cluster, or single-entity identity confirmation
          ids.forEach(eid => {
            this.setDisposition(eid, 'kept_distinct', reason);
            const ent = this.workingGold && this.workingGold.registry[eid];
            if (ent) {
              ent.mergedInto = '';
              ent.isDropped = false;
            }
          });
        }
      },
      focusQueueItem(item) {
        const { nextTick } = Vue;
        if (item.kind === 'prompt' && item.affected_chunk_ids && item.affected_chunk_ids.length) {
          this.focusedChunkId = item.affected_chunk_ids[0];
          this.reviewSubTab = 'chunks';
          this.chunkFilter = 'all';
          nextTick(() => {
            const el = document.getElementById('chunk-card-' + this.focusedChunkId);
            if (el && el.scrollIntoView) el.scrollIntoView({ behavior: 'smooth', block: 'start' });
          });
        } else if (item.kind === 'identity' || item.kind === 'state' || item.kind === 'lint') {
          this.entitySearch = (item.entity_ids || []).filter(Boolean)[0] || '';
          this.reviewSubTab = 'entities';
          if (item.kind === 'lint' && item.affected_chunk_ids && item.affected_chunk_ids.length) {
            this.focusedChunkId = item.affected_chunk_ids[0];
            this.reviewSubTab = 'chunks';
            nextTick(() => {
              const el = document.getElementById('chunk-card-' + this.focusedChunkId);
              if (el && el.scrollIntoView) el.scrollIntoView({ behavior: 'smooth', block: 'start' });
            });
          }
        }
      },
      stateEvents(item) {
        return (item.evidence && item.evidence.events) ||
               (item.evidence && item.evidence.event_id ? [item.evidence] : []);
      },
      decideStateEvent(item, action) {
        const reason = this._defaultReason(item, action);
        this.reviewActionError = '';
        this.stateEventIds(item).forEach(id => {
          this.stateEventReviews[id] = { action: action, reason: reason };
        });
      },
      decideOneStateEvent(item, eventId, action) {
        // Per-event exception inside a grouped card (e.g. confirm all but one).
        this.stateEventReviews[String(eventId)] = {
          action: action, reason: this._defaultReason(item, action) };
        this.stateEventReviews = Object.assign({}, this.stateEventReviews);
      },
      promptFindings(item) {
        const label = {
          prompt_missing_present_entity: 'prompt 未提到在场实体',
          empty_prompt: '该 chunk 的 prompt 为空',
          first_missing_current_crop: '首次出场实体缺本 chunk 证据 crop',
          present_missing_crop: '在场实体无可用 crop',
          character_missed_in_tracking: '起草看到角色但跟踪未检出',
        };
        return ((item.evidence && item.evidence.findings) || []).map(f =>
          (label[f.code] || f.code) + (f.entity_id ? '：' + f.entity_id : ''));
      },
      decidePrompt(item, action) {
        const reason = this._defaultReason(item, action);
        this.reviewActionError = '';
        this.promptReviews[item.id] = { action: action, reason: reason };
        if (action !== 'adequate') this.focusQueueItem(item);
      },
      mustReviewProgress() {
        const items = this.queueItems().filter(i => i.review_tier === 'must');
        const done = items.filter(i => this.isQueueItemDecided(i)).length;
        return { done: done, total: items.length };
      },
      previewPatch() {
        this.previewing = true;
        this.previewResult = null;
        fetch('/review/preview', { method: 'POST', headers: { 'Content-Type': 'application/json' },
          body: JSON.stringify(this.currentPatch) })
          .then(res => res.json())
          .then(data => { this.previewResult = data; this.previewing = false; })
          .catch(err => { this.previewResult = { ok: false, errors: [{ message: String(err) }] }; this.previewing = false; });
      },
      acceptGrayMerge(m) {
        if (!this.workingGold || !m) return;
        const src = this.workingGold.registry[m.merge];
        const dst = this.workingGold.registry[m.keep];
        if (!src || !dst) return;
        src.mergedInto = m.keep;
        src.isDropped = false;
        this.setDisposition(m.merge, 'merged',
          'accepted gray merge proposal (text=' + m.text_cos + ' body=' + m.body_cos + ')');
      },
      acceptAllGrayMerges() {
        (this.grayMerges || []).forEach(m => this.acceptGrayMerge(m));
      },
      fetchDraft() {
        fetch('/review/patch')
          .then(res => res.json())
          .then(patch => {
            if (!patch || Object.keys(patch).length === 0) return;
            if (!patch.schema_version && !patch.renames && !patch.drops && !patch.merges && !patch.field_edits
                && !patch.dispositions && !patch.state_event_reviews && !patch.prompt_reviews) return;
            if (patch.renames) {
              for (const eid in patch.renames) {
                if (this.workingGold.registry[eid]) this.workingGold.registry[eid].name = patch.renames[eid];
              }
            }
            if (patch.drops) {
              patch.drops.forEach(eid => {
                if (this.workingGold.registry[eid]) this.workingGold.registry[eid].isDropped = true;
              });
            }
            if (patch.merges) {
              patch.merges.forEach(([target, source]) => {
                if (this.workingGold.registry[source]) this.workingGold.registry[source].mergedInto = target;
              });
            }
            if (patch.field_edits) {
              patch.field_edits.forEach(fe => {
                const path = fe.path;
                const value = fe.value;
                if (path.startsWith('entities[')) {
                  const idx = path.indexOf('].description');
                  if (idx !== -1) {
                    const eid = path.substring(9, idx);
                    if (this.workingGold.registry[eid]) this.workingGold.registry[eid].description = value;
                  }
                } else if (path.startsWith('chunks[')) {
                  const idx = path.indexOf('].prompt');
                  if (idx !== -1) {
                    const cid = path.substring(7, idx);
                    if (this.workingGold.chunks[cid]) this.workingGold.chunks[cid].prompt = value;
                  }
                }
              });
            }
            if (patch.dispositions) this.reviewDispositions = Object.assign({}, patch.dispositions);
            if (patch.state_event_reviews) this.stateEventReviews = Object.assign({}, patch.state_event_reviews);
            if (patch.prompt_reviews) this.promptReviews = Object.assign({}, patch.prompt_reviews);
          })
          .catch(err => console.error('Error loading draft', err));
      },
      getMergeCandidates(currentEid) {
        if (!this.workingGold) return [];
        return Object.values(this.workingGold.registry).filter(e =>
          e.entity_id !== currentEid && !e.isDropped && (!e.mergedInto || e.mergedInto === ''));
      },
      resetReviewEdits() {
        if (!confirm('确定要重置当前所有的修改吗？未保存和应用的改动将全部回退。')) return;
        this.workingGold = JSON.parse(JSON.stringify(this.gold));
        this._primeWorkingFlags(this.workingGold);
        this.freezeError = null;
        this.freezeSuccess = false;
        this.previewResult = null;
        this.reviewDispositions = {};
        this.reviewNotes = {};
        this.stateEventReviews = {};
        this.promptReviews = {};
        this.focusedChunkId = null;
        Api.clearReviewDraft().catch(() => {});
      },
      saveDraft() {
        this.savingDraft = true;
        this.freezeError = null;
        this.freezeSuccess = false;
        fetch('/review/patch', {
          method: 'POST',
          headers: { 'Content-Type': 'application/json' },
          body: JSON.stringify(this.currentPatch)
        })
          .then(res => res.json())
          .then(data => {
            this.savingDraft = false;
            if (data.ok) alert('草稿暂存成功！');
            else alert('草稿保存失败: ' + data.error);
          })
          .catch(err => {
            this.savingDraft = false;
            alert('网络错误: ' + err);
          });
      },
      applyPatch() {
        this.applyingPatch = true;
        this.freezeError = null;
        this.freezeSuccess = false;
        fetch('/review/apply', {
          method: 'POST',
          headers: { 'Content-Type': 'application/json' },
          body: JSON.stringify(this.currentPatch)
        })
          .then(res => {
            if (res.status === 400 || res.status === 409) {
              return res.json().then(d => { throw new Error(d.error); });
            }
            return res.json();
          })
          .then(newGold => {
            const indexed = Api.indexGoldPayload(newGold);
            this.gold = indexed;
            this.workingGold = JSON.parse(JSON.stringify(indexed));
            this._primeWorkingFlags(this.workingGold);
            this.reviewDispositions = {};
            this.reviewNotes = {};
            this.stateEventReviews = {};
            this.promptReviews = {};
            this.focusedChunkId = null;
            this.fetchReviewQueue();
            return Api.clearReviewDraft().then(() => {
              this.applyingPatch = false;
              alert('已应用全部更改！已重置工作契约。');
            });
          })
          .catch(err => {
            this.applyingPatch = false;
            alert('应用补丁失败: ' + err.message);
          });
      },
      freezeGold() {
        this.freezing = true;
        this.freezeError = null;
        this.freezeSuccess = false;
        const { nextTick } = Vue;
        fetch('/review/freeze', {
          method: 'POST',
          headers: { 'Content-Type': 'application/json' },
          body: JSON.stringify({})
        })
          .then(res => res.json().then(data => {
            if (!res.ok) throw new Error(data.error);
            return data;
          }))
          .then(data => {
            this.freezing = false;
            if (data.ok) {
              this.freezeSuccess = true;
              if (this.gold) this.gold.human_reviewed = true;
              if (this.workingGold) this.workingGold.human_reviewed = true;
              alert('定稿冻结成功！Gold 数据已过全部 Lint 约束。');
            }
          })
          .catch(err => {
            this.freezing = false;
            this.freezeError = err.message;
            nextTick(() => {
              const box = this.$refs.freezeErrorBox;
              if (box && box.scrollIntoView) box.scrollIntoView({ behavior: 'smooth', block: 'nearest' });
              const panel = this.$refs.reviewPanel;
              if (panel) panel.scrollTop = panel.scrollHeight;
            });
          });
      },
      getFps() {
        if (this.workingGold && this.workingGold.layout) return this.workingGold.layout.fps || 24;
        return 24;
      },
      getKindClass(kind) {
        const k = kind ? kind.toLowerCase() : '';
        if (k === 'character') return 'bg-accent/15 border border-accent/25 text-accent';
        if (k === 'prop') return 'bg-violet/15 border border-violet/25 text-violet';
        if (k === 'location') return 'bg-success/15 border border-success/25 text-success';
        return 'bg-bg text-dimText border border-border';
      },
      getKindBadgeClass(kind) {
        const k = kind ? kind.toLowerCase() : '';
        if (k === 'character') return 'bg-accent/20 border-accent/40 text-accent';
        if (k === 'prop') return 'bg-violet/20 border-violet/40 text-violet';
        if (k === 'location') return 'bg-success/20 border-success/40 text-success';
        return 'bg-bg text-dimText';
      },
      getKindBadgeTextClass(kind) {
        const k = kind ? kind.toLowerCase() : '';
        if (k === 'character') return 'text-accent font-bold';
        if (k === 'prop') return 'text-violet font-bold';
        if (k === 'location') return 'text-success font-bold';
        return 'text-dimText font-bold';
      },
      getKindCount(kind) {
        if (!this.workingGold) return 0;
        return Object.values(this.workingGold.registry).filter(e => e.kind && e.kind.toLowerCase() === kind).length;
      },
      getFlaggedEntitiesCount() {
        if (!this.workingGold) return 0;
        return Object.values(this.workingGold.registry).filter(e => {
          const reps = e.representations || [];
          return reps.some(r => (r.qa && r.qa.flagged) || r.bbox_source === 'vlm_fallback' ||
            (r.bbox_source === 'grounding_dino' && r.qa && r.qa.grounding_score < 0.5));
        }).length;
      },
      getHighRiskChunksCount() {
        if (!this.workingGold) return 0;
        return Object.values(this.workingGold.chunks).filter(c => this.getChunkRiskScore(c) >= 3.0).length;
      },
      getEntityBorderHighlight(e) {
        if (e.isDropped) return '';
        const reps = e.representations || [];
        const hasFlaggedRep = reps.some(r => (r.qa && r.qa.flagged) || r.bbox_source === 'vlm_fallback' ||
          (r.bbox_source === 'grounding_dino' && r.qa && r.qa.grounding_score < 0.5));
        if (hasFlaggedRep) return 'border-warning/60 shadow-[0_0_12px_rgba(245,158,11,0.08)]';
        return '';
      },
      getEntityNameBorderClass(e) {
        if (this.gold && this.gold.registry[e.entity_id] && e.name !== this.gold.registry[e.entity_id].name) {
          return 'border-warning focus:border-warning';
        }
        return 'border-border/80 focus:border-accent';
      },
      getEntityDescBorderClass(e) {
        if (this.gold && this.gold.registry[e.entity_id] && e.description !== this.gold.registry[e.entity_id].description) {
          return 'border-warning focus:border-warning';
        }
        return 'border-border/80 focus:border-accent';
      },
      hasWarningReps(e) {
        const reps = e.representations || [];
        return reps.some(r => (r.qa && r.qa.flagged) || r.bbox_source === 'vlm_fallback' ||
          (r.bbox_source === 'grounding_dino' && r.qa && r.qa.grounding_score < 0.5));
      },
      getRepBorderClass(rep) {
        if (rep.qa && rep.qa.flagged) return 'border-danger/80';
        if (rep.bbox_source === 'vlm_fallback') return 'border-warning/80';
        if (rep.bbox_source === 'grounding_dino' && rep.qa && rep.qa.grounding_score < 0.5) return 'border-warning/50';
        return 'border-border/80';
      },
      getChunkRiskScore(c) {
        if (!this.workingGold) return 0.5;
        let risk = 0.5;
        for (const tag of (c.scenario_tags || [])) {
          if (tag === 'state-change' || tag === 'multi-instance' || tag === 're-appearance') risk += 2.0;
        }
        for (const eid of (c.present || [])) {
          const e = this.workingGold.registry[eid];
          if (!e) continue;
          for (const r of (e.representations || [])) {
            if (r.chunk_id !== c.chunk_id) continue;
            if (r.bbox_source === 'vlm_fallback') risk += 1.0;
            const gs = parseFloat((r.qa && r.qa.grounding_score) || 0.0);
            if (r.bbox_source === 'grounding_dino' && gs < 0.5) risk += 1.0;
          }
        }
        return risk;
      },
      getChunkRiskBadgeClass(c) {
        const risk = this.getChunkRiskScore(c);
        if (risk >= 4.0) return 'bg-danger/10 border border-danger/30 text-danger';
        if (risk >= 2.5) return 'bg-warning/10 border border-warning/30 text-warning';
        return 'bg-success/10 border border-success/30 text-success';
      },
      getChunkCardHighlight(c) {
        if (this.isQAChunkFlagged(c.chunk_id)) {
          return 'border-warning bg-warning/[0.01] shadow-[0_0_12px_rgba(245,158,11,0.06)]';
        }
        if (this.getChunkRiskScore(c) >= 4.0) return 'shadow-[0_0_8px_rgba(239,68,68,0.04)]';
        return '';
      },
      isPromptModified(c) {
        if (!this.gold || !this.gold.chunks[String(c.chunk_id)]) return false;
        return c.prompt !== this.gold.chunks[String(c.chunk_id)].prompt;
      },
      isQAChunkFlagged(cid) {
        if (!this.workingGold) return false;
        return this.workingGold.qa.some(q => q.chunk_id === cid && q.flagged);
      },
      getQAChunkErrors(cid) {
        if (!this.workingGold) return [];
        const item = this.workingGold.qa.find(q => q.chunk_id === cid && q.flagged);
        if (item && item.errors) return item.errors;
        if (item && item.detail) return [item.detail];
        return ['校验含有部分质量违规项'];
      },
      getEntityName(eid) {
        if (!this.workingGold || !this.workingGold.registry[eid]) return eid;
        return this.workingGold.registry[eid].name;
      },
      getEntityKindBadgeClass(eid) {
        if (!this.workingGold || !this.workingGold.registry[eid]) return 'bg-bg text-dimText';
        return this.getKindBadgeClass(this.workingGold.registry[eid].kind);
      },
      getTimelineEntities() {
        if (!this.workingGold) return [];
        return Object.values(this.workingGold.registry).filter(e => !e.isDropped);
      },
      getTimelineSpanClass(kind) {
        const k = kind ? kind.toLowerCase() : '';
        if (k === 'character') return 'bg-gradient-to-r from-blue-600/90 to-cyan-500/90';
        if (k === 'location') return 'bg-gradient-to-r from-emerald-600/90 to-teal-500/90';
        if (k === 'prop') return 'bg-gradient-to-r from-violet-600/90 to-fuchsia-500/90';
        return 'bg-slate-500';
      },
      showLightbox(src) { this.lightboxSrc = src; }
    }
  };
})(window);
