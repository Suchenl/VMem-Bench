(function () {
  'use strict';
  const api = window.MemStrataApi;
  const params = new URLSearchParams(location.search);
  const dataset = params.get('dataset') || '';
  const movieId = params.get('movie_id') || '';
  const state = {
    cards: [],
    // representation_id -> true means Keep (green). Missing/false means Reject.
    kept: {},
    alts: {},
    orderByEntity: {},
    expandAlts: new Set(),
    addSeq: 0,
    // Library crops removed from the board → reject on save.
    deleted: {}
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
    return String(value || '').replaceAll('&', '&amp;').replaceAll('<', '&lt;').replaceAll('>', '&gt;').replaceAll('"', '&quot;');
  }
  function mediaUrl(path) {
    return '/api/review/media?dataset=' + encodeURIComponent(dataset) +
      '&movie_id=' + encodeURIComponent(movieId) +
      '&path=' + encodeURIComponent(path);
  }

  function openLightbox(src) {
    if (!src) return;
    el('lightboxImg').src = src;
    el('lightbox').hidden = false;
  }
  function closeLightbox() {
    el('lightbox').hidden = true;
    el('lightboxImg').removeAttribute('src');
  }

  function cardImageSrc(card) {
    return card.image_url || (card.crop_path ? mediaUrl(card.crop_path) : '');
  }

  function isKept(id) {
    return state.kept[id] !== false;
  }

  function statusBadge(card, kept) {
    if (!kept) return '<span class="badge danger">不选</span>';
    if (card._reassigned) return '<span class="badge warn">已改归</span>';
    if (card._added) return '<span class="badge warn">新卡</span>';
    if (card._replacement && card._replacement.crop_path) {
      return '<span class="badge warn">已替换</span>';
    }
    return '<span class="badge ok">保留</span>';
  }

  function entityMetaById(entityId) {
    const hit = state.cards.find((c) => c.entity_id === entityId);
    if (!hit) {
      return { entity_id: entityId, name: entityId, kind: '', description: '' };
    }
    return {
      entity_id: entityId,
      name: hit.name || entityId,
      kind: hit.kind || '',
      description: hit.description || ''
    };
  }

  function removeFromAllOrders(repId) {
    for (const entityId of Object.keys(state.orderByEntity)) {
      state.orderByEntity[entityId] = (state.orderByEntity[entityId] || []).filter((id) => id !== repId);
    }
  }

  function insertIntoOrder(entityId, repId, beforeId) {
    const order = (state.orderByEntity[entityId] || []).filter((id) => id !== repId);
    if (beforeId) {
      const idx = order.indexOf(beforeId);
      if (idx >= 0) order.splice(idx, 0, repId);
      else order.push(repId);
    } else {
      order.push(repId);
    }
    state.orderByEntity[entityId] = order;
  }

  /**
   * Move a crop card to another entity (attribution fix), or reorder within the same entity.
   * Keeps representation_id; entity_id/name/kind follow the target group.
   */
  function reassignOrReorderCard(repId, targetEntityId, beforeId, targetMeta) {
    const card = state.cards.find((c) => c.representation_id === repId);
    if (!card || !targetEntityId) return false;
    const fromEntityId = card.entity_id;
    if (fromEntityId === targetEntityId) {
      if (beforeId && beforeId !== repId) {
        reorderWithinEntity(targetEntityId, repId, beforeId);
        return true;
      }
      return false;
    }
    const meta = targetMeta || entityMetaById(targetEntityId);
    if (
      state.cards.some(
        (c) =>
          c.representation_id !== repId &&
          c.entity_id === targetEntityId &&
          sameCropPath(c.crop_path, card.crop_path)
      )
    ) {
      toast('目标实体已有相同 crop，无法改归', 'fail');
      return false;
    }
    if (card.kind && meta.kind && card.kind !== meta.kind) {
      const ok = window.confirm(
        '跨类型改归？\n\n' +
          (card.name || fromEntityId) + ' (' + card.kind + ')\n→ ' +
          (meta.name || targetEntityId) + ' (' + meta.kind + ')'
      );
      if (!ok) return false;
    }
    if (!card._originalEntityId) {
      card._originalEntityId = fromEntityId;
    }
    card.entity_id = targetEntityId;
    card.name = meta.name || targetEntityId;
    card.kind = meta.kind || card.kind || '';
    card.description = meta.description || '';
    if (card._added && card._addProposal) {
      card._addProposal.entity_id = targetEntityId;
      card._addProposal.name = card.name;
      card._addProposal.kind = card.kind;
      card._addProposal.description = card.description;
    } else {
      card._reassigned = true;
    }
    delete state.alts[repId];
    state.expandAlts.delete(repId);
    removeFromAllOrders(repId);
    insertIntoOrder(targetEntityId, repId, beforeId);
    state.kept[repId] = true;
    toast('已改归到 ' + (card.name || targetEntityId), 'ok');
    return true;
  }

  function nextHumanAddId(entityId) {
    state.addSeq += 1;
    return String(entityId || 'entity') + '@human_add_' + String(state.addSeq).padStart(4, '0');
  }

  function normalizeCropKey(path) {
    return String(path || '').replace(/\\/g, '/').replace(/\/+/g, '/');
  }

  function sameCropPath(a, b) {
    const left = normalizeCropKey(a);
    const right = normalizeCropKey(b);
    if (!left || !right) return false;
    if (left === right) return true;
    // Absolute vs MemStrata-relative forms of the same file.
    return left.endsWith('/' + right) || right.endsWith('/' + left);
  }

  function chunkIdFromCropPath(path, fallback) {
    const leaf = String(path || '').split('/').pop() || '';
    const m = leaf.match(/^c(\d{5})_/);
    if (m) return parseInt(m[1], 10);
    return fallback;
  }

  function promoteAlt(sourceCard, alt) {
    const cropPath = alt && alt.crop_path;
    if (!cropPath) return;
    const entityId = sourceCard.entity_id;
    let existingId = alt.existing_representation_id;
    // Fallback: match an on-board / known card by crop leaf when API id is missing.
    if (!existingId) {
      const hit = state.cards.find(
        (c) => c.entity_id === entityId && sameCropPath(c.crop_path, cropPath)
      );
      if (hit) existingId = hit.representation_id;
    }
    const chunkId = chunkIdFromCropPath(cropPath, sourceCard.chunk_id);
    if (existingId) {
      let card = state.cards.find((c) => c.representation_id === existingId);
      if (!card) {
        card = {
          representation_id: existingId,
          entity_id: entityId,
          name: sourceCard.name,
          kind: sourceCard.kind,
          description: sourceCard.description || '',
          chunk_id: chunkId,
          segment_id: sourceCard.segment_id,
          task_kind: 'acquire',
          crop_path: cropPath,
          image_url: mediaUrl(cropPath)
        };
        state.cards.push(card);
        const order = state.orderByEntity[entityId] || [];
        order.push(existingId);
        state.orderByEntity[entityId] = order;
      }
      state.kept[existingId] = true;
      toast('已拉出已有 crop 卡 ' + existingId, 'ok');
      render();
      return;
    }
    if (state.cards.some((c) => c.entity_id === entityId && sameCropPath(c.crop_path, cropPath))) {
      toast('该候选已在本实体卡片中', 'fail');
      return;
    }
    const id = nextHumanAddId(entityId);
    const proposal = {
      representation_id: id,
      entity_id: entityId,
      kind: sourceCard.kind || 'character',
      name: sourceCard.name || entityId,
      description: sourceCard.description || '',
      chunk_id: chunkId,
      segment_id: sourceCard.segment_id,
      crop_path: cropPath,
      task_kind: 'acquire',
      action: 'acquire',
      reason: 'human_promoted_alt',
      accepted: true
    };
    state.cards.push({
      representation_id: id,
      entity_id: entityId,
      name: proposal.name,
      kind: proposal.kind,
      description: proposal.description,
      chunk_id: chunkId,
      segment_id: proposal.segment_id,
      task_kind: 'acquire',
      crop_path: cropPath,
      image_url: mediaUrl(cropPath),
      _added: true,
      _addProposal: proposal
    });
    state.kept[id] = true;
    const order = state.orderByEntity[entityId] || [];
    order.push(id);
    state.orderByEntity[entityId] = order;
    toast('已加为新 crop 卡 ' + id, 'ok');
    render();
  }

  async function load() {
    el('title').textContent = 'S6 · ' + dataset + '/' + movieId;
    const data = await api.reviewS6(dataset, movieId);
    state.cards = data.cards || [];
    const draft = ((data.draft || {}).decisions) || {};
    state.kept = {};
    state.alts = {};
    state.orderByEntity = {};
    state.expandAlts = new Set();
    state.addSeq = 0;
    state.deleted = {};
    for (const card of state.cards) {
      const id = card.representation_id;
      const prior = draft[id];
      if (prior && prior.action === 'reject') state.kept[id] = false;
      else if (prior && (prior.action === 'keep' || prior.action === 'replace' || prior.action === 'add' || prior.action === 'reassign')) {
        state.kept[id] = true;
      } else state.kept[id] = true;
      if (prior && prior.action === 'replace' && prior.replacement && prior.replacement.crop_path) {
        card.crop_path = prior.replacement.crop_path;
        card.image_url = mediaUrl(prior.replacement.crop_path);
        card._replacement = { crop_path: prior.replacement.crop_path };
      }
      if (prior && prior.action === 'reassign' && prior.entity_id) {
        card._originalEntityId = prior.from_entity_id || card.entity_id;
        card.entity_id = prior.entity_id;
        if (prior.name != null) card.name = prior.name;
        if (prior.kind != null) card.kind = prior.kind;
        if ('description' in prior) card.description = prior.description || '';
        card._reassigned = true;
        if (prior.replacement && prior.replacement.crop_path) {
          card.crop_path = prior.replacement.crop_path;
          card.image_url = mediaUrl(prior.replacement.crop_path);
          card._replacement = { crop_path: prior.replacement.crop_path };
        }
      }
    }
    for (const [id, prior] of Object.entries(draft)) {
      if (!prior || prior.action !== 'add' || !prior.proposal) continue;
      if (state.cards.some((c) => c.representation_id === id)) continue;
      const prop = prior.proposal;
      // Skip draft adds that duplicate an on-board crop (abs/rel path drift).
      if (state.cards.some(
        (c) => c.entity_id === prop.entity_id && sameCropPath(c.crop_path, prop.crop_path)
      )) {
        continue;
      }
      state.cards.push({
        representation_id: id,
        entity_id: prop.entity_id,
        name: prop.name,
        kind: prop.kind,
        description: prop.description || '',
        chunk_id: chunkIdFromCropPath(prop.crop_path, prop.chunk_id),
        task_kind: 'acquire',
        crop_path: prop.crop_path,
        image_url: mediaUrl(prop.crop_path),
        _added: true,
        _addProposal: prop
      });
      state.kept[id] = true;
      const m = String(id).match(/human_add_(\d+)/);
      if (m) state.addSeq = Math.max(state.addSeq, parseInt(m[1], 10));
    }
    const entities = new Set(state.cards.map((c) => c.entity_id).filter(Boolean));
    const nKeep = state.cards.filter((c) => isKept(c.representation_id)).length;
    el('meta').textContent = (data.available ? state.cards.length + ' 张待审 crop · ' + entities.size + ' 个实体' : '无 S6 队列') +
      ' · 绿框=保留 · 候选「选用」替换 · 拖出小图=新卡 · 卡片拖到其他实体=改归 · 删除可去掉卡片' +
      ' · 当前保留 ' + nKeep + '/' + state.cards.length +
      (data.audit && data.audit.human_reviewed ? ' · 已人工审核' : ' · 待审核');
    render();
  }

  function removeCard(id) {
    const card = state.cards.find((c) => c.representation_id === id);
    if (!card) return;
    if (!card._added) {
      state.deleted[id] = true;
    }
    state.cards = state.cards.filter((c) => c.representation_id !== id);
    delete state.kept[id];
    delete state.alts[id];
    state.expandAlts.delete(id);
    for (const entityId of Object.keys(state.orderByEntity)) {
      state.orderByEntity[entityId] = (state.orderByEntity[entityId] || []).filter((rid) => rid !== id);
    }
    toast(card._added ? '已删除新卡' : '已删除卡片（保存时记为不选）', 'ok');
    render();
  }

  function groupCardsByEntity(cards) {
    const groups = new Map();
    for (const card of cards) {
      const key = String(card.entity_id || card.representation_id || 'unknown');
      if (!groups.has(key)) {
        groups.set(key, {
          entity_id: key,
          name: card.name || key,
          kind: card.kind || '',
          description: card.description || '',
          cards: []
        });
      }
      groups.get(key).cards.push(card);
    }
    for (const group of groups.values()) {
      const order = state.orderByEntity[group.entity_id];
      if (order && order.length) {
        const byId = new Map(group.cards.map((c) => [c.representation_id, c]));
        const ordered = [];
        for (const id of order) {
          if (byId.has(id)) {
            ordered.push(byId.get(id));
            byId.delete(id);
          }
        }
        for (const card of byId.values()) ordered.push(card);
        group.cards = ordered;
      }
      state.orderByEntity[group.entity_id] = group.cards.map((c) => c.representation_id);
    }
    return Array.from(groups.values()).sort((a, b) => {
      const kindCmp = String(a.kind).localeCompare(String(b.kind));
      if (kindCmp) return kindCmp;
      return String(a.entity_id).localeCompare(String(b.entity_id));
    });
  }

  function render() {
    const root = el('cards');
    root.innerHTML = '';
    if (!state.cards.length) {
      root.innerHTML = '<div class="empty">没有待审 crop（机器已拒/无图/同图冗余已自动隐藏）。可「重跑 S5」或回控制台检查。</div>';
      return;
    }
    for (const group of groupCardsByEntity(state.cards)) {
      root.appendChild(renderEntityGroup(group));
    }
    const nKeep = state.cards.filter((c) => isKept(c.representation_id)).length;
    const entities = new Set(state.cards.map((c) => c.entity_id).filter(Boolean));
    el('meta').textContent =
      state.cards.length + ' 张待审 crop · ' + entities.size + ' 个实体' +
      ' · 绿框=保留 · 候选「选用」替换 · 拖出小图=新卡 · 卡片拖到其他实体=改归 · 删除可去掉卡片' +
      ' · 当前保留 ' + nKeep + '/' + state.cards.length;
  }

  function renderEntityGroup(group) {
    const section = document.createElement('section');
    section.className = 'entitySection s6EntitySection';
    section.dataset.entityId = group.entity_id;
    const nKeep = group.cards.filter((c) => isKept(c.representation_id)).length;
    const head = document.createElement('div');
    head.className = 'entitySectionHead';
    head.innerHTML =
      '<div><h2>' + escapeHtml(group.name) + ' · ' + escapeHtml(group.entity_id) + '</h2>' +
      '<div class="muted">' + escapeHtml(group.kind || '') +
      (group.description ? ' · ' + escapeHtml(group.description) : '') + '</div></div>' +
      '<div class="entitySectionStats">' +
      '<span class="badge">' + group.cards.length + ' crops</span>' +
      '<span class="badge ok">保留 ' + nKeep + '</span>' +
      '<span class="badge danger">不选 ' + (group.cards.length - nKeep) + '</span>' +
      '</div>';
    section.appendChild(head);

    const tools = document.createElement('div');
    tools.className = 'entityGroupTools';
    tools.innerHTML =
      '<button type="button" class="secondary" data-group-keep>本组全选</button>' +
      '<button type="button" class="secondary" data-group-clear>本组全不选</button>' +
      '<span class="muted s6DropHint">候选拖出=新卡 · 卡片拖到本组=改归</span>';
    tools.querySelector('[data-group-keep]').addEventListener('click', () => {
      for (const card of group.cards) state.kept[card.representation_id] = true;
      render();
    });
    tools.querySelector('[data-group-clear]').addEventListener('click', () => {
      for (const card of group.cards) state.kept[card.representation_id] = false;
      render();
    });
    section.appendChild(tools);

    function isInsideAltPanel(target) {
      return !!(target && target.closest && target.closest('.altGrid, .s6AltStrip, .s6AltItem'));
    }

    function hasType(types, mime) {
      return [...types].includes(mime);
    }

    section.addEventListener('dragover', (ev) => {
      const types = ev.dataTransfer.types;
      const isAlt = hasType(types, 'application/x-s6-alt');
      // Some browsers omit custom MIME from types during dragover; fall back to text/plain.
      const isCard = hasType(types, 'application/x-s6-card') ||
        (!isAlt && hasType(types, 'text/plain'));
      if (!isAlt && !isCard) return;
      if (isInsideAltPanel(ev.target)) {
        section.classList.remove('altDropTarget', 'cardDropTarget');
        ev.dataTransfer.dropEffect = 'none';
        return;
      }
      ev.preventDefault();
      if (isAlt) {
        ev.dataTransfer.dropEffect = 'copy';
        section.classList.add('altDropTarget');
        section.classList.remove('cardDropTarget');
      } else {
        ev.dataTransfer.dropEffect = 'move';
        section.classList.add('cardDropTarget');
        section.classList.remove('altDropTarget');
      }
    });
    section.addEventListener('dragleave', (ev) => {
      if (ev.relatedTarget && section.contains(ev.relatedTarget)) return;
      section.classList.remove('altDropTarget', 'cardDropTarget');
    });
    section.addEventListener('drop', (ev) => {
      section.classList.remove('altDropTarget', 'cardDropTarget');
      // Must leave the candidate panel; dropping inside it does nothing.
      if (isInsideAltPanel(ev.target)) {
        ev.preventDefault();
        ev.stopPropagation();
        return;
      }
      const altRaw = ev.dataTransfer.getData('application/x-s6-alt');
      if (altRaw) {
        ev.preventDefault();
        let payload;
        try {
          payload = JSON.parse(altRaw);
        } catch (_) {
          return;
        }
        if (payload.entityId !== group.entity_id) {
          toast('候选只能拖到同一实体组（卡片才可跨实体改归）', 'fail');
          return;
        }
        const source = state.cards.find((c) => c.representation_id === payload.sourceId);
        if (!source) return;
        promoteAlt(source, payload.alt || {
          crop_path: payload.crop_path,
          existing_representation_id: payload.existing_representation_id
        });
        return;
      }
      const cardRaw = ev.dataTransfer.getData('application/x-s6-card');
      if (!cardRaw) return;
      ev.preventDefault();
      let payload;
      try {
        payload = JSON.parse(cardRaw);
      } catch (_) {
        return;
      }
      if (!payload.id) return;
      if (reassignOrReorderCard(payload.id, group.entity_id, null, {
        entity_id: group.entity_id,
        name: group.name,
        kind: group.kind,
        description: group.description || ''
      })) render();
    });

    const grid = document.createElement('div');
    grid.className = 's6CropGrid';
    for (const card of group.cards) grid.appendChild(renderTile(card, group.entity_id));
    section.appendChild(grid);
    return section;
  }

  function renderTile(card, entityId) {
    const id = card.representation_id;
    const kept = isKept(id);
    const imgSrc = cardImageSrc(card);
    const altsOpen = state.expandAlts.has(id);
    const tile = document.createElement('article');
    tile.className = 's6CropTile' + (kept ? ' selected' : '') + (altsOpen ? ' altsOpen' : '');
    tile.draggable = !altsOpen;
    tile.dataset.rep = id;
    tile.dataset.entityId = entityId;
    tile.title = kept ? '已选中（保留）· 再点取消' : '未选中（不保留）· 点击选中';

    const mediaHtml = imgSrc
      ? '<img class="s6CropImg" loading="lazy" src="' + escapeHtml(imgSrc) + '" alt="">'
      : '<div class="emptySmall">无图</div>';

    const metaInner = [
      '<div class="cropStatusRow">',
      '<strong title="' + escapeHtml(id) + '">' + escapeHtml(id) + '</strong>',
      statusBadge(card, kept),
      '</div>',
      '<div class="muted">c' + escapeHtml(String(card.chunk_id ?? '')) +
      ' · ' + escapeHtml(card.task_kind || 'acquire') +
      (card._reassigned ? ' · 改归自 ' + escapeHtml(card._originalEntityId || '') : '') +
      (card._added ? ' · 人工拉出' : '') + '</div>',
      '<div class="s6TileActions">',
      '<button type="button" class="secondary s6AltBtn" data-toggle-alts="' + escapeHtml(id) + '">' +
      (altsOpen ? '收起候选' : '替换候选') + '</button>',
      '<button type="button" class="secondary s6DelBtn" data-delete="' + escapeHtml(id) + '">删除</button>',
      '</div>'
    ].join('');

    if (altsOpen) {
      // B: full-width strip — compact head + short wide candidate row
      tile.innerHTML =
        '<div class="s6AltStripHead">' +
        '<div class="s6CropMedia">' + mediaHtml + '</div>' +
        '<div class="s6CropMeta">' + metaInner + '</div>' +
        '</div>' +
        '<div class="altGrid" data-alts="' + escapeHtml(id) + '"></div>';
    } else {
      tile.innerHTML =
        '<div class="s6CropMedia">' + mediaHtml + '</div>' +
        '<div class="s6CropMeta">' + metaInner +
        '<div class="altGrid" data-alts="' + escapeHtml(id) + '" hidden></div>' +
        '</div>';
    }

    const img = tile.querySelector('.s6CropImg');
    if (img) {
      img.addEventListener('click', (ev) => {
        ev.preventDefault();
        ev.stopPropagation();
        openLightbox(img.currentSrc || img.src);
      });
    }

    tile.querySelector('[data-toggle-alts]').addEventListener('click', (ev) => {
      ev.stopPropagation();
      toggleAlts(id);
    });
    tile.querySelector('[data-delete]').addEventListener('click', (ev) => {
      ev.stopPropagation();
      removeCard(id);
    });

    tile.addEventListener('click', (ev) => {
      if (ev.target.closest('button,.altGrid,img,.s6AltStripHead')) return;
      if (altsOpen) return;
      state.kept[id] = !isKept(id);
      render();
    });

    tile.addEventListener('dragstart', (ev) => {
      if (altsOpen) {
        ev.preventDefault();
        return;
      }
      const payload = { id: id, entityId: entityId, type: 'card' };
      ev.dataTransfer.setData('application/x-s6-card', JSON.stringify(payload));
      ev.dataTransfer.setData('text/plain', JSON.stringify(payload));
      ev.dataTransfer.effectAllowed = 'move';
      tile.classList.add('dragging');
    });
    tile.addEventListener('dragend', () => tile.classList.remove('dragging'));
    tile.addEventListener('dragover', (ev) => {
      const types = ev.dataTransfer.types;
      if (![...types].includes('application/x-s6-card') && ![...types].includes('text/plain')) return;
      if ([...types].includes('application/x-s6-alt')) return;
      ev.preventDefault();
      ev.stopPropagation();
      ev.dataTransfer.dropEffect = 'move';
      tile.classList.add('dragOver');
    });
    tile.addEventListener('dragleave', () => tile.classList.remove('dragOver'));
    tile.addEventListener('drop', (ev) => {
      tile.classList.remove('dragOver');
      if (ev.dataTransfer.getData('application/x-s6-alt')) return;
      const raw = ev.dataTransfer.getData('application/x-s6-card') ||
        ev.dataTransfer.getData('text/plain') || '{}';
      let payload;
      try {
        payload = JSON.parse(raw);
      } catch (_) {
        return;
      }
      if (!payload.id || payload.type === 'alt') return;
      ev.preventDefault();
      ev.stopPropagation();
      if (reassignOrReorderCard(payload.id, entityId, id)) render();
    });

    if (altsOpen) paintAlts(tile.querySelector('[data-alts]'), id, card);
    return tile;
  }

  function reorderWithinEntity(entityId, fromId, toId) {
    const order = (state.orderByEntity[entityId] || []).slice();
    const fromIdx = order.indexOf(fromId);
    const toIdx = order.indexOf(toId);
    if (fromIdx < 0 || toIdx < 0 || fromIdx === toIdx) return;
    order.splice(fromIdx, 1);
    order.splice(toIdx, 0, fromId);
    state.orderByEntity[entityId] = order;
  }

  function paintAlts(container, id, card) {
    if (!container) return;
    container.hidden = false;
    container.innerHTML = '';
    // Swallow drops inside the candidate panel so a slight drag doesn't promote.
    container.addEventListener('dragover', (ev) => {
      ev.preventDefault();
      ev.stopPropagation();
      ev.dataTransfer.dropEffect = 'none';
    });
    container.addEventListener('drop', (ev) => {
      ev.preventDefault();
      ev.stopPropagation();
    });
    const alts = state.alts[id] || [];
    if (!alts.length) {
      container.innerHTML = '<div class="muted s6AltHint">无可用候选</div>';
      return;
    }
    const hint = document.createElement('div');
    hint.className = 'muted s6AltHint';
    hint.textContent = '「选用」=替换本卡 · 拖出候选区再放下 = 新卡';
    container.appendChild(hint);

    const strip = document.createElement('div');
    strip.className = 's6AltStrip';
    const chosen = (card && card._replacement && card._replacement.crop_path) || '';
    for (const alt of alts) {
      const wrap = document.createElement('div');
      wrap.className = 's6AltItem';
      wrap.draggable = true;
      const altSrc = mediaUrl(alt.crop_path);
      const isActive = chosen && chosen === alt.crop_path;
      const isUnmasked = alt.variant === 'unmasked';
      wrap.innerHTML =
        '<img loading="lazy" src="' + escapeHtml(altSrc) + '" alt="" draggable="false">' +
        '<button type="button" class="s6AltAction' + (isActive ? ' active' : '') + '" data-use>' +
        (isActive ? '已用' : (isUnmasked ? '选用未 mask' : '选用')) + '</button>';
      wrap.title = (isUnmasked ? '同一候选的未 mask 版本；' : '') +
        '拖出候选区再放到实体区才加新卡；点选用替换本卡；双击放大';

      wrap.addEventListener('dragstart', (ev) => {
        ev.stopPropagation();
        const payload = {
          sourceId: id,
          entityId: card.entity_id,
          crop_path: alt.crop_path,
          existing_representation_id: alt.existing_representation_id || '',
          alt: alt
        };
        ev.dataTransfer.setData('application/x-s6-alt', JSON.stringify(payload));
        ev.dataTransfer.setData('text/plain', JSON.stringify(payload));
        ev.dataTransfer.effectAllowed = 'copy';
        wrap.classList.add('dragging');
      });
      wrap.addEventListener('dragend', () => wrap.classList.remove('dragging'));

      wrap.querySelector('[data-use]').addEventListener('click', (ev) => {
        ev.stopPropagation();
        const target = state.cards.find((c) => c.representation_id === id);
        if (target) {
          target.crop_path = alt.crop_path;
          target.image_url = altSrc;
          target._replacement = { crop_path: alt.crop_path };
          delete target._added;
          delete target._addProposal;
        }
        state.kept[id] = true;
        toast('已选用替换图', 'ok');
        render();
      });
      wrap.querySelector('img').addEventListener('dblclick', (ev) => {
        ev.preventDefault();
        ev.stopPropagation();
        openLightbox(altSrc);
      });
      strip.appendChild(wrap);
    }
    container.appendChild(strip);
  }

  async function toggleAlts(id) {
    if (state.expandAlts.has(id)) {
      state.expandAlts.delete(id);
      render();
      return;
    }
    // Only one strip open at a time so the board stays readable.
    state.expandAlts = new Set([id]);
    if (!state.alts[id]) {
      try {
        const data = await api.reviewS6Alts(dataset, movieId, id);
        state.alts[id] = data.alternates || [];
        if (!state.alts[id].length) toast('没有可用候选', 'fail');
      } catch (err) {
        toast('加载候选失败: ' + err.message, 'fail');
        state.expandAlts.delete(id);
        render();
        return;
      }
    }
    render();
  }

  async function save() {
    try {
      const payload = {};
      for (const id of Object.keys(state.deleted)) {
        payload[id] = { action: 'reject', reason: 'human_deleted' };
      }
      for (const card of state.cards) {
        const id = card.representation_id;
        if (!isKept(id)) {
          if (card._added) continue;
          payload[id] = { action: 'reject', reason: 'human_deselected' };
          continue;
        }
        if (card._added && card._addProposal) {
          payload[id] = {
            action: 'add',
            reason: 'human_promoted_alt',
            proposal: card._addProposal
          };
        } else if (card._reassigned) {
          const decision = {
            action: 'reassign',
            entity_id: card.entity_id,
            name: card.name,
            kind: card.kind,
            description: card.description || '',
            from_entity_id: card._originalEntityId || '',
            reason: 'human_reassigned'
          };
          if (card._replacement && card._replacement.crop_path) {
            decision.replacement = card._replacement;
          }
          payload[id] = decision;
        } else if (card._replacement && card._replacement.crop_path) {
          payload[id] = {
            action: 'replace',
            reason: 'human_replaced',
            replacement: card._replacement
          };
        } else {
          payload[id] = { action: 'keep', reason: '' };
        }
      }
      const result = await api.applyS6({ dataset: dataset, movie_id: movieId, decisions: payload });
      state.deleted = {};
      toast('S6 已应用：accepted=' + result.accepted_count, 'ok');
    } catch (err) {
      toast('保存失败: ' + err.message, 'fail');
    }
  }

  function keepAll() {
    for (const card of state.cards) state.kept[card.representation_id] = true;
    render();
  }
  function clearAll() {
    for (const card of state.cards) state.kept[card.representation_id] = false;
    render();
  }

  async function cont() {
    const button = el('continueBtn');
    button.disabled = true;
    try {
      const result = await api.continueReview({
        dataset: dataset,
        movie_id: movieId,
        continue_from: 'after_s6'
      });
      toast('S7 冻结完成：' + (result.gold || movieId), 'ok');
      button.textContent = 'S7 已冻结';
    } catch (err) {
      button.disabled = false;
      toast('继续失败: ' + err.message, 'fail');
    }
  }

  async function rerunS5() {
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
      toast('已提交重跑 S5 ' + (job.job_id || ''), 'ok');
    } catch (err) {
      toast('重跑 S5 失败: ' + err.message, 'fail');
    }
  }

  el('saveBtn').addEventListener('click', save);
  el('keepAllBtn').addEventListener('click', keepAll);
  el('clearAllBtn').addEventListener('click', clearAll);
  el('continueBtn').addEventListener('click', cont);
  el('rerunS5Btn').addEventListener('click', () => {
    rerunS5().catch((err) => toast(err.message, 'fail'));
  });
  el('lightbox').addEventListener('click', (ev) => {
    if (ev.target === el('lightbox') || ev.target === el('lightboxClose')) closeLightbox();
  });
  el('lightboxClose').addEventListener('click', closeLightbox);
  document.addEventListener('keydown', (ev) => {
    if (ev.key === 'Escape' && !el('lightbox').hidden) closeLightbox();
  });
  load().catch((err) => toast(err.message, 'fail'));
})();
