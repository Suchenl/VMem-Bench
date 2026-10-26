/* VMem-Bench: plain browser JavaScript; all results come from the saved paper snapshots. */
(function () {
  'use strict';
  const data = window.VMEM_DATA;
  const cases = {
    stale: {id:'257ffe6d73',title:'The story changes.<br>The memory doesn’t.',description:'Axi’s scarf has changed to a short cape. Retrieved history and the generated result keep the old scarf.',tag:'Outdated appearance',entity:'Axi',captions:['The history we want to remember.','Relevant to the past.','The old look is carried forward.','Wrong for the present.']},
    forbidden: {id:'5cb7b5d1bf',title:'The character is gone.<br>The reference brings him back.',description:'Uncle Chaiyao has died. The current prompt mentions only his jacket, but the retrieved frame and generated output bring him back.',tag:'Forbidden entity',entity:'Uncle Chaiyao',captions:['A previously present character.','The frame includes the character.','He appears in the keyframe.','He should no longer be here.']},
    lookalike: {id:'995afac068',title:'A familiar shape.<br>The wrong identity.',description:'The story needs the orange dinghy seen earlier. Memory retrieves a white look-alike, which replaces it in the generated result.',tag:'Look-alike confusion',entity:'the dinghy',captions:['The required orange dinghy.','A white look-alike is retrieved.','The wrong identity is composed.','The requested dinghy is missing.']}
  };
  const panels = [['earlier','earlier'],['reference','ref_0'],['keyframe','keyframe'],['output','clip']];
  document.querySelectorAll('[data-case]').forEach(button => button.addEventListener('click', () => {
    const row = cases[button.dataset.case];
    document.querySelectorAll('[data-case]').forEach(b => {b.classList.toggle('active',b===button);b.setAttribute('aria-pressed',String(b===button));});
    document.getElementById('evidence-title').innerHTML = row.title;
    document.getElementById('case-description').textContent = row.description;
    document.getElementById('case-tag').textContent = row.tag;
    panels.forEach(([target, source],i) => {
      const img = document.getElementById('frame-'+target);
      img.src = 'assets/'+row.id+'_'+source+'.jpg';
      img.alt = row.entity+': '+row.captions[i];
      img.parentElement.querySelector('p').textContent = row.captions[i];
    });
  }));
  const metrics = [
    {key:'recall',name:'Entity recall',detail:'Required characters and props that appear.'},
    {key:'precision',name:'Entity precision',detail:'Present entities that belong in the current story.'},
    {key:'avoid_ok',name:'Forbidden-entity avoidance',detail:'Avoid-OK: forbidden entities remain absent.'}
  ];
  const percent = value => (value*100).toFixed(1)+'%';
  const pp = value => (value<0?'−':'+')+Math.abs(value*100).toFixed(1);
  function renderResults() {
    if (!data) return;
    const selected = data.conditions.find(c => c.id===document.getElementById('policy').value);
    const baseline = data.conditions[0];
    document.getElementById('metrics').innerHTML = metrics.map(metric => {
      const value=selected.values[metric.key], base=baseline.values[metric.key], delta=selected.delta[metric.key];
      return '<article class="metric-card"><h3 class="metric-name">'+metric.name+'</h3><p class="metric-detail">'+metric.detail+'</p><div class="metric-values"><strong class="metric-value">'+percent(value)+'</strong><span class="metric-delta">'+pp(delta.diff)+' pp</span></div><div class="metric-bar-row"><div class="metric-track"><i style="width:'+base*100+'%"></i></div><span>'+percent(base)+'</span></div><div class="metric-bar-row memory"><div class="metric-track"><i style="width:'+value*100+'%"></i></div><span>'+percent(value)+'</span></div><p class="metric-ci">Δ versus no memory: '+pp(delta.diff)+' percentage points.<br>95% CI: '+pp(delta.lo)+' to '+pp(delta.hi)+' pp.</p></article>';
    }).join('');
    document.getElementById('result-rows').innerHTML = data.conditions.map(c => '<tr'+(c.id===selected.id?' class="selected"':'')+'><th scope="row">'+c.name+'</th>'+metrics.map(m=>'<td>'+c.values[m.key].toFixed(3)+'</td>').join('')+'</tr>').join('');
  }
  document.getElementById('policy').addEventListener('change',renderResults);
  renderResults();
  if (data) {
    document.getElementById('prompt-original').textContent=percent(data.prompt.with_appearance);
    document.getElementById('prompt-stripped').textContent=percent(data.prompt.without_appearance);
    document.getElementById('prompt-original-bar').style.width=data.prompt.with_appearance*100+'%';
    document.getElementById('prompt-stripped-bar').style.width=data.prompt.without_appearance*100+'%';
  }
  // Progressive clipboard fallback for local file previews and older browsers.
  document.getElementById('copy-citation').addEventListener('click',async () => {
    const text=document.getElementById('bibtex').textContent;
    const status=document.getElementById('copy-status');
    try {
      if(navigator.clipboard && window.isSecureContext) await navigator.clipboard.writeText(text);
      else {
        const range=document.createRange();range.selectNodeContents(document.getElementById('bibtex'));
        const selection=window.getSelection();selection.removeAllRanges();selection.addRange(range);
        if(!document.execCommand('copy')) throw new Error('Select and copy manually');
        selection.removeAllRanges();
      }
      status.textContent='Copied';
    } catch (_) { status.textContent='Select the citation to copy'; }
  });
})();
