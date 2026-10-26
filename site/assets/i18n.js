/* Human-written translations. Scientific values and citation text stay unchanged. */
(function () {
  'use strict';
  const zh = {
    'Skip to content':'跳至正文',
    'Main navigation':'主导航',
    'Examples':'案例', 'Results':'结果', 'Benchmark':'基准', 'Resources':'资源',
    'Paper · forthcoming':'论文 · 即将发布',
    'VISUAL MEMORY · LONG-VIDEO GENERATION':'视觉记忆 · 长视频生成',
    'More memory.':'更多记忆。', 'Better videos?':'更好的视频？',
    'A relevant past can be wrong for the present.':'与过去相关，不代表适用于当下。',
    'We separate what memory retrieves from what a generator actually renders.':'我们区分记忆检索到了什么，以及生成器实际呈现了什么。',
    'More recall can come with unwanted entities and outdated appearances.':'更高的召回率，也可能伴随不该出现的实体和过时的外观。',
    'Code':'代码', 'Dataset':'数据集', 'Cite this work ↓':'引用本研究 ↓',
    '01 / SEE THE PROBLEM':'01 / 从案例看问题', 'Existing generated results':'已有生成结果',
    'Choose a memory failure':'选择记忆失败案例',
    '01 · An outdated look':'01 · 过时的外观', '02 · A forbidden return':'02 · 不该回归的角色', '03 · The wrong identity':'03 · 错误的身份',
    'The story changes.':'故事已经改变。', 'The memory doesn’t.':'记忆却停留在过去。',
    'The story changes.<br>The memory doesn’t.':'故事已经改变。<br>记忆却停留在过去。',
    'Axi’s scarf has changed to a short cape. Retrieved history and the generated result keep the old scarf.':'Axi的围巾已经换成短斗篷，但检索到的历史画面和生成结果仍保留着旧围巾。',
    'Outdated appearance':'过时的外观', 'Axi':'Axi',
    'Previous appearance':'先前外观', 'Retrieved reference':'检索参考帧', 'Composed keyframe':'合成关键帧', 'Generated clip':'生成片段', 'Mismatch':'不符',
    'The history we want to remember.':'我们希望记住的历史。', 'Relevant to the past.':'与过去相关。', 'The old look is carried forward.':'旧外观被延续。', 'Wrong for the present.':'却不适用于当下。',
    'Stills from earlier Track-B rollouts, shown in the paper’s Figure 1. One retrieved reference is displayed. These illustrate failure paths; they are not fixed-history causal interventions or samples from the main quantitative rerun.':'画面取自早期 Track-B 生成轨迹，对应论文图 1；此处仅展示一张检索参考帧。这些案例用于说明失败路径，既非固定历史的因果干预，也非主定量重跑中的样本。',
    'Remembering an entity is only half the problem.':'记住实体，只解决了一半问题。', 'Memory must also respect what has changed.':'记忆还必须尊重已经发生的变化。', 'Explore the measurements':'查看定量结果',
    '02 / THE TRADE-OFF':'02 / 收益与代价', 'Same shared generator':'使用相同的共享生成器',
    'More recall.':'更高的召回。', 'Less correctness.':'更低的正确性。',
    'Retrieval helps the right entities reappear. It also brings in entities that should be absent. A single consistency score misses this trade-off.':'检索帮助需要的实体重新出现，也会带回本应缺席的实体。单一的一致性分数无法反映这种权衡。',
    'Compare a retrieval policy with no memory':'选择检索策略，与无记忆条件比较',
    'No memory':'无记忆', 'Selected retrieval policy':'当前检索策略', 'Absolute rates · higher is better for all three':'绝对比例 · 三项指标均为越高越好',
    '30 stories · 2,824 five-second segments per condition · seed 2026 · FLUX.2 + Wan2.2. Differences and 95% intervals use paired story-clustered bootstrap; they do not cover generation-seed variability.':'30 个故事 · 每个条件 2,824 个五秒片段 · 随机种子 2026 · FLUX.2 + Wan2.2。差值及其 95% 置信区间采用按故事聚类的配对 bootstrap 计算，不涵盖生成随机种子带来的变异。',
    'See all four policies and the no-memory control':'查看全部四种策略与无记忆对照',
    'Absolute entity outcomes under the shared generator':'共享生成器下的实体指标绝对值',
    'Condition':'条件', 'Recall ↑':'召回率 ↑', 'Precision ↑':'精确率 ↑', 'Avoid-OK ↑':'禁止实体回避率 ↑',
    'Entity recall':'实体召回率', 'Required characters and props that appear.':'应出现的角色和道具实际出现的比例。',
    'Entity precision':'实体精确率', 'Present entities that belong in the current story.':'已出现实体中符合当前故事要求的比例。',
    'Forbidden-entity avoidance':'禁止实体回避率', 'Avoid-OK: forbidden entities remain absent.':'Avoid-OK：禁止实体保持缺席的比例。',
    '03 / A HIDDEN SHORTCUT':'03 / 隐藏的捷径', 'Appearance-stripping analysis':'移除外观描述的分析',
    'Did memory remember?':'是记忆记住了？', 'Or did the prompt tell it?':'还是提示词已经告诉了它？',
    'A character can return after a long gap even with no visual memory, because the current prompt describes the character’s appearance.':'即使没有视觉记忆，角色也可能在长间隔后重新出现，因为当前提示词已经描述了角色的外观。',
    'On the same 62 long-gap character-return events, removing the appearance information sharply reduces no-memory presence. Long-gap success alone is insufficient evidence of visual-memory use.':'在相同的 62 个长间隔角色回归事件上，移除外观信息会使无记忆条件下的角色出现率明显降低。仅凭长间隔后的成功回归，不足以证明模型利用了视觉记忆。',
    'Targeted prompt intervention; these are entity-presence rates, not proof of identity fidelity or a memory-capacity scaling law.':'这是针对提示词的干预；这里衡量的是实体出现率，不能据此证明身份保真度或记忆容量的缩放规律。',
    'NO MEMORY · LONG-GAP CHARACTER PRESENCE':'无记忆 · 长间隔角色出现率', 'Original prompt':'原始提示词', 'Appearance stripped':'移除外观描述',
    'Same event cohort.':'同一组事件。', 'Different information in the current prompt.':'当前提示词提供的信息不同。',
    '04 / TWO COMPLEMENTARY TRACKS':'04 / 两条互补的评估轨道', 'Selection → generated outcomes':'参考选择 → 生成结果',
    'Measure the evidence.':'先评估检索证据。', 'Then measure its use.':'再评估证据的使用效果。',
    'Track A evaluates reference selection before generation. Track B evaluates the generated pixels. They answer different questions on different instances.':'Track A 在生成前评估参考选择，Track B 评估生成画面。两条轨道使用不同的实例，回答不同的问题。',
    'What did memory retrieve?':'记忆检索到了什么？', 'Choose references before the current real segment is revealed; observe the segment afterward.':'在当前真实片段揭示前选择参考，之后再观察该片段并评分。',
    'Prompt':'提示词', 'Retrieve':'检索', 'Score references':'参考评分', 'real films':'部真实电影', 'segments':'个片段',
    'No generator confound. The scoring gold is never an input to the method.':'不受生成器影响。评分标准答案绝不会作为方法输入。',
    'What did the generator render?':'生成器实际呈现了什么？',
    'Generate causally from prompts and the system’s own history. Evaluate the entities and states that appear.':'仅依据当前及已有的提示词和系统自身历史逐步生成，评估画面中出现的实体及其状态。',
    'Generate':'生成', 'Observe':'观察', 'evaluated stories':'个评估故事', 'segments / condition':'个片段 / 条件',
    'Four retrieval policies, no memory and a separate gold-ID Last-seen diagnostic control.':'包含四种检索策略、无记忆对照，以及单独的 gold-ID Last-seen 诊断对照。',
    'The 30-story evaluated cohort is part of a larger 50-story authoring pool. The two tracks are not matched instances, so Track-A scores cannot be directly correlated with Track-B outcomes.':'评估的 30 个故事来自共 50 个故事的创作池。两条轨道的实例不匹配，因此不能将 Track-A 得分与 Track-B 结果直接进行相关分析。',
    '05 / WHAT TO TAKE AWAY':'05 / 可以带走的启发', 'For builders and evaluators':'面向系统开发者与评估者',
    'Ask three questions':'先问三个问题', 'before you retrieve.':'再进行检索。',
    'What information':'哪些信息', 'is valid now?':'现在仍然有效？',
    'Historical fidelity is useful only when the remembered state still agrees with the current story.':'只有当记住的状态仍符合当前故事时，忠实于历史才有价值。',
    'What else comes':'这张参考帧', 'with that reference?':'还会带来什么？',
    'A needed entity may share its frame with a forbidden character, a look-alike or an obsolete appearance.':'需要的实体可能与禁止出现的角色、相似但不同的实体或过时外观共处一帧。',
    'What did the prompt':'提示词已经', 'already supply?':'提供了哪些信息？',
    'Separate gains from visual history from successes explained by the current prompt itself.':'区分视觉历史带来的收益，与当前提示词本身即可解释的成功。',
    'What the study establishes, and what remains open':'研究支持哪些结论，还有哪些问题尚未解决',
    'The main comparison measures complete retrieval policies on one shared generator, conditional on one rollout per story. Histories diverge, so it does not isolate the causal effect of replacing a single reference. The paper also reports second-judge and eight-story H3 sensitivity checks. Absolute rates depend on the judge, and the backend check has limited scope. Last-seen has privileged gold-ID access and variable reference counts; it is not a deployable method or a budget-matched upper bound. The 224-unit mechanism audit was performed by one author and does not estimate population failure frequencies.':'主比较在一个共享生成器上评估完整检索策略，结论以每个故事的一次生成轨迹为条件。不同策略的历史会分化，因此不能隔离替换单张参考帧的因果效应。论文还报告了第二评判模型及八个故事上的 H3 后端敏感性检查。绝对比例依赖评判模型，后端检查的覆盖范围也有限。Last-seen 可访问特权 gold-ID，且参考数量可变；它不是可直接部署的方法，也不是匹配预算的性能上限。224 个单元的机制审查由一位作者执行，不能用于估计总体失败频率。',
    '06 / READ · USE · REPRODUCE':'06 / 阅读 · 使用 · 复现',
    'Build memory that knows':'让记忆学会判断', 'when the past is useful.':'何时过去才有用。',
    'The paper link will be added after the arXiv publication.':'论文在 arXiv 发布后将补充链接。',
    'Use the benchmark ↗':'使用评测基准 ↗', 'Scorers, method adapters, shared-generator testbed and reproduction guide.':'评分工具、方法适配器、共享生成器测试平台及复现指南。',
    'Explore the data ↗':'探索数据 ↗', 'Text gold and prompts. Track B needs no source films; Track A requires separate video access.':'文本标准答案与提示词。Track B 无需源视频；Track A 需要另行获取视频访问权限。',
    'THE PAPER':'论文', 'More Memory, Better Videos?':'更多记忆，更好的视频？',
    'A Controlled Study of Visual Memory in Long-Video Generation':'长视频生成中视觉记忆的受控研究',
    'University of Science and Technology of China':'中国科学技术大学', 'Tsinghua University':'清华大学',
    'The Hong Kong University of Science and Technology':'香港科技大学', 'City University of Hong Kong':'香港城市大学',
    '* Co-corresponding authors · ‡ Project leader':'* 共同通讯作者 · ‡ 项目负责人',
    '† This work was conducted during the author’s internship at KlingAI.':'† 本工作在作者于 KlingAI 实习期间完成。',
    'Copy citation':'复制引用', 'Copied':'已复制', 'Select the citation to copy':'请选中引用内容后复制',
    'Enable JavaScript to switch examples and compare policies. The paper and dataset links remain available.':'启用 JavaScript 后可切换案例、比较策略。论文和数据集链接仍可使用。',
    'The character is gone.<br>The reference brings him back.':'角色已经离场。<br>参考帧却将他带回。',
    'Uncle Chaiyao has died. The current prompt mentions only his jacket, but the retrieved frame and generated output bring him back.':'Chaiyao 叔叔已经去世。当前提示词仅提到他的夹克，但检索参考帧和生成结果却让他重新出现。',
    'Forbidden entity':'禁止出现的实体', 'Uncle Chaiyao':'Chaiyao 叔叔',
    'A previously present character.':'曾经出现过的角色。', 'The frame includes the character.':'参考帧中包含该角色。', 'He appears in the keyframe.':'他出现在关键帧中。', 'He should no longer be here.':'他本不该再次出现。',
    'A familiar shape.<br>The wrong identity.':'相似的外形。<br>错误的身份。',
    'The story needs the orange dinghy seen earlier. Memory retrieves a white look-alike, which replaces it in the generated result.':'故事需要先前出现过的橙色小艇，记忆却检索到一艘相似的白色小艇，并在生成结果中用它替代了目标。',
    'Look-alike confusion':'相似实体混淆', 'the dinghy':'小艇',
    'The required orange dinghy.':'需要的橙色小艇。', 'A white look-alike is retrieved.':'检索到相似的白色小艇。', 'The wrong identity is composed.':'合成了错误的实体。', 'The requested dinghy is missing.':'需要的小艇没有出现。',
    'Earlier generated appearance of Axi wearing a scarf':'Axi先前戴着围巾的生成画面',
    'An older reference preserving the scarf':'保留着旧围巾的历史参考帧',
    'Composed keyframe showing the old scarf':'仍显示旧围巾的合成关键帧',
    'Still from the generated clip retaining the obsolete scarf':'生成片段中仍保留旧围巾的画面',
    'A relevant past can be wrong for the present. VMem-Bench studies what visual memory retrieves and what a long-video generator actually uses.':'与过去相关，不代表适用于当下。VMem-Bench 研究视觉记忆检索到了什么，以及长视频生成器实际使用了什么。',
    'Retrieval improves recall, but can bring back the wrong entity or an obsolete appearance. Explore the evidence.':'检索提高了召回率，却也可能带回错误的实体或过时的外观。一起来看研究证据。'
  };
  let language = 'en';
  try { language = localStorage.getItem('vmem-language') || (navigator.language.startsWith('zh') ? 'zh-CN' : 'en'); } catch (_) {}
  if (language !== 'zh-CN') language = 'en';
  const t = value => language === 'zh-CN' ? (zh[value] || value) : value;
  const originals = [];
  const walker = document.createTreeWalker(document.body, NodeFilter.SHOW_TEXT);
  while (walker.nextNode()) {
    const node = walker.currentNode;
    if (node.parentElement.closest('script, style, pre, [data-lang]')) continue;
    const key = node.textContent.trim().replace(/\s*·$/, '');
    if (zh[key]) originals.push({node, text:node.textContent, key});
  }
  const attributes = [];
  document.querySelectorAll('[aria-label], img[alt], meta[name="description"], meta[property="og:description"]').forEach(node => {
    for (const attr of ['aria-label','alt','content']) {
      const value = node.getAttribute(attr);
      if (zh[value]) attributes.push({node, attr, value});
    }
  });
  function apply(next) {
    language = next === 'zh-CN' ? 'zh-CN' : 'en';
    document.documentElement.lang = language;
    document.title = language === 'zh-CN' ? 'VMem-Bench — 更多记忆，更好的视频？' : 'VMem-Bench — More Memory, Better Videos?';
    originals.forEach(({node,text,key}) => { node.textContent = text.replace(key,t(key)); });
    attributes.forEach(({node,attr,value}) => node.setAttribute(attr,t(value)));
    document.querySelectorAll('[data-lang]').forEach(button => button.setAttribute('aria-pressed',String(button.dataset.lang === language)));
    try { localStorage.setItem('vmem-language',language); } catch (_) {}
    document.dispatchEvent(new CustomEvent('vmem-language-change'));
  }
  window.VMEM_I18N = {t, get language() { return language; }};
  document.querySelectorAll('[data-lang]').forEach(button => button.addEventListener('click', () => apply(button.dataset.lang)));
  apply(language);
})();
