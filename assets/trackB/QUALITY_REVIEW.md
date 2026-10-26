# VMem-Bench TrackB Story Quality Review

Review date: 2026-07-26

Scope: independent review of the 50 `gt_source/*.json` story scripts after batch rewriting. This review did not edit story JSON, derived GT, SUT prompts, generator scripts, prompts, or manuscript files.

## Post-Optimization Addendum

Status after targeted optimization: the specific B/C issues identified below have
been addressed by follow-up rewrite/polish passes for `0004`, `0005`,
`0006`-`0011`, `0042`, `0043`, and `0045`-`0050`.

Post-optimization validation:

```text
python3 assets/trackB/scripts/complete_gt.py
python3 assets/trackB/scripts/get_sut_prompts.py
python3 assets/trackB/scripts/audit_quality.py --strict

stories=50 segments min=68 max=197 mean=123.2 buckets={'50-80': 8, '81-120': 19, '121-160': 8, '161-200': 15}
gt_warnings=0
sut_warnings=0
audit_quality=OK
```

The original review below is kept as an audit trail. The post-optimization pool
passes the enhanced residue audit and all mechanical hard-case gates. A final
freeze decision should still be based on a short second-pass qualitative review
of the changed stories, but the previously blocking B/C findings have been
targeted.

## Executive Verdict

The 50-story TrackB pool is **not freeze-ready as-is**.

The strict mechanical gate passes:

```text
python3 assets/trackB/scripts/audit_quality.py --strict
stories=50 segments min=68 max=197 mean=123.2 buckets={'50-80': 8, '81-120': 19, '121-160': 8, '161-200': 15}
OK
```

However, qualitative review finds that several scripts are still visibly template-derived and contain domain-inconsistent residue that would be embarrassing in a benchmark freeze candidate. Most stories from `0012` onward are concrete, filmable, and usable as freeze candidates, but the pool needs optimization before freeze.

Grade counts:

| Grade | Count | Meaning |
|---|---:|---|
| A | 34 | Freeze candidate as written. |
| B | 10 | Usable draft; needs targeted polish. |
| C | 6 | Not freeze-quality; needs substantial rewrite. |

Explicit optimization verdict: **optimization is needed**. At minimum, rewrite `0006`-`0011` and polish the B-grade stories listed below before freezing the 50-story pool.

## Per-Story Grades

| story_id | title | grade | rationale | required fixes |
|---|---|---:|---|---|
| `0001_lighthouse_keeper` | 灯塔守望录 | A | Exemplar-level script: concrete nautical locations, coherent multi-decade arc, and hard cases are naturally motivated by the lighthouse succession story. | None. |
| `0002_night_market_courier` | 霓虹夜市信使 | A | Filmable urban noir variant with strong visual anchors and motivated object/state changes despite sharing the broad mentor-successor pattern. | None. |
| `0003_desert_archaeologist` | 沙海残碑 | A | Coherent archaeological expedition with clear props, states, removals, lookalikes, and delayed returns. | None. |
| `0004_silk_road_caravan` | 丝路驼铃 | B | Domain is coherent and filmable, but many scene beats are visibly copied from the same scaffold and the prose leans on repeated phrases. | Vary scene order and wording; reduce stock lines such as "轻嗤一声", "说明新的过关章程", and repeated memorial/flashback phrasing. |
| `0005_opera_mask_atelier` | 面具工坊 | B | The opera-mask domain is concrete, but several beats are still scaffold-like and a few phrases are awkward or duplicated. | Polish copied disaster/repair/false-friend wording; fix awkward phrases such as "朱红朱漆戏箱" and make the repair sequence more craft-specific. |
| `0006_harbor_ice_cutter` | 碎冰港湾 | C | Contains major domain leakage from the caravan template: `驿馆檐下`, `一头灰帆渔船`, `沙窝`, `鼻铃`, `青铜狮钮印`, and `重新重新` are not credible harbor imagery. | Substantially rewrite around harbor-specific actions, classifiers, destruction, repair, flashbacks, and final handoff. |
| `0007_bamboo_paper_mill` | 竹纸水碓 | C | Severe copied residue breaks filmability: `驿馆檐下`, `驼峰上的月牙烙印`, `青铜狮钮印`, `沙窝`, `鼻铃`, `狮鬃略浅`, and `重新重新` appear in a paper-mill story. | Substantial rewrite; replace camel/seal/copper-smith logic with papermaking-specific handling, damage, repair, and memory anchors. |
| `0008_clocktower_horologist` | 钟楼修表匠 | C | Strong premise is undermined by nonsensical template artifacts:吊篮 has `驼峰/月牙烙印`, `沙窝`, `鼻铃`, `青铜狮钮印`, `狮鬃`, and repeated generic closure. | Substantial rewrite; recast all hard cases through clockwork, lift, gears, oil, and tower-specific visual actions. |
| `0009_tea_horse_station` | 茶马驿站 | C | Some horse/驿站 elements fit, but the script still mixes inappropriate camel and seal residue (`驼峰`, `青铜狮钮印`, `狮鬃`, `重新重新`) and repeated scaffold scenes. | Rewrite the memory spine around tea bricks, horse tack, pass ledgers, and fire-seal craft; remove camel/fire-seal-template residue. |
| `0010_subway_violinist` | 地铁琴声 | C | Musical subway story is not freeze-quality because it retains irrelevant `驿馆`, `驼峰/月牙烙印`, `沙窝`, `鼻铃`, `青铜狮钮印`, `狮鬃`, and male-beard phrasing for a female violinist. | Substantial rewrite with subway/music-specific false friends, damage, repair, and flashback anchors. |
| `0011_greenhouse_botanist` | 温室植物志 | C | Greenhouse setting contains the same non-domain residue (`驿馆`, `驼峰/月牙烙印`, `青铜狮钮印`, `狮鬃`, `过关章程`, `重新重新`) and reads as a noun-swapped template. | Substantial rewrite using greenhouse-specific propagation, tools, carts, labels, growth stages, and plausible repair/removal scenes. |
| `0012_puppet_shadow_troupe` | 影戏堂夜话 | A | Concrete village shadow-puppet world with naturally embedded lookalikes, count memory, state changes, destroyed cart, and late succession. | None. |
| `0013_salt_flat_photographer` | 盐沼曝光簿 | A | Distinct photographic salt-flat setting with filmable objects and plausible temporal returns. | None. |
| `0014_forge_swordsmith` | 霜纹砧上 | A | Strong craft domain; state changes and hard cases fit forge work rather than feeling pasted in. | None. |
| `0015_library_night_archivist` | 穹顶夜编目 | A | Library/archive visuals are specific and the long-gap/indirect-reference probes are natural to cataloging and preservation. | None. |
| `0016_hot_spring_innkeeper` | 檜木汤烟 | A | Distinct inn/hot-spring setting with coherent props, locations, and succession arc. | None. |
| `0017_reef_diver` | 浅滩采样簿 | A | Strong underwater/reef imagery; hard cases are motivated by boats, sample boxes, logs, and dive equipment. | None. |
| `0018_train_porter` | 月台号牌 | A | Railway setting supports lookalikes, removals, counts, and temporal references without obvious template residue. | None. |
| `0019_lantern_festival_maker` | 金鲤灯夜 | A | Lantern-making domain is visually rich and the memory probes fit props, festival staging, and handoff. | None. |
| `0020_observatory_intern` | 紫金山值夜 | A | Observatory story has clear filmable devices and coherent long-horizon progression. | None. |
| `0021_pottery_kiln_master` | 龙窑守火人 | A | Strong kiln/craft setting with concrete states, props, false friends, and successor arc. | None. |
| `0022_vineyard_cellarer` | 橡木窖酒人 | A | Vineyard/cellar objects and state changes are concrete and distinct from neighboring stories. | None. |
| `0023_firefighter_rookies` | 湿沥青上的新训 | A | Firefighter training context naturally motivates state changes, removals, count memory, and false friends. | None. |
| `0024_circus_acrobat` | 马戏篷下的钢丝 | A | Circus staging gives clear visual specificity and natural lookalike/state-change opportunities. | None. |
| `0025_museum_night_guard` | 石厅夜巡人 | A | Museum night-watch setting is coherent and well suited to long-gap object memory. | None. |
| `0026_rice_terrace_farmer` | 梯田水线 | A | Farming/terrace domain provides concrete locations, tools, and seasonal continuity. | None. |
| `0027_radio_station_dj` | 红灯夜班 | A | Radio-station props and time references support the hard cases without obvious awkwardness. | None. |
| `0028_shipyard_welder` | 船坞焊工 | A | Shipyard story is visually specific and the damaged/repaired artifacts are plausible. | None. |
| `0029_ink_wash_painter` | 水墨画室 | A | Ink-painting studio has distinct props, states, and visual anchors. | None. |
| `0030_ski_patrol` | 雪山巡逻队 | A | Snow patrol story is concrete and filmable; probes fit rescue gear, snow vehicles, and delayed succession. | None. |
| `0031_spice_bazaar_cook` | 香料市集厨 | A | Market kitchen domain is visually rich and narratively coherent. | None. |
| `0032_beekeeper_meadow` | 牧场养蜂人 | A | Beekeeping setting uses distinct objects and state transitions with natural hard-case placement. | None. |
| `0033_film_projectionist` | 胶片放映员 | A | Projection booth setting is well matched to stateful props, false friends, and temporal memory. | None. |
| `0034_canal_boatman` | 运河船夫 | A | Canal-boat story is coherent, visually grounded, and distinct enough from harbor/reef cases. | None. |
| `0035_windmill_miller` | 风车磨坊主 | A | Windmill/milling objects provide natural count memory, state changes, and long-gap returns. | None. |
| `0036_calligraphy_school` | 墨香私塾 | A | Calligraphy-school setting is concrete and coherent, with hard cases embedded in tools and student succession. | None. |
| `0037_mine_lamp_checker` | 矿灯检查员 | A | Mine-lamp context gives strong visual specificity and motivated removals/false friends. | None. |
| `0038_balloon_festival` | 云舟气球节 | A | Balloon-festival visuals are distinct and naturally support lookalikes, counts, and state changes. | None. |
| `0039_sushi_counter_apprentice` | 茂鮨柜台 | A | Sushi-counter craft setting is concrete, compact, and filmable. | None. |
| `0040_luthier_workshop` | 衡声制琴坊 | A | Luthier workshop is visually specific and mechanically suitable for memory probes. | None. |
| `0041_border_customs` | 戍边查验站 | A | Border/customs story has clear temporal arc and domain-specific props. | None. |
| `0042_glassblower_studio` | 灼窑吹制棚 | B | Mostly coherent glassblowing story, but some prose is generic and the `手记`/repair beats are repeated from nearby late-batch stories. | Polish generic record-keeping lines, make false-friend and repair scenes more glass-specific, and reduce repeated "手记" phrasing. |
| `0043_orchard_grafting` | 柏园嫁接记 | B | Good orchard/grafting premise, but shares late-batch `手记` scaffolding and generic record/repair phrasing. | Replace generic `手记` beats with orchard-specific observations and vary repair/false-friend sequence. |
| `0044_subway_map_designer` | 线网制图夜 | A | Distinct from the subway violinist case and appears coherent without the earlier copied residue. | None. |
| `0045_tide_pool_naturalist` | 潮池标本记 | B | Concrete tide-pool domain, but a few late-batch stock lines such as "放回原位" make the hard cases feel inserted. | Polish the generic object-return line and make indirect references more observational/naturalist-specific. |
| `0046_monastery_bell_ringer` | 古寺撞钟录 | B | Filmable monastery setting, but several lines are generic placeholders (`认真记下当日状况`, `关键记号`, `一件...细节不同`). | Replace generic record and false-friend phrasing with bell-ringing/monastery-specific actions. |
| `0047_cargo_crane_operator` | 码头吊机班 | B | Cargo-crane domain is usable, but late-batch placeholder lines (`认真记下当日状况`, `一件玩具吊臂模型...细节不同`) reduce benchmark polish. | Rewrite generic log and decoy lines; make false friends and removals operationally plausible for a port crane crew. |
| `0048_embroidery_guild` | 绣坊针谱 | B | Embroidery setting is concrete, but repeated placeholder record lines and `一件...细节不同` false-friend language need polish. | Replace generic bookkeeping/decoy phrasing with needlework-specific visual actions. |
| `0049_volcano_geologist` | 火山岩芯志 | B | Volcanology premise is strong, but repeated `地温手记`/generic record lines and `一件...细节不同` wording are too visible. | Vary the hand-note beats; make decoys and state changes more field-geology-specific and less template-like. |
| `0050_planetarium_guide` | 天象厅解说 | B | Planetarium story is filmable, but has late-batch generic lines (`认真记下当日状况`, `关键记号`, `一件...细节不同`) that fall below freeze polish. | Replace generic record/decoy wording with astronomy-show operations and more natural prompt language. |

## Systemic Issues

1. **Strict audit is necessary but insufficient.** It confirms segment range, hard-case counts, zero warnings, and prompt-rendering health, but it does not detect semantic nonsense such as camel imagery inside paper mills or violin cases.
2. **`0006`-`0011` preserve obvious copied source residue.** Repeated artifacts include `驿馆檐下`, `驿馆梁下`, `驼峰/月牙烙印`, `青铜狮钮印`, `狮鬃略浅`, `沙窝`, `鼻铃`, `一头...车/船/台`, `过关章程`, and `重新重新`.
3. **Broad story skeleton is over-regular.** Many stories follow the same order: opening with 5-count and 2-count objects, introduce apprentice and 3-count object, lookalike-only scene, lookalike共屏, ledger state change, filler season passage, destroyed vehicle/prop, disaster aftermath, repair, false friends, aged mentor, apprentice return, memorial, rival return, flashback, closure. This is acceptable for coverage but weakens perceived diversity.
4. **Late-batch stories `0046`-`0050` use placeholder prose.** Phrases such as `认真记下当日状况`, `关键记号`, `放回原位`, and `一件...细节不同` are fluent but generic and not freeze-grade.
5. **Some hard cases are visibly inserted rather than story-motivated.** The false-friend and count-memory segments are sometimes natural, but in B/C stories they read like benchmark checklist items rather than cinematic events.
6. **SUT suitability is mostly sound after mechanical checks.** Prompt text is chronological and generally natural, and the strict audit reports no SUT prompt warnings. The issue is content quality, not label leakage.

## Recommended Next Actions

1. **Block freeze until B/C optimization is complete.**
2. **Substantially rewrite `0006`-`0011`.** These should not be line-polished; rebuild their scene language from the actual domain.
3. **Target-polish B stories.** `0004`, `0005`, `0042`, `0043`, and `0045`-`0050` likely need only small localized edits.
4. **Add a semantic-residue audit.** Extend `audit_quality.py` or add a companion qualitative lint for cross-domain residue patterns such as `驼峰`, `青铜狮钮印`, `狮鬃`, `沙窝`, `鼻铃`, `重新重新`, `认真记下当日状况`, `关键记号`, and `一件...细节不同`.
5. **After edits, rerun strict audit and spot-review SUT prompt samples.** Mechanical coverage should remain unchanged, but prompt prose should be rechecked for natural long-video generation wording.
