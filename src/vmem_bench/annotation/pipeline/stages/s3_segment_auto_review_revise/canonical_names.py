"""Canonical entity-name coverage for S3 actions (name-anchored MemStrata contract).

Gold prompts must contain the roster ``name`` of every ``present_entity_id``.
IDs stay in structured fields; natural-language ``action`` uses stable names only.
Deterministic code validates model text but never manufactures an entity-list coda.
"""

from __future__ import annotations

import json
import re
from typing import Any

ACTION_MISSING_CANONICAL_NAME = "action_missing_canonical_name"
ACTION_ENTITY_LIST_CODA = "action_entity_list_coda"
ENTITY_EMPTY_CANONICAL_NAME = "entity_empty_canonical_name"

_ENTITY_LIST_CODA_RE = re.compile(
    r"(?:^|[，,。；;.!?]\s*)(?:可见|出场\s*[:：]|showing\b)[^。.!?]*[。.!?]?\s*$",
    flags=re.IGNORECASE,
)


def normalize_for_name_match(text: str) -> str:
    """Lowercase Latin; collapse whitespace; keep CJK characters intact."""
    return " ".join(str(text or "").split()).casefold()


def action_contains_canonical_name(action: str, name: str) -> bool:
    """True when ``name`` appears as a contiguous span in ``action`` (casefold)."""
    needle = normalize_for_name_match(name)
    if not needle:
        return False
    haystack = normalize_for_name_match(action)
    # Latin: require token boundaries so "art" does not match inside "start".
    if re.fullmatch(r"[a-z0-9][a-z0-9 _'-]*", needle):
        pattern = r"(?<![a-z0-9])" + re.escape(needle) + r"(?![a-z0-9])"
        return re.search(pattern, haystack) is not None
    return needle in haystack


def action_has_entity_list_coda(action: str) -> bool:
    """Return whether ``action`` ends in a mechanical entity-list clause."""
    return _ENTITY_LIST_CODA_RE.search(str(action or "").strip()) is not None


def missing_canonical_names(
    *,
    action: str,
    present_entity_ids: list[str],
    roster_by_id: dict[str, dict[str, str]],
) -> list[dict[str, str]]:
    """Return present entities whose roster name is absent from ``action``."""
    missing: list[dict[str, str]] = []
    seen: set[str] = set()
    for entity_id in present_entity_ids:
        eid = str(entity_id)
        if not eid or eid in seen:
            continue
        seen.add(eid)
        entry = roster_by_id.get(eid) or {}
        name = str(entry.get("name") or "").strip()
        if not name:
            missing.append(
                {
                    "entity_id": eid,
                    "name": "",
                    "kind": str(entry.get("kind") or ""),
                    "reason": ENTITY_EMPTY_CANONICAL_NAME,
                }
            )
            continue
        if not action_contains_canonical_name(action, name):
            missing.append(
                {
                    "entity_id": eid,
                    "name": name,
                    "kind": str(entry.get("kind") or ""),
                    "reason": ACTION_MISSING_CANONICAL_NAME,
                }
            )
    return missing


def _longest_unique_name_fragment(
    *,
    action: str,
    name: str,
    peer_names: list[str],
) -> str:
    """Find a safe generic CJK mention contained in one canonical name only."""
    if not name or not any("\u4e00" <= char <= "\u9fff" for char in name):
        return ""
    for width in range(len(name) - 1, 1, -1):
        for start in range(len(name) - width + 1):
            fragment = name[start : start + width]
            if fragment not in action:
                continue
            if sum(fragment in peer for peer in peer_names) == 1:
                return fragment
    return ""


def _location_prefix(name: str, action: str) -> str:
    if not any("\u4e00" <= char <= "\u9fff" for char in name + action):
        return f"In {name}, "
    if name.endswith(("草地", "小径", "道路", "屋顶", "地面", "高空")):
        return f"{name}上，"
    if name.endswith(("洞穴", "森林", "房间", "室内", "庭院")):
        return f"{name}中，"
    return f"在{name}中，"


def _replace_first_canonical(text: str, fragment: str, name: str) -> str:
    """Replace one generic mention while absorbing a matching name prefix."""
    index = text.find(fragment)
    if index < 0:
        return text
    name_prefix = name.split(fragment, 1)[0]
    overlap = 0
    for width in range(1, min(len(name_prefix), index) + 1):
        if text[index - width : index] == name_prefix[:width]:
            overlap = width
    start = index - overlap
    return text[:start] + name + text[index + len(fragment) :]


def _join_cjk_names(names: list[str]) -> str:
    unique = list(dict.fromkeys(name for name in names if name))
    if len(unique) <= 1:
        return "".join(unique)
    if len(unique) == 2:
        return "和".join(unique)
    return "、".join(unique[:-1]) + "和" + unique[-1]


def rewrite_action_canonical_mentions(
    *,
    action: str,
    present_entity_ids: list[str],
    roster_by_id: dict[str, dict[str, str]],
) -> tuple[str, list[dict[str, str]], list[dict[str, str]]]:
    """Naturally canonicalize already-mentioned entities without list codas.

    This deterministic layer only replaces an unambiguous generic mention
    (``兔子`` → ``大兔子``) or supplies one missing location as a sentence
    prefix. It never invents a missing character/prop mention.
    """
    text = " ".join(str(action or "").split())
    rewrites: list[dict[str, str]] = []
    missing = missing_canonical_names(
        action=text,
        present_entity_ids=present_entity_ids,
        roster_by_id=roster_by_id,
    )
    peer_names = [
        str((roster_by_id.get(str(entity_id)) or {}).get("name") or "")
        for entity_id in present_entity_ids
        if (roster_by_id.get(str(entity_id)) or {}).get("name")
    ]

    for item in missing:
        name = str(item.get("name") or "")
        if not name or action_contains_canonical_name(text, name):
            continue
        fragment = _longest_unique_name_fragment(
            action=text,
            name=name,
            peer_names=peer_names,
        )
        if fragment:
            text = _replace_first_canonical(text, fragment, name)
            rewrites.append(
                {
                    "entity_id": item["entity_id"],
                    "name": name,
                    "kind": item.get("kind") or "",
                    "operation": "replace",
                    "matched_phrase": fragment,
                }
            )
            continue
        if item.get("kind") == "location" and name.endswith("洞穴") and "洞内" in text:
            text = text.replace("洞内", f"{name}内", 1)
            rewrites.append(
                {
                    "entity_id": item["entity_id"],
                    "name": name,
                    "kind": "location",
                    "operation": "replace",
                    "matched_phrase": "洞内",
                }
            )
            continue
        aliases: tuple[str, ...] = ()
        if item.get("kind") == "location" and name == "高空":
            aliases = ("天空", "空中")
        elif item.get("kind") == "prop" and name.endswith("苹果"):
            aliases = ("水果",)
        for alias in aliases:
            if alias not in text:
                continue
            text = text.replace(alias, name, 1)
            rewrites.append(
                {
                    "entity_id": item["entity_id"],
                    "name": name,
                    "kind": item.get("kind") or "",
                    "operation": "replace",
                    "matched_phrase": alias,
                }
            )
            break

    still = missing_canonical_names(
        action=text,
        present_entity_ids=present_entity_ids,
        roster_by_id=roster_by_id,
    )
    missing_characters = [
        item for item in still if item.get("kind") == "character" and item.get("name")
    ]
    if len(missing_characters) >= 2:
        for group_phrase in ("三个小动物", "小动物们", "三个动物"):
            if group_phrase not in text:
                continue
            names = [str(item["name"]) for item in missing_characters]
            replacement = _join_cjk_names(names)
            text = text.replace(group_phrase, replacement, 1)
            rewrites.append(
                {
                    "entity_id": ",".join(item["entity_id"] for item in missing_characters),
                    "name": replacement,
                    "kind": "character",
                    "operation": "expand_group",
                    "matched_phrase": group_phrase,
                }
            )
            break

    still = missing_canonical_names(
        action=text,
        present_entity_ids=present_entity_ids,
        roster_by_id=roster_by_id,
    )
    missing_locations = [
        item for item in still if item.get("kind") == "location" and item.get("name")
    ]
    if len(missing_locations) == 1:
        item = missing_locations[0]
        name = str(item["name"])
        text = _location_prefix(name, text) + text
        rewrites.append(
            {
                "entity_id": item["entity_id"],
                "name": name,
                "kind": "location",
                "operation": "prefix",
                "matched_phrase": "",
            }
        )

    still = missing_canonical_names(
        action=text,
        present_entity_ids=present_entity_ids,
        roster_by_id=roster_by_id,
    )
    return text, rewrites, still


_LATIN_RUN_RE = re.compile(r"[A-Za-z][A-Za-z0-9 _'\-]*")
_CJK_2GRAM_RE = re.compile(r"[\u4e00-\u9fff]{2}")
_SLASH_REPEAT_RE = re.compile(r"([\u4e00-\u9fff]{1,3})/\1")


def _roster_latin_tokens(roster_by_id: dict[str, dict[str, str]]) -> set[str]:
    tokens: set[str] = set()
    for entry in roster_by_id.values():
        for run in _LATIN_RUN_RE.findall(str(entry.get("name") or "")):
            token = run.strip().casefold()
            if token:
                tokens.add(token)
    return tokens


def action_looks_canonically_clean(
    action: str,
    roster_by_id: dict[str, dict[str, str]],
) -> bool:
    """Conservative fluency guard for auto-adopting a rewritten action as gold.

    A struggling VLM often emits garbage while injecting canonical names —
    stray Latin filler ("亚istinguished亚裔少年"), repeated junk tokens
    ("故障/故障", "…中年女子在pool中年女子在pool…"). Such text still satisfies
    the name-coverage gate, so it must not be silently promoted to a PASS gold
    label. This check errs toward rejection: when in doubt the segment stays a
    BLOCK and reaches human review. It is used only to decide auto-adoption,
    never to mutate text.
    """
    text = str(action or "")
    if not text:
        return False
    allowed = _roster_latin_tokens(roster_by_id)
    for run in _LATIN_RUN_RE.findall(text):
        if run.strip().casefold() not in allowed:
            return False
    if _SLASH_REPEAT_RE.search(text):
        return False
    roster_names = [str(entry.get("name") or "") for entry in roster_by_id.values()]
    for token in set(_CJK_2GRAM_RE.findall(text)):
        if text.count(token) >= 3 and not any(token in name for name in roster_names):
            return False
    return True


def try_complete_canonical_action(
    *,
    action: str,
    present_entity_ids: list[str],
    roster_by_id: dict[str, dict[str, str]],
) -> str | None:
    """Deterministically finish a near-miss action, or refuse.

    Runs the same canonicalizer that protects the seed. Returns the completed
    text only when it fully covers every present entity's canonical name *and*
    passes :func:`action_looks_canonically_clean`; otherwise returns ``None`` so
    the caller keeps its conservative revert/flag path. Never invents an absent
    character/prop mention.
    """
    completed, _rewrites, still_missing = rewrite_action_canonical_mentions(
        action=action,
        present_entity_ids=present_entity_ids,
        roster_by_id=roster_by_id,
    )
    if still_missing or not completed:
        return None
    if not action_looks_canonically_clean(completed, roster_by_id):
        return None
    return completed


def format_missing_names_for_prompt(missing: list[dict[str, str]]) -> str:
    if not missing:
        return "[]"
    compact = [
        {
            "entity_id": item["entity_id"],
            "name": item.get("name") or "(EMPTY_NAME)",
            "kind": item.get("kind") or "",
        }
        for item in missing
    ]
    return json.dumps(compact, ensure_ascii=False)
