from __future__ import annotations

import asyncio
import math
import json
import re
import hashlib
import sqlite3
import time
from datetime import datetime
from pathlib import Path

from astrbot.api import logger
from astrbot.api.event import AstrMessageEvent, filter
from astrbot.api.provider import ProviderRequest
from astrbot.api.star import Context, Star, StarTools
from astrbot.core.agent.message import TextPart
from astrbot.core.message.message_event_result import MessageChain
from astrbot.api.message_components import Plain

DAY = 86400
PROMPT_TAGS = ["主人", "家人", "朋友", "自己", "恋人", "工作", "上班", "加班", "面试", "学习", "考试", "论文", "毕业", "编程", "代码", "调试", "报错", "部署", "模型", "电脑", "手机", "游戏", "动漫", "抽卡", "番剧", "音乐", "歌", "电影", "视频", "小说", "吃饭", "午饭", "晚饭", "夜宵", "零食", "咖啡", "奶茶", "做饭", "外卖", "睡觉", "熬夜", "失眠", "生病", "感冒", "医院", "吃药", "运动", "跑步", "累", "钱", "工资", "花钱", "购物", "快递", "房租", "旅行", "打车", "天气", "开心", "难过", "委屈", "生气", "焦虑", "孤独"]

TAXONOMY = ["主人", "家人", "朋友", "同学", "同事", "网友", "群友", "自己", "恋人", "老师", "学生", "室友", "邻居", "亲戚", "偶像", "粉丝", "陌生人", "小孩", "长辈", "工作", "上班", "加班", "面试", "实习", "学习", "考试", "作业", "论文", "课程", "成绩", "毕业", "求职", "摸鱼", "开会", "出差", "项目", "汇报", "任务", "deadline", "编程", "代码", "调试", "报错", "部署", "服务器", "数据库", "接口", "模型", "算法", "前端", "后端", "脚本", "插件", "配置", "电脑", "手机", "网络", "软件", "硬件", "游戏", "动漫", "漫画", "番剧", "手游", "端游", "抽卡", "通关", "攻略", "角色", "剧情", "周边", "同人", "cos", "音乐", "歌", "电影", "剧集", "综艺", "视频", "直播", "播客", "有声书", "小说", "读书", "文章", "新闻", "纪录片", "吃饭", "早饭", "午饭", "晚饭", "夜宵", "零食", "咖啡", "奶茶", "喝酒", "做饭", "外卖", "食堂", "水果", "辣", "甜", "睡觉", "起床", "熬夜", "午睡", "失眠", "通勤", "迟到", "生病", "感冒", "发烧", "头疼", "肚子疼", "医院", "吃药", "体检", "运动", "跑步", "健身", "减肥", "洗澡", "钱", "工资", "花钱", "购物", "网购", "快递", "房租", "水电", "存钱", "理财", "旅行", "高铁", "飞机", "打车", "地铁", "开车", "迷路", "景点", "酒店", "拍照", "家务", "打扫", "宠物", "植物", "装修", "搬家", "天气", "节日"]
FACTS_TITLE = "【长期记忆】"
DIARY_TITLE = "【云上日记】"
SHORT_ACK = {"嗯", "哦", "好", "好的", "在", "在吗", "hi", "在不在", "ok", "行", "收到", "?", "？", "草", "啊"}

REMEMBER_HINTS = ("记住", "别忘", "记一下", "要记得", "记牢", "答应我", "约好", "约定")
STOP = set("的了吗呢吧啊呀哦嗯是在和与就都也还又很太把被给让这那一个我你他她它们")


def split_sections(prompt: str, titles: list[str]) -> tuple[str, dict[str, str]]:
    base = prompt
    found: dict[str, str] = {}
    for title in titles:
        idx = base.find(title)
        if idx < 0:
            continue
        rest = base[idx:]
        nxt = len(rest)
        for other in titles:
            if other == title:
                continue
            j = rest.find(other, len(title))
            if 0 <= j < nxt:
                nxt = j
        found[title] = rest[:nxt].rstrip()
        base = (base[:idx] + rest[nxt:]).rstrip()
    return base, found


_BIGRAM_CACHE: dict = {}


def bigrams(text: str) -> frozenset:
    text = re.sub(r"[\s\W_]+", "", str(text or ""))
    cached = _BIGRAM_CACHE.get(text)
    if cached is not None:
        return cached
    if len(text) < 2:
        out = frozenset({text}) if text else frozenset()
    else:
        out = frozenset({text[i:i + 2] for i in range(len(text) - 1)} - STOP)
    if len(_BIGRAM_CACHE) > 20000:
        _BIGRAM_CACHE.clear()
    _BIGRAM_CACHE[text] = out
    return out


def jaccard(a: set[str], b: set[str]) -> float:
    if not a or not b:
        return 0.0
    return len(a & b) / len(a | b)


def cosine(a: list[float], b: list[float]) -> float:
    if not a or not b or len(a) != len(b):
        return 0.0
    dot = 0.0
    na = 0.0
    nb = 0.0
    for x, y in zip(a, b):
        dot += x * y
        na += x * x
        nb += y * y
    if na <= 0 or nb <= 0:
        return 0.0
    return dot / ((na ** 0.5) * (nb ** 0.5))


def text_key(text: str) -> str:
    import hashlib

    return hashlib.sha1(text.strip().encode("utf-8")).hexdigest()[:16]


class MemoryLayersPlugin(Star):
    def __init__(self, context: Context, config: dict | None = None):
        super().__init__(context)
        self.config = config or {}
        self.data_dir = Path(StarTools.get_data_dir("astrbot_plugin_waves"))
        self.data_dir.mkdir(parents=True, exist_ok=True)
        self.archive_dir = self.data_dir / "archive"
        self.archive_dir.mkdir(parents=True, exist_ok=True)
        self.state_path = self.data_dir / "state.json"
        self.l2_path = self.data_dir / "l2_notes.json"
        self.l3_path = self.data_dir / "l3_facts.json"
        self.graph_path = self.data_dir / "graph.json"
        self.l1_meta_path = self.data_dir / "l1_meta.json"
        self.state: dict = {"conv_offsets": {}, "archived": {}, "pending": []}
        self.l2: list[dict] = []
        self.l3: list[dict] = []
        self._task: asyncio.Task | None = None
        self._busy = False
        self._last_sync = 0.0
        self.recent_meta: list[dict] = []
        self.diary_entries: list[dict] = []
        self.last_diary: float = 0.0
        self._graph: dict = {"nodes": [], "links": []}
        self._adj: dict[str, list[str]] = {}
        self.emb_path = self.data_dir / "embeddings.json"
        self.emb: dict[str, list[float]] = {}
        self._load()

    # ---------- config ----------
    def cfg(self, key: str, default=None):
        try:
            value = self.config.get(key, default)
        except Exception:
            value = default
        return default if value is None else value

    def _int(self, key: str, default: int, minimum: int = 0) -> int:
        try:
            return max(minimum, int(self.cfg(key, default)))
        except Exception:
            return default

    # ---------- persist ----------
    def _load(self):
        for path, attr, default in (
            (self.state_path, "state", {}),
            (self.l2_path, "l2", []),
            (self.l3_path, "l3", []),
        ):
            try:
                if path.exists():
                    data = json.loads(path.read_text(encoding="utf-8"))
                    if isinstance(data, type(default)):
                        setattr(self, attr, data)
            except Exception as exc:
                logger.warning(f"[记忆] 读取 {path.name} 失败: {exc}")
        far = time.time() + 3650 * DAY
        for fact in self.l3:
            fact["expires_at"] = max(fact.get("expires_at", 0), far)
        try:
            if self.graph_path.exists():
                self._graph = json.loads(self.graph_path.read_text(encoding="utf-8"))
        except Exception:
            self._graph = {"nodes": [], "links": []}
        self._load_diary()
        logger.info("[Wave] diary loaded: %d entries (file exists=%s)" % (len(self.diary_entries), self._diary_state_path().exists()))

        try:
            if self.l1_meta_path.exists():
                data = json.loads(self.l1_meta_path.read_text(encoding="utf-8"))
                if isinstance(data, list):
                    self.recent_meta = data
        except Exception:
            self.recent_meta = []
        try:
            if self.emb_path.exists():
                data = json.loads(self.emb_path.read_text(encoding="utf-8"))
                if isinstance(data, dict):
                    self.emb = data
        except Exception:
            self.emb = {}

    def _save_state(self) -> None:
        self._save(self.state_path, self.state)
    def _save(self, path: Path, payload) -> None:
        try:
            path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
        except Exception as exc:
            logger.warning(f"[记忆] 写入 {path.name} 失败: {exc}")

    # ---------- L4 archive ----------
    def _db_path(self) -> Path:
        return self.data_dir.parents[1] / "data_v4.db"

    def sync_archive_sync(self) -> list[dict]:
        import sqlite3

        db = self._db_path()
        if not db.exists():
            return []
        prefix = str(self.cfg("session_prefix", "snowluma:FriendMessage:"))
        offsets = dict(self.state.get("conv_offsets") or {})
        archived = dict(self.state.get("archived") or {})
        added: list[dict] = []
        con = sqlite3.connect(f"file:{db}?mode=ro", uri=True, timeout=10)
        try:
            rows = con.execute(
                "select conversation_id, content from conversations where user_id like ? order by updated_at asc",
                (prefix + "%",),
            ).fetchall()
        finally:
            con.close()
        now = time.time()
        cache = getattr(self, "_conv_parse_cache", None)
        if cache is None:
            cache = self._conv_parse_cache = {}
        for conversation_id, raw in rows:
            key = (conversation_id, len(raw) if isinstance(raw, str) else id(raw))
            cached = cache.get(conversation_id)
            if cached and cached[0] == key:
                messages = cached[1]
            else:
                messages = None
            if messages is not None:
                pass
            try:
                if messages is None:
                    messages = json.loads(raw) if isinstance(raw, str) else (raw or [])
            except Exception:
                continue
            if not isinstance(messages, list):
                continue
            cache[conversation_id] = (key, messages)
            start = max(int(offsets.get(conversation_id, 0) or 0), int(archived.get(conversation_id, 0) or 0))
            if start > len(messages):
                start = 0
            for item in messages[start:]:
                role, text = self._parse(item)
                if not text:
                    continue
                rec = {"ts": now, "role": role, "text": text, "conv": conversation_id}
                added.append(rec)
                self._append(rec)
            offsets[conversation_id] = len(messages)
            archived[conversation_id] = len(messages)
        self.state["conv_offsets"] = offsets
        self.state["archived"] = archived
        return added

    def _append(self, rec: dict) -> None:
        day = datetime.fromtimestamp(rec["ts"]).strftime("%Y-%m-%d")
        try:
            with (self.archive_dir / f"{day}.jsonl").open("a", encoding="utf-8") as fh:
                fh.write(json.dumps(rec, ensure_ascii=False) + "\n")
        except Exception as exc:
            logger.warning(f"[记忆] 存档失败: {exc}")

    @staticmethod
    def _parse(item) -> tuple[str, str]:
        if not isinstance(item, dict):
            return "user", ""
        role = str(item.get("role") or "user")
        content = item.get("content")
        parts: list[str] = []
        if isinstance(content, str):
            parts.append(content)
        elif isinstance(content, list):
            for seg in content:
                if isinstance(seg, str):
                    parts.append(seg)
                elif isinstance(seg, dict) and seg.get("type") == "text":
                    parts.append(str(seg.get("text") or ""))
                elif isinstance(seg, dict) and seg.get("type") in ("image_url", "image"):
                    parts.append("[图片]")
                elif isinstance(seg, dict) and seg.get("type") in ("audio", "record"):
                    parts.append("[语音]")
        text = " ".join(p.strip() for p in parts if p and p.strip())
        while "<system_reminder>" in text:
            start = text.find("<system_reminder>")
            end = text.find("</system_reminder>")
            if end < 0:
                text = text[:start]
                break
            text = text[:start] + text[end + len("</system_reminder>"):]
        return role, " ".join(text.split())[:800]

    # ---------- llm ----------
    async def _resolve_provider(self) -> str:
        configured = str(self.cfg("cheap_provider_id", "") or "").strip()
        if configured:
            return configured
        try:
            conf = self.context.get_config()
            model = ((conf.get("agent_runner") or {}).get("config") or {}).get("model") or {}
            pid = str(model.get("provider_id") or "").strip()
            if pid:
                return pid
        except Exception:
            pass
        return ""

    async def _llm(self, prompt: str, system_prompt: str = "") -> str:
        pid = await self._resolve_provider()
        if not pid:
            raise RuntimeError("没有可用的对话模型")
        resp = await self.context.llm_generate(
            chat_provider_id=pid, prompt=prompt, system_prompt=system_prompt or None
        )
        return (getattr(resp, "completion_text", "") or "").strip()

    @staticmethod
    def _extract_json(raw: str) -> dict:
        if not raw:
            return {}
        match = re.search(r"\{.*\}", raw, re.S)
        try:
            data = json.loads(match.group(0) if match else raw)
            return data if isinstance(data, dict) else {}
        except Exception:
            return {}

    # ---------- L2/L3 ----------
    async def compress(self) -> str:
        if self._busy:
            return "busy"
        self._busy = True
        try:
            self._snapshot("before-compress")
            batch_size = self._int("compress_batch", 20, 5)
            pending = list(self.state.get("pending") or [])
            if not pending:
                return "empty"
            batch, rest = pending[:batch_size], pending[batch_size:]
            lines = [
                f"{i}. {'主人' if item.get('role') == 'user' else '我'}: {item.get('text', '')[:200]}"
                for i, item in enumerate(batch)
            ]
            tag_hint = self._int("tag_max", 3, 1)
            prompt = (
                "下面是你们最近的一段聊天。只输出 JSON：\n"
                '{"items":[{"i":序号,"score":0-10,"summary":"一句话","tags":["标签"],"emotion":{"label":"情绪","v":-1到1}}],'
                '"facts":[{"text":"值得长期记住的事实","tags":["标签"],"emotion":{"label":"情绪","v":-1到1}}]}\n'
                "score：10=明确喜好/约定/要我记住；6-9=情绪或关系重要信息；3-5=普通；0-2=寒暄。\n"
                "summary：不超过 25 字。\n"
                f"tags：每条**必须给 5 个**标签。优先从下面词表里选；如果词表实在覆盖不到，允许自造 1-2 个 2-4 字的短标签（不要生造长句）：" + "、".join(PROMPT_TAGS) + "\n"
                "用于以后联想，不要 # 号。"
                "emotion：这条消息的情绪标签（如 开心/委屈/期待/疲惫/生气/平静）和强度 v（-1 负到 1 正）。\n"
                "facts：只提取长期有效的信息（喜好、身份、习惯、约定、要我记住的事），"
                "每条带 1-2 个标签；不要一次性事件。\n"
                "不要输出 JSON 以外的内容。\n\n聊天：\n" + "\n".join(lines)
            )
            payload = self._extract_json(await self._llm(prompt, "你是精确的信息压缩与标签器，只输出 JSON。"))
            now = time.time()
            ttl_l2 = self._int("l2_ttl_hours", 24, 1) * 3600
            for entry in payload.get("items") or []:
                try:
                    idx = int(entry.get("i"))
                    score = int(entry.get("score", 0))
                except Exception:
                    continue
                if not (0 <= idx < len(batch)):
                    continue
                role = batch[idx].get("role")
                if role not in ("user", "assistant"):
                    continue
                summary = str(entry.get("summary") or "").strip()
                if not summary:
                    continue
                tags = [str(t).strip()[:8] for t in (entry.get("tags") or []) if str(t).strip()][:6]
                self.l2.append(
                    {"ts": now, "score": score, "summary": summary, "role": role, "tags": tags, "emotion": entry.get("emotion") or {}, "expires_at": now + ttl_l2}
                )
                text = str(batch[idx].get("text") or "")
                if role == "user" and any(h in text for h in REMEMBER_HINTS):
                    self._add_fact(text[:160], now, tags, entry.get("emotion"))
            for fact in payload.get("facts") or []:
                if isinstance(fact, dict):
                    text = str(fact.get("text") or "").strip()
                    tags = [str(t).strip()[:8] for t in (fact.get("tags") or []) if str(t).strip()][:6]
                else:
                    text, tags = str(fact or "").strip(), []
                if text:
                    self._add_fact(text, now, tags, (fact.get("emotion") if isinstance(fact, dict) else None))
            self.state["pending"] = rest
            self._save_state()
            self._prune_l2()
            self._save(self.l3_path, self.l3)
            for _note in self.l2[-len(batch) - 2:]:
                if not (_note.get("emotion") or {}).get("v"):
                    emo = self._emotion_of(str(_note.get("summary") or ""))
                    if emo:
                        _note["emotion"] = emo
            logger.debug(f"[记忆] 压缩 {len(batch)} 条 -> L2 {len(self.l2)} / L3 {len(self.l3)}")
            await self._sync_persona()
            await asyncio.to_thread(self.build_graph)
            asyncio.create_task(self._deferred_semantic())
            return f"压缩 {len(batch)} 条"
        finally:
            self._busy = False

    @staticmethod
    def _edit1(a: str, b: str) -> bool:
        """编辑距离 <= 1 的粗略判断，用于合并近似标签。"""
        if a == b:
            return True
        la, lb = len(a), len(b)
        if abs(la - lb) > 1:
            return False
        if la == lb:
            diff = sum(1 for x, y in zip(a, b) if x != y)
            return diff <= 1
        # 长度差 1：短串是否为长串删一字符
        short, long_ = (a, b) if la < lb else (b, a)
        i = j = 0
        skipped = False
        while i < len(short) and j < len(long_):
            if short[i] == long_[j]:
                i += 1
                j += 1
                continue
            if skipped:
                return False
            skipped = True
            j += 1
        return True

    def _norm_tags(self, tags: list[str]) -> list[str]:
        """把新标签归一到已有近似标签，避免「螺蛳粉/螺丝粉」这类稀释。"""
        try:
            known = set((self._tag_index or {}).keys())
            out = []
            for tag in tags:
                tag = str(tag).strip()[:8]
                if not tag:
                    continue
                hit = tag if tag in known else ""
                if not hit:
                    for k in known:
                        if self._edit1(tag, k):
                            hit = k
                            break
                out.append(hit or tag)
            return list(dict.fromkeys(out))
        except Exception:
            return tags

    async def _self_review(self) -> str:
        """定期自审：合并重复事实、改写含糊描述。"""
        try:
            facts = [f for f in self.l3 if not f.get("parked")]
            if len(facts) < 6:
                return "facts too few"
            listing = "\n".join(f"{i}. {f.get('text','')}  #标签:{'/'.join((f.get('tags') or [])[:4])}" for i, f in enumerate(facts[:50]))
            prompt = (
                "下面是长期记忆列表。请找出说同一件事的条目，并改写表述含糊的条目。只输出 JSON：\n"
                '{"merge":[[1,5],[3,9]],"rewrite":[{"i":2,"text":"更清楚的一句话"}]}\n'
                "规则：1) merge 里每组是同一件事的下标，保留信息最多的；2) 只在确实重复时才 merge；"
                "3) rewrite 只改含糊、缺主语、指代不明的；4) 数量要少，没有就给空数组。\n\n" + listing
            )
            raw = await self._llm(prompt, "你是记忆整理员，保守、精确，只输出 JSON。")
            payload = self._extract_json(raw)
            merged = 0
            drop = set()
            for group in (payload.get("merge") or []):
                idxs = [i for i in group if isinstance(i, int) and 0 <= i < len(facts)]
                if len(idxs) < 2:
                    continue
                keep = max(idxs, key=lambda i: len(str(facts[i].get("text") or "")))
                for i in idxs:
                    if i == keep:
                        continue
                    facts[keep]["tags"] = list(dict.fromkeys((facts[keep].get("tags") or []) + (facts[i].get("tags") or [])))[:7]
                    facts[keep]["hits"] = int(facts[keep].get("hits") or 1) + int(facts[i].get("hits") or 1)
                    facts[keep]["pinned"] = bool(facts[keep].get("pinned") or facts[i].get("pinned"))
                    drop.add(i)
                    merged += 1
            for item in (payload.get("rewrite") or []):
                try:
                    i = int(item.get("i"))
                    text = str(item.get("text") or "").strip()
                    if 0 <= i < len(facts) and 4 <= len(text) <= 120:
                        facts[i]["text"] = text
                except Exception:
                    continue
            if drop:
                parked = [f for f in self.l3 if f.get("parked")]
                self.l3 = [f for i, f in enumerate(facts) if i not in drop] + parked
                self._save(self.l3_path, self.l3)
                await self._apply_persona()
                await asyncio.to_thread(self.build_graph)
            self.state["last_review"] = time.time()
            self._save_state()
            msg = f"merged {merged}, rewrote {len(payload.get('rewrite') or [])}, now {len(self.l3)}"
            logger.info("[Wave] self-review: " + msg)
            return msg
        except Exception as exc:
            logger.warning(f"[Wave] self-review failed: {exc}")
            return ""

    def _promote_hot_notes(self, threshold: int = 3) -> int:
        """被反复提起的近期摘要，晋升成长期事实，避免到期被删就丢了。"""
        try:
            now = time.time()
            promoted = 0
            existing = [bigrams(str(f.get("text") or "")) for f in self.l3 if not f.get("parked")]
            for note in list(self.l2):
                if int(note.get("hits") or 1) < threshold:
                    continue
                text = str(note.get("summary") or "").strip()
                if len(text) < 8:
                    continue
                gram = bigrams(text)
                if any(jaccard(gram, g) >= 0.7 for g in existing):
                    continue
                fact = {
                    "text": text[:120],
                    "tags": list(note.get("tags") or [])[:7],
                    "created": float(note.get("ts") or now),
                    "last_seen": now,
                    "hits": int(note.get("hits") or 1),
                    "emotion": note.get("emotion") or {},
                    "expires_at": now + 3650 * DAY,
                    "source": "promoted",
                }
                self.l3.append(fact)
                existing.append(gram)
                promoted += 1
                if promoted >= 5:
                    break
            if promoted:
                self._save(self.l3_path, self.l3)
                logger.info(f"[Wave] promoted {promoted} hot notes to long-term")
            return promoted
        except Exception as exc:
            logger.debug(f"[Wave] promote failed: {exc}")
            return 0

    WEIGHT_BASE = ((("约定", "承诺", "身份", "边界", "情感", "关系"), 3.0),
                   (("偏好", "作息", "习惯", "口味"), 2.5),
                   (("梗", "表情", "玩笑"), 1.6))

    def _fact_weight(self, fact: dict, now: float | None = None) -> float:
        """权重 = 类型基础分 × 可信度 + 强化 - 时间衰减（每天 -0.1）。"""
        try:
            now = now or time.time()
            if float(fact.get("confidence") or 1.0) == 0:
                return -99.0
            vu = float(fact.get("valid_until") or 0)
            if vu and vu < now:
                return -99.0
            tags = "/".join(str(t) for t in (fact.get("tags") or []))
            base = 1.3
            for words, score in self.WEIGHT_BASE:
                if any(w in tags for w in words):
                    base = score
                    break
            if fact.get("pinned"):
                base = max(base, 3.5)
            try:
                conf = float(fact.get("confidence")) if fact.get("confidence") is not None else (1.0 if str(fact.get("source") or "") == "owner" else 0.65)
            except Exception:
                conf = 0.65
            if conf < 0.5:
                base *= 0.45
            elif conf < 0.75:
                base *= 0.7
            hits = int(fact.get("hits") or 1)
            last = float(fact.get("last_seen") or fact.get("created") or now)
            days = max(0.0, (now - last) / DAY)
            return round(base + 0.5 * min(6, hits - 1) - 0.1 * days, 3)
        except Exception:
            return 1.3

    def _snapshot(self, tag: str = "manual") -> None:
        """压缩前存一份状态快照，保留最近 10 份。"""
        try:
            import shutil
            root = self.data_dir / "snapshots"
            root.mkdir(parents=True, exist_ok=True)
            stamp = time.strftime("%Y%m%d-%H%M%S")
            dest = root / f"{stamp}-{tag}"
            dest.mkdir(exist_ok=True)
            for name in ("l3_facts.json", "l2_notes.json", "state.json", "diary_state.json"):
                src = self.data_dir / name
                if src.exists():
                    shutil.copy2(src, dest / name)
            snaps = sorted([p for p in root.iterdir() if p.is_dir()])
            keep = self._int("snapshot_keep", 10, 1)
            for old in snaps[: max(0, len(snaps) - keep)]:
                shutil.rmtree(old, ignore_errors=True)
        except Exception as exc:
            logger.debug(f"[Wave] snapshot failed: {exc}")

    def _archive_facts(self, victims: list[dict], reason: str) -> None:
        try:
            ap = self.data_dir / "facts_archive.json"
            arr = json.loads(ap.read_text(encoding="utf-8")) if ap.exists() else []
            for f in victims:
                arr.append({"text": f.get("text"), "tags": f.get("tags") or [],
                            "w": self._fact_weight(f), "reason": reason,
                            "archived_at": time.strftime("%Y-%m-%d %H:%M")})
            ap.write_text(json.dumps(arr[-500:], ensure_ascii=False, indent=2), encoding="utf-8")
        except Exception:
            pass

    def _expire_facts(self, now: float | None = None) -> int:
        """带到期时间的记忆，过期即移入归档。"""
        now = now or time.time()
        dead = [f for f in self.l3
                if not f.get("parked") and float(f.get("valid_until") or 0)
                and float(f.get("valid_until")) < now]
        if not dead:
            return 0
        self._archive_facts(dead, "expired")
        ids = {id(f) for f in dead}
        self.l3 = [f for f in self.l3 if id(f) not in ids]
        self._save(self.l3_path, self.l3)
        logger.info(f"[Wave] expired {len(dead)} facts")
        return len(dead)

    def _archive_capacity(self) -> int:
        """活跃事实超过上限时，把权重最低的收进归档。"""
        active = [f for f in self.l3 if not f.get("parked")]
        cap = self._int("fact_capacity", 120, 20)
        if len(active) <= cap:
            return 0
        batch = max(self._int("fact_archive_batch", 20, 5), len(active) - cap)
        victims = sorted(active, key=lambda f: self._fact_weight(f))[:batch]
        self._archive_facts(victims, "capacity")
        ids = {id(f) for f in victims}
        self.l3 = [f for f in self.l3 if id(f) not in ids]
        self._save(self.l3_path, self.l3)
        logger.info(f"[Wave] capacity archive: -{len(victims)}, left {len(self.l3)}")
        return len(victims)

    def _invalidate_fact(self, keyword: str) -> str:
        """主人说记错了：把匹配到的那条作废（不再注入、不进人格）。"""
        q = str(keyword or "").strip()
        if not q:
            return "要给个关键词我才知道撤回哪条"
        cands = [f for f in self.l3 if not f.get("parked") and q in str(f.get("text") or "")]
        if not cands:
            return f"没找到包含「{q}」的记忆"
        cands.sort(key=lambda f: len(str(f.get("text") or "")))
        hit = cands[0]
        hit["confidence"] = 0
        hit["w"] = 0
        hit["invalidated_at"] = time.time()
        self._save(self.l3_path, self.l3)
        asyncio.create_task(self._apply_persona())
        return f"已作废：{str(hit.get('text'))[:40]}"

    def _relationship_line(self) -> str:
        start = str(self.cfg("relationship_start", "") or "").strip()
        if not start:
            return ""
        try:
            import datetime as _dt
            d0 = _dt.datetime.strptime(start, "%Y-%m-%d").date()
            n = (_dt.date.today() - d0).days + 1
            if n > 0:
                return f"【关系】今天是认识的第 {n} 天（{start} 起）\n"
        except Exception:
            pass
        return ""

    def _dedupe_facts(self) -> int:
        """把说同一件事的多条事实合并掉，避免 L3 臃肿。"""
        try:
            facts = [f for f in self.l3 if not f.get("parked")]
            if len(facts) < 2:
                return 0
            merged = 0
            out: list[dict] = []
            for fact in facts:
                gram = bigrams(str(fact.get("text") or ""))
                target = None
                _num_now = ",".join(re.findall(r"\d+", str(fact.get("text") or "")))
                for kept in out:
                    _num_k = ",".join(re.findall(r"\d+", str(kept.get("text") or "")))
                    if _num_now != _num_k:
                        continue
                    if jaccard(gram, bigrams(str(kept.get("text") or ""))) >= 0.72:
                        target = kept
                        break
                if target is None:
                    out.append(fact)
                    continue
                # 合并：保留信息更全的描述，标签取并集，次数相加
                if len(str(fact.get("text") or "")) > len(str(target.get("text") or "")):
                    target["text"] = fact.get("text")
                tags = list(dict.fromkeys((target.get("tags") or []) + (fact.get("tags") or [])))[:7]
                target["tags"] = tags
                target["hits"] = int(target.get("hits") or 1) + int(fact.get("hits") or 1)
                target["pinned"] = bool(target.get("pinned") or fact.get("pinned"))
                merged += 1
            if merged:
                parked = [f for f in self.l3 if f.get("parked")]
                self.l3 = out + parked
                self._save(self.l3_path, self.l3)
                logger.info(f"[Wave] facts deduped: merged {merged}, now {len(out)}")
            return merged
        except Exception as exc:
            logger.debug(f"[Wave] dedupe failed: {exc}")
            return 0

    DURABLE = ("喜欢", "讨厌", "习惯", "约定", "答应", "以后", "每次", "总是", "从不",
               "不要", "别", "记住", "名字", "生日", "称呼", "怕", "过敏", "忌口", "梦想",
               "目标", "计划", "工作", "学校", "专业", "家人", "朋友", "边界", "隐私")
    TRANSIENT = ("今天", "刚才", "现在", "刚刚", "等下", "一会儿", "昨天", "这周", "好累", "有点")

    def _fact_worth(self, text: str) -> bool:
        """判断一条事实值不值得进 L3：有长期信号、不是纯当下情绪。"""
        try:
            t = str(text or "").strip()
            if len(t) < 8 or len(t) > 160:
                return False
            durable = sum(1 for k in self.DURABLE if k in t)
            transient = sum(1 for k in self.TRANSIENT if k in t)
            if durable == 0 and transient >= 1:
                return False
            if durable == 0 and len(t) < 14:
                return False
            return True
        except Exception:
            return True

    def _add_fact(self, text: str, now: float, tags: list[str] | None = None, emotion: dict | None = None, source: str | None = None, valid_until: float = 0.0, pinned: bool = False) -> None:
        text = " ".join(text.split())[:200]
        if not text:
            return
        if not pinned and not self._fact_worth(text):
            logger.debug("[Wave] skip low-value fact: " + text[:30])
            return
        if tags:
            tags = self._norm_tags(list(tags))
        if any(bad in text for bad in ("无法发送", "素材库为空", "合成失败", "暂时不能", "还不能发", "尚未接通")):
            return
        # 数字不同就别当同一件事（「抽玉溪」和「抽中华」不是一条）
        def _nums(t: str) -> str:
            return ",".join(re.findall(r"\d+", str(t or "")))

        # 语义相近的旧事实：用新的替换（覆盖过期/矛盾的说法）
        vec = self.emb.get(text_key(text)) or []
        for fact in self.l3:
            old_text = fact.get("text", "")
            if not old_text:
                continue
            same_text = text in old_text or old_text in text
            sim = 0.0
            if vec:
                old_vec = self.emb.get(text_key(old_text))
                if old_vec:
                    sim = cosine(vec, old_vec)
            else:
                sim = jaccard(bigrams(text), bigrams(old_text))
            if (same_text or sim >= 0.72) and _nums(text) == _nums(old_text):
                fact["text"] = text
                fact["last_seen"] = now
                fact["hits"] = int(fact.get("hits", 1)) + 1
                merged_tags = list(dict.fromkeys((fact.get("tags") or []) + (tags or [])))[:6]
                fact["tags"] = merged_tags
                return
        _src = source or ("owner" if pinned else "inferred")
        fact = {"text": text, "created": now, "last_seen": now, "hits": 1,
                "tags": tags or [], "emotion": emotion or {},
                "source": _src, "confidence": 1.0 if _src == "owner" else 0.65}
        if pinned:
            fact["pinned"] = True
        if valid_until:
            fact["valid_until"] = float(valid_until)
        self.l3.append(fact)

    def _prune_l2(self) -> None:
        now = time.time()
        alive = [n for n in self.l2 if n.get("expires_at", 0) > now][-200:]
        # 合并近重复摘要，避免同一句话反复出现
        merged: list[dict] = []
        for note in alive:
            text = str(note.get("summary") or "")
            hit = None
            for kept in merged:
                if jaccard(bigrams(text), bigrams(str(kept.get("summary") or ""))) >= 0.88:
                    hit = kept
                    break
            if hit is None:
                merged.append(note)
            else:
                hit["score"] = max(int(hit.get("score", 0)), int(note.get("score", 0)))
                hit["hits"] = int(hit.get("hits", 1)) + 1
                hit["ts"] = max(float(hit.get("ts", 0)), float(note.get("ts", 0)))
                hit["expires_at"] = max(float(hit.get("expires_at", 0)), float(note.get("expires_at", 0)))
        self.l2 = merged
        self._save(self.l2_path, self.l2)

    # ---------- cleanup ----------
    def _strength_of(self, ts: float, hits: int = 1) -> float:
        """艾宾浩斯式强度：越久越弱，被想起越多越稳。"""
        now = time.time()
        age_days = max(0.0, (now - float(ts or now)) / DAY)
        stability = 1.0 + 2.0 * min(6, int(hits or 0))
        return min(1.0, math.exp(-age_days / stability) + 0.06 * min(5, int(hits or 0)))

    def cleanup(self) -> str:
        """真正的遗忘：L2 强度过低即删，L3 弱化后下沉、沉太久真删，L4 永久。"""
        now = time.time()

        # L2：过期 / 强度过低 → 真的忘掉
        before_l2 = len(self.l2)
        self._prune_l2()
        expired = before_l2 - len(self.l2)
        floor_l2 = float(self.cfg("forget_l2_below", 0.08) or 0.08)
        kept = []
        weak = 0
        gone_notes: list[str] = []
        for note in self.l2:
            s = self._strength_of(float(note.get("ts") or now), int(note.get("hits", 1)))
            note["strength"] = round(s, 3)
            # 被反复提起的近期记忆多留几天（最长 3 天）
            try:
                if int(note.get("hits") or 1) >= 2:
                    left = float(note.get("expires_at") or 0) - now
                    cap = now + 3 * DAY
                    if left < 12 * 3600:
                        note["expires_at"] = min(cap, now + 48 * 3600)
            except Exception:
                pass
            if s < floor_l2 and not note.get("pinned"):
                weak += 1
                gone_notes.append(str(note.get("summary") or "")[:80])
                continue
            kept.append(note)
        if weak:
            self.l2 = kept
            self._save(self.l2_path, self.l2)

        # L3：弱化 → 下沉；下沉满一年 → 真删
        floor_l3 = float(self.cfg("forget_l3_below", 0.12) or 0.12)
        park_days = self._int("fact_park_days", 180, 30)
        alive = []
        parked = 0
        deleted = 0
        gone_facts: list[str] = []
        for fact in self.l3:
            s = self._strength_of(float(fact.get("last_seen") or fact.get("created") or now), int(fact.get("hits", 1)))
            fact["strength"] = round(s, 3)
            if fact.get("pinned") and fact.get("parked"):
                fact.pop("parked", None)
                fact.pop("parked_at", None)
            if fact.get("parked"):
                parked_at = float(fact.get("parked_at") or now)
                if (now - parked_at) / DAY > 365:
                    deleted += 1
                    gone_facts.append(str(fact.get("text") or "")[:80])
                    continue
                alive.append(fact)
                continue
            age_days = (now - float(fact.get("last_seen") or fact.get("created") or now)) / DAY
            _w = self._fact_weight(fact, now)
            if (_w < 1.0 or age_days > park_days) and not fact.get("pinned"):
                if parked >= 3:
                    continue
                fact["parked"] = True
                try:
                    ap = self.data_dir / "facts_archive.json"
                    arr = json.loads(ap.read_text(encoding="utf-8")) if ap.exists() else []
                    arr.append({"text": fact.get("text"), "tags": fact.get("tags") or [],
                                "weight": _w, "archived_at": time.strftime("%Y-%m-%d %H:%M")})
                    ap.write_text(json.dumps(arr[-500:], ensure_ascii=False, indent=2), encoding="utf-8")
                except Exception:
                    pass
                fact["parked_at"] = now
                parked += 1
            alive.append(fact)
        self.l3 = alive
        self._save(self.l3_path, self.l3)

        # 把这次真正忘掉的东西写进日志（删了就再也找不回来，除了 L4 原文）
        if gone_notes or gone_facts:
            try:
                with (self.data_dir / "forget_log.jsonl").open("a", encoding="utf-8") as fh:
                    for text in gone_notes:
                        fh.write(json.dumps({"t": now, "kind": "L2", "text": text}, ensure_ascii=False) + "\n")
                    for text in gone_facts:
                        fh.write(json.dumps({"t": now, "kind": "L3", "text": text}, ensure_ascii=False) + "\n")
            except Exception:
                pass

        pending = self.state.get("pending") or []
        if len(pending) > 300:
            self.state["pending"] = pending[-300:]
            self._save_state()

        active = [f for f in self.l3 if not f.get("parked")]
        if expired or weak or parked or deleted:
            logger.info(
                f"[记忆] 遗忘：L2 过期 {expired} / 弱化删除 {weak} 条；"
                f"L3 下沉 {parked} 条 / 彻底忘记 {deleted} 条（仍记得 {len(active)}）"
            )
            if time.time() - float(self.state.get("last_backup") or 0) > DAY:
                self._daily_backup()
                self.state["last_backup"] = time.time()
                self._save_state()
            if time.time() - float(self.state.get("last_index") or 0) > DAY:
                self._build_archive_index()
                self.state["last_index"] = time.time()
            self._promote_hot_notes()
            self._expire_facts()
            self._dedupe_facts()
            self._archive_capacity()
        return f"L2 -{expired + weak}, L3 下沉 {parked}, 忘记 {deleted}"
    # ---------- graph / association ----------
    TRIVIAL = ("晚安", "早", "在吗", "好的", "嗯", "哦", "哈哈", "收到", "1", "6", "？", "?")

    def _diary_nodes(self) -> list[dict]:
        """日记属于 L4 档案层，直接读内存里的 diary_entries。"""
        out: list[dict] = []
        try:
            for i, entry in enumerate((self.diary_entries or [])[-6:]):
                if not isinstance(entry, dict):
                    continue
                outline = [str(x) for x in (entry.get("outline") or [])][:6]
                date = entry.get("date") or ""
                title = entry.get("title") or ""
                tag = "合集" if entry.get("digest") else "日记"
                text = f"{date} {tag}《{title}》" + ("：" + " / ".join(outline[:3]) if outline else "")
                out.append({
                    "id": f"d{i}", "kind": "archive", "sub": "diary",
                    "text": text[:240], "tags": ["L4", tag],
                    "ts": float(entry.get("ts") or 0),
                    "date": date, "count": len(outline),
                })
        except Exception:
            pass
        return out

    def _day_emotion(self, day: str) -> dict:
        """统计某天 L2 摘要里的主导情绪。"""
        try:
            import datetime as _dt

            counter: dict[str, int] = {}
            for note in self.l2:
                ts = float(note.get("ts") or 0)
                if not ts:
                    continue
                if _dt.datetime.fromtimestamp(ts).strftime("%Y-%m-%d") != day:
                    continue
                label = str((note.get("emotion") or {}).get("label") or "").strip()
                if label:
                    counter[label] = counter.get(label, 0) + 1
            if not counter:
                return {}
            label = max(counter, key=counter.get)
            return {"label": label, "n": counter[label]}
        except Exception:
            return {}

    def _daily_backup(self, keep_days: int = 7) -> None:
        """把记忆数据每天复制一份，保留最近 7 天。"""
        try:
            today = time.strftime("%Y%m%d")
            root = self.data_dir / "backups" / today
            if root.exists():
                return
            root.mkdir(parents=True, exist_ok=True)
            import shutil
            for name in ("l1_meta.json", "l2_notes.json", "l3_facts.json", "diary_state.json",
                         "graph.json", "state.json", "archive_index.json"):
                src = self.data_dir / name
                if src.exists():
                    shutil.copy2(src, root / name)
            arc = self.data_dir / "archive"
            if arc.exists():
                shutil.copytree(arc, root / "archive", dirs_exist_ok=True)
            base = self.data_dir / "backups"
            olds = sorted([p for p in base.iterdir() if p.is_dir()], reverse=True)
            for old in olds[keep_days:]:
                shutil.rmtree(old, ignore_errors=True)
            logger.info(f"[Wave] 备份完成：{today}，保留 {min(len(olds), keep_days)} 份")
        except Exception as exc:
            logger.warning(f"[Wave] 备份失败: {exc}")

    def _build_archive_index(self) -> dict:
        """给 L4 每日存档建一份索引：每天的主题标签 + 一句代表内容。"""
        try:
            index = {}
            files = sorted(self.archive_dir.glob("*.jsonl"))[-400:]
            for path in files:
                day = path.stem
                tags, sample, count = [], "", 0
                try:
                    with path.open(encoding="utf-8") as fh:
                        for i, line in enumerate(fh):
                            count += 1
                            if i > 200:
                                break
                            try:
                                rec = json.loads(line)
                            except Exception:
                                continue
                            text = str(rec.get("text") or "")
                            if not sample and 8 <= len(text) <= 60:
                                sample = text[:60]
                except Exception:
                    continue
                # 用当天的 L2 摘要标签聚合主题
                try:
                    import datetime as _dt
                    counter = {}
                    for note in self.l2:
                        ts = float(note.get("ts") or 0)
                        if not ts:
                            continue
                        if _dt.datetime.fromtimestamp(ts).strftime("%Y-%m-%d") != day:
                            continue
                        for t in (note.get("tags") or []):
                            counter[t] = counter.get(t, 0) + 1
                    tags = [t for t, _ in sorted(counter.items(), key=lambda kv: -kv[1])[:4]]
                except Exception:
                    pass
                index[day] = {"tags": tags, "sample": sample, "count": count}
            self._save(self.data_dir / "archive_index.json", index)
            logger.debug("[Wave] archive index: %d days" % len(index))
            return index
        except Exception as exc:
            logger.debug(f"[Wave] archive index failed: {exc}")
            return {}

    def _archive_nodes(self) -> list[dict]:
        """L4：按天聚合的永久存档（原话），样本取当天信息量最大的一句。"""
        out: list[dict] = []
        try:
            for i, path in enumerate(sorted(self.archive_dir.glob("*.jsonl"), reverse=True)[:14]):
                lines = path.read_text(encoding="utf-8").splitlines()
                if not lines:
                    continue
                best = ""
                last_ts = 0.0
                for line in lines:
                    try:
                        rec = json.loads(line)
                    except Exception:
                        continue
                    ts = float(rec.get("ts") or 0)
                    if ts > last_ts:
                        last_ts = ts
                    text = str(rec.get("text") or "").strip()
                    if rec.get("role") != "user" or not (6 <= len(text) <= 60):
                        continue
                    if any(b in text for b in ("[系统任务", "[图片]", "[Image", "system_reminder", "主动消息", "Current datetime", "User ID")):
                        continue
                    if any(text.startswith(t) for t in self.TRIVIAL):
                        continue
                    if len(text) > len(best):
                        best = text
                emo = self._day_emotion(path.stem)
                day_tags = []
                try:
                    import datetime as _dt
                    counter = {}
                    for _note in self.l2:
                        _ts = float(_note.get("ts") or 0)
                        if not _ts:
                            continue
                        if _dt.datetime.fromtimestamp(_ts).strftime("%Y-%m-%d") != path.stem:
                            continue
                        for _t in (_note.get("tags") or []):
                            counter[_t] = counter.get(_t, 0) + 1
                    day_tags = [t for t, _ in sorted(counter.items(), key=lambda kv: -kv[1])[:4]]
                except Exception:
                    day_tags = []
                parts = [f"{path.stem} · {len(lines)} 句原话"]
                if best:
                    parts.append(best[:34])
                if emo:
                    parts.append(f"情绪偏「{emo['label']}」")
                out.append({
                    "id": f"a{i}",
                    "kind": "archive",
                    "sub": "day",
                    "text": " · ".join(parts),
                    "tags": ["L4"] + day_tags,
                    "ts": last_ts,
                    "count": len(lines),
                    "emotion": {"label": emo.get("label"), "v": 0} if emo else {},
                })
        except Exception:
            pass
        return out
    async def enrich_recent(self, added: list[dict]) -> None:
        """L1：每条对话都用便宜模型总结 + 打 5 个标签，值得记的自动记住。"""
        try:
            batch = [m for m in added if (m.get("text") or "").strip()][:15]
            if not batch:
                return
            texts = ["".join(ch for ch in str(m.get("text") or "") if not ch.isspace()) for m in batch]
            if all(len(t) <= 6 or t in SHORT_ACK for t in texts):
                # 全是寒暄/短句：不调模型，本地补个基础记录
                now0 = time.time()
                for m, t in zip(batch, texts):
                    if len(t) < 2:
                        continue
                    emo = self._emotion_of(t)
                    self.recent_meta.append({
                        "text": t[:60], "summary": t[:18], "tags": ["寒暄"] + ([emo["label"]] if emo.get("label") else []),
                        "ts": now0, "role": m.get("role") or "user", "keep": False, "emotion": emo,
                    })
                self.recent_meta = self.recent_meta[-80:]
                self._save(self.l1_meta_path, self.recent_meta)
                logger.debug(f"[Wave] skip llm for {len(batch)} trivial msgs")
                return
            items = [
                {"i": i, "role": "主人" if m.get("role") == "user" else "我", "t": m["text"][:120]}
                for i, m in enumerate(batch)
            ]
            prompt = (
                "给下面每条对话做三件事，只输出 JSON：\n"
                '{"items":[{"i":序号,"s":"不超过 18 字的摘要","tags":["标签"],"keep":false}]}\n'
                "tags：必须给 5 个，优先从词表选，词表覆盖不到可自造 1-2 个 2-4 字短标签：" + "、".join(PROMPT_TAGS) + "\n"
                "keep：只有当这条里主人明确要求记住、或包含长期有效的偏好/约定/身份时才为 true，其余 false。\n"
                "不要输出 JSON 以外的内容。\n\n对话：\n" + json.dumps(items, ensure_ascii=False)
            )
            raw = await self._llm(prompt, "你是精确的信息整理器，只输出 JSON。")
            payload = self._extract_json(raw)
            now = time.time()
            kept = 0
            for entry in payload.get("items") or []:
                try:
                    idx = int(entry.get("i"))
                except Exception:
                    continue
                if not (0 <= idx < len(batch)):
                    continue
                src = batch[idx]
                tags = [str(t).strip()[:8] for t in (entry.get("tags") or []) if str(t).strip()][:7]
                summary = str(entry.get("s") or src.get("text", ""))[:40]
                rec = {
                    "ts": float(src.get("ts") or now),
                    "role": src.get("role"),
                    "text": (src.get("text") or "")[:160],
                    "summary": summary,
                    "tags": tags,
                    "remember": bool(entry.get("keep")),
                }
                self.recent_meta.append(rec)
                if rec["remember"] and rec["role"] == "user":
                    self._add_fact(summary, now, tags)
                    kept += 1
                    logger.info(f"[记忆] L1 自动记住：{summary[:24]}")
            self.recent_meta = self.recent_meta[-80:]
            try:
                self.l1_meta_path.write_text(
                    json.dumps(self.recent_meta, ensure_ascii=False, indent=2), encoding="utf-8"
                )
            except Exception:
                pass
            self._save(self.l3_path, self.l3)
            logger.debug(f"[记忆] L1 整理 {len(batch)} 条（记住 {kept}，累计 {len(self.recent_meta)}）")
        except Exception as exc:
            logger.warning(f"[记忆] L1 整理失败: {exc}")

    def _read_conversation_sync(self) -> list[str]:
        """读取私聊会话，拼成 主人:/我: 的对话行，供日记使用。"""
        import sqlite3

        db = self._db_path()
        if not db.exists():
            return []
        prefix = str(self.cfg("session_prefix", "snowluma:FriendMessage:"))
        con = sqlite3.connect(f"file:{db}?mode=ro", uri=True, timeout=10)
        try:
            rows = con.execute(
                "select content from conversations where user_id like ? order by updated_at asc",
                (prefix + "%",),
            ).fetchall()
        finally:
            con.close()
        lines: list[str] = []
        for (raw,) in rows:
            try:
                messages = json.loads(raw) if isinstance(raw, str) else (raw or [])
            except Exception:
                continue
            if not isinstance(messages, list):
                continue
            for item in messages:
                role, text = self._parse_message(item)
                if not text:
                    continue
                who = "主人" if role == "user" else "我"
                lines.append(f"{who}: {text}")
        return lines

    def _recent_nodes(self, limit: int | None = None) -> list[dict]:
        """L1: recent messages with DS summary and 5 tags."""
        limit = limit or self._int("l1_window", 20, 2)
        out: list[dict] = []
        try:
            recent = [m for m in (self.recent_meta or []) if (m.get("text") or "").strip()][-limit:]
            if recent:
                for i, rec in enumerate(recent):
                    out.append({
                        "id": f"r{i}",
                        "kind": "recent",
                        "text": str(rec.get("summary") or rec.get("text") or "")[:120],
                        "tags": (rec.get("tags") or [])[:5] or ["L1"],
                        "ts": float(rec.get("ts") or 0),
                        "role": rec.get("role"),
                        "raw": str(rec.get("text") or "")[:160],
                    })
                return out
            files = sorted(self.archive_dir.glob("*.jsonl"), reverse=True)
            lines: list[str] = []
            for path in files:
                lines = path.read_text(encoding="utf-8").splitlines()[-limit:]
                if lines:
                    break
            for i, line in enumerate(lines):
                try:
                    rec = json.loads(line)
                except Exception:
                    continue
                text = str(rec.get("text") or "").strip()
                if not text:
                    continue
                out.append({
                    "id": f"r{i}",
                    "kind": "recent",
                    "text": text[:160],
                    "tags": ["L1"],
                    "ts": float(rec.get("ts") or 0),
                    "role": rec.get("role"),
                })
        except Exception:
            pass
        return out

    def _schedule_graph_rebuild(self, min_gap: float = 15.0) -> None:
        """重建联想图很贵（O(n^2)），改成后台节流执行，不挡住回复。"""
        try:
            now = time.time()
            if now - float(getattr(self, "_last_graph_build", 0) or 0) < min_gap:
                return
            if getattr(self, "_graph_task", None) and not self._graph_task.done():
                return
            self._last_graph_build = now

            async def _run():
                try:
                    await asyncio.to_thread(self.build_graph)
                except Exception as exc:
                    logger.debug(f"[Wave] async graph build failed: {exc}")

            self._graph_task = asyncio.create_task(_run())
        except Exception:
            pass

    def build_graph(self) -> dict:
        now = time.time()
        nodes: list[dict] = []
        for i, fact in enumerate([f for f in self.l3 if not f.get("parked")]):
            nodes.append({
                "id": f"f{i}", "kind": "fact", "text": fact.get("text", ""),
                "tags": fact.get("tags") or [],
                "ts": float(fact.get("last_seen") or fact.get("created") or now),
                "hits": int(fact.get("hits", 1)), "emotion": fact.get("emotion") or {},
            })
        for i, note in enumerate([n for n in self.l2 if n.get("expires_at", 0) > now][-120:]):
            nodes.append({
                "id": f"n{i}", "kind": "note", "text": note.get("summary", ""),
                "tags": note.get("tags") or [],
                "ts": float(note.get("ts") or now),
                "score": int(note.get("score", 0)),
                "hits": int(note.get("hits", 1)), "emotion": note.get("emotion") or {},
                "exp": float(note.get("expires_at") or 0),
            })
        nodes.extend(self._diary_nodes())
        nodes.extend(self._recent_nodes())
        nodes.extend(self._archive_nodes())

        # 去掉重复与无信息量的节点（大量「寒暄」会把图连成一团）
        seen: set[str] = set()
        uniq: list[dict] = []
        for node in nodes:
            text = str(node.get("text") or "").strip()
            if len(text) < 4 or text in ("寒暄", "表情包", "夸他", "撒娇"):
                continue
            if text in seen:
                continue
            seen.add(text)
            uniq.append(node)
        nodes = uniq

        # 人的记忆深浅：新近 + 被想起次数 → 强度，强度决定在星图里的位置与大小
        for node in nodes:
            ts = float(node.get("ts") or now)
            hits = int(node.get("hits") or node.get("hits", 0) or 0)
            age_days = max(0.0, (now - ts) / DAY)
            stability = 1.0 + 2.0 * min(6, hits)  # 每被想起一次，记忆更稳
            node["strength"] = round(min(1.0, math.exp(-age_days / stability) + 0.06 * min(5, hits)), 3)
            if node["kind"] == "note":
                node["exp"] = float(node.get("exp") or 0)

        tag_index = {}
        for _node in nodes:
            for _tag in (_node.get("tags") or []):
                tag_index.setdefault(str(_tag), []).append(_node["id"])
        self._tag_index = tag_index
        self._tag_df = {k: len(v) for k, v in tag_index.items()}
        self._link_weight = {}
        logger.debug("[Wave] tag index: %d tags" % len(tag_index))

        vecs = {n["id"]: bigrams(n["text"]) for n in nodes}
        tagsets = {n["id"]: set(n["tags"]) for n in nodes}
        links: list[dict] = []
        threshold = float(self.cfg("assoc_similarity", 0.38) or 0.38)
        degree: dict[str, int] = {}
        max_degree = self._int("graph_max_degree", 7, 1)
        for i in range(len(nodes)):
            a = nodes[i]
            for j in range(i + 1, len(nodes)):
                b = nodes[j]
                w = 0.0
                shared = tagsets[a["id"]] & tagsets[b["id"]]
                if shared:
                    _df = getattr(self, "_tag_df", {}) or {}
                    _idf = sum(1.0 + math.log(1.0 + 60.0 / max(1, _df.get(t, 1))) for t in shared) / max(1, len(shared))
                    w += 0.5 + 0.45 * min(3, len(shared)) * (0.6 + _idf * 0.4)
                sim = jaccard(vecs[a["id"]], vecs[b["id"]])
                if sim >= threshold:
                    w += sim
                if w >= 0.6:
                    links.append({"a": a["id"], "b": b["id"], "w": round(min(1.6, w), 2)})
        links.sort(key=lambda x: -x["w"])
        kept: list[dict] = []
        for link in links:
            if degree.get(link["a"], 0) >= max_degree or degree.get(link["b"], 0) >= max_degree:
                continue
            degree[link["a"]] = degree.get(link["a"], 0) + 1
            degree[link["b"]] = degree.get(link["b"], 0) + 1
            kept.append(link)
            if len(kept) >= self._int("graph_max_links", 260, 20):
                break

        # 跨层联想：L3 事实 ↔ L2 摘要 ↔ L4 档案，标签有交集就连一条“跨层边”
        cross = 0
        facts_nodes = [n for n in nodes if n["kind"] == "fact"]
        note_nodes = [n for n in nodes if n["kind"] == "note"]
        arch_nodes = [n for n in nodes if n["kind"] == "archive"]
        linked_pairs = set()
        for link in links:
            linked_pairs.add((link["a"], link["b"]))
            linked_pairs.add((link["b"], link["a"]))
        for fact in facts_nodes:
            best, best_w = None, 0
            for note in note_nodes:
                shared = set(fact.get("tags") or []) & set(note.get("tags") or [])
                w = len(shared)
                if w > best_w:
                    best, best_w = note, w
            if best and best_w >= 1 and (fact["id"], best["id"]) not in linked_pairs:
                kept.append({"a": fact["id"], "b": best["id"], "w": round(0.8 + 0.15 * best_w, 2), "kind": "cross"})
                cross += 1
        for note in note_nodes:
            best, best_w = None, 0
            for day in arch_nodes:
                shared = set(note.get("tags") or []) & set(day.get("tags") or [])
                w = len(shared)
                if w > best_w:
                    best, best_w = day, w
            if best and best_w >= 1 and (note["id"], best["id"]) not in linked_pairs:
                kept.append({"a": note["id"], "b": best["id"], "w": 0.9, "kind": "cross"})
                cross += 1
        if cross:
            logger.info(f"[记忆] 跨层联想边 +{cross}")
        # 时间衰减：越久没被提起的关联越淡
        decay_days = float(self.cfg("decay_days", 45) or 45)
        tsmap = {n["id"]: float(n.get("ts") or 0) for n in nodes}
        for link in kept:
            t = max(tsmap.get(link["a"], 0), tsmap.get(link["b"], 0)) or now
            age_days = max(0.0, (now - t) / DAY)
            link["w"] = round(max(0.05, link["w"] * math.exp(-age_days / decay_days)), 3)
            link["kind"] = link.get("kind") or "tag"
        kept = [l for l in kept if l["w"] >= 0.08]
        links = kept
        # L1 最近十句串成一条时间链，星图里能看到「刚刚聊的」
        # L4 档案链：按天串起来，日记挂到同一天
        arch_days = [n for n in nodes if n["kind"] == "archive" and n.get("sub") == "day"]
        for i in range(len(arch_days) - 1):
            links.append({"a": arch_days[i]["id"], "b": arch_days[i + 1]["id"], "w": 1.1, "kind": "archive"})
        for node in nodes:
            if node["kind"] == "archive" and node.get("sub") == "diary" and node.get("date"):
                for day in arch_days:
                    if str(day.get("text", "")).startswith(str(node["date"])):
                        links.append({"a": day["id"], "b": node["id"], "w": 1.7, "kind": "archive"})
        recent_ids = [n["id"] for n in nodes if n["kind"] == "recent"]
        for i in range(len(recent_ids) - 1):
            links.append({"a": recent_ids[i], "b": recent_ids[i + 1], "w": 1.4, "kind": "recent"})
        graph = {"nodes": nodes, "links": links, "built_at": now}
        # 邻接表：给检索用
        adj: dict[str, list[str]] = {}
        for link in links:
            adj.setdefault(link["a"], []).append(link["b"])
            adj.setdefault(link["b"], []).append(link["a"])
        self._graph = graph
        self._adj = adj
        self._save(self.graph_path, graph)
        logger.debug(f"[记忆] 联想图构建：{len(nodes)} 点 / {len(links)} 边")
        return graph

    # ---------- embeddings ----------
    async def _embed_provider(self):
        pid = str(self.cfg("embedding_provider_id", "bailian_embedding") or "")
        if not pid:
            return None
        try:
            return await self.context.provider_manager.get_provider_by_id(pid)
        except Exception as exc:
            logger.warning(f"[记忆] 取向量模型失败: {exc}")
            return None

    async def embed_texts(self, texts: list[str]) -> None:
        todo = [t for t in dict.fromkeys(str(x or "").strip() for x in texts) if t and text_key(t) not in self.emb]
        if not todo:
            return
        prov = await self._embed_provider()
        if not prov:
            try:
                keys = list((self.context.provider_manager.inst_map or {}).keys())
            except Exception:
                keys = []
            logger.debug(f"[记忆] 向量模型还没就绪（稍后重试），当前可用: {keys}")
            return
        batch = max(1, min(10, self._int("embed_batch", 8, 1)))
        added = 0
        for i in range(0, len(todo), batch):
            chunk = todo[i:i + batch]
            try:
                vecs = await prov.get_embeddings(chunk)
            except Exception as exc:
                logger.warning(f"[记忆] 向量化失败: {exc}")
                break
            for text, vec in zip(chunk, vecs or []):
                if vec:
                    self.emb[text_key(text)] = list(vec)
                    added += 1
        if added:
            if len(self.emb) > 900:
                keys = list(self.emb.keys())[-900:]
                self.emb = {k: self.emb[k] for k in keys}
            self._save(self.emb_path, self.emb)
            logger.debug(f"[记忆] 向量缓存 {len(self.emb)} 条（新增 {added}）")

    async def semantic_links(self) -> int:
        nodes = self._graph.get("nodes") or []
        if not nodes:
            return 0
        await self.embed_texts([n.get("text") for n in nodes])
        threshold = float(self.cfg("embed_min_sim", 0.74) or 0.74)
        links = list(self._graph.get("links") or [])
        existing = set()
        degree: dict[str, int] = {}
        for link in links:
            existing.add((link.get("a"), link.get("b")))
            existing.add((link.get("b"), link.get("a")))
            degree[link.get("a")] = degree.get(link.get("a"), 0) + 1
            degree[link.get("b")] = degree.get(link.get("b"), 0) + 1
        max_degree = self._int("graph_max_degree", 7, 1) + 3
        added: list[dict] = []
        for i in range(len(nodes)):
            a = nodes[i]
            va = self.emb.get(text_key(a.get("text", "")))
            if not va:
                continue
            for j in range(i + 1, len(nodes)):
                b = nodes[j]
                if (a["id"], b["id"]) in existing:
                    continue
                if degree.get(a["id"], 0) >= max_degree:
                    break
                if degree.get(b["id"], 0) >= max_degree:
                    continue
                vb = self.emb.get(text_key(b.get("text", "")))
                if not vb:
                    continue
                sim = cosine(va, vb)
                if sim < threshold:
                    continue
                added.append({"a": a["id"], "b": b["id"], "w": round(1.0 + sim, 2), "kind": "semantic"})
                degree[a["id"]] = degree.get(a["id"], 0) + 1
                degree[b["id"]] = degree.get(b["id"], 0) + 1
        if added:
            self._graph["links"] = links + added
            self._graph["built_at"] = time.time()
            self._save(self.graph_path, self._graph)
            logger.debug(f"[记忆] 语义边 +{len(added)}（阈值 {threshold}）")
        return len(added)

    async def _deferred_semantic(self):
        """等 provider 加载完再补语义边（插件初始化早于模型加载）。"""
        for attempt in range(5):
            await asyncio.sleep(20 + attempt * 15)
            try:
                added = await self.semantic_links()
                if added or self.emb:
                    try:
                        _idx = getattr(self, "_tag_index", {}) or {}
                        _top = sorted(_idx.items(), key=lambda kv: -len(kv[1]))[:1]
                        _q = _top[0][0] if _top else "主人"
                        logger.info("[Wave] self-test query: %s" % _q)
                        probe = await self.retrieve_async(_q)
                        if probe:
                            parts = ["%s#%s:%s" % (h["kind"], "/".join((h.get("tags") or [])[:2]), h["text"][:14]) for h in probe[:4]]
                            chain = self.reason_chain(probe)
                            logger.info("[Wave] recall self-test: " + " | ".join(parts) + (("  chain: " + chain) if chain else ""))
                        else:
                            logger.info("[Wave] recall self-test: empty")
                    except Exception as exc:
                        logger.warning(f"[Wave] self-test failed: {exc}")
                    return
            except Exception as exc:
                logger.warning(f"[记忆] 语义边重试失败: {exc}")

    def reason_chain(self, hits: list[dict]) -> str:
        """两跳联想链：只有靠「稀有标签」或强语义边连起来的才走，避免乱跳。"""
        try:
            if not hits:
                return ""
            nodes = {n["id"]: n for n in self._graph.get("nodes") or []}
            start = hits[0].get("id")
            if not start or start not in nodes:
                return ""
            total = max(1, len(nodes))
            df = getattr(self, "_tag_df", {}) or {}
            # 常见标签（主人/自己 这种）不配当桥梁，只有稀有标签才算"想到了一起"
            common_floor = max(5, int(total * 0.20))

            META = {"日程", "记忆", "日记", "提醒", "计划", "目标", "L4", "任务", "项目", "deadline"}

            def bridge_tags(a: dict, b: dict) -> list[str]:
                shared = set(a.get("tags") or []) & set(b.get("tags") or [])
                return [t for t in shared if df.get(t, 0) <= common_floor and t not in META]

            weight = {}
            for link in self._graph.get("links") or []:
                weight[(link.get("a"), link.get("b"))] = float(link.get("w") or 0)
                weight[(link.get("b"), link.get("a"))] = float(link.get("w") or 0)

            cur = start
            chain = [str(nodes[start].get("text") or "")[:14]]
            seen = {start}
            for _ in range(2):
                best, best_score, best_bridge = None, 0.0, []
                for nb in (self._adj.get(cur) or []):
                    if nb in seen or nb not in nodes:
                        continue
                    bridges = bridge_tags(nodes[cur], nodes[nb])
                    w = weight.get((cur, nb), 0.0)
                    if not bridges and w < 1.35:
                        continue          # 既没有稀有标签，语义边也不够强 → 不值得走
                    score = w + (0.6 if bridges else 0) + (0.4 if bridges and nodes[nb].get("kind") != nodes[cur].get("kind") else 0)
                    if score > best_score:
                        best, best_score, best_bridge = nb, score, bridges
                if not best or best_score < 1.0:
                    break
                seen.add(best)
                chain.append(str(nodes[best].get("text") or "")[:14] + (" ⟨" + "/".join(best_bridge[:2]) + "⟩" if best_bridge else ""))
                cur = best
            if len(chain) < 2:
                return ""
            return " → ".join(c for c in chain if c)
        except Exception:
            return ""

    def inference(self, hits: list[dict]) -> str:
        """基于命中的记忆做一句推断：共享标签 → 关联结论。"""
        try:
            if len(hits) < 2:
                return ""
            tags_a = set(hits[0].get("tags") or [])
            for other in hits[1:]:
                shared = tags_a & set(other.get("tags") or [])
                if shared:
                    tag = sorted(shared)[0]
                    return f"关于「{tag}」：{hits[0]['text'][:16]}，而{other['text'][:16]}"
            return ""
        except Exception:
            return ""

    EMOTION_WORDS = {
        "开心": 0.7, "高兴": 0.7, "喜欢": 0.6, "爱": 0.7, "谢谢": 0.4, "好耶": 0.8,
        "笑": 0.4, "哈哈": 0.4, "期待": 0.5, "舒服": 0.5, "棒": 0.6, "满足": 0.5,
        "感动": 0.6, "安心": 0.4, "放松": 0.5, "得意": 0.5, "好吃": 0.5,
        "顺利": 0.4, "赢了": 0.7, "被夸": 0.6, "不错": 0.45, "可爱": 0.5, "幸福": 0.7,
        "难过": -0.7, "委屈": -0.8, "累": -0.6, "烦": -0.6, "生气": -0.8, "气死": -0.9,
        "焦虑": -0.6, "怕": -0.6, "担心": -0.5, "哭": -0.8, "孤独": -0.7, "想哭": -0.8,
        "崩溃": -0.9, "压力": -0.6, "失眠": -0.6, "疼": -0.6, "生病": -0.5,
        "失望": -0.7, "后悔": -0.6, "被骂": -0.7, "吵架": -0.7,
        "好累": -0.7, "没意思": -0.5, "受不了": -0.8, "难受": -0.7, "emo": -0.6, "讨厌": -0.6,
    }

    NEUTRAL_WORDS = ("还行", "还好", "一般", "凑合", "随便", "都可以", "无所谓", "就那样")
    REPORT_WORDS = ("他说", "她说", "听说", "别人说", "有人说", "他们都", "她们都", "他老是", "她老是")
    EMOJI_SENTI = {"😭": -0.8, "😢": -0.7, "🥲": -0.3, "😡": -0.8, "😞": -0.6,
                   "😄": 0.7, "😊": 0.5, "🥰": 0.7, "😂": 0.45, "🙃": -0.1,
                   "唉": -0.3, "哎": -0.2}
    STRONG_TAIL = ("死了", "爆了", "极了", "坏了", "到爆", "得要命")
    WEAKEN_TAIL = ("一点", "一点点", "一下", "些许")
    CLAUSE_BOUND = "，,。！？!?；;…\n~～"

    # 含情绪字但不是那个意思的复合词
    EXCLUDE_COMPOUNDS = ("累积", "积累", "累计", "连累", "拖累", "麻烦", "烦请", "烦劳",
                         "开心果", "开心点", "爱心", "爱好", "热爱", "可爱", "怕生", "疼我")
    NEGATIONS = ("不", "没", "别", "无", "未", "毫不", "并没有", "没那么")
    NEG_TAILS = ("不起来", "不了", "不下去", "不动")
    INTENSIFY = {"超级": 1.5, "特别": 1.35, "非常": 1.35, "太": 1.3, "巨": 1.4,
                 "有点": 0.65, "稍微": 0.6, "一点点": 0.55, "略微": 0.55}

    def _mood_trend(self) -> dict:
        """近 7 天 L2 情绪的平均值（60 秒缓存，避免每轮重算）。"""
        try:
            _cached = getattr(self, "_mood_cache", None)
            if _cached and (time.time() - _cached[0]) < 60:
                return _cached[1]
            now = time.time()
            vals = []
            labels = {}
            for note in self.l2:
                ts = float(note.get("ts") or 0)
                if not ts or (now - ts) > 7 * DAY:
                    continue
                emo = note.get("emotion") or {}
                v = emo.get("v")
                if isinstance(v, (int, float)):
                    vals.append(float(v))
                label = str(emo.get("label") or "").strip()
                if label:
                    labels[label] = labels.get(label, 0) + 1
            if len(vals) < 3:
                return {}
            vals.sort()
            mid = len(vals) // 2
            avg = vals[mid] if len(vals) % 2 else (vals[mid - 1] + vals[mid]) / 2
            top = max(labels, key=labels.get) if labels else ""
            out = {"avg": round(avg, 2), "label": top, "n": len(vals)}
            self._mood_cache = (time.time(), out)
            return out
        except Exception:
            return {}

    def _clause_bounds(self, t: str, idx: int):
        start = 0
        for i in range(idx - 1, -1, -1):
            if t[i] in self.CLAUSE_BOUND:
                start = i + 1
                break
        end = len(t)
        for i in range(idx, len(t)):
            if t[i] in self.CLAUSE_BOUND:
                end = i
                break
        return start, end

    def _emotion_analysis(self, text: str) -> dict:
        """情绪分析 v3：分句作用域 + 转述降权 + 程度后缀 + emoji + 中性词。"""
        t = str(text or "")
        empty = {"v": 0.0, "label": "", "pos": 0, "neg": 0, "n": 0,
                 "neutral": False, "reported": False}
        if not t:
            return empty
        hits = []
        for word, base in self.EMOTION_WORDS.items():
            idx = t.find(word)
            while idx >= 0:
                nxt = idx + len(word)
                c_start, c_end = self._clause_bounds(t, idx)
                clause = t[c_start:c_end]
                span = t[max(c_start, idx - 2): min(len(t), nxt + 2)]
                if any(x != word and x in span for x in self.EXCLUDE_COMPOUNDS):
                    idx = t.find(word, nxt)
                    continue
                head = t[max(c_start, idx - 6): idx]
                near = head[-3:]
                flip = False
                if any(n in near for n in ("并没有", "没那么")):
                    flip = True
                elif any(n in near for n in ("不", "没", "别", "无", "未")):
                    flip = True
                if near.count("不") >= 2 or ("不是不" in head):
                    flip = False
                weaken = any(head.endswith(x) for x in ("不是很", "不太", "不怎么", "没那么"))
                tail = t[nxt: min(len(t), nxt + 3)]
                if tail.startswith(("不起来", "不了", "不下去", "不动")):
                    flip = not flip
                scale = 1.0
                for k, sc in self.INTENSIFY.items():
                    if k in head:
                        scale = sc
                        break
                if any(tail.startswith(x) for x in self.STRONG_TAIL):
                    scale *= 1.4
                elif any(tail.startswith(x) for x in self.WEAKEN_TAIL):
                    scale *= 0.7
                if weaken:
                    scale *= 0.55
                report = any(r in clause for r in self.REPORT_WORDS)
                val = base * scale * (-0.9 if flip else 1.0) * (0.45 if report else 1.0)
                hits.append((word, val, base, report))
                idx = t.find(word, nxt)

        for ch, v in self.EMOJI_SENTI.items():
            cnt = t.count(ch)
            if cnt:
                hits.append((ch, v * min(2, cnt), v, False))

        neutral_word = any(n in t for n in self.NEUTRAL_WORDS)
        reported = bool(hits) and all(h[3] for h in hits)
        if not hits:
            return {**empty, "neutral": neutral_word}

        pos = sum(v for _, v, _, _ in hits if v > 0)
        neg = -sum(v for _, v, _, _ in hits if v < 0)
        total = pos + neg
        v = ((pos - neg) / total) * min(1.0, total / 1.2) if total else 0.0
        if neutral_word and total < 1.0:
            v *= 0.5
        best = max(hits, key=lambda x: abs(x[1]))
        label = best[0] if (best[1] > 0) == (best[2] > 0) else ("正面" if best[1] > 0 else "负面")
        if pos > 0 and neg > 0:
            label = "复杂"
        return {"v": round(max(-1.0, min(1.0, v)), 2), "label": label,
                "pos": sum(1 for _, x, _, _ in hits if x > 0),
                "neg": sum(1 for _, x, _, _ in hits if x < 0),
                "n": len(hits), "neutral": neutral_word, "reported": reported}

    def _query_valence(self, text: str) -> float:
        """这句话的情绪倾向（-1 ~ 1）。"""
        return self._emotion_analysis(text)["v"]

    def _emotion_of(self, text: str) -> dict:
        """给一段话打情绪标签：{label, v}。"""
        try:
            r = self._emotion_analysis(text)
            if not r["label"] and abs(r["v"]) < 0.15:
                return {}
            return {"label": r["label"] or ("正面" if r["v"] > 0 else "负面"), "v": r["v"]}
        except Exception:
            return {}

    def reinforce_mentions(self, text: str) -> int:
        """用户这次又提到的事：被提起一次就更牢一点。"""
        try:
            qv = bigrams(text)
            if not qv:
                return 0
            now = time.time()
            bumped = 0
            for fact in self.l3:
                if fact.get("parked"):
                    continue
                if jaccard(qv, bigrams(fact.get("text", ""))) >= 0.18:
                    fact["hits"] = int(fact.get("hits", 1)) + 1
                    fact["last_seen"] = now
                    bumped += 1
            if bumped:
                self._save(self.l3_path, self.l3)
                logger.info(f"[记忆] 提到旧事，加深 {bumped} 条")
            return bumped
        except Exception:
            return 0

    @staticmethod
    def _short_when(self, ts) -> str:
        """把时间压成极短标签：今天/昨天/3天前/09-12。不占几个 token。"""
        try:
            t = float(ts or 0)
            if not t:
                return ""
            now = time.time()
            days = int((now - t) // DAY)
            if days <= 0:
                return "今天"
            if days == 1:
                return "昨天"
            if days < 30:
                return f"{days}天前"
            return time.strftime("%m-%d", time.localtime(t))
        except Exception:
            return ""

    def _now_line(self) -> str:
        """一行现在时间，让她对时间有概念。"""
        try:
            wd = "日一二三四五六"[int(time.strftime("%w"))]
            return f"【现在】{time.strftime('%Y-%m-%d')} 周{wd} {time.strftime('%H:%M')}\n"
        except Exception:
            return ""

    def _day_label(ts) -> str:
        """把时间戳说成 今天/昨天/N天前。"""
        try:
            import datetime as _dt

            d = _dt.datetime.fromtimestamp(float(ts or 0))
            delta = (_dt.date.today() - d.date()).days
            if delta <= 0:
                return "今天"
            if delta == 1:
                return "昨天"
            return f"{delta}天前"
        except Exception:
            return ""

    def _log_trace(self, query: str, hits: list[dict]) -> None:
        """记录一次联想：话题 → 想起来的记忆，星图的脑电波用它。"""
        try:
            rec = {
                "t": time.time(),
                "q": (query or "")[:60],
                "hits": [
                    {
                        "id": h.get("id"),
                        "kind": h.get("kind"),
                        "text": (h.get("text") or "")[:40],
                        "linked": bool(h.get("linked")),
                    }
                    for h in hits[:8]
                ],
            }
            with (self.data_dir / "assoc_log.jsonl").open("a", encoding="utf-8") as fh:
                fh.write(json.dumps(rec, ensure_ascii=False) + "\n")
        except Exception:
            pass

    async def _ensure_tag_vectors(self) -> dict:
        """把词表里的标签嵌入一次，用于「语义命中标签」。"""
        try:
            need = [t for t in TAXONOMY if ("tag::" + t) not in self.emb]
            if need:
                prov = await self._embed_provider()
                if prov is None:
                    return {}
                batch = max(1, min(10, self._int("embed_batch", 8, 1)))
                for i in range(0, len(need), batch):
                    chunk = need[i:i + batch]
                    try:
                        vecs = await prov.get_embeddings(chunk)
                    except Exception:
                        break
                    for name, vec in zip(chunk, vecs or []):
                        if vec:
                            self.emb["tag::" + name] = list(vec)
                if need:
                    self._save(self.emb_path, self.emb)
                    logger.info("[Wave] tag vectors ready: %d" % len([k for k in self.emb if k.startswith("tag::")]))
        except Exception as exc:
            logger.debug(f"[Wave] tag vector build failed: {exc}")
        return {t: self.emb.get("tag::" + t) for t in TAXONOMY if ("tag::" + t) in self.emb}

    def _semantic_tags(self, qvec: list[float]) -> list[str]:
        """用查询向量找语义上最接近的 3 个标签。"""
        try:
            if not qvec:
                return []
            scored = []
            for name in TAXONOMY:
                vec = self.emb.get("tag::" + name)
                if not vec:
                    continue
                sim = cosine(qvec, vec)
                if sim >= float(self.cfg("tag_semantic_min", 0.52) or 0.52):
                    scored.append((sim, name))
            scored.sort(reverse=True)
            return [n for _, n in scored[:3]]
        except Exception:
            return []

    async def retrieve_async(self, query: str, k: int | None = None) -> list[dict]:
        query = (query or "").strip()
        if not query:
            return []
        # 同一条查询 15 分钟内直接复用结果：省一次向量调用和一轮打分
        _cache = getattr(self, "_qcache", None)
        if _cache is None:
            _cache = self._qcache = {}
        _key = text_key(query)
        _now = time.time()
        _hit = _cache.get(_key)
        if _hit and (_now - _hit[0]) < 900:
            self._cache_hits = int(getattr(self, "_cache_hits", 0)) + 1
            return [dict(x) for x in _hit[1]]
        await self.embed_texts([query])
        qvec = self.emb.get(text_key(query))
        await self._ensure_tag_vectors()
        extra = self._semantic_tags(qvec) if qvec else []
        hits = self.retrieve(query, k=k, qvec=qvec, extra_tags=extra)
        try:
            topic = getattr(self, "_last_topic", "__none__")
            ts = self.state.setdefault("topic_stats", {})
            item = ts.get(topic) or {"n": 0, "hit": 0}
            item["n"] = int(item.get("n", 0)) + 1
            if hits:
                item["hit"] = int(item.get("hit", 0)) + 1
            ts[topic] = item
            if item["n"] >= 8:
                rate = item["hit"] / max(1, item["n"])
                sim = float((self.state.get("topic_sim") or {}).get(topic) or 0.55)
                if rate < 0.4:
                    sim = round(max(0.45, sim - 0.03), 2)   # 这个话老想不起来 → 放宽
                elif rate > 0.9 and item["n"] >= 15:
                    sim = round(min(0.70, sim + 0.02), 2)   # 总是有结果 → 收紧
                self.state.setdefault("topic_sim", {})[topic] = sim
                if item["n"] % 8 == 0:
                    logger.debug(f"[Wave] topic {topic}: hit {rate:.0%}, sim floor {sim}")
            self._save_state()
        except Exception:
            pass
        try:
            _cache[_key] = (_now, [dict(x) for x in hits])
            if len(_cache) > 200:
                for _k in list(_cache.keys())[:-200]:
                    _cache.pop(_k, None)
        except Exception:
            pass
        self._log_trace(query, hits)
        return hits

    # ---------- associative retrieval ----------
    def _vector_candidates(self, qvec: list[float], nodes: dict, top: int = 8, floor: float = 0.55) -> list[str]:
        """用查询向量在所有记忆里做近邻，补上标签漏掉的关联。"""
        try:
            if not qvec:
                return []
            scored = []
            for nid, node in nodes.items():
                vec = self.emb.get(text_key(node.get("text", "")))
                if not vec:
                    continue
                scored.append((cosine(qvec, vec), nid))
            scored.sort(reverse=True)
            return [nid for sim, nid in scored[:top] if sim >= floor]
        except Exception:
            return []

    def _co_tags(self, tag: str, limit: int = 2) -> list[str]:
        """和这个标签经常一起出现的兄弟标签（共现扩展）。"""
        try:
            idx = getattr(self, "_tag_index", {}) or {}
            ids = set(idx.get(tag) or [])
            if not ids:
                return []
            counter = {}
            for other, oids in idx.items():
                if other == tag:
                    continue
                overlap = len(ids & set(oids))
                if overlap >= 2:
                    counter[other] = overlap
            top = sorted(counter.items(), key=lambda kv: -kv[1])[:limit]
            return [t for t, _ in top]
        except Exception:
            return []

    def retrieve(self, query: str, k: int | None = None, qvec: list[float] | None = None, extra_tags: list[str] | None = None) -> list[dict]:
        """标签驱动的联想检索：先用标签词表定位候选，再在候选里排序，不再全量遍历。"""
        query = (query or "").strip()
        if not query:
            return []
        if not self._graph.get("nodes"):
            self.build_graph()
        k = k or self._int("recall_top_k", 6, 1)
        now = time.time()
        nodes = {n["id"]: n for n in self._graph.get("nodes") or []}
        index = getattr(self, "_tag_index", {}) or {}

        # 1) 从词表里找这句话命中的标签
        qtags = [t for t in TAXONOMY if t in query]
        if extra_tags:
            qtags = list(dict.fromkeys(qtags + list(extra_tags)))
        if qtags:
            _co = []
            for t in qtags[:2]:
                _co.extend(self._co_tags(t))
            qtags = list(dict.fromkeys(qtags + _co))
        cand: list[str] = []
        seen = set()
        for tag in qtags:
            for nid in (index.get(tag) or [])[:40]:
                if nid in seen or nid not in nodes:
                    continue
                seen.add(nid)
                cand.append(nid)

        # 话题自适应：每个话题有自己的相似度阈值
        _topic = qtags[0] if qtags else "__none__"
        self._last_topic = _topic
        _topic_sim = (self.state.get("topic_sim") or {}).get(_topic)
        _sim_floor = float(_topic_sim) if _topic_sim else 0.55
        _sim_floor = max(0.45, min(0.70, _sim_floor))

        # 1.5) 向量近邻：补上标签没覆盖到的关联
        for nid in self._vector_candidates(qvec, nodes, floor=_sim_floor):
            if nid not in seen:
                seen.add(nid)
                cand.append(nid)

        # 2) 没有标签命中时退化到字符相似（只扫最近 120 条，不全量）
        if not cand:
            recent = sorted(nodes.values(), key=lambda n: -(n.get("ts") or 0))[:60]
            qv = bigrams(query)
            for node in recent:
                if jaccard(qv, bigrams(node.get("text", ""))) > 0.12:
                    cand.append(node["id"])
        if not cand:
            return []

        # 3) 只对候选打分
        live_notes = {n.get("summary") for n in self.l2[-120:] if n.get("expires_at", 0) > now}
        known = self._known_tags()
        hit_tags = set(qtags) & known
        qvec = qvec or self.emb.get(text_key(query))
        qval = self._query_valence(query)
        scored = []
        for nid in cand[:60]:
            node = nodes[nid]
            if node["kind"] == "note" and node.get("text") not in live_notes:
                continue
            score = 0.0
            shared = set(node.get("tags") or []) & hit_tags
            if shared:
                import math as _m
                for _t in shared:
                    df = max(1, len(index.get(_t) or []))
                    score += 1.0 + _m.log(1.0 + 60.0 / df) * 1.4
            hits_n = int(node.get("hits") or 0)
            if hits_n:
                score += min(0.6, hits_n * 0.06)
            age_days = max(0.0, (now - float(node.get("ts") or now)) / DAY)
            score += max(0.0, 0.35 - age_days * 0.01)
            if node.get("sub") == "diary":
                score += 0.25
            if qvec:
                nv = self.emb.get(text_key(node.get("text", "")))
                if nv:
                    score += max(0.0, cosine(qvec, nv)) * 2.0
            sim = jaccard(bigrams(query), bigrams(node.get("text", "")))
            if sim > 0.05:
                score += sim * 1.2
            if node["kind"] == "fact":
                score += 0.35
            mem_v = float((node.get("emotion") or {}).get("v") or 0)
            if qval and mem_v:
                score += 0.9 * (1 - min(1, abs(qval - mem_v)))
            if score <= 0:
                continue
            # 证据门槛：不能只靠"一点点像"就注入
            _cos = 0.0
            if qvec:
                _nv = self.emb.get(text_key(node.get("text", "")))
                if _nv:
                    _cos = cosine(qvec, _nv)
            if not shared and _cos < 0.55 and sim < 0.25:
                continue
            # 太短的记忆信息量低，不注入
            _t = str(node.get("text") or "").strip()
            if len(_t) < 8:
                continue
            # 寒暄类摘要没有回忆价值
            if "寒暄" in (node.get("tags") or []):
                continue
            if any(k in _t for k in ("简单应声", "表示知道了", "没有回复", "已读", "只回了一个")):
                continue
            scored.append((score, nid))
        scored.sort(reverse=True)
        _floor = float(self.state.get("score_floor") or self.cfg("recall_score_floor", 1.2) or 1.2)
        _floor = max(0.8, min(1.5, _floor))
        scored = [x for x in scored if x[0] >= _floor]
        picked: list[str] = []
        picked_vecs: list[set] = []
        for _score, nid in scored:
            if len(picked) >= k:
                break
            grams = bigrams(nodes[nid].get("text", ""))
            if any(jaccard(grams, g) >= 0.72 for g in picked_vecs):
                continue
            picked.append(nid)
            picked_vecs.append(grams)

        extra: list[str] = []
        expand = self._int("assoc_expand", 2, 0)
        weight = getattr(self, "_link_weight", None)
        if not weight:
            weight = {}
            for link in self._graph.get("links") or []:
                weight[(link.get("a"), link.get("b"))] = float(link.get("w") or 0)
                weight[(link.get("b"), link.get("a"))] = float(link.get("w") or 0)
            self._link_weight = weight
        frontier = list(picked[:3])
        visited = set(picked)
        for _hop in range(2):
            nxt = []
            for nid in frontier:
                neigh = [x for x in (self._adj.get(nid) or []) if x not in visited and x in nodes]
                neigh.sort(key=lambda x: (
                    0 if (nodes.get(x, {}).get("kind") != nodes.get(nid, {}).get("kind")) else 1,
                    -weight.get((nid, x), 0),
                ))
                for nb in neigh[:2]:
                    visited.add(nb)
                    nxt.append(nb)
                    if nb not in extra:
                        extra.append(nb)
                    if len(extra) >= expand:
                        break
                if len(extra) >= expand:
                    break
            frontier = nxt
            if len(extra) >= expand:
                break
        if extra:
            extra.sort(key=lambda x: -weight.get((picked[0], x), 0) if picked else 0)

        results = []
        for nid in picked + extra:
            node = nodes.get(nid)
            if not node:
                continue
            results.append({
                "id": nid,
                "kind": node["kind"],
                "text": node["text"],
                "tags": node.get("tags") or [],
                "linked": nid in extra,
                "ts": float(node.get("ts") or 0),
                "score": round(float(next((sc for sc, x in scored if x == nid), 0.0)), 2),
            })
            if node["kind"] == "fact":
                for fact in self.l3:
                    if fact.get("text") == node["text"]:
                        fact["last_seen"] = now
                        fact["hits"] = int(fact.get("hits", 1)) + 1
                        break
        return results

    def _known_tags(self) -> set[str]:
        tags: set[str] = set()
        for fact in self.l3:
            tags.update(fact.get("tags") or [])
        for note in self.l2[-120:]:
            tags.update(note.get("tags") or [])
        return tags


    # ---------- 日记（原「云上日记」） ----------
    def _diary_state_path(self) -> Path:
        return self.data_dir / "diary_state.json"

    def _load_diary(self) -> None:
        try:
            p = self._diary_state_path()
            if p.exists():
                d = json.loads(p.read_text(encoding="utf-8"))
                if isinstance(d, dict):
                    self.diary_entries = list(d.get("entries") or [])
                    self.last_diary = float(d.get("last_diary") or 0)
        except Exception:
            pass

    def _save_diary(self) -> None:
        try:
            self._diary_state_path().write_text(
                json.dumps({"entries": self.diary_entries, "last_diary": self.last_diary},
                           ensure_ascii=False, indent=2), encoding="utf-8")
        except Exception as exc:
            logger.warning(f"[海浪] 保存日记失败: {exc}")

    async def _consolidate_diary(self) -> None:
        """超过 24 篇时，把最早的 24 篇合成一篇年度合集。"""
        limit = self._int("diary_digest_every", 24, 2)
        if len(self.diary_entries) <= limit:
            return
        batch = self.diary_entries[:limit]
        text = "\n\n".join(self._diary_text(e, True) for e in batch)
        prompt = (
            "下面是同一个人一年里写的 %d 篇日记。请合成一篇年度合集，只输出 JSON：\n"
            '{"title":"合集标题（12字内）","outline":["3-6条纲要，每条≤20字"],"body":"400-600字"}\n'
            "要求：保留重要事件与关系变化，按时间顺序，第一人称，不要编造，不要 markdown。\n\n"
            "%s" % (len(batch), text[:12000])
        )
        raw = await self._llm(prompt, "你是精确的日记汇编者，只输出 JSON。")
        payload = self._extract_json(raw)
        body = str(payload.get("body") or "").strip()
        if not body:
            logger.warning("[Wave] 日记合并失败：模型没返回正文")
            return
        digest = {
            "date": batch[-1].get("date") or "",
            "title": str(payload.get("title") or "年度合集").strip(),
            "outline": [str(x).strip() for x in (payload.get("outline") or []) if str(x).strip()][:6],
            "body": body,
            "ts": time.time(),
            "digest": True,
            "count": len(batch),
        }
        self.diary_entries = [digest] + self.diary_entries[limit:]
        self._save_diary()
        logger.info("[Wave] 日记合并：%d 篇 → 1 篇合集（剩 %d 篇）" % (len(batch), len(self.diary_entries)))

    def _diary_text(self, entry: dict, with_body: bool) -> str:
        date = entry.get("date") or ""
        title = entry.get("title") or ""
        outline = entry.get("outline") or []
        head = date + (f" · {title}" if title else "")
        lines = [head, "【大纲】"] + [f"- {x}" for x in outline]
        if with_body and entry.get("body"):
            lines += ["【正文】", entry["body"]]
        return "\n".join(lines)

    async def write_diary(self, force: bool = False) -> str:
        interval = self._int("diary_interval_days", 14, 1)
        now = time.time()
        if not force and self.last_diary and (now - self.last_diary) < interval * DAY:
            return "还没到写日记的日子。"
        lines = await asyncio.to_thread(self._read_conversation_sync)
        if not lines:
            return "这段时间没怎么说话，先不写了。"
        limit = max(2000, int(self.cfg("diary_max_input_chars", 12000) or 12000))
        transcript = "\n".join(lines)
        if len(transcript) > limit:
            transcript = "（前略）\n" + transcript[-limit:]
        prompt = (
            "下面是你们最近的聊天记录。请写一篇日记，只输出 JSON：\n"
            '{"title":"小标题（8字以内）","outline":["3-5条大纲，每条≤20字"],"body":"正文250-450字"}\n'
            "要求：1) 大纲在最前，用短句概括最重要的三四件事；"
            "2) 正文用第一人称，有情绪、可傲娇，只写真实发生的事，不编造；"
            "3) 分两到三段，纯文本，不要 markdown；4) 只输出 JSON。\n\n"
            f"聊天记录：\n{transcript}"
        )
        raw = await self._llm(prompt, "你是精确的日记写作者，只输出 JSON。")
        payload = self._extract_json(raw)
        body = str(payload.get("body") or "").strip()
        if not body:
            return "这次没写出来。"
        entry = {
            "date": datetime.now().strftime("%Y-%m-%d"),
            "title": str(payload.get("title") or "").strip(),
            "outline": [str(x).strip() for x in (payload.get("outline") or []) if str(x).strip()][:6],
            "ts": now,
            "body": body,
        }
        self.diary_entries.append(entry)
        self.last_diary = now
        self._save_diary()
        try:
            await self._consolidate_diary()
        except Exception as _exc:
            logger.warning(f"[Wave] diary consolidate failed: {_exc}")
        await self._apply_persona()
        logger.info(f"[海浪] 写成第 {len(self.diary_entries)} 篇日记")
        return self._diary_text(entry, True)

    # ---------- persona ----------
    def _facts_block(self) -> str:
        cap = self._int("fact_persona_max", 30, 5)
        picked = sorted([f for f in self.l3 if not f.get("parked")], key=lambda f: f.get("last_seen", 0))[-cap:]
        if not picked:
            return ""
        lines = []
        for fact in picked:
            tags = "".join(f"#{t} " for t in (fact.get("tags") or [])[:2])
            lines.append(f"· {tags}{fact['text']}")
        return FACTS_TITLE + "\n" + "\n".join(lines)

    async def _apply_persona(self) -> None:
        if not str(self.cfg("persona_id", "") or "").strip():
            logger.debug("[Wave] persona not configured, skip persona sync")
            return
        persona_id = str(self.cfg("persona_id", ""))
        try:
            persona = await self.context.persona_manager.get_persona(persona_id)
            if not persona:
                return
            base, sections = split_sections(persona.system_prompt or "", [FACTS_TITLE, DIARY_TITLE])
            parts = [base.strip()]
            block = self._facts_block()
            if block:
                parts.append(block)
            keep_n = self._int("diary_persona_entries", 2, 1)
            recent = self.diary_entries[-keep_n:]
            diary_blocks = []
            for idx, entry in enumerate(recent):
                diary_blocks.append(self._diary_text(entry, idx == len(recent) - 1))
            if diary_blocks:
                parts.append(DIARY_TITLE + "\n" + "\n\n".join(diary_blocks))
            elif sections.get(DIARY_TITLE):
                parts.append(sections[DIARY_TITLE])
            new_prompt = "\n\n".join(p for p in parts if p).strip()
            _h = hashlib.sha1(new_prompt.encode("utf-8")).hexdigest()[:16]
            if _h == str(self.state.get("persona_hash") or ""):
                self._last_sync = time.time()
                return
            await self.context.persona_manager.update_persona(persona_id, system_prompt=new_prompt)
            self.state["persona_hash"] = _h
            self._save_state()
            self._last_sync = time.time()
            logger.debug(f"[记忆] 人格已更新：{len(self.l3)} 条长期记忆（{len(new_prompt)} 字）")
        except Exception as exc:
            logger.error(f"[记忆] 更新人格失败: {exc}")

    async def _sync_persona(self, force: bool = False) -> None:
        if not bool(self.cfg("sync_facts_to_persona", True)):
            return
        gap = self._int("fact_sync_min_minutes", 10, 1) * 60
        if not force and (time.time() - self._last_sync) < gap:
            return
        await self._apply_persona()

    async def organize_facts(self) -> str:
        facts = [f for f in self.l3 if (f.get("text") or "").strip() and not f.get("parked")]
        if len(facts) < 2:
            await self._apply_persona()
            return "还不够整理"
        numbered = "\n".join(
            f"{i}. {f['text']}  标签: {','.join(f.get('tags') or [])}" for i, f in enumerate(facts)
        )
        prompt = (
            "下面是我关于主人的长期记忆。请整理，只输出 JSON：\n"
            '{"facts":[{"text":"整理后的条目","tags":["标签"]}]}\n'
            "1. 合并说同一件事的条目；2. 矛盾的只留最新；3. 删掉过期的能力/状态描述；"
            f"4. 每条≤40字，最多 {self._int('fact_persona_max', 30, 5)} 条；5. 每条给 1-2 个 2-4 字标签；6. 不新增信息。\n\n"
            f"条目：\n{numbered}"
        )
        payload = self._extract_json(await self._llm(prompt, "你是精确的记忆整理器，只输出 JSON。"))
        items = payload.get("facts") or []
        if items:
            now = time.time()
            old = {(f.get("text") or "").strip(): f for f in self.l3}
            rebuilt = []
            for item in items:
                if isinstance(item, dict):
                    text = str(item.get("text") or "").strip()
                    tags = [str(t).strip()[:8] for t in (item.get("tags") or []) if str(t).strip()][:2]
                else:
                    text, tags = str(item).strip(), []
                if not text:
                    continue
                src = old.get(text) or {}
                rebuilt.append(
                    {
                        "text": text,
                        "created": src.get("created", now),
                        "last_seen": src.get("last_seen", now),
                        "hits": src.get("hits", 1),
                        "tags": tags or (src.get("tags") or []),
                    }
                )
            self.l3 = rebuilt[: self._int("fact_persona_max", 30, 5)]
            self._save(self.l3_path, self.l3)
            logger.info(f"[记忆] 整理完成：{len(facts)} → {len(self.l3)}")
            await asyncio.to_thread(self.build_graph)
        await self._apply_persona()
        return f"{len(facts)} → {len(self.l3)}"

    async def _remember_on_demand(self, text: str) -> None:
        """主人明确说「记住」时，直接写进 L3 并刷新人格，不经过压缩。"""
        try:
            prompt = (
                "主人说了这句话：" + text[:300] + "\n\n"
                "请从中提取他要你长期记住的事实，只输出 JSON：\n"
                '{"fact":"一句话，第三人称，≤60字","tags":["5个标签"],"emotion":{"label":"情绪词","v":0.0}}\n'
                "标签优先用词表：吃饭 睡觉 工作 学习 游戏 情绪 委屈 开心 生气 难过 累 偏好 "
                "讨厌 喜欢 约定 称呼 家人 朋友 钱 考试 生病 表情包 语音 陪伴 边界 隐私。"
                "如果这句话是一时情绪或闲聊，fact 留空字符串。"
            )
            raw = await self._llm(prompt, "你是精确的记忆提取器，只输出 JSON。")
            payload = self._extract_json(raw)
            fact = str(payload.get("fact") or "").strip()
            if not fact or len(fact) < 4:
                logger.info("[记忆] 明确要求记住，但没提取到事实")
                return
            tags = [str(t).strip() for t in (payload.get("tags") or []) if str(t).strip()][:5]
            emo = payload.get("emotion") or {}
            self._add_fact(fact, time.time(), tags, emo if isinstance(emo, dict) else None)
            for f in self.l3:
                if f.get("text") == fact:
                    f["pinned"] = True
                    f["source"] = "on_demand"
                    break
            try:
                self._save(self.l3_path, self.l3)
            except Exception:
                pass
            await self._apply_persona()
            logger.info("[记忆] 明确要求记住 → 已写入长期：" + fact[:40])
        except Exception as exc:
            logger.warning(f"[记忆] 即时记忆失败: {exc}")

    def _private_chat_ok(self, umo: str) -> bool:
        """兼容多个平台：snowluma(OneBot) 与 qq_official(C2C)。"""
        allows = [x.strip() for x in str(self.cfg("platform_allow", "snowluma,qq_official") or "").split(",") if x.strip()]
        marks = [x.strip() for x in str(self.cfg("private_markers", "FriendMessage,C2CMessage") or "").split(",") if x.strip()]
        if allows and not any(a in umo for a in allows):
            return False
        return any(m in umo for m in marks)

    def _owner_ids(self) -> list[str]:
        raw = str(self.cfg("owner_id", "") or "")
        ids = [x.strip() for x in raw.replace("，", ",").split(",") if x.strip()]
        return ids

    def _note_owner_candidate(self, sid: str, umo: str) -> None:
        """记录见过但不在 owner_id 里的发送者，方便把官方机器人的 openid 填进配置。"""
        try:
            seen = self.state.setdefault("owner_candidates", [])
            key = f"{sid}|{umo.split(':')[0] if umo else ''}"
            if key in seen:
                return
            seen.append(key)
            self.state["owner_candidates"] = seen[-20:]
            self._save_state()
            logger.info(f"[记忆] 新的私聊来源 sender={sid} umo={umo}；如这是主人，把它填进 owner_id")
        except Exception:
            pass

    def _record_event_message(self, text: str, role: str = "user") -> None:
        """官方机器人没有本地会话库可读，直接按事件归档到 L4 + pending。"""
        try:
            text = str(text or "").strip()
            if not text:
                return
            day = time.strftime("%Y-%m-%d")
            rec = {"ts": time.time(), "role": role, "text": text[:500]}
            self.archive_dir.mkdir(parents=True, exist_ok=True)
            with (self.archive_dir / f"{day}.jsonl").open("a", encoding="utf-8") as fh:
                fh.write(json.dumps(rec, ensure_ascii=False) + "\n")
            pending = list(self.state.get("pending") or [])
            pending.append(rec)
            self.state["pending"] = pending[-500:]
            self._save_state()
        except Exception as exc:
            logger.debug(f"[记忆] 事件归档失败: {exc}")

    # 这些是"系统说明书"式的记忆，注入了她会用系统口吻说话
    META_NOISE = ("系统授权", "系统", "授权我", "指令", "插件", "配置项", "权限",
                  "token", "prompt", "接口", "部署", "运行状态", "省token")

    # ---------- injection ----------
    @filter.on_llm_request()
    def _provider_stats(self, since: float = 0) -> dict:
        """从 AstrBot 的 provider_stats 里读最近 20 次请求的缓存命中情况。"""
        try:
            db = Path(self.data_dir).parent.parent / "data_v4.db"
            if not db.exists():
                return {}
            con = sqlite3.connect(f"file:{db}?mode=ro", uri=True, timeout=2)
            try:
                if since:
                    rows = con.execute(
                        "select token_input_cached, token_input_other from provider_stats "
                        "where created_at > ? order by created_at desc", (since,)
                    ).fetchall()
                else:
                    rows = con.execute(
                        "select token_input_cached, token_input_other from provider_stats "
                        "order by created_at desc limit 20"
                    ).fetchall()
            finally:
                con.close()
            cached = sum(int(r[0] or 0) for r in rows)
            other = sum(int(r[1] or 0) for r in rows)
            total = cached + other
            if not rows or not total:
                return {}
            return {"cached": cached, "other": other, "n": len(rows),
                    "hit": round(100.0 * cached / total, 1),
                    "avg_in": int(total / len(rows))}
        except Exception:
            return {}

    def _cache_plan(self) -> dict:
        plan = self.state.get("cache_plan")
        if not isinstance(plan, dict):
            plan = {"chars": 240, "max": 4, "time": "auto",
                    "trial": None, "best": None, "samples": 0, "log": []}
            self.state["cache_plan"] = plan
        for k, v in (("chars", 240), ("max", 4), ("time", "auto"), ("samples", 0)):
            plan.setdefault(k, v)
        return plan

    def _cache_score(self, st: dict) -> float:
        """命中率越高越好，平均输入越短越好。"""
        if not st:
            return 0.0
        hit = float(st.get("hit") or 0) / 100.0
        avg = float(st.get("avg_in") or 60000)
        size = max(0.0, 1.0 - avg / 60000.0)
        return round(hit * 0.55 + size * 0.45, 4)

    def _cache_tune_tick(self) -> None:
        """每 10 轮评估一次：命中率 ↑ 且输入 ↓ 的方案胜出，逐步逼近最优。"""
        try:
            plan = self._cache_plan()
            plan["samples"] = int(plan.get("samples", 0)) + 1
            if plan["samples"] % 10 != 0:
                return
            st = self._provider_stats(float(plan.get("mark") or 0))
            if not st:
                return
            score = self._cache_score(st)
            cur = {"chars": plan["chars"], "max": plan["max"], "time": plan["time"]}
            plan["log"] = (plan.get("log") or [])[-9:] + [{
                "t": datetime.now().strftime("%m-%d %H:%M"),
                "hit": st["hit"], "avg_in": st["avg_in"], "score": score,
                "chars": cur["chars"], "max": cur["max"], "time": cur["time"],
            }]

            trial = plan.get("trial")
            if trial:
                # 试用期结束：和基线比
                base = plan.get("best") or {"score": -1}
                if score > float(base.get("score", -1)):
                    plan["best"] = {"score": score, **cur}
                    logger.info(f"[Wave] 缓存调优：保留 chars={cur['chars']} max={cur['max']} time={cur['time']}"
                                f"（命中 {st['hit']}%，均输入 {st['avg_in']}，score {score}）")
                else:
                    plan.update(trial["prev"])
                    logger.info(f"[Wave] 缓存调优：回退到 chars={plan['chars']} max={plan['max']} "
                                f"（新方案 score {score} 不如 {base.get('score')}）")
                plan["trial"] = None
                plan["mark"] = time.time()
                self._save_state()
                return

            # 选一个还没试过的方向
            order = plan.get("order") or ["chars-", "time-on", "max-", "chars+", "time-off", "max+"]
            step = order.pop(0) if order else "chars-"
            plan["order"] = order or ["chars-", "time-on", "max-", "chars+", "time-off", "max+"]
            plan["best"] = plan.get("best") or {"score": score, **cur}
            prev = dict(cur)
            if step == "chars-" and cur["chars"] > 160:
                plan["chars"] = cur["chars"] - 40
            elif step == "chars+" and cur["chars"] < 400:
                plan["chars"] = cur["chars"] + 40
            elif step == "max-" and cur["max"] > 2:
                plan["max"] = cur["max"] - 1
            elif step == "max+" and cur["max"] < 6:
                plan["max"] = cur["max"] + 1
            elif step == "time-on" and cur["time"] != "on":
                plan["time"] = "on"
            elif step == "time-off" and cur["time"] != "off":
                plan["time"] = "off"
            else:
                plan["chars"] = max(160, cur["chars"] - 40)
            plan["mark"] = time.time()
            plan["trial"] = {"step": step, "prev": prev}
            self._save_state()
            logger.info(f"[Wave] 缓存调优：试 {step} → chars={plan['chars']} max={plan['max']} time={plan['time']}")
        except Exception as exc:
            logger.debug(f"[Wave] 缓存调优跳过: {exc}")

    async def inject(self, event: AstrMessageEvent, req: ProviderRequest):
        try:
            umo = str(getattr(event, "unified_msg_origin", "") or "")
            if not self._private_chat_ok(umo):
                return
            sid = str(event.get_sender_id())
            owners = self._owner_ids()
            if owners and sid not in owners:
                self._note_owner_candidate(sid, umo)
                return
            # 实时收录：每条新消息立刻进 L1/L4，星图马上能看到
            _is_onebot = "snowluma" in umo or ":FriendMessage:" in umo
            if not _is_onebot:
                self._record_event_message(str(getattr(event, "message_str", "") or ""))
            try:
                added = await asyncio.to_thread(self.sync_archive_sync)
                if added:
                    pending = list(self.state.get("pending") or []) + added
                    self.state["pending"] = pending[-500:]
                    self._save_state()
                    self._schedule_graph_rebuild()
                    logger.debug(f"[记忆] 实时收录 {len(added)} 条（待整理 {len(self.state.get('pending') or [])}）")
                    if len(self.state.get("pending") or []) >= 2 and not self._busy:
                        asyncio.create_task(self.compress())
            except Exception as exc:
                logger.warning(f"[记忆] 实时收录失败: {exc}")
            query = str(getattr(event, "message_str", "") or "").strip()
            if query and any(h in query for h in REMEMBER_HINTS):
                seen = list(self.state.get("remembered") or [])
                if query not in seen:
                    seen.append(query)
                    self.state["remembered"] = seen[-60:]
                    self._save_state()
                    asyncio.create_task(self._remember_on_demand(query))
            if query:
                await asyncio.to_thread(self.reinforce_mentions, query)
            _plain = "".join(ch for ch in query if not ch.isspace())
            _skip = len(_plain) <= 3 or _plain.lower() in SHORT_ACK
            qval = 0.0 if _skip else self._query_valence(query)
            hits = [] if _skip else (await self.retrieve_async(query) if query else [])
            try:
                st = self.state.setdefault("recall_stats", {"n": 0, "hit": 0, "empty": 0, "avg": 0})
                _ch = int(getattr(self, "_cache_hits", 0))
                if _ch:
                    st["cache_hit"] = _ch
                st["n"] = int(st.get("n", 0)) + 1
                if hits:
                    st["hit"] = int(st.get("hit", 0)) + 1
                    st["avg"] = round((float(st.get("avg", 0)) * (st["hit"] - 1) + len(hits)) / st["hit"], 2)
                else:
                    st["empty"] = int(st.get("empty", 0)) + 1
                if st["n"] % 20 == 0 and st["n"] >= 20:
                    hit_rate = float(st["hit"]) / max(1, st["n"])
                    cur = float(self.state.get("score_floor") or 1.1)
                    if hit_rate < 0.55 and cur > 0.8:
                        cur = round(max(0.8, cur - 0.1), 2)
                    elif hit_rate > 0.9 and float(st.get("avg") or 0) >= 4.5 and cur < 1.5:
                        cur = round(min(1.5, cur + 0.1), 2)
                    self.state["score_floor"] = cur
                    self._save_state()
                    logger.info(f"[Wave] score floor -> {cur} (hit {hit_rate:.0%}, avg {st.get('avg')})")
                self._cache_tune_tick()
                if st["n"] % 100 == 0:
                    rate = round(100.0 * st["hit"] / max(1, st["n"]), 1)
                    logger.debug(f"[Wave] recall stats: {st['hit']}/{st['n']} hit ({rate}%), avg {st['avg']} items")
            except Exception:
                pass
            if not hits:
                # 没有命中就退回今天的摘要（少量）
                now = time.time()
                recent = [n for n in self.l2 if n.get("expires_at", 0) > now][-4:]
                if not recent:
                    return
                block = "\n\n" + (self._now_line() if self._cache_plan().get("time") != "off" else "") + self._relationship_line() + "【最近聊过的】\n" + "\n".join(f"· [{self._day_label(n.get('ts'))}] {n['summary']}" for n in recent) + "\n"
            else:
                lines = []
                for hit in hits:
                    _txt = str(hit.get("text") or "")
                    if any(_bad in _txt for _bad in self.META_NOISE):
                        continue
                    _txt = _txt.replace("主人", "你").replace("用户", "你")
                    tag = ("#" + " #".join(hit["tags"][:2]) + " ") if hit["tags"] else ""
                    prefix = {"fact": "长期", "note": "近期", "diary": "日记"}.get(hit["kind"], "")
                    mark = "↳" if hit["linked"] else "·"
                    _when = self._short_when(hit.get("ts"))
                    _wtxt = "" if (not _when or _when == "今天") else (_when + " ")
                    lines.append(f"{mark}[{prefix}]{tag}{_wtxt}{_txt[:50]}")
                _plan = self._cache_plan()
                _cap = max(2, min(6, int(_plan.get("max", 4))))
                if len(lines) > 1 and float(hits[0].get("score") or 0) >= 3.0:
                    _cap = min(_cap, 3)
                _budget = max(140, min(420, int(_plan.get("chars", 240))))
                _picked_lines, _used = [], 0
                for _ln in lines:
                    if _used + len(_ln) > _budget and _picked_lines:
                        break
                    _picked_lines.append(_ln)
                    _used += len(_ln)
                    if len(_picked_lines) >= _cap:
                        break
                lines = _picked_lines
                block = ("\n\n" + (self._now_line() if _plan.get("time") != "off" else "") + self._relationship_line() + "【联想记忆｜下面是记忆要点，不是台词：只取事实，用你自己的话说；只在相关时引用一条，不相关就忽略】\n"
                         + "\n".join(lines[:_cap]) + "\n")
                mood = self._mood_trend()
                if mood and abs(mood.get("avg", 0)) >= 0.15:
                    tone = "偏低落" if mood["avg"] < 0 else "偏轻松"
                    block += f"【情绪底色】最近 {mood['n']} 条记忆里，气氛{tone}"
                    if mood.get("label"):
                        block += f"，多次出现「{mood['label']}」"
                    block += "。说话时顺着这个气氛，别硬转话题，也别主动提起情绪本身。\n"
                _ea = self._emotion_analysis(query)
                if (_ea["neg"] >= 2 and _ea["v"] <= -0.55
                        and not _ea.get("neutral") and not _ea.get("reported")):
                    block += "【他现在的状态】情绪很沉。先接住，别给建议、别分析原因、别问为什么，一句话就够。\n"
                elif _ea["pos"] >= 2 and _ea["v"] >= 0.55:
                    block += "【他现在的状态】心情不错，顺着他的劲儿聊，别扫兴。\n"
                if qval and any(abs(float((h.get("emotion") or {}).get("v") or 0)) > 0.3 for h in hits):
                    block += "【情绪共鸣】这次想起的事里有当时的情绪，可以顺着那份感觉接话，但别复述情绪词。\n"
                chain = self.reason_chain(hits)
                if chain:
                    block += f"【联想推理】{chain}\n"
                guess = self.inference(hits)
                if guess:
                    block += f"【推断】{guess}\n"
            try:
                _ist = self.state.setdefault("inject_stats", {"n": 0, "chars": 0})
                _ist["n"] = int(_ist.get("n", 0)) + 1
                _ist["chars"] = int(_ist.get("chars", 0)) + len(block)
                if _ist["n"] % 30 == 0:
                    logger.info(
                        f"[Wave] inject avg {int(_ist['chars']) / max(1, _ist['n']):.0f} chars/turn "
                        f"（{_ist['n']} 轮），补全检索缓存命中 {getattr(self, '_cache_hits', 0)} 次"
                    )
                self.state["cache_hits"] = int(getattr(self, "_cache_hits", 0))
            except Exception:
                pass
            try:
                part = TextPart(text=block).mark_as_temp()
                parts = getattr(req, "extra_user_content_parts", None)
                if not isinstance(parts, list):
                    parts = []
                    req.extra_user_content_parts = parts
                parts.append(part)
            except Exception as _exc:
                logger.debug(f"[记忆] 临时内容注入失败: {_exc}")
            preview = " | ".join(l.strip() for l in block.strip().splitlines()[1:4])
            logger.debug(f"[记忆] 联想注入 {len(hits) or 4} 条（{len(block)} 字）：{preview[:110]}")
        except Exception as exc:
            logger.error(f"[记忆] 注入失败: {exc}", exc_info=True)

    # ---------- L4 recall ----------
    def search_sync(self, keyword: str, limit: int = 8) -> list[dict]:
        hits: list[dict] = []
        if not keyword:
            return hits
        for path in sorted(self.archive_dir.glob("*.jsonl"), reverse=True):
            try:
                for line in path.read_text(encoding="utf-8").splitlines():
                    if keyword not in line:
                        continue
                    try:
                        hits.append(json.loads(line))
                    except Exception:
                        continue
                    if len(hits) >= limit:
                        return hits
            except Exception:
                continue
        return hits

    @filter.llm_tool(name="memory_search")
    async def memory_search_tool(self, event: AstrMessageEvent, query: str) -> str:
        """在长期记忆里搜索和当前话题相关的记忆（L2 摘要、L3 事实、L4 档案）。

        不确定以前聊过什么、想不起细节时用它。返回条目、标签、时间和 id。

        Args:
            query(string): 要回忆的话题或关键词。
        """
        hits = await self.retrieve_async(query, k=8)
        out = [{
            "id": h.get("id"),
            "kind": h.get("kind"),
            "text": h.get("text"),
            "tags": h.get("tags"),
            "linked": h.get("linked"),
        } for h in hits]
        return json.dumps({"ok": True, "count": len(out), "hits": out}, ensure_ascii=False)

    @filter.llm_tool(name="memory_save")
    async def memory_save_tool(self, event: AstrMessageEvent, text: str, tags: str = "") -> str:
        """把一条重要信息写进长期记忆（永久保存、参与之后的联想）。

        适合记：主人的偏好、约定、身份、反复强调的事。不要拿它记一次性的闲聊。

        Args:
            text(string): 一句话写清楚要记住的事。
            tags(string): 逗号分隔的短标签，最多 3 个，可为空。
        """
        tag_list = [t.strip()[:8] for t in str(tags or "").replace("，", ",").split(",") if t.strip()][:3]
        plain = " ".join(str(text or "").split())[:200]
        if not plain:
            return json.dumps({"ok": False, "reason": "内容为空"}, ensure_ascii=False)
        self._add_fact(plain, time.time(), tag_list)
        self._save(self.l3_path, self.l3)
        await self._apply_persona()
        await asyncio.to_thread(self.build_graph)
        logger.info(f"[记忆] 主动记住：{plain[:30]}（标签 {tag_list}）")
        return json.dumps({"ok": True, "saved": plain, "tags": tag_list}, ensure_ascii=False)

    @filter.llm_tool(name="recall_archive")
    async def recall_archive(self, event: AstrMessageEvent, keyword: str) -> str:
        """从永久存档里搜索很久以前说过的原话。想不起细节时用它翻原文。

        Args:
            keyword(string): 要搜索的关键词，例如名字、地点、约定里出现过的词。
        """
        hits = await asyncio.to_thread(self.search_sync, keyword, 8)
        if not hits:
            return json.dumps({"ok": True, "hits": [], "note": "存档里没有相关记录"}, ensure_ascii=False)
        lines = []
        for rec in hits:
            who = "主人" if rec.get("role") == "user" else "我"
            stamp = datetime.fromtimestamp(rec.get("ts", 0)).strftime("%m-%d %H:%M")
            lines.append(f"[{stamp}] {who}: {rec.get('text', '')[:200]}")
        return json.dumps({"ok": True, "hits": lines}, ensure_ascii=False)

    # ---------- loop ----------
    async def _loop(self):
        await asyncio.sleep(15)
        while True:
            try:
                if bool(self.cfg("enabled", True)):
                    added = await asyncio.to_thread(self.sync_archive_sync)
                    if added:
                        await self.enrich_recent(added)
                        pending = list(self.state.get("pending") or []) + added
                        self.state["pending"] = pending[-500:]
                        self._save_state()
                    for _ in range(self._int("compress_max_batches", 8, 1)):
                        pending = list(self.state.get("pending") or [])
                        if len(pending) < 1:
                            break
                        await self.compress()
                        if len(self.state.get("pending") or []) >= len(pending):
                            break
                    _rv = float(self.state.get("last_review") or 0)
                    if (time.time() - _rv) > self._int("self_review_days", 7, 1) * DAY:
                        await self._self_review()
                    if True:
                        interval = self._int("diary_interval_days", 14, 1)
                        if self.last_diary and (time.time() - self.last_diary) >= interval * DAY:
                            await self.write_diary()
                    before = (len(self.l2), len(self.l3), len(self.state.get("pending") or []))
                    await asyncio.to_thread(self.cleanup)
                    after = (len(self.l2), len(self.l3), len(self.state.get("pending") or []))
                    if added or before != after:
                        await asyncio.to_thread(self.build_graph)
            except asyncio.CancelledError:
                raise
            except Exception as exc:
                logger.error(f"[记忆] 后台异常: {exc}")
            await asyncio.sleep(max(1, self._int("sync_interval_minutes", 5, 1)) * 60)


    # ---------- 内置星图页面 ----------
    def _web_payload(self) -> dict:
        now = time.time()
        facts = [f for f in self.l3 if not f.get("parked")]
        notes = [n for n in self.l2 if n.get("expires_at", 0) > now]
        waves = []
        try:
            p = self.data_dir / "assoc_log.jsonl"
            if p.exists():
                for line in p.read_text(encoding="utf-8").splitlines()[-40:]:
                    try:
                        waves.append(json.loads(line))
                    except Exception:
                        continue
        except Exception:
            pass
        diaries = [
            {"date": e.get("date"), "title": e.get("title"), "outline": e.get("outline") or []}
            for e in self.diary_entries[-5:]
        ]
        return {
            "facts": [{"text": f.get("text"), "tags": f.get("tags") or []} for f in facts],
            "notes": [{"text": n.get("summary"), "tags": n.get("tags") or []} for n in notes],
            "diaries": diaries,
            "graph": self._graph,
            "waves": waves,
        }

    def _start_web(self) -> None:
        try:
            import http.server
            import threading

            root = Path(__file__).parent / "web"
            if not root.exists():
                return
            plugin = self

            class Handler(http.server.SimpleHTTPRequestHandler):
                def __init__(self, *args, **kwargs):
                    super().__init__(*args, directory=str(root), **kwargs)

                def log_message(self, *args):
                    return

                def do_GET(self):
                    if self.path.startswith("/api/cloud"):
                        token = str(plugin.cfg("web_token", "") or plugin.state.get("web_token") or "").strip()
                        if token:
                            from urllib.parse import urlparse, parse_qs
                            qs = parse_qs(urlparse(self.path).query)
                            got = (qs.get("token") or [""])[0] or self.headers.get("X-Wave-Token", "")
                            if got != token:
                                self.send_response(401)
                                self.end_headers()
                                self.wfile.write(b'{"error":"unauthorized"}')
                                return
                        body = json.dumps({"memory": plugin._web_payload()}, ensure_ascii=False).encode("utf-8")
                        self.send_response(200)
                        self.send_header("Content-Type", "application/json; charset=utf-8")
                        self.send_header("Content-Length", str(len(body)))
                        self.end_headers()
                        self.wfile.write(body)
                        return
                    _path_only = self.path.split("?", 1)[0]
                    if _path_only in ("", "/", "/index.html"):
                        _qs = self.path.split("?", 1)[1] if "?" in self.path else ""
                        self.path = "/star_map.html" + (("?" + _qs) if _qs else "")
                    return super().do_GET()

            port = self._int("web_port", 8788, 1024)
            host = str(self.cfg("web_host", "0.0.0.0") or "0.0.0.0")
            server = http.server.ThreadingHTTPServer((host, port), Handler)
            threading.Thread(target=server.serve_forever, daemon=True).start()
            token = str(self.cfg("web_token", "") or "").strip()
            if not token:
                token = str(self.state.get("web_token") or "")
                if not token:
                    try:
                        import secrets as _secrets
                        token = _secrets.token_urlsafe(9)
                    except Exception:
                        token = str(int(time.time()))[-8:]
                    self.state["web_token"] = token
                    self._save_state()
            logger.info(f"[海浪] 记忆星图已启动：http://<host>:{port}/?token={token}")
        except Exception as exc:
            logger.warning(f"[海浪] 星图页面启动失败: {exc}")

    async def initialize(self):
        if not self._task or self._task.done():
            self._task = asyncio.create_task(self._loop())
        if len(self.diary_entries) > self._int("diary_digest_every", 24, 2):
            async def _startup_digest():
                await asyncio.sleep(20)
                try:
                    await self._consolidate_diary()
                except Exception as exc:
                    logger.warning(f"[Wave] startup digest failed: {exc}")
            asyncio.create_task(_startup_digest())
        try:
            if time.time() - float(self.state.get("last_backup") or 0) > 20 * 3600:
                self._daily_backup()
                self.state["last_backup"] = time.time()
                self._save_state()
        except Exception:
            pass
        self._start_web()
        try:
            await asyncio.to_thread(self.build_graph)
            asyncio.create_task(self._deferred_semantic())
        except Exception as exc:
            logger.warning(f"[记忆] 启动自检失败: {exc}")
        logger.info(
            f"[记忆] 已启动：L1 {self._int('l1_window', 10)} 句 / L2 {self._int('l2_ttl_hours', 24)}h / "
            f"L3 永久 / L4 存档 / 联想图 {len(self._graph.get('nodes') or [])} 点"
        )

    async def terminate(self):
        if self._task:
            self._task.cancel()
            self._task = None
        logger.info("[记忆] 已停止")

    # ---------- commands ----------
    @filter.command("记忆")
    async def memory_cmd(self, event: AstrMessageEvent):
        raw = (event.message_str or "").strip()
        for prefix in ("/记忆", "记忆"):
            if raw.startswith(prefix):
                raw = raw[len(prefix):]
                break
        action = raw.strip()

        if action.startswith("搜"):
            keyword = action[1:].strip()
            if not keyword:
                yield event.plain_result("要给个关键词呀，比如 /记忆 搜 名字")
                return
            hits = await asyncio.to_thread(self.search_sync, keyword, 5)
            if not hits:
                yield event.plain_result(f"存档里没找到「{keyword}」。")
                return
            lines = []
            for rec in hits:
                who = "主人" if rec.get("role") == "user" else "我"
                stamp = datetime.fromtimestamp(rec.get("ts", 0)).strftime("%m-%d %H:%M")
                lines.append(f"[{stamp}] {who}: {rec.get('text', '')[:120]}")
            yield event.plain_result("找到这些：\n" + "\n".join(lines))
            return

        if action.startswith("联想"):
            query = action[2:].strip() or "主人"
            hits = await self.retrieve_async(query, k=6)
            if not hits:
                yield event.plain_result("这条线索没想到什么。")
                return
            lines = [f"{'↳' if h['linked'] else '·'}[{h['kind']}] {h['text'][:60]}" for h in hits]
            chain = self.reason_chain(hits)
            if chain:
                lines.append(f'推理：{chain}')
            yield event.plain_result("\n".join(lines))
            return

        if action in ("事实", "长期"):
            if not self.l3:
                yield event.plain_result("长期记忆还是空的。")
                return
            lines = [f"· #{'/'.join(f.get('tags') or []) or '无'} {f['text']}" for f in sorted(self.l3, key=lambda x: x.get("last_seen", 0))]
            yield event.plain_result("我记得这些（已写进人格）：\n" + "\n".join(lines[-30:]))
            return

        if action in ("整理", "压缩"):
            if not event.is_admin():
                return
            yield event.plain_result("我把记忆理一理…")
            yield event.plain_result("整理完了：" + await self.organize_facts())
            return

        now = time.time()
        alive = [n for n in self.l2 if n.get("expires_at", 0) > now]
        total = 0
        files = list(self.archive_dir.glob("*.jsonl"))
        for path in files:
            try:
                total += sum(1 for _ in path.open(encoding="utf-8"))
            except Exception:
                pass
        yield event.plain_result(
            f"L1 最近：{self._int('l1_window', 10)} 句\n"
            f"L2 近 24 小时：{len(alive)} 条\n"
            f"L3 长期事实：{len(self.l3)} 条\n"
            f"L4 存档：{total} 句 / {len(files)} 天\n"
            f"联想图：{len(self._graph.get('nodes') or [])} 点 / {len(self._graph.get('links') or [])} 边\n"
            f"待压缩：{len(self.state.get('pending') or [])} 条\n遗忘记录：{(sum(1 for _ in (self.data_dir / 'forget_log.jsonl').open(encoding='utf-8')) if (self.data_dir / 'forget_log.jsonl').exists() else 0)} 条"
        )

    # 太书面/太像 AI 的说法 → 口语替换（只做很少的几条，避免误伤）
    STYLE_REWRITE = (
        ("这个口子不能开", "这事不能松口"),
        ("开这个口子", "通融这一回"),
        ("开个口子", "通融一下"),
        ("不能开这个口", "这事不能松口"),
        ("把口子堵上", "以后不这样了"),
        ("破例一次", "就这一次"),
        ("我理解你的感受", "我知道你难受"),
        ("你需要我做什么", "要我做点啥"),
        ("我可以为你提供", "我能帮你"),
        ("请告诉我更多", "再多说点"),
    )

    # 搭配错了的句子，用正则纠正（比字面表更灵活）
    STYLE_REGEX = (
        (r"课(?:要|如果)?是?不累", "要是课上得不累"),
        (r"课(?:要|如果)?是?累(?![得的])", "要是课上得累"),
        (r"回来还能剩(?:半个)?上午", "回来还能歇半天"),
        (r"上午还能剩", "上午还能空出来"),
        (r"作业吃(?:饱|饭)", "作业写完了"),
        (r"上午很(?:开心|高兴)", "上午过得挺舒服"),
    )

    # ---------- 出口清洗：去掉句尾句号 ----------
    async def _send_parts(self, event, parts):
        """把剩下的段落一条条补发（首条由主流程发）。"""
        try:
            await asyncio.sleep(1.2)
            for seg in (parts or [])[:3]:
                if not seg:
                    continue
                await event.send(MessageChain([Plain(seg)]))
                await asyncio.sleep(0.8)
        except Exception as exc:
            logger.debug(f"[Wave] 分段发送失败: {exc}")

    @filter.on_decorating_result()
    async def _drop_period(self, event: AstrMessageEvent):
        try:
            res = event.get_result()
            chain = getattr(res, "chain", None) if res else None
            if not chain:
                return
            for comp in chain:
                text = getattr(comp, "text", None)
                if not isinstance(text, str) or "。" not in text:
                    continue
                new = re.sub(r"。(?=\s*(?:\n|$))", "", text)
                for _bad, _good in self.STYLE_REWRITE:
                    if _bad in new:
                        new = new.replace(_bad, _good)
                for _pat, _rep in self.STYLE_REGEX:
                    try:
                        new = re.sub(_pat, _rep, new)
                    except Exception:
                        pass
                # 有空行分段 → 拆成多条消息发，而不是一条里带空行
                parts = [x.strip() for x in re.split(r"\n\s*\n+", new) if x.strip()]
                if len(parts) > 1 and bool(self.cfg("split_paragraphs", True)):
                    new = parts[0]
                    try:
                        asyncio.create_task(self._send_parts(event, parts[1:]))
                    except Exception:
                        pass
                if new != text:
                    try:
                        comp.text = new
                    except Exception:
                        pass
        except Exception:
            pass

    @filter.command("海浪")
    async def wave_cmd(self, event: AstrMessageEvent):
        now = time.time()
        alive = [n for n in self.l2 if n.get("expires_at", 0) > now]
        active = [f for f in self.l3 if not f.get("parked")]
        parked = [f for f in self.l3 if f.get("parked")]
        files = list(self.archive_dir.glob("*.jsonl"))
        total = 0
        for path in files:
            try:
                total += sum(1 for _ in path.open(encoding="utf-8"))
            except Exception:
                pass
        last = datetime.fromtimestamp(self.last_diary).strftime("%m-%d %H:%M") if self.last_diary else "还没写过"
        _floor = self.state.get("score_floor") or self.cfg("recall_score_floor", 1.2)
        _topics = self.state.get("topic_sim") or {}
        _bk = "有" if (self.data_dir / "backups").exists() else "无"
        yield event.plain_result(
            f"门槛 {_floor} · 话题阈值 {len(_topics)} 个 · 备份 {_bk}\n"
            f"L1 最近 {self._int('l1_window', 10)} 句 · L2 {len(alive)} 条 · "
            f"L3 {len(active)} 条（下沉 {len(parked)}）\n"
            f"L4 {total} 句 / {len(files)} 天 · 日记 {len(self.diary_entries)} 篇（上篇 {last}）\n"
            f"联想图 {len(self._graph.get('nodes') or [])} 点 / {len(self._graph.get('links') or [])} 边 · "
            f"标签 {len(getattr(self, '_tag_index', {}) or {})} 个\n"
            f"注入 {(int((self.state.get('inject_stats') or {}).get('chars', 0)) / max(1, int((self.state.get('inject_stats') or {}).get('n', 1)))):.0f} 字/轮 · "
            f"检索缓存命中 {self.state.get('cache_hits', 0)} 次\n"
            f"提示词缓存 {self._provider_stats().get('hit', '-')}% · "
            f"调优 chars={self._cache_plan().get('chars')} max={self._cache_plan().get('max')} "
            f"time={self._cache_plan().get('time')}"
        )

    @filter.command("日记")
    async def diary_cmd(self, event: AstrMessageEvent):
        raw = (event.message_str or "").strip()
        for prefix in ("/日记", "日记"):
            if raw.startswith(prefix):
                raw = raw[len(prefix):]
                break
        action = raw.strip()
        if action in ("合并", "digest"):
            if not event.is_admin():
                return
            yield event.plain_result("我把旧日记收一收…")
            await self._consolidate_diary()
            yield event.plain_result(f"现在共 {len(self.diary_entries)} 篇。")
            return
        if action in ("现在", "写"):
            if not event.is_admin():
                return
            yield event.plain_result("好，我翻翻最近的记录…")
            yield event.plain_result(await self.write_diary(force=True))
            return
        if not self.diary_entries:
            self._load_diary()
        if not self.diary_entries:
            yield event.plain_result("日记本还是空的。")
            return
        entry = self.diary_entries[-1]
        if action in ("大纲", "outline"):
            yield event.plain_result(self._diary_text(entry, False))
            return
        yield event.plain_result(self._diary_text(entry, True))
