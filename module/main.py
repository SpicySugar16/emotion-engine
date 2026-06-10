"""
Emotion Detection Module — Artic Protocol .amod

Plutchik 8-dimension emotion model with 16 compound emotions,
5 intensity levels, tone_map style injection, and autonomous
natural fluctuation (background timer, every 30 min).

Provides emotion.detect, emotion.style_inject, emotion.status,
and emotion.fluctuate services.
"""

import json
import os
import sys
import threading
import time
import random
import math
from pathlib import Path


DATA_DIR = Path(__file__).parent
EMOTION_MAP_PATH = DATA_DIR / "emotion_map.json"
TONE_MAP_PATH = DATA_DIR / "tone_map.json"
STATE_FILE = os.environ.get(
    "EMOTION_STATE_FILE",
    os.path.expanduser("~/.tremolite/data/emotion/state.json"),
)

PLUTCHIK_DIMS = [
    "joy", "sadness", "anger", "fear",
    "surprise", "disgust", "anticipation", "trust",
]

COMPOUND_PAIRS = [
    ("joy", "trust", "爱"),
    ("joy", "anticipation", "乐观"),
    ("joy", "surprise", "欣喜"),
    ("anger", "joy", "自豪"),
    ("trust", "fear", "服从"),
    ("fear", "surprise", "敬畏"),
    ("fear", "anticipation", "焦虑"),
    ("anger", "fear", "攻击性"),
    ("surprise", "anger", "愤怒"),
    ("surprise", "sadness", "不满"),
    ("disgust", "anger", "轻蔑"),
    ("sadness", "disgust", "悔恨"),
    ("sadness", "trust", "疏离"),
    ("anticipation", "disgust", "犬儒"),
    ("anticipation", "joy", "希望"),
    ("disgust", "joy", "病态"),
]

SINGLE_LABELS = {
    "joy": "快乐", "sadness": "悲伤", "anger": "愤怒",
    "fear": "恐惧", "surprise": "惊讶", "disgust": "厌恶",
    "anticipation": "期待", "trust": "信任",
}

SINGLE_THRESHOLDS = [85, 70, 55, 40]
COMPOUND_THRESHOLDS = [160, 130, 100, 70]

# 自然波动配置
CENTER = 50.0
SOFT_MIN = 10.0
SOFT_MAX = 90.0
FLUCTUATION_INTERVAL = 1800  # 30 分钟


class EmotionEngine:
    def __init__(self):
        self.plutchik = {d: 30.0 for d in PLUTCHIK_DIMS}
        self.plutchik["trust"] = 50.0
        self.plutchik["joy"] = 30.0
        self.plutchik["anticipation"] = 30.0
        self.last_update = time.time()
        self.last_fluctuation = None
        self._shutdown = threading.Event()
        self.emotion_map = self._load_json(EMOTION_MAP_PATH)
        self.tone_map = self._load_json(TONE_MAP_PATH)
        self._load_state()
        # 启动后台定时波动线程
        self._start_fluctuation_timer()

    @staticmethod
    def _load_json(path):
        if path.exists():
            with open(path, encoding="utf-8") as f:
                return json.load(f)
        return {}

    def _load_state(self):
        path = Path(STATE_FILE)
        if path.exists():
            try:
                with open(path, encoding="utf-8") as f:
                    data = json.load(f)
                for d in PLUTCHIK_DIMS:
                    if d in data.get("plutchik", {}):
                        self.plutchik[d] = float(data["plutchik"][d])
                self.last_update = data.get("last_update", time.time())
                lf = data.get("last_fluctuation")
                self.last_fluctuation = lf if lf else None
            except (json.JSONDecodeError, KeyError):
                pass

    def _save_state(self):
        path = Path(STATE_FILE)
        path.parent.mkdir(parents=True, exist_ok=True)
        data = {
            "plutchik": self.plutchik,
            "energy": 50.0,
            "last_update": time.time(),
        }
        if self.last_fluctuation is not None:
            data["last_fluctuation"] = self.last_fluctuation
        with open(path, "w", encoding="utf-8") as f:
            json.dump(data, f, indent=2, ensure_ascii=False)

    # ── 后台定时波动 ──────────────────────────────

    def _start_fluctuation_timer(self):
        """启动后台线程，每 30 分钟自动执行自然波动"""
        def _loop():
            while not self._shutdown.is_set():
                # 每 30 秒检查一次停止信号
                for _ in range(FLUCTUATION_INTERVAL // 30):
                    if self._shutdown.is_set():
                        return
                    time.sleep(30)
                self.natural_fluctuation()

        t = threading.Thread(target=_loop, daemon=True)
        t.start()

    def stop(self):
        """停止后台定时器（供框架 shutdown 时调用）"""
        self._shutdown.set()

    # ── 波动算法（全概率均值回归）────────────────

    def natural_fluctuation(self):
        """全概率均值回归 — 移植自 Hermes emotion-governor"""
        for dim in PLUTCHIK_DIMS:
            val = self.plutchik[dim]
            dist = abs(val - CENTER)
            normalized = min(1.0, dist / 50.0)

            # 回归概率：距中心越远越高
            p_toward = 0.5 + 0.42 * (normalized ** 0.9)

            # 软边界强化
            if val >= SOFT_MAX:
                edge = min(1.0, (val - SOFT_MAX) / 10.0) if val > SOFT_MAX else 0.35
                p_toward = max(p_toward, 0.82 + 0.16 * edge)
            elif val <= SOFT_MIN:
                edge = min(1.0, (SOFT_MIN - val) / 10.0) if val < SOFT_MIN else 0.35
                p_toward = max(p_toward, 0.82 + 0.16 * edge)

            # 幅度：中心附近波动大 → 远端波动小
            mean_mag = 1.2 + 4.8 * ((1.0 - normalized) ** 1.15)
            magnitude = max(1, min(6, round(random.gauss(mean_mag, 0.9))))

            # 外向阻尼（越靠近边界，向外走的幅度越小）
            if val >= SOFT_MAX:
                outward_damp = 0.35 if val <= 95 else 0.15
            elif val <= SOFT_MIN:
                outward_damp = 0.35 if val >= 5 else 0.15
            else:
                outward_damp = 1.0

            toward = random.random() < p_toward

            if val < CENTER:
                direction = 1 if toward else -1
            elif val > CENTER:
                direction = -1 if toward else 1
            else:
                direction = random.choice([-1, 1])

            step = float(magnitude)
            # 继续向外的步长受阻尼
            if (val >= CENTER and direction > 0) or (val <= CENTER and direction < 0):
                step = max(1.0, round(step * outward_damp))

            self.plutchik[dim] = max(0.0, min(100.0, val + direction * step))

        # 保险下限
        self.plutchik["joy"] = max(5.0, self.plutchik["joy"])
        self.plutchik["trust"] = max(20.0, self.plutchik["trust"])

        self.last_fluctuation = time.time()
        self._save_state()

    # ── 关键词检测 ────────────────────────────────

    def detect(self, text):
        if not text:
            return self.get_result()
        lower = text.lower()
        self._match_emotion_map(lower)
        self._match_builtin(lower)
        self._save_state()
        return self.get_result()

    def _match_emotion_map(self, text):
        for section in ("_verb_emotions", "_adjective_emotions"):
            words = self.emotion_map.get(section, {})
            for word, emotions in words.items():
                if word in text:
                    for dim, delta in emotions.items():
                        if dim in self.plutchik:
                            self.plutchik[dim] = min(
                                100.0, self.plutchik[dim] + abs(delta) * 0.3
                            )

    def _match_builtin(self, text):
        triggers = {
            ("开心", "高兴", "快乐", "幸福", "哈哈", "好棒", "太好了"): ("joy", 15),
            ("难过", "伤心", "好累", "不开心", "难受", "想哭"): ("sadness", 20),
            ("生气", "气死", "讨厌", "烦", "滚"): ("anger", 20),
            ("真的吗", "哇", "天哪", "想不到", "居然"): ("surprise", 20),
            ("相信", "交给你", "靠你了", "听话", "放心"): ("trust", 15),
            ("想要", "做吧", "开始", "继续"): ("anticipation", 15),
            ("害怕", "担心", "不安", "救命", "怕"): ("fear", 20),
            ("恶心", "臭", "难吃"): ("disgust", 20),
        }
        for keywords, (dim, delta) in triggers.items():
            for kw in keywords:
                if kw in text:
                    self.plutchik[dim] = min(100.0, self.plutchik[dim] + delta)
                    break

    # ── 复合情绪检测 ──────────────────────────────

    def get_result(self):
        best_compound = None
        best_score = 0
        best_dims = None
        for da, db, label in COMPOUND_PAIRS:
            va = self.plutchik.get(da, 0)
            vb = self.plutchik.get(db, 0)
            if va >= 40 and vb >= 40:
                total = va + vb
                if total >= 80 and total > best_score:
                    best_compound = label
                    best_score = total
                    best_dims = (da, db)
        if best_compound:
            return {
                "label": best_compound,
                "intensity": self._intensity(COMPOUND_THRESHOLDS, best_score),
                "score": best_score,
                "triggers": [(best_dims[0], self.plutchik[best_dims[0]]),
                             (best_dims[1], self.plutchik[best_dims[1]])],
                "plutchik": dict(self.plutchik),
            }
        dim = max(PLUTCHIK_DIMS, key=lambda d: self.plutchik[d])
        val = self.plutchik[dim]
        label = SINGLE_LABELS.get(dim, "平静")
        return {
            "label": label,
            "intensity": self._intensity(SINGLE_THRESHOLDS, val),
            "score": val,
            "triggers": [(dim, val)],
            "plutchik": dict(self.plutchik),
        }

    @staticmethod
    def _intensity(thresholds, val):
        for i, t in enumerate(thresholds):
            if val >= t:
                return ["极强", "强", "中", "弱"][i]
        return "微"

    # ── 风格注入 ──────────────────────────────────

    def get_style_injection(self, result=None):
        if result is None:
            result = self.get_result()
        label, intensity = result["label"], result["intensity"]
        entry = self.tone_map.get(label)
        if entry:
            level = entry.get("levels", {}).get(intensity)
            if level:
                style = level.get("style", "")
                emoji = level.get("emoji", "")
                tpl = level.get("模板", {})
                examples = tpl.get("句式示例", []) if isinstance(tpl, dict) else []
                particles = tpl.get("语气词", []) if isinstance(tpl, dict) else []
                forbidden = tpl.get("禁用词", []) if isinstance(tpl, dict) else []
                lines = [f"当前：{label} 强度：{intensity}", f"风格：{style}"]
                if examples:
                    lines.append("模板：" + "".join(f"「{s}」" for s in examples))
                if particles:
                    lines.append(f"语气词：{'、'.join(particles)}")
                if forbidden:
                    lines.append(f"禁用：{'、'.join(forbidden)}")
                if emoji:
                    lines.append(f"emoji：{emoji}")
                lines.extend(["", "以上为当前情绪和说话风格要求。",
                              "自然贴合以上风格。务必不能直接使用模板。"])
                return "\n".join(lines)
        return f"当前：{label} 强度：{intensity}\n风格：日常语气\n\n以上为当前情绪和说话风格要求。"


def main():
    engine = EmotionEngine()
    module_id = os.environ.get("ARTIC_MODULE_ID", "emotion.engine")
    log = lambda msg: print(f"[{module_id}] {msg}", file=sys.stderr)
    log("Emotion engine started (with autonomous fluctuation timer)")

    try:
        for line in sys.stdin:
            line = line.strip()
            if not line:
                continue
            try:
                envelope = json.loads(line)
            except json.JSONDecodeError:
                continue
            if envelope.get("kind") != "request":
                continue

            payload = envelope.get("payload", {})
            service = envelope.get("to", "")
            msg_id = envelope["id"]
            sender = envelope.get("from", "engine")

            if service in ("emotion.detect", "emotion.style_inject", "emotion.status", "emotion.fluctuate"):
                if service == "emotion.detect":
                    text = payload.get("text", "")
                    result = engine.detect(text)
                elif service == "emotion.style_inject":
                    text = payload.get("text", "")
                    if text:
                        engine.detect(text)
                    result = engine.get_result()
                    result = {"emotion": result, "injection": engine.get_style_injection(result)}
                elif service == "emotion.status":
                    result = {"emotion": engine.get_result(), "plutchik": engine.plutchik}
                else:
                    engine.natural_fluctuation()
                    result = {"emotion": engine.get_result(), "note": "fluctuation applied"}

                response = {
                    "id": msg_id, "from": module_id, "to": sender,
                    "kind": "response", "payload": result,
                    "reply_to": msg_id, "timestamp": int(time.time() * 1000),
                }
            else:
                response = {
                    "id": msg_id, "from": module_id, "to": sender,
                    "kind": "error",
                    "payload": {"code": "SERVICE_NOT_FOUND", "message": f"Unknown: {service}"},
                    "reply_to": msg_id, "timestamp": int(time.time() * 1000),
                }

            sys.stdout.write(json.dumps(response, ensure_ascii=False) + "\n")
            sys.stdout.flush()
    finally:
        engine.stop()


if __name__ == "__main__":
    main()
