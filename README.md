# Emotion Detection Engine (`.amod`)

An Artic Protocol module for emotion detection based on the **Plutchik wheel of emotions** model.

- **8 dimensions**: joy, sadness, anger, fear, surprise, disgust, anticipation, trust
- **16 compound emotions**: love, optimism, joy, pride, submission, awe, anxiety, aggression, anger, disapproval, contempt, remorse, detachment, cynicism, hope, morbidity
- **5 intensity levels**: 极强/strong, 强/medium-strong, 中/medium, 弱/weak, 微/trace
- **tone_map injection**: generates style guidance text for LLM response generation

## Services

| Service | Input | Output |
|---------|-------|--------|
| `emotion.detect` | `{"text": "..."}` | `{"label", "intensity", "score", "plutchik", ...}` |
| `emotion.style_inject` | `{"text": "..."}` | `{"emotion": {...}, "injection": "style text..."}` |
| `emotion.status` | `{}` | Current emotion state + plutchik vector |
| `emotion.fluctuate` | `{}` | Apply natural mean-reversion fluctuation |

## Quick Start

```bash
# Install the module
tremolite module install ./emotion-engine-v0.1.0.amod

# Test: detect emotion from text
echo '{"id":"t1","kind":"request","to":"emotion.detect","payload":{"text":"今天好开心"}}' | python3 module/main.py
```

## Package Contents

```
emotion-engine-v0.1.0/
├── manifest.toml          # Module metadata
├── module/
│   ├── main.py            # Entry point
│   ├── emotion_map.json   # 382 keyword→plutchik mappings
│   └── tone_map.json      # 24 emotion × 5 level style templates
├── tests/
│   └── test_detect.py     # Unit tests
└── README.md
```

## Algorithm

1. Text input → keyword matching (emotion_map.json) + built-in Chinese word triggers
2. Update 8-dimension Plutchik vector
3. Check 16 compound pairs: both dims ≥ 40 AND sum ≥ 80
4. Highest-scoring compound wins; fallback to single dimension
5. Look up tone_map for style injection text
