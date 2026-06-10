"""Tests for the emotion detection engine."""
import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "module"))

from main import EmotionEngine


def test_detect_joy():
    engine = EmotionEngine()
    result = engine.detect("今天好开心呀")
    assert result["label"] in ("快乐", "欣喜"), f"Expected joy/surprise, got {result['label']}"
    assert result["score"] > 30


def test_detect_sadness():
    engine = EmotionEngine()
    result = engine.detect("好难过想哭")
    assert result["label"] in ("悲伤",), f"Expected sadness, got {result['label']}"


def test_compound_emotion():
    engine = EmotionEngine()
    engine.plutchik["joy"] = 80.0
    engine.plutchik["trust"] = 80.0
    result = engine.get_result()
    assert result["label"] == "爱", f"Expected 爱, got {result['label']}"
    assert result["intensity"] == "强"


def test_style_injection():
    engine = EmotionEngine()
    engine.plutchik["joy"] = 80.0
    engine.plutchik["trust"] = 80.0
    injection = engine.get_style_injection()
    assert "当前：" in injection
    assert "爱" in injection


def test_natural_fluctuation():
    engine = EmotionEngine()
    before = engine.plutchik["joy"]
    engine.natural_fluctuation()
    assert 0 <= engine.plutchik["joy"] <= 100


def test_multiple_detects():
    engine = EmotionEngine()
    engine.detect("好开心～")
    engine.detect("但是又有点难过")
    result = engine.get_result()
    assert "score" in result


if __name__ == "__main__":
    test_detect_joy()
    test_detect_sadness()
    test_compound_emotion()
    test_style_injection()
    test_natural_fluctuation()
    test_multiple_detects()
    print("All tests passed!")
