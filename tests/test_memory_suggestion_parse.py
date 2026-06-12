"""_parse_suggestion_text must survive thinking-model output shapes.

The /api/memory/extract route asks the chat model for a JSON array of memory
suggestions, but reasoning models wrap it in ```json fences or prose, and some
replies skip JSON entirely and emit a bullet list. The helper has to recover a
clean list of strings from all of those, and never return fence/bracket noise
as "suggestions".
"""

from routes.memory_routes import _parse_suggestion_text


def test_plain_json_array():
    assert _parse_suggestion_text('["likes coffee", "plays guitar"]') == [
        "likes coffee", "plays guitar",
    ]


def test_fenced_json_array():
    raw = '```json\n["likes coffee", "plays guitar"]\n```'
    assert _parse_suggestion_text(raw) == ["likes coffee", "plays guitar"]


def test_array_embedded_in_prose():
    raw = 'Here are the suggestions:\n["likes coffee"]\nHope that helps!'
    assert _parse_suggestion_text(raw) == ["likes coffee"]


def test_dict_items_use_text_field():
    raw = '[{"text": "likes coffee"}, {"text": "plays guitar"}]'
    assert _parse_suggestion_text(raw) == ["likes coffee", "plays guitar"]


def test_bullet_list_fallback_strips_prefixes():
    raw = "- likes coffee\n* plays guitar\n1. owns a dog"
    assert _parse_suggestion_text(raw) == [
        "likes coffee", "plays guitar", "owns a dog",
    ]


def test_fence_and_bracket_noise_lines_are_dropped():
    # A reply that is *almost* JSON but unparseable must not surface stray
    # fences/brackets as suggestions in the fallback line split.
    raw = "```json\n[\nnot quite json\n]\n```"
    assert _parse_suggestion_text(raw) == ["not quite json"]


def test_empty_and_whitespace_return_empty():
    assert _parse_suggestion_text("") == []
    assert _parse_suggestion_text("   \n  ") == []
    assert _parse_suggestion_text(None) == []
