"""MemoryManager.load_all caches on the file's (mtime, size).

load_all runs on every chat turn (memory injection), so it must avoid
re-reading + re-parsing the whole file when nothing changed — while still
reflecting any write (internal save or external editor) and never handing back
the cached objects by reference.
"""


def _entry(text, owner="u1"):
    return {"text": text, "source": "user", "category": "fact", "owner": owner}


def test_load_all_uses_cache_until_file_changes(tmp_path, monkeypatch):
    from src.memory import MemoryManager

    mgr = MemoryManager(str(tmp_path))
    mgr.save([_entry("first fact")])

    # Prime the cache.
    first = mgr.load_all()
    assert any(e["text"] == "first fact" for e in first)

    # A second read with no file change must NOT re-open the file.
    reads = {"n": 0}
    real_open = open

    def counting_open(path, *a, **k):
        if str(path) == mgr.memory_file:
            reads["n"] += 1
        return real_open(path, *a, **k)

    monkeypatch.setattr("builtins.open", counting_open)
    cached = mgr.load_all()
    assert reads["n"] == 0  # served from cache, no file read
    assert [e["text"] for e in cached] == [e["text"] for e in first]


def test_load_all_returns_copies_not_cached_objects(tmp_path):
    from src.memory import MemoryManager

    mgr = MemoryManager(str(tmp_path))
    mgr.save([_entry("mutable fact")])

    a = mgr.load_all()
    a[0]["text"] = "MUTATED"
    b = mgr.load_all()
    # Mutating the returned list must not corrupt the cache.
    assert all(e["text"] != "MUTATED" for e in b)


def test_load_all_reparses_after_save(tmp_path):
    from src.memory import MemoryManager

    mgr = MemoryManager(str(tmp_path))
    mgr.save([_entry("one")])
    assert {e["text"] for e in mgr.load_all()} == {"one"}

    # A save bumps the file mtime/size → next load reflects the new entries.
    mgr.save([_entry("one"), _entry("two")])
    assert {e["text"] for e in mgr.load_all()} == {"one", "two"}
