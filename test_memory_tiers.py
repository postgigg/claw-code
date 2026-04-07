#!/usr/bin/env python3
"""Integration test for the tiered memory system (hot/warm/cold)."""

import json
import os
import shutil
import sys
import time
import datetime
from pathlib import Path

# Add project to path and import memory functions
sys.path.insert(0, os.path.dirname(__file__))
import claw_cli

MEMORY_DIR = claw_cli.MEMORY_DIR
BACKUP_DIR = MEMORY_DIR.parent / "memory_backup_test"

passed = 0
failed = 0

def test(name, condition, detail=""):
    global passed, failed
    if condition:
        print(f"  PASS: {name}")
        passed += 1
    else:
        print(f"  FAIL: {name} {detail}")
        failed += 1

def setup():
    """Backup existing memories and start clean."""
    if MEMORY_DIR.exists():
        if BACKUP_DIR.exists():
            shutil.rmtree(BACKUP_DIR)
        shutil.copytree(MEMORY_DIR, BACKUP_DIR)
        shutil.rmtree(MEMORY_DIR)
    claw_cli._ensure_memory_dir()
    claw_cli._memory_cache["entries"] = None
    claw_cli._memory_cache["mtime"] = 0

def teardown():
    """Restore original memories."""
    if MEMORY_DIR.exists():
        shutil.rmtree(MEMORY_DIR)
    if BACKUP_DIR.exists():
        shutil.copytree(BACKUP_DIR, MEMORY_DIR)
        shutil.rmtree(BACKUP_DIR)

# =========================================================================
print("\n=== Test 1: Save memory -> v2 schema with tier=warm ===")
setup()

result = claw_cli.tool_memory_save({
    "key": "tech_stack",
    "value": "Next.js + TypeScript + Supabase",
    "category": "project"
})
test("Save returns success", "Saved to memory" in result)

# Read the raw JSON
filepath = MEMORY_DIR / "project" / "tech_stack.json"
test("File created", filepath.exists())

entry = json.loads(filepath.read_text(encoding="utf-8"))
test("schema_version is 2", entry.get("schema_version") == 2)
# tier may be hot if maintenance ran (last_accessed=now -> classify as hot)
test("tier is hot (maintenance auto-promotes new entry)", entry.get("tier") == "hot", f"got {entry.get('tier')}")
test("pinned is False", entry.get("pinned") == False)
test("access_count is 0", entry.get("access_count") == 0)
test("has related_keywords", len(entry.get("related_keywords", [])) > 0)
test("has demotion_strikes", entry.get("demotion_strikes") == 0)
test("has saved_at", "saved_at" in entry)
test("has last_accessed", "last_accessed" in entry)
test("no _filepath in JSON", "_filepath" not in entry, f"found _filepath={entry.get('_filepath')}")

print(f"  Keywords: {entry.get('related_keywords')}")

# =========================================================================
print("\n=== Test 2: Search bumps access_count and promotes to hot ===")
claw_cli._memory_cache["entries"] = None  # reset cache

for i in range(3):
    result = claw_cli.tool_memory_search({"query": "tech_stack"})
    test(f"Search {i+1} finds entry", "tech_stack" in result)

# Re-read JSON
entry = json.loads(filepath.read_text(encoding="utf-8"))
test("access_count is 3 after 3 searches", entry.get("access_count") == 3, f"got {entry.get('access_count')}")
test("tier promoted to hot", entry.get("tier") == "hot", f"got {entry.get('tier')}")

# =========================================================================
print("\n=== Test 3: Tier tags in search results ===")
claw_cli._memory_cache["entries"] = None
result = claw_cli.tool_memory_search({"query": "tech_stack"})
test("Search result has /H tier tag", "/H]" in result, f"result: {result[:100]}")

# =========================================================================
print("\n=== Test 4: Hot memories appear in load_memories_for_context ===")
claw_cli._memory_cache["entries"] = None
context = claw_cli.load_memories_for_context()
test("Hot memory in context", "tech_stack" in context, f"context: {context[:200]}")

# =========================================================================
print("\n=== Test 5: Warm memory NOT in context (only hot) ===")
claw_cli.tool_memory_save({
    "key": "ui_framework",
    "value": "Tailwind CSS for styling",
    "category": "project"
})
claw_cli._memory_cache["entries"] = None
context = claw_cli.load_memories_for_context()
test("Warm memory NOT in hot context", "ui_framework" not in context)

# =========================================================================
print("\n=== Test 6: Auto-search warm memories by keyword overlap ===")
claw_cli._memory_cache["entries"] = None
warm_result = claw_cli._auto_search_warm_memories("I want to use tailwind CSS for styling the components")
test("Warm auto-search finds ui_framework", "ui_framework" in warm_result or "Tailwind" in warm_result,
     f"result: '{warm_result[:200]}'")

# =========================================================================
print("\n=== Test 7: Auto-search requires 2+ keyword matches ===")
claw_cli._memory_cache["entries"] = None
weak_result = claw_cli._auto_search_warm_memories("hello world")
test("Single-keyword message returns nothing", weak_result == "")

# =========================================================================
print("\n=== Test 8: /memory stats counts ===")
claw_cli._memory_cache["entries"] = None
entries = claw_cli._load_all_memories()
hot_count = sum(1 for e in entries if e.get("tier") == "hot")
warm_count = sum(1 for e in entries if e.get("tier") == "warm")
test("Hot count >= 1", hot_count >= 1, f"got {hot_count}")
test("Total entries >= 2", len(entries) >= 2, f"got {len(entries)}")

# =========================================================================
print("\n=== Test 9: Pin/unpin ===")
# Pin ui_framework
entries = claw_cli._load_all_memories()
for e in entries:
    if e.get("key") == "ui_framework":
        e["pinned"] = True
        e["tier"] = "hot"
        e["demotion_strikes"] = 0
        claw_cli._save_memory_entry(e, Path(e["_filepath"]))
        break

claw_cli._memory_cache["entries"] = None
context = claw_cli.load_memories_for_context()
test("Pinned entry appears in hot context", "ui_framework" in context)
test("Pinned entry shows [pinned]", "[pinned]" in context)

# =========================================================================
print("\n=== Test 10: V1 -> V2 migration ===")
# Write a v1 entry (no schema_version, no tier, no keywords, etc.)
v1_dir = MEMORY_DIR / "legacy"
v1_dir.mkdir(parents=True, exist_ok=True)
v1_entry = {
    "key": "old_decision",
    "value": "We chose PostgreSQL for the database",
    "category": "legacy",
    "saved_at": "2025-01-15T10:00:00"
}
v1_path = v1_dir / "old_decision.json"
v1_path.write_text(json.dumps(v1_entry, indent=2), encoding="utf-8")

claw_cli._memory_cache["entries"] = None
entries = claw_cli._load_all_memories()
migrated = [e for e in entries if e.get("key") == "old_decision"]
test("V1 entry found after migration", len(migrated) == 1)
if migrated:
    m = migrated[0]
    test("Migration adds schema_version=2", m.get("schema_version") == 2)
    test("Migration adds tier=warm", m.get("tier") == "warm")
    test("Migration adds pinned=False", m.get("pinned") == False)
    test("Migration adds access_count=0", m.get("access_count") == 0)
    test("Migration adds demotion_strikes=0", m.get("demotion_strikes") == 0)
    test("Migration extracts keywords", len(m.get("related_keywords", [])) > 0)
    test("Migration preserves saved_at", m.get("saved_at") == "2025-01-15T10:00:00")
    # V1 entry NOT rewritten to disk on migration-read (lazy writeback)
    raw = json.loads(v1_path.read_text(encoding="utf-8"))
    test("V1 file NOT rewritten on read (lazy)", "schema_version" not in raw)

# =========================================================================
print("\n=== Test 11: Demotion 2-strike rule ===")
# Create an entry that's been hot but hasn't been accessed in 5 days
stale_dir = MEMORY_DIR / "stale"
stale_dir.mkdir(parents=True, exist_ok=True)
five_days_ago = (datetime.datetime.now() - datetime.timedelta(days=5)).isoformat()
stale_entry = {
    "schema_version": 2,
    "key": "stale_hot",
    "value": "This was hot but is now stale",
    "category": "stale",
    "tier": "hot",
    "pinned": False,
    "saved_at": five_days_ago,
    "last_accessed": five_days_ago,
    "access_count": 1,
    "related_keywords": ["stale", "hot", "test"],
    "demotion_strikes": 0,
}
stale_path = stale_dir / "stale_hot.json"
stale_path.write_text(json.dumps(stale_entry, indent=2), encoding="utf-8")

# Run maintenance once - should get 1 strike, NOT demote
claw_cli._memory_cache["entries"] = None
claw_cli._run_memory_maintenance()

raw = json.loads(stale_path.read_text(encoding="utf-8"))
test("First strike: demotion_strikes=1", raw.get("demotion_strikes") == 1, f"got {raw.get('demotion_strikes')}")
test("First strike: still hot", raw.get("tier") == "hot", f"got {raw.get('tier')}")

# Run maintenance again - should demote to warm
claw_cli._memory_cache["entries"] = None
claw_cli._run_memory_maintenance()

raw = json.loads(stale_path.read_text(encoding="utf-8"))
test("Second strike: demoted to warm", raw.get("tier") == "warm", f"got {raw.get('tier')}")
test("Second strike: strikes reset to 0", raw.get("demotion_strikes") == 0, f"got {raw.get('demotion_strikes')}")

# =========================================================================
print("\n=== Test 12: Pinned entries never demoted ===")
pinned_dir = MEMORY_DIR / "pinned_test"
pinned_dir.mkdir(parents=True, exist_ok=True)
old_date = (datetime.datetime.now() - datetime.timedelta(days=60)).isoformat()
pinned_entry = {
    "schema_version": 2,
    "key": "pinned_important",
    "value": "This is pinned and should never be demoted",
    "category": "pinned_test",
    "tier": "hot",
    "pinned": True,
    "saved_at": old_date,
    "last_accessed": old_date,
    "access_count": 0,
    "related_keywords": ["pinned", "important"],
    "demotion_strikes": 0,
}
pinned_path = pinned_dir / "pinned_important.json"
pinned_path.write_text(json.dumps(pinned_entry, indent=2), encoding="utf-8")

claw_cli._memory_cache["entries"] = None
claw_cli._run_memory_maintenance()
claw_cli._memory_cache["entries"] = None
claw_cli._run_memory_maintenance()

raw = json.loads(pinned_path.read_text(encoding="utf-8"))
test("Pinned entry stays hot after 2 cycles", raw.get("tier") == "hot")
test("Pinned entry has 0 strikes", raw.get("demotion_strikes") == 0)

# =========================================================================
print("\n=== Test 13: Atomic write (no .tmp files left) ===")
tmp_files = list(MEMORY_DIR.rglob("*.json.tmp"))
test("No .tmp files remain", len(tmp_files) == 0, f"found {tmp_files}")

# =========================================================================
print("\n=== Test 14: Overwrite preserves access stats ===")
claw_cli._memory_cache["entries"] = None
# tech_stack had access_count=4+ from searches, now overwrite value
old_entry = json.loads((MEMORY_DIR / "project" / "tech_stack.json").read_text(encoding="utf-8"))
old_ac = old_entry.get("access_count", 0)

claw_cli.tool_memory_save({
    "key": "tech_stack",
    "value": "Next.js + TypeScript + Supabase + Redis",
    "category": "project"
})

new_entry = json.loads((MEMORY_DIR / "project" / "tech_stack.json").read_text(encoding="utf-8"))
test("Overwrite preserves access_count", new_entry.get("access_count") == old_ac,
     f"was {old_ac}, now {new_entry.get('access_count')}")
test("Overwrite updates value", "Redis" in new_entry.get("value", ""))

# =========================================================================
print("\n=== Test 15: Maintenance throttle ===")
claw_cli._last_maintenance = time.time()  # just ran
claw_cli._memory_cache["entries"] = None

# This should NOT run maintenance (within 5-min window)
import unittest.mock as mock
with mock.patch.object(claw_cli, '_run_memory_maintenance') as mocked:
    claw_cli._maybe_run_maintenance()
    test("Maintenance throttled (not called within 5min)", not mocked.called)

# Force expired
claw_cli._last_maintenance = time.time() - 301
with mock.patch.object(claw_cli, '_run_memory_maintenance') as mocked:
    claw_cli._maybe_run_maintenance()
    test("Maintenance runs when 5min elapsed", mocked.called)

# =========================================================================
print("\n=== Test 16: Keyword extraction ===")
kw = claw_cli._extract_keywords("api_key", "The Stripe payment integration uses webhooks")
test("Stop words filtered", "the" not in kw)
test("Domain words kept", "stripe" in kw or "payment" in kw)
test("Key included", "api_key" in kw)
test("Cap at 20", len(kw) <= 20)

# =========================================================================
print("\n=== Test 17: _classify_tier logic ===")
now = datetime.datetime.now()

# Accessed 1 day ago -> hot
e1 = {"last_accessed": (now - datetime.timedelta(days=1)).isoformat(), "access_count": 1}
test("1 day ago -> hot", claw_cli._classify_tier(e1) == "hot")

# 3+ accesses, 10 days ago -> hot
e2 = {"last_accessed": (now - datetime.timedelta(days=10)).isoformat(), "access_count": 5}
test("5 accesses, 10 days -> hot", claw_cli._classify_tier(e2) == "hot")

# 1 access, 15 days ago -> warm
e3 = {"last_accessed": (now - datetime.timedelta(days=15)).isoformat(), "access_count": 1}
test("1 access, 15 days -> warm", claw_cli._classify_tier(e3) == "warm")

# 0 accesses, 45 days ago -> cold
e4 = {"last_accessed": (now - datetime.timedelta(days=45)).isoformat(), "access_count": 0}
test("0 accesses, 45 days -> cold", claw_cli._classify_tier(e4) == "cold")


# =========================================================================
# Cleanup
teardown()

print(f"\n{'='*60}")
print(f"  Results: {passed} passed, {failed} failed out of {passed+failed} tests")
print(f"{'='*60}")
sys.exit(0 if failed == 0 else 1)
