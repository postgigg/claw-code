"""Tests for universal validation rules (Fixes 1-8)."""
import os
import sys
import unittest
from pathlib import Path
from unittest.mock import patch

# Ensure the project root is importable
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
import claw_cli


# ---------------------------------------------------------------------------
# Fix 1: Duplicate CSS @layer block detection
# ---------------------------------------------------------------------------

class TestDuplicateLayer(unittest.TestCase):
    def test_duplicate_layer_flags(self):
        css = "@layer base { color: red; }\n@layer base { color: blue; }\n"
        issues = claw_cli._check_css_idioms(css)
        self.assertTrue(any("Duplicate" in desc for _, desc in issues))

    def test_single_layer_clean(self):
        css = "@layer base { color: red; }\n@layer utilities { display: flex; }\n"
        issues = claw_cli._check_css_idioms(css)
        self.assertFalse(any("Duplicate" in desc for _, desc in issues))


# ---------------------------------------------------------------------------
# Fix 2: Self-import detection
# ---------------------------------------------------------------------------

class TestSelfImport(unittest.TestCase):
    def _setup_tree(self, tmp, structure):
        """Create files in tmp. structure = {relative_path: content}."""
        for rel, content in structure.items():
            p = tmp / rel
            p.parent.mkdir(parents=True, exist_ok=True)
            p.write_text(content, encoding="utf-8")

    def test_self_import_via_alias(self):
        import tempfile
        with tempfile.TemporaryDirectory() as tmp:
            tmp = Path(tmp)
            self._setup_tree(tmp, {
                "src/lib/utils.ts": "import { x } from '@/lib/utils'\nexport const x = 1\n",
            })
            content = (tmp / "src/lib/utils.ts").read_text()
            with patch.object(claw_cli, "CWD", str(tmp)):
                # Reset cache so it picks up new CWD
                claw_cli._tsconfig_paths_cache = None
                issues = claw_cli._check_import_coherence(content, str(tmp / "src/lib/utils.ts"))
            self.assertTrue(any("Self-import" in desc for _, desc in issues),
                            f"Expected Self-import issue, got: {issues}")

    def test_self_import_via_index(self):
        import tempfile
        with tempfile.TemporaryDirectory() as tmp:
            tmp = Path(tmp)
            self._setup_tree(tmp, {
                "src/components/index.ts": "import { x } from '@/components'\nexport const x = 1\n",
            })
            content = (tmp / "src/components/index.ts").read_text()
            with patch.object(claw_cli, "CWD", str(tmp)):
                claw_cli._tsconfig_paths_cache = None
                issues = claw_cli._check_import_coherence(content, str(tmp / "src/components/index.ts"))
            self.assertTrue(any("Self-import" in desc for _, desc in issues),
                            f"Expected Self-import issue, got: {issues}")

    def test_cross_import_clean(self):
        import tempfile
        with tempfile.TemporaryDirectory() as tmp:
            tmp = Path(tmp)
            self._setup_tree(tmp, {
                "src/lib/utils.ts": "import { x } from '@/lib/other'\nexport const y = 1\n",
                "src/lib/other.ts": "export const x = 1\n",
            })
            content = (tmp / "src/lib/utils.ts").read_text()
            with patch.object(claw_cli, "CWD", str(tmp)):
                claw_cli._tsconfig_paths_cache = None
                issues = claw_cli._check_import_coherence(content, str(tmp / "src/lib/utils.ts"))
            self.assertFalse(any("Self-import" in desc for _, desc in issues),
                             f"Unexpected Self-import issue: {issues}")


# ---------------------------------------------------------------------------
# Fix 3: Structural export conflict
# ---------------------------------------------------------------------------

class TestStructuralExportConflict(unittest.TestCase):
    def test_component_plus_post_flags(self):
        content = (
            "export default function Page() { return <div>Hello</div> }\n"
            "export async function POST() { return Response.json({}) }\n"
        )
        issues = claw_cli._check_js_idioms(content)
        self.assertTrue(any("Structural conflict" in desc for _, desc in issues))

    def test_component_only_clean(self):
        content = "export default function Page() { return <div>Hello</div> }\n"
        issues = claw_cli._check_js_idioms(content)
        self.assertFalse(any("Structural conflict" in desc for _, desc in issues))

    def test_route_only_clean(self):
        content = (
            "export async function GET() { return Response.json({}) }\n"
            "export async function POST() { return Response.json({}) }\n"
        )
        issues = claw_cli._check_js_idioms(content)
        self.assertFalse(any("Structural conflict" in desc for _, desc in issues))


# ---------------------------------------------------------------------------
# Fix 4: Untyped callback parameter
# ---------------------------------------------------------------------------

class TestUntypedCallback(unittest.TestCase):
    def test_untyped_callback_flags(self):
        content = "supabase.auth.setAll(cookiesToSet) {\n  // set cookies\n}\n"
        issues = claw_cli._check_js_idioms(content, filepath="test.ts")
        self.assertTrue(any("Untyped callback" in desc for _, desc in issues),
                        f"Expected untyped callback, got: {issues}")

    def test_typed_callback_clean(self):
        content = "supabase.auth.setAll(cookiesToSet: CookieOptions[]) {\n  // set cookies\n}\n"
        issues = claw_cli._check_js_idioms(content, filepath="test.ts")
        self.assertFalse(any("Untyped callback" in desc for _, desc in issues))


# ---------------------------------------------------------------------------
# Fix 5: Browser API in SSR
# ---------------------------------------------------------------------------

class TestBrowserAPIInSSR(unittest.TestCase):
    def test_browser_api_in_ssr_flags(self):
        content = "const url = window.location.href\n"
        issues = claw_cli._check_js_idioms(content, filepath="src/app/page.tsx")
        self.assertTrue(any("Browser API" in desc for _, desc in issues),
                        f"Expected Browser API issue, got: {issues}")

    def test_browser_api_with_use_client_clean(self):
        content = "'use client'\nconst url = window.location.href\n"
        issues = claw_cli._check_js_idioms(content, filepath="src/app/page.tsx")
        self.assertFalse(any("Browser API" in desc for _, desc in issues))

    def test_browser_api_outside_app_dir(self):
        content = "window.innerWidth\n"
        issues = claw_cli._check_js_idioms(content, filepath="src/utils/helpers.ts")
        # Should use old module-level check, not SSR check
        self.assertTrue(any("Unguarded browser API" in desc for _, desc in issues),
                        f"Expected module-level browser API warning, got: {issues}")


# ---------------------------------------------------------------------------
# Fix 6: Client-only library in server components
# ---------------------------------------------------------------------------

class TestClientOnlyLibrary(unittest.TestCase):
    def test_client_lib_in_server_flags(self):
        content = "import { BarChart } from 'recharts'\nexport default function Page() { return <div/> }\n"
        issues = claw_cli._check_js_idioms(content, filepath="src/app/page.tsx")
        self.assertTrue(any("Client-only library" in desc for _, desc in issues),
                        f"Expected client-only library issue, got: {issues}")

    def test_client_lib_with_use_client_clean(self):
        content = "'use client'\nimport { BarChart } from 'recharts'\nexport default function Page() { return <div/> }\n"
        issues = claw_cli._check_js_idioms(content, filepath="src/app/page.tsx")
        self.assertFalse(any("Client-only library" in desc for _, desc in issues))


# ---------------------------------------------------------------------------
# Fix 7: Nonexistent API endpoint detection
# ---------------------------------------------------------------------------

class TestNonexistentEndpoint(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        import tempfile
        cls._tmpdir = tempfile.mkdtemp()
        cls.tmp = Path(cls._tmpdir)
        # Create API route structure
        routes = [
            "src/app/api/tasks/route.ts",
            "src/app/api/tasks/[id]/route.ts",
            "src/app/api/storage/[...path]/route.ts",
        ]
        for r in routes:
            p = cls.tmp / r
            p.parent.mkdir(parents=True, exist_ok=True)
            p.write_text("export async function GET() {}", encoding="utf-8")

    @classmethod
    def tearDownClass(cls):
        import shutil
        shutil.rmtree(cls._tmpdir, ignore_errors=True)

    def _check(self, content):
        with patch.object(claw_cli, "CWD", str(self.tmp)):
            claw_cli._tsconfig_paths_cache = None
            return claw_cli._check_import_coherence(
                content, str(self.tmp / "src/app/page.tsx"))

    def test_missing_endpoint_flags(self):
        issues = self._check("fetch('/api/nonexistent')\n")
        self.assertTrue(any("no matching route.ts" in desc for _, desc in issues),
                        f"Expected missing route issue, got: {issues}")

    def test_existing_endpoint_clean(self):
        issues = self._check("fetch('/api/tasks')\n")
        self.assertFalse(any("no matching route.ts" in desc for _, desc in issues),
                         f"Unexpected missing route issue: {issues}")

    def test_dynamic_endpoint_template_clean(self):
        issues = self._check("fetch(`/api/tasks/${id}`)\n")
        self.assertFalse(any("no matching route.ts" in desc for _, desc in issues),
                         f"Unexpected missing route issue: {issues}")

    def test_catchall_endpoint_clean(self):
        issues = self._check("fetch('/api/storage/images/thumb/large')\n")
        self.assertFalse(any("no matching route.ts" in desc for _, desc in issues),
                         f"Unexpected missing route issue: {issues}")


# ---------------------------------------------------------------------------
# Fix 8: Query filter string interpolation
# ---------------------------------------------------------------------------

class TestQueryInterpolation(unittest.TestCase):
    def test_query_interpolation_flags(self):
        content = "const result = supabase.from('items').or(`id.eq.${var}`)\n"
        score, issues = claw_cli._slop_score(content, "test.ts")
        interp_issues = [i for i in issues if "interpolation" in i[1].lower()]
        self.assertTrue(len(interp_issues) > 0, f"Expected interpolation issue, got: {issues}")
        # Check that score was deducted by -2
        self.assertTrue(any(i[2] == -2 for i in interp_issues))

    def test_query_no_interpolation_clean(self):
        content = "const result = supabase.from('items').or('id.eq.fixed')\n"
        score, issues = claw_cli._slop_score(content, "test.ts")
        interp_issues = [i for i in issues if "interpolation" in i[1].lower()]
        self.assertEqual(len(interp_issues), 0)


# ---------------------------------------------------------------------------
# Integration test: score delta on a crafted file
# ---------------------------------------------------------------------------

class TestIntegrationScoreDelta(unittest.TestCase):
    def test_score_delta_with_query_interpolation(self):
        """A file with .or() interpolation should lose exactly -2 per hit from Fix 8."""
        content = (
            "import { createClient } from '@supabase/supabase-js'\n"
            "const supabase = createClient('url', 'key')\n"
            "const result = await supabase.from('items').or(`id.eq.${someVar}`)\n"
        )
        score_with, issues_with = claw_cli._slop_score(content, "test.ts")
        # Same content but without interpolation
        content_clean = content.replace("`id.eq.${someVar}`", "'id.eq.fixed'")
        score_clean, issues_clean = claw_cli._slop_score(content_clean, "test.ts")
        # The interpolation version should score lower
        delta = score_clean - score_with
        self.assertEqual(delta, 2, f"Expected delta of 2, got {delta}")
        # Should have an interpolation issue
        self.assertTrue(any("interpolation" in i[1].lower() for i in issues_with))


if __name__ == "__main__":
    unittest.main()
