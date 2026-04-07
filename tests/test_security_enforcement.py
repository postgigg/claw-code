"""Tests for security enforcement system — scan patterns, context builder, config enhancers."""
import os
import sys
import re
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

# Ensure the project root is importable
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
import claw_cli


# ---------------------------------------------------------------------------
# Test: _load_security_json
# ---------------------------------------------------------------------------

class TestLoadSecurityJson(unittest.TestCase):
    def test_loads_valid_json(self):
        # Reset cache to force reload
        claw_cli._security_json_cache = None
        data = claw_cli._load_security_json()
        self.assertIsNotNone(data, "security.json should load successfully")
        self.assertIn("header_configs", data)
        self.assertIn("security_rules", data)
        self.assertIn("security_scan_patterns", data)
        self.assertIsInstance(data["security_rules"], list)
        self.assertGreaterEqual(len(data["security_rules"]), 20)

    def test_caching_works(self):
        claw_cli._security_json_cache = None
        first = claw_cli._load_security_json()
        second = claw_cli._load_security_json()
        self.assertIs(first, second, "Subsequent calls should return cached object")


# ---------------------------------------------------------------------------
# Test: _detect_security_profile
# ---------------------------------------------------------------------------

class TestDetectSecurityProfile(unittest.TestCase):
    def test_saas_detection(self):
        self.assertEqual(claw_cli._detect_security_profile("saas dashboard for teams"), "saas")
        self.assertEqual(claw_cli._detect_security_profile("admin CRM panel"), "saas")

    def test_ecommerce_detection(self):
        self.assertEqual(claw_cli._detect_security_profile("ecommerce store with checkout"), "ecommerce")
        self.assertEqual(claw_cli._detect_security_profile("online shop for shoes"), "ecommerce")

    def test_default_is_saas(self):
        self.assertEqual(claw_cli._detect_security_profile("some random project"), "saas")
        self.assertEqual(claw_cli._detect_security_profile(""), "saas")


# ---------------------------------------------------------------------------
# Test: _build_security_context
# ---------------------------------------------------------------------------

class TestBuildSecurityContext(unittest.TestCase):
    def test_returns_string(self):
        claw_cli._security_json_cache = None
        ctx = claw_cli._build_security_context("saas dashboard")
        self.assertIsInstance(ctx, str)
        self.assertGreater(len(ctx), 100)

    def test_includes_headers(self):
        ctx = claw_cli._build_security_context("saas dashboard")
        self.assertIn("X-Content-Type-Options", ctx)
        self.assertIn("Content-Security-Policy", ctx)
        self.assertIn("X-Frame-Options", ctx)

    def test_empty_desc_works(self):
        ctx = claw_cli._build_security_context("")
        self.assertIsInstance(ctx, str)
        # Should still include rules and headers (default saas profile)
        self.assertIn("Security Rules", ctx)

    def test_respects_token_ceiling(self):
        ctx = claw_cli._build_security_context("saas dashboard with everything")
        # Should not exceed ~12K chars significantly
        self.assertLess(len(ctx), 15000)


# ---------------------------------------------------------------------------
# Test: _scan_security_quality
# ---------------------------------------------------------------------------

class TestScanSecurityQuality(unittest.TestCase):
    def _make_project(self, files):
        """Create temp project with given files dict {rel_path: content}."""
        tmp = tempfile.mkdtemp()
        pdir = Path(tmp)
        file_contents = {}
        for rel, content in files.items():
            fp = pdir / rel
            fp.parent.mkdir(parents=True, exist_ok=True)
            fp.write_text(content, encoding="utf-8")
            file_contents[fp] = content
        return pdir, file_contents

    def test_detects_sql_injection(self):
        claw_cli._SECURITY_SCAN_PATTERNS = None
        pdir, fc = self._make_project({
            "app/api/users/route.ts": 'const result = await db.query(`SELECT * FROM users WHERE id = ${req.params.id}`);\n'
        })
        issues = claw_cli._scan_security_quality(str(pdir), fc)
        types = [i["type"] for i in issues]
        self.assertIn("security_sql_injection", types)

    def test_detects_wildcard_cors(self):
        claw_cli._SECURITY_SCAN_PATTERNS = None
        pdir, fc = self._make_project({
            "app/api/config.ts": "const corsOptions = { origin: '*' };\n"
        })
        issues = claw_cli._scan_security_quality(str(pdir), fc)
        types = [i["type"] for i in issues]
        self.assertIn("security_wildcard_cors", types)

    def test_detects_exposed_server_secret_next(self):
        claw_cli._SECURITY_SCAN_PATTERNS = None
        pdir, fc = self._make_project({
            "lib/config.ts": "const key = NEXT_PUBLIC_SECRET_KEY;\n"
        })
        issues = claw_cli._scan_security_quality(str(pdir), fc)
        types = [i["type"] for i in issues]
        self.assertIn("security_exposed_server_secret", types)

    def test_detects_exposed_server_secret_vite(self):
        claw_cli._SECURITY_SCAN_PATTERNS = None
        pdir, fc = self._make_project({
            "src/config.ts": "const key = VITE_SECRET_KEY;\n"
        })
        issues = claw_cli._scan_security_quality(str(pdir), fc)
        types = [i["type"] for i in issues]
        self.assertIn("security_exposed_server_secret", types)

    def test_clean_code_passes(self):
        claw_cli._SECURITY_SCAN_PATTERNS = None
        pdir, fc = self._make_project({
            "app/api/users/route.ts": (
                "import { createClient } from '@/lib/supabase/server';\n"
                "export async function GET(request: Request) {\n"
                "  const supabase = createClient(cookies());\n"
                "  const { data } = await supabase.from('users').select('*');\n"
                "  return NextResponse.json(data);\n"
                "}\n"
            )
        })
        issues = claw_cli._scan_security_quality(str(pdir), fc)
        security_issues = [i for i in issues if i["type"].startswith("security_")]
        self.assertEqual(len(security_issues), 0, f"Clean code should have no security issues, got: {security_issues}")

    def test_middleware_suppresses_missing_auth(self):
        """Projects with middleware.ts containing auth should not flag missing_auth_check."""
        claw_cli._SECURITY_SCAN_PATTERNS = None
        pdir, fc = self._make_project({
            "middleware.ts": "import { getUser } from './lib/auth';\nexport default function middleware() { getUser(); }\n",
            "next.config.js": "module.exports = { headers() { return [{ key: 'X-Content-Type-Options', value: 'nosniff' }] } };\n",
            "app/api/data/route.ts": "export async function GET(req) { return Response.json({}); }\n"
        })
        issues = claw_cli._scan_security_quality(str(pdir), fc)
        auth_issues = [i for i in issues if i["type"] == "security_missing_auth_check"]
        self.assertEqual(len(auth_issues), 0, "middleware auth should suppress missing_auth_check")


# ---------------------------------------------------------------------------
# Test: cleartext_password (false positive avoidance)
# ---------------------------------------------------------------------------

class TestCleartextPassword(unittest.TestCase):
    def _make_project(self, files):
        tmp = tempfile.mkdtemp()
        pdir = Path(tmp)
        file_contents = {}
        for rel, content in files.items():
            fp = pdir / rel
            fp.parent.mkdir(parents=True, exist_ok=True)
            fp.write_text(content, encoding="utf-8")
            file_contents[fp] = content
        return pdir, file_contents

    def test_flags_hardcoded_password(self):
        claw_cli._SECURITY_SCAN_PATTERNS = None
        pdir, fc = self._make_project({
            "lib/seed.js": 'const password = "hunter2hunter2";\n'  # 14 chars, no bcrypt
        })
        # Remove seed from exceptions for this test by checking file without seed in name
        pdir2, fc2 = self._make_project({
            "lib/config.js": 'const password = "hunter2hunter2";\n'
        })
        issues = claw_cli._scan_security_quality(str(pdir2), fc2)
        types = [i["type"] for i in issues]
        self.assertIn("security_cleartext_password", types)

    def test_no_flag_for_variable_assignment(self):
        """password = req.body.password should NOT be flagged (no quotes on RHS)."""
        claw_cli._SECURITY_SCAN_PATTERNS = None
        pdir, fc = self._make_project({
            "app/api/auth/route.ts": "const password = req.body.password;\n"
        })
        issues = claw_cli._scan_security_quality(str(pdir), fc)
        pwd_issues = [i for i in issues if i["type"] == "security_cleartext_password"]
        self.assertEqual(len(pwd_issues), 0, "Variable assignment should not trigger cleartext_password")


# ---------------------------------------------------------------------------
# Test: extended _scan_api_validation
# ---------------------------------------------------------------------------

class TestExtendedApiValidation(unittest.TestCase):
    def _make_project(self, files):
        tmp = tempfile.mkdtemp()
        pdir = Path(tmp)
        file_contents = {}
        for rel, content in files.items():
            fp = pdir / rel
            fp.parent.mkdir(parents=True, exist_ok=True)
            fp.write_text(content, encoding="utf-8")
            file_contents[fp] = content
        return pdir, file_contents

    def test_patch_handler_flagged(self):
        pdir, fc = self._make_project({
            "app/api/items/route.ts": (
                "export async function PATCH(request: Request) {\n"
                "  const body = await request.json();\n"
                "  // no validation\n"
                "  await db.update(body);\n"
                "}\n"
            )
        })
        issues = claw_cli._scan_api_validation(str(pdir), fc)
        types = [i["type"] for i in issues]
        self.assertIn("unvalidated_api_input", types)

    def test_get_with_searchparams_flagged(self):
        pdir, fc = self._make_project({
            "app/api/search/route.ts": (
                "export async function GET(request: Request) {\n"
                "  const { searchParams } = new URL(request.url);\n"
                "  const query = searchParams.get('q');\n"
                "  const results = await db.search(query);\n"
                "  return Response.json(results);\n"
                "}\n"
            )
        })
        issues = claw_cli._scan_api_validation(str(pdir), fc)
        types = [i["type"] for i in issues]
        self.assertIn("unvalidated_get_params", types)

    def test_middleware_protected_route_not_flagged_for_auth(self):
        pdir, fc = self._make_project({
            "middleware.ts": "import { auth } from './lib/auth';\nexport function middleware() { auth(); }\n",
            "app/api/data/route.ts": (
                "export async function GET(request: Request) {\n"
                "  const data = await db.findAll();\n"
                "  return Response.json(data);\n"
                "}\n"
            )
        })
        issues = claw_cli._scan_api_validation(str(pdir), fc)
        auth_issues = [i for i in issues if i["type"] == "missing_api_auth"]
        self.assertEqual(len(auth_issues), 0, "Middleware-protected routes should not flag missing auth")


# ---------------------------------------------------------------------------
# Test: _enhance_next_config_security
# ---------------------------------------------------------------------------

class TestEnhanceNextConfigSecurity(unittest.TestCase):
    def test_injects_headers_esm(self):
        content = "const nextConfig = {\n  reactStrictMode: true,\n};\nexport default nextConfig;\n"
        result = claw_cli._enhance_next_config_security(content)
        self.assertIn("X-Content-Type-Options", result)
        self.assertIn("X-Frame-Options", result)
        self.assertIn("Content-Security-Policy", result)

    def test_skips_existing_headers(self):
        content = "const nextConfig = {\n  // X-Content-Type-Options already here\n};\n"
        result = claw_cli._enhance_next_config_security(content)
        self.assertEqual(result, content, "Should not modify if headers already exist")

    def test_skips_large_files(self):
        content = "const nextConfig = {\n" + ("  // padding\n" * 2000) + "};\n"
        result = claw_cli._enhance_next_config_security(content)
        self.assertEqual(result, content, "Should not modify large files")

    def test_handles_module_exports(self):
        content = "module.exports = {\n  reactStrictMode: true,\n};\n"
        result = claw_cli._enhance_next_config_security(content)
        self.assertIn("X-Content-Type-Options", result)
        self.assertIn("async headers()", result)

    def test_skips_wrapper_patterns(self):
        content = "export default defineConfig({\n  reactStrictMode: true,\n});\n"
        result = claw_cli._enhance_next_config_security(content)
        self.assertEqual(result, content, "Should not inject inside defineConfig()")


# ---------------------------------------------------------------------------
# Test: _enhance_html security meta tags
# ---------------------------------------------------------------------------

class TestEnhanceHtmlSecurity(unittest.TestCase):
    def test_injects_csp_meta_tag(self):
        html = (
            "<!DOCTYPE html>\n<html>\n<head>\n"
            "  <title>Test</title>\n"
            "  <script src=\"https://cdn.tailwindcss.com\"></script>\n"
            "</head>\n<body>\n<h1>Hello</h1>\n</body>\n</html>"
        )
        result = claw_cli._enhance_html(html)
        self.assertIn('Content-Security-Policy', result)
        self.assertIn('meta name="referrer"', result)


# ---------------------------------------------------------------------------
# Test: missing_input_length false positive avoidance
# ---------------------------------------------------------------------------

class TestMissingInputLength(unittest.TestCase):
    def _make_project(self, files):
        tmp = tempfile.mkdtemp()
        pdir = Path(tmp)
        file_contents = {}
        for rel, content in files.items():
            fp = pdir / rel
            fp.parent.mkdir(parents=True, exist_ok=True)
            fp.write_text(content, encoding="utf-8")
            file_contents[fp] = content
        return pdir, file_contents

    def test_checkbox_not_flagged(self):
        """Checkboxes should NOT trigger missing_input_length."""
        claw_cli._SECURITY_SCAN_PATTERNS = None
        pdir, fc = self._make_project({
            "components/Form.tsx": '<input type="checkbox" name="agree" />\n<input type="radio" name="choice" />\n'
        })
        issues = claw_cli._scan_security_quality(str(pdir), fc)
        length_issues = [i for i in issues if i["type"] == "security_missing_input_length"]
        self.assertEqual(len(length_issues), 0, f"Checkboxes/radios should not trigger missing_input_length, got: {length_issues}")


# ---------------------------------------------------------------------------
# Test: _detect_project_framework
# ---------------------------------------------------------------------------

class TestDetectProjectFramework(unittest.TestCase):
    def _make_project(self, files):
        tmp = tempfile.mkdtemp()
        pdir = Path(tmp)
        for rel, content in files.items():
            fp = pdir / rel
            fp.parent.mkdir(parents=True, exist_ok=True)
            fp.write_text(content, encoding="utf-8")
        return str(pdir)

    def test_nextjs_detection(self):
        pdir = self._make_project({"next.config.mjs": "export default {}"})
        self.assertEqual(claw_cli._detect_project_framework(pdir), "nextjs")

    def test_express_detection(self):
        pdir = self._make_project({
            "package.json": '{"dependencies": {"express": "^4.18.0"}}'
        })
        self.assertEqual(claw_cli._detect_project_framework(pdir), "express")

    def test_django_detection(self):
        pdir = self._make_project({"manage.py": "#!/usr/bin/env python"})
        self.assertEqual(claw_cli._detect_project_framework(pdir), "django")

    def test_fastapi_detection(self):
        pdir = self._make_project({"requirements.txt": "fastapi==0.100.0\nuvicorn"})
        self.assertEqual(claw_cli._detect_project_framework(pdir), "fastapi")

    def test_sveltekit_detection(self):
        pdir = self._make_project({"svelte.config.js": "export default {}"})
        self.assertEqual(claw_cli._detect_project_framework(pdir), "sveltekit")

    def test_unknown_project(self):
        pdir = self._make_project({"readme.md": "# Hello"})
        self.assertEqual(claw_cli._detect_project_framework(pdir), "unknown")

    def test_vite_detection(self):
        pdir = self._make_project({"vite.config.ts": "export default {}"})
        self.assertEqual(claw_cli._detect_project_framework(pdir), "vite")

    def test_remix_detection(self):
        pdir = self._make_project({
            "package.json": '{"dependencies": {"@remix-run/node": "^2.0.0"}}'
        })
        self.assertEqual(claw_cli._detect_project_framework(pdir), "remix")


# ---------------------------------------------------------------------------
# Test: Prompt registry
# ---------------------------------------------------------------------------

class TestPromptRegistry(unittest.TestCase):
    def test_load_registry(self):
        claw_cli._prompt_registry_cache = None
        registry = claw_cli._load_prompt_registry()
        self.assertIn("sections", registry)
        self.assertIn("core_identity", registry["sections"])
        self.assertIn("scaffold_rules", registry["sections"])

    def test_load_section(self):
        claw_cli._prompt_section_cache.clear()
        content = claw_cli._load_prompt_section("core_identity")
        self.assertIn("Rattlesnake", content)
        self.assertIn("{{CWD}}", content)

    def test_conversation_mode_includes_all(self):
        """Conversation mode should include core identity, discovery, security, etc."""
        claw_cli._prompt_registry_cache = None
        claw_cli._prompt_section_cache.clear()
        result = claw_cli._build_prompt_for_mode("conversation")
        self.assertIn("Rattlesnake", result)
        self.assertIn("Discovery", result)
        self.assertIn("ZERO COMPROMISE", result)
        self.assertIn("Inner Monologue", result)
        # Should have CWD substituted (not placeholder)
        self.assertNotIn("{{CWD}}", result)

    def test_scaffold_mode_excludes_conversation_only(self):
        """Scaffold mode should NOT include discovery flow, memory, communication, inner monologue."""
        claw_cli._prompt_registry_cache = None
        claw_cli._prompt_section_cache.clear()
        result = claw_cli._build_prompt_for_mode("scaffold", token_budget=2500)
        self.assertIn("Rattlesnake", result)
        # Scaffold should NOT include conversation-only sections
        self.assertNotIn("Inner Monologue", result)
        # Should still include workflow rules
        self.assertIn("SEARCH", result)

    def test_scaffold_mode_smaller_than_conversation(self):
        """Scaffold prompt should be significantly smaller than conversation prompt."""
        claw_cli._prompt_registry_cache = None
        claw_cli._prompt_section_cache.clear()
        conv = claw_cli._build_prompt_for_mode("conversation")
        scaffold = claw_cli._build_prompt_for_mode("scaffold", token_budget=2500)
        self.assertLess(len(scaffold), len(conv))
        # Scaffold should be under 10KB
        self.assertLess(len(scaffold), 10000)

    def test_scaffold_under_25k_ceiling(self):
        """Scaffold prompt should be under MAX_SCAFFOLD_CONTEXT."""
        claw_cli._prompt_registry_cache = None
        claw_cli._prompt_section_cache.clear()
        result = claw_cli._build_prompt_for_mode("scaffold", token_budget=2500)
        self.assertLess(len(result), claw_cli.MAX_SCAFFOLD_CONTEXT)

    def test_caching_works(self):
        claw_cli._prompt_registry_cache = None
        first = claw_cli._load_prompt_registry()
        second = claw_cli._load_prompt_registry()
        self.assertIs(first, second)


# ---------------------------------------------------------------------------
# Test: Express handler scanning
# ---------------------------------------------------------------------------

class TestExpressScanning(unittest.TestCase):
    def _make_project(self, files):
        tmp = tempfile.mkdtemp()
        pdir = Path(tmp)
        file_contents = {}
        for rel, content in files.items():
            fp = pdir / rel
            fp.parent.mkdir(parents=True, exist_ok=True)
            fp.write_text(content, encoding="utf-8")
            file_contents[fp] = content
        return pdir, file_contents

    def test_express_post_flagged(self):
        pdir, fc = self._make_project({
            "routes/users.js": (
                "const express = require('express');\n"
                "const router = express.Router();\n"
                "router.post('/api/users', async (req, res) => {\n"
                "  const data = req.body;\n"
                "  await db.insert(data);\n"
                "  res.json({ ok: true });\n"
                "});\n"
            )
        })
        issues = claw_cli._scan_api_validation(str(pdir), fc, framework="express")
        types = [i["type"] for i in issues]
        self.assertIn("unvalidated_api_input", types)

    def test_express_validated_not_flagged(self):
        pdir, fc = self._make_project({
            "routes/users.js": (
                "const express = require('express');\n"
                "const router = express.Router();\n"
                "router.post('/api/users', async (req, res) => {\n"
                "  const data = req.body;\n"
                "  const result = schema.safeParse(data);\n"
                "  if (!result.success) return res.status(400).json(result.error);\n"
                "  await db.insert(result.data);\n"
                "  res.json({ ok: true });\n"
                "});\n"
            )
        })
        issues = claw_cli._scan_api_validation(str(pdir), fc, framework="express")
        input_issues = [i for i in issues if i["type"] == "unvalidated_api_input"]
        self.assertEqual(len(input_issues), 0, "Validated Express route should not be flagged")


# ---------------------------------------------------------------------------
# Test: Broadened auth patterns
# ---------------------------------------------------------------------------

class TestBroadenedAuthPatterns(unittest.TestCase):
    def _make_project(self, files):
        tmp = tempfile.mkdtemp()
        pdir = Path(tmp)
        file_contents = {}
        for rel, content in files.items():
            fp = pdir / rel
            fp.parent.mkdir(parents=True, exist_ok=True)
            fp.write_text(content, encoding="utf-8")
            file_contents[fp] = content
        return pdir, file_contents

    def test_jwt_verify_matches(self):
        """jwt.verify should be recognized as auth check."""
        pdir, fc = self._make_project({
            "app/api/data/route.ts": (
                "export async function GET(request: Request) {\n"
                "  const token = request.headers.get('authorization');\n"
                "  const decoded = jwt.verify(token, process.env.SECRET);\n"
                "  return Response.json({ data: [] });\n"
                "}\n"
            )
        })
        issues = claw_cli._scan_api_validation(str(pdir), fc)
        auth_issues = [i for i in issues if i["type"] == "missing_api_auth"]
        self.assertEqual(len(auth_issues), 0, "jwt.verify should be recognized as auth")

    def test_req_user_matches(self):
        """req.user (Passport) should be recognized as auth check."""
        pdir, fc = self._make_project({
            "routes/api/data.js": (
                "router.get('/api/data', async (req, res) => {\n"
                "  if (!req.user) return res.status(401).json({ error: 'unauthorized' });\n"
                "  res.json({ data: [] });\n"
                "});\n"
            )
        })
        issues = claw_cli._scan_api_validation(str(pdir), fc, framework="express")
        auth_issues = [i for i in issues if i["type"] == "missing_api_auth"]
        self.assertEqual(len(auth_issues), 0, "req.user should be recognized as auth")

    def test_getServerSession_matches(self):
        """getServerSession (NextAuth) should be recognized as auth check."""
        pdir, fc = self._make_project({
            "app/api/data/route.ts": (
                "export async function GET(request: Request) {\n"
                "  const session = await getServerSession(authOptions);\n"
                "  if (!session) return NextResponse.json({ error: 'unauth' }, { status: 401 });\n"
                "  return NextResponse.json({ data: [] });\n"
                "}\n"
            )
        })
        issues = claw_cli._scan_api_validation(str(pdir), fc)
        auth_issues = [i for i in issues if i["type"] == "missing_api_auth"]
        self.assertEqual(len(auth_issues), 0, "getServerSession should be recognized as auth")


# ---------------------------------------------------------------------------
# Test: _enhance_config_security (renamed from _enhance_next_config_security)
# ---------------------------------------------------------------------------

class TestEnhanceConfigSecurity(unittest.TestCase):
    def test_nextjs_via_router(self):
        content = "const nextConfig = {\n  reactStrictMode: true,\n};\nexport default nextConfig;\n"
        result = claw_cli._enhance_config_security(content, framework="nextjs")
        self.assertIn("X-Content-Type-Options", result)

    def test_express_returns_unchanged(self):
        content = "const app = express();\n"
        result = claw_cli._enhance_config_security(content, framework="express")
        self.assertEqual(result, content, "Express should return unchanged (uses middleware, not config)")

    def test_default_framework_is_nextjs(self):
        content = "const nextConfig = {\n  reactStrictMode: true,\n};\nexport default nextConfig;\n"
        result = claw_cli._enhance_config_security(content)
        self.assertIn("X-Content-Type-Options", result)


# ---------------------------------------------------------------------------
# Test: _build_security_context with scaffold_mode and framework
# ---------------------------------------------------------------------------

class TestBuildSecurityContextParams(unittest.TestCase):
    def test_scaffold_mode_smaller(self):
        claw_cli._security_json_cache = None
        full = claw_cli._build_security_context("saas dashboard", scaffold_mode=False)
        scaffold = claw_cli._build_security_context("saas dashboard", scaffold_mode=True)
        # Scaffold mode omits descriptions, should be smaller or equal
        self.assertLessEqual(len(scaffold), len(full))

    def test_express_framework_header(self):
        claw_cli._security_json_cache = None
        ctx = claw_cli._build_security_context("api backend", framework="express")
        self.assertIn("express", ctx.lower())


if __name__ == "__main__":
    unittest.main()
