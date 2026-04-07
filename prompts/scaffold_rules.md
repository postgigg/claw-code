## Scaffold Execution Rules
You are in SCAFFOLD MODE — generating a complete project from a template.

**Execution discipline:**
- Create files ONE AT A TIME using write_file. Do NOT batch.
- Follow the BUILD SPEC file creation order exactly. Do NOT skip ahead.
- Do NOT edit package.json until AFTER creating all app files.
- If you need a new dependency, note it and add it at the END.
- Each file MUST compile — no TypeScript errors, no missing imports.

**Quality gates:**
- Use dark-mode-compatible colors everywhere — NEVER hardcode text-gray-900 or bg-white without dark: variants.
- Load custom fonts with the framework's font loader (next/font/google for Next.js).
- Every input/button MUST have working event handlers — no dead UI in server components.
- Every API route MUST have real working logic — no stubs, no TODOs.
- Data visualization components MUST derive display range from actual data, not hardcoded offsets.
- All API route handlers MUST wrap database operations in try/catch and return proper error responses.

**What NOT to do:**
- Do NOT rewrite template code (auth, webhooks, middleware) — it's already tested.
- Do NOT re-ask what the build spec already specifies.
- Do NOT show code blocks in text — use write_file for everything.
