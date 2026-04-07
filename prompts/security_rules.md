## 17. SECURITY — ZERO COMPROMISE. STATE OF THE ART.
This is NON-NEGOTIABLE. Every project you build must be secure by default.

**Authentication & Authorization:**
- NEVER store passwords in plaintext — ALWAYS use bcrypt (cost 12+) or argon2id
- ALWAYS validate JWTs server-side with getUser(), NEVER trust client-side session data
- ALWAYS implement proper RBAC — check permissions on EVERY protected route
- ALWAYS use httpOnly, secure, sameSite=strict cookies for session tokens
- ALWAYS implement rate limiting on auth endpoints (login, signup, reset password)
- NEVER expose user IDs in URLs — use UUIDs, NEVER sequential integers
- ALWAYS implement account lockout after 5-10 failed login attempts
- ALWAYS use CSRF tokens on all state-changing forms/endpoints

**Data Protection:**
- NEVER concatenate user input into SQL — ALWAYS use parameterized queries or ORM
- ALWAYS sanitize HTML output to prevent XSS — use DOMPurify or framework auto-escaping
- ALWAYS validate and sanitize ALL user input on the server (even if validated client-side)
- ALWAYS use Content-Security-Policy headers — block inline scripts where possible
- ALWAYS enable RLS on Supabase tables — data is PUBLIC without it
- NEVER log sensitive data (passwords, tokens, credit card numbers, PII)
- ALWAYS encrypt sensitive data at rest (AES-256-GCM) and in transit (TLS 1.3)

**Environment & Secrets:**
- NEVER hardcode API keys, secrets, or credentials in source code
- ALWAYS use environment variables for secrets — .env for local, vault for production
- ALWAYS create .env.example with placeholder values — NEVER commit .env to git
- ALWAYS add .env, .pem, .key files to .gitignore BEFORE first commit
- NEVER expose server-side secrets (STRIPE_SECRET_KEY, SERVICE_ROLE_KEY) to client code
- Client-safe vars MUST use proper prefixes: NEXT_PUBLIC_, REACT_APP_, VITE_
- Use env_manage tool to scan for missing env vars and generate .env.example templates
- ALWAYS rotate secrets if they may have been exposed — NEVER reuse compromised keys

**API & Network Security:**
- ALWAYS validate Content-Type on API routes — reject unexpected formats
- ALWAYS implement proper CORS — never use origin: '*' in production
- ALWAYS set security headers: X-Content-Type-Options, X-Frame-Options, Referrer-Policy
- ALWAYS verify webhook signatures (Stripe, GitHub, etc.) — NEVER trust unverified payloads
- ALWAYS use HTTPS — redirect HTTP to HTTPS
- ALWAYS validate file uploads: check MIME type, limit file size, sanitize filenames
- NEVER expose stack traces or internal errors to users — use generic error messages
- ALWAYS implement request size limits to prevent DoS

**Database Security:**
- ALWAYS use the db_schema tool when planning database architecture
- ALWAYS use foreign key constraints with proper ON DELETE behavior
- ALWAYS create indexes on frequently queried columns
- ALWAYS use transactions for multi-table operations
- ALWAYS use UUIDs for public-facing IDs (not sequential)
- ALWAYS store timestamps with timezone (TIMESTAMPTZ not TIMESTAMP)
- NEVER use the database service_role/admin credentials from client-side code
- ALWAYS apply the principle of least privilege to database roles

**Dependency & Supply Chain:**
- ALWAYS run npm audit / pip audit before deploying
- NEVER install packages from non-registry URLs without verification
- ALWAYS pin dependency versions in production (package-lock.json / pip freeze)
- ALWAYS review new dependency permissions and scope before installing
- NEVER use --ignore-scripts or --no-audit flags blindly

**Stripe-Specific Security:**
- ALWAYS verify webhook signatures with stripe.webhooks.constructEvent()
- NEVER trust client-side price calculations — always use server-side Price objects
- NEVER log full card numbers — Stripe handles PCI compliance, don't break it
- ALWAYS use test keys (sk_test_, pk_test_) in development, NEVER production keys
- ALWAYS handle subscription.deleted events — revoke access immediately
- ALWAYS use Stripe Checkout or Elements — NEVER build custom card forms
