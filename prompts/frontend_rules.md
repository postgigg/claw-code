### Frontend-specific (React, Vue, Svelte, HTML):
- **Directive correctness**: Next.js 'use client'/'use server', Nuxt auto-imports, Svelte runes — follow the framework's module system exactly.
- **SSR safety**: Guard browser APIs (window, document, localStorage) behind runtime checks. Never access them at module top level.
- **Animations**: NEVER fire-and-forget. Value-change animations need state-driven triggers (key prop, watchers, signals). Layout animations need overflow management.
- **CSS layers**: Custom @keyframes in @layer base (not utilities). No @apply outside global stylesheet.
- **3D transforms**: Parent needs perspective. Child needs transform-style: preserve-3d.
- **Tailwind**: v3 needs postcss.config.js + tailwind.config.ts. v4 uses CSS-native @config. Detect and match.
- **Images/fonts**: Use framework optimizers (next/image, nuxt-image, vite-imagetools) not raw <img>. Use system font stacks or framework font loaders.
- **Server module isolation**: NEVER import server-only modules (openai, better-sqlite3, pg, fs, crypto, bcrypt, jsonwebtoken, nodemailer) in 'use client' files. Extract server logic to API routes or server components.
- **Native module config**: If using native Node modules (better-sqlite3, sharp, bcrypt, argon2, canvas, sqlite3) in a Next.js project, ALWAYS add them to serverExternalPackages in next.config.

### UI Quality Standards (ALL frontend projects):
- **DARK MODE**: Default to dark theme. Use a cohesive color palette — not raw Tailwind colors (gray-800). Define custom theme colors in tailwind.config (background, surface, border, primary, text-primary, text-secondary, text-muted, danger).
- **TYPOGRAPHY**: Use Inter or system-ui for body, JetBrains Mono or monospace for code. Set font-feature-settings. Use rem, not px.
- **SPACING**: Consistent spacing scale. Use gap not margin for flex/grid children. Use max-w-3xl or max-w-4xl for readable content width.
- **ANIMATIONS**: Add transition-colors/transition-all to interactive elements. Dropdown/modal entry animations (scale+opacity). Skeleton loading states, not spinners.
- **SCROLLBARS**: Custom thin scrollbars on WebKit (6px, transparent track, rounded thumb matching theme).
- **CODE BLOCKS**: Syntax highlighting with a dark theme. Rounded corners, proper padding, overflow-x-auto.
- **EMPTY STATES**: Every list/container needs an empty state with icon + message. Never show a blank page.
- **RESPONSIVE**: Mobile-first. Sidebar collapsible on mobile with overlay backdrop. Touch-friendly tap targets (min 44px).
- **FOCUS STATES**: All interactive elements need visible focus rings (ring-2 ring-primary/30).
- **SELECTION**: Custom ::selection color matching the theme accent.
