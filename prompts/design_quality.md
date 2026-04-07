## 14. DESIGN QUALITY — ZERO SLOP TOLERANCE.
This is CRITICAL. You are NOT allowed to produce generic AI-looking output.
The design context above has exact class strings for every component. **Copy-paste them. Do not improvise.**

**BANNED (AI slop indicators):**
- Raw Tailwind color classes (gray-100, blue-500, etc.) — ONLY CSS variables from the palette
- shadow-lg/xl on cards or containers — border only; shadow reserved for modals, toasts, dropdowns
- rounded-2xl/3xl — max rounded-xl for modals, rounded-lg for cards, rounded-md for buttons/inputs
- Gradient backgrounds on containers — solid colors only
- Shadow AND border on same element — choose one
- More than 1 accent color — one accent, used sparingly
- Center-aligned body text — left-align everything except single-line hero headings
- Placeholder text (Lorem ipsum, "Your content here", "John Doe", "ACME Corp")
- Generic error messages ("Something went wrong") — be specific

**REQUIRED (production quality):**
- All colors via CSS custom properties: `bg-[var(--bg-primary)]`, `text-[var(--text-secondary)]`
- Loading skeleton for every async fetch — use `skeleton_loader` recipe
- Empty state with CTA for every list/table — use `empty_state` recipe
- Error state with retry for every data fetch — use `alert_banner` recipe
- `transition-colors duration-150` on all hover states
- `focus-visible:outline` or `focus-visible:ring-2` on all interactive elements
- Mobile-first responsive — follow responsive rules from design context
- Real, contextual copy — write actual text for the project
