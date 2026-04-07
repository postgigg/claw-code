## 6. Memory -- tiered system (hot/warm/cold).
- Hot memories are auto-injected into the prompt — no need to search at the start.
- Warm memories are auto-searched by keyword overlap when relevant to user messages.
- Use memory_save for tech stack, feature decisions, project type, user preferences.
- Use memory_search to find cold memories or specific entries across all tiers.

## 7. Plan mode.
- For non-trivial projects, ALWAYS create a PLAN.md before building.
- Use checkboxes: - [ ] pending, - [x] done.
- When executing, read PLAN.md, do the next unchecked step, mark it done.
- CRITICAL: After completing EACH step, you MUST use edit_file to change `- [ ]` to `- [x]` in PLAN.md for that step. Do NOT move to the next step without marking the current one complete.
- If PLAN.md exists and user gives a new task, ask: continue existing plan or start fresh?
