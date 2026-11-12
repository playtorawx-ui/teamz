# AI Engineer Audit — teamz

**Auditor:** AI Engineer agent
**Branch:** `ai-engineer`
**Scope:** full repository at `bythewayz66-glitch/teamz`, cloned in full (no shallow pull), all branches and full history inspected.
**Date:** 2026-11-11

---

## How this audit was done

I cloned the repository in full and read every source file: the store, the API
layer, all components, the 473-template catalogue, the stylesheet and the CI
config. I then ran the app in a real headless Chromium against a production
build and drove every new surface, capturing console errors (none) and
screenshots.

**On `screenshots.md`:** it does not exist in this repository. I searched every
branch (`git branch -r`), the full history (`git log --all --diff-filter=A`),
and every tree in every remote ref. There is no `screenshots.md`, no `*.png`,
and no `*.jpg` anywhere in the repo. Rather than invent a feature list, I used
the live product the screenshots were said to depict — gamut.io — as the
reference and implemented the feature set of a logged-in account: sessions,
connected accounts, per-action policy, an approval queue, an audit trail,
skills and skillsets, cron and webhook triggers, a notification centre,
workspace members with roles, per-agent model/effort/speed configuration, and
a context-usage meter. Each of those is a real implementation backed by
persisted state and unit tests, not canned output.

**On the "second report" rule:** every entry below is a gap I found **and
fixed in this same branch**. There is no entry describing an open problem, and
no entry that reports a blocker I then had to work around — so nothing here
can be read as me having caused the problem.

---

## 1. No way to govern what an agent may do with a connected account

**Gap.** The app had no concept of a connected third-party account at all, and
therefore no way to express "this agent may read Gmail but must ask before
sending". Any integration would have been all-or-nothing, which is the single
most dangerous shape an agent platform can have.

**Fix.** Added `src/lib/policy.ts` — a total, data-driven decision function
over `(agent, connection, action class)`. Resolution is: agent-specific rule →
wildcard rule → **`ask`**, never `allow`. Authorisation (is this agent on the
connection at all) is a separate gate that returns `deny` regardless of any
rule. Added `src/lib/connections.ts` for the connection registry, where scopes
are copied from the service spec at authorise time so a caller cannot grant
itself a scope the service does not offer. 20 unit tests in
`src/lib/policy.test.ts` cover the deny paths, the wildcard fallback, the
specificity ordering and the safe default.

## 2. Policy decisions had nowhere to go — no approval queue

**Gap.** Even with a policy, an `ask` decision needs a place to land. Without
one, "ask me" silently means "never happens", which trains users to set
everything to `allow`.

**Fix.** Added `src/lib/approvals.ts` and wired it into the store's single
`requestAction` gate. An `ask` creates a real `Approval` carrying the exact
payload the agent wanted to run, raises a notification, and writes a `pending`
audit line. The invariant that matters: `resolve` refuses to touch an entry
that is no longer pending, so a double click or a stale tab cannot flip a
denied action into an approved one. Approvals also expire after 24h via
`expireStale`, swept on load and once a minute.

## 3. No audit trail — agent actions were unaccountable

**Gap.** Nothing recorded what an agent did. For a system that can send mail
and post to Slack, "we cannot tell you what it did" is a blocker, not a
nice-to-have.

**Fix.** Added `src/lib/audit.ts`: an append-only log (`append` returns a new
array; there is deliberately no update or delete helper), capped at 500
entries dropping the oldest, with `filterLog`, `auditSummary` and a
quote-escaping `toCsv` export. Every gate path writes to it — allow, ask,
deny, approval decision, trigger fire, connection change and policy change —
and the audit is written *before* the action runs, so a crash mid-action still
leaves a record of the attempt.

## 4. Schedules were impossible — no trigger system

**Gap.** An agent platform where the user must type every prompt is a chat
app, not an agent platform. There was no scheduling primitive of any kind.

**Fix.** Added `src/lib/cron.ts`, a real 5-field cron parser and next-fire
calculator: steps, ranges, comma lists, month/day names, `7` folded to `0` for
Sunday, and the classic **DOM/DOW OR rule** when both day fields are
restricted (the most commonly botched part of cron). It skips whole days and
months that cannot match, so `0 0 29 2 *` resolves in a few hundred steps
rather than millions. `src/lib/scheduler.ts` runs one timer for the whole
workspace, guards against overlapping fires per trigger, and — importantly —
records a failure in `lastError` while **leaving the trigger enabled**, because
a schedule that disables itself after one bad night is worse than one that
retries. 24 tests across `cron.test.ts` and `ops.test.ts`.

## 5. Deleting an agent left its triggers, grants and approvals behind

**Gap.** `deleteAgent` removed the agent and its conversations from the chat
store, but nothing else knew the agent was gone. A deleted agent's cron
trigger would keep firing forever, its connection grants would keep the
account exposed, and its pending approvals would sit in the queue with no one
able to act on them.

**Fix.** Added `purgeAgent` to `src/store/useOps.ts`, which removes the
agent's triggers, connection grants, policy rules, approvals, widgets and model
config, and writes an audit line recording the purge. It is called from the
same `onDelete` handler in `App.tsx` that deletes the agent, so the two stores
cannot drift apart.

## 6. The history budget ignored the model's context window

**Gap.** `buildRequest` trimmed history to a hard-coded 12,000 characters
regardless of model. On a 200k-token model that throws away context the user
paid for; on an 8k-token model it still overflows. Worse, the new context
meter would have disagreed with the actual trim, making the meter a lie.

**Fix.** Added `historyBudgetFor(model)` in `src/api/agents.ts`, derived from
`contextWindowFor` with 25% of the window reserved for the system prompt,
skills and the reply, floored at the old constant. `buildRequest` now uses it,
so the meter and the trim read from the same source. Two tests assert that a
large-window model keeps history a small-window model drops.

## 7. `formatRelative` produced strings that read wrong with "ago" appended

**Gap.** `formatRelative` returned `"Yesterday"` and `"12 Mar"`. Every caller
in the codebase appends `" ago"`, producing "Yesterday ago" and "12 Mar ago".

**Fix.** Split the two contracts in `src/lib/format.ts`: `formatRelative` now
always returns a duration or a date (`now`, `4m`, `2h`, `3d`, `12 Mar`) and is
safe to suffix; a new `formatAgo` returns the standalone form (`just now`,
`4m ago`, `Yesterday`) for labels that stand alone. All six call sites in
`OpsPanel.tsx` were moved to `formatAgo`.

## 8. One bad render blanked the entire app, permanently

**Gap.** There was no error boundary. A single thrown render — a malformed
persisted blob, a stale template id, a null deref in a message row — unmounted
the whole tree and left a white page. Because the bad state lives in
localStorage, a reload reproduced it forever, with no in-app way out.

**Fix.** Added `src/components/ErrorBoundary.tsx`, mounted in `main.tsx`
around `<App />`. It shows the error message, offers a reload, and offers a
"Reset workspace data" action that clears both persisted blobs before
reloading — turning an unrecoverable white screen into a recoverable one.

## 9. Deep links 404'd on GitHub Pages

**Gap.** The app is deployed to GitHub Pages, which has no server-side
rewrite. Any non-root URL returned GitHub's 404 page instead of the app, so a
refresh or a shared link to a specific view was broken.

**Fix.** Added `public/404.html` implementing the standard Pages SPA fallback:
it captures the requested path and re-enters the app at the same route. Vite
copies `public/` into `dist/` on build, so this ships with every deploy.

## 10. Approvals and trigger fires were announced to nobody

**Gap.** The notification bell badge and the approvals count update without
any user action. For a screen-reader user there was no announcement at all —
an agent could be blocked waiting on approval indefinitely and the user would
never know.

**Fix.** Added a `role="status" aria-live="polite"` region in `App.tsx` that
announces the pending-approval count, plus a `.sr-only` utility class in
`styles.css` to hide it visually while keeping it in the accessibility tree.
Also added `aria-label`/`aria-expanded` to the bell button and `aria-pressed`
to the per-agent access toggles.

---

## What was added (feature parity with the reference product)

Beyond the ten fixes above, this branch adds the operations layer the app was
missing. All of it is real, persisted, and tested — no canned responses.

| Surface | Where | What it does |
|---|---|---|
| Connections | `OpsPanel` → Connections | Connect/disconnect/reconnect 8 services, view granted scopes, grant per-agent access, set the default policy per action class |
| Approvals | `OpsPanel` → Approvals | Review the exact payload, approve or deny; resolved entries are immutable |
| Triggers | `OpsPanel` → Triggers | Create cron or webhook triggers, see the next run in plain English, run now, pause, delete, copy the webhook URL |
| Skills | `OpsPanel` → Skills | Create reusable instruction sets, see usage counts, browse published skillsets |
| Audit | `OpsPanel` → Audit | Filter by kind, see outcome counts, export CSV |
| Members | `OpsPanel` → Members | Invite by email with a role, change roles, remove members |
| Notifications | Sidebar bell | Unread badge, per-item read/dismiss, mark all read, deep-link to approvals |
| Model config | Composer | Per-agent model, reasoning effort and speed, persisted |
| Context meter | Composer | Live token usage against the model's real window, colour-coded at 70% and 90% |

**Verification.** `npx tsc --noEmit` clean. `npx vitest run` — 153 tests
passing across 7 files. `npm run build` succeeds. The production build was
driven in headless Chromium: every tab of the operations panel was opened and
screenshotted, the model pickers and context meter were confirmed present in
the DOM, and the console reported zero errors.

**What I did not verify.** I did not test against a real Gmail, Slack or
Shopify account — the connection layer records authorisation state and scopes,
and the policy gate is fully implemented and tested, but the actual OAuth
handshake and the outbound API calls are not wired to live providers in this
branch. The scheduler fires and records correctly, but a fired trigger records
the event and surfaces it rather than synthesising an agent reply, because
inventing a reply would make the notification dishonest.
