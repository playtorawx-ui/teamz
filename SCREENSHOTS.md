# Screenshot -> Implementation Tracker

Repo: **`bythewayz66-glitch/teamz`**
Surface: the **Settings** area, reached from the sidebar footer (**Settings**) or directly at `#/settings/<page-id>`.

Each screenshot is the spec. This file maps every screenshot to the route/component that implements it, so later batches build on this instead of duplicating it.

## How the surface is wired

| Piece | File | Role |
|---|---|---|
| Nav model | `src/settings/nav.ts` | 4 groups, 24 pages, ids + labels + icons. Single source of truth for the sidebar and the router. |
| Persisted state | `src/settings/store.ts` | `localStorage`-backed settings state, one slice per page, with defaults + validation. |
| Icons | `src/settings/icons.tsx` | Settings-surface icon set (1.75px stroke, `currentColor`). |
| Styles | `src/settings/settings.css` | Light-theme stylesheet, scoped under `.settings-root` so the dark chat UI is untouched. |
| Primitives | `src/settings/ui.tsx` | `PageHeader`, `Card`, `Field`, `Toggle`, `Radio`, `Segmented`, `EmptyState`, `Table`, `Badge`, `Button`. |
| Pages | `src/settings/pages/{app,capabilities,infra}.tsx` | The 24 page bodies. |
| Shell | `src/settings/SettingsShell.tsx` | Hash router + sidebar + page dispatch. |
| Entry | `src/App.tsx`, `src/components/Sidebar.tsx` | `#/settings/...` renders the shell; sidebar footer links to it. |

## Screenshot map

Batch 1 - screenshots 0..19 (20 screenshots).

| # | Screenshot shows | Route | Component | Status |
|---|---|---|---|---|
| 0 | **Users** - stuck on a loading state | `#/settings/users` | `pages/app.tsx` -> `UsersPage` | done |
| 1 | **Auth** - session timeout, lockout policy, trusted origins | `#/settings/auth` | `pages/app.tsx` -> `AuthPage` | done |
| 2 | **Admin** - Danger Zone, Factory Reset | `#/settings/admin` | `pages/app.tsx` -> `AdminPage` | done |
| 3 | **Admin** - Danger Zone + Runtime & VM Status + Containers | `#/settings/admin` | `pages/app.tsx` -> `AdminPage` | done |
| 4 | **Mobile** - pair a device (QR / direct connection) | `#/settings/mobile` | `pages/capabilities.tsx` -> `MobilePage` | done |
| 5 | **Connections** - Add New Connection modal, **All** tab | `#/settings/connections` | `pages/capabilities.tsx` -> `AddConnectionModal` | done |
| 6 | **Connections** - modal, **MCPs** tab | `#/settings/connections` | `pages/capabilities.tsx` -> `AddConnectionModal` | done |
| 7 | **Connections** - modal, **APIs** tab | `#/settings/connections` | `pages/capabilities.tsx` -> `AddConnectionModal` | done |
| 8 | **Connections** - default policies + existing connections (GitHub) | `#/settings/connections` | `pages/capabilities.tsx` -> `ConnectionsPage` | done |
| 9 | **Drives** - empty state + Add drive | `#/settings/drives` | `pages/capabilities.tsx` -> `DrivesPage` | done |
| 10 | **Web Search** - provider selection | `#/settings/web-search` | `pages/capabilities.tsx` -> `WebSearchPage` | done |
| 11 | **Logins** - empty state | `#/settings/logins` | `pages/capabilities.tsx` -> `LoginsPage` | done |
| 12 | **Skillsets** - GitHub repo URL + token + list | `#/settings/skillsets` | `pages/capabilities.tsx` -> `SkillsetsPage` | done |
| 13 | **Subagents** - launch policies (tri-state) | `#/settings/subagents` | `pages/capabilities.tsx` -> `SubagentsPage` | done |
| 14 | **Account Provider** - Composio API key | `#/settings/account-provider` | `pages/infra.tsx` -> `AccountProviderPage` | done |
| 15 | **Global Guidance** - instructions applied to all agents | `#/settings/global-guidance` | `pages/infra.tsx` -> `GlobalGuidancePage` | done |
| 16 | **Model Providers** - org credits + global agent model | `#/settings/model-providers` | `pages/infra.tsx` -> `ModelProvidersPage` | done |
| 17 | **Voice** - input/output/speed/provider | `#/settings/voice` | `pages/infra.tsx` -> `VoicePage` | done |
| 18 | **Container Runtime** - resource limits + env vars | `#/settings/container-runtime` | `pages/infra.tsx` -> `ContainerRuntimePage` | done |
| 19 | **Browser Use** - password manager + browser host | `#/settings/browser-use` | `pages/infra.tsx` -> `BrowserUsePage` | done |

## Verification

- `npm test` -> 121 passed (6 files), including `src/settings/nav.test.ts` and `src/settings/store.test.ts`.
- `npm run typecheck` -> clean (`tsc --noEmit`).
- `npm run build` -> clean.
- Rendered every page with `scripts/shot-settings.mjs` (Playwright) and compared against the source screenshots. Two defects found and fixed: a layout void to the right of the content column, and `Unlimited` rendered as an input value instead of a placeholder.
