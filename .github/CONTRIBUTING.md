# Contributing Guide — Sioux Falls Network Simulator

## Branch Strategy

```
main          ← Always deployable; direct push is prohibited
  └─ feature/<topic>   New functionality
  └─ fix/<topic>       Bug fixes
  └─ refactor/<topic>  Code improvements with no behavior change
  └─ docs/<topic>      Documentation or comment changes only
  └─ chore/<topic>     Build, config, or dependency changes
```

Branch names use **lowercase letters and hyphens** only.

```
# Good
feature/passenger-boarding
fix/bus-idle-loop
refactor/engine-tick-order

# Bad
Feature/PassengerBoarding
fix_bus_idle
```

---

## Commit Message Rules

### Format

```
<type>(<scope>): <subject>

[body — optional]

[footer — optional]
```

### Type

| Type | When to use |
|------|-------------|
| `feat` | New feature |
| `fix` | Bug fix |
| `refactor` | Code change with no behavior change |
| `style` | Formatting, whitespace (no logic change) |
| `docs` | Documentation or comments only |
| `chore` | Build, dependencies, or config |
| `perf` | Performance improvement |
| `test` | Adding or updating tests |

### Scope (optional)

Indicate the area of impact inside parentheses.

```
engine, store, network2d, network3d, controls, infopanel,
pathfinding, graph, types, data, ci
```

### Subject

- Start with a present-tense verb (Add / Fix / Remove / Update / Refactor …)
- Capitalize the first letter; no trailing period
- 72 characters or fewer

### Examples

```
feat(engine): Add passenger boarding before bus departure
fix(network2d): Separate riding passenger dots below bus circle
refactor(store): Swap processPassengers and stepBus order
chore: Bump vite to 5.4
```

---

## Pull Request Rules

### PR Title

Follow the same format as the commit subject line.

```
feat(engine): Add passenger boarding before bus departure
```

### PR Body

Use the template in `.github/PULL_REQUEST_TEMPLATE.md`.

### Checklist (required before merge)

- [ ] `npm run build` passes with zero TypeScript errors
- [ ] Verified in the browser
- [ ] No unrelated file changes included
- [ ] Commit messages follow the format above

### Merge Method

| Situation | Method |
|-----------|--------|
| Feature branch → `main` | **Squash and Merge** (clean history) |
| Hotfix (single commit) | **Rebase and Merge** |
| Never use | ~~Merge commit~~ (avoids noise on `main`) |

### Merge Permissions

- Direct push to `main` is not allowed — always open a PR
- Self-merge is permitted on this project (small team)

---

## Code Style

- **TypeScript strict mode** — no `any`
- Component files: `PascalCase.tsx`
- Utility files: `camelCase.ts`
- Constants: `UPPER_SNAKE_CASE`
- Functions: `camelCase`, starting with a verb

---

## Commit Frequency

- Keep each commit logically independent
- Squash WIP commits before merging a PR
- Avoid meaningless messages like `b7d42cc .`
