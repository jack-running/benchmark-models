# Releasing

How this repo publishes GitHub releases, with best practices and the exact
commands. Two equivalent paths are documented: the **`gh` CLI** (recommended;
cleanest) and the **GitHub API via your stored Git Credential Manager token**
(the path used to publish v0.1.0, since `gh` is not installed on this machine).

## Versioning policy

- Use [SemVer](https://semver.org/). This repo is pre-1.0, so the current
  series is `0.x.y`: `0.1.0`, `0.2.0`, … Patch bumps (`0.1.1`) for fixes only;
  minor bumps (`0.2.0`) for features or breaking changes while `0.x`.
  Once `1.0.0` is reached, semantic rules apply strictly (breaking → `1.x.0`,
  fixes → `x.1.x`).
- **One commit owns the version.** Before releasing, the changelog and docs
  for the new version should be committed on `main`. The tag then points at
  that exact commit; never tag a commit that omits the release notes.
- Drive the version from **`CHANGELOG.md`** (Keep a Changelog). When a
  release is cut, move its section out of "Unreleased" — do not invent the
  notes at release time.

## Best-practice checklist

1. `main` is green: `python -m pytest tests/ -q` passes.
2. CHANGELOG updated and committed (`Unreleased` → versioned section).
3. Pick `NEXT` version from the changelog, not from memory.
4. Create an **annotated** tag (carries a message and its own identity; a
   lightweight tag has neither).
5. Push the tag **before** creating the release (a draft release can exist
   without a pushed tag, but publishing requires the tag to be reachable).
6. Write release notes from the changelog section. Keep the first line the
   one-line summary; group the rest (New / Changed / Fixed / Removed).
7. Publish a **draft** first and review it in the browser for anything
   user-facing; flip to published only when confident. `gh` supports
   `--draft`; the API takes `"draft": true`.
8. For a beta/RC use a **prerelease**: `v0.2.0-rc.1` (dotted, so a later
   `v0.2.0-rc.2` sorts correctly) and `"prerelease": true`.
9. Never delete or re-point an already-published tag. If a release is broken,
   cut a patch (`0.1.1`), do not mutate `0.1.0`.
10. A release is **immutable provenance**: once published, treat it as a
    permanent record.

## Path A — `gh` CLI (recommended)

Install once: `winget install --id GitHub.cli` then `gh auth login`.

```bash
# 1. bump version in the changelog, commit it on main, then:
export NEXT=v0.1.0

# 2. annotated tag + push it
git tag -a "$NEXT" -m "Release $NEXT"
git push origin "$NEXT"

# 3. draft the release from that tag (edit notes, publish in browser)
gh release create "$NEXT" --target main --draft --generate-notes

# 4. to publish without the browser:
gh release edit "$NEXT" --draft=false
```

Other useful commands:

```bash
gh release list                        # what's already published
gh release view "$NEXT"                # inspect one release
gh release create "$NEXT" --prerelease --title "v0.2.0-rc.1"
gh release delete "$NEXT" --yes        # only if never published
```

## Path B — GitHub API with the stored GCM token (no `gh` install)

The machine authenticates git pushes through **Git Credential Manager**, which
stores an OAuth token (`gho_…`) for this account. The same token can drive the
API. Retrieve it, then create the release — all in one shell invocation,
because the token is scoped to that process (never write it to a file):

```bash
TOKEN=$(printf 'protocol=https\nhost=github.com\n\n' | git credential fill | awk -F= '/^password=/{print $2}')

# 2. annotated tag + push it
git tag -a v0.1.0 -m "Release v0.1.0"
git push origin v0.1.0

# 3. create the release (body from a file)
export NEXT=v0.1.0
GITHUB_TOKEN="$TOKEN" python - <<'PY'
import os, json, urllib.request
body = open("release_body.md", encoding="utf-8").read()
payload = {"tag_name": os.environ["NEXT"], "target_commitish": "main",
           "name": os.environ["NEXT"], "body": body,
           "draft": False, "prerelease": False}
req = urllib.request.Request(
    "https://api.github.com/repos/jack-running/benchmark-models/releases",
    data=json.dumps(payload).encode(), method="POST",
    headers={"Authorization": "Bearer " + os.environ["GITHUB_TOKEN"],
             "Accept": "application/vnd.github+json",
             "Content-Type": "application/json"})
try:
    with urllib.request.urlopen(req) as r:
        d = json.loads(r.read())
        print("CREATED:", d["html_url"])
except urllib.error.HTTPError as e:
    print("HTTP", e.code, e.read().decode()[:500])
PY
```

Verify the publish:

```bash
git tag --list
git ls-remote --tags origin | grep v0.1.0        # tag reachable on the remote
# and view the release in the browser at:
#   https://github.com/jack-running/benchmark-models/releases/tag/v0.1.0
```

If the GCM token was granted without `repo` scope, the API POST will return
`403`; re-auth with `gh auth login` (Path A) instead of chasing scopes by
hand.

## Minimal sequence (for reference)

```bash
python -m pytest tests/ -q            # gate
# ... commit the changelog bump ...
git tag -a v0.1.0 -m "Release v0.1.0"
git push origin main v0.1.0
gh release create v0.1.0 --target main   # or the API path above
```