#!/bin/bash
# Submit-time gate for Turing DGX jobs. The DGX sibling of isambard_submit.sh.
#
# WHY THIS EXISTS
# ---------------
# Isambard has had a mandatory six-leg gate since 2026-08-26 and its failures
# went away. The DGX never had one. Every DGX job was launched by a per-study
# script under notes/, each re-inventing its own checks, and the lessons never
# accumulated:
#   2026-09-01  two concurrent jobs took the whole box down (the CUDA pool IS
#               system RAM), killing every terminal session on it.
#   2026-09-02  the overnight eval queue burned 2h35m and produced nothing,
#               twice, on faults knowable in seconds. A preflight was written --
#               into outputs/eval_queue/, which is GITIGNORED.
#   2026-09-15  a card-sharing measurement died because the .hydra/overrides.yaml
#               files it reads were not on the box.
#   2026-09-15  a 9-way MPS launch was projected 4-5x wrong and was stopped
#               after 15 minutes.
#   2026-09-16  the 27-cell grid waited 5h40m for the card, won it, and died
#               FOUR SECONDS later on a stale module: the caller had been copied
#               across, the callee it imports had not. ~40 GPU-hours idle.
#
# ⛔⛔ THE ROOT CAUSE IS DURABILITY, NOT CARELESSNESS. `notes/` carries 0 tracked
# files and `outputs` is gitignored, so EVERY DGX safety device this project had
# built lived outside version control -- unreviewable, undiscoverable, and
# re-invented per study. This gate and its legs are tracked, in their own
# repository (`~/dev/turing/preflight`), so they outlive any one project.
#
# LEG 1 -- the box runs the same code as this clone.
#   assert_code_parity.py walks the AST closure from the entrypoint and
#   checksums every first-party file against the box. It follows imports inside
#   function bodies, which is what makes it able to catch the 2026-09-16 failure
#   when --status and --dry-run structurally could not: the launcher defers the
#   heavy import on purpose, so no run of the program reaches the stale module
#   until the real launch, i.e. after the queue wait is spent.
#   ⛔ git cannot do this job here -- the files that broke live under notes/,
#   which git does not carry, so hand-copying is forced and must be VERIFIED.
#   A file may differ if declared: --allow-diff 'path=reason'.
#
# LEG 2 -- the entrypoint actually imports ON THE BOX.
#   Parity proves the bytes match. It does not prove they run: the box has its
#   own venv, its own datasets and its own paths. Leg 2 imports the entrypoint's
#   module graph remotely, under the box's own interpreter.
#
# LEG 3 -- the GPU is free.
#   One GPU job at a time per DGX (owner's standing rule; 2026-09-01). The guard
#   is PIPED to the box from this clone, so the box cannot run a stale copy of
#   the check itself -- the failure mode this whole gate exists for.
#
# LEG 4 -- the host memory guard is in the launch command.
#   The CUDA pool is system RAM, so an uncapped job can take the box down. The
#   guard is applied by PYTHONPATH; a job launched without it is uncovered
#   (Q12). Declared exceptions are refused -- this one has no good reason.
#
# CONCURRENCY. Leg 3 defaults to the standing rule -- one job at a time -- and
# --max-jobs N raises it. Raise it only against a MEASURED sharing result for
# the workload in hand: 9-at-once was tried on 2026-09-15 and stopped after 15
# minutes (370 s/epoch against a recorded 82, i.e. a 4-5x optimistic
# projection), and sharing bought only ~1.5x on this GPU-bound grid. Another
# USER on the card is refused separately and always, unless
# --allow-other-users 'reason': the sharing daemon has a measured deadlock when
# a second user's client appears, which wedged it for 9 h 44 m.
#
# AUTOCAST_DGX_GATE_CHECK_ONLY=1 runs every leg and stops before launching.
# Running with no command runs the legs and reports, launching nothing.
set -euo pipefail

HOST="${AUTOCAST_DGX_HOST:-site-b}"
REMOTE_ROOT="${AUTOCAST_DGX_ROOT:-/data/site/user1/project-a}"
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

# When the box IS this machine (AUTOCAST_DGX_HOST=localhost) every leg runs here
# directly. It used to `ssh localhost`, which only worked while a forwarded agent was
# attached, so a chain armed for later (2026-10-02, owner away) could not gate itself.
on_box() {
    if [[ "$HOST" == localhost ]]; then bash -c "$1"; else ssh -o BatchMode=yes "$HOST" "$1"; fi
}

# ⛔ This gate lives in its OWN repository, NOT inside the project it checks --
# `project-a` reconciles with a public repo eventually, so a machine-level
# gate must not travel in it. It therefore cannot derive the project root from
# its own location the way it did before the 2026-09-18 split. Order: --repo,
# then $AUTOCAST_DGX_REPO, then the git top level of wherever it was invoked.
repo_arg=""

entry=""; searches=(); allows=(); alsos=(); cmd=(); gpu_args=(); resident_given=0
while [[ $# -gt 0 ]]; do
    case "$1" in
        --repo)              repo_arg="$2"; shift 2 ;;
        --entry)             entry="$2"; shift 2 ;;
        --search)            searches+=("--search" "$2"); shift 2 ;;
        --allow-diff)        allows+=("--allow-diff" "$2"); shift 2 ;;
        --also)              alsos+=("--also" "$2"); shift 2 ;;
        --max-jobs)          gpu_args+=("--max-jobs" "$2"); shift 2 ;;
        --allow-other-users) gpu_args+=("--allow-other-users" "$2"); shift 2 ;;
        --resident)          gpu_args+=("--resident" "$2"); resident_given=1; shift 2 ;;
        --)                  shift; cmd=("$@"); break ;;
        *) echo "[dgx] unknown argument: $1" >&2; exit 2 ;;
    esac
done

# AUTOCAST_GPU_RESIDENT='<name>=<reason>' is the standing way to declare the
# owner's resident process (2026-09-25); an explicit --resident wins over it.
if [[ "$resident_given" == 0 && -n "${AUTOCAST_GPU_RESIDENT:-}" ]]; then
    gpu_args+=("--resident" "$AUTOCAST_GPU_RESIDENT")
fi

REPO="${repo_arg:-${AUTOCAST_DGX_REPO:-$(git rev-parse --show-toplevel 2>/dev/null || true)}}"
if [[ -z "$REPO" || ! -d "$REPO" ]]; then
    echo "[dgx] cannot tell which project to check. Run from inside its git" >&2
    echo "[dgx] checkout, or pass --repo <path> / set AUTOCAST_DGX_REPO." >&2
    exit 2
fi
REPO="$(cd "$REPO" && pwd)"

if [[ -z "$entry" ]]; then
    echo "usage: dgx_submit.sh [--repo <project>] --entry <script.py> [--search dir]..." >&2
    echo "                     [--also <file>]... [--allow-diff 'path=reason']..." >&2
    echo "                     [--max-jobs N] [--allow-other-users <reason>]" >&2
    echo "                     [--resident '<name>=<reason>'] [-- command...]" >&2
    exit 2
fi
[[ -f "$REPO/$entry" ]] || { echo "[dgx] no such entrypoint: $entry" >&2; exit 2; }

echo "[dgx] gate for $entry on $HOST (project: $REPO)"

# ---- LEG 1 -------------------------------------------------------------------
echo "[dgx] leg 1/4: checking the box runs the same code as this clone ..."
stem="$(basename "$entry" .py)"
STAGE="outputs/dgx_gate/$stem"
manifest="$REPO/$STAGE/manifest.json"
if ! python3 "$HERE/assert_code_parity.py" "$REPO/$entry" \
        --host "$HOST" --remote-root "$REMOTE_ROOT" --local-root "$REPO" \
        --write-manifest "$manifest" \
        "${searches[@]+"${searches[@]}"}" "${allows[@]+"${allows[@]}"}" \
        "${alsos[@]+"${alsos[@]}"}"; then
    echo "[dgx] REFUSED by leg 1. Nothing was launched." >&2
    exit 1
fi

# Stage the manifest AND the checker onto the box. The supervisor that actually
# launches runs THERE, hours later, and cannot diff against this clone -- so it
# re-checks these digests instead. Both files are copied by the gate itself on
# every arm, so neither can be the stale hand-copy this gate exists to prevent.
on_box "mkdir -p '$REMOTE_ROOT/$STAGE'"
if [[ "$HOST" == localhost ]]; then
    for f in "$manifest" "$HERE/assert_code_parity.py"; do
        [[ "$f" -ef "$REMOTE_ROOT/$STAGE/$(basename "$f")" ]] || cp "$f" "$REMOTE_ROOT/$STAGE/"
    done
else
    scp -q "$manifest" "$HERE/assert_code_parity.py" "$HOST:$REMOTE_ROOT/$STAGE/"
fi
echo "[dgx] staged manifest + checker to $HOST:$REMOTE_ROOT/$STAGE"
echo "[dgx] launch-time re-check (put this in the supervisor, before it launches):"
echo "[dgx]   python3 $STAGE/assert_code_parity.py --check-manifest $STAGE/manifest.json --local-root \"\$PWD\""

# ---- LEG 2 -------------------------------------------------------------------
echo "[dgx] leg 2/4: importing the entrypoint on $HOST ..."
entry_dir="$(dirname "$entry")"
entry_mod="$(basename "$entry" .py)"
search_paths="$REMOTE_ROOT/src:$REMOTE_ROOT/$entry_dir"
for ((i = 0; i < ${#searches[@]}; i += 2)); do
    search_paths="$search_paths:$REMOTE_ROOT/${searches[i + 1]}"
done
if ! on_box \
        "cd '$REMOTE_ROOT' && PYTHONPATH='$search_paths' \
         .venv/bin/python -c 'import $entry_mod' 2>&1 | tail -20"; then
    echo "[dgx] REFUSED by leg 2: the entrypoint does not import on $HOST." >&2
    exit 3
fi
echo "[dgx] leg 2/4: OK"

# ---- LEG 3 -------------------------------------------------------------------
echo "[dgx] leg 3/4: checking the GPU is free on $HOST ..."
# Piped from this clone so the box cannot run a stale copy of the guard itself.
# ⛔ QUOTE EACH ARGUMENT FOR THE REMOTE SHELL. `${gpu_args[*]}` was interpolated
# raw into the ssh command string, so a declared reason containing a space was
# split into words and one containing `(`, `)` or `;` was a remote SYNTAX ERROR
# -- and the gate then reported "the GPU is not free", blaming the card for a
# quoting bug in itself. Found 2026-09-21 on the first multi-word
# --allow-other-users. A gate that refuses a correctly-declared input is one
# that gets switched off, which is worse than no gate.
gpu_remote=""
if [[ ${#gpu_args[@]} -gt 0 ]]; then
    gpu_remote="$(printf ' %q' "${gpu_args[@]}")"
fi
if ! on_box \
        "python3 -$gpu_remote" < "$HERE/assert_gpu_free.py"; then
    echo "[dgx] REFUSED by leg 3: the GPU on $HOST is not free." >&2
    exit 4
fi

# ---- LEG 4 -------------------------------------------------------------------
echo "[dgx] leg 4/4: checking the host memory guard is in the launch ..."
# (a) The project venv on the box loads the guard at EVERY interpreter start (hostguard.pth),
# and it is byte-identical to this clone's. Run by the venv's own python with PYTHONPATH
# cleared, so it sees what an unwrapped smoke test would get -- the 2026-10-06 13:51 hang
# came from one. Piped from here, like leg 3, so the box cannot run a stale checker.
guard_sha="$(python3 -c 'import hashlib,sys; print(hashlib.sha256(open(sys.argv[1],"rb").read()).hexdigest())' \
    "$HERE/../hostguard/hostguard.py")"
if ! on_box "cd '$REMOTE_ROOT' && env -u PYTHONPATH .venv/bin/python - --expect-sha $guard_sha" \
        < "$HERE/assert_hostguard.py"; then
    echo "[dgx] REFUSED by leg 4: the venv on $HOST does not load the current memory guard." >&2
    exit 5
fi
if [[ ${#cmd[@]} -eq 0 ]]; then
    echo "[dgx] leg 4/4: SKIPPED -- no command given, so there is nothing to launch."
else
    if [[ ! "${cmd[*]}" =~ hostguard ]]; then
        echo "[dgx] REFUSED by leg 4: the command carries no hostguard on" >&2
        echo "[dgx] PYTHONPATH. The CUDA pool on this box IS system RAM and an" >&2
        echo "[dgx] uncapped job took the whole machine down on 2026-09-01." >&2
        exit 5
    fi
    echo "[dgx] leg 4/4: OK"
fi

if [[ "${AUTOCAST_DGX_GATE_CHECK_ONLY:-0}" == "1" || ${#cmd[@]} -eq 0 ]]; then
    echo "[dgx] ALL LEGS PASSED. Nothing launched (check-only)."
    exit 0
fi

echo "[dgx] all legs passed; launching ..."
if [[ "$HOST" == localhost ]]; then
    # Recheck occupancy under the shared lock and hold it for the actual job.
    exec bash "$HERE/run_gpu_job.sh" "${gpu_args[@]+"${gpu_args[@]}"}" -- "${cmd[@]}"
fi
if [[ ! "${cmd[*]}" =~ run_gpu_job.sh ]]; then
    echo "[dgx] REFUSED: remote jobs must run inside run_gpu_job.sh on $HOST." >&2
    exit 5
fi
exec "${cmd[@]}"
