"""Harbor agent wrappers that capture the agent's work as `model.patch`.

Why this exists
---------------

DeepSWE tasks set ``[verifier] environment_mode = "separate"``: the verifier
runs in a *different container* than the agent, so in-place edits to ``/app``
are invisible to it. The only channel between the two is the artifact declared
in ``task.toml``::

    artifacts = ["/logs/artifacts/model.patch"]

Upstream DeepSWE creates that file from a ``pre_artifacts.sh`` hook. That hook
belongs to their own runner (``pier``); **Harbor has no equivalent lifecycle
step** — it collects the declared artifact and nothing ever creates it. Running
DeepSWE under Harbor without this wrapper produces a silent, uniform zero:
every verifier grades a pristine checkout, so every task reports ``f2p 0`` with
``p2p 1.0``, and both arms of an A/B return *byte-identical* scores. That is a
broken instrument that reads exactly like a legitimate hard-benchmark result.

Why not just run `pre_artifacts.sh`
-----------------------------------

It captures ``git diff base..HEAD``, which is empty unless the agent committed.
The task prompt does ask it to ("work on this in a new branch ... and commit
everything"), but a *model complying with a prompt* is not a mechanism. A
non-compliant trial would produce an empty patch and score zero — reproducing
the same silent-wrong-answer failure that motivated this file.

So the capture commits on the agent's behalf before diffing, and always logs a
byte count. A zero-byte patch is a harness error, not a score of 0.00.

The base commit is read from the live container at setup time rather than from
``task.toml``'s ``base_commit_hash``, so it reflects the tree the agent actually
started from.
"""

from harbor.agents.installed.claude_code import ClaudeCode
from harbor.agents.installed.codex import Codex
from harbor.agents.oracle import OracleAgent
from harbor.environments.base import BaseEnvironment

#: Where the pre-agent HEAD is stashed inside the container.
BASE_REF_FILE = "/tmp/.kbench_base_commit"

_RECORD_BASE = f"""
set -uo pipefail
cd /app 2>/dev/null || {{ echo '[capture] no /app; nothing to record'; exit 0; }}
git config --global --add safe.directory /app 2>/dev/null || true
git rev-parse HEAD > {BASE_REF_FILE} 2>/dev/null || true
echo "[capture] base commit $(cat {BASE_REF_FILE} 2>/dev/null || echo '<none>')"
"""

# `;` rather than `&&` throughout: `set -o pipefail` is active and a nonzero
# exit from the agent must never skip the capture.
_CAPTURE = f"""
set -uo pipefail
cd /app 2>/dev/null || {{ echo '[capture] no /app; skipping'; exit 0; }}
git config --global --add safe.directory /app 2>/dev/null || true
BASE=$(cat {BASE_REF_FILE} 2>/dev/null || true)
if [ -z "$BASE" ]; then
  echo '[capture] FATAL: no base commit was recorded at setup'
  exit 3
fi
mkdir -p /logs/artifacts
git add -A 2>/dev/null || true
git -c user.email=bench@local -c user.name=kbench \
    commit -q -m 'kbench submission' 2>/dev/null || true
git diff --binary "$BASE" HEAD > /logs/artifacts/model.patch 2>/dev/null || true
echo "[capture] wrote $(wc -c < /logs/artifacts/model.patch) bytes (base $BASE)"
"""


class _CaptureMixin:
    """Records the pre-agent HEAD, then diffs against it once the agent stops."""

    async def _record_base(self, environment: BaseEnvironment) -> None:
        result = await environment.exec(command=_RECORD_BASE, user="root")
        self.logger.info(f"[capture] record-base rc={result.return_code}")

    async def _capture(self, environment: BaseEnvironment) -> None:
        try:
            result = await environment.exec(command=_CAPTURE, user="root")
            self.logger.info(f"[capture] capture rc={result.return_code}")
        except Exception as exc:  # noqa: BLE001 - never mask the agent's own error
            self.logger.error(f"[capture] failed to capture model.patch: {exc}")

    async def setup(self, environment: BaseEnvironment) -> None:
        await super().setup(environment)
        await self._record_base(environment)

    async def run(self, instruction, environment: BaseEnvironment, context) -> None:
        # `finally`, so a crashed or timed-out agent still submits whatever it
        # managed to write. A partial patch is a real (losing) attempt; a
        # missing one is an unmeasured trial, and those are indistinguishable
        # downstream if we let the capture be skipped.
        try:
            await super().run(instruction, environment, context)
        finally:
            await self._capture(environment)


class CaptureClaudeCode(_CaptureMixin, ClaudeCode):
    """`claude-code`, plus the model.patch capture DeepSWE's verifier needs."""


class CaptureCodex(_CaptureMixin, Codex):
    """`codex`, plus the same capture."""


class CaptureOracle(_CaptureMixin, OracleAgent):
    """`oracle` (runs the task's own solution), plus the same capture.

    This is the gate. Because it shares the capture path with the real agent,
    an oracle trial scoring ``reward == 1.0`` proves the whole chain — solve →
    commit → diff → collect → upload → apply → grade. Hand-placing a
    ``model.patch`` would prove only the grader and leave the capture, which is
    the half that was actually broken, untested.
    """
