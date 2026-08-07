"""The doc-code lock for ``sld.experiments.protocol`` (PROTOCOL.md sections
12.2, 12.3, 15.1): the committed prose and the module's constants must agree,
or CI fails.

Follows ``tests/test_demotion_banners.py``'s own precedent -- a test that
inspects PROTOCOL.md's TEXT, not its numbers alone -- so a wording change
that silently drops a pinned declaration is caught the same way a missing
demotion banner is: substring containment, positive AND negative (REGLES
section 7.1), the negative built by string substitution in memory, never by
mutating the file on disk.
"""

from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

from sld.experiments import protocol

REPO_ROOT = Path(__file__).resolve().parents[2]
PROTOCOL_MD = REPO_ROOT / "PROTOCOL.md"

# The four pinned substrings, one per locked constant -- committed wording,
# not just committed numbers, so a rewrite of the sentence (not only the
# digit) is caught too. Sources:
#   H_GATE, GAMMA  -- PROTOCOL.md section 12.3
#   TAU_VPT        -- PROTOCOL.md section 12.2
#   GAMMA_1        -- PROTOCOL.md section 15.1
_H_GATE_SUBSTRING = "`H_gate = 25`"
_GAMMA_SUBSTRING = "`γ = 5`"
_TAU_VPT_SUBSTRING = "`τ = 0.5`"
_GAMMA_1_SUBSTRING = "`gamma_1 = 2`"


def _check_locks(consts, doc_text: str) -> list[str]:
    """Verify ``consts`` (anything carrying ``H_GATE``/``GAMMA``/``TAU_VPT``/
    ``GAMMA_1`` attributes) against both the committed doc text and the
    literal values PROTOCOL.md declares. Returns the list of failures (empty
    means everything locks); never raises itself, so both the positive and
    the negative call site can decide what to do with the result.
    """
    failures: list[str] = []

    if _H_GATE_SUBSTRING not in doc_text:
        failures.append(f"PROTOCOL.md missing {_H_GATE_SUBSTRING!r} (section 12.3)")
    if consts.H_GATE != 25:
        failures.append(f"H_GATE == {consts.H_GATE!r}, expected 25 (PROTOCOL.md section 12.3)")

    if _GAMMA_SUBSTRING not in doc_text:
        failures.append(f"PROTOCOL.md missing {_GAMMA_SUBSTRING!r} (section 12.3)")
    if consts.GAMMA != 5.0:
        failures.append(f"GAMMA == {consts.GAMMA!r}, expected 5.0 (PROTOCOL.md section 12.3)")

    if _TAU_VPT_SUBSTRING not in doc_text:
        failures.append(f"PROTOCOL.md missing {_TAU_VPT_SUBSTRING!r} (section 12.2)")
    if consts.TAU_VPT != 0.5:
        failures.append(f"TAU_VPT == {consts.TAU_VPT!r}, expected 0.5 (PROTOCOL.md section 12.2)")

    if _GAMMA_1_SUBSTRING not in doc_text:
        failures.append(f"PROTOCOL.md missing {_GAMMA_1_SUBSTRING!r} (section 15.1)")
    if consts.GAMMA_1 != 2.0:
        failures.append(f"GAMMA_1 == {consts.GAMMA_1!r}, expected 2.0 (PROTOCOL.md section 15.1)")

    return failures


# --------------------------------------------------------------------------
# Positive
# --------------------------------------------------------------------------


def test_locks_pass_against_the_real_module_and_the_real_doc():
    doc_text = PROTOCOL_MD.read_text(encoding="utf-8")
    failures = _check_locks(protocol, doc_text)
    assert not failures, f"doc-code lock broken: {failures}"


# --------------------------------------------------------------------------
# Negative -- the lock has teeth
# --------------------------------------------------------------------------


def test_locks_fail_against_a_broken_namespace():
    """A namespace carrying a wrong H_GATE (24, not 25) must be CAUGHT --
    proves the lock actually compares values rather than only checking the
    doc text is present. No mutation of the real module: constructed
    standalone, exactly like ``test_demotion_banners.py``'s in-memory
    ``.replace(...)`` builds its negative case.
    """
    doc_text = PROTOCOL_MD.read_text(encoding="utf-8")
    broken = SimpleNamespace(H_GATE=24, GAMMA=5.0, TAU_VPT=0.5, GAMMA_1=2.0)
    failures = _check_locks(broken, doc_text)
    assert failures == ["H_GATE == 24, expected 25 (PROTOCOL.md section 12.3)"]


def test_locks_fail_when_the_doc_text_drops_a_pinned_declaration():
    """Mirror of ``test_demotion_banners.py``'s
    ``test_banner_disappearing_from_one_file_is_caught``: strip ONE pinned
    substring from the doc text (in memory, the file on disk is untouched)
    and check the lock reports exactly that failure, not a false pass.
    """
    doc_text = PROTOCOL_MD.read_text(encoding="utf-8")
    assert _GAMMA_1_SUBSTRING in doc_text  # precondition
    stripped = doc_text.replace(_GAMMA_1_SUBSTRING, "nothing to see here")
    assert _GAMMA_1_SUBSTRING not in stripped

    failures = _check_locks(protocol, stripped)
    assert failures == [f"PROTOCOL.md missing {_GAMMA_1_SUBSTRING!r} (section 15.1)"]
