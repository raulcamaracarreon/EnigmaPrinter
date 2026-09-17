import ast
import hashlib
import math
from pathlib import Path
from types import SimpleNamespace


ROOT_DIR = Path(__file__).parent.parent.parent
WEBUI_MAIN = ROOT_DIR / "webui" / "Main.py"


class _UploadedAudio:
    def __init__(self, content: bytes):
        self._content = content

    def getbuffer(self):
        return memoryview(self._content)


def _load_uploaded_narration_helpers(session_state):
    tree = ast.parse(WEBUI_MAIN.read_text(encoding="utf-8"))
    wanted = {
        "_uploaded_narration_fingerprint",
        "_get_matching_uploaded_narration_preview",
    }
    functions = [
        node
        for node in tree.body
        if isinstance(node, ast.FunctionDef) and node.name in wanted
    ]
    assert {node.name for node in functions} == wanted
    module = ast.Module(body=functions, type_ignores=[])
    namespace = {
        "hashlib": hashlib,
        "math": math,
        "st": SimpleNamespace(session_state=session_state),
    }
    exec(compile(module, str(WEBUI_MAIN), "exec"), namespace)
    return (
        namespace["_uploaded_narration_fingerprint"],
        namespace["_get_matching_uploaded_narration_preview"],
    )


def test_uploaded_narration_fingerprint_tracks_script_and_audio_bytes():
    fingerprint, _ = _load_uploaded_narration_helpers({})
    first_audio = _UploadedAudio(b"same audio")
    changed_audio = _UploadedAudio(b"different audio")

    original = fingerprint("La fotografía estaba sobre el andén.", first_audio)
    same = fingerprint("  La fotografía estaba sobre el andén.  ", first_audio)
    changed_script = fingerprint("La fotografía seguía sobre el andén.", first_audio)
    changed_bytes = fingerprint("La fotografía estaba sobre el andén.", changed_audio)

    assert original == same
    assert original != changed_script
    assert original != changed_bytes


def test_matching_uploaded_preview_requires_exact_current_identity_and_real_timeline():
    session_state = {}
    fingerprint, get_matching = _load_uploaded_narration_helpers(session_state)
    audio = _UploadedAudio(b"voiceover bytes")
    params = SimpleNamespace(video_script="Una mujer esperaba en el andén.")
    identity = fingerprint(params.video_script, audio)

    cached = {
        "preview_type": "uploaded",
        "fingerprint": identity,
        "duration": 12.5,
        "narration_timeline": {
            "audio_duration": 12.5,
            "timing_source": "derived",
            "segments": [
                {
                    "index": 1,
                    "start": 0.3,
                    "end": 3.2,
                    "text": "Una mujer esperaba en el andén.",
                    "timing_source": "derived",
                }
            ],
            "alignment_units": [],
        },
    }
    session_state["voice_preview_audio"] = cached

    assert get_matching(params, audio) is cached

    params.video_script = "El texto cambió."
    assert get_matching(params, audio) is None

    params.video_script = "Una mujer esperaba en el andén."
    session_state["voice_preview_audio"] = dict(cached, duration=0.0)
    assert get_matching(params, audio) is None

    session_state["voice_preview_audio"] = dict(
        cached,
        narration_timeline={"audio_duration": 12.5, "segments": []},
    )
    assert get_matching(params, audio) is None
