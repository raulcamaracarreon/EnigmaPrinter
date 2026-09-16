from __future__ import annotations

import json
import textwrap
from pathlib import Path


ROOT = Path(__file__).resolve().parent.parent
MAIN = ROOT / "webui" / "Main.py"
EN_LOCALE = ROOT / "webui" / "i18n" / "en.json"
ES_LOCALE = ROOT / "webui" / "i18n" / "es.json"


def replace_once(source: str, old: str, new: str, label: str) -> str:
    count = source.count(old)
    if count != 1:
        raise RuntimeError(f"{label}: expected exactly one match, found {count}")
    return source.replace(old, new, 1)


def add_translation(path: Path, values: dict[str, str]) -> None:
    data = json.loads(path.read_text(encoding="utf-8"))
    translations = data.setdefault("Translation", {})
    translations.update(values)
    path.write_text(
        json.dumps(data, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )


def patch_main() -> None:
    source = MAIN.read_text(encoding="utf-8")

    source = replace_once(
        source,
        "    metaso_minimax,\n    narration_timeline,\n",
        "    metaso_minimax,\n    narration_alignment,\n    narration_timeline,\n",
        "narration_alignment import",
    )

    # Reuse the exact existing Full Audio downstream pipeline. The extracted block
    # contains Narration Timeline -> Semantic Scene Timeline -> Active Shot Plan ->
    # optional refinement -> Visual Prompts/Media Plan. Keeping one renderer prevents
    # uploaded audio and internal TTS from drifting into separate implementations.
    timeline_start_marker = (
        '            timeline = cached_preview.get("narration_timeline")\n'
    )
    next_function_marker = "\n\ndef _video_term_lines_for_timeline(value) -> list[str]:\n"
    start = source.index(timeline_start_marker)
    end = source.index(next_function_marker, start)
    original_block = source[start:end]
    if "Semantic Scene Timeline" not in original_block or "Active Shot Plan" not in original_block:
        raise RuntimeError("audio-first preview block markers changed unexpectedly")

    helper_body = textwrap.indent(textwrap.dedent(original_block), "    ")
    helper = (
        "\n\ndef _render_audio_first_timeline_plan(params, cached_preview):\n"
        "    \"\"\"Render the shared audio-first planning chain for any timed narration.\"\"\"\n"
        f"{helper_body}"
    )
    source = (
        source[:start]
        + "            _render_audio_first_timeline_plan(params, cached_preview)\n"
        + helper
        + source[end:]
    )

    upload_helpers = r'''

def _uploaded_narration_fingerprint(script: str, uploaded_audio_file) -> str:
    """Bind uploaded narration analysis to both the exact script and audio bytes."""
    digest = hashlib.sha256()
    digest.update(b"enigmaprinter-uploaded-narration-v1\0")
    digest.update(str(script or "").strip().encode("utf-8"))
    digest.update(b"\0")
    digest.update(uploaded_audio_file.getbuffer())
    return digest.hexdigest()


def _invalidate_uploaded_narration_downstream_state():
    """Drop plans that were derived from a previous script/audio pair."""
    for key in (
        "media_shot_refinement_preview",
        "visual_shot_plan_preview",
        "media_plan_preview",
        "applied_visual_shot_plan_fingerprint",
        "applied_visual_shot_plan_voice_fingerprint",
    ):
        st.session_state.pop(key, None)


def _analyze_uploaded_narration(params, uploaded_audio_file) -> dict:
    script_content = str(params.video_script or "").strip()
    if not script_content:
        raise narration_alignment.NarrationAlignmentError(
            "script must contain narration text"
        )

    audio_bytes = bytes(uploaded_audio_file.getbuffer())
    fingerprint = _uploaded_narration_fingerprint(
        script_content,
        uploaded_audio_file,
    )
    temp_dir = utils.storage_dir("temp", create=True)
    temp_audio_path = _build_uploaded_file_path(
        uploaded_audio_file,
        temp_dir,
        CUSTOM_AUDIO_EXTENSIONS,
        "uploaded-narration-analysis",
    )
    try:
        with open(temp_audio_path, "wb") as file:
            file.write(audio_bytes)
        alignment = narration_alignment.align_external_narration(
            audio_file=temp_audio_path,
            script=script_content,
        )
    finally:
        try:
            os.remove(temp_audio_path)
        except FileNotFoundError:
            pass
        except OSError as exc:
            logger.warning(
                "failed to delete uploaded narration analysis file: "
                f"path={temp_audio_path}, error={exc}"
            )

    timeline = alignment.timeline.to_dict()
    duration = float(timeline.get("audio_duration", 0.0) or 0.0)
    mime_type = str(getattr(uploaded_audio_file, "type", "") or "").strip()
    if not mime_type.startswith("audio/"):
        mime_type = (
            mimetypes.guess_type(str(uploaded_audio_file.name or ""))[0]
            or "audio/mpeg"
        )

    return {
        "preview_type": "uploaded",
        "fingerprint": fingerprint,
        "duration": duration,
        "mime_type": mime_type,
        "narration_timeline": timeline,
        "content_digest": hashlib.sha256(
            script_content.encode("utf-8")
        ).hexdigest(),
        "alignment_summary": {
            "matched": alignment.matched_word_count,
            "script": alignment.script_word_count,
            "recognized": alignment.recognized_word_count,
            "script_coverage": alignment.script_coverage,
            "recognized_coverage": alignment.recognized_coverage,
        },
    }


def _get_matching_uploaded_narration_preview(
    params,
    uploaded_audio_file,
) -> dict | None:
    if uploaded_audio_file is None:
        return None
    script_content = str(params.video_script or "").strip()
    if not script_content:
        return None
    try:
        expected_fingerprint = _uploaded_narration_fingerprint(
            script_content,
            uploaded_audio_file,
        )
    except Exception:
        return None

    cached_preview = st.session_state.get("voice_preview_audio")
    if (
        not isinstance(cached_preview, dict)
        or cached_preview.get("preview_type") != "uploaded"
        or cached_preview.get("fingerprint") != expected_fingerprint
        or not isinstance(cached_preview.get("narration_timeline"), dict)
        or not cached_preview["narration_timeline"].get("segments")
    ):
        return None

    duration = cached_preview.get("duration")
    if (
        not isinstance(duration, (int, float))
        or not math.isfinite(duration)
        or duration <= 0
    ):
        return None
    return cached_preview


def _render_uploaded_narration_analysis(params, uploaded_audio_file):
    """Analyze uploaded voiceover against the current script, then reuse audio-first UI."""
    if uploaded_audio_file is None:
        return

    script_content = str(params.video_script or "").strip()
    if not script_content:
        st.caption(tr("Voiceover Script Required"))
        return

    current_fingerprint = _uploaded_narration_fingerprint(
        script_content,
        uploaded_audio_file,
    )
    cached_preview = st.session_state.get("voice_preview_audio")
    if (
        isinstance(cached_preview, dict)
        and cached_preview.get("preview_type") == "uploaded"
        and cached_preview.get("fingerprint") != current_fingerprint
    ):
        st.session_state.pop("voice_preview_audio", None)
        _invalidate_uploaded_narration_downstream_state()
        cached_preview = None

    analyze_requested = st.button(
        tr("Analyze Uploaded Narration"),
        key="analyze_uploaded_narration_button",
        icon=":material/graphic_eq:",
        use_container_width=True,
        help=tr("Analyze Uploaded Narration Help"),
    )
    if analyze_requested and (
        not isinstance(cached_preview, dict)
        or cached_preview.get("fingerprint") != current_fingerprint
    ):
        try:
            with st.spinner(tr("Analyzing Uploaded Narration")):
                preview_result = _analyze_uploaded_narration(
                    params,
                    uploaded_audio_file,
                )
        except narration_alignment.NarrationAlignmentError as exc:
            logger.warning(f"uploaded narration alignment rejected: {exc}")
            if (
                isinstance(cached_preview, dict)
                and cached_preview.get("preview_type") == "uploaded"
            ):
                st.session_state.pop("voice_preview_audio", None)
            _invalidate_uploaded_narration_downstream_state()
            st.error(
                tr("Uploaded Narration Alignment Failed").format(error=str(exc))
            )
            return
        except Exception as exc:
            logger.exception("uploaded narration analysis failed")
            _invalidate_uploaded_narration_downstream_state()
            st.error(
                tr("Uploaded Narration Alignment Failed").format(error=str(exc))
            )
            return
        else:
            _invalidate_uploaded_narration_downstream_state()
            st.session_state["voice_preview_audio"] = preview_result
            cached_preview = preview_result

    cached_preview = _get_matching_uploaded_narration_preview(
        params,
        uploaded_audio_file,
    )
    if not cached_preview:
        return

    summary = cached_preview.get("alignment_summary", {}) or {}
    st.success(
        tr("Uploaded Narration Alignment Summary").format(
            matched=int(summary.get("matched", 0) or 0),
            script=int(summary.get("script", 0) or 0),
            coverage=float(summary.get("script_coverage", 0.0) or 0.0),
            recognized_coverage=float(
                summary.get("recognized_coverage", 0.0) or 0.0
            ),
        )
    )
    _render_audio_first_timeline_plan(params, cached_preview)
'''

    source = replace_once(
        source,
        "\n\ndef _render_audio_settings(panel, params):\n",
        upload_helpers + "\n\ndef _render_audio_settings(panel, params):\n",
        "uploaded narration helpers",
    )

    source = replace_once(
        source,
        '''                    st.info(
                        tr(
                            "Custom audio will be used directly. TTS synthesis will be skipped for this task."
                        )
                    )
''',
        '''                    st.info(
                        tr(
                            "Custom audio will be used directly. TTS synthesis will be skipped for this task."
                        )
                    )
                    _render_uploaded_narration_analysis(
                        params,
                        uploaded_audio_file,
                    )
''',
        "upload narration analysis renderer",
    )

    # A reusable TTS preview and an uploaded narration analysis now share one
    # generic timing identity for locked shot validation. Only the TTS preview is
    # still forwarded to the task worker as synthesized audio/SubMaker state.
    source = replace_once(
        source,
        '''        "audio_bytes": bytes(cached_preview["audio_bytes"]),
        "duration": float(duration),
''',
        '''        "audio_bytes": bytes(cached_preview["audio_bytes"]),
        "duration": float(duration),
        "fingerprint": expected_fingerprint,
''',
        "reusable full preview fingerprint",
    )

    function_start = source.index("def _matching_applied_media_shot_timeline(")
    function_end = source.index("\n\ndef _get_reusable_full_voice_preview", function_start)
    function = source[function_start:function_end]
    function = function.replace("reusable_voice_preview", "narration_preview")
    function = function.replace(
        '''    current_voice_preview = st.session_state.get("voice_preview_audio")
    current_voice_fingerprint = str(
        current_voice_preview.get("fingerprint", "")
        if isinstance(current_voice_preview, dict)
        else ""
    )
''',
        '''    current_voice_fingerprint = str(
        narration_preview.get("fingerprint", "")
        if isinstance(narration_preview, dict)
        else ""
    )
''',
    )
    function = function.replace(
        "the applied Visual Shot Plan requires the matching Full Audio preview; "
        "generate Full Audio again before starting timeline-aware generation",
        "the applied Visual Shot Plan requires matching narration timing; "
        "analyze or generate the current voiceover again before timeline-aware generation",
    )
    function = function.replace(
        "the applied Visual Shot Plan belongs to a different voiceover preview; "
        "regenerate the Visual Shot Plan after the current Full Audio",
        "the applied Visual Shot Plan belongs to different narration timing; "
        "regenerate the Visual Shot Plan after the current voiceover",
    )
    if "reusable_voice_preview" in function:
        raise RuntimeError("generic narration preview rename was incomplete")
    source = source[:function_start] + function + source[function_end:]

    source = replace_once(
        source,
        '''        reusable_voice_preview = _get_reusable_full_voice_preview(
            params,
            voice_mode,
        )

''',
        '''        reusable_voice_preview = _get_reusable_full_voice_preview(
            params,
            voice_mode,
        )
        timeline_voice_preview = reusable_voice_preview
        if voice_mode == VOICE_MODE_UPLOAD:
            timeline_voice_preview = _get_matching_uploaded_narration_preview(
                params,
                uploaded_audio_file,
            )

''',
        "generation narration timing selection",
    )
    source = replace_once(
        source,
        '''            params.media_shot_timeline = _matching_applied_media_shot_timeline(
                params,
                reusable_voice_preview,
            )
''',
        '''            params.media_shot_timeline = _matching_applied_media_shot_timeline(
                params,
                timeline_voice_preview,
            )
''',
        "locked timeline narration context",
    )

    MAIN.write_text(source, encoding="utf-8")


def main() -> None:
    patch_main()
    add_translation(
        EN_LOCALE,
        {
            "Analyze Uploaded Narration": "Analyze Uploaded Narration",
            "Analyze Uploaded Narration Help": (
                "Align the uploaded voiceover to the current script and build the same "
                "Narration Timeline, Semantic Scene Timeline, and Active Shot Plan used by internal TTS."
            ),
            "Analyzing Uploaded Narration": "Analyzing Uploaded Narration",
            "Uploaded Narration Alignment Failed": (
                "Uploaded narration alignment failed: {error}"
            ),
            "Uploaded Narration Alignment Summary": (
                "Uploaded narration aligned: {matched}/{script} script words matched "
                "({coverage:.0%}); {recognized_coverage:.0%} of recognized speech belongs to the script."
            ),
        },
    )
    add_translation(
        ES_LOCALE,
        {
            "Analyze Uploaded Narration": "Analizar narración cargada",
            "Analyze Uploaded Narration Help": (
                "Alinea la narración cargada con el guion actual y construye el mismo "
                "Timeline de narración, Timeline semántico de escenas y Active Shot Plan del TTS interno."
            ),
            "Analyzing Uploaded Narration": "Analizando narración cargada",
            "Uploaded Narration Alignment Failed": (
                "No se pudo alinear la narración cargada: {error}"
            ),
            "Uploaded Narration Alignment Summary": (
                "Narración cargada alineada: {matched}/{script} palabras del guion coinciden "
                "({coverage:.0%}); {recognized_coverage:.0%} del habla reconocida pertenece al guion."
            ),
        },
    )


if __name__ == "__main__":
    main()
