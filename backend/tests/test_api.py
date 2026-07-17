from __future__ import annotations

import json
import time

from fastapi.testclient import TestClient

from backend.app.episodes import upgrade_rarity
from backend.app.main import create_app
from backend.app.schemas import WorkerResult, WorkerRole
from backend.tests.conftest import wait_until_ready


def begin_conversation(client: TestClient) -> dict:
    started = client.post("/api/session/start").json()["session"]
    session_id = started["session_id"]
    consented = client.post(
        f"/api/session/{session_id}/consent",
        json={"voice_clone_consent": True},
    ).json()["session"]
    assert consented["state"] == "device_check"
    checked = client.post(
        f"/api/session/{session_id}/device-check/complete",
        json={
            "camera_width": 1280,
            "camera_height": 720,
            "camera_fps": 30,
            "face_check_supported": False,
            "face_detected": None,
            "brightness": 120,
        },
    ).json()["session"]
    assert checked["state"] == "conversation"
    assert (
        checked["current_question_text"] == "未来の自分に何か聞きたいことはありますか？"
    )
    return checked


def test_session_flow_capture_and_cleanup(fast_config, monkeypatch) -> None:
    app = create_app(fast_config)

    with TestClient(app) as client:
        wait_until_ready(client)
        started = client.post("/api/session/start").json()["session"]
        session_id = started["session_id"]
        session_dir = fast_config.session_root / session_id
        assert started["state"] == "consent"
        assert session_dir.exists()

        refused = client.post(
            f"/api/session/{session_id}/consent",
            json={"voice_clone_consent": False},
        )
        assert refused.status_code == 422

        consented = client.post(
            f"/api/session/{session_id}/consent",
            json={"voice_clone_consent": True},
        ).json()["session"]
        assert consented["state"] == "device_check"

        video = client.post(
            f"/api/session/{session_id}/media/chunk",
            params={"kind": "video", "sequence": 0, "mime_type": "video/webm"},
            content=b"video-chunk",
        )
        assert video.status_code == 200

        checked = client.post(
            f"/api/session/{session_id}/device-check/complete",
            json={
                "camera_width": 1280,
                "camera_height": 720,
                "camera_fps": 30,
                "face_check_supported": True,
                "face_detected": True,
                "brightness": 120,
            },
        ).json()["session"]
        assert checked["state"] == "conversation"

        listening = client.post(
            f"/api/session/{session_id}/conversation/ai-finished"
        ).json()["session"]
        assert listening["conversation_phase"] == "listening"

        audio = client.post(
            f"/api/session/{session_id}/media/chunk",
            params={"kind": "audio", "sequence": 0, "mime_type": "audio/webm"},
            content=b"audio-answer",
        )
        assert audio.status_code == 200

        answered = client.post(
            f"/api/session/{session_id}/conversation/answer-complete",
            json={
                "sequence": 0,
                "duration_ms": 1200,
                "silence_reason": "silence",
                "byte_count": 12,
            },
        ).json()["session"]
        assert answered["answer_count"] == 1
        assert answered["capture_stats"]["video_chunk_count"] == 1
        assert answered["capture_stats"]["audio_segment_count"] == 1

        utterance = client.post(
            f"/api/session/{session_id}/conversation/utterance",
            json={"text": "未来の技術に興味があります。"},
        ).json()["session"]
        assert utterance["visitor_char_count"] > 0

        generating = client.post(
            f"/api/session/{session_id}/conversation/finish"
        ).json()["session"]
        assert generating["state"] == "generating"
        assert generating["generation_elapsed_seconds"] >= 0

        review = client.post(
            f"/api/session/{session_id}/generation/complete"
        ).json()["session"]
        assert review["state"] == "review"
        assert review["video_artifact"]["implemented"] is False
        assert review["video_artifact"]["metadata_path"] == "output/video-placeholder.json"
        assert review["base_rarity"] in {"R", "SR", "SSR", "UR"}
        assert review["final_rarity"] in {"R", "SR", "SSR", "UR"}
        metadata_path = session_dir / "output" / "video-placeholder.json"
        assert metadata_path.exists()
        debug_artifacts = client.get(
            f"/api/session/{session_id}/debug/artifacts"
        )
        assert debug_artifacts.status_code == 200
        artifact_paths = {
            item["path"] for item in debug_artifacts.json()["artifacts"]
        }
        assert "output/video-placeholder.json" in artifact_paths
        previous_episode_id = review["selected_episode_id"]
        eligible_episodes = app.state.episodes.eligible(fast_config.staff)
        replacement_episode = next(
            episode
            for episode in eligible_episodes
            if episode.id != previous_episode_id
        )
        replacement_effect = app.state.episodes.eligible_effects(fast_config.staff)[0]
        replacement_rarity = upgrade_rarity(
            replacement_episode.base_rarity,
            replacement_effect.rarity_upgrade_steps,
        )
        monkeypatch.setattr(
            app.state.episodes,
            "select",
            lambda settings, rng=None: (
                replacement_episode,
                replacement_effect,
                replacement_rarity,
            ),
        )
        stale_marker = session_dir / "intermediate" / "stale.txt"
        stale_marker.write_text("old generation", encoding="utf-8")
        fast_config.developer.pipeline.stub_step_delay_seconds = 0.2

        regenerating_response = client.post(
            f"/api/session/{session_id}/review/regenerate"
        )
        assert regenerating_response.status_code == 200
        regenerating = regenerating_response.json()["session"]
        assert regenerating["state"] == "generating"
        assert regenerating["completion_reason"] == "operator_regenerated"
        assert regenerating["selected_episode_id"] == replacement_episode.id
        assert regenerating["selected_effect_id"] == replacement_effect.id
        assert not stale_marker.exists()
        assert (session_dir / "input" / "video" / "chunks" / "video-000000.webm").exists()
        assert (session_dir / "input" / "audio" / "answers" / "audio-000000.webm").exists()

        regenerated = client.post(
            f"/api/session/{session_id}/generation/complete"
        ).json()["session"]
        assert regenerated["state"] == "review"
        regenerated_metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
        assert regenerated_metadata["episode_id"] == replacement_episode.id
        assert regenerated_metadata["effect_id"] == replacement_effect.id

        reset = client.post("/api/control/reset").json()
        assert reset["session"] is None
        assert not session_dir.exists()


def test_automatic_close_waits_for_capture_finalize(fast_config) -> None:
    app = create_app(fast_config)

    with TestClient(app) as client:
        wait_until_ready(client)
        conversation = begin_conversation(client)
        session_id = conversation["session_id"]
        closing = client.post(
            f"/api/session/{session_id}/conversation/utterance",
            json={"text": "未来の研究を楽しみにしています。" * 60},
        ).json()["session"]

        assert closing["state"] == "conversation"
        assert closing["conversation_phase"] == "closing"
        time.sleep(0.6)
        still_closing = client.get("/api/session/current").json()["session"]
        assert still_closing["state"] == "conversation"
        assert still_closing["conversation_phase"] == "closing"

        generating = client.post(
            f"/api/session/{session_id}/conversation/finish"
        ).json()["session"]
        assert generating["state"] == "generating"
        assert generating["completion_reason"] == "target_transcript_reached"
        repeated = client.post(
            f"/api/session/{session_id}/conversation/finish"
        )
        assert repeated.status_code == 200



def test_active_session_is_reused_and_settings_are_snapshotted(fast_config) -> None:
    app = create_app(fast_config)

    with TestClient(app) as client:
        wait_until_ready(client)
        first = client.post("/api/session/start").json()["session"]
        second = client.post("/api/session/start").json()["session"]
        assert first["session_id"] == second["session_id"]

        settings = client.get("/api/settings").json()["settings"]
        changed = {**settings, "generation_time_limit_seconds": 3600}
        response = client.put("/api/settings", json=changed)
        assert response.status_code == 200

        current = client.get("/api/session/current").json()["session"]
        assert current["generation_time_limit_seconds"] == 1800


def test_emergency_stop_deletes_files_and_restarts_preparation(fast_config) -> None:
    app = create_app(fast_config)

    with TestClient(app) as client:
        wait_until_ready(client)
        session = client.post("/api/session/start").json()["session"]
        session_dir = fast_config.session_root / session["session_id"]

        stopped = client.post("/api/control/emergency-stop").json()

        assert stopped["session"] is None
        assert stopped["preparation"]["state"] == "loading"
        assert not session_dir.exists()
        wait_until_ready(client)


def test_interview_abandon_deletes_session(fast_config) -> None:
    app = create_app(fast_config)

    with TestClient(app) as client:
        wait_until_ready(client)
        conversation = begin_conversation(client)
        session_id = conversation["session_id"]
        session_dir = fast_config.session_root / session_id

        response = client.post(f"/api/session/{session_id}/abandon").json()

        assert response["deleted"] is True
        assert not session_dir.exists()
        assert client.get("/api/session/current").json()["session"] is None


def test_invalid_fixed_episode_is_not_saved(fast_config) -> None:
    app = create_app(fast_config)
    original = fast_config.staff.model_dump(mode="json")

    with TestClient(app) as client:
        wait_until_ready(client)
        invalid = {
            **original,
            "episode_selection": "fixed",
            "fixed_episode_id": "biohazard-survivor",
        }
        response = client.put("/api/settings", json=invalid)

        assert response.status_code == 422
        assert fast_config.staff.model_dump(mode="json") == original



def test_answer_complete_adds_asr_transcript(fast_config, monkeypatch) -> None:
    app = create_app(fast_config)

    async def fake_interview_workers(session, role, **kwargs):
        if role == WorkerRole.STREAMING_ASR:
            return WorkerResult(
                request_id="asr-test",
                worker=role,
                backend="fake-asr",
                model_id="fake/model",
                model_revision="test",
                implemented=True,
                metadata={"text": "未来の乗り物を研究してみたいです。"},
            )
        assert role == WorkerRole.INTERVIEW_LLM
        return WorkerResult(
            request_id="llm-test",
            worker=role,
            backend="fake-llm",
            model_id="fake/model",
            model_revision="test",
            implemented=True,
            metadata={
                "interview_turn": {
                    "acquired_information": {
                        "future_wishes": ["未来の乗り物の研究"]
                    },
                    "asked_topics": ["research-motivation"],
                    "next_topics": ["ideal-future"],
                    "visitor_char_count": 18,
                    "elapsed_seconds": 10,
                    "next_utterance": "どんな乗り物を作れたら一番うれしいですか？",
                }
            },
        )

    monkeypatch.setattr(
        app.state.workers,
        "run_prepared_role",
        fake_interview_workers,
    )
    with TestClient(app) as client:
        wait_until_ready(client)
        session = begin_conversation(client)
        session_id = session["session_id"]
        client.post(f"/api/session/{session_id}/conversation/ai-finished")
        uploaded = client.post(
            f"/api/session/{session_id}/media/chunk",
            params={"kind": "audio", "sequence": 0, "mime_type": "audio/webm"},
            content=b"audio-answer",
        )
        assert uploaded.status_code == 200

        response = client.post(
            f"/api/session/{session_id}/conversation/answer-complete",
            json={
                "sequence": 0,
                "duration_ms": 1200,
                "silence_reason": "silence",
                "byte_count": 12,
            },
        )
        assert response.status_code == 200
        body = response.json()
        assert body["warning"] is None
        assert body["session"]["visitor_char_count"] == len(
            "未来の乗り物を研究してみたいです。"
        )
        assert (
            body["session"]["latest_visitor_transcript"]
            == "未来の乗り物を研究してみたいです。"
        )
        assert (
            body["session"]["current_question_text"]
            == "どんな乗り物を作れたら一番うれしいですか？"
        )
        transcript_path = (
            fast_config.session_root / session_id / "input" / "transcript.json"
        )
        transcript = json.loads(transcript_path.read_text(encoding="utf-8"))
        visitor_entries = [
            entry for entry in transcript if entry["speaker"] == "visitor"
        ]
        assert visitor_entries[-1]["text"] == "未来の乗り物を研究してみたいです。"


def test_debug_error_retains_artifacts_until_reset(fast_config) -> None:
    app = create_app(fast_config)

    with TestClient(app) as client:
        wait_until_ready(client)
        session = client.post("/api/session/start").json()["session"]
        session_id = session["session_id"]
        session_dir = fast_config.session_root / session_id

        script_path = (
            session_dir
            / "intermediate"
            / "script_design_llm_worker"
            / "script-design.json"
        )
        image_path = (
            session_dir
            / "intermediate"
            / "image_generation_worker"
            / "future-image.png"
        )
        audio_path = (
            session_dir
            / "intermediate"
            / "voice_clone_tts_worker"
            / "generated-audio.wav"
        )
        script_path.parent.mkdir(parents=True, exist_ok=True)
        image_path.parent.mkdir(parents=True, exist_ok=True)
        audio_path.parent.mkdir(parents=True, exist_ok=True)
        script_path.write_text(
            json.dumps({"future_world": "軌道都市"}, ensure_ascii=False),
            encoding="utf-8",
        )
        image_path.write_bytes(b"debug-image")
        audio_path.write_bytes(b"debug-audio")

        failed = client.post(f"/api/session/{session_id}/debug/error")
        assert failed.status_code == 200
        assert failed.json()["session"]["state"] == "error"
        assert session_dir.is_dir()

        response = client.get(f"/api/session/{session_id}/debug/artifacts")
        assert response.status_code == 200
        body = response.json()
        assert body["retained"] is True
        artifacts = {artifact["path"]: artifact for artifact in body["artifacts"]}
        script_relative = "intermediate/script_design_llm_worker/script-design.json"
        image_relative = "intermediate/image_generation_worker/future-image.png"
        audio_relative = "intermediate/voice_clone_tts_worker/generated-audio.wav"
        assert json.loads(artifacts[script_relative]["text_preview"])["future_world"] == "軌道都市"
        assert artifacts[image_relative]["kind"] == "image"
        assert artifacts[audio_relative]["kind"] == "audio"

        image_response = client.get(f"/{artifacts[image_relative]['media_url']}")
        assert image_response.status_code == 200
        assert image_response.content == b"debug-image"

        client.post("/api/control/reset")
        assert not session_dir.exists()
