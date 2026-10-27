from __future__ import annotations

import hashlib
import json
import os
import sys
from pathlib import Path
from typing import Any


PROTOCOL_VERSION = 1
CACHE_NAMESPACE = "desktop-pet-laya-s256-k8-plugin"


def _send(message: dict[str, Any]) -> None:
    print(json.dumps({"protocol_version": PROTOCOL_VERSION, **message},
                      ensure_ascii=False, separators=(",", ":"), allow_nan=False), flush=True)


def _load_session(root: Path):
    from som_probe.assets import load_lock, sha256_file
    from som_probe.tryout import (
        _ExclusiveModelLock, _LayaSession, _SOURCE_RUN_ID, _question_templates,
    )

    templates, canaries = _question_templates(root)
    lock = _ExclusiveModelLock()
    lock.__enter__()
    try:
        session = _LayaSession(
            root, _SOURCE_RUN_ID, templates, canaries, execution_device="NPU",
            cache_namespace=CACHE_NAMESPACE,
        )
    except Exception:
        lock.__exit__(*sys.exc_info())
        raise
    try:
        model_lock = load_lock(root)
        source = model_lock["sources"]["reference_checkpoint"]
        ir_path = session._ir_path
        ir_bin = ir_path.with_suffix(".bin")
        import openvino as ov
        metadata = {
            "model_repository": source["repository"],
            "model_revision": source["revision"],
            "model_lock_sha256": sha256_file(root / "configs" / "models.lock.json"),
            "profile": "s256-k8",
            "question_definition": "dev-v3 + desktop-pet-actions-v1",
            "precision": "FP16",
            "backend": f"OpenVINO {ov.__version__}",
            "requested_device": "NPU",
            "worker_pid": os.getpid(),
            "execution_devices": list(session._execution_devices),
            "device_architecture": str(session._actual_architecture),
            "compiler_type": str(session._properties.get("NPU_COMPILER_TYPE", "未取得")),
            "compiler_version": str(session._properties.get("NPU_COMPILER_VERSION", "未取得")),
            "loaded_from_cache": session._properties.get("LOADED_FROM_CACHE"),
            "source_run_id": _SOURCE_RUN_ID,
            "ir_xml_sha256": sha256_file(ir_path),
            "ir_bin_sha256": sha256_file(ir_bin),
            "dev_v3_sha256": sha256_file(root / "datasets" / "dev-v3.jsonl"),
        }
    except Exception:
        error_info = sys.exc_info()
        try:
            session.close()
        except Exception:
            pass
        lock.__exit__(*error_info)
        raise
    return session, lock, metadata


def _handle_infer(session: Any, request: dict[str, Any]) -> dict[str, Any]:
    state = request.get("state")
    request_id = request.get("request_id")
    if not isinstance(request_id, int) or not isinstance(state, str):
        return {"type": "error", "request_id": request_id, "code": "WORKER_REQUEST_INVALID"}
    state_hash = hashlib.sha256(state.encode("utf-8")).hexdigest()
    actions: dict[str, str] = {}
    outputs: dict[str, Any] = {}
    traces: dict[str, Any] = {}
    from desktop_pet.behavior import QUESTIONS

    for question_id, question in QUESTIONS.items():
        output, trace = session.predict_choice(question_id, question, state)
        decision = output.get("decision")
        if not isinstance(decision, str):
            return {"type": "error", "request_id": request_id,
                    "code": "WORKER_ACTION_CONTRACT_INVALID", "state_sha256": state_hash}
        actions[question_id] = decision
        outputs[question_id] = output
        traces[question_id] = {
            "input_sha256": trace["input_sha256"],
            "required_total_tokens": trace["required_total_tokens"],
            "state_truncated": trace["state_truncated"],
            "candidate_ids": trace["candidate_ids"],
        }
    return {"type": "inference", "request_id": request_id,
            "state_sha256": state_hash, "actions": actions,
            "model_outputs": outputs, "input_traces": traces}


def run(root: Path) -> int:
    session = None
    lock = None
    try:
        _send({"type": "status", "status": "loading", "device": "NPU",
               "precision": "FP16", "pid": os.getpid()})
        session, lock, metadata = _load_session(root)
        _send({"type": "ready", "metadata": metadata})
        for line in sys.stdin:
            try:
                request = json.loads(line)
            except json.JSONDecodeError:
                _send({"type": "error", "code": "WORKER_JSON_INVALID"})
                continue
            if not isinstance(request, dict) or request.get("protocol_version") != PROTOCOL_VERSION:
                _send({"type": "error", "code": "WORKER_PROTOCOL_MISMATCH"})
                continue
            if request.get("type") == "shutdown":
                _send({"type": "shutdown-complete"})
                break
            if request.get("type") == "infer":
                try:
                    _send(_handle_infer(session, request))
                except Exception:
                    _send({"type": "error", "request_id": request.get("request_id"),
                           "code": "WORKER_INFERENCE_FAILED"})
    except Exception as error:
        code = getattr(error, "code", None)
        _send({"type": "failure", "code": code or type(error).__name__})
        return 1
    finally:
        if session is not None:
            try:
                session.close()
            except Exception:
                pass
        if lock is not None:
            try:
                lock.__exit__(None, None, None)
            except Exception:
                pass
    return 0


def main() -> int:
    app_root = Path(__file__).resolve().parents[1]
    root = Path(__file__).resolve().parents[3] / "som-probe"
    if len(sys.argv) == 3 and sys.argv[1] == "--root":
        root = Path(sys.argv[2]).resolve()
    if not root.is_dir():
        _send({"type": "failure", "code": "SOM_PROBE_ROOT_MISSING"})
        return 2
    sys.path.insert(0, str(root))
    sys.path.insert(0, str(app_root))
    return run(root)


if __name__ == "__main__":
    raise SystemExit(main())
