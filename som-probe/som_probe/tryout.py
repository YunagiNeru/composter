"""Local, non-recording interactive trial for the pinned Laya and Sokudan models."""

from __future__ import annotations

import ctypes
import hashlib
import json
import math
import os
import tarfile
from pathlib import Path
from typing import Any, Callable

from .assets import load_lock, sha256_file, verify_assets
from .contracts import Profile
from .dataset import load_dataset
from .runtime import (
    _execution_device_names,
    _input_arrays,
    _input_digest,
    _load_run_conversion,
    _select_outputs,
    _verify_saved_ir,
    decode_laya_output,
    inspect_device,
)
from .storage import open_run
from .sokudan_cpu import (
    _candidate_ids,
    _question_kind,
    _prepare_question_inputs,
    _run_environment,
    _validate_calibration,
    adapt_official_answer,
)


_TYPES = ("choice", "score", "noul")
_SOURCE_RUN_ID = "20261025-041751-58713e9e"
_MUTEX_NAME = "Local\\som-probe-interactive-model"
_DISPLAY_CHOICE = {"task": "未完了", "none": "対象外", "need_info": "確認待ち"}
_DISPLAY_SCORE = (
    "記載なし、または時期が未確定・曖昧",
    "日付・相対日付・曜日・期間だけ、または時刻だけ",
    "日付（相対日付を含む）と時刻の両方",
)


class TryoutError(RuntimeError):
    def __init__(self, code: str, *, recoverable: bool = False):
        super().__init__(code)
        self.code = code
        self.recoverable = recoverable


class _ExclusiveModelLock:
    """Keep the two interactive model sessions mutually exclusive without a lock file."""

    def __init__(self) -> None:
        self._kernel32 = None
        self._handle = None

    def __enter__(self) -> _ExclusiveModelLock:
        if os.name != "nt":
            raise TryoutError("WINDOWS_RUNTIME_REQUIRED")
        self._kernel32 = ctypes.WinDLL("kernel32", use_last_error=True)
        create_mutex = self._kernel32.CreateMutexW
        create_mutex.argtypes = (ctypes.c_void_p, ctypes.c_bool, ctypes.c_wchar_p)
        create_mutex.restype = ctypes.c_void_p
        close_handle = self._kernel32.CloseHandle
        close_handle.argtypes = (ctypes.c_void_p,)
        close_handle.restype = ctypes.c_bool

        ctypes.set_last_error(0)
        self._handle = create_mutex(None, True, _MUTEX_NAME)
        error = ctypes.get_last_error()
        if not self._handle:
            raise TryoutError("MODEL_SESSION_LOCK_FAILED")
        if error == 183:  # ERROR_ALREADY_EXISTS
            close_handle(self._handle)
            self._handle = None
            raise TryoutError("ANOTHER_MODEL_SESSION_IS_RUNNING")
        return self

    def __exit__(self, exc_type, exc, traceback) -> bool:
        if self._handle:
            release_mutex = self._kernel32.ReleaseMutex
            release_mutex.argtypes = (ctypes.c_void_p,)
            release_mutex.restype = ctypes.c_bool
            released = bool(release_mutex(self._handle))
            self._kernel32.CloseHandle(self._handle)
            self._handle = None
            if not released and exc_type is None:
                raise TryoutError("MODEL_SESSION_LOCK_RELEASE_FAILED")
        return False


def _canonical(value: Any) -> str:
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"), allow_nan=False)


class _PinnedLayaTokenizer:
    """Use the pinned tokenizer JSON without importing Laya's Torch runtime."""

    def __init__(self, tokenizer_path: Path, tokenizer_config: dict[str, Any]) -> None:
        try:
            from tokenizers import Tokenizer

            self._tokenizer = Tokenizer.from_file(str(tokenizer_path))
            self._tokenizer.no_truncation()
            self._tokenizer.no_padding()
            self.mask_token = tokenizer_config["mask_token"]
            self.cls_token_id = self._token_id(tokenizer_config["cls_token"])
            self.sep_token_id = self._token_id(tokenizer_config["sep_token"])
            self.mask_token_id = self._token_id(self.mask_token)
            self.pad_token_id = self._token_id(tokenizer_config["pad_token"])
        except TryoutError:
            raise
        except Exception:
            raise TryoutError("LAYA_TOKENIZER_LOAD_FAILED") from None

    def _token_id(self, token: str) -> int:
        token_id = self._tokenizer.token_to_id(token)
        if token_id is None:
            raise TryoutError("LAYA_TOKENIZER_SPECIAL_TOKEN_MISSING")
        return int(token_id)

    def encode(self, text: str) -> list[int]:
        return list(self._tokenizer.encode(text, add_special_tokens=False).ids)


def _question_templates(root: Path) -> tuple[dict[str, dict[str, Any]], dict[str, dict[str, Any]]]:
    snapshot = load_dataset(root, "dev-v3", Profile(sequence_length=256))
    templates: dict[str, dict[str, Any]] = {}
    canaries: dict[str, dict[str, Any]] = {}
    template_ids: dict[str, str] = {}
    for case in snapshot["cases"]:
        if not isinstance(case.get("state"), str) or len(case.get("questions", {})) != 1:
            raise TryoutError("TRYOUT_TEMPLATE_CONTRACT_INVALID")
        question_id, question = next(iter(case["questions"].items()))
        kind = _question_kind(question)
        if kind not in _TYPES:
            raise TryoutError("TRYOUT_TEMPLATE_TYPE_UNSUPPORTED")
        if kind not in templates:
            templates[kind] = question
            canaries[kind] = {"case_id": case["case_id"], "state": case["state"],
                              "question_id": question_id, "question": question}
            template_ids[kind] = question_id
        elif template_ids[kind] != question_id:
            raise TryoutError("TRYOUT_TEMPLATE_DEFINITIONS_DIFFER")
        elif kind == "choice":
            shared = templates[kind]
            instruction = question.get("instructions")
            shared_instruction = shared.get("instructions")
            if (_canonical(shared.get("criteria")) != _canonical(question.get("criteria"))
                    or not isinstance(instruction, str) or not isinstance(shared_instruction, str)
                    or (instruction != shared_instruction
                        and not instruction.startswith(shared_instruction + " "))):
                raise TryoutError("TRYOUT_TEMPLATE_DEFINITIONS_DIFFER")
        elif _canonical(templates[kind]) != _canonical(question):
            raise TryoutError("TRYOUT_TEMPLATE_DEFINITIONS_DIFFER")
    if set(templates) != set(_TYPES):
        raise TryoutError("TRYOUT_TEMPLATE_TYPE_MISSING")
    return templates, canaries


def _verify_extracted_laya_source(root: Path, source_dir: Path, commit: str) -> None:
    archive = root / ".local" / "sources" / f"laya-{commit}.tar.gz"
    prefix = f"laya-{commit}/"
    checked = 0
    try:
        with tarfile.open(archive, "r:gz") as bundle:
            for member in bundle.getmembers():
                if (not member.isfile() or not member.name.startswith(prefix)
                        or not member.name.endswith(".py")):
                    continue
                relative = member.name.removeprefix(prefix)
                if not relative.startswith("laya/"):
                    continue
                extracted = bundle.extractfile(member)
                local_path = source_dir / Path(*relative.split("/"))
                if extracted is None or not local_path.is_file():
                    raise TryoutError("PINNED_LAYA_SOURCE_MISMATCH")
                with extracted:
                    if extracted.read() != local_path.read_bytes():
                        raise TryoutError("PINNED_LAYA_SOURCE_MISMATCH")
                checked += 1
    except TryoutError:
        raise
    except Exception:
        raise TryoutError("PINNED_LAYA_SOURCE_UNREADABLE") from None
    if checked == 0:
        raise TryoutError("PINNED_LAYA_SOURCE_MISSING")


def _validate_laya_source(root: Path, run_id: str) -> tuple[Any, dict[str, Any], Path, dict[str, Any]]:
    """Read the previous Run only for its pinned IR and compatibility fingerprints."""
    if run_id != _SOURCE_RUN_ID:
        raise TryoutError("LAYA_SOURCE_RUN_NOT_APPROVED")
    try:
        source_run = open_run(root, run_id)
        manifest = source_run.read_manifest()
        conversion = _load_run_conversion(root, source_run)
        npu_result = json.loads((source_run.path / "npu-result.json").read_text(encoding="utf-8"))
    except TryoutError:
        raise
    except Exception:
        raise TryoutError("LAYA_SOURCE_RUN_UNREADABLE") from None

    expected_profile = {"name": "s256-k8", "batch": 1, "sequence_length": 256, "candidate_slots": 8}
    stages = manifest.get("stages", {})
    if (manifest.get("status") != "COMPLETE" or manifest.get("worktree_dirty") is not False
            or manifest.get("profile_config") != expected_profile
            or stages.get("G2-cpu-check", {}).get("status") != "PASS"
            or stages.get("G3", {}).get("status") != "PASS"):
        raise TryoutError("LAYA_SOURCE_RUN_NOT_COMPLETE")
    if (conversion.get("profile") != "s256-k8" or conversion.get("sequence_length") != 256
            or npu_result.get("status") != "PASS" or npu_result.get("explicit_device") != "NPU"
            or npu_result.get("execution_devices") != ["NPU"]
            or npu_result.get("device_architecture") != "3720"):
        raise TryoutError("LAYA_SOURCE_IR_CONTRACT_MISMATCH")
    requested = npu_result.get("requested_compile_configuration", {})
    actual = npu_result.get("compiled_model_properties", {})
    if (requested.get("NPU_COMPILER_TYPE") != "PLUGIN"
            or not str(actual.get("NPU_COMPILER_TYPE", "")).upper().endswith("PLUGIN")):
        raise TryoutError("LAYA_SOURCE_COMPILER_MISMATCH")

    lock = load_lock(root)
    recorded_hashes = manifest.get("input_sha256", {})
    for relative in (
        "configs/probe.json", "configs/models.lock.json", "requirements/run-win-x64.lock",
        "datasets/manifest-v3.json", "datasets/dev-v3.jsonl", "configs/evaluation-contract-v2.json",
        "datasets/dev-v3-revisions.json",
    ):
        if recorded_hashes.get(relative) != sha256_file(root / relative):
            raise TryoutError("LAYA_SOURCE_INPUT_FINGERPRINT_MISMATCH")
    g0_assets = stages.get("G0", {}).get("verified_artifact_sha256", {})
    if g0_assets != lock.get("artifacts"):
        raise TryoutError("LAYA_SOURCE_MODEL_FINGERPRINT_MISMATCH")
    if conversion.get("source_model_sha256") != lock.get("artifacts", {}).get("onnx/model.onnx"):
        raise TryoutError("LAYA_SOURCE_MODEL_FINGERPRINT_MISMATCH")
    for key in ("source_model_sha256", "fp32_xml_sha256", "fp32_bin_sha256",
                "fp16_xml_sha256", "fp16_bin_sha256"):
        if manifest.get("converted_ir_sha256", {}).get(key) != conversion.get(key):
            raise TryoutError("LAYA_SOURCE_IR_FINGERPRINT_MISMATCH")

    xml = (root / conversion["fp16_xml"]).resolve()
    binary = xml.with_suffix(".bin")
    root_path = root.resolve()
    if root_path not in xml.parents or not xml.is_file() or not binary.is_file():
        raise TryoutError("LAYA_SOURCE_IR_MISSING")
    if (sha256_file(xml) != conversion.get("fp16_xml_sha256")
            or sha256_file(binary) != conversion.get("fp16_bin_sha256")):
        raise TryoutError("LAYA_SOURCE_IR_HASH_MISMATCH")
    ir_contract = _verify_saved_ir(root, xml, 256)
    if (ir_contract != conversion.get("saved_ir_input_contracts", {}).get("fp16")
            or ir_contract != conversion.get("input_contract")):
        raise TryoutError("LAYA_SOURCE_IR_INPUT_CONTRACT_MISMATCH")
    verify_assets(root)
    return source_run, conversion, xml, npu_result


def _laya_input(tokenizer: _PinnedLayaTokenizer, config: dict[str, Any],
                question_id: str, raw_question: dict[str, Any], state: str) -> dict[str, Any]:
    import numpy as np

    question = _laya_internal_question(raw_question)
    ids, markers, options = _laya_head(tokenizer, question)
    state_ids = tokenizer.encode(state.replace(tokenizer.mask_token, " "))
    required_tokens = len(ids) + len(state_ids) + 1
    if required_tokens > 256:
        raise TryoutError("INPUT_EXCEEDS_LAYA_SHAPE", recoverable=True)
    if (options["tokens_per_option"] is not None
            or options["options_distinct"] != options["options"]):
        raise TryoutError("LAYA_CANDIDATE_LAYOUT_INVALID")

    candidate_ids = _candidate_ids(raw_question)
    if not 2 <= len(candidate_ids) <= 8 or len(markers) != len(candidate_ids):
        raise TryoutError("LAYA_CANDIDATE_LAYOUT_INVALID")
    question_type = question["t"]
    qtype = {"choice": 0, "score": 1, "noul": 2}.get(question_type)
    if qtype is None:
        raise TryoutError("LAYA_QUESTION_TYPE_INVALID")
    size = len(candidate_ids)
    raw_temperatures = config.get("temperature", [1.0, 1.0, 1.0])
    if not isinstance(raw_temperatures, (list, tuple)) or len(raw_temperatures) != 3:
        raise TryoutError("LAYA_TEMPERATURE_CONFIG_INVALID")
    temperatures = [_clamp_laya_temperature(value) for value in raw_temperatures]
    temperature_by_options = {key: _clamp_laya_temperature(value)
                              for key, value in config.get("temperature_by_options", {}).items()}
    bucket = _laya_temperature_bucket(question_type, size)
    temperature = float(temperature_by_options.get(bucket, temperatures[qtype]))

    special_ids = {"cls": int(tokenizer.cls_token_id), "sep": int(tokenizer.sep_token_id),
                   "mask": int(tokenizer.mask_token_id), "pad": int(tokenizer.pad_token_id)}
    input_ids = np.full((1, 256), special_ids["pad"], dtype=np.int64)
    attention_mask = np.zeros((1, 256), dtype=np.int64)
    marker_pos = np.zeros((1, 8), dtype=np.int64)
    marker_mask = np.zeros((1, 8), dtype=np.bool_)
    sequence = ids + state_ids + [special_ids["sep"]]
    input_ids[0, :len(sequence)] = np.asarray(sequence, dtype=np.int64)
    attention_mask[0, :len(sequence)] = 1
    marker_pos[0, :len(markers)] = np.asarray(markers, dtype=np.int64)
    marker_mask[0, :len(markers)] = True
    inputs = {"input_ids": input_ids.tolist(), "attention_mask": attention_mask.tolist(),
              "marker_pos": marker_pos.tolist(), "marker_mask": marker_mask.tolist(),
              "qtype": [qtype]}
    input_sha256 = _input_digest(inputs)
    row = {"case_id": "interactive", "question_id": question_id,
           "question_type": question_type,
           "candidate_ids": candidate_ids, "candidate_count": size,
           "temperature": temperature, "input_sequence_length": 256,
           "required_total_tokens": required_tokens, "special_token_ids": special_ids,
           "state_truncated": False, "state_tokens_dropped": 0,
           "input_sha256": input_sha256, "inputs": inputs,
           "state_token_ids": state_ids, "active_sequence_ids": sequence,
           "question_contract": raw_question,
           "candidate_descriptions": _render_laya_options(question),
           "question_text": question["ins"]}
    row["arrays"] = _input_arrays(row)
    return row


def _verify_laya_input_canaries(source_run: Any, canaries: dict[str, dict[str, Any]],
                                tokenizer: _PinnedLayaTokenizer, config: dict[str, Any]) -> None:
    wanted = {(item["case_id"], item["question_id"]): item for item in canaries.values()}
    saved: dict[tuple[str, str], dict[str, Any]] = {}
    input_path = source_run.path / "inputs.jsonl"
    try:
        with input_path.open("r", encoding="utf-8") as stream:
            for line in stream:
                row = json.loads(line)
                key = (row.get("case_id"), row.get("question_id"))
                if key in wanted:
                    saved[key] = row
                    if len(saved) == len(wanted):
                        break
    except Exception:
        raise TryoutError("LAYA_SOURCE_INPUTS_UNREADABLE") from None
    if saved.keys() != wanted.keys():
        raise TryoutError("LAYA_SOURCE_INPUT_CANARY_MISSING")
    for key, canary in wanted.items():
        current = _laya_input(tokenizer, config, canary["question_id"],
                              canary["question"], canary["state"])
        previous = saved[key]
        if (current["input_sha256"] != previous.get("input_sha256")
                or current["inputs"] != previous.get("inputs")
                or current["candidate_ids"] != previous.get("candidate_ids")
                or current["question_type"] != previous.get("question_type")
                or current["temperature"] != previous.get("temperature")):
            raise TryoutError("LAYA_SOURCE_INPUT_FINGERPRINT_MISMATCH")


def _render_laya_criterion(value: Any) -> str:
    return value if isinstance(value, str) else json.dumps(
        value, ensure_ascii=False, separators=(", ", ": "), default=str
    )


def _laya_internal_question(raw_question: dict[str, Any]) -> dict[str, Any]:
    question_type = "noul" if raw_question.get("type") == "bool" else raw_question.get("type")
    instructions = raw_question.get("instructions")
    if not isinstance(instructions, str):
        instructions = json.dumps(instructions, ensure_ascii=False)
    internal = {"t": question_type, "ins": instructions, "crit": raw_question.get("criteria")}
    if "labels" in raw_question:
        internal["labels"] = raw_question["labels"]
    if "option_order" in raw_question:
        internal["option_order"] = [int(index) for index in raw_question["option_order"]]
    return internal


def _render_laya_options(question: dict[str, Any]) -> list[str]:
    question_type = question["t"]
    criteria = question.get("crit")
    if question_type == "choice":
        return [str(key) if value is None or value == "" else
                f"{key}: {_render_laya_criterion(value)}" for key, value in criteria.items()]
    if question_type == "score":
        return [f"level {index}: {_render_laya_criterion(value)}"
                for index, value in enumerate(criteria)]
    criteria = criteria or {}
    labels = question.get("labels", {"false": "false", "true": "true"})
    false_criterion = criteria.get("false")
    true_criterion = criteria.get("true")
    return [
        labels["false"] + ": " + (_render_laya_criterion(false_criterion)
                                   if false_criterion not in (None, "")
                                   else "no, the statement does not hold"),
        labels["true"] + ": " + (_render_laya_criterion(true_criterion)
                                  if true_criterion not in (None, "")
                                  else "yes, the statement holds"),
    ]


def _laya_head(tokenizer: _PinnedLayaTokenizer, question: dict[str, Any],
               head_max_len: int = 192) -> tuple[list[int], list[int], dict[str, Any]]:
    options = _render_laya_options(question)
    instruction = str(question["ins"]).replace(tokenizer.mask_token, " ")
    head_tokens = tokenizer.encode(f"{question['t']} question: {instruction}")
    rendered_option_tokens = [
        tokenizer.encode(" " + option.replace(tokenizer.mask_token, " ")) for option in options
    ]
    if any(len(tokens) > 48 for tokens in rendered_option_tokens):
        raise TryoutError("OPTION_TRUNCATION_REQUIRED")
    option_tokens = [[tokenizer.mask_token_id] + tokens for tokens in rendered_option_tokens]
    option_budget = head_max_len - sum(map(len, option_tokens))
    tokens_per_option = None
    if option_budget < 16:
        tokens_per_option = max(4, (head_max_len - 16) // max(1, len(option_tokens)))
        if any(len(tokens) > tokens_per_option for tokens in option_tokens):
            raise TryoutError("OPTION_TRUNCATION_REQUIRED")
        option_tokens = [tokens[:tokens_per_option] for tokens in option_tokens]
        option_budget = head_max_len - sum(map(len, option_tokens))
    if len(head_tokens) > max(8, option_budget):
        raise TryoutError("INSTRUCTION_TRUNCATION_REQUIRED")
    head_tokens = head_tokens[:max(8, option_budget)]
    ids = [tokenizer.cls_token_id, *head_tokens, tokenizer.sep_token_id]
    markers: list[int] = []
    for option in option_tokens:
        markers.append(len(ids))
        ids.extend(option)
    ids.append(tokenizer.sep_token_id)
    stats = {"options": len(option_tokens),
             "options_distinct": len({tuple(tokens) for tokens in option_tokens}),
             "tokens_per_option": tokens_per_option}
    return ids, markers, stats


def _clamp_laya_temperature(value: Any, lo: float = 0.5, hi: float = 5.0) -> float:
    if isinstance(value, bool):
        return 1.0
    try:
        temperature = float(value)
    except (TypeError, ValueError):
        return 1.0
    if not math.isfinite(temperature):
        return 1.0
    return min(hi, max(lo, temperature))


def _laya_temperature_bucket(question_type: str, candidate_count: int) -> str:
    size = "2" if candidate_count <= 2 else "3-5" if candidate_count <= 5 else "6-10" if candidate_count <= 10 else "11+"
    return f"{question_type}:{size}"


class _LayaSession:
    def __init__(self, root: Path, source_run_id: str,
                 templates: dict[str, dict[str, Any]], canaries: dict[str, dict[str, Any]], *,
                 execution_device: str = "NPU", diagnostic_run_id: str | None = None) -> None:
        if execution_device not in {"CPU", "NPU"}:
            raise TryoutError("LAYA_DEVICE_INVALID")
        source_run, conversion, _, _ = _validate_laya_source(root, source_run_id)
        lock = load_lock(root)
        source = lock["sources"]["laya_code"]
        source_dir = root / ".local" / "sources" / f"laya-pinned-{source['commit']}"
        if not (source_dir / "laya" / "agent.py").is_file():
            raise TryoutError("PINNED_LAYA_SOURCE_MISSING")
        _verify_extracted_laya_source(root, source_dir, source["commit"])
        os.environ["HF_HUB_OFFLINE"] = "1"
        os.environ["TRANSFORMERS_OFFLINE"] = "1"
        os.environ["HF_HUB_DISABLE_TELEMETRY"] = "1"
        os.environ["TOKENIZERS_PARALLELISM"] = "false"
        try:
            import openvino as ov
        except Exception:
            raise TryoutError("LAYA_RUNTIME_IMPORT_FAILED") from None

        # The historical Run contributes only its fixed IR and saved input canaries.
        # Its reference-model copy may have been normalized by the old loader, so load
        # the tokenizer and settings from the hash-locked source artifacts instead.
        checkpoint = root / ".local" / "models" / "reference"
        config_path = checkpoint / "rl_agent_config.json"
        tokenizer_path = checkpoint / "tokenizer"
        expected = lock["artifacts"]
        tokenizer_config_path = tokenizer_path / "tokenizer_config.json"
        if (not config_path.is_file() or not tokenizer_path.is_dir()
                or sha256_file(config_path) != expected["reference/rl_agent_config.json"]
                or sha256_file(tokenizer_path / "tokenizer.json") != expected["reference/tokenizer/tokenizer.json"]
                or not tokenizer_config_path.is_file()
                or sha256_file(tokenizer_config_path) != expected["reference/tokenizer/tokenizer_config.json"]):
            raise TryoutError("LAYA_TOKENIZER_FINGERPRINT_MISMATCH")
        config = json.loads(config_path.read_text(encoding="utf-8"))
        tokenizer_config = json.loads(tokenizer_config_path.read_text(encoding="utf-8"))
        tokenizer = _PinnedLayaTokenizer(tokenizer_path / "tokenizer.json", tokenizer_config)
        _verify_laya_input_canaries(source_run, canaries, tokenizer, config)

        xml_key = "fp16_xml" if execution_device == "NPU" else "fp32_xml"
        hash_key = f"{xml_key}_sha256"
        bin_hash_key = f"{xml_key.removesuffix('_xml')}_bin_sha256"
        xml = (root / conversion[xml_key]).resolve()
        if (not xml.is_file() or not xml.with_suffix(".bin").is_file()
                or sha256_file(xml) != conversion[hash_key]
                or sha256_file(xml.with_suffix(".bin")) != conversion[bin_hash_key]):
            raise TryoutError("LAYA_SOURCE_IR_HASH_MISMATCH")
        device: dict[str, Any] = {"status": "PASS", "device_properties": {}}
        if execution_device == "NPU":
            device = inspect_device("NPU", "3720")
            if device.get("status") != "PASS":
                raise TryoutError("LAYA_NPU_3720_UNAVAILABLE")
        try:
            core = ov.Core()
            model = core.read_model(str(xml))
            cache_dir = None
            if execution_device == "NPU":
                cache_dir = (root / ".local" / "tryout-cache" / "laya-s256-k8-plugin"
                             if diagnostic_run_id is None else
                             root / ".local" / "tryout-cache" / f"diagnostic-{diagnostic_run_id}-laya-plugin").resolve()
                cache_dir.mkdir(parents=True, exist_ok=True)
                compiled = core.compile_model(
                    model, "NPU", {"CACHE_DIR": str(cache_dir), "NPU_COMPILER_TYPE": "PLUGIN"}
                )
                properties = {
                    key: compiled.get_property(key)
                    for key in ("EXECUTION_DEVICES", "NPU_COMPILER_TYPE", "NPU_COMPILER_VERSION",
                                "RUNTIME_REQUIREMENTS", "LOADED_FROM_CACHE")
                }
            else:
                compiled = core.compile_model(model, "CPU")
                properties = {"EXECUTION_DEVICES": compiled.get_property("EXECUTION_DEVICES")}
        except Exception:
            raise TryoutError("LAYA_NPU_COMPILE_FAILED" if execution_device == "NPU"
                              else "LAYA_CPU_COMPILE_FAILED") from None
        execution_devices = _execution_device_names(properties["EXECUTION_DEVICES"])
        if execution_devices != [execution_device]:
            raise TryoutError("LAYA_EXECUTION_DEVICE_MISMATCH")
        compiler_type = "未適用"
        actual_arch = "対象外"
        if execution_device == "NPU":
            compiler_type = str(properties["NPU_COMPILER_TYPE"])
            if not compiler_type.upper().endswith("PLUGIN"):
                raise TryoutError("LAYA_COMPILER_MISMATCH")
            actual_arch = device.get("device_properties", {}).get("DEVICE_ARCHITECTURE")
            if str(actual_arch) != "3720":
                raise TryoutError("LAYA_ARCHITECTURE_MISMATCH")

        self._compiled = compiled
        self._core = core
        self._tokenizer = tokenizer
        self._config = config
        self._templates = templates
        self._canaries = canaries
        self.execution_device = execution_device
        self._cache_dir = cache_dir
        self._properties = properties
        self._execution_devices = execution_devices
        self._actual_architecture = actual_arch
        self._device_properties = device.get("device_properties", {})
        model_source = lock["sources"]["reference_checkpoint"]
        precision = "FP16" if execution_device == "NPU" else "FP32"
        compiler_version = (str(properties.get("NPU_COMPILER_VERSION", "未取得"))
                            if execution_device == "NPU" else "未適用")
        startup_lines = [
            "【試用するモデル】",
            f"Laya｜{execution_device}で実行｜{precision}｜PID：{os.getpid()}",
            "【実行情報】",
            f"モデル：{model_source['repository']}@{model_source['revision']}",
            f"バックエンド：OpenVINO {ov.__version__}",
            f"要求デバイス：{execution_device}｜実デバイス：{', '.join(execution_devices)}",
            f"NPUアーキテクチャ：要求3720／実値{actual_arch}｜コンパイラー：{compiler_type}｜版：{compiler_version}",
            f"入力プロファイル：s256-k8｜質問定義：dev-v3",
            f"固定IR取得元：{source_run_id}｜XML SHA-256：{conversion[hash_key]}｜BIN SHA-256：{conversion[bin_hash_key]}",
            f"質問データ SHA-256：{sha256_file(root / 'datasets' / 'dev-v3.jsonl')}",
        ]
        if cache_dir is not None:
            startup_lines.append(f"診断キャッシュ：{cache_dir}")
        self.startup = "\n".join(startup_lines)

    def predict(self, state: str) -> dict[str, dict[str, Any]]:
        return self.predict_with_trace(state)[0]

    def display_context(self) -> dict[str, Any]:
        return {kind: {"candidate_ids": _candidate_ids(self._templates[kind]),
                       "candidate_descriptions": _candidate_descriptions(self._templates[kind])}
                for kind in _TYPES}

    def predict_with_trace(self, state: str) -> tuple[dict[str, dict[str, Any]], dict[str, Any]]:
        outputs: dict[str, dict[str, Any]] = {}
        traces: dict[str, Any] = {}
        for kind in _TYPES:
            question_id = self._canaries[kind]["question_id"]
            question = self._templates[kind]
            row = _laya_input(self._tokenizer, self._config,
                              question_id, question, state)
            result = self._compiled({name: row["arrays"][name] for name in
                                     ("input_ids", "attention_mask", "marker_pos", "marker_mask", "qtype")})
            named = {port.get_any_name(): value for port, value in result.items()}
            logits, actions = _select_outputs(named)
            outputs[kind] = decode_laya_output(row["question_type"], row["candidate_ids"],
                                                row["temperature"], logits, actions)
            traces[kind] = {
                "question_id": question_id, "question_type": row["question_type"],
                "question_text": row["question_text"],
                "question_contract": row["question_contract"],
                "candidate_ids": row["candidate_ids"],
                "candidate_descriptions": row["candidate_descriptions"],
                "state_token_ids": row["state_token_ids"],
                "active_sequence_ids": row["active_sequence_ids"],
                "input_tensors": row["inputs"],
                "special_token_ids": row["special_token_ids"],
                "language_parameter": None,
                "temperature_bucket": _laya_temperature_bucket(
                    row["question_type"], len(row["candidate_ids"]),
                ),
                "temperature": row["temperature"],
                "required_total_tokens": row["required_total_tokens"],
                "state_truncated": row["state_truncated"],
                "state_tokens_dropped": row["state_tokens_dropped"],
                "input_sha256": row["input_sha256"],
                "logits": [float(value) for value in logits[:row["candidate_count"]]],
                "action_logits": [float(value) for value in actions],
            }
        return outputs, traces

    def close(self) -> None:
        self._compiled = None
        self._core = None
        self._tokenizer = None
        self._config = None


class _SokudanSession:
    def __init__(self, root: Path, templates: dict[str, dict[str, Any]],
                 canaries: dict[str, dict[str, Any]]) -> None:
        _, sokudan, lock, verification = _run_environment(root)
        runtime_path = (root / verification["runtime_model"]["directory"]).resolve()
        try:
            agent = sokudan.load(runtime_path, backend="torch", device="cpu", dtype="float32")
        except Exception:
            raise TryoutError("SOKUDAN_MODEL_LOAD_FAILED") from None
        actual = (getattr(agent.backend, "name", None), str(agent.device), str(agent.dtype))
        parameter_dtypes = sorted({str(parameter.dtype) for parameter in agent.model.parameters()
                                   if parameter.is_floating_point()})
        if actual != ("torch", "cpu", "float32") or parameter_dtypes != ["torch.float32"]:
            raise TryoutError("SOKUDAN_EXECUTION_MISMATCH")
        calibration_path = runtime_path / "calibration.json"
        calibration = json.loads(calibration_path.read_text(encoding="utf-8"))
        self._expected_temperature = float(calibration["temperatures"]["bool/2"])
        self._calibration_path = calibration_path
        self._agent = agent
        self._templates = templates
        self._canaries = canaries
        self._lock = lock
        self._verification = verification
        self.startup = "\n".join((
            "【試用するモデル】",
            f"Sokudan｜CPUで実行・NPU未対応｜FP32｜PID：{os.getpid()}",
            "【実行情報】",
            f"モデル：{lock['model']['repository']}@{lock['model']['revision']}",
            f"パッケージ：sokudan {lock['package']['version']}｜ソース：{lock['package']['source_commit']}",
            f"バックエンド：torch｜要求デバイス：CPU｜実デバイス：{self._agent.device}｜精度：{self._agent.dtype}",
            "入力プロファイル：モデル既定（固定Shapeなし）｜質問定義：dev-v3",
            f"校正 SHA-256：{sha256_file(calibration_path)}｜質問データ SHA-256：{sha256_file(root / 'datasets' / 'dev-v3.jsonl')}",
        ))

    def predict(self, state: str) -> dict[str, dict[str, Any]]:
        return self.predict_with_trace(state)[0]

    def display_context(self) -> dict[str, Any]:
        return {kind: {"candidate_ids": _candidate_ids(self._templates[kind]),
                       "candidate_descriptions": _candidate_descriptions(self._templates[kind])}
                for kind in _TYPES}

    def predict_with_trace(self, state: str) -> tuple[dict[str, dict[str, Any]], dict[str, Any]]:
        questions = _questions_for_templates(self._templates, self._canaries)
        kinds = {self._canaries[kind]["question_id"]: kind for kind in _TYPES}
        counts = {qid: len(_candidate_ids(question)) for qid, question in questions.items()}
        prepared_inputs: dict[str, Any] = {}
        for qid, question in questions.items():
            prepared = _prepare_question_inputs(self._agent, state, {qid: question}, include_tokens=True)[qid]
            if prepared["truncated"] or prepared["required_total_tokens"] > 1024:
                raise TryoutError("INPUT_EXCEEDS_SOKUDAN_LIMIT", recoverable=True)
            prepared_inputs[qid] = prepared
        before = _canonical(questions)
        try:
            official = self._agent.predict(state, questions, order_marginalize=False)
        except Exception:
            raise TryoutError("SOKUDAN_INFERENCE_FAILED") from None
        if (_canonical(questions) != before
                or official.get("usage", {}).get("state_truncated") is not False
                or official.get("encoding") != self._agent.backend.encoding
                or set(official.get("answers", {})) != set(questions)):
            raise TryoutError("SOKUDAN_OUTPUT_CONTRACT_FAILED")
        _validate_calibration(self._agent, official, list(questions), kinds, counts,
                              self._expected_temperature)
        adapted = {kind: adapt_official_answer(kind, official["answers"][qid],
                                               _candidate_ids(questions[qid]))
                   for qid, kind in kinds.items()}
        trace = {"questions": questions, "prepared_inputs": prepared_inputs,
                 "official_output": official, "calibration_temperature": self._expected_temperature,
                 "calibration_sha256": sha256_file(self._calibration_path)}
        return adapted, trace

    def close(self) -> None:
        self._agent = None


def _probability_rows(result: dict[str, Any]) -> list[tuple[Any, float]]:
    if "candidate_probabilities" in result:
        return [(row["candidate_id"], float(row["probability"]))
                for row in result["candidate_probabilities"]]
    values = result.get("api_display_probabilities")
    ids = result.get("candidate_ids", [])
    if not isinstance(values, list) or len(values) != len(ids):
        return []
    return list(zip(ids, (float(value) for value in values)))


def _questions_for_templates(templates: dict[str, dict[str, Any]],
                             canaries: dict[str, dict[str, Any]]) -> dict[str, dict[str, Any]]:
    if set(templates) != set(_TYPES) or set(canaries) != set(_TYPES):
        raise TryoutError("TRYOUT_TEMPLATE_TYPE_MISSING")
    questions = {canaries[kind]["question_id"]: templates[kind] for kind in _TYPES}
    if len(questions) != len(_TYPES):
        raise TryoutError("TRYOUT_TEMPLATE_ID_DUPLICATE")
    return questions


def _candidate_descriptions(question: dict[str, Any]) -> dict[Any, str]:
    kind = _question_kind(question)
    criteria = question.get("criteria") or {}
    if kind == "choice":
        return {key: _render_laya_criterion(value) for key, value in criteria.items()}
    if kind == "score":
        return {index: _render_laya_criterion(value) for index, value in enumerate(criteria)}
    labels = question.get("labels", {"false": "いいえ", "true": "はい"})
    return {False: str(labels.get("false", "いいえ")), True: str(labels.get("true", "はい"))}


def _candidate_rows(result: dict[str, Any]) -> tuple[list[tuple[Any, float]], str]:
    raw = result.get("candidate_probabilities")
    if isinstance(raw, list):
        try:
            rows = [(row["candidate_id"], float(row["probability"])) for row in raw]
        except (KeyError, TypeError, ValueError):
            raise TryoutError("DISPLAY_PROBABILITY_CONTRACT_INVALID") from None
        candidate_ids = result.get("candidate_ids")
        if candidate_ids is not None and [candidate_id for candidate_id, _ in rows] != candidate_ids:
            raise TryoutError("DISPLAY_CANDIDATE_ORDER_MISMATCH")
        return rows, "モデル出力値"
    values = result.get("api_display_probabilities")
    candidate_ids = result.get("candidate_ids", [])
    if values is None:
        return [], "未取得"
    if not isinstance(values, list) or not isinstance(candidate_ids, list) or len(values) != len(candidate_ids):
        raise TryoutError("DISPLAY_PROBABILITY_CONTRACT_INVALID")
    try:
        return list(zip(candidate_ids, (float(value) for value in values))), "公式API返却値（小数4桁）"
    except (TypeError, ValueError):
        raise TryoutError("DISPLAY_PROBABILITY_CONTRACT_INVALID") from None


def _candidate_id_text(value: Any) -> str:
    return json.dumps(value, ensure_ascii=False, separators=(",", ":"), allow_nan=False)


def _exact_number(value: Any) -> str:
    if type(value) not in (int, float) or not math.isfinite(float(value)):
        return "未取得"
    return repr(value)


def _display_candidate_label(kind: str, candidate_id: Any, context: dict[str, Any]) -> str:
    labels = context.get("candidate_descriptions", {})
    if candidate_id in labels:
        return str(labels[candidate_id])
    defaults = _DISPLAY_CHOICE if kind == "choice" else {}
    if kind == "score" and type(candidate_id) is int and 0 <= candidate_id < len(_DISPLAY_SCORE):
        return _DISPLAY_SCORE[candidate_id]
    if kind == "noul" and type(candidate_id) is bool:
        return "はい" if candidate_id else "いいえ"
    return defaults.get(candidate_id, "説明未取得")


def _format_candidate_detail(kind: str, result: dict[str, Any], context: dict[str, Any],
                             selected: Any) -> tuple[list[str], list[str]]:
    rows, origin = _candidate_rows(result)
    expected_order = context.get("candidate_ids")
    if rows and expected_order is not None and [candidate_id for candidate_id, _ in rows] != expected_order:
        raise TryoutError("DISPLAY_CANDIDATE_ORDER_MISMATCH")
    detail: list[str] = []
    numeric: list[str] = []
    if rows:
        detail.append(f"  候補確率（表示は小数2桁、取得元：{origin}）")
        for candidate_id, probability in rows:
            if not math.isfinite(probability) or not 0.0 <= probability <= 1.0:
                raise TryoutError("DISPLAY_PROBABILITY_CONTRACT_INVALID")
            label = _display_candidate_label(kind, candidate_id, context)
            mark = " ← 選択" if _candidate_id_text(candidate_id) == _candidate_id_text(selected) else ""
            detail.append(f"    {_candidate_id_text(candidate_id)}：{label}　{probability * 100:.2f}%{mark}")
        numeric.append(f"  候補確率取得値（{origin}）: " + ", ".join(
            f"{_candidate_id_text(candidate_id)}={_exact_number(result_value)}"
            for (candidate_id, _), result_value in zip(rows, _probability_source_values(result, rows))))
        top = max(probability for _, probability in rows)
        tied = [candidate_id for candidate_id, probability in rows if probability == top]
        if len(tied) > 1:
            detail.append("    同率候補あり（選択はAPI／モデルが返した候補を表示）")
    else:
        detail.append("  候補確率：未取得")
    return detail, numeric


def _probability_source_values(result: dict[str, Any], rows: list[tuple[Any, float]]) -> list[Any]:
    raw = result.get("candidate_probabilities")
    if isinstance(raw, list):
        return [row.get("probability") for row in raw]
    values = result.get("api_display_probabilities")
    return list(values) if isinstance(values, list) and len(values) == len(rows) else []


def format_results(results: dict[str, dict[str, Any]], context: dict[str, Any] | None = None) -> str:
    context = context or {}
    choice = results["choice"]
    score = results["score"]
    noul = results["noul"]
    choice_slot = choice.get("decision_slot")
    choice_ids = choice.get("candidate_ids")
    if choice.get("decision") is None and isinstance(choice_slot, int) and isinstance(choice_ids, list):
        choice_decision = choice_ids[choice_slot] if 0 <= choice_slot < len(choice_ids) else None
    else:
        choice_decision = choice.get("decision")
    if choice_decision not in _DISPLAY_CHOICE:
        raise TryoutError("CHOICE_OUTPUT_UNKNOWN")
    if isinstance(choice_ids, list) and choice_decision not in choice_ids:
        raise TryoutError("CHOICE_OUTPUT_CONTRACT_INVALID")
    choice_context = context.get("choice", {})
    choice_detail, choice_numeric = _format_candidate_detail(
        "choice", choice, choice_context, choice_decision)
    stage = (score.get("decision_slot") if score.get("question_type") == "score"
             else score.get("stage_prediction"))
    if type(stage) is not int or not 0 <= stage <= 2:
        raise TryoutError("SCORE_OUTPUT_UNKNOWN")
    boolean = noul.get("boolean_result", noul.get("evaluator_boolean"))
    if type(boolean) is not bool:
        raise TryoutError("NOUL_OUTPUT_UNKNOWN")
    score_context = context.get("score", {})
    score_id = score.get("decision", stage)
    score_detail, score_numeric = _format_candidate_detail("score", score, score_context, score_id)
    raw_expected = score.get("score_expected_unrounded")
    returned_score = score.get("returned_score", score.get("api_returned_value"))
    for value in (raw_expected, returned_score):
        if value is not None and (type(value) not in (int, float) or not math.isfinite(float(value))):
            raise TryoutError("SCORE_OUTPUT_CONTRACT_INVALID")
    score_precision = "OpenVINO/内部計算の返却値" if "returned_score" in score else "公式API返却値"
    noul_detail, noul_numeric = _format_candidate_detail(
        "noul", noul, context.get("noul", {}), True if boolean else False)
    raw_probability = noul.get("true_probability")
    returned_probability = noul.get("returned_noul", noul.get("api_returned_value"))
    for value in (raw_probability, returned_probability):
        if value is not None and (type(value) not in (int, float)
                                  or not math.isfinite(float(value)) or not 0.0 <= float(value) <= 1.0):
            raise TryoutError("NOUL_OUTPUT_CONTRACT_INVALID")
    boolean_matches_value = (returned_probability is None
                             or boolean is (float(returned_probability) >= 0.5))
    probability_is_api_rounded = "api_returned_value" in noul and "true_probability" not in noul
    if raw_probability is not None:
        notification_percent = f"P(true)の表示：{float(raw_probability) * 100:.2f}%（モデル出力値）"
        raw_true = _exact_number(raw_probability)
    elif returned_probability is not None:
        probability_source = ("公式API返却値・小数4桁丸め済み" if probability_is_api_rounded
                              else "モデル返却値")
        notification_percent = (f"P(true)の表示：{float(returned_probability) * 100:.2f}%"
                                f"（{probability_source}）")
        raw_true = "未取得"
    else:
        notification_percent = "P(true)の表示：未取得"
        raw_true = "未取得"
    bool_rule = "判定に使うP(true)が0.5以上なら「はい」、未満なら「いいえ」"
    if returned_probability is not None and float(returned_probability) == 0.5:
        bool_rule += "（今回値は境界の0.5）"

    lines = [
        "【モデルの判断】",
        f"作業の状態：{_DISPLAY_CHOICE[choice_decision]}",
        f"日時の具体性：{stage} — {_DISPLAY_SCORE[stage]}",
        f"通知候補：{'はい' if boolean else 'いいえ'}"
        + ("（返却値との真偽契約に不一致）" if not boolean_matches_value else ""),
        "",
        "【判断の内訳】",
        f"作業（選択ID：{_candidate_id_text(choice_decision)}）",
        *choice_detail,
        f"日時（選択段階：argmaxで選択した段階 {stage}）",
        *score_detail,
        (f"  段階の平均値（確率期待値、表示は小数4桁）：{float(raw_expected):.4f}／2"
         if raw_expected is not None else "  段階の平均値（確率期待値）：未取得"),
        "通知",
        *noul_detail,
        f"  {notification_percent}",
        (f"  判定に使う値：{'公式API返却値' if probability_is_api_rounded else 'モデル返却値'} "
         f"{_exact_number(returned_probability)}"
         if returned_probability is not None else "  判定に使う値：未取得"),
        f"  判定規則：{bool_rule}",
        *([] if boolean_matches_value else [
            "  契約エラー：返却値から計算する真偽とモデルの真偽出力が一致しません。両方の値をそのまま表示しています。"
        ]),
        "",
        "【数値照合用】",
        *choice_numeric,
        *score_numeric,
        f"  Score選択段階（argmax）：{stage}",
        f"  Score段階期待値（丸め前）：{_exact_number(raw_expected)}",
        f"  Score API返却値（{score_precision}）：{_exact_number(returned_score)}",
        f"  Noul P(true)（丸め前モデル値）：{raw_true}",
        f"  Noul {'公式API返却値' if probability_is_api_rounded else 'モデル返却値'}："
        f"{_exact_number(returned_probability)}",
        f"  Noul評価器判定：{'true' if boolean else 'false'}（返却値が0.5以上）",
    ]
    return "\n".join(lines)


def _tryout_error_text(exc: TryoutError) -> str:
    descriptions = {
        "INPUT_EXCEEDS_LAYA_SHAPE": "s256入力上限を超えたため拒否しました。文章は切り捨てていません。",
        "INPUT_EXCEEDS_SOKUDAN_LIMIT": "Sokudanの入力上限を超えたため拒否しました。文章は切り捨てていません。",
        "LAYA_CANDIDATE_LAYOUT_INVALID": "固定質問の候補位置または候補数が契約と一致しません。",
        "SOKUDAN_OUTPUT_CONTRACT_FAILED": "公式APIの返却値または入力契約が一致しません。",
        "DISPLAY_CANDIDATE_ORDER_MISMATCH": "画面用候補順とモデル出力の候補順が一致しません。",
        "DISPLAY_PROBABILITY_CONTRACT_INVALID": "確率が契約外のため、結果を表示できません。",
        "CHOICE_OUTPUT_CONTRACT_INVALID": "選択候補が返却された候補一覧にありません。",
        "SCORE_OUTPUT_CONTRACT_INVALID": "Scoreの取得値が数値契約に合いません。",
        "NOUL_OUTPUT_CONTRACT_INVALID": "通知候補の取得値が確率契約に合いません。",
    }
    detail = descriptions.get(exc.code, "固定実行契約に合わないため、結果を採用しません。")
    return f"推論できませんでした（{exc.code}）：{detail}"


def _interactive_loop(session: Any, input_fn: Callable[[str], str],
                      output_fn: Callable[[str], None]) -> None:
    while True:
        try:
            state = input_fn("文章> ")
        except KeyboardInterrupt:
            output_fn("Ctrl+Cを受け取りました。モデルを解放して終了します。")
            return
        except EOFError:
            output_fn("入力が閉じられました。モデルを解放して終了します。")
            return
        if state.strip() == ":quit":
            output_fn("終了します。")
            return
        if not state.strip():
            output_fn("空の文章は推論せず、入力待ちを続けます。")
            continue
        try:
            results = session.predict(state)
            display_context = session.display_context() if hasattr(session, "display_context") else None
            output_fn(f"入力：{state}")
            output_fn(format_results(results, display_context))
        except TryoutError as exc:
            output_fn(_tryout_error_text(exc))
            if not exc.recoverable:
                return
        except Exception:
            # Do not print exception text: upstream errors can include the submitted text.
            output_fn("推論に失敗しました（INFERENCE_FAILED）。入力待ちを終了します。")
            return
        output_fn("入力待ち（モデルを保持中。終了は :quit）")


def run_tryout(root: Path, model: str, source_run_id: str | None,
               input_fn: Callable[[str], str] | None = None,
               output_fn: Callable[[str], None] | None = None,
               session_factory: Callable[[], Any] | None = None) -> None:
    if model not in {"laya", "sokudan"}:
        raise TryoutError("MODEL_CHOICE_INVALID")
    if model == "laya" and not source_run_id:
        raise TryoutError("LAYA_SOURCE_RUN_REQUIRED")
    if model == "sokudan" and source_run_id:
        raise TryoutError("SOKUDAN_SOURCE_RUN_NOT_ALLOWED")
    input_fn = input if input_fn is None else input_fn
    output_fn = print if output_fn is None else output_fn

    with _ExclusiveModelLock():
        if session_factory is None:
            templates, canaries = _question_templates(root)
            if model == "laya":
                session_factory = lambda: _LayaSession(root, source_run_id, templates, canaries)
            else:
                session_factory = lambda: _SokudanSession(root, templates, canaries)
        try:
            session = session_factory()
        except TryoutError:
            raise
        except Exception:
            raise TryoutError("MODEL_LOAD_FAILED") from None
        try:
            output_fn(f"ロード完了：{session.startup}")
            output_fn("確率は正答率ではなく、各候補に対するモデル出力値です。")
            output_fn("表示はモデル出力の要約です。正解や事実認定ではありません。")
            output_fn("文章と結果は保存しません。理由はモデル出力に含まれないため生成しません。")
            output_fn("入力待ち（モデルを保持中。終了は :quit）")
            _interactive_loop(session, input_fn, output_fn)
        finally:
            try:
                session.close()
            except Exception:
                pass
