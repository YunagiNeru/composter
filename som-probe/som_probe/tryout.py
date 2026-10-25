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
    "日時情報なし、または時期未確定・曖昧",
    "日付・相対日付・曜日・期間だけ、または時計時刻だけ",
    "日付または相対日付と時計時刻の両方が明記",
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
           "input_sha256": input_sha256, "inputs": inputs}
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
                 templates: dict[str, dict[str, Any]], canaries: dict[str, dict[str, Any]]) -> None:
        source_run, _, xml, _ = _validate_laya_source(root, source_run_id)
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

        device = inspect_device("NPU", "3720")
        if device.get("status") != "PASS":
            raise TryoutError("LAYA_NPU_3720_UNAVAILABLE")
        try:
            core = ov.Core()
            cache_dir = (root / ".local" / "tryout-cache" / "laya-s256-k8-plugin").resolve()
            cache_dir.mkdir(parents=True, exist_ok=True)
            model = core.read_model(str(xml))
            compiled = core.compile_model(
                model, "NPU", {"CACHE_DIR": str(cache_dir), "NPU_COMPILER_TYPE": "PLUGIN"}
            )
            properties = {
                key: compiled.get_property(key)
                for key in ("EXECUTION_DEVICES", "NPU_COMPILER_TYPE", "NPU_COMPILER_VERSION",
                            "RUNTIME_REQUIREMENTS", "LOADED_FROM_CACHE")
            }
        except Exception:
            raise TryoutError("LAYA_NPU_COMPILE_FAILED") from None
        execution_devices = _execution_device_names(properties["EXECUTION_DEVICES"])
        if execution_devices != ["NPU"]:
            raise TryoutError("LAYA_EXECUTION_DEVICE_MISMATCH")
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
        model_source = lock["sources"]["reference_checkpoint"]
        self.startup = (f"Laya / FP16 / NPU / 3720 / {compiler_type} / "
                        f"model={model_source['repository']}@{model_source['revision']} / "
                        f"IR元Run={source_run_id} / s256-k8")

    def predict(self, state: str) -> dict[str, dict[str, Any]]:
        outputs: dict[str, dict[str, Any]] = {}
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
        return outputs

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
        calibration = json.loads((runtime_path / "calibration.json").read_text(encoding="utf-8"))
        self._expected_temperature = float(calibration["temperatures"]["bool/2"])
        self._agent = agent
        self._templates = templates
        self._canaries = canaries
        self.startup = (f"Sokudan {lock['package']['version']} / torch / CPU / FP32 / "
                        f"model={lock['model']['repository']}@{lock['model']['revision']}")

    def predict(self, state: str) -> dict[str, dict[str, Any]]:
        questions = _questions_for_templates(self._templates, self._canaries)
        kinds = {self._canaries[kind]["question_id"]: kind for kind in _TYPES}
        counts = {qid: len(_candidate_ids(question)) for qid, question in questions.items()}
        for qid, question in questions.items():
            prepared = _prepare_question_inputs(self._agent, state, {qid: question})[qid]
            if prepared["truncated"] or prepared["required_total_tokens"] > 1024:
                raise TryoutError("INPUT_EXCEEDS_SOKUDAN_LIMIT", recoverable=True)
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
        return {kind: adapt_official_answer(kind, official["answers"][qid],
                                            _candidate_ids(questions[qid]))
                for qid, kind in kinds.items()}

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


def format_results(results: dict[str, dict[str, Any]]) -> str:
    choice = results["choice"]
    score = results["score"]
    noul = results["noul"]
    choice_label = _DISPLAY_CHOICE.get(choice.get("decision"))
    if choice_label is None:
        raise TryoutError("CHOICE_OUTPUT_UNKNOWN")
    stage = (score.get("decision_slot") if score.get("question_type") == "score"
             else score.get("stage_prediction"))
    if type(stage) is not int or not 0 <= stage <= 2:
        raise TryoutError("SCORE_OUTPUT_UNKNOWN")
    boolean = noul.get("boolean_result", noul.get("evaluator_boolean"))
    if type(boolean) is not bool:
        raise TryoutError("NOUL_OUTPUT_UNKNOWN")

    lines = [
        f"作業の状態：{choice_label}",
        f"日時の具体性：{stage} — {_DISPLAY_SCORE[stage]}",
        f"通知候補：{'はい' if boolean else 'いいえ'}",
        "数値（参考）:",
    ]
    choice_names = {"task": "未完了", "none": "対象外", "need_info": "確認待ち"}
    choice_probs = _probability_rows(choice)
    if choice_probs:
        lines.append("  Choice確率: " + "、".join(
            f"{choice_names.get(key, key)}={value:.4f}" for key, value in choice_probs))
    score_probs = _probability_rows(score)
    if score_probs:
        lines.append("  Score段階確率: " + "、".join(
            f"{key}={value:.4f}" for key, value in score_probs))
    returned_score = score.get("returned_score", score.get("api_returned_value"))
    if returned_score is not None:
        lines.append(f"  返却Score={returned_score!r}")
    raw_expected = score.get("score_expected_unrounded")
    if raw_expected is not None:
        lines.append(f"  丸め前期待値={raw_expected!r}")
    p_true = noul.get("true_probability", noul.get("api_returned_value"))
    if p_true is not None:
        lines.append(f"  P(true)={float(p_true):.8f}")
    returned_noul = noul.get("returned_noul", noul.get("api_returned_value"))
    if returned_noul is not None:
        lines.append(f"  API返却P(true)={float(returned_noul):.4f}")
    lines.append("  画面の確率表示は小数4桁です。モデルの返却値は変更していません。")
    return "\n".join(lines)


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
            output_fn(format_results(results))
        except TryoutError as exc:
            output_fn(f"推論できませんでした（{exc.code}）。")
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
            output_fn(f"推論プロセス PID：{os.getpid()}")
            output_fn("dev-v3の固定質問定義を使用。文章と結果は保存しません。")
            output_fn("入力待ち（モデルを保持中。終了は :quit）")
            _interactive_loop(session, input_fn, output_fn)
        finally:
            try:
                session.close()
            except Exception:
                pass
