import unittest
import json
from pathlib import Path

from desktop_pet.behavior import QUESTIONS
from desktop_pet.npu_worker import _handle_infer
from desktop_pet.worker_client import NpuWorkerClient
from PySide6.QtCore import QProcess


class FakeProcess:
    def __init__(self, data):
        self.data = data

    def state(self):
        return QProcess.ProcessState.Running

    def readAllStandardOutput(self):
        value, self.data = self.data, b""
        return value


class FakeSession:
    def predict_choice(self, question_id, question, state):
        candidates = list(question["criteria"])
        return ({"decision": candidates[0], "candidate_ids": candidates,
                 "candidate_probabilities": [{"candidate_id": key, "probability": 1 / len(candidates)}
                                              for key in candidates]},
                {"input_sha256": "a" * 64, "required_total_tokens": 24,
                 "state_truncated": False, "candidate_ids": candidates})


class WorkerProtocolTests(unittest.TestCase):
    def test_inference_replies_with_actions_and_hash_not_structured_input_text(self):
        private_state = "cpu_usage_percent: 89.0\npersonality: AFFECTIONATE"
        result = _handle_infer(FakeSession(), {"request_id": 3, "state": private_state})
        self.assertEqual(result["type"], "inference")
        self.assertEqual(set(result["actions"]), set(QUESTIONS))
        self.assertNotIn(private_state, repr(result))
        self.assertEqual(result["state_sha256"], __import__("hashlib").sha256(
            private_state.encode("utf-8")).hexdigest())

    def test_invalid_request_is_rejected_without_running_inference(self):
        result = _handle_infer(FakeSession(), {"request_id": "bad", "state": 12})
        self.assertEqual(result["code"], "WORKER_REQUEST_INVALID")

    def test_client_reports_model_holding_worker_pid_not_qprocess_launcher(self):
        client = NpuWorkerClient(Path("."), Path("."), Path("python.exe"))
        client.process = FakeProcess((json.dumps({
            "protocol_version": 1, "type": "status", "status": "loading",
            "device": "NPU", "precision": "FP16", "pid": 43127,
        }) + "\n").encode("utf-8"))
        client._read_stdout()
        self.assertEqual(client.pid, 43127)


if __name__ == "__main__":
    unittest.main()
