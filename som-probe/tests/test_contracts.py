import unittest

from som_probe.contracts import ContractError, Profile, softmax, validate_dataset_case, validate_request


def request(question):
    return {
        "schema_version": 1,
        "request_id": "case-001",
        "state": "架空の依頼です。",
        "state_version": 0,
        "questions": {"intent": question},
    }


class ContractTests(unittest.TestCase):
    def test_accepts_choice(self):
        value = request({"type": "choice", "instructions": "どれですか", "criteria": {"a": "A", "b": "B"}})
        self.assertEqual(validate_request(value), value)

    def test_rejects_empty_choice(self):
        with self.assertRaisesRegex(ContractError, "候補"):
            validate_request(request({"type": "choice", "instructions": "どれですか", "criteria": {}}))

    def test_rejects_duplicate_rendered_candidates(self):
        with self.assertRaises(ContractError) as caught:
            validate_request(request({"type": "choice", "instructions": "選ぶ", "criteria": {"a": "同じ", "b": "同じ"}}))
        self.assertEqual(caught.exception.code, "DUPLICATE_CANDIDATE")

    def test_rejects_more_than_fixed_slots(self):
        choices = {f"c{i}": str(i) for i in range(9)}
        with self.assertRaises(ContractError) as caught:
            validate_request(request({"type": "choice", "instructions": "選ぶ", "criteria": choices}), Profile())
        self.assertEqual(caught.exception.code, "INVALID_CANDIDATES")

    def test_rejects_unknown_keys_and_nonfinite_json(self):
        value = request({"type": "bool", "instructions": "通知が必要ですか"})
        value["secret_field"] = "unexpected"
        with self.assertRaises(ContractError):
            validate_request(value)

    def test_softmax_is_normalized_and_rejects_nan(self):
        result = softmax([1.0, 2.0])
        self.assertAlmostEqual(sum(result), 1.0)
        with self.assertRaises(ContractError):
            softmax([float("nan"), 1.0])

    def test_dataset_label_must_name_a_real_answer_and_remain_unverified(self):
        case = {
            "case_id": "case-001",
            "state": "架空の状態",
            "questions": {"intent": {"type": "choice", "instructions": "分類する",
                                       "criteria": {"task": "作業", "none": "なし"}}},
            "expected": {"intent": {"allowed_answers": ["missing"], "rationale": "理由",
                                     "target_errors": ["誤分類"],
                                     "verification_status": "ai_draft_unverified"}},
        }
        with self.assertRaises(ContractError) as caught:
            validate_dataset_case(case)
        self.assertEqual(caught.exception.code, "LABEL_ANSWER_OUT_OF_RANGE")
        case["expected"]["intent"]["allowed_answers"] = ["task"]
        case["expected"]["intent"]["verification_status"] = "human_verified"
        with self.assertRaises(ContractError) as caught:
            validate_dataset_case(case)
        self.assertEqual(caught.exception.code, "INVALID_VERIFICATION_STATE")


if __name__ == "__main__":
    unittest.main()
