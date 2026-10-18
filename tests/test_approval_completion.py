from copy import deepcopy
import json
import unittest

from scripts.config.project_config import PROJECT_ROOT
from scripts.rag.approval_completion import APPROVAL_VERSION, complete_approval_proof
from scripts.rag.rag_workflow import build_workflow, validate_answer


def fixture():
    quotes = ["Lina maintains Lantern's rule catalogue.",
              "Decision D-90 requires Lantern for Project Vega's review process.",
              "Omar approved D-90 on October 9, 2026."]
    sources = [{"id": f"chunk{i}", "citation": f"S{i}", "text": text} for i, text in enumerate(quotes, 1)]
    def step(edge, relation, source, target, index):
        return {"edge_id": edge, "relation": relation, "source": source, "target": target,
            "source_label": source, "target_label": target,
            "evidence": [{"chunk_id": f"chunk{index}", "excerpt": quotes[index-1]}]}
    steps = [step("maintains", "MAINTAINS_RULES", "Lina", "Lantern", 1),
             step("about", "ABOUT_PROJECT", "D-90", "Project Vega", 2),
             step("requires", "REQUIRES_TOOL", "D-90", "Lantern", 2),
             step("approves", "APPROVED_BY", "D-90", "Omar", 3)]
    retrieval = {"graph_selection": {"mentions": [{"status": "resolved", "candidates": [
        {"id": "Project Vega", "type": "Project"}]}], "paths": [{"steps": steps}]}}
    answer = {"status": "insufficient_evidence", "answer": quotes[0] + " [S1]\n\nThe selected evidence does not establish every requested part of this question.", "citations": ["S1"]}
    result = {"answer": answer, "selected_excerpt_answer": deepcopy(answer)}
    question = "Who maintains the rule catalogue for the checker required by Project Vega's review decision, and who approved that decision?"
    return result, retrieval, sources, question


class ApprovalCompletionTests(unittest.TestCase):
    def test_unique_bound_proof_completes_and_preserves_original_selection(self):
        result, retrieval, sources, question = fixture()
        fixed = complete_approval_proof(result, retrieval, sources, question)
        self.assertEqual(fixed["answer"]["status"], "answered")
        self.assertIn("Omar approved D-90", fixed["answer"]["answer"])
        self.assertNotIn("does not establish", fixed["answer"]["answer"])
        self.assertEqual(fixed["selected_excerpt_answer"], result["answer"])
        self.assertEqual(result["answer"]["status"], "insufficient_evidence")
        self.assertEqual(len(fixed["approval_completion"]["added_excerpts"]), 2)
        validate_answer(json.dumps(fixed["answer"]), sources)
        repeated = complete_approval_proof(fixed, retrieval, sources, question)
        self.assertEqual(repeated["answer"], fixed["answer"])

    def test_no_cross_decision_approval_or_project_substitution(self):
        for key, value in [("source", "D-91"), ("relation", "PREPARED_BY")]:
            result, retrieval, sources, question = fixture()
            retrieval["graph_selection"]["paths"][0]["steps"][-1][key] = value
            fixed = complete_approval_proof(result, retrieval, sources, question)
            self.assertEqual(fixed["answer"], result["answer"])
        result, retrieval, sources, question = fixture()
        retrieval["graph_selection"]["mentions"][0]["candidates"][0]["id"] = "Project Willow"
        self.assertEqual(complete_approval_proof(result, retrieval, sources, question)["answer"], result["answer"])

    def test_missing_or_altered_packed_source_cannot_upgrade_status(self):
        for remove in [False, True]:
            result, retrieval, sources, question = fixture()
            if remove:
                sources.pop()
            else:
                sources[-1]["text"] = "This source no longer contains the reviewed approval."
            fixed = complete_approval_proof(result, retrieval, sources, question)
            self.assertEqual(fixed["approval_completion"]["status"], "missing_packed_evidence")
            self.assertEqual(fixed["answer"], result["answer"])

    def test_multiple_decisions_are_not_resolved_by_source_availability(self):
        result, retrieval, sources, question = fixture()
        steps = retrieval["graph_selection"]["paths"][0]["steps"]
        extra = deepcopy(steps[1:])
        for step in extra:
            step["edge_id"] += "-other"
            step["source"] = "D-91"
            step["evidence"] = [{"chunk_id": "not-packed", "excerpt": "An additional reviewed decision source."}]
        steps.extend(extra)
        fixed = complete_approval_proof(result, retrieval, sources, question)
        self.assertEqual(fixed["approval_completion"]["status"], "ambiguous")
        self.assertEqual(fixed["answer"], result["answer"])

    def test_names_alone_and_extra_question_parts_do_not_trigger_completion(self):
        result, retrieval, sources, question = fixture()
        changed = deepcopy(result)
        changed["answer"]["answer"] = "Lina and Lantern appear in a source [S1]."
        self.assertEqual(complete_approval_proof(changed, retrieval, sources, question)["answer"], changed["answer"])
        for query in [question + " What date was it approved?", question + " Why?",
                      question.replace("the checker", "the checker and what is its cost"), "Who approved Lantern?"]:
            self.assertEqual(complete_approval_proof(result, retrieval, sources, query)["approval_completion"]["status"], "not_applicable")

    def test_historical_q07_runs_through_shared_workflow_without_a_model_call(self):
        root = PROJECT_ROOT / "outputs/evaluation/runs/revised-20261018T155637344296Z"
        report = json.loads((root / "Q07-graph.json").read_text())["report"]
        class SavedSelection:
            prompt_version = APPROVAL_VERSION
            def generate(self, question, context, sources):
                return {"answer": report["selected_excerpt_answer"], "structured_claims": report["structured_claims"]}
        workflow = build_workflow(lambda question: {"evidence": report["sources"],
            "graph_selection": report["graph_selection"]}, SavedSelection())
        fixed = workflow.invoke({"question": report["question"]})
        self.assertEqual(fixed["answer"]["status"], "answered")
        self.assertIn("Elena Brooks approved DEC-004", fixed["answer"]["answer"])
        self.assertNotIn("supporting DEC-004 recommendation", fixed["answer"]["answer"])
        self.assertNotIn("approved DEC-001", fixed["answer"]["answer"])
        self.assertEqual(fixed["selected_excerpt_answer"], report["selected_excerpt_answer"])
        self.assertEqual(fixed["answer_checks"]["missing_explicit_roles"], [])
        self.assertEqual(fixed["approval_completion"]["decision_id"], "decision:dec-004")

    def test_only_trailing_application_warning_is_removed_not_identical_source_words(self):
        result, retrieval, sources, question = fixture()
        message = "The selected evidence does not establish every requested part of this question."
        quote = sources[0]["text"] + " " + message
        sources[0]["text"] = quote
        retrieval["graph_selection"]["paths"][0]["steps"][0]["evidence"][0]["excerpt"] = quote
        result["answer"]["answer"] = quote + " [S1]\n\n" + message
        result["selected_excerpt_answer"] = deepcopy(result["answer"])
        fixed = complete_approval_proof(result, retrieval, sources, question)
        self.assertTrue(fixed["answer"]["answer"].startswith(quote + " [S1]"))
        self.assertEqual(len(fixed["approval_completion"]["added_excerpts"]), 2)
