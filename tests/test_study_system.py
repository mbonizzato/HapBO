from pathlib import Path
import tempfile
import textwrap
import unittest

from src.experiments.catalog import StudyConfigurationError, load_catalog, load_study
from src.experiments.planner import build_study_plan


PROJECT_ROOT = Path(__file__).resolve().parents[1]


class StudySystemTests(unittest.TestCase):
    def test_catalog_and_every_study_manifest_validate(self):
        catalog = load_catalog()
        self.assertIn("neuromosaic-dataset1-rat-flexor", catalog.benchmarks)
        self.assertIn("hpobench-tabular", catalog.benchmarks)
        self.assertIn("solar-single-objective", catalog.benchmarks)
        self.assertIn("cnn-leak-of", catalog.benchmarks)
        self.assertEqual(
            catalog.methods["alpha-pi-raw-ei-matched"]["strategy"], "alpha-pi"
        )
        self.assertEqual(
            catalog.methods["alpha-pi-raw-ei-matched"]["acquisition"],
            "expected-improvement",
        )

        paths = sorted((PROJECT_ROOT / "experiments" / "studies").glob("**/study.yaml"))
        # Shipped manifests: 41 HPOBench, 12 cortical, 1 SOLAR (scripts/make_studies.py) and the
        # 5 CNN leak-control studies. Every shipped study must expand into a valid plan.
        self.assertEqual(len(paths), 59)
        for path in paths:
            build_study_plan(load_study(path), catalog)

    def test_cnn_leak_campaign_expands_to_one_method_per_process_and_unique_directory(self):
        catalog = load_catalog()
        # CNN leak-control campaign: 1 benchmark (cnn-leak-of) x 3 methods x 1 budget (30)
        # x 10 seeds = 30 runs, all executable (shipped prior).
        study = load_study(
            PROJECT_ROOT / "experiments" / "studies" / "cnn-leak" / "cnn-leak-of-campaign"
        )
        plan = build_study_plan(study, catalog)

        self.assertEqual(len(plan.runs), 30)
        self.assertEqual(len({run.run_id for run in plan.runs}), len(plan.runs))
        self.assertEqual(
            len({run.result_directory for run in plan.runs}), len(plan.runs)
        )
        executable = [run for run in plan.runs if run.executable]
        self.assertEqual(len(executable), 30)
        self.assertTrue(
            all(run.command[1] == "scripts/run_experiment.py" for run in executable)
        )
        self.assertTrue(
            all(run.command[-1] == str(run.result_directory) for run in executable)
        )
        self.assertEqual(
            {run.method_id for run in plan.runs},
            {"alpha-pi-raw-ei-matched", "vanilla-bo-ei", "random-search"},
        )

    def test_missing_priors_are_exposed_as_blocked(self):
        # The planner's missing-prior blocking on a synthetic manifest: a
        # prior-requiring method (alpha-pi-raw-ei-matched) is blocked both when the prior
        # is unassigned (no prior_path) and when it points at a nonexistent
        # file, while a prior-free method (vanilla-bo-ei) stays executable.
        catalog = load_catalog()
        manifest = textwrap.dedent(
            """
            artifact_format: llm-bo-study-v1
            study_id: synthetic-missing-prior-fixture
            phase: fixture
            status: ready
            execution_mode: optimization
            design:
              budgets: [8]
              seeds: [1, 2]
              methods:
                - alpha-pi-raw-ei-matched
                - vanilla-bo-ei
              optimizer: {n_initial: 3}
              benchmarks:
                - benchmark_id: hpobench-tabular
                  include: [lr-australian]
                - benchmark_id: cnn-leak-of
                  include: all
                  prior_path: experiments/priors/does-not-exist/prior/prior.json
            """
        ).strip() + "\n"

        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "study.yaml"
            path.write_text(manifest, encoding="utf-8", newline="\n")
            plan = build_study_plan(load_study(path), catalog)

        self.assertGreater(plan.blocked_count, 0)
        reasons = {reason for run in plan.runs for reason in run.blocked_reasons}
        self.assertIn("prior path is not assigned", reasons)
        self.assertTrue(
            any(reason.startswith("prior file does not exist") for reason in reasons),
            reasons,
        )
        # Only the prior-requiring method is blocked; the prior-free one runs.
        blocked_methods = {
            run.method_id for run in plan.runs if not run.executable
        }
        self.assertEqual(blocked_methods, {"alpha-pi-raw-ei-matched"})
        self.assertTrue(
            any(run.method_id == "vanilla-bo-ei" and run.executable for run in plan.runs)
        )

    def test_study_must_be_a_ready_optimization_study(self):
        header = "artifact_format: llm-bo-study-v1\nstudy_id: fixture\nphase: fixture\n"
        for fields in ("status: draft\nexecution_mode: optimization\n",
                       "status: ready\nexecution_mode: module\n"):
            with self.subTest(fields=fields), tempfile.TemporaryDirectory() as directory:
                path = Path(directory) / "study.yaml"
                path.write_text(header + fields, encoding="utf-8", newline="\n")
                with self.assertRaises(StudyConfigurationError):
                    load_study(path)


if __name__ == "__main__":
    unittest.main()
