import importlib.util
import os

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))

# Load the baseline runner module from disk
spec = importlib.util.spec_from_file_location(
    'baseline_runner',
    os.path.join(ROOT, 'eval', 'baseline', 'run_baseline.py')
)
rb = importlib.util.module_from_spec(spec)
spec.loader.exec_module(rb)

# Patch the graph policy at runtime to allow only one verification pass.
# After one failed verify, the graph stops instead of retrying generation.
import src.generation.graph as graph_mod

def one_pass_should_retry(state):
    if state.get('verification_passed'):
        return graph_mod.END
    if state.get('retries', 0) >= 1:
        return 'flag_unverified'
    return 'generate'

graph_mod.should_retry = one_pass_should_retry

retrieval_summary, per_question = rb.compute_retrieval_metrics()
results, usage_log = rb.run_end_to_end()
stress_summary = rb.run_stress_test()
rb.build_report(retrieval_summary, per_question, results, stress_summary)

print('--- RETRIEVAL SUMMARY ---')
print(retrieval_summary)
print('--- RESULTS COUNT ---')
print(len(results))
print('--- VERIFIED ---')
print(sum(1 for r in results if r.get('verification_passed')))
print('--- USAGE LOG ENTRIES ---')
print(len(usage_log))
print('--- STRESS SUMMARY ---')
print(stress_summary)
