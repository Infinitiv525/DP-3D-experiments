"""Read-only input validation and recipe export, never submits any allocation."""
import argparse
from pathlib import Path
from main_protocol import (validate_manifest, shared_plan, search_plan, training_config, read, save, digest, identity)
from configurator_contract_v3 import SYSTEM_SHA256
from configurator_transport_v1 import POLICY


def run(a):
    manifest = read(a.manifest); validate_manifest(manifest)
    if manifest['execution_ready']: raise ValueError('This command is only candidate preparation, not execution')
    a.output.mkdir(parents=True, exist_ok=True)
    data = a.root / manifest['dataset_relative']; audit = read(data / 'pressure_gauge_audit.json')
    norm_path = a.root / manifest['normalization_source_relative']; norm = read(norm_path)
    inputs = {str(data / 'split.json'): digest(data / 'split.json'),
              str(data / 'pressure_gauge_audit.json'): digest(data / 'pressure_gauge_audit.json'),
              str(norm_path): digest(norm_path)}
    checks = {'corrected_cache': audit['status'] == 'FLOWBENCH_NATIVE_GAUGE64_CACHE_COMPLETE',
        'revision_and_split': audit['source_revision'] == manifest['revision'] and inputs[str(data / 'split.json')] == manifest['split_sha256'],
        'train_only_norm_identity': norm['identity']['split_sha256'] == manifest['split_sha256'] and norm['identity']['audit_sha256'] == digest(data / 'pressure_gauge_audit.json'),
        'execution_disabled': manifest['execution_ready'] is False and manifest['requires_separate_user_approval'] is True}
    for relative, status in (('results/flowbench/data_audit_123590/summary.json', 'FLOWBENCH_DATA_AUDIT_PASSED'),
        ('results/llm_search_prepare/transport_review_123872/summary.json', 'FOUR_FAMILY_TRANSPORT_CONFIRMATION_QC_PASSED')):
        path = a.root / relative; record = read(path); inputs[str(path)] = digest(path)
        checks[relative] = record['status'] == status and all(record['checks'].values())
    llm_path = a.root / manifest['llm_config_relative']
    models = read(llm_path); inputs[str(llm_path)] = digest(llm_path)
    checks['fixed_common_generation_settings'] = models['generation'] == dict(seed=20261021, max_new_tokens=512,
        max_time_seconds=90, max_input_tokens=8192, do_sample=False, max_repairs=1)
    for spec in models['models']:
        path = a.root / manifest['model_preparation_relative'] / (spec['label'] + '.json')
        proof = read(path); inputs[str(path)] = digest(path)
        checks[spec['label'] + '_pinned_model_manifest'] = proof['status'] == 'LOCAL_MODEL_VERIFIED' and proof['spec'] == spec
    shared = shared_plan(manifest); searches = search_plan(manifest)
    save(a.output / 'shared_plan.json', shared); save(a.output / 'search_plan.json', searches)
    configs = a.output / 'shared_configs'; configs.mkdir(exist_ok=True)
    for job in shared: save(configs / (job['id'] + '.json'), training_config(manifest, job))
    sources = ['llm/main_protocol.py', 'llm/test_main_protocol.py', 'llm/prepare_main_candidate.py',
        'training/train_flowbench_main.py', 'training/train_flowbench_protocol.py', 'training/train_flowbench_smoke.py',
        'training/fno_model.py', 'training/flowbench_physics.py', 'training/flowbench_protocol_metrics.py',
        'llm/configurator_contract_v3.py', 'llm/configurator_transport_v1.py', 'llm/propose_configuration_transport_v1.py',
        'llm/propose_configuration_v3.py', 'llm/persistent_search.py', 'llm/search_controls.py', 'llm/allocation_budget.py',
        'llm/settle_allocation_uid.py', 'scripts/116_prepare_main_candidate.sbatch']
    frozen = {'status': 'MAIN_CANDIDATE_SNAPSHOT_NOT_EXECUTION_FREEZE', 'manifest_file_sha256': digest(a.manifest),
              'source_files': {str(a.root / 'test' / name): digest(a.root / 'test' / name) for name in sources},
              'inputs': inputs, 'system_prompt_sha256': SYSTEM_SHA256, 'transport_policy': POLICY}
    save(a.output / 'candidate_snapshot.json', frozen); save(a.output / 'manifest.json', manifest)
    report = {'status': 'MAIN_CANDIDATE_PLANNING_CHECKS_PASSED' if all(checks.values()) else 'MAIN_CANDIDATE_PLANNING_CHECKS_NOT_PASSED',
        'checks': checks, 'manifest_sha256': identity(manifest), 'shared_training_count': len(shared),
        'independent_search_count': len(searches), 'adaptive_attempt_count_upper_bound': 18 * 9,
        'maximum_final_training_references': 95, 'final_unique_training_count': 'depends on validation-selected exact duplicates,5..95',
        'scope': manifest['scope'], 'main_execution_authorized': False, 'new_gpu_jobs': 0, 'test_evaluated': False,
        'not_yet_released': manifest['not_yet_released'],
        'warning': 'This is a planning/source snapshot, NOT MAIN_SOURCE_FREEZE_PASSED or user approval. Numerical main adapter not GPU-exercised in this step. Model manifests checked; model weights not rehashed here.'}
    save(a.output / 'summary.json', report)
    lines = ['PRÍPRAVA HLAVNÉHO EXPERIMENTU', report['status'],
        '9nových spoločných20epoch tréningov;18hľadaní,každé max12pokusov vrátane3shared.',
        'Najviac162adaptívnych návrhov a95finálnych tréningových použití, s5novými seedmi.',
        'Presné finálne duplikáty sa vykonajú iba raz, ale všetky priradenia zostanú uložené.',
        'Pilotné výsledky sa nedajú použiť ako hlavné výsledky. Test sa neotvára.',
        'Pripravené sú recepty, výber a deduplikácia, seedové pravidlá a approval-gated tréningový adaptér.',
        'Zatiaľ NIE je uvoľnené celé main riadenie ani test. Neexistuje používateľovo70GPU-h schválenie.',
        'Tento krok nespustil žiadnu GPU úlohu.']
    (a.output / 'vysledok.txt').write_text('\n'.join(lines) + '\n')
    print(report['status'], flush=True); return all(checks.values())


if __name__ == '__main__':
    p = argparse.ArgumentParser(); p.add_argument('--root', type=Path, required=True); p.add_argument('--manifest', type=Path, required=True)
    p.add_argument('--output', type=Path, required=True); raise SystemExit(0 if run(p.parse_args()) else 1)
