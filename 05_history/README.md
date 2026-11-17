# Vývoj a pilotné pokusy

V archívoch sú od hlavného porovnania oddelené aj dostupné staršie siete, rozbehy a technické pokusy. Súbory best, last a training_state sú rôzne stavy tej istej siete; ich počet nie je počtom nezávislých experimentov. Presné zaradenie určuje index modelov a manifest príloh.

## Podklady jednotlivých pokusov

Cesty sú skrátené pre čitateľnosť a kompatibilitu s Windows. Obsah súborov ostal byte za bytom rovnaký; pôvodné cesty a SHA sú v `LOCAL_SCIENTIFIC_RECORDS.json`.

| Pokus | Počet súborov |
|---|---:|
| [01_flowbench_3d_fno](pokusy/01_flowbench_3d_fno/README.md) | 17 |
| [02_vlastne_3d_cfd](pokusy/02_vlastne_3d_cfd/README.md) | 13 |
| [PRECITAJ_MA.txt](pokusy/PRECITAJ_MA.txt/README.md) | 1 |
| [accounting_closure_124666](pokusy/accounting_closure_124666/README.md) | 1 |
| [box_fidelity_review_122974](pokusy/box_fidelity_review_122974/README.md) | 2 |
| [comparative_report_20261022_v1](pokusy/comparative_report_20261022_v1/README.md) | 14 |
| [controller_integration_123651](pokusy/controller_integration_123651/README.md) | 14 |
| [controller_latency_review_123656](pokusy/controller_latency_review_123656/README.md) | 2 |
| [cylinder3d](pokusy/cylinder3d/README.md) | 19 |
| [data_audit_123590](pokusy/data_audit_123590/README.md) | 4 |
| [dataset_a](pokusy/dataset_a/README.md) | 170 |
| [environment](pokusy/environment/README.md) | 1 |
| [final_measurement_replay_20261022](pokusy/final_measurement_replay_20261022/README.md) | 7 |
| [final_measurement_start_20261022](pokusy/final_measurement_start_20261022/README.md) | 2 |
| [final_models_review_20261022](pokusy/final_models_review_20261022/README.md) | 3 |
| [final_window_check_124859](pokusy/final_window_check_124859/README.md) | 2 |
| [flowbench_2d_audit_20261030_v1](pokusy/flowbench_2d_audit_20261030_v1/README.md) | 2 |
| [flowbench_boundary_audit_121701](pokusy/flowbench_boundary_audit_121701/README.md) | 1 |
| [flowbench_evaluation_121349](pokusy/flowbench_evaluation_121349/README.md) | 2 |
| [flowbench_gauge_probe_122477](pokusy/flowbench_gauge_probe_122477/README.md) | 1 |
| [flowbench_header_probe.json](pokusy/flowbench_header_probe.json/README.md) | 1 |
| [flowbench_pair_review_122474](pokusy/flowbench_pair_review_122474/README.md) | 2 |
| [flowbench_refine_control_122413](pokusy/flowbench_refine_control_122413/README.md) | 2 |
| [flowbench_refine_divergence_122412](pokusy/flowbench_refine_divergence_122412/README.md) | 2 |
| [flowbench_review_121345](pokusy/flowbench_review_121345/README.md) | 4 |
| [flowbench_smoke_121348](pokusy/flowbench_smoke_121348/README.md) | 2 |
| [flowbench_uniform64_baseline_122336](pokusy/flowbench_uniform64_baseline_122336/README.md) | 3 |
| [flowbench_uniform64_cache](pokusy/flowbench_uniform64_cache/README.md) | 1 |
| [flowbench_uniform64_control_121721](pokusy/flowbench_uniform64_control_121721/README.md) | 3 |
| [four_family_confirmation_123872](pokusy/four_family_confirmation_123872/README.md) | 2 |
| [kopie_manifest.json](pokusy/kopie_manifest.json/README.md) | 1 |
| [latency_current_123650](pokusy/latency_current_123650/README.md) | 2 |
| [llm_feasibility_123301](pokusy/llm_feasibility_123301/README.md) | 36 |
| [llm_feedback_cycle_123438](pokusy/llm_feedback_cycle_123438/README.md) | 40 |
| [llm_feedback_repair_123449](pokusy/llm_feedback_repair_123449/README.md) | 9 |
| [main_candidate_124484](pokusy/main_candidate_124484/README.md) | 15 |
| [main_candidate_124485](pokusy/main_candidate_124485/README.md) | 15 |
| [main_release_124502](pokusy/main_release_124502/README.md) | 7 |
| [main_workflow_check_124489](pokusy/main_workflow_check_124489/README.md) | 2 |
| [ministral_transport_123871](pokusy/ministral_transport_123871/README.md) | 12 |
| [native_cylinder_full_20261025_v3](pokusy/native_cylinder_full_20261025_v3/README.md) | 4 |
| [native_cylinder_refinement_20261023](pokusy/native_cylinder_refinement_20261023/README.md) | 3 |
| [native_cylinder_spatial_20261024](pokusy/native_cylinder_spatial_20261024/README.md) | 2 |
| [native_cylinder_temporal_20261023](pokusy/native_cylinder_temporal_20261023/README.md) | 2 |
| [native_gauge_refine_control_122976](pokusy/native_gauge_refine_control_122976/README.md) | 2 |
| [native_gauge_refine_divergence_122975](pokusy/native_gauge_refine_divergence_122975/README.md) | 2 |
| [native_quality_20261022_v1](pokusy/native_quality_20261022_v1/README.md) | 1 |
| [native_quality_review_20261023](pokusy/native_quality_review_20261023/README.md) | 1 |
| [search_launch_124667_124684](pokusy/search_launch_124667_124684/README.md) | 5 |
| [search_preparation_123454_123457](pokusy/search_preparation_123454_123457/README.md) | 15 |
| [search_recovery_123598_123599](pokusy/search_recovery_123598_123599/README.md) | 4 |
| [search_review_20261022](pokusy/search_review_20261022/README.md) | 19 |
| [sequential_v3_123757_123763](pokusy/sequential_v3_123757_123763/README.md) | 3 |
| [smoke_101584](pokusy/smoke_101584/README.md) | 1 |
| [training](pokusy/training/README.md) | 20 |
| [transport_replay_123870](pokusy/transport_replay_123870/README.md) | 1 |
| [verification](pokusy/verification/README.md) | 32 |
