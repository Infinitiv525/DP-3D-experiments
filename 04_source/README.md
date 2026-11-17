# Zdrojový kód

Pôvodné vedecké implementácie sú v `original/test/`. Zachované názvy uľahčujú spojenie s konfiguráciami a históriou. Tréningové a vyhodnocovacie postupy používajú pôvodné absolútne pracovné cesty; pri reprodukcii ich nastav v pracovnej kópii. Prevádzkové prihlásenia a nasadzovacie pomocníky sa nezverejňujú.

## Hlavné súbory

- [Definícia FNO](original/test/training/fno_model.py).
- [Hlavný tréning](original/test/training/train_flowbench_main.py).
- [Záverečné meranie](original/test/training/measure_final_models_v3.py).
- Vstupy vlastných CFD pilotov a tvorba geometrie sú v `original/test/cases`, `original/test/meshing` a `original/test/native`.
