# Staršie 3D experimenty prúdenia

## Úplný prehľad dostupných dát

**[Katalóg výsledkov, váh a numerických polí](01_results/DOSTUPNE_DATA.md)** uvádza všetky publikované balíky, priame odkazy a aj výnimky z archivácie. Veľké súbory sú v Releases a nevyžadujú stiahnutie celého repozitára.

V tejto štúdii lokálne LLM navrhovali konfiguráciu FNO pre ustálené 3D prúdenie FlowBench. Osobitne sú zachované vlastné dynamické CFD piloty; nie sú hotovým porovnaním LLM na dynamickom prúdení.

Finálne porovnanie má **70 fyzických sietí a 95 logických odkazov**, pretože niektoré konfigurácie zdieľajú model. Najlepšie skóre dosiahlo TPE; testované LLM ho v tomto konkrétnom experimente neprekonali. [Čitateľný výsledok](01_results/final/report.txt) a [presné čísla](01_results/final/comparison_numbers.json).

Výstup je ustálené priestorové pole, nejde o časový rollout. Kritérium 5 ms sa vzťahuje na deklarovaný rozsah merania na H200; rôzne merané režimy sa nezhŕňajú do jedného času. Vlastné CFD piloty sú v `05_history/cfd_pilots`. Rozpracovaný text článku sa nezverejňuje.

## Kde začať

| Priečinok | Čo v ňom nájdeš |
|---|---|
| [01_results](01_results) | Hotové porovnanie, grafy a číselné tabuľky |
| [02_models](02_models) | Index sietí a odkazy na skutočné váhy |
| [03_methodology](03_methodology) | Konfigurácie, metriky a podmienky experimentu |
| [04_source](04_source) | Vedecký zdrojový kód |
| [05_history](05_history) | Oddelená história pilotov a neúspešných pokusov |

## Modely a podrobné záznamy

[Stiahnuť modely a experimentálne záznamy](https://github.com/Infinitiv525/DP-3D-experiments/releases/tag/research-artifacts-v1). Archívy sú prílohy Release; bežný `git clone` ich nesťahuje. [Index modelov](02_models/model_index.csv) určuje príslušný diel archívu, pôvod a SHA256 každého súboru. [Manifest](02_models/ARTIFACT_MANIFEST.json) pokrýva aj zverejnené experimentálne záznamy.

Zachovaných je **284 checkpointových súborov**, vrátane vybraných váh, dostupných posledných stavov a obnovovacích stavov. Toto číslo nie je počtom nezávislých tréningov. Finálne siete a história sa v archíve aj indexe rozlišujú. Vývojové a neúspešné pokusy nenahrádzajú finálne výsledky.

Úplné veľké vstupné FlowBench/CFD polia a stiahnuté pôvodné LLM sa neduplikujú; ich zdroje a pripnuté verzie sú v metodike. Surové natívne predikčné polia možno znovu vypočítať zo zachovaných sietí. Kód obsahuje pôvodné pracovné cesty a vyžaduje nastavenie prostredia pre daný stroj. Archívy zachovávajú váhy byte za bytom.

## Súvisiace experimenty

- [3D experimenty](https://github.com/Infinitiv525/DP-3D-experiments)
- [Porovnanie LLM](https://github.com/Infinitiv525/DP-LLM-comparison)
- [Porovnanie architektúr](https://github.com/Infinitiv525/DP-architecture-comparison)

Verejný vedecký export oddeľuje tieto podklady od súkromnej rozpracovanej dizertácie a prevádzkovej histórie. Referenčné dáta pochádzajú z [FlowBench](https://huggingface.co/datasets/BGLab/FlowBench); metódy autorov opisuje [pôvodný projekt](https://github.com/baskargroup/flowbench-tools). Licencie a zdroje externých vstupov sú v [metodike](03_methodology/DATA_AND_REUSE.md).

## Doplnkové vedecké dáta

Ďalšie surové merania a podklady na reprodukciu sú v [doplnkovom balíku](01_results/doplnkove_data/README.md).

## Historické numerické výstupy

Pôvodné polia a diagnostika starších pokusov sú v [samostatnom balíku](01_results/historicke_numericke_vystupy/README.md).

## Ďalšie výskumné záznamy

Doplnené pôvodné diagnostiky, protokoly, obrázky a zdroje sú v [samostatnom balíku](01_results/dalsie_vyskumne_zaznamy/README.md).

## Pôvodné Slurm logy

Pôvodné výstupné a chybové záznamy jednotlivých úloh Slurm sú v [samostatnom balíku](01_results/slurm_logy/README.md); zahŕňajú aj prípravné a neúspešné pokusy.

## Pôvodné CFD referenčné dáta

Dostupné pôvodné CFD referencie, trajektórie a diagnostika sú v [samostatnom dátovom balíku](01_results/vlastne_CFD_data/README.md).
