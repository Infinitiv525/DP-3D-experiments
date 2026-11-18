# Kde sú dáta a čo je archivované

Zachované sú modely a výsledky starších 3D pokusov aj vlastné CFD referenčné exporty, trajektórie, siete výpočtovej oblasti, sondy, solverové logy a kontroly kvality. Úspešné, čiastočné a nahradené piloty sa rozlišujú pôvodnou cestou; ich prítomnosť neznamená úspešný finálny experiment.

V tomto repozitári: **284 checkpointových súborov** a **70 vybraných modelov**. Checkpointy zahŕňajú aj priebežné a posledné stavy; ich počet nie je počet nezávislých finálnych sietí.

| Balík | Pôvodných súborov | Archív na GitHube |
|---|---:|---:|
| [Modely a základné výsledky](https://github.com/Infinitiv525/DP-3D-experiments/releases/tag/research-artifacts-v1) | 5267 | 0.466 GB |
| [Doplnené vedecké záznamy a merania](https://github.com/Infinitiv525/DP-3D-experiments/releases/tag/scientific-data-supplement-20261118) | 1262 | 0.038 GB |
| [Historické numerické výstupy](https://github.com/Infinitiv525/DP-3D-experiments/releases/tag/historical-numerical-outputs-20261118) | 24 | 0.169 GB |
| [Ďalšie pôvodné diagnostiky a zdroje](https://github.com/Infinitiv525/DP-3D-experiments/releases/tag/research-records-completion-20261118) | 714 | 0.209 GB |
| [Pôvodné stdout a stderr úloh Slurm](https://github.com/Infinitiv525/DP-3D-experiments/releases/tag/scheduler-logs-20261118) | 1056 | 0.001 GB |
| [Vlastné CFD referencie a pôvodné piloty](https://github.com/Infinitiv525/DP-3D-experiments/releases/tag/own-cfd-reference-data-20261118) | 16483 | 22.345 GB |

Veľké váhy a numerické polia sú **v Releases**, aby sa pri práci so zdrojovým kódom nemuseli automaticky sťahovať. [Zoznam všetkých priamych odkazov a SHA256](PUBLIC_DOWNLOADS.json). Archívy sú členené podľa modelov alebo pôvodných pokusov. Každý balík má manifest s pôvodnými cestami; historické a neúspešné pokusy zostávajú oddelené od finálneho porovnania. V starších nemenných manifestoch sa pôvodné vylúčenia vzťahujú iba na daný balík; dodatočné vydania ich rozširujú podľa tejto tabuľky.

## Ako sa zorientovať

- [Výsledky a tabuľky](.)
- [Vybrané modely a odkazy na váhy](../02_models/README.md)
- [Metodika](../03_methodology/)
- [Zdrojový kód](../04_source/)
- [História pokusov](../05_history/)

## Overenie a rozsah

Každý uložený súbor má SHA256. Overené bolo rozbalenie archívov, serverový SHA256 a úplné spätné prečítanie archívov na Perune. Aktuálne priame odkazy boli samostatne overené bez prihlásenia. Počas dopĺňania sa dáta prenášali priamo Perun → GitHub, bez modelov alebo archívov cez PC, bez nového tréningu a bez nového počítania polí.

Toto je archív vlastných výskumných artefaktov, nie úplná kópia súborového systému Perunu. Neduplikujú sa verejné vstupné FlowBench dáta, obnoviteľné tréningové a vstupné cache, stiahnuté základné LLM váhy ani softvérové prostredia. Nezverejňujú sa prístupové údaje, autentifikačné pomocné súbory ani rozpracovaná dizertácia. Staré surové paralelné/reštartovacie CFD adresáre nie sú úplne zrkadlené; spracované referenčné exporty a diagnostiky sú zahrnuté. Pri čítaní starej reštartovacej metadátovej vetvy sa pozorovalo čakanie na NFS; túto výnimku audit uvádza otvorene. Pôvodné výskumné dáta na Perune neboli zmazané.

Pôvodný [FlowBench](https://huggingface.co/datasets/BGLab/FlowBench) a ďalšie cudzie podklady si ponechávajú svoje autorstvo a licencie; archív im nepriraďuje novú licenciu.

## Súvisiace experimenty

- [Staršie 3D pokusy](https://github.com/Infinitiv525/DP-3D-experiments)
- [Porovnanie LLM](https://github.com/Infinitiv525/DP-LLM-comparison)
- [Porovnanie architektúr](https://github.com/Infinitiv525/DP-architecture-comparison)

Podrobný textový záznam: [DOKUMENTACIA_ARCHIVACIE.txt](DOKUMENTACIA_ARCHIVACIE.txt).
