"""Reproducible Slovak report from frozen numeric metadata; CPU plotting only."""
import argparse
import hashlib
import html
import json
import math
from pathlib import Path
import statistics as st
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

METHODS=['random','tpe','qwen3_8b','ministral3_8b','phi4_mini','granite33_8b']
NAMES={'manual':'Manuálne','random':'Random','tpe':'TPE','qwen3_8b':'Qwen3-8B',
       'ministral3_8b':'Ministral-3-8B','phi4_mini':'Phi-4-mini','granite33_8b':'Granite-3.3-8B'}
PARAMS=['learning_rate','weight_decay','pressure_weight','divergence_weight']


def read(p):return json.loads(p.read_text(encoding='utf-8'))
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def save(p,x):p.write_text(json.dumps(x,ensure_ascii=False,indent=2,allow_nan=False)+'\n',encoding='utf-8')
def key(p):return tuple(float(p[k]) for k in PARAMS)
def fmt(v,n=4):return f'{v:.{n}f}'.replace('.',',')


def table(headers,rows):
    rows=[[str(x) for x in row] for row in rows]
    widths=[max(len(str(h)),*(len(r[j]) for r in rows)) for j,h in enumerate(headers)]
    def line(row):return ' | '.join(str(v).ljust(widths[j]) for j,v in enumerate(row))
    return '\n'.join([line(headers),'-+-'.join('-'*w for w in widths)]+[line(r) for r in rows])


def run(folder):
    data=read(folder/'report_inputs.json');receipt=read(folder/'input_receipt.json')
    assert sha(folder/'report_inputs.json')==receipt['report_inputs_sha256']
    assert data['verification']['status']=='FLOWBENCH_MAIN_FINAL_MEASUREMENTS_VERIFIED'
    a=data['accuracy'];l=data['latency'];paired=data['paired'];catalog=data['catalog']
    assert a['model_count']==l['model_count']==70 and len(a['references'])==95
    assert len(data['proposals'])==108 and len(data['studies'])==18
    metrics={};costs={};behavior={};resources={};winners=[]
    for method in ['manual']+METHODS:
        refs=[r for r in a['references'] if r['method']==method]
        values=[a['models'][r['training_id']]['metrics'] for r in refs]
        row={k:st.mean(v[k] for v in values) for k in ['S','velocity_error','pressure_error']}
        row['logical_final_references']=len(refs)
        if method=='manual':
            row['seed_mean_S']=[a['models'][r['training_id']]['metrics']['S'] for r in refs]
        else:
            row['repeat_mean_S']=[st.mean(a['models'][r['training_id']]['metrics']['S'] for r in refs if r['repeat']==rep) for rep in range(3)]
            row['repeat_sd_S']=st.stdev(row['repeat_mean_S'])
        row['mean_final_training_process_seconds']=st.mean(catalog['trainings'][r['training_id']]['result']['seconds'] for r in refs)
        metrics[method]=row
        if method=='manual':continue
        studies=sorted([s for s in data['studies'] if s['method']==method],key=lambda s:s['repeat'])
        costs[method]=dict(training_process_mean_seconds=st.mean(s['training_seconds_spent'] for s in studies),
            llm_decision_process_mean_seconds=st.mean(s['decision_seconds_spent'] for s in studies))
        costs[method]['combined_recorded_process_mean_seconds']=sum(costs[method].values())
        allocations=[r for r in data['budget']['payload']['allocations'] if r['label'].startswith('search:'+method+'_r')]
        assert len(allocations)==3
        costs[method]['actual_adaptive_search_allocation_mean_seconds']=st.mean(r['charged_gpu_seconds'] for r in allocations)
        unique=[];novel=0;records=0;shared_wins=0;curves=[]
        for s in studies:
            trials=s['trials'];seen=set();best=float('inf');curve=[]
            assert len(trials)==12 and all(t['status']=='complete' for t in trials)
            winner=min(trials,key=lambda t:(t['S'],t['number']))
            winners.append(dict(method=method,repeat=s['repeat'],**winner))
            shared_wins+=winner['number']<3
            for t in trials:
                k=key(t['parameters'])
                if t['number']>=3:
                    novel+=k not in seen;records+=t['S']<best
                seen.add(k);best=min(best,t['S']);curve.append(best)
            unique.append(len(seen));curves.append(curve)
        behavior[method]=dict(unique_configurations_per_repeat=unique,
            unique_mean=st.mean(unique),novel_adaptive_configs=novel,duplicate_adaptive_configs=27-novel,
            new_validation_records=records,shared_start_winners=shared_wins,validation_best_so_far=curves)
        proposals=[p for p in data['proposals'] if p['method']==method]
        if proposals:
            assert math.isclose(sum(p['process_seconds'] for p in proposals),
                sum(s['decision_seconds_spent'] for s in studies),rel_tol=1e-12)
            raw=sum(p['attempts'][0]['transport'].get('raw_strict_json_valid',False) for p in proposals)
            repaired=sum(len(p['attempts'])>1 for p in proposals)
            valid_evidence=0
            for p in proposals:
                history_ids={h['id'] for h in p['feedback_history'] if h['status']=='complete'}
                valid_evidence+=set(p['configuration_with_reason']['evidence_ids']).issubset(history_ids)
            assert valid_evidence==27
            gpu=[p['peak_sampled_nvml_gpu_bytes'] for p in proposals if p['peak_sampled_nvml_gpu_bytes'] is not None]
            resources[method]=dict(proposals=27,one_repair=repaired,initial_raw_strict_json=raw,
                initial_without_repair=27-repaired,all_final_valid=27,evidence_ids_in_available_history=valid_evidence,
                median_process_seconds=st.median(p['process_seconds'] for p in proposals),
                median_load_seconds=st.median(p['load_seconds'] for p in proposals),
                median_generation_seconds=st.median(sum(x['generation_seconds'] for x in p['attempts']) for p in proposals),
                max_sampled_gpu_gib=max(gpu)/2**30 if gpu else None,
                max_sampled_rss_gib=max(p['peak_sampled_rss_bytes'] for p in proposals)/2**30,
                total_cpu_seconds=sum(p['cpu_seconds'] for p in proposals),
                total_input_tokens=sum(x['input_tokens'] for p in proposals for x in p['attempts']),
                total_output_tokens=sum(x['output_tokens'] for p in proposals for x in p['attempts']))
    for m,row in metrics.items():
        row['relative_S_difference_vs_tpe_percent']=100*(row['S']/metrics['tpe']['S']-1)
    # Check the reported means against all geometry x search x final-seed cells.
    for i,m in enumerate(METHODS):
        for metric in ['S','velocity_error','pressure_error']:
            assert math.isclose(np.asarray(paired['tensors'][metric])[i].mean(),metrics[m][metric],abs_tol=1e-12)
    examples=[]
    for method in METHODS[2:]:
        p=next(p for p in data['proposals'] if p['method']==method and p['repeat']==0 and p['number']==3)
        examples.append(dict(method=method,selection_rule='First adaptive proposal in repeat0; not selected for favorability',
            configuration=p['configuration_with_reason'],observed_validation_S=p['observed_validation_S'],
            prior_best_validation_S=p['prior_best_validation_S']))
    summary=dict(method_metrics=metrics,search_costs=costs,search_behavior=behavior,llm_resources=resources,
        winners=winners,rationale_examples=examples,prespecified_statistics=data['statistics'],
        latency=data['verification']['worst_round_p99_wall_ms'],
        inference_scope='One fixed steady3D task, fixed architecture and limited search; no broad claim against LLMs.',
        input_sha256=sha(folder/'report_inputs.json'),builder_sha256=sha(Path(__file__)))
    save(folder/'comparison_numbers.json',summary)

    plt.rcParams.update({'font.family':'DejaVu Sans','font.size':10,'axes.spines.top':False,'axes.spines.right':False})
    figures=[]
    def finish(fig,name):
        fig.tight_layout();fig.savefig(folder/(name+'.png'),dpi=160,bbox_inches='tight')
        fig.savefig(folder/(name+'.svg'),bbox_inches='tight');plt.close(fig);figures.append(name)
    order=['manual']+METHODS;xx=np.arange(7)
    colors=['#7c858d','#8472ac','#17877e']+['#497bad']*4
    fig,ax=plt.subplots(figsize=(10.8,4.8))
    ax.bar(xx,[metrics[m]['S'] for m in order],color=colors,alpha=.8,width=.65)
    for j,m in enumerate(order):
        vals=metrics[m].get('repeat_mean_S',metrics[m].get('seed_mean_S'))
        ax.scatter(np.linspace(j-.15,j+.15,len(vals)),vals,c='#151d26',s=22,zorder=3)
    ax.set_xticks(xx,[NAMES[m] for m in order],rotation=15,ha='right')
    ax.set_ylabel('Testovacie skóre S (nižšie je lepšie)');ax.set_ylim(0,.29)
    ax.set_title('Výber konfigurácie: testovacia chyba po finálnych tréningoch z čerstvej inicializácie')
    fig.text(.5,-.015,'Body: 3 priemery search opakovaní; pri manuálnom nastavení 5 finálnych seedov. Nie intervaly spoľahlivosti.',ha='center',fontsize=9)
    finish(fig,'01_testovacia_presnost')
    contrasts=data['statistics']['contrasts'];fig,ax=plt.subplots(figsize=(9.3,4.6))
    for j,c in enumerate(contrasts):
        ax.errorbar(c['mean_difference'],j,xerr=[[c['mean_difference']-c['lower']],[c['upper']-c['mean_difference']]],fmt='o',capsize=5,color='#35618c')
    ax.axvline(0,color='black',linewidth=1);ax.set_yticks(range(4),[NAMES[c['method']] for c in contrasts]);ax.invert_yaxis()
    ax.set_xlim(-.008,.095);ax.set_xlabel('Rozdiel S: LLM − TPE (kladná hodnota = vyššia chyba LLM)')
    ax.set_title('Vopred určené 98,75 % bootstrap intervaly; 4 porovnania')
    fig.text(.5,-.01,'Bonferroni; približné nominálne rodinné pokrytie 95 %. Iba 3 search opakovania.',ha='center',fontsize=9)
    finish(fig,'02_porovnanie_llm_tpe')
    fig,ax=plt.subplots(figsize=(10.5,4.8));x=np.arange(6)
    base=np.array([costs[m]['training_process_mean_seconds']/60 for m in METHODS])
    extra=np.array([costs[m]['llm_decision_process_mean_seconds']/60 for m in METHODS])
    ax.bar(x,base,color='#528c9c',label='12 tréningových procesov vrátane 3 štartov')
    ax.bar(x,extra,bottom=base,color='#e6ae62',label='LLM rozhodovacie procesy')
    ax.set_xticks(x,[NAMES[m] for m in METHODS],rotation=15,ha='right');ax.set_ylabel('Minúty na jedno search opakovanie (priemer)')
    ax.set_title('Zaznamenaný čas procesov pri výbere konfigurácie');ax.legend(loc='upper left',fontsize=8)
    ax.set_ylim(0,145)
    fig.text(.5,-.015,'Logické porovnanie procesov, nie fakturácia GPU. Neobsahuje finálne tréningy; TPE/Random bookkeeping nie je v LLM čase.',ha='center',fontsize=8)
    finish(fig,'03_naklady_vyberu')
    fig,ax=plt.subplots(figsize=(10,4.8))
    for m in METHODS:
        ys=np.mean(behavior[m]['validation_best_so_far'],axis=0)
        ax.plot(range(1,13),ys,marker='o',markersize=3,label=NAMES[m])
    ax.axvline(3.5,color='gray',linestyle=':',linewidth=1);ax.set_xticks(range(1,13))
    ax.set_xlabel('Počet započítaných pokusov (prvé 3 sú spoločné štarty)');ax.set_ylabel('Najlepšie validačné S doteraz; priemer 3 behov')
    ax.set_title('Priebeh vyhľadávania: validačné dáta, nie test');ax.legend(fontsize=8,ncol=2)
    finish(fig,'04_priebeh_hladania')
    modes=['model_only','gpu_pipeline','host_to_host'];fig,ax=plt.subplots(figsize=(8.8,4.7))
    vals=[summary['latency'][m]['max'] for m in modes]
    ax.bar(range(3),vals,color=['#7195a5','#17877e','#9b88b4']);ax.axhline(5,color='#943e3e',linestyle='--',label='Hranica 5 ms')
    for j,v in enumerate(vals):ax.text(j,v+.08,f'{v:.3f} ms',ha='center')
    ax.set_xticks(range(3),['Iba sieť na GPU','GPU pipeline (hlavný)','CPU → GPU → CPU'])
    ax.set_ylim(0,5.7);ax.set_ylabel('Najhoršie p99 zo všetkých modelov a kôl [ms]')
    ax.set_title('NVIDIA H200, batch 1, 64³, FP32');ax.legend()
    fig.text(.5,-.01,'Bez diskového I/O, tvorby geometrie, vykresľovania a rozhodovania LLM.',ha='center',fontsize=9)
    finish(fig,'05_latencia')

    accuracy_rows=[[NAMES[m],fmt(metrics[m]['S']),fmt(metrics[m]['velocity_error']),fmt(metrics[m]['pressure_error']),
        ('—' if m=='tpe' else ('+' if metrics[m]['relative_S_difference_vs_tpe_percent']>=0 else '')+fmt(metrics[m]['relative_S_difference_vs_tpe_percent'],1)+' %')] for m in order]
    contrast_rows=[[NAMES[c['method']],fmt(c['mean_difference']),f"[{fmt(c['lower'])}; {fmt(c['upper'])}]",'celý interval nad 0'] for c in contrasts]
    cost_rows=[[NAMES[m],fmt(costs[m]['training_process_mean_seconds']/60,2),fmt(costs[m]['llm_decision_process_mean_seconds']/60,2),fmt(costs[m]['combined_recorded_process_mean_seconds']/60,2)] for m in METHODS]
    behavior_rows=[[NAMES[m],'/'.join(map(str,behavior[m]['unique_configurations_per_repeat'])),str(behavior[m]['duplicate_adaptive_configs'])+'/27',str(behavior[m]['shared_start_winners'])+'/3'] for m in METHODS]
    resource_rows=[[NAMES[m],fmt(resources[m]['median_process_seconds'],1),fmt(resources[m]['median_generation_seconds'],1),fmt(resources[m]['max_sampled_gpu_gib'],2),str(resources[m]['initial_raw_strict_json'])+'/27',str(resources[m]['one_repair'])+'/27'] for m in METHODS[2:]]
    latency_rows=[[m,fmt(summary['latency'][m]['max'],4),'70/70'] for m in modes]
    sections=[]
    def section(title,text,headers=None,rows=None,figure=None):sections.append(dict(title=title,text=text,headers=headers,rows=rows,figure=figure))
    section('1. Čo sme skúmali',
        'Skúmali sme, či lokálne LLM dokážu pri obmedzenom počte pokusov vybrať lepšie tréningové nastavenia pevnej neurónovej siete než manuálne nastavenie, náhodné hľadanie a TPE. LLM nepočítali prúdenie, nevytvárali rovnice ani nemenili architektúru. Zo zaznamenanej validačnej histórie navrhovali learning_rate, weight_decay, pressure_weight a divergence_weight.\n\n'
        'Úloha: ustálené 3D prúdenie FlowBench na mriežke 64³. Vstupom je geometria opísaná SDF a Reynoldsovo číslo, výstupom u, v, w a tlak. Ide o dátovo trénovanú FNO s voliteľným divergenčným regularizačným členom v stratovej funkcii, nie o experiment s generovaním PDE alebo všeobecný dôkaz o PINN. V tomto reporte nie je vlastné časovo závislé CFD ani voľná hladina.\n\n'
        'Použité lokálne modely: Qwen/Qwen3-8B, mistralai/Ministral-3-8B-Instruct-2512, microsoft/Phi-4-mini-instruct a ibm-granite/granite-3.3-8b-instruct. Presné revízie, loader a generovanie sú uložené v report_inputs.json; bez plateného LLM API. Spoločné generovanie: do_sample=false, max_new_tokens=512, najviac jedna oprava formátu. Qwen a Granite mali vypnutý thinking režim podľa konfigurácie.')
    section('2. Ako bolo porovnanie urobené',
        'Dáta sa delili podľa geometrie: 700 tréningových, 150 validačných a 150 testovacích prípadov. Test tvorí 15 geometrií × 10 Reynoldsových čísel. Normalizácia je výlučne z tréningu; výber konfigurácie a checkpointu používal iba validáciu.\n\n'
        'Každá zo šiestich vyhľadávacích metód mala 3 opakovania a 12 započítaných pokusov: 3 spoločné štarty + 9 ďalších návrhov. Každý tréning mal 20 epoch a rovnakú FNO: width=12, layers=2, modes=4, batch=1. Limit tréningových procesov bol 7200 s na metódu/opakovanie; LLM mali samostatný rozhodovací limit 1890 s. Všetky behy skončili na limite 12 pokusov, nie vyčerpaním tréningového času. Preto ide hlavne o porovnanie pri rovnakom počte pokusov, NIE pri rovnakom celkovom end-to-end čase.\n\n'
        'Najlepšia validačná konfigurácia z každého hľadania sa trénovala znovu s 5 čerstvými seedmi 410001–410005; manuálna konfigurácia tiež s 5 seedmi. Spolu je 95 logických výsledkov, ale len 70 fyzických modelov: 25 dodatočných odkazov opakovane používa identický model. Aliasy nie sú nové nezávislé vzorky. Celá tréningová fáza obsahovala 9 spoločných + 162 adaptívnych + 70 finálnych tréningov.\n\n'
        'S = 0,5 Eu + 0,5 Ep; menšie je lepšie. Eu je relatívna L2 chyba spoločného vektora rýchlosti, Ep relatívna L2 chyba tlaku po odstránení tlakovej konštanty, iba v kvapaline. Redukcie sú float64. Najprv sa priemeruje 10 Re v geometrii a potom geometrie rovnakou váhou. Divergencia je len diagnostika; nie dôkaz fyzikálnej správnosti.')
    section('3. Testovacia presnosť',
        'Nasledujúce hodnoty sú priemery cez všetkých 15 testovacích geometrií, 3 search opakovania a 5 finálnych seedov; pri manuálnom nastavení cez geometrie a 5 seedov. Percento je relatívny rozdiel priemerného skóre S voči TPE, nie percento nesprávne predikovaných buniek.\n\n'
        'TPE malo najnižšie priemerné S aj obe zložky chyby. Random bol v priemere druhý; toto poradie je opisné, nie nové potvrdzujúce štatistické porovnanie. Všetky LLM mali vyššiu chybu než TPE. Ich priemerné S bolo nižšie než manuálny základ, ale ich chyba rýchlosti bola vyššia než pri manuálnom základe: nižšie kombinované S preto netreba zamieňať za zlepšenie každej fyzikálnej veličiny.',
        ['Metóda','S','Eu','Ep','Rozdiel S vs. TPE'],accuracy_rows,'01_testovacia_presnost')
    section('4. Vopred určené porovnania LLM proti TPE',
        'Použil sa pôvodný krížený párovaný bootstrap: 10 000 replikácií, seed 510001, samostatné resamplovanie osí geometria × search opakovanie × finálny seed; rovnaké indexy pre všetky metódy. Štyri dvojstranné 98,75 % intervaly používajú Bonferroniho úpravu s približným nominálnym rodinným pokrytím 95 %.\n\n'
        'Všetky štyri intervaly rozdielu S(LLM) − S(TPE) sú nad nulou. V rámci vopred určeného vyhodnotenia teda výsledok neukazuje prínos LLM oproti TPE; ukazuje smer k vyššej chybe LLM. Je to výsledok tejto úlohy a protokolu, nie tvrdenie, že LLM nemôžu pomôcť pri iných úlohách. Pokrytie bootstrapu je približné a iba 3 search opakovania obmedzujú istotu záveru. Formálne znenie dizertačných hypotéz sa tu nemení.',
        ['LLM','Priemer LLM − TPE','98,75 % interval','Smer'],contrast_rows,'02_porovnanie_llm_tpe')
    section('5. Koľko stál výber konfigurácie',
        'Časy nižšie sú priemerné zaznamenané časy procesov na jedno search opakovanie. Tréning zahŕňa všetkých 12 pokusov vrátane logicky započítaných spoločných štartov. Rozhodovanie LLM zahŕňa samostatné spustenie návrhového procesu, načítanie modelu, generovanie a prípadnú opravu. Generovanie samotné je vykázané zvlášť v ďalšej tabuľke.\n\n'
        'Toto nie je priama fakturácia GPU ani nástenný čas paralelného experimentu. Spoločné štarty sa fyzicky spustili iba raz pre opakovanie. Random/TPE nemajú samostatne meraný LLM proces; nula neznamená nulové CPU náklady algoritmu a obsluhy. Finálne tréningy a záverečné merania nie sú zahrnuté v tejto tabuľke. Presné alokačné a finálne tréningové údaje sú v comparison_numbers.json a report_inputs.json. Manuálny základ nemal automatické hľadanie; ľudský čas prípravy sa nemeral.\n\n'
        'Použitý proces načítaval lokálny LLM pre každý návrh znovu. Studený štart a súborová cache ovplyvnili časy; perzistentne načítaný LLM by mohol znížiť prevádzkovú réžiu. Tento experiment však takú optimalizáciu nemeral a neukazuje, že by sama zlepšila kvalitu návrhov.',
        ['Metóda','Tréning [min]','LLM procesy [min]','Súčet [min]'],cost_rows,'03_naklady_vyberu')
    section('6. Platnosť odpovedí a merané zdroje LLM',
        'Všetkých 108 návrhov bolo po povolenej oprave použiteľných; 103 nepotrebovalo opakovanie generovania a 5 potrebovalo jednu opravu, všetkých päť pri Granite. To nie je to isté ako 100 % striktný JSON hneď v surovom prvom výstupe: spoločná transportná vrstva vedela odstrániť jeden úplný JSON code fence. Ministral vo všetkých 27 prvých odpovediach použil JSON code fence, ktorý odstránila transportná vrstva bez opakovania generovania. Počty surového JSON sú v tabuľke; tokeny sú zachované v číselných prílohách. Všetky evidence_ids odkazovali na dostupnú dokončenú históriu; to nepotvrdzuje pravdivosť slovného zdôvodnenia. Adaptívne tréningy nemali neúspešný pokus.\n\n'
        'Čas je medián na jeden z 27 návrhov danej rodiny. Generovanie zahŕňa aj opravu, ak bola potrebná. VRAM je maximum vzorkovaného procesového NVML merania, nie garantovaný presný okamžitý peak; rovnaké obmedzenie platí pre vzorkovanú RAM. CPU čas, RAM a počty tokenov sú tiež uložené v comparison_numbers.json.',
        ['LLM','Proces [s]','Generovanie [s]','Max. vzork. VRAM [GiB]','Prvý raw JSON','Opravy'],resource_rows)
    section('7. Čo LLM pri hľadaní robili',
        'Výrazným pozorovaním je opakovanie rovnakých konfigurácií. Počet jedinečných konfigurácií v troch behoch je uvedený oddelene; porovnávajú sa presné numerické hodnoty všetkých štyroch parametrov. Duplicitný adaptívny návrh už bol v histórii toho istého behu; stále sa trénoval s iným seedom a spotreboval celý pokus.\n\n'
        'Niektoré rodiny tak venovali väčšinu pokusov opakovanému tréningu podobného alebo rovnakého nastavenia, zatiaľ čo Random a TPE skúšali nové konfigurácie. Je to možný mechanizmus slabšieho výsledku, nie preukázaná príčina: pri každom tréningu sa menil aj seed a neprebehla ablačná štúdia so zákazom duplicít. Víťaz zo spoločných štartov znamená, že dodatočné návrhy v danom behu neprekonali najlepší spoločný validačný výsledok.',
        ['Metóda','Jedinečné z 12: behy 1/2/3','Duplicitné nové návrhy','Víťaz zo štartov'],behavior_rows,'04_priebeh_hladania')
    example_lines=[]
    for ex in examples:
        c=ex['configuration'];example_lines.append(NAMES[ex['method']]+':\n'
            +'  reason: '+c['reason']+'\n  expected_effect: '+c['expected_effect']+'\n  risk: '+c['risk']
            +'\n  evidence_ids: '+', '.join(c['evidence_ids'])
            +'\n  Najlepšie pred návrhom S='+fmt(ex['prior_best_validation_S'],6)
            +'; následný validačný tréning návrhu S='+fmt(ex['observed_validation_S'],6)+'.')
    section('8. Sú uložené aj zdôvodnenia?',
        'Áno. Uložený je verejný štruktúrovaný výstup reason, evidence_ids, expected_effect a risk, vstupná história, surové odpovede, chyby a opravy. Nie je to skrytý vnútorný myšlienkový postup modelu. Uvedené tvrdenia o očakávanom účinku sú návrhy LLM, nie kauzálne potvrdené vysvetlenia.\n\n'
        'Nasledujúce príklady boli vybrané pevným pravidlom: prvý adaptívny návrh v prvom opakovaní každej rodiny, nie podľa toho, či vyzeral úspešne. Originálne anglické formulácie sú ponechané bez prikrášlenia. Rozdiel validačného výsledku jedného tréningu je ovplyvnený aj seedom.\n\n'+'\n\n'.join(example_lines))
    section('9. Splnenie hranice 5 ms',
        'Všetkých 70 fyzických modelov splnilo hlavné kritérium: GPU pipeline synchronized wall p99 ≤ 5 ms V KAŽDOM z troch kôl. Uložených a znovu overených je 630 000 jednotlivých časovaní plus 63 000 zahrievacích volaní. Tabuľka uvádza najhorší p99 naprieč všetkými modelmi a kolami, nie najhorší jednotlivý čas. p99 nie je tvrdá garancia, že každé volanie bude pod hranicou.\n\n'
        'Hlavný GPU režim zahŕňa normalizáciu, koordinátové vstupy, sieť a denormalizáciu, pričom vstupy sú už na GPU. CPU → GPU → CPU zahŕňa prenosy a je sekundárny. Nie je zahrnuté čítanie z disku, vytváranie SDF, vizualizácia ani rozhodovanie LLM. Hardvér je H200, batch 1, 64³, FP32, TF32 vypnuté. Tri kolá časovania nie sú tri nezávislé vedecké replikácie.\n\n'
        'Rýchlosť medzi metódami veľmi nerozlišuje, pretože všetky používajú tú istú architektúru a veľkosť vstupu. Splnenie 5 ms preto nemožno pripísať LLM ani ním dokazovať prínos LLM pri hľadaní konfigurácie.',
        ['Režim','Najhorší p99 [ms]','Modely s p99 ≤ 5 ms v každom kole'],latency_rows,'05_latencia')
    section('10. Čo musíme priznať a čo ešte nevieme',
        'Meranie malo dve technické prerušenia. Prvé bolo pred dátami pre import pomocnej knižnice. Druhé už otvorilo prvý testovací prípad [10] a zlyhalo pri nesprávnom rozmere masky diagnostiky divergencie. Neuložilo skóre modelu a neslúžilo na zmenu modelov. Opravený výpočet sa zhodoval s pôvodným trénerom na všetkých 150 validačných prípadoch; až po výslovnom súhlase používateľa sa test zopakoval. Staré claims, autorizácie a chyby ostali zachované. Test nesmieme opisovať ako nikdy predtým ani čiastočne otvorený.\n\n'
        'Rozsah záveru obmedzuje jedna úloha, jedna pevná FNO, 3 search opakovania, 20-epochový tréning a krátke vyhľadávanie. Test má nové geometrie z existujúceho rozdelenia; nepreukazuje všeobecnú generalizáciu na nové rodiny geometrie alebo iné fyzikálne režimy. Referenčné dáta sú prevzorkované na 64³; nejde o nový dôkaz vernosti voči pôvodnému numerickému riešeniu. Štúdia nepokrýva voľnú hladinu, časový rollout ani generovanie rovníc.\n\n'
        'Podmienky generovania boli spoločné, no rodiny sa líšia veľkosťou a loaderom. Hodnotíme konkrétne lokálne modely s konkrétnym promptom a dekódovaním, nie čistý izolovaný účinok architektúry LLM. Vyhľadávacia stabilita, duplicity a racionalizácie sú dodatočný opis už hotových behov, nie nové potvrdzujúce hypotézy.')
    section('11. Záver a najbližší postup',
        'Technicky je táto FlowBench sada dokončená, vrátane presnosti, zdrojov, zdôvodnení a rýchlosti. V tomto protokole LLM neprekonali TPE a pridali čas na rozhodovanie. Pozitívny výsledok je reprodukovateľná prevádzka štyroch lokálnych rodín a splnenie cieľa rýchlej inferencie; to však nenahrádza výsledok porovnania presnosti. Aj dobre zdokumentovaný negatívny výsledok má výskumnú hodnotu.\n\n'
        'Teraz treba uzavrieť túto sadu ako nemenný výsledok, skontrolovať tabuľky a grafy s vedúcim a až potom rozhodnúť o ďalšej nezávislej úlohe. Ak by sme skúmali napríklad zákaz duplicitných návrhov alebo perzistentný LLM proces, pôjde o nový, vopred opísaný experiment; nepoužijeme tento už otvorený test na ladenie a nebudeme ho zamieňať za opravu pôvodného výsledku. Žiadne nové GPU úlohy tento report nespúšťa.\n\n'
        'Zaúčtovanie celého hlavného experimentu: 37,0225 GPU-h z interného limitu 70 h; zostáva 32,9775 GPU-h. Súčet obsahuje aj osem starších konzervatívnych plných rezervácií, nie pri každej historickej položke presný runtime. Úspešná validačná kontrola + finálny test + latencia spotrebovali spolu 717 GPU-s; technické neúspešné pokusy zostali započítané.')
    section('12. Podklady a reprodukovateľnosť',
        'Všetky číselné podklady sú v report_inputs.json (iba malé metadáta a reporty, nie CFD polia alebo váhy) a odvodené tabuľky v comparison_numbers.json. report_inputs.json obsahuje SHA-256 všetkých 359 načítaných podkladov vrátane jednotlivých LLM návrhov a ich spätnej väzby. input_receipt.json overuje celý balík. Sú zachované surové verejné odpovede, tokeny, zdroje a všetkých 108 návrhov.\n\n'
        'Grafy sú uložené ako PNG aj SVG. Kód: test/analysis/collect_main_report_inputs_v1.py a build_comparative_report_v1.py. Prehľad reportu je v report.txt a report.html. JSON čísla majú plnú presnosť, textové tabuľky sú zaokrúhlené. Pôvodné merania ostávajú na Perune vo final_measurements_v3. Žiadne veľké polia ani modelové váhy neboli stiahnuté.\n\n'
        'SHA-256 vstupného balíka: '+summary['input_sha256'])
    title='POROVNÁVACÍ REPORT: LOKÁLNE LLM PRI VÝBERE KONFIGURÁCIE FNO\nFlowBench, ustálené 3D prúdenie; výsledky k 22. 10. 2026\n'
    chunks=[title]
    for s in sections:
        chunks.extend([s['title'].upper(),s['text']])
        if s['headers']:chunks.append(table(s['headers'],s['rows']))
        if s['figure']:chunks.append('Graf: '+s['figure']+'.png (aj SVG)')
    (folder/'report.txt').write_text('\n\n'.join(chunks)+'\n',encoding='utf-8')
    blocks=[]
    for s in sections:
        b='<section><h2>'+html.escape(s['title'])+'</h2>'
        b+=''.join('<p>'+html.escape(p).replace('\n','<br>')+'</p>' for p in s['text'].split('\n\n'))
        if s['headers']:
            b+='<div class="table"><table><thead><tr>'+''.join('<th>'+html.escape(str(v))+'</th>' for v in s['headers'])+'</tr></thead><tbody>'
            b+=''.join('<tr>'+''.join('<td>'+html.escape(str(v))+'</td>' for v in row)+'</tr>' for row in s['rows'])+'</tbody></table></div>'
        if s['figure']:b+='<img src="'+s['figure']+'.png" alt="'+html.escape(s['title'])+'">'
        blocks.append(b+'</section>')
    page='<!doctype html><html lang="sk"><meta charset="utf-8"><meta name="viewport" content="width=device-width"><title>Porovnávací report LLM a FNO</title><style>body{font:16px/1.65 system-ui,sans-serif;color:#233345;background:#f3f5f7;margin:0}main{max-width:1100px;background:white;margin:24px auto;padding:36px 48px}h1{line-height:1.2}h2{color:#1b6963;margin-top:36px}p{max-width:100ch}table{border-collapse:collapse;width:100%;font-size:14px}td,th{text-align:left;padding:10px;border-bottom:1px solid #d7e0e4}th{background:#eaf1f1}.table{overflow:auto}img{width:100%;height:auto;margin:18px 0}small{color:#627080}@media print{body{background:white}main{margin:0;padding:0}section{break-inside:auto}img,table{break-inside:avoid}}</style><main><h1>Lokálne LLM pri výbere konfigurácie FNO</h1><small>Porovnávací report · FlowBench · 22. 10. 2026 · bez nových GPU behov</small>'+''.join(blocks)+'</main></html>'
    (folder/'report.html').write_text(page,encoding='utf-8')
    fig,axs=plt.subplots(3,2,figsize=(16,15))
    for ax,name in zip(axs.flat,figures):ax.imshow(plt.imread(folder/(name+'.png')));ax.axis('off')
    axs.flat[-1].axis('off');fig.tight_layout();fig.savefig(folder/'contact_sheet.png',dpi=100);plt.close(fig)
    files=['report.txt','report.html','comparison_numbers.json','contact_sheet.png']+[f+'.'+ext for f in figures for ext in ['png','svg']]
    save(folder/'output_receipt.json',dict(input_sha256=summary['input_sha256'],builder_sha256=summary['builder_sha256'],
        figures=figures,files={f:dict(sha256=sha(folder/f),bytes=(folder/f).stat().st_size) for f in files}))
    print('COMPARATIVE_REPORT_BUILT',len(sections),'sections',len(figures),'figures',flush=True)


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('folder',type=Path);run(p.parse_args().folder)
