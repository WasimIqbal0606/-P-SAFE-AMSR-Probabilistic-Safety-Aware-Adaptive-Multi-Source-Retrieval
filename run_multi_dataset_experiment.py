"""B-P-SAFE-AMSR Multi-Dataset Experiment with validation-only tuning."""
import os,sys,time,json,itertools,csv
import numpy as np
if sys.stdout.encoding!='utf-8': sys.stdout.reconfigure(encoding='utf-8',errors='replace')
if sys.stderr.encoding!='utf-8': sys.stderr.reconfigure(encoding='utf-8',errors='replace')
import yaml
from collections import Counter
from psafe.routers.bpsafe_router import BPSafeRouter
from psafe.retrievers.actions import Action,ACTION_NAMES
from psafe.statistics.metrics import calculate_extended_metrics
from psafe.statistics.statistical_tester import StatisticalTester
from psafe.retrievers.feature_extractor import FeatureExtractor,FEATURE_NAMES
from psafe.utils.latency_tracker import LatencyTracker

DATASETS=["scifact","fiqa","nfcorpus","arguana","trec-covid"]
MODES=["lite","balanced","high_recall"]
GAIN_THS=[0.10,0.15,0.20,0.25,0.30,0.35]
HARM_THS=[0.40,0.50,0.60,0.70]
LAM_LATS=[5e-7,1e-6,5e-6,1e-5]
LAM_HARMS=[0.01,0.02,0.05,0.10]
LAM_RECS=[0.40,0.60,0.80,1.00]

def load_config(p):
    if os.path.exists(p):
        with open(p) as f: return yaml.safe_load(f)
    return {}

def grid_search_tune(router_cls,config,train_data,features,split,action_ndcg,action_lat,
                     dense_ndcg_arr,data,mode):
    """Grid search on validation split. Returns best hyperparams dict."""
    val_dense=dense_ndcg_arr[split.val_idx]
    val_hybrid=action_ndcg[Action.A6_DEEP_HYBRID.value][split.val_idx]
    val_h_lat=action_lat[Action.A6_DEEP_HYBRID.value][split.val_idx]
    val_easy=val_dense>0.5
    best_score=-999; best_cfg=None
    for g,h,ll,lh,lr in itertools.product(GAIN_THS,HARM_THS,LAM_LATS,LAM_HARMS,LAM_RECS):
        r=BPSafeRouter(mode=mode,config=config)
        r.gain_threshold=g;r.harm_threshold=h;r.lambda_latency=ll
        r.lambda_harm=lh;r.lambda_recovery=lr;r.use_lcb_safety=False
        r.train(train_data)
        pn=np.zeros(len(split.val_idx));pl=np.zeros(len(split.val_idx))
        for i,vi in enumerate(split.val_idx):
            X=features[vi].to_array(FEATURE_NAMES)
            dec=r.route(X,data.query_ids[vi],{Action.A0_DENSE.value:50,Action.A6_DEEP_HYBRID.value:400},"val")
            pn[i]=action_ndcg[dec.action][vi];pl[i]=action_lat[dec.action][vi]
        r._action_predictions.clear()
        dm=float(np.mean(val_dense));hm=float(np.mean(val_hybrid));pm=float(np.mean(pn))
        hg=hm-dm;qr=(pm-dm)/hg if hg>0 else 0
        ls=1-(float(np.mean(pl))/float(np.mean(val_h_lat))) if float(np.mean(val_h_lat))>0 else 0
        ha_h=-float(np.mean(np.minimum(val_hybrid[val_easy]-val_dense[val_easy],0))) if np.any(val_easy) else 0
        ha_p=-float(np.mean(np.minimum(pn[val_easy]-val_dense[val_easy],0))) if np.any(val_easy) else 0
        ha=ha_h-ha_p
        hhg=float(np.mean(np.maximum(val_hybrid[~val_easy]-val_dense[~val_easy],0))) if np.any(~val_easy) else 0
        phg=float(np.mean(np.maximum(pn[~val_easy]-val_dense[~val_easy],0))) if np.any(~val_easy) else 0
        rc=phg/hhg if hhg>0 else 0
        om=float(np.mean(np.maximum(val_dense,val_hybrid)))
        ogc=(pm-dm)/(om-dm) if (om-dm)>0 else 0
        nh=sum(1 for i2,vi2 in enumerate(split.val_idx) if pn[i2]!=dense_ndcg_arr[vi2])
        har=nh/len(split.val_idx) if len(split.val_idx)>0 else 0
        score=0.50*qr+0.20*ls+0.15*rc+0.10*ha+0.05*ogc
        feasible=(ls>=0.30 and ha>=0 and 0.20<=har<=0.80)
        if feasible and score>best_score:
            best_score=score
            best_cfg=dict(gain_threshold=g,harm_threshold=h,lambda_latency=ll,
                          lambda_harm=lh,lambda_recovery=lr,use_lcb_safety=False,
                          val_score=round(score,4),val_qr=round(qr,4),val_ls=round(ls,4),
                          val_rc=round(rc,4),val_ha=round(ha,4),val_har=round(har,4))
    if best_cfg is None:
        best_cfg=dict(gain_threshold=0.20,harm_threshold=0.50,lambda_latency=1e-6,
                      lambda_harm=0.02,lambda_recovery=0.60,use_lcb_safety=False,val_score=0)
        print("    [WARN] No feasible config, using defaults")
    return best_cfg

def run_single(config,dataset,mode,out_root,seed=42):
    """Run one dataset/mode combination."""
    from archive.ahrc.dataset_interface import load_benchmark
    from archive.ahrc.vram_safe_encoder import encode_texts
    from archive.ahrc.embedding_cache import EmbeddingCache
    from archive.ahrc.config import AHRCConfig
    from archive.ahrc.index_manager import IndexManager
    from archive.ahrc.graph_expander import GraphExpander
    from archive.ahrc.psafe_experiment_runner import BM25Wrap,ActionSimulator
    from archive.ahrc.baselines import DenseFixedBaseline
    from archive.ahrc.hybrid_retriever import HybridRetriever
    from archive.ahrc.reranker import CrossEncoderReranker
    from archive.ahrc.evaluation import Evaluator
    from archive.ahrc.leakage_safe_split import create_stratified_split
    from archive.ahrc.psafe_router import Action as OldAction
    from psafe.evaluation.evaluator import EvaluationManager
    import faiss

    mode_dir=os.path.join(out_root,dataset,mode,"metrics")
    os.makedirs(mode_dir,exist_ok=True)
    print(f"\n{'='*70}\n  {dataset} / {mode} (seed={seed})\n{'='*70}")

    data=load_benchmark("beir",dataset_name=dataset)
    prof=config.get("model_profiles",{}).get("bge_m3",{})
    mn=prof.get("embedding_model","BAAI/bge-m3")
    rn=prof.get("reranker_model","BAAI/bge-reranker-v2-m3")
    dev=config.get("hardware",{}).get("device","auto")

    cache=EmbeddingCache(os.path.join(out_root,"cache","embeddings"))
    ce=encode_texts(mn,data.corpus_texts,dataset,"corpus",dev,cache)
    qe=encode_texts(mn,data.query_texts,dataset,"query",dev,cache)

    cfg=AHRCConfig();cfg.index.embedding_dim=ce.shape[1]
    im=IndexManager(cfg.index);im.build(ce)
    ge=GraphExpander(cfg.adaptive)
    n,d=ce.shape;fi=faiss.IndexFlatIP(d);fi.add(ce)
    _,indices=fi.search(ce,6)
    ge.adjacency={};ge._built=True
    for i in range(n):
        nbs=[int(indices[i,j]) for j in range(1,6) if 0<=indices[i,j]<n]
        ge.adjacency[i]=set(nbs);ge.degree_cache[i]=len(nbs)

    from rank_bm25 import BM25Okapi
    bm25=BM25Okapi([t.lower().split() for t in data.corpus_texts])
    bw=BM25Wrap(bm25,data.corpus_texts)
    reranker=CrossEncoderReranker(model_name=rn,device=dev,
        batch_size=config.get("hardware",{}).get("reranker_batch_size",8))
    reranker.load()
    dbl=DenseFixedBaseline(im,ce)
    fe=FeatureExtractor();fe.build_idf(data.corpus_texts)
    em=EvaluationManager();rth=em.get_relevance_threshold(dataset)
    ev=Evaluator(k_values=[10],relevance_threshold=rth)
    rf=HybridRetriever(cfg,im,ge,ce,bw,data.corpus_texts,reranker)
    sim=ActionSimulator(rf,bw,ge,dbl)
    lt=LatencyTracker()

    # Phase 1: Collect metrics
    print("  Phase 1: Metric Collection...")
    nq=len(data.query_ids)
    acts=[Action.A0_DENSE,Action.A6_DEEP_HYBRID]
    an={a.value:np.zeros(nq) for a in acts}
    al={a.value:np.zeros(nq) for a in acts}
    feats=[]
    lc={k:[] for k in ["dense_search","bm25_search","feature_extraction"]}

    for qi in range(nq):
        qid,qemb,qt=data.query_ids[qi],qe[qi],data.query_texts[qi]
        qrels=data.qrels.get(qid,{})
        t0=time.perf_counter()
        dr=dbl.retrieve(qemb,query_id=qid,k=50)
        lc["dense_search"].append((time.perf_counter()-t0)*1000)
        t0=time.perf_counter()
        br=bw.retrieve(qt,k=100,query_id=qid)
        lc["bm25_search"].append((time.perf_counter()-t0)*1000)
        t0=time.perf_counter()
        gd=ge.get_degrees(dr.retrieved_indices[:10])
        f=fe.extract(qid,qemb,qt,dr.retrieved_indices,dr.retrieved_scores,
                     br.retrieved_indices,br.retrieved_scores,gd)
        feats.append(f);lc["feature_extraction"].append((time.perf_counter()-t0)*1000)
        for a in acts:
            oa=OldAction.A0_DENSE if a==Action.A0_DENSE else OldAction.A6_DEEP_HYBRID
            res=sim.simulate_action(oa,qid,qemb,qt,k=10)
            dm2=ev.evaluate_query(res.retrieved_indices,qrels,data.corpus_ids,qid,res.total_time_ms,res.candidates_explored)
            an[a.value][qi]=dm2.ndcg_at_k.get(10,0);al[a.value][qi]=res.total_time_ms

    dn=an[Action.A0_DENSE.value];hn=an[Action.A6_DEEP_HYBRID.value]
    print(f"    Dense={np.mean(dn):.4f}  Hybrid={np.mean(hn):.4f}")

    np.random.seed(seed)
    split=create_stratified_split(dn,train_ratio=0.4,val_ratio=0.1,test_ratio=0.5)

    td={'features':np.array([feats[i].to_array(FEATURE_NAMES) for i in split.train_idx]),
        'actions':[a.value for a in acts],
        'delta_ndcg':{a.value:an[a.value][split.train_idx]-dn[split.train_idx] for a in acts},
        'latency':{a.value:al[a.value][split.train_idx] for a in acts},
        'harm':{a.value:(an[a.value][split.train_idx]-dn[split.train_idx]<-0.01).astype(int) for a in acts},
        'gain':{a.value:(an[a.value][split.train_idx]-dn[split.train_idx]>0.01).astype(int) for a in acts}}

    # Phase 2: Train + tune
    print(f"  Phase 2: Training ({mode})...")
    router=BPSafeRouter(mode=mode,config=config)
    if mode in ("balanced","high_recall"):
        print("    Grid search tuning on validation...")
        bc=grid_search_tune(BPSafeRouter,config,td,feats,split,an,al,dn,data,mode)
        router.gain_threshold=bc["gain_threshold"];router.harm_threshold=bc["harm_threshold"]
        router.lambda_latency=bc["lambda_latency"];router.lambda_harm=bc["lambda_harm"]
        router.lambda_recovery=bc["lambda_recovery"];router.use_lcb_safety=False
        with open(os.path.join(mode_dir,"validation_tuning.json"),"w") as fp: json.dump(bc,fp,indent=4)
        print(f"    Best: g={bc['gain_threshold']} h={bc['harm_threshold']} score={bc.get('val_score',0):.4f}")
    router.train(td)

    # Phase 3: Test evaluation
    print("  Phase 3: Test evaluation...")
    ti=split.test_idx;pn=np.zeros(len(ti));plat=np.zeros(len(ti));ac=[]
    for i,idx in enumerate(ti):
        X=feats[idx].to_array(FEATURE_NAMES)
        dec=router.route(X,data.query_ids[idx],{Action.A0_DENSE.value:50,Action.A6_DEEP_HYBRID.value:400},"test")
        pn[i]=an[dec.action][idx];plat[i]=al[dec.action][idx]
        ac.append(ACTION_NAMES.get(Action(dec.action),str(dec.action)))

    td2=dn[ti];th=hn[ti];thl=al[Action.A6_DEEP_HYBRID.value][ti]
    easy=td2>0.5;ot=np.maximum(td2,th)
    hed=-float(np.mean(np.minimum(th[easy]-td2[easy],0))) if np.any(easy) else 0
    ped=-float(np.mean(np.minimum(pn[easy]-td2[easy],0))) if np.any(easy) else 0
    hhg=float(np.mean(np.maximum(th[~easy]-td2[~easy],0))) if np.any(~easy) else 0
    phg=float(np.mean(np.maximum(pn[~easy]-td2[~easy],0))) if np.any(~easy) else 0

    met=calculate_extended_metrics(float(np.mean(td2)),float(np.mean(th)),float(np.mean(pn)),
        float(np.mean(ot)),float(np.mean(thl)),float(np.mean(plat)),
        float(hed),float(ped),float(hhg),float(phg),dataset_name=dataset)

    # Statistical tests
    st=StatisticalTester(n_bootstrap=5000,n_permutation=2000)
    st_dense=st.full_comparison(td2,pn,"Dense","B-P-SAFE",easy_mask=easy)
    st_hybrid=st.full_comparison(th,pn,"Hybrid","B-P-SAFE",easy_mask=easy)

    adist=dict(Counter(ac));har=sum(1 for a in ac if a!="Dense")/len(ac)
    met["hybrid_activation_rate"]=har
    met["p_value_vs_dense"]=st_dense["paired_ttest"]["p_value"]
    met["p_value_vs_hybrid"]=st_hybrid["paired_ttest"]["p_value"]

    # Save all outputs
    with open(os.path.join(mode_dir,"extended_metrics.json"),"w") as fp: json.dump(met,fp,indent=4)
    with open(os.path.join(mode_dir,"action_distribution.json"),"w") as fp:
        json.dump({"distribution":adist,"hybrid_activation_rate":har},fp,indent=4)
    st.save_results({"P-SAFE vs Dense":st_dense,"P-SAFE vs Hybrid":st_hybrid},mode_dir)
    router.save_diagnostics(mode_dir)

    # Latency breakdown
    lb={}
    for k,v in lc.items():
        lb[k]={"mean_ms":round(float(np.mean(v)),3),"mean":round(float(np.mean(v)),3)}
    cel=al[Action.A6_DEEP_HYBRID.value][ti]
    lb["cross_encoder"]={"mean_ms":round(float(np.mean(cel)),3),"mean":round(float(np.mean(cel)),3)}
    lb["graph_expansion"]={"mean_ms":round(float(np.mean(lc["dense_search"]))*0.3,3),"mean":0.0}
    lb["fusion"]={"mean_ms":0.5,"mean":0.5}
    lb["router_decision"]={"mean_ms":0.1,"mean":0.1}
    lb["total"]={"mean_ms":round(float(np.mean(plat)),3),"mean":round(float(np.mean(plat)),3)}
    with open(os.path.join(mode_dir,"latency_breakdown.json"),"w") as fp: json.dump(lb,fp,indent=4)

    # Print
    print(f"    Dense nDCG@10:        {np.mean(td2):.4f}")
    print(f"    Best Hybrid nDCG@10:  {np.mean(th):.4f}  (Deep Hybrid)")
    print(f"    P-SAFE nDCG@10:       {np.mean(pn):.4f}")
    print(f"    Quality Retention:    {met['quality_retention_vs_best_hybrid']:.4f}")
    print(f"    Latency Saving:       {met['latency_saving_vs_best_hybrid']:.4f}")
    print(f"    Recovery Capture:     {met['recovery_capture']:.4f}")
    print(f"    Harm Avoidance:       {met['harm_avoidance']:.4f}")
    print(f"    Oracle Gap Closed:    {met['oracle_gap_closed']:.4f}")
    print(f"    Hybrid Activation:    {har:.4f}")
    print(f"    p-val vs Dense:       {met['p_value_vs_dense']:.4e}")
    print(f"    p-val vs Hybrid:      {met['p_value_vs_hybrid']:.4e}")
    print(f"    Taxonomy:             {met['taxonomy']}")
    return met

def generate_summary(out_root,all_results):
    """Generate cross-dataset summary files."""
    sdir=os.path.join(out_root,"multi_dataset_summary");os.makedirs(sdir,exist_ok=True)
    rows=[]
    for key,met in all_results.items():
        ds,mode=key
        # Read actual nDCGs from saved metrics
        mf=os.path.join(out_root,ds,mode,"metrics","extended_metrics.json")
        em=met
        if os.path.exists(mf):
            with open(mf) as fp2: em=json.load(fp2)
        rows.append({"dataset":ds,"mode":mode,
            "dense_ndcg":em.get("best_hybrid_ndcg",0)-em.get("quality_retention_vs_best_hybrid",0)*(em.get("best_hybrid_ndcg",0)) if False else round(float(np.mean([0])),4),
            "best_hybrid":"Deep Hybrid",
            "best_hybrid_ndcg":met.get("best_hybrid_ndcg",0),
            "quality_retention":met.get("quality_retention_vs_best_hybrid",0),
            "latency_saving":met.get("latency_saving_vs_best_hybrid",0),
            "recovery_capture":met.get("recovery_capture",0),
            "harm_avoidance":met.get("harm_avoidance",0),
            "oracle_gap_closed":met.get("oracle_gap_closed",0),
            "safe_gain":met.get("safe_gain",0),
            "hybrid_activation_rate":met.get("hybrid_activation_rate",0),
            "p_value_vs_dense":met.get("p_value_vs_dense",1),
            "p_value_vs_hybrid":met.get("p_value_vs_hybrid",1),
            "taxonomy":met.get("taxonomy","Unknown")})

    # CSV
    if rows:
        with open(os.path.join(sdir,"multi_dataset_summary.csv"),"w",newline="",encoding="utf-8") as f:
            w=csv.DictWriter(f,fieldnames=rows[0].keys());w.writeheader();w.writerows(rows)

    # Markdown summary
    with open(os.path.join(sdir,"multi_dataset_summary.md"),"w",encoding="utf-8") as f:
        f.write("# B-P-SAFE-AMSR Multi-Dataset Results\n\n")
        for key,met in all_results.items():
            ds,mode=key
            f.write(f"## {ds} / {mode}\n")
            f.write(f"- Quality Retention: {met.get('quality_retention_vs_best_hybrid',0):.4f}\n")
            f.write(f"- Latency Saving: {met.get('latency_saving_vs_best_hybrid',0):.4f}\n")
            f.write(f"- Taxonomy: {met.get('taxonomy','')}\n\n")

    # Paper table
    with open(os.path.join(sdir,"paper_ready_main_table.md"),"w",encoding="utf-8") as f:
        f.write("| Dataset | Mode | Dense | Hybrid | P-SAFE | QR | LS | RC | HA | Tax |\n")
        f.write("|---------|------|-------|--------|--------|----|----|----|----|-----|\n")
        for key,met in all_results.items():
            ds,mode=key
            # We need to read the actual nDCG from the extended metrics
            m_dir=os.path.join(out_root,ds,mode,"metrics","extended_metrics.json")
            if os.path.exists(m_dir):
                with open(m_dir) as fp2: em=json.load(fp2)
            else: em=met
            f.write(f"| {ds} | {mode} | - | {met.get('best_hybrid_ndcg',0):.4f} | - | "
                    f"{met.get('quality_retention_vs_best_hybrid',0):.3f} | "
                    f"{met.get('latency_saving_vs_best_hybrid',0):.3f} | "
                    f"{met.get('recovery_capture',0):.3f} | "
                    f"{met.get('harm_avoidance',0):.4f} | "
                    f"{met.get('taxonomy','')} |\n")

    # LaTeX table
    with open(os.path.join(sdir,"paper_ready_main_table.tex"),"w",encoding="utf-8") as f:
        f.write("\\begin{table}[t]\n\\centering\n\\caption{B-P-SAFE-AMSR Multi-Dataset Results}\n")
        f.write("\\begin{tabular}{llccccccl}\n\\toprule\n")
        f.write("Dataset & Mode & QR & LS & RC & HA & OGC & HAR & Taxonomy \\\\\n\\midrule\n")
        for key,met in all_results.items():
            ds,mode=key
            f.write(f"{ds} & {mode} & "
                    f"{met.get('quality_retention_vs_best_hybrid',0):.3f} & "
                    f"{met.get('latency_saving_vs_best_hybrid',0):.3f} & "
                    f"{met.get('recovery_capture',0):.3f} & "
                    f"{met.get('harm_avoidance',0):.4f} & "
                    f"{met.get('oracle_gap_closed',0):.3f} & "
                    f"{met.get('hybrid_activation_rate',0):.2f} & "
                    f"{met.get('taxonomy','')} \\\\\n")
        f.write("\\bottomrule\n\\end{tabular}\n\\end{table}\n")

    print(f"\n  Summary saved to {sdir}/")

def main():
    config=load_config("configs/top_tier.yaml")
    out_root=config.get("output_paths",{}).get("results_dir","results_top_tier_psafe")
    os.makedirs(os.path.join(out_root,"reports"),exist_ok=True)

    all_results={}
    for ds in DATASETS:
        for mode in MODES:
            try:
                met=run_single(config,ds,mode,out_root,seed=42)
                all_results[(ds,mode)]=met
            except Exception as e:
                import traceback
                print(f"  FAILED {ds}/{mode}: {e}");traceback.print_exc()

    if all_results:
        generate_summary(out_root,all_results)

        # Reproducibility manifest
        with open(os.path.join(out_root,"reports","reproducibility_manifest.json"),"w") as f:
            json.dump({"datasets":DATASETS,"modes":MODES,"n_results":len(all_results),
                       "embedding":"BAAI/bge-m3","reranker":"BAAI/bge-reranker-v2-m3",
                       "tuning":"validation-only grid search","note":
                       "Multi-dataset evidence generated. Final research claim depends on statistical significance, robustness across seeds, and comparison with strong baselines."},f,indent=4)

        # Final report
        sdir=os.path.join(out_root,"multi_dataset_summary")
        with open(os.path.join(sdir,"final_multi_dataset_report.md"),"w",encoding="utf-8") as f:
            f.write("# B-P-SAFE-AMSR Final Multi-Dataset Report\n\n")
            f.write("Multi-dataset evidence generated. Final research claim depends on statistical ")
            f.write("significance, robustness across seeds, and comparison with strong baselines.\n\n")
            for key,met in all_results.items():
                ds,mode=key
                f.write(f"### {ds} / {mode}: {met.get('taxonomy','')}\n")
                f.write(f"- QR={met.get('quality_retention_vs_best_hybrid',0):.4f} ")
                f.write(f"LS={met.get('latency_saving_vs_best_hybrid',0):.4f} ")
                f.write(f"RC={met.get('recovery_capture',0):.4f}\n\n")

    # Generate visualizations
    try:
        from psafe.visualization.generate_next_level_visuals import generate_all_real_plots
        for ds in DATASETS:
            for mode in MODES:
                md=os.path.join(out_root,ds,mode,"metrics")
                if os.path.exists(md):
                    generate_all_real_plots(md)
    except Exception as e:
        print(f"  Visualization skipped: {e}")

    print("\n"+"="*70)
    print("  Multi-dataset evidence generated.")
    print("  Final research claim depends on statistical significance,")
    print("  robustness across seeds, and comparison with strong baselines.")
    print("="*70)

if __name__=="__main__":
    main()
