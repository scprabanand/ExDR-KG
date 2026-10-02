"""Step 2 (model 5): TxGNN official implementation, scored on our test pairs.

The manuscript says every graph baseline runs on the same unified HKG. The official TxGNN code (data/raw/txgnn) is built
around its own PrimeKG-derived data (TxData), so it cannot read unified_kg.pt directly. This script therefore does two
things and records both, so the manuscript can say exactly what was done:

  1. TxGNN-official: train the official model with the official pipeline on its own KG, using OUR split as the
     train/valid/test drug-disease indications (mapped through --id-map), and score OUR test pairs.
  2. Report coverage: test pairs whose drug or disease has no PrimeKG counterpart cannot be scored by the official
     model; they receive the training-set base rate and are counted in <name>.meta.json (report this number).

--id-map is a CSV with columns: drug_id (DrugBank), disease_id (DOID), txgnn_drug (PrimeKG drug id), txgnn_disease (PrimeKG disease id).
Build it from the crosswalks shipped with TxGNN (data/raw/txgnn/txgnn/data_splits/mondo_references.csv and DrugBank ids in
PrimeKG's drug nodes); pairs without a mapping are listed in the coverage report.

Usage
  python -m exdrkg_rerun.baselines.txgnn --root . --split standard_degree_matched --id-map data/txgnn_id_map.csv --txgnn-data data/raw/txgnn/data
Official hyper-parameters (TxGNN paper / repository README): n_hid=100, n_inp=100, n_out=100, proto=True, proto_num=3, sim_measure='all_nodes_profile', bert_measure='disease_name', agg_measure='rarity', num_walks=200, walk_mode='bit', path_length=2; 500 pretrain epochs (n_epoch=2, learning_rate=1e-3 for pretraining; finetune 500 epochs, lr 5e-4).
"""
import argparse, json, sys
from pathlib import Path
import numpy as np, pandas as pd

sys.path.insert(0, "data/raw/txgnn")          # official package shipped in the repository
from txgnn import TxData, TxGNN               # noqa: E402


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--root", default="."); ap.add_argument("--split", default="standard")
    ap.add_argument("--id-map", required=True); ap.add_argument("--txgnn-data", required=True); ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--device", default="cuda:0"); ap.add_argument("--pretrain-epochs", type=int, default=500); ap.add_argument("--finetune-epochs", type=int, default=500)
    a = ap.parse_args(); root = Path(a.root)
    idmap = pd.read_csv(a.id_map); sp = {k: pd.read_csv(root / "data/splits" / a.split / f"{k}.csv") for k in ("train", "val", "test")}
    mapped = idmap.set_index(["drug_id", "disease_id"])
    def cover(df): return df.join(mapped[["txgnn_drug", "txgnn_disease"]], on=["drug_id", "disease_id"])
    tr, va, te = cover(sp["train"]), cover(sp["val"]), cover(sp["test"])
    covered = te.txgnn_drug.notna() & te.txgnn_disease.notna()
    data = TxData(data_folder_path=a.txgnn_data); data.prepare_split(split="random", seed=a.seed)
    # replace the random split with ours: positives only are drug-disease indications in TxGNN's format (x_idx, y_idx)
    def pos(df): d = df[(df.label == 1) & df.txgnn_drug.notna() & df.txgnn_disease.notna()]; return d[["txgnn_drug", "txgnn_disease"]].rename(columns={"txgnn_drug": "x_id", "txgnn_disease": "y_id"})
    data.df_train, data.df_valid, data.df_test = pos(tr), pos(va), pos(te)
    model = TxGNN(data=data, weight_bias_track=False, proj_name="ExDR-KG-reviewer", exp_name=f"txgnn_{a.split}", device=a.device)
    model.model_initialize(n_hid=100, n_inp=100, n_out=100, proto=True, proto_num=3, attention=False, sim_measure="all_nodes_profile",
                           bert_measure="disease_name", agg_measure="rarity", num_walks=200, walk_mode="bit", path_length=2)
    model.pretrain(n_epoch=a.pretrain_epochs, learning_rate=1e-3, batch_size=1024, train_print_per_n=20)
    model.finetune(n_epoch=a.finetune_epochs, learning_rate=5e-4, train_print_per_n=5, valid_per_n=20)
    # score our test pairs with the fine-tuned model; TxGNN exposes per-pair drug-disease scores via its predict helper
    base_rate = float(sp["train"].label.mean()); scores = np.full(len(te), base_rate)
    idx = np.where(covered.values)[0]
    preds = model.predict(te.loc[idx, ["txgnn_drug", "txgnn_disease"]].rename(columns={"txgnn_drug": "x_id", "txgnn_disease": "y_id"}))
    scores[idx] = np.asarray(preds, dtype=float)
    out = root / "results/rerun/main/scores"; out.mkdir(parents=True, exist_ok=True)
    pd.DataFrame({"drug_id": te.drug_id, "disease_id": te.disease_id, "label": te.label, "score": scores}).to_csv(out / "TxGNN.csv", index=False)
    json.dump({"pairs": len(te), "covered": int(covered.sum()), "uncovered_given_base_rate": int((~covered).sum()), "base_rate": base_rate,
               "positives_covered": int(te[covered].label.sum()), "positives_total": int(te.label.sum())}, open(out / "TxGNN.meta.json", "w"), indent=1)
    print("TxGNN scored;", int(covered.sum()), "of", len(te), "test pairs mapped to the official KG")


if __name__ == "__main__":
    main()
