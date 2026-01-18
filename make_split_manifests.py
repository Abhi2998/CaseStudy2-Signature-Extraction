import csv
import json
from pathlib import Path

COMPLETE_CSV = Path(r"C:\Users\Dell\Desktop\Case Study 2\manifest_pairs_complete.csv")
SPLITS_JSON  = Path(r"C:\Users\Dell\Desktop\Case Study 2\splits.json")

OUT_DIR = Path(r"C:\Users\Dell\Desktop\Case Study 2\split_manifests")
OUT_DIR.mkdir(parents=True, exist_ok=True)

def load_splits(path: Path):
    data = json.loads(path.read_text(encoding="utf-8"))
    return set(data["train_groups"]), set(data["val_groups"]), set(data["test_groups"])

def main():
    train_groups, val_groups, test_groups = load_splits(SPLITS_JSON)

    out_train = (OUT_DIR / "train_pairs.csv").open("w", newline="", encoding="utf-8")
    out_val   = (OUT_DIR / "val_pairs.csv").open("w", newline="", encoding="utf-8")
    out_test  = (OUT_DIR / "test_pairs.csv").open("w", newline="", encoding="utf-8")

    with COMPLETE_CSV.open("r", newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        if not reader.fieldnames:
            raise RuntimeError("Complete CSV has no header row.")

        w_train = csv.DictWriter(out_train, fieldnames=reader.fieldnames); w_train.writeheader()
        w_val   = csv.DictWriter(out_val,   fieldnames=reader.fieldnames); w_val.writeheader()
        w_test  = csv.DictWriter(out_test,  fieldnames=reader.fieldnames); w_test.writeheader()

        n_train = n_val = n_test = 0

        for row in reader:
            g = row["group_id"]
            if g in train_groups:
                w_train.writerow(row); n_train += 1
            elif g in val_groups:
                w_val.writerow(row); n_val += 1
            elif g in test_groups:
                w_test.writerow(row); n_test += 1
            else:
                # group not in splits.json => should not happen
                pass

    out_train.close(); out_val.close(); out_test.close()

    print("[split_manifests] wrote:", OUT_DIR)
    print("[split_manifests] train rows:", n_train)
    print("[split_manifests] val rows:", n_val)
    print("[split_manifests] test rows:", n_test)
    print("[split_manifests] total:", n_train + n_val + n_test)

if __name__ == "__main__":
    main()
