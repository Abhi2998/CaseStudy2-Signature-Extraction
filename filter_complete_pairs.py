import csv
from pathlib import Path

# CHANGE THESE if your files are elsewhere
IN_CSV = Path(r"C:\Users\Dell\Desktop\Case Study 2\manifest_pairs.csv")
OUT_CSV = Path(r"C:\Users\Dell\Desktop\Case Study 2\manifest_pairs_complete.csv")

def to_bool(x: str) -> bool:
    # your CSV contains True/False strings
    return str(x).strip().lower() in {"true", "1", "yes", "y"}

def main():
    kept = 0
    dropped = 0

    with IN_CSV.open("r", newline="", encoding="utf-8") as f_in, \
         OUT_CSV.open("w", newline="", encoding="utf-8") as f_out:

        reader = csv.DictReader(f_in)
        if reader.fieldnames is None:
            raise RuntimeError("Input CSV has no header row.")

        writer = csv.DictWriter(f_out, fieldnames=reader.fieldnames)
        writer.writeheader()

        for row in reader:
            has_input = to_bool(row.get("has_input", "False"))
            has_target = to_bool(row.get("has_target", "False"))

            if has_input and has_target:
                writer.writerow(row)
                kept += 1
            else:
                dropped += 1

    print(f"[filter] input:  {IN_CSV}")
    print(f"[filter] output: {OUT_CSV}")
    print(f"[filter] kept complete pairs: {kept}")
    print(f"[filter] dropped incomplete rows: {dropped}")

if __name__ == "__main__":
    main()
