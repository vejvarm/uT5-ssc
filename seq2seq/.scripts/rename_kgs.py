import argparse
from pathlib import Path

dbid_to_filename = {
    "allergies":           "allergy",
    "careplans":           "careplan",
    "claims":              "claim",
    "claims_transactions": "claim_transaction",
    "conditions":          "condition",
    "devices":             "device",
    "encounters":          "encounter",
    "imaging_studies":     "imaging_study",
    "immunizations":       "immunization",
    "medications":         "medication",
    "observations":        "observation",
    "organizations":       "organization",
    "patients":            "patient",
    "patient_expenses":   "patient_expenses",
    "payers":              "payer",
    "payer_transitions":  "payer_transition",
    "procedures":          "procedure",
    "providers":           "provider",
    "supplies":            "supply",
}

def main():
    parser = argparse.ArgumentParser(description="Rename Synthea KGS files to plural db_id names.")
    parser.add_argument("data_dir", type=Path, help="Directory containing the files to rename")
    parser.add_argument("--dry-run", action="store_true", help="Only show what would be renamed")
    args = parser.parse_args()

    for dbid, singular in dbid_to_filename.items():
        for ext in [".csv", ".ttl"]:
            src = args.data_dir / f"{singular}{ext}"
            dst = args.data_dir / f"{dbid}{ext}"
            if src.exists() and not dst.exists():
                if args.dry_run:
                    print(f"Would rename: {src} -> {dst}")
                else:
                    print(f"Renaming: {src} -> {dst}")
                    src.rename(dst)
            elif not src.exists():
                print(f"Source missing: {src}")
            elif dst.exists():
                print(f"Target already exists: {dst}")

if __name__ == "__main__":
    main()
