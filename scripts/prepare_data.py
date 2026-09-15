from pathlib import Path
from datasets import load_dataset


# ------------------------------------------------------------
# Paths
# ------------------------------------------------------------

OUTPUT_DIR = Path(__file__).parent.parent / "data" / "processed"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


# ------------------------------------------------------------
# Load IWSLT17 En-De
# ------------------------------------------------------------

base_url = (
    r"https://huggingface.co/datasets/IWSLT/iwslt2017/"
    r"resolve/refs%2Fconvert%2Fparquet/iwslt2017-en-de/"
)

data_files = {
    "train": base_url + "train/0000.parquet",
    "validation": base_url + "validation/0000.parquet",
    "test": base_url + "test/0000.parquet"
}

dataset = load_dataset(
    "parquet",
    data_files=data_files,
)


# ------------------------------------------------------------
# Basic filtering
# ------------------------------------------------------------

def is_valid_pair(src: str, tgt: str) -> bool:
    if src is None or tgt is None:
        return False

    src = src.strip()
    tgt = tgt.strip()

    if len(src) == 0 or len(tgt) == 0:
        return False

    return True


# ------------------------------------------------------------
# Save one dataset split
# ------------------------------------------------------------

def save_split(split_name: str):
    split = dataset[split_name]

    src_path = OUTPUT_DIR / f"{split_name}.en"
    tgt_path = OUTPUT_DIR / f"{split_name}.de"

    num_saved = 0

    with (
        src_path.open("w", encoding="utf-8") as src_file,
        tgt_path.open("w", encoding="utf-8") as tgt_file
    ):
        for exmaple in split:
            translation = exmaple["translation"]

            src = translation["en"]
            tgt = translation["de"]

            if not is_valid_pair(src, tgt):
                continue

            src = src.strip()
            tgt = tgt.strip()

            src_file.write(src + "\n")
            tgt_file.write(tgt + "\n")

            num_saved += 1

    print(f"{split_name}: {num_saved} sentence pairs saved.")


# ------------------------------------------------------------
# Main
# ------------------------------------------------------------

def main():
    print(dataset)

    for split_name in [
        "train",
        "validation",
        "test"
    ]:
        save_split(split_name)

    print("\nData preparation completed.")
    print(f"Saved to {OUTPUT_DIR.resolve()}")


if __name__ == "__main__":
    main()