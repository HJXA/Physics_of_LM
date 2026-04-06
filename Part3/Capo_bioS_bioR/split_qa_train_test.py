import argparse
from pathlib import Path

import pandas as pd


def split_single_dataset(src_file: Path, train_file: Path, test_file: Path, seed: int) -> None:
    df = pd.read_parquet(src_file)

    if len(df) == 0:
        raise ValueError(f"Empty dataset: {src_file}")

    # 不打乱，直接取前一半为 train，其余为 test
    df = df.reset_index(drop=True)
    split_idx = len(df) // 2

    train_df = df.iloc[:split_idx]
    test_df = df.iloc[split_idx:]

    train_df.to_parquet(train_file, index=False)
    test_df.to_parquet(test_file, index=False)

    print(
        f"[DONE] {src_file.parent.name}: total={len(df)}, "
        f"train={len(train_df)}, test={len(test_df)}"
    )

    return len(train_df), len(test_df)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Split each QA parquet into 50/50 train and test sets."
    )
    parser.add_argument(
        "--qa-root",
        type=str,
        default="/ruilab/jxhe/CoE_Monitor/Physics_of_LM/Part3/datasets/QA",
        help="Root directory of QA datasets (contains q*/data.parquet).",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed for shuffling before split.",
    )
    args = parser.parse_args()

    qa_root = Path(args.qa_root)
    train_dir = qa_root / "train"
    test_dir = qa_root / "test"
    train_dir.mkdir(parents=True, exist_ok=True)
    test_dir.mkdir(parents=True, exist_ok=True)

    dataset_dirs = sorted(
        p for p in qa_root.iterdir()
        if p.is_dir() and p.name.startswith("q") and p.name not in {"train", "test"}
    )

    if not dataset_dirs:
        raise ValueError(f"No dataset folders found under: {qa_root}")

    total_train = 0
    total_test = 0
    per_dataset_counts = {}

    for d in dataset_dirs:
        src_file = d / "data.parquet"
        if not src_file.exists():
            print(f"[SKIP] Missing file: {src_file}")
            continue

        out_name = f"{d.name}.parquet"
        train_file = train_dir / out_name
        test_file = test_dir / out_name

        t_count, v_count = split_single_dataset(src_file, train_file, test_file, seed=args.seed)
        total_train += t_count
        total_test += v_count
        per_dataset_counts[d.name] = (t_count, v_count)

    print(f"\nOutput train dir: {train_dir}")
    print(f"Output test dir:  {test_dir}")

    print("\nPer-dataset counts:")
    for name, (tc, vc) in per_dataset_counts.items():
        print(f" - {name}: train={tc}, test={vc}")

    print(f"\nTotal samples: train={total_train}, test={total_test}")


if __name__ == "__main__":
    main()
