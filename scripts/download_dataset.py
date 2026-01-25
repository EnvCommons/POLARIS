"""
Download and convert POLARIS dataset to local parquet format.

This script downloads the POLARIS-Dataset-53K from HuggingFace and
converts it to a local parquet file for use by the environment.

Run once before starting the environment server:
    python scripts/download_dataset.py

Output: data/polaris_tasks.parquet (~12 MB)
"""

import datasets
from pathlib import Path
import sys


def download_polaris_dataset():
    """Download and convert POLARIS dataset"""

    print("="*60)
    print("POLARIS Dataset Download Script")
    print("="*60)

    # Download from HuggingFace
    print("\n[1/4] Downloading POLARIS-Dataset-53K from HuggingFace...")
    print("      This may take a few minutes on first run...")

    try:
        dataset = datasets.load_dataset(
            "POLARIS-Project/Polaris-Dataset-53K",
            streaming=False
        )
    except Exception as e:
        print(f"\n❌ Error downloading dataset: {e}")
        print("\nTroubleshooting:")
        print("  - Check internet connection")
        print("  - Verify HuggingFace datasets library is installed: pip install datasets")
        print("  - Try clearing cache: rm -rf ~/.cache/huggingface/datasets")
        sys.exit(1)

    # Get train split
    print("[2/4] Extracting train split...")
    train_data = dataset["train"]

    # Convert to pandas
    print("[3/4] Converting to pandas DataFrame...")
    df = train_data.to_pandas()

    # Validate data structure
    expected_columns = {"problem", "answer", "difficulty"}
    actual_columns = set(df.columns)

    if not expected_columns.issubset(actual_columns):
        print(f"\n⚠️  Warning: Expected columns {expected_columns}, got {actual_columns}")

    # Create data directory
    script_dir = Path(__file__).parent
    data_dir = script_dir.parent / "data"
    data_dir.mkdir(exist_ok=True)

    # Save to parquet
    print("[4/4] Saving to parquet...")
    output_path = data_dir / "polaris_tasks.parquet"
    df.to_parquet(output_path, index=False)

    # Report success
    file_size_mb = output_path.stat().st_size / (1024 * 1024)

    print("\n" + "="*60)
    print("✅ Dataset Download Complete!")
    print("="*60)
    print(f"Output file: {output_path}")
    print(f"File size:   {file_size_mb:.2f} MB")
    print(f"Total tasks: {len(df):,}")
    print(f"Columns:     {list(df.columns)}")
    print(f"\nDifficulty distribution:")
    print(df["difficulty"].value_counts().to_string())

    print(f"\n📝 Sample task:")
    print("-" * 60)
    sample = df.iloc[0]
    print(f"Problem:    {sample['problem'][:200]}...")
    print(f"Answer:     {sample['answer']}")
    print(f"Difficulty: {sample['difficulty']}")
    print("-" * 60)

    print("\n✅ Ready to start server: python server.py")


if __name__ == "__main__":
    download_polaris_dataset()
