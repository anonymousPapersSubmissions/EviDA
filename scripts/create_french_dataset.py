# """
# Create synthetic French dataset by translating twitter + fakeddit to French.

# Uses Helsinki-NLP/opus-mt-en-fr — free, runs fully offline on the cluster,
# no API key needed.

# Usage
# -----
# # Translate both sources, auto-detect text column
# python scripts/create_french_dataset.py --data_dir data --sources twitter fakeddit

# # Limit rows per split (useful for a quick test)
# python scripts/create_french_dataset.py --data_dir data --sources twitter fakeddit --max_rows 500

# # Use GPU for faster translation
# python scripts/create_french_dataset.py --data_dir data --sources twitter fakeddit --device cuda

# Output
# ------
# data/french_synthetic/
#     train.csv
#     val.csv
#     test.csv   (if source test splits exist)

# The images are NOT copied — the french_synthetic source reuses the original
# image files from twitter/ and fakeddit/ via a new column `image_source` that
# records which original dataset each image came from.  Your FlexibleImageLoader
# and dataset code will need a small update to honour this column; see the
# NOTE at the bottom of this file.
# """

# import sys
# import argparse
# import logging
# import shutil
# from pathlib import Path

# import pandas as pd
# import torch
# from tqdm import tqdm
# from transformers import MarianMTModel, MarianTokenizer

# sys.path.append(str(Path(__file__).parent.parent))
# from src.utils.logger import setup_logger

# logger = setup_logger('french_dataset', 'logs/french_dataset.log')

# # Helsinki-NLP English → French model
# MODEL_NAME = 'Helsinki-NLP/opus-mt-en-fr'


# # ─────────────────────────────────────────────────────────────────────────────
# # Translation helpers
# # ─────────────────────────────────────────────────────────────────────────────

# def load_translator(device: str):
#     """Load Helsinki-NLP EN→FR model and tokenizer."""
#     logger.info(f'Loading translation model: {MODEL_NAME}')
#     tokenizer = MarianTokenizer.from_pretrained(MODEL_NAME)
#     model     = MarianMTModel.from_pretrained(MODEL_NAME).to(device)
#     model.eval()
#     logger.info(f'  Model loaded on {device}')
#     return tokenizer, model


# def translate_batch(texts: list[str], tokenizer, model, device: str, max_length: int = 512) -> list[str]:
#     """Translate a batch of strings from English to French."""
#     # MarianMT expects a list of strings
#     inputs = tokenizer(
#         texts,
#         return_tensors  = 'pt',
#         padding         = True,
#         truncation      = True,
#         max_length      = max_length,
#     ).to(device)

#     with torch.no_grad():
#         translated = model.generate(**inputs, max_length=max_length)

#     return [tokenizer.decode(t, skip_special_tokens=True) for t in translated]


# def translate_column(
#     series: pd.Series,
#     tokenizer,
#     model,
#     device: str,
#     batch_size: int = 16,
#     max_length: int = 512,
# ) -> pd.Series:
#     """Translate an entire DataFrame column in batches."""
#     texts     = series.fillna('').astype(str).tolist()
#     results   = []
#     n_batches = (len(texts) + batch_size - 1) // batch_size

#     for i in tqdm(range(n_batches), desc='  Translating', unit='batch'):
#         batch = texts[i * batch_size : (i + 1) * batch_size]
#         results.extend(translate_batch(batch, tokenizer, model, device, max_length))

#     return pd.Series(results, index=series.index)


# # ─────────────────────────────────────────────────────────────────────────────
# # Column detection
# # ─────────────────────────────────────────────────────────────────────────────

# def detect_text_column(df: pd.DataFrame, source: str) -> str | None:
#     """Return the name of the primary text column."""
#     candidates = [col for col in df.columns
#                   if any(t in col.lower() for t in ['text', 'content', 'tweet', 'title'])
#                   and 'id' not in col.lower()]
#     if not candidates:
#         logger.error(f'  [{source}] No text column found. Columns: {list(df.columns)}')
#         return None
#     if len(candidates) > 1:
#         logger.warning(f'  [{source}] Multiple text columns found: {candidates}. Using "{candidates[0]}".')
#     return candidates[0]


# def detect_label_column(df: pd.DataFrame, source: str) -> str | None:
#     """Return the name of the label column."""
#     candidates = [col for col in df.columns
#                   if any(t in col.lower() for t in ['label', 'class', 'category'])]
#     if not candidates:
#         logger.error(f'  [{source}] No label column found. Columns: {list(df.columns)}')
#         return None
#     return candidates[0]


# def detect_image_column(df: pd.DataFrame, source: str) -> str | None:
#     """Return the name of the image-id column."""
#     if 'image_id' in df.columns:
#         return 'image_id'
#     if 'image' in df.columns:
#         return 'image'
#     candidates = [col for col in df.columns
#                   if any(t in col.lower() for t in ['img', 'photo', 'pic'])]
#     if not candidates:
#         logger.warning(f'  [{source}] No image column found — image_id will be empty.')
#         return None
#     return candidates[0]


# # ─────────────────────────────────────────────────────────────────────────────
# # Per-source processing
# # ─────────────────────────────────────────────────────────────────────────────

# def process_source(
#     source: str,
#     data_dir: Path,
#     tokenizer,
#     model,
#     device: str,
#     splits: list[str],
#     max_rows: int | None,
#     batch_size: int,
# ) -> dict[str, pd.DataFrame]:
#     """Translate all available splits for one source. Returns {split: df}."""
#     source_dir = data_dir / source
#     result = {}

#     for split in splits:
#         csv_path = source_dir / f'{split}.csv'
#         if not csv_path.exists():
#             logger.warning(f'  [{source}] {split}.csv not found — skipping.')
#             continue

#         logger.info(f'\n[{source}] Processing {split} split...')
#         df = pd.read_csv(csv_path)

#         if max_rows:
#             df = df.head(max_rows)
#             logger.info(f'  Limiting to {max_rows} rows.')

#         text_col  = detect_text_column(df, source)
#         label_col = detect_label_column(df, source)
#         image_col = detect_image_column(df, source)

#         if text_col is None or label_col is None:
#             logger.error(f'  [{source}/{split}] Skipping — required columns missing.')
#             continue

#         logger.info(f'  Rows: {len(df)}  |  text="{text_col}"  label="{label_col}"'
#                     + (f'  image="{image_col}"' if image_col else ''))

#         # Translate
#         translated_text = translate_column(
#             df[text_col], tokenizer, model, device, batch_size
#         )

#         # Build output DataFrame with standardised column names
#         out = pd.DataFrame()
#         out['text']         = translated_text
#         out['label']        = df[label_col].values
#         out['image_id']     = df[image_col].values if image_col else ''
#         out['image_source'] = source          # which dataset owns the image file
#         out['original_text'] = df[text_col].values  # keep for reference

#         result[split] = out
#         logger.info(f'  [{source}/{split}] ✓ {len(out)} rows translated.')

#     return result


# # ─────────────────────────────────────────────────────────────────────────────
# # Main
# # ─────────────────────────────────────────────────────────────────────────────

# def main():
#     parser = argparse.ArgumentParser(
#         description='Create synthetic French dataset from English sources.',
#         formatter_class=argparse.ArgumentDefaultsHelpFormatter,
#     )
#     parser.add_argument('--data_dir',   type=str, default='data',
#                         help='Root data directory')
#     parser.add_argument('--sources',    nargs='+', default=['twitter', 'fakeddit'],
#                         help='Source datasets to translate')
#     parser.add_argument('--output',     type=str, default='french_synthetic',
#                         help='Output subdirectory name inside data_dir')
#     parser.add_argument('--splits',     nargs='+', default=['train', 'val', 'test'],
#                         help='Splits to process')
#     parser.add_argument('--max_rows',   type=int, default=None,
#                         help='Max rows per source/split (None = all)')
#     parser.add_argument('--batch_size', type=int, default=16,
#                         help='Translation batch size')
#     parser.add_argument('--device',     type=str,
#                         default='cuda' if torch.cuda.is_available() else 'cpu',
#                         help='Device for translation model')
#     parser.add_argument('--hf_cache',   type=str, default=None,
#                         help='Optional HuggingFace cache dir (avoids home quota)')
#     args = parser.parse_args()

#     # Redirect HF cache to scratch if specified (avoids home dir quota)
#     if args.hf_cache:
#         import os
#         os.environ['TRANSFORMERS_CACHE'] = args.hf_cache
#         os.environ['HF_HOME']            = args.hf_cache
#         logger.info(f'HF cache → {args.hf_cache}')

#     data_dir   = Path(args.data_dir)
#     output_dir = data_dir / args.output
#     output_dir.mkdir(parents=True, exist_ok=True)

#     logger.info('='*80)
#     logger.info('SYNTHETIC FRENCH DATASET CREATION')
#     logger.info('='*80)
#     logger.info(f'Sources:    {args.sources}')
#     logger.info(f'Splits:     {args.splits}')
#     logger.info(f'Output:     {output_dir}')
#     logger.info(f'Device:     {args.device}')
#     logger.info(f'Max rows:   {args.max_rows or "all"}')
#     logger.info(f'Batch size: {args.batch_size}')
#     logger.info('='*80 + '\n')

#     # Load translation model once
#     tokenizer, model = load_translator(args.device)

#     # Collect translated DataFrames per split
#     all_splits: dict[str, list[pd.DataFrame]] = {s: [] for s in args.splits}

#     for source in args.sources:
#         source_dfs = process_source(
#             source      = source,
#             data_dir    = data_dir,
#             tokenizer   = tokenizer,
#             model       = model,
#             device      = args.device,
#             splits      = args.splits,
#             max_rows    = args.max_rows,
#             batch_size  = args.batch_size,
#         )
#         for split, df in source_dfs.items():
#             all_splits[split].append(df)

#     # Merge sources per split and save
#     logger.info('\n' + '='*80)
#     logger.info('SAVING OUTPUT')
#     logger.info('='*80)

#     for split, dfs in all_splits.items():
#         if not dfs:
#             logger.warning(f'No data for split "{split}" — skipping.')
#             continue

#         merged = pd.concat(dfs, ignore_index=True)
#         # Shuffle so twitter and fakeddit rows are interleaved
#         merged = merged.sample(frac=1, random_state=42).reset_index(drop=True)

#         out_path = output_dir / f'{split}.csv'
#         merged.to_csv(out_path, index=False)

#         # Label distribution
#         dist = merged['label'].value_counts().to_dict()
#         logger.info(f'\n✓ {split}.csv → {len(merged)} rows saved to {out_path}')
#         logger.info(f'  Label distribution: {dist}')

#     logger.info('\n' + '='*80)
#     logger.info('DONE')
#     logger.info(f'French synthetic dataset saved to: {output_dir}')
#     logger.info("""
# NOTE — Image loading for french_synthetic
# -----------------------------------------
# Each row has two columns:
#   image_id     : the original filename/id (from twitter or fakeddit)
#   image_source : which dataset owns the image ("twitter" or "fakeddit")

# Your FlexibleImageLoader.get_image_path() currently takes (source, image_id, split).
# For french_synthetic rows you should pass image_source instead of "french_synthetic"
# so it looks in the correct original directory.

# Simplest fix in your Dataset __getitem__:
#   actual_source = row.get('image_source', self.source)
#   image_path = self.image_loader.get_image_path(actual_source, image_id, split)
# """)
#     logger.info('='*80 + '\n')


# if __name__ == '__main__':
#     main()


"""
Create synthetic French dataset by translating twitter + fakeddit to French.

Uses Helsinki-NLP/opus-mt-en-fr — free, runs fully offline on the cluster,
no API key needed.

Usage
-----
# Translate both sources and copy images (default)
python scripts/create_french_dataset.py --data_dir data --sources twitter fakeddit

# Limit rows per split (useful for a quick test)
python scripts/create_french_dataset.py --data_dir data --sources twitter fakeddit --max_rows 500

# Use GPU for faster translation
python scripts/create_french_dataset.py --data_dir data --sources twitter fakeddit --device cuda

# Skip image copying (e.g. disk space is tight)
python scripts/create_french_dataset.py --data_dir data --sources twitter fakeddit --no_copy_images

Output
------
data/french_synthetic/
    train.csv
    val.csv
    test.csv        (if source test splits exist)
    images/         all images copied from source datasets (flat, deduplicated)

Images from all translated sources are copied into a single flat
data/french_synthetic/images/ folder so no code changes are needed anywhere.
"""

import sys
import argparse
import logging
import shutil
from pathlib import Path

import pandas as pd
import torch
from tqdm import tqdm
from transformers import MarianMTModel, MarianTokenizer

sys.path.append(str(Path(__file__).parent.parent))
from src.utils.logger import setup_logger

logger = setup_logger('french_dataset', 'logs/french_dataset.log')

# Helsinki-NLP English → French model
MODEL_NAME = 'Helsinki-NLP/opus-mt-en-fr'

IMAGE_EXTENSIONS = {'.jpg', '.jpeg', '.png', '.gif', '.bmp', '.webp'}


# ─────────────────────────────────────────────────────────────────────────────
# Translation helpers
# ─────────────────────────────────────────────────────────────────────────────

def load_translator(device: str):
    """Load Helsinki-NLP EN→FR model and tokenizer."""
    logger.info(f'Loading translation model: {MODEL_NAME}')
    tokenizer = MarianTokenizer.from_pretrained(MODEL_NAME)
    model     = MarianMTModel.from_pretrained(MODEL_NAME).to(device)
    model.eval()
    logger.info(f'  Model loaded on {device}')
    return tokenizer, model


def translate_batch(texts: list, tokenizer, model, device: str, max_length: int = 512) -> list:
    """Translate a batch of strings from English to French."""
    inputs = tokenizer(
        texts,
        return_tensors = 'pt',
        padding        = True,
        truncation     = True,
        max_length     = max_length,
    ).to(device)

    with torch.no_grad():
        translated = model.generate(**inputs, max_length=max_length)

    return [tokenizer.decode(t, skip_special_tokens=True) for t in translated]


def translate_column(series, tokenizer, model, device, batch_size=16, max_length=512):
    """Translate an entire DataFrame column in batches."""
    texts     = series.fillna('').astype(str).tolist()
    results   = []
    n_batches = (len(texts) + batch_size - 1) // batch_size

    for i in tqdm(range(n_batches), desc='  Translating', unit='batch'):
        batch = texts[i * batch_size : (i + 1) * batch_size]
        results.extend(translate_batch(batch, tokenizer, model, device, max_length))

    return pd.Series(results, index=series.index)


# ─────────────────────────────────────────────────────────────────────────────
# Column detection
# ─────────────────────────────────────────────────────────────────────────────

def detect_text_column(df, source):
    candidates = [col for col in df.columns
                  if any(t in col.lower() for t in ['text', 'content', 'tweet', 'title'])
                  and 'id' not in col.lower()]
    if not candidates:
        logger.error(f'  [{source}] No text column found. Columns: {list(df.columns)}')
        return None
    if len(candidates) > 1:
        logger.warning(f'  [{source}] Multiple text columns: {candidates}. Using "{candidates[0]}".')
    return candidates[0]


def detect_label_column(df, source):
    candidates = [col for col in df.columns
                  if any(t in col.lower() for t in ['label', 'class', 'category'])]
    if not candidates:
        logger.error(f'  [{source}] No label column found. Columns: {list(df.columns)}')
        return None
    return candidates[0]


def detect_image_column(df, source):
    if 'image_id' in df.columns:
        return 'image_id'
    if 'image' in df.columns:
        return 'image'
    candidates = [col for col in df.columns
                  if any(t in col.lower() for t in ['img', 'photo', 'pic'])]
    if not candidates:
        logger.warning(f'  [{source}] No image column found — image_id will be empty.')
        return None
    return candidates[0]


# ─────────────────────────────────────────────────────────────────────────────
# Image copying
# ─────────────────────────────────────────────────────────────────────────────

def copy_images(sources: list, data_dir: Path, output_dir: Path) -> dict:
    """
    Copy all images from each source dataset into output_dir/images/.

    Files are copied flat (no subdirectories) and deduplicated by filename.
    If two sources share a filename, the first one wins and a warning is logged.
    Twitter and fakeddit use different naming conventions so collisions are rare.

    Returns {source: n_copied} summary dict.
    """
    images_dir = output_dir / 'images'
    images_dir.mkdir(parents=True, exist_ok=True)

    summary = {}
    seen    = {}   # basename -> source that first provided it

    for source in sources:
        source_dir = data_dir / source
        if not source_dir.exists():
            logger.warning(f'  [{source}] Source directory not found — skipping image copy.')
            summary[source] = 0
            continue

        n_copied  = 0
        n_skipped = 0

        for img_path in sorted(source_dir.rglob('*')):
            if img_path.suffix.lower() not in IMAGE_EXTENSIONS:
                continue

            dest = images_dir / img_path.name

            if dest.exists():
                n_skipped += 1
                if img_path.name not in seen:
                    logger.warning(
                        f'  [{source}] Collision: {img_path.name} already copied '
                        f'from {seen.get(img_path.name, "unknown")} — skipping.'
                    )
                continue

            shutil.copy2(img_path, dest)
            seen[img_path.name] = source
            n_copied += 1

            if n_copied % 5000 == 0:
                logger.info(f'  [{source}] {n_copied} images copied so far...')

        summary[source] = n_copied
        logger.info(
            f'  [{source}] ✓ {n_copied} images copied'
            + (f'  ({n_skipped} skipped — filename already present)' if n_skipped else '')
        )

    logger.info(f'  Total images in {images_dir}: {sum(summary.values())}')
    return summary


# ─────────────────────────────────────────────────────────────────────────────
# Per-source processing
# ─────────────────────────────────────────────────────────────────────────────

def process_source(source, data_dir, tokenizer, model, device, splits, max_rows, batch_size):
    """Translate all available splits for one source. Returns {split: df}."""
    source_dir = data_dir / source
    result     = {}

    for split in splits:
        csv_path = source_dir / f'{split}.csv'
        if not csv_path.exists():
            logger.warning(f'  [{source}] {split}.csv not found — skipping.')
            continue

        logger.info(f'\n[{source}] Processing {split} split...')
        df = pd.read_csv(csv_path)

        if max_rows:
            df = df.head(max_rows)
            logger.info(f'  Limiting to {max_rows} rows.')

        text_col  = detect_text_column(df, source)
        label_col = detect_label_column(df, source)
        image_col = detect_image_column(df, source)

        if text_col is None or label_col is None:
            logger.error(f'  [{source}/{split}] Skipping — required columns missing.')
            continue

        logger.info(f'  Rows: {len(df)}  |  text="{text_col}"  label="{label_col}"'
                    + (f'  image="{image_col}"' if image_col else ''))

        translated_text = translate_column(
            df[text_col], tokenizer, model, device, batch_size
        )

        out = pd.DataFrame()
        out['text']          = translated_text
        out['label']         = df[label_col].values
        out['image_id']      = df[image_col].values if image_col else ''
        out['original_text'] = df[text_col].values   # keep for reference

        result[split] = out
        logger.info(f'  [{source}/{split}] ✓ {len(out)} rows translated.')

    return result


# ─────────────────────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description='Create synthetic French dataset from English sources.',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument('--data_dir',        type=str, default='data')
    parser.add_argument('--sources',         nargs='+', default=['twitter', 'fakeddit'])
    parser.add_argument('--output',          type=str, default='french_synthetic')
    parser.add_argument('--splits',          nargs='+', default=['train', 'val', 'test'])
    parser.add_argument('--max_rows',        type=int, default=None)
    parser.add_argument('--batch_size',      type=int, default=16)
    parser.add_argument('--device',          type=str,
                        default='cuda' if torch.cuda.is_available() else 'cpu')
    parser.add_argument('--hf_cache',        type=str, default=None,
                        help='HuggingFace cache dir (avoids home quota on cluster)')
    parser.add_argument('--no_copy_images',  action='store_true', default=False,
                        help='Skip copying images (useful if disk space is tight)')
    args = parser.parse_args()

    if args.hf_cache:
        import os
        os.environ['TRANSFORMERS_CACHE'] = args.hf_cache
        os.environ['HF_HOME']            = args.hf_cache
        logger.info(f'HF cache → {args.hf_cache}')

    data_dir   = Path(args.data_dir)
    output_dir = data_dir / args.output
    output_dir.mkdir(parents=True, exist_ok=True)

    logger.info('='*80)
    logger.info('SYNTHETIC FRENCH DATASET CREATION')
    logger.info('='*80)
    logger.info(f'Sources:      {args.sources}')
    logger.info(f'Splits:       {args.splits}')
    logger.info(f'Output:       {output_dir}')
    logger.info(f'Device:       {args.device}')
    logger.info(f'Max rows:     {args.max_rows or "all"}')
    logger.info(f'Batch size:   {args.batch_size}')
    logger.info(f'Copy images:  {not args.no_copy_images}')
    logger.info('='*80 + '\n')

    # ── Translation ───────────────────────────────────────────────────────
    tokenizer, model = load_translator(args.device)

    all_splits = {s: [] for s in args.splits}

    for source in args.sources:
        source_dfs = process_source(
            source     = source,
            data_dir   = data_dir,
            tokenizer  = tokenizer,
            model      = model,
            device     = args.device,
            splits     = args.splits,
            max_rows   = args.max_rows,
            batch_size = args.batch_size,
        )
        for split, df in source_dfs.items():
            all_splits[split].append(df)

    # ── Save CSVs ─────────────────────────────────────────────────────────
    logger.info('\n' + '='*80)
    logger.info('SAVING CSVs')
    logger.info('='*80)

    for split, dfs in all_splits.items():
        if not dfs:
            logger.warning(f'No data for split "{split}" — skipping.')
            continue

        merged = pd.concat(dfs, ignore_index=True)
        merged = merged.sample(frac=1, random_state=42).reset_index(drop=True)

        out_path = output_dir / f'{split}.csv'
        merged.to_csv(out_path, index=False)

        dist = merged['label'].value_counts().to_dict()
        logger.info(f'✓ {split}.csv  {len(merged)} rows  labels={dist}')

    # ── Copy images ───────────────────────────────────────────────────────
    if not args.no_copy_images:
        logger.info('\n' + '='*80)
        logger.info('COPYING IMAGES')
        logger.info('='*80)
        logger.info(f'Destination: {output_dir}/images/')
        logger.info('(This may take a while for large datasets...)')
        copy_images(args.sources, data_dir, output_dir)
    else:
        logger.info('\n  ↷ Image copying skipped (--no_copy_images)')

    # ── Done ──────────────────────────────────────────────────────────────
    logger.info('\n' + '='*80)
    logger.info('DONE')
    logger.info(f'Output: {output_dir}')
    logger.info(f'  CSVs:   train.csv / val.csv / test.csv')
    if not args.no_copy_images:
        logger.info(f'  Images: {output_dir}/images/')
    logger.info('='*80 + '\n')


if __name__ == '__main__':
    main()