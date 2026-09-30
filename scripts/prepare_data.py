"""
Data preparation and validation script with support for nested image structures
Validates data integrity, image availability, and directory structures
INCLUDES: Automatic validation split creation + Debug logging
"""

import argparse
import sys
from pathlib import Path
import pandas as pd
import logging
import json
from collections import defaultdict
import numpy as np

sys.path.append(str(Path(__file__).parent.parent))

from config.config import DataConfig
from src.data.image_loader import FlexibleImageLoader
from src.utils.logger import setup_logger

logger = setup_logger('data_prep', 'logs/data_prep.log')


def debug_image_matching(image_loader: FlexibleImageLoader, source: str, sample_ids: list, split: str):
    """
    Debug function to check why images aren't being found
    """
    logger.info(f"\n    DEBUG: Testing image matching for {source}/{split}...")
    
    structure = image_loader.get_structure_info(source)
    if not structure:
        logger.error(f"    No structure info for {source}")
        return
    
    logger.info(f"    Structure type: {structure['type']}")
    logger.info(f"    Total images in cache: {structure['total_images']}")
    logger.info(f"    Sample from basename_map: {list(structure['basename_map'].keys())[:5]}")
    
    # Test first 3 sample IDs
    for img_id in sample_ids[:3]:
        img_path = image_loader.get_image_path(source, img_id, split)
        if img_path:
            logger.info(f"    ✓ Found: {img_id} -> {Path(img_path).name}")
        else:
            logger.error(f"    ✗ NOT FOUND: {img_id}")
            # Try to diagnose
            img_id_base = str(img_id).strip()
            logger.error(f"      Searching for: '{img_id_base}'")
            logger.error(f"      In basename_map: {img_id_base in structure['basename_map']}")
            logger.error(f"      Keys containing '{img_id_base}': {[k for k in structure['basename_map'].keys() if img_id_base in k][:5]}")


def create_validation_split(train_csv_path: Path, val_csv_path: Path, val_ratio: float = 0.2):
    """Create validation split from training data if it doesn't exist"""
    try:
        df = pd.read_csv(train_csv_path)
        total_samples = len(df)
        
        label_cols = [col for col in df.columns 
                     if any(term in col.lower() for term in ['label', 'class'])]
        
        if not label_cols:
            logger.warning(f"No label column found in {train_csv_path}, using random split")
            df = df.sample(frac=1, random_state=42).reset_index(drop=True)
            split_idx = int(len(df) * (1 - val_ratio))
            train_df = df.iloc[:split_idx]
            val_df = df.iloc[split_idx:]
        else:
            label_col = label_cols[0]
            train_dfs = []
            val_dfs = []
            
            for label in df[label_col].unique():
                label_df = df[df[label_col] == label].sample(frac=1, random_state=42)
                split_idx = int(len(label_df) * (1 - val_ratio))
                train_dfs.append(label_df.iloc[:split_idx])
                val_dfs.append(label_df.iloc[split_idx:])
            
            train_df = pd.concat(train_dfs, ignore_index=True).sample(frac=1, random_state=42)
            val_df = pd.concat(val_dfs, ignore_index=True).sample(frac=1, random_state=42)
        
        train_df.to_csv(train_csv_path, index=False)
        val_df.to_csv(val_csv_path, index=False)
        
        logger.info(f"  ✓ Created validation split:")
        logger.info(f"    - Training: {len(train_df)} samples ({len(train_df)/total_samples*100:.1f}%)")
        logger.info(f"    - Validation: {len(val_df)} samples ({len(val_df)/total_samples*100:.1f}%)")
        
        return True
    
    except Exception as e:
        logger.error(f"  ✗ Failed to create validation split: {e}")
        return False


def analyze_csv_structure(csv_path: Path) -> dict:
    """Analyze CSV structure and column names"""
    try:
        df = pd.read_csv(csv_path)
        
        analysis = {
            'rows': len(df),
            'columns': list(df.columns),
            'missing_values': {k: int(v) for k, v in df.isnull().sum().to_dict().items()},
            'dtypes': df.dtypes.astype(str).to_dict()
        }
        
        # FIXED: More precise column detection
        # Text columns - exclude 'post_id' which is just an ID
        text_cols = [col for col in df.columns 
                    if any(term in col.lower() for term in ['text', 'content', 'tweet']) 
                    and 'id' not in col.lower()]  # Exclude columns with 'id'
        
        # Image columns - prioritize exact matches
        image_cols = []
        # First, look for exact 'image_id' or 'image' column
        if 'image_id' in df.columns:
            image_cols = ['image_id']
        elif 'image' in df.columns:
            image_cols = ['image']
        else:
            # Fall back to pattern matching
            image_cols = [col for col in df.columns 
                         if any(term in col.lower() for term in ['img', 'photo', 'pic'])
                         and 'id' in col.lower()]  # Must have 'id' for image identifier
        
        # Label columns
        label_cols = [col for col in df.columns 
                     if any(term in col.lower() for term in ['label', 'class', 'category'])]
        
        analysis['text_columns'] = text_cols
        analysis['image_columns'] = image_cols
        analysis['label_columns'] = label_cols
        
        if label_cols:
            label_col = label_cols[0]
            label_dist = df[label_col].value_counts().to_dict()
            analysis['label_distribution'] = {str(k): int(v) for k, v in label_dist.items()}
        
        return analysis
    
    except Exception as e:
        logger.error(f"Error analyzing CSV {csv_path}: {e}")
        return None


def validate_images(
    df: pd.DataFrame,
    image_loader: FlexibleImageLoader,
    source: str,
    split: str,
    sample_size: int = 100,
    debug: bool = False,
    csv_analysis: dict = None  # NEW: Pass the analysis
) -> dict:
    """Validate image availability with support for nested directories"""
    
    # Use the image column from csv_analysis if provided
    if csv_analysis and csv_analysis.get('image_columns'):
        image_col = csv_analysis['image_columns'][0]
    else:
        # Fallback to detection (prioritize exact matches)
        if 'image_id' in df.columns:
            image_col = 'image_id'
        elif 'image' in df.columns:
            image_col = 'image'
        else:
            # Pattern matching as last resort
            image_cols = [col for col in df.columns 
                         if any(term in col.lower() for term in ['img', 'photo', 'pic'])]
            if not image_cols:
                logger.error(f"No image column found in {source}/{split}")
                return {'error': 'No image column found'}
            image_col = image_cols[0]
    
    # Debug: Show what column we're using and sample values
    if debug:
        logger.info(f"    DEBUG: Using image column: '{image_col}'")
        logger.info(f"    DEBUG: Sample values from '{image_col}': {df[image_col].head(5).tolist()}")
    
    sample_size = min(sample_size, len(df))
    sample_indices = np.linspace(0, len(df)-1, sample_size, dtype=int)
    
    missing_images = []
    found_images = []
    sample_ids = []
    
    for idx in sample_indices:
        image_id = str(df.iloc[idx][image_col])
        sample_ids.append(image_id)
        
        image_path = image_loader.get_image_path(
            source=source,
            image_id=image_id,
            split=split
        )
        
        if image_path is None:
            missing_images.append(image_id)
        else:
            found_images.append(image_id)
    
    # Add debug logging if images not found
    if debug and len(found_images) == 0 and len(sample_ids) > 0:
        debug_image_matching(image_loader, source, sample_ids, split)
    
    return {
        'total_checked': sample_size,
        'found': len(found_images),
        'missing': len(missing_images),
        'missing_examples': missing_images[:5],
        'found_rate': len(found_images) / sample_size if sample_size > 0 else 0
    }

def validate_data_structure(
    data_dir: str,
    sources: list = None,
    sample_size: int = 100,
    output_report: str = None,
    auto_create_val: bool = True,
    val_ratio: float = 0.2,
    debug: bool = False
) -> bool:
    """Comprehensive data structure validation"""
    
    logger.info("="*80)
    logger.info("DATA STRUCTURE VALIDATION")
    if auto_create_val:
        logger.info("(Auto-creating validation splits if missing)")
    logger.info("="*80)
    
    data_path = Path(data_dir)
    
    if not data_path.exists():
        logger.error(f"Data directory does not exist: {data_dir}")
        return False
    
    if sources is None:
        sources = [d.name for d in data_path.iterdir() 
                  if d.is_dir() and not d.name.startswith('.')]
    
    logger.info(f"\nFound {len(sources)} source directories: {sources}\n")
    
    # Create image loader
    source_paths = {source: str(data_path / source) for source in sources}
    image_loader = FlexibleImageLoader(source_paths, cache_structure=True)
    
    # Print structure summary
    image_loader.print_structure_summary()
    
    validation_report = {
        'data_dir': str(data_dir),
        'sources': {},
        'overall_status': 'passed',
        'warnings': [],
        'errors': []
    }
    
    all_valid = True
    
    for source in sources:
        logger.info(f"\n{'='*80}")
        logger.info(f"VALIDATING {source.upper()}")
        logger.info(f"{'='*80}")
        
        source_dir = data_path / source
        source_report = {
            'path': str(source_dir),
            'splits': {},
            'status': 'passed'
        }
        
        structure_info = image_loader.get_structure_info(source)
        if structure_info:
            source_report['image_structure'] = {
                'type': structure_info['type'],
                'base_path': str(structure_info['base_path']),
                'total_images': structure_info['total_images']
            }
        
        csv_files = {
            'train': source_dir / 'train.csv',
            'val': source_dir / 'val.csv',
            'test': source_dir / 'test.csv'
        }
        
        if not csv_files['val'].exists() and csv_files['train'].exists() and auto_create_val:
            logger.info(f"\n  ⚠ val.csv not found, creating from train.csv...")
            if create_validation_split(csv_files['train'], csv_files['val'], val_ratio):
                logger.info(f"  ✓ Validation split created successfully")
            else:
                logger.warning(f"  ✗ Failed to create validation split")
        
        for split_name, csv_path in csv_files.items():
            split_report = {
                'csv_path': str(csv_path),
                'exists': csv_path.exists(),
                'status': 'passed'
            }
            
            if not csv_path.exists():
                logger.warning(f"  ✗ {split_name}.csv not found")
                split_report['status'] = 'missing'
                validation_report['warnings'].append(f"{source}/{split_name}.csv not found")
                source_report['splits'][split_name] = split_report
                continue
            
            logger.info(f"\n  Validating {split_name}.csv...")
            
            csv_analysis = analyze_csv_structure(csv_path)
            
            if csv_analysis is None:
                logger.error(f"    ✗ Failed to analyze CSV")
                split_report['status'] = 'error'
                all_valid = False
                validation_report['errors'].append(f"{source}/{split_name}.csv could not be analyzed")
                source_report['splits'][split_name] = split_report
                continue
            
            split_report['analysis'] = csv_analysis
            
            logger.info(f"    ✓ Loaded {csv_analysis['rows']} rows")
            logger.info(f"    ✓ Columns: {', '.join(csv_analysis['columns'])}")
            
            if not csv_analysis['text_columns']:
                logger.error(f"    ✗ No text column found")
                split_report['status'] = 'error'
                all_valid = False
                validation_report['errors'].append(f"{source}/{split_name}.csv: No text column")
            else:
                logger.info(f"    ✓ Text columns: {csv_analysis['text_columns']}")
            
            if not csv_analysis['image_columns']:
                logger.error(f"    ✗ No image column found")
                split_report['status'] = 'error'
                all_valid = False
                validation_report['errors'].append(f"{source}/{split_name}.csv: No image column")
            else:
                logger.info(f"    ✓ Image columns: {csv_analysis['image_columns']}")
            
            if not csv_analysis['label_columns']:
                logger.error(f"    ✗ No label column found")
                split_report['status'] = 'error'
                all_valid = False
                validation_report['errors'].append(f"{source}/{split_name}.csv: No label column")
            else:
                logger.info(f"    ✓ Label columns: {csv_analysis['label_columns']}")
                
                if 'label_distribution' in csv_analysis:
                    logger.info(f"    Label distribution:")
                    for label, count in csv_analysis['label_distribution'].items():
                        pct = (count / csv_analysis['rows']) * 100
                        logger.info(f"      - {label}: {count} ({pct:.1f}%)")
            
            logger.info(f"\n    Checking image availability...")
            df = pd.read_csv(csv_path)
            
            image_validation = validate_images(
                df, image_loader, source, split_name, sample_size, debug=debug
            )
            
            split_report['image_validation'] = image_validation
            
            if 'error' in image_validation:
                logger.error(f"    ✗ Image validation failed: {image_validation['error']}")
                split_report['status'] = 'error'
                all_valid = False
            else:
                found_rate = image_validation['found_rate']
                logger.info(f"    ✓ Found {image_validation['found']}/{image_validation['total_checked']} "
                          f"sampled images ({found_rate*100:.1f}%)")
                
                if image_validation['missing'] > 0:
                    logger.warning(f"    ⚠ {image_validation['missing']} images not found")
                    logger.warning(f"      Examples: {image_validation['missing_examples']}")
                    validation_report['warnings'].append(
                        f"{source}/{split_name}: {image_validation['missing']} images missing"
                    )
                    
                    if found_rate < 0.5:
                        logger.warning(f"    ⚠ Low image availability ({found_rate*100:.1f}%)")
                        validation_report['warnings'].append(
                            f"{source}/{split_name}: Low image availability ({found_rate*100:.1f}%)"
                        )
            
            source_report['splits'][split_name] = split_report
        
        if not (csv_files['train'].exists() and csv_files['val'].exists()):
            logger.error(f"  ✗ Source missing required splits (train and val)")
            source_report['status'] = 'error'
            all_valid = False
            validation_report['errors'].append(f"{source}: Missing required splits (train/val)")
        
        validation_report['sources'][source] = source_report
    
    logger.info("\n" + "="*80)
    logger.info("VALIDATION SUMMARY")
    logger.info("="*80)
    
    if all_valid and len(validation_report['errors']) == 0:
        logger.info("✓ DATA VALIDATION PASSED")
        validation_report['overall_status'] = 'passed'
    elif len(validation_report['errors']) > 0:
        logger.error("✗ DATA VALIDATION FAILED")
        validation_report['overall_status'] = 'failed'
        all_valid = False
    else:
        logger.warning("⚠ DATA VALIDATION PASSED WITH WARNINGS")
        validation_report['overall_status'] = 'passed_with_warnings'
    
    if validation_report['warnings']:
        logger.info(f"\nWarnings ({len(validation_report['warnings'])}):")
        for warning in validation_report['warnings'][:10]:
            logger.info(f"  - {warning}")
        if len(validation_report['warnings']) > 10:
            logger.info(f"  ... and {len(validation_report['warnings']) - 10} more")
    
    if validation_report['errors']:
        logger.error(f"\nErrors ({len(validation_report['errors'])}):")
        for error in validation_report['errors']:
            logger.error(f"  - {error}")
    
    logger.info("="*80 + "\n")
    
    if output_report:
        output_path = Path(output_report)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        
        with open(output_path, 'w') as f:
            json.dump(validation_report, f, indent=2)
        
        logger.info(f"Validation report saved to: {output_report}\n")
    
    return all_valid


def generate_statistics(data_dir: str, sources: list = None):
    """Generate comprehensive statistics about the dataset"""
    logger.info("\n" + "="*80)
    logger.info("DATASET STATISTICS")
    logger.info("="*80)
    
    data_path = Path(data_dir)
    
    if sources is None:
        sources = [d.name for d in data_path.iterdir() 
                  if d.is_dir() and not d.name.startswith('.')]
    
    total_stats = defaultdict(lambda: defaultdict(int))
    
    for source in sources:
        logger.info(f"\n{source.upper()}:")
        source_dir = data_path / source
        
        for split in ['train', 'val', 'test']:
            csv_path = source_dir / f'{split}.csv'
            
            if not csv_path.exists():
                continue
            
            df = pd.read_csv(csv_path)
            
            label_cols = [col for col in df.columns 
                         if any(term in col.lower() for term in ['label', 'class'])]
            
            if label_cols:
                label_col = label_cols[0]
                label_dist = df[label_col].value_counts()
                
                logger.info(f"  {split}:")
                logger.info(f"    Total: {len(df)}")
                for label, count in label_dist.items():
                    pct = (count / len(df)) * 100
                    logger.info(f"      - {label}: {count} ({pct:.1f}%)")
                    total_stats[split][str(label)] += count
                total_stats[split]['total'] += len(df)
    
    logger.info("\n" + "-"*80)
    logger.info("OVERALL STATISTICS:")
    logger.info("-"*80)
    
    for split in ['train', 'val', 'test']:
        if split in total_stats:
            logger.info(f"\n{split.upper()}:")
            logger.info(f"  Total: {total_stats[split]['total']}")
            
            labels = {k: v for k, v in total_stats[split].items() if k != 'total'}
            for label, count in labels.items():
                pct = (count / total_stats[split]['total']) * 100
                logger.info(f"    - {label}: {count} ({pct:.1f}%)")
    
    logger.info("\n" + "="*80 + "\n")


def main():
    parser = argparse.ArgumentParser(
        description='Validate and prepare data with flexible image structures',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    
    parser.add_argument('--data_dir', type=str, default='data',
                       help='Root data directory')
    parser.add_argument('--sources', nargs='+', default=None,
                       help='Specific sources to validate (default: all)')
    parser.add_argument('--sample_size', type=int, default=100,
                       help='Number of images to sample for validation per split')
    parser.add_argument('--output_report', type=str, default='data_validation_report.json',
                       help='Path to save validation report')
    parser.add_argument('--stats', action='store_true',
                       help='Generate dataset statistics')
    parser.add_argument('--no_auto_val', action='store_true',
                       help='Disable automatic validation split creation')
    parser.add_argument('--val_ratio', type=float, default=0.2,
                       help='Validation split ratio (default: 0.2 = 20%%)')
    parser.add_argument('--debug', action='store_true',
                       help='Enable debug logging for image matching')
    
    args = parser.parse_args()
    
    success = validate_data_structure(
        args.data_dir,
        args.sources,
        args.sample_size,
        args.output_report,
        auto_create_val=not args.no_auto_val,
        val_ratio=args.val_ratio,
        debug=args.debug
    )
    
    if args.stats:
        generate_statistics(args.data_dir, args.sources)
    
    sys.exit(0 if success else 1)


if __name__ == '__main__':
    main()