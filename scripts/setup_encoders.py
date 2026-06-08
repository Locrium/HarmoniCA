"""
Set up HarmoniCA model files in models/.

Each construct has its own HuggingFace repository containing:
  - {model_type}_model.pkl        (small, ~40 KB)
  - {model_type}_model_encoder/   (large, ~1-1.5 GB)

HF repo IDs are configured in harmonica/config.py (HF_REPOS dict).
Set them to your repo IDs before using the --download or --upload options.

Options
-------
A) Copy from a local source directory (e.g. your fine-tuning output):
   python scripts/setup_encoders.py --source /path/to/models-final

B) Download all constructs from HuggingFace (uses HF_REPOS from config.py):
   python scripts/setup_encoders.py --download

C) Download a single construct:
   python scripts/setup_encoders.py --download --construct depression

D) Upload a construct to its HuggingFace repo:
   python scripts/setup_encoders.py --upload --source /path/to/models-final --construct depression

E) Upload all constructs:
   python scripts/setup_encoders.py --upload --source /path/to/models-final
"""

import sys
import argparse
import shutil
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

from harmonica.config import HF_REPOS

CONSTRUCT_MODEL = {
    'depression':     'contrastive_model',
    'apathy':         'contrastive_model',
    'psychosis':      'contrastive_model',
    'anxiety':        'contrastive_model',
    'sleep':          'contrastive_model',
    'impulse_control':'prototype_model',
}

DEST_DIR = REPO_ROOT / 'models'


def _constructs_to_process(args_construct):
    if args_construct:
        if args_construct not in CONSTRUCT_MODEL:
            raise ValueError(f"Unknown construct '{args_construct}'. "
                             f"Choose from: {list(CONSTRUCT_MODEL)}")
        return [args_construct]
    return list(CONSTRUCT_MODEL)


def copy_from_local(source_dir: Path, constructs: list):
    print(f"Copying model files from {source_dir} ...")
    for construct in constructs:
        model_stem   = CONSTRUCT_MODEL[construct]
        src_base     = source_dir / construct
        dest_base    = DEST_DIR / construct
        dest_base.mkdir(parents=True, exist_ok=True)

        # PKL
        pkl_src  = src_base / f'{model_stem}.pkl'
        pkl_dest = dest_base / f'{model_stem}.pkl'
        if pkl_src.exists():
            shutil.copy2(pkl_src, pkl_dest)
            print(f"  {construct}: copied {model_stem}.pkl")
        else:
            print(f"  {construct}: WARNING — {pkl_src} not found, skipping")
            continue

        # Encoder directory
        enc_src  = src_base / f'{model_stem}_encoder'
        enc_dest = dest_base / f'{model_stem}_encoder'
        if enc_src.exists():
            if enc_dest.exists():
                shutil.rmtree(enc_dest)
            shutil.copytree(enc_src, enc_dest)
            print(f"  {construct}: copied {model_stem}_encoder/")
        else:
            print(f"  {construct}: WARNING — {enc_src} not found")

    print("Done.")


def download_from_hub(constructs: list):
    from huggingface_hub import snapshot_download

    for construct in constructs:
        repo_id = HF_REPOS.get(construct)
        if repo_id is None:
            print(f"  {construct}: SKIP — HF_REPOS['{construct}'] is not set in config.py")
            continue

        dest = DEST_DIR / construct
        dest.mkdir(parents=True, exist_ok=True)
        print(f"  {construct}: downloading from {repo_id} ...")
        snapshot_download(repo_id=repo_id, local_dir=str(dest))
        print(f"  {construct}: done")

    print("Download complete.")


def upload_to_hub(source_dir: Path, constructs: list):
    from huggingface_hub import HfApi, create_repo

    api = HfApi()

    for construct in constructs:
        repo_id = HF_REPOS.get(construct)
        if repo_id is None:
            print(f"  {construct}: SKIP — HF_REPOS['{construct}'] is not set in config.py")
            continue

        model_stem = CONSTRUCT_MODEL[construct]
        src_base   = source_dir / construct
        pkl_src    = src_base / f'{model_stem}.pkl'
        enc_src    = src_base / f'{model_stem}_encoder'

        if not pkl_src.exists():
            print(f"  {construct}: SKIP — {pkl_src} not found")
            continue

        create_repo(repo_id=repo_id, repo_type='model', exist_ok=True)
        print(f"  {construct}: uploading to {repo_id} ...")

        api.upload_file(
            path_or_fileobj=str(pkl_src),
            path_in_repo=f'{model_stem}.pkl',
            repo_id=repo_id,
            repo_type='model',
        )

        if enc_src.exists():
            api.upload_folder(
                folder_path=str(enc_src),
                path_in_repo=f'{model_stem}_encoder',
                repo_id=repo_id,
                repo_type='model',
            )

        print(f"  {construct}: uploaded")

    print("Upload complete.")


def main():
    parser = argparse.ArgumentParser(
        description='Set up HarmoniCA model files.',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument('--source', type=str,
                       help='Copy from a local models directory (Option A / D-source)')
    group.add_argument('--download', action='store_true',
                       help='Download from HuggingFace Hub (uses HF_REPOS in config.py)')
    group.add_argument('--upload', action='store_true',
                       help='Upload to HuggingFace Hub (requires --source)')

    parser.add_argument('--construct', type=str, default=None,
                        help='Process a single construct (default: all)')
    parser.add_argument('--upload-source', type=str, dest='upload_source',
                        help='Local models directory when using --upload')

    args = parser.parse_args()
    constructs = _constructs_to_process(args.construct)

    if args.source:
        copy_from_local(Path(args.source), constructs)

    elif args.download:
        download_from_hub(constructs)

    elif args.upload:
        src = Path(args.upload_source) if args.upload_source else None
        if src is None:
            parser.error('--upload requires --upload-source <path>')
        upload_to_hub(src, constructs)


if __name__ == '__main__':
    main()
