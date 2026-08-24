from datasets import load_dataset
import pandas as pd
import warnings
warnings.filterwarnings("ignore", message=".*promote_options.*")
warnings.filterwarnings("ignore", message=".*precompiled_charsmap.*")

TSV_FILE_PATH = "./data/Train_GCC-training_with_header.tsv"
TXT_FILE_PATH = "./data/prompts.txt"


def get_gcc3m(num_samples):
    try:
        df = pd.read_csv(TSV_FILE_PATH, sep='\t', encoding='utf-8')
        print(f"Successfully read TSV file: {TSV_FILE_PATH}")
        print(f"Total number of rows: {len(df)}")
        print(f"Column names: {list(df.columns)}")

        prompts = df['caption'].head(num_samples).tolist()
        print(f"Extracted {len(prompts)} prompts from GCC3M")

        return prompts

    except FileNotFoundError:
        print(f"Error: File {TSV_FILE_PATH} does not exist.")
        return []
    except Exception as e:
        print(f"Error occurred while reading the file: {str(e)}")
        return []


def get_prompts_txt(num_samples, path=TXT_FILE_PATH):
    """Load calibration prompts from a plain text file, one prompt per line.
    Used as a substitute for GCC3M when the TSV/image URLs aren't available --
    OBS-Diff's Hessian only ever consumes the caption text, never the image,
    so a flat prompt list is a drop-in replacement for calibration purposes."""
    try:
        with open(path, 'r', encoding='utf-8') as f:
            # strip blank lines in case the file has trailing/interspersed empties
            prompts = [line.strip() for line in f if line.strip()]

        print(f"Successfully read prompts file: {path}")
        print(f"Total number of prompts available: {len(prompts)}")

        prompts = prompts[:num_samples]
        print(f"Using {len(prompts)} prompts for calibration")

        if len(prompts) < num_samples:
            print(f"Warning: requested {num_samples} samples but only "
                  f"{len(prompts)} available in {path}")

        return prompts

    except FileNotFoundError:
        print(f"Error: File {path} does not exist.")
        return []
    except Exception as e:
        print(f"Error occurred while reading the file: {str(e)}")
        return []


def get_loaders(name, num_samples=50):
    if name == 'gcc3m':
        return get_gcc3m(num_samples)
    if name == 'prompts_txt':
        return get_prompts_txt(num_samples)
    raise ValueError(f"Unknown dataset: {name}")
