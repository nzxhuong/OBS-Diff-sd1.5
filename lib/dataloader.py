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

        captions = df['caption'].head(num_samples).tolist()
        print(f"Extracted {len(captions)} prompts from GCC3M")

        # normalized to (prompt, negative) pairs, same shape as get_prompts_txt --
        # GCC3M captions have no associated negative prompt, so "" throughout
        return [(c, "") for c in captions]

    except FileNotFoundError:
        print(f"Error: File {TSV_FILE_PATH} does not exist.")
        return []
    except Exception as e:
        print(f"Error occurred while reading the file: {str(e)}")
        return []


def get_prompts_txt(num_samples, path=TXT_FILE_PATH, delimiter="||"):
    """Load calibration prompts from a plain text file.

    Each line is either:
      - just a positive prompt, or
      - "positive_prompt||negative_prompt" (delimiter configurable)

    Returns a list of (prompt: str, negative_prompt: str) tuples -- negative
    is "" if the line had no delimiter or an empty right-hand side.
    """
    try:
        with open(path, 'r', encoding='utf-8') as f:
            raw_lines = [line.rstrip('\n') for line in f if line.strip()]

        print(f"Successfully read prompts file: {path}")
        print(f"Total number of lines available: {len(raw_lines)}")

        pairs = []
        for line in raw_lines:
            if delimiter in line:
                prompt, negative = line.split(delimiter, 1)
                pairs.append((prompt.strip(), negative.strip()))
            else:
                pairs.append((line.strip(), ""))

        pairs = pairs[:num_samples]
        print(f"Using {len(pairs)} prompt/negative pairs for calibration")

        if len(pairs) < num_samples:
            print(f"Warning: requested {num_samples} samples but only "
                  f"{len(pairs)} available in {path}")

        return pairs

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
