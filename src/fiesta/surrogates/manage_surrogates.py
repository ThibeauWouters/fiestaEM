import os
from pathlib import Path
from shutil import copy2, rmtree

from fiesta.logging import logger

from huggingface_hub import hf_hub_download, HfApi
from huggingface_hub.errors import EntryNotFoundError
from huggingface_hub.utils import HfHubHTTPError

HF_REPO_ID = "nuclear-multimessenger-astronomy/fiesta-surrogates"
HF_REVISION = "main"

###########################
### BUILT-IN SURROGATES ###
###########################


def built_in_surrogates():
    surrogate_dir = Path(__file__).resolve().parent

    for transient_dir in sorted(surrogate_dir.iterdir()):
        if not transient_dir.is_dir():
            continue
        if ".cache" in transient_dir.parts:
            continue
        transient_type = transient_dir.name

        for model_dir in sorted(transient_dir.iterdir()):
            if not model_dir.is_dir():
                continue
            model_name = model_dir.name

            if not model_name.startswith("_"):
                yield model_name, model_dir, transient_type

def print_built_in_surrogates():
    logger.info(f"Available built-in surrogates in fiesta are:")
    for model_name, _, transient_type in built_in_surrogates():
        logger.info(f"\t {model_name} ({transient_type})")


#################################
### DOWNLOAD FROM HUGGINGFACE ###
#################################

def download_surrogate(
        name: str,
        directory: str | None = None,
    ) -> tuple[bool, str | None]:
    """
    Downloads a surrogate from the fiesta hugging-face repository.

    Args:
        name (str): Which surrogate to download. Available downloads can be checked with ``print_downloadable_surrogates``.
        directory (str | None): Where to download the surrogate. 
        Defaults to ``None``, in which case the surrogate will be downloaded to the installation directory from where it can be loaded automatically.

    Returns:
        download_ok (bool): Whether the download was successful.
        surrogate_dir (str): Location where the surrogate was downloaded to.
    """
    
    if name.endswith("_lc"):
        raise ValueError("Light curve models are not supported for automatic download at the moment. Please download manually from Hugging Face.")

    logger.info(f"Attempting to download {name} from Hugging Face ({HF_REPO_ID}).")

    download_ok = False
    for transient in ["KN", "GRB"]:

        try:
            metadata_path = f"{transient}/{name}/model/{name}_metadata.pkl"
            downloaded_metadata = hf_hub_download(
                    repo_id=HF_REPO_ID,
                    revision=HF_REVISION,
                    filename=metadata_path,
                )

            if directory is None:
                download_dir = Path(__file__).resolve().parent
                Path(download_dir / f"{transient}/{name}/model").mkdir(parents=True, exist_ok=True)
            else:
                download_dir = Path(directory)
                Path(download_dir).mkdir(parents=True, exist_ok=True)

            if directory is not None:
                metadata_path = f"{name}_metadata.pkl"
            if (download_dir / metadata_path).exists():
                logger.warning(f"Surrogate metadata for {name} already present in {download_dir / metadata_path}. Will be overwritten through download.")

            copy2(downloaded_metadata, download_dir / metadata_path)
            download_ok = True
            logger.info(f"Found {metadata_path}. Downloading model ...")
            break

        except EntryNotFoundError:
            continue

        except HfHubHTTPError:
            logger.exception(f"Hugging Face lookup failed for transient={transient}, model={name}.")
            raise

    if not download_ok:
        return download_ok, None

    model_path = f"{transient}/{name}/model/{name}.pkl"
    try:
        downloaded_pkl = hf_hub_download(
            repo_id=HF_REPO_ID,
            revision=HF_REVISION,
            filename=model_path,
        )

        if directory is not None:
            model_path = f"{name}.pkl"

        copy2(downloaded_pkl, download_dir / model_path)

    except EntryNotFoundError:
        logger.warning(f"Model file not found on Hugging Face: {model_path}")
        return False, None
    
    except HfHubHTTPError:
        logger.exception(f"Hugging Face model download failed for transient={transient}, model={name}.")
        raise

    if directory is None:
        download_dir = download_dir / transient / name
    logger.info(f"Successfully downloaded {name} to {download_dir}.")

    return download_ok, download_dir

def download_recommended_surrogates():

    download_surrogate("Bu2026_MLP")
    download_surrogate("afgpy_gaussian_CVAE")
    download_surrogate("pbag_gaussian_CVAE")

def print_downloadable_surrogates():

    files = HfApi().list_repo_files(
        repo_id=HF_REPO_ID,
        revision=HF_REVISION,
    )

    available = {"KN": set(), "GRB": set()}

    for path in files:
        # Expected structure:
        # {transient}/<name>/model/<name>_metadata.pkl
        parts = path.split("/")
        if len(parts) == 4 and parts[2] == "model" and parts[3].endswith("_metadata.pkl"):
            transient = parts[0]
            name = parts[1]
            available[transient].add(name)

    if not available:
        print("No surrogate models found.")
        return

    logger.info(f"Downloadable surrogates for fiesta are:")
    for transient, models in available.items():
        logger.info(f"\t ==================== ")
        logger.info(f"\t \t-{transient}-")
        for name in models:
            logger.info(f"\t {name}")
        logger.info(f"\t ==================== \n \n")
