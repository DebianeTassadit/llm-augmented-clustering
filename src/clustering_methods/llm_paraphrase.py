import pickle
import os
import numpy as np
import pandas as pd
import concurrent.futures
from typing import List, Dict, Tuple, Optional
from sklearn.cluster import KMeans
from sklearn.preprocessing import normalize
from tqdm import tqdm
from src.llm_service import LLMService, ParaphraseList


def _process_doc(
    doc_index: int,
    document: str,
    llm_service: LLMService,
    prompt_template: str,
) -> Tuple[int, str, List[str]]:
    """Query LLM for paraphrases of a document."""
    try:
        prompt = prompt_template.format(text=document)
        response = llm_service.get_chat_completion(
            prompt, output_structure=ParaphraseList
        )
        paraphrases = (
            response.paraphrases
            if response and hasattr(response, "paraphrases")
            else []
        )
        return (doc_index, document, paraphrases)
    except Exception:
        return (doc_index, document, [])


def cluster_via_llm_paraphrase(
    documents: List[str],
    features: np.ndarray,
    n_clusters: int,
    llm_service: LLMService,
    prompt_template: str,
    output_path: str = "paraphrase_output.csv",
    output_dir: str = "results/",
    max_workers: int = 50,
) -> Optional[np.ndarray]:
    """Cluster using LLM-generated paraphrase ensemble.

    Phase 1: LLM generates N paraphrases per doc (cached in paraphrase_cache.pkl).
    Phase 2: Embed original + all paraphrases; mean-pool → L2-normalize → KMeans.
    """
    print("\n--- Running LLM Paraphrase Ensemble Clustering ---")
    n_samples = len(documents)

    os.makedirs(output_dir, exist_ok=True)
    cache_file = os.path.join(output_dir, "paraphrase_cache.pkl")
    paraphrase_map: Dict[int, List[str]] = {}

    if os.path.exists(cache_file):
        try:
            with open(cache_file, "rb") as f:
                paraphrase_map = pickle.load(f)
            print(
                f"  Loaded paraphrases from cache: {cache_file} ({len(paraphrase_map)} docs)"
            )
        except Exception as e:
            print(f"  Paraphrase cache load failed: {e}. Re-querying LLM.")
            paraphrase_map = {}

    if not paraphrase_map:
        if not llm_service.generation_available():
            print("  No paraphrase cache and no generation model. Cannot run.")
            return None

        print(
            f"  Querying LLM for paraphrases ({n_samples} docs, up to {max_workers} workers)..."
        )
        workers = min(max_workers, n_samples)
        raw_results: List[Optional[Tuple]] = [None] * n_samples

        with concurrent.futures.ThreadPoolExecutor(max_workers=workers) as executor:
            futures = {
                executor.submit(
                    _process_doc, i, documents[i], llm_service, prompt_template
                ): i
                for i in range(n_samples)
            }
            for future in tqdm(
                concurrent.futures.as_completed(futures),
                total=n_samples,
                desc="Paraphrasing",
            ):
                raw_results[futures[future]] = future.result()

        for result in raw_results:
            if result:
                doc_index, _, paraphrases = result
                paraphrase_map[doc_index] = paraphrases

        with open(cache_file, "wb") as f:
            pickle.dump(paraphrase_map, f)
        print(f"  Paraphrases cached to: {cache_file}")

        pd.DataFrame(
            [
                {
                    "document_index": i,
                    "original": documents[i],
                    "paraphrases": " | ".join(paraphrase_map.get(i, [])),
                }
                for i in range(n_samples)
            ]
        ).to_csv(output_path, index=False)

    print(f"  Embedding paraphrase ensembles ({n_samples} docs)...")
    # Flatten [doc, para_1, ..., para_m] for all docs into one batch, then
    # mean-pool each doc's slice back into a single ensemble vector.
    texts: List[str] = []
    offsets = [0]
    for i in range(n_samples):
        texts.extend([documents[i]] + paraphrase_map.get(i, []))
        offsets.append(len(texts))
    all_embeddings = llm_service.get_embeddings(texts)
    ensemble_features = np.array(
        [all_embeddings[offsets[i] : offsets[i + 1]].mean(axis=0) for i in range(n_samples)]
    )
    ensemble_features = normalize(ensemble_features, axis=1, norm="l2")

    try:
        clusters = KMeans(
            n_clusters=n_clusters, random_state=0, n_init="auto"
        ).fit_predict(ensemble_features)
        return clusters
    except Exception as e:
        print(f"  Clustering error: {e}")
        return None
