import pickle
import os
import numpy as np
import pandas as pd
import concurrent.futures
from typing import List, Dict, Tuple, Optional
from sklearn.cluster import KMeans
from sklearn.preprocessing import normalize
from langchain_core.prompts import ChatPromptTemplate
from tqdm import tqdm
from src.llm_service import LLMService, KeyphraseList


def process_document(
    doc_index: int,
    document: str,
    llm_service: LLMService,
    prompt_template: ChatPromptTemplate,
) -> Tuple[int, str, List[str]]:
    """Query the LLM for keyphrases for one document."""
    try:
        prompt = prompt_template.format(document_text=document)
        llm_response = llm_service.get_chat_completion(
            prompt, output_structure=KeyphraseList
        )
        keyphrases = (
            llm_response.keyphrases
            if llm_response and hasattr(llm_response, "keyphrases")
            else []
        )
        return (doc_index, document, keyphrases)
    except Exception:
        return (doc_index, document, [])


def cluster_via_keyphrase_expansion(
    documents: List[str],
    features: np.ndarray,
    n_clusters: int,
    llm_service: LLMService,
    keyphrase_prompt_template: str,
    keyphrase_output_csv_path: str = "results/keyphrase_expansions.csv",
) -> Dict[str, np.ndarray]:
    """Cluster using LLM-generated keyphrase expansions.

    Keyphrases are cached to {output_dir}/keyphrases_cache.pkl; on subsequent
    runs the LLM is skipped and only local re-embedding is performed (fast).

    Returns a dict of variant_name → cluster_assignments for variants:
    'concatenated', 'average', 'weighted_0.1' … 'weighted_1.0'.
    """
    n_samples = len(documents)

    output_dir = os.path.dirname(keyphrase_output_csv_path)
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)
    cache_file = (
        os.path.join(output_dir, "keyphrases_cache.pkl")
        if output_dir
        else "keyphrases_cache.pkl"
    )

    keyphrases_map: Dict[int, List[str]] = {}

    if os.path.exists(cache_file):
        try:
            with open(cache_file, "rb") as f:
                keyphrases_map = pickle.load(f)
            print(
                f"  Loaded keyphrases from cache: {cache_file} ({len(keyphrases_map)} docs)"
            )
        except Exception as e:
            print(f"  Keyphrase cache load failed: {e}. Re-querying LLM.")
            keyphrases_map = {}

    if not keyphrases_map:
        if not llm_service.generation_available():
            print(
                "  No keyphrase cache and no generation model. Cannot run keyphrase expansion."
            )
            return {"concatenated": None, "average": None}

        print(f"  Querying LLM for keyphrases ({n_samples} docs, up to 50 workers)...")
        prompt_template = ChatPromptTemplate.from_template(
            keyphrase_prompt_template + "\nDocument: {document_text}"
        )
        max_workers = min(50, n_samples)
        raw_results: List[Tuple] = [None] * n_samples

        with concurrent.futures.ThreadPoolExecutor(max_workers=max_workers) as executor:
            futures = {
                executor.submit(
                    process_document,
                    i,
                    documents[i],
                    llm_service,
                    prompt_template,
                ): i
                for i in range(n_samples)
            }
            for future in tqdm(
                concurrent.futures.as_completed(futures),
                total=n_samples,
                desc="Keyphrases",
            ):
                raw_results[futures[future]] = future.result()

        for result in raw_results:
            if result:
                doc_index, _, keyphrases = result
                keyphrases_map[doc_index] = keyphrases

        with open(cache_file, "wb") as f:
            pickle.dump(keyphrases_map, f)
        print(f"  Keyphrases cached to: {cache_file}")

        pd.DataFrame(
            [
                {
                    "document_index": i,
                    "document_text": documents[i],
                    "generated_keyphrases": ", ".join(keyphrases_map.get(i, [])),
                }
                for i in range(n_samples)
            ]
        ).to_csv(keyphrase_output_csv_path, index=False)

    # Docs whose keyphrase query failed are left unassigned (-1).
    successful_indices = [i for i in range(n_samples) if keyphrases_map.get(i)]
    if not successful_indices:
        return {"concatenated": None, "average": None}

    print(f"  Embedding keyphrase expansions ({len(successful_indices)} docs)...")
    expanded_texts = [
        ", ".join([documents[i]] + keyphrases_map[i]) for i in successful_indices
    ]
    exp_arr = llm_service.get_embeddings(expanded_texts)
    orig_arr = normalize(features[successful_indices], axis=1, norm="l2")
    full_assignments = np.full(n_samples, -1, dtype=int)

    def run_clustering(feat: np.ndarray) -> Optional[np.ndarray]:
        try:
            clusters = KMeans(
                n_clusters=n_clusters, random_state=0, n_init="auto"
            ).fit_predict(feat)
            result = full_assignments.copy()
            for i, idx in enumerate(successful_indices):
                result[idx] = clusters[i]
            return result
        except Exception:
            return None

    cluster_results: Dict[str, Optional[np.ndarray]] = {}
    cluster_results["concatenated"] = run_clustering(np.hstack([orig_arr, exp_arr]))
    cluster_results["average"] = run_clustering((orig_arr + exp_arr) / 2)

    for weight in [round(w, 1) for w in np.arange(0.1, 1.1, 0.1)]:
        cluster_results[f"weighted_{weight}"] = run_clustering(
            (1 - weight) * orig_arr + weight * exp_arr
        )

    return cluster_results
