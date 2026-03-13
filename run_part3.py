import argparse
from pathlib import Path
import numpy as np
from src.retrieval_extended import PaperRetrieverExtended
from src.part2_utils import load_jsonl
from src.feedback_logic import apply_rocchio
from src.qa_engine import QAGenerator

def normalize_id(paper_id: str) -> str:
    """
    Standardize ArXiv IDs by removing 'arxiv:' prefix and version suffixes like 'v1'.
    """
    pid = str(paper_id).lower().strip()
    if pid.startswith("arxiv:"):
        pid = pid.replace("arxiv:", "")
    if 'v' in pid:
        pid = pid.split('v')[0]
    return pid

def main():
    parser = argparse.ArgumentParser(description="Part 3: Relevance Feedback & RAG")
    parser.add_argument("--queries", default="data/queries_val.jsonl")
    parser.add_argument("--rounds", type=int, default=1, help="Number of feedback iterations")
    parser.add_argument("--top_k", type=int, default=5, help="Number of documents to retrieve")
    parser.add_argument("--use_pseudo", action="store_true", help="Enable PRF if no ground truth is found")
    args = parser.parse_args()

    # Initialize Retrieval and QA Engine
    # Reminder: Must be on UM VPN for the GPT-oss-120B server
    retriever = PaperRetrieverExtended()
    qa_gen = QAGenerator()
    query_rows = load_jsonl(Path(args.queries))

    print(f"Executing Part 3 evaluation on {len(query_rows)} queries...")

    for row in query_rows:
        query_text = row["query_text"]
        relevant_ids = {normalize_id(idx) for idx in row.get("relevant_arxiv_ids", [])}
        
        print(f"\n{'='*80}")
        print(f"QUERY: {query_text}")
        print(f"{'='*80}")
        
        # 1. Generate initial query vector
        current_vec = retriever.encode_query(query_text)
        
        final_results = []
        for r in range(args.rounds + 1):
            # 2. Retrieve papers
            results = retriever.retrieve_by_vector(current_vec, k=args.top_k)
            final_results = results
            
            # Calculate Precision based on normalized IDs
            hits = [res for res in results if normalize_id(res['arxiv_id']) in relevant_ids]
            precision = len(hits) / args.top_k
            print(f"Round {r} | Precision@{args.top_k}: {precision:.2f}")

            # 3. Apply Rocchio Feedback Logic
            if r < args.rounds:
                if hits:
                    # Standard Rocchio using ground truth hits
                    pos_vecs = [retriever.dense_embeddings[retriever.paper_ids.index(h['paper_id'])] for h in hits]
                    current_vec = apply_rocchio(current_vec, pos_vecs)
                    print(f"  --> Vector updated using {len(hits)} relevant papers.")
                elif args.use_pseudo:
                    # Pseudo-Relevance Feedback (PRF) for demonstration
                    print(f"  --> Applying Pseudo-Feedback using the top-ranked document.")
                    top_1_id = results[0]['paper_id']
                    top_1_vec = retriever.dense_embeddings[retriever.paper_ids.index(top_1_id)]
                    current_vec = apply_rocchio(current_vec, [top_1_vec])

        # 4. Generate grounded answer using retrieved context and UM Server
        print("\n[RAG] Generating answer via UM GPT-oss-120B...")
        context = qa_gen.format_context(final_results)
        answer = qa_gen.generate_answer(query_text, context)
        print(f"\n[AI Answer]:\n{answer}\n")

if __name__ == "__main__":
    main()