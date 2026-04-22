from openai import OpenAI
from typing import Dict, List


class QAGenerator:
    def __init__(self):
        # Ensure you are on UM VPN to access this server.
        self.client = OpenAI(
            base_url="http://promaxgb10-d473.eecs.umich.edu:8000/v1",
            api_key="api_IcLlffdxoWOSgBPWW3X3zS15YSBHim5a"
        )
        self.model_name = "openai/gpt-oss-120b"

    def format_context(self, results: List[Dict]) -> str:
        """Combine paper abstracts or chunk-level evidence into a single context string."""
        context_blocks = []
        for i, res in enumerate(results):
            if "chunk_text" in res:
                block = (
                    f"Source [{i+1}] (ID: {res['arxiv_id']}, chunk {res.get('chunk_index', '?')}):\n"
                    f"Title: {res.get('title', '')}\n"
                    f"Evidence: {res['chunk_text']}"
                )
            else:
                block = (
                    f"Source [{i+1}] (ID: {res['arxiv_id']}):\n"
                    f"Title: {res['title']}\n"
                    f"Abstract: {res['abstract']}"
                )
            context_blocks.append(block)
        return "\n\n".join(context_blocks)

    def generate_answer(self, query: str, context: str) -> str:
        """Generate a grounded answer using the provided evidence."""
        prompt = f"""You are a research assistant. Answer the user's question based ONLY on the provided arXiv evidence.
If the answer is not in the context, say "I do not have enough information."
Always cite your sources using [Source X] notation at the end of relevant sentences.

CONTEXT:
{context}

QUESTION:
{query}

ANSWER:"""

        try:
            response = self.client.chat.completions.create(
                model=self.model_name,
                messages=[{"role": "user", "content": prompt}],
                max_tokens=800,
                temperature=0.1,
            )
            return response.choices[0].message.content
        except Exception as e:
            return f"Error connecting to UM Server: {str(e)}"
