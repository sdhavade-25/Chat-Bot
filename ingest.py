"""Build the vector index from a folder of documents.

Run:  python ingest.py                     # index docs/
      python ingest.py --docs sample_docs  # index the bundled fictional documents
"""
import argparse

import rag

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Build the vector index from a folder of documents.")
    parser.add_argument("--docs", default=str(rag.DOCS_DIR),
                        help="folder to read .pdf / .md / .txt files from (default: docs/)")
    args = parser.parse_args()
    print(f"Indexed {rag.ingest(args.docs)} chunks from {args.docs}")
