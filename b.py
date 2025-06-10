import os
import ast
import networkx as nx
import chromadb
from chromadb.config import Settings
from sentence_transformers import SentenceTransformer
from collections import defaultdict
import re
import glob
import json

# Define supported file extensions
SUPPORTED_EXTENSIONS = [".py", ".go", ".js", ".ts", ".java"]

# Initialize ChromaDB client
chroma_client = chromadb.Client(Settings(persist_directory="./chromadb", chroma_db_impl="duckdb"))

# Create a Chroma collection for embeddings
collection = chroma_client.get_or_create_collection("code_embeddings")

# Load embedding model
embedding_model = SentenceTransformer("all-MiniLM-L6-v2")

# Parse Python files using AST
def parse_python(file_path):
    with open(file_path, "r", encoding="utf-8", errors="ignore") as f:
        tree = ast.parse(f.read())
    func_defs = []
    func_calls = []
    imports = []

    for node in ast.walk(tree):
        if isinstance(node, ast.FunctionDef):
            func_defs.append((node.name, ast.unparse(node)))
        elif isinstance(node, ast.Call) and isinstance(node.func, ast.Name):
            func_calls.append(node.func.id)
        elif isinstance(node, ast.Import):
            for alias in node.names:
                imports.append(alias.name)
        elif isinstance(node, ast.ImportFrom):
            module = node.module or ""
            for alias in node.names:
                imports.append(f"{module}.{alias.name}")

    return func_defs, func_calls, imports

# Parse other languages (Go, JS, TS, Java) using regex
def parse_generic(file_path, lang):
    with open(file_path, "r", encoding="utf-8", errors="ignore") as f:
        content = f.read()

    func_defs = re.findall(r"(?:func|function|\bdef\b|void|\bint\b|\bString\b)\s+(\w+)\s*\(.*?\).*?{", content)
    func_calls = re.findall(r"\b(\w+)\s*\(", content)
    imports = re.findall(r"(?:import|require)\s+["']?([\w\.\-/]+)["']?", content)

    return [(func, None) for func in func_defs], func_calls, imports

# Unified parsing function
def parse_file(file_path):
    ext = os.path.splitext(file_path)[1]
    if ext == ".py":
        return parse_python(file_path)
    elif ext in [".go", ".js", ".ts", ".java"]:
        return parse_generic(file_path, ext[1:])
    return [], [], []

# Create a knowledge graph (KG) from the repositories
def create_knowledge_graph(root_dir):
    G = nx.DiGraph()
    chunks = defaultdict(list)

    for repo_name in os.listdir(root_dir):
        repo_path = os.path.join(root_dir, repo_name)
        if not os.path.isdir(repo_path):
            continue

        for file_path in glob.glob(os.path.join(repo_path, "**"), recursive=True):
            if os.path.splitext(file_path)[1] in SUPPORTED_EXTENSIONS:
                file_path = os.path.abspath(file_path)
                func_defs, func_calls, imports = parse_file(file_path)

                for func_name, func_body in func_defs:
                    # Add function nodes
                    G.add_node(func_name, type="function", body=func_body, file=file_path, repo=repo_name)
                    chunks[file_path].append({"name": func_name, "body": func_body})

                # Add function call edges
                for call in func_calls:
                    if call in G.nodes:
                        G.add_edge(func_name, call, relation="calls")

                # Add imports as nodes and edges
                for imp in imports:
                    G.add_node(imp, type="import", file=file_path, repo=repo_name)
                    for func_name, _ in func_defs:
                        G.add_edge(imp, func_name, relation="imports")

    return G, chunks

# Generate enriched embeddings
def generate_enriched_chunks(graph, chunks):
    enriched_chunks = []

    for file_path, func_list in chunks.items():
        for func in func_list:
            func_name = func["name"]
            func_body = func["body"]
            repo = graph.nodes[func_name].get("repo")
            calls = list(graph.successors(func_name))
            imports = [n for n in graph.predecessors(func_name) if graph.nodes[n].get('type') == 'import']

            enriched_text = (
                f"Function: {func_name}\n"
                f"File: {file_path}\n"
                f"Repo: {repo}\n"
                f"Calls: {', '.join(calls)}\n"
                f"Imports: {', '.join(imports)}\n"
                f"---\n"
                f"{func_body}"
            )

            enriched_chunks.append({
                "text": enriched_text,
                "metadata": {
                    "function": func_name,
                    "file": file_path,
                    "repo": repo,
                    "calls": calls,
                    "imports": imports
                }
            })

    return enriched_chunks

# Generate embeddings and store them in ChromaDB
def store_embeddings_in_chromadb(enriched_chunks):
    for chunk in enriched_chunks:
        text = chunk["text"]
        metadata = chunk["metadata"]
        embedding = embedding_model.encode(text, convert_to_tensor=False)

        # Add to ChromaDB
        collection.add(
            embeddings=[embedding.tolist()],
            metadatas=[metadata],
            documents=[text]
        )

# Main workflow
if __name__ == "__main__":
    ROOT_DIR = "/path/to/cloned/repos"  # Change to actual root directory

    print("Creating Knowledge Graph...")
    knowledge_graph, function_chunks = create_knowledge_graph(ROOT_DIR)

    print("Generating Enriched Chunks...")
    enriched_chunks = generate_enriched_chunks(knowledge_graph, function_chunks)

    print("Storing Embeddings in ChromaDB...")
    store_embeddings_in_chromadb(enriched_chunks)

    print("Process Completed. Embeddings and metadata stored in ChromaDB.")
