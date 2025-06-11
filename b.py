import os
import ast
import networkx as nx
from langchain.vectorstores import Chroma
from langchain.embeddings import HuggingFaceEmbeddings
from collections import defaultdict
import re
import glob

# Define supported file extensions
SUPPORTED_EXTENSIONS = [".py", ".go", ".js", ".ts", ".java"]

# Load embedding model
embedding_model = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")

# Initialize Chroma with LangChain's interface
chroma_db = Chroma(embedding_function=embedding_model, persist_directory="./chromadb")

# Parse Python files using AST
def parse_python(file_path):
    with open(file_path, "r", encoding="utf-8", errors="ignore") as f:
        source_code = f.read()
        tree = ast.parse(source_code)

    func_defs = []
    func_calls = []
    imports = []

    for node in ast.walk(tree):
        if isinstance(node, ast.FunctionDef):
            # Extract function body using ast.get_source_segment
            func_body = ast.get_source_segment(source_code, node) or "No body found"
            func_defs.append((node.name, func_body.strip()))
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

    # Regex to capture the entire function, including body
    func_body_pattern = r"(?:func|function|\bdef\b|void|\bint\b|\bString\b)\s+(\w+)\s*\(.*?\)\s*{(.*?)}"
    func_defs = re.findall(func_body_pattern, content, re.DOTALL)
    func_calls = re.findall(r"\b(\w+)\s*\(", content)
    imports = re.findall(r"(?:import|require)\s+['\"]?([\w\.\-/]+)['\"]?", content)

    return [(func.strip(), body.strip()) for func, body in func_defs], func_calls, imports

# Unified parsing function
def parse_file(file_path):
    ext = os.path.splitext(file_path)[1]
    if ext == ".py":
        return parse_python(file_path)
    elif ext in [".go", ".js", ".ts", ".java"]:
        return parse_generic(file_path, ext[1:])
    return [], [], []

def is_test_file(file_path):
    # Check for common test file patterns
    file_name = os.path.basename(file_path).lower()
    if "test" in file_name or file_name.startswith("test_") or file_name.endswith("_test.py"):
        return True

    # Check if the file is in a test-related directory
    test_directories = ["tests", "__tests__", "test"]
    path_parts = [part.lower() for part in file_path.split(os.sep)]
    if any(test_dir in path_parts for test_dir in test_directories):
        return True

    return False

# Create a knowledge graph (KG) from the repositories
def create_knowledge_graph(root_dir):
    G = nx.DiGraph()
    chunks = defaultdict(list)

    for repo_name in os.listdir(root_dir):
        repo_path = os.path.join(root_dir, repo_name)
        if not os.path.isdir(repo_path):
            continue

        for file_path in glob.glob(os.path.join(repo_path, "**"), recursive=True):
            if (
                os.path.isfile(file_path)
                and os.path.splitext(file_path)[1] in SUPPORTED_EXTENSIONS
                and not is_test_file(file_path)  # Skip test files
            ):
                file_path = os.path.abspath(file_path)
                func_defs, func_calls, imports = parse_file(file_path)

                for func_name, func_body in func_defs:
                    # Add function nodes with full body
                    G.add_node(func_name, type="function", body=func_body, file=file_path, repo=repo_name)
                    chunks[file_path].append({"name": func_name, "body": func_body})

                for call in func_calls:
                    for func_name, _ in func_defs:
                        if func_name in G.nodes:
                            G.add_edge(func_name, call, relation="calls")

                for imp in imports:
                    G.add_node(imp, type="import", file=file_path, repo=repo_name)
                    for func_name, _ in func_defs:
                        if func_name in G.nodes:
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

# Store embeddings in ChromaDB
def store_embeddings_in_chromadb(enriched_chunks):
    for chunk in enriched_chunks:
        text = chunk["text"]
        metadata = chunk["metadata"]

        # Add to ChromaDB
        chroma_db.add_texts(
            texts=[text],
            metadatas=[metadata]
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
