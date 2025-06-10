import os
import re
import ast
import networkx as nx
from collections import defaultdict
import glob

SUPPORTED_EXTENSIONS = [".py", ".go", ".js", ".ts", ".java"]

# Function to parse Python files
def parse_python(file_path):
    with open(file_path, "r", encoding="utf-8", errors="ignore") as f:
        tree = ast.parse(f.read())
    func_defs = []
    func_calls = []
    imports = []

    for node in ast.walk(tree):
        if isinstance(node, ast.FunctionDef):
            func_defs.append(node.name)
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

# Function to parse Go/JavaScript/TypeScript/Java files using regex
def parse_generic(file_path, lang):
    with open(file_path, "r", encoding="utf-8", errors="ignore") as f:
        content = f.read()

    if lang in ["go", "js", "ts", "java"]:
        func_defs = re.findall(r"(?:func|function|def|\bvoid\b|\bint\b|\bString\b)\s+(\w+)\s*\(", content)
        func_calls = re.findall(r"\b(\w+)\s*\(", content)
        imports = re.findall(r"(?:import|require)\s+["']?([\w\.\-/]+)["']?", content)
        return func_defs, func_calls, imports
    return [], [], []

# Unified parsing function
def parse_file(file_path):
    ext = os.path.splitext(file_path)[1]
    if ext == ".py":
        return parse_python(file_path)
    elif ext in [".go", ".js", ".ts", ".java"]:
        return parse_generic(file_path, ext[1:])
    return [], [], []

# Function to create a knowledge graph
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

                # Add functions as nodes
                for func in func_defs:
                    G.add_node(func, type="function", file=file_path)

                # Add function calls as edges
                for call in func_calls:
                    if call in func_defs:  # Only add calls within the known functions
                        G.add_edge(func, call, relation="calls")

                # Add imports as nodes and edges
                for imp in imports:
                    G.add_node(imp, type="import", file=file_path)
                    for func in func_defs:
                        G.add_edge(imp, func, relation="imports")

                # Chunk functions by keeping them unbroken
                for func in func_defs:
                    chunks[file_path].append(func)

    return G, chunks

# Example usage
ROOT_DIR = "/path/to/cloned/repos"  # Change to the actual root directory
knowledge_graph, function_chunks = create_knowledge_graph(ROOT_DIR)

# Print summary of the knowledge graph
print(f"Knowledge Graph: {knowledge_graph.number_of_nodes()} nodes, {knowledge_graph.number_of_edges()} edges")
for file, funcs in function_chunks.items():
    print(f"Chunks for {file}: {funcs}")
    
    
    
    
    
    
from sentence_transformers import SentenceTransformer
import numpy as np

# Load embedding model
embedding_model = SentenceTransformer("all-MiniLM-L6-v2")

def generate_enriched_chunks(graph, chunks):
    enriched_chunks = []
    for file_path, funcs in chunks.items():
        for func in funcs:
            # Enrich each function node with context from the KG
            if graph.has_node(func):
                calls = list(graph.successors(func))
                imports = [n for n in graph.predecessors(func) if graph.nodes[n].get('type') == 'import']
                enriched_text = (
                    f"Function: {func}\n"
                    f"File: {file_path}\n"
                    f"Calls: {', '.join(calls)}\n"
                    f"Imports: {', '.join(imports)}\n"
                    "---\n"
                    f"{func} implementation here..."
                )
                enriched_chunks.append(enriched_text)
    return enriched_chunks

def generate_embeddings(enriched_chunks):
    # Generate embeddings for each enriched chunk
    embeddings = embedding_model.encode(enriched_chunks, convert_to_tensor=True)
    return embeddings

# Generate enriched chunks using KG
enriched_chunks = generate_enriched_chunks(knowledge_graph, function_chunks)

# Generate embeddings
embeddings = generate_embeddings(enriched_chunks)

# Example query embedding for comparison
query = "How is an order processed in the order service?"
query_embedding = embedding_model.encode(query, convert_to_tensor=True)

# Find most relevant chunks (using cosine similarity or dot product)
similarity_scores = np.dot(embeddings, query_embedding)
top_k_indices = np.argsort(similarity_scores)[::-1][:5]
top_k_chunks = [enriched_chunks[i] for i in top_k_indices]

# Output top chunks
for chunk in top_k_chunks:
    print(chunk)
