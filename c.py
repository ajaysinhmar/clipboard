def parse_java(file_path):
    """Parse Java files using javalang."""
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            content = f.read()

        tree = javalang.parse.parse(content)
        func_defs = []
        func_calls = []
        MAX_IMPORT_DEPTH = 1
        imports = []
        
        # Process imports with depth limit
        for imp in tree.imports:
            path_parts = imp.path.split('.')
            if len(path_parts) <= MAX_IMPORT_DEPTH:
                imports.append(imp.path)

        for _, node in tree.filter(javalang.tree.MethodDeclaration):
            name = node.name
            if node.body:
                start_line = node.position.line
                # Fetch the end line using the last statement in the body
                end_line = node.body[-1].position.line if node.body[-1].position else start_line
                func_body = "\n".join(content.splitlines()[start_line - 1:end_line])
                func_defs.append((name, func_body))
            else:
                func_defs.append((name, "{}"))

        for _, node in tree.filter(javalang.tree.MethodInvocation):
            func_calls.append(node.member)

        return func_defs, func_calls, imports

    except Exception as e:
        print(f"[ERROR] Parsing Java file failed for {file_path}: {e}")
        return [], [], []