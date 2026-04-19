import os

OUTPUT_FILE = "django_project_dump.txt"

# file types we care about
INCLUDE_EXTENSIONS = {".py", ".html", ".txt", ".md"}

# folders to skip
EXCLUDE_DIRS = {"venv", "__pycache__", ".git", "node_modules", "migrations"}

def should_include_file(filename):
    return any(filename.endswith(ext) for ext in INCLUDE_EXTENSIONS)

def get_project_structure(root_dir):
    structure = []
    for root, dirs, files in os.walk(root_dir):
        # remove excluded dirs
        dirs[:] = [d for d in dirs if d not in EXCLUDE_DIRS]

        level = root.replace(root_dir, "").count(os.sep)
        indent = " " * 4 * level
        structure.append(f"{indent}{os.path.basename(root)}/")

        sub_indent = " " * 4 * (level + 1)
        for f in files:
            structure.append(f"{sub_indent}{f}")

    return "\n".join(structure)

def dump_files(root_dir):
    content = []

    for root, dirs, files in os.walk(root_dir):
        dirs[:] = [d for d in dirs if d not in EXCLUDE_DIRS]

        for file in files:
            if should_include_file(file):
                file_path = os.path.join(root, file)

                try:
                    with open(file_path, "r", encoding="utf-8") as f:
                        file_content = f.read()

                    content.append(f"\n{'='*80}")
                    content.append(f"FILE: {file_path}")
                    content.append(f"{'='*80}\n")
                    content.append(file_content)

                except Exception as e:
                    content.append(f"\nCould not read {file_path}: {e}")

    return "\n".join(content)

def main():
    root_dir = os.getcwd()

    with open(OUTPUT_FILE, "w", encoding="utf-8") as out:
        out.write("PROJECT STRUCTURE\n")
        out.write("="*80 + "\n")
        out.write(get_project_structure(root_dir))
        out.write("\n\nFILE CONTENTS\n")
        out.write("="*80 + "\n")
        out.write(dump_files(root_dir))

    print(f"\n✅ Done! Output saved to {OUTPUT_FILE}")

if __name__ == "__main__":
    main()