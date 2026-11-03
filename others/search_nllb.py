import os

def search_files(directory):
    print(f"Scanning directory: {directory}")
    nllb_files = []
    all_files_count = 0
    for root, dirs, files in os.walk(directory):
        for file in files:
            all_files_count += 1
            full_path = os.path.join(root, file)
            if "nllb" in file.lower() or "nllb" in root.lower():
                nllb_files.append(full_path)
    
    print(f"Total files scanned: {all_files_count}")
    print("Found NLLB-related files:")
    for f in nllb_files:
        print(f" - {f} ({os.path.getsize(f)} bytes)")

if __name__ == "__main__":
    search_files(r"d:\New models Lokesh sir")
