import os

def remove_zone_identifier_files(folder_path):
    try:
        for root, _, files in os.walk(folder_path):
            for file in files:
                if file.endswith(":Zone.Identifier"):
                    full_path = os.path.join(root, file)
                    try:
                        os.remove(full_path)
                        print(f"Removed: {full_path}")
                    except Exception as e:
                        print(f"Could not remove {full_path}: {e}")
    except Exception as e:
        print(f"An error occurred: {e}")

# Example usage
# folder_path = input("Enter the folder path: ")
remove_zone_identifier_files("/home/madhu/lucene_code/trec_data/lucene_data/trec678rb/documents")
