import os

def list_files_in_directory(folder_path):
    try:
        # Check if the folder exists
        if not os.path.exists(folder_path):
            print(f"The folder '{folder_path}' does not exist.")
            return

        # List all files and directories in the folder
        entries = os.listdir(folder_path)
        files = [entry for entry in entries if os.path.isfile(os.path.join(folder_path, entry))]

        print(f"Files in '{folder_path}':")
        if files:
            for file in files:
                print(file)
        else:
            print("No files found in the folder.")
    except Exception as e:
        print(f"An error occurred: {e}")

# Example usage
# folder_path = input("Enter the folder path: ")
list_files_in_directory("/home/madhu/lucene_code/trec_data/lucene_data/trec678rb/documents")
