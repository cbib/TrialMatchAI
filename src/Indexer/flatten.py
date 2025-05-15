import os
import shutil

def flatten_folder_structure(root_dir, output_dir, separator="_", handle_duplicates=True):
    os.makedirs(output_dir, exist_ok=True)
    filename_set = set()

    for subdir, dirs, files in os.walk(root_dir):
        # Skip the root directory itself
        if subdir == root_dir:
            continue
        
        parent_folder = os.path.basename(subdir)
        
        for file in files:
            old_path = os.path.join(subdir, file)
            new_filename = f"{parent_folder}{separator}{file}"
            new_path = os.path.join(output_dir, new_filename)

            if handle_duplicates:
                base, ext = os.path.splitext(new_filename)
                counter = 1
                while new_path in filename_set or os.path.exists(new_path):
                    new_filename = f"{base}_{counter}{ext}"
                    new_path = os.path.join(output_dir, new_filename)
                    counter += 1

            filename_set.add(new_path)
            shutil.copy2(old_path, new_path)
            print(f"Copied: {old_path} → {new_path}")

# Example usage:
root_directory = "/home/mabdallah/scratch/TrialMatchAI/src/Indexer/processed_criteria"
output_directory = "/home/mabdallah/scratch/TrialMatchAI/src/Indexer/processed_criteria_flattened"
flatten_folder_structure(root_directory, output_directory)
