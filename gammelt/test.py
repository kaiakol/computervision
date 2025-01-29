import os

# Define the folder where the files are located
folder_path = 'data/pushup/incorrect'

# Get a list of all files in the folder
files = os.listdir(folder_path)

# Filter out directories (optional)
files = [f for f in files if os.path.isfile(os.path.join(folder_path, f))]

# Define the new naming format
for idx, old_name in enumerate(files):
    # Create a new file name with a specific format
    # For example, 'file_001.txt'
    new_name = f'pushup_incorrect_{idx + 1:03d}{os.path.splitext(old_name)[1]}'

    # Build the full path for old and new filenames
    old_path = os.path.join(folder_path, old_name)
    new_path = os.path.join(folder_path, new_name)

    # Rename the file
    os.rename(old_path, new_path)

    print(f'Renamed: {old_name} -> {new_name}')
