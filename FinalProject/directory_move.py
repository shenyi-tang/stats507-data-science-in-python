import kagglehub
import shutil
import os


# Download latest version
path = kagglehub.dataset_download("odins0n/top-20-play-store-app-reviews-daily-update")

print("Path to dataset files:", path)

# remove the dataset from the download directory to specified directory
current_path = path
destination_path = "/Users/tshenyi/UMich/2024fall/stats507/stats507-data-science-in-python/FinalProject/data"

shutil.move(current_path, destination_path)
print(f"Current path: {destination_path}")

# delete the sub folder
parent_folder = "/Users/tshenyi/UMich/2024fall/stats507/stats507-data-science-in-python/FinalProject/data"
sub_folder = "/Users/tshenyi/UMich/2024fall/stats507/stats507-data-science-in-python/FinalProject/data/761"

if os.path.exists(sub_folder):
    # Traverse through all files under the sub_folder
    for file_name in os.listdir(sub_folder):
        file_path = os.path.join(sub_folder, file_name)

        # make sure the object is a file
        if os.path.isfile(file_path):
            shutil.move(file_path, parent_folder)

    os.rmdir(sub_folder)

