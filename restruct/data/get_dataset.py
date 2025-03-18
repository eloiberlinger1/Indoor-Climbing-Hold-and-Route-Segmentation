import kagglehub

# Download latest version
path = kagglehub.dataset_download("tomasslama/indoor-climbing-gym-hold-segmentation")

print("Path to dataset files:", path)