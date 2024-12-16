import kagglehub

kagglehub.login()

# Download latest version
path = kagglehub.model_download("google/movenet/tfLite/singlepose-lightning")

print("Path to model files:", path)