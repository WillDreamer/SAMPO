from huggingface_hub import create_repo, upload_folder

repo_name = "xxxxx"
repo_id = repo_name  
create_repo(repo_id, exist_ok=True)

upload_folder(
    repo_id=repo_id,
    folder_path="xxxxxx/checkpoints/Qwen3-4B-alfworld-e1",  
    path_in_repo=".",                
)
