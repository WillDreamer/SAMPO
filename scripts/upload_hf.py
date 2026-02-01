# python -m verl.model_merger merge \
#     --backend fsdp \
#     --local_dir /workspace/AgentRL/sft_ckpt/global_step_56 \
#     --target_dir /workspace/Qwen-4B-webshop


from huggingface_hub import create_repo, upload_folder

# 设置模型名称（在你 HuggingFace 账号下）
repo_name = "willamazon1/Qwen3-4B-webshop-gpt"

# 可选：如果模型属于某个组织（例如 huggingface/your-org-model）
# repo_id = "your-org/my-sft-model"
repo_id = repo_name  # 若是上传到你自己账号

# 创建仓库（自动跳过已存在）
create_repo(repo_id, exist_ok=True)

# 上传整个模型文件夹
upload_folder(
    repo_id=repo_id,
    folder_path="/workspace/Qwen3-4B-webshop",  # 你的本地模型路径
    path_in_repo=".",                # 上传到仓库的根目录
)
