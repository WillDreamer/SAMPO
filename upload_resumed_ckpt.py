#!/usr/bin/env python3
"""
Script to upload complete AgentRL checkpoint to Hugging Face Hub for training continuation
This preserves all training state including optimizer states, learning rates, etc.
"""

import os
import sys
import shutil
import tempfile
import json
from pathlib import Path
import torch
from huggingface_hub import HfApi, create_repo, upload_folder, upload_file
import argparse
import tarfile

def create_checkpoint_archive(checkpoint_dir, output_path, compress=False):
    """
    Create an archive of the complete checkpoint (optionally compressed)
    """
    print("📦 Creating checkpoint archive...")
    
    # Use uncompressed tar for speed, or compressed if requested
    mode = 'w:gz' if compress else 'w'
    compression_info = " (compressed)" if compress else " (uncompressed for speed)"
    print(f"Archive mode: {mode}{compression_info}")
    
    # Get all files first and show progress
    all_files = []
    total_size = 0
    for root, dirs, files in os.walk(checkpoint_dir):
        for file in files:
            file_path = os.path.join(root, file)
            file_size = os.path.getsize(file_path)
            total_size += file_size
            arcname = os.path.relpath(file_path, checkpoint_dir)
            all_files.append((file_path, arcname, file_size))
    
    print(f"Found {len(all_files)} files, total size: {total_size / (1024**3):.1f} GB")
    
    with tarfile.open(output_path, mode) as tar:
        for i, (file_path, arcname, file_size) in enumerate(all_files, 1):
            print(f"  [{i}/{len(all_files)}] Adding: {arcname} ({file_size / (1024**3):.1f} GB)")
            tar.add(file_path, arcname=arcname)
    
    final_size = os.path.getsize(output_path)
    print(f"✅ Archive created: {output_path}")
    print(f"   Original size: {total_size / (1024**3):.1f} GB")
    print(f"   Archive size: {final_size / (1024**3):.1f} GB")
    if compress:
        print(f"   Compression ratio: {(1 - final_size/total_size)*100:.1f}%")
    return True


def upload_complete_checkpoint(checkpoint_dir, repo_name, token=None, private=False, use_archive=True):
    """
    Upload complete checkpoint to Hugging Face Hub
    
    Args:
        use_archive: If True, create tar archive. If False, upload files directly.
    """
    print(f"🚀 Uploading complete checkpoint to: {repo_name}")
    print(f"Upload method: {'Archive' if use_archive else 'Direct files'}")
    
    # Initialize HF API
    api = HfApi(token=token)
    
    try:
        # Create repository
        print("📝 Creating repository...")
        create_repo(repo_name, token=token, private=private, exist_ok=True, repo_type="model")
        
        if use_archive:
            # Method 1: Create tar archive and upload
            with tempfile.TemporaryDirectory() as temp_dir:
                # Create checkpoint archive (uncompressed for speed)
                archive_path = os.path.join(temp_dir, "checkpoint.tar")
                create_checkpoint_archive(checkpoint_dir, archive_path, compress=False)
                
                # Copy huggingface directory for inference
                hf_source = os.path.join(checkpoint_dir, "huggingface")
                if os.path.exists(hf_source):
                    hf_dest = os.path.join(temp_dir, "huggingface")
                    shutil.copytree(hf_source, hf_dest)
                    print("📋 Copied Hugging Face inference files")
                
                # Upload everything
                print("⬆️ Uploading files...")
                upload_folder(
                    folder_path=temp_dir,
                    repo_id=repo_name,
                    token=token,
                    commit_message="Upload complete AgentRL training checkpoint"
                )
        else:
            # Method 2: Upload files directly (faster, no local storage needed)
            print("⬆️ Uploading files directly...")
            
            # Create a temporary structure
            with tempfile.TemporaryDirectory() as temp_dir:
                # Create checkpoint subdirectory
                checkpoint_temp = os.path.join(temp_dir, "checkpoint")
                shutil.copytree(checkpoint_dir, checkpoint_temp)
                
                # Copy huggingface directory for inference
                hf_source = os.path.join(checkpoint_dir, "huggingface")
                if os.path.exists(hf_source):
                    hf_dest = os.path.join(temp_dir, "huggingface")
                    shutil.copytree(hf_source, hf_dest)
                    print("📋 Copied Hugging Face inference files")
                
                # Upload the entire structure
                upload_folder(
                    folder_path=temp_dir,
                    repo_id=repo_name,
                    token=token,
                    commit_message="Upload complete AgentRL training checkpoint (direct files)"
                )
        
        print(f"✅ Successfully uploaded to https://huggingface.co/{repo_name}")
        print("\n📖 Usage Instructions:")
        if use_archive:
            print("For training continuation:")
            print(f"  wget https://huggingface.co/{repo_name}/resolve/main/checkpoint.tar")
            print("  tar -xf checkpoint.tar")
        else:
            print("For training continuation:")
            print(f"  git clone https://huggingface.co/{repo_name}")
            print(f"  # Use the checkpoint/ subdirectory")
        print("\nFor inference only:")
        print(f"  model = AutoModelForCausalLM.from_pretrained('{repo_name}', subfolder='huggingface')")
        
        return True
        
    except Exception as e:
        print(f"❌ Upload failed: {str(e)}")
        return False

def download_and_extract_checkpoint(repo_name, output_dir, token=None, method="auto"):
    """
    Helper function to download checkpoint for training continuation
    
    Args:
        method: "auto", "git", "archive", or "hf_hub"
    """
    print(f"⬇️ Downloading checkpoint from {repo_name}")
    print(f"Download method: {method}")
    
    try:
        if method == "git" or method == "auto":
            # Method 1: Use git clone (best for direct file uploads)
            print("Using git clone method...")
            import subprocess
            
            clone_dir = os.path.join(output_dir, repo_name.split('/')[-1])
            
            # Build git clone command
            if token:
                # Use token for private repos
                repo_url = f"https://{token}@huggingface.co/{repo_name}"
            else:
                repo_url = f"https://huggingface.co/{repo_name}"
            
            cmd = ["git", "clone", repo_url, clone_dir]
            
            result = subprocess.run(cmd, capture_output=True, text=True)
            
            if result.returncode == 0:
                checkpoint_dir = os.path.join(clone_dir, "checkpoint")
                if os.path.exists(checkpoint_dir):
                    print(f"✅ Checkpoint downloaded to: {checkpoint_dir}")
                    print("🎯 You can now use this checkpoint for training continuation")
                    return checkpoint_dir
                else:
                    print(f"✅ Repository cloned to: {clone_dir}")
                    print("📁 Checkpoint files are in the root directory")
                    return clone_dir
            else:
                print(f"Git clone failed: {result.stderr}")
                if method == "auto":
                    print("Falling back to archive method...")
                    return download_and_extract_checkpoint(repo_name, output_dir, token, "archive")
                return None
                
        elif method == "archive":
            # Method 2: Download tar archive (for archive uploads)
            print("Using archive download method...")
            from huggingface_hub import hf_hub_download
            
            archive_path = hf_hub_download(
                repo_id=repo_name,
                filename="checkpoint.tar",
                token=token,
                local_dir=output_dir
            )
            
            # Extract archive
            extract_dir = os.path.join(output_dir, "checkpoint")
            os.makedirs(extract_dir, exist_ok=True)
            
            with tarfile.open(archive_path, 'r') as tar:
                tar.extractall(extract_dir)
            
            print(f"✅ Checkpoint extracted to: {extract_dir}")
            print("🎯 You can now use this checkpoint for training continuation")
            return extract_dir
            
        elif method == "hf_hub":
            # Method 3: Download individual files using HF Hub
            print("Using Hugging Face Hub download method...")
            from huggingface_hub import snapshot_download
            
            download_dir = snapshot_download(
                repo_id=repo_name,
                local_dir=os.path.join(output_dir, repo_name.split('/')[-1]),
                token=token
            )
            
            checkpoint_dir = os.path.join(download_dir, "checkpoint")
            if os.path.exists(checkpoint_dir):
                print(f"✅ Checkpoint downloaded to: {checkpoint_dir}")
                return checkpoint_dir
            else:
                print(f"✅ Repository downloaded to: {download_dir}")
                return download_dir
        
    except Exception as e:
        print(f"❌ Download failed: {str(e)}")
        if method == "auto":
            print("Trying alternative download method...")
            return download_and_extract_checkpoint(repo_name, output_dir, token, "hf_hub")
        return None

def main():
    parser = argparse.ArgumentParser(description="Upload/Download AgentRL checkpoint for training continuation")
    subparsers = parser.add_subparsers(dest='command', help='Available commands')
    
    # Upload command
    upload_parser = subparsers.add_parser('upload', help='Upload checkpoint to Hugging Face')
    upload_parser.add_argument("--checkpoint_dir", required=True, help="Path to checkpoint directory")
    upload_parser.add_argument("--repo_name", required=True, help="Hugging Face repository name")
    upload_parser.add_argument("--token", help="Hugging Face token")
    upload_parser.add_argument("--no-archive", action="store_true", help="Upload files directly without creating archive (faster)")
    upload_parser.add_argument("--private", action="store_true", help="Make repository private (default: public)")
    
    # Download command
    download_parser = subparsers.add_parser('download', help='Download checkpoint from Hugging Face')
    download_parser.add_argument("--repo_name", required=True, help="Hugging Face repository name")
    download_parser.add_argument("--output_dir", required=True, help="Output directory for checkpoint")
    download_parser.add_argument("--method", choices=["auto", "git", "archive", "hf_hub"], default="auto", 
                                help="Download method: auto (try git first), git (git clone), archive (tar file), hf_hub (HF download)")
    download_parser.add_argument("--token", help="Hugging Face token")
    
    args = parser.parse_args()
    
    if not args.command:
        parser.print_help()
        return 1
    
    # Get HF token
    token = args.token or os.getenv('HF_TOKEN')
    
    if args.command == 'upload':
        if not token:
            print("❌ Please provide Hugging Face token via --token or HF_TOKEN environment variable")
            return 1
        
        checkpoint_dir = Path(args.checkpoint_dir)
        if not checkpoint_dir.exists():
            print(f"❌ Checkpoint directory not found: {checkpoint_dir}")
            return 1
        
        if not upload_complete_checkpoint(str(checkpoint_dir), args.repo_name, token, args.private, use_archive=not args.no_archive):
            return 1
            
    elif args.command == 'download':
        output_dir = Path(args.output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        
        if not download_and_extract_checkpoint(args.repo_name, str(output_dir), token, args.method):
            return 1
    
    print("🎉 Process completed successfully!")
    return 0

if __name__ == "__main__":
    sys.exit(main())
