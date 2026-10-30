"""
    Synthetic defect image generation for synthetic Inspector.
"""

import os
import json
import sys
import time
import subprocess
from pathlib import Path
from dotenv import load_dotenv

load_dotenv()

from src.config import (
    WORKFLOW_YAML, DATA_IMAGE_DIR, DATA_ANNOTATION_DIR,
    S3_BUCKET, S3_PREFIX, JOB_TIMEOUT_SECONDS, MAX_IMAGES,
    IMAGES_PER_CLASS, DEFECT_PROMPTS, validate_secrets,
)


SECRET_ENVS = ["HF_TOKEN", "AWS_ACCESS_KEY_ID", "AWS_SECRET_ACCESS_KEY"]

def _submit_workflow(prompt: str, prefix: str) -> subprocess.CompletedProcess:
    """
        Build the npa command, run it, return the completedProcess.
    """
    cmd = [
        "npa", "workbench", "workflow", "submit",
        WORKFLOW_YAML,
        "--var", f"bucket={S3_BUCKET}",
        "--var", f"prefix={prefix}",
        "--var", f"prompt={prompt}",
        "--runtime",
    ]
    for name in SECRET_ENVS:
        cmd.extend(["--secret-env", name])
    return subprocess.run(
        cmd,
        capture_output=True,
        text=True,
        timeout=JOB_TIMEOUT_SECONDS,
    )



def smoke_test() -> bool:
    """
        Generate ONE image. Verify it is a real file. Return True/False.
        Do not proceed to batch generation until this returns True.
    """
    validate_secrets(required=("S3_BUCKET",) + tuple(SECRET_ENVS))
    print("[smoke] Single test generation")
    result = _submit_workflow(DEFECT_PROMPTS["scratch"], prefix=f"{S3_PREFIX}/smoke")
    
    if result.returncode !=0:
        print(f"[smoke] FAILED (exit {result.returncode})")
        print(f"[smoke] stderr:\n{result.stderr[:1000]}")
        return False

    print("[smoke] Job succeeded. stdout (first 500 chars):")
    print(result.stdout[:500])
    return True



def submit_one_generation(prompt: str, image_path: Path, annotation_path: Path) -> bool:
    """
        Submit ONE workflow job, wait for completion, download the image, write the annotation. Return True on success.
        It is a thin wrapper. 
    """
    run_id = image_path.stem
    result = _submit_workflow(prompt, prefix=f"{S3_PREFIX}/{run_id}")
    
    if result.returncode !=0:
        print(f"    job failed: {result.stderr[:300]}")
        return False

    
    s3_uri = f"s3://{S3_BUCKET}/{S3_PREFIX}/{run_id}/generated/vision.jpg"

    download = subprocess.run(
        ["aws", "s3", "cp", s3_uri, str(image_path)],
        capture_output=True,
        text=True,
        timeout=120,
    )

    if download.returncode !=0:
        print(f"    download failed: {download.stderr[:300]}")
        return False 

    annotation = {
        "image_id": image_path.name,
        "class": image_path.stem.rsplit("_", 1)[0],
        "prompt" : prompt,
        "s3_uri" : s3_uri,
    }
    annotation_path.write_text(json.dumps(annotation, indent=2))
    return True

    
def main() -> None:
    validate_secrets(required=("S3_BUCKET",) + tuple(SECRET_ENVS))
    DATA_IMAGE_DIR.mkdir(parents=True, exist_ok=True)
    DATA_ANNOTATION_DIR.mkdir(parents=True, exist_ok=True)

    total_generated = 0

    for defect_class, prompt in DEFECT_PROMPTS.items():
        print(f"\n[class] {defect_class}: {IMAGES_PER_CLASS} images")
        for i in range(IMAGES_PER_CLASS):
            if total_generated >= MAX_IMAGES:
                print(f"[cap] Reached MAX_IMAGES={MAX_IMAGES}. Stopping.")
                print(f"[done] Generated {total_generated} images total.")
                return 

            image_path = DATA_IMAGE_DIR / f"{defect_class}_{i:04d}.png"
            annotation_path = DATA_ANNOTATION_DIR / f"{defect_class}_{i:04d}.json"
            
            if image_path.exists() and annotation_path.exists():
                continue
            print(f" [{total_generated + 1}/{MAX_IMAGES}] {defect_class}_{i:04d}")

            success = submit_one_generation(prompt, image_path, annotation_path)
            if not success:
                print(f" FAILED: {defect_class}_{i:04d}. Continuing...")
                continue
            total_generated +=1
    print(f"\n[done] Generated {total_generated} images")




if __name__ =="__main__":
    if "--smoke" in sys.argv:
        ok = smoke_test()
        sys.exit(0 if ok else 1)
    main()