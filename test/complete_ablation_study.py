"""
Script to check training status and complete ablation study once all models are ready.
This will wait for training to complete and then generate the full ablation table.
"""

import os
import sys
import time
import subprocess
import glob
import json

def find_latest_checkpoint(log_dirs, config_name):
    """Find the latest checkpoint in log directories."""
    all_checkpoints = []
    for log_dir in log_dirs if isinstance(log_dirs, list) else [log_dirs]:
        pattern = os.path.join(log_dir, config_name, "*", "best_model-*.ckpt")
        checkpoints = glob.glob(pattern)
        all_checkpoints.extend(checkpoints)
    if all_checkpoints:
        all_checkpoints.sort(key=os.path.getmtime, reverse=True)
        return all_checkpoints[0]
    return None

def check_training_complete(log_dirs, config_name, min_file_size=1000):
    """Check if training has produced a valid checkpoint."""
    checkpoint = find_latest_checkpoint(log_dirs, config_name)
    if checkpoint and os.path.exists(checkpoint):
        # Check if file is large enough (not just created)
        if os.path.getsize(checkpoint) > min_file_size:
            # Check if file was modified recently (within last 5 minutes means still training)
            mod_time = os.path.getmtime(checkpoint)
            time_since_mod = time.time() - mod_time
            # If modified more than 5 minutes ago, training is likely complete
            if time_since_mod > 300:  # 5 minutes
                return True, checkpoint
            else:
                return False, checkpoint  # Still training
    return False, None

def main():
    project_root = os.path.dirname(os.path.dirname(__file__))
    
    # Config paths
    base_config = os.path.join(project_root, 'pe_uncert_models/configs/crispAIPE_conf1.json')
    transformer_config = os.path.join(project_root, 'pe_uncert_models/configs/crispAIPE_transformer_only_conf.json')
    cnn_config = os.path.join(project_root, 'pe_uncert_models/configs/crispAIPE_cnn_only_conf.json')
    
    # Load configs to get log directories
    with open(transformer_config, 'r') as f:
        transformer_config_data = json.load(f)
    with open(cnn_config, 'r') as f:
        cnn_config_data = json.load(f)
    
    log_dir = transformer_config_data['training_parameters']['log_dir']
    transformer_config_name = os.path.basename(transformer_config).replace('.json', '')
    cnn_config_name = os.path.basename(cnn_config).replace('.json', '')
    
    # Resolve log_dir relative to config location
    if not os.path.isabs(log_dir):
        log_dir = os.path.join(os.path.dirname(transformer_config), log_dir)
    log_dir = os.path.abspath(log_dir)
    
    # Also check Desktop/logs as fallback (config uses ../logs which resolves there)
    desktop_logs = os.path.join(os.path.expanduser("~"), "Desktop", "logs")
    if os.path.exists(desktop_logs):
        # Try both locations
        log_dirs_to_check = [log_dir, desktop_logs]
    else:
        log_dirs_to_check = [log_dir]
    
    print("="*70)
    print("ABLATION STUDY COMPLETION CHECKER")
    print("="*70)
    print(f"Log directory: {log_dir}")
    print(f"Transformer config: {transformer_config_name}")
    print(f"CNN config: {cnn_config_name}")
    print("="*70 + "\n")
    
    # Check hybrid checkpoint
    with open(base_config, 'r') as f:
        base_config_data = json.load(f)
    base_log_dir = base_config_data['training_parameters']['log_dir']
    if not os.path.isabs(base_log_dir):
        base_log_dir = os.path.join(os.path.dirname(base_config), base_log_dir)
    base_log_dir = os.path.abspath(base_log_dir)
    base_config_name = os.path.basename(base_config).replace('.json', '')
    
    # Check both possible log locations for hybrid
    base_log_dirs = [base_log_dir]
    desktop_base_logs = os.path.join(os.path.expanduser("~"), "Desktop", "logs")
    if os.path.exists(desktop_base_logs):
        base_log_dirs.append(desktop_base_logs)
    hybrid_checkpoint = find_latest_checkpoint(base_log_dirs, base_config_name)
    if hybrid_checkpoint:
        print(f"✓ Hybrid checkpoint found: {hybrid_checkpoint}\n")
    else:
        print("✗ Hybrid checkpoint not found!\n")
        return 1
    
    # Check transformer-only
    transformer_complete, transformer_checkpoint = check_training_complete(
        log_dirs_to_check, transformer_config_name
    )
    if transformer_complete:
        print(f"✓ Transformer-only training complete: {transformer_checkpoint}\n")
    elif transformer_checkpoint:
        print(f"⏳ Transformer-only still training: {transformer_checkpoint}\n")
    else:
        print("✗ Transformer-only checkpoint not found\n")
    
    # Check CNN-only
    cnn_complete, cnn_checkpoint = check_training_complete(
        log_dirs_to_check, cnn_config_name
    )
    if cnn_complete:
        print(f"✓ CNN-only training complete: {cnn_checkpoint}\n")
    elif cnn_checkpoint:
        print(f"⏳ CNN-only still training: {cnn_checkpoint}\n")
    else:
        print("✗ CNN-only checkpoint not found\n")
    
    # If all are ready, run ablation evaluation
    if hybrid_checkpoint and transformer_complete and cnn_complete:
        print("="*70)
        print("All models ready! Running ablation evaluation...")
        print("="*70 + "\n")
        
        cmd = [
            sys.executable,
            'test/model_ablation.py',
            '--config', base_config,
            '--hybrid_checkpoint', hybrid_checkpoint,
            '--transformer_checkpoint', transformer_checkpoint,
            '--cnn_checkpoint', cnn_checkpoint,
            '--output_dir', 'test/figures/ablation_study'
        ]
        
        result = subprocess.run(cmd, cwd=project_root)
        
        if result.returncode == 0:
            print("\n" + "="*70)
            print("ABLATION STUDY COMPLETE!")
            print("="*70)
            print("Results saved to: test/figures/ablation_study/")
            print("  - ablation_table_s3.csv")
            print("  - ablation_table_s3.png/pdf")
            print("  - ablation_comparison.png/pdf")
            print("  - ablation_results.json")
            print("="*70)
            return 0
        else:
            print("\nERROR: Ablation evaluation failed")
            return 1
    else:
        print("\n" + "="*70)
        print("Training not complete yet.")
        print("="*70)
        if not transformer_complete:
            print("  - Transformer-only model still training or not found")
        if not cnn_complete:
            print("  - CNN-only model still training or not found")
        print("\nRun this script again once training completes.")
        print("="*70)
        return 1

if __name__ == "__main__":
    exit(main())

