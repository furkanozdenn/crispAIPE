"""
Finalize ablation study with target-disjoint split (revised).
Waits for retraining to complete and generates the final ablation table.
"""

import os
import sys
import time
import subprocess
import glob
import json


def find_latest_checkpoint(log_dir, config_name):
    """Find the latest checkpoint in a log directory."""
    pattern = os.path.join(log_dir, config_name, "*", "best_model-*.ckpt")
    checkpoints = glob.glob(pattern)
    if checkpoints:
        checkpoints.sort(key=os.path.getmtime, reverse=True)
        return checkpoints[0]
    return None


def check_training_complete(log_dir, config_name, min_file_size=1000, wait_time=300):
    """Check if training has produced a valid checkpoint that hasn't been updated recently."""
    checkpoint = find_latest_checkpoint(log_dir, config_name)
    if checkpoint and os.path.exists(checkpoint):
        if os.path.getsize(checkpoint) > min_file_size:
            mod_time = os.path.getmtime(checkpoint)
            time_since_mod = time.time() - mod_time
            if time_since_mod > wait_time:
                return True, checkpoint
            else:
                return False, checkpoint
    return False, None


def main():
    project_root = os.path.dirname(os.path.dirname(__file__))

    base_config = os.path.join(project_root, 'pe_uncert_models/configs/crispAIPE_revs_conf.json')
    transformer_config = os.path.join(project_root, 'pe_uncert_models/configs/crispAIPE_transformer_only_revs_conf.json')
    cnn_config = os.path.join(project_root, 'pe_uncert_models/configs/crispAIPE_cnn_only_revs_conf.json')

    with open(transformer_config, 'r') as f:
        transformer_config_data = json.load(f)
    with open(cnn_config, 'r') as f:
        cnn_config_data = json.load(f)

    log_dir = transformer_config_data['training_parameters']['log_dir']
    if not os.path.isabs(log_dir):
        log_dir = os.path.join(os.path.dirname(transformer_config), log_dir)
    log_dir = os.path.abspath(log_dir)

    log_dirs_to_check = [log_dir]
    desktop_logs = os.path.join(os.path.expanduser("~"), "Desktop", "logs")
    if os.path.exists(desktop_logs):
        log_dirs_to_check.append(desktop_logs)

    # Config names for checkpoint search
    hybrid_config_name = "crispAIPE_revs_conf"
    transformer_config_name = os.path.basename(transformer_config).replace('.json', '')
    cnn_config_name = os.path.basename(cnn_config).replace('.json', '')

    # Find hybrid checkpoint
    hybrid_checkpoint = None
    for log_dir_check in log_dirs_to_check:
        hybrid_checkpoint = find_latest_checkpoint(log_dir_check, hybrid_config_name)
        if hybrid_checkpoint:
            break

    print("=" * 70)
    print("FINALIZING ABLATION STUDY — REVISED (TARGET-DISJOINT SPLIT)")
    print("=" * 70)
    print(f"Base config: {base_config}")
    print(f"Hybrid checkpoint: {hybrid_checkpoint}")
    print("=" * 70 + "\n")

    if not hybrid_checkpoint or not os.path.exists(hybrid_checkpoint):
        print("ERROR: Hybrid checkpoint not found. Train with crispAIPE_revs_conf.json first.")
        return 1

    print(f"  Hybrid checkpoint found: {hybrid_checkpoint}\n")

    # Check transformer-only
    transformer_complete = False
    transformer_checkpoint = None
    for log_dir_check in log_dirs_to_check:
        complete, checkpoint = check_training_complete(
            log_dir_check, transformer_config_name, wait_time=300
        )
        if complete:
            transformer_complete = True
            transformer_checkpoint = checkpoint
            break
        elif checkpoint:
            transformer_checkpoint = checkpoint

    if transformer_complete:
        print(f"  Transformer-only training complete: {transformer_checkpoint}\n")
    elif transformer_checkpoint:
        print(f"  Transformer-only still training: {transformer_checkpoint}\n")
    else:
        print("  Transformer-only checkpoint not found\n")

    # Check CNN-only
    cnn_complete = False
    cnn_checkpoint = None
    for log_dir_check in log_dirs_to_check:
        complete, checkpoint = check_training_complete(
            log_dir_check, cnn_config_name, wait_time=300
        )
        if complete:
            cnn_complete = True
            cnn_checkpoint = checkpoint
            break
        elif checkpoint:
            cnn_checkpoint = checkpoint

    if cnn_complete:
        print(f"  CNN-only training complete: {cnn_checkpoint}\n")
    elif cnn_checkpoint:
        print(f"  CNN-only still training: {cnn_checkpoint}\n")
    else:
        print("  CNN-only checkpoint not found\n")

    if hybrid_checkpoint and transformer_complete and cnn_complete:
        print("=" * 70)
        print("All models ready! Running ablation evaluation...")
        print("=" * 70 + "\n")

        cmd = [
            sys.executable,
            'test/model_ablation.py',
            '--config', base_config,
            '--hybrid_checkpoint', hybrid_checkpoint,
            '--transformer_checkpoint', transformer_checkpoint,
            '--cnn_checkpoint', cnn_checkpoint,
            '--output_dir', 'test/figures_revs/ablation_study'
        ]

        result = subprocess.run(cmd, cwd=project_root)

        if result.returncode == 0:
            print("\n" + "=" * 70)
            print("ABLATION STUDY COMPLETE (TARGET-DISJOINT SPLIT)!")
            print("=" * 70)
            print("Results saved to: test/figures_revs/ablation_study/")
            print("=" * 70)
            return 0
        else:
            print("\nERROR: Ablation evaluation failed")
            return 1
    else:
        print("\n" + "=" * 70)
        print("Training not complete yet.")
        print("=" * 70)
        if not transformer_complete:
            print("  - Transformer-only model still training or not found")
        if not cnn_complete:
            print("  - CNN-only model still training or not found")
        print("\nRun this script again once training completes.")
        print("=" * 70)
        return 1


if __name__ == "__main__":
    exit(main())
