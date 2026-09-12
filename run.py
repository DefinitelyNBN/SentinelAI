"""Main entrypoint for SentinelAI Phase 6 training and evaluation."""
import sys
from pathlib import Path
from src.utils import load_config, set_seed
from src.experiments import run_phase6_pipeline

def main():
    cfg = load_config()
    set_seed(cfg.get("seed", 42))
    
    manifest_path = Path(cfg["dataset"]["manifest"])
    if not manifest_path.exists():
        print(f"Error: Manifest not found at {manifest_path}. Please run src/generate_manifest.py first.")
        return 1
        
    print(f"Starting SentinelAI Phase 6 Pipeline: {cfg['project_name']} ({cfg['track']})")
    summary = run_phase6_pipeline(cfg)
    print("\nPhase 6 execution finished successfully.")
    return 0

if __name__ == "__main__":
    sys.exit(main())
