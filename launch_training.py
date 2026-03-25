"""Launch parallel RL training — starts N workers + 1 central trainer.

Each worker needs its own BizHawk instance running with a Lua bridge
pointing to its bridge_N/ directory.

Usage:
    python launch_training.py --workers 2
    python launch_training.py --workers 4 --threshold 1024

Setup before running:
    1. Start BizHawk instance 0 with Lua bridge dir = bridge_0/
    2. Start BizHawk instance 1 with Lua bridge dir = bridge_1/
    3. (etc. for more workers)
    4. Run this script
"""

import argparse
import os
import subprocess
import sys
import signal
import time


def main():
    parser = argparse.ArgumentParser(description="Launch parallel RL training")
    parser.add_argument("--workers", type=int, default=4, help="Number of worker instances")
    args = parser.parse_args()

    python = sys.executable
    processes = []

    # Create bridge directories
    for i in range(args.workers):
        bridge_dir = f"bridge_{i}"
        os.makedirs(bridge_dir, exist_ok=True)
        print(f"Bridge directory ready: {bridge_dir}/")

    os.makedirs("rl_checkpoints", exist_ok=True)

    print(f"\n{'='*60}")
    print(f"  Launching {args.workers} workers + 1 trainer")
    print(f"{'='*60}")
    print(f"\nMake sure you have {args.workers} BizHawk instances running,")
    print(f"each with Lua bridge pointing to bridge_0/, bridge_1/, etc.\n")

    try:
        # Start trainer
        trainer_cmd = [python, "rl_trainer.py"]
        print(f"Starting trainer: {' '.join(trainer_cmd)}")
        trainer_proc = subprocess.Popen(trainer_cmd)
        processes.append(("Trainer", trainer_proc))
        time.sleep(1)

        # Start workers
        for i in range(args.workers):
            worker_cmd = [python, "rl_worker.py", "--worker-id", str(i)]
            print(f"Starting worker {i}: {' '.join(worker_cmd)}")
            proc = subprocess.Popen(worker_cmd)
            processes.append((f"Worker {i}", proc))
            time.sleep(0.5)

        print(f"\nAll processes started. Press Ctrl+C to stop.\n")

        # Wait for any process to exit
        while True:
            for name, proc in processes:
                ret = proc.poll()
                if ret is not None:
                    print(f"\n{name} exited with code {ret}")
                    raise KeyboardInterrupt
            time.sleep(2)

    except KeyboardInterrupt:
        print("\n\nShutting down all processes...")
        for name, proc in processes:
            if proc.poll() is None:
                proc.terminate()
                print(f"  Terminated {name}")

        # Wait for graceful shutdown
        for name, proc in processes:
            try:
                proc.wait(timeout=5)
            except subprocess.TimeoutExpired:
                proc.kill()
                print(f"  Killed {name}")

        print("All processes stopped.")


if __name__ == "__main__":
    main()
