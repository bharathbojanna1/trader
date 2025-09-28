#!/usr/bin/env python3
"""
Example usage of the Enhanced Intraday Trading RL Agent
Demonstrates different configuration options and features
"""

import subprocess
import sys
import os

def run_example(name: str, args: list):
    """Run an example configuration."""
    print(f"\n{'='*60}")
    print(f"🚀 Running Example: {name}")
    print(f"{'='*60}")
    
    cmd = [sys.executable, "intraday_agent.py"] + args
    print(f"Command: {' '.join(cmd)}")
    print("-" * 60)
    
    try:
        result = subprocess.run(cmd, capture_output=True, text=True, timeout=300)
        if result.returncode == 0:
            print("✅ Example completed successfully!")
            print("\nOutput:")
            print(result.stdout[-1000:])  # Last 1000 characters
        else:
            print("❌ Example failed!")
            print("Error:", result.stderr)
    except subprocess.TimeoutExpired:
        print("⏰ Example timed out (5 minutes)")
    except Exception as e:
        print(f"❌ Error running example: {e}")

def main():
    """Run various example configurations."""
    
    print("🎯 Enhanced Intraday Trading RL Agent - Examples")
    print("This script demonstrates different configuration options")
    
    # Check if main script exists
    if not os.path.exists("intraday_agent.py"):
        print("❌ intraday_agent.py not found in current directory")
        return
    
    # Example 1: Minimal configuration (fast)
    run_example(
        "Minimal Configuration (Fast)",
        [
            "--max_steps", "1000",
            "--days", "2",
            "--no_multi_transformer",
            "--no_technical_indicators", 
            "--no_sentiment",
            "--no_log_timestep_details",
            "--print_every", "200"
        ]
    )
    
    # Example 2: Full-featured configuration
    run_example(
        "Full-Featured Configuration",
        [
            "--max_steps", "2000",
            "--days", "3",
            "--use_multi_transformer",
            "--use_technical_indicators",
            "--use_sentiment",
            "--log_dir", "example_logs",
            "--print_every", "500"
        ]
    )
    
    # Example 3: Custom architecture
    run_example(
        "Custom Architecture",
        [
            "--max_steps", "1500",
            "--days", "2",
            "--transformer_d_model", "32",
            "--transformer_nhead", "2",
            "--transformer_num_layers", "1",
            "--indicator_window", "10",
            "--sentiment_window", "5",
            "--print_every", "300"
        ]
    )
    
    # Example 4: Risk-focused configuration
    run_example(
        "Risk-Focused Configuration",
        [
            "--max_steps", "1500",
            "--days", "2",
            "--risk_penalty_lambda", "0.2",
            "--sharpe_bonus_weight", "0.1",
            "--trailing_stop_pct", "0.005",
            "--print_every", "300"
        ]
    )
    
    print(f"\n{'='*60}")
    print("🎉 All examples completed!")
    print("📝 Check the generated log files for detailed results")
    print("📊 Use the CSV files for further analysis and backtesting")
    print(f"{'='*60}")

if __name__ == "__main__":
    main()
