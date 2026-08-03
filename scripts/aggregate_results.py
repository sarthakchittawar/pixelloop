#!/usr/bin/env python3
"""
Aggregate benchmarking results from multiple scenes and goal images.
Reads results_summary.txt files and compares different strategies.
Generates CSV and Markdown tables.
"""

import os
import sys
import pandas as pd
import numpy as np
from pathlib import Path
import argparse

# TODO: Adjust these parameters as needed
EDGE_CULLING_MODE = "EMST_SINGLE"
NODE_CULLING_MODE = "NONE"
NODE_CULLING_FACTOR = 1
DEPTH = "mast3r"
MATCHES = "mast3r"

def find_latest_run(base_path, scene, exp_name, goal_img):
    search_path = base_path / exp_name / scene / f"goalImg{goal_img}" / f"EC_{EDGE_CULLING_MODE}_NC_{NODE_CULLING_MODE}_NCF_{NODE_CULLING_FACTOR}_DEPTH_{DEPTH}_MATCHES_{MATCHES}" / "benchmark_default"
    if not search_path.exists():
        return None
    timestamp_folders = [d for d in search_path.iterdir() if d.is_dir()]
    if not timestamp_folders:
        return None
    latest = max(timestamp_folders, key=lambda x: x.stat().st_mtime)
    return latest

def read_results_summary(run_folder):
    summary_file = run_folder / "metrics_summary.txt"
    if not summary_file.exists():
        return None
    try:
        with open(summary_file, 'r') as f:
            lines = f.readlines()
        metrics = {}
        for line in lines:
            line = line.strip()
            if not line or ':' not in line:
                continue
            parts = line.split(':', 1)
            if len(parts) != 2:
                continue
            key, value = parts[0].strip(), parts[1].strip()
            try:
                if '.' in value:
                    metrics[key] = float(value)
                else:
                    metrics[key] = int(value)
            except ValueError:
                metrics[key] = value
        return metrics
    except Exception as e:
        print(f"Error reading {summary_file}: {e}")
        return None

def read_episode_results(run_folder):
    """Read per-episode results from CSV file."""
    csv_file = run_folder / "results_summary.csv"
    if not csv_file.exists():
        return None
    try:
        df = pd.read_csv(csv_file)
        episodes = []
        for _, row in df.iterrows():
            episode_data = {
                'spl': row.get('spl', 0.0),
                'soft_spl': row.get('soft_spl', 0.0),
                'success': row.get('success_status', False) if 'success_status' in df.columns else (row.get('spl', 0.0) > 0),
            }
            episodes.append(episode_data)
        return episodes
    except Exception as e:
        print(f"Error reading {csv_file}: {e}")
        return None

def aggregate_scene_goal(summary):
    if not summary:
        return None
    result = {
        'num_successes': summary.get('successful_episodes', 0),
        'num_failures': summary.get('failed_episodes', 0),
        'total_episodes': summary.get('total_episodes', 0),
        'exceeded_steps': summary.get('exceeded_steps', 0),
        'collision': summary.get('stuck', 0),
        'other_failures': 0,
    }
    min_spl = summary.get('successful_min_spl', np.nan)
    max_spl = summary.get('successful_max_spl', np.nan)
    mean_spl = summary.get('mean_spl', np.nan)
    mean_sspl = summary.get('mean_soft_spl', np.nan)
    successful_mean_spl = summary.get('successful_mean_spl', np.nan)
    successful_mean_sspl = summary.get('successful_mean_soft_spl', np.nan)
    success_rate = summary.get('success_rate', np.nan)
    
    result['success_rate_pct'] = success_rate if not pd.isna(success_rate) else np.nan
    result['spl_min'] = min_spl * 100 if not pd.isna(min_spl) else np.nan
    result['spl_max'] = max_spl * 100 if not pd.isna(max_spl) else np.nan
    result['spl_mean'] = mean_spl * 100 if not pd.isna(mean_spl) else np.nan
    result['sspl_mean'] = mean_sspl * 100 if not pd.isna(mean_sspl) else np.nan
    result['spl_success_mean'] = successful_mean_spl * 100 if not pd.isna(successful_mean_spl) else np.nan
    result['sspl_success_mean'] = successful_mean_sspl * 100 if not pd.isna(successful_mean_sspl) else np.nan
    result['mean_collisions'] = summary.get('mean_collisions', np.nan)
    return result

def process_experiment(base_path, scenes_goals, exp_name):
    results = []
    all_episodes = []  # Collect all episode-level data for weighted averaging
    for scene, goal_imgs in scenes_goals.items():
        for goal_img in goal_imgs:
            print(f"Processing {scene} - Goal {goal_img} - {exp_name}...")
            run_folder = find_latest_run(base_path, scene, exp_name, goal_img)
            if run_folder is None:
                print(f"  ⚠ No run found for {scene} goal {goal_img}")
                continue
            print(f"  Found: {run_folder}")
            summary = read_results_summary(run_folder)
            if not summary:
                print(f"  ⚠ No metrics_summary.txt found in {run_folder}")
                continue
            total = summary.get('total_episodes', 0)
            success = summary.get('successful_episodes', 0)
            print(f"  Episodes: {success}/{total} successful")
            
            # Read per-episode data for weighted averaging
            episodes = read_episode_results(run_folder)
            if episodes:
                all_episodes.extend(episodes)
            
            agg = aggregate_scene_goal(summary)
            if agg:
                agg['scene'] = scene
                agg['goal_img'] = goal_img
                agg['exp_name'] = exp_name
                results.append(agg)
    return results, all_episodes

def create_table(results_list, title, all_episodes=None):
    """Create results table with properly weighted averages.
    
    Args:
        results_list: List of per-scene-goal metrics
        title: Table title
        all_episodes: List of all episode-level data for weighted averaging
    """
    if not results_list:
        return None
    df = pd.DataFrame(results_list)
    columns = ['scene', 'goal_img', 'success_rate_pct',
               'spl_mean', 'spl_success_mean', 'sspl_mean']
    df = df[columns]
    numeric_cols = ['success_rate_pct', 'spl_mean', 'spl_success_mean', 'sspl_mean']
    for col in numeric_cols:
        df[col] = pd.to_numeric(df[col], errors='coerce')
    
    # Compute weighted averages from raw episode data if available
    if all_episodes and len(all_episodes) > 0:
        all_spl = [ep['spl'] for ep in all_episodes]
        all_sspl = [ep['soft_spl'] for ep in all_episodes]
        successful_spl = [ep['spl'] for ep in all_episodes if ep['success']=='success']
        total_episodes = len(all_episodes)
        num_successes = sum(1 for ep in all_episodes if ep['success']=='success')
        
        avg_row = {
            'scene': 'Average (All Episodes)',
            'goal_img': '',
            'success_rate_pct': (num_successes / total_episodes * 100) if total_episodes > 0 else np.nan,
            'spl_mean': np.mean(all_spl) * 100 if all_spl else np.nan,
            'spl_success_mean': (np.sum(successful_spl) / num_successes * 100) if num_successes > 0 else np.nan,
            'sspl_mean': np.mean(all_sspl) * 100 if all_sspl else np.nan,
        }
    else:
        # Fallback to simple averaging if no episode data available
        # For success-only metrics (spl_min, spl_max, spl_success_mean), only average over rows with successes
        df_with_success = df[df['success_rate_pct'] > 0].copy() if 'success_rate_pct' in df.columns else df
        
        avg_row = {
            'scene': 'Average',
            'goal_img': '',
            'success_rate_pct': df['success_rate_pct'].mean(skipna=True),
            'spl_mean': df['spl_mean'].mean(skipna=True),
            'spl_success_mean': df_with_success['spl_success_mean'].mean(skipna=True),
            'sspl_mean': df['sspl_mean'].mean(skipna=True),
        }
    df = pd.concat([df, pd.DataFrame([avg_row])], ignore_index=True)
    return df

def format_markdown_table(df, title):
    if df is None:
        return f"## {title}\n\nNo data available.\n\n"
    md = f"## {title}\n\n"
    md += "| Scene | Goal | Success Rate (%) | "
    md += "Mean SPL | Mean SPL (Success Only) | Mean SSPL |\n"
    md += "|" + "---|" * 6 + "\n"
    for _, row in df.iterrows():
        scene = row['scene']
        goal = str(row['goal_img']) if row['goal_img'] != '' else ''
        def fmt(val):
            if pd.isna(val):
                return "N/A"
            elif isinstance(val, float):
                if val == int(val):
                    return str(int(val))
                return f"{val:.2f}"
            return str(val)
        md += f"| {scene} | {goal} | {fmt(row['success_rate_pct'])} | "
        md += f"{fmt(row['spl_mean'])} | {fmt(row['spl_success_mean'])} | {fmt(row['sspl_mean'])} |\n"
    md += "\n"
    return md

def main():
    parser = argparse.ArgumentParser(description='Aggregate benchmarking results')
    parser.add_argument('--base-path', type=str, required=True, help='Base path to output directory', default='/scratch/sarthakc/mast3rnav/sg_habitat/output')
    parser.add_argument('--output', type=str, default='benchmarking_results', help='Output filename (without extension)')
    args = parser.parse_args()
    base_path = Path(args.base_path)
    scenes_goals = {
        "1W61QJVDBqe": [104, 77, 321],
        "CETmJJqkhcK": [65, 205, 278],
        "Wo6kuutE9i7": [43, 104, 171],
        "3UDjdrwcqMb": [278, 193, 114],
        "Nfvxx8J5NCo": [107, 349, 272],
        "PaQrTquNd2v": [263, 302, 113]
    }
    # scenes_goals = {
    #     "53jtKd53a1X": [81, 290],
    #     "9DnDAhJ7qcj": [90, 207, 320],
    #     "b2e31HFFizw": [246],
    #     "qQgcM8T4hiD": [160, 238],
    #     "enfahKs8XHw": [261],
    #     "XiJhRLvpKpX": [247],
    # }
    # scenes_goals = {
    #     "3UDjdrwcqMb": [278, 193, 114],
    #     "Nfvxx8J5NCo": [107, 349, 272],
    #     "PaQrTquNd2v": [263, 302, 113]
    # }
    experiments = {
        'multi_scene_runs_noLC': 'Without LCs',
        'multi_scene_runs_LC_oracle': 'With Oracle (<1m threshold) LCs',
        'multi_scene_runs_LC_seqvlad': 'With SeqVLAD (<1m threshold) LCs',
    }
    all_results = {}
    all_episodes_data = {}  # Store episode-level data for each experiment
    for exp_name, exp_title in experiments.items():
        print(f"\n{'='*60}")
        print(f"Processing: {exp_title}")
        print('='*60)
        results, episodes = process_experiment(base_path, scenes_goals, exp_name)
        all_results[exp_title] = results
        all_episodes_data[exp_title] = episodes
        print(f"  Total episodes collected: {len(episodes)}")
    
    # Create output directory
    output_dir = Path(args.output) / f"EC_{EDGE_CULLING_MODE}_NC_{NODE_CULLING_MODE}_NCF_{NODE_CULLING_FACTOR}_DEPTH_{DEPTH}_MATCHES_{MATCHES}"
    output_dir.mkdir(parents=True, exist_ok=True)
    
    output_md = output_dir / "summary.md"
    print(f"\nGenerating markdown report: {output_md}")
    with open(output_md, 'w') as f:
        f.write("# Benchmarking Results: A2B Navigation Study\n\n")
        f.write("SPL = Success weighted by Path Lengths  \n")
        f.write("SSPL = Soft SPL  \n")
        f.write("**Note:** Averages are weighted by episode count (all episodes treated equally)\n\n")
        for exp_title, results in all_results.items():
            episodes = all_episodes_data.get(exp_title, [])
            df = create_table(results, exp_title, episodes)
            f.write(format_markdown_table(df, exp_title))
    for exp_title, results in all_results.items():
        episodes = all_episodes_data.get(exp_title, [])
        df = create_table(results, exp_title, episodes)
        if df is not None:
            csv_file = output_dir / f"{exp_title.replace(' ', '_').lower()}.csv"
            df.to_csv(csv_file, index=False)
            print(f"Saved: {csv_file}")
    print(f"\n✓ Results saved to {output_dir}/")
    print(f"  - Markdown summary: {output_md.name}")
    print(f"  - CSV files: {len(all_results)} experiment file(s)")

if __name__ == '__main__':
    main()