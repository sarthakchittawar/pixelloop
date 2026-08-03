import time
from collections import defaultdict
from typing import Optional
import numpy as np
from functools import wraps


class GlobalTimingManager:
    """
    Centralized timing manager for cross-file benchmarking
    """

    _instance = None

    def __new__(cls):
        if cls._instance is None:
            cls._instance = super(GlobalTimingManager, cls).__new__(cls)
            cls._instance.enabled = False
            cls._instance.step_timings = defaultdict(list)
            cls._instance.episode_timings = defaultdict(list)
            cls._instance.function_timings = defaultdict(list)
            cls._instance.current_step = 0
            cls._instance.current_episode = 0
        return cls._instance

    def enable(self):
        self.enabled = True

    def disable(self):
        self.enabled = False

    def start_timer(self, operation_name: str, step: Optional[int] = None):
        if not self.enabled:
            return
        key = f"{operation_name}_start"
        if step is not None:
            key = f"step_{step}_{key}"
        setattr(self, key, time.time())

    def end_timer(self, operation_name: str, step: Optional[int] = None) -> float:
        if not self.enabled:
            return 0.0

        key = f"{operation_name}_start"
        if step is not None:
            key = f"step_{step}_{key}"

        if hasattr(self, key):
            elapsed = time.time() - getattr(self, key)
            delattr(self, key)  # Clean up

            # Store timing
            self.function_timings[operation_name].append(elapsed)
            if step is not None:
                self.step_timings[f"{operation_name}_step_{step}"].append(elapsed)

            return elapsed
        return 0.0

    def print_last_function_timings(self):
        """
        Print the most recent timing value for each function that was called.
        """
        if not self.function_timings:
            print("No timing data collected for this step.")
            return
        print("\nStep Timing:")
        for func, times in self.function_timings.items():
            if times:
                print(f"  {func:40}: {times[-1]:.4f}s (this step)")

    def get_function_stats(self, operation_name: str) -> dict:
        times = self.function_timings.get(operation_name, [])
        if not times:
            return {}

        return {
            "mean": np.mean(times),
            "min": np.min(times),
            "max": np.max(times),
            "total": np.sum(times),
            "count": len(times),
            "std": np.std(times),
        }

    def print_function_summary(self, operation_name: str):
        stats = self.get_function_stats(operation_name)
        if not stats:
            print(f"No timing data for {operation_name}")
            return

        print(f"\n{operation_name} Timing Summary:")
        print(f"  Calls: {stats['count']}")
        print(f"  Mean:  {stats['mean']:.4f}s")
        print(f"  Min:   {stats['min']:.4f}s")
        print(f"  Max:   {stats['max']:.4f}s")
        print(f"  Total: {stats['total']:.3f}s")
        print(f"  Std:   {stats['std']:.4f}s")

    def print_all_function_summaries(self):
        if not self.function_timings:
            print("No function timing data collected")
            return

        print("\n" + "=" * 60)
        print("FUNCTION-SPECIFIC TIMING ANALYSIS")
        print("=" * 60)

        for operation_name in sorted(self.function_timings.keys()):
            self.print_function_summary(operation_name)

    def save_function_timings_csv(self, csv_path):
        """
        Save function timing statistics to a CSV file.
        """
        import csv

        if not self.function_timings:
            print("No function timing data to save.")
            return

        # csv_path = save_path / 'function_timing_analysis.csv'
        with open(csv_path, "w", newline="") as csvfile:
            csvwriter = csv.writer(csvfile)
            csvwriter.writerow(
                [
                    "Function",
                    "Calls",
                    "Mean_Time_s",
                    "Min_Time_s",
                    "Max_Time_s",
                    "Total_Time_s",
                    "Std_Dev_s",
                ]
            )
            for func_name in sorted(self.function_timings.keys()):
                stats = self.get_function_stats(func_name)
                if stats:
                    csvwriter.writerow(
                        [
                            func_name,
                            stats["count"],
                            f"{stats['mean']:.6f}",
                            f"{stats['min']:.6f}",
                            f"{stats['max']:.6f}",
                            f"{stats['total']:.6f}",
                            f"{stats['std']:.6f}",
                        ]
                    )
        print(f"Function timing analysis saved to: {csv_path}")

    def clear_episode_data(self):
        """Clear data for the current episode"""
        self.step_timings.clear()

    def clear_all_data(self):
        """Clear all timing data"""
        self.step_timings.clear()
        self.episode_timings.clear()
        self.function_timings.clear()


# Global instance
timing_manager = GlobalTimingManager()


def time_function(operation_name: str = None):
    """
    Decorator to automatically time functions
    """

    def decorator(func):
        @wraps(func)
        def wrapper(*args, **kwargs):
            if not timing_manager.enabled:
                return func(*args, **kwargs)

            func_name = operation_name or f"{func.__module__}.{func.__name__}"
            timing_manager.start_timer(func_name)

            try:
                result = func(*args, **kwargs)
                return result
            finally:
                timing_manager.end_timer(func_name)

        return wrapper

    return decorator