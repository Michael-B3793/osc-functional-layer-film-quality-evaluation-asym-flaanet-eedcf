import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from plot_target_correlation_methods import run_for_target


if __name__ == "__main__":
    run_for_target("Voc (V)", "voc_correlation_plots", "Voc (V)")
