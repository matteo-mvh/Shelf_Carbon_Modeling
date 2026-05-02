from helpers.common import create_timestamped_results
from helpers.model_runners import run_dic_calibration_and_model

if __name__ == "__main__":
    out = create_timestamped_results("dic_calibration")
    run_dic_calibration_and_model(out, plot_only_socat=True)
    print(f"Results saved to: {out}")
