from helpers.common import create_timestamped_results
from helpers.model_runners import run_dic_full, run_dic_doc_full

MODEL_TYPE = "DIC"  # options: "DIC", "DIC_DOC"

if __name__ == "__main__":
    out = create_timestamped_results(f"plots_{MODEL_TYPE.lower()}")
    if MODEL_TYPE == "DIC":
        run_dic_full(out)
    elif MODEL_TYPE == "DIC_DOC":
        run_dic_doc_full(out)
    else:
        raise ValueError("MODEL_TYPE must be 'DIC' or 'DIC_DOC'")
    print(f"Results saved to: {out}")
