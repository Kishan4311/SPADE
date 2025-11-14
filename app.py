# streamlit run /home/iitb/Kishan_SpecDec/Archived/app.py
import streamlit as st
import time
import pandas as pd
import matplotlib.pyplot as plt
from appInference2 import InferenceCLI, ModelsCatalog, GreedyProcessor  # ensure appInference2.py exports these

st.set_page_config(layout="wide", page_title="SPADE UI", initial_sidebar_state="expanded")

st.title("SPADE — Speculative Decoding for Precise & Low‑Cost Distributed Edge‑Cloud Inference")

# -------------------------
# Sidebar: settings & model selection
# -------------------------
st.sidebar.header("Model & Device Settings")

model_options = ModelsCatalog.list_models()
target_model_key = st.sidebar.selectbox("Target model (key)", model_options, index=model_options.index("llama-3b") if "llama-3b" in model_options else 0)
drafter_model_key = st.sidebar.selectbox("Drafter model (key)", model_options, index=model_options.index("llama-1b") if "llama-1b" in model_options else 0)

target_model_id = ModelsCatalog.model_id(target_model_key)
drafter_model_id = ModelsCatalog.model_id(drafter_model_key)

device_default = "cuda:0"
device = st.sidebar.text_input("Default device", device_default)
device_target = st.sidebar.text_input("Target device (leave empty to use default)", "")
device_drafter = st.sidebar.text_input("Drafter device (leave empty to use default)", "")

st.sidebar.markdown("---")
st.sidebar.header("Generation settings")
gamma = st.sidebar.number_input("Gamma (drafter drafts)", min_value=1, max_value=16, value=6, step=1)
gen_len = st.sidebar.number_input("Max gen length (tokens)", min_value=2, max_value=2048, value=256, step=2)
use_spec = st.sidebar.checkbox("Enable speculative decoding", value=True)
use_target = st.sidebar.checkbox("Enable target AR", value=True)
use_drafter = st.sidebar.checkbox("Enable drafter AR", value=False)

st.sidebar.markdown("---")
if st.sidebar.button("Load models"):
    # instantiate and load models
    with st.spinner("Loading models (this can take a while)..."):
        try:
            cli = InferenceCLI(
                device=device,
                device_target=(device_target or None),
                device_drafter=(device_drafter or None),
                target_model=target_model_id,
                drafter_model=drafter_model_id,
            )
            # configure generation settings
            cli.gamma = gamma
            cli.gen_len = gen_len
            cli.spec = use_spec
            cli.dr = use_drafter
            cli.target_gen = use_target
            cli.processor = GreedyProcessor()
            # store in session state
            st.session_state["cli"] = cli
            st.success("Models loaded and stored in session state.")
        except Exception as e:
            st.error(f"Error loading models: {e}")

# show loaded status
if "cli" in st.session_state:
    st.sidebar.success("Models loaded ✓")
else:
    st.sidebar.warning("No models loaded yet.")


# -------------------------
# Main layout: left = prompt, right = outputs
# -------------------------
left_col, right_col = st.columns([1, 1.5])

with left_col:
    st.subheader("Prompt")
    prompt = st.text_area("Enter your prompt here", height=100)
    if "cli" in st.session_state:
        # allow toggling chat mode per-run
        chat_mode = st.checkbox("Chat template mode", value=True)
        st.session_state["cli"].chat = chat_mode

    if st.button("Generate"):
        if "cli" not in st.session_state:
            st.error("Please load models first in the sidebar.")
        else:
            cli = st.session_state["cli"]
            # update cli settings from sidebar before run
            cli.gamma = gamma
            cli.gen_len = gen_len
            cli.spec = use_spec
            cli.dr = use_drafter
            cli.target_gen = use_target
            # run once (non blocking)
            with st.spinner("Running generation..."):
                t0 = time.time()
                try:
                    result = cli.run_once(prompt)
                    t1 = time.time()
                except Exception as e:
                    st.error(f"Generation error: {e}")
                    result = None

            if result:
                st.success(f"Done in {t1-t0:.2f}s")
                # store latest result in session for display
                st.session_state["last_result"] = result

    # Gamma sweep widget
    st.markdown("---")
    st.subheader("Gamma sweep")
    sweep_gammas = st.text_input("Comma separated gamma values (e.g. 2,4,6,8)", value="2,4,6,8")
    if st.button("Run gamma sweep"):
        if "cli" not in st.session_state:
            st.error("Load models first.")
        else:
            cli = st.session_state["cli"]
            try:
                gamma_list = [int(x.strip()) for x in sweep_gammas.split(",") if x.strip()]
            except ValueError:
                st.error("Invalid gamma list. Use integers separated by commas.")
                gamma_list = []
            if gamma_list:
                # run sweep and collect results
                sweep_results = {}
                with st.spinner("Running gamma sweep... this runs generation multiple times"):
                    res = cli.run_gamma_sweep_cloud(prompt,gamma_list)
                    sweep_results = res
                st.session_state["sweep_results"] = sweep_results
                st.success("Sweep finished.")
                df = pd.DataFrame(sweep_results)
                # data = df.drop(columns=['accept_rate'])
                st.dataframe(df)


with right_col:
    st.subheader("Outputs")

    if "last_result" in st.session_state:
        result = st.session_state["last_result"]

        with st.expander("Our Model (Speculative) output", expanded=True):
            if result.get("speculative") is not None:
                st.markdown(f"**Output:**\n\n{result['speculative']}")
                st.markdown(f"- Acceptance rate: `{result['spec_accept_rate']}`")
                st.markdown(f"- Throughput: `{result['spec_throughput']:.1f}` tokens/s")
            else:
                st.info("Speculative output not enabled or not available.")

        with st.expander("Target (autoregressive) output", expanded=False):
            if result.get("target") is not None:
                st.markdown(f"**Output:**\n\n{result['target']}")
                st.markdown(f"- Throughput: `{result['target_throughput']:.1f}` tokens/s")
            else:
                st.info("Target output not enabled or not available.")

        with st.expander("Drafter (autoregressive) output", expanded=False):
            if result.get("drafter") is not None:
                st.markdown(f"**Output:**\n\n{result['drafter']}")
                st.markdown(f"- Throughput: `{result['drafter_throughput']:.1f}` tokens/s")
            else:
                st.info("Drafter output not enabled or not available.")
    else:
        st.info("No generation yet. Enter a prompt and press Generate.")

    # show sweep plot/results if available
    if "sweep_results" in st.session_state:
        st.markdown("---")
        st.subheader("Gamma sweep results")
        sweep_results = st.session_state["sweep_results"]

        # Ensure gammas are sorted numbers; if keys are strings like "0.1", convert to float
        try:
            gammas = sorted([float(k) for k in sweep_results.keys()])
        except Exception:
            # fallback to default sorting if keys are already numbers or non-convertible
            gammas = sorted(sweep_results.keys())

        # Safely extract values with .get and a fallback of 0
        our_model = [sweep_results[g]["cloud_runtime"] if ("cloud_runtime" in sweep_results[g] and sweep_results[g]["cloud_runtime"] is not None) else 0
                 for g in gammas]

        # Use length of gammas (was sweep_gammas which is undefined)
        target_model = [1] * len(gammas)

        # Plot using figure/ax so we have a fig object to pass to st.pyplot
        fig, ax = plt.subplots(figsize=(6, 4))
        ax.plot(gammas, our_model, marker='o', linewidth=2, markersize=8, label='Our Model')
        ax.plot(gammas, target_model, marker='s', linewidth=2, markersize=8, label='Target Model')

        ax.set_xlabel("Proposal d", fontsize=12, fontweight='bold')
        ax.set_ylabel("Cloud Runtime", fontsize=12, fontweight='bold')
        # ax.set_title("Gamma sweep: Cloud Runtime")
        ax.set_xticks(gammas)
        ax.legend(loc='best', prop={'size': 12, 'weight': 'bold'})
        ax.grid(True, linestyle='--', alpha=0.6)
        fig.tight_layout()

        st.pyplot(fig)

st.markdown("---")
st.caption(" Models: Use small model as Drafer, Big model as Verifier and both model from same family. Ensure you have GPU memory and HF access tokens set up.") 