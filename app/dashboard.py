"""Dashboard: upload a panoramic X-ray, get the AI dental chart. Research use only.

    DENTCHART_MODEL_DIRS=/path/to/WORK streamlit run app/dashboard.py

WORK is the notebook's results folder (runs/teeth/best.pt, runs/findings/best.pt, results.json);
several folders can be given, separated by ':'. In Colab, Part 6 of the notebook starts it.
The X-ray is prepared and charted by dentchart/infer.py.
"""

import os
import sys
from pathlib import Path

import streamlit as st
from PIL import Image

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
from dentchart import infer  # noqa: E402
from dentchart.fdi import CODES  # noqa: E402

MODEL_DIRS = tuple(p for p in os.environ.get("DENTCHART_MODEL_DIRS", "").split(os.pathsep) if p) or (
    "/content/drive/MyDrive/panoramic-dental-charting",)


@st.cache_resource(show_spinner="Loading the trained models...")
def load(dirs):
    return infer.load_models(dirs)


def main():
    st.set_page_config(page_title="AI dental charting", page_icon="🦷", layout="wide")
    st.title("🦷 AI dental charting on panoramic X-rays")
    st.write("Upload a panoramic X-ray (OPG) to number the teeth and find caries, deep caries, periapical "
             "lesions and impacted teeth. Research use only, not a diagnosis.")
    try:
        teeth_model, find_model, device, saved_t, info = load(MODEL_DIRS)
    except Exception as e:  # noqa: BLE001 - shown on the page
        st.error(f"Could not load the trained models: {e}")
        st.stop()
    st.success("Trained models loaded: " + info)

    with st.sidebar:
        st.header("Settings")
        thresh = st.slider("Finding score threshold", 0.05, 0.95, float(round(saved_t, 2)), 0.05,
                           help="Default: the threshold chosen on the validation X-rays.")
        fit = st.checkbox("Fit the scale to the training X-rays", True,
                          help="Helps with crops, screenshots and other machines.")
        crop = st.checkbox("Crop scanner label strips at the edges", True)
        show_teeth = st.checkbox("Show tooth numbers on the X-ray", True)
        show_unassigned = st.checkbox("Show findings not on a detected tooth", False)

    uploaded = st.file_uploader("Choose an OPG image", type=["png", "jpg", "jpeg", "tif", "tiff", "bmp"])
    if uploaded is None:
        return
    with st.spinner("Analysing the X-ray..."):
        result = infer.chart_opg(Image.open(uploaded), teeth_model, find_model, find_thresh=thresh,
                                 device=device, fit=fit, crop=crop)

    col1, col2 = st.columns([2, 1])
    with col1:
        st.subheader("AI detection findings")
        st.pyplot(infer.plot_result(result, title=uploaded.name, show_teeth=show_teeth,
                                    show_unassigned=show_unassigned, report=False))
    with col2:
        st.subheader("Diagnostic report")
        st.markdown(f"**Teeth found:** {len(result['present'])} of 32")
        if result["findings"]:
            for f in result["findings"]:
                st.markdown(f"- **Tooth {f['fdi']}**: {f['diagnosis']} ({CODES.get(f['diagnosis'], '')}), "
                            f"score {f['score']:.2f}")
        else:
            st.write("No findings above the threshold.")
        if result["unassigned"]:
            st.caption(f"{len(result['unassigned'])} finding(s) not on a detected tooth were not charted "
                       "(see the sidebar to show them).")
        for note in result["notes"]:
            st.info(note)
        for warning in result["warnings"]:
            st.warning(warning)
        st.caption("Research use only - not a medical device and not a diagnosis.")


main()
