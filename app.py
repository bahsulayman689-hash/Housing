import streamlit as st
import pandas as pd
import joblib
import numpy as np
import os
import time
from datetime import datetime

# Gemini LLM (optional — app degrades gracefully if not configured)
try:
    import google.generativeai as genai
    GENAI_AVAILABLE = True
except ImportError:
    GENAI_AVAILABLE = False

# 1. Page Configuration
st.set_page_config(
    page_title="Enterprise Valuation Engine & Observability Platform",
    page_icon="🏠",
    layout="wide"
)

# ==========================================================
# ENTERPRISE SLATE GRAY THEME INJECTION
# ==========================================================
st.markdown("""
    <style>
    .stApp {
        background-color: #262626 !important;
        color: #E0E0E0 !important;
    }
    section[data-testid="stSidebar"] {
        background-color: #1C1C1C !important;
        border-right: 1px solid #444444 !important;
    }
    div[data-baseweb="input"] > div, div[data-baseweb="select"] > div {
        background-color: #333333 !important;
        color: #FFFFFF !important;
        border: 1px solid #555555 !important;
    }
    h1, h2, h3, h4, h5, h6, label {
        color: #D3D3D3 !important;
    }
    button[kind="primary"] {
        background-color: #4F5257 !important;
        color: #FFFFFF !important;
        border: 1px solid #70757C !important;
    }
    button[kind="primary"]:hover {
        background-color: #61656C !important;
    }
    .gray-paper {
        background-color: #383A3E !important;
        padding: 25px !important;
        border-radius: 8px !important;
        border: 1px solid #4E5157 !important;
        box-shadow: inset 0px 1px 3px rgba(0,0,0,0.2), 0px 4px 10px rgba(0,0,0,0.3);
        margin-bottom: 20px !important;
    }
    /* Floating Chat Container Styling */
    div[data-testid="stVerticalBlock"] > div:has(div.floating-chat-anchor) {
        position: fixed;
        bottom: 20px;
        right: 20px;
        z-index: 999999;
        background-color: #2E3033 !important;
        padding: 15px;
        border-radius: 10px;
        box-shadow: 0px 4px 15px rgba(0,0,0,0.6);
        border: 1px solid #4A4D52 !important;
        width: 380px;
    }
    /* Keep the chat_input itself inside the floating box's visual bounds */
    div[data-testid="stVerticalBlock"] > div:has(div.floating-chat-anchor) [data-testid="stChatInput"] {
        position: relative !important;
        bottom: 0 !important;
        background-color: #2E3033 !important;
        padding: 0 !important;
    }
    </style>
""", unsafe_allow_html=True)

# 2. State & Session Management
if "telemetry_logs" not in st.session_state:
    st.session_state.telemetry_logs = pd.DataFrame(
        columns=["Timestamp", "Latency_ms", "Predicted_Value", "Status"]
    )

if "chat_history" not in st.session_state:
    st.session_state.chat_history = [
        {"role": "assistant", "content": "Hello! I am your ML Pipeline Assistant. Adjust the parameters on the dashboard, run a metric evaluation, and I can explain the valuation logic, or answer structural feature questions for you."}
    ]

if "chat_open" not in st.session_state:
    st.session_state.chat_open = False

# 3. Load the Unified Model Pipeline (Scaler + XGBoost Model)
@st.cache_resource
def load_pipeline():
    try:
        return joblib.load("best_model.pkl")
    except Exception as e:
        st.error(f"❌ Failed to load the model artifact: {str(e)}")
        return None

model_pipeline = load_pipeline()

# 3b. Configure Gemini (reads from env var or st.secrets — never hardcode a key)
def get_gemini_model():
    if not GENAI_AVAILABLE:
        return None, "google-generativeai package not installed in this environment"

    api_key = os.environ.get("GEMINI_API_KEY")
    if not api_key:
        try:
            api_key = st.secrets["GEMINI_API_KEY"]
        except Exception as e:
            return None, f"could not read GEMINI_API_KEY from st.secrets: {e}"

    if not api_key:
        return None, "GEMINI_API_KEY was empty"

    try:
        genai.configure(api_key=api_key)
        model = genai.GenerativeModel("gemini-3.6-flash")
        return model, None
    except Exception as e:
        return None, f"genai.configure/GenerativeModel failed: {e}"

gemini_model, gemini_error = get_gemini_model()

# ==========================================================
# SIDEBAR: DEVELOPER PROFILE
# ==========================================================
with st.sidebar:
    st.subheader("🛠️ Core Systems Engineer")

    profile_pic_path = "IMG-20260704-WA0629.jpg"
    if os.path.exists(profile_pic_path):
        st.image(profile_pic_path, width=120, use_container_width=False)
    else:
        st.caption("💡 Tip: Upload an 'avatar.jpg' to render your profile picture here.")

    st.markdown("**ML & DL Systems Engineer**")
    st.caption("Tabular Framework Optimization & Low-Latency Inference Deployments")
    st.divider()
    st.markdown("🌐 **Connect or Follow Execution:**")
    st.caption("🔗 [GitHub Production Profile](https://github.com)")
    st.caption("💼 [LinkedIn Professional Network](https://linkedin.com)")
    st.caption("Email test ")

# ==========================================================
# MAIN INTERFACE
# ==========================================================
logo_path = "logo.png"
if os.path.exists(logo_path):
    st.image(logo_path, width=80)

st.title("🏠 Advanced House Price Prediction Engine & Observability Suite")

tab1, tab2, tab3, tab4 = st.tabs([
    "🚀 Engine Inference Console",
    "📈 Live System Monitoring",
    "🧠 Model Specs & Telemetry",
    "📋 Raw Diagnostics & Data Rules"
])

# ==========================================================
# TAB 1: INFERENCE CONSOLE
# ==========================================================
with tab1:
    if model_pipeline is not None:
        st.subheader("📋 Property Dimensional Entry Matrix")

        col1, col2 = st.columns(2)

        with col1:
            st.markdown("#### **Structural Spaces**")
            bedrooms = st.number_input("Bedrooms Total", min_value=0, max_value=20, value=3, step=1)
            bathrooms = st.number_input("Bathrooms Total", min_value=0.0, max_value=10.0, value=2.0, step=0.25)
            floors = st.number_input("Total Floors Layout", min_value=1.0, max_value=4.0, value=1.0, step=0.5)

            st.markdown("#### **Footprint Metrics (Square Footage)**")
            sqft_living = st.number_input("Living Area (Sqft)", min_value=100, max_value=20000, value=2000, step=50)
            sqft_above = st.number_input("Sqft Above Ground Level", min_value=100, max_value=15000, value=1700, step=50)
            sqft_basement = st.number_input("Sqft Basement Area", min_value=0, max_value=10000, value=300, step=50)

        with col2:
            st.markdown("#### **Site & Amenities Profile**")
            waterfront = st.selectbox("Waterfront Location?", options=[0, 1], format_func=lambda x: "Yes (1)" if x == 1 else "No (0)")
            view = st.slider("Visual View Quality Rating", min_value=0, max_value=4, value=0, step=1)
            condition = st.slider("Structural Condition Score", min_value=1, max_value=5, value=3, step=1)
            sqft_lot = st.number_input("Total Lot Property Size (Sqft)", min_value=100, max_value=1000000, value=5000, step=100)

            st.markdown("#### **Temporal Variables**")
            yr_built = st.number_input("Original Year Built", min_value=1800, max_value=2026, value=2000, step=1)
            yr_renovated = st.number_input("Year Renovated (Set 0 if None)", min_value=0, max_value=2026, value=0, step=1)

        st.divider()

        if st.button("💰 Compute Fair Market Valuation", type="primary", use_container_width=True):
            input_dict = {
                "bedrooms": bedrooms, "bathrooms": bathrooms, "sqft_living": sqft_living,
                "sqft_lot": sqft_lot, "floors": floors, "waterfront": waterfront,
                "view": view, "condition": condition, "sqft_above": sqft_above,
                "sqft_basement": sqft_basement, "yr_built": yr_built, "yr_renovated": yr_renovated
            }
            input_df = pd.DataFrame([input_dict])

            start_time = time.time()
            with st.spinner("Processing through pipeline matrices..."):
                try:
                    prediction = model_pipeline.predict(input_df)[0]
                    latency = (time.time() - start_time) * 1000

                    new_log = pd.DataFrame([{
                        "Timestamp": datetime.now().strftime("%H:%M:%S"),
                        "Latency_ms": round(latency, 2),
                        "Predicted_Value": round(float(prediction), 2),
                        "Status": "Success"
                    }])
                    st.session_state.telemetry_logs = pd.concat([st.session_state.telemetry_logs, new_log], ignore_index=True)
                    # Keep the last input+prediction around so the chatbot can ground its answers in it
                    st.session_state.last_prediction = {
                        "inputs": input_dict,
                        "value": float(prediction)
                    }

                    st.balloons()
                    st.success("🎉 Pipeline Inference Execution Complete!")

                    res_col1, res_col2 = st.columns(2)
                    res_col1.metric(label="Estimated Asset Market Value (USD)", value=f"${prediction:,.2f}")
                    res_col2.metric(label="Cost Per Sqft (Living)", value=f"${(prediction / max(1, sqft_living)):,.2f}")

                except Exception as e:
                    latency = (time.time() - start_time) * 1000
                    error_log = pd.DataFrame([{
                        "Timestamp": datetime.now().strftime("%H:%M:%S"),
                        "Latency_ms": round(latency, 2),
                        "Predicted_Value": 0.0,
                        "Status": f"Error: {str(e)}"
                    }])
                    st.session_state.telemetry_logs = pd.concat([st.session_state.telemetry_logs, error_log], ignore_index=True)
                    st.error(f"Inference Boundary Exception Encountered: {str(e)}")

# ==========================================================
# TAB 2: LIVE SYSTEM MONITORING
# ==========================================================
with tab2:
    st.subheader("📊 Live Pipeline Telemetry Tracker")

    if st.session_state.telemetry_logs.empty:
        st.info("⚡ System Idle. Awaiting inference calls inside the 'Engine Inference Console' to compile live charts.")
    else:
        m_col1, m_col2, m_col3 = st.columns(3)
        m_col1.metric(label="Total Logged Predictions", value=len(st.session_state.telemetry_logs))
        m_col2.metric(label="Mean Inference Latency", value=f"{st.session_state.telemetry_logs['Latency_ms'].mean():.2f} ms")
        m_col3.metric(label="Last Forecast Valuation", value=f"${st.session_state.telemetry_logs['Predicted_Value'].iloc[-1]:,.2f}")

        st.divider()

        c_col1, c_col2 = st.columns(2)
        with c_col1:
            st.markdown("#### 🕒 Compute Latency Trend Curve (ms)")
            st.line_chart(st.session_state.telemetry_logs, x="Timestamp", y="Latency_ms")
        with c_col2:
            st.markdown("#### 💲 Computed Asset Valuations History")
            st.bar_chart(st.session_state.telemetry_logs, x="Timestamp", y="Predicted_Value")

        st.markdown("#### 📝 Raw Logs Stack Transaction History")
        st.dataframe(st.session_state.telemetry_logs.iloc[::-1], use_container_width=True)

# ==========================================================
# TAB 3: MODEL SPECS & TELEMETRY
# ==========================================================
with tab3:
    st.subheader("🔍 Deep Architecture & Evaluation Logs")
    st.markdown("* **Algorithm Strategy:** Gradient Boosted Decision Trees (GBDT) via XGBoost.")
    st.markdown("* **Preprocessing Pipeline:** Integrated StandardScaler executing in-memory feature normalization prior to model input.")

# ==========================================================
# TAB 4: RAW DIAGNOSTICS & DATA RULES
# ==========================================================
with tab4:
    st.markdown('<div class="gray-paper">', unsafe_allow_html=True)
    st.subheader("🛡️ Production Pipeline Constraints & Data Rules")
    st.markdown("""
    The training dataset intentionally dropped spatial high-cardinality components (`street`, `city`, `statezip`, `country`)
    alongside internal target parameters (`price`, `date`) during optimization splits to enforce dimensional balance.
    All incoming raw JSON matrices routed to this endpoint must maintain exact datatypes to prevent matrix calculation shifts.
    """)
    st.markdown("#### Sample Schema Validation Blueprint")
    st.json({
        "bedrooms": 3.0,
        "bathrooms": 2.25,
        "sqft_living": 2150,
        "sqft_lot": 5200,
        "floors": 1.5,
        "waterfront": 0,
        "view": 1,
        "condition": 4,
        "sqft_above": 1850,
        "sqft_basement": 300,
        "yr_built": 1995,
        "yr_renovated": 0
    })
    st.markdown('</div>', unsafe_allow_html=True)

# ==========================================================
# HELPER: build a grounded response (LLM if available, rule-based fallback)
# ==========================================================
def generate_bot_response(user_input: str) -> str:
    last_pred = st.session_state.get("last_prediction")

    # Build grounding context from real pipeline state — this is what makes the
    # bot answer buyers accurately instead of giving generic canned text.
    context_lines = [
        "You are a helpful assistant embedded in a real-estate price-prediction dashboard.",
        "The prediction model is an XGBoost regressor wrapped with a StandardScaler in a single scikit-learn Pipeline.",
        "Features used: bedrooms, bathrooms, sqft_living, sqft_lot, floors, waterfront, view, condition, sqft_above, sqft_basement, yr_built, yr_renovated.",
        "street/city/statezip/country and price/date were dropped before training.",
    ]
    if last_pred:
        context_lines.append(f"The most recent valuation run produced: {last_pred['value']:,.2f} USD for inputs: {last_pred['inputs']}.")
    else:
        context_lines.append("No valuation has been run yet in this session.")

    if not st.session_state.telemetry_logs.empty:
        context_lines.append(
            f"Session stats so far — predictions run: {len(st.session_state.telemetry_logs)}, "
            f"mean latency: {st.session_state.telemetry_logs['Latency_ms'].mean():.2f} ms."
        )

    system_context = "\n".join(context_lines)

    # Preferred path: real LLM grounded in the pipeline's actual state
    if gemini_model is not None:
        try:
            prompt = (
                f"{system_context}\n\n"
                f"A prospective buyer asks: \"{user_input}\"\n\n"
                "Answer clearly and concisely (2-4 sentences), using the numbers above when relevant. "
                "If the question can't be answered from the context, say so honestly rather than inventing figures."
            )
            response = gemini_model.generate_content(prompt)
            text = (response.text or "").strip()
            if text:
                return text
        except Exception as e:
            # Fall through to rule-based fallback; surface the failure quietly in the response
            return f"(LLM temporarily unavailable — {str(e)[:80]}) Falling back to basic answer: " + rule_based_response(user_input)

    # Fallback when no API key / library configured
    return rule_based_response(user_input)


def rule_based_response(user_input: str) -> str:
    lower_input = user_input.lower()
    last_pred = st.session_state.get("last_prediction")
    if "xgboost" in lower_input or "model" in lower_input:
        return "Our architecture deploys an optimized XGBoost Regressor connected sequentially behind a StandardScaler."
    elif "price" in lower_input or "valuation" in lower_input:
        if last_pred:
            return f"The last computed valuation was ${last_pred['value']:,.2f} based on the inputs you entered. Check the Live System Monitoring tab for the full trend."
        return "Run a valuation in the Engine Inference Console tab first, then ask me about the result."
    elif "scaler" in lower_input or "leakage" in lower_input:
        return "Data leakage is entirely circumvented by wrapping the scaler and model inside a unified scikit-learn Pipeline wrapper."
    else:
        return "I can answer questions about our XGBoost model, the last price calculation, or data preprocessing — try asking about one of those."

# ==========================================================
# DYNAMIC FLOATING LOWER RIGHT CHATBOT WIDGET
# ==========================================================
with st.container():
    st.markdown('<div class="floating-chat-anchor"></div>', unsafe_allow_html=True)

    if st.button("🤖 Model Assistant Chatbot", use_container_width=True):
        st.session_state.chat_open = not st.session_state.chat_open
        st.rerun()

    if st.session_state.chat_open:
        st.divider()

        chat_container = st.container(height=250)
        with chat_container:
            for msg in st.session_state.chat_history:
                st.chat_message(msg["role"]).write(msg["content"])

            # FIX: chat_input must live *inside* the height-bound container to stay
            # pinned within the floating box. Previously it sat outside chat_container
            # (but still inside the floating wrapper), which made Streamlit dock it to
            # the bottom of the whole page as a full-width bar instead of the widget.
            if user_input := st.chat_input("Ask assistant a context question...", key="floating_chat_input"):
                st.session_state.chat_history.append({"role": "user", "content": user_input})
                bot_response = generate_bot_response(user_input)
                st.session_state.chat_history.append({"role": "assistant", "content": bot_response})
                st.rerun()

        if gemini_model is None:
            st.caption(f"💡 LLM not active — {gemini_error}. Using rule-based fallback for now.")