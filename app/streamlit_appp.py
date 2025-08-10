import streamlit as st
# --- add these lines at the top ---
import os, sys
ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)
# --- end patch ---
import json
from datetime import datetime, timedelta
import pandas as pd

from core.utils.text_loader import load_docs
from core.stage1_qa.qa import answer_question
from core.stage1_sentiment.sentiment import analyze_sentiment
from core.stage1_anomaly.anomaly import compute_anomalies
from core.stage2_forecast.forecast import forecast_returns
from core.stage3_decision.decision import decide
from core.schemas import Stage1QA, Stage1Sentiment, Stage1Anomaly, Stage2Forecast, Stage3Decision

st.set_page_config(page_title='FinDocGPT', layout='wide')
st.markdown("<h1 style='text-align:center; margin-top:0;'>FinDocGPT</h1>", unsafe_allow_html=True)

st.sidebar.header('Navigation')
page = st.sidebar.radio(
    "Choose a stage",
    ["Stage 1: Docs & Q&A", "Stage 2: Forecast", "Stage 3: Decision", "Run All: Demo Pipeline", "Chat: Decision Q&A", "About"]
)

DATA_DIR = os.path.join(os.path.dirname(__file__), '..', 'data', 'sample_docs')

def ensure_session():
    for k, v in [('docs', []), ('qa_json', None), ('sent_json', None), ('an_json', None), ('fc_json', None), ('dec_json', None)]:
        if k not in st.session_state:
            st.session_state[k] = v
ensure_session()

def use_sample_docs():
    sample_paths = [os.path.join(DATA_DIR, fn) for fn in os.listdir(DATA_DIR) if fn.lower().endswith(('.txt', '.pdf'))]
    st.session_state.docs = load_docs(sample_paths, default_company='SampleCo', default_period='Sample')
    st.success(f'Loaded {len(st.session_state.docs)} sample doc(s).')

# -----------------------
# Stage 1
# -----------------------
if page.startswith('Stage 1'):
    st.header('Stage 1 - Upload Docs, Ask Q&A, Sentiment & Anomalies')

    col1, col2 = st.columns(2)
    with col1:
        uploaded = st.file_uploader('Upload .txt or .pdf (you can add multiple)', type=['txt', 'pdf'], accept_multiple_files=True)
        if uploaded:
            tmp_paths = []
            for f in uploaded:
                path = os.path.join('/tmp', f.name)
                with open(path, 'wb') as out:
                    out.write(f.read())
                tmp_paths.append(path)
            st.session_state.docs = load_docs(tmp_paths, default_company='UnknownCo', default_period='Unknown')
            st.success(f'Loaded {len(st.session_state.docs)} document(s).')
        st.button('Use sample docs', on_click=use_sample_docs)

    with col2:
        use_embeddings = st.checkbox('Use Embeddings (FAISS)')
        use_gpt41 = st.checkbox('Use GPT-4.1 synthesis')
        question = st.text_input("Question (e.g., 'What was revenue in 2023?')", value="What risks are mentioned?")

    if st.button('Run Q&A'):
        try:
            if use_gpt41:
                from core.stage1_qa.qa_gpt import answer_question_gpt41
                qa: Stage1QA = answer_question_gpt41(question, st.session_state.docs, k=3, model="gpt-4.1")
            elif use_embeddings:
                from core.stage1_qa.qa_embed import answer_question_embed
                qa: Stage1QA = answer_question_embed(question, st.session_state.docs)
            else:
                qa: Stage1QA = answer_question(question, st.session_state.docs)
            st.session_state.qa_json = qa.model_dump()
        except Exception as e:
            st.error(f"Q&A error: {e}")


        if st.session_state.qa_json:
            st.subheader('Q&A JSON')
            st.json(st.session_state.qa_json)

            # Optional: Top-k evidence (only when embeddings are used)
            if 'use_embeddings' in locals() and use_embeddings and st.session_state.docs:
                try:
                    from core.stage1_qa.qa_embed import topk_passages_embed
                    topk = topk_passages_embed(question, st.session_state.docs, k=3)
                    if topk:
                        st.subheader("Top-k Embedding Evidence")
                        for item in topk:
                            st.markdown(f"**#{item['rank']}** • Score: {item['score']:.3f} • Source: `{item['source']}`")
                            st.write(item['snippet'])
                            st.divider()
                    else:
                        st.caption("Top-k evidence unavailable (embeddings not installed or no passages).")
                except Exception as e:
                    st.caption(f"Top-k evidence error: {e}")

    st.divider()
    st.subheader('Sentiment')
    if st.button('Analyze sentiment'):
        try:
            sent: Stage1Sentiment = analyze_sentiment(st.session_state.docs)
            st.session_state.sent_json = sent.model_dump()
        except Exception as e:
            st.error(f"Sentiment error: {e}")

    if st.session_state.sent_json:
        st.json(st.session_state.sent_json)

    st.divider()
    st.subheader('Anomalies (enter a small series)')
    with st.form('an_form'):
        st.write('Enter periods and values (comma-separated). Example: 2023-Q1,2023-Q2,2023-Q3,2023-Q4')
        periods = st.text_input('Periods', value='2023-Q1,2023-Q2,2023-Q3,2023-Q4,2024-Q1')
        values = st.text_input('Values', value='100,120,118,180,130')
        submitted = st.form_submit_button('Detect anomalies')
        if submitted:
            try:
                per = [p.strip() for p in periods.split(',') if p.strip()]
                vals = [float(v.strip()) for v in values.split(',') if v.strip()]
                series = [{'period': p, 'value': v} for p, v in zip(per, vals)]
                an = compute_anomalies(series, z_threshold=2.0)
                st.session_state.an_json = an.model_dump()
            except Exception as e:
                st.error(f'Error: {e}')

    if st.session_state.an_json:
        st.json(st.session_state.an_json)

    st.divider()
    st.subheader("Tables (experimental)")
    if st.button("Extract tables from PDFs"):
        try:
            from core.utils.table_extract import extract_tables_pdf
            any_found = False
            for d in st.session_state.docs:
                p = d.get("path","")
                if p and p.lower().endswith(".pdf") and os.path.exists(p):
                    st.markdown(f"**Source:** `{d.get('id','')}`")
                    dfs = extract_tables_pdf(p, max_tables=2)
                    if dfs:
                        any_found = True
                        for j, df in enumerate(dfs, start=1):
                            st.write(f"Table {j}")
                            st.dataframe(df)
                            st.caption("If numbers look off, PDF extraction may need tuning.")
                            st.divider()
            if not any_found:
                st.info("No PDF tables found or PDF parsing libs not available.")
        except Exception as e:
            st.error(f"Table extraction error: {e}")

# -----------------------
# Stage 2
# -----------------------
elif page.startswith('Stage 2'):
    st.header('Stage 2 - Forecast (ARIMA on daily returns)')
    ticker = st.text_input('Ticker', value='AAPL')
    c1, c2, c3 = st.columns(3)
    with c1:
        start = st.date_input('Start date', value=(datetime.today() - timedelta(days=365)))
    with c2:
        end = st.date_input('End date', value=datetime.today())
    with c3:
        horizon = st.number_input('Horizon (days)', min_value=1, max_value=30, value=5)

    if st.button('Fetch & Forecast'):
        with st.spinner("Fetching data & fitting model..."):
            try:
                fc: Stage2Forecast = forecast_returns(ticker, start.isoformat(), end.isoformat(), horizon=int(horizon))
                st.session_state.fc_json = fc.model_dump()
                st.success('Forecast complete.')
            except Exception as e:
                st.error(f'Error: {e}')

    if st.session_state.fc_json:
        st.subheader('Forecast JSON')
        st.json(st.session_state.fc_json)
        # Backtest hint
        bt = (st.session_state.fc_json or {}).get("answer", {}).get("backtest", {})
        if bt:
            st.caption("Backtest metrics shown above: lower **RMSE** is better. sMAPE/MAPE are % error metrics (lower is better).")

        # Forecast chart
        try:
            import plotly.graph_objects as go
            fc_list = st.session_state.fc_json["answer"]["forecast"]
            df = pd.DataFrame(fc_list)
            df["l80"] = df["pi80"].apply(lambda x: x[0])
            df["u80"] = df["pi80"].apply(lambda x: x[1])
            df["l95"] = df["pi95"].apply(lambda x: x[0])
            df["u95"] = df["pi95"].apply(lambda x: x[1])
            fig = go.Figure()
            # 95% band
            fig.add_traces([
                go.Scatter(x=df["date"], y=df["u95"], line=dict(width=0), name="95% Upper", showlegend=False),
                go.Scatter(x=df["date"], y=df["l95"], line=dict(width=0), name="95% Lower", fill="tonexty", opacity=0.2, showlegend=False),
            ])
            # 80% band
            fig.add_traces([
                go.Scatter(x=df["date"], y=df["u80"], line=dict(width=0), name="80% Upper", showlegend=False),
                go.Scatter(x=df["date"], y=df["l80"], line=dict(width=0), name="80% Lower", fill="tonexty", opacity=0.3, showlegend=False),
            ])
            # Mean
            fig.add_trace(go.Scatter(x=df["date"], y=df["mean"], name="Mean Forecast"))
            fig.update_layout(title="Forecast Chart", xaxis_title="Date", yaxis_title="Daily Return")
            st.subheader("Forecast Chart")
            st.plotly_chart(fig, use_container_width=True)
        except Exception as e:
            st.caption(f"Chart unavailable: {e}")

# -----------------------
# Stage 3
# -----------------------
elif page.startswith('Stage 3'):
    st.header('Stage 3 - Decision (Buy/Sell/Hold)')
    st.write("We use the last forecast point and any anomaly flags as signals.")

    # Decision strictness selector
    dec_mode = st.selectbox("Decision strictness", ["strict", "medium", "aggressive"], index=0)

    # Build signals
    signals = {}
    if st.session_state.fc_json:
        last = st.session_state.fc_json["answer"]["forecast"][-1]
        signals["forecast_return_next_h"] = {"mean": last["mean"], "pi95": last["pi95"], "pi80": last.get("pi80")}
    else:
        st.info("No forecast yet; go to Stage 2 to create one. Using demo values.")
        signals["forecast_return_next_h"] = {"mean": 0.01, "pi95": [-0.005, 0.025]}

    if st.session_state.an_json and st.session_state.an_json["answer"]["flags"]:
        signals["risk_flags"] = [f['label'] for f in st.session_state.an_json["answer"]["flags"] if f['label'] == "risk"]
    else:
        signals["risk_flags"] = []

    # Add sentiment signal if available
    if st.session_state.sent_json:
        overall = st.session_state.sent_json.get("answer", {}).get("overall")
        if overall:
            signals["sentiment_overall"] = overall  # expects {'label':..., 'score':...}

    st.subheader("Signals")
    st.code(json.dumps(signals, indent=2))

    # Auto-decide once signals exist (ensures a decision shows up on deploy too)
    if signals and not st.session_state.get("dec_json"):
        try:
            dec = decide(signals, policy={"max_position_pct": 5, "stop_loss_pct": 8, "take_profit_pct": 15, "min_confidence": "medium", "decisiveness": dec_mode})
            st.session_state.dec_json = dec.model_dump()
        except Exception as e:
            st.error(f"Decision error: {e}")

    # Button to re-run decision if user changes strictness
    if st.button('Decide / Recalculate'):
        try:
            dec = decide(signals, policy={"max_position_pct": 5, "stop_loss_pct": 8, "take_profit_pct": 15, "min_confidence": "medium", "decisiveness": dec_mode})
            st.session_state.dec_json = dec.model_dump()
        except Exception as e:
            st.error(f"Decision error: {e}")

    if st.session_state.dec_json:
        st.subheader('Decision JSON')
        st.json(st.session_state.dec_json)

        # Risk note in UI
        if signals.get("risk_flags"):
            st.warning("Risk flags detected from anomalies: " + ", ".join(signals["risk_flags"]))

        # Rationale bullets
        try:
            bullets = st.session_state.dec_json["answer"].get("rationale_bullets", [])
            if bullets:
                st.subheader("Rationales")
                for b in bullets:
                    st.write(f"• {b}")
        except Exception:
            pass

        # Optional: GPT-4.1 rationale refinement (only if pkg+key available)
        import importlib.util, os
        has_openai_pkg = importlib.util.find_spec("openai") is not None
        has_openai_key = bool(os.environ.get("OPENAI_API_KEY")) or ("OPENAI_API_KEY" in getattr(st, "secrets", {}))
        if has_openai_pkg and has_openai_key:
            if st.checkbox("Refine rationales with GPT-4.1"):
                try:
                    from core.stage3_decision.rationales_gpt import refine_rationales
                    signals_local = signals
                    raw = st.session_state.dec_json["answer"].get("rationale_bullets", [])
                    better = refine_rationales(signals_local, raw, model="gpt-4.1")
                    if better:
                        st.session_state.dec_json["answer"]["rationale_bullets"] = better
                        st.success("Rationales refined.")
                except Exception as e:
                    st.error(f"Rationale polish error: {e}")
        else:
            st.caption("Rationale refinement needs the 'openai' package and an OPENAI_API_KEY. Feature disabled.")

        # Download all results
        try:
            combined = {
                "qa": st.session_state.qa_json or {},
                "sentiment": st.session_state.sent_json or {},
                "anomalies": st.session_state.an_json or {},
                "forecast": st.session_state.fc_json or {},
                "decision": st.session_state.dec_json or {},
            }
            st.download_button(
                label="Download all results (JSON)",
                data=json.dumps(combined, indent=2),
                file_name="fin-docgpt_output.json",
                mime="application/json"
            )
        except Exception as e:
            st.caption(f"Download unavailable: {e}")


# -----------------------
# Run All
# -----------------------
elif page.startswith('Run All'):
    st.header('Run All — End-to-End Demo')
    st.write("This runs Q&A → Sentiment → Anomalies → Forecast → Decision automatically.")

    # Ensure we have some docs
    if not st.session_state.docs:
        try:
            sample_paths = [os.path.join(DATA_DIR, fn) for fn in os.listdir(DATA_DIR) if fn.lower().endswith(('.txt', '.pdf'))]
            if sample_paths:
                st.session_state.docs = load_docs(sample_paths, default_company='SampleCo', default_period='Sample')
                st.info(f"Loaded {len(st.session_state.docs)} sample doc(s). You can upload your own in Stage 1.")
        except Exception as e:
            st.error(f"Could not load sample docs: {e}")

    # Params (editable for live demo)
    question = st.text_input("Question", "What risks are mentioned?")
    ticker = st.text_input("Ticker", "AAPL").upper()
    c1, c2, c3 = st.columns(3)
    with c1:
        start = st.date_input("Start", (datetime.today() - timedelta(days=365))).isoformat()
    with c2:
        end = st.date_input("End", datetime.today()).isoformat()
    with c3:
        horizon = st.number_input("Horizon (days)", min_value=1, max_value=30, value=5)

    if st.button("Run end-to-end"):
        with st.spinner("Running all stages..."):
            # Stage 1: Q&A
            qa = None
            try:
                qa = answer_question(question, st.session_state.docs).model_dump()
                st.session_state.qa_json = qa
            except Exception as e:
                st.error(f"Q&A error: {e}")

            # Stage 1: Sentiment
            sent = None
            try:
                sent = analyze_sentiment(st.session_state.docs).model_dump()
                st.session_state.sent_json = sent
            except Exception as e:
                st.error(f"Sentiment error: {e}")

            # Stage 1: Anomalies (demo series)
            an = None
            try:
                series = [
                    {"period": "2023-Q1", "value": 100},
                    {"period": "2023-Q2", "value": 120},
                    {"period": "2023-Q3", "value": 118},
                    {"period": "2023-Q4", "value": 180},
                    {"period": "2024-Q1", "value": 130},
                ]
                an = compute_anomalies(series, z_threshold=2.0).model_dump()
                st.session_state.an_json = an
            except Exception as e:
                st.error(f"Anomaly error: {e}")

            # Stage 2: Forecast
            fc = None
            try:
                fc = forecast_returns(ticker, start, end, horizon=int(horizon)).model_dump()
                st.session_state.fc_json = fc
            except Exception as e:
                st.error(f"Forecast error: {e}")

            # Stage 3: Decision
            dec = None
            try:
                signals = {}
                # Forecast signal
                if fc and fc.get("answer", {}).get("forecast"):
                    last = fc["answer"]["forecast"][-1]
                    signals["forecast_return_next_h"] = {"mean": last["mean"], "pi95": last["pi95"]}
                else:
                    signals["forecast_return_next_h"] = {"mean": 0.01, "pi95": [-0.005, 0.025]}

                # Risk flags
                if an and an.get("answer", {}).get("flags"):
                    signals["risk_flags"] = [f['label'] for f in an["answer"]["flags"] if f['label'] == "risk"]
                else:
                    signals["risk_flags"] = []

                # Sentiment (from this run or prior session state as fallback)
                sent_overall = None
                if sent and sent.get("answer", {}).get("overall"):
                    sent_overall = sent["answer"]["overall"]
                elif st.session_state.get("sent_json") and st.session_state["sent_json"].get("answer", {}).get("overall"):
                    sent_overall = st.session_state["sent_json"]["answer"]["overall"]
                if sent_overall:
                    signals["sentiment_overall"] = sent_overall

                dec = decide(
                    signals,
                    policy={"max_position_pct": 5, "stop_loss_pct": 8, "take_profit_pct": 15, "min_confidence": "medium"}
                ).model_dump()
                st.session_state.dec_json = dec
            except Exception as e:
                st.error(f"Decision error: {e}")

            # Summary badge
            if dec:
                action = dec["answer"]["action"]
                st.success(f"Action: {action}  |  Position: {dec['answer']['position_size_pct']}%  |  Stop: {dec['answer']['risk_management']['stop_loss_pct']}%  |  TP: {dec['answer']['risk_management']['take_profit_pct']}%")

            # Show all JSONs
            st.subheader("Q&A"); st.json(qa or {})
            st.subheader("Sentiment"); st.json(sent or {})
            st.subheader("Anomalies"); st.json(an or {})
            st.subheader("Forecast"); st.json(fc or {})
            st.subheader("Decision"); st.json(dec or {})

            # Backtest hint
            bt = (fc or {}).get("answer", {}).get("backtest", {})
            if bt:
                st.caption("Backtest metrics shown above: lower **RMSE** is better. sMAPE/MAPE are % error metrics (lower is better).")

            # Forecast chart (Run All)
            try:
                if fc and fc.get("answer", {}).get("forecast"):
                    import plotly.graph_objects as go
                    df = pd.DataFrame(fc["answer"]["forecast"])
                    df["l80"] = df["pi80"].apply(lambda x: x[0])
                    df["u80"] = df["pi80"].apply(lambda x: x[1])
                    df["l95"] = df["pi95"].apply(lambda x: x[0])
                    df["u95"] = df["pi95"].apply(lambda x: x[1])
                    fig = go.Figure()
                    fig.add_traces([
                        go.Scatter(x=df["date"], y=df["u95"], line=dict(width=0), name="95% Upper", showlegend=False),
                        go.Scatter(x=df["date"], y=df["l95"], line=dict(width=0), name="95% Lower", fill="tonexty", opacity=0.2, showlegend=False),
                    ])
                    fig.add_traces([
                        go.Scatter(x=df["date"], y=df["u80"], line=dict(width=0), name="80% Upper", showlegend=False),
                        go.Scatter(x=df["date"], y=df["l80"], line=dict(width=0), name="80% Lower", fill="tonexty", opacity=0.3, showlegend=False),
                    ])
                    fig.add_trace(go.Scatter(x=df["date"], y=df["mean"], name="Mean Forecast"))
                    fig.update_layout(title="Forecast Chart (Run All)", xaxis_title="Date", yaxis_title="Daily Return")
                    st.subheader("Forecast Chart (Run All)")
                    st.plotly_chart(fig, use_container_width=True)
            except Exception as e:
                st.caption(f"Chart unavailable: {e}")

            # Download All JSON
            try:
                combined = {
                    "qa": qa or {},
                    "sentiment": sent or {},
                    "anomalies": an or {},
                    "forecast": fc or {},
                    "decision": dec or {},
                }
                st.subheader("Download All JSON")
                st.download_button(
                    label="Download results as JSON",
                    data=json.dumps(combined, indent=2),
                    file_name="fin-docgpt_output.json",
                    mime="application/json"
                )
            except Exception as e:
                st.caption(f"Download unavailable: {e}")
# -----------------------
# Chat
# -----------------------
elif page.startswith('Chat'):
    st.header("Chat — Ask about the Decision & Framework")
    st.caption("Grounded in your latest run (Q&A, sentiment, anomalies, forecast, decision). Not financial advice.")

    # Check API key
    import os
    if not os.environ.get("OPENAI_API_KEY"):
        st.info("Set OPENAI_API_KEY to enable chat. Example (PowerShell):  `$env:OPENAI_API_KEY=\"sk-...\"`")
    else:
        # Init history
        if "chat_history" not in st.session_state:
            st.session_state.chat_history = []

        # Small context preview
        colA, colB, colC = st.columns(3)
        with colA:
            dec = (st.session_state.dec_json or {}).get("answer", {})
            st.metric("Decision", dec.get("action", "—"))
        with colB:
            sent = (st.session_state.sent_json or {}).get("answer", {}).get("overall") or {}
            st.metric("Sentiment", f"{sent.get('label','—')}", sent.get("score"))
        with colC:
            bt = (st.session_state.fc_json or {}).get("answer", {}).get("backtest", {}) or {}
            st.metric("RMSE (lower better)", bt.get("RMSE", "—"))

        # Render history
        for m in st.session_state.chat_history:
            st.chat_message(m["role"]).write(m["content"])

        # Input box
        user_msg = st.chat_input("Ask about the action, risk, backtest, or how the pipeline works…")
        if user_msg:
            st.session_state.chat_history.append({"role": "user", "content": user_msg})
            with st.chat_message("assistant"):
                try:
                    from core.chat.chatbot import chat_reply
                    reply = chat_reply(user_msg, st.session_state, model="gpt-4.1")
                except Exception as e:
                    reply = f"Chat error: {e}"
                st.write(reply)
            st.session_state.chat_history.append({"role": "assistant", "content": reply})

        # Tools
        cols = st.columns(2)
        if cols[0].button("Clear chat"):
            st.session_state.chat_history = []
        if cols[1].button("Paste latest run summary"):
            # Quick helper to drop a summary into the chat
            from core.chat.chatbot import _build_context
            ctx = _build_context(st.session_state)
            st.session_state.chat_history.append({
                "role": "assistant",
                "content": f"Here is the latest run context:\n\n```json\n{json.dumps(ctx, indent=2)}\n```"
            })
            st.experimental_rerun()

# -----------------------
# About
# -----------------------
else:
    st.header('About & Next Steps')
    st.write('This is a Demo Phase of FinDocGPT. Suggested upgrades: embeddings Q&A, better sentiment, exogenous features in forecasting, and a single Run All Stages button.')

# --- Footer (always visible) ---
st.markdown("---")
st.caption("FinDocGPT is a demo for a hackathon. .")
