import os
import streamlit as st
from main import travel_graph, llm, memory
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers.string import StrOutputParser
from dateutil import parser as date_parser
import re

# ✅ Enable LangSmith tracing
os.environ["LANGCHAIN_TRACING_V2"] = "true"
os.environ["LANGCHAIN_PROJECT"] = "travel-planner-ai"

# 🧳 UI Setup
st.title("🧳 AI Travel Planner")

if "msgs" not in st.session_state:
    st.session_state.msgs = []
if "chat_history" not in st.session_state:
    st.session_state.chat_history = []

parser = StrOutputParser()

# ✅ Normalize full user message: convert any dates into 'YYYY-MM-DD'
def normalize_dates_in_text(text):
    date_pattern = r"\b\d{1,2}(?:st|nd|rd|th)?\s+\w+\s+\d{4}|\b\w+\s+\d{1,2},?\s+\d{4}"
    matches = re.findall(date_pattern, text)

    for match in matches:
        try:
            parsed_date = date_parser.parse(match, fuzzy=True)
            iso_date = parsed_date.strftime("%Y-%m-%d")
            text = text.replace(match, iso_date)
        except Exception:
            continue

    return text

# 🔍 Detect whether it's travel planning or casual chat
detect_prompt = ChatPromptTemplate.from_template("""
You are a smart assistant that determines if the user is engaging in a travel planning conversation or just chatting.

Conversation so far:
{history}

Latest user message:
"{input}"

Based on the conversation and the message, respond with one word:
- travel (if the message is related to any ongoing travel planning (even implicitly))
- chat (if it's just casual unrelated conversation)

ONLY respond with: travel or chat.
""")
detect_chain = detect_prompt | llm | parser

# 🤖 Ask for missing details
extract_prompt = ChatPromptTemplate.from_template("""
You are a friendly travel assistant.
Based on the full conversation so far:
{history}

Please determine if you have the following details:
- departure_city
- destination_country
- budget
- preferences
- outbound_date
- return_date

If any are missing, ask for them nicely.
Otherwise only output : ready
""")
extract_chain = extract_prompt | llm | parser

# ✍️ User message input
user_input = st.chat_input("Hi! Planning something fun or just chatting?")

if user_input:
    # ✅ Normalize dates BEFORE saving message
    normalized_input = normalize_dates_in_text(user_input)

    # Save to state
    st.session_state.msgs.append({"role": "user", "content": normalized_input})
    st.session_state.chat_history.append(("user", normalized_input))

    # Full history
    full_chat = "\n".join(f"{r}: {c}" for r, c in st.session_state.chat_history)

    # 🚦 Determine context: chat vs travel
    mode = detect_chain.invoke({"input": normalized_input, "history": full_chat}).strip().lower()

    if mode == "chat":
        response = llm.invoke(st.session_state.chat_history)
        st.session_state.msgs.append({"role": "assistant", "content": response.content})
        st.session_state.chat_history.append(("assistant", response.content))

    elif mode == "travel":
        extracted = extract_chain.invoke({"history": full_chat}).strip()

        if "ready" in extracted.lower():
            # ✅ Build state for travel_graph
            state = {"user_input": full_chat}

            # Dates are already normalized in chat history, so nothing to fix here
            result = travel_graph.invoke(state)

            itinerary = result.get("itinerary", "Sorry, I couldn't generate a plan.")
            st.session_state.msgs.append({"role": "assistant", "content": itinerary})
            st.session_state.chat_history.append(("assistant", itinerary))
        else:
            # Ask user for missing data
            st.session_state.msgs.append({"role": "assistant", "content": extracted})
            st.session_state.chat_history.append(("assistant", extracted))

# 💬 Render chat UI
for msg in st.session_state.msgs:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])
