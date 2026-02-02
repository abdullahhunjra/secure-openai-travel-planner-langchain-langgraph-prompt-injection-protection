import os
import uuid
import streamlit as st
from main import travel_graph, llm, memory
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers.string import StrOutputParser
from dateutil import parser as date_parser
import re

# Security imports
from security.middleware import SecurityMiddleware
from security.prompt.hardened_templates import HardenedPromptTemplates
from security.logging.security_logger import SecurityLogger, SecurityEventType

# Enable LangSmith tracing
os.environ["LANGCHAIN_TRACING_V2"] = "true"
os.environ["LANGCHAIN_PROJECT"] = "travel-planner-ai"

# Initialize security components
security_logger = SecurityLogger()
security_middleware = SecurityMiddleware(logger=security_logger)

# UI Setup
st.title("AI Travel Planner")

# Initialize session state
if "msgs" not in st.session_state:
    st.session_state.msgs = []
if "chat_history" not in st.session_state:
    st.session_state.chat_history = []
if "session_id" not in st.session_state:
    st.session_state.session_id = str(uuid.uuid4())

parser = StrOutputParser()


# Normalize full user message: convert any dates into 'YYYY-MM-DD'
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


# Detect whether it's travel planning or casual chat (using hardened prompt)
detect_prompt = HardenedPromptTemplates.get_detect_mode_prompt()
detect_chain = detect_prompt | llm | parser

# Ask for missing details (using hardened prompt)
extract_prompt = HardenedPromptTemplates.get_extract_details_prompt()
extract_chain = extract_prompt | llm | parser


def display_security_warning(message: str):
    """Display a security warning to the user."""
    st.warning(f"Security Notice: {message}")


def process_user_input(user_input: str) -> tuple[bool, str, str]:
    """
    Process user input through security middleware.

    Returns:
        Tuple of (is_allowed, sanitized_input, error_message)
    """
    session_id = st.session_state.session_id

    # Run security checks (skip topic check for initial chat detection)
    security_result = security_middleware.check(
        user_input=user_input,
        session_id=session_id,
        skip_topic_check=True  # We'll do topic classification with the LLM
    )

    if not security_result.is_allowed:
        return False, "", security_result.block_reason or "Security check failed"

    # Log warnings if any
    if security_result.warnings:
        for warning in security_result.warnings:
            security_logger.log_event(
                event_type=SecurityEventType.SECURITY_CHECK_PASSED,
                session_id=session_id,
                details={"warning": warning},
                severity="warning"
            )

    return True, security_result.sanitized_input, ""


# User message input
user_input = st.chat_input("Hi! Planning something fun or just chatting?")

if user_input:
    session_id = st.session_state.session_id

    # Run security checks on input
    is_allowed, sanitized_input, error_message = process_user_input(user_input)

    if not is_allowed:
        # Display error and add to chat history
        error_response = f"I couldn't process that request. {error_message}"
        st.session_state.msgs.append({"role": "user", "content": user_input})
        st.session_state.msgs.append({"role": "assistant", "content": error_response})
        st.session_state.chat_history.append(("user", user_input))
        st.session_state.chat_history.append(("assistant", error_response))
    else:
        # Normalize dates in sanitized input
        normalized_input = normalize_dates_in_text(sanitized_input)

        # Save to state
        st.session_state.msgs.append({"role": "user", "content": normalized_input})
        st.session_state.chat_history.append(("user", normalized_input))

        # Full history
        full_chat = "\n".join(f"{r}: {c}" for r, c in st.session_state.chat_history)

        # Sanitize input for prompt
        sanitized_for_prompt = HardenedPromptTemplates.sanitize_for_prompt(normalized_input)
        sanitized_history = HardenedPromptTemplates.sanitize_for_prompt(full_chat)

        # Determine context: chat vs travel
        mode = detect_chain.invoke({
            "input": sanitized_for_prompt,
            "history": sanitized_history
        }).strip().lower()

        if mode == "chat":
            response = llm.invoke(st.session_state.chat_history)
            st.session_state.msgs.append({"role": "assistant", "content": response.content})
            st.session_state.chat_history.append(("assistant", response.content))

        elif mode == "travel":
            extracted = extract_chain.invoke({"history": sanitized_history}).strip()

            if "ready" in extracted.lower():
                # Build state for travel_graph with session_id
                state = {
                    "user_input": full_chat,
                    "session_id": session_id
                }

                # Run travel graph (which now includes security gate)
                result = travel_graph.invoke(state)

                itinerary = result.get("itinerary", "Sorry, I couldn't generate a plan.")
                st.session_state.msgs.append({"role": "assistant", "content": itinerary})
                st.session_state.chat_history.append(("assistant", itinerary))
            else:
                # Ask user for missing data
                st.session_state.msgs.append({"role": "assistant", "content": extracted})
                st.session_state.chat_history.append(("assistant", extracted))

# Render chat UI
for msg in st.session_state.msgs:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])
