import streamlit as st
from self_rag import self_reflective_rag
from basic_rag import basic_rag

st.set_page_config(
    page_title="Interactive Chat System",
    layout="wide",
)

st.title("Chat with the knowledge base")

# Add a dropdown to select the document
option = st.selectbox(
    "Select a document:",
    ["Annual Report (Self reflective RAG)", "Airbus Annual Report (Basic RAG)"]
)

# Initialize or retrieve the conversation history
if "messages" not in st.session_state:
    st.session_state.messages = []

# Function to clear session state
def clear_session_state():
    st.session_state.messages = []

# Check if the dropdown selection has changed
if "selected_option" in st.session_state and st.session_state.selected_option != option:
    clear_session_state()

# Store the selected option in session state
st.session_state.selected_option = option

for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        try:
            with st.expander("Backend details"):
                st.markdown(message["backend_details"])
        except KeyError:
            pass
        st.markdown(message["content"].replace("$", r"\$"))

if prompt := st.chat_input("Ask me about the document..."):
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    with st.chat_message("assistant"):
        if option == "Annual Report (Self reflective RAG)":
            backend_details_placeholder = st.expander("Backend details", expanded=True)
            message_placeholder = st.empty()

            full_response = ""
            backend_details = ""
            full_backend_details = ""
            message_placeholder.markdown("Thinking...")

            for update in self_reflective_rag(prompt, st.session_state.messages): 
                full_response = update["full_response"]
                backend_details = update["backend_details"]
                full_backend_details += backend_details
                # Update the placeholders with the new content
                backend_details_placeholder.markdown(backend_details)

            message_placeholder.markdown(full_response.replace("$", r"\$"))

            st.session_state.messages.append(
                {
                    "role": "assistant",
                    "content": full_response,
                    "backend_details": full_backend_details,
                }
            )
        elif option == "Airbus Annual Report (Basic RAG)":
            message_placeholder = st.empty()

            full_response = ""
            message_placeholder.markdown("Thinking...")

            for update in basic_rag(prompt, st.session_state.messages): 
                full_response = update["full_response"]
                message_placeholder.markdown(full_response.replace("$", r"\$"))

            st.session_state.messages.append(
                {"role": "assistant", "content": full_response}
            )
