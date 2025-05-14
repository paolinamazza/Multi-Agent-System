import streamlit as st
import asyncio
import matplotlib.pyplot as plt
from matplotlib.figure import Figure
from main import agent_with_memory, conversation_agent

# Fullscreen layout, light blue aesthetic, custom emoji icon
st.set_page_config(page_title="NoiPA Chatbot", page_icon="🤖", layout="wide")

# Custom styling
st.markdown("""
    <style>
    .stApp {
        background-color: #e6f2ff;
        color: #000000;
    }
    .stTextInput > label, .stMarkdown, .stButton {
        color: #000000 !important;
    }
    .stChatMessage {
        border-radius: 12px;
        padding: 10px 15px;
        margin-bottom: 10px;
    }
    .stChatMessage.user {
        background-color: #e1f5fe;
        color: #000000;
        border: 1px solid #b3e5fc;
    }
    .stChatMessage.assistant {
        background-color: #ffffff;
        color: #000000;
        border: 1px solid #b3d9ff;
    }
    input[type="text"] {
        color: #000000 !important;
        background-color: #ffffff !important;
        border: 1px solid #b3d9ff !important;
    }
    .clear-button {
        position: absolute;
        top: 1rem;
        right: 1rem;
    }
    </style>
""", unsafe_allow_html=True)

# Logo
st.image("noipalogo.png", width=800)

# Initialize memory and history
if "memory_buffer" not in st.session_state:
    st.session_state.memory_buffer = []
if "chat_history" not in st.session_state:
    st.session_state.chat_history = []

max_token_length = 3000

# Agent runner
def run_sync(query):
    loop = asyncio.new_event_loop()
    asyncio.set_event_loop(loop)

    async def _internal():
        return await agent_with_memory(query, conversation_agent, st.session_state.memory_buffer, max_token_length)

    result = loop.run_until_complete(_internal())
    st.session_state.memory_buffer = result["memory_buffer"]
    st.session_state.chat_history.append({
        "user": query,
        "assistant": result
    })
    return result

# Title & intro
st.title("NoiPA Chatbot")
st.markdown("Ask me anything! I will assist you with analysis, insights, and visualizations.")

# Top-level action buttons
col1, col2 = st.columns([1, 1])

with col1:
    if st.button("🗑️ Clear Conversation", help="Reset chat history and memory"):
        st.session_state.chat_history = []
        st.session_state.memory_buffer = []

with col2:
    if st.button("🔁 Re-run Last Query", help="Re-run the most recent query"):
        if st.session_state.chat_history:
            last_query = st.session_state.chat_history[-1]["user"]
            with st.spinner("Re-processing the last request..."):
                result = run_sync(last_query)

            with st.chat_message("user"):
                st.markdown(f"**👤 You (Re-run):** {last_query}")

            with st.chat_message("assistant"):
                if isinstance(result, dict):
                    for key, val in result.items():
                        if key != "memory_buffer" and key != "fig":
                            label = "**🤖 ChatBot:**" if key == "output" else f"**{key}:**"
                            st.markdown(label)
                            st.write(val)
                        elif key == "fig":
                            if isinstance(val, list):
                                for fig in val:
                                    if isinstance(fig, Figure):
                                        st.pyplot(fig)
                            elif isinstance(val, Figure):
                                st.pyplot(val)
                else:
                    st.write(result)
            if plt.get_fignums():
                st.pyplot(plt.gcf())



# Chat history
for idx, chat in enumerate(st.session_state.chat_history):
    with st.chat_message("user"):
        st.markdown(f"**user:** {chat['user']}")

    with st.chat_message("assistant"):
        response = chat["assistant"]
        if isinstance(response, dict):
            for key, val in response.items():
                if key != "memory_buffer" and key != "fig":
                    st.markdown(f"**{key}:**")
                    st.write(val)
                elif key == "fig":
                    if isinstance(val, list):
                        for fig in val:
                            if isinstance(fig, Figure):
                                st.pyplot(fig)
                    elif isinstance(val, Figure):
                        st.pyplot(val)
        else:
            st.write(response)

# 💬 Input + spinner + live display
query = st.chat_input(placeholder="Type your request here and press Enter...")

if query:
    with st.spinner(" I am processing your request... "):
        result = run_sync(query)

    with st.chat_message("user"):
        st.markdown(f"**👤 You:** {query}")

    with st.chat_message("assistant"):
        if isinstance(result, dict):
            for key, val in result.items():
                if key != "memory_buffer" and key != "fig":
                    label = "**🤖 ChatBot:**" if key == "output" else f"**{key}:**"
                    st.markdown(label)
                    st.write(val)
                elif key == "fig":
                    if isinstance(val, list):
                        for fig in val:
                            if isinstance(fig, Figure):
                                st.pyplot(fig)
                    elif isinstance(val, Figure):
                        st.pyplot(val)
        else:
            st.write(result)

    output_text = result.get("output", "") if isinstance(result, dict) else str(result)

    if plt.get_fignums():
        st.pyplot(plt.gcf())
