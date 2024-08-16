import os
import pickle
import time
import panel as pn

pn.extension()
pn.config.console_output = 'both'

from ragQA import ragQA, prepare_vectorDB
from LLM import load_LLM, load_embeddings, run_llm

######################################################

# Initialize empty list of file input widgets
file_inputs = []

llm_options = ["GPT-3.5 Turbo", "Meta-Llama-3.1-8B"]
llm_dropdown = pn.widgets.Select(name='Choose LLM: ', options=llm_options)
llm = load_LLM(llm_dropdown.value)

embeddings_options = ["OpenAI Embeddings", "Google GenerativeAI Embeddings"]
embeddings_dropdown = pn.widgets.Select(name='Choose embeddings: ', options=embeddings_options)
embeddings = load_embeddings(embeddings_dropdown.value)

vectorstore = None

def add_file_input(event):
    # Add a new FileInput widget with a unique label
    new_index = len(file_inputs) + 1
    new_file_input = pn.widgets.FileInput(accept=".txt,.docx,.pdf,.dat", height=50)
    file_inputs.append(new_file_input)
    
    # Update sidebar with new file input
    update_sidebar()

def process_files(event):
    global vectorstore
    vectorstore = prepare_vectorDB(file_inputs, load_embeddings(embeddings_dropdown.value))
    chat_interface.send("Files processed successfully. Now you can ask questions about this context. ", user="System", respond=False)

def confirm_choices(event):
    # Set the chat interface to use the selected LLM and embeddings
    chat_interface.callback = chatBot_dynamic
    chat_interface.send(
        "Choices of LLM and embeddings are confirmed. You can now either \n1) upload files and ask specific questions about their content, or \n2) send a message to get a reply from the LLM without context. ", 
        user="System", 
        respond=False
    )
    chat_interface.callback_user=f"{llm_dropdown.value} (retrieval mode)" if any(f.value for f in file_inputs) else f"{llm_dropdown.value}"

    global llm
    global embeddings
    llm = load_LLM(llm_dropdown.value)
    embeddings = load_embeddings(embeddings_dropdown.value)

def create_labeled_file_inputs():
    # Create a list of labeled file input widgets
    labeled_file_inputs = []
    for i, file_input in enumerate(file_inputs, start=1):
        label = pn.pane.Markdown(f"**File {i}:**")
        labeled_file_inputs.append(pn.Row(label, file_input))
    return labeled_file_inputs

def update_sidebar():
    # Update the sidebar layout with the current list of labeled file inputs and keep the buttons visible
    sidebar_pane.objects = [
        "## Settings",
        llm_dropdown,
        embeddings_dropdown,
        confirm_button, 
        "\n", "\n", "\n", 
        pn.Row("### Upload files for retrieval:\nAllowed formats: .txt, .docx, .pdf, .dat", add_file_button), 
        *create_labeled_file_inputs(),  # Add labeled file inputs to the sidebar
        process_files_button
    ]

def chatBot_dynamic(question, user, instance):
    # llm = load_LLM(llm_dropdown.value)
    # embeddings = load_embeddings(embeddings_dropdown.value)

    # Check for uploaded files
    if len(file_inputs) > 0:
        response = ragQA(question, user, instance, vectorstore, llm)
    else:
        response = run_llm(question, user, None, llm)

    return response

chat_interface = pn.chat.ChatInterface(
    callback=chatBot_dynamic,
    callback_user=f"{llm_dropdown.value} (retrieval mode)" if any(f.value for f in file_inputs) else f"{llm_dropdown.value}"
)
chat_interface.send(
    "Please first choose an LLM and an embedding. Click **\"Confirm choices\"** to continue.", 
    user="System", 
    respond=False
)

# Buttons for additional functionality
add_file_button = pn.widgets.Button(name="Attach another file", button_type="primary")
add_file_button.on_click(add_file_input)

process_files_button = pn.widgets.Button(name="Process files", button_type="success")
process_files_button.on_click(process_files)

confirm_button = pn.widgets.Button(name="Confirm choices", button_type="primary")
confirm_button.on_click(confirm_choices)

# Initial sidebar setup with only the button
sidebar_pane = pn.Column(
    "## Settings",
    llm_dropdown,
    embeddings_dropdown,
    confirm_button, 
    "\n", "\n", "\n", 
    pn.Row("### Upload files for retrieval:\nAllowed formats: .txt, .docx, .pdf, .dat", add_file_button), 
    process_files_button,
    width=370,  # Fixed width for sidebar
    height=800, 
    sizing_mode="fixed",  # Ensure the sidebar has a fixed width
    margin=(10, 10, 10, 10),  # Add some margin to the sidebar
)

main_content = pn.Column(
    pn.Row("# ResearchMate", "# Chat with me! "), 
    chat_interface,
    sizing_mode="stretch_both",  # Allow the main content to stretch to available space
    margin=(10, 10, 10, 10)  # Add margin to ensure content doesn't touch the edges
)

page = pn.Row(
    sidebar_pane,
    main_content,
    sizing_mode="stretch_both"  # Ensure the row stretches to the size of the window
)
page.servable()
