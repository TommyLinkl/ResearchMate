import os
import pickle
import time
import panel as pn

pn.extension()
pn.config.console_output = 'both'

from ragQA import ragQA, prepare_vectorDB_files, prepare_vectorDB_URLs
from LLM import load_LLM, load_embeddings, run_llm

######################################################

# Initialize empty list of file input widgets
selected_files = None
URL_inputs = []

llm_options = ["GPT-3.5 Turbo", "Meta-Llama-3.1-8B"]
llm_dropdown = pn.widgets.Select(name='Choose LLM: ', options=llm_options)
llm = load_LLM(llm_dropdown.value)

embeddings_options = ["OpenAI Embeddings", "Google GenerativeAI Embeddings"]
embeddings_dropdown = pn.widgets.Select(name='Choose embeddings: ', options=embeddings_options)
embeddings = load_embeddings(embeddings_dropdown.value)

vectorstore = None

def confirm_choices(event):
    # Set the chat interface to use the selected LLM and embeddings
    chat_interface.callback = chatBot_dynamic
    chat_interface.send(
    pn.pane.Markdown(
        """
        Choices of LLM and embeddings are confirmed. You can now do the following:

        1) **Retrieval-based question answering**  
            - Select local files to upload via the `Attach files` button, or  
            - Add several URLs of the websites via the `Add URL` button,  
            After clicking the `process files` or `process URLs` button, you may ask specific questions about this content.

        2) **Pure LLM inference**  
            - Send a message to get a reply from the LLM without any additional context.
        """),
    user="System",
    respond=False
)
    chat_interface.callback_user=f"{llm_dropdown.value} (retrieval mode)" if any(u.value for u in URL_inputs) or selected_files is not None else f"{llm_dropdown.value}"

    global llm
    global embeddings
    llm = load_LLM(llm_dropdown.value)
    embeddings = load_embeddings(embeddings_dropdown.value)

def select_files(event):
    global selected_files
    file_selector = pn.widgets.FileSelector('./') # , sizing_mode='fixed', width=100)
    selected_files = file_selector  # Store the file selector widget

    # Update sidebar with new file input
    file_container.objects = [
        pn.Row(pn.pane.Markdown("### Upload files for retrieval:\nAllowed formats: .txt, .pdf"), cancel_file_button), 
        selected_files,  # Add file selector widgets
        process_files_button
    ]
    update_sidebar()

def cancel_files(event):
    file_container.objects = [
        pn.Row(pn.pane.Markdown("### Upload files for retrieval:\nAllowed formats: .txt, .pdf"),  add_file_button), 
        process_files_button
    ]
    update_sidebar()

def process_files(event):
    global vectorstore
    global selected_files
    print(selected_files)

    file_container.objects = [
        pn.pane.Markdown("### Upload files for retrieval:\nAllowed formats: .txt, .pdf"), 
        "Processing files...", 
        process_files_button
    ]
    update_sidebar()

    vectorstore = prepare_vectorDB_files(selected_files.value, vectorstore, load_embeddings(embeddings_dropdown.value))
    chat_interface.send(f"{len(selected_files.value)} files processed successfully. Now you can ask questions about them. ", user="System", respond=False)

    file_container.objects = [
        pn.Column(
            pn.pane.Markdown("### Upload files for retrieval:\nAllowed formats: .txt, .pdf"), 
            f"Processing files... {len(selected_files.value)} files processed successfully. "
        ), 
        process_files_button
    ]
    update_sidebar()

def add_URL_input(event):
    global URL_inputs
    new_URL_input = pn.widgets.TextInput(placeholder='http://example.com')
    URL_inputs.append(new_URL_input)

    # Update sidebar with new file input
    url_container.objects = [
        pn.Row("### Input URLs: ", add_URL_button), 
        *create_labeled_URL_inputs(),  # Add URL inputs
        process_URLs_button
    ]
    update_sidebar()

def process_URLs(event):
    global vectorstore

    url_container.objects = [
        pn.Row("### Input URLs: "), 
        *create_labeled_URL_inputs(),  # Add URL inputs
        "Processing URLs...", 
        process_URLs_button
    ]
    update_sidebar()

    vectorstore = prepare_vectorDB_URLs(URL_inputs, vectorstore, load_embeddings(embeddings_dropdown.value))
    chat_interface.send(f"{len(URL_inputs)} URLs processed successfully. Now you can ask questions about them. ", user="System", respond=False)

    url_container.objects = [
        pn.Row("### Input URLs: "), 
        *create_labeled_URL_inputs(),  # Add URL inputs
        f"Processing URLs... {len(URL_inputs)} URLs processed successfully. ", 
        process_URLs_button
    ]
    update_sidebar()

def create_labeled_URL_inputs():
    global URL_inputs
    labeled_URL_inputs = []
    for i, URL_input in enumerate(URL_inputs, start=1):
        labeled_URL_inputs.append(pn.Row(f"**URL {i}:**", URL_input))
    return labeled_URL_inputs

def update_sidebar():
    # Combine all containers into the sidebar pane
    sidebar_pane.objects = [
        "## Settings",
        llm_dropdown,
        embeddings_dropdown,
        confirm_button,
        "\n", "\n", "\n", "\n",
        file_container,
        "\n", "\n", "\n", "\n",
        url_container
    ]

def chatBot_dynamic(question, user, instance):
    # Check for uploaded files
    if any(u.value for u in URL_inputs) or selected_files is not None:
        response = ragQA(question, user, instance, vectorstore, llm)
    else:
        response = run_llm(question, user, None, llm)

    return response

chat_interface = pn.chat.ChatInterface(
    callback=chatBot_dynamic,
    callback_user=f"{llm_dropdown.value} (retrieval mode)" if any(u.value for u in URL_inputs) or selected_files is not None else f"{llm_dropdown.value}"
)
chat_interface.send(
    "Please first choose an LLM and an embedding. Click **\"Confirm choices\"** to continue.", 
    user="System", 
    respond=False
)

# Buttons for additional functionality
confirm_button = pn.widgets.Button(name="Confirm choices", button_type="primary")
confirm_button.on_click(confirm_choices)

add_file_button = pn.widgets.Button(name="Attach files", button_type="primary")
add_file_button.on_click(select_files)

cancel_file_button = pn.widgets.Button(name="Cancel files", button_type="warning")
cancel_file_button.on_click(cancel_files)

process_files_button = pn.widgets.Button(name="Process files", button_type="success")
process_files_button.on_click(process_files)

add_URL_button = pn.widgets.Button(name="Add another URL", button_type="primary")
add_URL_button.on_click(add_URL_input)

process_URLs_button = pn.widgets.Button(name="Process URLs", button_type="success")
process_URLs_button.on_click(process_URLs)

# Create containers for the sidebar elements
file_container = pn.Column(
    pn.Row(pn.pane.Markdown("### Upload files for retrieval:\nAllowed formats: .txt, .pdf"), add_file_button), 
    process_files_button
)

url_container = pn.Column(
    pn.Row("### Input URLs: ", add_URL_button), 
    process_URLs_button
)

# Initial sidebar setup with only the button
sidebar_pane = pn.Column(
    "## Settings",
    llm_dropdown,
    embeddings_dropdown,
    confirm_button, 
    "\n", "\n", "\n", "\n", 
    file_container, 
    "\n", "\n", "\n", "\n", 
    url_container, 
    width=370,  # Fixed width for sidebar
    height=800, 
    sizing_mode="fixed", 
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
