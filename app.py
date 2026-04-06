import streamlit as st
import subprocess
import time
import os
import platform
import uuid
import re
import glob
from concurrent.futures import ThreadPoolExecutor
import pyvista as pv
from stpyvista import stpyvista
from openai import OpenAI
from google import genai

# --- Configuration & Setup ---
st.set_page_config(page_title="The AI CAD Battleground", layout="wide", page_icon="⚙️")

# Initialize Session State
if 'results' not in st.session_state:
    st.session_state.results = None

# Models Definition
MODELS = {
    "Model A": {"id": "llama-3.1-8b-instant", "provider": "groq", "tier": "Small"},
    "Model B": {"id": "openai/gpt-oss-20b", "provider": "groq", "tier": "Medium-Small"},
    "Model C": {"id": "gemma-3-12b-it", "provider": "gemini", "tier": "Small"}
}

# --- System Prompt ---
def get_system_prompt():
    return """You are an expert mechanical CAD engineer. Your task is to write ONLY valid OpenSCAD code to generate a 3D part based on the user's description.

CRITICAL RULES:
1. OUTPUT ONLY CODE: No markdown formatting blocks (do not use ```openscad or ```), no explanations, no conversational filler.
2. INSTANTIATE AT THE END: If you define a module, you MUST call it at the very bottom of the script. If you do not call the module, the "top level object is empty" error occurs and compilation fails.
3. 3D ONLY: You are creating a 3D STL file. Do not mix 2D shapes (circle, square) with 3D shapes (cylinder, cube) in boolean operations like difference(). ALWAYS use linear_extrude() on 2D shapes before 3D booleans.
4. HIGH RESOLUTION: Always include `$fn=100;` at the top of your file for smooth curves.
5. CLEAN BOOLEANS: When using difference() to create holes, make the cutting object slightly larger/longer than the hole depth and translate it properly to avoid Z-fighting (zero-thickness walls).

EXAMPLE STRUCTURE TO FOLLOW:
$fn=100;

module user_requested_part() {
    // ... geometry logic goes here ...
}

// YOU MUST INSTANTIATE THE GEOMETRY AT THE VERY END:
user_requested_part();
"""

# --- API Calling Functions ---
def clean_code(raw_text):
    """Removes markdown code blocks if the LLM ignores instructions."""
    code = raw_text.strip()
    code = re.sub(r"^```[a-zA-Z]*\n", "", code)
    code = re.sub(r"```$", "", code)
    return code.strip()

def call_groq(model_id, prompt, system_prompt):
    # Relies on st.secrets["GROQ_API_KEY"]
    client = OpenAI(api_key=st.secrets.get("GROQ_API_KEY", ""), base_url="https://api.groq.com/openai/v1")
    response = client.chat.completions.create(
        model=model_id,
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": prompt}
        ],
        temperature=0.1
    )
    return clean_code(response.choices[0].message.content)

def call_gemini(model_id, prompt, system_prompt):
    # Relies on st.secrets["GEMINI_API_KEY"]
    client = genai.Client(api_key=st.secrets.get("GEMINI_API_KEY", ""))
    
    # FIX: Prepend system prompt to user prompt to bypass "Developer instruction not enabled" on Gemma models
    combined_prompt = f"{system_prompt}\n\nUser Request: {prompt}"
    
    response = client.models.generate_content(
        model=model_id,
        contents=combined_prompt,
        config=genai.types.GenerateContentConfig(
            temperature=0.1
        )
    )
    return clean_code(response.text)

def generate_code(model_name, model_info, prompt):
    start_time = time.time()
    try:
        sys_prompt = get_system_prompt()
        if model_info["provider"] == "groq":
            code = call_groq(model_info["id"], prompt, sys_prompt)
        else:
            code = call_gemini(model_info["id"], prompt, sys_prompt)
        
        elapsed = time.time() - start_time
        return {"model": model_name, "code": code, "time": elapsed, "error": None}
    except Exception as e:
        return {"model": model_name, "code": None, "time": time.time() - start_time, "error": str(e)}

def cleanup_temp_files():
    """Deletes old temporary STL and SCAD files from previous runs."""
    for file_path in glob.glob("temp_*.scad") + glob.glob("temp_*.stl"):
        try:
            os.remove(file_path)
        except Exception:
            pass

# --- Rendering Functions ---
def render_openscad(code):
    uid = str(uuid.uuid4())
    scad_file = f"temp_{uid}.scad"
    stl_file = f"temp_{uid}.stl"
    
    with open(scad_file, "w") as f:
        f.write(code)
    
    try:
        # Check OS to use the correct OpenSCAD command locally (Windows) vs cloud (Linux)
        openscad_cmd = r"C:\Program Files\OpenSCAD (Nightly)\openscad.exe" if platform.system() == "Windows" else "openscad"
        
        # Calls the openscad CLI tool
        result = subprocess.run([openscad_cmd, "-o", stl_file, scad_file], capture_output=True, text=True, timeout=30)
        if result.returncode != 0:
            raise Exception(f"OpenSCAD Error: {result.stderr}")
        return stl_file
    finally:
        if os.path.exists(scad_file):
            os.remove(scad_file)

# --- Main App UI ---
st.title("🛡️ The AI CAD Battleground")
st.markdown("Evaluating small-tier LLMs on Engineering CAD Generation. Enter a prompt and compare the generated models side-by-side!")

# Inputs (Wrapped in a form to enable Ctrl+Enter submission)
with st.form("generation_form", border=False):
    col_input, col_settings = st.columns([3, 1])
    with col_input:
        user_prompt = st.text_area("Describe the CAD part you want to generate: (PLEASE MAKE IT SIMPLE AND SPECIFIC or these lightweight models may struggle to generate a renderable model)", placeholder="e.g., A hex nut with an M8 internal thread and 15mm outer diameter...")
    with col_settings:
        st.markdown("<br>", unsafe_allow_html=True) # Adds a little spacing so the button aligns nicely with the text box
        generate_btn = st.form_submit_button("🚀 Generate Models", use_container_width=True, type="primary")

# Execution block
if generate_btn and user_prompt:
    
    # Clean up old files before starting a new run
    cleanup_temp_files()
    
    with st.spinner("Dispatching prompts to AI models simultaneously..."):
        # Run API calls concurrently
        with ThreadPoolExecutor(max_workers=3) as executor:
            futures = [
                executor.submit(generate_code, name, info, user_prompt)
                for name, info in MODELS.items()
            ]
            api_results = [f.result() for f in futures]
        
        # Process and Render
        st.session_state.results = {}
        for res in api_results:
            model_name = res["model"]
            if res["error"]:
                st.session_state.results[model_name] = {"status": "error", "message": res["error"]}
                continue
            
            # Attempt to render
            try:
                stl_path = render_openscad(res["code"])
                
                st.session_state.results[model_name] = {
                    "status": "success",
                    "code": res["code"],
                    "time": res["time"],
                    "stl_path": stl_path,
                    "info": MODELS[model_name]
                }
            except Exception as e:
                st.session_state.results[model_name] = {
                    "status": "error", 
                    "message": str(e),
                    "code": res["code"]
                }

# Display Results
if st.session_state.results:
    st.markdown("---")
    cols = st.columns(3)
    
    for idx, (model_name, data) in enumerate(st.session_state.results.items()):
        with cols[idx]:
            # Wrap each result in a modern UI card/container
            with st.container(border=True):
                info = MODELS[model_name]
                st.subheader(f"{model_name}")
                st.caption(f"**Engine:** {info['id']} | **Time:** {data.get('time', 0):.2f}s")
                
                if data["status"] == "error":
                    st.error("Failed to render geometry.")
                    if "message" in data:
                        with st.expander("View Error Log"):
                            st.code(data["message"])
                    if "code" in data and data["code"]:
                        with st.expander("View Generated Code"):
                            st.code(data["code"], language="openscad")
                else:
                    # Render 3D Object
                    stl_path = data["stl_path"]
                    try:
                        plotter = pv.Plotter(window_size=[300, 300])
                        # Changed background to white and model to a sleek blue
                        plotter.background_color = "white"
                        mesh = pv.read(stl_path)
                        
                        # FIX: Removed smooth_shading=True. This forces "flat shading", 
                        # which eliminates the black shadow glitches on sharp CAD corners!
                        plotter.add_mesh(mesh, color='#1f77b4', specular=0.5)
                        
                        plotter.view_isometric()
                        
                        # Show interactive 3D window
                        stpyvista(plotter, key=f"plot_{model_name}_{uuid.uuid4().hex[:8]}")
                    except Exception as e:
                        st.error(f"Render engine error: {e}")

                    # Download & Code buttons
                    with open(stl_path, "rb") as file:
                        st.download_button(
                            label="⬇️ Download STL",
                            data=file,
                            file_name=f"{model_name}_OpenSCAD.stl",
                            mime="application/octet-stream",
                            use_container_width=True,
                            key=f"dl_{model_name}"
                        )
                    
                    with st.expander("🔍 View Source Code"):
                        st.code(data["code"], language="openscad")