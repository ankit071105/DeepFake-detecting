# app.py
import streamlit as st
import requests
import time
import os

# Set page configuration
st.set_page_config(
    page_title="DeepFake Detection App",
    page_icon="🔍",
    layout="wide"
)

# Custom CSS for styling
st.markdown("""
    <style>
    .main-header {
        font-size: 3rem;
        color: #1f77b4;
        text-align: center;
        margin-bottom: 2rem;
    }
    .sub-header {
        font-size: 1.5rem;
        color: #2c3e50;
        margin-bottom: 1rem;
    }
    .result-box {
        padding: 20px;
        border-radius: 10px;
        margin: 20px 0;
        box-shadow: 0 4px 6px rgba(0, 0, 0, 0.1);
    }
    .fake-result {
        background-color: #ffebee;
        border-left: 5px solid #f44336;
    }
    .real-result {
        background-color: #e8f5e9;
        border-left: 5px solid #4caf50;
    }
    .confidence-meter {
        height: 20px;
        background-color: #e0e0e0;
        border-radius: 10px;
        margin: 10px 0;
    }
    .confidence-fill {
        height: 100%;
        border-radius: 10px;
        transition: width 0.5s ease-in-out;
    }
    .uploaded-video {
        max-width: 100%;
        border-radius: 10px;
        margin: 20px 0;
    }
    .stProgress > div > div > div > div {
        background-color: #1f77b4;
    }
    </style>
""", unsafe_allow_html=True)

# App title and description
st.markdown('<h1 class="main-header">🔍 DeepFake Detection</h1>', unsafe_allow_html=True)
st.markdown("""
    <p style='text-align: center; font-size: 1.2rem;'>
    Upload a video to analyze whether it's authentic or AI-generated using our advanced Deep Learning models.
    </p>
""", unsafe_allow_html=True)

# Initialize session state
if 'result' not in st.session_state:
    st.session_state.result = None
if 'uploaded_file' not in st.session_state:
    st.session_state.uploaded_file = None
if 'processing' not in st.session_state:
    st.session_state.processing = False

# File upload section
st.markdown('<h2 class="sub-header">Upload a Video for Analysis</h2>', unsafe_allow_html=True)
uploaded_file = st.file_uploader(
    "Choose a video file", 
    type=['mp4', 'mov', 'avi', 'mkv'],
    help="Supported formats: MP4, MOV, AVI, MKV"
)

# Display uploaded video
if uploaded_file is not None:
    st.session_state.uploaded_file = uploaded_file
    st.video(uploaded_file)
    st.success(f"Video uploaded successfully: {uploaded_file.name}")

# Analysis button
if st.button("Analyze Video", type="primary", disabled=st.session_state.uploaded_file is None) and not st.session_state.processing:
    st.session_state.processing = True
    st.session_state.result = None
    
    # Show processing indicator
    progress_bar = st.progress(0)
    status_text = st.empty()
    
    for percent in range(100):
        # Simulate processing steps
        if percent < 30:
            status_text.text("Uploading video to server...")
        elif percent < 60:
            status_text.text("Extracting frames...")
        elif percent < 90:
            status_text.text("Analyzing with AI model...")
        else:
            status_text.text("Finalizing results...")
        
        progress_bar.progress(percent + 1)
        time.sleep(0.05)
    
    try:
        # Send request to FastAPI backend
        files = {"file": (st.session_state.uploaded_file.name, 
                         st.session_state.uploaded_file.getvalue(), 
                         st.session_state.uploaded_file.type)}
        
        response = requests.post("http://localhost:8000/predict", files=files)
        
        if response.status_code == 200:
            st.session_state.result = response.json()
            status_text.text("Analysis complete!")
            progress_bar.progress(100)
        else:
            st.error(f"Error: {response.json().get('detail', 'Unknown error')}")
            
    except requests.exceptions.ConnectionError:
        st.error("Could not connect to the analysis server. Please make sure the FastAPI backend is running.")
    except Exception as e:
        st.error(f"An error occurred: {str(e)}")
        
    st.session_state.processing = False
    time.sleep(0.5)
    progress_bar.empty()
    status_text.empty()

# Display results
if st.session_state.result:
    result = st.session_state.result
    fake_confidence = result['fake_confidence']
    real_confidence = result['real_confidence']
    is_fake = result['prediction'] == 'FAKE'
    
    # Result box
    result_class = "fake-result" if is_fake else "real-result"
    st.markdown(f'<div class="result-box {result_class}">', unsafe_allow_html=True)
    
    st.markdown(f"### Prediction: **{result['prediction']}**")
    
    # Show note if using fallback
    if 'note' in result.get('details', {}):
        st.warning(result['details']['note'])
    
    # Confidence meters
    st.markdown(f"#### Confidence Levels:")
    
    st.markdown("**Real**")
    st.markdown(f'<div class="confidence-meter"><div class="confidence-fill" style="width: {real_confidence*100}%; background-color: #4caf50;"></div></div>', unsafe_allow_html=True)
    st.markdown(f"{real_confidence*100:.2f}%")
    
    st.markdown("**Fake**")
    st.markdown(f'<div class="confidence-meter"><div class="confidence-fill" style="width: {fake_confidence*100}%; background-color: #f44336;"></div></div>', unsafe_allow_html=True)
    st.markdown(f"{fake_confidence*100:.2f}%")
    
    st.markdown('</div>', unsafe_allow_html=True)
    
    # Additional details
    with st.expander("View Analysis Details"):
        st.json(result)

# Information section
st.markdown("---")
st.markdown("""
    ### How It Works
    This application uses advanced Deep Learning techniques to detect AI-generated (DeepFake) videos:
    
    1. **Frame Extraction**: The video is broken down into individual frames
    2. **Feature Analysis**: Each frame is analyzed for tell-tale signs of manipulation
    3. **Temporal Analysis**: The relationship between frames is examined for inconsistencies
    4. **Ensemble Prediction**: Multiple models contribute to the final prediction
    
    ### Technology Stack
    - **Backend**: FastAPI with PyTorch deep learning models
    - **Frontend**: Streamlit for user-friendly interface
    - **Computer Vision**: Custom frame extraction without OpenCV
    - **Apple Silicon Support**: Optimized for M1/M2 chips
    
    ### Model Architecture
    Our system uses a 3D CNN model:
    - **3D Convolutions**: Extract spatiotemporal features from video frames
    - **Training**: Model was trained on synthetic data (replace with real dataset for production)
    
    ### Note
    This is a demonstration application. In a production environment, you would:
    - Use a real dataset of DeepFake videos
    - Implement more sophisticated architectures
    - Add continuous learning capabilities
""")

# Footer
st.markdown("---")
st.markdown(
    "<div style='text-align: center; color: #777;'><i>DeepFake Detection System • Powered by PyTorch • Apple Silicon Optimized</i></div>",
    unsafe_allow_html=True
)