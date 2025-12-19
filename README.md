# 🧠 Pokémon Battle Advisor - AI-Powered Strategy Generator

## 📋 Project Overview

The **Pokémon Battle Advisor** is an intelligent application that combines computer vision and advanced AI reasoning techniques to help users build optimal Pokémon battle strategies. The system identifies Pokémon from uploaded images and generates strategic team compositions and battle plans to counter opponent teams using a multi-agent reasoning pipeline.

### 🎯 Project Goals

- **Pokémon Recognition**: Automatically identify Pokémon from user-uploaded images using a fine-tuned EfficientNetB0 model
- **Strategic Team Building**: Generate optimal team compositions based on type advantages, abilities, and matchups
- **Battle Strategy Generation**: Create detailed battle plans using advanced AI reasoning techniques
- **User-Friendly Interface**: Provide an intuitive Streamlit-based web application for seamless interaction

---

## 👥 Project Members

- Rayan Grégoire
- Corentin Gaudé
- Ikram Amine
- Alexis Boulic

---

## 🏗️ Project Structure

```
GenAI-Project-Pokemon/
│
├── README.md                      
├── app.py                         # Main Streamlit application
├── requirements.txt               # Python dependencies
├── .env                           # Environment variables (API key)
│
├── data/                          # Data storage
│   ├── pokedex/                   # Pokémon JSON data files
│   ├── pokemon-dataset-1000/      # Pokémon images dataset
│   ├── dataset_embedding/         # Pre-computed image embeddings
│   └── chroma_pokedex/            # Vector database for RAG
│
├── models/                        # Machine learning model
│   └── finetuned_efficientnetb0_pour_pokemon.h5
│
├── Embedding/                     # Image embedding & similarity
│   ├── fonction_embedding_image_solo.py  # Generate embeddings
│   ├── loading_model.py                  # Load fine-tuned model
│   └── recherche_similarity.py           # Pokémon prediction
│
├── Rag_Agent/                     # Multi-agent reasoning system
│   ├── llm.py                     # LLM configuration
│   ├── retrieval.py               # Vector database retrieval
│   ├── pipeline.py                # Agent orchestration
│   ├── prompts.py                 # Agent prompts
│   ├── agent_treeofthoughts.py    # Tree-of-Thoughts agent
│   ├── agent_react.py             # ReAct agent
│   ├── agent_selfcorrection.py    # Self-correction agent
│   ├── json_retriever.py          # Pokémon data loader
│   └── utils.py                   # Utility functions
│
└── Jupyter_files/                 # Notebooks for experimentation
    ├── build_chroma.ipynb
    ├── creation_pokedex.ipynb
    ├── embedding_dataset_images_pokemon.ipynb
    └── finetunning.ipynb
```

### 📁 Folder Descriptions

- **`data/`**: Contains Pokémon datasets, embeddings, and vector database
- **`models/`**: Stores the fine-tuned EfficientNetB0 model for Pokémon recognition
- **`Embedding/`**: Handles image embedding generation and similarity matching
- **`Rag_Agent/`**: Multi-agent system for strategic reasoning and decision-making
- **`Jupyter_files/`**: Research and experimentation notebooks

---

## 🧩 Reasoning Technique: Multi-Agent Pipeline

Our application uses a sophisticated **three-stage reasoning pipeline** that combines multiple AI agent architectures to generate only optimal battle strategies:

### 1️⃣ Tree-of-Thoughts (ToT) Agent
**Purpose**: Initial team composition and strategic exploration

- Explores multiple reasoning paths simultaneously
- Evaluates different team compositions in parallel
- Considers type matchups, abilities, and synergies
- Selects the most promising strategy based on depth-first exploration
- **Output**: Initial team composition and high-level strategy

### 2️⃣ ReAct (Reasoning + Acting) Agent
**Purpose**: Tactical refinement and step-by-step planning

- Generates detailed turn-by-turn battle plans
- Interleaves reasoning with action planning
- Considers move selection, switching strategies, and predictions
- Uses retrieval-augmented generation (RAG) for move data and matchups
- **Output**: Detailed tactical strategy with reasoning traces

### 3️⃣ Self-Correction Agent
**Purpose**: Strategy validation and refinement

- Reviews the ReAct-generated strategy for weaknesses
- Identifies potential counter-strategies from opponents
- Suggests improvements and contingency plans
- Ensures robustness against edge cases
- **Output**: Final validated and improved strategy

### 🔄 Pipeline Flow

```
User Input → Pokémon Recognition → ToT Agent → ReAct Agent → Self-Correction Agent → Final Strategy
```

This curated multi-stage approach ensures:
- **Comprehensive exploration** (ToT)
- **Detailed planning** (ReAct)
- **Robust validation** (Self-Correction)

---

## 🚀 How to Run the Project

### Prerequisites

- Python 3.10 or higher
- Virtual environment (recommended)
- Google Gemini API key

### Installation Steps

1. **Clone the repository**
   ```bash
   git clone https://github.com/yourusername/GenAI-Project-Pokemon.git
   cd GenAI-Project-Pokemon
   ```

2. **Create and activate a virtual environment**
   ```bash
   # Windows
   python -m venv venv
   .\venv\Scripts\activate

   # macOS/Linux
   python3 -m venv venv
   source venv/bin/activate
   ```

3. **Install dependencies**
   ```bash
   pip install -r requirements.txt
   ```

4. **Set up environment variables**
   
   Create a `.env` file in the project root:
   ```
   GEMINI_API_KEY=your_gemini_api_key_here
   ```

   Get your Gemini API key from: https://makersuite.google.com/app/apikey

5. **Run the application**
   ```bash
   streamlit run app.py
   ```

6. **Access the application**
   
   Open your browser and navigate to: `http://localhost:8501`

---

## 💡 Key Features

### 🔍 Pokémon Recognition
- **Fine-tuned EfficientNetB0** model for accurate identification
- **Top-3 similarity matching** with confidence scores
- **Image visualization** of similar Pokémon
- Supports common image formats (JPG, PNG, JPEG)

### 🎯 Strategic Team Building
- **Type advantage analysis** using comprehensive Pokémon data
- **Ability and moveset consideration**
- **Synergy detection** between team members
- **Counter-team generation** based on opponent composition

### 🧠 AI-Powered Strategy
- **Multi-agent reasoning** for comprehensive analysis
- **Retrieval-Augmented Generation (RAG)** for accurate Pokémon data
- **Step-by-step battle plans** with reasoning transparency
- **Self-validation** to ensure strategy robustness

### 🖼️ Interactive UI
- **Real-time Pokémon image display** for uploaded, enemy, and recommended teams
- **Visual team composition** with images and names
- **Confidence scores** as percentages (0-100%)
- **Reasoning trace viewer** for advanced users

---

## 🛠️ Tech Stack

### Machine Learning & AI
- **TensorFlow/Keras**: Fine-tuned EfficientNetB0 for image recognition
- **Sentence Transformers**: Text embeddings for RAG
- **Scikit-learn**: Cosine similarity for image matching
- **NumPy**: Numerical computations

### LLM & Reasoning
- **LangChain**: Agent orchestration and RAG pipeline
- **Google Gemini 2.5 Flash Lite**: Fast, efficient language model
- **ChromaDB**: Vector database for semantic search

### Web Application
- **Streamlit**: Interactive web interface
- **Python-dotenv**: Environment variable management

---

## 📊 Performance Optimizations

- **Model caching** with `@st.cache_resource` for instant load times
- **Data caching** with `@st.cache_data` for faster UI interactions
- **Reduced retrieval count** (6 documents instead of 12) for faster agent processing
- **Optimized LLM settings** with 1500 token limit and 30s timeout
- **Percentage-based confidence scores** for better interpretability

---

## 🔮 Future Improvements

- Support for additional Pokémon generations
- Multiplayer battle simulation
- Historical battle analysis and learning
- Mobile-responsive design
