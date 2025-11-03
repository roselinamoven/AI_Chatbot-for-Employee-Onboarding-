# 🔥 KIA – AI Employee Onboarding Chatbot

Hi! This is **KIA**, your intelligent onboarding assistant. This AI assistant guides new employees through company policies, answers questions about internal regulations, and provides personalized responses based on employee data. Think of it as a “Pensieve” for your onboarding journey — illuminating knowledge, unlocking secrets, and making onboarding magical! ✨🪄

---

##  Features

- Answer employee questions about company policies, procedures, and regulations.  
- Personalized responses based on employee information.  
- Semantic search over PDF documents using a vector store.  
- Powered by local LLMs via Ollama or HuggingFace models.  
- Streamlit web interface for an interactive, friendly experience.  
- Customizable company branding (name, logo, theme).  

---

##  Tech Stack

- **Python 3.11+**  
- **Streamlit** – Web interface  
- **LangChain** – AI assistant framework  
- **Ollama** – Run LLMs locally  
- **HuggingFace Transformers** – NLP models & embeddings  
- **ChromaDB** – Vector store for semantic search  
- **dotenv** – Environment variable management  

---

##  Quickstart

###  Clone the Repository

```bash
git clone https://github.com/yourusername/AI_Onboarding_Chatbot.git
cd AI_Onboarding_Chatbot


---

###  Create virtual environment
python3 -m venv chatbot_env
source chatbot_env/bin/activate   # macOS/Linux
# OR
chatbot_env\Scripts\activate  

### 1 install dependencies
pip install -r requirements.txt


##Ollama does not require a token if running models locally.

###5 Install & Run Ollama

Install Ollama for your OS: https://ollama.com/docs/installation

Pull a model for the chatbot:

ollama pull llama3
ollama serve


### run the chatbot
streamlit run app.py
