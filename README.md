# 🧳 AI Travel Planner

An **intelligent travel booking assistant** powered by **LangChain**, **LangGraph**, **LangSmith**, and **Streamlit**. This project demonstrates how modern LLM frameworks can orchestrate multi-step reasoning with multiple tools (flights, hotels, weather, budgets, and search) to build an end-to-end conversational booking system.

---

## 🚀 Features

* **Conversational Travel Planning**: Chat with an AI agent to plan your trip interactively.
* **Multi-Agent Orchestration (LangGraph)**: Tools for flights, hotels, weather, budgets, and search are composed in a graph-based workflow.
* **External API Integrations**:

  * ✈️ Flight search (custom API tool)
  * 🏨 Hotel search (custom API tool)
  * 🌤️ Weather lookup (OpenWeatherMap)
  * 🔎 Web search (SerpAPI)
* **Budget Calculation**: Get estimated trip costs based on flight + hotel + preferences.
* **Conversation Memory**: Maintains chat history using `ConversationBufferMemory`.
* **LangChain Expression Language (LCEL)**: Concise orchestration of LLM chains.
* **LangSmith Tracing**: End-to-end observability for debugging and evaluation.
* **Streamlit UI**: Clean, interactive web interface for chatting with the travel planner.

---

## 📂 Project Structure

```
├── app.py        # Streamlit frontend (chat interface)
├── main.py       # Core LangGraph + LLM orchestration
├── requirements.txt
├── .env.example  # Example environment variables
└── README.md
```

---

## ⚙️ Tech Stack

* **LLM Framework**: [LangChain](https://www.langchain.com/) + [LangGraph](https://github.com/langchain-ai/langgraph)
* **Experiment Tracking**: [LangSmith](https://smith.langchain.com/)
* **LLM Provider**: OpenAI (via `langchain-openai`)
* **Memory**: Conversation Buffer Memory
* **Orchestration**: LangChain Expression Language (LCEL)
* **Frontend**: [Streamlit](https://streamlit.io/)
* **APIs**:

  * SerpAPI (web search)
  * OpenWeatherMap (weather)
  * Custom flight & hotel APIs (stubbed for demo / extendable)

---

## 🏗️ Setup

### 1️⃣ Clone the Repository

```bash
git clone https://github.com/abdullahhunjra/ai-travel-assistant-multiagent-orchestration-langchain-langgraph-lcel.git
cd ai-travel-planner
```

### 3️⃣ Install Dependencies

```bash
pip install -r requirements.txt
```

### 4️⃣ Configure Environment Variables

Create a `.env` file (based on `.env.example`) and add your keys:

```bash
OPENAI_API_KEY=your_openai_key
SERPAPI_API_KEY=your_serpapi_key
OPENWEATHERMAP_API_KEY=your_openweathermap_key
LANGSMITH_API_KEY=your_langsmith_key
```

### 5️⃣ Run the App

```bash
streamlit run app.py
```


---

## 💡 Usage

1. Open the Streamlit interface.
2. Start chatting with the AI:

   * "Find me flights from Paris to Tokyo in December."
   * "Book me a hotel near Shibuya under \$150 per night."
   * "What’s the weather in Tokyo during December?"
   * "What’s my estimated budget for 5 days in Tokyo?"
3. The assistant uses **multi-step reasoning** to:

   * Normalize dates & locations.
   * Query APIs for flights/hotels/weather.
   * Aggregate responses into a trip plan.
   * Maintain chat context across multiple queries.

---

## 🧩 Architecture

```
[Streamlit UI] ⇄ [LangGraph Orchestration]
       │                   │
       │                   ├── Flight Search Tool
       │                   ├── Hotel Search Tool
       │                   ├── Weather API (OpenWeatherMap)
       │                   ├── Web Search (SerpAPI)
       │                   └── Budget Calculator
       │
   [Conversation Memory]
       │
   [LangSmith Tracing]
```

---

## 📊 Example Queries

* "Find me a round trip from NYC to London for June 2025."
* "What hotels under \$200 per night are available in London near Hyde Park?"
* "How much should I budget for flights + hotels for 7 days in London?"
* "What’s the weather forecast in London next June?"

---

## 🔮 Future Improvements

* Add **real booking APIs** (e.g., Skyscanner, Amadeus).
* Extend hotel search with filtering (stars, amenities, neighborhoods).
* Add personalized recommendations (based on past queries/preferences).
* Deploy as a hosted app (Streamlit Cloud / Docker).

---

## 👨‍💻 Author

**Abdullah Hanjra**  
📧 Email: [abdullahshahzadhunjra@gmail.com](mailto:abdullahshahzadhunjra@gmail.com)  
🔗 GitHub: [github.com/abdullahhunjra](https://github.com/abdullahhunjra)  
🔗 LinkedIn: [linkedin.com/in/abdullahhunjra](https://linkedin.com/in/abdullahhunjra)


---

## 📝 License

This project is licensed under the MIT License.
