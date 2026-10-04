# 🚀 CourseMate AI

### Enterprise-Grade Multimodal RAG & AI Learning Assistant

Developed by Arjav Jain.

---

## 👋 About the Project

**CourseMate AI** is a full-stack, production-ready Generative AI application that transforms static documents (PDFs, DOC/DOCX) into an **interactive, conversational learning environment**.

Designed to solve the inefficiency of traditional document navigation, CourseMate AI goes beyond standard keyword searches. By leveraging **Retrieval-Augmented Generation (RAG)**, secure user isolation, and **Multi-Modal Vision OCR**, it reads, understands, and accurately answers questions grounded strictly in your academic or professional documents.

This marks the evolution from a prototype into a highly resilient, cloud-deployable Single Page Application (SPA).

---

## 🔥 v2.0 Architectural Advancements 

This project has been completely re-architected for production:

* **Dual-Engine Vision OCR Pipeline:** Natively processes scanned PDFs and images using **Gemini 2.5 Flash** as the primary vision extractor, with automatic failover to **Mistral Pixtral-12B** if rate limits are hit.
* **Secure User Identity & Isolation:** Integrated **Firebase Authentication** (Google Sign-In) with strict backend JWT validation. Every user gets a private, encrypted workspace and dedicated chat histories.
* **Persistent Chat History:** Migrated to **SQLAlchemy (PostgreSQL/SQLite)**. Conversations are saved sequentially, allowing users to return to previous document sessions seamlessly.
* **Ephemeral Data Security:** Uploaded documents and vector blocks are automatically wiped from server disks after 10 minutes of inactivity to ensure strict data privacy and prevent memory bloat.
* **Resilient LLM Failovers:** If the primary LLM (Mistral) experiences capacity limits (429 errors), the backend instantly falls back to Gemini, ensuring zero downtime for the end user.
* **SPA-Grade Frontend:** Completely refactored Vanilla JS interface with glassmorphic UI, responsive mobile sidebars, real-time typing effects, and asynchronous DOM updates (zero page reloads).

---

## ⚙️ Tech Stack

### 🧠 AI / ML Layer
* **LangChain** → Orchestrates the RAG and chunking pipelines
* **ChromaDB** → Local vector database for rapid semantic retrieval
* **HuggingFace** → `all-MiniLM-L6-v2` for dense vector embeddings
* **Mistral AI & Google Gemini** → Dynamic, dual-engine LLM & OCR architecture

### ⚡ Backend Architecture
* **FastAPI** → High-performance, async API routing
* **SQLAlchemy** → ORM for PostgreSQL/SQLite relational mapping
* **Firebase Admin SDK** → Cryptographic JWT validation
* **PyMuPDF (fitz) & python-docx** → Deep document parsing

### 🎨 Frontend UI
* **HTML5, CSS3, Vanilla JavaScript**
* **Firebase Client SDK** → OAuth 2.0 Google Sign-In
* Fully responsive Single Page Application (SPA) design

---

## 🔄 The Multimodal RAG Flow

1. A signed-in user uploads a PDF or DOC/DOCX file.
2. The backend extracts text and OCR content, then stores searchable vector embeddings.
3. Questions are answered from the uploaded document with conversation history and citations.
4. Temporary uploaded files and vector data are cleaned up after the session expires.

## ⚙️ Configuration

Copy `.env.example` to `.env` and provide the required service credentials locally. Never commit `.env`, Firebase service-account JSON files, or API keys.

Required services:

- Firebase Authentication with Google Sign-In enabled.
- A Firebase Admin service-account JSON file for backend token verification.
- Mistral, Google Gemini, and Hugging Face API keys.
- An optional PostgreSQL `DATABASE_URL`; SQLite is used by default.

The Firebase web configuration and deployed backend URL are currently declared in `app.html`. Update those values for your own Firebase project and deployment; they must not contain personal names, contact details, or private keys.

## Deployment

### Backend deployment

The FastAPI backend can be deployed to Render, Railway, Fly.io, or another Python web-service host.

1. Create a web service from this repository.
2. Use `pip install -r requirements.txt` as the build command.
3. Use `uvicorn app:app --host 0.0.0.0 --port $PORT` as the start command.
4. Configure `MISTRAL_API_KEY`, `GOOGLE_API_KEY`, and `HUGGINGFACEHUB_API_TOKEN`.
5. Configure `FIREBASE_CREDENTIALS_JSON` with the contents of a Firebase Admin service-account JSON file. Keep it server-side only.
6. Set `DATABASE_URL` to a persistent PostgreSQL database URL for production. SQLite is suitable only for local development.
7. Confirm the service at `/health` before connecting the frontend.

`render.yaml` contains the equivalent Render service definition.

### GitHub repository and GitHub Pages

GitHub stores the source code, but GitHub Pages cannot run the FastAPI backend. If you use GitHub Pages for the static frontend, copy `config.example.js` to an untracked `config.js` beside `app.html` and fill in the backend URL and Firebase web configuration:

```js
window.COURSEMATE_API_URL = "https://your-backend.example.com";
```

The frontend then uses that backend URL. Add the GitHub Pages domain and your backend domain to Firebase Authentication's authorized domains, and configure backend CORS if you restrict origins. Do not put backend API keys or the Firebase Admin service-account JSON in GitHub Pages.

For the simplest deployment, serve `index.html` and `app.html` from the FastAPI backend itself; same-origin API requests require no `config.js`.

### Firebase setup

Create a Firebase project, enable Google Authentication, register the web app, and copy its public web configuration into `config.js`. Add the deployed frontend domains to Firebase's authorized domains. The web configuration identifies the Firebase project but does not replace the private Admin service-account credentials required by the backend.

## License and attribution

CourseMate AI is developed by Arjav Jain.
