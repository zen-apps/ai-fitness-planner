# AI Fitness Planner

A production-ready GenAI system for personalized fitness and nutrition planning using LangGraph workflows, real USDA nutrition data, and coordinated AI agents.

## 🚀 Quick Start

**Choose your setup mode:**

### Option 1: Demo Mode (2 minutes) - Perfect for Blog Features
```bash
make setup-demo
```
- ✅ Instant gratification with curated sample data  
- ✅ Covers all major food categories (proteins, carbs, fats, vegetables, fruits)
- ✅ Perfect for LangGraph workflow testing
- ✅ Blog-ready demonstrations

### Option 2: Full Mode (15 minutes) - Production Ready  
```bash  
make setup-full
```
- 🔥 Complete USDA nutrition database (300K+ foods)
- 🔥 Real-world data processing pipeline
- 🔥 Production-ready configuration
- 🔥 Perfect for LangChain showcase

## 🏗️ Architecture

### **Tiered Reproducibility Design**
1. **Demo Mode**: 2-minute setup with curated foods for instant results
2. **Full Mode**: 15-minute complete USDA data pipeline  
3. **Enterprise Ready**: Scalable to hosted API (future enhancement)

### **Core Components**  
- **LangGraph Workflows**: Orchestrated AI agents for meal planning
- **Real USDA Data**: 300K+ foods with enhanced nutrition calculations
- **Semantic Search**: FAISS-powered nutrition matching
- **MongoDB**: Optimized nutrition database with smart indexing
- **FastAPI**: Production-ready API layer

## 📊 Key Features

### **Data Engineering Pipeline**
- Automated USDA Branded Foods download & processing
- Enhanced nutrition calculations (per-100g normalization)  
- Macro breakdown analysis (high-protein/fat/carb classification)
- Optimized MongoDB indexing for sub-second search

### **AI Agent Workflows**
- Parallel agent execution with LangGraph
- Coordinated meal timing with workout planning
- Personalized nutrition recommendations (bulk/cut/maintenance)
- Real-time nutrition search and matching

### **Production Patterns**
- Docker containerization
- Environment-based configuration
- LangSmith observability integration
- Comprehensive error handling & logging

## 🛠️ Development

### Database Management
```bash
make db-stats          # Show database statistics
make test-search       # Test nutrition search functionality  
make clean-db          # Reset database
```

### Service Management
```bash
make up               # Start all services
make logs             # View all service logs
make logs-be          # Backend logs only
make logs-fe          # Frontend logs only
```

### Help
```bash
make help             # Show all available commands
```

## 🎯 Perfect for LangChain Blog Features

### **Immediate Impact Story**
1. **2-minute setup** → Working AI fitness planner
2. **Real data processing** → 300K foods in 15 minutes  
3. **LangGraph workflows** → Coordinated AI agents
4. **Production patterns** → Scalable GenAI architecture

### **Technical Innovation Highlights**
- **Hybrid Search**: Semantic + traditional nutrition matching
- **Agent Coordination**: Parallel meal and workout planning
- **Data Engineering**: Complete USDA pipeline automation
- **Vector Similarity**: FAISS-powered food recommendations

### **Metrics to Showcase**
- 300K+ foods processed with enhanced nutrition calculations
- Sub-second semantic search across complete nutrition database  
- Parallel agent workflows reducing plan generation time
- Real-world scenarios: bulk/cut/maintenance nutrition planning

## 📈 Blog Narrative Arc

1. **Hook**: "Building a production GenAI system in 15 minutes"
2. **Problem**: Complex nutrition planning requires orchestrated AI agents  
3. **Solution**: LangGraph + real USDA data + coordinated workflows
4. **Demo**: `make setup-demo` → immediate results
5. **Scale**: `make setup-full` → production showcase
6. **Results**: Complete personalized fitness planning system

---

## Legacy API Endpoints (Still Available)

- **Download USDA data**: `@nutrition_setup.get("/load_usda_data/")`
- **Import to MongoDB**: `@nutrition_setup.post("/import_usda_data/")`  
- **Search products**: `@nutrition_setup.get("/search_nutrition/")`


 # Environment Setup

## Quick Start

1. **Copy the environment template:**
   ```bash
   cp .env.example .env
   ```

2. **Update the `.env` file with your actual values:**
   - Replace placeholder values with your real credentials
   - Generate secure passwords for database and admin interfaces
   - Add your API keys where needed

## Required Environment Variables

### Database Setup
- `DB_USER`: Your PostgreSQL username
- `DB_PASSWORD`: Secure password for your database
- `DATABASE_URL`: Complete PostgreSQL connection string

### Security Keys
- `SECRET_KEY`: Generate a secure random key for JWT tokens
  ```bash
  # Generate a secure secret key:
  python -c "import secrets; print(secrets.token_urlsafe(32))"
  ```

### Jupyter Password Hash
Generate your Jupyter password hash:
```bash
jupyter server password
# Copy the generated hash to JUPYTER_PASSWORD_HASH
```

### API Keys
- `OPENAI_API_KEY`: Your OpenAI API key (if using AI features)
- `LANGSMITH_API_KEY`: Your LangSmith key (optional, for monitoring)

## Important Security Notes

- **Never commit your `.env` file to git**
- The `.env` file is already in `.gitignore`
- Use strong, unique passwords for all services
- Regularly rotate your API keys and passwords
- For production, use environment-specific values

## Docker Setup

If using Docker, the environment variables will be automatically loaded from your `.env` file when running:

```bash
docker-compose up
```

## Verification

After setting up your `.env` file, verify it works by checking if the application starts without environment variable errors.