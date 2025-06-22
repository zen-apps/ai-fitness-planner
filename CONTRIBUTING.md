# Contributing to AI Fitness Planner

Thank you for your interest in contributing! This project demonstrates production-ready GenAI patterns with LangGraph workflows.

## 🚀 Quick Setup

### Prerequisites
- Docker and Docker Compose
- Git
- OpenAI API key (for AI features)

### Get Started
```bash
# Clone and setup
git clone https://github.com/your-username/ai-fitness-planner.git
cd ai-fitness-planner
cp .env.example .env

# Add your OpenAI API key to .env
# Start the application
make setup-demo
```

## 🎯 Ways to Contribute

### **New AI Agents** (Most Valuable)
Add specialized agents to enhance the workflow:

```python
@traceable(name="your_new_agent")
def your_new_agent(state: FitnessWorkflowState):
    """Your agent description"""
    # Your logic here
    return {"new_data": result}
```

**Ideas:**
- **Supplement Advisor**: Evidence-based recommendations
- **Progress Tracker**: Monitor results and adjust plans
- **Recovery Optimizer**: Sleep and rest day planning

### **Performance Improvements**
- Optimize meal planning (currently 83% of execution time)
- Improve vector search caching
- Database query optimization

### **Documentation & Examples**
- Tutorial notebooks
- New use case examples
- Performance optimization guides

## 📝 Contribution Process

### Simple Changes
For small fixes, documentation, or examples:
1. Fork the repository
2. Make your changes
3. Submit a pull request

### New Features
For new agents or major changes:
1. Open an issue first to discuss
2. Fork and create a feature branch
3. Add basic tests if adding code
4. Update documentation
5. Submit a pull request

## 🧪 Testing

```bash
# Run tests (optional but appreciated)
make test

# Check your code works
make up
# Test your changes at http://localhost:8526
```

## 📋 Code Standards

### Simple Guidelines
- Use `@traceable` decorator for new agents
- Include docstrings for new functions
- Follow existing code patterns
- Test your changes manually

### Commit Messages
```bash
feat: add supplement advisor agent
fix: resolve meal planning timeout
docs: add custom agent example
```

## 🤝 Getting Help

- **Questions**: Open a GitHub issue
- **Ideas**: Start a GitHub discussion
- **Bugs**: Create an issue with steps to reproduce

## 📊 Current Performance (From LangSmith)
- **0% error rate** across 66 runs
- **~3 minutes** average plan generation
- **$0.07** average cost per plan
- **Meal planning bottleneck**: 83% of execution time

Contributions that improve these metrics are especially welcome!

## 📄 License

By contributing, you agree your contributions will be licensed under the MIT License.

---

**Thanks for helping make this project better!** 🎉

*This project serves as both a working fitness planner and an educational example of LangGraph patterns.*