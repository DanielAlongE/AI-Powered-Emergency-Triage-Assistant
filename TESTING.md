# Emergency Triage Agent Testing Framework

This document describes the testing framework for evaluating different emergency triage agents against ESI (Emergency Severity Index) test cases.

## Overview

The testing framework provides a modular, extensible system for:
- Implementing different triage agents (Random, Rule-based, LLM-based, Hybrid)
- Running systematic evaluations against test data
- Visualizing results with confusion matrices and performance charts
- Comparing agent performance across multiple metrics
- Analyzing error patterns and confidence calibration

## Architecture

### Core Components

```
server/
├── src/                       
│   ├── agents/                # Triage agent implementations
│   │   ├── base.py           # Abstract base class
│   │   └── implementations/
│   │       ├── random_agent.py    # Baseline random agent
│   │       ├── rule_based_agent.py # Keyword/rule-based agent
│   │       └── llm_agent.py       # LLM and hybrid agents
│   ├── models/
│   │   └── esi_assessment.py  # Data models for ESI assessment
│   ├── app/                   # FastAPI server code
│   └── playground/            # Development utilities
├── testing/                   # Testing framework (separate)
│   ├── runner.py              # Main test runner
│   └── utils/
│       ├── data_loader.py     # Test data loading
│       ├── metrics.py         # Performance metrics calculation
│       └── visualization.py   # Result visualization
```

## Available Agents

### 1. RandomTriageAgent
**Purpose**: Baseline for comparison
- Randomly assigns ESI levels (1-5)
- Configurable weights for different ESI level probabilities
- Reproducible results with seed parameter

```python
agent = RandomTriageAgent("Random", {
    'seed': 42,
    'weights': [0.1, 0.2, 0.4, 0.2, 0.1]  # Favor middle ESI levels
})
```

### 2. RuleBasedTriageAgent
**Purpose**: Keyword-based clinical rules
- Uses predefined keywords for each ESI level
- Pattern matching with confidence scoring
- Customizable keyword dictionaries
- Fast execution suitable for real-time use

```python
agent = RuleBasedTriageAgent("Rule-Based")
agent.add_custom_keywords(1, 'emergency', ['cardiac arrest', 'stroke'])
```

### 3. LLMTriageAgent
**Purpose**: AI-powered assessment using LLama
- Uses Ollama/LLama 3.2 for conversation analysis
- Structured JSON output with rationale
- Configurable temperature and model parameters
- Requires Ollama service running locally

```python
agent = LLMTriageAgent("LLM", {
    'model': 'llama3.2',
    'temperature': 0.3
})
```

### 4. HybridTriageAgent
**Purpose**: Combines rule-based and LLM approaches
- Weighted combination of rule and LLM assessments
- Takes most urgent assessment for critical cases (ESI 1-2)
- Configurable weights for different components
- Fallback to rule-based if LLM fails

```python
agent = HybridTriageAgent("Hybrid", {
    'rule_weight': 0.3,
    'llm_weight': 0.7
})
```

## Test Data Format

The framework uses ESI test cases in JSON format:

```json
[
  {
    "type": "Practice",
    "number": 1,
    "expected": {
      "esi_level": 2,
      "esi_rationale": "High-risk chest pain requiring urgent evaluation"
    },
    "conversation": [
      {
        "speaker": "nurse",
        "message": "What brings you in today?"
      },
      {
        "speaker": "patient",
        "message": "I'm having severe chest pain"
      }
    ]
  }
]
```

## Quick Start

### 1. Setup Dependencies

```bash
cd server
poetry install
```

### 2. Launch Dashboard

```bash
# while in server directory
poetry run dashboard
# Access at http://localhost:8501
```

## Modal GPU Setup (Optional - For Fast Inference)

### 1. Install & Setup Modal

```bash
# Install Modal (already included in dependencies)
poetry run modal setup  # Follow prompts to authenticate
```

### 2. Deploy Ollama Service to GPU

```bash
# From server/ directory
poetry run modal deploy modal_ollama_service.py
```

**Copy the endpoint URL** from deployment output (looks like: `https://your-app--ollama-serve.modal.run`)

### 3. Configure Dashboard

1. Select **"Handbook RAG + Ollama Agent"**
2. Set **Inference Location** to **"Auto"** or **"Modal"**
3. Paste your **Modal endpoint URL**
4. Choose **T4 GPU** for cost-effectiveness

**That's it!** The dashboard handles everything else automatically.

## Usage Examples

### Dashboard Visualization

Use the web dashboard for interactive analysis:

```bash
cd server
poetry run dashboard
```

**Features:**
- Agent testing
- Performance charts
- Confusion matrices
- Results comparison
- Export capabilities

### Custom Agent Development

```python
from agents.base import BaseTriageAgent
from models.esi_assessment import ESIAssessment

class MyCustomAgent(BaseTriageAgent):
    def triage(self, conversation):
        # Your custom logic here
        esi_level = analyze_conversation(conversation)

        return ESIAssessment(
            esi_level=esi_level,
            confidence=0.8,
            rationale="Custom analysis result",
            agent_name=self.name
        )

# Test your agent
custom_agent = MyCustomAgent("My Agent")
results = runner.test_agent(custom_agent, limit=50)
```

## Evaluation Metrics

### Primary Metrics
- **Accuracy**: Percentage of correct ESI level predictions
- **Per-ESI Precision/Recall/F1**: Performance for each ESI level
- **Confusion Matrix**: Detailed prediction vs actual breakdown
- **Processing Time**: Average time per assessment

## Configuration Options

### Test Runner Configuration
```python
runner = TriageTestRunner(
    test_data_path="path/to/test_data.json",
    max_workers=4  # Parallel processing
)
```

### Agent Configuration Examples

```python
# Random agent with custom distribution
RandomTriageAgent("Random", {
    'seed': 42,
    'weights': [0.05, 0.15, 0.6, 0.15, 0.05]  # Emphasize ESI 3
})

# Rule-based with custom keywords
RuleBasedTriageAgent("Custom Rules", {
    'keywords': {
        1: {'critical': ['cardiac arrest', 'not breathing']},
        2: {'urgent': ['chest pain', 'stroke']}
    }
})

# LLM with specific model
LLMTriageAgent("LLM", {
    'model': 'llama3.2',
    'temperature': 0.1  # More deterministic
})

# Hybrid with custom weights
HybridTriageAgent("Hybrid", {
    'rule_weight': 0.4,
    'llm_weight': 0.6
})
```

## Best Practices

### Agent Development
1. **Inherit from BaseTriageAgent** for consistency
2. **Include confidence scoring** for reliability assessment
3. **Provide meaningful rationale** for debugging and validation
4. **Handle edge cases gracefully** with safe defaults
5. **Test thoroughly** with unit tests and real data

### Testing Strategy
1. **Start with baseline** Random agent for comparison
2. **Use consistent test data** for fair evaluation
3. **Test with limited cases first** for rapid iteration
4. **Analyze confusion matrices** to identify weaknesses
5. **Consider processing time** for real-world deployment

### Performance Optimization
1. **Cache preprocessing** for repeated evaluations
2. **Use parallel processing** for large test sets
3. **Implement early stopping** for critical cases
4. **Monitor resource usage** especially for LLM agents
5. **Profile bottlenecks** in custom implementations

## Troubleshooting

### Common Issues

**"Blank testing dashboard page"**
- First time might take a while to load, wait a little
- Confirm no errors were shown on the terminal
- Confirm the loading icon in the top right corner is shown

**"Test data file not found"**
- Ensure `tests/data/esi_test_cases.json` exists
- Check file permissions and path
- Verify JSON format is valid

**"Ollama connection failed"**
- Start Ollama service: `ollama serve`
- Pull required model: `ollama pull llama3.2`
- Check network connectivity and ports

**"Slow performance"**
- Reduce test case limit for development
- Use fewer workers for memory constraints
- Consider simpler agents for real-time use
- Profile code for bottlenecks

### Debug Mode

Enable detailed logging:

```python
import logging
logging.basicConfig(level=logging.DEBUG)

runner = TriageTestRunner()
runner.load_test_data()
# Detailed output for debugging
```

## Contributing

### Adding New Agents
1. Create new file in `agents/implementations/`
2. Inherit from `BaseTriageAgent`
3. Implement `triage()` method
4. Update imports in `__init__.py` files
5. Update the testing dashboard to support the new agent