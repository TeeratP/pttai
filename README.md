# Agentic Framework

A framework for building agent-based AI systems using LangGraph. This framework provides a flexible and type-safe way to construct graphs of language model agents that can make decisions and process information.

## Features

- **Node System**: 
  - `AgentNode`: Process messages and generate responses using language models
  - `DecisionNode`: Make branching decisions based on language model output
  - Extensible base `Node` class for custom node types

- **Graph Management**: 
  - Automated graph construction from node relationships
  - Support for conditional edges and decision-based routing
  - Type-safe node connections and state management

- **State Management**:
  - Conversation history tracking
  - Message-based state updates
  - Structured state validation

## Requirements

- Python 3.6+
- LangGraph
- LangChain
- Pydantic

## Installation

```bash
pip install -r requirements.txt
```

## Usage

Here's an example of creating a simple agent workflow:

```python
from agentic_framework.nodes import AgentNode, DecisionNode
from agentic_framework.graph import AgenticGraph

# Create nodes
agent = AgentNode(
    name="processor",
    llm=your_llm,  # Your language model instance
    node_prompt="Process the input and provide a detailed response"
)

decision = DecisionNode(
    name="classifier",
    llm=your_llm,
    node_prompt="Classify the response as either 'positive' or 'negative'",
    choices=["positive", "negative"]
)

positive_handler = AgentNode(
    name="positive_handler",
    llm=your_llm,
    node_prompt="Handle positive response"
)

negative_handler = AgentNode(
    name="negative_handler",
    llm=your_llm,
    node_prompt="Handle negative response"
)

# Connect nodes
agent > decision
decision["positive"] > positive_handler
decision["negative"] > negative_handler

# Create and build graph
graph = AgenticGraph(
    start_node=agent,
    end_nodes={positive_handler, negative_handler}
)
```
