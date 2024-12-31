# Agentic Framework

This project aims to create a tool that will help in building agentic AI based on LangGraph. The framework will include several classes such as `Node` (which will extend to different types of nodes) and a `Graph` constructor.

## Features

- **Node Class**: Base class for different types of nodes.
- **Graph Constructor**: Tool to construct and manage the graph of nodes.

## Requirements

Ensure you have Python 3.6+ installed. You can check your Python version by running:

```bash
python --version
```

## Installation

To install the required packages, run:

```bash
pip install -r requirements.txt
```

## Usage

To use the framework, import the necessary classes and start building your graph:

```python
from agentic_framework import Node, Graph

# Example usage
graph = Graph()
node_a = Node('A')
node_b = Node('B')

graph.add_node(node_a)
graph.add_node(node_b)
graph.connect_nodes(node_a, node_b)
```