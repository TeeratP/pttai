# The compile-time validator

Every node in a nae graph declares the state keys it reads and writes. That
gives `AgenticGraph` enough information to run a static **dataflow analysis**
over the wired nodes at construction, *before compiling* — so a node that reads
a key nothing produces, or a decision branch left unwired, fails the build at
construction time, not three LLM calls into a run.

```python
graph = AgenticGraph(start_node=a, end_nodes={z})   # validate() runs HERE
```

If validation finds an error, the constructor raises `GraphValidationError`; if
it finds only warnings, the graph builds but you can inspect them with
[`summary()`](api/graph/agenticgraph.md). Pass `validate=False` to skip the
check (escape hatch), or call `graph.validate(strict=True)` to promote warnings
to errors.

!!! note
    The snippets on this page assume `llm = ...` is any LangChain chat model.

## Why this matters for LLM pipelines

Agent graphs fail differently from ordinary programs: a broken edge or a
misspelled state key doesn't crash immediately — it surfaces *after* the model
has already run, once execution reaches the broken node. By then you've spent
tokens, latency, and (with tools) real side effects on a run that was doomed at
build time.

The validator turns those runtime `KeyError`s and `InvalidUpdateError`s into a
**build-time exception with zero token cost**. It is a dataflow analysis over
your *declared* node reads/writes and the graph edges — not a soundness proof.
It catches read-before-write (including loop-carried reads in cyclic graphs),
undeclared reads/writes, and the structural bugs below. It does **not** see
state keys touched only inside opaque callables — `ConditionNode` predicates
(arbitrary lambdas), tool bodies, or other custom functions — so those are
checked by declared annotations only, not by reading their code. "Compiles" is
therefore a strong signal, not a guarantee, that the dataflow is sound.

## What it checks

In plain terms, the validator walks the graph from the start, tracking which
state keys can exist at each node: keys that *might* be there (produced on at
least one incoming path) and keys that are *certain* to be there (produced on
every incoming path). A read of a key that might be missing is worth a warning;
a read of a key that cannot be there yet is an error.

Formally, the core is a **forward dataflow fixpoint** over a tagged edge list
recorded during the build. It computes, per node:

- `may` — keys available on *some* path into the node (drives hard **errors**).
  Loop-back edges are excluded, so a key produced only after the loop cycles
  back is not counted as available.
- `must` — keys guaranteed on *all* paths (drives **warnings**).

`may`/`must` union across AND-parallel joins (`fanout`) and intersect within an
exclusive `DecisionNode` choice-group, so the analysis understands your
branching. On top of that dataflow it runs a handful of structural checks.

Each bug class below shows a minimal snippet and the **verbatim** message nae
raises. Classes 1–4 and 7 are validator findings (`GraphValidationError`);
classes 5 and 6 are caught earlier, while the graph is being wired, as plain
**`ValueError`s** — you get them even with `validate=False`.

### 1. Read before write (the headline check)

A node reads a state key that no upstream node has produced yet — the producer
runs downstream or on a sibling branch. This is the classic ordering bug.

```python
from nae import AgentNode, AgenticGraph

# `writer` produces `summary`; `reader` needs it — but reader runs FIRST.
reader = AgentNode(name="reader", llm=llm, reads=["summary"],
                   node_prompt="Use this: {summary}")
writer = AgentNode(name="writer", llm=llm, writes={"summary": str})

reader > writer
AgenticGraph(start_node=reader, end_nodes={writer})
```

```
nae.validation.GraphValidationError: AgenticGraph 'graph': 1 error(s), 0 warning(s)
  [error] reader: reads computed key 'summary' but no upstream node produces it before this node (produced by: ['writer'], none of which are upstream); available keys here: ['log', 'messages', 'token']
```

If the key is declared in the schema but **no node writes it at all** (and it
isn't an input), the message instead points you to `inputs=`:

```
  [error] reader: reads 'summary' but no node produces it and it is not an input key; produce it upstream or declare it in inputs=... if you supply it at invoke(); available keys here: ['log', 'messages', 'token']
```

### 2. Undeclared key (typo)

A node reads or writes a key the state schema doesn't declare. LangGraph would
**silently** read nothing / drop the write; nae catches it.

Read side:

```
  [error] reader: reads 'sumary' which is not declared in the state schema ['log', 'messages', 'token']; available keys here: ['log', 'messages']
```

Write side:

```
  [error] writer: writes key 'sumary' which is not declared in the state schema ['log', 'messages', 'token']; LangGraph silently drops unknown-key writes
```

(Note: in the *schema-free* default, an undeclared key a node writes is
auto-registered as a plain channel, so this error is what you get when you pass
an explicit `state=` schema that omits the key.)

### 3. Concurrent write with no reducer

Two nodes on parallel branches (`fanout`) write the same plain key. At runtime
LangGraph raises `InvalidUpdateError`; nae catches it at build time.

```python
from typing import Annotated, TypedDict
import operator
from langgraph.graph.message import add_messages
from nae import AgentNode, AgenticGraph, fanout

class S(TypedDict):
    messages: Annotated[list, add_messages]
    log: Annotated[list, operator.add]
    result: str                    # plain — no reducer

start = AgentNode(name="start", llm=llm)
a = AgentNode(name="a", llm=llm, writes=["result"])
b = AgentNode(name="b", llm=llm, writes=["result"])
join = AgentNode(name="join", llm=llm)

start > fanout(a, b) > join        # a and b write `result` concurrently
AgenticGraph(start_node=start, end_nodes={join}, state=S)
```

```
  [error] a: concurrently writes 'result' with node 'b', but 'result' has no reducer in the schema; concurrent writes to a plain key raise InvalidUpdateError at runtime
```

Fix: give `result` a reducer in the schema (`Annotated[list, operator.add]`),
or have only one branch write it.

### 4. Dangling decision choice

A `DecisionNode` (or `ConditionNode`) declares a choice you never wired to a
handler. The model could route to a branch that goes nowhere.

```python
from nae import DecisionNode, AgentNode, AgenticGraph

decide = DecisionNode(name="decide", llm=llm,
                      node_prompt="Is the sentiment positive or negative?",
                      choices=["positive", "negative"])
happy = AgentNode(name="happy", llm=llm)

decide["positive"] > happy         # "negative" is never wired
AgenticGraph(start_node=decide, end_nodes={happy})
```

```
  [error] decide: choice 'negative' has no connected node; wire it, e.g. `decision['negative'] > some_node`
```

### 5. Non-end node with no child (dead end)

Every node that isn't in `end_nodes` must lead somewhere. A node with no child
that isn't declared terminal would leave the graph with nowhere to go. This is a
**build-time `ValueError`** (raised as the graph is wired, before the dataflow
pass):

```python
p = AgentNode(name="p", llm=llm)
q = AgentNode(name="q", llm=llm)
r = AgentNode(name="r", llm=llm)
p > [q, r]                                   # q and r both run after p
AgenticGraph(start_node=p, end_nodes={r})    # but only r is terminal — q dead-ends
```

```
ValueError: Node 'q' has no children and is not an end node.
```

Fix: add `q` to `end_nodes`, or wire `q` onward.

### 6. Duplicate node names

Node names must be unique within a graph (the build keys nodes by name, so a
collision would silently merge two distinct nodes). Detected during name
resolution / the build walk as a **`ValueError`**:

```python
a = AgentNode(name="worker", llm=llm)
b = AgentNode(name="worker", llm=llm)   # same explicit name, different node
a > b
```

```
ValueError: Duplicate node name 'worker': two distinct nodes share this name. Node names must be unique within a graph.
```

(Auto/inferred names never collide — they get suffixed `_1`, `_2`. Only two
*explicit* identical names, or an explicit name clashing with a distinct node,
raise.)

### 7. Prompt-placeholder / scalar-read mismatch

A `node_prompt` with a `{placeholder}` that isn't a declared scalar read is a
guaranteed runtime `KeyError` when the prompt is interpolated — a hard error:

```python
AgentNode(name="n", llm=llm, reads=["topic"],
          node_prompt="Write about {topic} in {style}.")   # {style} not read
```

```
  [error] n: node_prompt references placeholder {style} but 'style' is not a declared scalar read ['topic']; this raises KeyError when the prompt is interpolated
```

The reverse — a declared scalar read the prompt never interpolates — is a
**warning** (a dead read, fetched and silently ignored).

## Warnings (build succeeds)

These don't fail the build; they surface via `graph.validate()` /
`summary()`, and `strict=True` promotes them to errors:

- **Some-paths read** — a key produced on some branches but not all (`may` but
  not `must`): `reads 'x' which is only produced on SOME paths; guaranteed on
  all paths: [...]`.
- **Optional read** (`reads=["x?"]`) never produced — explicit opt-out, at most
  a warning.
- **Unreachable node** — `is unreachable from the start node`.
- **End node with children** — `is an end node but still has children; the
  children are unreachable from it`.

## Inspecting the analysis: `summary()`

`graph.summary()` prints a table of every node's `reads`, `writes`, and
available keys, plus the error/warning counts — the same `may`/`must` sets the
validator computes:

```
AgenticGraph 'graph'   state=AgenticState
initial: log, messages, token
------------------------------------------------------------
node    type       reads     writes         available
frame   AgentNode  messages  log,messages   log,messages
...
------------------------------------------------------------
5 nodes · 0 errors · 0 warning(s)
```

See [State & observability](state.md) for the full state-channel model the
analysis runs over.
