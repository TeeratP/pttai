# Node types

pttai ships four node types. All are callables (`__call__(state) -> delta`)
invoked by LangGraph with the shared state; each returns **only the keys it
updates** and never mutates state in place, which is what keeps checkpointing,
parallel-branch merges, and subgraph composition correct. This page is a
conceptual tour of when to reach for each — for full signatures and every
keyword argument, see the [API Reference](api/nodes.md), generated from the
docstrings.

## `AgentNode`

The workhorse. It prepends your `node_prompt` as a system message, calls the
LLM, and returns the result. Reach for it for any "have a model do something"
step. Give it `tools=[...]` and it runs a built-in tool-call loop (execute
tool → append result → re-call the model, until the model stops), folding away
the `ToolNode` + conditional-edge + loop-back plumbing you'd write by hand. Give
it `writes={"score": int}` and it returns native-typed structured output instead
of appending to the conversation.

## `DecisionNode`

LLM branching. Use it when *the model* should pick the path — classify, triage,
route by intent. It calls the model with constrained (structured) output over a
`Literal` of your `choices`, so the model can only return a valid branch; the
label goes to a private per-node channel, never into the conversation. Wire each
branch by indexing a choice: `decision["positive"] > handler`. It can also take
`tools=[...]` to gather context first and *then* route.

## `ConditionNode`

Deterministic branching with **no LLM**. Reach for it when the branch is
decidable in plain Python — a length check, a loop counter, a flag another node
wrote. A predicate `condition(state) -> str` returns one of `choices`; routing is
free, reproducible, and prompt-less. Wired exactly like a `DecisionNode`
(`route["retry"] > worker`), which makes it the natural tool for capping loops.

## `HumanNode`

Resumable human-in-the-loop, backed by LangGraph's `interrupt()`. Use it when a
person must review or supply something mid-run. It pauses and surfaces a payload
for review; when you invoke again with `Command(resume=value)` on the same
thread, the reply flows back into state (as a `HumanMessage`, or into any key a
router can gate on). Requires a `checkpointer` for the pause/resume to persist.

See these nodes in action, one at a time, in the [Basics gallery](examples/basics.md).
