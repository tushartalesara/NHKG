# Hindi Graph QnA Demo

This demo uses the existing NHKG event graph as a small Hindi question-answering backend.

Important positioning:
- this is a graph-grounded demo
- it is best for structured event questions
- it is not a full open-domain Hindi chatbot

## What it uses

Default graph:
- `final_outputs/demo_graph/canonical_demo_graph.nq`

Demo script:
- [hindi_graph_qa_demo.py](d:\Acads\MTP_Sem_10\NHKG\demo\hindi_graph_qa_demo.py)

## How to run

Print example questions:

```powershell
D:\Python310\python.exe -X utf8 demo\hindi_graph_qa_demo.py --examples
```

Ask one question directly:

```powershell
D:\Python310\python.exe -X utf8 demo\hindi_graph_qa_demo.py --question "राम कहाँ गए?"
```

Start interactive mode:

```powershell
D:\Python310\python.exe -X utf8 demo\hindi_graph_qa_demo.py
```

Use a different graph:

```powershell
D:\Python310\python.exe -X utf8 demo\hindi_graph_qa_demo.py --graph "PATH_TO_GRAPH.nq" --question "पिताजी क्या लाए?"
```

## Suggested professor demo questions

These are grounded in the current local graph content.

1. `राम कहाँ गए?`
2. `राम से जुड़ी घटनाएँ क्या हैं?`
3. `मोहन ने क्या दिया?`
4. `माँ ने क्या बनाया?`
5. `पिताजी क्या लाए?`
6. `रवि ने क्या बेचा?`
7. `चोर को किसने पकड़ा?`
8. `पुलिस ने क्या किया?`
9. `न्यायाधीश ने क्या किया?`

## What to say during the demo

Simple explanation:

- The question is interpreted as a small structured intent.
- The script looks up matching event nodes in the graph.
- It answers using trigger text, role links, and source sentence evidence.
- So the answer is grounded in the graph, not invented from scratch.

## Best use cases

- who did what
- what happened to whom
- where someone went
- list of events connected to an entity

## Current limitations

- The graph is structurally strong, but semantic specificity is not perfect.
- Some surviving events are generic.
- Argument quality is weaker than trigger quality.
- So this demo is best presented as a graph-backed prototype for Hindi QnA, not as a final production QA engine.
