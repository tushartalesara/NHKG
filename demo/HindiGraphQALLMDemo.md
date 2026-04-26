# LLM + KG Hindi QnA Demo

This is the professor-facing version that is closest to the original vision:

- retrieve evidence from the NHKG graph
- give only that evidence to an LLM
- generate a grounded Hindi answer

## Architecture

Flow:

1. Hindi question
2. Graph retrieval over event nodes
3. Top event evidence block
4. Local `llama_cpp` model
5. Grounded Hindi answer + evidence used

So this is not a free-form chatbot. It is a graph-grounded QnA demo.

## Script

- [hindi_graph_qa_llm.py](d:\Acads\MTP_Sem_10\NHKG\demo\hindi_graph_qa_llm.py)

## Requirements

- a local GGUF model path
- `llama-cpp-python` available in the environment

You can provide the model in either way:

1. pass `--model PATH_TO_MODEL.gguf`
2. or set environment variable `NHKG_QA_MODEL`

## Run examples

Print sample questions:

```powershell
D:\Python310\python.exe -X utf8 demo\hindi_graph_qa_llm.py --examples
```

Ask one grounded question:

```powershell
D:\Python310\python.exe -X utf8 demo\hindi_graph_qa_llm.py --model "PATH_TO_MODEL.gguf" --question "??? ???? ???"
```

Interactive mode:

```powershell
D:\Python310\python.exe -X utf8 demo\hindi_graph_qa_llm.py --model "PATH_TO_MODEL.gguf"
```

Using the environment variable:

```powershell
$env:NHKG_QA_MODEL="PATH_TO_MODEL.gguf"
D:\Python310\python.exe -X utf8 demo\hindi_graph_qa_llm.py --question "???? ?? ???? ?????"
```

## Suggested demo questions

1. `??? ???? ???`
2. `??? ?? ????? ?????? ???? ????`
3. `???? ?? ???? ?????`
4. `??? ?? ???? ??????`
5. `?????? ???? ????`
6. `??? ?? ???? ?????`
7. `??? ?? ????? ??????`
8. `????? ?? ???? ?????`
9. `????????? ?? ???? ?????`

## What to tell your professor

Use this simple explanation:

- The graph stores structured events, participants, and evidence sentences.
- The retriever first finds relevant event nodes from the graph.
- Only those retrieved facts are shown to the LLM.
- The LLM then produces a fluent Hindi answer grounded in graph evidence.

## Why this is better than the rule-only demo

- more natural Hindi answers
- closer to the original KG-for-QnA goal
- still safer than a raw chatbot because retrieval happens first

## Honest limitation

The graph is structurally strong, but some event frames are still semantically generic.
So this should be presented as a grounded prototype for Hindi QnA, not a perfect QA engine.
