# Handle the Imports
import os, sys
from dotenv import load_dotenv

from pydantic import BaseModel, Field, computed_field

from langchain_core.prompts import ChatPromptTemplate
from langgraph.func import entrypoint, task
from langchain_openai import ChatOpenAI

load_dotenv()
openai_api_key = os.getenv("OPENAI_API_KEY")
os.environ["OPENAI_API_KEY"] = openai_api_key

print("Imports successful!")

llm = ChatOpenAI(model="gpt-4.1-mini")

TEST_CASES = [
    # Lavender
    """
    Lavender is a perennial aromatic plant valued for its distinctive fragrance, resilience, and biochemical complexity. Native to the Mediterranean region, it thrives in dry, well-drained soils and intense sunlight—conditions that force the plant to concentrate essential oils as a survival mechanism. These oils, rich in compounds such as linalool and linalyl acetate, are responsible for lavender's characteristic scent and many of its functional properties. Structurally modest but chemically potent, lavender is a textbook example of how environmental stress can drive biological efficiency.

    Beyond its botanical profile, lavender occupies a rare intersection between traditional use and modern evidence-based application. It has long been employed in herbal medicine for its calming, antiseptic, and anti-inflammatory effects, and contemporary research largely supports these claims, particularly in the domains of anxiety reduction and sleep quality. Unlike many botanical remedies that rely on anecdote, lavender demonstrates measurable physiological effects, including modulation of the nervous system and mild analgesic action. That said, it is not a panacea, and exaggerated claims—especially in wellness marketing—should be treated with skepticism.

    Economically and culturally, lavender's significance far exceeds its physical footprint in the field. It underpins industries ranging from perfumery and cosmetics to pharmaceuticals and culinary arts, with cultivation practices refined for both yield and oil composition. However, quality varies dramatically depending on species, growing conditions, and extraction methods, and synthetic substitutes frequently dilute its reputation. The practical takeaway is straightforward: lavender is valuable precisely because it is specific—specific chemistry, specific conditions, specific effects—and when those specifics are ignored, what remains is little more than a pleasant smell.
    """,

    # Time
    """
    Time is the fundamental dimension that orders events, enables causality, and gives structure to change. Unlike space, it is experienced unidirectionally—past to present to future—an asymmetry that defines both physical processes and human perception. In classical mechanics, time is treated as uniform and absolute; in modern physics, it is intertwined with space, variable under motion and gravity, and inseparable from the dynamics it governs. Despite centuries of study, time remains less an object we observe than a framework within which observation itself becomes possible.

    From a physical standpoint, time's arrow is anchored in entropy. Systems evolve toward disorder, not because they must philosophically, but because statistics make the reverse overwhelmingly improbable. This thermodynamic bias explains why memory works backward, why causes precede effects, and why aging is irreversible. Relativity further complicates intuition by showing that simultaneity is not universal—two events can be ordered differently depending on the observer. These results are not semantic tricks; they are experimentally verified constraints that force a rethinking of what “now” even means.

    Practically and psychologically, time is both a resource and a distortion. Humans measure it obsessively, manage it poorly, and experience it subjectively—compressed under urgency, stretched under boredom, warped by emotion. Modern culture treats time as something to be optimized, monetized, or conquered, a framing that is intellectually shallow and operationally flawed. Time does not yield to efficiency hacks; it only exposes trade-offs. The blunt reality is that time is not scarce in the abstract—attention and prioritization are. Misunderstanding that distinction is why so much effort feels busy yet accomplishes little.
    """,

    # Wrinkles
    """
    Wrinkles are visible creases, folds, or ridges in the skin that develop as a result of structural and biochemical changes over time. At a fundamental level, they emerge when the skin loses collagen, elastin, and hyaluronic acid—key components responsible for firmness, elasticity, and hydration. As cell turnover slows with age, the dermis thins and becomes less resilient, making it easier for repetitive mechanical stresses, such as facial expressions, to leave permanent lines. In short, wrinkles are not a surface defect; they are a manifestation of deeper tissue-level degradation.

    While aging is inevitable, the rate and severity of wrinkle formation are heavily influenced by environmental and behavioral factors. Chronic exposure to ultraviolet radiation is the dominant accelerator, as UV light damages collagen fibers and disrupts normal skin repair mechanisms—a process known as photoaging. Smoking compounds this damage by reducing blood flow and introducing oxidative stress, while poor nutrition, dehydration, and inadequate sleep further impair the skin's ability to regenerate. These factors explain why two individuals of the same age can exhibit dramatically different skin textures.

    From a clinical and cosmetic standpoint, wrinkles are increasingly understood as modifiable outcomes rather than fixed consequences of aging. Preventive strategies—sun protection, topical retinoids, antioxidants, and lifestyle interventions—are demonstrably more effective than corrective measures applied late. Procedural treatments such as chemical peels, laser resurfacing, and injectable fillers can improve appearance, but they do not reverse the underlying biology. The hard truth is that wrinkle management is a long-term systems problem: neglect accumulates silently, and by the time wrinkles dominate the surface, the damage underneath is already well established.
    """,
]

import operator
from typing import Annotated

class ZettelLinks(BaseModel):
    links: dict[str, list[str]]

class TopicList(BaseModel):
    topics: list[str]

class Zettel(BaseModel):
    id: str
    title: str
    content: str
    links: list[str] = Field(default_factory=list)

class ZettelWorkerState(BaseModel):
    paragraph: str
    topic: str

class State(BaseModel):
    paragraph: str
    topics: list[str] = Field(default_factory=list)
    zettels: Annotated[list, operator.add]

# Prompts

GLOBAL_PROMPT = """
You are part of a knowledge-compilation system, not a chatbot.

Your task is to produce structured, reusable knowledge units.
Do NOT explain your reasoning.
Do NOT refer to the input text directly.
Do NOT add filler, examples, or conversational language.
Be precise, concise, and formal.

If the task cannot be completed exactly as requested, fail explicitly.
"""

TOPIC_DISCOVERY_PROMPT = """
You are a Topic Discovery Agent.

Task:
Extract a list of DISTINCT CONCEPTUAL TOPICS from the input text.

Give only 3 topics max

Rules:
- Each topic must be a noun or noun phrase.
- Topics must represent concepts, not sentences.
- Do NOT summarize or explain.
- Do NOT repeat overlapping topics.
- Do NOT include examples, applications, or conclusions.
- Output ONLY the topic list.

Good topics:
- "Gradient Descent"
- "Learning Rate"
- "Loss Function"

Bad topics:
- "How gradient descent works"
- "This paragraph discusses optimization"
- Full sentences

Return a JSON object matching this schema:
{{
  "topics": ["topic_1", "topic_2", ...]
}}

Input text:
{paragraph}
"""

ZETTEL_WRITER_PROMPT = """
You are a Zettelkasten Note Writer.

Task:
Write ONE atomic note about the assigned topic.

Assigned topic:
{topic}

Rules (STRICT):
- The note MUST express exactly ONE idea.
- If the topic requires multiple ideas, write only the core one.
- Do NOT reference the input text or its structure.
- Do NOT include examples unless essential.
- Do NOT include links or references to other notes.
- Use neutral, timeless, academic language.
- The note must be understandable in isolation.

Atomicity test:
"If this note were removed, would exactly one idea be lost?"
If not, the note is invalid.

Return a JSON object matching this schema:
{{
  "id": "<unique identifier>",
  "title": "<concise concept name>",
  "content": "<single-idea explanation>",
  "topics": ["<domain tags>"],
  "links": []
}}

Input text:
{paragraph}
"""

ZETTEL_LINKER_PROMPT = """
You are a Knowledge Graph Linker.

Task:
Identify CONCEPTUAL DEPENDENCIES between notes.

Definition:
Note A depends on Note B if understanding A requires understanding B.

Rules:
- Do NOT link based on similarity alone.
- Do NOT create links unless dependency is necessary.
- Prefer fewer, stronger links.
- Links must be directional.
- Use note titles as identifiers.

Return ONLY a JSON object in this format:
{{
  "links": {{
    "Note A Title": ["Required Note B", "Required Note C"],
    "Note D Title": []
  }}
}}

Notes:
{zettels}
"""

topic_discovery_prompt = ChatPromptTemplate.from_messages(
    [
        ("system", GLOBAL_PROMPT),
        ("system", TOPIC_DISCOVERY_PROMPT)
    ]
)
topic_discovery_chain = topic_discovery_prompt | llm.with_structured_output(TopicList)

zettel_writer_prompt = ChatPromptTemplate.from_messages(
    [
        ("system", GLOBAL_PROMPT),
        ("system", ZETTEL_WRITER_PROMPT)
    ]
)
zettel_writer_chain = zettel_writer_prompt | llm.with_structured_output(Zettel)

zettel_linker_prompt = ChatPromptTemplate.from_messages(
    [
        ("system", GLOBAL_PROMPT),
        ("system", ZETTEL_LINKER_PROMPT)
    ]
)
zettel_linker_chain = zettel_linker_prompt | llm.with_structured_output(ZettelLinks)

from langgraph.types import Send

def topic_discovery_node(state: State):
    topics = topic_discovery_chain.invoke({"paragraph": state.paragraph}).topics
    return {"topics": topics}

def zettel_writing_worker(state: ZettelWorkerState):
    print("writing zettel")
    zettel = zettel_writer_chain.invoke({"paragraph": state.paragraph, "topic": state.topic})
    return {"zettels": [zettel]}

def zettel_writing_node(state: State):
    return [Send("zettel_worker", {"paragraph": state.paragraph, "topic": topic}) for topic in state.topics]
    

from langgraph.graph import StateGraph, START, END
from langgraph.checkpoint.memory import InMemorySaver

checkpointer = InMemorySaver()
graph = StateGraph(State)

graph.add_node("zettel_writing", zettel_writing_node)
graph.add_node("zettel_worker", zettel_writing_worker)
graph.add_node("topic_discovery", topic_discovery_node)

graph.add_edge(START, "topic_discovery")
graph.add_edge("topic_discovery", "zettel_writing")
graph.add_edge("zettel_worker", END)

compiled_graph = graph.compile(checkpointer=checkpointer)
# Workflow is now the compiled graph

result = compiled_graph.invoke({"paragraph": TEST_CASES[0]}, {"configurable": {"thread_id": "1"}})
result




