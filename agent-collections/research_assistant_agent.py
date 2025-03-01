import nest_asyncio
import os
import dotenv
import asyncio
from dataclasses import dataclass, field
from typing import List, Optional
from pydantic import BaseModel
from pydantic_ai import Agent
from pydantic_graph import BaseNode, GraphRunContext, End, Graph
from pydantic_ai.common_tools.duckduckgo import duckduckgo_search_tool

# Load environment variables
dotenv.load_dotenv()
os.environ["GOOGLE_API_KEY"] = os.getenv("GEMINI_API_KEY")

# Apply nest_asyncio to allow nested event loops (useful in notebooks)
nest_asyncio.apply()

## Define the State Class to Hold Data
@dataclass
class State:
    user_name: str = ""
    user_research_topic: str = ""
    search_results: List[dict] = field(default_factory=list)
    relevant_papers: List[dict] = field(default_factory=list)
    research_summary: Optional[dict] = None

## Define the Nodes (Steps in Workflow)
@dataclass
class GetUserResearchTopic(BaseNode[State]):
    async def run(self, ctx: GraphRunContext[State]) -> "SearchForInformation":
        name = input("Enter your name: ")
        topic = input("Enter your research topic: ")

        ctx.state.user_name = name
        ctx.state.user_research_topic = topic

        return SearchForInformation()

@dataclass
class SearchForInformation(BaseNode[State]):
    async def run(self, ctx: GraphRunContext[State]) -> "FilterRelevantPapers":
        topic = ctx.state.user_research_topic
        
        # Initialize search agent with DuckDuckGo tool
        search_agent = Agent(
            'google-gla:gemini-1.5-flash',
            tools=[duckduckgo_search_tool()],
            system_prompt="Search for comprehensive information on the given research topic. Focus on finding academic papers, research articles, and authoritative sources."
        )
        
        # Run the search
        search_query = f"latest research papers on {topic} academic articles"
        search_result = await search_agent.run(search_query)
        
        # Store search results in state
        ctx.state.search_results = search_result.data
        
        return FilterRelevantPapers()

class PaperInfo(BaseModel):
    title: str
    authors: Optional[List[str]] = None
    publication_date: Optional[str] = None
    source: str
    url: str
    relevance_score: int  # 1-10 scale
    key_topics: List[str]

class RelevantPapers(BaseModel):
    papers: List[PaperInfo]
    search_topic: str

@dataclass
class FilterRelevantPapers(BaseNode[State]):
    async def run(self, ctx: GraphRunContext[State]) -> "GenerateResearchSummary":
        # Initialize filtering agent
        filter_agent = Agent(
            'google-gla:gemini-1.5-flash',
            system_prompt="""You are a research paper filtering assistant. 
            Analyze the search results and identify the most relevant academic papers and research articles.
            Rate each paper's relevance on a scale of 1-10 and extract key information.
            Focus on recent, peer-reviewed, and high-quality sources.""",
            result_type=RelevantPapers
        )
        
        # Run the filtering
        filter_result = await filter_agent.run(
            f"""Filter these search results for the most relevant papers on "{ctx.state.user_research_topic}":
            {ctx.state.search_results}"""
        )
        
        # Store filtered papers in state
        ctx.state.relevant_papers = filter_result.data.papers
        
        return GenerateResearchSummary()

class ResearchSummary(BaseModel):
    title: str
    summary: str
    key_findings: List[str]
    methodology_insights: List[str]
    future_research_directions: List[str]
    bibliography: List[str]

@dataclass
class GenerateResearchSummary(BaseNode[State]):
    async def run(self, ctx: GraphRunContext[State]) -> "End":
        # Initialize research summary agent
        summary_agent = Agent(
            'google-gla:gemini-1.5-flash',
            system_prompt="""You are an expert research assistant specializing in creating comprehensive research summaries.
            Analyze the provided papers and synthesize the information into a cohesive summary.
            Identify key findings, methodological approaches, and future research directions.
            Create a properly formatted bibliography of the sources.""",
            result_type=ResearchSummary
        )
        
        # Run the summary generation
        summary_result = await summary_agent.run(
            f"""Create a comprehensive research summary on "{ctx.state.user_research_topic}" for {ctx.state.user_name} 
            based on these papers:
            {ctx.state.relevant_papers}"""
        )
        
        # Store summary in state
        ctx.state.research_summary = summary_result.data
        
        return End(ctx.state.research_summary)

## Define the Pydantic Graph
research_assistant_graph = Graph(
    nodes=[GetUserResearchTopic, SearchForInformation, FilterRelevantPapers, GenerateResearchSummary]
)

## Execute the Workflow
async def main():
    state = State()
    ctx = GraphRunContext(state=state, deps={})
    result = await research_assistant_graph.run(GetUserResearchTopic(), state=state)

    print("\n=== Research Summary ===")
    print(f"Title: {result.output.title}")
    print(f"\nSummary: {result.output.summary}")
    
    print("\nKey Findings:")
    for finding in result.output.key_findings:
        print(f"- {finding}")
    
    print("\nMethodology Insights:")
    for insight in result.output.methodology_insights:
        print(f"- {insight}")
    
    print("\nFuture Research Directions:")
    for direction in result.output.future_research_directions:
        print(f"- {direction}")
    
    print("\nBibliography:")
    for source in result.output.bibliography:
        print(f"- {source}")

if __name__ == "__main__":
    asyncio.run(main())

# Visualize the workflow
print("\n=== Workflow Diagram ===")
print(research_assistant_graph.mermaid_code(start_node=GetUserResearchTopic))
