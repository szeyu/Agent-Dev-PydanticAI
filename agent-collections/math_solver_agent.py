import nest_asyncio
import os
import dotenv
import asyncio
import re
from dataclasses import dataclass, field
from typing import List, Optional, Dict, Any, Union
from pydantic import BaseModel, Field
from pydantic_ai import Agent, BinaryContent
from pydantic_graph import BaseNode, GraphRunContext, End, Graph

# Load environment variables
dotenv.load_dotenv()
os.environ["GOOGLE_API_KEY"] = os.getenv("GEMINI_API_KEY")
os.environ["OPENAI_API_KEY"] = os.getenv("OPENAI_API_KEY")

# Apply nest_asyncio to allow nested event loops (useful in notebooks)
nest_asyncio.apply()

## Define the State Class to Hold Data
@dataclass
class State:
    math_problem: str = ""
    problem_type: str = ""
    solution_plan: List[str] = field(default_factory=list)
    intermediate_steps: List[Dict[str, Any]] = field(default_factory=list)
    python_code: str = ""
    code_execution_result: str = ""
    final_solution: Dict[str, Any] = field(default_factory=dict)
    has_image: bool = False
    image_path: Optional[str] = None

## Define the Models for structured outputs with simpler schemas
class ProblemAnalysis(BaseModel):
    problem_type: str
    key_variables: Dict[str, str] = Field(default_factory=dict)
    constraints: List[str] = Field(default_factory=list)
    expected_output: str
    complexity_level: str

class SolutionPlan(BaseModel):
    steps: List[str] = Field(default_factory=list)
    mathematical_concepts: List[str] = Field(default_factory=list)
    formulas_needed: List[str] = Field(default_factory=list)

class PythonSolution(BaseModel):
    code: str
    explanation: str
    libraries_needed: List[str] = Field(default_factory=list)

class FinalSolution(BaseModel):
    answer: str
    units: Optional[str] = None
    step_by_step_explanation: List[str] = Field(default_factory=list)
    alternative_approaches: Optional[List[str]] = None
    visualization_code: Optional[str] = None

## Define the Nodes (Steps in Workflow)
@dataclass
class GetMathProblem(BaseNode[State]):
    async def run(self, ctx: GraphRunContext[State]) -> Union["AnalyzeProblem", "ProcessImageProblem"]:        
        has_image = input("Does your problem include an image? (yes/no): ").lower() == "yes"
        ctx.state.has_image = has_image
        
        if has_image:
            image_path = input("Enter the path to your image: ")
            ctx.state.image_path = image_path
            problem_text = input("Enter any additional text for your math problem: ")
            ctx.state.math_problem = problem_text
            return ProcessImageProblem()
        else:
            problem = input("Enter your math problem: ")
            ctx.state.math_problem = problem
            return AnalyzeProblem()

@dataclass
class ProcessImageProblem(BaseNode[State]):
    async def run(self, ctx: GraphRunContext[State]) -> Union["AnalyzeProblem", End]:
        try:            
            # Read the image file as binary data
            with open(ctx.state.image_path, "rb") as image_file:
                image_data = image_file.read()
            
            # Initialize image processing agent with Gemini
            image_agent = Agent(
                model='google-gla:gemini-1.5-flash',
                system_prompt="""You are a math problem interpreter. 
                Analyze the image containing a math problem and extract all relevant information.
                Identify equations, diagrams, graphs, and any text in the image.
                Convert the visual information into a clear textual description of the math problem."""
            )
            
            # Process the image using BinaryContent
            image_result = await image_agent.run([
                f"Extract and interpret the math problem from this image. Additional context: {ctx.state.math_problem}",
                BinaryContent(data=image_data, media_type='image/jpeg')
            ])
            
            # Combine the extracted information with any text provided
            full_problem = f"{ctx.state.math_problem}\n\nExtracted from image: {image_result.data}"
            ctx.state.math_problem = full_problem
            
            print("Successfully processed image with Gemini 1.5 Flash.")
            
        except Exception as e:
            print(f"Error processing image: {str(e)}")
            print("Exiting workflow...")

            # Create a proper error message in the final solution
            ctx.state.final_solution = {
                "answer": "Error: Unable to process the image",
                "units": None,
                "step_by_step_explanation": [f"The workflow encountered an error: {str(e)}"],
                "alternative_approaches": None,
                "visualization_code": None
            }
            
            # End the workflow with the error information
            return End(ctx.state.final_solution)
        
        return AnalyzeProblem()

@dataclass
class AnalyzeProblem(BaseNode[State]):
    async def run(self, ctx: GraphRunContext[State]) -> "PlanSolution":
        try:
            # Initialize analysis agent
            analysis_agent = Agent(
                'openai:gpt-4o-mini',
                system_prompt="""You are a math problem analyzer. 
                Carefully examine the given math problem and identify its type, key variables, constraints, and expected output.
                Assess the complexity level and determine what mathematical concepts are involved.
                
                For key_variables, always return a dictionary mapping variable names to descriptions.
                For example: {"x": "unknown value", "r": "radius of circle"}
                
                For constraints, return a list of constraints as strings.
                For example: ["x must be positive", "angle must be in radians"]
                
                For complexity_level, use one of: "basic", "intermediate", or "advanced".""",
                result_type=ProblemAnalysis,
                retries=3
            )
            
            # Run the analysis
            analysis_result = await analysis_agent.run(
                f"""Analyze this math problem thoroughly:
                {ctx.state.math_problem}
                
                Return a structured analysis with problem_type, key_variables (as a dictionary), 
                constraints (as a list), expected_output, and complexity_level."""
            )
            
            # Store problem type in state
            ctx.state.problem_type = analysis_result.data.problem_type
            
        except Exception as e:
            # Fallback if structured output fails
            print(f"Warning: Structured analysis failed with error: {str(e)}")
            print("Continuing with basic analysis...")
            
            # Use a simpler approach without structured output
            basic_agent = Agent(
                'openai:gpt-4o-mini',
                system_prompt="You are a math problem analyzer. Identify the type of math problem."
            )
            
            basic_result = await basic_agent.run(
                f"What type of mathematical problem is this? Just return the problem type in one short phrase: {ctx.state.math_problem}"
            )
            
            ctx.state.problem_type = basic_result.text.strip()
        
        return PlanSolution()

@dataclass
class PlanSolution(BaseNode[State]):
    async def run(self, ctx: GraphRunContext[State]) -> "GeneratePythonCode":
        try:
            # Initialize planning agent
            planning_agent = Agent(
                'openai:gpt-4o-mini',
                system_prompt="""You are a math solution planner.
                Create a detailed step-by-step plan to solve the given math problem.
                Identify all necessary mathematical concepts, formulas, and approaches.
                Break down complex problems into manageable steps.""",
                result_type=SolutionPlan,
                retries=3
            )
            
            # Run the planning
            plan_result = await planning_agent.run(
                f"""Create a detailed solution plan for this {ctx.state.problem_type} problem:
                {ctx.state.math_problem}
                
                Return a structured plan with steps (list of steps), mathematical_concepts (list), 
                and formulas_needed (list)."""
            )
            
            # Store solution plan in state
            ctx.state.solution_plan = plan_result.data.steps
            
        except Exception as e:
            # Fallback if structured output fails
            print(f"Warning: Structured planning failed with error: {str(e)}")
            print("Continuing with basic planning...")
            
            # Use a simpler approach without structured output
            basic_agent = Agent(
                'openai:gpt-4o-mini',
                system_prompt="You are a math solution planner. Create a step-by-step plan to solve math problems."
            )
            
            basic_result = await basic_agent.run(
                f"Create a step-by-step plan to solve this {ctx.state.problem_type} problem: {ctx.state.math_problem}"
            )
            
            # Extract steps from the text response
            steps_text = basic_result.text.strip()
            steps = [step.strip() for step in re.split(r'\d+\.|\n-|\n\*', steps_text) if step.strip()]
            ctx.state.solution_plan = steps
        
        return GeneratePythonCode()

@dataclass
class GeneratePythonCode(BaseNode[State]):
    async def run(self, ctx: GraphRunContext[State]) -> "ExecuteCode":
        try:
            # Initialize code generation agent
            code_agent = Agent(
                'openai:gpt-4o-mini',
                system_prompt="""You are a Python code generator for mathematical problems.
                Create clear, efficient, and well-commented Python code to solve the given math problem.
                Use appropriate libraries like NumPy, SymPy, or Matplotlib when beneficial.
                Include print statements to show intermediate steps and explanations.
                
                Format your output with:
                - code: The complete Python code solution
                - explanation: Brief explanation of how the code works
                - libraries_needed: List of Python libraries required""",
                result_type=PythonSolution,
                retries=3
            )
            
            # Run the code generation
            code_result = await code_agent.run(
                f"""Generate Python code to solve this {ctx.state.problem_type} problem:
                Problem: {ctx.state.math_problem}
                
                Solution Plan:
                {ctx.state.solution_plan}
                
                Create code that shows all steps clearly and includes explanations.
                Label each step with "Step X:" where X is the step number.
                Make sure to print the final answer clearly."""
            )
            
            # Store generated code in state
            ctx.state.python_code = code_result.data.code
            
        except Exception as e:
            # Fallback if structured output fails
            print(f"Warning: Structured code generation failed with error: {str(e)}")
            print("Continuing with basic code generation...")
            
            # Use a simpler approach without structured output
            basic_agent = Agent(
                'openai:gpt-4o-mini',
                system_prompt="You are a Python code generator for mathematical problems."
            )
            
            basic_result = await basic_agent.run(
                f"""Generate Python code to solve this {ctx.state.problem_type} problem:
                Problem: {ctx.state.math_problem}
                
                Solution Plan:
                {ctx.state.solution_plan}
                
                Create code that shows all steps clearly with print statements.
                Label each step with "Step X:" where X is the step number.
                Make sure to print the final answer clearly."""
            )
            
            # Extract code from the text response
            code_text = basic_result.text
            code_match = re.search(r'```python\s*(.*?)\s*```', code_text, re.DOTALL)
            if code_match:
                ctx.state.python_code = code_match.group(1)
            else:
                ctx.state.python_code = code_text
        
        return ExecuteCode()

@dataclass
class ExecuteCode(BaseNode[State]):
    async def run(self, ctx: GraphRunContext[State]) -> "GenerateFinalSolution":
        try:
            # Prepare the code execution
            from llm_sandbox import SandboxSession
            
            # Determine required libraries from the code
            libraries = []
            if "numpy" in ctx.state.python_code:
                libraries.append("numpy")
            if "sympy" in ctx.state.python_code:
                libraries.append("sympy")
            if "matplotlib" in ctx.state.python_code:
                libraries.append("matplotlib")
            if "scipy" in ctx.state.python_code:
                libraries.append("scipy")
            if "pandas" in ctx.state.python_code:
                libraries.append("pandas")
            if "math" in ctx.state.python_code:
                libraries.append("math")
            
            # Execute the code in a sandbox
            with SandboxSession(image="python:3.10", lang="python", keep_template=True) as session:
                result = session.run(ctx.state.python_code, libraries)
                ctx.state.code_execution_result = result.text
            
            # Track intermediate steps from the output
            steps = []
            step_pattern = re.compile(r"Step (\d+):(.*?)(?=Step \d+:|$)", re.DOTALL)
            matches = step_pattern.findall(ctx.state.code_execution_result)
            
            for i, (step_num, step_content) in enumerate(matches):
                steps.append({
                    "step_number": int(step_num),
                    "description": step_content.strip().split("\n")[0] if step_content.strip() else "",
                    "calculation": "\n".join(step_content.strip().split("\n")[1:-1]) if len(step_content.strip().split("\n")) > 2 else "",
                    "result": step_content.strip().split("\n")[-1] if len(step_content.strip().split("\n")) > 1 else "",
                    "explanation": step_content.strip()
                })
            
            ctx.state.intermediate_steps = steps
            
        except Exception as e:
            # If execution fails, record the error
            ctx.state.code_execution_result = f"Error executing code: {str(e)}"
        
        return GenerateFinalSolution()

@dataclass
class GenerateFinalSolution(BaseNode[State]):
    async def run(self, ctx: GraphRunContext[State]) -> "End":
        try:
            # Initialize final solution agent
            solution_agent = Agent(
                'openai:gpt-4o-mini',
                system_prompt="""You are a math solution synthesizer.
                Create a comprehensive final solution based on the executed code and intermediate steps.
                Provide clear explanations for each step and the final answer.
                Include alternative approaches when applicable and ensure the solution is accurate.
                
                Format your output with:
                - answer: The final answer to the problem (as a string)
                - units: Units of the answer, if applicable (or null)
                - step_by_step_explanation: List of explanation steps
                - alternative_approaches: List of alternative solution methods (or null)
                - visualization_code: Python code to visualize the solution, if applicable (or null)""",
                result_type=FinalSolution,
                retries=3
            )
            
            # Run the solution generation
            solution_result = await solution_agent.run(
                f"""Create a final solution for this {ctx.state.problem_type} problem:
                
                Problem: {ctx.state.math_problem}
                
                Solution Plan: {ctx.state.solution_plan}
                
                Python Code:
                ```python
                {ctx.state.python_code}
                ```
                
                Code Execution Result:
                {ctx.state.code_execution_result}
                
                Provide a comprehensive solution with clear step-by-step explanations."""
            )
            
            # Store final solution in state
            ctx.state.final_solution = solution_result.data.dict()
            
        except Exception as e:
            # Fallback if structured output fails
            print(f"Warning: Structured final solution failed with error: {str(e)}")
            print("Continuing with basic solution generation...")
            
            # Use a simpler approach without structured output
            basic_agent = Agent(
                'openai:gpt-4o-mini',
                system_prompt="You are a math solution synthesizer. Create clear explanations of math solutions."
            )
            
            basic_result = await basic_agent.run(
                f"""Create a final solution for this {ctx.state.problem_type} problem:
                
                Problem: {ctx.state.math_problem}
                
                Code Execution Result:
                {ctx.state.code_execution_result}
                
                Provide the final answer and a step-by-step explanation."""
            )
            
            # Create a basic solution structure
            answer_match = re.search(r'(?:answer|result|solution):\s*(.*?)(?:\n|$)', basic_result.text, re.IGNORECASE)
            answer = answer_match.group(1) if answer_match else "See explanation below"
            
            # Extract steps from the text response
            steps_text = basic_result.text.strip()
            steps = [step.strip() for step in re.split(r'\d+\.|\n-|\n\*', steps_text) if step.strip()]
            
            ctx.state.final_solution = {
                "answer": answer,
                "units": None,
                "step_by_step_explanation": steps,
                "alternative_approaches": None,
                "visualization_code": None
            }
        
        return End(ctx.state.final_solution)

## Define the Pydantic Graph
math_solver_graph = Graph(
    nodes=[GetMathProblem, ProcessImageProblem, AnalyzeProblem, PlanSolution, 
           GeneratePythonCode, ExecuteCode, GenerateFinalSolution]
)

## Execute the Workflow
async def main():
    state = State()
    result = await math_solver_graph.run(GetMathProblem(), state=state)

    print("\n=== Math Problem Solution ===")
    print(f"Problem: {state.math_problem}")
    
    # Check if result.output exists and has the expected structure
    if hasattr(result, 'output') and isinstance(result.output, dict) and 'answer' in result.output:
        print(f"\nFinal Answer: {result.output['answer']}")
        if result.output.get('units'):
            print(f"Units: {result.output['units']}")
        
        print("\nStep-by-Step Explanation:")
        for i, step in enumerate(result.output.get('step_by_step_explanation', []), 1):
            print(f"{i}. {step}")
        
        if result.output.get('alternative_approaches'):
            print("\nAlternative Approaches:")
            for approach in result.output['alternative_approaches']:
                print(f"- {approach}")
        
        print("\nPython Code Used:")
        print(f"```python\n{state.python_code}\n```")
        
        print("\nCode Execution Result:")
        print(state.code_execution_result)
    else:
        print("\nWorkflow ended early or with an error.")
        if hasattr(result, 'output'):
            print(f"Output: {result.output}")

if __name__ == "__main__":
    # Apply nest_asyncio to allow running asyncio in Jupyter notebooks
    nest_asyncio.apply()
    asyncio.run(main())

# Visualize the workflow
print("\n=== Workflow Diagram ===")
print(math_solver_graph.mermaid_code(start_node=GetMathProblem))