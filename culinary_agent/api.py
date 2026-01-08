"""FastAPI server for the Culinary Agent Backend.

Run with: uvicorn api:app --reload
Docs available at: http://127.0.0.1:8000/docs
"""

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
from typing import Optional, Literal
from contextlib import asynccontextmanager
import sys
from pathlib import Path

# Add src directory to path for imports
src_path = Path(__file__).parent / "src"
sys.path.insert(0, str(src_path))

# Import your existing backend modules
from agent.core import create_agent, CulinaryAgent
from agent.prompts import SYSTEM_PROMPT

# --- Data Models (Request/Response Contract) ---

class MealPlanRequest(BaseModel):
    """Request model for meal plan generation/modification."""
    
    diet_constraints: str = Field(
        ..., 
        example="high protein, vegan",
        description="Dietary preferences and constraints"
    )
    mode: Literal["Full Day", "Breakfast", "Lunch", "Dinner"] = Field(
        default="Full Day",
        description="What to generate/update"
    )
    previous_plan: Optional[str] = Field(
        None, 
        description="The text of the previous plan if updating specific meals"
    )
    
    class Config:
        json_schema_extra = {
            "example": {
                "diet_constraints": "vegan, gluten-free",
                "mode": "Full Day",
                "previous_plan": None
            }
        }

class MealPlanResponse(BaseModel):
    """Response model for meal plan generation."""
    
    status: Literal["success", "error"]
    result: str
    mode: str
    diet_constraints: str


class HealthResponse(BaseModel):
    """Health check response."""
    
    status: Literal["healthy", "initializing", "error"]
    model: str
    version: str

# --- Lifecycle Management (Efficient Loading) ---
# Load the heavy Agent (FAISS/Models) only ONCE when the server starts.

agent_instance: Optional[CulinaryAgent] = None

@asynccontextmanager
async def lifespan(app: FastAPI):
    """Manage application lifecycle - load agent on startup, cleanup on shutdown."""
    global agent_instance
    print("🍳 Initializing Culinary Agent Backend...")
    try:
        # Reusing your existing create_agent logic
        agent_instance = create_agent(top_k_recipes=1, max_steps=15)
        print("✅ Agent is ready to cook!")
    except Exception as e:
        print(f"❌ Failed to load agent: {e}")
        print("Server will start but /generate-plan will return 503")
    
    yield
    
    # Cleanup logic (if needed) goes here
    print("👋 Shutting down Culinary Agent.")

# --- The FastAPI App ---

app = FastAPI(
    title="Culinary Agent API",
    description="Smart Daily Meal Planner with AI-powered recipe generation and adaptation",
    version="1.0.0",
    lifespan=lifespan
)

# Add CORS middleware for frontend integration
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # In production, specify actual origins
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# --- API Endpoints ---

@app.get("/", tags=["Root"])
def read_root():
    """Root endpoint - API information."""
    return {
        "message": "Culinary Agent API",
        "version": "1.0.0",
        "endpoints": {
            "health": "/health",
            "generate_plan": "/generate-plan",
            "docs": "/docs"
        }
    }

@app.get("/health", response_model=HealthResponse, tags=["Health"])
def health_check():
    """Check if the server and agent are running properly."""
    if not agent_instance:
        return HealthResponse(
            status="initializing",
            model="Qwen/Qwen2.5-14B-Instruct",
            version="1.0.0"
        )
    
    return HealthResponse(
        status="healthy",
        model="Qwen/Qwen2.5-14B-Instruct",
        version="1.0.0"
    )

@app.post("/generate-plan", response_model=MealPlanResponse, tags=["Meal Planning"])
def generate_meal_plan(request: MealPlanRequest):
    """
    Generate or modify meal plans based on dietary constraints.
    
    - **Full Day**: Creates a complete meal plan (Breakfast, Lunch, Dinner)
    - **Breakfast/Lunch/Dinner**: Updates only the specified meal, keeping others unchanged
    
    Args:
        request: MealPlanRequest with diet_constraints, mode, and optional previous_plan
        
    Returns:
        MealPlanResponse with generated/modified meal plan
        
    Raises:
        HTTPException 503: If agent is not initialized
        HTTPException 500: If plan generation fails
    """
    if not agent_instance:
        raise HTTPException(
            status_code=503, 
            detail="Agent not ready. Please wait for initialization or check server logs."
        )

    # 1. Construct the 'User Request' string based on Mode
    if request.mode == "Full Day":
        user_intent = f"Create a full daily meal plan. Constraint: {request.diet_constraints}"
    else:
        user_intent = f"Update ONLY the {request.mode}. Constraint: {request.diet_constraints}"

    # 2. Construct the Full Prompt (Logic ported from main.py)
    if not request.previous_plan or request.mode == "Full Day":
        # Fresh Start
        full_prompt = f"{SYSTEM_PROMPT}\n\nUSER REQUEST: {user_intent}"
    else:
        # Context-Aware Update
        full_prompt = f"""
            {SYSTEM_PROMPT}

            CONTEXT: The user has an existing plan.
            PREVIOUS PLAN:
            {request.previous_plan}

            USER REQUEST: {user_intent}

            TASK: 
            1. If the user asked to UPDATE a specific meal, keep the other meals from PREVIOUS PLAN and only rewrite the requested one.
            2. Maintain the format.
            """

    # 3. Run the Agent
    print(f"🚀 Processing request: {user_intent}")
    try:
        # Use run_with_retry for robustness
        if hasattr(agent_instance, 'run_with_retry'):
            response_text = agent_instance.run_with_retry(full_prompt)
        else:
            response_text = agent_instance.run(full_prompt)
        
        return MealPlanResponse(
            status="success",
            result=str(response_text),
            mode=request.mode,
            diet_constraints=request.diet_constraints
        )
    except Exception as e:
        print(f"❌ Error generating plan: {e}")
        raise HTTPException(
            status_code=500, 
            detail=f"Plan generation failed: {str(e)}"
        )


# --- Run Instructions ---
if __name__ == "__main__":
    import uvicorn
    print("Starting Culinary Agent API server...")
    print("Docs available at: http://127.0.0.1:8000/docs")
    uvicorn.run(app, host="0.0.0.0", port=8000)