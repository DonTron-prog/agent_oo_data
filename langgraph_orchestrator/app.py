# langgraph_orchestrator/app.py
import json
import os
from typing import List, Dict, Optional

from langchain_aws import ChatBedrock # Or your preferred LLM integration
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.pydantic_v1 import BaseModel, Field

from langgraph.graph import StateGraph, END
from langgraph.checkpoint.dynamodb import DynamoDBSaver # For state persistence

# --- 1. Define Agent State (as defined in Step 1) ---
class AgentState(TypedDict):
    original_request: str
    plan: Optional[List[str]]
    target_urls: Optional[List[str]]
    scraped_data: Optional[Dict[str, str]]
    structured_data: Optional[Dict[str, dict]]
    insights: Optional[str]
    error_message: Optional[str]

# --- 2. Initialize LLM (using Amazon Bedrock for this example) ---
# Ensure your Lambda has permissions to invoke Bedrock models
# and the necessary environment variables are set (AWS_REGION, etc.)
# You might need to adjust model_id based on availability and your needs
llm = ChatBedrock(
    model_id="anthropic.claude-3-sonnet-20240229-v1:0", # Or another model
    # model_id="anthropic.claude-3-haiku-20240307-v1:0", # Cheaper/Faster option for some tasks
    model_kwargs={"temperature": 0.1}
)

# --- 3. Define Pydantic model for structured LLM output for planning ---
class PlanOutput(BaseModel):
    plan_steps: List[str] = Field(description="A list of high-level steps to address the request.")
    initial_urls_to_scrape: List[str] = Field(description="A list of initial URLs that should be scraped to gather information. Only include if explicitly mentioned or obviously inferable from the request.")
    search_queries: Optional[List[str]] = Field(description="Keywords or questions for a search engine if direct URLs are not known.")

# --- 4. Input Processing & Planning Node Function ---
def input_processing_and_planning_node(state: AgentState) -> AgentState:
    print("--- Executing Input Processing & Planning Node ---")
    original_request = state.get("original_request")
    if not original_request:
        return {"error_message": "Original request is missing."}

    prompt_template = ChatPromptTemplate.from_messages([
        ("system", "You are an expert research assistant. Your goal is to understand a user's request, formulate a plan, and identify initial data sources like URLs or search queries. Respond using the provided JSON schema."),
        ("human", "Based on the following user request, please generate a plan and identify initial URLs to scrape or search queries:\n\nUser Request: \"{request}\"\n\nIf specific URLs are mentioned, prioritize them. If the request is broad, suggest relevant search queries.")
    ])
    
    # Using structured output to get reliable JSON from the LLM
    structured_llm = llm.with_structured_output(PlanOutput)
    planner_chain = prompt_template | structured_llm

    try:
        response: PlanOutput = planner_chain.invoke({"request": original_request})
        
        print(f"LLM Planner Response: {response}")
        
        updated_state = {
            "plan": response.plan_steps,
            "target_urls": response.initial_urls_to_scrape
            # You might also store response.search_queries if you plan to use a search tool
        }
        # If you have search queries and no URLs, you might immediately decide the next step
        # is a search tool rather than your BeautifulSoupScraper.
        # For now, we assume URLs will be found for the scraper.

        return updated_state
    except Exception as e:
        print(f"Error in planning node: {e}")
        return {"error_message": f"Failed to process request with LLM: {str(e)}"}

# --- (Other nodes like your scraper invoker will be added here later) ---

# --- 5. Define the Graph ---
# For now, we'll have a simple graph: input_planner -> END
# This allows us to test the planner node in isolation.

workflow = StateGraph(AgentState)
workflow.add_node("input_planner", input_processing_and_planning_node)
workflow.set_entry_point("input_planner")

# To test this single node, you can just end the graph here.
# Later, you'll add edges to other nodes like the scraper.
workflow.add_edge("input_planner", END) # Or a "print_state_node" for debugging

# --- 6. Setup Checkpointer (Essential for Lambda's statelessness) ---
# You'll need to create a DynamoDB table for this.
# Table name: e.g., 'langgraph-agent-checkpoints'
# Primary key: 'thread_id' (String)
# Ensure your Lambda's IAM role has GetItem, PutItem, UpdateItem, DeleteItem permissions on this table.
DYNAMODB_TABLE_NAME = os.environ.get("DYNAMODB_CHECKPOINT_TABLE", "langgraph-agent-checkpoints")
memory = DynamoDBSaver.from_conn_string(DYNAMODB_TABLE_NAME) # Uses boto3 default session

app = workflow.compile(checkpointer=memory)

# --- 7. Lambda Handler ---
def lambda_handler(event, context):
    print(f"Received event: {json.dumps(event)}")

    # Assuming the input request comes in the event body
    # For API Gateway, event['body'] might be a JSON string
    try:
        if isinstance(event.get('body'), str):
            body = json.loads(event.get('body'))
        else:
            body = event.get('body', {}) # Or directly from event if not API Gateway with body mapping
        
        original_request = body.get('original_request')
        thread_id = body.get('thread_id') # Important for conversations/persistent state

        if not original_request:
            return {"statusCode": 400, "body": json.dumps({"error": "original_request is required in the body"})}
        if not thread_id: # Generate one if not provided, or enforce it for specific use cases
            import uuid
            thread_id = f"lambda-thread-{uuid.uuid4()}"
            print(f"Generated new thread_id: {thread_id}")


        # Config for the graph invocation, especially the thread_id
        config = {"configurable": {"thread_id": thread_id}}
        
        # Prepare inputs for the graph
        inputs = {"original_request": original_request}

        # Invoke the graph
        # For a non-streaming, single response:
        final_state = app.invoke(inputs, config=config)
        
        # If you use `app.stream(...)` you'd iterate through the results
        # and the final state is more complex to get directly here.
        # For now, invoke is simpler.

        print(f"Final state from LangGraph: {final_state}")

        # Determine response based on final state
        if final_state.get("error_message"):
            return {
                "statusCode": 500,
                "body": json.dumps({
                    "error": final_state["error_message"],
                    "thread_id": thread_id,
                    "final_state": final_state # for debugging
                })
            }
        else:
            return {
                "statusCode": 200,
                "body": json.dumps({
                    "message": "Processing complete.",
                    "thread_id": thread_id,
                    "plan": final_state.get("plan"),
                    "target_urls": final_state.get("target_urls")
                    # Add other relevant parts of the state you want to return
                })
            }

    except Exception as e:
        print(f"Error in lambda_handler: {str(e)}")
        return {"statusCode": 500, "body": json.dumps({"error": f"Internal server error: {str(e)}"})}

# --- Example for local testing (optional, won't run in Lambda directly like this) ---
if __name__ == "__main__":
    print("Local testing of the Input Processing & Planning Node...")
    # Mock event and context for local testing
    test_event_body = {
        "original_request": "Find information about recent AI advancements in healthcare and identify some key research papers or articles.",
        "thread_id": "local-test-thread-123"
    }
    mock_event = {"body": json.dumps(test_event_body)}
    
    # Set dummy environment variable for local testing if table doesn't exist
    # (though for DynamoDBSaver it will try to connect unless mocked)
    if "DYNAMODB_CHECKPOINT_TABLE" not in os.environ:
        print("WARNING: DYNAMODB_CHECKPOINT_TABLE not set. DynamoDBSaver might fail if table doesn't exist or AWS creds aren't configured.")
        # For true local testing with DynamoDB, you might use DynamoDB Local.
        # For now, this test will likely show if the node logic itself has Python errors.
    
    response = lambda_handler(mock_event, None)
    print("\nLambda handler response (local test):")
    print(json.dumps(response, indent=2))

    # You can also directly test the node function:
    # test_state_input = {"original_request": "Tell me about LangGraph."}
    # node_output = input_processing_and_planning_node(test_state_input)
    # print("\nDirect node output:")
    # print(node_output)