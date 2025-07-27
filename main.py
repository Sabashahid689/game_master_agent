import os
from dotenv import load_dotenv
from agents import Agent, Runner, AsyncOpenAI, OpenAIChatCompletionsModel
from agents.run import RunConfig
from game_tool import roll_dice, generate_event

# Load environment variables from .env file
load_dotenv()

# Initialize the OpenAI Async Client
client = AsyncOpenAI(
    api_key=os.getenv("GEMINI_API_KEY"),
    base_url="https://generativelanguage.googleapis.com/v1beta/openai/"  # Adjust if needed
)

# Configure the model to use
model = OpenAIChatCompletionsModel(
    model="gemini-2.0-flash",
    openai_client=client
)

# Create a configuration object for running agents
config = RunConfig(
    model=model,
    tracing_disabled=True
)

# --- Define Game Agents ---

narrator_agent = Agent(
    name="NarratorAgent",
    instructions="You narrate the fantasy adventure and ask the player for choices.",
    model=model
)

monster_agent = Agent(
    name="MonsterAgent",
    instructions="You handle monster encounters using roll_dice and generate_event tools.",
    model=model,
    tools=[roll_dice, generate_event]
)

item_agent = Agent(
    name="ItemAgent",
    instructions="You provide rewards or items to the player after successful events.",
    model=model
)

# --- Main Game Logic ---

def main():
    """
    Starts the adventure game using agent-based narration and event simulation.
    """
    print("🎮 Welcome to the Fantasy Adventure Game!")
    choice = input("🌲 Do you enter the forest or turn back? -> ").strip()

    # Step 1: Narration
    result1 = Runner.run_sync(narrator_agent, choice, run_config=config)
    print("\n📖 Story:", result1.final_output)

    # Step 2: Monster Encounter
    result2 = Runner.run_sync(monster_agent, "Start encounter", run_config=config)
    print("\n👹 Encounter:", result2.final_output)

    # Step 3: Reward
    result3 = Runner.run_sync(item_agent, "Give reward to player", run_config=config)
    print("\n🎁 Reward:", result3.final_output)

# Entry point
if __name__ == "__main__":
    main()
    # print("Game Over. Thanks for playing!")
