from agents import function_tool
import random

@function_tool
def roll_dice() -> str:
    return f"You rolled a {random.randint(1, 6)}!"

@function_tool
def generate_event()-> str:
    events = [
        "You found a treasure chest!",
        "A wild monster appears!",
        "You discovered a hidden path!",
        "You encountered a friendly NPC!",
        "A storm is brewing in the distance!"
    ]
    return random.choice(events)