"""
Embedding Derivative Visualization

This script visualizes the change in cosine similarity between consecutive state descriptions
and between each state and the final state. The user provides a beginning and end stage,
and the model generates descriptions for specific steps along that transition.
"""

import asyncio
import numpy as np
import matplotlib.pyplot as plt
from typing import List, Optional
from pydantic import BaseModel, Field
from ollama import AsyncClient
import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(__file__)))
from utils import get_embedding, cosine_similarity
from gemini import get_gemini_output
import time

class StateDescription(BaseModel):
    """Model for storing state descriptions and their embeddings."""
    description: str = Field(..., description="The text description of the state")
    embedding: Optional[List[float]] = Field(None, description="Normalized embedding vector")
    step: int = Field(..., description="Step number in the transition")
    total_steps: int = Field(..., description="Total number of steps in the transition")


class EmbeddingDerivative:
    """Main class for handling embedding derivative calculations and visualization."""
    
    def __init__(self):
        self.client = AsyncClient()
        self.states: List[StateDescription] = []
        self.beginning_stage: str = ""
        self.end_stage: str = ""
    
    async def generate_description(self, step: int, total_steps: int) -> str:
        """Generate a description for a specific step in the transition."""
        try:
            prompt = f"""You are a writing assistant. You will be given a beginning state and end state.
            Your job is to write a description for a single intermediate state.


This is the beginning: {self.beginning_stage}
This is the end: {self.end_stage}


Write the description for the state {step} out of{total_steps}.

The description must:
- Describe only the current state.
- Describe the state *as it is in this exact moment*
- Avoid comparative language.
- NOT write about progression.
- NOT write any notes or follow-up questions.

Limit your response to 1000 characters.
"""
            
            # Check if model starts with 'gemini' and use appropriate client
            if self.model.lower().startswith('gemini'):
                response = get_gemini_output(prompt)
                return response.strip()
            else:
                response = await self.client.generate(model=self.model, prompt=prompt)
                time.sleep(10)
                return response['response'].strip()
        except Exception as e:
            print(f"Error generating description for step {step}/{total_steps}: {e}")
            return f"Error: {e}"
    
    def plot_similarities(self):
        """Create plots for cosine similarities."""
        if len(self.states) < 2:
            print("Need at least 2 states to plot similarities.")
            return
        
        # Prepare data for plotting
        steps = [state.step for state in self.states]
        total_steps = self.states[0].total_steps
        
        # Calculate consecutive similarities (N to N+1)
        consecutive_similarities = []
        for i in range(len(self.states) - 1):
            sim = cosine_similarity(
                self.states[i].embedding, 
                self.states[i + 1].embedding
            )
            consecutive_similarities.append(sim)
        
        # Calculate similarities to final state (N to last)
        final_similarities = []
        final_embedding = self.states[-1].embedding
        for i in range(len(self.states) - 1):  # Exclude the last state itself
            sim = cosine_similarity(
                self.states[i].embedding, 
                final_embedding
            )
            final_similarities.append(sim)
        
        # Create the plots
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 10))
        
        # Plot 1: Consecutive similarities (N to N+1)
        ax1.plot(steps[:-1], consecutive_similarities, 'bo-', linewidth=2, markersize=8)
        ax1.set_xlabel('Step Number')
        ax1.set_ylabel('Cosine Similarity')
        ax1.set_title(f'Cosine Similarity: Step N to Step N+1 (Total Steps: {total_steps})')
        ax1.grid(True, alpha=0.3)
        ax1.set_ylim(-1, 1)
        
        # Set x-axis to show step numbers clearly
        ax1.set_xticks(steps[:-1])
        
        # Add value labels on points
        for i, sim in enumerate(consecutive_similarities):
            ax1.annotate(f'{sim:.3f}', (steps[i], sim), 
                        textcoords="offset points", xytext=(0,10), ha='center')
        
        # Plot 2: Similarities to final state (N to last)
        ax2.plot(steps[:-1], final_similarities, 'ro-', linewidth=2, markersize=8)
        ax2.set_xlabel('Step Number')
        ax2.set_ylabel('Cosine Similarity')
        ax2.set_title(f'Cosine Similarity: Step N to Final Step {total_steps}')
        ax2.grid(True, alpha=0.3)
        ax2.set_ylim(-1, 1)
        
        # Set x-axis to show step numbers clearly
        ax2.set_xticks(steps[:-1])
        
        # Add value labels on points
        for i, sim in enumerate(final_similarities):
            ax2.annotate(f'{sim:.3f}', (steps[i], sim), 
                        textcoords="offset points", xytext=(0,10), ha='center')
        
        plt.tight_layout()
        plt.show()
        
        # Print summary
        print("\n=== Similarity Summary ===")
        print(f"Transition: {self.beginning_stage} → {self.end_stage}")
        print(f"Total steps: {total_steps}")
        print("\nConsecutive similarities (Step N to Step N+1):")
        for i, sim in enumerate(consecutive_similarities):
            print(f"  Step {steps[i]} → Step {steps[i+1]}: {sim:.4f}")
        
        print("\nSimilarities to final step (Step N to Final Step):")
        for i, sim in enumerate(final_similarities):
            print(f"  Step {steps[i]} → Final Step {total_steps}: {sim:.4f}")


async def main():
    """Main function to run the embedding derivative visualization."""
    print("This is to visualize the change in cosine of a state's description")
    print("We describe the state of something and then we plot the cosine difference from one state to another")
    print()
    
    # Get user input for beginning and end stages
    print("Please provide the beginning and end stages for the transition:")
    beginning_stage = input("Beginning stage: ").strip()
    if not beginning_stage:
        print("Beginning stage cannot be empty.")
        return
    
    end_stage = input("End stage: ").strip()
    if not end_stage:
        print("End stage cannot be empty.")
        return
    
    print()
    print(f"Transition: {beginning_stage} → {end_stage}")
    print()
    
    # Get number of steps
    try:
        total_steps = int(input("How many steps should be generated? "))
        if total_steps < 2:
            print("Need at least 2 steps to calculate similarities.")
            return
    except ValueError:
        print("Please enter a valid number.")
        return
    
    # Get model
    model = input("Model: ").strip()
    if not model:
        print("Model cannot be empty.")
        return
    
    print(f"\nGenerating {total_steps} steps using model '{model}'...")
    print(f"Beginning: {beginning_stage}")
    print(f"End: {end_stage}")
    print()
    
    # Initialize the embedding derivative handler
    handler = EmbeddingDerivative()
    handler.beginning_stage = beginning_stage
    handler.end_stage = end_stage
    handler.model = model
    
    # PHASE 1: Generate all descriptions first (keep LLM model in memory)
    print("=== PHASE 1: Generating all descriptions ===")
    for step in range(1, total_steps + 1):
        print(f"*** Generating description for step {step}/{total_steps}...")
        description = await handler.generate_description(step, total_steps)
        print(f"Step {step}/{total_steps}: {description}")
        print()
        
        # Store the state (without embedding for now)
        state = StateDescription(
            description=description, 
            step=step, 
            total_steps=total_steps
        )
        handler.states.append(state)
    
    print("✓ All descriptions generated successfully!")
    print()
    
    # PHASE 2: Get all embeddings in batch (load embedding model once)
    print("=== PHASE 2: Getting embeddings for all descriptions ===")
    print("Loading embedding model and processing all descriptions...")
    
    # Get embeddings for all descriptions in sequence
    for i, state in enumerate(handler.states):
        print(f"Getting embedding for step {state.step}/{state.total_steps}...")
        embedding = await get_embedding(state.description, handler.client)
        state.embedding = embedding
    
    print("✓ All embeddings generated successfully!")
    print()
    
    # PHASE 3: Generate visualization
    print("=== PHASE 3: Generating visualization ===")
    handler.plot_similarities()
    
    print("\nDone!")


if __name__ == "__main__":
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        print("\nOperation cancelled by user.")
    except Exception as e:
        print(f"An error occurred: {e}")
        sys.exit(1)