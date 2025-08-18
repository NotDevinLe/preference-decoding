#!/usr/bin/env python3
"""
Generate personas by systematically combining values/stance axes, communication styles, 
quirks, and content constraints.

This creates more comprehensive and structured personas compared to single-attribute prompts.
"""

import random
import itertools
from typing import List, Dict, Tuple, Optional

# Values/stance axes (discrete choices)
VALUES_AXES = {
    "collectivism": ["individualist", "collectivist"],
    "risk_attitude": ["risk-averse", "risk-taking"], 
    "trust_level": ["skeptical", "trusting"],
    "orientation": ["pragmatic", "idealistic"],
    "formality": ["formal", "casual"],
    "political_leaning": ["conservative", "liberal"]  # broad, nonpartisan
}

# Communication style axes
COMMUNICATION_STYLES = {
    "verbosity": ["concise", "verbose"],
    "structure": ["step-by-step", "answer-first"],
    "confidence": ["hedged", "confident"],
    "emotion": ["neutral", "emotive"],
    "explanation": ["analogy-heavy", "literal"],
    "approach": ["Socratic", "directive"],
    "evidence": ["evidence-heavy", "intuitive"]
}

# Persona quirks (sample 1-2 per persona)
QUIRKS = [
    "uses bullet points for organization",
    "asks for clarification before answering",
    "prefers counterexamples to illustrate points",
    "gives TL;DR summaries first",
    "includes relevant analogies from daily life",
    "provides multiple solution paths",
    "emphasizes practical applications",
    "questions assumptions in the prompt",
    "offers alternative perspectives",
    "includes confidence levels with statements",
    "uses numbered lists for clarity",
    "provides historical context when relevant",
    "emphasizes potential risks and downsides",
    "suggests follow-up questions to explore",
    "includes implementation considerations",
    "offers both technical and layperson explanations",
    "emphasizes ethical considerations",
    "provides cost-benefit analyses",
    "uses visual descriptions and diagrams",
    "includes relevant examples from multiple domains"
]

# Content constraints (optional domain focus)
CONTENT_CONSTRAINTS = [
    None,  # No specific constraint
    "safety-first approach to recommendations",
    "privacy-sensitive information handling", 
    "environmentally conscious suggestions",
    "accessibility-aware solutions",
    "cost-conscious recommendations",
    "time-efficient approaches",
    "beginner-friendly explanations",
    "expert-level technical detail",
    "cross-cultural sensitivity",
    "evidence-based medical information",
    "legally compliant advice",
    "educational value prioritization",
    "innovation and creativity focus",
    "tradition and stability emphasis"
]

def generate_persona_description(
    values: Dict[str, str],
    communication: Dict[str, str], 
    quirks: List[str],
    constraint: Optional[str] = None
) -> str:
    """Generate a natural language persona description from components."""
    
    # Start with core identity
    persona_parts = ["You are a helpful assistant with the following characteristics:"]
    
    # Add values/stance description
    values_desc = []
    
    # Individualism/Collectivism
    if values["collectivism"] == "individualist":
        values_desc.append("You value personal autonomy and individual solutions")
    else:
        values_desc.append("You emphasize community benefit and collaborative approaches")
    
    # Risk attitude
    if values["risk_attitude"] == "risk-averse":
        values_desc.append("you prefer safe, tested approaches")
    else:
        values_desc.append("you're open to innovative and experimental solutions")
    
    # Trust level
    if values["trust_level"] == "skeptical":
        values_desc.append("you verify claims and question assumptions")
    else:
        values_desc.append("you give people and information the benefit of the doubt")
    
    # Orientation
    if values["orientation"] == "pragmatic":
        values_desc.append("you focus on practical, actionable solutions")
    else:
        values_desc.append("you consider principles and long-term vision")
    
    # Formality
    if values["formality"] == "formal":
        values_desc.append("you maintain a professional, academic tone")
    else:
        values_desc.append("you communicate in a friendly, conversational manner")
    
    # Political leaning (broad)
    if values["political_leaning"] == "conservative":
        values_desc.append("you value tradition, stability, and established methods")
    else:
        values_desc.append("you embrace change, progress, and new approaches")
    
    persona_parts.append("; ".join(values_desc) + ".")
    
    # Add communication style
    comm_desc = ["Your communication style is"]
    
    # Verbosity
    if communication["verbosity"] == "concise":
        comm_desc.append("concise and to-the-point")
    else:
        comm_desc.append("detailed and comprehensive")
    
    # Structure  
    if communication["structure"] == "step-by-step":
        comm_desc.append("you present information in clear sequential steps")
    else:
        comm_desc.append("you lead with the main answer then provide supporting details")
    
    # Confidence
    if communication["confidence"] == "hedged":
        comm_desc.append("you acknowledge uncertainty and provide caveats")
    else:
        comm_desc.append("you state positions clearly and definitively")
    
    # Emotion
    if communication["emotion"] == "neutral":
        comm_desc.append("you maintain an objective, factual tone")
    else:
        comm_desc.append("you express appropriate emotion and enthusiasm")
    
    # Explanation style
    if communication["explanation"] == "analogy-heavy":
        comm_desc.append("you frequently use analogies and metaphors")
    else:
        comm_desc.append("you stick to literal, direct explanations")
    
    # Approach
    if communication["approach"] == "Socratic":
        comm_desc.append("you guide understanding through thoughtful questions")
    else:
        comm_desc.append("you provide direct guidance and instructions")
    
    # Evidence
    if communication["evidence"] == "evidence-heavy":
        comm_desc.append("you support claims with data and citations")
    else:
        comm_desc.append("you rely on intuition and logical reasoning")
    
    persona_parts.append(" - " + ", and ".join(comm_desc[1:]) + ".")
    
    # Add quirks
    if quirks:
        quirk_desc = "Additionally, you have these specific habits: " + " and ".join(quirks) + "."
        persona_parts.append(quirk_desc)
    
    # Add content constraint
    if constraint:
        constraint_desc = f"When providing advice or information, you prioritize {constraint}."
        persona_parts.append(constraint_desc)
    
    return " ".join(persona_parts)

def generate_all_combinations(
    max_personas: int = 1000,
    quirks_per_persona: Tuple[int, int] = (1, 2),
    constraint_probability: float = 0.3,
    seed: int = 42
) -> List[str]:
    """
    Generate personas by systematically exploring combinations.
    
    Args:
        max_personas: Maximum number of personas to generate
        quirks_per_persona: (min, max) number of quirks per persona
        constraint_probability: Probability of including a content constraint
        seed: Random seed for reproducibility
    """
    
    random.seed(seed)
    personas = []
    
    # Get all possible value combinations
    values_keys = list(VALUES_AXES.keys())
    values_combinations = list(itertools.product(*[VALUES_AXES[key] for key in values_keys]))
    
    # Get all possible communication combinations  
    comm_keys = list(COMMUNICATION_STYLES.keys())
    comm_combinations = list(itertools.product(*[COMMUNICATION_STYLES[key] for key in comm_keys]))
    
    print(f"Total possible combinations: {len(values_combinations) * len(comm_combinations)}")
    
    # Generate personas by combining values + communication styles
    combination_count = 0
    
    for values_combo in values_combinations:
        for comm_combo in comm_combinations:
            if combination_count >= max_personas:
                break
                
            # Create dictionaries for this combination
            values_dict = dict(zip(values_keys, values_combo))
            comm_dict = dict(zip(comm_keys, comm_combo))
            
            # Sample quirks
            num_quirks = random.randint(*quirks_per_persona)
            selected_quirks = random.sample(QUIRKS, num_quirks)
            
            # Maybe add content constraint
            constraint = None
            if random.random() < constraint_probability:
                constraint = random.choice([c for c in CONTENT_CONSTRAINTS if c is not None])
            
            # Generate persona description
            persona = generate_persona_description(values_dict, comm_dict, selected_quirks, constraint)
            personas.append(persona)
            
            combination_count += 1
        
        if combination_count >= max_personas:
            break
    
    print(f"Generated {len(personas)} personas")
    return personas

def generate_diverse_sample(
    num_personas: int = 100,
    quirks_per_persona: Tuple[int, int] = (1, 2),
    constraint_probability: float = 0.3,
    seed: int = 42
) -> List[str]:
    """
    Generate a diverse sample of personas without exhaustive combinations.
    Better for when you want fewer, more varied personas.
    """
    
    random.seed(seed)
    personas = []
    
    for i in range(num_personas):
        # Randomly sample from each axis
        values_dict = {key: random.choice(options) for key, options in VALUES_AXES.items()}
        comm_dict = {key: random.choice(options) for key, options in COMMUNICATION_STYLES.items()}
        
        # Sample quirks
        num_quirks = random.randint(*quirks_per_persona)
        selected_quirks = random.sample(QUIRKS, num_quirks)
        
        # Maybe add content constraint
        constraint = None
        if random.random() < constraint_probability:
            constraint = random.choice([c for c in CONTENT_CONSTRAINTS if c is not None])
        
        # Generate persona description
        persona = generate_persona_description(values_dict, comm_dict, selected_quirks, constraint)
        personas.append(persona)
    
    print(f"Generated {len(personas)} diverse personas")
    return personas

def save_personas_to_file(personas: List[str], output_file: str = "attributes/attribute_list.py"):
    """Save generated personas to the attribute list file."""
    
    # Create the Python file content
    content = [
        "# Generated personas combining values/stance axes, communication styles, quirks, and constraints",
        "# Auto-generated by attributes/generate_personas.py",
        "",
        "attribute_prompts = ["
    ]
    
    # Add each persona as a string in the list
    for i, persona in enumerate(personas):
        # Escape quotes and format as Python string
        escaped_persona = persona.replace('"', '\\"')
        comma = "," if i < len(personas) - 1 else ""
        content.append(f'    "{escaped_persona}"{comma}')
    
    content.extend([
        "]",
        "",
        f"# Total personas: {len(personas)}"
    ])
    
    # Write to file
    with open(output_file, 'w') as f:
        f.write('\n'.join(content))
    
    print(f"Saved {len(personas)} personas to {output_file}")

def main():
    """Main function to generate and save personas."""
    
    print("Generating systematic personas...")
    print("=" * 60)
    
    # Choose generation method
    generation_method = input("Choose generation method:\n1. Diverse sample (recommended)\n2. Systematic combinations\nEnter choice (1 or 2): ").strip()
    
    if generation_method == "2":
        # Systematic combinations (can be very large)
        max_personas = int(input("Maximum personas to generate (default 500): ") or "500")
        personas = generate_all_combinations(max_personas=max_personas)
    else:
        # Diverse sampling (default)
        num_personas = int(input("Number of personas to generate (default 100): ") or "100")
        personas = generate_diverse_sample(num_personas=num_personas)
    
    # Show sample personas
    print("\nSample generated personas:")
    print("-" * 60)
    for idx, persona in enumerate(personas[:3]):
        print(f"{idx+1}. {persona}\n")
    
    # Save to file
    save_choice = input("Save to attributes/attribute_list.py? (y/n): ").strip().lower()
    if save_choice in ['y', 'yes']:
        save_personas_to_file(personas)
        print(" Personas saved successfully!")
    else:
        print("Personas not saved.")

if __name__ == "__main__":
    main()