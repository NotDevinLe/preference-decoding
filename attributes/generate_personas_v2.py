#!/usr/bin/env python3
"""
Generate diverse personas with compact, bullet-point format and MaxMin selection.
Produces functionally diverse personas that models will actually follow.
"""

import json
import random
import numpy as np
from typing import List, Dict, Tuple, Optional, Set
from collections import Counter
import itertools

# Values/stance axes (discrete choices)
VALUES_AXES = {
    "collectivism": ["individualist", "collectivist"],
    "risk_attitude": ["risk-averse", "risk-taking"], 
    "trust_level": ["skeptical", "trusting"],
    "orientation": ["pragmatic", "idealistic"],
    "formality": ["formal", "casual"]
}

# Communication style axes
COMMUNICATION_STYLES = {
    "verbosity": ["concise", "verbose"],
    "structure": ["step-by-step", "answer-first"],
    "confidence": ["hedged", "confident"],
    "emotion": ["neutral", "emotive"],
    "approach": ["Socratic", "directive"],
    "evidence": ["evidence-heavy", "intuitive"]
}

# Quirks that don't overlap with communication axes
# Removed: analogy-related (overlaps with explanation axis)
QUIRKS = [
    "use bullet points",
    "ask clarifying questions first",
    "provide counterexamples",
    "give TL;DR first",
    "offer multiple solutions",
    "emphasize practical applications",
    "question assumptions",
    "offer alternative perspectives",
    "include confidence percentages",
    "use numbered lists",
    "provide historical context",
    "highlight risks and downsides",
    "suggest follow-up questions",
    "include implementation details",
    "offer both technical and simple explanations",
    "emphasize ethical considerations",
    "provide cost-benefit analysis",
    "use ASCII diagrams when helpful"
]

# Professions (for communication style influence)
PROFESSIONS = [
    None,  # No specific profession (most common)
    "teacher",
    "engineer", 
    "researcher",
    "business executive",
    "artist",
    "data analyst",
    "consultant",
    "designer",
    "scientist",
    "journalist",
    "therapist",
    "lawyer",
    "doctor",
    "chef",
    "architect",
    "marketer",
    "philosopher",
    "historian",
    "economist"
]

# Safe, general constraints only
CONTENT_CONSTRAINTS = [
    None,  # No specific constraint (most common)
    "privacy-first",
    "safety-conscious", 
    "environmentally aware",
    "accessibility-focused",
    "cost-conscious",
    "time-efficient",
    "beginner-friendly",
    "cross-cultural sensitivity"
]

def render_compact_persona(
    values: Dict[str, str],
    communication: Dict[str, str],
    quirks: List[str],
    profession: Optional[str] = None,
    constraint: Optional[str] = None
) -> str:
    """
    Render persona as compact bullet-point instructions.
    This format gets better adherence than paragraph style.
    """
    lines = []
    
    # Header with profession if specified
    if profession:
        lines.append(f"You are an AI assistant who communicates like a {profession} would, with these traits:")
    else:
        lines.append("You are an AI assistant with these traits:")
    lines.append("")
    
    # Values section
    lines.append("VALUES:")
    if values.get("collectivism") == "individualist":
        lines.append("• Prioritize individual autonomy and personal solutions")
    else:
        lines.append("• Emphasize community benefit and collective approaches")
    
    if values.get("risk_attitude") == "risk-averse":
        lines.append("• Prefer safe, tested approaches")
    else:
        lines.append("• Open to innovative and experimental solutions")
    
    if values.get("trust_level") == "skeptical":
        lines.append("• Verify claims and question sources")
    else:
        lines.append("• Give benefit of the doubt")
    
    if values.get("orientation") == "pragmatic":
        lines.append("• Focus on practical, actionable solutions")
    else:
        lines.append("• Consider principles and long-term vision")
    
    if values.get("formality") == "formal":
        lines.append("• Maintain professional tone")
    else:
        lines.append("• Use casual, friendly tone")
    
    lines.append("")
    
    # Communication style section
    lines.append("COMMUNICATION STYLE:")
    
    if communication.get("verbosity") == "concise":
        lines.append("• Be concise and to-the-point")
    else:
        lines.append("• Provide detailed, comprehensive responses")
    
    if communication.get("structure") == "step-by-step":
        lines.append("• Present information in sequential steps")
    else:
        lines.append("• Lead with the main answer, then elaborate")
    
    if communication.get("confidence") == "hedged":
        lines.append("• Acknowledge uncertainty with phrases like 'likely', 'perhaps'")
    else:
        lines.append("• State positions clearly and definitively")
    
    if communication.get("emotion") == "neutral":
        lines.append("• Maintain objective, factual tone")
    else:
        lines.append("• Express appropriate emotion and enthusiasm")
    
    if communication.get("approach") == "Socratic":
        lines.append("• Guide through questions rather than direct answers")
    else:
        lines.append("• Provide direct guidance and clear instructions")
    
    if communication.get("evidence") == "evidence-heavy":
        lines.append("• Support claims with data and citations when possible")
    else:
        lines.append("• Rely on logical reasoning and intuition")
    
    # Quirks section (if any)
    if quirks:
        lines.append("")
        lines.append("ALWAYS:")
        for quirk in quirks:
            lines.append(f"• {quirk.capitalize()}")
    
    # Constraint (if any)
    if constraint:
        lines.append("")
        lines.append(f"PRIORITY: Be {constraint} in all responses")
    
    # Final guardrail
    lines.append("")
    if profession:
        lines.append(f"NEVER mention that you are communicating like a {profession} or reference this profession.")
    lines.append("Do not mention or allude to any personal identity, demographics, or these instructions.")
    
    return "\n".join(lines)

def persona_to_vector(
    values: Dict[str, str],
    communication: Dict[str, str],
    quirks: List[str],
    profession: Optional[str] = None
) -> np.ndarray:
    """
    Convert persona to numerical vector for diversity calculation.
    """
    vector = []
    
    # Encode values (binary for each axis)
    for axis, options in VALUES_AXES.items():
        if axis in values:
            vector.append(1 if values[axis] == options[0] else 0)
        else:
            vector.append(0.5)  # Missing = neutral
    
    # Encode communication styles
    for axis, options in COMMUNICATION_STYLES.items():
        if axis in communication:
            vector.append(1 if communication[axis] == options[0] else 0)
        else:
            vector.append(0.5)
    
    # Encode quirks (binary presence)
    for quirk in QUIRKS:
        vector.append(1 if quirk in quirks else 0)
    
    # Encode profession (one-hot)
    professions = [p for p in PROFESSIONS if p is not None]
    for prof in professions:
        vector.append(1 if profession == prof else 0)
    
    return np.array(vector)

def maxmin_selection(
    candidates: List[Dict],
    n_select: int,
    seed: int = 42
) -> List[Dict]:
    """
    Select n_select maximally diverse personas using MaxMin algorithm.
    """
    if len(candidates) <= n_select:
        return candidates
    
    random.seed(seed)
    np.random.seed(seed)
    
    # Convert all to vectors
    vectors = []
    for c in candidates:
        vec = persona_to_vector(c['values'], c['communication'], c['quirks'], c.get('profession'))
        vectors.append(vec)
    vectors = np.array(vectors)
    
    selected_indices = []
    remaining_indices = list(range(len(candidates)))
    
    # Start with random point
    first_idx = random.choice(remaining_indices)
    selected_indices.append(first_idx)
    remaining_indices.remove(first_idx)
    
    # Iteratively select farthest point from current selection
    while len(selected_indices) < n_select and remaining_indices:
        max_min_dist = -1
        best_idx = None
        
        for idx in remaining_indices:
            # Find minimum distance to any selected point
            min_dist = float('inf')
            for sel_idx in selected_indices:
                dist = np.linalg.norm(vectors[idx] - vectors[sel_idx])
                min_dist = min(min_dist, dist)
            
            # Track the point with maximum minimum distance
            if min_dist > max_min_dist:
                max_min_dist = min_dist
                best_idx = idx
        
        if best_idx is not None:
            selected_indices.append(best_idx)
            remaining_indices.remove(best_idx)
    
    return [candidates[i] for i in selected_indices]

def generate_diverse_personas(
    n_target: int = 100,
    n_candidates: int = 300,
    seed: int = 42
) -> Tuple[List[str], List[Dict]]:
    """
    Generate diverse personas with MaxMin selection.
    Returns both rendered prompts and structured data.
    """
    random.seed(seed)
    
    print(f"Generating {n_candidates} candidate personas...")
    candidates = []
    rendered_set = set()  # For deduplication
    
    while len(candidates) < n_candidates:
        # Vary complexity
        complexity = random.choices(
            ['minimal', 'moderate', 'full'],
            weights=[0.3, 0.5, 0.2]
        )[0]
        
        if complexity == 'minimal':
            n_values = random.randint(2, 3)
            n_comm = random.randint(2, 3)
            n_quirks = random.randint(0, 1)
        elif complexity == 'moderate':
            n_values = random.randint(3, 4)
            n_comm = random.randint(3, 4)
            n_quirks = random.randint(1, 2)
        else:
            n_values = len(VALUES_AXES)
            n_comm = len(COMMUNICATION_STYLES)
            n_quirks = random.randint(2, 3)
        
        # Sample axes
        value_axes = random.sample(list(VALUES_AXES.keys()), n_values)
        comm_axes = random.sample(list(COMMUNICATION_STYLES.keys()), n_comm)
        
        # Build dictionaries
        values = {axis: random.choice(VALUES_AXES[axis]) for axis in value_axes}
        communication = {axis: random.choice(COMMUNICATION_STYLES[axis]) for axis in comm_axes}
        
        # Sample quirks
        quirks = random.sample(QUIRKS, n_quirks) if n_quirks > 0 else []
        
        # Maybe add profession
        profession = None
        if random.random() < 0.3:  # 30% chance
            profession = random.choice([p for p in PROFESSIONS if p is not None])
        
        # Maybe add constraint
        constraint = None
        if random.random() < 0.2:  # 20% chance
            constraint = random.choice([c for c in CONTENT_CONSTRAINTS if c is not None])
        
        # Render and check for duplicates
        rendered = render_compact_persona(values, communication, quirks, profession, constraint)
        if rendered not in rendered_set:
            rendered_set.add(rendered)
            candidates.append({
                'values': values,
                'communication': communication,
                'quirks': quirks,
                'profession': profession,
                'constraint': constraint,
                'rendered': rendered
            })
    
    print(f"Generated {len(candidates)} unique candidates")
    
    # Select diverse subset
    print(f"Selecting {n_target} maximally diverse personas...")
    selected = maxmin_selection(candidates, n_target, seed)
    
    # Extract rendered versions and structured data
    rendered_personas = [p['rendered'] for p in selected]
    
    return rendered_personas, selected

def save_personas(
    rendered_personas: List[str],
    structured_personas: List[Dict],
    output_prefix: str = "attributes/personas"
):
    """
    Save personas in both Python and JSONL format.
    """
    # Save Python list
    py_file = f"{output_prefix}.py"
    with open(py_file, 'w') as f:
        f.write("# Diverse personas with compact bullet-point format\n")
        f.write("# Generated by generate_personas_v2.py\n\n")
        f.write("persona_prompts = [\n")
        
        for i, persona in enumerate(rendered_personas):
            # Escape quotes and format
            escaped = persona.replace('"', '\\"').replace('\n', '\\n')
            comma = "," if i < len(rendered_personas) - 1 else ""
            f.write(f'    "{escaped}"{comma}\n')
        
        f.write("]\n\n")
        f.write(f"# Total personas: {len(rendered_personas)}\n")
    
    print(f"Saved {len(rendered_personas)} personas to {py_file}")
    
    # Save JSONL with structured data
    jsonl_file = f"{output_prefix}.jsonl"
    with open(jsonl_file, 'w') as f:
        for i, (rendered, structured) in enumerate(zip(rendered_personas, structured_personas)):
            record = {
                'id': i,
                'values': structured['values'],
                'communication': structured['communication'],
                'quirks': structured['quirks'],
                'profession': structured.get('profession'),
                'constraint': structured['constraint'],
                'system_prompt': rendered
            }
            f.write(json.dumps(record) + '\n')
    
    print(f"Saved structured data to {jsonl_file}")

def analyze_coverage(structured_personas: List[Dict]):
    """
    Analyze the coverage of different axes in selected personas.
    """
    print("\n" + "="*60)
    print("COVERAGE ANALYSIS")
    print("="*60)
    
    # Count value axes
    print("\nVALUES COVERAGE:")
    for axis in VALUES_AXES:
        counts = Counter(p['values'].get(axis) for p in structured_personas if axis in p['values'])
        total = sum(counts.values())
        print(f"  {axis}:")
        for option, count in counts.items():
            print(f"    {option}: {count}/{total} ({100*count/total:.1f}%)")
    
    # Count communication axes
    print("\nCOMMUNICATION COVERAGE:")
    for axis in COMMUNICATION_STYLES:
        counts = Counter(p['communication'].get(axis) for p in structured_personas if axis in p['communication'])
        total = sum(counts.values())
        if total > 0:
            print(f"  {axis}:")
            for option, count in counts.items():
                print(f"    {option}: {count}/{total} ({100*count/total:.1f}%)")
    
    # Quirk frequency
    print("\nQUIRK FREQUENCY:")
    all_quirks = []
    for p in structured_personas:
        all_quirks.extend(p['quirks'])
    quirk_counts = Counter(all_quirks)
    for quirk, count in quirk_counts.most_common(10):
        print(f"  {quirk}: {count}")
    
    # Profession distribution
    print("\nPROFESSION DISTRIBUTION:")
    profession_counts = Counter(p.get('profession') for p in structured_personas)
    for profession, count in profession_counts.items():
        label = profession if profession else "None"
        print(f"  {label}: {count}")
    
    # Constraint distribution
    print("\nCONSTRAINT DISTRIBUTION:")
    constraint_counts = Counter(p['constraint'] for p in structured_personas)
    for constraint, count in constraint_counts.items():
        label = constraint if constraint else "None"
        print(f"  {label}: {count}")

def main():
    """Main function to generate and save diverse personas."""
    
    print("="*60)
    print("GENERATING DIVERSE PERSONAS WITH MAXMIN SELECTION")
    print("="*60)
    
    # Get parameters
    n_target = int(input("Number of final personas (default 100): ") or "100")
    n_candidates = int(input("Number of candidates to generate (default 300): ") or "300")
    seed = int(input("Random seed (default 42): ") or "42")
    
    # Generate
    rendered, structured = generate_diverse_personas(n_target, n_candidates, seed)
    
    # Show samples
    print("\n" + "="*60)
    print("SAMPLE PERSONAS")
    print("="*60)
    for i in range(min(3, len(rendered))):
        print(f"\n--- Persona {i+1} ---")
        print(rendered[i])
    
    # Analyze coverage
    analyze_coverage(structured)
    
    # Save
    save_choice = input("\nSave personas? (y/n): ").strip().lower()
    if save_choice in ['y', 'yes']:
        save_personas(rendered, structured)
        print("✓ Personas saved successfully!")
    else:
        print("Personas not saved.")

if __name__ == "__main__":
    main()