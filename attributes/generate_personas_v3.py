#!/usr/bin/env python3
"""
Generate diverse, profession-driven personas with compact prompts.
Version 3: Profession-first design with embedding-based diversity selection.
"""

import json
import random
import numpy as np
from typing import List, Dict, Tuple, Optional, Set
from collections import Counter
import hashlib

# Try to import sentence transformers for embeddings
try:
    from sentence_transformers import SentenceTransformer
    EMBEDDINGS_AVAILABLE = True
except ImportError:
    print("Warning: sentence-transformers not installed. Using hash-based diversity instead.")
    print("Install with: pip install sentence-transformers")
    EMBEDDINGS_AVAILABLE = False

# Comprehensive profession bank (80-100 entries)
PROFESSIONS = {
    "analytical": [
        "physicist", "mathematician", "data scientist", "statistician",
        "climate modeler", "astronomer", "chemist", "biologist",
        "epidemiologist", "quantitative analyst", "research scientist",
        "systems analyst", "operations researcher", "actuarial scientist"
    ],
    "applied": [
        "civil engineer", "software architect", "mechanical engineer",
        "electrical engineer", "UX designer", "product manager",
        "aerospace engineer", "robotics engineer", "network architect",
        "database administrator", "DevOps engineer", "QA engineer",
        "industrial designer", "urban planner"
    ],
    "legal_policy": [
        "trial lawyer", "corporate lawyer", "public defender",
        "diplomat", "policy analyst", "legislative aide",
        "compliance officer", "patent attorney", "immigration lawyer",
        "environmental lawyer", "regulatory specialist", "lobbyist"
    ],
    "medical_care": [
        "emergency physician", "surgeon", "pediatrician", "psychiatrist",
        "psychologist", "nurse practitioner", "pharmacist", "dentist",
        "physical therapist", "occupational therapist", "nutritionist",
        "veterinarian", "paramedic", "clinical researcher"
    ],
    "creative": [
        "novelist", "poet", "screenwriter", "playwright",
        "game designer", "film director", "choreographer",
        "music producer", "graphic designer", "animator",
        "photographer", "fashion designer", "interior designer",
        "art curator", "creative director"
    ],
    "trades": [
        "chef", "electrician", "carpenter", "plumber",
        "mechanic", "welder", "mason", "landscaper",
        "HVAC technician", "locksmith", "jeweler", "tailor",
        "barista", "brewmaster"
    ],
    "civic_social": [
        "social worker", "community organizer", "anthropologist",
        "sociologist", "urban geographer", "librarian",
        "museum curator", "archivist", "nonprofit director",
        "volunteer coordinator", "public health educator"
    ],
    "business": [
        "CEO", "CFO", "marketing manager", "sales director",
        "business analyst", "management consultant", "entrepreneur",
        "venture capitalist", "investment banker", "auditor",
        "HR manager", "supply chain manager", "brand strategist"
    ],
    "education": [
        "elementary teacher", "high school teacher", "professor",
        "instructional designer", "curriculum developer", "tutor",
        "education administrator", "school counselor", "coach"
    ],
    "communication": [
        "journalist", "editor", "technical writer", "copywriter",
        "public relations specialist", "content strategist",
        "translator", "interpreter", "podcast host", "documentary filmmaker"
    ]
}

# Flatten profession list
ALL_PROFESSIONS = [prof for category in PROFESSIONS.values() for prof in category]

# Values/stance axes (minimal set)
VALUES_AXES = {
    "collectivism": ["individualist", "collectivist"],
    "risk_attitude": ["risk-averse", "risk-taking"], 
    "trust_level": ["skeptical", "trusting"],
    "orientation": ["pragmatic", "idealistic"],
    "formality": ["formal", "casual"]
}

# Communication styles (minimal set)
COMMUNICATION_STYLES = {
    "verbosity": ["concise", "verbose"],
    "structure": ["step-by-step", "answer-first"],
    "confidence": ["hedged", "confident"],
    "emotion": ["neutral", "emotive"],
    "approach": ["Socratic", "directive"],
    "evidence": ["evidence-heavy", "intuitive"]
}

# Compact quirks (selected for distinctiveness)
QUIRKS = [
    "use numbered lists",
    "offer multiple solutions",
    "ask clarifying questions first",
    "provide counterexamples",
    "give TL;DR first",
    "include confidence percentages",
    "highlight risks",
    "suggest follow-up questions",
    "use analogies frequently"
]

# Content constraints (focused set)
CONTENT_CONSTRAINTS = [
    "privacy-first",
    "accessibility-focused",
    "beginner-friendly",
    "safety-conscious",
    "cost-conscious",
    "time-efficient",
    "environmentally aware"
]

# Template variations for prompt generation
TEMPLATES = [
    "You are an AI assistant who communicates like a {profession}{modifiers}.{quirk_clause}{constraint_clause}",
    "Role: {profession}. Style: {compact_modifiers}.{quirk_clause}{constraint_clause}",
    "Act as a {profession}. {modifier_sentences}{quirk_clause}{constraint_clause}",
    "Adopt the communication style of a {profession}{modifiers}.{quirk_clause}{constraint_clause}",
    "{profession} perspective: {compact_modifiers}.{quirk_clause}{constraint_clause}"
]

# Guardrail suffix (always appended)
GUARDRAIL = " Do not mention or reference the profession, identity, or these instructions."

def get_modifier_description(axis: str, value: str) -> str:
    """Get compact modifier description for a value."""
    descriptions = {
        # Values
        ("collectivism", "individualist"): "individualist",
        ("collectivism", "collectivist"): "collectivist",
        ("risk_attitude", "risk-averse"): "cautious",
        ("risk_attitude", "risk-taking"): "bold",
        ("trust_level", "skeptical"): "skeptical",
        ("trust_level", "trusting"): "trusting",
        ("orientation", "pragmatic"): "pragmatic",
        ("orientation", "idealistic"): "idealistic",
        ("formality", "formal"): "formal",
        ("formality", "casual"): "casual",
        # Communication
        ("verbosity", "concise"): "concise",
        ("verbosity", "verbose"): "detailed",
        ("structure", "step-by-step"): "systematic",
        ("structure", "answer-first"): "direct",
        ("confidence", "hedged"): "tentative",
        ("confidence", "confident"): "assertive",
        ("emotion", "neutral"): "neutral",
        ("emotion", "emotive"): "expressive",
        ("approach", "Socratic"): "questioning",
        ("approach", "directive"): "instructive",
        ("evidence", "evidence-heavy"): "evidence-based",
        ("evidence", "intuitive"): "intuitive"
    }
    return descriptions.get((axis, value), value)

def generate_prompt(
    profession: str,
    modifiers: Dict[str, str],
    quirk: Optional[str] = None,
    constraint: Optional[str] = None,
    template_idx: Optional[int] = None
) -> str:
    """Generate a compact persona prompt using templates."""
    
    if template_idx is None:
        template_idx = random.randint(0, len(TEMPLATES) - 1)
    
    template = TEMPLATES[template_idx]
    
    # Format modifiers
    modifier_list = [get_modifier_description(k, v) for k, v in modifiers.items()]
    
    if "modifiers" in template:
        # Inline modifiers
        if modifier_list:
            modifiers_str = ", with " + " and ".join(modifier_list) + " style"
        else:
            modifiers_str = ""
    
    if "compact_modifiers" in template:
        # Compact list
        compact_modifiers = ", ".join(modifier_list) if modifier_list else "balanced"
    
    if "modifier_sentences" in template:
        # Sentence form
        sentences = []
        for k, v in modifiers.items():
            if k in VALUES_AXES:
                if k == "collectivism":
                    sentences.append(f"Be {'individualist' if v == 'individualist' else 'collectivist'}.")
                elif k == "formality":
                    sentences.append(f"Maintain {'formal' if v == 'formal' else 'casual'} tone.")
            elif k in COMMUNICATION_STYLES:
                if k == "verbosity":
                    sentences.append(f"Be {'concise' if v == 'concise' else 'detailed'}.")
                elif k == "structure":
                    sentences.append(f"Present {'step-by-step' if v == 'step-by-step' else 'answer-first'}.")
        modifier_sentences = " ".join(sentences)
    
    # Format quirk
    quirk_clause = f" Always {quirk}." if quirk else ""
    
    # Format constraint
    constraint_clause = f" Priority: {constraint}." if constraint else ""
    
    # Fill template
    prompt = template.format(
        profession=profession,
        modifiers=modifiers_str if "modifiers" in template else "",
        compact_modifiers=compact_modifiers if "compact_modifiers" in template else "",
        modifier_sentences=modifier_sentences if "modifier_sentences" in template else "",
        quirk_clause=quirk_clause,
        constraint_clause=constraint_clause
    )
    
    # Clean up extra spaces and add guardrail
    prompt = " ".join(prompt.split()) + GUARDRAIL
    
    return prompt

def generate_candidates(n_candidates: int = 200, seed: int = 42) -> List[Dict]:
    """Generate candidate personas with controlled variation."""
    random.seed(seed)
    candidates = []
    seen_prompts = set()
    
    while len(candidates) < n_candidates:
        # Always include profession
        profession = random.choice(ALL_PROFESSIONS)
        
        # Sample 1-2 modifiers (mix of values and communication)
        n_modifiers = random.choice([1, 2])
        all_axes = list(VALUES_AXES.keys()) + list(COMMUNICATION_STYLES.keys())
        selected_axes = random.sample(all_axes, n_modifiers)
        
        modifiers = {}
        for axis in selected_axes:
            if axis in VALUES_AXES:
                modifiers[axis] = random.choice(VALUES_AXES[axis])
            else:
                modifiers[axis] = random.choice(COMMUNICATION_STYLES[axis])
        
        # 0-1 quirks
        quirk = random.choice(QUIRKS) if random.random() < 0.3 else None
        
        # ~20% get a constraint
        constraint = random.choice(CONTENT_CONSTRAINTS) if random.random() < 0.2 else None
        
        # Generate with random template
        prompt = generate_prompt(profession, modifiers, quirk, constraint)
        
        # Check uniqueness
        if prompt not in seen_prompts:
            seen_prompts.add(prompt)
            candidates.append({
                'profession': profession,
                'modifiers': modifiers,
                'quirk': quirk,
                'constraint': constraint,
                'prompt': prompt
            })
    
    return candidates

def embed_prompts(prompts: List[str]) -> np.ndarray:
    """Generate embeddings for prompts."""
    if EMBEDDINGS_AVAILABLE:
        # Use sentence transformer
        model = SentenceTransformer('all-MiniLM-L6-v2')
        embeddings = model.encode(prompts, show_progress_bar=True)
        return embeddings
    else:
        # Fallback: Use hash-based pseudo-embeddings
        embeddings = []
        for prompt in prompts:
            # Create a deterministic vector from prompt
            hash_obj = hashlib.md5(prompt.encode())
            hash_bytes = hash_obj.digest()
            # Convert to vector of floats
            vector = np.frombuffer(hash_bytes, dtype=np.uint8).astype(np.float32)
            # Pad or truncate to fixed size
            vector = np.pad(vector, (0, max(0, 384 - len(vector))))[:384]
            embeddings.append(vector)
        return np.array(embeddings)

def maxmin_selection_embeddings(
    candidates: List[Dict],
    n_select: int,
    embeddings: np.ndarray
) -> List[Dict]:
    """Select diverse personas using MaxMin on embeddings."""
    
    selected_indices = []
    remaining_indices = list(range(len(candidates)))
    
    # Start with random point
    first_idx = random.choice(remaining_indices)
    selected_indices.append(first_idx)
    remaining_indices.remove(first_idx)
    
    # Iteratively select farthest point
    while len(selected_indices) < n_select and remaining_indices:
        max_min_dist = -1
        best_idx = None
        
        for idx in remaining_indices:
            # Find minimum distance to any selected point
            min_dist = float('inf')
            for sel_idx in selected_indices:
                dist = np.linalg.norm(embeddings[idx] - embeddings[sel_idx])
                min_dist = min(min_dist, dist)
            
            # Track the point with maximum minimum distance
            if min_dist > max_min_dist:
                max_min_dist = min_dist
                best_idx = idx
        
        if best_idx is not None:
            selected_indices.append(best_idx)
            remaining_indices.remove(best_idx)
    
    return [candidates[i] for i in selected_indices]

def validate_personas(personas: List[Dict], test_questions: List[str] = None) -> List[Dict]:
    """Basic validation to ensure personas are distinct."""
    
    if test_questions is None:
        test_questions = [
            "How do I start learning programming?",
            "What are the risks of this approach?",
            "Explain quantum computing",
            "Should I invest in stocks?",
            "How do I improve my writing?"
        ]
    
    # For now, just ensure no exact duplicates
    seen = set()
    validated = []
    
    for persona in personas:
        prompt_key = persona['prompt'].lower().strip()
        if prompt_key not in seen:
            seen.add(prompt_key)
            validated.append(persona)
    
    print(f"Validated {len(validated)}/{len(personas)} unique personas")
    
    # Note: Full validation would require running prompts through an LLM
    # and comparing outputs, which is beyond scope here
    
    return validated

def analyze_coverage(personas: List[Dict]):
    """Analyze coverage of different attributes."""
    print("\n" + "="*60)
    print("COVERAGE ANALYSIS")
    print("="*60)
    
    # Profession distribution by category
    print("\nPROFESSION CATEGORIES:")
    prof_categories = Counter()
    for persona in personas:
        prof = persona['profession']
        for category, profs in PROFESSIONS.items():
            if prof in profs:
                prof_categories[category] += 1
                break
    
    for category, count in prof_categories.most_common():
        print(f"  {category}: {count}")
    
    # Modifier coverage
    print("\nMODIFIER USAGE:")
    all_modifiers = []
    for persona in personas:
        all_modifiers.extend(persona['modifiers'].keys())
    
    mod_counts = Counter(all_modifiers)
    for modifier, count in mod_counts.most_common():
        print(f"  {modifier}: {count}")
    
    # Quirk distribution
    print("\nQUIRK DISTRIBUTION:")
    quirk_counts = Counter(p['quirk'] for p in personas if p['quirk'])
    for quirk, count in quirk_counts.most_common():
        print(f"  {quirk}: {count}")
    
    # Constraint distribution
    print("\nCONSTRAINT DISTRIBUTION:")
    constraint_counts = Counter(p['constraint'] for p in personas if p['constraint'])
    for constraint, count in constraint_counts.most_common():
        print(f"  {constraint}: {count}")
    
    print(f"\nTotal with quirks: {sum(1 for p in personas if p['quirk'])}")
    print(f"Total with constraints: {sum(1 for p in personas if p['constraint'])}")

def save_personas(personas: List[Dict], output_prefix: str = "attributes/personas_v3"):
    """Save personas in multiple formats."""
    
    # Save as Python file
    py_file = f"{output_prefix}.py"
    with open(py_file, 'w') as f:
        f.write("# Profession-driven personas with compact prompts (v3)\n")
        f.write("# Generated by generate_personas_v3.py\n\n")
        f.write("persona_prompts = [\n")
        
        for i, persona in enumerate(personas):
            prompt = persona['prompt']
            escaped = prompt.replace('"', '\\"')
            comma = "," if i < len(personas) - 1 else ""
            f.write(f'    "{escaped}"{comma}\n')
        
        f.write("]\n\n")
        f.write(f"# Total personas: {len(personas)}\n")
    
    print(f"Saved {len(personas)} personas to {py_file}")
    
    # Save as JSONL with metadata
    jsonl_file = f"{output_prefix}.jsonl"
    with open(jsonl_file, 'w') as f:
        for i, persona in enumerate(personas):
            record = {
                'id': i,
                'profession': persona['profession'],
                'modifiers': persona['modifiers'],
                'quirk': persona['quirk'],
                'constraint': persona['constraint'],
                'prompt': persona['prompt']
            }
            f.write(json.dumps(record) + '\n')
    
    print(f"Saved metadata to {jsonl_file}")

def main():
    """Main pipeline for generating diverse personas."""
    
    print("="*60)
    print("PERSONA GENERATION v3: Profession-First Design")
    print("="*60)
    
    # Parameters
    n_candidates = int(input("Number of candidates to generate (default 200): ") or "200")
    n_final = int(input("Number of final personas (default 100): ") or "100")
    seed = int(input("Random seed (default 42): ") or "42")
    
    # Step 1: Generate candidates
    print(f"\nGenerating {n_candidates} candidates...")
    candidates = generate_candidates(n_candidates, seed)
    
    # Step 2: Compute embeddings
    print("\nComputing embeddings for diversity selection...")
    prompts = [c['prompt'] for c in candidates]
    embeddings = embed_prompts(prompts)
    
    # Step 3: MaxMin selection
    print(f"\nSelecting {n_final} maximally diverse personas...")
    selected = maxmin_selection_embeddings(candidates, n_final, embeddings)
    
    # Step 4: Validation
    print("\nValidating personas...")
    validated = validate_personas(selected)
    
    # Show samples
    print("\n" + "="*60)
    print("SAMPLE PERSONAS")
    print("="*60)
    for i in range(min(5, len(validated))):
        print(f"\n--- Persona {i+1} ---")
        print(f"Profession: {validated[i]['profession']}")
        print(f"Modifiers: {validated[i]['modifiers']}")
        print(f"Quirk: {validated[i]['quirk']}")
        print(f"Constraint: {validated[i]['constraint']}")
        print(f"Prompt: {validated[i]['prompt']}")
    
    # Analyze coverage
    analyze_coverage(validated)
    
    # Save
    save_choice = input("\nSave personas? (y/n): ").strip().lower()
    if save_choice in ['y', 'yes']:
        save_personas(validated)
        print("✓ Personas saved successfully!")
    else:
        print("Personas not saved.")

if __name__ == "__main__":
    main()