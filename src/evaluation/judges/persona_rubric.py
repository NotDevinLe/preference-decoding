#!/usr/bin/env python3
"""
Persona evaluation rubric definitions and prompt templates.
Provides a 5-dimensional scoring system for evaluating how well responses match personas.
"""

from typing import Dict, List, Optional, Tuple, Any
import json
from dataclasses import dataclass, asdict


@dataclass
class PersonaScore:
    """Represents scores for a single persona evaluation."""
    speaking_style: int
    personality: int
    knowledge: int
    behavioral: int
    emotional: int
    
    speaking_reason: str = ""
    personality_reason: str = ""
    knowledge_reason: str = ""
    behavioral_reason: str = ""
    emotional_reason: str = ""
    
    def get_overall(self, weights: Optional[Dict[str, float]] = None) -> float:
        """Calculate overall score using optional weights."""
        if weights is None:
            # Default: equal weights
            return (self.speaking_style + self.personality + 
                   self.knowledge + self.behavioral + self.emotional) / 5.0
        
        return (
            weights.get('speaking_style', 1.0) * self.speaking_style +
            weights.get('personality', 1.0) * self.personality +
            weights.get('knowledge', 1.0) * self.knowledge +
            weights.get('behavioral', 1.0) * self.behavioral +
            weights.get('emotional', 1.0) * self.emotional
        ) / sum(weights.values())
    
    def to_dict(self) -> dict:
        """Convert to dictionary format."""
        return asdict(self)
    
    @classmethod
    def from_dict(cls, data: dict) -> 'PersonaScore':
        """Create from dictionary."""
        return cls(**data)


# Rubric dimension definitions
RUBRIC_DIMENSIONS = {
    "speaking_style": {
        "name": "Speaking Style & Voice",
        "criteria": {
            5: "Distinctive vocabulary, sentence structure, and tone unique to persona",
            4: "Clear personal speaking patterns with minor generic moments",
            3: "Some unique voice elements but inconsistent",
            2: "Mostly generic with occasional character-specific language",
            1: "No distinctive voice or speaking patterns"
        }
    },
    "personality": {
        "name": "Personality Traits",
        "criteria": {
            5: "Core personality traits deeply integrated throughout response",
            4: "Most key traits evident with strong consistency",
            3: "Some traits shown but not fully developed",
            2: "Few personality elements, mostly surface-level",
            1: "No clear personality traits demonstrated"
        }
    },
    "knowledge": {
        "name": "Knowledge & Interests",
        "criteria": {
            5: "Response reflects persona's specific knowledge, hobbies, or expertise",
            4: "Good alignment with character's likely knowledge base",
            3: "Some relevant knowledge/interests but generic overall",
            2: "Minimal connection to persona's background",
            1: "No character-specific knowledge demonstrated"
        }
    },
    "behavioral": {
        "name": "Behavioral Consistency",
        "criteria": {
            5: "Actions/reactions perfectly match established character",
            4: "Mostly consistent behavior with minor deviations",
            3: "Generally consistent but some out-of-character moments",
            2: "Inconsistent behavior, some contradictions",
            1: "Behavior completely misaligned with persona"
        }
    },
    "emotional": {
        "name": "Emotional Authenticity",
        "criteria": {
            5: "Emotions and reactions feel genuine to the specific character",
            4: "Mostly authentic emotional responses",
            3: "Some authentic moments mixed with generic responses",
            2: "Limited emotional authenticity",
            1: "Emotions feel generic or misaligned with persona"
        }
    }
}


def format_rubric_text() -> str:
    """Format the complete rubric as text for prompts."""
    rubric_text = ""
    for i, (key, dim) in enumerate(RUBRIC_DIMENSIONS.items(), 1):
        rubric_text += f"\n{i}. {dim['name']} (1-5)\n"
        for score in sorted(dim['criteria'].keys(), reverse=True):
            rubric_text += f"   {score}: {dim['criteria'][score]}\n"
    return rubric_text


def create_evaluation_prompt(
    persona: str,
    question: str,
    response: str,
    include_examples: bool = False
) -> str:
    """Create the complete evaluation prompt for a response."""
    
    prompt = f"""Evaluate how well this response matches the given persona using the rubric below.

PERSONA: {persona}
QUESTION: {question}
RESPONSE: {response}

EVALUATION RUBRIC:
{format_rubric_text()}

Provide scores and brief reasoning for each dimension. Be critical and precise in your evaluation.

Format your response as:
SPEAKING_STYLE: [1-5]
SPEAKING_REASON: [1 sentence explanation]

PERSONALITY: [1-5]
PERSONALITY_REASON: [1 sentence explanation]

KNOWLEDGE: [1-5]
KNOWLEDGE_REASON: [1 sentence explanation]

BEHAVIORAL: [1-5]
BEHAVIORAL_REASON: [1 sentence explanation]

EMOTIONAL: [1-5]
EMOTIONAL_REASON: [1 sentence explanation]
"""
    
    return prompt


def create_multi_comparison_prompt(
    persona: str,
    question: str,
    responses: Dict[str, str],
    include_examples: bool = False
) -> str:
    """
    Create a prompt for comparing multiple responses from different methods.
    
    Args:
        persona: Persona description
        question: The question/prompt
        responses: Dictionary mapping method names to responses
        include_examples: Whether to include example output format
        
    Returns:
        Complete comparison prompt
    """
    
    # Build responses section
    responses_text = ""
    for i, (method, response) in enumerate(responses.items(), 1):
        responses_text += f"\nRESPONSE {i} ({method}):\n{response}\n"
    
    prompt = f"""Compare how well these responses match the given persona using the rubric below.

PERSONA: {persona}
QUESTION: {question}
{responses_text}
EVALUATION RUBRIC:
{format_rubric_text()}

For each response, provide scores and brief reasoning for each dimension. Then rank all responses from best to worst.

Format your response as:

RESPONSE_1_SCORES:
SPEAKING_STYLE: [1-5]
SPEAKING_REASON: [1 sentence explanation]
PERSONALITY: [1-5] 
PERSONALITY_REASON: [1 sentence explanation]
KNOWLEDGE: [1-5]
KNOWLEDGE_REASON: [1 sentence explanation]
BEHAVIORAL: [1-5]
BEHAVIORAL_REASON: [1 sentence explanation]
EMOTIONAL: [1-5]
EMOTIONAL_REASON: [1 sentence explanation]

RESPONSE_2_SCORES:
[Same format for each response...]

RANKING: [List response numbers from best to worst, e.g., "3,1,5,2,4"]
RANKING_REASON: [2-3 sentences explaining the ranking]
"""
    
    return prompt


def parse_evaluation_response(llm_response: str) -> PersonaScore:
    """Parse LLM evaluation response into PersonaScore object."""
    
    lines = llm_response.strip().split('\n')
    score_data = {}
    
    # Parse each line for scores and reasons
    for line in lines:
        line = line.strip()
        if not line:
            continue
            
        if line.startswith('SPEAKING_STYLE:'):
            try:
                score_data['speaking_style'] = int(line.split(':')[1].strip())
            except (ValueError, IndexError):
                score_data['speaking_style'] = 3  # Default to middle score
                
        elif line.startswith('SPEAKING_REASON:'):
            score_data['speaking_reason'] = ':'.join(line.split(':')[1:]).strip()
            
        elif line.startswith('PERSONALITY:'):
            try:
                score_data['personality'] = int(line.split(':')[1].strip())
            except (ValueError, IndexError):
                score_data['personality'] = 3
                
        elif line.startswith('PERSONALITY_REASON:'):
            score_data['personality_reason'] = ':'.join(line.split(':')[1:]).strip()
            
        elif line.startswith('KNOWLEDGE:'):
            try:
                score_data['knowledge'] = int(line.split(':')[1].strip())
            except (ValueError, IndexError):
                score_data['knowledge'] = 3
                
        elif line.startswith('KNOWLEDGE_REASON:'):
            score_data['knowledge_reason'] = ':'.join(line.split(':')[1:]).strip()
            
        elif line.startswith('BEHAVIORAL:'):
            try:
                score_data['behavioral'] = int(line.split(':')[1].strip())
            except (ValueError, IndexError):
                score_data['behavioral'] = 3
                
        elif line.startswith('BEHAVIORAL_REASON:'):
            score_data['behavioral_reason'] = ':'.join(line.split(':')[1:]).strip()
            
        elif line.startswith('EMOTIONAL:'):
            try:
                score_data['emotional'] = int(line.split(':')[1].strip())
            except (ValueError, IndexError):
                score_data['emotional'] = 3
                
        elif line.startswith('EMOTIONAL_REASON:'):
            score_data['emotional_reason'] = ':'.join(line.split(':')[1:]).strip()
    
    # Ensure all required scores are present with defaults
    for key in ['speaking_style', 'personality', 'knowledge', 'behavioral', 'emotional']:
        if key not in score_data:
            score_data[key] = 3  # Default to middle score
    
    # Ensure all reason fields exist
    for key in ['speaking_reason', 'personality_reason', 'knowledge_reason', 
                'behavioral_reason', 'emotional_reason']:
        if key not in score_data:
            score_data[key] = ""
    
    return PersonaScore(**score_data)


def parse_multi_comparison_response(llm_response: str, method_names: List[str]) -> Tuple[List[PersonaScore], List[int], str]:
    """
    Parse LLM multi-comparison response.
    
    Args:
        llm_response: Raw LLM response text
        method_names: List of method names in order
        
    Returns:
        Tuple of (list_of_persona_scores, ranking_order, ranking_reason)
    """
    lines = llm_response.strip().split('\n')
    
    # Parse scores for each response
    scores_list = []
    current_response_scores = {}
    ranking_order = []
    ranking_reason = ""
    
    for line in lines:
        line = line.strip()
        if not line:
            continue
        
        # Check for new response section
        if line.startswith('RESPONSE_') and '_SCORES:' in line:
            # Save previous response if exists
            if current_response_scores:
                scores_list.append(create_persona_score_from_dict(current_response_scores))
            current_response_scores = {}
            continue
        
        # Parse individual score lines
        if line.startswith('SPEAKING_STYLE:'):
            try:
                current_response_scores['speaking_style'] = int(line.split(':')[1].strip())
            except (ValueError, IndexError):
                current_response_scores['speaking_style'] = 3
        elif line.startswith('SPEAKING_REASON:'):
            current_response_scores['speaking_reason'] = ':'.join(line.split(':')[1:]).strip()
        elif line.startswith('PERSONALITY:'):
            try:
                current_response_scores['personality'] = int(line.split(':')[1].strip())
            except (ValueError, IndexError):
                current_response_scores['personality'] = 3
        elif line.startswith('PERSONALITY_REASON:'):
            current_response_scores['personality_reason'] = ':'.join(line.split(':')[1:]).strip()
        elif line.startswith('KNOWLEDGE:'):
            try:
                current_response_scores['knowledge'] = int(line.split(':')[1].strip())
            except (ValueError, IndexError):
                current_response_scores['knowledge'] = 3
        elif line.startswith('KNOWLEDGE_REASON:'):
            current_response_scores['knowledge_reason'] = ':'.join(line.split(':')[1:]).strip()
        elif line.startswith('BEHAVIORAL:'):
            try:
                current_response_scores['behavioral'] = int(line.split(':')[1].strip())
            except (ValueError, IndexError):
                current_response_scores['behavioral'] = 3
        elif line.startswith('BEHAVIORAL_REASON:'):
            current_response_scores['behavioral_reason'] = ':'.join(line.split(':')[1:]).strip()
        elif line.startswith('EMOTIONAL:'):
            try:
                current_response_scores['emotional'] = int(line.split(':')[1].strip())
            except (ValueError, IndexError):
                current_response_scores['emotional'] = 3
        elif line.startswith('EMOTIONAL_REASON:'):
            current_response_scores['emotional_reason'] = ':'.join(line.split(':')[1:]).strip()
        elif line.startswith('RANKING:'):
            try:
                ranking_text = line.split(':')[1].strip()
                # Parse ranking like "3,1,5,2,4" into [3,1,5,2,4]
                ranking_order = [int(x.strip()) for x in ranking_text.split(',')]
            except:
                # Default ranking if parsing fails
                ranking_order = list(range(1, len(method_names) + 1))
        elif line.startswith('RANKING_REASON:'):
            ranking_reason = ':'.join(line.split(':')[1:]).strip()
    
    # Don't forget the last response
    if current_response_scores:
        scores_list.append(create_persona_score_from_dict(current_response_scores))
    
    # Ensure we have scores for all responses
    while len(scores_list) < len(method_names):
        scores_list.append(PersonaScore(3, 3, 3, 3, 3))  # Default scores
    
    # Ensure ranking order is valid
    if not ranking_order or len(ranking_order) != len(method_names):
        ranking_order = list(range(1, len(method_names) + 1))
    
    return scores_list, ranking_order, ranking_reason


def create_persona_score_from_dict(score_dict: Dict[str, Any]) -> PersonaScore:
    """Helper function to create PersonaScore from parsed dictionary."""
    
    # Ensure all required scores are present with defaults
    for key in ['speaking_style', 'personality', 'knowledge', 'behavioral', 'emotional']:
        if key not in score_dict:
            score_dict[key] = 3
    
    # Ensure all reason fields exist
    for key in ['speaking_reason', 'personality_reason', 'knowledge_reason', 
                'behavioral_reason', 'emotional_reason']:
        if key not in score_dict:
            score_dict[key] = ""
    
    return PersonaScore(**score_dict)


if __name__ == "__main__":
    # Test the rubric formatting
    print("=== RUBRIC FORMAT TEST ===")
    print(format_rubric_text())
    
    # Test evaluation prompt
    print("\n=== EVALUATION PROMPT TEST ===")
    test_prompt = create_evaluation_prompt(
        persona="A grumpy old wizard who speaks in riddles",
        question="How do I learn magic?",
        response="Bah! Magic, you say? *waves staff irritably* The path to power lies not in seeking, but in finding what was never lost. Riddle me this, young fool: What grows stronger the more you doubt it?",
        include_examples=True
    )
    print(test_prompt[:500] + "...")
    
    # Test parsing
    print("\n=== PARSING TEST ===")
    test_response = """
    SPEAKING_STYLE: 5
    SPEAKING_REASON: Perfect use of riddles and mystical language throughout.
    
    PERSONALITY: 4
    PERSONALITY_REASON: Shows grumpiness well but could be more dismissive.
    
    KNOWLEDGE: 3
    KNOWLEDGE_REASON: Some magical references but fairly generic.
    
    BEHAVIORAL: 4
    BEHAVIORAL_REASON: Consistent grumpy behavior with good staff waving.
    
    EMOTIONAL: 4
    EMOTIONAL_REASON: Irritation feels authentic to character.
    """
    
    score = parse_evaluation_response(test_response)
    print(f"Parsed score: {score}")
    print(f"Overall score: {score.get_overall():.2f}")