from dataclasses import dataclass, field
from typing import Optional, List
from enum import Enum

# ==========================================
# 1. METRICAL STRUCTURE
# Expresses the regular, hierarchical alternation of strong and weak beats.
# ==========================================
@dataclass(frozen=True)
class MetricalGrid:
    beat_index: int
    level: int  # Higher number = stronger beat (e.g., 4 = downbeat, 1 = 16th note)

@dataclass(frozen=True)
class MusicalEvent:
    root_pc: int
    quality: str
    bass_pc: int
    duration: float
    meter: MetricalGrid  


# ==========================================
# 2. GROUPING STRUCTURE
# Expresses a hierarchical segmentation into motives, phrases, and sections.
# ==========================================
@dataclass
class Group:
    level_name: str  # e.g., "motive", "phrase", "section"
    events: List[MusicalEvent] = field(default_factory=list)
    sub_groups: List['Group'] = field(default_factory=list)
    
    def get_all_events(self) -> List[MusicalEvent]:
        if not self.sub_groups:
            return self.events
        all_events = []
        for sg in self.sub_groups:
            all_events.extend(sg.get_all_events())
        return all_events


# ==========================================
# 3. TIME-SPAN REDUCTION
# Assigns structural importance, subordinating events to "heads".
# ==========================================
@dataclass
class TimeSpanNode:
    head: Optional[MusicalEvent] = None
    children: List['TimeSpanNode'] = field(default_factory=list)

def reduce_time_span(group: Group) -> TimeSpanNode:
    """
    Converts a Grouping Structure into a Time-Span Reduction Tree.
    (Simplified logic: normally evaluates preference rules to pick the 'head').
    """
    node = TimeSpanNode()
    if not group.sub_groups:
        # Base case: evaluate events in this smallest group to find the head
        # (In a full implementation, this uses TSRPR 1, 2a, 2b, etc.)
        node.head = group.events[0] # Stub: picking the first event as the head
        return node
    
    # Recursive case
    for sub_group in group.sub_groups:
        child_node = reduce_time_span(sub_group)
        node.children.append(child_node)
        
    # Stub: compare child_node heads to find the dominant head for this larger span
    node.head = node.children[0].head 
    return node


# ==========================================
# 4. PROLONGATIONAL REDUCTION
# Expresses tension/relaxation via right-branching (tensing) and left-branching (relaxing).
# ==========================================
class BranchType(Enum):
    RIGHT_TENSING = "Right-Branching (Tensing/Departure)"
    LEFT_RELAXING = "Left-Branching (Relaxing/Arrival)"
    STRONG_PROLONGATION = "Strong Prolongation (Static)"

@dataclass
class ProlongationalNode:
    event: MusicalEvent
    branch_type: Optional[BranchType] = None
    children: List['ProlongationalNode'] = field(default_factory=list)

def assign_prolongational_branching(tsr_node: TimeSpanNode, target_event: MusicalEvent) -> ProlongationalNode:
    """
    Takes the structurally important heads from the Time-Span Reduction
    and assigns tension/relaxation branches.
    """
    p_node = ProlongationalNode(event=target_event)
    
    # Example logic: If moving from I to V, it's a departure (Right/Tensing)
    # If moving from V to I, it's a resolution (Left/Relaxing)
    for child in tsr_node.children:
        if child.head:
            child_p_node = ProlongationalNode(event=child.head)
            
            # Simplified Tension/Relaxation logic
            if target_event.root_pc == 0 and child.head.root_pc == 7: # I to V
                child_p_node.branch_type = BranchType.RIGHT_TENSING
            elif target_event.root_pc == 7 and child.head.root_pc == 0: # V to I
                child_p_node.branch_type = BranchType.LEFT_RELAXING
            else:
                child_p_node.branch_type = BranchType.STRONG_PROLONGATION
                
            p_node.children.append(child_p_node)
            
    return p_node