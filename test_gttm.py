import unittest
from gttm_features import (
    MetricalGrid,
    MusicalEvent,
    Group,
    TimeSpanNode,
    reduce_time_span,
    BranchType,
    ProlongationalNode,
    assign_prolongational_branching
)

class TestGTTMCoreComponents(unittest.TestCase):

    def setUp(self):
        """
        Set up the musical surface before each test.
        1. METRICAL STRUCTURE: Define the metrical grid.
        """
        self.strong_downbeat = MetricalGrid(beat_index=0, level=4)
        self.weak_beat = MetricalGrid(beat_index=1, level=1)
        self.medium_beat = MetricalGrid(beat_index=2, level=2)

        # Define Musical Events linked to the metrical grid
        self.chord_I = MusicalEvent(
            root_pc=0, quality="maj", bass_pc=0, duration=2.0, meter=self.strong_downbeat
        )
        self.chord_V = MusicalEvent(
            root_pc=7, quality="dom7", bass_pc=7, duration=1.0, meter=self.weak_beat
        )
        self.chord_I_end = MusicalEvent(
            root_pc=0, quality="maj", bass_pc=0, duration=2.0, meter=self.medium_beat
        )

    def test_grouping_structure(self):
        """
        2. GROUPING STRUCTURE: Check that hierarchical grouping properly 
        collects all events across motives and phrases.
        """
        motive1 = Group(level_name="motive", events=[self.chord_I, self.chord_V])
        motive2 = Group(level_name="motive", events=[self.chord_I_end])
        phrase = Group(level_name="phrase", sub_groups=[motive1, motive2])

        all_events = phrase.get_all_events()
        self.assertEqual(len(all_events), 3, "The phrase should contain exactly 3 events.")
        self.assertEqual(all_events[0].root_pc, 0, "First event should be the I chord.")
        self.assertEqual(all_events[1].root_pc, 7, "Second event should be the V chord.")

    def test_time_span_reduction(self):
        """
        3. TIME-SPAN REDUCTION: Check that the reduction builds a tree 
        directly from the grouping boundaries.
        """
        motive1 = Group(level_name="motive", events=[self.chord_I, self.chord_V])
        motive2 = Group(level_name="motive", events=[self.chord_I_end])
        phrase = Group(level_name="phrase", sub_groups=[motive1, motive2])

        # Execute reduction
        tsr_tree = reduce_time_span(phrase)
        
        self.assertIsInstance(tsr_tree, TimeSpanNode)
        self.assertEqual(len(tsr_tree.children), 2, "The phrase node should have two children corresponding to the two motives.")
        
        # Based on the current stub logic, it promotes the first event of the first subgroup
        self.assertEqual(tsr_tree.head, self.chord_I, "The Time-Span head should be extracted correctly.")

    def test_prolongational_reduction(self):
        """
        4. PROLONGATIONAL REDUCTION: Check that the tree correctly assigns 
        tension (Right-Branching) and relaxation (Left-Branching).
        """
        # Manually set up a Time-Span Reduction tree for testing
        child1 = TimeSpanNode(head=self.chord_V)       # V chord
        child2 = TimeSpanNode(head=self.chord_I_end)   # I chord
        
        tsr_node_I = TimeSpanNode(head=self.chord_I, children=[child1, child2])

        # Apply prolongational branching treating chord_I as the target
        prolongational_tree = assign_prolongational_branching(tsr_node_I, self.chord_I)

        self.assertEqual(len(prolongational_tree.children), 2)
        
        # Test I -> V logic (Departure/Tensing)
        self.assertEqual(
            prolongational_tree.children[0].branch_type, 
            BranchType.RIGHT_TENSING,
            "Motion from I to V should be a right-branching (tensing) departure."
        )
        
        # Test I -> I logic (Static/Prolongation)
        self.assertEqual(
            prolongational_tree.children[1].branch_type, 
            BranchType.STRONG_PROLONGATION,
            "Motion from I to I should be a strong prolongation."
        )

    def test_prolongational_resolution(self):
        """
        Test the resolution logic (V -> I) in Prolongational Reduction.
        """
        # Setup TSR node where the target event is V, and the child is I
        tsr_node_V = TimeSpanNode(head=self.chord_V, children=[TimeSpanNode(head=self.chord_I)])
        
        prol_tree_V = assign_prolongational_branching(tsr_node_V, self.chord_V)
        
        # Test V -> I logic (Arrival/Relaxing)
        self.assertEqual(
            prol_tree_V.children[0].branch_type, 
            BranchType.LEFT_RELAXING,
            "Motion from V to I should be a left-branching (relaxing) resolution."
        )

if __name__ == "__main__":
    unittest.main()