import unittest

import networkx as nx

import uav_gym.utils as gym_utils


class TestMakeGraphFromLocs(unittest.TestCase):
    def test_edges_present(self):
        """
        Test that the correct edges are in the graph.
        """
        comm_range = 500
        home_loc = [0, 0]

        uav_locs = [
            [200, 300],
            [1000, 1000],
            [500, 1000]
        ]

        g = gym_utils.make_graph_from_locs(uav_locs, home_loc, comm_range)

        # edge between home and first uav at [200, 300]
        self.assertIn((0, 1), g.edges)
        # edge between second uav at [1000, 1000] and third uav at [500, 1000]
        self.assertIn((2, 3), g.edges)

    def test_edges_abscent(self):
        """
        Test that there aren't incorrect edges present.
        """
        comm_range = 500
        home_loc = [0, 0]

        uav_locs = [
            [200, 300],
            [1000, 1000],
            [500, 1000]
        ]

        g = gym_utils.make_graph_from_locs(uav_locs, home_loc, comm_range)
        # no edge between home and second uav at [1000, 1000]
        self.assertNotIn((0, 2), g.edges)
        # no edge between first uav at [200, 300] and second uav at [1000, 1000]
        self.assertNotIn((1, 2), g.edges)


class TestGetDisconnectedCount(unittest.TestCase):
    def test_no_disconnected(self):
        """
        Test that when there are none disconnected the count is 0
        """
        comm_range = 500
        home_loc = [0, 0]

        uav_locs = [
            [200, 300],
            [500, 500],
            [100, 100]
        ]

        g = gym_utils.make_graph_from_locs(uav_locs, home_loc, comm_range)

        d_count = gym_utils.get_disconnected_count(g)

        self.assertEqual(d_count, 0)

    def test_some_disconnected(self):
        """
        Test for some disconnected
        """
        comm_range = 500
        home_loc = [0, 0]

        uav_locs = [
            [200, 300],
            [800, 800],
            [1000, 1000]
        ]

        g = gym_utils.make_graph_from_locs(uav_locs, home_loc, comm_range)

        d_count = gym_utils.get_disconnected_count(g)

        self.assertEqual(d_count, 2)

    def test_all_disconnected(self):
        """
        Test for all disconnected
        """
        comm_range = 500
        home_loc = [0, 0]

        uav_locs = [
            [900, 900],
            [800, 800],
            [1000, 1000]
        ]

        g = gym_utils.make_graph_from_locs(uav_locs, home_loc, comm_range)

        d_count = gym_utils.get_disconnected_count(g)

        self.assertEqual(d_count, 3)


if __name__ == '__main__':
    unittest.main()
