import unittest

from episode_lengths import group_episode_lengths


def episode(index, seconds, fps=30):
    return {
        "episode_index": index,
        "length_seconds": seconds,
        "frames": round(seconds * fps),
    }


class EpisodeLengthGroupingTests(unittest.TestCase):
    def test_groups_and_clamps_last_edge_like_viewer(self):
        groups = group_episode_lengths([
            episode(166, 200.0),
            episode(171, 201.3),
            episode(172, 224.9),
            episode(180, 225.0),
            episode(190, 249.9),
            episode(208, 425.0),
        ])
        non_empty = [group for group in groups if group["episodes"]]

        self.assertEqual("200.0–225.0s", non_empty[0]["label"])
        self.assertEqual([166, 171, 172], [
            row["episode_index"] for row in non_empty[0]["episodes"]
        ])
        self.assertEqual("225.0–250.0s", non_empty[1]["label"])
        self.assertEqual("400.0–425.0s", non_empty[-1]["label"])
        self.assertEqual([208], [
            row["episode_index"] for row in non_empty[-1]["episodes"]
        ])

    def test_single_length_uses_one_expanded_group(self):
        groups = group_episode_lengths([
            episode(2, 10.0), episode(1, 10.0),
        ])

        self.assertEqual(1, len(groups))
        self.assertEqual("10.0s", groups[0]["label"])
        self.assertEqual([1, 2], [
            row["episode_index"] for row in groups[0]["episodes"]
        ])

    def test_empty_input_has_no_groups(self):
        self.assertEqual([], group_episode_lengths([]))


if __name__ == "__main__":
    unittest.main()
