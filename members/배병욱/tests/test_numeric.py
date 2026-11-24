"""집계·방어·공격·서버 검증을 작은 수치 예제로 확인한다.

실행 (members/배병욱 에서): python -m unittest discover -s tests -t .
"""
import json
import unittest

import torch

from fedsim import aggregation as A
from fedsim.attacks import collusion
from fedsim.data import label_skew_split
from fedsim.environment import Schedule
from fedsim.serialization import to_bytes
from fedsim.server import create_app

T = lambda *v: torch.tensor(v, dtype=torch.float32)  # noqa: E731


class TestAggregation(unittest.TestCase):
    def test_fedavg_weighted_by_samples(self):
        # (1·[1,2] + 3·[3,4]) / 4 = [2.5, 3.5]
        out = A.fedavg([T(1, 2), T(3, 4)], [1, 3])
        self.assertTrue(torch.allclose(out, T(2.5, 3.5)))

    def test_clip_scales_only_above_bound(self):
        v, n = A.clip(T(3, 4), 1.0)          # 노름 5 → 1
        self.assertAlmostEqual(n, 5.0)
        self.assertTrue(torch.allclose(v, T(0.6, 0.8)))
        v, _ = A.clip(T(0.3, 0.4), 1.0)      # 노름 0.5 → 그대로
        self.assertTrue(torch.allclose(v, T(0.3, 0.4)))

    def test_clip_fedavg(self):
        # [6,8] → [0.6,0.8]로 잘린 뒤 [0,0]과 같은 가중 평균 → [0.3,0.4]
        out, after = A.aggregate("clip_fedavg", [T(6, 8), T(0, 0)], [1, 1], 1.0)
        self.assertTrue(torch.allclose(out, T(0.3, 0.4)))
        self.assertAlmostEqual(after[0], 1.0, places=5)

    def test_median_odd_and_even(self):
        self.assertTrue(torch.allclose(A.coordinate_median([T(1, 9), T(5, 1), T(3, 4)]), T(3, 4)))
        # 짝수 개: 가운데 두 값 평균. 좌표0: 1,2,10,20 → 6 / 좌표1: -1,0,0,100 → 0
        out = A.coordinate_median([T(1, 0), T(2, -1), T(10, 100), T(20, 0)])
        self.assertTrue(torch.allclose(out, T(6, 0)))

    def test_median_ignores_two_attackers_among_ten(self):
        honest = [T(1.0, 1.0)] * 8
        attack = [T(-50.0, -50.0)] * 2
        self.assertTrue(torch.allclose(A.coordinate_median(honest + attack), T(1, 1)))


class TestCollusion(unittest.TestCase):
    def test_two_attackers_submit_same_vector(self):
        out = collusion({3: T(1, 1), 7: T(3, 3)})
        self.assertTrue(torch.allclose(out[3], T(-10, -10)))
        self.assertTrue(torch.allclose(out[7], T(-10, -10)))

    def test_one_attacker_scales_own_update(self):
        self.assertTrue(torch.allclose(collusion({2: T(1, -2)})[2], T(-5, 10)))

    def test_no_attacker(self):
        self.assertEqual(collusion({}), {})


class TestEnvironment(unittest.TestCase):
    def test_same_malicious_across_envs(self):
        ids = {e: Schedule(e, 0).malicious for e in ("baseline", "delay", "partial", "label_skew")}
        self.assertEqual(len(set(map(tuple, ids.values()))), 1)

    def test_partial_selects_five_and_is_reproducible(self):
        s1, s2 = Schedule("partial", 1), Schedule("partial", 1)
        self.assertEqual(len(s1.participants(4)), 5)
        self.assertEqual(s1.participants(4), s2.participants(4))

    def test_delay_only_three_late(self):
        s = Schedule("delay", 0)
        late = [c for c in range(10) if s.virtual_finish(c, 1) > s.deadline]
        self.assertEqual(late, s.delayed)
        self.assertEqual(len(late), 3)

    def test_label_skew_keeps_images_and_sizes(self):
        labels = [i % 10 for i in range(200)]
        clients = [list(range(k * 20, k * 20 + 20)) for k in range(10)]
        out = label_skew_split(clients, labels, seed=0, skew=0.8, shards_per_client=2)
        self.assertEqual([len(c) for c in out], [20] * 10)
        self.assertEqual(sorted(sum(out, [])), list(range(200)))


class TestServer(unittest.TestCase):
    def setUp(self):
        self.c = create_app().test_client()
        sched = {"env": "delay", "seed": 0, "num_clients": 10}
        meta = {"aggregation": "fedavg", "min_accept": 3, "num_clients": 10, "schedule": sched}
        self.c.post("/reset", data=to_bytes(torch.zeros(2)), headers={"X-Meta": json.dumps(meta)})
        self.late = Schedule("delay", 0).delayed

    def submit(self, cid, vec, rnd=1, version=0, vf=1.0, ns=10):
        m = {"client_id": cid, "round": rnd, "version": version, "num_samples": ns, "virtual_finish": vf}
        return self.c.post("/update", data=to_bytes(vec), headers={"X-Meta": json.dumps(m)}).get_json()

    def test_rejections_with_reason(self):
        self.assertTrue(self.submit(0, T(1, 1))["accepted"])
        self.assertEqual(self.submit(0, T(1, 1))["reason"], "duplicate")
        self.assertEqual(self.submit(1, T(1, 1), rnd=2)["reason"], "wrong_round")
        self.assertEqual(self.submit(1, T(1, 1), version=5)["reason"], "stale_version")
        self.assertEqual(self.submit(42, T(1, 1))["reason"], "unknown_client")
        self.assertEqual(self.submit(2, T(1, 1, 1))["reason"], "bad_shape")
        self.assertEqual(self.submit(3, T(float("nan"), 1))["reason"], "non_finite")
        self.assertEqual(self.submit(4, T(1, 1), vf=9.0)["reason"], "late")
        self.assertEqual(self.submit(5, T(1, 1), ns=0)["reason"], "bad_num_samples")

    def test_too_few_accepted_keeps_model(self):
        self.submit(0, T(1, 1))
        self.submit(1, T(1, 1))
        info = self.c.post("/aggregate").get_json()
        self.assertFalse(info["updated"])
        self.assertIn("too_few_accepted", info["reason"])
        self.assertEqual(self.c.get("/status").get_json()["version"], 0)

    def test_fedavg_updates_global(self):
        for cid, v in [(0, T(1, 0)), (1, T(0, 1)), (2, T(1, 1))]:
            self.submit(cid, v, ns=10)
        info = self.c.post("/aggregate").get_json()
        self.assertTrue(info["updated"])
        st = self.c.get("/status").get_json()
        self.assertEqual((st["round"], st["version"]), (2, 1))


if __name__ == "__main__":
    unittest.main()
