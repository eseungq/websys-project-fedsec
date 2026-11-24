"""운영 환경 일정: 악성 ID, 지연 클라이언트, 라운드별 참여자, 가상 시간.

같은 시드면 환경이 달라도 악성 ID가 같고, 같은 환경·시드면 참여·지연 일정이 같다.
일정마다 독립된 난수 흐름([seed, 종류, ...])을 써서 한 요인을 바꿔도 다른 일정이 흔들리지 않는다.
"""
import numpy as np

ENVS = ("baseline", "delay", "partial", "label_skew")


class Schedule:
    def __init__(self, env, seed, num_clients=10, num_malicious=2, num_delayed=3,
                 participants=5, base_time=1.0, jitter=0.2, delay=5.0, deadline=3.0):
        if env not in ENVS:
            raise ValueError(f"알 수 없는 환경: {env}")
        self.env, self.seed, self.num_clients = env, seed, num_clients
        self.malicious = sorted(np.random.default_rng([seed, 1]).choice(
            num_clients, num_malicious, replace=False).tolist())
        self.delayed = sorted(np.random.default_rng([seed, 2]).choice(
            num_clients, num_delayed, replace=False).tolist()) if env == "delay" else []
        self.num_participants = participants if env == "partial" else num_clients
        self.base_time, self.jitter, self.delay, self.deadline = base_time, jitter, delay, deadline

    def participants(self, rnd):
        if self.num_participants >= self.num_clients:
            return list(range(self.num_clients))
        rng = np.random.default_rng([self.seed, 3, rnd])
        return sorted(rng.choice(self.num_clients, self.num_participants, replace=False).tolist())

    def virtual_finish(self, cid, rnd):
        """가상 완료 시각 (라운드 시작 = 0). 실제 순차 실행 시간과 무관하다."""
        t = np.random.default_rng([self.seed, 4, rnd, cid]).uniform(self.base_time, self.base_time + self.jitter)
        if cid in self.delayed:
            t += self.delay
        return float(t)

    def to_dict(self):
        return {"env": self.env, "seed": self.seed, "malicious": self.malicious, "delayed": self.delayed,
                "num_participants": self.num_participants, "deadline": self.deadline,
                "base_time": self.base_time, "jitter": self.jitter, "delay": self.delay}
