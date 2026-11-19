import pytest
import torch

from fedsec.aggregation.fedavg import apply_update, weighted_average
from fedsec.server.app import create_app
from fedsec.server.serialization import decode_state, encode_state
from fedsec.server.state import ServerPolicy, ServerState, initial_global_state

GLOBAL = {"w": torch.tensor([1.0, 2.0]), "b": torch.tensor([0.5])}
NUM_CLIENTS = 4


def _delta(w0: float, w1: float, b: float) -> dict:
    return {"w": torch.tensor([w0, w1]), "b": torch.tensor([b])}


def _payload(cid: int, delta: dict | None = None, **overrides) -> dict:
    body = {"client_id": cid, "round": 1, "model_version": 0, "num_samples": 100,
            "delta_weights": encode_state(delta or _delta(0.1, 0.1, 0.1))}
    return {**body, **overrides}


def _server(**kwargs):
    state = ServerState(GLOBAL, num_clients=NUM_CLIENTS, **kwargs)
    return state, create_app(state).test_client()


def _submit_all(client, deltas, samples):
    for cid, (delta, n) in enumerate(zip(deltas, samples)):
        assert client.post("/update", json=_payload(cid, delta, num_samples=n)).status_code == 200


def test_model_snapshot_is_same_within_round():
    _, client = _server()

    first = client.get("/model").get_json()["data"]
    client.post("/update", json=_payload(0))
    second = client.get("/model").get_json()["data"]

    assert first == second
    assert (first["round"], first["model_version"], first["participants"]) == (1, 0, [0, 1, 2, 3])
    assert torch.equal(decode_state(first["weights"])["w"], GLOBAL["w"])


def test_aggregate_matches_weighted_average():
    deltas = [_delta(1.0, 0.0, 0.0), _delta(4.0, 3.0, 1.0), _delta(-2.0, 1.0, 2.0)]
    samples = [100, 300, 100]
    _, client = _server()
    _submit_all(client, deltas, samples)

    record = client.post("/aggregate").get_json()["data"]
    new_global = decode_state(client.get("/model").get_json()["data"]["weights"])

    # w[0] = 1 + (1·100 + 4·300 − 2·100) / 500 = 3.2,  w[1] = 2 + (0·100 + 3·300 + 1·100) / 500 = 4.0
    expected = apply_update(GLOBAL, weighted_average(deltas, samples))
    assert torch.allclose(new_global["w"], torch.tensor([3.2, 4.0]))
    assert all(torch.equal(new_global[k], expected[k]) for k in GLOBAL)
    assert record["updated"] and record["accepted_clients"] == [0, 1, 2] and record["total_samples"] == 500


def test_aggregate_order_does_not_depend_on_submission_order():
    deltas = [_delta(0.3, 0.1, 0.2), _delta(0.7, 0.9, 0.1), _delta(0.2, 0.4, 0.6)]
    forward, client_f = _server()
    backward, client_b = _server()
    _submit_all(client_f, deltas, [1, 1, 1])
    for cid in (2, 1, 0):
        client_b.post("/update", json=_payload(cid, deltas[cid], num_samples=1))

    client_f.post("/aggregate")
    client_b.post("/aggregate")

    assert all(torch.equal(forward.global_state[k], backward.global_state[k]) for k in GLOBAL)


def test_round_and_version_advance_after_aggregate():
    state, client = _server()
    _submit_all(client, [_delta(0.1, 0.1, 0.1)] * 3, [1, 1, 1])

    client.post("/aggregate")
    snapshot = client.get("/model").get_json()["data"]

    assert (snapshot["round"], snapshot["model_version"]) == (2, 1)
    assert state.pending == {}
    old_round = client.post("/update", json=_payload(0))
    assert old_round.get_json()["data"]["reason"] == "round_mismatch"


@pytest.mark.parametrize("num_accepted, reason", [(0, "no_accepted"), (2, "too_few_accepted")])
def test_fewer_than_min_accepted_keeps_model(num_accepted, reason):
    state, client = _server()
    _submit_all(client, [_delta(5.0, 5.0, 5.0)] * num_accepted, [1] * num_accepted)

    record = client.post("/aggregate").get_json()["data"]

    assert (record["updated"], record["skip_reason"], record["num_accepted"]) == (False, reason, num_accepted)
    assert all(torch.equal(state.global_state[k], GLOBAL[k]) for k in GLOBAL)
    assert (state.round, state.model_version) == (2, 0)  # 라운드는 넘어가고 버전은 그대로


def test_min_accepted_is_configurable():
    state, client = _server(policy=ServerPolicy(min_accepted=1))
    _submit_all(client, [_delta(1.0, 1.0, 1.0)], [1])

    assert client.post("/aggregate").get_json()["data"]["updated"]
    assert state.model_version == 1


def test_duplicate_submission_is_excluded_and_first_kept():
    state, client = _server()
    client.post("/update", json=_payload(0, _delta(1.0, 1.0, 1.0)))

    reply = client.post("/update", json=_payload(0, _delta(9.0, 9.0, 9.0)))

    assert reply.status_code == 409
    assert reply.get_json()["data"]["reason"] == "duplicate"
    assert torch.equal(state.pending[0].update["w"], torch.tensor([1.0, 1.0]))
    assert [e["reason"] for e in state.exclusions] == ["duplicate"]


NAN_DELTA = _delta(float("nan"), 0.0, 0.0)
WRONG_SHAPE = {"w": torch.zeros(3), "b": torch.zeros(1)}
WRONG_KEYS = {"w": torch.zeros(2), "bias": torch.zeros(1)}
WRONG_DTYPE = {"w": torch.zeros(2, dtype=torch.float64), "b": torch.zeros(1)}


@pytest.mark.parametrize("body, reason, status", [
    ([1, 2], "bad_json", 400),
    ({k: v for k, v in _payload(0).items() if k != "num_samples"}, "missing_field", 400),
    (_payload(0, round="1"), "bad_type", 400),
    (_payload(True), "bad_type", 400),
    (_payload(0, delta_weights=3), "bad_type", 400),
    (_payload(7), "unknown_client", 403),
    (_payload(0, round=2), "round_mismatch", 409),
    (_payload(0, model_version=5), "version_mismatch", 409),
    (_payload(0, num_samples=0), "bad_num_samples", 400),
    (_payload(0, num_samples=4.5), "bad_num_samples", 400),
    (_payload(0, delta_weights="!!!"), "bad_payload", 400),
    (_payload(0, WRONG_SHAPE), "bad_shape", 400),
    (_payload(0, WRONG_KEYS), "bad_shape", 400),
    (_payload(0, WRONG_DTYPE), "bad_shape", 400),
    (_payload(0, NAN_DELTA), "non_finite", 400),
    (_payload(0, _delta(float("inf"), 0.0, 0.0)), "non_finite", 400),
])
def test_invalid_submission_is_excluded_with_reason(body, reason, status):
    state, client = _server()

    reply = client.post("/update", json=body)

    body_json = reply.get_json()
    assert (reply.status_code, body_json["ok"], body_json["data"]["reason"]) == (status, False, reason)
    assert body_json["error"]
    assert state.pending == {}
    assert state.exclusions[-1]["reason"] == reason
    assert state.exclusions[-1]["round"] == 1


def test_non_json_body_is_bad_json():
    state, client = _server()

    reply = client.post("/update", data="not json", content_type="text/plain")

    assert reply.get_json()["data"]["reason"] == "bad_json"


def test_oversized_request_is_rejected_before_parsing():
    state, client = _server(max_update_bytes=1000)

    reply = client.post("/update", json=_payload(0, {"w": torch.zeros(200_000), "b": torch.zeros(1)}))

    assert reply.status_code == 413
    assert reply.get_json()["data"]["reason"] == "too_large"
    assert state.exclusions[-1]["client_id"] is None


def test_oversized_update_inside_request_limit_is_too_large():
    state, client = _server(max_update_bytes=1000)

    reply = client.post("/update", json=_payload(0, {"w": torch.zeros(400), "b": torch.zeros(1)}))

    assert (reply.status_code, reply.get_json()["data"]["reason"]) == (413, "too_large")


def test_round_record_counts_exclusions():
    _, client = _server()
    _submit_all(client, [_delta(0.1, 0.1, 0.1)] * 3, [1, 1, 1])
    client.post("/update", json=_payload(0))
    client.post("/update", json=_payload(3, NAN_DELTA))

    record = client.post("/aggregate").get_json()["data"]
    status = client.get("/status").get_json()["data"]

    assert (record["num_accepted"], record["num_excluded"]) == (3, 2)
    assert record["exclusion_reasons"] == {"duplicate": 1, "non_finite": 1}
    assert status["history"] == [record] and len(status["exclusions"]) == 2


def test_injected_aggregator_is_used():
    def take_first(updates, weights):
        return updates[0]

    state, client = _server(policy=ServerPolicy(aggregator=take_first))
    _submit_all(client, [_delta(1.0, 1.0, 1.0), _delta(9.0, 9.0, 9.0), _delta(9.0, 9.0, 9.0)], [1, 1, 1])

    record = client.post("/aggregate").get_json()["data"]

    assert record["aggregator"] == "take_first"
    assert torch.equal(state.global_state["w"], torch.tensor([2.0, 3.0]))


def test_participants_policy_limits_who_can_submit():
    state, client = _server(policy=ServerPolicy(participants=lambda rnd: [3, 1]))

    snapshot = client.get("/model").get_json()["data"]
    reply = client.post("/update", json=_payload(0))

    assert snapshot["participants"] == [1, 3]
    assert (reply.status_code, reply.get_json()["data"]["reason"]) == (403, "not_selected")


@pytest.mark.parametrize("finish, reason", [(None, "missing_field"), (12.5, "late")])
def test_deadline_policy_excludes_late_or_missing_time(finish, reason):
    _, client = _server(policy=ServerPolicy(deadline=lambda rnd: 10.0))
    body = _payload(0) if finish is None else _payload(0, virtual_finish_time=finish)

    reply = client.post("/update", json=body)

    assert reply.get_json()["data"]["reason"] == reason


def test_deadline_policy_accepts_on_time():
    state, client = _server(policy=ServerPolicy(deadline=lambda rnd: 10.0))

    reply = client.post("/update", json=_payload(0, virtual_finish_time=10.0))

    assert reply.status_code == 200
    assert state.pending[0].virtual_finish_time == 10.0


def test_unknown_route_uses_envelope():
    _, client = _server()

    reply = client.get("/nope")

    assert reply.status_code == 404
    assert reply.get_json()["ok"] is False


def test_server_log_writes_round_and_exclusion_files(tmp_path):
    from fedsec.logger.server_log import ServerLog

    _, client = _server(log=ServerLog(tmp_path))
    _submit_all(client, [_delta(0.1, 0.1, 0.1)] * 3, [1, 1, 1])
    client.post("/update", json=_payload(0))
    client.post("/aggregate")

    exclusions = (tmp_path / "exclusions.csv").read_text(encoding="utf-8").splitlines()
    rounds = (tmp_path / "server_rounds.csv").read_text(encoding="utf-8").splitlines()
    assert len(exclusions) == 2 and "duplicate" in exclusions[1]
    assert len(rounds) == 2 and '"[0,1,2]"' in rounds[1]


def test_initial_global_state_is_reproducible():
    first = initial_global_state(seed=0, model_name="simple_cnn")
    second = initial_global_state(seed=0, model_name="simple_cnn")
    other = initial_global_state(seed=1, model_name="simple_cnn")

    assert all(torch.equal(first[k], second[k]) for k in first)
    assert not all(torch.equal(first[k], other[k]) for k in first)
