"""웹 대시보드(web/fl_server): 결과 폴더 찾기·읽기, 비교표, DLG, 경로 거부, /api/start 검증, 한 번에 한 실행.

실제 학습은 하지 않는다 (하위 프로세스는 가짜 Popen).
"""

import io
import json
from dataclasses import replace
from pathlib import Path

import pytest
import torch

from fedsec.client.program import check_server, client_plan
from fedsec.environments.config import load_env_config
from fedsec.experiment.config import load_attack_defense, load_config
from fedsec.server.app import create_app
from fedsec.server.state import ServerState
from web.fl_server.dashboard import create_dashboard
from web.fl_server.live import (
    LazyServerLog,
    LiveRunner,
    StateHolder,
    build_server_state,
    parse_start,
)

ROOT = Path(__file__).resolve().parents[1]
ROUNDS_HEADER = ("experiment,round,val_loss,val_acc,num_selected,num_accepted,num_late,malicious_selected,"
                 "malicious_accepted,overall_malicious_ratio,accepted_malicious_ratio,updated,skip_reason,aggregator")
CLIENTS_HEADER = "experiment,round,client,malicious,selected,accepted,reason,num_samples,update_norm,norm_defended"


def write(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")


def make_run(run_dir: Path, summary: dict, rounds: int = 2) -> None:
    write(run_dir / "summary.json", json.dumps(summary))
    rows = [f"x,{r},1.5,0.{r}5,10,9,1,2,2,0.2,0.2222,True,,fedavg" for r in range(1, rounds + 1)]
    write(run_dir / "rounds.csv", "\n".join([ROUNDS_HEADER, *rows]) + "\n")
    write(run_dir / "clients.csv", f"{CLIENTS_HEADER}\nx,1,0,False,True,True,,4500,1.7,1.7\nx,1,3,True,True,False,late,4500,8.5,8.5\n")
    write(run_dir / "class_acc.csv", "round,split,class,correct,total,acc\n1,val,0,1,2,0.5\n2,val,0,2,2,1.0\n")
    write(run_dir / "config.yaml", "seed: 0\nplan:\n  malicious: [2, 3]\n  delayed: []\n")


@pytest.fixture
def experiments(tmp_path: Path) -> Path:
    base = tmp_path / "experiments"
    make_run(base / "matrix" / "baseline-attack-fedavg-s0",
             {"env": "baseline", "attack": True, "aggregator": "fedavg", "seed": 0, "final_val_acc": 0.25,
              "final_test_acc": None, "diverged_round": 3})
    make_run(base / "replay" / "r1", {"env": "baseline", "attack": True, "aggregator": "median", "seed": 0})
    write(base / "replay" / "r1" / "replay.json",
          json.dumps({"source": "matrix/baseline-attack-fedavg-s0", "from_round": 2, "change": {"aggregator": "median"}}))
    write(base / "matrix" / "baseline-attack-fedavg-s0" / "models" / "nested" / "summary.json", "{}")
    write(base / "matrix" / "_tables" / "comparison.csv",
          "env,metric,defense,seed,value_pp,reference_pp,diverged_rounds\n"
          "baseline,attack_damage,,0,,66.16,13\nbaseline,attack_damage,,1,,66.43,11\n"
          "baseline,normal_loss,median,0,0.25,0.25,\nbaseline,normal_loss,median,1,0.15,0.15,\n")
    write(base / "matrix" / "_tables" / "accuracy.svg", "<svg xmlns='http://www.w3.org/2000/svg'/>")
    write(base / "matrix" / "runs.csv", "name,env,attack,aggregator,seed,status,final_test_acc,diverged_round\n"
                                        "a,baseline,normal,fedavg,0,done,0.7616,\nb,baseline,attack,fedavg,0,diverged,0.1,13\n")
    dlg = base / "dlg" / "sweep"
    write(dlg / "results.csv", "image_index,label,class,sigma,iterations,mse,label_correct\n"
                               "1,9,truck,0.0,300,0.0001,True\n1,9,truck,0.01,300,0.05,True\n")
    write(dlg / "config.yaml", "client: 0\nimages: [1]\nsigmas: [0.0, 0.01]\niterations: [300]\n")
    (dlg / "dlg_images").mkdir()
    (dlg / "dlg_images" / "img1_original.png").write_bytes(b"\x89PNG fake")
    (dlg / "summary.png").write_bytes(b"\x89PNG fake")
    write(tmp_path / "secret.txt", "비밀")
    return base


class FakeProcess:
    def __init__(self, command: list[str], **kwargs):
        self.command = command
        self.kwargs = kwargs
        self.returncode = None
        self.stdout = io.StringIO("device=cpu\n")

    def poll(self):
        return self.returncode

    def terminate(self):
        self.returncode = -15

    def kill(self):
        self.returncode = -9

    def wait(self, timeout=None):
        return self.returncode


def tiny_state(*args, **kwargs) -> ServerState:
    return ServerState({"w": torch.zeros(2)}, num_clients=10, config={"env": args[1] if len(args) > 1 else "baseline"})


@pytest.fixture
def setup(experiments: Path):
    holder = StateHolder(tiny_state())
    processes = []

    def popen(command, **kwargs):
        processes.append(FakeProcess(command, **kwargs))
        return processes[-1]

    runner = LiveRunner(holder, ROOT, experiments, "http://127.0.0.1:8765", popen=popen, state_factory=tiny_state)
    app = create_app(holder)
    app.register_blueprint(create_dashboard(experiments, runner))
    return app.test_client(), processes, holder


def data_of(response):
    body = response.get_json()
    assert body is not None, response.data[:200]
    return body


# ---- 조회형 ----

def test_runs_discovers_summary_folders_including_replay_and_skips_models(setup):
    client, _, _ = setup

    runs = data_of(client.get("/api/runs"))["data"]

    ids = [run["id"] for run in runs]
    assert ids == ["matrix/baseline-attack-fedavg-s0", "replay/r1"]
    replay = next(run for run in runs if run["id"] == "replay/r1")
    assert replay["replay"] == {"source": "matrix/baseline-attack-fedavg-s0", "from_round": 2,
                                "change": {"aggregator": "median"}}
    assert replay["group"] == "replay"


def test_run_detail_returns_parsed_rounds_clients_and_last_class_acc(setup):
    client, _, _ = setup

    detail = data_of(client.get("/api/runs/matrix/baseline-attack-fedavg-s0"))["data"]

    assert detail["summary"]["diverged_round"] == 3
    assert [row["round"] for row in detail["rounds"]] == [1, 2]
    assert detail["rounds"][0]["accepted_malicious_ratio"] == 0.2222
    assert detail["rounds"][0]["updated"] is True
    assert detail["clients"][1]["reason"] == "late"
    assert detail["class_acc"] == [{"round": 2, "split": "val", "class": 0, "correct": 2, "total": 2, "acc": 1.0}]
    assert detail["plan"]["malicious"] == [2, 3]


def test_run_detail_turns_nan_into_null(setup, experiments):
    client, _, _ = setup
    write(experiments / "matrix" / "baseline-attack-fedavg-s0" / "summary.json", '{"final_val_acc": NaN}')

    response = client.get("/api/runs/matrix/baseline-attack-fedavg-s0")

    assert b"NaN" not in response.data
    assert data_of(response)["data"]["summary"]["final_val_acc"] is None


@pytest.mark.parametrize("path", ["/api/runs/../secret.txt", "/api/runs/..%2Fsecret.txt", "/api/runs/%2E%2E/%2E%2E/etc",
                                  "/api/runs/matrix/baseline-attack-fedavg-s0/models/nested",
                                  "/api/dlg/..%2F..%2Fsecret.txt/images/summary.png",
                                  "/api/dlg/sweep/images/..%2F..%2F..%2Fsecret.txt",
                                  "/api/dlg/sweep/images/config.yaml"])
def test_paths_outside_results_are_rejected(setup, path):
    client, _, _ = setup

    response = client.get(path)

    assert response.status_code in (400, 404)
    assert "비밀".encode() not in response.data


def test_missing_run_is_404(setup):
    client, _, _ = setup

    response = client.get("/api/runs/matrix/nope")

    assert response.status_code == 404
    assert data_of(response)["ok"] is False


def test_comparison_groups_seeds_and_means(setup):
    client, _, _ = setup

    data = data_of(client.get("/api/comparison"))["data"]

    assert data["available"] is True and data["has_chart"] is True
    loss = next(g for g in data["metrics"] if g["metric"] == "normal_loss")
    assert loss["defense"] == "median" and loss["mean_value_pp"] == 0.2
    damage = next(g for g in data["metrics"] if g["metric"] == "attack_damage")
    assert damage["mean_value_pp"] is None          # 발산한 칸은 평균 없음
    assert damage["mean_reference_pp"] == 66.3
    assert [s["diverged_rounds"] for s in damage["seeds"]] == [13, 11]
    attack = next(c for c in data["accuracy"] if c["attack"] == "attack")
    assert attack["seeds"][0]["diverged_round"] == 13


def test_comparison_chart_is_svg_without_script_permission(setup):
    client, _, _ = setup

    response = client.get("/api/comparison/accuracy.svg")

    assert response.status_code == 200
    assert response.mimetype == "image/svg+xml"
    assert response.headers["Content-Security-Policy"].startswith("default-src 'none'")


def test_dlg_lists_results_with_existing_image_names(setup):
    client, _, _ = setup

    runs = data_of(client.get("/api/dlg"))["data"]

    assert [run["id"] for run in runs] == ["sweep"]
    first, second = runs[0]["results"]
    assert first["original"] == "img1_original.png"
    assert first["reconstructed"] is None              # img1_sigma0_it300.png 파일이 없음
    assert second["sigma"] == 0.01 and runs[0]["has_summary"] is True


def test_dlg_image_is_served(setup):
    client, _, _ = setup

    response = client.get("/api/dlg/sweep/images/img1_original.png")

    assert response.status_code == 200 and response.mimetype == "image/png"


def test_index_page_is_served_with_csp(setup):
    client, _, _ = setup

    response = client.get("/")

    assert response.status_code == 200
    assert "script-src 'self'" in response.headers["Content-Security-Policy"]
    assert "안태현 PR #8".encode() in response.data


def test_browse_mode_does_not_write_into_experiments(setup, experiments):
    client, _, _ = setup
    before = sorted(p.as_posix() for p in experiments.rglob("*"))

    for path in ("/api/runs", "/api/runs/replay/r1", "/api/comparison", "/api/dlg", "/api/status"):
        client.get(path)

    assert sorted(p.as_posix() for p in experiments.rglob("*")) == before


# ---- 실행형 ----

@pytest.mark.parametrize("body", [
    {"env": "../x"}, {"attack": "dlg"}, {"aggregator": "krum"}, {"seed": -1}, {"seed": 1.5}, {"seed": "1"},
    {"seed": True}, {"rounds": 0}, {"rounds": 11}, {"name": "../evil"}, {"name": "a" * 41}, {"extra": 1}, [1, 2],
])
def test_parse_start_rejects_bad_input(body):
    with pytest.raises((TypeError, ValueError)):
        parse_start(body)


def test_parse_start_defaults():
    params = parse_start({})

    assert (params.env, params.attack, params.aggregator, params.seed, params.rounds) == ("baseline", False, "fedavg", 0, 3)
    assert params.label == "baseline-normal-fedavg-s0"


def test_start_rejects_invalid_body_with_400(setup):
    client, processes, _ = setup

    response = client.post("/api/start", json={"rounds": 50})

    assert response.status_code == 400
    assert processes == []


def test_start_rejects_non_object_json_with_400(setup):
    client, processes, _ = setup

    response = client.post("/api/start", json=[1, 2])

    assert response.status_code == 400
    assert processes == []


def test_start_requires_json(setup):
    client, processes, _ = setup

    response = client.post("/api/start", data="env=baseline", content_type="application/x-www-form-urlencoded")

    assert response.status_code == 415
    assert processes == []


def test_start_swaps_state_and_launches_run_clients(setup):
    client, processes, holder = setup
    old_state = holder.current

    response = client.post("/api/start", json={"env": "partial", "attack": "attack", "aggregator": "median",
                                               "seed": 2, "rounds": 4})

    data = data_of(response)["data"]
    assert response.status_code == 200
    assert data["run_id"].startswith("live/") and data["run_id"].endswith("-partial-attack-median-s2")
    assert holder.current is not old_state
    assert client.get("/status").get_json()["data"]["config"]["env"] == "partial"   # FL API가 새 상태를 씀
    command = processes[0].command
    assert command[1].endswith("run_clients.py")
    for flag, value in (("--name", data["run_id"]), ("--server-url", "http://127.0.0.1:8765"), ("--seed", "2"),
                        ("--rounds", "4"), ("--env", "configs/env/partial.yaml"), ("--aggregator", "median")):
        assert command[command.index(flag) + 1] == value
    assert "--attack" in command


def test_start_rejected_while_running_and_allowed_after_stop(setup):
    client, processes, _ = setup
    client.post("/api/start", json={})

    busy = client.post("/api/start", json={})
    stopped = client.post("/api/stop", json={})
    again = client.post("/api/start", json={"name": "second"})

    assert busy.status_code == 409
    assert data_of(stopped)["data"]["stopped"] is True
    assert again.status_code == 200
    assert len(processes) == 2


def test_status_reports_progress_from_run_files(setup, experiments):
    client, processes, _ = setup
    run_id = data_of(client.post("/api/start", json={"rounds": 2}))["data"]["run_id"]
    make_run(experiments / run_id, {"final_val_acc": 0.5}, rounds=1)
    (experiments / run_id / "summary.json").unlink()   # 아직 진행 중

    running = data_of(client.get("/api/status"))["data"]
    processes[0].returncode = 0
    done = data_of(client.get("/api/status"))["data"]

    assert running["state"] == "running" and running["rounds_done"] == 1 and running["rounds_total"] == 2
    assert [row["client"] for row in running["clients"]] == [0, 3]
    assert running["plan"]["malicious"] == [2, 3]
    assert done["state"] == "completed"


def test_status_idle_before_any_run(setup):
    client, _, _ = setup

    assert data_of(client.get("/api/status"))["data"]["state"] == "idle"


def test_stop_without_run_is_noop(setup):
    client, _, _ = setup

    assert data_of(client.post("/api/stop", json={}))["data"] == {"stopped": False}


def test_lazy_server_log_creates_folder_on_first_record(tmp_path):
    log = LazyServerLog(tmp_path / "run" / "server")
    assert not (tmp_path / "run").exists()   # run_clients.py가 결과 폴더를 먼저 만들 수 있게

    log.log_exclusion({"round": 1, "client_id": 3, "reason": "late"})

    assert (tmp_path / "run" / "server" / "exclusions.csv").is_file()


@pytest.mark.parametrize("env,attack,aggregator", [("baseline", False, "fedavg"), ("partial", True, "median"),
                                                   ("delay", True, "clip_fedavg")])
def test_built_state_passes_client_server_check(env, attack, aggregator):
    # run_clients.py는 서버 설정이 다르면 멈추므로, serve.py와 같은 설정이 /status에 나와야 한다
    state = build_server_state(ROOT, env, attack, aggregator, seed=1)
    cfg = load_config(ROOT / "configs" / "base.yaml").with_overrides(seed=1)
    env_cfg = load_env_config(ROOT / "configs" / "env" / f"{env}.yaml")
    attack_defense = load_attack_defense(ROOT / "configs" / "attack_defense.yaml").with_overrides(
        attack=attack, aggregator=aggregator)
    plan = client_plan("live/x", env_cfg, attack_defense, cfg.seed, 10)

    check_server(state.status(), replace(cfg, split=env_cfg.split), 10, plan)   # 다르면 RuntimeError
