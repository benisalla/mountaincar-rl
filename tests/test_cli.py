import json

from mountaincar_rl import compare, evaluate, play, train


def test_train_eval_play_cli(tmp_path, capsys):
    out = tmp_path / "run"
    train.main(["--algo", "sarsa", "--episodes", "3", "--eval-episodes", "2", "--output-dir", str(out)])
    assert (out / "agent.pt").exists() and (out / "history.json").exists()
    evaluate.main([str(out), "--episodes", "2"])
    text = capsys.readouterr().out
    assert json.loads(text[text.index("{"):])["episodes"] == 2
    gif = tmp_path / "car.gif"
    play.main([str(out), "--gif", str(gif)])
    assert gif.stat().st_size > 0


def test_compare_runs_a_suite_and_draws_figures(tmp_path, monkeypatch):
    monkeypatch.setitem(compare.SUITES, "unit", [("q", "q-learning", False, {}), ("cem", "cem", False,
                                                                                    {"cem_population": 2})])
    rows = compare.run_suite("unit", out=str(tmp_path / "runs"), episodes=3, verbose=False)
    assert [r["run"] for r in rows] == ["q", "cem"]
    files = compare.make_figures("unit", out=str(tmp_path / "runs"), assets=str(tmp_path / "assets"))
    assert all(f.exists() for f in files)
    assert "| `q` |" in (tmp_path / "assets" / "unit_table.md").read_text()
