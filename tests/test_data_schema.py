from app.data.pipeline import load_and_validate_training_csv, profile_dataframe, write_profile
from app.data.schema import expected_training_columns, validate_training_dataframe
from scripts.generate_demo_data import generate_dataset


def test_expected_columns_unique_and_include_target():
    cols = expected_training_columns()
    assert len(cols) == len(set(cols))
    assert "target_class" in cols


def test_demo_frame_validates_strict():
    df = generate_dataset(120, random_seed=7)
    r = validate_training_dataframe(df, strict_categories=True, min_rows=50)
    assert r.ok
    assert r.row_count == 120
    assert r.target_positive_rate is not None


def test_missing_column_fails():
    df = generate_dataset(60, random_seed=8)
    df = df.drop(columns=["rating"])
    r = validate_training_dataframe(df, min_rows=10)
    assert not r.ok
    assert any("Missing required" in e for e in r.errors)


def test_bad_target_fails():
    df = generate_dataset(60, random_seed=9)
    df.loc[0, "target_class"] = 2
    r = validate_training_dataframe(df, min_rows=10)
    assert not r.ok


def test_strict_unknown_category_fails():
    df = generate_dataset(60, random_seed=10)
    df.loc[0, "tariff_type"] = "unknown_tariff"
    r = validate_training_dataframe(df, strict_categories=True, min_rows=10)
    assert not r.ok


def test_profile_roundtrip(tmp_path):
    df = generate_dataset(40, random_seed=11)
    p = profile_dataframe(df, "target_class")
    assert p["n_rows"] == 40
    out = tmp_path / "prof.json"
    write_profile(p, out)
    assert out.exists()


def test_load_and_validate_integration(tmp_path):
    df = generate_dataset(80, random_seed=12)
    path = tmp_path / "t.csv"
    df.to_csv(path, index=False)
    _, r = load_and_validate_training_csv(path, strict_categories=True, min_rows=30)
    assert r.ok
