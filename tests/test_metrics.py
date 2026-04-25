import pandas as pd
import pytest
from src.metrics import (
    attrition_rate,
    attrition_by_department,
    attrition_by_overtime,
    average_income_by_attrition,
    satisfaction_summary,
)


@pytest.fixture
def sample_df():
    return pd.DataFrame({
        "employee_id":      [1,       2,       3,       4,    5,             6],
        "department":       ["Sales", "Sales", "Sales", "HR", "Engineering", "Engineering"],
        "monthly_income":   [4000,    5000,    6000,    7000, 9000,          11000],
        "job_satisfaction": [1,       1,       3,       3,    2,             4],
        "overtime":         ["Yes",   "Yes",   "No",    "No", "Yes",         "No"],
        "attrition":        ["Yes",   "Yes",   "No",    "No", "Yes",         "No"],
    })


# --- attrition_rate ---

def test_attrition_rate_returns_expected_percent():
    df = pd.DataFrame({
        "employee_id": [1, 2, 3, 4],
        "department": ["Sales", "Sales", "HR", "HR"],
        "attrition": ["Yes", "No", "No", "Yes"],
    })
    assert attrition_rate(df) == 50.0


def test_attrition_rate_no_leavers():
    df = pd.DataFrame({"employee_id": [1, 2], "attrition": ["No", "No"]})
    assert attrition_rate(df) == 0.0


def test_attrition_rate_all_leavers():
    df = pd.DataFrame({"employee_id": [1, 2], "attrition": ["Yes", "Yes"]})
    assert attrition_rate(df) == 100.0


# --- attrition_by_department ---

def test_attrition_by_department_returns_expected_columns():
    df = pd.DataFrame({
        "employee_id": [1, 2, 3, 4],
        "department": ["Sales", "Sales", "HR", "HR"],
        "attrition": ["Yes", "No", "No", "Yes"],
    })
    result = attrition_by_department(df)
    assert list(result.columns) == ["department", "employees", "leavers", "attrition_rate"]


def test_attrition_by_department_correct_rates(sample_df):
    result = attrition_by_department(sample_df)
    rates = dict(zip(result["department"], result["attrition_rate"]))
    assert rates["Sales"] == round(2 / 3 * 100, 2)
    assert rates["Engineering"] == 50.0
    assert rates["HR"] == 0.0


def test_attrition_by_department_sorted_descending(sample_df):
    result = attrition_by_department(sample_df)
    rates = result["attrition_rate"].tolist()
    assert rates == sorted(rates, reverse=True)


# --- attrition_by_overtime ---

def test_attrition_by_overtime_returns_expected_columns(sample_df):
    result = attrition_by_overtime(sample_df)
    assert list(result.columns) == ["overtime", "employees", "leavers", "attrition_rate"]


def test_attrition_by_overtime_correct_rates(sample_df):
    result = attrition_by_overtime(sample_df)
    rates = dict(zip(result["overtime"], result["attrition_rate"]))
    assert rates["Yes"] == 100.0
    assert rates["No"] == 0.0


# --- average_income_by_attrition ---

def test_average_income_by_attrition_correct_values(sample_df):
    result = average_income_by_attrition(sample_df)
    avgs = dict(zip(result["attrition"], result["avg_monthly_income"]))
    assert avgs["Yes"] == 6000.0   # (4000 + 5000 + 9000) / 3
    assert avgs["No"] == 8000.0    # (6000 + 7000 + 11000) / 3


def test_average_income_leavers_earn_less_than_stayers(sample_df):
    result = average_income_by_attrition(sample_df)
    avgs = dict(zip(result["attrition"], result["avg_monthly_income"]))
    assert avgs["Yes"] < avgs["No"]


# --- satisfaction_summary ---

def test_satisfaction_summary_correct_attrition_rates(sample_df):
    result = satisfaction_summary(sample_df)
    rates = dict(zip(result["job_satisfaction"], result["attrition_rate"]))
    assert rates[1] == 100.0   # both employees with score 1 left
    assert rates[2] == 100.0   # the one employee with score 2 left
    assert rates[3] == 0.0     # neither employee with score 3 left
    assert rates[4] == 0.0     # the one employee with score 4 stayed


def test_satisfaction_summary_sorted_by_score(sample_df):
    result = satisfaction_summary(sample_df)
    scores = result["job_satisfaction"].tolist()
    assert scores == sorted(scores)


def test_satisfaction_summary_rates_are_per_group_not_share_of_leavers(sample_df):
    result = satisfaction_summary(sample_df)
    # If rates were share-of-leavers they would sum to ~100.
    # As true per-group attrition rates they should not.
    assert result["attrition_rate"].sum() != pytest.approx(100.0)
