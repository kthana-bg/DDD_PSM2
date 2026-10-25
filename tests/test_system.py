import os, sys, tempfile
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
os.environ["DRIVER_DB_PATH"] = os.path.join(tempfile.mkdtemp(), "t.db")
from utils.auth_utils import hash_password, verify_password
from utils.alerts import AlertManager
from utils.model_loader import available_models, load_results


def test_password():
    h = hash_password("abc123")
    assert verify_password("abc123", h) and not verify_password("x", h)


def test_alert_escalation():
    a = AlertManager()
    assert a.update("Alert") is None
    assert a.update("Warning") == "soft"
    assert a.update("Danger") == "loud"


def test_results_loaded():
    assert len(load_results()) == 3 and isinstance(available_models(), dict)
