import os

def test_app_exists():
    assert os.path.exists("app.py"), "app.py should exist"
    
    with open("app.py", "r") as f:
        content = f.read()
        assert "import streamlit as st" in content
        assert "Baseline" in content
        assert "Monitor" in content
        assert "Heal" in content
        assert "Export" in content

if __name__ == "__main__":
    test_app_exists()
    print("test_app_exists passed!")
