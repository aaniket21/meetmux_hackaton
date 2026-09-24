import os

def test_project_structure_exists():
    assert os.path.exists("core"), "core directory should exist"
    assert os.path.exists("scripts"), "scripts directory should exist"
    assert os.path.exists("data"), "data directory should exist"
    assert os.path.exists("requirements.txt"), "requirements.txt should exist"

if __name__ == "__main__":
    test_project_structure_exists()
    print("Project structure is correct!")
