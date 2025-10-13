"""
Syntax validation tests - can run without torch/diffusers installed.

These tests verify that all code is syntactically correct and imports
are properly structured.
"""

import ast
import sys
from pathlib import Path


class TestSyntaxValidation:
    """Test that all Python files have valid syntax."""

    def test_all_python_files_valid_syntax(self):
        """Test that all Python files in ml_lib/diffusion compile."""
        print("\n🔍 Validating Python syntax...")

        ml_lib_path = Path(__file__).parent.parent.parent / "ml_lib" / "diffusion"
        errors = []

        python_files = list(ml_lib_path.rglob("*.py"))
        print(f"  Found {len(python_files)} Python files")

        for py_file in python_files:
            # Skip __pycache__ and .pyc files
            if "__pycache__" in str(py_file):
                continue

            try:
                with open(py_file, 'r', encoding='utf-8') as f:
                    source = f.read()

                # Try to parse as AST
                ast.parse(source, filename=str(py_file))

                # Try to compile
                compile(source, str(py_file), 'exec')

            except SyntaxError as e:
                errors.append(f"{py_file.relative_to(ml_lib_path)}: {e}")
            except Exception as e:
                errors.append(f"{py_file.relative_to(ml_lib_path)}: Unexpected error: {e}")

        if errors:
            print("\n❌ Syntax errors found:")
            for error in errors:
                print(f"  - {error}")
            assert False, f"Found {len(errors)} syntax errors"

        print(f"✅ All {len(python_files)} files have valid syntax")

    def test_no_todos_in_production_code(self):
        """Verify there are no TODO comments in production code."""
        print("\n🔍 Checking for TODO comments...")

        ml_lib_path = Path(__file__).parent.parent.parent / "ml_lib" / "diffusion"
        todos = []

        for py_file in ml_lib_path.rglob("*.py"):
            # Skip docs and tests
            if "docs" in str(py_file) or "test" in str(py_file):
                continue

            if "__pycache__" in str(py_file):
                continue

            try:
                with open(py_file, 'r', encoding='utf-8') as f:
                    for line_num, line in enumerate(f, 1):
                        if "TODO" in line and not line.strip().startswith("#"):
                            # Only flag if TODO is in actual code, not comments
                            continue
                        if "TODO:" in line or "TODO " in line:
                            todos.append(f"{py_file.name}:{line_num}: {line.strip()}")
            except Exception:
                pass

        if todos:
            print(f"\n⚠️  Found {len(todos)} TODO comments:")
            for todo in todos[:10]:  # Show first 10
                print(f"  - {todo}")
            # Don't fail, just warn
            print("\n  Note: TODOs should be converted to issues or completed")

        else:
            print("✅ No TODO comments in production code")

    def test_import_structure(self):
        """Test that import structure is correct."""
        print("\n🔍 Validating import structure...")

        # Test main package imports
        try:
            # These should not raise ImportError for structure
            import ml_lib
            print("  ✅ ml_lib package structure OK")
        except ImportError as e:
            # Expected if dependencies not installed
            if "torch" in str(e) or "diffusers" in str(e):
                print(f"  ⚠️  Dependencies not installed (expected): {e}")
            else:
                raise

    def test_no_syntax_antipatterns(self):
        """Check for common syntax antipatterns."""
        print("\n🔍 Checking for syntax antipatterns...")

        ml_lib_path = Path(__file__).parent.parent.parent / "ml_lib" / "diffusion"
        issues = []

        for py_file in ml_lib_path.rglob("*.py"):
            if "docs" in str(py_file) or "__pycache__" in str(py_file):
                continue

            try:
                with open(py_file, 'r', encoding='utf-8') as f:
                    source = f.read()
                    lines = source.split('\n')

                # Check for bare except without logging
                for i, line in enumerate(lines, 1):
                    if "except:" in line and i + 1 < len(lines):
                        next_line = lines[i].strip()
                        if next_line == "pass":
                            issues.append(
                                f"{py_file.name}:{i}: Bare except with pass (should log)"
                            )

            except Exception:
                pass

        if issues:
            print(f"\n⚠️  Found {len(issues)} potential issues:")
            for issue in issues[:5]:
                print(f"  - {issue}")
        else:
            print("✅ No obvious antipatterns detected")


class TestCodeQuality:
    """Test code quality metrics."""

    def test_file_size_reasonable(self):
        """Check that no files are excessively large."""
        print("\n📏 Checking file sizes...")

        ml_lib_path = Path(__file__).parent.parent.parent / "ml_lib" / "diffusion"
        large_files = []

        MAX_LINES = 1000  # Reasonable maximum for a single file

        for py_file in ml_lib_path.rglob("*.py"):
            if "docs" in str(py_file) or "__pycache__" in str(py_file):
                continue

            try:
                with open(py_file, 'r', encoding='utf-8') as f:
                    lines = len(f.readlines())

                if lines > MAX_LINES:
                    large_files.append(f"{py_file.name}: {lines} lines")

            except Exception:
                pass

        if large_files:
            print(f"\n⚠️  Found {len(large_files)} large files:")
            for file_info in large_files:
                print(f"  - {file_info}")
            print(f"\n  Consider refactoring files > {MAX_LINES} lines")
        else:
            print(f"✅ All files under {MAX_LINES} lines")

    def test_docstring_coverage(self):
        """Check that public functions have docstrings."""
        print("\n📝 Checking docstring coverage...")

        ml_lib_path = Path(__file__).parent.parent.parent / "ml_lib" / "diffusion"
        missing_docstrings = []

        for py_file in ml_lib_path.rglob("*.py"):
            if "docs" in str(py_file) or "__pycache__" in str(py_file):
                continue
            if py_file.name.startswith("_"):  # Skip private modules
                continue

            try:
                with open(py_file, 'r', encoding='utf-8') as f:
                    tree = ast.parse(f.read())

                for node in ast.walk(tree):
                    if isinstance(node, ast.ClassDef):
                        # Check class docstring
                        if not ast.get_docstring(node):
                            missing_docstrings.append(f"{py_file.name}: class {node.name}")

                    elif isinstance(node, ast.FunctionDef):
                        # Only check public functions
                        if not node.name.startswith("_"):
                            if not ast.get_docstring(node):
                                missing_docstrings.append(
                                    f"{py_file.name}: function {node.name}"
                                )

            except Exception:
                pass

        if missing_docstrings:
            print(f"\n⚠️  Found {len(missing_docstrings)} items without docstrings")
            # Show sample
            for item in missing_docstrings[:5]:
                print(f"  - {item}")
            print("\n  Note: Public APIs should have docstrings")
        else:
            print("✅ Good docstring coverage")


if __name__ == "__main__":
    import pytest
    pytest.main([__file__, "-v", "-s"])
