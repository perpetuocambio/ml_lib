import sys
import re
from pathlib import Path
from typing import List, Tuple


def check_code_quality(ml_lib_path: Path) -> Tuple[bool, List[str]]:
    """
    Verifica principios de calidad de código.

    Returns:
        (tiene_errores, lista_de_warnings)
    """
    warnings = []

    # Patrón para detectar Dict[str, Any] en return types
    dict_any_pattern = re.compile(r'-> .*Dict\[str,\s*Any\]')

    # Patrón para detectar tuplas largas (>2 elementos) en return types
    long_tuple_pattern = re.compile(r'-> .*Tuple\[([^\]]+)\]')

    for py_file in ml_lib_path.rglob("*.py"):
        if py_file.name.startswith("test_") or "__pycache__" in str(py_file):
            continue

        try:
            content = py_file.read_text(encoding='utf-8')
            lines = content.split('\n')

            for line_num, line in enumerate(lines, 1):
                # Detectar Dict[str, Any] en return types
                if dict_any_pattern.search(line):
                    warnings.append(
                        f"⚠️  Calidad: {py_file.relative_to(ml_lib_path.parent)}:{line_num}\n"
                        f"    Detectado Dict[str, Any] en tipo de retorno.\n"
                        f"    Considera usar una dataclass en su lugar.\n"
                        f"    → {line.strip()}"
                    )

                # Detectar tuplas largas (>2 elementos)
                match = long_tuple_pattern.search(line)
                if match:
                    tuple_content = match.group(1)
                    num_elements = tuple_content.count(',') + 1
                    if num_elements > 2:
                        warnings.append(
                            f"⚠️  Calidad: {py_file.relative_to(ml_lib_path.parent)}:{line_num}\n"
                            f"    Detectada tupla con {num_elements} elementos en tipo de retorno.\n"
                            f"    Considera usar una dataclass para mayor claridad.\n"
                            f"    → {line.strip()}"
                        )

        except Exception as e:
            warnings.append(f"⚠️  Error leyendo {py_file}: {e}")

    return len(warnings) > 0, warnings


def main():
    """
    Verifica la estructura del proyecto y de los módulos en ml_lib.

    Reglas:
    1. No debe haber ficheros .py en la raíz del proyecto.
    2. Dentro de un módulo de ml_lib, no debe haber ficheros .py en su raíz (excepto __init__.py).
    3. Los ficheros .py de un módulo solo pueden estar dentro de subdirectorios llamados:
       'services', 'interfaces', 'models', 'handlers'.
    """
    error_found = False
    project_root = Path(".")

    # Regla 1: No debe haber ficheros .py en la raíz del proyecto
    print("Verificando ficheros Python en la raíz del proyecto...")
    for py_file in project_root.glob("*.py"):
        if py_file.is_file():
            print(
                f"Error de Estructura: Fichero Python encontrado en la raíz del proyecto: {py_file}. "
                f"Todo el código Python debe estar dentro de los módulos de 'ml_lib'.",
                file=sys.stderr,
            )
            error_found = True

    # --- Verificación de estructura de módulos ---
    print("\nVerificando la estructura de módulos de ml_lib...")
    ml_lib_path = Path("ml_lib")
    if not ml_lib_path.is_dir():
        print(
            f"Error: El directorio '{ml_lib_path}' no fue encontrado.", file=sys.stderr
        )
        sys.exit(1)

    allowed_subdirs = {"services", "interfaces", "models", "handlers", "__pycache__"}

    # Iteramos sobre cada módulo potencial dentro de ml_lib
    for module_path in ml_lib_path.iterdir():
        if not module_path.is_dir() or not (module_path / "__init__.py").exists():
            continue

        # Regla 2: Buscamos ficheros .py directamente en la raíz del módulo
        for py_file in module_path.glob("*.py"):
            if py_file.name != "__init__.py":
                print(
                    f"Error de Estructura: Fichero Python encontrado en la raíz del módulo '{module_path.name}'. "
                    f"Debe estar en un subdirectorio (services, models, etc.): {py_file}",
                    file=sys.stderr,
                )
                error_found = True

        # Regla 3: Verificamos que los ficheros .py estén solo en los subdirectorios permitidos
        for py_file in module_path.rglob("*.py"):
            if py_file.name == "__init__.py":
                continue

            parent_dir_name = py_file.parent.name
            if (
                parent_dir_name != module_path.name
                and parent_dir_name not in allowed_subdirs
            ):
                print(
                    f"Error de Estructura: Fichero Python en un subdirectorio no permitido '{parent_dir_name}'. "
                    f"Fichero: {py_file}",
                    file=sys.stderr,
                )
                error_found = True

    # Verificación de calidad de código
    print("\nVerificando calidad de código...")
    has_quality_issues, quality_warnings = check_code_quality(ml_lib_path)

    if quality_warnings:
        print("\n📋 Advertencias de Calidad de Código:")
        print("=" * 80)
        for warning in quality_warnings:
            print(warning)
            print()

    if error_found:
        print("\n❌ Fallo la validación de estructura.", file=sys.stderr)
        sys.exit(1)
    elif has_quality_issues:
        print("\n⚠️  La estructura es correcta, pero hay advertencias de calidad.")
        print("    Consulta docs/architecture/INTERFACE_IMPROVEMENTS.md para mejores prácticas.")
        sys.exit(0)  # No fallar por advertencias, solo informar
    else:
        print("\n✅ La estructura del proyecto y la calidad de código son correctas.")
        sys.exit(0)


if __name__ == "__main__":
    main()
