import sys
from pathlib import Path

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
    project_root = Path('.')

    # Regla 1: No debe haber ficheros .py en la raíz del proyecto
    print("Verificando ficheros Python en la raíz del proyecto...")
    for py_file in project_root.glob('*.py'):
        if py_file.is_file():
            print(
                f"Error de Estructura: Fichero Python encontrado en la raíz del proyecto: {py_file}. "
                f"Todo el código Python debe estar dentro de los módulos de 'ml_lib'.",
                file=sys.stderr
            )
            error_found = True

    # --- Verificación de estructura de módulos ---
    print("\nVerificando la estructura de módulos de ml_lib...")
    ml_lib_path = Path('ml_lib')
    if not ml_lib_path.is_dir():
        print(f"Error: El directorio '{ml_lib_path}' no fue encontrado.", file=sys.stderr)
        sys.exit(1)

    allowed_subdirs = {'services', 'interfaces', 'models', 'handlers', '__pycache__'}

    # Iteramos sobre cada módulo potencial dentro de ml_lib
    for module_path in ml_lib_path.iterdir():
        if not module_path.is_dir() or not (module_path / '__init__.py').exists():
            continue

        # Regla 2: Buscamos ficheros .py directamente en la raíz del módulo
        for py_file in module_path.glob('*.py'):
            if py_file.name != '__init__.py':
                print(
                    f"Error de Estructura: Fichero Python encontrado en la raíz del módulo '{module_path.name}'. "
                    f"Debe estar en un subdirectorio (services, models, etc.): {py_file}",
                    file=sys.stderr
                )
                error_found = True

        # Regla 3: Verificamos que los ficheros .py estén solo en los subdirectorios permitidos
        for py_file in module_path.rglob('*.py'):
            if py_file.name == '__init__.py':
                continue
            
            parent_dir_name = py_file.parent.name
            if parent_dir_name != module_path.name and parent_dir_name not in allowed_subdirs:
                print(
                    f"Error de Estructura: Fichero Python en un subdirectorio no permitido '{parent_dir_name}'. "
                    f"Fichero: {py_file}",
                    file=sys.stderr
                )
                error_found = True

    if error_found:
        print("\nFallo la validación de estructura.", file=sys.stderr)
        sys.exit(1)
    else:
        print("\nLa estructura del proyecto y los módulos es correcta.")
        sys.exit(0)

if __name__ == "__main__":
    main()